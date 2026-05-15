use chrono::Utc;
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
/// MemoryPilot v4.0 Database Engine — SQLite + FTS5.
/// Features: dedup, importance, TTL, bulk ops, export, auto-prompt, lazy embedding, content hash.
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use uuid::Uuid;

#[path = "db/benchmark.rs"]
mod benchmark;
#[path = "db/benchmark_fr.rs"]
mod benchmark_fr;
#[path = "db/benchmark_longmemeval.rs"]
mod benchmark_longmemeval;
#[path = "db/compaction.rs"]
mod compaction;
#[path = "db/embed_worker.rs"]
mod embed_worker;
#[path = "db/export.rs"]
mod export;
#[path = "db/schema.rs"]
mod schema;
#[path = "db/transcript.rs"]
mod transcript;

use embed_worker::{
    cached_embed_text, queue_access_update, queue_embedding_job, set_embed_ann_index,
    set_embed_db_path,
};

const DB_DIR: &str = ".MemoryPilot";
const DB_FILE: &str = "memory.db";
const PROMPT_FILE: &str = "GLOBAL_PROMPT.md";
const DEDUP_THRESHOLD: f64 = 0.85;

pub(crate) fn content_hash(text: &str) -> String {
    let mut h: u64 = 14695981039346656037;
    for b in text.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(1099511628211);
    }
    format!("{:016x}", h)
}

#[derive(Debug, Clone, Default)]
struct QueryIntent {
    preference: bool,
    temporal: bool,
    user_turn: bool,
    assistant_turn: bool,
    update_or_correction: bool,
    technical: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    pub id: String,
    pub content: String,
    pub kind: String,
    pub project: Option<String>,
    pub tags: Vec<String>,
    pub source: String,
    pub importance: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<String>,
    pub created_at: String,
    pub updated_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_accessed_at: Option<String>,
    pub access_count: i32,
}
#[derive(Debug, Clone, Serialize)]
pub struct SearchResult {
    pub memory: Memory,
    pub score: f64,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub sources: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Project {
    pub name: String,
    pub path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub created_at: String,
    pub memory_count: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecallMode {
    Safe,
    Default,
    Full,
}

impl RecallMode {
    pub fn from_str(value: Option<&str>) -> Result<Self, String> {
        match value.unwrap_or("safe").trim().to_ascii_lowercase().as_str() {
            "safe" => Ok(Self::Safe),
            "default" => Ok(Self::Default),
            "full" => Ok(Self::Full),
            other => Err(format!(
                "Invalid recall mode '{}'. Use safe, default, or full.",
                other
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Safe => "safe",
            Self::Default => "default",
            Self::Full => "full",
        }
    }

    pub fn includes_credentials(self) -> bool {
        matches!(self, Self::Full)
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryScope {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thread_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub window_id: Option<String>,
}

impl MemoryScope {
    pub fn is_empty(&self) -> bool {
        self.session_id.is_none() && self.thread_id.is_none() && self.window_id.is_none()
    }
}

/// Number of read-only SQLite connections per Database handle. Each handle
/// (one per HTTP worker / MCP client) keeps this many readers warm so that
/// concurrent searches no longer serialize on a single Mutex<Connection>.
/// SQLite WAL allows N readers + 1 writer concurrently, so a larger pool
/// only costs file descriptors and ~80 KB of cache per connection.
const READ_POOL_SIZE: usize = 16;

pub struct Database {
    conn: Connection,
    read_pool: Vec<Mutex<Connection>>,
    ann: Option<Arc<crate::ann::AnnIndex>>,
    /// Set to `true` once the detached `spawn_ann_warmup` thread has
    /// finished hydrating the in-memory ANN index from SQLite. Lets
    /// callers opt into a deterministic search path via
    /// [`Database::wait_for_ann_warm`] / [`Database::open_at_warm`]
    /// without paying that cost on the default `open_at`.
    ann_warm_complete: Arc<std::sync::atomic::AtomicBool>,
}

/// RAII helper used by the ANN warm-up thread: ensures the
/// completion flag is published exactly once, no matter which exit
/// path the worker takes (success, prepare failure, query failure).
struct WarmupGuard {
    flag: Arc<std::sync::atomic::AtomicBool>,
}

impl Drop for WarmupGuard {
    fn drop(&mut self) {
        self.flag
            .store(true, std::sync::atomic::Ordering::Release);
    }
}

impl Database {
    pub fn open() -> Result<Self, String> {
        let dir = dirs::home_dir()
            .ok_or("Cannot find home directory")?
            .join(DB_DIR);
        std::fs::create_dir_all(&dir).map_err(|e| format!("Cannot create dir: {}", e))?;
        Self::open_at(&dir.join(DB_FILE))
    }

    pub fn open_at(path: &Path) -> Result<Self, String> {
        let conn = Connection::open(path).map_err(|e| format!("SQLite open: {}", e))?;
        schema::configure_connection(&conn)?;
        set_embed_db_path(path);

        let read_pool = schema::open_read_pool(path, READ_POOL_SIZE)?;

        let ann = Self::open_ann_index(path);
        set_embed_ann_index(ann.clone());

        let ann_warm_complete = Arc::new(std::sync::atomic::AtomicBool::new(
            ann.as_ref().map(|index| !index.is_empty()).unwrap_or(true),
        ));
        let db = Self {
            conn,
            read_pool,
            ann: ann.clone(),
            ann_warm_complete: ann_warm_complete.clone(),
        };
        db.init_schema()?;
        db.upgrade_schema()?;
        db.normalize_project_identities()?;
        // If the user just swapped the embedding model (or upgraded
        // from v4.1 small-model blobs to a v4.2 default), the on-disk
        // blob length no longer matches the active model. Wipe those
        // stale blobs so the regular backfill pass below re-embeds
        // them with the new model. Cheap one-shot SQL — no scan
        // happens once everything is up to date.
        let _ = db.invalidate_stale_embeddings();
        let _ = db.backfill_embeddings();
        let _ = db.migrate_fts_to_stemmed();
        Self::spawn_ann_warmup(ann, path.to_path_buf(), ann_warm_complete);
        Ok(db)
    }

    /// Set `embedding = NULL` on every row whose blob length doesn't
    /// match the active model's expected length. Keeps the row, just
    /// queues a re-embed.
    fn invalidate_stale_embeddings(&self) -> Result<usize, String> {
        let expected = crate::embedding::quantized_blob_len() as i64;
        let invalidated = self
            .conn
            .execute(
                "UPDATE memories
                    SET embedding = NULL
                  WHERE embedding IS NOT NULL
                    AND length(embedding) <> ?1",
                params![expected],
            )
            .map_err(|error| format!("Stale embedding invalidate: {}", error))?;
        if invalidated > 0 {
            eprintln!(
                "[MemoryPilot] Invalidated {} embeddings produced by a different model — they will be re-embedded with the active model.",
                invalidated
            );
        }
        Ok(invalidated)
    }

    /// Open the database **and block until the ANN index is fully
    /// hydrated in RAM**. The non-warm `open_at` returns immediately
    /// and lets the SQL fallback serve the first few searches while
    /// the ANN thread catches up — fine for an interactive process,
    /// but disastrous for two specific scenarios:
    ///
    /// - benchmarks, where intermittent ANN hydration causes scoring
    ///   to flicker between the ANN-pruned path and the full SQL scan
    ///   (the source of the ±10pp variance on `--benchmark-fr`),
    /// - cold-start sensitive workloads where the first 5 queries
    ///   per process were paying the warm-up tail (p99 ~6 s in the
    ///   concurrency bench).
    ///
    /// `open_at_warm` pays the warm-up upfront. Returns the same
    /// `Database` handle as `open_at` once the index is ready.
    pub fn open_at_warm(path: &Path) -> Result<Self, String> {
        let db = Self::open_at(path)?;
        db.wait_for_ann_warm(std::time::Duration::from_secs(120));
        Ok(db)
    }

    /// Block until the detached ANN warm-up thread reports completion
    /// (or `timeout` elapses). Cheap polling — we only spin every 25 ms
    /// because the warm-up pass is bottlenecked on SQLite I/O, not on
    /// the polling loop. Returns `true` if the warm-up finished within
    /// the budget, `false` on timeout.
    pub fn wait_for_ann_warm(&self, timeout: std::time::Duration) -> bool {
        let started = std::time::Instant::now();
        while !self.ann_warm_complete.load(std::sync::atomic::Ordering::Acquire) {
            if started.elapsed() >= timeout {
                return false;
            }
            std::thread::sleep(std::time::Duration::from_millis(25));
        }
        true
    }

    /// One-shot migration that rewrites every memory's FTS5 entry to
    /// include the Snowball-stemmed projection of its content. Idempotent:
    /// guarded by a `fts_stem_version=1` row in the `config` table.
    fn migrate_fts_to_stemmed(&self) -> Result<(), String> {
        let current: String = self
            .conn
            .query_row(
                "SELECT value FROM config WHERE key='fts_stem_version'",
                [],
                |row| row.get(0),
            )
            .unwrap_or_default();
        if current == "1" {
            return Ok(());
        }

        let mut stmt = self
            .conn
            .prepare("SELECT m.rowid, m.content, m.tags, m.kind, COALESCE(m.project, '') FROM memories m")
            .map_err(|error| format!("FTS stem migration prepare: {}", error))?;
        let rows = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                ))
            })
            .map_err(|error| format!("FTS stem migration query: {}", error))?;

        let entries: Vec<(i64, String, String, String, String)> = rows
            .filter_map(|row| row.ok())
            .collect();
        drop(stmt);

        let tx = self
            .conn
            .unchecked_transaction()
            .map_err(|error| format!("FTS stem migration tx: {}", error))?;
        for (rowid, content, tags_json, kind, project) in entries {
            let fts_content = Self::fts_index_content(&content);
            tx.execute("DELETE FROM memories_fts WHERE rowid=?1", params![rowid])
                .ok();
            tx.execute(
                "INSERT INTO memories_fts (rowid,content,tags,kind,project) VALUES (?1,?2,?3,?4,?5)",
                params![rowid, fts_content, tags_json, kind, project],
            )
            .ok();
        }
        tx.execute(
            "INSERT OR REPLACE INTO config (key,value) VALUES ('fts_stem_version','1')",
            [],
        )
        .map_err(|error| format!("FTS stem migration version: {}", error))?;
        tx.commit()
            .map_err(|error| format!("FTS stem migration commit: {}", error))?;
        Ok(())
    }

    /// Hydrate the ANN index in a detached worker so `open_at` does not pay the
    /// upfront I/O cost. Until warm-up finishes, searches transparently fall
    /// back to the deterministic SQL scan path. The `complete` flag is set
    /// once the worker exits so [`Database::wait_for_ann_warm`] can block on
    /// it for benchmarks and cold-start-sensitive workloads.
    fn spawn_ann_warmup(
        ann: Option<Arc<crate::ann::AnnIndex>>,
        db_path: PathBuf,
        complete: Arc<std::sync::atomic::AtomicBool>,
    ) {
        let Some(index) = ann else {
            complete.store(true, std::sync::atomic::Ordering::Release);
            return;
        };
        if !index.is_empty() {
            complete.store(true, std::sync::atomic::Ordering::Release);
            return;
        }
        std::thread::Builder::new()
            .name("memorypilot-ann-warmup".into())
            .spawn(move || {
                // Always mark complete on exit, even on early returns,
                // so callers blocking on `wait_for_ann_warm` are never
                // stranded if the SQL probe fails for any reason.
                let _guard = WarmupGuard {
                    flag: complete.clone(),
                };
                let conn = match Connection::open(&db_path) {
                    Ok(conn) => conn,
                    Err(_) => return,
                };
                let _ = conn
                    .execute_batch("PRAGMA journal_mode = WAL; PRAGMA synchronous = NORMAL;");
                let mut stmt = match conn
                    .prepare("SELECT id, embedding FROM memories WHERE embedding IS NOT NULL")
                {
                    Ok(stmt) => stmt,
                    Err(_) => return,
                };
                let rows = match stmt.query_map([], |row| {
                    let id: String = row.get(0)?;
                    let blob: Vec<u8> = row.get(1)?;
                    Ok((id, blob))
                }) {
                    Ok(rows) => rows,
                    Err(_) => return,
                };
                let mut added = 0usize;
                for row in rows.flatten() {
                    let (id, blob) = row;
                    let vector = crate::embedding::blob_to_vec(&blob);
                    if vector.is_empty() {
                        continue;
                    }
                    if index.add(&id, &vector).is_ok() {
                        added += 1;
                    }
                }
                if added > 0 {
                    let _ = index.persist();
                }
            })
            .ok();
    }

    fn open_ann_index(path: &Path) -> Option<Arc<crate::ann::AnnIndex>> {
        let mut ann_path = path.to_path_buf();
        let file_name = ann_path
            .file_name()
            .and_then(|name| name.to_str())
            .map(|name| format!("{}.ann.usearch", name))
            .unwrap_or_else(|| "memorypilot.ann.usearch".to_string());
        ann_path.set_file_name(file_name);
        match crate::ann::AnnIndex::open(Some(ann_path)) {
            Ok(index) => Some(Arc::new(index)),
            Err(error) => {
                eprintln!("[MemoryPilot] ANN index disabled: {}", error);
                None
            }
        }
    }

    #[allow(dead_code)]
    fn warm_ann_index(&self) -> Result<(), String> {
        let Some(ann) = self.ann.clone() else {
            return Ok(());
        };
        if !ann.is_empty() {
            return Ok(());
        }
        let mut stmt = self
            .conn
            .prepare("SELECT id, embedding FROM memories WHERE embedding IS NOT NULL")
            .map_err(|error| format!("ANN warm prepare: {}", error))?;
        let rows = stmt
            .query_map([], |row| {
                let id: String = row.get(0)?;
                let blob: Vec<u8> = row.get(1)?;
                Ok((id, blob))
            })
            .map_err(|error| format!("ANN warm query: {}", error))?;
        let mut added = 0usize;
        for row in rows.flatten() {
            let (id, blob) = row;
            let vector = crate::embedding::blob_to_vec(&blob);
            if vector.is_empty() {
                continue;
            }
            if ann.add(&id, &vector).is_ok() {
                added += 1;
            }
        }
        if added > 0 {
            let _ = ann.persist();
        }
        Ok(())
    }

    #[allow(dead_code)]
    pub fn ann_index(&self) -> Option<Arc<crate::ann::AnnIndex>> {
        self.ann.clone()
    }

    /// Build the text we feed into the FTS5 `content` column.
    ///
    /// We append a Snowball-stemmed projection of the raw content so that
    /// inflected matches survive BM25 (e.g. French "messages" vs query
    /// "message", or English "running" vs query "run"). The raw content
    /// is preserved verbatim, so exact-phrase, NEAR, and BM25 statistics
    /// on the original text continue to work unchanged. The stemmed
    /// fragment is delimited by an ASCII NUL so it is unlikely to be
    /// matched by any user query phrase.
    fn fts_index_content(content: &str) -> String {
        let stem = crate::stemming::stem_text(content);
        if stem.is_empty() {
            content.to_string()
        } else {
            format!("{} {}", content, stem)
        }
    }

    fn read_conn(&self) -> std::sync::MutexGuard<'_, Connection> {
        for pooled in &self.read_pool {
            if let Ok(guard) = pooled.try_lock() {
                return guard;
            }
        }
        // All busy — wait on first available, handling poison
        for pooled in &self.read_pool {
            if let Ok(guard) = pooled.lock() {
                return guard;
            }
        }
        // Last resort: clear poison on pool[0]
        self.read_pool[0].clear_poison();
        self.read_pool[0].lock().expect("read pool irrecoverable")
    }

    fn canonical_project_name(name: &str) -> Option<String> {
        let trimmed = name.trim();
        if trimmed.is_empty() {
            return None;
        }

        let mut slug = String::new();
        let mut previous_was_separator = false;

        for character in trimmed.chars() {
            if character.is_ascii_alphanumeric() {
                slug.push(character.to_ascii_lowercase());
                previous_was_separator = false;
            } else if !slug.is_empty() && !previous_was_separator {
                slug.push('-');
                previous_was_separator = true;
            }
        }

        while slug.ends_with('-') {
            slug.pop();
        }

        if slug.is_empty() {
            None
        } else {
            Some(slug)
        }
    }

    fn canonical_project(project: Option<&str>) -> Option<String> {
        project.and_then(Self::canonical_project_name)
    }

    fn normalize_path(path: &str) -> String {
        let trimmed = path.trim();
        if trimmed.is_empty() {
            return String::new();
        }

        let canonical = std::fs::canonicalize(trimmed).unwrap_or_else(|_| PathBuf::from(trimmed));
        let rendered = canonical.to_string_lossy().replace('\\', "/");
        if rendered == "/" {
            rendered
        } else {
            rendered.trim_end_matches('/').to_string()
        }
    }

    fn infer_project_root(working_dir: &str) -> String {
        let normalized_dir = Self::normalize_path(working_dir);
        if normalized_dir.is_empty() {
            return normalized_dir;
        }

        let normalized_path = PathBuf::from(&normalized_dir);
        let search_start = if normalized_path.is_file() {
            normalized_path
                .parent()
                .map(Path::to_path_buf)
                .unwrap_or_else(|| normalized_path.clone())
        } else {
            normalized_path.clone()
        };

        for ancestor in search_start.ancestors() {
            if ancestor.join(".git").exists()
                || ancestor.join("Cargo.toml").exists()
                || ancestor.join("package.json").exists()
                || ancestor.join("pyproject.toml").exists()
                || ancestor.join("go.mod").exists()
            {
                return Self::normalize_path(&ancestor.to_string_lossy());
            }
        }

        normalized_dir
    }

    fn infer_project_slug_from_root(project_root: &str) -> Option<String> {
        let root = Path::new(project_root);

        let cargo_manifest = root.join("Cargo.toml");
        if cargo_manifest.exists() {
            if let Ok(content) = std::fs::read_to_string(&cargo_manifest) {
                let mut in_package_section = false;
                for line in content.lines() {
                    let trimmed = line.trim();
                    if trimmed.starts_with('[') {
                        in_package_section = trimmed == "[package]";
                        continue;
                    }
                    if in_package_section && trimmed.starts_with("name") {
                        if let Some((_, raw_value)) = trimmed.split_once('=') {
                            if let Some(slug) =
                                Self::canonical_project_name(raw_value.trim().trim_matches('"'))
                            {
                                return Some(slug);
                            }
                        }
                    }
                }
            }
        }

        let package_manifest = root.join("package.json");
        if package_manifest.exists() {
            if let Ok(content) = std::fs::read_to_string(&package_manifest) {
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
                    if let Some(name) = json.get("name").and_then(|value| value.as_str()) {
                        if let Some(slug) = Self::canonical_project_name(name) {
                            return Some(slug);
                        }
                    }
                }
            }
        }

        root.file_name()
            .and_then(|name| name.to_str())
            .and_then(Self::canonical_project_name)
    }

    fn path_matches(working_dir: &str, project_path: &str) -> bool {
        if project_path.is_empty() {
            return false;
        }

        working_dir == project_path || working_dir.starts_with(&format!("{project_path}/"))
    }

    fn should_include_in_context(memory: &Memory, mode: RecallMode) -> bool {
        mode.includes_credentials() || memory.kind != "credential"
    }

    fn metadata_object(
        metadata: Option<&serde_json::Value>,
    ) -> serde_json::Map<String, serde_json::Value> {
        match metadata.cloned() {
            Some(serde_json::Value::Object(object)) => object,
            Some(other) => {
                let mut object = serde_json::Map::new();
                object.insert("value".into(), other);
                object
            }
            None => serde_json::Map::new(),
        }
    }

    fn merge_metadata(
        base: Option<&serde_json::Value>,
        overlay: Option<&serde_json::Value>,
    ) -> Option<serde_json::Value> {
        let mut object = Self::metadata_object(base);
        for (key, value) in Self::metadata_object(overlay) {
            object.insert(key, value);
        }
        if object.is_empty() {
            None
        } else {
            Some(serde_json::Value::Object(object))
        }
    }

    fn apply_scope_to_metadata(
        metadata: Option<&serde_json::Value>,
        scope: &MemoryScope,
    ) -> Option<serde_json::Value> {
        let mut object = Self::metadata_object(metadata);
        if let Some(session_id) = &scope.session_id {
            object.insert(
                "session_id".into(),
                serde_json::Value::String(session_id.clone()),
            );
        }
        if let Some(thread_id) = &scope.thread_id {
            object.insert(
                "thread_id".into(),
                serde_json::Value::String(thread_id.clone()),
            );
        }
        if let Some(window_id) = &scope.window_id {
            object.insert(
                "window_id".into(),
                serde_json::Value::String(window_id.clone()),
            );
        }
        if object.is_empty() {
            None
        } else {
            Some(serde_json::Value::Object(object))
        }
    }

    fn metadata_scope(metadata: Option<&serde_json::Value>) -> MemoryScope {
        let Some(serde_json::Value::Object(object)) = metadata else {
            return MemoryScope::default();
        };
        MemoryScope {
            session_id: object
                .get("session_id")
                .and_then(|value| value.as_str())
                .map(String::from),
            thread_id: object
                .get("thread_id")
                .and_then(|value| value.as_str())
                .map(String::from),
            window_id: object
                .get("window_id")
                .and_then(|value| value.as_str())
                .map(String::from),
        }
    }

    fn scope_match_score(memory: &Memory, scope: &MemoryScope) -> i32 {
        if scope.is_empty() {
            return 0;
        }

        let memory_scope = Self::metadata_scope(memory.metadata.as_ref());
        let mut score = 0;
        if scope.thread_id.is_some() && scope.thread_id == memory_scope.thread_id {
            score += 4;
        }
        if scope.window_id.is_some() && scope.window_id == memory_scope.window_id {
            score += 2;
        }
        if scope.session_id.is_some() && scope.session_id == memory_scope.session_id {
            score += 1;
        }
        score
    }

    fn list_scope_memories(
        &self,
        project: Option<&str>,
        scope: &MemoryScope,
        limit: usize,
    ) -> Result<Vec<Memory>, String> {
        if scope.is_empty() {
            return Ok(Vec::new());
        }

        let candidate_limit = limit.max(50).min(200);
        let (memories, _) =
            self.list_memories(project, None, Some("transcript"), candidate_limit, 0)?;
        let mut scoped: Vec<(i32, Memory)> = memories
            .into_iter()
            .filter_map(|memory| {
                let score = Self::scope_match_score(&memory, scope);
                if score > 0 {
                    Some((score, memory))
                } else {
                    None
                }
            })
            .collect();

        scoped.sort_by(|left, right| {
            right
                .0
                .cmp(&left.0)
                .then_with(|| right.1.updated_at.cmp(&left.1.updated_at))
        });

        Ok(scoped
            .into_iter()
            .take(limit)
            .map(|(_, memory)| memory)
            .collect())
    }

    fn memory_age_days(memory: &Memory) -> Option<i64> {
        let updated_at = chrono::DateTime::parse_from_rfc3339(&memory.updated_at).ok()?;
        Some((Utc::now() - updated_at.with_timezone(&Utc)).num_days())
    }

    fn memory_haystack(memory: &Memory) -> String {
        format!(
            "{} {} {}",
            memory.content.to_ascii_lowercase(),
            memory.tags.join(" ").to_ascii_lowercase(),
            memory.source.to_ascii_lowercase()
        )
    }

    fn entity_overlap_keys(
        content: &str,
        project: Option<&str>,
    ) -> std::collections::HashSet<String> {
        let mut keys = std::collections::HashSet::new();
        for entity in crate::graph::extract_entities(content, project) {
            keys.insert(entity.value.to_ascii_lowercase());
        }
        keys
    }

    fn global_context_score(
        memory: &Memory,
        hint_terms: &[String],
        hint_entity_keys: &std::collections::HashSet<String>,
        project: Option<&str>,
    ) -> i32 {
        let haystack = Self::memory_haystack(memory);
        let hint_overlap = hint_terms
            .iter()
            .filter(|term| haystack.contains(term.as_str()))
            .count() as i32;
        let entity_overlap =
            crate::graph::extract_entities(&memory.content, memory.project.as_deref())
                .into_iter()
                .filter(|entity| hint_entity_keys.contains(&entity.value.to_ascii_lowercase()))
                .count() as i32;
        let recency_bonus = match Self::memory_age_days(memory) {
            Some(age_days) if age_days <= 14 => 2,
            Some(age_days) if age_days <= 45 => 1,
            Some(age_days) if age_days >= 180 => -1,
            _ => 0,
        };
        let access_bonus = memory.access_count.min(3);
        let kind_bias = match memory.kind.as_str() {
            "preference" => 3,
            "pattern" => 2,
            "decision" => 1,
            _ => 0,
        };
        let project_bonus = project
            .map(|project_name| haystack.contains(project_name) as i32)
            .unwrap_or(0);
        let contextual = project.is_some() || !hint_terms.is_empty();
        let generic_penalty = if contextual && hint_overlap == 0 && entity_overlap == 0 {
            if project.is_some() && hint_terms.len() >= 2 {
                8
            } else {
                5
            }
        } else {
            0
        };

        kind_bias
            + (hint_overlap * 4)
            + (entity_overlap * 3)
            + access_bonus
            + recency_bonus
            + project_bonus
            - generic_penalty
    }

    fn select_global_context_memories(
        memories: Vec<Memory>,
        hint_terms: &[String],
        hint_entity_keys: &std::collections::HashSet<String>,
        project: Option<&str>,
        limit: usize,
    ) -> Vec<Memory> {
        let contextual = project.is_some() || !hint_terms.is_empty();
        let mut ranked: Vec<(i32, Memory)> = memories
            .into_iter()
            .map(|memory| {
                (
                    Self::global_context_score(&memory, hint_terms, hint_entity_keys, project),
                    memory,
                )
            })
            .collect();

        ranked.sort_by(|left, right| {
            right
                .0
                .cmp(&left.0)
                .then_with(|| right.1.importance.cmp(&left.1.importance))
                .then_with(|| right.1.updated_at.cmp(&left.1.updated_at))
        });

        ranked
            .into_iter()
            .filter(|(score, _)| !contextual || *score > 0)
            .take(limit)
            .map(|(_, memory)| memory)
            .collect()
    }

    fn recency_boost(memory: &Memory) -> f64 {
        let age_days = Self::memory_age_days(memory).unwrap_or(365).max(0) as f64;
        ((30.0 - age_days).max(0.0) / 30.0 * 0.25 * 100.0).round() / 100.0
    }

    fn access_boost(memory: &Memory) -> f64 {
        ((memory.access_count.min(10) as f64) / 10.0 * 0.2 * 100.0).round() / 100.0
    }

    fn build_link_boosts(&self) -> std::collections::HashMap<String, f64> {
        self.build_link_boosts_for(&[])
    }

    fn build_link_boosts_for(
        &self,
        candidate_ids: &[&String],
    ) -> std::collections::HashMap<String, f64> {
        let mut link_boosts = std::collections::HashMap::new();
        let mut rows_data: Vec<(String, String)> = Vec::new();

        if candidate_ids.is_empty() || candidate_ids.len() > 200 {
            if let Ok(mut stmt) = self
                .conn
                .prepare("SELECT target_id, relation_type FROM memory_links")
            {
                if let Ok(rows) = stmt.query_map([], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                }) {
                    for r in rows.flatten() {
                        rows_data.push(r);
                    }
                }
            }
        } else {
            let placeholders: Vec<String> = (1..=candidate_ids.len())
                .map(|i| format!("?{}", i))
                .collect();
            let sql = format!(
                "SELECT target_id, relation_type FROM memory_links WHERE target_id IN ({})",
                placeholders.join(",")
            );
            if let Ok(mut stmt) = self.conn.prepare(&sql) {
                let param_refs: Vec<&dyn rusqlite::types::ToSql> = candidate_ids
                    .iter()
                    .map(|id| id as &dyn rusqlite::types::ToSql)
                    .collect();
                if let Ok(rows) = stmt.query_map(param_refs.as_slice(), |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                }) {
                    for r in rows.flatten() {
                        rows_data.push(r);
                    }
                }
            }
        }

        for (target_id, relation_type) in rows_data {
            let boost: f64 = match relation_type.as_str() {
                "deprecates" => -0.6,
                "depends_on" | "implements" | "resolves" | "resolved_by" | "fixed_by" | "fixes" => {
                    0.08
                }
                "shares_topic" => 0.06,
                "same_agent" => 0.05,
                "same_origin" => 0.02,
                _ => 0.03,
            };
            let total = link_boosts.entry(target_id).or_insert(0.0);
            *total = (*total + boost).clamp(-0.8_f64, 0.25_f64);
        }
        link_boosts
    }

    fn get_kg_expansion_terms(&self, query: &str) -> Vec<String> {
        let words: Vec<&str> = query.split_whitespace().collect();
        if words.len() > 15 {
            return Vec::new();
        }

        let mut terms: Vec<String> = Vec::new();
        let query_lower = query.to_lowercase();

        for word in &words {
            if word.len() < 3 {
                continue;
            }
            let lower = word.to_lowercase();

            // KG triples: related subjects/objects
            if let Ok(mut stmt) = self.conn.prepare(
                "SELECT DISTINCT object FROM knowledge_triples WHERE lower(subject) = ?1 AND valid_to IS NULL LIMIT 3"
            ) {
                if let Ok(rows) = stmt.query_map(params![&lower], |r| r.get::<_, String>(0)) {
                    for r in rows.flatten() {
                        if !query_lower.contains(&r.to_lowercase()) && r.len() >= 2 { terms.push(r.to_lowercase()); }
                    }
                }
            }
            if let Ok(mut stmt) = self.conn.prepare(
                "SELECT DISTINCT subject FROM knowledge_triples WHERE lower(object) = ?1 AND valid_to IS NULL LIMIT 3"
            ) {
                if let Ok(rows) = stmt.query_map(params![&lower], |r| r.get::<_, String>(0)) {
                    for r in rows.flatten() {
                        if !query_lower.contains(&r.to_lowercase()) && r.len() >= 2 { terms.push(r.to_lowercase()); }
                    }
                }
            }

            // Entity co-occurrence
            if let Ok(mut stmt) = self.conn.prepare(
                "SELECT DISTINCT b.entity_value FROM memory_entities a \
                 JOIN memory_entities b ON a.memory_id = b.memory_id AND a.entity_value != b.entity_value \
                 WHERE lower(a.entity_value) = ?1 LIMIT 4"
            ) {
                if let Ok(rows) = stmt.query_map(params![lower], |r| r.get::<_, String>(0)) {
                    for r in rows.flatten() {
                        if !query_lower.contains(&r.to_lowercase()) && r.len() >= 2 { terms.push(r.to_lowercase()); }
                    }
                }
            }
        }

        terms.truncate(12);
        terms
    }

    fn build_adjacency_set(&self, ids: &[String]) -> std::collections::HashSet<(String, String)> {
        let mut adj = std::collections::HashSet::new();
        if ids.is_empty() {
            return adj;
        }
        let placeholders: Vec<String> = (1..=ids.len()).map(|i| format!("?{}", i)).collect();
        let sql = format!(
            "SELECT source_id, target_id FROM memory_links WHERE source_id IN ({0}) AND target_id IN ({0})",
            placeholders.join(",")
        );
        if let Ok(mut stmt) = self.conn.prepare(&sql) {
            // Bind each id twice (once for source_id IN, once for target_id IN)
            let mut params: Vec<&dyn rusqlite::types::ToSql> = Vec::with_capacity(ids.len());
            for id in ids {
                params.push(id as &dyn rusqlite::types::ToSql);
            }
            if let Ok(rows) = stmt.query_map(params.as_slice(), |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
            }) {
                for row in rows.flatten() {
                    adj.insert(row);
                }
            }
        }

        // Also connect via shared entities
        let ent_sql = format!(
            "SELECT DISTINCT a.memory_id, b.memory_id FROM memory_entities a \
             JOIN memory_entities b ON a.entity_value = b.entity_value AND a.entity_kind = b.entity_kind AND a.memory_id != b.memory_id \
             WHERE a.memory_id IN ({0}) AND b.memory_id IN ({0})",
            placeholders.join(",")
        );
        if let Ok(mut stmt) = self.conn.prepare(&ent_sql) {
            let mut params: Vec<&dyn rusqlite::types::ToSql> = Vec::with_capacity(ids.len());
            for id in ids {
                params.push(id as &dyn rusqlite::types::ToSql);
            }
            if let Ok(rows) = stmt.query_map(params.as_slice(), |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
            }) {
                for row in rows.flatten() {
                    adj.insert(row);
                }
            }
        }
        adj
    }

    fn batch_triple_counts(
        &self,
        candidate_ids: &[&String],
    ) -> std::collections::HashMap<String, (i64, i64)> {
        let mut counts: std::collections::HashMap<String, (i64, i64)> =
            std::collections::HashMap::new();
        if candidate_ids.is_empty() {
            return counts;
        }
        let placeholders: Vec<String> = (1..=candidate_ids.len())
            .map(|i| format!("?{}", i))
            .collect();
        let sql = format!(
            "SELECT source_memory_id, \
             SUM(CASE WHEN valid_to IS NULL THEN 1 ELSE 0 END), \
             SUM(CASE WHEN valid_to IS NOT NULL THEN 1 ELSE 0 END) \
             FROM knowledge_triples WHERE source_memory_id IN ({}) GROUP BY source_memory_id",
            placeholders.join(",")
        );
        if let Ok(mut stmt) = self.conn.prepare(&sql) {
            let param_refs: Vec<&dyn rusqlite::types::ToSql> = candidate_ids
                .iter()
                .map(|id| id as &dyn rusqlite::types::ToSql)
                .collect();
            if let Ok(rows) = stmt.query_map(param_refs.as_slice(), |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, i64>(1)?,
                    r.get::<_, i64>(2)?,
                ))
            }) {
                for row in rows.flatten() {
                    counts.insert(row.0, (row.1, row.2));
                }
            }
        }
        counts
    }

    fn preview_snippet(content: &str) -> String {
        let trimmed = content.trim().replace('\n', " ");
        if trimmed.len() <= 120 {
            trimmed
        } else {
            format!("{}...", trimmed.chars().take(117).collect::<String>())
        }
    }

    fn hygiene_report(&self) -> crate::gc::HygieneReport {
        let stale_threshold = (Utc::now() - chrono::Duration::days(30)).to_rfc3339();
        crate::gc::HygieneReport {
            projects_missing_path: self.conn.query_row(
                "SELECT COUNT(*) FROM projects WHERE trim(path) = ''",
                [],
                |row| row.get(0),
            ).unwrap_or(0),
            memory_project_mismatches: self.conn.query_row(
                "SELECT COUNT(*) FROM memories m WHERE m.project IS NOT NULL AND NOT EXISTS (SELECT 1 FROM projects p WHERE p.name = m.project)",
                [],
                |row| row.get(0),
            ).unwrap_or(0),
            never_accessed_memories: self.conn.query_row(
                "SELECT COUNT(*) FROM memories WHERE access_count = 0",
                [],
                |row| row.get(0),
            ).unwrap_or(0),
            stale_low_value_memories: self.conn.query_row(
                "SELECT COUNT(*) FROM memories WHERE importance <= 2 AND access_count = 0 AND updated_at < ?1",
                params![stale_threshold],
                |row| row.get(0),
            ).unwrap_or(0),
            orphan_entities: self.conn.query_row(
                "SELECT COUNT(*) FROM memory_entities WHERE memory_id NOT IN (SELECT id FROM memories)",
                [],
                |row| row.get(0),
            ).unwrap_or(0),
            orphan_links: self.conn.query_row(
                "SELECT COUNT(*) FROM memory_links WHERE source_id NOT IN (SELECT id FROM memories) OR target_id NOT IN (SELECT id FROM memories)",
                [],
                |row| row.get(0),
            ).unwrap_or(0),
            credential_memories: self.conn.query_row(
                "SELECT COUNT(*) FROM memories WHERE kind = 'credential'",
                [],
                |row| row.get(0),
            ).unwrap_or(0),
            global_memories: self.conn.query_row(
                "SELECT COUNT(*) FROM memories WHERE project IS NULL",
                [],
                |row| row.get(0),
            ).unwrap_or(0),
        }
    }

    fn recall_explanation(
        &self,
        memory: &Memory,
        selection_source: &str,
        project: Option<&str>,
        mode: RecallMode,
        search_score: Option<f64>,
        link_boosts: &std::collections::HashMap<String, f64>,
    ) -> serde_json::Value {
        let graph_boost = link_boosts.get(&memory.id).copied().unwrap_or(0.0);
        serde_json::json!({
            "id": memory.id,
            "kind": memory.kind,
            "project": memory.project,
            "selection_source": selection_source,
            "importance": memory.importance,
            "access_count": memory.access_count,
            "updated_at": memory.updated_at,
            "search_score": search_score,
            "reason": {
                "mode": mode.as_str(),
                "project_match": project.is_some() && memory.project.as_deref() == project,
                "importance_weight": ((memory.importance as f64 / 3.0) * 100.0).round() / 100.0,
                "recency_boost": Self::recency_boost(memory),
                "access_boost": Self::access_boost(memory),
                "graph_boost": (graph_boost * 100.0).round() / 100.0,
                "age_days": Self::memory_age_days(memory),
            }
        })
    }

    fn remember_project_path_if_known(&self, project_name: &str, path: &str) -> Result<(), String> {
        if project_name.is_empty() || path.is_empty() {
            return Ok(());
        }

        let known_project: bool = self
            .conn
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM projects WHERE name = ?1)",
                params![project_name],
                |row| row.get(0),
            )
            .unwrap_or(false);
        let known_memories: bool = self
            .conn
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM memories WHERE project = ?1)",
                params![project_name],
                |row| row.get(0),
            )
            .unwrap_or(false);

        if !known_project && !known_memories {
            return Ok(());
        }

        let now = Utc::now().to_rfc3339();
        self.conn
            .execute(
                "INSERT INTO projects (name, path, created_at) VALUES (?1, ?2, ?3)
                 ON CONFLICT(name) DO UPDATE SET
                    path = CASE
                        WHEN projects.path = '' AND excluded.path != '' THEN excluded.path
                        ELSE projects.path
                    END",
                params![project_name, path, now],
            )
            .map_err(|error| format!("Remember path: {}", error))?;
        Ok(())
    }

    fn normalize_project_identities(&self) -> Result<(), String> {
        #[derive(Clone)]
        struct CanonicalProjectRecord {
            path: String,
            description: Option<String>,
            created_at: String,
        }

        let now = Utc::now().to_rfc3339();
        let mut canonical_projects: std::collections::BTreeMap<String, CanonicalProjectRecord> =
            std::collections::BTreeMap::new();
        let mut project_names_to_delete: std::collections::BTreeSet<String> =
            std::collections::BTreeSet::new();

        {
            let mut statement = self
                .conn
                .prepare("SELECT name, path, description, created_at FROM projects")
                .map_err(|error| format!("Normalize projects: {}", error))?;
            let rows = statement
                .query_map([], |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, Option<String>>(2)?,
                        row.get::<_, String>(3)?,
                    ))
                })
                .map_err(|error| format!("Normalize projects: {}", error))?;

            for row in rows.flatten() {
                let (name, path, description, created_at) = row;
                match Self::canonical_project_name(&name) {
                    Some(slug) => {
                        let normalized_path = Self::normalize_path(&path);
                        let entry = canonical_projects.entry(slug.clone()).or_insert_with(|| {
                            CanonicalProjectRecord {
                                path: normalized_path.clone(),
                                description: description
                                    .clone()
                                    .filter(|value| !value.trim().is_empty()),
                                created_at: created_at.clone(),
                            }
                        });

                        if entry.path.is_empty() && !normalized_path.is_empty() {
                            entry.path = normalized_path.clone();
                        }
                        if entry.description.is_none() {
                            entry.description =
                                description.clone().filter(|value| !value.trim().is_empty());
                        }
                        if created_at < entry.created_at {
                            entry.created_at = created_at.clone();
                        }

                        if slug != name || normalized_path != path {
                            project_names_to_delete.insert(name);
                        }
                    }
                    None => {
                        project_names_to_delete.insert(name);
                    }
                }
            }
        }

        let mut distinct_memory_projects: Vec<String> = Vec::new();
        {
            let mut statement = self
                .conn
                .prepare("SELECT DISTINCT project FROM memories WHERE project IS NOT NULL AND trim(project) != ''")
                .map_err(|error| format!("Normalize memory projects: {}", error))?;
            let rows = statement
                .query_map([], |row| row.get::<_, String>(0))
                .map_err(|error| format!("Normalize memory projects: {}", error))?;
            for row in rows.flatten() {
                distinct_memory_projects.push(row);
            }
        }

        for raw_project in &distinct_memory_projects {
            if let Some(slug) = Self::canonical_project_name(raw_project) {
                canonical_projects
                    .entry(slug.clone())
                    .or_insert_with(|| CanonicalProjectRecord {
                        path: String::new(),
                        description: None,
                        created_at: now.clone(),
                    });
                if slug != *raw_project {
                    project_names_to_delete.insert(raw_project.clone());
                }
            }
        }

        if canonical_projects.is_empty() && project_names_to_delete.is_empty() {
            return Ok(());
        }

        let transaction = self
            .conn
            .unchecked_transaction()
            .map_err(|error| format!("Normalize transaction: {}", error))?;

        for raw_project in &distinct_memory_projects {
            match Self::canonical_project_name(raw_project) {
                Some(slug) if slug != *raw_project => {
                    transaction
                        .execute(
                            "UPDATE memories SET project = ?1 WHERE project = ?2",
                            params![slug, raw_project],
                        )
                        .map_err(|error| format!("Normalize memories: {}", error))?;
                    transaction
                        .execute(
                            "UPDATE memories_fts SET project = ?1 WHERE project = ?2",
                            params![slug, raw_project],
                        )
                        .map_err(|error| format!("Normalize memories_fts: {}", error))?;
                }
                Some(_) => {}
                None => {
                    transaction
                        .execute(
                            "UPDATE memories SET project = NULL WHERE project = ?1",
                            params![raw_project],
                        )
                        .map_err(|error| format!("Nullify memories project: {}", error))?;
                    transaction
                        .execute(
                            "UPDATE memories_fts SET project = '' WHERE project = ?1",
                            params![raw_project],
                        )
                        .map_err(|error| format!("Nullify memories_fts project: {}", error))?;
                }
            }
        }

        for (slug, record) in &canonical_projects {
            transaction
                .execute(
                    "INSERT INTO projects (name, path, description, created_at) VALUES (?1, ?2, ?3, ?4)
                     ON CONFLICT(name) DO UPDATE SET
                        path = CASE
                            WHEN excluded.path != '' THEN excluded.path
                            ELSE projects.path
                        END,
                        description = COALESCE(projects.description, excluded.description)",
                    params![slug, record.path, record.description, record.created_at],
                )
                .map_err(|error| format!("Upsert canonical project: {}", error))?;
        }

        for legacy_name in project_names_to_delete {
            if Self::canonical_project_name(&legacy_name).as_deref() != Some(legacy_name.as_str()) {
                transaction
                    .execute("DELETE FROM projects WHERE name = ?1", params![legacy_name])
                    .map_err(|error| format!("Delete legacy project: {}", error))?;
            }
        }

        transaction
            .commit()
            .map_err(|error| format!("Normalize commit: {}", error))?;
        Ok(())
    }
    fn init_schema(&self) -> Result<(), String> {
        self.conn.execute_batch("
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                kind TEXT NOT NULL DEFAULT 'fact',
                project TEXT,
                tags TEXT NOT NULL DEFAULT '[]',
                source TEXT NOT NULL DEFAULT 'cursor',
                importance INTEGER NOT NULL DEFAULT 3,
                expires_at TEXT,
                metadata TEXT,
                embedding BLOB,
                content_hash TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_accessed_at TEXT,
                access_count INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS memory_links (
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relation_type TEXT NOT NULL DEFAULT 'relates_to',
                valid_from TEXT,
                valid_to TEXT,
                confidence REAL DEFAULT 1.0,
                created_at TEXT NOT NULL,
                PRIMARY KEY (source_id, target_id),
                FOREIGN KEY (source_id) REFERENCES memories(id) ON DELETE CASCADE,
                FOREIGN KEY (target_id) REFERENCES memories(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_links_source ON memory_links(source_id);
            CREATE INDEX IF NOT EXISTS idx_links_target ON memory_links(target_id);

            CREATE TABLE IF NOT EXISTS memory_entities (
                memory_id TEXT NOT NULL,
                entity_kind TEXT NOT NULL,
                entity_value TEXT NOT NULL,
                valid_from TEXT,
                valid_to TEXT,
                FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_entities_value ON memory_entities(entity_value);
            CREATE INDEX IF NOT EXISTS idx_entities_memory ON memory_entities(memory_id);

            CREATE TABLE IF NOT EXISTS knowledge_triples (
                id TEXT PRIMARY KEY,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                valid_from TEXT,
                valid_to TEXT,
                confidence REAL DEFAULT 1.0,
                source_memory_id TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (source_memory_id) REFERENCES memories(id) ON DELETE SET NULL
            );
            CREATE INDEX IF NOT EXISTS idx_triples_subject ON knowledge_triples(subject);
            CREATE INDEX IF NOT EXISTS idx_triples_object ON knowledge_triples(object);
            CREATE INDEX IF NOT EXISTS idx_triples_valid ON knowledge_triples(valid_from, valid_to);

            CREATE INDEX IF NOT EXISTS idx_memories_project ON memories(project);
            CREATE INDEX IF NOT EXISTS idx_memories_kind ON memories(kind);
            CREATE INDEX IF NOT EXISTS idx_memories_updated ON memories(updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_memories_expires ON memories(expires_at) WHERE expires_at IS NOT NULL;

            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                content, tags, kind, project,
                content_rowid='rowid',
                tokenize='unicode61 remove_diacritics 2'
            );

            CREATE TABLE IF NOT EXISTS projects (
                name TEXT PRIMARY KEY,
                path TEXT NOT NULL DEFAULT '',
                description TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
        ").map_err(|e| format!("Schema: {}", e))
    }
    /// Upgrade schema for existing databases (add new columns if missing).
    fn upgrade_schema(&self) -> Result<(), String> {
        // Check if importance column exists
        let has_importance: bool = self
            .conn
            .prepare("SELECT importance FROM memories LIMIT 0")
            .is_ok();
        if !has_importance {
            let _ = self.conn.execute_batch(
                "ALTER TABLE memories ADD COLUMN importance INTEGER NOT NULL DEFAULT 3;
                 ALTER TABLE memories ADD COLUMN expires_at TEXT;",
            );
        }
        // v3.0 columns
        let has_embedding: bool = self
            .conn
            .prepare("SELECT embedding FROM memories LIMIT 0")
            .is_ok();
        if !has_embedding {
            let _ = self.conn.execute_batch(
                "ALTER TABLE memories ADD COLUMN embedding BLOB;
                 ALTER TABLE memories ADD COLUMN last_accessed_at TEXT;
                 ALTER TABLE memories ADD COLUMN access_count INTEGER NOT NULL DEFAULT 0;",
            );
        }
        // v4.0: knowledge_triples + temporal columns
        let _ = self.conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS memory_links (
                 source_id TEXT NOT NULL,
                 target_id TEXT NOT NULL,
                 relation_type TEXT NOT NULL DEFAULT 'relates_to',
                 valid_from TEXT,
                 valid_to TEXT,
                 confidence REAL DEFAULT 1.0,
                 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                 PRIMARY KEY (source_id, target_id),
                 FOREIGN KEY (source_id) REFERENCES memories(id) ON DELETE CASCADE,
                 FOREIGN KEY (target_id) REFERENCES memories(id) ON DELETE CASCADE
             );
             CREATE INDEX IF NOT EXISTS idx_links_source ON memory_links(source_id);
             CREATE INDEX IF NOT EXISTS idx_links_target ON memory_links(target_id);
             CREATE TABLE IF NOT EXISTS memory_entities (
                 memory_id TEXT NOT NULL,
                 entity_kind TEXT NOT NULL,
                 entity_value TEXT NOT NULL,
                 valid_from TEXT,
                 valid_to TEXT,
                 FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
             );
             CREATE INDEX IF NOT EXISTS idx_entities_value ON memory_entities(entity_value);
             CREATE INDEX IF NOT EXISTS idx_entities_memory ON memory_entities(memory_id);
             CREATE TABLE IF NOT EXISTS knowledge_triples (
                 id TEXT PRIMARY KEY,
                 subject TEXT NOT NULL,
                 predicate TEXT NOT NULL,
                 object TEXT NOT NULL,
                 valid_from TEXT,
                 valid_to TEXT,
                 confidence REAL DEFAULT 1.0,
                 source_memory_id TEXT,
                 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                 FOREIGN KEY (source_memory_id) REFERENCES memories(id) ON DELETE SET NULL
             );
             CREATE INDEX IF NOT EXISTS idx_triples_subject ON knowledge_triples(subject);
             CREATE INDEX IF NOT EXISTS idx_triples_object ON knowledge_triples(object);
             CREATE INDEX IF NOT EXISTS idx_triples_valid ON knowledge_triples(valid_from, valid_to);"
        );
        // v4.0.1: content_hash column
        let has_content_hash: bool = self
            .conn
            .prepare("SELECT content_hash FROM memories LIMIT 0")
            .is_ok();
        if !has_content_hash {
            let _ = self
                .conn
                .execute_batch("ALTER TABLE memories ADD COLUMN content_hash TEXT;");
        }
        Ok(())
    }

    // ─── DEDUP ────────────────────────────────────────

    /// Normalize text for comparison: lowercase, collapse whitespace, strip punctuation.
    fn normalize(text: &str) -> String {
        text.to_lowercase()
            .chars()
            .map(|c| {
                if c.is_alphanumeric() || c == ' ' {
                    c
                } else {
                    ' '
                }
            })
            .collect::<String>()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Jaccard similarity between two normalized strings (word-level).
    fn similarity(a: &str, b: &str) -> f64 {
        let a_words: std::collections::HashSet<&str> = a.split_whitespace().collect();
        let b_words: std::collections::HashSet<&str> = b.split_whitespace().collect();
        if a_words.is_empty() && b_words.is_empty() {
            return 1.0;
        }
        let intersection = a_words.intersection(&b_words).count() as f64;
        let union = a_words.union(&b_words).count() as f64;
        if union == 0.0 {
            0.0
        } else {
            intersection / union
        }
    }
    /// Find a near-duplicate in the same project/scope.
    fn find_duplicate(
        &self,
        content: &str,
        project: Option<&str>,
    ) -> Result<Option<Memory>, String> {
        // Fast path: exact content match via hash
        let hash = content_hash(content);
        let exact = if let Some(p) = project {
            self.conn.prepare("SELECT id,content,kind,project,tags,source,importance,expires_at,metadata,created_at,updated_at,last_accessed_at,access_count FROM memories WHERE content_hash=?1 AND project=?2 LIMIT 1")
                .ok().and_then(|mut s| s.query_row(params![&hash, p], |r| Ok(row_to_memory(r))).ok())
        } else {
            self.conn.prepare("SELECT id,content,kind,project,tags,source,importance,expires_at,metadata,created_at,updated_at,last_accessed_at,access_count FROM memories WHERE content_hash=?1 AND project IS NULL LIMIT 1")
                .ok().and_then(|mut s| s.query_row(params![&hash], |r| Ok(row_to_memory(r))).ok())
        };
        if let Some(mem) = exact {
            return Ok(Some(mem));
        }

        // Slow path: Jaccard fuzzy match on recent memories
        let norm = Self::normalize(content);
        let memories: Vec<Memory> = if let Some(p) = project {
            let mut stmt = self.conn.prepare(
                "SELECT id,content,kind,project,tags,source,importance,expires_at,metadata,created_at,updated_at,last_accessed_at,access_count FROM memories WHERE project=?1 ORDER BY updated_at DESC LIMIT 200"
            ).map_err(|e| format!("Dedup: {}", e))?;
            let rows = stmt
                .query_map(params![p], |r| Ok(row_to_memory(r)))
                .map_err(|e| format!("Dedup: {}", e))?;
            rows.flatten().collect()
        } else {
            let mut stmt = self.conn.prepare(
                "SELECT id,content,kind,project,tags,source,importance,expires_at,metadata,created_at,updated_at,last_accessed_at,access_count FROM memories WHERE project IS NULL ORDER BY updated_at DESC LIMIT 200"
            ).map_err(|e| format!("Dedup: {}", e))?;
            let rows = stmt
                .query_map([], |r| Ok(row_to_memory(r)))
                .map_err(|e| format!("Dedup: {}", e))?;
            rows.flatten().collect()
        };
        for mem in memories {
            let mem_norm = Self::normalize(&mem.content);
            if Self::similarity(&norm, &mem_norm) >= DEDUP_THRESHOLD {
                return Ok(Some(mem));
            }
        }
        Ok(None)
    }
    // ─── KNOWLEDGE GRAPH ──────────────────────────────

    pub fn rebuild_links(&self, memory: &Memory) -> Result<(), String> {
        let entities = crate::graph::extract_entities(&memory.content, memory.project.as_deref());

        // 1. Update entities table
        let _ = self.conn.execute(
            "DELETE FROM memory_entities WHERE memory_id = ?1",
            params![memory.id],
        );
        for entity in &entities {
            let _ = self.conn.execute(
                "INSERT OR IGNORE INTO memory_entities (memory_id, entity_kind, entity_value) VALUES (?1, ?2, ?3)",
                params![memory.id, entity.kind, entity.value],
            );
        }

        // 2. Find related memories via shared entities
        let mut target_ids = std::collections::HashSet::new();
        for entity in &entities {
            if !crate::graph::is_reliable_link_entity(entity) {
                continue;
            }

            let relation_hint = crate::graph::relation_for_entity_kind(entity.kind);
            let limit = match entity.kind {
                "topic" => 8,
                "agent" => 6,
                "platform" | "origin" => 4,
                _ => 10,
            };
            let cross_project_topic = entity.kind == "topic";

            if let Some(project_name) = memory.project.as_deref() {
                if let Ok(mut stmt) = self.conn.prepare(
                    "SELECT DISTINCT m.id, m.kind FROM memory_entities e
                     JOIN memories m ON e.memory_id = m.id
                     WHERE e.entity_kind = ?1
                       AND e.entity_value = ?2
                       AND e.memory_id != ?3
                       AND (?4 = 1 OR m.project = ?5 OR m.project IS NULL)
                     LIMIT ?6",
                ) {
                    let allow_cross_project = if cross_project_topic { 1 } else { 0 };
                    if let Ok(rows) = stmt.query_map(
                        params![
                            entity.kind,
                            entity.value,
                            memory.id,
                            allow_cross_project,
                            project_name,
                            limit
                        ],
                        |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
                    ) {
                        for row in rows.flatten() {
                            target_ids.insert((row.0, row.1, relation_hint.to_string()));
                        }
                    }
                }
            } else {
                if let Ok(mut stmt) = self.conn.prepare(
                    "SELECT DISTINCT m.id, m.kind FROM memory_entities e
                     JOIN memories m ON e.memory_id = m.id
                     WHERE e.entity_kind = ?1
                       AND e.entity_value = ?2
                       AND e.memory_id != ?3
                       AND m.project IS NULL
                     LIMIT ?4",
                ) {
                    if let Ok(rows) = stmt.query_map(
                        params![entity.kind, entity.value, memory.id, limit],
                        |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
                    ) {
                        for row in rows.flatten() {
                            target_ids.insert((row.0, row.1, relation_hint.to_string()));
                        }
                    }
                }
            }
        }

        let _ = self.conn.execute(
            "DELETE FROM memory_links WHERE source_id = ?1 OR target_id = ?1",
            params![memory.id],
        );

        let created_at = Utc::now().to_rfc3339();
        for (target_id, target_kind, relation_hint) in target_ids {
            let rel = if relation_hint == "relates_to" {
                crate::graph::infer_relation(&memory.kind, &target_kind)
            } else {
                relation_hint.as_str()
            };
            let _ = self.conn.execute(
                "INSERT OR IGNORE INTO memory_links (source_id, target_id, relation_type, created_at) VALUES (?1, ?2, ?3, ?4)",
                params![memory.id, target_id, rel, &created_at]
            );
            let rev_rel = if relation_hint == "relates_to" {
                crate::graph::infer_relation(&target_kind, &memory.kind)
            } else {
                relation_hint.as_str()
            };
            let _ = self.conn.execute(
                "INSERT OR IGNORE INTO memory_links (source_id, target_id, relation_type, created_at) VALUES (?1, ?2, ?3, ?4)",
                params![target_id, memory.id, rev_rel, &created_at]
            );
        }
        Ok(())
    }

    // ─── KNOWLEDGE TRIPLES ─────────────────────────────

    pub fn add_triple(
        &self,
        subject: &str,
        predicate: &str,
        object: &str,
        valid_from: Option<&str>,
        valid_to: Option<&str>,
        confidence: Option<f64>,
        source_memory_id: Option<&str>,
    ) -> Result<serde_json::Value, String> {
        let sub = subject.to_lowercase().replace(' ', "_");
        let pred = predicate.to_lowercase().replace(' ', "_");
        let obj = object.to_lowercase().replace(' ', "_");

        let existing: Option<String> = self.conn.prepare(
            "SELECT id FROM knowledge_triples WHERE subject=?1 AND predicate=?2 AND object=?3 AND valid_to IS NULL"
        ).ok().and_then(|mut s| s.query_row(params![&sub, &pred, &obj], |r| r.get(0)).ok());

        if let Some(id) = existing {
            return Ok(
                serde_json::json!({"triple_id": id, "already_exists": true, "fact": format!("{} -> {} -> {}", subject, predicate, object)}),
            );
        }

        let id = format!(
            "t_{}_{}_{}_{}",
            &sub,
            &pred,
            &obj,
            &Uuid::new_v4().to_string()[..8]
        );
        let conf = confidence.unwrap_or(1.0);
        let now = Utc::now().to_rfc3339();
        self.conn.execute(
            "INSERT INTO knowledge_triples (id, subject, predicate, object, valid_from, valid_to, confidence, source_memory_id, created_at) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9)",
            params![&id, &sub, &pred, &obj, valid_from, valid_to, conf, source_memory_id, &now]
        ).map_err(|e| format!("add_triple: {}", e))?;
        Ok(
            serde_json::json!({"triple_id": id, "fact": format!("{} -> {} -> {}", subject, predicate, object)}),
        )
    }

    pub fn invalidate_triple(
        &self,
        subject: &str,
        predicate: &str,
        object: &str,
        ended: Option<&str>,
    ) -> Result<serde_json::Value, String> {
        let sub = subject.to_lowercase().replace(' ', "_");
        let pred = predicate.to_lowercase().replace(' ', "_");
        let obj = object.to_lowercase().replace(' ', "_");
        let end_date = ended
            .unwrap_or(&Utc::now().format("%Y-%m-%d").to_string())
            .to_string();
        let changed = self.conn.execute(
            "UPDATE knowledge_triples SET valid_to=?1 WHERE subject=?2 AND predicate=?3 AND object=?4 AND valid_to IS NULL",
            params![&end_date, &sub, &pred, &obj]
        ).map_err(|e| format!("invalidate_triple: {}", e))?;
        Ok(
            serde_json::json!({"invalidated": changed, "fact": format!("{} -> {} -> {}", subject, predicate, object), "ended": end_date}),
        )
    }

    pub fn query_kg_entity(
        &self,
        name: &str,
        as_of: Option<&str>,
        direction: &str,
    ) -> Result<serde_json::Value, String> {
        let eid = name.to_lowercase().replace(' ', "_");
        let mut facts = Vec::new();

        if direction == "outgoing" || direction == "both" {
            let mut sql = "SELECT subject, predicate, object, valid_from, valid_to, confidence, source_memory_id FROM knowledge_triples WHERE subject = ?1".to_string();
            let mut param_vals: Vec<Box<dyn rusqlite::types::ToSql>> = vec![Box::new(eid.clone())];
            if let Some(date) = as_of {
                sql += &format!(" AND (valid_from IS NULL OR valid_from <= ?{n}) AND (valid_to IS NULL OR valid_to >= ?{n})", n=param_vals.len()+1);
                param_vals.push(Box::new(date.to_string()));
            }
            if let Ok(mut stmt) = self.conn.prepare(&sql) {
                let refs: Vec<&dyn rusqlite::types::ToSql> =
                    param_vals.iter().map(|p| p.as_ref()).collect();
                if let Ok(rows) = stmt.query_map(refs.as_slice(), |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, String>(2)?,
                        r.get::<_, Option<String>>(3)?,
                        r.get::<_, Option<String>>(4)?,
                        r.get::<_, f64>(5)?,
                    ))
                }) {
                    for row in rows.flatten() {
                        facts.push(serde_json::json!({"direction":"outgoing","subject":row.0,"predicate":row.1,"object":row.2,"valid_from":row.3,"valid_to":row.4,"confidence":row.5,"current":row.4.is_none()}));
                    }
                }
            }
        }
        if direction == "incoming" || direction == "both" {
            let mut sql = "SELECT subject, predicate, object, valid_from, valid_to, confidence FROM knowledge_triples WHERE object = ?1".to_string();
            let mut param_vals: Vec<Box<dyn rusqlite::types::ToSql>> = vec![Box::new(eid.clone())];
            if let Some(date) = as_of {
                sql += &format!(" AND (valid_from IS NULL OR valid_from <= ?{n}) AND (valid_to IS NULL OR valid_to >= ?{n})", n=param_vals.len()+1);
                param_vals.push(Box::new(date.to_string()));
            }
            if let Ok(mut stmt) = self.conn.prepare(&sql) {
                let refs: Vec<&dyn rusqlite::types::ToSql> =
                    param_vals.iter().map(|p| p.as_ref()).collect();
                if let Ok(rows) = stmt.query_map(refs.as_slice(), |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, String>(2)?,
                        r.get::<_, Option<String>>(3)?,
                        r.get::<_, Option<String>>(4)?,
                        r.get::<_, f64>(5)?,
                    ))
                }) {
                    for row in rows.flatten() {
                        facts.push(serde_json::json!({"direction":"incoming","subject":row.0,"predicate":row.1,"object":row.2,"valid_from":row.3,"valid_to":row.4,"confidence":row.5,"current":row.4.is_none()}));
                    }
                }
            }
        }
        Ok(
            serde_json::json!({"entity": name, "as_of": as_of, "facts": facts, "count": facts.len()}),
        )
    }

    pub fn kg_timeline(&self, entity: Option<&str>) -> Result<serde_json::Value, String> {
        let mut results = Vec::new();
        let sql = if let Some(name) = entity {
            let eid = name.to_lowercase().replace(' ', "_");
            let mut stmt = self.conn.prepare(
                "SELECT subject, predicate, object, valid_from, valid_to, confidence FROM knowledge_triples WHERE subject = ?1 OR object = ?1 ORDER BY valid_from ASC NULLS LAST"
            ).map_err(|e| format!("kg_timeline: {}", e))?;
            let rows = stmt
                .query_map(params![&eid], |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, String>(2)?,
                        r.get::<_, Option<String>>(3)?,
                        r.get::<_, Option<String>>(4)?,
                    ))
                })
                .map_err(|e| format!("kg_timeline: {}", e))?;
            for r in rows.flatten() {
                results.push(serde_json::json!({"subject":r.0,"predicate":r.1,"object":r.2,"valid_from":r.3,"valid_to":r.4,"current":r.4.is_none()}));
            }
            return Ok(
                serde_json::json!({"entity": name, "timeline": results, "count": results.len()}),
            );
        } else {
            "SELECT subject, predicate, object, valid_from, valid_to FROM knowledge_triples ORDER BY valid_from ASC NULLS LAST LIMIT 100"
        };
        if entity.is_none() {
            let mut stmt = self
                .conn
                .prepare(sql)
                .map_err(|e| format!("kg_timeline: {}", e))?;
            let rows = stmt
                .query_map([], |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, String>(2)?,
                        r.get::<_, Option<String>>(3)?,
                        r.get::<_, Option<String>>(4)?,
                    ))
                })
                .map_err(|e| format!("kg_timeline: {}", e))?;
            for r in rows.flatten() {
                results.push(serde_json::json!({"subject":r.0,"predicate":r.1,"object":r.2,"valid_from":r.3,"valid_to":r.4,"current":r.4.is_none()}));
            }
        }
        Ok(serde_json::json!({"entity": "all", "timeline": results, "count": results.len()}))
    }

    pub fn kg_stats(&self) -> Result<serde_json::Value, String> {
        let total: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM knowledge_triples", [], |r| r.get(0))
            .unwrap_or(0);
        let current: i64 = self
            .conn
            .query_row(
                "SELECT COUNT(*) FROM knowledge_triples WHERE valid_to IS NULL",
                [],
                |r| r.get(0),
            )
            .unwrap_or(0);
        let expired = total - current;
        let mut predicates = Vec::new();
        if let Ok(mut stmt) = self
            .conn
            .prepare("SELECT DISTINCT predicate FROM knowledge_triples ORDER BY predicate")
        {
            if let Ok(rows) = stmt.query_map([], |r| r.get::<_, String>(0)) {
                for p in rows.flatten() {
                    predicates.push(p);
                }
            }
        }
        let mut entities = std::collections::HashSet::new();
        if let Ok(mut stmt) = self.conn.prepare("SELECT DISTINCT subject FROM knowledge_triples UNION SELECT DISTINCT object FROM knowledge_triples") {
            if let Ok(rows) = stmt.query_map([], |r| r.get::<_, String>(0)) {
                for e in rows.flatten() { entities.insert(e); }
            }
        }
        Ok(serde_json::json!({
            "entities": entities.len(),
            "triples": total,
            "current_facts": current,
            "expired_facts": expired,
            "relationship_types": predicates,
        }))
    }

    // ─── CRUD ────────────────────────────────────────

    /// Add memory with dedup check. Returns (memory, was_merged).
    pub fn add_memory(
        &self,
        content: &str,
        kind: &str,
        project: Option<&str>,
        tags: &[String],
        source: &str,
        importance: i32,
        expires_at: Option<&str>,
        metadata: Option<&serde_json::Value>,
        scope: &MemoryScope,
    ) -> Result<(Memory, bool), String> {
        self.add_memory_with_id(
            None, content, kind, project, tags, source, importance, expires_at, metadata, scope,
        )
    }

    /// Same as [`Self::add_memory`] but lets the caller pin the row's
    /// primary key. Used by benchmarks (and only by benchmarks) so that
    /// the memory id is reproducible across runs — without it, the
    /// random UUID drives the deterministic id-based tie-break we now
    /// apply in `search`, which produced visible run-to-run variance
    /// on `--benchmark-fr`.
    #[allow(clippy::too_many_arguments)]
    pub fn add_memory_with_id(
        &self,
        explicit_id: Option<&str>,
        content: &str,
        kind: &str,
        project: Option<&str>,
        tags: &[String],
        source: &str,
        importance: i32,
        expires_at: Option<&str>,
        metadata: Option<&serde_json::Value>,
        scope: &MemoryScope,
    ) -> Result<(Memory, bool), String> {
        let canonical_project = Self::canonical_project(project);
        let scoped_metadata = Self::apply_scope_to_metadata(metadata, scope);
        // Check for near-duplicate
        if let Some(existing) = self.find_duplicate(content, canonical_project.as_deref())? {
            // Merge: update content if newer is longer, bump updated_at
            let new_content = if content.len() > existing.content.len() {
                content
            } else {
                &existing.content
            };
            let new_importance = importance.max(existing.importance);
            let mut merged_tags: Vec<String> = existing.tags.clone();
            for t in tags {
                if !merged_tags.contains(t) {
                    merged_tags.push(t.clone());
                }
            }
            let merged_metadata =
                Self::merge_metadata(existing.metadata.as_ref(), scoped_metadata.as_ref());
            let updated = self.update_memory_full(
                &existing.id,
                Some(new_content),
                None,
                Some(&merged_tags),
                Some(new_importance),
                expires_at,
                merged_metadata.as_ref(),
            )?;
            return Ok((updated.unwrap_or(existing), true));
        }

        let id = explicit_id
            .map(|value| value.to_string())
            .unwrap_or_else(|| Uuid::new_v4().to_string());
        let now = Utc::now().to_rfc3339();
        let tags_json = serde_json::to_string(tags).unwrap_or_else(|_| "[]".into());
        let meta_json = scoped_metadata
            .as_ref()
            .map(|m| serde_json::to_string(m).unwrap_or_default());
        let imp = importance.clamp(1, 5);
        let hash = content_hash(content);

        self.conn.execute(
            "INSERT INTO memories (id,content,kind,project,tags,source,importance,expires_at,metadata,embedding,content_hash,created_at,updated_at,access_count)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,NULL,?10,?11,?12,0)",
            params![id, content, kind, canonical_project.as_deref(), tags_json, source, imp, expires_at, meta_json, &hash, now, now],
        ).map_err(|e| format!("Insert: {}", e))?;

        // Queue embedding for background computation
        queue_embedding_job(&id, content);

        // FTS index (raw content + stemmed projection for FR/EN recall)
        let rowid = self.conn.last_insert_rowid();
        let fts_content = Self::fts_index_content(content);
        self.conn.execute(
            "INSERT INTO memories_fts (rowid,content,tags,kind,project) VALUES (?1,?2,?3,?4,?5)",
            params![rowid, fts_content, tags_json, kind, canonical_project.as_deref().unwrap_or("")],
        ).map_err(|e| format!("FTS insert: {}", e))?;

        if let Some(proj) = canonical_project.as_deref() {
            let _ = self.ensure_project(proj);
        }

        let mem = Memory {
            id,
            content: content.into(),
            kind: kind.into(),
            project: canonical_project,
            tags: tags.to_vec(),
            source: source.into(),
            importance: imp,
            expires_at: expires_at.map(String::from),
            created_at: now.clone(),
            updated_at: now,
            metadata: scoped_metadata,
            last_accessed_at: None,
            access_count: 0,
        };
        let _ = self.rebuild_links(&mem);

        // Auto-compaction: trigger GC when memory count exceeds threshold
        self.maybe_auto_compact();

        Ok((mem, false))
    }

    /// Trigger auto-compaction if memory count exceeds threshold.
    /// Debounced: runs max once per 5 minutes.
    fn maybe_auto_compact(&self) {
        compaction::maybe_auto_compact(self);
    }
    /// Full update with all fields.
    pub fn update_memory_full(
        &self,
        id: &str,
        content: Option<&str>,
        kind: Option<&str>,
        tags: Option<&[String]>,
        importance: Option<i32>,
        expires_at: Option<&str>,
        metadata: Option<&serde_json::Value>,
    ) -> Result<Option<Memory>, String> {
        let existing = match self.get_memory(id)? {
            Some(m) => m,
            None => return Ok(None),
        };
        let now = Utc::now().to_rfc3339();
        let new_content = content.unwrap_or(&existing.content);
        let new_kind = kind.unwrap_or(&existing.kind);
        let new_tags = tags
            .map(|t| t.to_vec())
            .unwrap_or_else(|| existing.tags.clone());
        let tags_json = serde_json::to_string(&new_tags).unwrap_or_else(|_| "[]".into());
        let new_imp = importance.unwrap_or(existing.importance).clamp(1, 5);
        let new_exp = if expires_at.is_some() {
            expires_at.map(String::from)
        } else {
            existing.expires_at.clone()
        };
        let new_metadata = metadata.cloned().or_else(|| existing.metadata.clone());
        let metadata_json = new_metadata
            .as_ref()
            .map(|value| serde_json::to_string(value).unwrap_or_default());
        let new_hash = content_hash(new_content);
        let content_changed = content.is_some() && new_content != existing.content;

        if content_changed {
            self.conn.execute(
                "UPDATE memories SET content=?1,kind=?2,tags=?3,importance=?4,expires_at=?5,metadata=?6,updated_at=?7,embedding=NULL,content_hash=?8 WHERE id=?9",
                params![new_content, new_kind, tags_json, new_imp, new_exp, metadata_json, now, &new_hash, id],
            ).map_err(|e| format!("Update: {}", e))?;
            queue_embedding_job(id, new_content);
        } else {
            self.conn.execute(
                "UPDATE memories SET content=?1,kind=?2,tags=?3,importance=?4,expires_at=?5,metadata=?6,updated_at=?7 WHERE id=?8",
                params![new_content, new_kind, tags_json, new_imp, new_exp, metadata_json, now, id],
            ).map_err(|e| format!("Update: {}", e))?;
        }

        // Rebuild FTS
        if let Ok(rowid) = self.conn.query_row::<i64, _, _>(
            "SELECT rowid FROM memories WHERE id=?1",
            params![id],
            |r| r.get(0),
        ) {
            let _ = self
                .conn
                .execute("DELETE FROM memories_fts WHERE rowid=?1", params![rowid]);
            let proj = existing.project.as_deref().unwrap_or("");
            let fts_content = Self::fts_index_content(new_content);
            let _ = self.conn.execute(
                "INSERT INTO memories_fts (rowid,content,tags,kind,project) VALUES (?1,?2,?3,?4,?5)",
                params![rowid, fts_content, tags_json, new_kind, proj]);
        }

        let mem = Memory {
            id: id.into(),
            content: new_content.into(),
            kind: new_kind.into(),
            project: existing.project,
            tags: new_tags,
            source: existing.source,
            importance: new_imp,
            expires_at: new_exp,
            created_at: existing.created_at,
            updated_at: now,
            metadata: new_metadata,
            last_accessed_at: existing.last_accessed_at,
            access_count: existing.access_count,
        };
        let _ = self.rebuild_links(&mem);
        Ok(Some(mem))
    }

    pub fn delete_memory(&self, id: &str) -> Result<bool, String> {
        if let Ok(rowid) = self.conn.query_row::<i64, _, _>(
            "SELECT rowid FROM memories WHERE id=?1",
            params![id],
            |r| r.get(0),
        ) {
            let _ = self
                .conn
                .execute("DELETE FROM memories_fts WHERE rowid=?1", params![rowid]);
        }
        let affected = self
            .conn
            .execute("DELETE FROM memories WHERE id=?1", params![id])
            .map_err(|e| format!("Delete: {}", e))?;
        if let Some(ann) = self.ann.as_ref() {
            let _ = ann.remove(id);
        }
        Ok(affected > 0)
    }

    pub fn get_memory(&self, id: &str) -> Result<Option<Memory>, String> {
        let mut stmt = self.conn.prepare(
            "SELECT id,content,kind,project,tags,source,importance,expires_at,metadata,created_at,updated_at,last_accessed_at,access_count FROM memories WHERE id=?1"
        ).map_err(|e| format!("Prepare: {}", e))?;
        let mut rows = stmt
            .query(params![id])
            .map_err(|e| format!("Query: {}", e))?;
        match rows.next().map_err(|e| format!("Next: {}", e))? {
            Some(row) => Ok(Some(row_to_memory(row))),
            None => Ok(None),
        }
    }

    // ─── BULK ADD ─────────────────────────────────────

    /// Add multiple memories in one call, with dedup per item. Returns (added, merged, skipped).
    pub fn add_memories_bulk(
        &self,
        items: &[BulkItem],
    ) -> Result<(Vec<Memory>, usize, usize), String> {
        let mut added: Vec<Memory> = Vec::new();
        let mut merged = 0usize;
        let mut skipped = 0usize;
        for item in items {
            if item.content.trim().is_empty() {
                skipped += 1;
                continue;
            }
            let tags: Vec<String> = item.tags.clone().unwrap_or_default();
            let imp = item.importance.unwrap_or(3);
            let exp = item.expires_at.as_deref();
            match self.add_memory(
                &item.content,
                &item.kind,
                item.project.as_deref(),
                &tags,
                &item.source,
                imp,
                exp,
                item.metadata.as_ref(),
                &item.scope(),
            ) {
                Ok((mem, was_merged)) => {
                    if was_merged {
                        merged += 1;
                    } else {
                        added.push(mem);
                    }
                }
                Err(_) => {
                    skipped += 1;
                }
            }
        }
        Ok((added, merged, skipped))
    }

    // ─── SEARCH (FTS5 BM25 × importance) ──────────────

    fn infer_query_intent(query: &str) -> QueryIntent {
        let lower = query.to_ascii_lowercase();
        let contains_any = |needles: &[&str]| needles.iter().any(|needle| lower.contains(needle));
        QueryIntent {
            preference: contains_any(&[
                "prefer",
                "preference",
                "like",
                "favorite",
                "favourite",
                "would rather",
                "préfère",
                "préférence",
                "aime",
                "favori",
                "plutôt",
            ]),
            temporal: contains_any(&[
                "when",
                "before",
                "after",
                "first",
                "last",
                "latest",
                "recent",
                "currently",
                "now",
                "timeline",
                "previously",
                "earlier",
                "ensuite",
                "avant",
                "après",
                "dernier",
                "récemment",
                "maintenant",
                "actuellement",
            ]),
            user_turn: contains_any(&[
                "user",
                "human",
                "i said",
                "i told",
                "j'ai dit",
                "je voulais",
                "utilisateur",
            ]),
            assistant_turn: contains_any(&[
                "assistant",
                "you said",
                "claude",
                "cursor",
                "chatgpt",
                "tu as dit",
            ]),
            update_or_correction: contains_any(&[
                "changed",
                "updated",
                "correction",
                "actually",
                "instead",
                "not anymore",
                "now uses",
                "switch",
                "switched",
                "remplace",
                "corrige",
                "modifié",
                "désormais",
                "maintenant",
                "en fait",
                "plutôt",
            ]),
            technical: contains_any(&[
                "bug",
                "error",
                "stack",
                "architecture",
                "api",
                "database",
                "file",
                "function",
                "component",
                "deploy",
                "build",
                "benchmark",
                "embedding",
                "reranker",
            ]),
        }
    }

    fn query_intent_factor(intent: &QueryIntent, memory: &Memory) -> f64 {
        let content = memory.content.to_ascii_lowercase();
        let mut factor: f64 = 1.0;

        if intent.preference {
            let pref_signal = memory.kind == "preference"
                || content.contains("prefer")
                || content.contains("préf")
                || content.contains("like")
                || content.contains("aime")
                || content.contains("favorite")
                || content.contains("plutôt");
            if pref_signal {
                factor *= 1.16;
            }
        }

        if intent.user_turn && content.trim_start().starts_with("user:") {
            factor *= 1.12;
        }
        if intent.assistant_turn && content.trim_start().starts_with("assistant:") {
            factor *= 1.12;
        }

        if intent.update_or_correction {
            let update_signal = content.contains("changed")
                || content.contains("updated")
                || content.contains("actually")
                || content.contains("instead")
                || content.contains("correction")
                || content.contains("switch")
                || content.contains("maintenant")
                || content.contains("désormais")
                || content.contains("en fait");
            if update_signal || matches!(memory.kind.as_str(), "decision" | "architecture" | "fact")
            {
                factor *= 1.1;
            }
        }

        if intent.temporal {
            if let Some(turn) = memory
                .id
                .split("__t")
                .last()
                .and_then(|value| value.parse::<usize>().ok())
            {
                if turn >= 20 {
                    factor *= 1.08;
                } else if turn <= 2
                    && (content.contains("first")
                        || content.contains("initial")
                        || content.contains("started"))
                {
                    factor *= 1.05;
                }
            }
            if content.contains("today")
                || content.contains("yesterday")
                || content.contains("now")
                || content.contains("maintenant")
            {
                factor *= 1.05;
            }
        }

        if intent.technical
            && matches!(
                memory.kind.as_str(),
                "architecture" | "decision" | "bug" | "snippet" | "pattern"
            )
        {
            factor *= 1.07;
        }

        factor.min(1.35)
    }

    fn cognitive_activation_factor(memory: &Memory, now_ts: f64) -> f64 {
        let frequency = ((memory.access_count.max(0) as f64 + 1.0).ln() / 21.0_f64.ln()).min(1.0);
        let last_touch = memory
            .last_accessed_at
            .as_deref()
            .unwrap_or(memory.updated_at.as_str());
        let recency = chrono::DateTime::parse_from_rfc3339(last_touch)
            .ok()
            .map(|timestamp| {
                let age_days = ((now_ts - timestamp.timestamp() as f64) / 86400.0).max(0.0);
                (1.0 / (1.0 + age_days / 14.0)).min(1.0)
            })
            .unwrap_or(0.0);

        1.0 + ((frequency * 0.07) + (recency * 0.05)).min(0.12)
    }

    pub fn search(
        &self,
        query: &str,
        limit: usize,
        project: Option<&str>,
        kind: Option<&str>,
        tags: Option<&[String]>,
        watcher_keywords: Option<&[String]>,
    ) -> Result<Vec<SearchResult>, String> {
        let canonical_project = Self::canonical_project(project);

        let telemetry_enabled = crate::telemetry::is_enabled();
        let search_start = std::time::Instant::now();
        let mut trace = if telemetry_enabled {
            let mut t = crate::telemetry::RetrievalTrace::default();
            t.query_truncated = crate::telemetry::truncate_query(query);
            t.query_chars = query.chars().count();
            t.project = canonical_project.clone();
            t.kind = kind.map(|k| k.to_string());
            t.tags_count = tags.map(|t| t.len()).unwrap_or(0);
            t.limit = limit;
            Some(t)
        } else {
            None
        };

        let fts_variants = crate::fts::fts5_query_variants(query);
        if fts_variants.is_empty() {
            return Ok(Vec::new());
        }
        if let Some(t) = trace.as_mut() {
            t.fts_variants = fts_variants.len();
        }

        let _ = self.cleanup_expired();

        let embed_start = std::time::Instant::now();
        let query_emb = cached_embed_text(query);
        if let Some(t) = trace.as_mut() {
            t.timing_ms_embed_query = embed_start.elapsed().as_secs_f64() * 1000.0;
        }

        // Pre-compute KG expansion terms for post-retrieval scoring boost
        let kg_expansion = self.get_kg_expansion_terms(query);

        let mut bm25_results = std::collections::HashMap::new();
        let mut all_memories = std::collections::HashMap::new();
        let mut candidate_sources: std::collections::HashMap<
            String,
            std::collections::HashSet<String>,
        > = std::collections::HashMap::new();

        let bm25_start = std::time::Instant::now();
        // 1. FTS5 search. Run the normal prefix query plus exact phrase/proximity variants.
        let run_fts_variant =
            |fts_query: &str,
             source: &'static str,
             rank_offset: usize,
             limit_rows: usize,
             return_err_on_failure: bool,
             bm25_results: &mut std::collections::HashMap<String, usize>,
             all_memories: &mut std::collections::HashMap<String, Memory>,
             candidate_sources: &mut std::collections::HashMap<
                String,
                std::collections::HashSet<String>,
             >|
             -> Result<usize, String> {
                let mut conditions = vec!["memories_fts MATCH ?1".to_string()];
                let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> =
                    vec![Box::new(fts_query.to_string())];

                if let Some(p) = canonical_project.as_deref() {
                    conditions.push(format!("m.project = ?{}", param_values.len() + 1));
                    param_values.push(Box::new(p.to_string()));
                }
                if let Some(k) = kind {
                    conditions.push(format!("m.kind = ?{}", param_values.len() + 1));
                    param_values.push(Box::new(k.to_string()));
                }

                let where_clause = conditions.join(" AND ");
                let sql = format!(
                    "SELECT m.id,m.content,m.kind,m.project,m.tags,m.source,m.importance,m.expires_at,m.metadata,m.created_at,m.updated_at,m.last_accessed_at,m.access_count,
                            bm25(memories_fts, 8.0, 5.0, 2.0, 4.0) AS bm25_score
                     FROM memories_fts f
                     JOIN memories m ON m.rowid = f.rowid
                     WHERE {}
                     ORDER BY bm25_score ASC
                     LIMIT {}", where_clause, limit_rows);

                let mut stmt = match self.conn.prepare(&sql) {
                    Ok(stmt) => stmt,
                    Err(error) if return_err_on_failure => {
                        return Err(format!("Search prepare: {}", error));
                    }
                    Err(_) => return Ok(0),
                };
                let param_refs: Vec<&dyn rusqlite::types::ToSql> =
                    param_values.iter().map(|p| p.as_ref()).collect();
                let rows = match stmt.query_map(param_refs.as_slice(), |row| {
                    let mem = row_to_memory(row);
                    let bm25: f64 = row.get(13)?;
                    Ok((mem, bm25))
                }) {
                    Ok(rows) => rows,
                    Err(error) if return_err_on_failure => {
                        return Err(format!("Search: {}", error));
                    }
                    Err(_) => return Ok(0),
                };

                let mut produced = 0usize;
                for (rank_index, row) in rows.flatten().enumerate() {
                    let (mem, _) = row;
                    let rank = rank_index + 1 + rank_offset;
                    let entry = bm25_results.entry(mem.id.clone()).or_insert(rank);
                    *entry = (*entry).min(rank);
                    candidate_sources
                        .entry(mem.id.clone())
                        .or_default()
                        .insert(source.to_string());
                    all_memories.entry(mem.id.clone()).or_insert(mem);
                    produced += 1;
                }
                Ok(produced)
            };

        for (variant_index, (fts_query, source)) in fts_variants.iter().enumerate() {
            let limit_rows = if variant_index == 0 { 150 } else { 75 };
            let return_err_on_failure = variant_index == 0;
            run_fts_variant(
                fts_query,
                *source,
                0,
                limit_rows,
                return_err_on_failure,
                &mut bm25_results,
                &mut all_memories,
                &mut candidate_sources,
            )?;
        }

        // NOTE: lexical synonym expansion (`fts5_synonym_variants`) is
        // intentionally NOT invoked here. Three feeding strategies were
        // measured on memorypilot-fr-30:
        //
        //   - Eager OR into BM25 pool       : R@5 73.3% → 63.3% (-10)
        //   - Penalised rank, capped pool   : R@5 73.3% → 70.0% (-3)
        //   - Candidate-only (vector judge) : R@5 73.3% → 63.3% (-10)
        //
        // All three regress precision because the curated thesaurus is
        // too generic for a small corpus and pulls in long-tail noise.
        // The right fix for cross-lingual / synonym retrieval is the
        // dense cross-encoder reranker, not BM25 widening. The
        // dictionary remains in `query_expansion.rs` for callers that
        // want it (CLI tools, HTTP debugging) and may resurface later
        // for very large corpora where the long tail dilutes naturally.

        // 2a. Optional ANN pre-filter. Marks candidates with `vector_ann` so the
        // explain output exposes which retrieval stage surfaced them. The full
        // scan below stays authoritative for ranking — ANN only annotates here.
        let ann_hits: Vec<(String, f32)> = self
            .ann
            .as_ref()
            .map(|index| index.search(&query_emb, 200))
            .unwrap_or_default();
        for (id, _) in &ann_hits {
            candidate_sources
                .entry(id.clone())
                .or_default()
                .insert("vector_ann".to_string());
        }

        // 2b. Vector Search (read pool connection, scoped). When the ANN index is
        // populated past `ANN_BYPASS_THRESHOLD`, restrict the SQL scan to the union
        // of ANN top-K and BM25 hits — avoids loading every embedding blob just to
        // score it. Below the threshold the full scan is kept (provably no recall
        // regression).
        const ANN_BYPASS_THRESHOLD: usize = 5_000;
        let ann_active_for_bypass = self
            .ann
            .as_ref()
            .map(|index| index.len() >= ANN_BYPASS_THRESHOLD && !ann_hits.is_empty())
            .unwrap_or(false);
        let restricted_ids: Option<Vec<String>> = if ann_active_for_bypass {
            let mut ids: std::collections::HashSet<String> = bm25_results.keys().cloned().collect();
            for (id, _) in &ann_hits {
                ids.insert(id.clone());
            }
            Some(ids.into_iter().collect())
        } else {
            None
        };

        if let Some(t) = trace.as_mut() {
            t.timing_ms_bm25 = bm25_start.elapsed().as_secs_f64() * 1000.0;
            t.candidates_bm25 = bm25_results.len();
        }

        let vector_start = std::time::Instant::now();
        let vector_results = {
            let mut vec_conditions = Vec::new();
            let mut vec_params: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
            if let Some(p) = canonical_project.as_deref() {
                vec_conditions.push(format!("project = ?{}", vec_params.len() + 1));
                vec_params.push(Box::new(p.to_string()));
            }
            if let Some(k) = kind {
                vec_conditions.push(format!("kind = ?{}", vec_params.len() + 1));
                vec_params.push(Box::new(k.to_string()));
            }
            if let Some(ids) = restricted_ids.as_ref() {
                let placeholders = (1..=ids.len())
                    .map(|i| format!("?{}", vec_params.len() + i))
                    .collect::<Vec<_>>()
                    .join(",");
                vec_conditions.push(format!("id IN ({})", placeholders));
                for id in ids {
                    vec_params.push(Box::new(id.clone()));
                }
            }
            let vec_where = if vec_conditions.is_empty() {
                String::new()
            } else {
                format!("WHERE {}", vec_conditions.join(" AND "))
            };
            let vec_sql = format!("SELECT id,content,kind,project,tags,source,importance,expires_at,metadata,created_at,updated_at,last_accessed_at,access_count,embedding FROM memories {}", vec_where);
            let rconn = self.read_conn();
            let mut stmt2 = rconn
                .prepare(&vec_sql)
                .map_err(|e| format!("Vector Search: {}", e))?;
            let vec_refs: Vec<&dyn rusqlite::types::ToSql> =
                vec_params.iter().map(|p| p.as_ref()).collect();

            let mut vector_scores: Vec<(String, f32)> = Vec::new();
            let rows2 = stmt2
                .query_map(vec_refs.as_slice(), |row| {
                    let mem = row_to_memory(row);
                    let blob: Option<Vec<u8>> = row.get(13)?;
                    Ok((mem, blob))
                })
                .map_err(|e| format!("Vector Search error: {}", e))?;

            for r in rows2.flatten() {
                let (mem, blob) = r;
                all_memories
                    .entry(mem.id.clone())
                    .or_insert_with(|| mem.clone());
                if let Some(b) = blob {
                    candidate_sources
                        .entry(mem.id.clone())
                        .or_default()
                        .insert("vector".to_string());
                    let score = crate::embedding::similarity_with_blob(&query_emb, &b);
                    vector_scores.push((mem.id, score));
                } else {
                    candidate_sources
                        .entry(mem.id.clone())
                        .or_default()
                        .insert("vector_pending".to_string());
                    vector_scores.push((mem.id, 0.0));
                }
            }

            vector_scores
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let mut vr = std::collections::HashMap::new();
            for (i, (id, _)) in vector_scores.iter().take(100).enumerate() {
                vr.insert(id.clone(), i + 1);
            }
            vr
        };

        if let Some(t) = trace.as_mut() {
            t.timing_ms_vector = vector_start.elapsed().as_secs_f64() * 1000.0;
            t.candidates_vector = vector_results.len();
            t.candidates_total_unique = all_memories.len();
            t.candidates_ann = candidate_sources
                .values()
                .filter(|set| set.iter().any(|src| src == "ann"))
                .count();
            t.kg_expansion_terms = kg_expansion.len();
        }
        let fusion_start = std::time::Instant::now();

        // 3. RRF Fusion
        let mut rrf_scores: Vec<(String, f64)> = Vec::new();

        // Fetch graph links for PageRank-like boost (scoped to candidates)
        let candidate_ids: Vec<&String> = all_memories.keys().collect();
        let link_boosts = self.build_link_boosts_for(&candidate_ids);

        // Batch-query knowledge triple counts (avoids N+1)
        let triple_counts = self.batch_triple_counts(&candidate_ids);

        let now_ts = Utc::now().timestamp() as f64;

        let query_tokens: Vec<String> = query
            .split_whitespace()
            .map(|w| w.to_lowercase())
            .filter(|w| w.len() >= 3)
            .collect();
        let query_intent = Self::infer_query_intent(query);

        for (id, mem) in &all_memories {
            let bm25_rank = bm25_results.get(id).copied().unwrap_or(1000);
            let vec_rank = vector_results.get(id).copied().unwrap_or(1000);
            let mut score = crate::embedding::rrf_score(bm25_rank, vec_rank);

            // Exact term coverage: boost if a high fraction of query terms appear in the memory content
            if !query_tokens.is_empty() {
                let content_lower = mem.content.to_lowercase();
                let match_frac = query_tokens
                    .iter()
                    .filter(|t| content_lower.contains(t.as_str()))
                    .count() as f64
                    / query_tokens.len() as f64;
                if match_frac >= 0.8 {
                    score *= 1.0 + (match_frac - 0.8) * 0.5; // up to +10% for 100% match
                }
            }

            score *= Self::query_intent_factor(&query_intent, mem);
            score *= Self::cognitive_activation_factor(mem, now_ts);

            // Importance tiebreaker: subtle nudge, never overrides relevance
            // imp 5 → 1.06x, imp 4 → 1.03x, imp 3 → 1.0x, imp 2 → 0.98x, imp 1 → 0.96x
            let imp_factor = 1.0 + (mem.importance as f64 - 3.0) * 0.03;
            score *= imp_factor;

            // Temporal recency: gentle boost, decaying over 30 days
            if let Ok(updated) = chrono::DateTime::parse_from_rfc3339(&mem.updated_at) {
                let age_days = (now_ts - updated.timestamp() as f64) / 86400.0;
                let recency = if age_days <= 3.0 {
                    1.05
                } else if age_days <= 30.0 {
                    1.0 + 0.05 * ((30.0 - age_days) / 27.0)
                } else {
                    1.0
                };
                score *= recency;
            }

            // KG expansion: subtle relevance signal from entity co-occurrence
            if !kg_expansion.is_empty() {
                let content_lower = mem.content.to_lowercase();
                let kg_hits = kg_expansion
                    .iter()
                    .filter(|t| content_lower.contains(t.as_str()))
                    .count();
                if kg_hits > 0 {
                    score *= 1.0 + (kg_hits as f64 * 0.04).min(0.15);
                }
            }

            // PageRank-like link boost
            if let Some(lb) = link_boosts.get(id) {
                if *lb < 0.0 {
                    score *= 1.0 + lb; // penalty (e.g. 1.0 - 0.9 = 0.1x score)
                } else {
                    score *= 1.0 + lb; // boost
                }
            }

            // Watcher boost (dynamic context)
            if let Some(keywords) = watcher_keywords {
                let content_lower = mem.content.to_lowercase();
                let match_count = keywords
                    .iter()
                    .filter(|w| content_lower.contains(w.to_lowercase().as_str()))
                    .count();
                if match_count > 0 {
                    score *= 1.0 + (match_count as f64 * 0.2); // +20% per matching keyword
                }
            }

            // Also boost if tag match
            if let Some(filter_tags) = tags {
                let filter_set: std::collections::HashSet<String> =
                    filter_tags.iter().map(|t| t.to_lowercase()).collect();
                if mem
                    .tags
                    .iter()
                    .any(|t| filter_set.contains(&t.to_lowercase()))
                {
                    score *= 1.5;
                } else {
                    score *= 0.1; // penalize if tags are requested but don't match
                }
            }

            // Penalize memories linked to expired knowledge triples (batch-preloaded)
            if let Some(&(active, expired)) = triple_counts.get(id.as_str()) {
                if expired > 0 {
                    if active == 0 {
                        score *= 0.3;
                    } else {
                        let ratio = active as f64 / (active + expired) as f64;
                        score *= 0.5 + (ratio * 0.5);
                    }
                }
            }

            rrf_scores.push((id.clone(), score));
        }

        // Deterministic tie-break on the memory id. Without it, two
        // candidates that happen to score identically (a common case
        // when the cross-encoder normalises into a small dynamic
        // range) would be ordered by the unstable HashMap iteration
        // order, producing different top-K from one process to the
        // next on the same input. The id-based break costs nothing
        // and turns the bench output into a reproducible artefact.
        rrf_scores.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });

        let mut results: Vec<SearchResult> = Vec::new();
        let session_candidate_limit = if crate::session_fusion::should_expand_candidates(query) {
            (limit * 3).clamp(limit, 30)
        } else {
            limit
        };
        for (id, score) in rrf_scores.into_iter().take(session_candidate_limit) {
            if let Some(mem) = all_memories.remove(&id) {
                let mut sources = candidate_sources
                    .remove(&id)
                    .map(|items| items.into_iter().collect::<Vec<_>>())
                    .unwrap_or_default();
                sources.sort();
                results.push(SearchResult {
                    memory: mem,
                    score: (score * 10000.0).round() / 10000.0,
                    sources,
                });
            }
        }

        // 4. GraphRAG Traversal (Expand context based on top matches)
        let top_ids: Vec<String> = results
            .iter()
            .take(3)
            .map(|r| r.memory.id.clone())
            .collect();
        if let Ok(related_ids) = crate::graph::traverse_graph(&self.conn, &top_ids, 1) {
            for rel_id in related_ids {
                // If it's not already in results, fetch it and add it
                if !results.iter().any(|r| r.memory.id == rel_id) {
                    if let Ok(Some(mem)) = self.get_memory(&rel_id) {
                        results.push(SearchResult {
                            memory: mem,
                            // Give it a slightly lower score than the original match that pulled it
                            score: 0.1,
                            sources: vec!["graph".to_string()],
                        });
                    }
                }
            }
        }

        // 5. Combinatorial Reranker — boost connected clusters
        // Greedy subgraph selection: prefer memories that are connected to other selected memories
        if results.len() > 2 {
            let result_ids: Vec<String> = results.iter().map(|r| r.memory.id.clone()).collect();
            let adjacency = self.build_adjacency_set(&result_ids);

            let mut selected: Vec<usize> = Vec::with_capacity(results.len());
            let mut remaining: Vec<usize> = (0..results.len()).collect();

            // Seed: pick the highest-scoring result
            selected.push(remaining.remove(0));

            while !remaining.is_empty() && selected.len() < results.len() {
                let mut best_idx = 0;
                let mut best_combined = f64::NEG_INFINITY;

                for (ri, &cand) in remaining.iter().enumerate() {
                    let base_score = results[cand].score;
                    let cand_id = &results[cand].memory.id;

                    let conn_count = selected
                        .iter()
                        .filter(|&&si| {
                            let sel_id = &results[si].memory.id;
                            adjacency.contains(&(cand_id.clone(), sel_id.clone()))
                                || adjacency.contains(&(sel_id.clone(), cand_id.clone()))
                        })
                        .count();

                    // connectivity_bonus: tiebreaker only — 5% per connected memory, capped at 15%
                    let connectivity_bonus = (conn_count as f64 * 0.05).min(0.15);
                    let combined = base_score * (1.0 + connectivity_bonus);

                    if combined > best_combined {
                        best_combined = combined;
                        best_idx = ri;
                    }
                }

                selected.push(remaining.remove(best_idx));
            }

            let mut reranked: Vec<SearchResult> = selected
                .into_iter()
                .map(|i| {
                    std::mem::replace(
                        &mut results[i],
                        SearchResult {
                            memory: Memory {
                                id: String::new(),
                                content: String::new(),
                                kind: String::new(),
                                project: None,
                                tags: vec![],
                                source: String::new(),
                                importance: 0,
                                expires_at: None,
                                metadata: None,
                                created_at: String::new(),
                                updated_at: String::new(),
                                last_accessed_at: None,
                                access_count: 0,
                            },
                            score: 0.0,
                            sources: Vec::new(),
                        },
                    )
                })
                .collect();

            // Ensure monotonically decreasing scores
            for i in 1..reranked.len() {
                let prev = reranked[i - 1].score;
                if reranked[i].score > prev {
                    reranked[i].score = prev * 0.99;
                }
            }

            results = reranked;
        }

        crate::reranking::rerank_local(query, &mut results);
        crate::reranking::rerank_cross_encoder_if_enabled(query, &mut results);
        results = crate::session_fusion::fuse_sessions(query, results, limit);

        if let Some(mut t) = trace.take() {
            t.timing_ms_fusion = fusion_start.elapsed().as_secs_f64() * 1000.0;
            t.timing_ms_total = search_start.elapsed().as_secs_f64() * 1000.0;
            t.results_returned = results.len();
            if let Some(top) = results.first() {
                t.top_score = top.score;
                t.top_sources = top.sources.clone();
            }
            crate::telemetry::emit(&t);
        }

        // Defer access-count + last_accessed updates to the embed worker
        // so the hot search path never holds the WAL writer lock. The
        // worker drains the queue every ~2 s in a single batch UPDATE.
        for res in &results {
            queue_access_update(res.memory.id.clone());
        }

        Ok(results)
    }
    // ─── LIST ─────────────────────────────────────────

    pub fn list_memories(
        &self,
        project: Option<&str>,
        kind: Option<&str>,
        exclude_kind: Option<&str>,
        limit: usize,
        offset: usize,
    ) -> Result<(Vec<Memory>, i64), String> {
        let _ = self.cleanup_expired();
        let canonical_project = Self::canonical_project(project);

        let mut conditions: Vec<String> = Vec::new();
        let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();

        if let Some(p) = canonical_project.as_deref() {
            conditions.push(format!("project = ?{}", param_values.len() + 1));
            param_values.push(Box::new(p.to_string()));
        }
        if let Some(k) = kind {
            conditions.push(format!("kind = ?{}", param_values.len() + 1));
            param_values.push(Box::new(k.to_string()));
        }
        if let Some(ek) = exclude_kind {
            conditions.push(format!("kind != ?{}", param_values.len() + 1));
            param_values.push(Box::new(ek.to_string()));
        }

        let where_clause = if conditions.is_empty() {
            String::new()
        } else {
            format!(" WHERE {}", conditions.join(" AND "))
        };

        let count_sql = format!("SELECT COUNT(*) FROM memories{}", where_clause);
        let param_refs: Vec<&dyn rusqlite::types::ToSql> =
            param_values.iter().map(|p| p.as_ref()).collect();
        let total: i64 = self
            .conn
            .query_row(&count_sql, param_refs.as_slice(), |r| r.get(0))
            .map_err(|e| format!("Count: {}", e))?;

        let data_sql = format!(
            "SELECT id,content,kind,project,tags,source,importance,expires_at,metadata,created_at,updated_at,last_accessed_at,access_count FROM memories{} ORDER BY updated_at DESC LIMIT ?{} OFFSET ?{}",
            where_clause, param_values.len() + 1, param_values.len() + 2);
        param_values.push(Box::new(limit as i64));
        param_values.push(Box::new(offset as i64));
        let param_refs2: Vec<&dyn rusqlite::types::ToSql> =
            param_values.iter().map(|p| p.as_ref()).collect();

        let mut stmt = self
            .conn
            .prepare(&data_sql)
            .map_err(|e| format!("List: {}", e))?;
        let memories: Vec<Memory> = stmt
            .query_map(param_refs2.as_slice(), |r| Ok(row_to_memory(r)))
            .map_err(|e| format!("List query: {}", e))?
            .filter_map(|r| r.ok())
            .collect();
        Ok((memories, total))
    }
    // ─── TTL / EXPIRATION ─────────────────────────────

    pub fn cleanup_expired(&self) -> Result<usize, String> {
        static LAST_CLEANUP: OnceLock<Mutex<std::time::Instant>> = OnceLock::new();
        let last = LAST_CLEANUP.get_or_init(|| {
            Mutex::new(std::time::Instant::now() - std::time::Duration::from_secs(120))
        });
        if let Ok(mut ts) = last.lock() {
            if ts.elapsed() < std::time::Duration::from_secs(60) {
                return Ok(0);
            }
            *ts = std::time::Instant::now();
        }
        let now = Utc::now().to_rfc3339();
        let _ = self.conn.execute(
            "DELETE FROM memories_fts WHERE rowid IN (SELECT rowid FROM memories WHERE expires_at IS NOT NULL AND expires_at < ?1)",
            params![now]);
        let affected = self
            .conn
            .execute(
                "DELETE FROM memories WHERE expires_at IS NOT NULL AND expires_at < ?1",
                params![now],
            )
            .map_err(|e| format!("Cleanup: {}", e))?;
        Ok(affected)
    }

    // ─── GC & COMPRESSION ─────────────────────────────

    pub fn run_gc(
        &self,
        config: &crate::gc::GcConfig,
        dry_run: bool,
    ) -> Result<crate::gc::GcReport, String> {
        #[derive(Clone)]
        struct GcCandidate {
            id: String,
            content: String,
            project: Option<String>,
            importance: i32,
            age_days: i64,
            gc_score: f64,
        }

        let db_path = dirs::home_dir()
            .unwrap_or_default()
            .join(DB_DIR)
            .join(DB_FILE);
        let size_before = std::fs::metadata(&db_path).map(|m| m.len()).unwrap_or(0);

        let mut expired_removed = 0;
        if !dry_run {
            expired_removed = self.cleanup_expired()?;
        }

        // Find mergeable candidates
        let now = chrono::Utc::now();
        let mut groups_merged = 0;
        let mut memories_compressed = 0;
        let mut preview_candidates: Vec<crate::gc::GcPreviewCandidate> = Vec::new();

        for kind in &config.compressible_kinds {
            let sql = "SELECT id, content, project, importance, updated_at FROM memories WHERE kind = ?1 AND tags NOT LIKE '%pinned%'";
            if let Ok(mut stmt) = self.conn.prepare(&sql) {
                if let Ok(rows) = stmt.query_map(params![kind], |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, Option<String>>(2)?,
                        r.get::<_, i32>(3)?,
                        r.get::<_, String>(4)?,
                    ))
                }) {
                    let mut by_project: std::collections::HashMap<
                        Option<String>,
                        Vec<GcCandidate>,
                    > = std::collections::HashMap::new();
                    for row in rows.flatten() {
                        let updated_at = chrono::DateTime::parse_from_rfc3339(&row.4)
                            .unwrap_or_else(|_| chrono::Utc::now().into());
                        let age_days = (now - updated_at.with_timezone(&chrono::Utc)).num_days();

                        let score = crate::gc::gc_score(row.3, age_days, kind, config);
                        if score > 0.6
                            && row.3 < config.importance_threshold
                            && age_days >= config.age_days
                        {
                            by_project
                                .entry(row.2.clone())
                                .or_default()
                                .push(GcCandidate {
                                    id: row.0,
                                    content: row.1,
                                    project: row.2,
                                    importance: row.3,
                                    age_days,
                                    gc_score: score,
                                });
                        }
                    }

                    for (proj, mut items) in by_project {
                        if items.len() > 1 {
                            items.truncate(config.max_merge_group);
                            let contents: Vec<String> =
                                items.iter().map(|item| item.content.clone()).collect();
                            let merged_content =
                                crate::gc::merge_memories(&contents, kind, proj.as_deref());

                            let ids_to_delete: Vec<String> =
                                items.iter().map(|item| item.id.clone()).collect();
                            let gc_score_avg = items.iter().map(|item| item.gc_score).sum::<f64>()
                                / items.len() as f64;
                            let age_days_min =
                                items.iter().map(|item| item.age_days).min().unwrap_or(0);
                            let age_days_max =
                                items.iter().map(|item| item.age_days).max().unwrap_or(0);
                            let importance_min =
                                items.iter().map(|item| item.importance).min().unwrap_or(0);
                            let importance_max =
                                items.iter().map(|item| item.importance).max().unwrap_or(0);

                            preview_candidates.push(crate::gc::GcPreviewCandidate {
                                kind: kind.clone(),
                                project: proj
                                    .clone()
                                    .or_else(|| items.iter().find_map(|item| item.project.clone())),
                                memory_ids: ids_to_delete.clone(),
                                sample_contents: items
                                    .iter()
                                    .take(3)
                                    .map(|item| Self::preview_snippet(&item.content))
                                    .collect(),
                                confidence_score: (gc_score_avg * 100.0).round() / 100.0,
                                gc_score_avg: (gc_score_avg * 100.0).round() / 100.0,
                                age_days_min,
                                age_days_max,
                                importance_min,
                                importance_max,
                            });

                            if !dry_run {
                                if self
                                    .add_memory(
                                        &merged_content,
                                        kind,
                                        proj.as_deref(),
                                        &["merged".to_string()],
                                        "gc_compressor",
                                        3,
                                        None,
                                        None,
                                        &MemoryScope::default(),
                                    )
                                    .is_ok()
                                {
                                    for id in ids_to_delete {
                                        let _ = self.delete_memory(&id);
                                        memories_compressed += 1;
                                    }
                                    groups_merged += 1;
                                }
                            } else {
                                memories_compressed += ids_to_delete.len();
                                groups_merged += 1;
                            }
                        }
                    }
                }
            }
        }

        let mut orphan_links_removed = 0;
        if !dry_run {
            orphan_links_removed += self
                .conn
                .execute(
                    "DELETE FROM memory_entities WHERE memory_id NOT IN (SELECT id FROM memories)",
                    [],
                )
                .unwrap_or(0);

            orphan_links_removed += self.conn.execute(
                "DELETE FROM memory_links WHERE source_id NOT IN (SELECT id FROM memories) OR target_id NOT IN (SELECT id FROM memories)",
                []
            ).unwrap_or(0);
        }

        let size_after = std::fs::metadata(&db_path).map(|m| m.len()).unwrap_or(0);
        let hygiene = self.hygiene_report();

        Ok(crate::gc::GcReport {
            expired_removed,
            groups_merged,
            memories_compressed,
            orphan_links_removed: orphan_links_removed as usize,
            db_size_before: size_before,
            db_size_after: size_after,
            preview_mode: dry_run,
            preview_candidates,
            hygiene,
        })
    }

    // ─── MEMORY CAPSULES ──────────────────────────────

    /// Compress old low-importance memories into dense capsules.
    /// Groups by project + age, merges into ~100-200 token summaries.
    /// Preserves Knowledge Graph links. Returns (capsules_created, memories_compressed).
    pub fn compact_to_capsules(
        &self,
        age_days: i64,
        importance_max: i32,
    ) -> Result<(usize, usize), String> {
        let now = chrono::Utc::now();
        let sql = "SELECT id, content, kind, project, importance, updated_at FROM memories WHERE kind NOT IN ('credential', 'architecture', 'transcript_chunk') AND importance <= ?1";
        let mut stmt = self
            .conn
            .prepare(sql)
            .map_err(|e| format!("Capsule prepare: {}", e))?;
        let all_rows: Vec<(String, String, String, Option<String>, i32, String)> = stmt
            .query_map(params![importance_max], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                    r.get::<_, Option<String>>(3)?,
                    r.get::<_, i32>(4)?,
                    r.get::<_, String>(5)?,
                ))
            })
            .map_err(|e| format!("Capsule query: {}", e))?
            .filter_map(|r| r.ok())
            .collect();

        let rows: Vec<&(String, String, String, Option<String>, i32, String)> = all_rows
            .iter()
            .filter(|(_id, _content, _kind, _proj, _imp, updated_at)| {
                chrono::DateTime::parse_from_rfc3339(updated_at)
                    .map(|dt| (now - dt.with_timezone(&chrono::Utc)).num_days() >= age_days)
                    .unwrap_or(false)
            })
            .collect();

        if rows.len() < 3 {
            return Ok((0, 0));
        }

        // Group by project
        let mut by_project: std::collections::HashMap<
            Option<String>,
            Vec<(String, String, String)>,
        > = std::collections::HashMap::new();
        for (id, content, kind, project, _imp, _updated) in rows {
            by_project.entry(project.clone()).or_default().push((
                id.clone(),
                content.clone(),
                kind.clone(),
            ));
        }

        let mut capsules_created = 0usize;
        let mut memories_compressed = 0usize;

        for (project, items) in by_project {
            // Process in chunks of 10
            for chunk in items.chunks(10) {
                if chunk.len() < 2 {
                    continue;
                }
                let contents: Vec<String> = chunk.iter().map(|(_, c, _)| c.clone()).collect();
                let kinds: Vec<String> = chunk.iter().map(|(_, _, k)| k.clone()).collect();
                let capsule = crate::gc::capsule_summary(&contents, &kinds, project.as_deref());

                if self
                    .add_memory(
                        &capsule,
                        "note",
                        project.as_deref(),
                        &["capsule".to_string(), "compressed".to_string()],
                        "auto_capsule",
                        3,
                        None,
                        None,
                        &MemoryScope::default(),
                    )
                    .is_ok()
                {
                    for (id, _, _) in chunk {
                        let _ = self.delete_memory(id);
                        memories_compressed += 1;
                    }
                    capsules_created += 1;
                }
            }
        }

        Ok((capsules_created, memories_compressed))
    }

    // ─── FIND RELATED ─────────────────────────────────

    pub fn find_related(&self, id: &str, depth: u32) -> Result<serde_json::Value, String> {
        use serde_json::json;
        let related_ids = crate::graph::traverse_graph(&self.conn, &[id.to_string()], depth)?;
        let mut results: Vec<serde_json::Value> = Vec::new();
        for rid in &related_ids {
            if rid == id {
                continue;
            }
            if let Ok(Some(mem)) = self.get_memory(rid) {
                results.push(json!({
                    "id": mem.id,
                    "kind": mem.kind,
                    "project": mem.project,
                    "importance": mem.importance,
                    "preview": if mem.content.len() > 150 { format!("{}...", &mem.content[..150]) } else { mem.content.clone() },
                    "tags": mem.tags,
                }));
            }
        }
        Ok(json!({
            "source_id": id,
            "depth": depth,
            "related_count": results.len(),
            "related": results,
        }))
    }

    // ─── BULK DELETE ──────────────────────────────────

    pub fn bulk_delete(
        &self,
        kind: Option<&str>,
        project: Option<&str>,
        tag: Option<&str>,
        older_than_days: Option<i64>,
        importance_max: Option<i32>,
    ) -> Result<usize, String> {
        let canonical_project = Self::canonical_project(project);
        let mut conditions: Vec<String> = Vec::new();
        let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();

        if let Some(k) = kind {
            conditions.push(format!("kind = ?{}", param_values.len() + 1));
            param_values.push(Box::new(k.to_string()));
        }
        if let Some(p) = canonical_project.as_deref() {
            conditions.push(format!("project = ?{}", param_values.len() + 1));
            param_values.push(Box::new(p.to_string()));
        }
        if let Some(imp) = importance_max {
            conditions.push(format!("importance <= ?{}", param_values.len() + 1));
            param_values.push(Box::new(imp));
        }
        if let Some(days) = older_than_days {
            let cutoff = (chrono::Utc::now() - chrono::Duration::days(days)).to_rfc3339();
            conditions.push(format!("updated_at < ?{}", param_values.len() + 1));
            param_values.push(Box::new(cutoff));
        }

        // Never bulk-delete pinned memories
        conditions.push("tags NOT LIKE '%pinned%'".to_string());

        if conditions.is_empty() {
            return Err("At least one filter required".to_string());
        }

        // First collect IDs to delete (for cascading entity/link cleanup)
        let select_sql = format!("SELECT id FROM memories WHERE {}", conditions.join(" AND "));
        let mut stmt = self
            .conn
            .prepare(&select_sql)
            .map_err(|e| format!("Bulk select: {}", e))?;
        let params_ref: Vec<&dyn rusqlite::types::ToSql> =
            param_values.iter().map(|p| p.as_ref()).collect();
        let ids: Vec<String> = stmt
            .query_map(params_ref.as_slice(), |r| r.get::<_, String>(0))
            .map_err(|e| format!("Bulk query: {}", e))?
            .filter_map(|r| r.ok())
            .collect();

        // Filter by tag if specified (needs JSON parsing)
        let ids_to_delete: Vec<String> = if let Some(tag_filter) = tag {
            ids.into_iter()
                .filter(|id| {
                    self.get_memory(id)
                        .ok()
                        .flatten()
                        .map(|m| m.tags.iter().any(|t| t == tag_filter))
                        .unwrap_or(false)
                })
                .collect()
        } else {
            ids
        };

        let mut deleted = 0;
        for id in &ids_to_delete {
            if self.delete_memory(id).unwrap_or(false) {
                deleted += 1;
            }
        }
        Ok(deleted)
    }

    // ─── MEMORY HEALTH REPORT ────────────────────────

    pub fn memory_health_report(&self) -> Result<serde_json::Value, String> {
        use serde_json::json;
        let now = chrono::Utc::now();

        let total: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))
            .unwrap_or(0);

        // By kind
        let mut by_kind: Vec<(String, i64)> = Vec::new();
        if let Ok(mut stmt) = self
            .conn
            .prepare("SELECT kind, COUNT(*) FROM memories GROUP BY kind ORDER BY COUNT(*) DESC")
        {
            if let Ok(rows) =
                stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)))
            {
                by_kind = rows.filter_map(|r| r.ok()).collect();
            }
        }

        // By project
        let mut by_project: Vec<(String, i64)> = Vec::new();
        if let Ok(mut stmt) = self.conn.prepare("SELECT COALESCE(project, '(global)'), COUNT(*) FROM memories GROUP BY project ORDER BY COUNT(*) DESC") {
            if let Ok(rows) = stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?))) {
                by_project = rows.filter_map(|r| r.ok()).collect();
            }
        }

        // By importance
        let mut by_importance: Vec<(i32, i64)> = Vec::new();
        if let Ok(mut stmt) = self.conn.prepare("SELECT importance, COUNT(*) FROM memories GROUP BY importance ORDER BY importance DESC") {
            if let Ok(rows) = stmt.query_map([], |r| Ok((r.get::<_, i32>(0)?, r.get::<_, i64>(1)?))) {
                by_importance = rows.filter_map(|r| r.ok()).collect();
            }
        }

        // Pinned count
        let pinned: i64 = self
            .conn
            .query_row(
                "SELECT COUNT(*) FROM memories WHERE tags LIKE '%pinned%'",
                [],
                |r| r.get(0),
            )
            .unwrap_or(0);

        // Stale memories (> 30 days, importance <= 2)
        let cutoff_30 = (now - chrono::Duration::days(30)).to_rfc3339();
        let stale: i64 = self
            .conn
            .query_row(
                "SELECT COUNT(*) FROM memories WHERE updated_at < ?1 AND importance <= 2",
                params![cutoff_30],
                |r| r.get(0),
            )
            .unwrap_or(0);

        // Compression potential (compressible old memories)
        let compressible: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM memories WHERE updated_at < ?1 AND importance <= 3 AND kind NOT IN ('credential', 'architecture', 'transcript_chunk') AND tags NOT LIKE '%pinned%'",
            params![cutoff_30], |r| r.get(0),
        ).unwrap_or(0);

        // Average age in days
        let avg_age: f64 = self
            .conn
            .query_row(
                "SELECT AVG(julianday('now') - julianday(created_at)) FROM memories",
                [],
                |r| r.get(0),
            )
            .unwrap_or(0.0);

        // Orphan entities and links
        let orphan_entities: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM memory_entities WHERE memory_id NOT IN (SELECT id FROM memories)", [], |r| r.get(0),
        ).unwrap_or(0);
        let orphan_links: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM memory_links WHERE source_id NOT IN (SELECT id FROM memories) OR target_id NOT IN (SELECT id FROM memories)", [], |r| r.get(0),
        ).unwrap_or(0);

        // DB size
        let db_path = dirs::home_dir()
            .unwrap_or_default()
            .join(DB_DIR)
            .join(DB_FILE);
        let db_size = std::fs::metadata(&db_path).map(|m| m.len()).unwrap_or(0);

        // Capsule count
        let capsules: i64 = self
            .conn
            .query_row(
                "SELECT COUNT(*) FROM memories WHERE tags LIKE '%capsule%'",
                [],
                |r| r.get(0),
            )
            .unwrap_or(0);

        Ok(json!({
            "total_memories": total,
            "pinned": pinned,
            "capsules": capsules,
            "stale_low_value": stale,
            "compression_potential": compressible,
            "orphan_entities": orphan_entities,
            "orphan_links": orphan_links,
            "avg_age_days": (avg_age * 10.0).round() / 10.0,
            "db_size_bytes": db_size,
            "db_size_mb": (db_size as f64 / 1_048_576.0 * 100.0).round() / 100.0,
            "by_kind": by_kind.into_iter().map(|(k, c)| json!({"kind": k, "count": c})).collect::<Vec<_>>(),
            "by_project": by_project.into_iter().map(|(p, c)| json!({"project": p, "count": c})).collect::<Vec<_>>(),
            "by_importance": by_importance.into_iter().map(|(i, c)| json!({"importance": i, "count": c})).collect::<Vec<_>>(),
        }))
    }

    // ─── DEDUPE REPORT ───────────────────────────────

    pub fn dedupe_report(
        &self,
        project: Option<&str>,
        threshold: f64,
    ) -> Result<serde_json::Value, String> {
        use serde_json::json;
        let canonical_project = Self::canonical_project(project);
        let memories: Vec<(String, String, String, i32)> = if let Some(proj) =
            canonical_project.as_deref()
        {
            let mut stmt = self.conn.prepare("SELECT id, content, kind, importance FROM memories WHERE project = ?1 ORDER BY created_at DESC LIMIT 500")
                .map_err(|e| format!("Dedupe prepare: {}", e))?;
            let result: Vec<(String, String, String, i32)> = stmt
                .query_map(params![proj], |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, String>(2)?,
                        r.get::<_, i32>(3)?,
                    ))
                })
                .map_err(|e| format!("Dedupe query: {}", e))?
                .filter_map(|r| r.ok())
                .collect();
            result
        } else {
            let mut stmt = self.conn.prepare("SELECT id, content, kind, importance FROM memories ORDER BY created_at DESC LIMIT 500")
                .map_err(|e| format!("Dedupe prepare: {}", e))?;
            let result: Vec<(String, String, String, i32)> = stmt
                .query_map([], |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, String>(2)?,
                        r.get::<_, i32>(3)?,
                    ))
                })
                .map_err(|e| format!("Dedupe query: {}", e))?
                .filter_map(|r| r.ok())
                .collect();
            result
        };

        let mut groups: Vec<serde_json::Value> = Vec::new();
        let mut seen: std::collections::HashSet<usize> = std::collections::HashSet::new();

        for i in 0..memories.len() {
            if seen.contains(&i) {
                continue;
            }
            let mut group: Vec<serde_json::Value> = Vec::new();
            let words_i: std::collections::HashSet<String> = memories[i]
                .1
                .split_whitespace()
                .map(|w: &str| {
                    w.trim_matches(|c: char| !c.is_alphanumeric())
                        .to_lowercase()
                })
                .filter(|w: &String| w.len() > 2)
                .collect();
            if words_i.is_empty() {
                continue;
            }

            for j in (i + 1)..memories.len() {
                if seen.contains(&j) {
                    continue;
                }
                let words_j: std::collections::HashSet<String> = memories[j]
                    .1
                    .split_whitespace()
                    .map(|w: &str| {
                        w.trim_matches(|c: char| !c.is_alphanumeric())
                            .to_lowercase()
                    })
                    .filter(|w: &String| w.len() > 2)
                    .collect();
                if words_j.is_empty() {
                    continue;
                }

                let intersection = words_i.intersection(&words_j).count();
                let union = words_i.union(&words_j).count();
                let jaccard = if union > 0 {
                    intersection as f64 / union as f64
                } else {
                    0.0
                };

                if jaccard >= threshold {
                    if group.is_empty() {
                        group.push(json!({
                            "id": memories[i].0,
                            "kind": memories[i].2,
                            "importance": memories[i].3,
                            "preview": if memories[i].1.len() > 120 { format!("{}...", &memories[i].1[..120]) } else { memories[i].1.clone() },
                        }));
                    }
                    group.push(json!({
                        "id": memories[j].0,
                        "kind": memories[j].2,
                        "importance": memories[j].3,
                        "similarity": (jaccard * 100.0).round() / 100.0,
                        "preview": if memories[j].1.len() > 120 { format!("{}...", &memories[j].1[..120]) } else { memories[j].1.clone() },
                    }));
                    seen.insert(j);
                }
            }

            if !group.is_empty() {
                seen.insert(i);
                groups.push(json!({
                    "group_size": group.len(),
                    "memories": group,
                }));
            }
        }

        Ok(json!({
            "threshold": threshold,
            "duplicate_groups": groups.len(),
            "total_duplicates": groups.iter().map(|g| g["group_size"].as_u64().unwrap_or(0)).sum::<u64>(),
            "groups": groups,
        }))
    }

    // ─── EXPORT ───────────────────────────────────────

    pub fn export_memories(&self, project: Option<&str>, format: &str) -> Result<String, String> {
        export::export_memories(self, project, format)
    }

    pub fn export_session_markdown(
        &self,
        session_id: Option<&str>,
        thread_id: Option<&str>,
        window_id: Option<&str>,
        project: Option<&str>,
    ) -> Result<String, String> {
        export::export_session_markdown(self, session_id, thread_id, window_id, project)
    }
    // ─── PROJECTS ─────────────────────────────────────

    fn ensure_project(&self, name: &str) -> Result<(), String> {
        let Some(canonical_name) = Self::canonical_project_name(name) else {
            return Ok(());
        };
        let now = Utc::now().to_rfc3339();
        self.conn
            .execute(
                "INSERT OR IGNORE INTO projects (name,path,created_at) VALUES (?1,'',?2)",
                params![canonical_name, now],
            )
            .map_err(|e| format!("Ensure: {}", e))?;
        Ok(())
    }

    pub fn register_project(
        &self,
        name: &str,
        path: &str,
        description: Option<&str>,
    ) -> Result<Project, String> {
        let canonical_name =
            Self::canonical_project_name(name).ok_or("Project name cannot be empty")?;
        let project_root = Self::infer_project_root(path);
        let now = Utc::now().to_rfc3339();
        self.conn
            .execute(
                "INSERT INTO projects (name,path,description,created_at) VALUES (?1,?2,?3,?4)
             ON CONFLICT(name) DO UPDATE SET path=?2, description=COALESCE(?3,description)",
                params![&canonical_name, &project_root, description, &now],
            )
            .map_err(|e| format!("Register: {}", e))?;
        let count: i64 = self
            .conn
            .query_row(
                "SELECT COUNT(*) FROM memories WHERE project=?1",
                params![&canonical_name],
                |r| r.get(0),
            )
            .unwrap_or(0);
        Ok(Project {
            name: canonical_name,
            path: project_root,
            description: description.map(String::from),
            created_at: now,
            memory_count: count,
        })
    }

    pub fn list_projects(&self) -> Result<Vec<Project>, String> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT p.name, p.path, p.description, p.created_at, COUNT(m.id) as cnt
             FROM projects p LEFT JOIN memories m ON m.project = p.name
             GROUP BY p.name ORDER BY cnt DESC",
            )
            .map_err(|e| format!("List projects: {}", e))?;
        let projects = stmt
            .query_map([], |row| {
                Ok(Project {
                    name: row.get(0)?,
                    path: row.get(1)?,
                    description: row.get(2)?,
                    created_at: row.get(3)?,
                    memory_count: row.get(4)?,
                })
            })
            .map_err(|e| format!("Projects: {}", e))?
            .filter_map(|r| r.ok())
            .collect();
        Ok(projects)
    }

    pub fn detect_project(&self, working_dir: &str) -> Result<Option<String>, String> {
        let normalized_dir = Self::normalize_path(working_dir);
        if normalized_dir.is_empty() {
            return Ok(None);
        }
        let project_root = Self::infer_project_root(&normalized_dir);

        let mut stmt = self
            .conn
            .prepare("SELECT name, path FROM projects WHERE path != '' ORDER BY length(path) DESC")
            .map_err(|e| format!("Detect: {}", e))?;
        let projects: Vec<(String, String)> = stmt
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
            .map_err(|e| format!("Detect2: {}", e))?
            .filter_map(|r| r.ok())
            .collect();
        for (name, path) in &projects {
            let normalized_path = Self::normalize_path(path);
            if Self::path_matches(&project_root, &normalized_path)
                || Self::path_matches(&normalized_dir, &normalized_path)
            {
                return Ok(Some(name.clone()));
            }
        }
        let project_name = Self::infer_project_slug_from_root(&project_root);
        if let Some(ref project_name) = project_name {
            let _ = self.remember_project_path_if_known(project_name, &project_root);
        }
        Ok(project_name)
    }
    // ─── STATS ────────────────────────────────────────

    pub fn stats(&self) -> Result<serde_json::Value, String> {
        let total: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))
            .unwrap_or(0);
        let global: i64 = self
            .conn
            .query_row(
                "SELECT COUNT(*) FROM memories WHERE project IS NULL",
                [],
                |r| r.get(0),
            )
            .unwrap_or(0);
        let projects: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM projects", [], |r| r.get(0))
            .unwrap_or(0);
        let expired: i64 = self
            .conn
            .query_row(
                "SELECT COUNT(*) FROM memories WHERE expires_at IS NOT NULL AND expires_at < ?1",
                params![Utc::now().to_rfc3339()],
                |r| r.get(0),
            )
            .unwrap_or(0);

        let mut by_kind = serde_json::Map::new();
        if let Ok(mut stmt) = self
            .conn
            .prepare("SELECT kind, COUNT(*) FROM memories GROUP BY kind")
        {
            if let Ok(rows) =
                stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)))
            {
                for row in rows.flatten() {
                    by_kind.insert(row.0, serde_json::json!(row.1));
                }
            }
        }
        let mut by_project = serde_json::Map::new();
        if let Ok(mut stmt) = self.conn.prepare(
            "SELECT COALESCE(project,'__global__'), COUNT(*) FROM memories GROUP BY project",
        ) {
            if let Ok(rows) =
                stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)))
            {
                for row in rows.flatten() {
                    by_project.insert(row.0, serde_json::json!(row.1));
                }
            }
        }
        let db_path = dirs::home_dir()
            .unwrap_or_default()
            .join(DB_DIR)
            .join(DB_FILE);
        let size = std::fs::metadata(&db_path).map(|m| m.len()).unwrap_or(0);
        let size_str = if size < 1024 {
            format!("{} B", size)
        } else if size < 1048576 {
            format!("{} KB", size / 1024)
        } else {
            format!("{:.1} MB", size as f64 / 1048576.0)
        };
        let hygiene = self.hygiene_report();

        Ok(
            serde_json::json!({ "total_memories": total, "global_memories": global, "projects": projects,
            "expired_pending": expired, "by_kind": by_kind, "by_project": by_project, "db_size": size_str,
            "hygiene": hygiene }),
        )
    }
    // ─── CONFIG ───────────────────────────────────────

    pub fn get_config(&self, key: &str) -> Option<String> {
        self.conn
            .query_row("SELECT value FROM config WHERE key=?1", params![key], |r| {
                r.get(0)
            })
            .ok()
    }

    pub fn set_config(&self, key: &str, value: &str) -> Result<(), String> {
        self.conn.execute("INSERT INTO config (key,value) VALUES (?1,?2) ON CONFLICT(key) DO UPDATE SET value=?2",
            params![key, value]).map_err(|e| format!("Config: {}", e))?;
        Ok(())
    }

    // ─── GLOBAL PROMPT (auto-scan) ────────────────────

    pub fn get_global_prompt(
        &self,
        project: Option<&str>,
        working_dir: Option<&str>,
    ) -> Option<String> {
        let canonical_project = Self::canonical_project(project);
        let mut prompts: Vec<String> = Vec::new();

        // Helper to read file if modified since last cache, or use cache
        fn get_cached_prompt(path: &std::path::Path) -> Option<String> {
            if !path.exists() {
                return None;
            }
            let metadata = std::fs::metadata(path).ok()?;
            let modified = metadata.modified().ok()?;

            let mut cache = crate::PROMPT_CACHE.lock().unwrap();
            let path_str = path.to_string_lossy().to_string();

            if let Some((last_mod, content)) = cache.get(&path_str) {
                if last_mod == &modified {
                    return Some(content.clone());
                }
            }

            if let Ok(content) = std::fs::read_to_string(path) {
                cache.insert(path_str, (modified, content.clone()));
                Some(content)
            } else {
                None
            }
        }

        // 1. Check configured path
        if let Some(path_str) = self.get_config("global_prompt_path") {
            let path = std::path::Path::new(&path_str);
            if let Some(content) = get_cached_prompt(path) {
                prompts.push(content);
            }
        }

        // 2. Auto-scan ~/.MemoryPilot/GLOBAL_PROMPT.md
        let home_prompt = dirs::home_dir().map(|h| h.join(DB_DIR).join(PROMPT_FILE));
        if let Some(path) = &home_prompt {
            if let Some(content) = get_cached_prompt(path) {
                if !prompts.iter().any(|p| p == &content) {
                    prompts.push(content);
                }
            }
        }

        // 3. Auto-scan project root GLOBAL_PROMPT.md
        let proj_dir: Option<String> = working_dir.map(Self::infer_project_root).or_else(|| {
            let proj_name = canonical_project.as_deref()?;
            let mut stmt = self
                .conn
                .prepare("SELECT path FROM projects WHERE name=?1")
                .ok()?;
            stmt.query_row(params![proj_name], |r| r.get::<_, String>(0))
                .ok()
        });

        if let Some(dir) = proj_dir {
            let proj_prompt = std::path::Path::new(&dir).join(PROMPT_FILE);
            if let Some(content) = get_cached_prompt(&proj_prompt) {
                if !prompts.iter().any(|p| p == &content) {
                    prompts.push(content);
                }
            }
        }

        if prompts.is_empty() {
            None
        } else {
            Some(prompts.join("\n\n---\n\n"))
        }
    }
    // ─── PROJECT CONTEXT ──────────────────────────────

    pub fn backfill_embeddings(&self) -> Result<usize, String> {
        self.backfill_embeddings_inner(false)
    }

    pub fn backfill_embeddings_force(&self) -> Result<usize, String> {
        self.backfill_embeddings_inner(true)
    }

    fn backfill_embeddings_inner(&self, force: bool) -> Result<usize, String> {
        let sql = if force {
            "SELECT id, content, content_hash FROM memories"
        } else {
            "SELECT id, content, content_hash FROM memories WHERE embedding IS NULL"
        };
        let mut stmt = self
            .conn
            .prepare(sql)
            .map_err(|e| format!("Backfill prepare: {}", e))?;

        let rows = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, Option<String>>(2)?,
                ))
            })
            .map_err(|e| format!("Backfill query: {}", e))?;

        let mut to_embed: Vec<(String, String)> = Vec::new();
        let mut skipped = 0usize;
        for r in rows.flatten() {
            let (id, content, existing_hash) = r;
            if !force {
                let new_hash = content_hash(&content);
                if existing_hash.as_deref() == Some(&new_hash) {
                    let has_emb: bool = self
                        .conn
                        .query_row(
                            "SELECT embedding IS NOT NULL FROM memories WHERE id = ?1",
                            params![&id],
                            |r| r.get(0),
                        )
                        .unwrap_or(false);
                    if has_emb {
                        skipped += 1;
                        continue;
                    }
                }
            }
            to_embed.push((id, content));
        }

        if to_embed.is_empty() {
            if skipped > 0 {
                eprintln!(
                    "  Skipped {} memories (content unchanged, embedding exists)",
                    skipped
                );
            }
            return Ok(0);
        }

        eprintln!(
            "  Computing embeddings for {} memories (skipped {} unchanged)...",
            to_embed.len(),
            skipped
        );

        let texts: Vec<&str> = to_embed.iter().map(|(_, c)| c.as_str()).collect();
        let embeddings = crate::embedding::embed_batch(&texts);
        let mut count = 0;
        for ((id, content), emb) in to_embed.iter().zip(embeddings.iter()) {
            let blob = crate::embedding::vec_to_blob(emb);
            let hash = content_hash(content);
            let _ = self.conn.execute(
                "UPDATE memories SET embedding = ?1, content_hash = ?2 WHERE id = ?3",
                params![blob, &hash, id],
            );
            if let Some(ann) = self.ann.as_ref() {
                let _ = ann.add(id, emb);
            }
            count += 1;
        }
        if count > 0 {
            if let Some(ann) = self.ann.as_ref() {
                let _ = ann.persist();
            }
        }
        Ok(count)
    }

    // ─── AAAK-STYLE COMPRESSION ─────────────────────────

    fn compress_memory(mem: &Memory) -> String {
        let kind_short = match mem.kind.as_str() {
            "decision" => "DEC",
            "preference" => "PREF",
            "bug" => "BUG",
            "pattern" => "PAT",
            "credential" => "CRED",
            "snippet" => "SNIP",
            "todo" => "TODO",
            "note" => "NOTE",
            "fact" => "FACT",
            "transcript" => "TXN",
            "architecture" => "ARCH",
            "milestone" => "MILE",
            "problem" => "PROB",
            _ => "MEM",
        };
        let truncated = if mem.content.len() > 200 {
            format!("{}...", &mem.content[..200])
        } else {
            mem.content.clone()
        };
        let tags_str = if mem.tags.is_empty() {
            String::new()
        } else {
            format!(" | tags:{}", mem.tags.join(","))
        };
        let proj_str = mem
            .project
            .as_ref()
            .map(|p| format!(" | proj:{}", p))
            .unwrap_or_default();
        format!(
            "[{}:{}] {}{}{}",
            kind_short,
            mem.importance,
            truncated.replace('\n', " "),
            tags_str,
            proj_str
        )
    }

    fn compress_memories(mems: &[Memory]) -> String {
        mems.iter()
            .map(Self::compress_memory)
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn compress_strings(kind: &str, items: &[String]) -> String {
        if items.is_empty() {
            return String::new();
        }
        let tag = kind.to_uppercase();
        items
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let truncated = if s.len() > 150 {
                    format!("{}...", &s[..150])
                } else {
                    s.clone()
                };
                format!("[{}:{}] {}", tag, i + 1, truncated.replace('\n', " "))
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub fn get_project_brain(
        &self,
        project: &str,
        max_tokens: Option<usize>,
        compact: bool,
    ) -> Result<serde_json::Value, String> {
        let canonical_project =
            Self::canonical_project_name(project).ok_or("Project name cannot be empty")?;
        let max_t = max_tokens.unwrap_or(1500);
        let max_chars = max_t * 4;
        let mut current_chars = 0;

        let mut tech_stack = Vec::new();
        if let Ok(mut stmt) = self.conn.prepare("SELECT DISTINCT entity_value FROM memory_entities e JOIN memories m ON e.memory_id = m.id WHERE m.project = ?1 AND e.entity_kind = 'tech' LIMIT 15") {
            if let Ok(rows) = stmt.query_map(params![&canonical_project], |r| r.get::<_, String>(0)) {
                for tech in rows.flatten() {
                    let len = tech.len();
                    if current_chars + len > max_chars { break; }
                    current_chars += len;
                    tech_stack.push(tech);
                }
            }
        }

        let (core_arch, _) =
            self.list_memories(Some(&canonical_project), Some("architecture"), None, 10, 0)?;
        let mut arch_content = Vec::new();
        for m in core_arch {
            if current_chars + m.content.len() > max_chars {
                break;
            }
            current_chars += m.content.len();
            arch_content.push(m.content);
        }

        let (decisions, _) =
            self.list_memories(Some(&canonical_project), Some("decision"), None, 10, 0)?;
        let mut dec_content = Vec::new();
        for m in decisions {
            if current_chars + m.content.len() > max_chars {
                break;
            }
            current_chars += m.content.len();
            dec_content.push(m.content);
        }

        let (bugs, _) = self.list_memories(Some(&canonical_project), Some("bug"), None, 10, 0)?;
        let mut bug_content = Vec::new();
        for m in bugs {
            if current_chars + m.content.len() > max_chars {
                break;
            }
            current_chars += m.content.len();
            bug_content.push(m.content);
        }

        let mut recent_content = Vec::new();
        if let Ok(mut stmt) = self.conn.prepare("SELECT content FROM memories WHERE project = ?1 AND updated_at > datetime('now','-7 days') ORDER BY updated_at DESC LIMIT 10") {
            if let Ok(rows) = stmt.query_map(params![&canonical_project], |r| r.get::<_, String>(0)) {
                for content in rows.flatten() {
                    if current_chars + content.len() > max_chars { break; }
                    current_chars += content.len();
                    recent_content.push(content);
                }
            }
        }

        let mut key_components = Vec::new();
        if let Ok(mut stmt) = self.conn.prepare("SELECT DISTINCT entity_value FROM memory_entities e JOIN memories m ON e.memory_id = m.id WHERE m.project = ?1 AND e.entity_kind IN ('component', 'file') LIMIT 15") {
            if let Ok(rows) = stmt.query_map(params![&canonical_project], |r| r.get::<_, String>(0)) {
                for comp in rows.flatten() {
                    let len = comp.len();
                    if current_chars + len > max_chars { break; }
                    current_chars += len;
                    key_components.push(comp);
                }
            }
        }

        // Team members (person entities)
        let mut team_members = Vec::new();
        if let Ok(mut stmt) = self.conn.prepare("SELECT DISTINCT entity_value FROM memory_entities e JOIN memories m ON e.memory_id = m.id WHERE m.project = ?1 AND e.entity_kind = 'person' LIMIT 20") {
            if let Ok(rows) = stmt.query_map(params![&canonical_project], |r| r.get::<_, String>(0)) {
                for person in rows.flatten() {
                    team_members.push(person);
                }
            }
        }

        if compact {
            let mut lines = Vec::new();
            lines.push(format!("# {} brain", canonical_project));
            if !tech_stack.is_empty() {
                lines.push(format!("STACK: {}", tech_stack.join(", ")));
            }
            if !key_components.is_empty() {
                lines.push(format!("COMPONENTS: {}", key_components.join(", ")));
            }
            if !team_members.is_empty() {
                lines.push(format!("TEAM: {}", team_members.join(", ")));
            }
            if !arch_content.is_empty() {
                lines.push(Self::compress_strings("ARCH", &arch_content));
            }
            if !dec_content.is_empty() {
                lines.push(Self::compress_strings("DEC", &dec_content));
            }
            if !bug_content.is_empty() {
                lines.push(Self::compress_strings("BUG", &bug_content));
            }
            if !recent_content.is_empty() {
                lines.push(Self::compress_strings("RECENT", &recent_content));
            }
            return Ok(
                serde_json::json!({ "compact": lines.join("\n"), "approx_tokens_used": lines.join("\n").len() / 4 }),
            );
        }

        Ok(serde_json::json!({
            "project": canonical_project,
            "tech_stack": tech_stack,
            "core_architecture": arch_content,
            "current_critical_decisions": dec_content,
            "active_bugs_known": bug_content,
            "recent_changes": recent_content,
            "key_components": key_components,
            "team_members": team_members,
            "approx_tokens_used": current_chars / 4
        }))
    }

    pub fn get_project_context(
        &self,
        project: Option<&str>,
        working_dir: Option<&str>,
        mode: RecallMode,
        scope: &MemoryScope,
    ) -> Result<serde_json::Value, String> {
        let proj_name = match Self::canonical_project(project) {
            Some(p) => Some(p),
            None => match working_dir {
                Some(wd) => self.detect_project(wd)?,
                None => None,
            },
        };
        let proj_ref = proj_name.as_deref();
        let (proj_memories, proj_total) = if let Some(p) = proj_ref {
            let (memories, total) =
                self.list_memories(Some(p), None, Some("transcript"), 100, 0)?;
            (
                memories
                    .into_iter()
                    .filter(|memory| Self::should_include_in_context(memory, mode))
                    .collect::<Vec<_>>(),
                total,
            )
        } else {
            (vec![], 0)
        };
        let (prefs, _) = self.list_memories(None, Some("preference"), None, 50, 0)?;
        let prefs = prefs
            .into_iter()
            .filter(|memory| memory.project.is_none())
            .collect::<Vec<_>>();
        let (patterns, _) = self.list_memories(None, Some("pattern"), None, 50, 0)?;
        let patterns = patterns
            .into_iter()
            .filter(|memory| memory.project.is_none())
            .collect::<Vec<_>>();
        let (snippets, _) = self.list_memories(None, Some("snippet"), None, 20, 0)?;
        let scope_memories = self.list_scope_memories(proj_ref, scope, 20)?;

        Ok(serde_json::json!({
            "mode": mode.as_str(),
            "project": proj_ref.unwrap_or("none"),
            "project_memories": proj_total,
            "global_preferences": prefs.len(),
            "global_patterns": patterns.len(),
            "context": {
                "scope": scope_memories.iter().map(|m| serde_json::json!({"kind":m.kind,"content":m.content,"tags":m.tags,"importance":m.importance})).collect::<Vec<_>>(),
                "project": proj_memories.iter().map(|m| serde_json::json!({"kind":m.kind,"content":m.content,"tags":m.tags,"importance":m.importance})).collect::<Vec<_>>(),
                "preferences": prefs.iter().map(|m| &m.content).collect::<Vec<_>>(),
                "patterns": patterns.iter().map(|m| serde_json::json!({"content":m.content,"tags":m.tags})).collect::<Vec<_>>(),
                "snippets": snippets.iter().map(|m| serde_json::json!({"content":m.content,"tags":m.tags})).collect::<Vec<_>>(),
            }
        }))
    }
    // ─── RECALL (auto-context loader) ─────────────────

    /// One-shot context loader for new conversations.
    /// Combines: project context, global prompt, critical memories, and optional hint search.
    pub fn recall(
        &self,
        project: Option<&str>,
        working_dir: Option<&str>,
        hints: Option<&str>,
        mode: RecallMode,
        explain: bool,
        compact: bool,
        scope: &MemoryScope,
    ) -> Result<serde_json::Value, String> {
        // ~4 chars per token — 800 token budget = 3200 chars for memories
        const TOKEN_BUDGET_CHARS: usize = 3200;

        // Auto-detect project
        let proj_name = match Self::canonical_project(project) {
            Some(p) => Some(p),
            None => match working_dir {
                Some(wd) => self.detect_project(wd)?,
                None => None,
            },
        };
        let proj_ref = proj_name.as_deref();
        let hint_terms: Vec<String> = hints
            .unwrap_or_default()
            .split_whitespace()
            .map(|term: &str| {
                term.trim_matches(|character: char| {
                    !character.is_alphanumeric() && character != '-' && character != '_'
                })
                .to_ascii_lowercase()
            })
            .filter(|term: &String| term.len() > 2)
            .collect();
        let hint_entity_keys = Self::entity_overlap_keys(hints.unwrap_or_default(), proj_ref);

        let mut seen_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut used_chars: usize = 0;
        let link_boosts = if explain {
            self.build_link_boosts()
        } else {
            std::collections::HashMap::new()
        };
        let mut selected_memories_for_explain: Vec<serde_json::Value> = Vec::new();
        let credentials_hidden = if mode.includes_credentials() {
            0
        } else {
            self.conn
                .query_row(
                    "SELECT COUNT(*) FROM memories WHERE kind = 'credential'",
                    [],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap_or(0)
        };

        // Helper: check budget and dedup, returns true if item should be included
        macro_rules! budget_check {
            ($id:expr, $content:expr) => {{
                if seen_ids.contains($id) {
                    false
                } else if used_chars + $content.len() > TOKEN_BUDGET_CHARS {
                    false
                } else {
                    seen_ids.insert($id.to_string());
                    used_chars += $content.len();
                    true
                }
            }};
        }

        // 1. Critical memories first, but keep them project-safe and hint-aware.
        let critical: Vec<Memory> = {
            let has_contextual_hint = !hint_terms.is_empty();
            let mut stmt = self.conn.prepare(
                "SELECT id,content,kind,project,tags,source,importance,expires_at,metadata,created_at,updated_at,last_accessed_at,access_count \
                 FROM memories WHERE importance >= 4 \
                 AND (expires_at IS NULL OR expires_at > datetime('now')) \
                 ORDER BY importance DESC, updated_at DESC LIMIT 30"
            ).map_err(|e| format!("Recall critical: {}", e))?;
            let rows = stmt
                .query_map([], |r| Ok(row_to_memory(r)))
                .map_err(|e| format!("Recall critical: {}", e))?;
            let mut candidates: Vec<(i32, Memory)> = rows
                .flatten()
                .filter(|memory| Self::should_include_in_context(memory, mode))
                .map(|memory| {
                    let lowered_content = memory.content.to_ascii_lowercase();
                    let tag_text = memory.tags.join(" ").to_ascii_lowercase();
                    let hint_overlap = hint_terms
                        .iter()
                        .filter(|term| {
                            lowered_content.contains(term.as_str())
                                || tag_text.contains(term.as_str())
                        })
                        .count() as i32;
                    (memory, hint_overlap)
                })
                .filter(|(memory, hint_overlap)| {
                    if let Some(project_name) = proj_ref {
                        if memory.project.as_deref() == Some(project_name) {
                            return true;
                        }
                        memory.project.is_none() && *hint_overlap > 0
                    } else {
                        true
                    }
                })
                .map(|(memory, hint_overlap)| {
                    let project_match =
                        if proj_ref.is_some() && memory.project.as_deref() == proj_ref {
                            8
                        } else if memory.project.is_none() && hint_overlap > 0 {
                            2
                        } else {
                            0
                        };
                    let score = project_match
                        + (hint_overlap * 4)
                        + (memory.importance * 2)
                        + memory.access_count.min(5);
                    (score, memory)
                })
                .collect();

            candidates.sort_by(|left, right| {
                right
                    .0
                    .cmp(&left.0)
                    .then_with(|| right.1.updated_at.cmp(&left.1.updated_at))
            });

            let critical_limit = if has_contextual_hint {
                3
            } else if proj_ref.is_some() {
                6
            } else {
                12
            };
            candidates
                .into_iter()
                .filter(|(score, _)| *score > 0 || (proj_ref.is_none() && !has_contextual_hint))
                .map(|(_, memory)| memory)
                .take(critical_limit)
                .filter(|memory| budget_check!(&memory.id, &memory.content))
                .collect()
        };

        // 1b. Pinned memories — always included regardless of score/hints
        let pinned_memories: Vec<Memory> = {
            let mut stmt = self.conn.prepare(
                "SELECT id,content,kind,project,tags,source,importance,expires_at,metadata,created_at,updated_at,last_accessed_at,access_count \
                 FROM memories WHERE tags LIKE '%pinned%' ORDER BY importance DESC, updated_at DESC"
            ).map_err(|e| format!("Recall pinned: {}", e))?;
            let result: Vec<Memory> = stmt
                .query_map([], |r| Ok(row_to_memory(r)))
                .map_err(|e| format!("Recall pinned: {}", e))?
                .flatten()
                .filter(|m| Self::should_include_in_context(m, mode))
                .filter(|m| budget_check!(&m.id, &m.content))
                .collect();
            result
        };

        // 2. Hint-based search (most relevant to current task)
        let hint_results: Vec<SearchResult> = if let Some(h) = hints {
            if !h.trim().is_empty() {
                self.search(h, 10, proj_ref, None, None, None)
                    .unwrap_or_default()
                    .into_iter()
                    .filter(|result| result.memory.kind != "transcript")
                    .filter(|result| Self::should_include_in_context(&result.memory, mode))
                    .filter(|r| budget_check!(&r.memory.id, &r.memory.content))
                    .collect()
            } else {
                vec![]
            }
        } else {
            vec![]
        };

        // 3. Scope memories from the same session / thread / window
        let scope_memories = self
            .list_scope_memories(proj_ref, scope, 15)?
            .into_iter()
            .filter(|memory| Self::should_include_in_context(memory, mode))
            .filter(|memory| budget_check!(&memory.id, &memory.content))
            .collect::<Vec<_>>();

        // 4. Project memories (excluding transcripts — too verbose)
        let (proj_memories, proj_total) = if let Some(p) = proj_ref {
            let (all, total) = self.list_memories(Some(p), None, Some("transcript"), 50, 0)?;
            let filtered: Vec<Memory> = all
                .into_iter()
                .filter(|memory| Self::should_include_in_context(memory, mode))
                .filter(|m| budget_check!(&m.id, &m.content))
                .collect();
            (filtered, total)
        } else {
            (vec![], 0)
        };

        // 5. Global preferences + patterns + decisions (with remaining budget)
        let (prefs, _) = self.list_memories(None, Some("preference"), None, 20, 0)?;
        let prefs: Vec<Memory> = Self::select_global_context_memories(
            prefs
                .into_iter()
                .filter(|memory| memory.project.is_none())
                .collect(),
            &hint_terms,
            &hint_entity_keys,
            proj_ref,
            12,
        )
        .into_iter()
        .filter(|m| budget_check!(&m.id, &m.content))
        .collect();

        let (patterns, _) = self.list_memories(None, Some("pattern"), None, 15, 0)?;
        let patterns: Vec<Memory> = Self::select_global_context_memories(
            patterns
                .into_iter()
                .filter(|memory| memory.project.is_none())
                .collect(),
            &hint_terms,
            &hint_entity_keys,
            proj_ref,
            10,
        )
        .into_iter()
        .filter(|m| budget_check!(&m.id, &m.content))
        .collect();

        let (decisions, _) = self.list_memories(None, Some("decision"), None, 15, 0)?;
        let decisions: Vec<Memory> = Self::select_global_context_memories(
            decisions
                .into_iter()
                .filter(|memory| memory.project.is_none())
                .collect(),
            &hint_terms,
            &hint_entity_keys,
            proj_ref,
            10,
        )
        .into_iter()
        .filter(|m| budget_check!(&m.id, &m.content))
        .collect();

        // 6. Global prompt — lazy: only load if budget allows and file exists
        let global_prompt = if used_chars < TOKEN_BUDGET_CHARS {
            self.get_global_prompt(proj_ref, working_dir)
        } else {
            None
        };

        // 7. Stats (cheap, always included)
        let total: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))
            .unwrap_or(0);
        let projects_count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM projects", [], |r| r.get(0))
            .unwrap_or(0);
        let project_context_memories_count = proj_memories.len();

        if explain {
            for result in &hint_results {
                let mut explanation = self.recall_explanation(
                    &result.memory,
                    "hint",
                    proj_ref,
                    mode,
                    Some(result.score),
                    &link_boosts,
                );
                if let Some(object) = explanation.as_object_mut() {
                    object.insert(
                        "retrieval_sources".into(),
                        serde_json::json!(result.sources),
                    );
                }
                selected_memories_for_explain.push(explanation);
            }
            for memory in &scope_memories {
                selected_memories_for_explain.push(self.recall_explanation(
                    memory,
                    "scope",
                    proj_ref,
                    mode,
                    None,
                    &link_boosts,
                ));
            }
            for memory in &proj_memories {
                selected_memories_for_explain.push(self.recall_explanation(
                    memory,
                    "project",
                    proj_ref,
                    mode,
                    None,
                    &link_boosts,
                ));
            }
            for memory in &critical {
                selected_memories_for_explain.push(self.recall_explanation(
                    memory,
                    "critical",
                    proj_ref,
                    mode,
                    None,
                    &link_boosts,
                ));
            }
            for memory in &prefs {
                selected_memories_for_explain.push(self.recall_explanation(
                    memory,
                    "preference",
                    proj_ref,
                    mode,
                    None,
                    &link_boosts,
                ));
            }
            for memory in &patterns {
                selected_memories_for_explain.push(self.recall_explanation(
                    memory,
                    "pattern",
                    proj_ref,
                    mode,
                    None,
                    &link_boosts,
                ));
            }
            for memory in &decisions {
                selected_memories_for_explain.push(self.recall_explanation(
                    memory,
                    "decision",
                    proj_ref,
                    mode,
                    None,
                    &link_boosts,
                ));
            }
        }

        if compact {
            let mut lines = Vec::new();
            lines.push(format!(
                "# recall | proj:{} | mode:{}",
                proj_ref.unwrap_or("none"),
                mode.as_str()
            ));
            if !critical.is_empty() {
                lines.push(format!(
                    "--- critical ---\n{}",
                    Self::compress_memories(&critical)
                ));
            }
            let hint_mems: Vec<Memory> = hint_results.iter().map(|r| r.memory.clone()).collect();
            if !hint_mems.is_empty() {
                lines.push(format!(
                    "--- hints ---\n{}",
                    Self::compress_memories(&hint_mems)
                ));
            }
            if !scope_memories.is_empty() {
                lines.push(format!(
                    "--- scope ---\n{}",
                    Self::compress_memories(&scope_memories)
                ));
            }
            if !proj_memories.is_empty() {
                lines.push(format!(
                    "--- project ---\n{}",
                    Self::compress_memories(&proj_memories)
                ));
            }
            if !prefs.is_empty() {
                lines.push(format!(
                    "--- prefs ---\n{}",
                    Self::compress_memories(&prefs)
                ));
            }
            if !patterns.is_empty() {
                lines.push(format!(
                    "--- patterns ---\n{}",
                    Self::compress_memories(&patterns)
                ));
            }
            if !decisions.is_empty() {
                lines.push(format!(
                    "--- decisions ---\n{}",
                    Self::compress_memories(&decisions)
                ));
            }
            if let Some(gp) = &global_prompt {
                let gp_short = if gp.len() > 500 {
                    format!("{}...", &gp[..500])
                } else {
                    gp.clone()
                };
                lines.push(format!("--- global_prompt ---\n{}", gp_short));
            }
            return Ok(serde_json::json!({
                "status": "recalled",
                "mode": mode.as_str(),
                "project": proj_ref.unwrap_or("none"),
                "compact": lines.join("\n"),
                "stats": { "total_memories": total, "projects": projects_count },
            }));
        }

        let mut response = serde_json::json!({
            "status": "recalled",
            "mode": mode.as_str(),
            "project": proj_ref.unwrap_or("none"),
            "stats": { "total_memories": total, "projects": projects_count, "project_memories": proj_total },
            "hint_results": hint_results.iter().map(|r| serde_json::json!({
                "content": r.memory.content, "kind": r.memory.kind, "project": r.memory.project
            })).collect::<Vec<_>>(),
            "critical_memories": critical.iter().map(|m| serde_json::json!({
                "content": m.content, "kind": m.kind, "project": m.project,
                "importance": m.importance
            })).collect::<Vec<_>>(),
            "pinned_memories": pinned_memories.iter().map(|m| serde_json::json!({
                "id": m.id, "content": m.content, "kind": m.kind, "project": m.project,
                "importance": m.importance
            })).collect::<Vec<_>>(),
            "scope_context": scope_memories.iter().map(|m| serde_json::json!({
                "content": m.content, "kind": m.kind, "importance": m.importance
            })).collect::<Vec<_>>(),
            "project_context": proj_memories.iter().map(|m| serde_json::json!({
                "content": m.content, "kind": m.kind, "importance": m.importance
            })).collect::<Vec<_>>(),
            "preferences": prefs.iter().map(|m| m.content.as_str()).collect::<Vec<_>>(),
            "patterns": patterns.iter().map(|m| m.content.as_str()).collect::<Vec<_>>(),
            "decisions": decisions.iter().map(|m| m.content.as_str()).collect::<Vec<_>>(),
            "global_prompt": global_prompt.as_deref().unwrap_or(""),
        });

        if explain {
            response.as_object_mut().map(|object| {
                object.insert(
                    "explain".into(),
                    serde_json::json!({
                        "budget_chars_used": used_chars,
                        "budget_chars_limit": TOKEN_BUDGET_CHARS,
                        "selected_count": selected_memories_for_explain.len(),
                        "project_detected": proj_ref,
                        "credentials_hidden_in_safe_mode": credentials_hidden,
                        "category_counts": {
                            "critical": critical.len(),
                            "pinned": pinned_memories.len(),
                            "hint": hint_results.len(),
                            "scope": scope_memories.len(),
                            "project": project_context_memories_count,
                            "preferences": prefs.len(),
                            "patterns": patterns.len(),
                            "decisions": decisions.len(),
                        },
                        "selected_memories": selected_memories_for_explain,
                    }),
                );
            });
        }

        Ok(response)
    }

    // ─── IMPORT / MIGRATE ─────────────────────────────

    pub fn import_batch(
        &self,
        memories: &[(String, String, Option<String>, Vec<String>, String)],
    ) -> Result<usize, String> {
        let tx = self
            .conn
            .unchecked_transaction()
            .map_err(|e| format!("Tx: {}", e))?;
        let mut count = 0;
        for (content, kind, project, tags, source) in memories {
            let exists: bool = tx
                .query_row(
                    "SELECT EXISTS(SELECT 1 FROM memories WHERE content=?1)",
                    params![content],
                    |r| r.get(0),
                )
                .unwrap_or(false);
            if exists {
                continue;
            }
            let canonical_project = project.as_deref().and_then(Self::canonical_project_name);
            let id = Uuid::new_v4().to_string();
            let now = Utc::now().to_rfc3339();
            let tags_json = serde_json::to_string(tags).unwrap_or_else(|_| "[]".into());
            let emb = crate::embedding::embed_text(content);
            let emb_blob = crate::embedding::vec_to_blob(&emb);
            tx.execute(
                "INSERT INTO memories (id,content,kind,project,tags,source,importance,embedding,created_at,updated_at,access_count) VALUES (?1,?2,?3,?4,?5,?6,3,?7,?8,?9,0)",
                params![id, content, kind, canonical_project.as_deref(), tags_json, source, emb_blob, now, now],
            ).map_err(|e| format!("Import: {}", e))?;
            let rowid = tx.last_insert_rowid();
            let fts_content = Self::fts_index_content(content);
            tx.execute(
                "INSERT INTO memories_fts (rowid,content,tags,kind,project) VALUES (?1,?2,?3,?4,?5)",
                params![rowid, fts_content, tags_json, kind, canonical_project.as_deref().unwrap_or("")],
            ).map_err(|e| format!("FTS: {}", e))?;
            if let Some(p) = canonical_project.as_deref() {
                let _ = tx.execute(
                    "INSERT OR IGNORE INTO projects (name,path,created_at) VALUES (?1,'',?2)",
                    params![p, now],
                );
            }
            count += 1;
        }
        tx.commit().map_err(|e| format!("Commit: {}", e))?;
        Ok(count)
    }
    pub fn migrate_from_v1(&self) -> Result<usize, String> {
        let v1_dir = dirs::home_dir().ok_or("No home")?.join(DB_DIR);
        let mut batch: Vec<(String, String, Option<String>, Vec<String>, String)> = Vec::new();

        // Load global.json
        let global_path = v1_dir.join("global.json");
        if global_path.exists() {
            if let Ok(content) = std::fs::read_to_string(&global_path) {
                if let Ok(store) = serde_json::from_str::<serde_json::Value>(&content) {
                    if let Some(memories) = store.get("memories").and_then(|v| v.as_array()) {
                        for m in memories {
                            parse_v1_memory(m, None, &mut batch);
                        }
                    }
                }
            }
        }
        // Load projects/*.json
        let projects_dir = v1_dir.join("projects");
        if projects_dir.exists() {
            if let Ok(entries) = std::fs::read_dir(&projects_dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.extension().and_then(|e| e.to_str()) != Some("json") {
                        continue;
                    }
                    let proj_name = path
                        .file_stem()
                        .and_then(|n| n.to_str())
                        .unwrap_or("unknown")
                        .to_string();
                    if let Ok(content) = std::fs::read_to_string(&path) {
                        if let Ok(store) = serde_json::from_str::<serde_json::Value>(&content) {
                            if let Some(memories) = store.get("memories").and_then(|v| v.as_array())
                            {
                                for m in memories {
                                    parse_v1_memory(m, Some(proj_name.clone()), &mut batch);
                                }
                            }
                        }
                    }
                }
            }
        }
        self.import_batch(&batch)
    }
} // end impl Database

// ─── Supporting types ─────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct TranscriptAddReport {
    pub transcript_id: String,
    pub chunks_total: usize,
    pub chunk_added: usize,
    pub chunk_merged: usize,
    pub chunk_skipped: usize,
    pub distilled_candidates: usize,
    pub distilled_added: usize,
    pub distilled_merged: usize,
    pub distilled_skipped: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BulkItem {
    pub content: String,
    #[serde(default = "default_kind")]
    pub kind: String,
    pub project: Option<String>,
    pub tags: Option<Vec<String>>,
    #[serde(default = "default_source")]
    pub source: String,
    pub importance: Option<i32>,
    pub expires_at: Option<String>,
    pub metadata: Option<serde_json::Value>,
    pub session_id: Option<String>,
    pub thread_id: Option<String>,
    pub window_id: Option<String>,
}
fn default_kind() -> String {
    "fact".into()
}
fn default_source() -> String {
    "cursor".into()
}

impl BulkItem {
    fn scope(&self) -> MemoryScope {
        MemoryScope {
            session_id: self.session_id.clone(),
            thread_id: self.thread_id.clone(),
            window_id: self.window_id.clone(),
        }
    }
}

// ─── Row helper ───────────────────────────────────

fn row_to_memory(row: &rusqlite::Row) -> Memory {
    let tags_str: String = row.get(4).unwrap_or_default();
    let tags: Vec<String> = serde_json::from_str(&tags_str).unwrap_or_default();
    let meta_str: Option<String> = row.get(8).unwrap_or(None);
    let metadata = meta_str.and_then(|s| serde_json::from_str(&s).ok());
    Memory {
        id: row.get(0).unwrap_or_default(),
        content: row.get(1).unwrap_or_default(),
        kind: row.get(2).unwrap_or_default(),
        project: row.get(3).unwrap_or(None),
        tags,
        source: row.get(5).unwrap_or_default(),
        importance: row.get(6).unwrap_or(3),
        expires_at: row.get(7).unwrap_or(None),
        metadata,
        created_at: row.get(9).unwrap_or_default(),
        updated_at: row.get(10).unwrap_or_default(),
        last_accessed_at: row.get(11).unwrap_or(None),
        access_count: row.get(12).unwrap_or(0),
    }
}

fn parse_v1_memory(
    m: &serde_json::Value,
    project: Option<String>,
    batch: &mut Vec<(String, String, Option<String>, Vec<String>, String)>,
) {
    let c = m
        .get("content")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    if c.is_empty() {
        return;
    }
    let k = m
        .get("kind")
        .or(m.get("type"))
        .and_then(|v| v.as_str())
        .unwrap_or("fact");
    let kind = match k {
        "context" => "fact",
        "architecture" => "decision",
        "component" | "workflow" => "pattern",
        o => o,
    }
    .to_string();
    let tags: Vec<String> = m
        .get("tags")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    let source = m
        .get("source")
        .and_then(|v| v.as_str())
        .unwrap_or("v1-import")
        .to_string();
    batch.push((c, kind, project, tags, source));
}

#[cfg(test)]
mod tests {
    use super::{Database, MemoryScope, RecallMode};
    use rusqlite::params;
    use std::path::PathBuf;
    use uuid::Uuid;

    fn temp_db_path(test_name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("memory-pilot-{test_name}-{}.db", Uuid::new_v4()))
    }

    fn cleanup_db_files(path: &PathBuf) {
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(format!("{}-wal", path.display()));
        let _ = std::fs::remove_file(format!("{}-shm", path.display()));
    }

    #[test]
    fn normalize_project_identity_merges_case_variants() {
        let path = temp_db_path("normalize-project-identity");
        {
            let db = Database::open_at(&path).expect("db");
            let now = chrono::Utc::now().to_rfc3339();

            db.conn
                .execute(
                    "INSERT INTO projects (name, path, created_at) VALUES (?1, '', ?2)",
                    params!["Planify", &now],
                )
                .expect("insert project");
            db.conn
                .execute(
                    "INSERT INTO memories (id, content, kind, project, tags, source, importance, created_at, updated_at, access_count)
                     VALUES (?1, ?2, 'fact', 'Planify', '[]', 'test', 3, ?3, ?3, 0)",
                    params![Uuid::new_v4().to_string(), "Uppercase project", &now],
                )
                .expect("insert memory");
            db.conn
                .execute(
                    "INSERT INTO memories_fts (rowid, content, tags, kind, project)
                     SELECT rowid, content, tags, kind, project FROM memories WHERE project = 'Planify'",
                    [],
                )
                .expect("insert fts");

            db.normalize_project_identities().expect("normalize");

            let projects = db.list_projects().expect("list projects");
            assert_eq!(projects.len(), 1);
            assert_eq!(projects[0].name, "planify");

            let count: i64 = db
                .conn
                .query_row(
                    "SELECT COUNT(*) FROM memories WHERE project = 'planify'",
                    [],
                    |row| row.get(0),
                )
                .expect("count memories");
            assert_eq!(count, 1);
        }
        cleanup_db_files(&path);
    }

    #[test]
    fn detect_project_backfills_missing_path_from_root() {
        let path = temp_db_path("detect-project-root");
        let project_root = std::env::temp_dir().join(format!("planify-root-{}", Uuid::new_v4()));
        let nested_dir = project_root.join("apps").join("web");

        std::fs::create_dir_all(&nested_dir).expect("mkdir");
        std::fs::write(
            project_root.join("Cargo.toml"),
            "[package]\nname = \"planify\"\nversion = \"0.1.0\"\n",
        )
        .expect("write cargo");

        {
            let db = Database::open_at(&path).expect("db");
            let empty_tags: Vec<String> = Vec::new();
            let _ = db
                .add_memory(
                    "Planify architecture note",
                    "fact",
                    Some("Planify"),
                    &empty_tags,
                    "test",
                    3,
                    None,
                    None,
                    &MemoryScope::default(),
                )
                .expect("add memory");

            let detected = db
                .detect_project(&nested_dir.to_string_lossy())
                .expect("detect");
            assert_eq!(detected.as_deref(), Some("planify"));

            let stored_path: String = db
                .conn
                .query_row(
                    "SELECT path FROM projects WHERE name = 'planify'",
                    [],
                    |row| row.get(0),
                )
                .expect("stored path");
            assert_eq!(
                stored_path,
                Database::normalize_path(&project_root.to_string_lossy())
            );
        }
        let _ = std::fs::remove_dir_all(&project_root);
        cleanup_db_files(&path);
    }

    #[test]
    fn recall_safe_excludes_credentials_but_full_keeps_them() {
        let path = temp_db_path("recall-safe");
        {
            let db = Database::open_at(&path).expect("db");
            let empty_tags: Vec<String> = Vec::new();

            let _ = db
                .add_memory(
                    "API token is sk_live_secret",
                    "credential",
                    Some("Planify"),
                    &empty_tags,
                    "test",
                    5,
                    None,
                    None,
                    &MemoryScope::default(),
                )
                .expect("add credential");
            let _ = db
                .add_memory(
                    "Planify uses SvelteKit",
                    "decision",
                    Some("planify"),
                    &empty_tags,
                    "test",
                    4,
                    None,
                    None,
                    &MemoryScope::default(),
                )
                .expect("add decision");

            let safe = db
                .recall(
                    Some("Planify"),
                    None,
                    Some("token"),
                    RecallMode::Safe,
                    false,
                    false,
                    &MemoryScope::default(),
                )
                .expect("safe recall");
            let full = db
                .recall(
                    Some("Planify"),
                    None,
                    Some("token"),
                    RecallMode::Full,
                    false,
                    false,
                    &MemoryScope::default(),
                )
                .expect("full recall");

            let safe_text = serde_json::to_string(&safe).expect("safe json");
            let full_text = serde_json::to_string(&full).expect("full json");
            assert!(!safe_text.contains("sk_live_secret"));
            assert!(full_text.contains("sk_live_secret"));
        }
        cleanup_db_files(&path);
    }

    #[test]
    fn recall_explain_includes_selection_reasons() {
        let path = temp_db_path("recall-explain");
        {
            let db = Database::open_at(&path).expect("db");
            let empty_tags: Vec<String> = Vec::new();

            let _ = db
                .add_memory(
                    "Planify uses SvelteKit and Rust",
                    "decision",
                    Some("Planify"),
                    &empty_tags,
                    "test",
                    5,
                    None,
                    None,
                    &MemoryScope::default(),
                )
                .expect("add decision");

            let explained = db
                .recall(
                    Some("Planify"),
                    None,
                    Some("SvelteKit"),
                    RecallMode::Safe,
                    true,
                    false,
                    &MemoryScope::default(),
                )
                .expect("explained recall");

            let explain = explained.get("explain").expect("explain block");
            assert!(explain
                .get("selected_memories")
                .and_then(|value| value.as_array())
                .map(|items| !items.is_empty())
                .unwrap_or(false));
            let rendered = serde_json::to_string(explain).expect("render explain");
            assert!(rendered.contains("selection_source"));
            assert!(rendered.contains("graph_boost"));
            assert!(rendered.contains("importance_weight"));
        }
        cleanup_db_files(&path);
    }

    #[test]
    fn gc_preview_returns_candidates_and_hygiene() {
        let path = temp_db_path("gc-preview");
        {
            let db = Database::open_at(&path).expect("db");
            let now = (chrono::Utc::now() - chrono::Duration::days(90)).to_rfc3339();

            for idx in 0..2 {
                let id = Uuid::new_v4().to_string();
                db.conn
                    .execute(
                        "INSERT INTO memories (id, content, kind, project, tags, source, importance, created_at, updated_at, access_count)
                         VALUES (?1, ?2, 'note', 'planify', '[]', 'test', 1, ?3, ?3, 0)",
                        params![id, format!("Old note {}", idx), &now],
                    )
                    .expect("insert gc memory");
                db.conn
                    .execute(
                        "INSERT INTO memories_fts (rowid, content, tags, kind, project)
                         SELECT rowid, content, tags, kind, project FROM memories WHERE id = ?1",
                        params![id],
                    )
                    .expect("insert gc fts");
            }

            let report = db
                .run_gc(&crate::gc::GcConfig::default(), true)
                .expect("gc preview");
            assert!(report.preview_mode);
            assert!(!report.preview_candidates.is_empty());
            assert!(report.hygiene.never_accessed_memories >= 2);
        }
        cleanup_db_files(&path);
    }

    #[test]
    fn graph_links_stay_within_project_scope() {
        let path = temp_db_path("graph-scope");
        {
            let db = Database::open_at(&path).expect("db");
            let empty_tags: Vec<String> = Vec::new();

            let (planify_a, _) = db
                .add_memory(
                    "Planify bug in src/routes/settings.ts for SettingsPanel component",
                    "bug",
                    Some("Planify"),
                    &empty_tags,
                    "test",
                    4,
                    None,
                    None,
                    &MemoryScope::default(),
                )
                .expect("add planify a");
            let (onlyst, _) = db
                .add_memory(
                    "Onlyst bug in src/routes/settings.ts for SettingsPanel component",
                    "bug",
                    Some("Onlyst"),
                    &empty_tags,
                    "test",
                    4,
                    None,
                    None,
                    &MemoryScope::default(),
                )
                .expect("add onlyst");
            let (planify_b, _) = db
                .add_memory(
                    "Planify fix touches src/routes/settings.ts and SettingsPanel rendering",
                    "decision",
                    Some("planify"),
                    &empty_tags,
                    "test",
                    4,
                    None,
                    None,
                    &MemoryScope::default(),
                )
                .expect("add planify b");

            let same_project_links: i64 = db
                .conn
                .query_row(
                    "SELECT COUNT(*) FROM memory_links WHERE source_id = ?1 AND target_id = ?2",
                    params![&planify_a.id, &planify_b.id],
                    |row| row.get(0),
                )
                .expect("same project links");
            let cross_project_links: i64 = db
                .conn
                .query_row(
                    "SELECT COUNT(*) FROM memory_links WHERE source_id = ?1 AND target_id = ?2",
                    params![&planify_a.id, &onlyst.id],
                    |row| row.get(0),
                )
                .expect("cross project links");

            assert!(same_project_links > 0);
            assert_eq!(cross_project_links, 0);
        }
        cleanup_db_files(&path);
    }

    #[test]
    fn recall_scope_prioritizes_same_thread_memories() {
        let path = temp_db_path("scope-recall");
        {
            let db = Database::open_at(&path).expect("db");
            let empty_tags: Vec<String> = Vec::new();
            let scope = MemoryScope {
                session_id: Some("session-a".into()),
                thread_id: Some("thread-a".into()),
                window_id: Some("window-a".into()),
            };

            let _ = db
                .add_memory(
                    "Planify thread-specific architecture note",
                    "decision",
                    Some("planify"),
                    &empty_tags,
                    "test",
                    3,
                    None,
                    None,
                    &scope,
                )
                .expect("add scoped memory");
            let _ = db
                .add_memory(
                    "Planify generic architecture note",
                    "decision",
                    Some("planify"),
                    &empty_tags,
                    "test",
                    5,
                    None,
                    None,
                    &MemoryScope::default(),
                )
                .expect("add generic memory");

            let recall = db
                .recall(
                    Some("planify"),
                    None,
                    None,
                    RecallMode::Safe,
                    true,
                    false,
                    &scope,
                )
                .expect("scoped recall");

            let scope_context = recall
                .get("scope_context")
                .and_then(|value| value.as_array())
                .expect("scope context");
            assert!(!scope_context.is_empty());

            let explain_text = serde_json::to_string(recall.get("explain").expect("explain"))
                .expect("render explain");
            assert!(explain_text.contains("\"selection_source\":\"scope\""));
        }
        cleanup_db_files(&path);
    }

    #[test]
    fn benchmark_recall_reports_core_metrics() {
        let path = temp_db_path("benchmark-recall");
        {
            let db = Database::open_at(&path).expect("db");
            let empty_tags: Vec<String> = Vec::new();

            let _ = db
                .add_memory(
                    "Planify uses SvelteKit and Rust for the settings dashboard",
                    "decision",
                    Some("planify"),
                    &empty_tags,
                    "test",
                    5,
                    None,
                    None,
                    &MemoryScope::default(),
                )
                .expect("add planify");
            let _ = db
                .add_memory(
                    "Onlyst uses Swift and Supabase for member onboarding",
                    "decision",
                    Some("onlyst"),
                    &empty_tags,
                    "test",
                    5,
                    None,
                    None,
                    &MemoryScope::default(),
                )
                .expect("add onlyst");
            let _ = db
                .add_memory(
                    "MemoryPilot keeps safe mode by default and hides credentials unless full mode is explicitly requested.",
                    "decision",
                    Some("memorypilot"),
                    &empty_tags,
                    "test",
                    5,
                    None,
                    None,
                    &MemoryScope::default(),
                )
                .expect("add memorypilot preference");
            let _ = db
                .add_memory(
                    "MemoryPilot adds a benchmark_recall CLI command to measure top1 and top5 hit rate on recall quality.",
                    "decision",
                    Some("memorypilot"),
                    &empty_tags,
                    "test",
                    5,
                    None,
                    None,
                    &MemoryScope::default(),
                )
                .expect("add memorypilot benchmark decision");

            let report = db.benchmark_recall(10).expect("benchmark");
            assert!(
                report
                    .get("scenario_count")
                    .and_then(|value| value.as_u64())
                    .unwrap_or(0)
                    >= 2
            );
            assert!(
                report
                    .get("golden_defined_count")
                    .and_then(|value| value.as_u64())
                    .unwrap_or(0)
                    >= 10
            );
            assert!(report
                .get("top5_hit_rate")
                .and_then(|value| value.as_f64())
                .is_some());
            assert!(report
                .get("cross_project_leak_rate")
                .and_then(|value| value.as_f64())
                .is_some());
            assert!(report
                .get("credential_leak_rate_safe")
                .and_then(|value| value.as_f64())
                .is_some());
            assert!(
                report
                    .get("scenario_source_counts")
                    .and_then(|value| value.get("golden"))
                    .and_then(|value| value.as_u64())
                    .unwrap_or(0)
                    >= 1
            );
        }
        cleanup_db_files(&path);
    }

    #[test]
    fn benchmark_recall_uses_golden_scenarios_when_available() {
        let path = temp_db_path("benchmark-golden");
        {
            let db = Database::open_at(&path).expect("db");
            let empty_tags: Vec<String> = Vec::new();

            let _ = db
                .add_memory(
                    "MemoryPilot keeps safe mode by default and hides credentials unless full mode is explicitly requested.",
                    "decision",
                    Some("memorypilot"),
                    &empty_tags,
                    "test",
                    5,
                    None,
                    None,
                    &MemoryScope::default(),
                )
                .expect("add safe mode preference");
            let _ = db
                .add_memory(
                    "MemoryPilot adds a benchmark_recall CLI command to measure top1 and top5 hit rate on recall quality.",
                    "decision",
                    Some("memorypilot"),
                    &empty_tags,
                    "test",
                    5,
                    None,
                    None,
                    &MemoryScope::default(),
                )
                .expect("add benchmark decision");

            let report = db.benchmark_recall(12).expect("benchmark");
            let golden_count = report
                .get("scenario_source_counts")
                .and_then(|value| value.get("golden"))
                .and_then(|value| value.as_u64())
                .unwrap_or(0);
            assert!(golden_count >= 2);
            assert!(
                report
                    .get("golden_executed_count")
                    .and_then(|value| value.as_u64())
                    .unwrap_or(0)
                    >= 2
            );
            assert!(report
                .get("scenarios")
                .and_then(|value| value.as_array())
                .map(|items| items.iter().any(|item| item
                    .get("scenario_source")
                    .and_then(|value| value.as_str())
                    == Some("golden")))
                .unwrap_or(false));
        }
        cleanup_db_files(&path);
    }

    #[test]
    fn add_transcript_distills_structured_memories() {
        let path = temp_db_path("transcript-distill");
        {
            let db = Database::open_at(&path).expect("db");
            let scope = MemoryScope {
                session_id: Some("session-1".into()),
                thread_id: Some("thread-1".into()),
                window_id: Some("window-1".into()),
            };
            let tags = vec!["chat".to_string()];
            let transcript = "\
USER: We should always keep safe mode by default and never expose credentials in recall.\n\
ASSISTANT: MemoryPilot will keep safe mode by default and only expose credentials in full mode.\n\
USER: Add a benchmark_recall command to measure top1 and top5 hit rate.\n\
ASSISTANT: We'll add a benchmark_recall tool and a --benchmark-recall CLI command.\n\
USER: There is a bug in src/db.rs where cross-project leakage pollutes recall results.\n";

            let report = db
                .add_transcript(transcript, Some("MemoryPilot"), &tags, "test", &scope, true)
                .expect("ingest transcript");

            assert!(report.chunks_total >= 1);
            assert!(report.distilled_candidates >= 2);
            assert!(report.distilled_added >= 2);

            let (memories, _) = db
                .list_memories(Some("memorypilot"), None, Some("transcript"), 20, 0)
                .expect("list memories");

            assert!(memories
                .iter()
                .any(|memory| memory.kind == "preference" && memory.content.contains("safe mode")));
            assert!(memories
                .iter()
                .any(|memory| memory.kind == "decision"
                    && memory.content.contains("benchmark_recall")));
            assert!(memories
                .iter()
                .any(|memory| memory.kind == "bug"
                    && memory.content.contains("cross-project leakage")));
            assert!(memories.iter().all(|memory| memory.kind != "transcript"));
        }
        cleanup_db_files(&path);
    }

    #[test]
    fn distilled_transcript_memories_improve_recall_without_loading_chunks() {
        let path = temp_db_path("transcript-recall");
        {
            let db = Database::open_at(&path).expect("db");
            let scope = MemoryScope {
                session_id: Some("session-2".into()),
                thread_id: Some("thread-2".into()),
                window_id: Some("window-2".into()),
            };
            let transcript = "\
USER: We should always keep safe mode by default and never expose credentials in recall.\n\
ASSISTANT: MemoryPilot will keep safe mode by default and only expose credentials in full mode.\n\
USER: Add a benchmark_recall command to measure top1 and top5 hit rate.\n\
ASSISTANT: We'll add a benchmark_recall tool and a --benchmark-recall CLI command.\n";

            let report = db
                .add_transcript(transcript, Some("MemoryPilot"), &[], "test", &scope, true)
                .expect("ingest transcript");
            assert!(report.distilled_added >= 1);

            let recall = db
                .recall(
                    Some("memorypilot"),
                    None,
                    Some("benchmark_recall safe mode credentials"),
                    RecallMode::Safe,
                    true,
                    false,
                    &scope,
                )
                .expect("recall");

            let rendered = serde_json::to_string(&recall).expect("render recall");
            assert!(rendered.contains("benchmark_recall") || rendered.contains("safe mode"));
            assert!(!rendered.contains("\"kind\":\"transcript\""));
        }
        cleanup_db_files(&path);
    }

    #[test]
    fn ingest_session_distills_without_raw_chunks_by_default() {
        let path = temp_db_path("session-ingest");
        {
            let db = Database::open_at(&path).expect("db");
            let scope = MemoryScope {
                session_id: Some("session-ingest-1".into()),
                thread_id: Some("thread-ingest-1".into()),
                window_id: None,
            };
            let transcript = "\
USER: We should always keep MemoryPilot as the only MCP server through MCP Hub.\n\
ASSISTANT: MemoryPilot will stay the single MCP server and expose new capabilities as tools.\n\
USER: There is a bug in src/tools.rs where duplicate MCP servers would confuse setup.\n";

            let report = db
                .ingest_session_transcript(
                    transcript,
                    Some("MemoryPilot"),
                    &["session-ingest".to_string()],
                    "test-session",
                    &scope,
                    false,
                )
                .expect("ingest session");

            assert_eq!(report.chunks_total, 0);
            assert_eq!(report.chunk_added, 0);
            assert!(report.distilled_added >= 1);

            let (memories, _) = db
                .list_memories(Some("memorypilot"), None, None, 20, 0)
                .expect("list memories");
            assert!(memories.iter().all(|memory| memory.kind != "transcript"));
            assert!(memories.iter().any(|memory| {
                memory.content.contains("only MCP server")
                    || memory.content.contains("duplicate MCP servers")
            }));
        }
        cleanup_db_files(&path);
    }

    #[test]
    fn search_reports_lexical_candidate_sources() {
        let path = temp_db_path("search-sources");
        {
            let db = Database::open_at(&path).expect("db");
            let tags = vec!["SettingsPanel".to_string(), "render".to_string()];
            let _ = db
                .add_memory(
                    "SettingsPanel render bug happens in src/routes/settings.ts when the modal opens.",
                    "bug",
                    Some("MemoryPilot"),
                    &tags,
                    "test",
                    4,
                    None,
                    None,
                    &MemoryScope::default(),
                )
                .expect("add memory");

            let results = db
                .search(
                    "SettingsPanel render bug",
                    5,
                    Some("memorypilot"),
                    Some("bug"),
                    None,
                    None,
                )
                .expect("search");

            assert!(!results.is_empty());
            assert!(results[0]
                .sources
                .iter()
                .any(|source| source.starts_with("fts_")));
        }
        cleanup_db_files(&path);
    }
}
