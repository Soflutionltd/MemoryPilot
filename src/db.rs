/// MemoryPilot v4.0 Database Engine — SQLite + FTS5.
/// Features: dedup, importance, TTL, bulk ops, export, auto-prompt, lazy embedding, content hash.
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use chrono::Utc;

const DB_DIR: &str = ".MemoryPilot";
const DB_FILE: &str = "memory.db";
const PROMPT_FILE: &str = "GLOBAL_PROMPT.md";
const DEDUP_THRESHOLD: f64 = 0.85;
const EMBED_CACHE_SIZE: usize = 64;

fn content_hash(text: &str) -> String {
    let mut h: u64 = 14695981039346656037;
    for b in text.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(1099511628211);
    }
    format!("{:016x}", h)
}

// ─── LAZY EMBEDDING QUEUE ────────────────────────────
// add_memory inserts with embedding=NULL and pushes a job here.
// A background thread drains the queue and writes embeddings to DB.

struct EmbedJob {
    id: String,
    content: String,
}

static EMBED_QUEUE: OnceLock<Mutex<Vec<EmbedJob>>> = OnceLock::new();
static EMBED_DB_PATH: OnceLock<PathBuf> = OnceLock::new();

fn embed_queue() -> &'static Mutex<Vec<EmbedJob>> {
    EMBED_QUEUE.get_or_init(|| Mutex::new(Vec::new()))
}

fn queue_embedding_job(id: &str, content: &str) {
    if let Ok(mut q) = embed_queue().lock() {
        q.push(EmbedJob { id: id.to_string(), content: content.to_string() });
    }
    // Signal the background thread (fire-and-forget)
    ensure_embed_worker();
}

static EMBED_WORKER_STARTED: OnceLock<()> = OnceLock::new();

fn ensure_embed_worker() {
    EMBED_WORKER_STARTED.get_or_init(|| {
        std::thread::Builder::new()
            .name("embed-worker".into())
            .spawn(embed_worker_loop)
            .ok();
    });
}

fn embed_worker_loop() {
    loop {
        std::thread::sleep(std::time::Duration::from_millis(100));

        let jobs: Vec<EmbedJob> = {
            let mut q = match embed_queue().lock() {
                Ok(q) => q,
                Err(_) => continue,
            };
            q.drain(..).collect()
        };

        if jobs.is_empty() {
            std::thread::sleep(std::time::Duration::from_millis(500));
            continue;
        }

        let db_path = match EMBED_DB_PATH.get() {
            Some(p) => p.clone(),
            None => continue,
        };

        let conn = match Connection::open(&db_path) {
            Ok(c) => c,
            Err(_) => continue,
        };
        let _ = conn.execute_batch("PRAGMA journal_mode = WAL; PRAGMA synchronous = NORMAL;");

        let texts: Vec<&str> = jobs.iter().map(|j| j.content.as_str()).collect();
        let embeddings = crate::embedding::embed_batch(&texts);

        for (job, emb) in jobs.iter().zip(embeddings.iter()) {
            let blob = crate::embedding::vec_to_blob(emb);
            let hash = content_hash(&job.content);
            let _ = conn.execute(
                "UPDATE memories SET embedding = ?1, content_hash = ?2 WHERE id = ?3 AND embedding IS NULL",
                params![blob, &hash, &job.id],
            );
        }
    }
}

// ─── EMBEDDING CACHE (LRU) ──────────────────────────
// Caches query embeddings so repeated searches don't recompute.

struct EmbedCache {
    entries: Vec<(String, Vec<f32>)>,
}

impl EmbedCache {
    fn new() -> Self {
        Self { entries: Vec::with_capacity(EMBED_CACHE_SIZE) }
    }

    fn get(&self, text: &str) -> Option<&Vec<f32>> {
        self.entries.iter().find(|(k, _)| k == text).map(|(_, v)| v)
    }

    fn insert(&mut self, text: String, emb: Vec<f32>) {
        if let Some(pos) = self.entries.iter().position(|(k, _)| k == &text) {
            self.entries.remove(pos);
        }
        if self.entries.len() >= EMBED_CACHE_SIZE {
            self.entries.remove(0);
        }
        self.entries.push((text, emb));
    }
}

static EMBED_CACHE: OnceLock<Mutex<EmbedCache>> = OnceLock::new();

fn embed_cache() -> &'static Mutex<EmbedCache> {
    EMBED_CACHE.get_or_init(|| Mutex::new(EmbedCache::new()))
}

fn cached_embed_text(text: &str) -> Vec<f32> {
    if let Ok(cache) = embed_cache().lock() {
        if let Some(emb) = cache.get(text) {
            return emb.clone();
        }
    }
    let emb = crate::embedding::embed_text(text);
    if let Ok(mut cache) = embed_cache().lock() {
        cache.insert(text.to_string(), emb.clone());
    }
    emb
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
            other => Err(format!("Invalid recall mode '{}'. Use safe, default, or full.", other)),
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

const READ_POOL_SIZE: usize = 4;

pub struct Database {
    conn: Connection,
    read_pool: Vec<Mutex<Connection>>,
}

impl Database {
    pub fn open() -> Result<Self, String> {
        let dir = dirs::home_dir().ok_or("Cannot find home directory")?.join(DB_DIR);
        std::fs::create_dir_all(&dir).map_err(|e| format!("Cannot create dir: {}", e))?;
        Self::open_at(&dir.join(DB_FILE))
    }

    pub fn open_at(path: &Path) -> Result<Self, String> {
        let conn = Connection::open(path).map_err(|e| format!("SQLite open: {}", e))?;
        conn.execute_batch("
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = NORMAL;
            PRAGMA cache_size = -8000;
            PRAGMA foreign_keys = ON;
        ").map_err(|e| format!("Pragma: {}", e))?;
        let _ = EMBED_DB_PATH.set(path.to_path_buf());

        let mut read_pool = Vec::with_capacity(READ_POOL_SIZE);
        for _ in 0..READ_POOL_SIZE {
            let rc = Connection::open_with_flags(path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX)
                .map_err(|e| format!("Read pool open: {}", e))?;
            let _ = rc.execute_batch("PRAGMA cache_size = -4000;");
            read_pool.push(Mutex::new(rc));
        }

        let db = Self { conn, read_pool };
        db.init_schema()?;
        db.upgrade_schema()?;
        db.normalize_project_identities()?;
        let _ = db.backfill_embeddings();
        Ok(db)
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
                            if let Some(slug) = Self::canonical_project_name(raw_value.trim().trim_matches('"')) {
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

    fn metadata_object(metadata: Option<&serde_json::Value>) -> serde_json::Map<String, serde_json::Value> {
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

    fn merge_metadata(base: Option<&serde_json::Value>, overlay: Option<&serde_json::Value>) -> Option<serde_json::Value> {
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

    fn apply_scope_to_metadata(metadata: Option<&serde_json::Value>, scope: &MemoryScope) -> Option<serde_json::Value> {
        let mut object = Self::metadata_object(metadata);
        if let Some(session_id) = &scope.session_id {
            object.insert("session_id".into(), serde_json::Value::String(session_id.clone()));
        }
        if let Some(thread_id) = &scope.thread_id {
            object.insert("thread_id".into(), serde_json::Value::String(thread_id.clone()));
        }
        if let Some(window_id) = &scope.window_id {
            object.insert("window_id".into(), serde_json::Value::String(window_id.clone()));
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
            session_id: object.get("session_id").and_then(|value| value.as_str()).map(String::from),
            thread_id: object.get("thread_id").and_then(|value| value.as_str()).map(String::from),
            window_id: object.get("window_id").and_then(|value| value.as_str()).map(String::from),
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

    fn list_scope_memories(&self, project: Option<&str>, scope: &MemoryScope, limit: usize) -> Result<Vec<Memory>, String> {
        if scope.is_empty() {
            return Ok(Vec::new());
        }

        let candidate_limit = limit.max(50).min(200);
        let (memories, _) = self.list_memories(project, None, Some("transcript"), candidate_limit, 0)?;
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

        Ok(scoped.into_iter().take(limit).map(|(_, memory)| memory).collect())
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

    fn entity_overlap_keys(content: &str, project: Option<&str>) -> std::collections::HashSet<String> {
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
        let entity_overlap = crate::graph::extract_entities(&memory.content, memory.project.as_deref())
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
            if project.is_some() && hint_terms.len() >= 2 { 8 } else { 5 }
        } else {
            0
        };

        kind_bias + (hint_overlap * 4) + (entity_overlap * 3) + access_bonus + recency_bonus + project_bonus - generic_penalty
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

    fn build_link_boosts_for(&self, candidate_ids: &[&String]) -> std::collections::HashMap<String, f64> {
        let mut link_boosts = std::collections::HashMap::new();
        let mut rows_data: Vec<(String, String)> = Vec::new();

        if candidate_ids.is_empty() || candidate_ids.len() > 200 {
            if let Ok(mut stmt) = self.conn.prepare("SELECT target_id, relation_type FROM memory_links") {
                if let Ok(rows) = stmt.query_map([], |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))) {
                    for r in rows.flatten() { rows_data.push(r); }
                }
            }
        } else {
            let placeholders: Vec<String> = (1..=candidate_ids.len()).map(|i| format!("?{}", i)).collect();
            let sql = format!("SELECT target_id, relation_type FROM memory_links WHERE target_id IN ({})", placeholders.join(","));
            if let Ok(mut stmt) = self.conn.prepare(&sql) {
                let param_refs: Vec<&dyn rusqlite::types::ToSql> = candidate_ids.iter().map(|id| id as &dyn rusqlite::types::ToSql).collect();
                if let Ok(rows) = stmt.query_map(param_refs.as_slice(), |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))) {
                    for r in rows.flatten() { rows_data.push(r); }
                }
            }
        }

        for (target_id, relation_type) in rows_data {
            let boost: f64 = match relation_type.as_str() {
                "deprecates" => -0.6,
                "depends_on" | "implements" | "resolves" | "resolved_by" | "fixed_by" | "fixes" => 0.08,
                _ => 0.03,
            };
            let total = link_boosts.entry(target_id).or_insert(0.0);
            *total = (*total + boost).clamp(-0.8_f64, 0.25_f64);
        }
        link_boosts
    }

    fn get_kg_expansion_terms(&self, query: &str) -> Vec<String> {
        let words: Vec<&str> = query.split_whitespace().collect();
        if words.len() > 15 { return Vec::new(); }

        let mut terms: Vec<String> = Vec::new();
        let query_lower = query.to_lowercase();

        for word in &words {
            if word.len() < 3 { continue; }
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
        if ids.is_empty() { return adj; }
        let placeholders: Vec<String> = (1..=ids.len()).map(|i| format!("?{}", i)).collect();
        let sql = format!(
            "SELECT source_id, target_id FROM memory_links WHERE source_id IN ({0}) AND target_id IN ({0})",
            placeholders.join(",")
        );
        if let Ok(mut stmt) = self.conn.prepare(&sql) {
            // Bind each id twice (once for source_id IN, once for target_id IN)
            let mut params: Vec<&dyn rusqlite::types::ToSql> = Vec::with_capacity(ids.len());
            for id in ids { params.push(id as &dyn rusqlite::types::ToSql); }
            if let Ok(rows) = stmt.query_map(params.as_slice(), |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
            }) {
                for row in rows.flatten() { adj.insert(row); }
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
            for id in ids { params.push(id as &dyn rusqlite::types::ToSql); }
            if let Ok(rows) = stmt.query_map(params.as_slice(), |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
            }) {
                for row in rows.flatten() { adj.insert(row); }
            }
        }
        adj
    }

    fn batch_triple_counts(&self, candidate_ids: &[&String]) -> std::collections::HashMap<String, (i64, i64)> {
        let mut counts: std::collections::HashMap<String, (i64, i64)> = std::collections::HashMap::new();
        if candidate_ids.is_empty() { return counts; }
        let placeholders: Vec<String> = (1..=candidate_ids.len()).map(|i| format!("?{}", i)).collect();
        let sql = format!(
            "SELECT source_memory_id, \
             SUM(CASE WHEN valid_to IS NULL THEN 1 ELSE 0 END), \
             SUM(CASE WHEN valid_to IS NOT NULL THEN 1 ELSE 0 END) \
             FROM knowledge_triples WHERE source_memory_id IN ({}) GROUP BY source_memory_id",
            placeholders.join(",")
        );
        if let Ok(mut stmt) = self.conn.prepare(&sql) {
            let param_refs: Vec<&dyn rusqlite::types::ToSql> = candidate_ids.iter().map(|id| id as &dyn rusqlite::types::ToSql).collect();
            if let Ok(rows) = stmt.query_map(param_refs.as_slice(), |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?, r.get::<_, i64>(2)?))
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
                        let entry = canonical_projects.entry(slug.clone()).or_insert_with(|| CanonicalProjectRecord {
                            path: normalized_path.clone(),
                            description: description.clone().filter(|value| !value.trim().is_empty()),
                            created_at: created_at.clone(),
                        });

                        if entry.path.is_empty() && !normalized_path.is_empty() {
                            entry.path = normalized_path.clone();
                        }
                        if entry.description.is_none() {
                            entry.description = description.clone().filter(|value| !value.trim().is_empty());
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
        let has_importance: bool = self.conn
            .prepare("SELECT importance FROM memories LIMIT 0")
            .is_ok();
        if !has_importance {
            let _ = self.conn.execute_batch(
                "ALTER TABLE memories ADD COLUMN importance INTEGER NOT NULL DEFAULT 3;
                 ALTER TABLE memories ADD COLUMN expires_at TEXT;"
            );
        }
        // v3.0 columns
        let has_embedding: bool = self.conn
            .prepare("SELECT embedding FROM memories LIMIT 0")
            .is_ok();
        if !has_embedding {
            let _ = self.conn.execute_batch(
                "ALTER TABLE memories ADD COLUMN embedding BLOB;
                 ALTER TABLE memories ADD COLUMN last_accessed_at TEXT;
                 ALTER TABLE memories ADD COLUMN access_count INTEGER NOT NULL DEFAULT 0;"
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
        let has_content_hash: bool = self.conn
            .prepare("SELECT content_hash FROM memories LIMIT 0")
            .is_ok();
        if !has_content_hash {
            let _ = self.conn.execute_batch("ALTER TABLE memories ADD COLUMN content_hash TEXT;");
        }
        Ok(())
    }

    // ─── DEDUP ────────────────────────────────────────

    /// Normalize text for comparison: lowercase, collapse whitespace, strip punctuation.
    fn normalize(text: &str) -> String {
        text.to_lowercase()
            .chars()
            .map(|c| if c.is_alphanumeric() || c == ' ' { c } else { ' ' })
            .collect::<String>()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Jaccard similarity between two normalized strings (word-level).
    fn similarity(a: &str, b: &str) -> f64 {
        let a_words: std::collections::HashSet<&str> = a.split_whitespace().collect();
        let b_words: std::collections::HashSet<&str> = b.split_whitespace().collect();
        if a_words.is_empty() && b_words.is_empty() { return 1.0; }
        let intersection = a_words.intersection(&b_words).count() as f64;
        let union = a_words.union(&b_words).count() as f64;
        if union == 0.0 { 0.0 } else { intersection / union }
    }
    /// Find a near-duplicate in the same project/scope.
    fn find_duplicate(&self, content: &str, project: Option<&str>) -> Result<Option<Memory>, String> {
        // Fast path: exact content match via hash
        let hash = content_hash(content);
        let exact = if let Some(p) = project {
            self.conn.prepare("SELECT id,content,kind,project,tags,source,importance,expires_at,metadata,created_at,updated_at,last_accessed_at,access_count FROM memories WHERE content_hash=?1 AND project=?2 LIMIT 1")
                .ok().and_then(|mut s| s.query_row(params![&hash, p], |r| Ok(row_to_memory(r))).ok())
        } else {
            self.conn.prepare("SELECT id,content,kind,project,tags,source,importance,expires_at,metadata,created_at,updated_at,last_accessed_at,access_count FROM memories WHERE content_hash=?1 AND project IS NULL LIMIT 1")
                .ok().and_then(|mut s| s.query_row(params![&hash], |r| Ok(row_to_memory(r))).ok())
        };
        if let Some(mem) = exact { return Ok(Some(mem)); }

        // Slow path: Jaccard fuzzy match on recent memories
        let norm = Self::normalize(content);
        let memories: Vec<Memory> = if let Some(p) = project {
            let mut stmt = self.conn.prepare(
                "SELECT id,content,kind,project,tags,source,importance,expires_at,metadata,created_at,updated_at,last_accessed_at,access_count FROM memories WHERE project=?1 ORDER BY updated_at DESC LIMIT 200"
            ).map_err(|e| format!("Dedup: {}", e))?;
            let rows = stmt.query_map(params![p], |r| Ok(row_to_memory(r)))
                .map_err(|e| format!("Dedup: {}", e))?;
            rows.flatten().collect()
        } else {
            let mut stmt = self.conn.prepare(
                "SELECT id,content,kind,project,tags,source,importance,expires_at,metadata,created_at,updated_at,last_accessed_at,access_count FROM memories WHERE project IS NULL ORDER BY updated_at DESC LIMIT 200"
            ).map_err(|e| format!("Dedup: {}", e))?;
            let rows = stmt.query_map([], |r| Ok(row_to_memory(r)))
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
        let _ = self.conn.execute("DELETE FROM memory_entities WHERE memory_id = ?1", params![memory.id]);
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

            if let Some(project_name) = memory.project.as_deref() {
                if let Ok(mut stmt) = self.conn.prepare(
                    "SELECT DISTINCT m.id, m.kind FROM memory_entities e
                     JOIN memories m ON e.memory_id = m.id
                     WHERE e.entity_kind = ?1
                       AND e.entity_value = ?2
                       AND e.memory_id != ?3
                       AND (m.project = ?4 OR m.project IS NULL)
                     LIMIT 10",
                ) {
                    if let Ok(rows) = stmt.query_map(params![entity.kind, entity.value, memory.id, project_name], |row| {
                        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                    }) {
                        for row in rows.flatten() {
                            target_ids.insert((row.0, row.1));
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
                     LIMIT 10",
                ) {
                    if let Ok(rows) = stmt.query_map(params![entity.kind, entity.value, memory.id], |row| {
                        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                    }) {
                        for row in rows.flatten() {
                            target_ids.insert((row.0, row.1));
                        }
                    }
                }
            }
        }
        
        let _ = self.conn.execute("DELETE FROM memory_links WHERE source_id = ?1 OR target_id = ?1", params![memory.id]);
        
        let created_at = Utc::now().to_rfc3339();
        for (target_id, target_kind) in target_ids {
            let rel = crate::graph::infer_relation(&memory.kind, &target_kind);
            let _ = self.conn.execute(
                "INSERT OR IGNORE INTO memory_links (source_id, target_id, relation_type, created_at) VALUES (?1, ?2, ?3, ?4)",
                params![memory.id, target_id, rel, &created_at]
            );
            let rev_rel = crate::graph::infer_relation(&target_kind, &memory.kind);
            let _ = self.conn.execute(
                "INSERT OR IGNORE INTO memory_links (source_id, target_id, relation_type, created_at) VALUES (?1, ?2, ?3, ?4)",
                params![target_id, memory.id, rev_rel, &created_at]
            );
        }
        Ok(())
    }

    // ─── KNOWLEDGE TRIPLES ─────────────────────────────

    pub fn add_triple(&self, subject: &str, predicate: &str, object: &str,
                      valid_from: Option<&str>, valid_to: Option<&str>,
                      confidence: Option<f64>, source_memory_id: Option<&str>) -> Result<serde_json::Value, String> {
        let sub = subject.to_lowercase().replace(' ', "_");
        let pred = predicate.to_lowercase().replace(' ', "_");
        let obj = object.to_lowercase().replace(' ', "_");

        let existing: Option<String> = self.conn.prepare(
            "SELECT id FROM knowledge_triples WHERE subject=?1 AND predicate=?2 AND object=?3 AND valid_to IS NULL"
        ).ok().and_then(|mut s| s.query_row(params![&sub, &pred, &obj], |r| r.get(0)).ok());

        if let Some(id) = existing {
            return Ok(serde_json::json!({"triple_id": id, "already_exists": true, "fact": format!("{} -> {} -> {}", subject, predicate, object)}));
        }

        let id = format!("t_{}_{}_{}_{}", &sub, &pred, &obj, &Uuid::new_v4().to_string()[..8]);
        let conf = confidence.unwrap_or(1.0);
        let now = Utc::now().to_rfc3339();
        self.conn.execute(
            "INSERT INTO knowledge_triples (id, subject, predicate, object, valid_from, valid_to, confidence, source_memory_id, created_at) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9)",
            params![&id, &sub, &pred, &obj, valid_from, valid_to, conf, source_memory_id, &now]
        ).map_err(|e| format!("add_triple: {}", e))?;
        Ok(serde_json::json!({"triple_id": id, "fact": format!("{} -> {} -> {}", subject, predicate, object)}))
    }

    pub fn invalidate_triple(&self, subject: &str, predicate: &str, object: &str, ended: Option<&str>) -> Result<serde_json::Value, String> {
        let sub = subject.to_lowercase().replace(' ', "_");
        let pred = predicate.to_lowercase().replace(' ', "_");
        let obj = object.to_lowercase().replace(' ', "_");
        let end_date = ended.unwrap_or(&Utc::now().format("%Y-%m-%d").to_string()).to_string();
        let changed = self.conn.execute(
            "UPDATE knowledge_triples SET valid_to=?1 WHERE subject=?2 AND predicate=?3 AND object=?4 AND valid_to IS NULL",
            params![&end_date, &sub, &pred, &obj]
        ).map_err(|e| format!("invalidate_triple: {}", e))?;
        Ok(serde_json::json!({"invalidated": changed, "fact": format!("{} -> {} -> {}", subject, predicate, object), "ended": end_date}))
    }

    pub fn query_kg_entity(&self, name: &str, as_of: Option<&str>, direction: &str) -> Result<serde_json::Value, String> {
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
                let refs: Vec<&dyn rusqlite::types::ToSql> = param_vals.iter().map(|p| p.as_ref()).collect();
                if let Ok(rows) = stmt.query_map(refs.as_slice(), |r| {
                    Ok((r.get::<_,String>(0)?, r.get::<_,String>(1)?, r.get::<_,String>(2)?, r.get::<_,Option<String>>(3)?, r.get::<_,Option<String>>(4)?, r.get::<_,f64>(5)?))
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
                let refs: Vec<&dyn rusqlite::types::ToSql> = param_vals.iter().map(|p| p.as_ref()).collect();
                if let Ok(rows) = stmt.query_map(refs.as_slice(), |r| {
                    Ok((r.get::<_,String>(0)?, r.get::<_,String>(1)?, r.get::<_,String>(2)?, r.get::<_,Option<String>>(3)?, r.get::<_,Option<String>>(4)?, r.get::<_,f64>(5)?))
                }) {
                    for row in rows.flatten() {
                        facts.push(serde_json::json!({"direction":"incoming","subject":row.0,"predicate":row.1,"object":row.2,"valid_from":row.3,"valid_to":row.4,"confidence":row.5,"current":row.4.is_none()}));
                    }
                }
            }
        }
        Ok(serde_json::json!({"entity": name, "as_of": as_of, "facts": facts, "count": facts.len()}))
    }

    pub fn kg_timeline(&self, entity: Option<&str>) -> Result<serde_json::Value, String> {
        let mut results = Vec::new();
        let sql = if let Some(name) = entity {
            let eid = name.to_lowercase().replace(' ', "_");
            let mut stmt = self.conn.prepare(
                "SELECT subject, predicate, object, valid_from, valid_to, confidence FROM knowledge_triples WHERE subject = ?1 OR object = ?1 ORDER BY valid_from ASC NULLS LAST"
            ).map_err(|e| format!("kg_timeline: {}", e))?;
            let rows = stmt.query_map(params![&eid], |r| {
                Ok((r.get::<_,String>(0)?, r.get::<_,String>(1)?, r.get::<_,String>(2)?, r.get::<_,Option<String>>(3)?, r.get::<_,Option<String>>(4)?))
            }).map_err(|e| format!("kg_timeline: {}", e))?;
            for r in rows.flatten() {
                results.push(serde_json::json!({"subject":r.0,"predicate":r.1,"object":r.2,"valid_from":r.3,"valid_to":r.4,"current":r.4.is_none()}));
            }
            return Ok(serde_json::json!({"entity": name, "timeline": results, "count": results.len()}));
        } else {
            "SELECT subject, predicate, object, valid_from, valid_to FROM knowledge_triples ORDER BY valid_from ASC NULLS LAST LIMIT 100"
        };
        if entity.is_none() {
            let mut stmt = self.conn.prepare(sql).map_err(|e| format!("kg_timeline: {}", e))?;
            let rows = stmt.query_map([], |r| {
                Ok((r.get::<_,String>(0)?, r.get::<_,String>(1)?, r.get::<_,String>(2)?, r.get::<_,Option<String>>(3)?, r.get::<_,Option<String>>(4)?))
            }).map_err(|e| format!("kg_timeline: {}", e))?;
            for r in rows.flatten() {
                results.push(serde_json::json!({"subject":r.0,"predicate":r.1,"object":r.2,"valid_from":r.3,"valid_to":r.4,"current":r.4.is_none()}));
            }
        }
        Ok(serde_json::json!({"entity": "all", "timeline": results, "count": results.len()}))
    }

    pub fn kg_stats(&self) -> Result<serde_json::Value, String> {
        let total: i64 = self.conn.query_row("SELECT COUNT(*) FROM knowledge_triples", [], |r| r.get(0)).unwrap_or(0);
        let current: i64 = self.conn.query_row("SELECT COUNT(*) FROM knowledge_triples WHERE valid_to IS NULL", [], |r| r.get(0)).unwrap_or(0);
        let expired = total - current;
        let mut predicates = Vec::new();
        if let Ok(mut stmt) = self.conn.prepare("SELECT DISTINCT predicate FROM knowledge_triples ORDER BY predicate") {
            if let Ok(rows) = stmt.query_map([], |r| r.get::<_, String>(0)) {
                for p in rows.flatten() { predicates.push(p); }
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
    pub fn add_memory(&self, content: &str, kind: &str, project: Option<&str>,
                      tags: &[String], source: &str, importance: i32,
                      expires_at: Option<&str>,
                      metadata: Option<&serde_json::Value>,
                      scope: &MemoryScope) -> Result<(Memory, bool), String> {
        let canonical_project = Self::canonical_project(project);
        let scoped_metadata = Self::apply_scope_to_metadata(metadata, scope);
        // Check for near-duplicate
        if let Some(existing) = self.find_duplicate(content, canonical_project.as_deref())? {
            // Merge: update content if newer is longer, bump updated_at
            let new_content = if content.len() > existing.content.len() { content } else { &existing.content };
            let new_importance = importance.max(existing.importance);
            let mut merged_tags: Vec<String> = existing.tags.clone();
            for t in tags { if !merged_tags.contains(t) { merged_tags.push(t.clone()); } }
            let merged_metadata = Self::merge_metadata(existing.metadata.as_ref(), scoped_metadata.as_ref());
            let updated = self.update_memory_full(&existing.id, Some(new_content), None,
                Some(&merged_tags), Some(new_importance), expires_at, merged_metadata.as_ref())?;
            return Ok((updated.unwrap_or(existing), true));
        }

        let id = Uuid::new_v4().to_string();
        let now = Utc::now().to_rfc3339();
        let tags_json = serde_json::to_string(tags).unwrap_or_else(|_| "[]".into());
        let meta_json = scoped_metadata.as_ref().map(|m| serde_json::to_string(m).unwrap_or_default());
        let imp = importance.clamp(1, 5);
        let hash = content_hash(content);

        self.conn.execute(
            "INSERT INTO memories (id,content,kind,project,tags,source,importance,expires_at,metadata,embedding,content_hash,created_at,updated_at,access_count)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,NULL,?10,?11,?12,0)",
            params![id, content, kind, canonical_project.as_deref(), tags_json, source, imp, expires_at, meta_json, &hash, now, now],
        ).map_err(|e| format!("Insert: {}", e))?;

        // Queue embedding for background computation
        queue_embedding_job(&id, content);

        // FTS index
        let rowid = self.conn.last_insert_rowid();
        self.conn.execute(
            "INSERT INTO memories_fts (rowid,content,tags,kind,project) VALUES (?1,?2,?3,?4,?5)",
            params![rowid, content, tags_json, kind, canonical_project.as_deref().unwrap_or("")],
        ).map_err(|e| format!("FTS insert: {}", e))?;

        if let Some(proj) = canonical_project.as_deref() { let _ = self.ensure_project(proj); }

        let mem = Memory { id, content: content.into(), kind: kind.into(), project: canonical_project,
            tags: tags.to_vec(), source: source.into(), importance: imp, expires_at: expires_at.map(String::from),
            created_at: now.clone(), updated_at: now, metadata: scoped_metadata, last_accessed_at: None, access_count: 0 };
        let _ = self.rebuild_links(&mem);
        Ok((mem, false))
    }
    /// Full update with all fields.
    pub fn update_memory_full(&self, id: &str, content: Option<&str>, kind: Option<&str>,
                              tags: Option<&[String]>, importance: Option<i32>,
                              expires_at: Option<&str>,
                              metadata: Option<&serde_json::Value>) -> Result<Option<Memory>, String> {
        let existing = match self.get_memory(id)? { Some(m) => m, None => return Ok(None) };
        let now = Utc::now().to_rfc3339();
        let new_content = content.unwrap_or(&existing.content);
        let new_kind = kind.unwrap_or(&existing.kind);
        let new_tags = tags.map(|t| t.to_vec()).unwrap_or_else(|| existing.tags.clone());
        let tags_json = serde_json::to_string(&new_tags).unwrap_or_else(|_| "[]".into());
        let new_imp = importance.unwrap_or(existing.importance).clamp(1, 5);
        let new_exp = if expires_at.is_some() { expires_at.map(String::from) } else { existing.expires_at.clone() };
        let new_metadata = metadata.cloned().or_else(|| existing.metadata.clone());
        let metadata_json = new_metadata.as_ref().map(|value| serde_json::to_string(value).unwrap_or_default());
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
            "SELECT rowid FROM memories WHERE id=?1", params![id], |r| r.get(0)) {
            let _ = self.conn.execute("DELETE FROM memories_fts WHERE rowid=?1", params![rowid]);
            let proj = existing.project.as_deref().unwrap_or("");
            let _ = self.conn.execute(
                "INSERT INTO memories_fts (rowid,content,tags,kind,project) VALUES (?1,?2,?3,?4,?5)",
                params![rowid, new_content, tags_json, new_kind, proj]);
        }

        let mem = Memory { id: id.into(), content: new_content.into(), kind: new_kind.into(),
            project: existing.project, tags: new_tags, source: existing.source,
            importance: new_imp, expires_at: new_exp,
            created_at: existing.created_at, updated_at: now, metadata: new_metadata, 
            last_accessed_at: existing.last_accessed_at, access_count: existing.access_count };
        let _ = self.rebuild_links(&mem);
        Ok(Some(mem))
    }



    pub fn delete_memory(&self, id: &str) -> Result<bool, String> {
        if let Ok(rowid) = self.conn.query_row::<i64, _, _>(
            "SELECT rowid FROM memories WHERE id=?1", params![id], |r| r.get(0)) {
            let _ = self.conn.execute("DELETE FROM memories_fts WHERE rowid=?1", params![rowid]);
        }
        let affected = self.conn.execute("DELETE FROM memories WHERE id=?1", params![id])
            .map_err(|e| format!("Delete: {}", e))?;
        Ok(affected > 0)
    }

    pub fn get_memory(&self, id: &str) -> Result<Option<Memory>, String> {
        let mut stmt = self.conn.prepare(
            "SELECT id,content,kind,project,tags,source,importance,expires_at,metadata,created_at,updated_at,last_accessed_at,access_count FROM memories WHERE id=?1"
        ).map_err(|e| format!("Prepare: {}", e))?;
        let mut rows = stmt.query(params![id]).map_err(|e| format!("Query: {}", e))?;
        match rows.next().map_err(|e| format!("Next: {}", e))? {
            Some(row) => Ok(Some(row_to_memory(row))),
            None => Ok(None),
        }
    }

    // ─── BULK ADD ─────────────────────────────────────

    /// Add multiple memories in one call, with dedup per item. Returns (added, merged, skipped).
    pub fn add_memories_bulk(&self, items: &[BulkItem]) -> Result<(Vec<Memory>, usize, usize), String> {
        let mut added: Vec<Memory> = Vec::new();
        let mut merged = 0usize;
        let mut skipped = 0usize;
        for item in items {
            if item.content.trim().is_empty() { skipped += 1; continue; }
            let tags: Vec<String> = item.tags.clone().unwrap_or_default();
            let imp = item.importance.unwrap_or(3);
            let exp = item.expires_at.as_deref();
            match self.add_memory(&item.content, &item.kind, item.project.as_deref(),
                                  &tags, &item.source, imp, exp, item.metadata.as_ref(), &item.scope()) {
                Ok((mem, was_merged)) => {
                    if was_merged { merged += 1; } else { added.push(mem); }
                }
                Err(_) => { skipped += 1; }
            }
        }
        Ok((added, merged, skipped))
    }

    fn split_transcript_chunks(content: &str, chunk_size: usize) -> Vec<String> {
        let mut chunks = Vec::new();
        let mut current_chunk = String::new();

        for word in content.split_whitespace() {
            if current_chunk.len() + word.len() + 1 > chunk_size && !current_chunk.is_empty() {
                chunks.push(current_chunk.trim().to_string());
                current_chunk.clear();
            }
            if !current_chunk.is_empty() {
                current_chunk.push(' ');
            }
            current_chunk.push_str(word);
        }

        if !current_chunk.trim().is_empty() {
            chunks.push(current_chunk.trim().to_string());
        }

        chunks
    }

    fn transcript_segments(content: &str) -> Vec<(Option<&'static str>, String)> {
        let mut segments = Vec::new();

        for raw_line in content.lines() {
            let trimmed_line = raw_line.trim();
            if trimmed_line.is_empty() {
                continue;
            }

            let lower = trimmed_line.to_ascii_lowercase();
            let (role, body) = if lower.starts_with("user:") {
                (Some("user"), trimmed_line[5..].trim())
            } else if lower.starts_with("assistant:") {
                (Some("assistant"), trimmed_line[10..].trim())
            } else if lower.starts_with("system:") {
                (Some("system"), trimmed_line[7..].trim())
            } else if lower.starts_with("developer:") {
                (Some("developer"), trimmed_line[10..].trim())
            } else {
                (None, trimmed_line)
            };

            let mut current = String::new();
            let mut chars = body.chars().peekable();
            while let Some(character) = chars.next() {
                current.push(character);
                if matches!(character, '.' | '!' | '?') {
                    let next_is_space = chars.peek().map(|next| next.is_whitespace()).unwrap_or(true);
                    if next_is_space {
                        let candidate = current.trim();
                        if !candidate.is_empty() {
                            segments.push((role, candidate.to_string()));
                        }
                        current.clear();
                    }
                }
            }

            let remaining = current.trim();
            if !remaining.is_empty() {
                segments.push((role, remaining.to_string()));
            }
        }

        segments
    }

    fn normalized_transcript_segment(segment: &str) -> Option<String> {
        let cleaned = segment
            .trim()
            .trim_matches(|character: char| matches!(character, '"' | '\'' | '`' | '-' | '*' | ':' | ';'))
            .replace("  ", " ");
        if cleaned.len() < 24 || cleaned.len() > 260 {
            return None;
        }

        let lowered = cleaned.to_ascii_lowercase();
        if lowered.starts_with("http://")
            || lowered.starts_with("https://")
            || lowered.starts_with("tool:")
            || lowered.starts_with("[tool")
            || lowered.starts_with("[thinking")
            || lowered.starts_with("```")
            || cleaned.chars().filter(|character| !character.is_alphanumeric() && !character.is_whitespace()).count() > cleaned.len() / 3
        {
            return None;
        }

        Some(cleaned)
    }

    fn transcript_candidate_limit(kind: &str) -> usize {
        match kind {
            "preference" => 3,
            "decision" => 3,
            "milestone" => 2,
            "bug" => 2,
            "todo" => 2,
            "fact" => 3,
            _ => 1,
        }
    }

    fn build_transcript_candidate(
        segment: &str,
        role: Option<&str>,
        project: Option<&str>,
        tags: &[String],
        source: &str,
        transcript_id: &str,
        scope: &MemoryScope,
    ) -> Option<DistilledTranscriptCandidate> {
        let content = Self::normalized_transcript_segment(segment)?;
        let lowered = content.to_ascii_lowercase();
        if lowered.ends_with('?') {
            return None;
        }

        let has_file = content.contains('/') && content.contains('.');
        let entities = crate::graph::extract_entities(&content, project);
        let entity_bonus = entities.len() as i32;

        let preference_markers = [
            "always", "never", "prefer", "do not", "don't", "must", "should", "need to",
            "my rule", "i like", "i hate", "convention",
            "toujours", "ne jamais", "je veux", "il faut", "pas de", "je préfère",
        ];
        let decision_markers = [
            "decided", "chose", "switched to", "instead of", "trade-off", "because we",
            "we will", "we'll", "will use", "use ", "uses ", "keep ", "switch ", "migrate ",
            "supports ", "store ", "now supports",
            "on va", "garder", "utiliser", "choisir", "on a décidé",
        ];
        let todo_markers = [
            "todo", "next step", "follow up", "remaining", "pending", "should add",
            "prochaine étape", "reste à", "à faire",
        ];
        let bug_markers = [
            "bug", "error", "failed", "fails", "failing", "panic", "crash", "not found",
            "cannot", "doesn't work", "does not work", "broken", "root cause", "workaround",
            "ne marche", "erreur", "échoue", "cassé",
        ];
        let milestone_markers = [
            "it works", "breakthrough", "figured out", "shipped", "deployed", "released",
            "completed", "done", "finally", "working now", "launched", "resolved",
            "ça marche", "terminé", "fini", "déployé", "livré",
        ];
        let resolution_markers = ["fixed", "solved", "resolved", "patched", "corrected", "réglé", "corrigé"];

        let contains_any = |markers: &[&str]| markers.iter().any(|marker| lowered.contains(marker));
        let count_matches = |markers: &[&str]| markers.iter().filter(|marker| lowered.contains(**marker)).count() as i32;

        let pref_score = count_matches(&preference_markers);
        let dec_score = count_matches(&decision_markers);
        let bug_score = count_matches(&bug_markers);
        let todo_score = count_matches(&todo_markers);
        let mile_score = count_matches(&milestone_markers);
        let resolution_hits = count_matches(&resolution_markers);

        // Disambiguate: a bug with resolution markers becomes a milestone
        let is_resolved_problem = bug_score > 0 && resolution_hits > 0;

        let (kind, importance, score) = if contains_any(&preference_markers) && role == Some("user") && pref_score >= 1 {
            ("preference", 5, 16 + entity_bonus + pref_score)
        } else if is_resolved_problem || (mile_score >= 1 && bug_score == 0) {
            ("milestone", 4, 15 + entity_bonus + mile_score + resolution_hits)
        } else if bug_score >= 1 && !is_resolved_problem {
            ("bug", 4, 14 + entity_bonus + bug_score + if has_file { 2 } else { 0 })
        } else if dec_score >= 1 && (has_file || entity_bonus > 0 || role.is_some()) {
            ("decision", 4, 13 + entity_bonus + dec_score + if role == Some("user") { 1 } else { 0 })
        } else if todo_score >= 1 || (role == Some("assistant") && lowered.contains("next")) {
            ("todo", 3, 11 + entity_bonus + todo_score)
        } else if (entity_bonus >= 2 || has_file) && lowered.contains("safe")
            || lowered.contains("mode")
            || lowered.contains("benchmark")
            || lowered.contains("scope")
            || lowered.contains("project")
            || lowered.contains("credential")
        {
            ("fact", 3, 9 + entity_bonus + if has_file { 2 } else { 0 })
        } else {
            return None;
        };

        if score < 9 {
            return None;
        }

        let mut candidate_tags = tags.to_vec();
        for tag in ["transcript", "distilled", kind] {
            if !candidate_tags.iter().any(|existing| existing == tag) {
                candidate_tags.push(tag.to_string());
            }
        }

        let metadata = serde_json::json!({
            "transcript_id": transcript_id,
            "distilled_from": "transcript",
            "speaker_role": role,
            "distillation_score": score,
        });

        Some(DistilledTranscriptCandidate {
            score,
            normalized_key: content.to_ascii_lowercase(),
            item: BulkItem {
                content,
                kind: kind.to_string(),
                project: project.map(String::from),
                tags: Some(candidate_tags),
                source: format!("{}:transcript-distilled", source),
                importance: Some(importance),
                expires_at: None,
                metadata: Some(metadata),
                session_id: scope.session_id.clone(),
                thread_id: scope.thread_id.clone(),
                window_id: scope.window_id.clone(),
            },
        })
    }

    fn distill_transcript_memories(
        content: &str,
        project: Option<&str>,
        tags: &[String],
        source: &str,
        transcript_id: &str,
        scope: &MemoryScope,
    ) -> Vec<BulkItem> {
        let mut candidates = Vec::new();
        for (role, segment) in Self::transcript_segments(content) {
            if let Some(candidate) = Self::build_transcript_candidate(&segment, role, project, tags, source, transcript_id, scope) {
                candidates.push(candidate);
            }
        }

        candidates.sort_by(|left, right| {
            right
                .score
                .cmp(&left.score)
                .then_with(|| right.item.importance.cmp(&left.item.importance))
                .then_with(|| right.item.content.len().cmp(&left.item.content.len()))
        });

        let mut seen = std::collections::HashSet::new();
        let mut kind_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        let mut selected = Vec::new();

        for candidate in candidates {
            if selected.len() >= 10 {
                break;
            }
            if !seen.insert(candidate.normalized_key.clone()) {
                continue;
            }

            let kind_limit = Self::transcript_candidate_limit(&candidate.item.kind);
            let count = kind_counts.entry(candidate.item.kind.clone()).or_insert(0);
            if *count >= kind_limit {
                continue;
            }

            *count += 1;
            selected.push(candidate.item);
        }

        selected
    }

    pub fn add_transcript(
        &self,
        content: &str,
        project: Option<&str>,
        tags: &[String],
        source: &str,
        scope: &MemoryScope,
        distill: bool,
    ) -> Result<TranscriptAddReport, String> {
        let transcript_id = Uuid::new_v4().to_string();
        let chunks = Self::split_transcript_chunks(content, 2000);

        let transcript_items = chunks
            .iter()
            .enumerate()
            .map(|(index, chunk)| BulkItem {
                content: chunk.clone(),
                kind: "transcript".to_string(),
                project: project.map(String::from),
                tags: Some(tags.to_vec()),
                source: source.to_string(),
                importance: Some(3),
                expires_at: None,
                metadata: Some(serde_json::json!({
                    "transcript_id": transcript_id,
                    "chunk_index": index,
                    "total_chunks": chunks.len()
                })),
                session_id: scope.session_id.clone(),
                thread_id: scope.thread_id.clone(),
                window_id: scope.window_id.clone(),
            })
            .collect::<Vec<_>>();

        let (chunk_added, chunk_merged, chunk_skipped) = match self.add_memories_bulk(&transcript_items) {
            Ok((added, merged, skipped)) => (added.len(), merged, skipped),
            Err(error) => return Err(error),
        };

        let distilled_items = if distill {
            Self::distill_transcript_memories(content, project, tags, source, &transcript_id, scope)
        } else {
            Vec::new()
        };
        let distilled_candidates = distilled_items.len();
        let (distilled_added, distilled_merged, distilled_skipped) = if distilled_items.is_empty() {
            (0, 0, 0)
        } else {
            match self.add_memories_bulk(&distilled_items) {
                Ok((added, merged, skipped)) => (added.len(), merged, skipped),
                Err(error) => return Err(error),
            }
        };

        Ok(TranscriptAddReport {
            transcript_id,
            chunks_total: chunks.len(),
            chunk_added,
            chunk_merged,
            chunk_skipped,
            distilled_candidates,
            distilled_added,
            distilled_merged,
            distilled_skipped,
        })
    }
    // ─── SEARCH (FTS5 BM25 × importance) ──────────────

    pub fn search(&self, query: &str, limit: usize, project: Option<&str>,
                  kind: Option<&str>, tags: Option<&[String]>, watcher_keywords: Option<&[String]>) -> Result<Vec<SearchResult>, String> {
        let canonical_project = Self::canonical_project(project);

        let sanitized_query = query.replace('(', " ").replace(')', " ");
        let fts_terms: String = sanitized_query.split_whitespace()
            .map(|w| {
                let clean = w.replace('"', "\"\"");
                if clean.trim().is_empty() { return String::new(); }
                format!("\"{}\"*", clean)
            })
            .filter(|t| !t.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        if fts_terms.is_empty() { return Ok(Vec::new()); }

        let _ = self.cleanup_expired();

        let query_emb = cached_embed_text(query);

        // Pre-compute KG expansion terms for post-retrieval scoring boost
        let kg_expansion = self.get_kg_expansion_terms(query);

        // 1. BM25 Search
        let mut conditions = vec!["memories_fts MATCH ?1".to_string()];
        let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> = vec![Box::new(fts_terms.clone())];

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
                    bm25(memories_fts, 10.0, 3.0, 1.0, 2.0) AS bm25_score
             FROM memories_fts f
             JOIN memories m ON m.rowid = f.rowid
             WHERE {}
             ORDER BY bm25_score ASC
             LIMIT 100", where_clause);
             
        let mut stmt = self.conn.prepare(&sql).map_err(|e| format!("Search prepare: {}", e))?;
        let param_refs: Vec<&dyn rusqlite::types::ToSql> = param_values.iter().map(|p| p.as_ref()).collect();
        let mut bm25_results = std::collections::HashMap::new();
        let rows = stmt.query_map(param_refs.as_slice(), |row| {
            let mem = row_to_memory(row);
            let bm25: f64 = row.get(13)?;
            Ok((mem, bm25))
        }).map_err(|e| format!("Search: {}", e))?;
        
        let mut rank = 1;
        let mut all_memories = std::collections::HashMap::new();
        for r in rows.flatten() {
            let (mem, _) = r;
            bm25_results.insert(mem.id.clone(), rank);
            all_memories.insert(mem.id.clone(), mem);
            rank += 1;
        }

        // 2. Vector Search (read pool connection, scoped)
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
            let vec_where = if vec_conditions.is_empty() { String::new() } else { format!("WHERE {}", vec_conditions.join(" AND ")) };
            let vec_sql = format!("SELECT id,content,kind,project,tags,source,importance,expires_at,metadata,created_at,updated_at,last_accessed_at,access_count,embedding FROM memories {}", vec_where);
            let rconn = self.read_conn();
            let mut stmt2 = rconn.prepare(&vec_sql).map_err(|e| format!("Vector Search: {}", e))?;
            let vec_refs: Vec<&dyn rusqlite::types::ToSql> = vec_params.iter().map(|p| p.as_ref()).collect();

            let mut vector_scores: Vec<(String, f32)> = Vec::new();
            let rows2 = stmt2.query_map(vec_refs.as_slice(), |row| {
                let mem = row_to_memory(row);
                let blob: Option<Vec<u8>> = row.get(13)?;
                Ok((mem, blob))
            }).map_err(|e| format!("Vector Search error: {}", e))?;

            for r in rows2.flatten() {
                let (mem, blob) = r;
                all_memories.entry(mem.id.clone()).or_insert_with(|| mem.clone());
                if let Some(b) = blob {
                    let emb = crate::embedding::blob_to_vec(&b);
                    let score = crate::embedding::cosine_similarity(&query_emb, &emb);
                    vector_scores.push((mem.id, score));
                } else {
                    vector_scores.push((mem.id, 0.0));
                }
            }

            vector_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let mut vr = std::collections::HashMap::new();
            for (i, (id, _)) in vector_scores.iter().take(100).enumerate() {
                vr.insert(id.clone(), i + 1);
            }
            vr
        };

        // 3. RRF Fusion
        let mut rrf_scores: Vec<(String, f64)> = Vec::new();
        
        // Fetch graph links for PageRank-like boost (scoped to candidates)
        let candidate_ids: Vec<&String> = all_memories.keys().collect();
        let link_boosts = self.build_link_boosts_for(&candidate_ids);

        // Batch-query knowledge triple counts (avoids N+1)
        let triple_counts = self.batch_triple_counts(&candidate_ids);
        
        let now_ts = Utc::now().timestamp() as f64;

        let query_tokens: Vec<String> = query.split_whitespace()
            .map(|w| w.to_lowercase())
            .filter(|w| w.len() >= 3)
            .collect();

        for (id, mem) in &all_memories {
            let bm25_rank = bm25_results.get(id).copied().unwrap_or(1000);
            let vec_rank = vector_results.get(id).copied().unwrap_or(1000);
            let mut score = crate::embedding::rrf_score(bm25_rank, vec_rank);

            // Exact term coverage: boost if a high fraction of query terms appear in the memory content
            if !query_tokens.is_empty() {
                let content_lower = mem.content.to_lowercase();
                let match_frac = query_tokens.iter().filter(|t| content_lower.contains(t.as_str())).count() as f64 / query_tokens.len() as f64;
                if match_frac >= 0.8 {
                    score *= 1.0 + (match_frac - 0.8) * 0.5; // up to +10% for 100% match
                }
            }

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
                let kg_hits = kg_expansion.iter().filter(|t| content_lower.contains(t.as_str())).count();
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
                let match_count = keywords.iter().filter(|w| content_lower.contains(w.to_lowercase().as_str())).count();
                if match_count > 0 {
                    score *= 1.0 + (match_count as f64 * 0.2); // +20% per matching keyword
                }
            }
            
            // Also boost if tag match
            if let Some(filter_tags) = tags {
                let filter_set: std::collections::HashSet<String> = filter_tags.iter().map(|t| t.to_lowercase()).collect();
                if mem.tags.iter().any(|t| filter_set.contains(&t.to_lowercase())) {
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

        rrf_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut results: Vec<SearchResult> = Vec::new();
        for (id, score) in rrf_scores.into_iter().take(limit) {
            if let Some(mem) = all_memories.remove(&id) {
                results.push(SearchResult { memory: mem, score: (score * 10000.0).round() / 10000.0 });
            }
        }
        
        // 4. GraphRAG Traversal (Expand context based on top matches)
        let top_ids: Vec<String> = results.iter().take(3).map(|r| r.memory.id.clone()).collect();
        if let Ok(related_ids) = crate::graph::traverse_graph(&self.conn, &top_ids, 1) {
            for rel_id in related_ids {
                // If it's not already in results, fetch it and add it
                if !results.iter().any(|r| r.memory.id == rel_id) {
                    if let Ok(Some(mem)) = self.get_memory(&rel_id) {
                        results.push(SearchResult { 
                            memory: mem, 
                            // Give it a slightly lower score than the original match that pulled it
                            score: 0.1 
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

                    let conn_count = selected.iter()
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

            let mut reranked: Vec<SearchResult> = selected.into_iter()
                .map(|i| std::mem::replace(&mut results[i], SearchResult { memory: Memory { id: String::new(), content: String::new(), kind: String::new(), project: None, tags: vec![], source: String::new(), importance: 0, expires_at: None, metadata: None, created_at: String::new(), updated_at: String::new(), last_accessed_at: None, access_count: 0 }, score: 0.0 }))
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

        // Update access count and timestamp for returned results
        for res in &results {
            let _ = self.conn.execute("UPDATE memories SET access_count = access_count + 1, last_accessed_at = ?1 WHERE id = ?2", 
                params![chrono::Utc::now().to_rfc3339(), res.memory.id]);
        }

        Ok(results)
    }
    // ─── LIST ─────────────────────────────────────────

    pub fn list_memories(&self, project: Option<&str>, kind: Option<&str>, exclude_kind: Option<&str>,
                         limit: usize, offset: usize) -> Result<(Vec<Memory>, i64), String> {
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

        let where_clause = if conditions.is_empty() { String::new() }
            else { format!(" WHERE {}", conditions.join(" AND ")) };

        let count_sql = format!("SELECT COUNT(*) FROM memories{}", where_clause);
        let param_refs: Vec<&dyn rusqlite::types::ToSql> = param_values.iter().map(|p| p.as_ref()).collect();
        let total: i64 = self.conn.query_row(&count_sql, param_refs.as_slice(), |r| r.get(0))
            .map_err(|e| format!("Count: {}", e))?;

        let data_sql = format!(
            "SELECT id,content,kind,project,tags,source,importance,expires_at,metadata,created_at,updated_at,last_accessed_at,access_count FROM memories{} ORDER BY updated_at DESC LIMIT ?{} OFFSET ?{}",
            where_clause, param_values.len() + 1, param_values.len() + 2);
        param_values.push(Box::new(limit as i64));
        param_values.push(Box::new(offset as i64));
        let param_refs2: Vec<&dyn rusqlite::types::ToSql> = param_values.iter().map(|p| p.as_ref()).collect();

        let mut stmt = self.conn.prepare(&data_sql).map_err(|e| format!("List: {}", e))?;
        let memories: Vec<Memory> = stmt.query_map(param_refs2.as_slice(), |r| Ok(row_to_memory(r)))
            .map_err(|e| format!("List query: {}", e))?
            .filter_map(|r| r.ok())
            .collect();
        Ok((memories, total))
    }
    // ─── TTL / EXPIRATION ─────────────────────────────

    pub fn cleanup_expired(&self) -> Result<usize, String> {
        static LAST_CLEANUP: OnceLock<Mutex<std::time::Instant>> = OnceLock::new();
        let last = LAST_CLEANUP.get_or_init(|| Mutex::new(std::time::Instant::now() - std::time::Duration::from_secs(120)));
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
        let affected = self.conn.execute(
            "DELETE FROM memories WHERE expires_at IS NOT NULL AND expires_at < ?1", params![now]
        ).map_err(|e| format!("Cleanup: {}", e))?;
        Ok(affected)
    }

    // ─── GC & COMPRESSION ─────────────────────────────
    
    pub fn run_gc(&self, config: &crate::gc::GcConfig, dry_run: bool) -> Result<crate::gc::GcReport, String> {
        #[derive(Clone)]
        struct GcCandidate {
            id: String,
            content: String,
            project: Option<String>,
            importance: i32,
            age_days: i64,
            gc_score: f64,
        }

        let db_path = dirs::home_dir().unwrap_or_default().join(DB_DIR).join(DB_FILE);
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
            let sql = "SELECT id, content, project, importance, updated_at FROM memories WHERE kind = ?1";
            if let Ok(mut stmt) = self.conn.prepare(&sql) {
                if let Ok(rows) = stmt.query_map(params![kind], |r| {
                    Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?, r.get::<_, Option<String>>(2)?, r.get::<_, i32>(3)?, r.get::<_, String>(4)?))
                }) {
                    let mut by_project: std::collections::HashMap<Option<String>, Vec<GcCandidate>> = std::collections::HashMap::new();
                    for row in rows.flatten() {
                        let updated_at = chrono::DateTime::parse_from_rfc3339(&row.4).unwrap_or_else(|_| chrono::Utc::now().into());
                        let age_days = (now - updated_at.with_timezone(&chrono::Utc)).num_days();
                        
                        let score = crate::gc::gc_score(row.3, age_days, kind, config);
                        if score > 0.6 && row.3 < config.importance_threshold && age_days >= config.age_days {
                            by_project.entry(row.2.clone()).or_default().push(GcCandidate {
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
                            let contents: Vec<String> = items.iter().map(|item| item.content.clone()).collect();
                            let merged_content = crate::gc::merge_memories(&contents, kind, proj.as_deref());
                            
                            let ids_to_delete: Vec<String> = items.iter().map(|item| item.id.clone()).collect();
                            let gc_score_avg = items.iter().map(|item| item.gc_score).sum::<f64>() / items.len() as f64;
                            let age_days_min = items.iter().map(|item| item.age_days).min().unwrap_or(0);
                            let age_days_max = items.iter().map(|item| item.age_days).max().unwrap_or(0);
                            let importance_min = items.iter().map(|item| item.importance).min().unwrap_or(0);
                            let importance_max = items.iter().map(|item| item.importance).max().unwrap_or(0);

                            preview_candidates.push(crate::gc::GcPreviewCandidate {
                                kind: kind.clone(),
                                project: proj.clone().or_else(|| items.iter().find_map(|item| item.project.clone())),
                                memory_ids: ids_to_delete.clone(),
                                sample_contents: items.iter().take(3).map(|item| Self::preview_snippet(&item.content)).collect(),
                                confidence_score: (gc_score_avg * 100.0).round() / 100.0,
                                gc_score_avg: (gc_score_avg * 100.0).round() / 100.0,
                                age_days_min,
                                age_days_max,
                                importance_min,
                                importance_max,
                            });
                            
                            if !dry_run {
                                if self.add_memory(&merged_content, kind, proj.as_deref(), &["merged".to_string()], "gc_compressor", 3, None, None, &MemoryScope::default()).is_ok() {
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
            orphan_links_removed += self.conn.execute(
                "DELETE FROM memory_entities WHERE memory_id NOT IN (SELECT id FROM memories)",
                []
            ).unwrap_or(0);
            
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

    // ─── EXPORT ───────────────────────────────────────

    pub fn export_memories(&self, project: Option<&str>, format: &str) -> Result<String, String> {
        let canonical_project = Self::canonical_project(project);
        let (memories, _) = self.list_memories(canonical_project.as_deref(), None, None, 10000, 0)?;
        match format {
            "json" => serde_json::to_string_pretty(&memories).map_err(|e| format!("JSON: {}", e)),
            "markdown" | "md" => {
                let mut md = String::new();
                let title = canonical_project.as_deref().unwrap_or("All Memories");
                md.push_str(&format!("# MemoryPilot Export: {}\n\n", title));
                md.push_str(&format!("Total: {} memories\n\n", memories.len()));

                let mut by_kind: std::collections::BTreeMap<String, Vec<&Memory>> = std::collections::BTreeMap::new();
                for m in &memories { by_kind.entry(m.kind.clone()).or_default().push(m); }

                for (kind, mems) in &by_kind {
                    md.push_str(&format!("## {} ({})\n\n", kind, mems.len()));
                    for m in mems {
                        let tags = if m.tags.is_empty() { String::new() }
                            else { format!(" `{}`", m.tags.join("` `")) };
                        let imp = "★".repeat(m.importance as usize);
                        md.push_str(&format!("- [{}] {}{}\n", imp, m.content, tags));
                    }
                    md.push('\n');
                }
                Ok(md)
            }
            _ => Err(format!("Unknown format '{}'. Use 'json' or 'markdown'.", format)),
        }
    }
    // ─── PROJECTS ─────────────────────────────────────

    fn ensure_project(&self, name: &str) -> Result<(), String> {
        let Some(canonical_name) = Self::canonical_project_name(name) else {
            return Ok(());
        };
        let now = Utc::now().to_rfc3339();
        self.conn.execute("INSERT OR IGNORE INTO projects (name,path,created_at) VALUES (?1,'',?2)", params![canonical_name, now])
            .map_err(|e| format!("Ensure: {}", e))?;
        Ok(())
    }

    pub fn register_project(&self, name: &str, path: &str, description: Option<&str>) -> Result<Project, String> {
        let canonical_name = Self::canonical_project_name(name).ok_or("Project name cannot be empty")?;
        let project_root = Self::infer_project_root(path);
        let now = Utc::now().to_rfc3339();
        self.conn.execute(
            "INSERT INTO projects (name,path,description,created_at) VALUES (?1,?2,?3,?4)
             ON CONFLICT(name) DO UPDATE SET path=?2, description=COALESCE(?3,description)",
            params![&canonical_name, &project_root, description, &now],
        ).map_err(|e| format!("Register: {}", e))?;
        let count: i64 = self.conn.query_row("SELECT COUNT(*) FROM memories WHERE project=?1", params![&canonical_name], |r| r.get(0)).unwrap_or(0);
        Ok(Project { name: canonical_name, path: project_root, description: description.map(String::from), created_at: now, memory_count: count })
    }

    pub fn list_projects(&self) -> Result<Vec<Project>, String> {
        let mut stmt = self.conn.prepare(
            "SELECT p.name, p.path, p.description, p.created_at, COUNT(m.id) as cnt
             FROM projects p LEFT JOIN memories m ON m.project = p.name
             GROUP BY p.name ORDER BY cnt DESC"
        ).map_err(|e| format!("List projects: {}", e))?;
        let projects = stmt.query_map([], |row| {
            Ok(Project { name: row.get(0)?, path: row.get(1)?, description: row.get(2)?,
                created_at: row.get(3)?, memory_count: row.get(4)? })
        }).map_err(|e| format!("Projects: {}", e))?.filter_map(|r| r.ok()).collect();
        Ok(projects)
    }

    pub fn detect_project(&self, working_dir: &str) -> Result<Option<String>, String> {
        let normalized_dir = Self::normalize_path(working_dir);
        if normalized_dir.is_empty() {
            return Ok(None);
        }
        let project_root = Self::infer_project_root(&normalized_dir);

        let mut stmt = self.conn.prepare("SELECT name, path FROM projects WHERE path != '' ORDER BY length(path) DESC")
            .map_err(|e| format!("Detect: {}", e))?;
        let projects: Vec<(String, String)> = stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
            .map_err(|e| format!("Detect2: {}", e))?.filter_map(|r| r.ok()).collect();
        for (name, path) in &projects {
            let normalized_path = Self::normalize_path(path);
            if Self::path_matches(&project_root, &normalized_path) || Self::path_matches(&normalized_dir, &normalized_path) {
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
        let total: i64 = self.conn.query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0)).unwrap_or(0);
        let global: i64 = self.conn.query_row("SELECT COUNT(*) FROM memories WHERE project IS NULL", [], |r| r.get(0)).unwrap_or(0);
        let projects: i64 = self.conn.query_row("SELECT COUNT(*) FROM projects", [], |r| r.get(0)).unwrap_or(0);
        let expired: i64 = self.conn.query_row("SELECT COUNT(*) FROM memories WHERE expires_at IS NOT NULL AND expires_at < ?1",
            params![Utc::now().to_rfc3339()], |r| r.get(0)).unwrap_or(0);

        let mut by_kind = serde_json::Map::new();
        if let Ok(mut stmt) = self.conn.prepare("SELECT kind, COUNT(*) FROM memories GROUP BY kind") {
            if let Ok(rows) = stmt.query_map([], |r| Ok((r.get::<_,String>(0)?, r.get::<_,i64>(1)?))) {
                for row in rows.flatten() { by_kind.insert(row.0, serde_json::json!(row.1)); }
            }
        }
        let mut by_project = serde_json::Map::new();
        if let Ok(mut stmt) = self.conn.prepare("SELECT COALESCE(project,'__global__'), COUNT(*) FROM memories GROUP BY project") {
            if let Ok(rows) = stmt.query_map([], |r| Ok((r.get::<_,String>(0)?, r.get::<_,i64>(1)?))) {
                for row in rows.flatten() { by_project.insert(row.0, serde_json::json!(row.1)); }
            }
        }
        let db_path = dirs::home_dir().unwrap_or_default().join(DB_DIR).join(DB_FILE);
        let size = std::fs::metadata(&db_path).map(|m| m.len()).unwrap_or(0);
        let size_str = if size < 1024 { format!("{} B", size) }
            else if size < 1048576 { format!("{} KB", size / 1024) }
            else { format!("{:.1} MB", size as f64 / 1048576.0) };
        let hygiene = self.hygiene_report();

        Ok(serde_json::json!({ "total_memories": total, "global_memories": global, "projects": projects,
            "expired_pending": expired, "by_kind": by_kind, "by_project": by_project, "db_size": size_str,
            "hygiene": hygiene }))
    }
    // ─── CONFIG ───────────────────────────────────────

    pub fn get_config(&self, key: &str) -> Option<String> {
        self.conn.query_row("SELECT value FROM config WHERE key=?1", params![key], |r| r.get(0)).ok()
    }

    pub fn set_config(&self, key: &str, value: &str) -> Result<(), String> {
        self.conn.execute("INSERT INTO config (key,value) VALUES (?1,?2) ON CONFLICT(key) DO UPDATE SET value=?2",
            params![key, value]).map_err(|e| format!("Config: {}", e))?;
        Ok(())
    }

    // ─── GLOBAL PROMPT (auto-scan) ────────────────────

    pub fn get_global_prompt(&self, project: Option<&str>, working_dir: Option<&str>) -> Option<String> {
        let canonical_project = Self::canonical_project(project);
        let mut prompts: Vec<String> = Vec::new();

        // Helper to read file if modified since last cache, or use cache
        fn get_cached_prompt(path: &std::path::Path) -> Option<String> {
            if !path.exists() { return None; }
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
            if let Some(content) = get_cached_prompt(path) { prompts.push(content); }
        }

        // 2. Auto-scan ~/.MemoryPilot/GLOBAL_PROMPT.md
        let home_prompt = dirs::home_dir().map(|h| h.join(DB_DIR).join(PROMPT_FILE));
        if let Some(path) = &home_prompt {
            if let Some(content) = get_cached_prompt(path) {
                if !prompts.iter().any(|p| p == &content) { prompts.push(content); }
            }
        }

        // 3. Auto-scan project root GLOBAL_PROMPT.md
        let proj_dir: Option<String> = working_dir.map(Self::infer_project_root).or_else(|| {
            let proj_name = canonical_project.as_deref()?;
            let mut stmt = self.conn.prepare("SELECT path FROM projects WHERE name=?1").ok()?;
            stmt.query_row(params![proj_name], |r| r.get::<_,String>(0)).ok()
        });
        
        if let Some(dir) = proj_dir {
            let proj_prompt = std::path::Path::new(&dir).join(PROMPT_FILE);
            if let Some(content) = get_cached_prompt(&proj_prompt) {
                if !prompts.iter().any(|p| p == &content) { prompts.push(content); }
            }
        }

        if prompts.is_empty() { None } else { Some(prompts.join("\n\n---\n\n")) }
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
        let mut stmt = self.conn.prepare(sql)
            .map_err(|e| format!("Backfill prepare: {}", e))?;

        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?, row.get::<_, Option<String>>(2)?))
        }).map_err(|e| format!("Backfill query: {}", e))?;

        let mut to_embed: Vec<(String, String)> = Vec::new();
        let mut skipped = 0usize;
        for r in rows.flatten() {
            let (id, content, existing_hash) = r;
            if !force {
                let new_hash = content_hash(&content);
                if existing_hash.as_deref() == Some(&new_hash) {
                    let has_emb: bool = self.conn.query_row(
                        "SELECT embedding IS NOT NULL FROM memories WHERE id = ?1",
                        params![&id], |r| r.get(0)
                    ).unwrap_or(false);
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
                eprintln!("  Skipped {} memories (content unchanged, embedding exists)", skipped);
            }
            return Ok(0);
        }

        eprintln!("  Computing embeddings for {} memories (skipped {} unchanged)...", to_embed.len(), skipped);

        let texts: Vec<&str> = to_embed.iter().map(|(_, c)| c.as_str()).collect();
        let embeddings = crate::embedding::embed_batch(&texts);
        let mut count = 0;
        for ((id, content), emb) in to_embed.iter().zip(embeddings.iter()) {
            let blob = crate::embedding::vec_to_blob(emb);
            let hash = content_hash(content);
            let _ = self.conn.execute(
                "UPDATE memories SET embedding = ?1, content_hash = ?2 WHERE id = ?3",
                params![blob, &hash, id]
            );
            count += 1;
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
        let tags_str = if mem.tags.is_empty() { String::new() } else { format!(" | tags:{}", mem.tags.join(",")) };
        let proj_str = mem.project.as_ref().map(|p| format!(" | proj:{}", p)).unwrap_or_default();
        format!("[{}:{}] {}{}{}", kind_short, mem.importance, truncated.replace('\n', " "), tags_str, proj_str)
    }

    fn compress_memories(mems: &[Memory]) -> String {
        mems.iter().map(Self::compress_memory).collect::<Vec<_>>().join("\n")
    }

    fn compress_strings(kind: &str, items: &[String]) -> String {
        if items.is_empty() { return String::new(); }
        let tag = kind.to_uppercase();
        items.iter().enumerate().map(|(i, s)| {
            let truncated = if s.len() > 150 { format!("{}...", &s[..150]) } else { s.clone() };
            format!("[{}:{}] {}", tag, i + 1, truncated.replace('\n', " "))
        }).collect::<Vec<_>>().join("\n")
    }

    pub fn get_project_brain(&self, project: &str, max_tokens: Option<usize>, compact: bool) -> Result<serde_json::Value, String> {
        let canonical_project = Self::canonical_project_name(project).ok_or("Project name cannot be empty")?;
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
        
        let (core_arch, _) = self.list_memories(Some(&canonical_project), Some("architecture"), None, 10, 0)?;
        let mut arch_content = Vec::new();
        for m in core_arch {
            if current_chars + m.content.len() > max_chars { break; }
            current_chars += m.content.len();
            arch_content.push(m.content);
        }
        
        let (decisions, _) = self.list_memories(Some(&canonical_project), Some("decision"), None, 10, 0)?;
        let mut dec_content = Vec::new();
        for m in decisions {
            if current_chars + m.content.len() > max_chars { break; }
            current_chars += m.content.len();
            dec_content.push(m.content);
        }
        
        let (bugs, _) = self.list_memories(Some(&canonical_project), Some("bug"), None, 10, 0)?;
        let mut bug_content = Vec::new();
        for m in bugs {
            if current_chars + m.content.len() > max_chars { break; }
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
            if !tech_stack.is_empty() { lines.push(format!("STACK: {}", tech_stack.join(", "))); }
            if !key_components.is_empty() { lines.push(format!("COMPONENTS: {}", key_components.join(", "))); }
            if !team_members.is_empty() { lines.push(format!("TEAM: {}", team_members.join(", "))); }
            if !arch_content.is_empty() { lines.push(Self::compress_strings("ARCH", &arch_content)); }
            if !dec_content.is_empty() { lines.push(Self::compress_strings("DEC", &dec_content)); }
            if !bug_content.is_empty() { lines.push(Self::compress_strings("BUG", &bug_content)); }
            if !recent_content.is_empty() { lines.push(Self::compress_strings("RECENT", &recent_content)); }
            return Ok(serde_json::json!({ "compact": lines.join("\n"), "approx_tokens_used": lines.join("\n").len() / 4 }));
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

    pub fn get_project_context(&self, project: Option<&str>, working_dir: Option<&str>, mode: RecallMode, scope: &MemoryScope) -> Result<serde_json::Value, String> {
        let proj_name = match Self::canonical_project(project) {
            Some(p) => Some(p),
            None => match working_dir { Some(wd) => self.detect_project(wd)?, None => None }
        };
        let proj_ref = proj_name.as_deref();
        let (proj_memories, proj_total) = if let Some(p) = proj_ref {
            let (memories, total) = self.list_memories(Some(p), None, Some("transcript"), 100, 0)?;
            (
                memories
                    .into_iter()
                    .filter(|memory| Self::should_include_in_context(memory, mode))
                    .collect::<Vec<_>>(),
                total,
            )
        } else { (vec![], 0) };
        let (prefs, _) = self.list_memories(None, Some("preference"), None, 50, 0)?;
        let prefs = prefs.into_iter().filter(|memory| memory.project.is_none()).collect::<Vec<_>>();
        let (patterns, _) = self.list_memories(None, Some("pattern"), None, 50, 0)?;
        let patterns = patterns.into_iter().filter(|memory| memory.project.is_none()).collect::<Vec<_>>();
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
    pub fn recall(&self, project: Option<&str>, working_dir: Option<&str>, hints: Option<&str>, mode: RecallMode, explain: bool, compact: bool, scope: &MemoryScope) -> Result<serde_json::Value, String> {
        // ~4 chars per token — 800 token budget = 3200 chars for memories
        const TOKEN_BUDGET_CHARS: usize = 3200;

        // Auto-detect project
        let proj_name = match Self::canonical_project(project) {
            Some(p) => Some(p),
            None => match working_dir { Some(wd) => self.detect_project(wd)?, None => None }
        };
        let proj_ref = proj_name.as_deref();
        let hint_terms: Vec<String> = hints
            .unwrap_or_default()
            .split_whitespace()
            .map(|term: &str| {
                term.trim_matches(|character: char| !character.is_alphanumeric() && character != '-' && character != '_')
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
            let rows = stmt.query_map([], |r| Ok(row_to_memory(r)))
                .map_err(|e| format!("Recall critical: {}", e))?;
            let mut candidates: Vec<(i32, Memory)> = rows
                .flatten()
                .filter(|memory| Self::should_include_in_context(memory, mode))
                .map(|memory| {
                    let lowered_content = memory.content.to_ascii_lowercase();
                    let tag_text = memory.tags.join(" ").to_ascii_lowercase();
                    let hint_overlap = hint_terms
                        .iter()
                        .filter(|term| lowered_content.contains(term.as_str()) || tag_text.contains(term.as_str()))
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
                    let project_match = if proj_ref.is_some() && memory.project.as_deref() == proj_ref {
                        8
                    } else if memory.project.is_none() && hint_overlap > 0 {
                        2
                    } else {
                        0
                    };
                    let score = project_match + (hint_overlap * 4) + (memory.importance * 2) + memory.access_count.min(5);
                    (score, memory)
                })
                .collect();

            candidates.sort_by(|left, right| {
                right
                    .0
                    .cmp(&left.0)
                    .then_with(|| right.1.updated_at.cmp(&left.1.updated_at))
            });

            let critical_limit = if has_contextual_hint { 3 } else if proj_ref.is_some() { 6 } else { 12 };
            candidates
                .into_iter()
                .filter(|(score, _)| *score > 0 || (proj_ref.is_none() && !has_contextual_hint))
                .map(|(_, memory)| memory)
                .take(critical_limit)
                .filter(|memory| budget_check!(&memory.id, &memory.content))
                .collect()
        };

        // 2. Hint-based search (most relevant to current task)
        let hint_results: Vec<SearchResult> = if let Some(h) = hints {
            if !h.trim().is_empty() {
                self.search(h, 10, proj_ref, None, None, None).unwrap_or_default()
                    .into_iter()
                    .filter(|result| result.memory.kind != "transcript")
                    .filter(|result| Self::should_include_in_context(&result.memory, mode))
                    .filter(|r| budget_check!(&r.memory.id, &r.memory.content))
                    .collect()
            } else { vec![] }
        } else { vec![] };

        // 3. Scope memories from the same session / thread / window
        let scope_memories = self.list_scope_memories(proj_ref, scope, 15)?
            .into_iter()
            .filter(|memory| Self::should_include_in_context(memory, mode))
            .filter(|memory| budget_check!(&memory.id, &memory.content))
            .collect::<Vec<_>>();

        // 4. Project memories (excluding transcripts — too verbose)
        let (proj_memories, proj_total) = if let Some(p) = proj_ref {
            let (all, total) = self.list_memories(Some(p), None, Some("transcript"), 50, 0)?;
            let filtered: Vec<Memory> = all.into_iter()
                .filter(|memory| Self::should_include_in_context(memory, mode))
                .filter(|m| budget_check!(&m.id, &m.content))
                .collect();
            (filtered, total)
        } else { (vec![], 0) };

        // 5. Global preferences + patterns + decisions (with remaining budget)
        let (prefs, _) = self.list_memories(None, Some("preference"), None, 20, 0)?;
        let prefs: Vec<Memory> = Self::select_global_context_memories(
            prefs.into_iter().filter(|memory| memory.project.is_none()).collect(),
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
            patterns.into_iter().filter(|memory| memory.project.is_none()).collect(),
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
            decisions.into_iter().filter(|memory| memory.project.is_none()).collect(),
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
        let total: i64 = self.conn.query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0)).unwrap_or(0);
        let projects_count: i64 = self.conn.query_row("SELECT COUNT(*) FROM projects", [], |r| r.get(0)).unwrap_or(0);
        let project_context_memories_count = proj_memories.len();

        if explain {
            for result in &hint_results {
                selected_memories_for_explain.push(self.recall_explanation(&result.memory, "hint", proj_ref, mode, Some(result.score), &link_boosts));
            }
            for memory in &scope_memories {
                selected_memories_for_explain.push(self.recall_explanation(memory, "scope", proj_ref, mode, None, &link_boosts));
            }
            for memory in &proj_memories {
                selected_memories_for_explain.push(self.recall_explanation(memory, "project", proj_ref, mode, None, &link_boosts));
            }
            for memory in &critical {
                selected_memories_for_explain.push(self.recall_explanation(memory, "critical", proj_ref, mode, None, &link_boosts));
            }
            for memory in &prefs {
                selected_memories_for_explain.push(self.recall_explanation(memory, "preference", proj_ref, mode, None, &link_boosts));
            }
            for memory in &patterns {
                selected_memories_for_explain.push(self.recall_explanation(memory, "pattern", proj_ref, mode, None, &link_boosts));
            }
            for memory in &decisions {
                selected_memories_for_explain.push(self.recall_explanation(memory, "decision", proj_ref, mode, None, &link_boosts));
            }
        }

        if compact {
            let mut lines = Vec::new();
            lines.push(format!("# recall | proj:{} | mode:{}", proj_ref.unwrap_or("none"), mode.as_str()));
            if !critical.is_empty() { lines.push(format!("--- critical ---\n{}", Self::compress_memories(&critical))); }
            let hint_mems: Vec<Memory> = hint_results.iter().map(|r| r.memory.clone()).collect();
            if !hint_mems.is_empty() { lines.push(format!("--- hints ---\n{}", Self::compress_memories(&hint_mems))); }
            if !scope_memories.is_empty() { lines.push(format!("--- scope ---\n{}", Self::compress_memories(&scope_memories))); }
            if !proj_memories.is_empty() { lines.push(format!("--- project ---\n{}", Self::compress_memories(&proj_memories))); }
            if !prefs.is_empty() { lines.push(format!("--- prefs ---\n{}", Self::compress_memories(&prefs))); }
            if !patterns.is_empty() { lines.push(format!("--- patterns ---\n{}", Self::compress_memories(&patterns))); }
            if !decisions.is_empty() { lines.push(format!("--- decisions ---\n{}", Self::compress_memories(&decisions))); }
            if let Some(gp) = &global_prompt {
                let gp_short = if gp.len() > 500 { format!("{}...", &gp[..500]) } else { gp.clone() };
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

    fn benchmark_query_for_memory(memory: &Memory) -> Option<String> {
        let mut terms: Vec<String> = Vec::new();

        for entity in crate::graph::extract_entities(&memory.content, memory.project.as_deref()) {
            if entity.kind == "project" {
                continue;
            }
            let value = entity
                .value
                .rsplit('/')
                .next()
                .unwrap_or(entity.value.as_str())
                .trim_matches(|character: char| !character.is_alphanumeric() && character != '-' && character != '_')
                .to_ascii_lowercase();
            if value.len() > 2 && !terms.contains(&value) {
                terms.push(value);
            }
        }

        for tag in &memory.tags {
            let normalized = tag.trim().to_ascii_lowercase();
            if normalized.len() > 2 && !terms.contains(&normalized) {
                terms.push(normalized);
            }
        }

        for word in memory.content.split_whitespace() {
            let normalized = word
                .trim_matches(|character: char| !character.is_alphanumeric() && character != '-' && character != '_')
                .to_ascii_lowercase();
            if normalized.len() <= 3 {
                continue;
            }
            if memory.project.as_deref() == Some(normalized.as_str()) {
                continue;
            }
            if matches!(normalized.as_str(), "this" | "that" | "with" | "from" | "have" | "been" | "will" | "would" | "could" | "into" | "using" | "when" | "where" | "what" | "which" | "dans" | "pour" | "avec" | "cette" | "sont" | "mais" | "plus" | "todo" | "note" | "fact" | "decision" | "pattern") {
                continue;
            }
            if !terms.contains(&normalized) {
                terms.push(normalized);
            }
            if terms.len() >= 4 {
                break;
            }
        }

        if terms.is_empty() {
            None
        } else {
            Some(terms.into_iter().take(4).collect::<Vec<_>>().join(" "))
        }
    }

    fn curated_benchmark_scenarios() -> Vec<BenchmarkScenarioSpec> {
        vec![
            BenchmarkScenarioSpec {
                name: "memorypilot-safe-mode-default",
                query: "safe mode credentials recall",
                project: Some("memorypilot"),
                expected_kind: Some("decision"),
                match_substrings: &["safe mode", "credential"],
                mode: RecallMode::Safe,
            },
            BenchmarkScenarioSpec {
                name: "memorypilot-benchmark-cli",
                query: "benchmark_recall top1 top5",
                project: Some("memorypilot"),
                expected_kind: Some("decision"),
                match_substrings: &["benchmark_recall", "top1", "top5"],
                mode: RecallMode::Safe,
            },
            BenchmarkScenarioSpec {
                name: "memorypilot-cross-project-pollution",
                query: "cross project pollution recall",
                project: Some("memorypilot"),
                expected_kind: Some("decision"),
                match_substrings: &["cross-project", "pollution"],
                mode: RecallMode::Safe,
            },
            BenchmarkScenarioSpec {
                name: "zed-fork-observability-metrics",
                query: "mcp mcphub observability metrics",
                project: Some("zed-fork"),
                expected_kind: Some("decision"),
                match_substrings: &["observability", "metrics"],
                mode: RecallMode::Safe,
            },
            BenchmarkScenarioSpec {
                name: "zed-fork-health-check",
                query: "mcp health check",
                project: Some("zed-fork"),
                expected_kind: Some("bug"),
                match_substrings: &["health check"],
                mode: RecallMode::Safe,
            },
            BenchmarkScenarioSpec {
                name: "mcphub-schema-cache",
                query: "mcp cargo schema cache",
                project: Some("mcphub"),
                expected_kind: Some("decision"),
                match_substrings: &["schema", "cache"],
                mode: RecallMode::Safe,
            },
            BenchmarkScenarioSpec {
                name: "planify-testflight-build",
                query: "testflight ios build sociomator",
                project: Some("planify"),
                expected_kind: Some("fact"),
                match_substrings: &["testflight", "ios"],
                mode: RecallMode::Safe,
            },
            BenchmarkScenarioSpec {
                name: "planify-instagram-cache",
                query: "instagram cache flutter shared_preferences",
                project: Some("planify"),
                expected_kind: Some("decision"),
                match_substrings: &["instagram", "shared_preferences"],
                mode: RecallMode::Safe,
            },
            BenchmarkScenarioSpec {
                name: "memorypilot-session-scope",
                query: "session thread window scope recall",
                project: Some("memorypilot"),
                expected_kind: Some("decision"),
                match_substrings: &["session", "thread", "window"],
                mode: RecallMode::Safe,
            },
            BenchmarkScenarioSpec {
                name: "memorypilot-transcript-distillation",
                query: "transcript distillation recall",
                project: Some("memorypilot"),
                expected_kind: Some("decision"),
                match_substrings: &["transcript", "distill"],
                mode: RecallMode::Safe,
            },
            BenchmarkScenarioSpec {
                name: "memorypilot-credential-safety",
                query: "credential leakage safe mode",
                project: Some("memorypilot"),
                expected_kind: Some("decision"),
                match_substrings: &["credential", "safe"],
                mode: RecallMode::Safe,
            },
            BenchmarkScenarioSpec {
                name: "memorypilot-recall-explain",
                query: "recall explain search score graph boost",
                project: Some("memorypilot"),
                expected_kind: Some("decision"),
                match_substrings: &["recall", "explain"],
                mode: RecallMode::Safe,
            },
        ]
    }

    fn resolve_benchmark_memory(&self, spec: &BenchmarkScenarioSpec) -> Result<Option<Memory>, String> {
        let canonical_project = spec.project.and_then(Self::canonical_project_name);
        let (memories, _) = self.list_memories(
            canonical_project.as_deref(),
            spec.expected_kind,
            Some("transcript"),
            250,
            0,
        )?;

        let mut matches: Vec<Memory> = memories
            .into_iter()
            .filter(|memory| {
                let haystack = format!(
                    "{} {}",
                    memory.content.to_ascii_lowercase(),
                    memory.tags.join(" ").to_ascii_lowercase()
                );
                spec.match_substrings
                    .iter()
                    .all(|term| haystack.contains(term))
            })
            .collect();

        matches.sort_by(|left, right| {
            right
                .importance
                .cmp(&left.importance)
                .then_with(|| right.updated_at.cmp(&left.updated_at))
        });

        Ok(matches.into_iter().next())
    }

    fn generated_benchmark_runs(
        &self,
        limit: usize,
        excluded_ids: &std::collections::HashSet<String>,
    ) -> Result<Vec<BenchmarkScenarioRun>, String> {
        if limit == 0 {
            return Ok(Vec::new());
        }

        let (memories, _) = self.list_memories(None, None, Some("transcript"), limit * 6, 0)?;
        let mut runs = Vec::new();
        let mut seen_projects = std::collections::HashSet::new();

        for memory in memories.into_iter() {
            if excluded_ids.contains(&memory.id) {
                continue;
            }
            if !matches!(memory.kind.as_str(), "decision" | "bug" | "pattern" | "fact" | "snippet") {
                continue;
            }
            if let Some(project_name) = memory.project.as_deref() {
                if !seen_projects.insert((project_name.to_string(), memory.kind.clone())) {
                    continue;
                }
            }
            let Some(query) = Self::benchmark_query_for_memory(&memory) else {
                continue;
            };
            let stable_results = self.search(
                &query,
                5,
                memory.project.as_deref(),
                Some(memory.kind.as_str()),
                None,
                None,
            ).unwrap_or_default();
            let is_stable_candidate = stable_results.iter().any(|result| result.memory.id == memory.id);
            if !is_stable_candidate {
                continue;
            }

            runs.push(BenchmarkScenarioRun {
                name: format!(
                    "generated:{}:{}",
                    memory.project.clone().unwrap_or_else(|| "global".into()),
                    memory.kind
                ),
                source: "generated".into(),
                query,
                mode: RecallMode::Safe,
                expected_memory: memory,
            });

            if runs.len() >= limit {
                break;
            }
        }

        Ok(runs)
    }

    fn benchmark_percentage(count: usize, total: usize) -> f64 {
        ((count as f64 / total.max(1) as f64) * 100.0).round() / 100.0
    }

    pub fn benchmark_recall(&self, scenario_limit: usize) -> Result<serde_json::Value, String> {
        let candidate_limit = scenario_limit.max(5).min(30);
        let golden_specs = Self::curated_benchmark_scenarios();
        let considered_golden_specs = golden_specs.into_iter().take(candidate_limit).collect::<Vec<_>>();
        let mut scenarios = Vec::new();
        let mut skipped_golden = Vec::new();
        let mut expected_ids = std::collections::HashSet::new();

        for spec in &considered_golden_specs {
            match self.resolve_benchmark_memory(spec)? {
                Some(memory) => {
                    expected_ids.insert(memory.id.clone());
                    scenarios.push(BenchmarkScenarioRun {
                        name: spec.name.into(),
                        source: "golden".into(),
                        query: spec.query.into(),
                        mode: spec.mode,
                        expected_memory: memory,
                    });
                }
                None => skipped_golden.push(serde_json::json!({
                    "name": spec.name,
                    "project": spec.project,
                    "kind": spec.expected_kind,
                    "query": spec.query,
                    "reason": "missing_expected_memory"
                })),
            }
        }

        let golden_executed_count = scenarios.len();
        if scenarios.len() < candidate_limit {
            scenarios.extend(self.generated_benchmark_runs(candidate_limit - scenarios.len(), &expected_ids)?);
        }

        let mut hits_top1 = 0usize;
        let mut hits_top5 = 0usize;
        let mut cross_project_leaks = 0usize;
        let mut credential_leaks_safe = 0usize;
        let mut explain_with_search_score = 0usize;
        let mut scenario_results = Vec::new();
        let mut golden_run_count = 0usize;
        let mut generated_run_count = 0usize;

        for scenario in scenarios {
            let recall = self.recall(
                scenario.expected_memory.project.as_deref(),
                None,
                Some(&scenario.query),
                scenario.mode,
                true,
                false,
                &MemoryScope::default(),
            )?;

            let explain_block = recall
                .get("explain")
                .and_then(|value| value.as_object())
                .cloned()
                .unwrap_or_default();
            let selected = explain_block
                .get("selected_memories")
                .and_then(|value| value.as_array())
                .cloned()
                .unwrap_or_default();

            let selected_ids: Vec<String> = selected
                .iter()
                .filter_map(|value| value.get("id").and_then(|id| id.as_str()).map(String::from))
                .collect();

            let top1_hit = selected_ids.first().map(|id| id == &scenario.expected_memory.id).unwrap_or(false);
            let top5_hit = selected_ids.iter().take(5).any(|id| id == &scenario.expected_memory.id);
            if top1_hit {
                hits_top1 += 1;
            }
            if top5_hit {
                hits_top5 += 1;
            }

            let cross_leak_count = selected
                .iter()
                .take(5)
                .filter(|value| {
                    let selected_project = value.get("project").and_then(|project| project.as_str());
                    scenario
                        .expected_memory
                        .project
                        .as_deref()
                        .map(|expected_project| selected_project.is_some() && selected_project != Some(expected_project))
                        .unwrap_or(false)
                })
                .count();
            if cross_leak_count > 0 {
                cross_project_leaks += 1;
            }

            let credential_leak_count = selected
                .iter()
                .filter(|value| value.get("kind").and_then(|kind| kind.as_str()) == Some("credential"))
                .count();
            if credential_leak_count > 0 {
                credential_leaks_safe += 1;
            }

            let has_search_score = selected
                .iter()
                .any(|value| value.get("search_score").and_then(|score| score.as_f64()).is_some());
            if has_search_score {
                explain_with_search_score += 1;
            }

            if scenario.source == "golden" {
                golden_run_count += 1;
            } else {
                generated_run_count += 1;
            }

            scenario_results.push(serde_json::json!({
                "scenario_name": scenario.name,
                "scenario_source": scenario.source,
                "mode": scenario.mode.as_str(),
                "project": scenario.expected_memory.project,
                "kind": scenario.expected_memory.kind,
                "query": scenario.query,
                "expected_memory_id": scenario.expected_memory.id,
                "top1_hit": top1_hit,
                "top5_hit": top5_hit,
                "cross_project_leak_count": cross_leak_count,
                "credential_leak_count_safe": credential_leak_count,
                "selected_memory_ids": selected_ids.into_iter().take(5).collect::<Vec<_>>(),
            }));
        }

        let scenario_count = scenario_results.len();

        Ok(serde_json::json!({
            "status": "ok",
            "scenario_count": scenario_count,
            "golden_defined_count": considered_golden_specs.len(),
            "golden_executed_count": golden_executed_count,
            "golden_skipped_count": skipped_golden.len(),
            "golden_skipped": skipped_golden,
            "scenario_source_counts": {
                "golden": golden_run_count,
                "generated": generated_run_count
            },
            "top1_hit_rate": Self::benchmark_percentage(hits_top1, scenario_count),
            "top5_hit_rate": Self::benchmark_percentage(hits_top5, scenario_count),
            "cross_project_leak_rate": Self::benchmark_percentage(cross_project_leaks, scenario_count),
            "credential_leak_rate_safe": Self::benchmark_percentage(credential_leaks_safe, scenario_count),
            "explain_consistency_rate": Self::benchmark_percentage(explain_with_search_score, scenario_count),
            "scenarios": scenario_results,
        }))
    }

    // ─── SEARCH QUALITY BENCHMARK ─────────────────────

    pub fn benchmark_search(&self, scenario_limit: usize) -> Result<serde_json::Value, String> {
        let all_memories = self.list_all_memories_for_benchmark()?;
        if all_memories.len() < 5 {
            return Ok(serde_json::json!({
                "status": "insufficient_data",
                "memory_count": all_memories.len(),
                "message": "Need at least 5 memories to run search benchmark"
            }));
        }

        let scenario_count = scenario_limit.min(all_memories.len()).min(50);
        let step = all_memories.len() / scenario_count;

        let mut hits_r5 = 0usize;
        let mut hits_r10 = 0usize;
        let mut ndcg_sum = 0.0f64;
        let mut cluster_coherence_sum = 0.0f64;
        let mut avg_search_ms = 0.0f64;
        let mut scenarios = Vec::new();

        for i in 0..scenario_count {
            let target = &all_memories[i * step];
            let query_words: Vec<&str> = target.content.split_whitespace().take(8).collect();
            if query_words.len() < 2 { continue; }
            let query = query_words.join(" ");

            let start = std::time::Instant::now();
            let results = self.search(&query, 10, target.project.as_deref(), None, None, None)?;
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            avg_search_ms += elapsed_ms;

            let result_ids: Vec<&str> = results.iter().map(|r| r.memory.id.as_str()).collect();

            // R@5
            let in_top5 = result_ids.iter().take(5).any(|id| *id == target.id);
            if in_top5 { hits_r5 += 1; }

            // R@10
            let in_top10 = result_ids.iter().take(10).any(|id| *id == target.id);
            if in_top10 { hits_r10 += 1; }

            // NDCG@10 (single relevant item)
            let ndcg = if let Some(pos) = result_ids.iter().position(|id| *id == target.id) {
                if pos < 10 { 1.0 / (pos as f64 + 2.0).log2() } else { 0.0 }
            } else {
                0.0
            };
            ndcg_sum += ndcg;

            // Cluster coherence: what fraction of top-5 results share at least one entity?
            let coherence = self.measure_cluster_coherence(&results);
            cluster_coherence_sum += coherence;

            scenarios.push(serde_json::json!({
                "query": query,
                "target_id": target.id,
                "target_kind": target.kind,
                "target_project": target.project,
                "r5_hit": in_top5,
                "r10_hit": in_top10,
                "ndcg10": (ndcg * 1000.0).round() / 1000.0,
                "cluster_coherence": (coherence * 1000.0).round() / 1000.0,
                "search_ms": (elapsed_ms * 100.0).round() / 100.0,
                "results_returned": results.len(),
            }));
        }

        let actual_count = scenarios.len().max(1);
        let r5_rate = (hits_r5 as f64 / actual_count as f64 * 1000.0).round() / 10.0;
        let r10_rate = (hits_r10 as f64 / actual_count as f64 * 1000.0).round() / 10.0;
        let ndcg10 = (ndcg_sum / actual_count as f64 * 1000.0).round() / 10.0;
        let coherence = (cluster_coherence_sum / actual_count as f64 * 1000.0).round() / 10.0;
        avg_search_ms = (avg_search_ms / actual_count as f64 * 100.0).round() / 100.0;

        Ok(serde_json::json!({
            "status": "ok",
            "memory_count": all_memories.len(),
            "scenario_count": actual_count,
            "metrics": {
                "R@5": format!("{}%", r5_rate),
                "R@10": format!("{}%", r10_rate),
                "NDCG@10": format!("{}%", ndcg10),
                "cluster_coherence": format!("{}%", coherence),
                "avg_search_ms": format!("{:.2}ms", avg_search_ms),
            },
            "scenarios": scenarios,
        }))
    }

    fn list_all_memories_for_benchmark(&self) -> Result<Vec<Memory>, String> {
        let mut stmt = self.conn.prepare(
            "SELECT id,content,kind,project,tags,source,importance,expires_at,metadata,created_at,updated_at,last_accessed_at,access_count \
             FROM memories WHERE kind != 'transcript_chunk' AND length(content) > 20 ORDER BY created_at DESC LIMIT 500"
        ).map_err(|e| format!("Benchmark list: {}", e))?;
        let rows = stmt.query_map([], |r| Ok(row_to_memory(r))).map_err(|e| format!("Benchmark query: {}", e))?;
        Ok(rows.flatten().collect())
    }

    fn measure_cluster_coherence(&self, results: &[SearchResult]) -> f64 {
        if results.len() < 2 { return 1.0; }
        let top5: Vec<&str> = results.iter().take(5).map(|r| r.memory.id.as_str()).collect();
        if top5.len() < 2 { return 1.0; }

        let placeholders: Vec<String> = (1..=top5.len()).map(|i| format!("?{}", i)).collect();
        let sql = format!(
            "SELECT COUNT(DISTINCT a.memory_id || ':' || b.memory_id) FROM memory_entities a \
             JOIN memory_entities b ON a.entity_value = b.entity_value AND a.entity_kind = b.entity_kind AND a.memory_id < b.memory_id \
             WHERE a.memory_id IN ({0}) AND b.memory_id IN ({0})",
            placeholders.join(",")
        );

        let connected_pairs: i64 = if let Ok(mut stmt) = self.conn.prepare(&sql) {
            let params: Vec<&dyn rusqlite::types::ToSql> = top5.iter().map(|id| id as &dyn rusqlite::types::ToSql).collect();
            stmt.query_row(params.as_slice(), |r| r.get(0)).unwrap_or(0)
        } else { 0 };

        let max_pairs = (top5.len() * (top5.len() - 1)) / 2;
        if max_pairs == 0 { return 1.0; }
        (connected_pairs as f64 / max_pairs as f64).min(1.0)
    }

    // ─── IMPORT / MIGRATE ─────────────────────────────

    pub fn import_batch(&self, memories: &[(String, String, Option<String>, Vec<String>, String)]) -> Result<usize, String> {
        let tx = self.conn.unchecked_transaction().map_err(|e| format!("Tx: {}", e))?;
        let mut count = 0;
        for (content, kind, project, tags, source) in memories {
            let exists: bool = tx.query_row(
                "SELECT EXISTS(SELECT 1 FROM memories WHERE content=?1)", params![content], |r| r.get(0)
            ).unwrap_or(false);
            if exists { continue; }
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
            tx.execute(
                "INSERT INTO memories_fts (rowid,content,tags,kind,project) VALUES (?1,?2,?3,?4,?5)",
                params![rowid, content, tags_json, kind, canonical_project.as_deref().unwrap_or("")],
            ).map_err(|e| format!("FTS: {}", e))?;
            if let Some(p) = canonical_project.as_deref() {
                let _ = tx.execute("INSERT OR IGNORE INTO projects (name,path,created_at) VALUES (?1,'',?2)", params![p, now]);
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
                        for m in memories { parse_v1_memory(m, None, &mut batch); }
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
                    if path.extension().and_then(|e| e.to_str()) != Some("json") { continue; }
                    let proj_name = path.file_stem().and_then(|n| n.to_str()).unwrap_or("unknown").to_string();
                    if let Ok(content) = std::fs::read_to_string(&path) {
                        if let Ok(store) = serde_json::from_str::<serde_json::Value>(&content) {
                            if let Some(memories) = store.get("memories").and_then(|v| v.as_array()) {
                                for m in memories { parse_v1_memory(m, Some(proj_name.clone()), &mut batch); }
                            }
                        }
                    }
                }
            }
        }
        self.import_batch(&batch)
    }
    // ─── LongMemEval Benchmark (ICLR 2025) ────────────────────
    // Pure retrieval evaluation on the standard academic benchmark.
    // For each question: index ~48 sessions into a temp DB, search, check if gold session is in top-K.

    pub fn benchmark_longmemeval(dataset_path: &str, limit: Option<usize>) -> Result<serde_json::Value, String> {
        use serde_json::json;

        eprintln!("[LongMemEval] Loading dataset: {}", dataset_path);
        let raw = std::fs::read_to_string(dataset_path)
            .map_err(|e| format!("Cannot read dataset: {}", e))?;

        let entries: Vec<serde_json::Value> = serde_json::from_str(&raw)
            .map_err(|e| format!("Invalid JSON: {}", e))?;

        let total = entries.len();
        eprintln!("[LongMemEval] {} questions loaded", total);

        // Filter out abstention questions (question_id ending with _abs)
        let questions: Vec<&serde_json::Value> = entries.iter()
            .filter(|e| {
                let qid = e.get("question_id").and_then(|v| v.as_str()).unwrap_or("");
                !qid.ends_with("_abs")
            })
            .collect();

        let abstention_count = total - questions.len();
        let eval_count = if let Some(lim) = limit { lim.min(questions.len()) } else { questions.len() };
        eprintln!("[LongMemEval] Evaluating {} questions ({} abstention skipped)", eval_count, abstention_count);

        let mut hits_r5 = 0usize;
        let mut hits_r10 = 0usize;
        let mut ndcg_sum = 0.0f64;
        let mut mrr_sum = 0.0f64;
        let mut total_search_ms = 0.0f64;
        let mut category_stats: std::collections::HashMap<String, (usize, usize, usize, f64, f64)> = std::collections::HashMap::new();

        for (qi, entry) in questions.iter().take(eval_count).enumerate() {
            let question = entry.get("question").and_then(|v| v.as_str()).unwrap_or("");
            let question_type = entry.get("question_type").and_then(|v| v.as_str()).unwrap_or("unknown");
            let answer_session_ids: Vec<String> = entry.get("answer_session_ids")
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                .unwrap_or_default();
            let sessions = entry.get("haystack_sessions").and_then(|v| v.as_array());
            let session_ids = entry.get("haystack_session_ids").and_then(|v| v.as_array());

            if sessions.is_none() || session_ids.is_none() || question.is_empty() {
                eprintln!("[LongMemEval] Skipping question {} (missing fields)", qi);
                continue;
            }
            let sessions = sessions.unwrap();
            let session_ids = session_ids.unwrap();

            // Create temp DB
            let tmp_path = std::env::temp_dir().join(format!("memorypilot_lme_{}.db", qi));
            let _ = std::fs::remove_file(&tmp_path);
            let _ = std::fs::remove_file(tmp_path.with_extension("db-wal"));
            let _ = std::fs::remove_file(tmp_path.with_extension("db-shm"));

            let db = match Self::open_lme_db(&tmp_path) {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("[LongMemEval] Q{}: DB open failed: {}", qi, e);
                    continue;
                }
            };

            eprintln!("[LongMemEval] Q{}/{} ({}): indexing {} sessions...", qi + 1, questions.len(), question_type, sessions.len());

            // Index sessions at turn-level granularity with batch embedding
            let mut all_texts: Vec<String> = Vec::new();
            let mut all_ids: Vec<String> = Vec::new();

            for (si, (session, sid_val)) in sessions.iter().zip(session_ids.iter()).enumerate() {
                let fallback_sid = format!("session_{}", si);
                let sid = sid_val.as_str().unwrap_or(&fallback_sid);
                let turns = match session.as_array() {
                    Some(t) => t,
                    None => continue,
                };
                for (ti, turn) in turns.iter().enumerate() {
                    let role = turn.get("role").and_then(|v| v.as_str()).unwrap_or("user");
                    let content = turn.get("content").and_then(|v| v.as_str()).unwrap_or("");
                    if content.is_empty() { continue; }
                    all_texts.push(format!("{}: {}", role, content));
                    all_ids.push(format!("{}__t{}", sid, ti));
                }
            }

            if !all_texts.is_empty() {
                eprintln!("[LongMemEval] Q{}: embedding {} turns in batches...", qi + 1, all_texts.len());
                let now = Utc::now().to_rfc3339();
                let batch_size = 128;

                for chunk_start in (0..all_texts.len()).step_by(batch_size) {
                    let chunk_end = (chunk_start + batch_size).min(all_texts.len());
                    let chunk_refs: Vec<&str> = all_texts[chunk_start..chunk_end].iter().map(|s| s.as_str()).collect();
                    let chunk_embeddings = crate::embedding::embed_batch(&chunk_refs);

                    for (ci, emb) in chunk_embeddings.into_iter().enumerate() {
                        let idx = chunk_start + ci;
                        let blob = crate::embedding::vec_to_blob(&emb);
                        let hash = content_hash(&all_texts[idx]);
                        let _ = db.conn.execute(
                            "INSERT INTO memories (id, content, kind, project, tags, source, importance, embedding, content_hash, created_at, updated_at)
                             VALUES (?1, ?2, 'session', NULL, '[]', 'longmemeval', 3, ?3, ?4, ?5, ?5)",
                            params![all_ids[idx], all_texts[idx], blob, hash, now],
                        );
                        let _ = db.conn.execute(
                            "INSERT INTO memories_fts (rowid, content, tags, kind, project) VALUES ((SELECT rowid FROM memories WHERE id=?1), ?2, '[]', 'session', '')",
                            params![all_ids[idx], all_texts[idx]],
                        );
                    }
                }
                eprintln!("[LongMemEval] Q{}: indexed {} turns ✓", qi + 1, all_texts.len());
            }

            // Search
            let start = std::time::Instant::now();
            let results = match db.search(question, 10, None, None, None, None) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("[LongMemEval] Q{}: search failed: {}", qi, e);
                    let _ = std::fs::remove_file(&tmp_path);
                    continue;
                }
            };
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            total_search_ms += elapsed_ms;

            let result_ids: Vec<&str> = results.iter().map(|r| r.memory.id.as_str()).collect();

            // recall_any@K: does ANY turn from a gold session appear in top-K?
            // Turn IDs are formatted as "session_id__tN", match by prefix
            let matches_gold = |id: &&str| -> bool {
                answer_session_ids.iter().any(|gold| id.starts_with(gold))
            };

            let in_top5 = result_ids.iter().take(5).any(matches_gold);
            let in_top10 = result_ids.iter().take(10).any(matches_gold);

            if in_top5 { hits_r5 += 1; }
            if in_top10 { hits_r10 += 1; }

            // NDCG@10 (single relevant item — best rank)
            let best_rank = result_ids.iter().take(10).position(matches_gold);
            let ndcg = match best_rank {
                Some(pos) => 1.0 / (pos as f64 + 2.0).log2(),
                None => 0.0,
            };
            ndcg_sum += ndcg;

            // MRR
            let rr = match best_rank {
                Some(pos) => 1.0 / (pos as f64 + 1.0),
                None => 0.0,
            };
            mrr_sum += rr;

            // Per-category stats: (count, r5_hits, r10_hits, ndcg_sum, mrr_sum)
            let cat = category_stats.entry(question_type.to_string()).or_insert((0, 0, 0, 0.0, 0.0));
            cat.0 += 1;
            if in_top5 { cat.1 += 1; }
            if in_top10 { cat.2 += 1; }
            cat.3 += ndcg;
            cat.4 += rr;

            if (qi + 1) % 10 == 0 || qi + 1 == eval_count {
                let running_r5 = hits_r5 as f64 / (qi + 1) as f64 * 100.0;
                let running_r10 = hits_r10 as f64 / (qi + 1) as f64 * 100.0;
                eprintln!("[LongMemEval] {}/{} — R@5: {:.1}% R@10: {:.1}% ({} sessions indexed)",
                    qi + 1, eval_count, running_r5, running_r10, sessions.len());
            }

            // Cleanup temp DB
            let _ = std::fs::remove_file(&tmp_path);
            let _ = std::fs::remove_file(tmp_path.with_extension("db-wal"));
            let _ = std::fs::remove_file(tmp_path.with_extension("db-shm"));
        }

        let actual_count = eval_count.max(1) as f64;
        let r5 = (hits_r5 as f64 / actual_count * 1000.0).round() / 10.0;
        let r10 = (hits_r10 as f64 / actual_count * 1000.0).round() / 10.0;
        let ndcg10 = (ndcg_sum / actual_count * 1000.0).round() / 10.0;
        let mrr = (mrr_sum / actual_count * 1000.0).round() / 10.0;
        let avg_ms = (total_search_ms / actual_count * 100.0).round() / 100.0;

        let mut by_category = serde_json::Map::new();
        for (cat, (count, r5h, r10h, ndcg_s, mrr_s)) in &category_stats {
            let c = *count as f64;
            by_category.insert(cat.clone(), json!({
                "count": count,
                "recall_at_5": format!("{:.1}%", *r5h as f64 / c * 100.0),
                "recall_at_10": format!("{:.1}%", *r10h as f64 / c * 100.0),
                "ndcg_at_10": format!("{:.1}%", ndcg_s / c * 100.0),
                "mrr": format!("{:.1}%", mrr_s / c * 100.0),
            }));
        }

        Ok(json!({
            "benchmark": "LongMemEval-S (ICLR 2025)",
            "dataset": dataset_path,
            "questions_total": total,
            "questions_evaluated": eval_count,
            "questions_abstention_skipped": abstention_count,
            "granularity": "turn",
            "embedding_model": "multilingual-e5-small (384-dim)",
            "search_engine": "BM25 + cosine RRF (k=40)",
            "metrics": {
                "recall_at_5": format!("{}%", r5),
                "recall_at_10": format!("{}%", r10),
                "ndcg_at_10": format!("{}%", ndcg10),
                "mrr": format!("{}%", mrr),
                "avg_search_latency_ms": avg_ms,
            },
            "by_category": by_category,
        }))
    }

    /// Open a lightweight DB for LongMemEval benchmarking (no backfill, no read pool)
    fn open_lme_db(path: &Path) -> Result<Self, String> {
        let conn = Connection::open(path).map_err(|e| format!("SQLite open: {}", e))?;
        conn.execute_batch("
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = NORMAL;
            PRAGMA cache_size = -4000;
            PRAGMA foreign_keys = ON;
        ").map_err(|e| format!("Pragma: {}", e))?;

        let mut read_pool = Vec::with_capacity(1);
        let rc = Connection::open_with_flags(path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX)
            .map_err(|e| format!("Read pool: {}", e))?;
        let _ = rc.execute_batch("PRAGMA cache_size = -2000;");
        read_pool.push(Mutex::new(rc));

        let db = Self { conn, read_pool };
        db.init_schema()?;
        Ok(db)
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

#[derive(Debug, Clone)]
struct DistilledTranscriptCandidate {
    score: i32,
    normalized_key: String,
    item: BulkItem,
}

#[derive(Debug, Clone)]
struct BenchmarkScenarioSpec {
    name: &'static str,
    query: &'static str,
    project: Option<&'static str>,
    expected_kind: Option<&'static str>,
    match_substrings: &'static [&'static str],
    mode: RecallMode,
}

#[derive(Debug, Clone)]
struct BenchmarkScenarioRun {
    name: String,
    source: String,
    query: String,
    mode: RecallMode,
    expected_memory: Memory,
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
fn default_kind() -> String { "fact".into() }
fn default_source() -> String { "cursor".into() }

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

fn parse_v1_memory(m: &serde_json::Value, project: Option<String>, batch: &mut Vec<(String, String, Option<String>, Vec<String>, String)>) {
    let c = m.get("content").and_then(|v| v.as_str()).unwrap_or("").to_string();
    if c.is_empty() { return; }
    let k = m.get("kind").or(m.get("type")).and_then(|v| v.as_str()).unwrap_or("fact");
    let kind = match k { "context"=>"fact", "architecture"=>"decision", "component"|"workflow"=>"pattern", o=>o }.to_string();
    let tags: Vec<String> = m.get("tags").and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect()).unwrap_or_default();
    let source = m.get("source").and_then(|v| v.as_str()).unwrap_or("v1-import").to_string();
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
            assert_eq!(stored_path, Database::normalize_path(&project_root.to_string_lossy()));
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
                .recall(Some("Planify"), None, Some("token"), RecallMode::Safe, false, false, &MemoryScope::default())
                .expect("safe recall");
            let full = db
                .recall(Some("Planify"), None, Some("token"), RecallMode::Full, false, false, &MemoryScope::default())
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
                .recall(Some("Planify"), None, Some("SvelteKit"), RecallMode::Safe, true, false, &MemoryScope::default())
                .expect("explained recall");

            let explain = explained.get("explain").expect("explain block");
            assert!(explain.get("selected_memories").and_then(|value| value.as_array()).map(|items| !items.is_empty()).unwrap_or(false));
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
                .recall(Some("planify"), None, None, RecallMode::Safe, true, false, &scope)
                .expect("scoped recall");

            let scope_context = recall
                .get("scope_context")
                .and_then(|value| value.as_array())
                .expect("scope context");
            assert!(!scope_context.is_empty());

            let explain_text = serde_json::to_string(recall.get("explain").expect("explain")).expect("render explain");
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
            assert!(report.get("scenario_count").and_then(|value| value.as_u64()).unwrap_or(0) >= 2);
            assert!(report.get("golden_defined_count").and_then(|value| value.as_u64()).unwrap_or(0) >= 10);
            assert!(report.get("top5_hit_rate").and_then(|value| value.as_f64()).is_some());
            assert!(report.get("cross_project_leak_rate").and_then(|value| value.as_f64()).is_some());
            assert!(report.get("credential_leak_rate_safe").and_then(|value| value.as_f64()).is_some());
            assert!(report.get("scenario_source_counts").and_then(|value| value.get("golden")).and_then(|value| value.as_u64()).unwrap_or(0) >= 1);
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
            assert!(report.get("golden_executed_count").and_then(|value| value.as_u64()).unwrap_or(0) >= 2);
            assert!(
                report
                    .get("scenarios")
                    .and_then(|value| value.as_array())
                    .map(|items| items.iter().any(|item| item.get("scenario_source").and_then(|value| value.as_str()) == Some("golden")))
                    .unwrap_or(false)
            );
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

            assert!(memories.iter().any(|memory| memory.kind == "preference" && memory.content.contains("safe mode")));
            assert!(memories.iter().any(|memory| memory.kind == "decision" && memory.content.contains("benchmark_recall")));
            assert!(memories.iter().any(|memory| memory.kind == "bug" && memory.content.contains("cross-project leakage")));
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
}