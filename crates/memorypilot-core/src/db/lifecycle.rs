//! Opening the database: connections, startup embedding sync and backfill, ANN warm-up.

use super::*;

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
        // from an older default), the stored vectors no longer share
        // the active model's geometry. Wipe them so the backfill pass
        // below re-embeds everything with the new model. Cheap one-shot
        // SQL — no scan happens once everything is up to date.
        let _ = db.sync_embedding_model();
        db.backfill_missing_at_startup();
        let _ = db.migrate_fts_to_stemmed();
        if let Err(error) = db.migrate_credential_masking() {
            eprintln!("[MemoryPilot] Credential masking migration: {}", error);
        }
        Self::spawn_ann_warmup(ann, path.to_path_buf(), ann_warm_complete);
        Ok(db)
    }

    /// Opens the database at `path`, applies this version's schema
    /// upgrades and counts the memories, without loading a model or the
    /// ANN index. The updater runs it with the new binary on a copy of the
    /// user's database before switching binaries (`--self-test`).
    pub fn check_schema_at(path: &Path) -> Result<usize, String> {
        let db = Self::open_maintenance(path)?;
        db.init_schema()?;
        db.upgrade_schema()?;
        db.normalize_project_identities()?;
        db.migrate_fts_to_stemmed()?;
        let count: i64 = db
            .conn
            .query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0))
            .map_err(|error| format!("count memories: {}", error))?;
        usize::try_from(count).map_err(|_| format!("invalid memory count {}", count))
    }

    /// A second handle on the same database for background maintenance
    /// (episodic rollup). One writer connection, one reader, no ANN
    /// index, none of the startup work `open_at` does — cheap enough to
    /// open per job on a thread and drop when the job is done.
    pub(crate) fn open_maintenance(path: &Path) -> Result<Self, String> {
        let conn = Connection::open(path).map_err(|e| format!("SQLite open: {}", e))?;
        schema::configure_connection(&conn)?;
        let read_pool = schema::open_read_pool(path, 1)?;
        Ok(Self {
            conn,
            read_pool,
            ann: None,
            ann_warm_complete: Arc::new(std::sync::atomic::AtomicBool::new(true)),
        })
    }

    /// Reconcile the stored embedding fingerprint with the active model.
    ///
    /// The database remembers which encoder produced its vectors
    /// (`config.embedding_model`). When the active model differs — or
    /// the fingerprint is missing on a database that already holds
    /// vectors, i.e. an upgrade from a pre-fingerprint release — every
    /// embedding is set to NULL and the ANN index is emptied, then the
    /// fingerprint is rewritten. Blob length alone cannot do this job:
    /// multilingual-e5-base and jina-nano are both 768-dim.
    pub(super) fn sync_embedding_model(&self) -> Result<usize, String> {
        let active = crate::embedding::model_label();
        let stored = self.get_config(EMBEDDING_MODEL_KEY);
        let mut invalidated = self.invalidate_stale_embeddings()?;
        if stored.as_deref() == Some(active) {
            return Ok(invalidated);
        }
        let with_vectors = self
            .conn
            .execute(
                "UPDATE memories SET embedding = NULL WHERE embedding IS NOT NULL",
                [],
            )
            .map_err(|error| format!("Embedding model swap invalidate: {}", error))?;
        // Episode vectors come from the same encoder: rebuild them with the new one.
        let _ = self.conn.execute("DELETE FROM episodes", []);
        if with_vectors > 0 {
            invalidated += with_vectors;
            if let Some(ann) = self.ann.as_ref() {
                let _ = ann.clear();
            }
            eprintln!(
                "[MemoryPilot] Embedding model changed ({} → {}): {} vectors dropped and queued for re-embedding.",
                stored.as_deref().unwrap_or("unknown"),
                active,
                with_vectors
            );
        }
        self.set_config(EMBEDDING_MODEL_KEY, active)?;
        Ok(invalidated)
    }

    /// Embed every memory that lacks a vector. A handful is done inline
    /// so a fresh install returns fully searchable; a whole corpus (model
    /// swap, upgrade) goes to the background embed worker so `open_at`
    /// — and therefore the MCP handshake — returns at once. BM25 keeps
    /// answering while vectors stream back in.
    pub(super) fn backfill_missing_at_startup(&self) {
        let mut stmt = match self
            .conn
            .prepare("SELECT id, content, kind, project, created_at FROM memories WHERE embedding IS NULL")
        {
            Ok(stmt) => stmt,
            Err(_) => return,
        };
        let missing: Vec<(String, String, String)> = match stmt.query_map([], |row| {
            let id: String = row.get(0)?;
            let content: String = row.get(1)?;
            let kind: String = row.get(2)?;
            let project: Option<String> = row.get(3)?;
            let created_at: String = row.get(4)?;
            let embed_text =
                Self::embedding_input(&kind, project.as_deref(), &created_at, &content);
            Ok((id, content, embed_text))
        }) {
            Ok(rows) => rows.flatten().collect(),
            Err(_) => return,
        };
        if missing.is_empty() {
            return;
        }
        if missing.len() <= SYNC_BACKFILL_MAX {
            let _ = self.backfill_embeddings();
            return;
        }
        let count = missing.len();
        if queue_startup_backfill(missing) {
            eprintln!(
                "[MemoryPilot] Re-embedding {} memories in the background with {} — vector recall ramps up over the next minutes, BM25 serves meanwhile.",
                count,
                crate::embedding::model_label()
            );
        }
    }

    /// Set `embedding = NULL` on every row whose blob length doesn't
    /// match the active model's expected length. Keeps the row, just
    /// queues a re-embed.
    pub(super) fn invalidate_stale_embeddings(&self) -> Result<usize, String> {
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
    pub(super) fn migrate_fts_to_stemmed(&self) -> Result<(), String> {
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
    pub(super) fn spawn_ann_warmup(
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

    pub(super) fn open_ann_index(path: &Path) -> Option<Arc<crate::ann::AnnIndex>> {
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
    pub(super) fn warm_ann_index(&self) -> Result<(), String> {
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
    pub(super) fn fts_index_content(content: &str) -> String {
        let stem = crate::stemming::stem_text(content);
        if stem.is_empty() {
            content.to_string()
        } else {
            format!("{} {}", content, stem)
        }
    }

    pub(super) fn read_conn(&self) -> std::sync::MutexGuard<'_, Connection> {
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
}
