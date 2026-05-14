use rusqlite::{params, Connection};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, RwLock};

use super::content_hash;
use crate::ann::AnnIndex;

const EMBED_CACHE_SIZE: usize = 256;
const DISK_CACHE_SOFT_CAP: usize = 8_192;

struct EmbedJob {
    id: String,
    content: String,
}

static EMBED_QUEUE: OnceLock<Mutex<Vec<EmbedJob>>> = OnceLock::new();
static EMBED_DB_PATH: OnceLock<PathBuf> = OnceLock::new();
static EMBED_WORKER_STARTED: OnceLock<()> = OnceLock::new();
static EMBED_ANN_INDEX: OnceLock<RwLock<Option<Arc<AnnIndex>>>> = OnceLock::new();

struct EmbedCache {
    entries: Vec<(String, Vec<f32>)>,
}

static EMBED_CACHE: OnceLock<Mutex<EmbedCache>> = OnceLock::new();

/// Async access-count queue.
///
/// Hot-path searches push memory ids here instead of issuing an
/// `UPDATE memories SET access_count = ...` directly. Flushing is
/// done in batch by the embed worker thread, which removes WAL writer
/// contention from the search path. Precision trade-off: many accesses
/// to the same id within the flush window count as a single increment.
static ACCESS_QUEUE: OnceLock<Mutex<Vec<String>>> = OnceLock::new();
const ACCESS_FLUSH_INTERVAL: std::time::Duration = std::time::Duration::from_secs(2);

fn access_queue() -> &'static Mutex<Vec<String>> {
    ACCESS_QUEUE.get_or_init(|| Mutex::new(Vec::new()))
}

pub(super) fn queue_access_update(memory_id: String) {
    if let Ok(mut q) = access_queue().lock() {
        q.push(memory_id);
    }
}

fn last_access_flush() -> &'static Mutex<std::time::Instant> {
    static LAST: OnceLock<Mutex<std::time::Instant>> = OnceLock::new();
    LAST.get_or_init(|| Mutex::new(std::time::Instant::now() - ACCESS_FLUSH_INTERVAL))
}

fn flush_access_queue(conn: &Connection) {
    let due = {
        let mut last = match last_access_flush().lock() {
            Ok(g) => g,
            Err(_) => return,
        };
        if last.elapsed() < ACCESS_FLUSH_INTERVAL {
            return;
        }
        *last = std::time::Instant::now();
        true
    };
    if !due {
        return;
    }

    let drained: Vec<String> = match access_queue().lock() {
        Ok(mut q) => q.drain(..).collect(),
        Err(_) => return,
    };
    if drained.is_empty() {
        return;
    }

    // Deduplicate while preserving uniqueness; we trade a small amount of
    // counter precision for one round-trip instead of N.
    let mut unique: std::collections::HashSet<String> =
        std::collections::HashSet::with_capacity(drained.len());
    unique.extend(drained);
    let ids: Vec<String> = unique.into_iter().collect();
    if ids.is_empty() {
        return;
    }

    let placeholders: Vec<String> = (1..=ids.len()).map(|i| format!("?{}", i + 1)).collect();
    let sql = format!(
        "UPDATE memories \
         SET access_count = access_count + 1, last_accessed_at = ?1 \
         WHERE id IN ({})",
        placeholders.join(", ")
    );
    let now = chrono::Utc::now().to_rfc3339();
    let mut params_vec: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::with_capacity(ids.len() + 1);
    params_vec.push(Box::new(now));
    for id in &ids {
        params_vec.push(Box::new(id.clone()));
    }
    let refs: Vec<&dyn rusqlite::types::ToSql> = params_vec.iter().map(|b| &**b).collect();
    let _ = conn.execute(&sql, refs.as_slice());
}

pub(super) fn set_embed_db_path(path: &Path) {
    let _ = EMBED_DB_PATH.set(path.to_path_buf());
}

pub(super) fn set_embed_ann_index(ann: Option<Arc<AnnIndex>>) {
    let slot = EMBED_ANN_INDEX.get_or_init(|| RwLock::new(None));
    if let Ok(mut guard) = slot.write() {
        *guard = ann;
    }
}

fn embed_ann_index() -> Option<Arc<AnnIndex>> {
    EMBED_ANN_INDEX
        .get()
        .and_then(|slot| slot.read().ok().and_then(|guard| guard.clone()))
}

pub(super) fn queue_embedding_job(id: &str, content: &str) {
    if let Ok(mut queue) = embed_queue().lock() {
        queue.push(EmbedJob {
            id: id.to_string(),
            content: content.to_string(),
        });
    }
    ensure_embed_worker();
}

pub(super) fn cached_embed_text(text: &str) -> Vec<f32> {
    if let Ok(cache) = embed_cache().lock() {
        if let Some(embedding) = cache.get(text) {
            return embedding.clone();
        }
    }

    if let Some(embedding) = read_disk_query_cache(text) {
        if let Ok(mut cache) = embed_cache().lock() {
            cache.insert(text.to_string(), embedding.clone());
        }
        return embedding;
    }

    let embedding = crate::embedding::embed_text(text);
    if let Ok(mut cache) = embed_cache().lock() {
        cache.insert(text.to_string(), embedding.clone());
    }
    write_disk_query_cache(text, &embedding);
    embedding
}

fn query_cache_path() -> Option<PathBuf> {
    let db_path = EMBED_DB_PATH.get()?;
    let mut path = db_path.clone();
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .map(|name| format!("{}.query_cache.sqlite", name))
        .unwrap_or_else(|| "memorypilot.query_cache.sqlite".to_string());
    path.set_file_name(file_name);
    Some(path)
}

fn open_query_cache() -> Option<Connection> {
    let path = query_cache_path()?;
    let conn = Connection::open(&path).ok()?;
    let _ = conn.execute_batch(
        "PRAGMA journal_mode = WAL;
         PRAGMA synchronous = NORMAL;
         CREATE TABLE IF NOT EXISTS query_cache (
             text_hash TEXT PRIMARY KEY,
             embedding BLOB NOT NULL,
             last_used INTEGER NOT NULL
         );",
    );
    Some(conn)
}

fn read_disk_query_cache(text: &str) -> Option<Vec<f32>> {
    let conn = open_query_cache()?;
    let key = content_hash(text);
    let blob: Vec<u8> = conn
        .query_row(
            "SELECT embedding FROM query_cache WHERE text_hash = ?1",
            params![&key],
            |row| row.get(0),
        )
        .ok()?;
    let now = chrono::Utc::now().timestamp();
    let _ = conn.execute(
        "UPDATE query_cache SET last_used = ?1 WHERE text_hash = ?2",
        params![now, &key],
    );
    let vector = crate::embedding::blob_to_vec(&blob);
    if vector.is_empty() {
        None
    } else {
        Some(vector)
    }
}

fn write_disk_query_cache(text: &str, embedding: &[f32]) {
    let Some(conn) = open_query_cache() else {
        return;
    };
    let key = content_hash(text);
    let blob = crate::embedding::vec_to_blob(embedding);
    let now = chrono::Utc::now().timestamp();
    let _ = conn.execute(
        "INSERT OR REPLACE INTO query_cache (text_hash, embedding, last_used) VALUES (?1, ?2, ?3)",
        params![&key, blob, now],
    );

    // Soft trim: when the cache grows past the cap, drop the LRU half in one shot.
    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM query_cache", [], |row| row.get(0))
        .unwrap_or(0);
    if count as usize > DISK_CACHE_SOFT_CAP {
        let target = (DISK_CACHE_SOFT_CAP / 2) as i64;
        let _ = conn.execute(
            "DELETE FROM query_cache WHERE text_hash IN (
                 SELECT text_hash FROM query_cache ORDER BY last_used ASC LIMIT ?1
             )",
            params![target],
        );
    }
}

fn embed_queue() -> &'static Mutex<Vec<EmbedJob>> {
    EMBED_QUEUE.get_or_init(|| Mutex::new(Vec::new()))
}

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
            let mut queue = match embed_queue().lock() {
                Ok(queue) => queue,
                Err(_) => continue,
            };
            queue.drain(..).collect()
        };

        // Even with no embed jobs we still want to drain the access
        // queue periodically so search counters do not grow stale.
        let db_path = match EMBED_DB_PATH.get() {
            Some(path) => path.clone(),
            None => {
                std::thread::sleep(std::time::Duration::from_millis(500));
                continue;
            }
        };

        let conn = match Connection::open(&db_path) {
            Ok(conn) => conn,
            Err(_) => {
                std::thread::sleep(std::time::Duration::from_millis(500));
                continue;
            }
        };
        let _ = conn.execute_batch("PRAGMA journal_mode = WAL; PRAGMA synchronous = NORMAL;");

        flush_access_queue(&conn);

        if jobs.is_empty() {
            std::thread::sleep(std::time::Duration::from_millis(500));
            continue;
        }

        let texts: Vec<&str> = jobs.iter().map(|job| job.content.as_str()).collect();
        let embeddings = crate::embedding::embed_batch(&texts);

        let ann = embed_ann_index();
        let mut ann_pushed = 0usize;
        for (job, embedding) in jobs.iter().zip(embeddings.iter()) {
            let blob = crate::embedding::vec_to_blob(embedding);
            let hash = content_hash(&job.content);
            let _ = conn.execute(
                "UPDATE memories SET embedding = ?1, content_hash = ?2 WHERE id = ?3 AND embedding IS NULL",
                params![blob, &hash, &job.id],
            );
            if let Some(index) = ann.as_ref() {
                if index.add(&job.id, embedding).is_ok() {
                    ann_pushed += 1;
                }
            }
        }
        if ann_pushed > 0 {
            if let Some(index) = ann.as_ref() {
                // Debounce ANN persistence so the embed worker does not
                // serialize every batch onto disk, which used to block
                // concurrent searches on the WAL writer lock. We persist
                // at most once every 5 seconds; the in-memory index is
                // always up to date for searches.
                let should_persist = {
                    let mut last = last_ann_persist().lock().expect("ann persist lock");
                    if last.elapsed() >= std::time::Duration::from_secs(5) {
                        *last = std::time::Instant::now();
                        true
                    } else {
                        false
                    }
                };
                if should_persist {
                    let _ = index.persist();
                }
            }
        }
    }
}

fn last_ann_persist() -> &'static std::sync::Mutex<std::time::Instant> {
    static LAST: std::sync::OnceLock<std::sync::Mutex<std::time::Instant>> =
        std::sync::OnceLock::new();
    LAST.get_or_init(|| {
        std::sync::Mutex::new(std::time::Instant::now() - std::time::Duration::from_secs(60))
    })
}

impl EmbedCache {
    fn new() -> Self {
        Self {
            entries: Vec::with_capacity(EMBED_CACHE_SIZE),
        }
    }

    fn get(&self, text: &str) -> Option<&Vec<f32>> {
        self.entries
            .iter()
            .find(|(key, _)| key == text)
            .map(|(_, value)| value)
    }

    fn insert(&mut self, text: String, embedding: Vec<f32>) {
        if let Some(position) = self.entries.iter().position(|(key, _)| key == &text) {
            self.entries.remove(position);
        }
        if self.entries.len() >= EMBED_CACHE_SIZE {
            self.entries.remove(0);
        }
        self.entries.push((text, embedding));
    }
}

fn embed_cache() -> &'static Mutex<EmbedCache> {
    EMBED_CACHE.get_or_init(|| Mutex::new(EmbedCache::new()))
}
