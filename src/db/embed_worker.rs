use rusqlite::{params, Connection};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use super::content_hash;

const EMBED_CACHE_SIZE: usize = 64;

struct EmbedJob {
    id: String,
    content: String,
}

static EMBED_QUEUE: OnceLock<Mutex<Vec<EmbedJob>>> = OnceLock::new();
static EMBED_DB_PATH: OnceLock<PathBuf> = OnceLock::new();
static EMBED_WORKER_STARTED: OnceLock<()> = OnceLock::new();

struct EmbedCache {
    entries: Vec<(String, Vec<f32>)>,
}

static EMBED_CACHE: OnceLock<Mutex<EmbedCache>> = OnceLock::new();

pub(super) fn set_embed_db_path(path: &Path) {
    let _ = EMBED_DB_PATH.set(path.to_path_buf());
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

    let embedding = crate::embedding::embed_text(text);
    if let Ok(mut cache) = embed_cache().lock() {
        cache.insert(text.to_string(), embedding.clone());
    }
    embedding
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

        if jobs.is_empty() {
            std::thread::sleep(std::time::Duration::from_millis(500));
            continue;
        }

        let db_path = match EMBED_DB_PATH.get() {
            Some(path) => path.clone(),
            None => continue,
        };

        let conn = match Connection::open(&db_path) {
            Ok(conn) => conn,
            Err(_) => continue,
        };
        let _ = conn.execute_batch("PRAGMA journal_mode = WAL; PRAGMA synchronous = NORMAL;");

        let texts: Vec<&str> = jobs.iter().map(|job| job.content.as_str()).collect();
        let embeddings = crate::embedding::embed_batch(&texts);

        for (job, embedding) in jobs.iter().zip(embeddings.iter()) {
            let blob = crate::embedding::vec_to_blob(embedding);
            let hash = content_hash(&job.content);
            let _ = conn.execute(
                "UPDATE memories SET embedding = ?1, content_hash = ?2 WHERE id = ?3 AND embedding IS NULL",
                params![blob, &hash, &job.id],
            );
        }
    }
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
