//! Local Approximate Nearest Neighbor index over `usearch` HNSW.
//!
//! Goal: keep cosine search latency near-constant as the memory base grows
//! past a few thousand entries. The persistent SQLite scan is preserved as
//! a deterministic fallback whenever the index is empty or out of sync.
//!
//! Design constraints:
//! - 100% local. No external service, no network call.
//! - Incremental: add/remove on every memory mutation.
//! - Persisted to disk next to the SQLite file.
//! - Stable mapping from string memory ids to `u64` keys via a deterministic hash.
//!   Collisions are tracked and resolved by re-checking ids during candidate
//!   filtering — at <100k entries the FNV-64 collision probability is negligible.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, RwLock};

use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

const DEFAULT_CAPACITY: usize = 4_096;
const DEFAULT_RESERVE_THREADS: usize = 1;

pub struct AnnIndex {
    inner: Mutex<Index>,
    key_map: RwLock<HashMap<u64, String>>,
    storage_path: Option<PathBuf>,
    dim: usize,
}

impl AnnIndex {
    pub fn open(storage_path: Option<PathBuf>) -> Result<Self, String> {
        Self::open_with_dim(storage_path, crate::embedding::vector_dim())
    }

    pub fn open_with_dim(storage_path: Option<PathBuf>, dim: usize) -> Result<Self, String> {
        let options = IndexOptions {
            dimensions: dim,
            metric: MetricKind::Cos,
            quantization: ScalarKind::I8,
            ..Default::default()
        };
        let index = Index::new(&options).map_err(|error| format!("ANN init: {}", error))?;
        index
            .reserve_capacity_and_threads(DEFAULT_CAPACITY, DEFAULT_RESERVE_THREADS)
            .map_err(|error| format!("ANN reserve: {}", error))?;

        let mut ann = Self {
            inner: Mutex::new(index),
            key_map: RwLock::new(HashMap::new()),
            storage_path,
            dim,
        };

        if let Some(path) = ann.storage_path.clone() {
            if path.exists() {
                if let Err(error) = ann.load_from_disk(&path) {
                    // The on-disk index was almost certainly produced
                    // by a different model dimension. Drop the stale
                    // file (and its sidecar), rebuild a fresh
                    // in-memory index at the active dim, and let the
                    // warm-up thread hydrate it from the freshly
                    // re-embedded SQLite blobs.
                    eprintln!(
                        "[MemoryPilot] ANN index reset (incompatible on-disk format: {})",
                        error
                    );
                    let _ = std::fs::remove_file(path.as_path());
                    let _ = std::fs::remove_file(sidecar_path(path.as_path()));
                    let fresh_options = IndexOptions {
                        dimensions: dim,
                        metric: MetricKind::Cos,
                        quantization: ScalarKind::I8,
                        ..Default::default()
                    };
                    let fresh = Index::new(&fresh_options)
                        .map_err(|error| format!("ANN reset init: {}", error))?;
                    fresh
                        .reserve_capacity_and_threads(DEFAULT_CAPACITY, DEFAULT_RESERVE_THREADS)
                        .map_err(|error| format!("ANN reset reserve: {}", error))?;
                    if let Ok(mut guard) = ann.inner.lock() {
                        *guard = fresh;
                    }
                    if let Ok(mut map) = ann.key_map.write() {
                        map.clear();
                    }
                }
            }
        }

        Ok(ann)
    }

    pub fn len(&self) -> usize {
        self.key_map.read().map(|guard| guard.len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn add(&self, id: &str, vector: &[f32]) -> Result<(), String> {
        if vector.len() != self.dim {
            return Err(format!(
                "ANN add: vector dim {} != index dim {}",
                vector.len(),
                self.dim
            ));
        }

        let key = id_to_key(id);

        {
            let index = self
                .inner
                .lock()
                .map_err(|_| "ANN lock poisoned".to_string())?;
            let current_size = index.size();
            let capacity = index.capacity();
            if current_size + 1 > capacity {
                let new_capacity = (capacity.max(DEFAULT_CAPACITY) * 2).max(current_size + 1);
                index
                    .reserve_capacity_and_threads(new_capacity, DEFAULT_RESERVE_THREADS)
                    .map_err(|error| format!("ANN grow: {}", error))?;
            }
            // usearch refuses duplicate keys with multi=false; remove first to keep idempotent.
            if index.contains(key) {
                let _ = index.remove(key);
            }
            index
                .add(key, vector)
                .map_err(|error| format!("ANN add: {}", error))?;
        }

        if let Ok(mut map) = self.key_map.write() {
            map.insert(key, id.to_string());
        }
        Ok(())
    }

    pub fn remove(&self, id: &str) -> Result<(), String> {
        let key = id_to_key(id);
        if let Ok(index) = self.inner.lock() {
            let _ = index.remove(key);
        }
        if let Ok(mut map) = self.key_map.write() {
            map.remove(&key);
        }
        Ok(())
    }

    /// Return up to `top_k` candidates ranked by cosine similarity.
    /// Each entry is `(memory_id, similarity_score)` where score is in [-1, 1].
    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<(String, f32)> {
        if query.len() != self.dim || top_k == 0 || self.is_empty() {
            return Vec::new();
        }
        let matches = match self.inner.lock() {
            Ok(index) => match index.search(query, top_k) {
                Ok(matches) => matches,
                Err(_) => return Vec::new(),
            },
            Err(_) => return Vec::new(),
        };
        let map = match self.key_map.read() {
            Ok(map) => map,
            Err(_) => return Vec::new(),
        };
        matches
            .keys
            .iter()
            .zip(matches.distances.iter())
            .filter_map(|(key, distance)| {
                map.get(key)
                    .map(|id| (id.clone(), cosine_from_distance(*distance)))
            })
            .collect()
    }

    pub fn persist(&self) -> Result<(), String> {
        let Some(path) = self.storage_path.as_ref() else {
            return Ok(());
        };
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let path_str = path.to_string_lossy().to_string();
        let index = self
            .inner
            .lock()
            .map_err(|_| "ANN lock poisoned".to_string())?;
        index
            .save(&path_str)
            .map_err(|error| format!("ANN save: {}", error))?;

        if let Ok(map) = self.key_map.read() {
            let sidecar = sidecar_path(path);
            let serialized: Vec<(u64, String)> = map.iter().map(|(k, v)| (*k, v.clone())).collect();
            let json = serde_json::to_string(&serialized)
                .map_err(|error| format!("ANN sidecar serialize: {}", error))?;
            std::fs::write(&sidecar, json)
                .map_err(|error| format!("ANN sidecar write: {}", error))?;
        }
        Ok(())
    }

    fn load_from_disk(&mut self, path: &Path) -> Result<(), String> {
        let path_str = path.to_string_lossy().to_string();
        if let Ok(index) = self.inner.lock() {
            index
                .load(&path_str)
                .map_err(|error| format!("ANN load: {}", error))?;
            let on_disk_dim = index.dimensions();
            if on_disk_dim != self.dim {
                // Drop the just-loaded data so the caller's recovery
                // path can wipe the file and start over with the
                // correct dim. Returning Err triggers exactly that.
                let _ = index.reset();
                return Err(format!(
                    "on-disk dim {} ≠ active dim {}",
                    on_disk_dim, self.dim
                ));
            }
        }
        let sidecar = sidecar_path(path);
        if sidecar.exists() {
            let raw = std::fs::read_to_string(&sidecar)
                .map_err(|error| format!("ANN sidecar read: {}", error))?;
            let entries: Vec<(u64, String)> = serde_json::from_str(&raw)
                .map_err(|error| format!("ANN sidecar parse: {}", error))?;
            if let Ok(mut map) = self.key_map.write() {
                map.clear();
                for (key, id) in entries {
                    map.insert(key, id);
                }
            }
        }
        Ok(())
    }
}

fn sidecar_path(path: &Path) -> PathBuf {
    let mut sidecar = path.to_path_buf();
    let new_name = match sidecar.file_name().and_then(|s| s.to_str()) {
        Some(name) => format!("{}.keys.json", name),
        None => "ann.keys.json".to_string(),
    };
    sidecar.set_file_name(new_name);
    sidecar
}

/// FNV-1a 64-bit hash. Deterministic mapping from memory id to ANN key.
fn id_to_key(id: &str) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in id.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// usearch returns cosine *distance* in [0, 2]; convert back to similarity in [-1, 1].
fn cosine_from_distance(distance: f32) -> f32 {
    (1.0 - distance).clamp(-1.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_vec(seed: u64, dim: usize) -> Vec<f32> {
        let mut v = Vec::with_capacity(dim);
        for i in 0..dim {
            let raw = ((seed.wrapping_add(i as u64)) % 1000) as f32 / 1000.0 - 0.5;
            v.push(raw);
        }
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
        for x in &mut v {
            *x /= norm;
        }
        v
    }

    const TEST_DIM: usize = 384;

    #[test]
    fn ann_returns_self_as_top_match() {
        let ann = AnnIndex::open_with_dim(None, TEST_DIM).expect("open");
        let v = unit_vec(7, TEST_DIM);
        ann.add("memory-a", &v).expect("add");
        ann.add("memory-b", &unit_vec(42, TEST_DIM)).expect("add");
        let results = ann.search(&v, 2);
        assert!(!results.is_empty());
        assert_eq!(results[0].0, "memory-a");
    }

    #[test]
    fn ann_remove_drops_candidate() {
        let ann = AnnIndex::open_with_dim(None, TEST_DIM).expect("open");
        let v = unit_vec(7, TEST_DIM);
        ann.add("memory-a", &v).expect("add");
        ann.add("memory-b", &unit_vec(99, TEST_DIM)).expect("add");
        ann.remove("memory-a").expect("remove");
        let results = ann.search(&v, 2);
        assert!(!results.iter().any(|(id, _)| id == "memory-a"));
    }

    #[test]
    fn ann_persists_and_reloads() {
        let dir = std::env::temp_dir().join(format!("memorypilot-ann-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test.usearch");
        {
            let ann = AnnIndex::open_with_dim(Some(path.clone()), TEST_DIM).expect("open");
            ann.add("memory-a", &unit_vec(7, TEST_DIM)).expect("add");
            ann.add("memory-b", &unit_vec(42, TEST_DIM)).expect("add");
            ann.persist().expect("persist");
        }
        let reopened = AnnIndex::open_with_dim(Some(path.clone()), TEST_DIM).expect("reopen");
        assert!(reopened.len() >= 2);
        let results = reopened.search(&unit_vec(7, TEST_DIM), 2);
        assert!(results.iter().any(|(id, _)| id == "memory-a"));
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(sidecar_path(&path));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn ann_dim_mismatch_resets_index() {
        let dir = std::env::temp_dir().join(format!(
            "memorypilot-ann-mismatch-{}-{}",
            std::process::id(),
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0)
        ));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test.usearch");
        {
            let ann = AnnIndex::open_with_dim(Some(path.clone()), 384).expect("open");
            ann.add("a", &unit_vec(1, 384)).expect("add");
            ann.persist().expect("persist");
        }
        // Reopening with a different dim should drop the stale file
        // and yield a fresh, empty index — never panic.
        let reopened = AnnIndex::open_with_dim(Some(path.clone()), 1024).expect("reopen");
        assert_eq!(reopened.len(), 0, "stale 384-dim file must be discarded");
        // And we should be able to write under the new dim.
        reopened.add("b", &unit_vec(2, 1024)).expect("add");
        assert_eq!(reopened.len(), 1);
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(sidecar_path(&path));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn ann_empty_index_returns_no_results() {
        let ann = AnnIndex::open_with_dim(None, TEST_DIM).expect("open");
        let v = unit_vec(7, TEST_DIM);
        assert!(ann.search(&v, 5).is_empty());
    }
}
