/// MemoryPilot v4.2 — Embedding Engine.
///
/// fastembed transformer, multilingual-e5-large by default (1024-dim,
/// local ONNX inference, 100+ languages). Selectable at runtime via
/// `MEMORYPILOT_EMBED_MODEL` so users can downgrade to the smaller
/// `multilingual-e5-small` (384-dim) if RAM and binary footprint are
/// the priority.
///
/// Stored embeddings are int8-quantized (4-byte scale + N i8 bytes,
/// 4× smaller than f32). The on-disk blob length is `4 + dim`. The
/// legacy 384-dim layouts (quantized 388 bytes and f32 1536 bytes)
/// remain readable so existing databases keep working — but vectors
/// produced by a different model are mathematically incompatible with
/// queries produced by another, so a `--backfill-force` is required
/// when the user changes models.
use std::sync::{Condvar, Mutex, OnceLock};

/// Legacy 384-dim layouts kept for backwards compatibility on existing
/// on-disk blobs (small model). Anything else is rejected as unknown.
const LEGACY_SMALL_DIM: usize = 384;
const LEGACY_SMALL_QUANTIZED_BLOB_LEN: usize = 4 + LEGACY_SMALL_DIM;
const LEGACY_SMALL_F32_BLOB_LEN: usize = LEGACY_SMALL_DIM * 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SelectedModel {
    E5Large,
    E5Base,
    E5Small,
    BGEM3,
}

impl SelectedModel {
    fn dim(self) -> usize {
        match self {
            SelectedModel::E5Large | SelectedModel::BGEM3 => 1024,
            SelectedModel::E5Base => 768,
            SelectedModel::E5Small => 384,
        }
    }

    fn fastembed(self) -> fastembed::EmbeddingModel {
        match self {
            SelectedModel::E5Large => fastembed::EmbeddingModel::MultilingualE5Large,
            SelectedModel::E5Base => fastembed::EmbeddingModel::MultilingualE5Base,
            SelectedModel::E5Small => fastembed::EmbeddingModel::MultilingualE5Small,
            SelectedModel::BGEM3 => fastembed::EmbeddingModel::BGEM3,
        }
    }
}

fn selected_model() -> SelectedModel {
    static CACHED: OnceLock<SelectedModel> = OnceLock::new();
    *CACHED.get_or_init(|| {
        match std::env::var("MEMORYPILOT_EMBED_MODEL")
            .ok()
            .as_deref()
            .map(str::trim)
            .map(str::to_ascii_lowercase)
            .as_deref()
        {
            Some("e5-large") | Some("multilingual-e5-large") | Some("large") => {
                SelectedModel::E5Large
            }
            Some("e5-base") | Some("multilingual-e5-base") | Some("base") => SelectedModel::E5Base,
            Some("bge-m3") | Some("bgem3") | Some("baai/bge-m3") => SelectedModel::BGEM3,
            // Default: small multilingual model. The big cousins
            // (e5-large, BGE-M3) add only +3-6 pp R@5 on
            // memorypilot-fr-v2 once the cross-encoder rerank kicks
            // in, but cost ~1.4 GB extra resident RAM, ~30 ms per
            // embedding and an order-of-magnitude longer LongMemEval
            // run. Power users can opt in via `MEMORYPILOT_EMBED_MODEL`,
            // the default stays sensible for a local-first MCP server.
            _ => SelectedModel::E5Small,
        }
    })
}

/// Active embedding dimension. Cheap to call (one atomic load).
pub fn vector_dim() -> usize {
    selected_model().dim()
}

/// Length of the int8-quantized blob written to SQLite (4-byte scale + dim i8 bytes).
pub fn quantized_blob_len() -> usize {
    4 + vector_dim()
}

/// Pool of fastembed model instances.
///
/// fastembed wraps an ONNX runtime session that is not safe to call from
/// multiple threads at once, so a single global Mutex serializes every
/// embed() call. That is a hard ceiling on concurrent throughput in MCP
/// servers that handle several clients in parallel.
///
/// `EmbedPool` keeps `MEMORYPILOT_EMBED_POOL_SIZE` (default: 4)
/// independent ONNX sessions and hands them out one at a time. A
/// lightweight Condvar wakeup avoids busy spinning.
///
/// Memory footprint depends on the selected model. multilingual-e5-large
/// (default, 1024-dim) is ~1.4 GB per session resident once the arena
/// has stabilised; the smaller `multilingual-e5-small` (384-dim) is
/// closer to ~700 MB. The pool default of 4 keeps us under ~6 GB
/// resident with the large model and ~3.5 GB with the small model,
/// which is the right operating point for a local-first server.
/// Users who care about throughput more than RAM can opt into a
/// bigger pool via `MEMORYPILOT_EMBED_POOL_SIZE` (capped at 8 to
/// keep memory predictable).
struct EmbedPool {
    available: Mutex<Vec<fastembed::TextEmbedding>>,
    notify: Condvar,
}

static EMBED_POOL: OnceLock<EmbedPool> = OnceLock::new();

fn fastembed_cache_dir() -> std::path::PathBuf {
    let cache_dir = std::env::var("FASTEMBED_CACHE_PATH").unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
        format!("{}/.cache/fastembed", home)
    });
    let path = std::path::PathBuf::from(&cache_dir);
    std::fs::create_dir_all(&path).ok();
    path
}

fn build_model() -> fastembed::TextEmbedding {
    let model = selected_model();
    let opts = fastembed::InitOptions::new(model.fastembed())
        .with_show_download_progress(true)
        .with_cache_dir(fastembed_cache_dir());
    fastembed::TextEmbedding::try_new(opts)
        .unwrap_or_else(|error| {
            panic!(
                "[MemoryPilot] fastembed init failed for model {:?}: {} — \
                 cannot start without embedding engine",
                model, error
            )
        })
}

fn embed_pool() -> &'static EmbedPool {
    EMBED_POOL.get_or_init(|| {
        // Default of 4: matches the comment, mirrors the read pool
        // sizing that empirically saturates SQLite WAL on a 4-core
        // worker, and keeps RAM under ~3.5 GB on the concurrency
        // bench. Going to 8 only gives marginal extra throughput
        // (the cross-encoder mutex becomes the next bottleneck).
        let pool_size = std::env::var("MEMORYPILOT_EMBED_POOL_SIZE")
            .ok()
            .and_then(|raw| raw.parse::<usize>().ok())
            .unwrap_or(4)
            .clamp(1, 8);

        let mut models = Vec::with_capacity(pool_size);
        for _ in 0..pool_size {
            models.push(build_model());
        }
        EmbedPool {
            available: Mutex::new(models),
            notify: Condvar::new(),
        }
    })
}

struct PooledModel {
    inner: Option<fastembed::TextEmbedding>,
}

impl PooledModel {
    fn new() -> Self {
        let pool = embed_pool();
        let mut guard = pool.available.lock().expect("embed pool poisoned");
        while guard.is_empty() {
            guard = pool.notify.wait(guard).expect("embed pool wait poisoned");
        }
        let model = guard.pop().expect("pool was non-empty under guard");
        PooledModel { inner: Some(model) }
    }

    fn model(&mut self) -> &mut fastembed::TextEmbedding {
        self.inner.as_mut().expect("pooled model dropped")
    }
}

impl Drop for PooledModel {
    fn drop(&mut self) {
        if let Some(model) = self.inner.take() {
            let pool = embed_pool();
            if let Ok(mut guard) = pool.available.lock() {
                guard.push(model);
                pool.notify.notify_one();
            }
        }
    }
}

pub fn embed_text(text: &str) -> Vec<f32> {
    let mut pooled = PooledModel::new();
    let mut embeddings = pooled
        .model()
        .embed(vec![text], None)
        .expect("fastembed embed failed");
    embeddings.pop().unwrap_or_else(|| vec![0.0; vector_dim()])
}

pub fn embed_batch(texts: &[&str]) -> Vec<Vec<f32>> {
    if texts.is_empty() {
        return vec![];
    }
    let mut pooled = PooledModel::new();
    pooled
        .model()
        .embed(texts.to_vec(), None)
        .expect("fastembed batch embed failed")
}

// ─── Shared Utilities ──────────────────────────────

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

pub fn rrf_score(bm25_rank: usize, vector_rank: usize) -> f64 {
    let k = 40.0;
    (1.0 / (k + bm25_rank as f64)) + (1.0 / (k + vector_rank as f64))
}

/// Quantize a normalized embedding to 4-byte scale + N i8 bytes
/// (4 + dim bytes total). E5 embeddings are L2-normalized so values
/// fit in [-1, 1]; int8 keeps ~3 decimals.
pub fn quantize_to_blob(v: &[f32]) -> Vec<u8> {
    let max_abs = v.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
    let mut out = Vec::with_capacity(4 + v.len());
    out.extend_from_slice(&scale.to_le_bytes());
    for &x in v {
        let q = (x / scale).round().clamp(-127.0, 127.0) as i8;
        out.push(q as u8);
    }
    out
}

fn dequantize_from_blob(blob: &[u8], dim: usize) -> Vec<f32> {
    if blob.len() != 4 + dim {
        return Vec::new();
    }
    let scale = f32::from_le_bytes([blob[0], blob[1], blob[2], blob[3]]);
    blob[4..]
        .iter()
        .map(|&b| (b as i8) as f32 * scale)
        .collect()
}

/// Default codec: write quantized int8 blob.
pub fn vec_to_blob(v: &[f32]) -> Vec<u8> {
    quantize_to_blob(v)
}

/// Auto-detect blob format and decode to f32. Supports the active
/// model's int8 layout (4 + dim bytes) plus the legacy small-model
/// formats (388 bytes int8, 1536 bytes f32) so older databases keep
/// returning vectors during the one-shot re-embed pass triggered by
/// `--backfill-force` after a model swap. A vector returned from a
/// legacy layout is dimensionally incompatible with the active query
/// vector — `similarity_with_blob` and `cosine_similarity` already
/// short-circuit on length mismatch, so legacy rows transparently
/// score 0 until the backfill rewrites them.
pub fn blob_to_vec(blob: &[u8]) -> Vec<f32> {
    let active = quantized_blob_len();
    match blob.len() {
        len if len == active => dequantize_from_blob(blob, vector_dim()),
        LEGACY_SMALL_QUANTIZED_BLOB_LEN if active != LEGACY_SMALL_QUANTIZED_BLOB_LEN => {
            dequantize_from_blob(blob, LEGACY_SMALL_DIM)
        }
        LEGACY_SMALL_F32_BLOB_LEN => blob
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        _ => Vec::new(),
    }
}

/// Fast similarity directly from blob, avoiding the intermediate Vec allocation.
/// Falls back to the regular cosine path for legacy f32 blobs or unknown layouts.
pub fn similarity_with_blob(query: &[f32], blob: &[u8]) -> f32 {
    let active_dim = vector_dim();
    let active_blob_len = quantized_blob_len();
    if query.len() == active_dim && blob.len() == active_blob_len {
        let scale = f32::from_le_bytes([blob[0], blob[1], blob[2], blob[3]]);
        let mut sum = 0.0f32;
        for (q, &b) in query.iter().zip(blob[4..].iter()) {
            sum += q * (b as i8) as f32;
        }
        sum * scale
    } else {
        let stored = blob_to_vec(blob);
        cosine_similarity(query, &stored)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_similar_texts() {
        let v1 = embed_text("authentication login Supabase auth JWT");
        let v2 = embed_text("user login authentication with JWT tokens");
        let v3 = embed_text("CSS grid layout flexbox styling");
        let sim_related = cosine_similarity(&v1, &v2);
        let sim_unrelated = cosine_similarity(&v1, &v3);
        assert!(
            sim_related > sim_unrelated,
            "Related texts should have higher similarity"
        );
    }

    #[test]
    fn test_blob_roundtrip() {
        let v = embed_text("test embedding roundtrip");
        let blob = vec_to_blob(&v);
        assert_eq!(blob.len(), quantized_blob_len());
        let restored = blob_to_vec(&blob);
        assert_eq!(v.len(), restored.len());
        let mut max_err = 0.0f32;
        for (a, b) in v.iter().zip(restored.iter()) {
            max_err = max_err.max((a - b).abs());
        }
        assert!(
            max_err < 0.02,
            "int8 quantization error too high: {}",
            max_err
        );
    }

    #[test]
    fn test_quantization_preserves_ranking() {
        let q = embed_text("authentication login Supabase auth JWT");
        let related = embed_text("user login authentication with JWT tokens");
        let unrelated = embed_text("CSS grid layout flexbox styling");

        let related_blob = vec_to_blob(&related);
        let unrelated_blob = vec_to_blob(&unrelated);

        let sim_related = similarity_with_blob(&q, &related_blob);
        let sim_unrelated = similarity_with_blob(&q, &unrelated_blob);

        assert!(
            sim_related > sim_unrelated,
            "Quantized similarity must preserve relative ranking"
        );
    }

    #[test]
    fn test_legacy_f32_blob_still_readable() {
        // Legacy f32 layout was small-model only (1536 bytes = 384 × 4).
        // The decoder must keep returning a 384-dim vector regardless
        // of the active model so older databases survive an upgrade
        // until `--backfill-force` rewrites them.
        let synthetic: Vec<f32> = (0..LEGACY_SMALL_DIM)
            .map(|i| (i as f32 / LEGACY_SMALL_DIM as f32) - 0.5)
            .collect();
        let legacy_blob: Vec<u8> = synthetic.iter().flat_map(|f| f.to_le_bytes()).collect();
        assert_eq!(legacy_blob.len(), LEGACY_SMALL_F32_BLOB_LEN);
        let restored = blob_to_vec(&legacy_blob);
        assert_eq!(restored.len(), LEGACY_SMALL_DIM);
        for (a, b) in synthetic.iter().zip(restored.iter()) {
            assert!((a - b).abs() < 1e-7);
        }
    }
}
