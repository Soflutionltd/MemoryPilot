/// MemoryPilot v4.0 — Embedding Engine.
/// fastembed transformer (multilingual-e5-small, 384-dim, local ONNX inference).
/// Supports 100+ languages including French and English natively.
///
/// Stored embeddings are int8-quantized (4-byte scale + 384 i8 = 388 bytes,
/// 4× smaller than f32). Legacy f32 blobs (1536 bytes) remain readable.
use std::sync::{Mutex, OnceLock};

const VECTOR_DIM: usize = 384;
const QUANTIZED_BLOB_LEN: usize = 4 + VECTOR_DIM;
const F32_BLOB_LEN: usize = VECTOR_DIM * 4;

static FASTEMBED_MODEL: OnceLock<Mutex<fastembed::TextEmbedding>> = OnceLock::new();

fn get_model() -> &'static Mutex<fastembed::TextEmbedding> {
    FASTEMBED_MODEL.get_or_init(|| {
        // Use a stable, writable cache dir so fastembed finds the ONNX model
        // even when launched from sandboxed environments (Claude Desktop, etc.)
        let cache_dir = std::env::var("FASTEMBED_CACHE_PATH").unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
            format!("{}/.cache/fastembed", home)
        });
        let cache_path = std::path::PathBuf::from(&cache_dir);
        std::fs::create_dir_all(&cache_path).ok();

        let opts = fastembed::InitOptions::new(fastembed::EmbeddingModel::MultilingualE5Small)
            .with_show_download_progress(false)
            .with_cache_dir(cache_path);
        let model = fastembed::TextEmbedding::try_new(opts)
            .expect("[MemoryPilot] fastembed init failed — cannot start without embedding engine");
        Mutex::new(model)
    })
}

pub fn embed_text(text: &str) -> Vec<f32> {
    let mut model = get_model().lock().expect("fastembed lock poisoned");
    let mut embeddings = model
        .embed(vec![text], None)
        .expect("fastembed embed failed");
    embeddings.pop().unwrap_or_else(|| vec![0.0; VECTOR_DIM])
}

pub fn embed_batch(texts: &[&str]) -> Vec<Vec<f32>> {
    if texts.is_empty() {
        return vec![];
    }
    let mut model = get_model().lock().expect("fastembed lock poisoned");
    model
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

/// Quantize a normalized embedding to 4-byte scale + 384 i8 bytes (388 bytes total).
/// E5 embeddings are L2-normalized so values fit in [-1, 1]; int8 keeps ~3 decimals.
pub fn quantize_to_blob(v: &[f32]) -> Vec<u8> {
    let max_abs = v.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
    let mut out = Vec::with_capacity(QUANTIZED_BLOB_LEN);
    out.extend_from_slice(&scale.to_le_bytes());
    for &x in v {
        let q = (x / scale).round().clamp(-127.0, 127.0) as i8;
        out.push(q as u8);
    }
    out
}

fn dequantize_from_blob(blob: &[u8]) -> Vec<f32> {
    if blob.len() != QUANTIZED_BLOB_LEN {
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

/// Auto-detect blob format and decode to f32. Supports both quantized int8 (388 bytes)
/// and legacy f32 (1536 bytes) layouts so existing databases keep working.
pub fn blob_to_vec(blob: &[u8]) -> Vec<f32> {
    match blob.len() {
        QUANTIZED_BLOB_LEN => dequantize_from_blob(blob),
        F32_BLOB_LEN => blob
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        _ => Vec::new(),
    }
}

/// Fast similarity directly from blob, avoiding the intermediate Vec allocation.
/// Falls back to the regular cosine path for legacy f32 blobs or unknown layouts.
pub fn similarity_with_blob(query: &[f32], blob: &[u8]) -> f32 {
    if query.len() == VECTOR_DIM && blob.len() == QUANTIZED_BLOB_LEN {
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
        assert_eq!(blob.len(), QUANTIZED_BLOB_LEN);
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
        let v = embed_text("legacy compatibility check");
        let legacy_blob: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
        assert_eq!(legacy_blob.len(), F32_BLOB_LEN);
        let restored = blob_to_vec(&legacy_blob);
        assert_eq!(v.len(), restored.len());
        for (a, b) in v.iter().zip(restored.iter()) {
            assert!((a - b).abs() < 1e-7);
        }
    }
}
