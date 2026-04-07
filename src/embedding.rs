/// MemoryPilot v4.0 — Embedding Engine.
/// fastembed transformer (all-MiniLM-L6-v2, 384-dim, local ONNX inference).
use std::sync::{Mutex, OnceLock};

const VECTOR_DIM: usize = 384;

static FASTEMBED_MODEL: OnceLock<Mutex<fastembed::TextEmbedding>> = OnceLock::new();

fn get_model() -> &'static Mutex<fastembed::TextEmbedding> {
    FASTEMBED_MODEL.get_or_init(|| {
        let opts = fastembed::InitOptions::new(fastembed::EmbeddingModel::AllMiniLML6V2)
            .with_show_download_progress(false);
        let model = fastembed::TextEmbedding::try_new(opts)
            .expect("[MemoryPilot] fastembed init failed — cannot start without embedding engine");
        Mutex::new(model)
    })
}

pub fn embed_text(text: &str) -> Vec<f32> {
    let mut model = get_model().lock().expect("fastembed lock poisoned");
    let mut embeddings = model.embed(vec![text], None).expect("fastembed embed failed");
    embeddings.pop().unwrap_or_else(|| vec![0.0; VECTOR_DIM])
}

pub fn embed_batch(texts: &[&str]) -> Vec<Vec<f32>> {
    if texts.is_empty() { return vec![]; }
    let mut model = get_model().lock().expect("fastembed lock poisoned");
    model.embed(texts.to_vec(), None).expect("fastembed batch embed failed")
}

// ─── Shared Utilities ──────────────────────────────

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() { return 0.0; }
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

pub fn rrf_score(bm25_rank: usize, vector_rank: usize) -> f64 {
    let k = 60.0;
    (1.0 / (k + bm25_rank as f64)) + (1.0 / (k + vector_rank as f64))
}

pub fn vec_to_blob(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

pub fn blob_to_vec(blob: &[u8]) -> Vec<f32> {
    blob.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
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
        assert!(sim_related > sim_unrelated, "Related texts should have higher similarity");
    }

    #[test]
    fn test_blob_roundtrip() {
        let v = embed_text("test embedding roundtrip");
        let blob = vec_to_blob(&v);
        let restored = blob_to_vec(&blob);
        assert_eq!(v.len(), restored.len());
        for (a, b) in v.iter().zip(restored.iter()) {
            assert!((a - b).abs() < 1e-7);
        }
    }
}
