/// MemoryPilot v4.4 — Embedding Engine.
///
/// Default model: **jina-embeddings-v5-text-nano-retrieval** (768-dim,
/// int8 ONNX, ~250 MB on disk, 100+ languages, 8k context, last-token
/// pooling done *inside* the exported graph via its `sentence_embedding`
/// output). It is the strongest sub-300M multilingual retrieval encoder
/// available today and beats multilingual-e5-small by a wide margin on
/// MMTEB — especially on French queries, which is where MemoryPilot's
/// users live. Licence: CC-BY-NC-4.0; the model is downloaded straight
/// from Jina's Hugging Face repository at first run, MemoryPilot does not
/// redistribute it.
///
/// The jina model is driven by `ort` directly (not through fastembed):
/// fastembed only knows CLS/mean pooling and loads "bring your own"
/// graphs from memory, which would duplicate the 250 MB weights while
/// the session hydrates. Talking to ONNX Runtime ourselves lets us load
/// from file (external-data friendly), pick the pre-pooled output and
/// cap the intra-op threads so the server stays a good citizen.
///
/// The legacy fastembed models remain selectable via
/// `MEMORYPILOT_EMBED_MODEL` (`e5-small`, `e5-base`, `e5-large`,
/// `bge-m3`) for users who need an Apache-2.0 encoder; `jina-q4`
/// selects the 4-bit jina variant (~140 MB) for constrained machines.
///
/// Stored embeddings are int8-quantized (4-byte scale + N i8 bytes,
/// 4× smaller than f32). The on-disk blob length is `4 + dim`. The
/// legacy 384-dim layouts (quantized 388 bytes and f32 1536 bytes)
/// remain readable so existing databases keep working — but vectors
/// produced by one model are mathematically incompatible with queries
/// produced by another, so the database records the active model label
/// and re-embeds everything when it changes (see `Database::open_at`).
use std::path::PathBuf;
use std::sync::OnceLock;

/// Legacy 384-dim layouts kept for backwards compatibility on existing
/// on-disk blobs (small model). Anything else is rejected as unknown.
const LEGACY_SMALL_DIM: usize = 384;
const LEGACY_SMALL_QUANTIZED_BLOB_LEN: usize = 4 + LEGACY_SMALL_DIM;
const LEGACY_SMALL_F32_BLOB_LEN: usize = LEGACY_SMALL_DIM * 4;

const JINA_REPO: &str = "jinaai/jina-embeddings-v5-text-nano-retrieval";
const JINA_QUERY_PREFIX: &str = "Query: ";
const JINA_DOCUMENT_PREFIX: &str = "Document: ";
/// Memories are short (a few hundred tokens at most); capping the
/// sequence keeps the attention buffers small and predictable instead of
/// letting one 8k-token outlier balloon the arena.
const JINA_MAX_TOKENS: usize = 512;
/// Padded tokens per ONNX run (batch × longest sequence). Texts are
/// sorted by length and packed under this budget, so a run of short
/// memories holds a handful of texts while a 512-token outlier goes
/// alone: the activation footprint is the same either way. Measured on
/// the int8 graph with arena shrinkage on: 512 → ~200 MB of
/// activations at peak, 1024 → ~300 MB, 4096 → ~1.5 GB, for the *same*
/// throughput — CPU inference gains nothing from wider batches, so the
/// budget is kept small on purpose.
const JINA_TOKEN_BUDGET: usize = 512;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SelectedModel {
    JinaV5Nano,
    JinaV5NanoQ4,
    E5Large,
    E5Base,
    E5Small,
    BGEM3,
}

impl SelectedModel {
    fn dim(self) -> usize {
        match self {
            SelectedModel::E5Large | SelectedModel::BGEM3 => 1024,
            SelectedModel::JinaV5Nano | SelectedModel::JinaV5NanoQ4 | SelectedModel::E5Base => {
                768
            }
            SelectedModel::E5Small => 384,
        }
    }

    fn is_jina(self) -> bool {
        matches!(self, SelectedModel::JinaV5Nano | SelectedModel::JinaV5NanoQ4)
    }

    fn fastembed(self) -> fastembed::EmbeddingModel {
        match self {
            SelectedModel::E5Large => fastembed::EmbeddingModel::MultilingualE5Large,
            SelectedModel::E5Base => fastembed::EmbeddingModel::MultilingualE5Base,
            SelectedModel::E5Small => fastembed::EmbeddingModel::MultilingualE5Small,
            SelectedModel::BGEM3 => fastembed::EmbeddingModel::BGEM3,
            SelectedModel::JinaV5Nano | SelectedModel::JinaV5NanoQ4 => {
                unreachable!("jina models are served by the ort engine")
            }
        }
    }

    /// ONNX graph file inside the jina repository. Both variants ship
    /// their weights as an external `.onnx_data` sibling, resolved by
    /// ONNX Runtime relative to the graph path.
    fn jina_onnx_file(self) -> &'static str {
        match self {
            SelectedModel::JinaV5Nano => "onnx/model_quantized.onnx",
            SelectedModel::JinaV5NanoQ4 => "onnx/model_q4.onnx",
            _ => unreachable!("not a jina model"),
        }
    }

    /// Human-readable label "name (dim-dim)" for logs and the HTTP /info
    /// endpoint. Reflects the *active* model so diagnostics never lie.
    /// Also persisted in the database as the embedding fingerprint, so
    /// changing it for an existing variant triggers a full re-embed.
    fn label(self) -> &'static str {
        match self {
            SelectedModel::JinaV5Nano => "jina-embeddings-v5-text-nano-retrieval int8, 768-dim",
            SelectedModel::JinaV5NanoQ4 => "jina-embeddings-v5-text-nano-retrieval q4, 768-dim",
            SelectedModel::E5Large => "multilingual-e5-large, 1024-dim",
            SelectedModel::E5Base => "multilingual-e5-base, 768-dim",
            SelectedModel::E5Small => "multilingual-e5-small, 384-dim",
            SelectedModel::BGEM3 => "BGE-M3, 1024-dim",
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
            Some("e5-small") | Some("multilingual-e5-small") | Some("small") => {
                SelectedModel::E5Small
            }
            Some("bge-m3") | Some("bgem3") | Some("baai/bge-m3") => SelectedModel::BGEM3,
            Some("jina-q4") | Some("jina-nano-q4") | Some("light") | Some("lite") => {
                SelectedModel::JinaV5NanoQ4
            }
            // Default: jina-embeddings-v5-text-nano-retrieval, int8.
            // Best-in-class multilingual recall for its size, ~250 MB
            // download, one ONNX session ≈ 400 MB resident. `quality`
            // and `max` used to select BGE-M3 (2 GB download, 1.4 GB
            // resident); jina-nano now outranks it on MMTEB retrieval at
            // a fraction of the footprint, so they map here too.
            _ => SelectedModel::JinaV5Nano,
        }
    })
}

/// Active model label, e.g. "jina-embeddings-v5-text-nano-retrieval
/// int8, 768-dim". Used by logs, the HTTP info endpoint and as the
/// on-disk embedding fingerprint.
pub fn model_label() -> &'static str {
    selected_model().label()
}

/// Active embedding dimension. Cheap to call (one atomic load).
pub fn vector_dim() -> usize {
    selected_model().dim()
}

/// Length of the int8-quantized blob written to SQLite (4-byte scale + dim i8 bytes).
pub fn quantized_blob_len() -> usize {
    4 + vector_dim()
}

/// Intra-op threads handed to ONNX Runtime. Defaults to
/// `min(4, available cores)`: embedding a memory is latency-bound on a
/// handful of matmuls, and a background MCP server has no business
/// pinning every core of a laptop. `MEMORYPILOT_EMBED_THREADS` overrides.
fn intra_threads() -> usize {
    let available = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    std::env::var("MEMORYPILOT_EMBED_THREADS")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .unwrap_or_else(|| available.min(4))
        .clamp(1, available.max(1))
}

pub(crate) fn fastembed_cache_dir() -> PathBuf {
    let cache_dir = std::env::var("FASTEMBED_CACHE_PATH").unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
        format!("{}/.cache/fastembed", home)
    });
    let path = PathBuf::from(&cache_dir);
    std::fs::create_dir_all(&path).ok();
    path
}

/// Fetch one file of a Hugging Face model repository into the fastembed
/// cache (same layout fastembed uses, so a user's cache is shared). Files
/// of one repo land in one snapshot directory, which is what lets ONNX
/// Runtime resolve the `.onnx_data` external weights next to the graph.
pub(crate) fn hf_file(repo: &str, file: &str, show_progress: bool) -> Result<PathBuf, String> {
    let api = hf_hub::api::sync::ApiBuilder::new()
        .with_cache_dir(fastembed_cache_dir())
        .with_progress(show_progress)
        .build()
        .map_err(|error| format!("Hugging Face API init: {}", error))?;
    api.model(repo.to_string())
        .get(file)
        .map_err(|error| format!("download {}/{}: {}", repo, file, error))
}

// ─── ONNX Runtime engine (jina) ─────────────────────

struct OrtEmbedder {
    session: ort::session::Session,
    tokenizer: crate::tokenizer::ByteLevelBpe,
    run_options: ort::session::RunOptions,
    dim: usize,
}

impl OrtEmbedder {
    fn load(model: SelectedModel) -> Result<Self, String> {
        let onnx = model.jina_onnx_file();
        let graph_path = hf_file(JINA_REPO, onnx, true)?;
        // External weights: fetched for their side effect (present next
        // to the graph); ONNX Runtime opens them by relative path.
        hf_file(JINA_REPO, &format!("{}_data", onnx), true)?;
        let tokenizer_path = hf_file(JINA_REPO, "tokenizer.json", true)?;
        let tokenizer = crate::tokenizer::ByteLevelBpe::from_file(&tokenizer_path)
            .map_err(|error| format!("tokenizer load: {}", error))?;

        use ort::session::builder::GraphOptimizationLevel;
        let session = ort::session::Session::builder()
            .map_err(|error| format!("ort builder: {}", error))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|error| format!("ort optimization level: {}", error))?
            .with_intra_threads(intra_threads())
            .map_err(|error| format!("ort intra threads: {}", error))?
            // Memory patterns pre-plan one big arena block per input
            // shape; with variable-length text every new shape adds a
            // block that is never released. Per-op allocation + arena
            // shrinkage (below) keeps the footprint flat.
            .with_memory_pattern(false)
            .map_err(|error| format!("ort memory pattern: {}", error))?
            // Let the device allocator own the initializers instead of
            // copying them into the arena: one copy of the weights in
            // memory, not two, while the session hydrates.
            .with_config_entry("session.use_device_allocator_for_initializers", "1")
            .map_err(|error| format!("ort config entry: {}", error))?
            .commit_from_file(&graph_path)
            .map_err(|error| format!("ort session load {}: {}", graph_path.display(), error))?;

        // Give the arena back after each run: a 512-token chunk grows
        // it by a few hundred MB that would otherwise stay resident for
        // the life of the process even though 99% of runs are tiny.
        let mut run_options = ort::session::RunOptions::new()
            .map_err(|error| format!("ort run options: {}", error))?;
        run_options
            .set("memory.enable_memory_arena_shrinkage", "cpu:0")
            .map_err(|error| format!("ort arena shrinkage: {}", error))?;

        Ok(Self {
            session,
            tokenizer,
            run_options,
            dim: model.dim(),
        })
    }

    /// Tokenize everything once, sort by length, pack under the token
    /// budget, run each pack, and scatter the vectors back in input
    /// order.
    fn embed(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        let encodings: Vec<Vec<u32>> = texts
            .iter()
            .map(|text| self.tokenizer.encode(text, JINA_MAX_TOKENS))
            .collect();

        let mut order: Vec<usize> = (0..encodings.len()).collect();
        order.sort_by_key(|&index| std::cmp::Reverse(encodings[index].len()));

        let mut out: Vec<Vec<f32>> = vec![Vec::new(); texts.len()];
        let mut start = 0;
        while start < order.len() {
            // The first item of a pack is the longest, so it fixes the
            // padded width for the whole pack.
            let width = encodings[order[start]].len().max(1);
            let rows = (JINA_TOKEN_BUDGET / width).max(1);
            let end = (start + rows).min(order.len());
            let pack = &order[start..end];
            let vectors = self.run_pack(&encodings, pack, width)?;
            for (&index, vector) in pack.iter().zip(vectors) {
                out[index] = vector;
            }
            start = end;
        }
        Ok(out)
    }

    fn run_pack(
        &mut self,
        encodings: &[Vec<u32>],
        pack: &[usize],
        width: usize,
    ) -> Result<Vec<Vec<f32>>, String> {
        let batch = pack.len();
        let pad_id = self.tokenizer.pad_id() as i64;
        let mut ids = Vec::with_capacity(batch * width);
        let mut mask = Vec::with_capacity(batch * width);
        for &index in pack {
            let encoding = &encodings[index];
            let len = encoding.len().min(width);
            ids.extend(encoding[..len].iter().map(|&id| id as i64));
            ids.extend(std::iter::repeat(pad_id).take(width - len));
            mask.extend(std::iter::repeat(1i64).take(len));
            mask.extend(std::iter::repeat(0i64).take(width - len));
        }
        let shape = [batch, width];
        let ids = ort::value::Tensor::from_array((shape, ids))
            .map_err(|error| format!("input_ids tensor: {}", error))?;
        let mask = ort::value::Tensor::from_array((shape, mask))
            .map_err(|error| format!("attention_mask tensor: {}", error))?;
        let outputs = self
            .session
            .run_with_options(
                ort::inputs!["input_ids" => ids, "attention_mask" => mask],
                &self.run_options,
            )
            .map_err(|error| format!("ort run: {}", error))?;
        // `sentence_embedding` is the graph's own last-token pooling,
        // already L2-normalised (verified against a manual gather on
        // `last_hidden_state`: max abs diff 1.5e-8).
        let (out_shape, data) = outputs["sentence_embedding"]
            .try_extract_tensor::<f32>()
            .map_err(|error| format!("sentence_embedding extract: {}", error))?;
        let dims: Vec<i64> = out_shape.iter().copied().collect();
        if dims.len() != 2 || dims[0] as usize != batch || dims[1] as usize != self.dim {
            return Err(format!(
                "sentence_embedding shape {:?}, expected [{}, {}]",
                dims, batch, self.dim
            ));
        }
        Ok(data
            .chunks_exact(self.dim)
            .map(|row| {
                let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 && (norm - 1.0).abs() > 1e-3 {
                    row.iter().map(|x| x / norm).collect()
                } else {
                    row.to_vec()
                }
            })
            .collect())
    }
}

// ─── Engine pool ────────────────────────────────────

enum Engine {
    Fastembed(fastembed::TextEmbedding),
    Ort(OrtEmbedder),
}

impl Engine {
    fn embed(&mut self, texts: Vec<String>) -> Vec<Vec<f32>> {
        match self {
            Engine::Fastembed(model) => model
                .embed(texts, None)
                .expect("fastembed embed failed"),
            Engine::Ort(model) => model
                .embed(&texts)
                .unwrap_or_else(|error| panic!("[MemoryPilot] jina embed failed: {}", error)),
        }
    }
}

/// Pool of embedding engines.
///
/// An ONNX runtime session is not safe to call from multiple threads at
/// once, so each session is handed out to one caller at a time. The pool
/// keeps at most `MEMORYPILOT_EMBED_POOL_SIZE` independent sessions,
/// builds them on first use and drops them again after
/// `MEMORYPILOT_MODEL_IDLE_SECS` without a caller (see `crate::pool`).
///
/// Default pool size is **1**: one jina session is ~250 MB resident and
/// embedding is a short burst, so a second session buys throughput only
/// under heavy concurrent writes — users who want that opt in (capped
/// at 8 to keep memory predictable). The legacy fastembed models keep
/// their historical defaults (4, or 2 for the 1024-dim models).
static EMBED_POOL: OnceLock<crate::pool::IdlePool<Engine>> = OnceLock::new();

fn build_engine() -> Engine {
    let model = selected_model();
    if model.is_jina() {
        return Engine::Ort(OrtEmbedder::load(model).unwrap_or_else(|error| {
            panic!(
                "[MemoryPilot] jina embedding init failed ({}): {} — cannot start without embedding engine",
                model.label(),
                error
            )
        }));
    }
    let opts = fastembed::TextInitOptions::new(model.fastembed())
        .with_show_download_progress(true)
        .with_cache_dir(fastembed_cache_dir())
        .with_intra_threads(intra_threads());
    Engine::Fastembed(fastembed::TextEmbedding::try_new(opts).unwrap_or_else(|error| {
        panic!(
            "[MemoryPilot] fastembed init failed for model {:?}: {} — \
             cannot start without embedding engine",
            model, error
        )
    }))
}

fn embed_pool() -> &'static crate::pool::IdlePool<Engine> {
    let pool = EMBED_POOL.get_or_init(|| {
        let model = selected_model();
        let default_pool = if model.is_jina() {
            1
        } else if vector_dim() >= 1024 {
            2
        } else {
            4
        };
        let pool_size = std::env::var("MEMORYPILOT_EMBED_POOL_SIZE")
            .ok()
            .and_then(|raw| raw.parse::<usize>().ok())
            .unwrap_or(default_pool)
            .clamp(1, 8);
        crate::pool::IdlePool::new("embedder", pool_size, build_engine)
    });
    static EVICTOR: OnceLock<()> = OnceLock::new();
    EVICTOR.get_or_init(|| pool.spawn_evictor());
    pool
}

/// Ask the allocator to hand freed pages back to the OS. Dropping an
/// ONNX session frees hundreds of megabytes through `malloc`, which on
/// macOS keeps them in the zone until told otherwise.
pub fn release_freed_memory() {
    #[cfg(target_os = "macos")]
    unsafe {
        extern "C" {
            fn malloc_zone_pressure_relief(zone: *mut std::ffi::c_void, goal: usize) -> usize;
        }
        malloc_zone_pressure_relief(std::ptr::null_mut(), 0);
    }
    #[cfg(target_os = "linux")]
    unsafe {
        extern "C" {
            fn malloc_trim(pad: usize) -> i32;
        }
        malloc_trim(0);
    }
}

/// Text actually fed to the encoder. The jina retrieval model is
/// asymmetric: queries and documents carry different prefixes, and
/// skipping them costs several points of recall. The fastembed models
/// are used exactly as before (no prefix) so their stored vectors stay
/// valid.
fn with_prefix(text: &str, prefix: &str) -> String {
    if selected_model().is_jina() {
        let mut prefixed = String::with_capacity(prefix.len() + text.len());
        prefixed.push_str(prefix);
        prefixed.push_str(text);
        prefixed
    } else {
        text.to_string()
    }
}

fn embed_prefixed(texts: Vec<String>) -> Vec<Vec<f32>> {
    if texts.is_empty() {
        return Vec::new();
    }
    let mut pooled = embed_pool().acquire();
    pooled.get().embed(texts)
}

/// Embed a *search query*. Use this for the text a user is looking
/// with, never for stored content.
pub fn embed_query(text: &str) -> Vec<f32> {
    embed_prefixed(vec![with_prefix(text, JINA_QUERY_PREFIX)])
        .pop()
        .unwrap_or_else(|| vec![0.0; vector_dim()])
}

/// Embed one stored memory (document side).
pub fn embed_text(text: &str) -> Vec<f32> {
    embed_prefixed(vec![with_prefix(text, JINA_DOCUMENT_PREFIX)])
        .pop()
        .unwrap_or_else(|| vec![0.0; vector_dim()])
}

/// Embed a batch of stored memories (document side).
pub fn embed_batch(texts: &[&str]) -> Vec<Vec<f32>> {
    embed_prefixed(
        texts
            .iter()
            .map(|text| with_prefix(text, JINA_DOCUMENT_PREFIX))
            .collect(),
    )
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
/// (4 + dim bytes total). The encoders are L2-normalized so values
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
/// returning vectors during the one-shot re-embed pass triggered by a
/// model swap. A vector returned from a legacy layout is dimensionally
/// incompatible with the active query vector — `similarity_with_blob`
/// and `cosine_similarity` already short-circuit on length mismatch, so
/// legacy rows transparently score 0 until the backfill rewrites them.
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
        let v1 = embed_query("authentication login Supabase auth JWT");
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
    fn test_cross_lingual_query_finds_french_document() {
        let query = embed_query("Which planet is known as the Red Planet?");
        let french = embed_text(
            "Mars, connue pour sa couleur rougeâtre, est souvent appelée la planète rouge.",
        );
        let unrelated = embed_text("Vénus est souvent appelée la jumelle de la Terre.");
        assert!(
            cosine_similarity(&query, &french) > cosine_similarity(&query, &unrelated),
            "EN query must retrieve the matching FR document"
        );
    }

    #[test]
    fn test_embeddings_are_unit_length() {
        let v = embed_text("Décision technique : transport daemon SSE avec fallback stdio.");
        assert_eq!(v.len(), vector_dim());
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-3, "norm {}", norm);
    }

    #[test]
    fn test_batch_matches_single() {
        let texts = ["first memory about Cloudflare Pages", "second memory about SQLite WAL"];
        let batch = embed_batch(&texts);
        assert_eq!(batch.len(), 2);
        let single = embed_text(texts[1]);
        let sim = cosine_similarity(&batch[1], &single);
        assert!(sim > 0.99, "batch/single drift: {}", sim);
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
        let q = embed_query("authentication login Supabase auth JWT");
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
        // until the re-embed pass rewrites them.
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
