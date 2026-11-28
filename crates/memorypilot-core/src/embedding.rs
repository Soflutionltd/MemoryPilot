/// MemoryPilot v4.4 — Embedding Engine.
///
/// Default model: **jina-embeddings-v5-text-nano-retrieval** (768-dim,
/// int8 ONNX, ~250 MB on disk, 100+ languages, 8k context, last-token
/// pooling done *inside* the exported graph via its `sentence_embedding`
/// output). It is the strongest sub-300M multilingual retrieval encoder
/// available today and beats multilingual-e5-small by a wide margin on
/// MMTEB — especially on French queries, which is where MemoryPilot's
/// users live. Licence: CC-BY-NC-4.0 (non-commercial use only); the model
/// is downloaded straight from Jina's Hugging Face repository at first
/// run, MemoryPilot does not redistribute it.
///
/// The jina model is driven by `ort` directly (not through fastembed):
/// fastembed only knows CLS/mean pooling and loads "bring your own"
/// graphs from memory, which would duplicate the 250 MB weights while
/// the session hydrates. Talking to ONNX Runtime ourselves lets us load
/// from file (external-data friendly), pick the pre-pooled output and
/// cap the intra-op threads so the server stays a good citizen.
///
/// Only the jina models are supported. `MEMORYPILOT_EMBED_MODEL=jina-q4`
/// selects the 4-bit variant (~140 MB) for constrained machines; any
/// other value falls back to the int8 default (a database built with an
/// older model is re-embedded once, see `Database::sync_embedding_model`).
///
/// Stored embeddings are int8-quantized (4-byte scale + N i8 bytes,
/// 4× smaller than f32). The on-disk blob length is `4 + dim`. The
/// legacy 384-dim layouts (quantized 388 bytes and f32 1536 bytes)
/// remain readable so existing databases keep working — but vectors
/// produced by one model are mathematically incompatible with queries
/// produced by another, so the database records the active model label
/// and re-embeds everything when it changes (see `Database::open_at`).
use crate::model_pin::PinnedFile;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

/// Legacy 384-dim layouts kept for backwards compatibility on existing
/// on-disk blobs (small model). Anything else is rejected as unknown.
const LEGACY_SMALL_DIM: usize = 384;
const LEGACY_SMALL_QUANTIZED_BLOB_LEN: usize = 4 + LEGACY_SMALL_DIM;
const LEGACY_SMALL_F32_BLOB_LEN: usize = LEGACY_SMALL_DIM * 4;

/// The embedding model, at the revision its files below were hashed at:
/// a repository followed on its main branch can change under an
/// installation. Moving to a later revision means new prints here.
const JINA: crate::model_pin::PinnedRepo = crate::model_pin::PinnedRepo {
    repo: "jinaai/jina-embeddings-v5-text-nano-retrieval",
    revision: "ac5d898c8d382b17167c33e5c8af644a3519b47d",
};
const JINA_TOKENIZER: PinnedFile = PinnedFile {
    path: "tokenizer.json",
    bytes: 17_210_235,
    sha256: "98d4a1d32152d6cedf85b5e88f3b205106dca1fe72aaab34e0ac13c238421069",
};
const JINA_QUANTIZED: [PinnedFile; 2] = [
    PinnedFile {
        path: "onnx/model_quantized.onnx",
        bytes: 131_365,
        sha256: "ac93a7417c216e5076e37da2b3599f7ef16513934098a477680440c09f735a08",
    },
    PinnedFile {
        path: "onnx/model_quantized.onnx_data",
        bytes: 247_006_208,
        sha256: "ee7870eb143a7353be08b33f79992a51de3e32b41f684ccd82953a710c2f2f9c",
    },
];
const JINA_Q4: [PinnedFile; 2] = [
    PinnedFile {
        path: "onnx/model_q4.onnx",
        bytes: 129_099,
        sha256: "a44811eb61f90bfcc24720c04e774ce9f5dc1d1539fe32a814aa40b16efbaa1f",
    },
    PinnedFile {
        path: "onnx/model_q4.onnx_data",
        bytes: 137_824_256,
        sha256: "6f63e71508c585ac8221f0d6a3c1afd3200ab2952cbd08421e10e9aa1f9e7c46",
    },
];
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
}

impl SelectedModel {
    fn dim(self) -> usize {
        768
    }

    /// The ONNX graph and its weights, an external `.onnx_data` sibling
    /// ONNX Runtime resolves relative to the graph path, then the tokenizer.
    fn files(self) -> [PinnedFile; 3] {
        let [graph, weights] = match self {
            SelectedModel::JinaV5Nano => JINA_QUANTIZED,
            SelectedModel::JinaV5NanoQ4 => JINA_Q4,
        };
        [graph, weights, JINA_TOKENIZER]
    }

    /// The files of the model in the cache, checked against their prints
    /// (hashed once, see `model_pin`), downloaded on first use. Returns the
    /// graph and the tokenizer.
    fn fetch(self, show_progress: bool) -> Result<(PathBuf, PathBuf), String> {
        let cache = fastembed_cache_dir();
        let [graph, weights, tokenizer] = self.files();
        let graph_path = JINA.file(&cache, &graph, show_progress)?;
        JINA.file(&cache, &weights, show_progress)?;
        let tokenizer_path = JINA.file(&cache, &tokenizer, show_progress)?;
        Ok((graph_path, tokenizer_path))
    }

    /// Human-readable label "name (dim-dim)" for logs and the HTTP /info
    /// endpoint. Reflects the *active* model so diagnostics never lie.
    /// Also persisted in the database as the embedding fingerprint, so
    /// changing it for an existing variant triggers a full re-embed.
    fn label(self) -> &'static str {
        match self {
            SelectedModel::JinaV5Nano => "jina-embeddings-v5-text-nano-retrieval int8, 768-dim",
            SelectedModel::JinaV5NanoQ4 => "jina-embeddings-v5-text-nano-retrieval q4, 768-dim",
        }
    }
}

fn selected_model() -> SelectedModel {
    static CACHED: OnceLock<SelectedModel> = OnceLock::new();
    *CACHED.get_or_init(|| {
        let requested = std::env::var("MEMORYPILOT_EMBED_MODEL")
            .ok()
            .map(|value| value.trim().to_ascii_lowercase())
            .filter(|value| !value.is_empty());
        match requested.as_deref() {
            Some("jina-q4") | Some("jina-nano-q4") | Some("light") | Some("lite") => {
                SelectedModel::JinaV5NanoQ4
            }
            None | Some("jina") | Some("jina-nano") | Some("jina-v5") | Some("default") => SelectedModel::JinaV5Nano,
            Some(other) => {
                eprintln!(
                    "[MemoryPilot] MEMORYPILOT_EMBED_MODEL={} is no longer supported: only the jina models are. Using jina-embeddings-v5-text-nano-retrieval int8.",
                    other
                );
                SelectedModel::JinaV5Nano
            }
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

pub fn fastembed_cache_dir() -> PathBuf {
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
    /// Loads the model. When it does not load, its files are hashed again:
    /// a damaged one is removed and the load is tried once more, which
    /// downloads it. A model whose files are sound fails for another
    /// reason, reported as it is.
    fn load(model: SelectedModel) -> Result<Self, String> {
        match Self::load_from_cache(model) {
            Ok(loaded) => Ok(loaded),
            Err(error) => {
                let removed = JINA.repair(&fastembed_cache_dir(), &model.files());
                if removed == 0 {
                    return Err(error);
                }
                eprintln!(
                    "[MemoryPilot] {} file(s) of the embedding model were damaged ({}): removed, downloading again",
                    removed, error
                );
                Self::load_from_cache(model)
            }
        }
    }

    fn load_from_cache(model: SelectedModel) -> Result<Self, String> {
        let (graph_path, tokenizer_path) = model.fetch(true)?;
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
    Ort(Box<OrtEmbedder>),
    /// The model could not be loaded: why, and when it was tried.
    Unavailable(String, std::time::Instant),
}

/// A model that failed to load is tried again after this long.
const UNAVAILABLE_RETRY: std::time::Duration = std::time::Duration::from_secs(60);
/// How long a caller waits for the model to load. From the cache it takes
/// under a second; a first download goes on in the background past this,
/// and is picked up by the first call made after it ends. Below the ten
/// seconds an agent gives a prompt hook.
const LOAD_DEADLINE: std::time::Duration = std::time::Duration::from_secs(8);

impl Engine {
    fn embed(&mut self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, String> {
        if matches!(self, Engine::Unavailable(_, tried) if tried.elapsed() >= UNAVAILABLE_RETRY) || (matches!(self, Engine::Unavailable(..)) && load_has_ended()) {
            *self = build_engine();
        }
        match self {
            Engine::Ort(model) => model.embed(&texts).map_err(|error| format!("embedding failed: {}", error)),
            Engine::Unavailable(error, _) => Err(error.clone()),
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
/// at 8 to keep memory predictable). 
static EMBED_POOL: OnceLock<crate::pool::IdlePool<Engine>> = OnceLock::new();

/// A load that outlived `LOAD_DEADLINE` (a first download, a network that
/// accepts and does not answer), with when it began: the next attempt reads
/// its outcome instead of starting a second one over the same files.
struct RunningLoad {
    began: std::time::Instant,
    outcome: std::sync::mpsc::Receiver<Result<OrtEmbedder, String>>,
}

static PENDING_LOAD: Mutex<Option<RunningLoad>> = Mutex::new(None);

/// A load that has not ended after this long is given up: its thread is
/// left to itself and a new attempt is made.
const LOAD_ABANDONED: std::time::Duration = std::time::Duration::from_secs(600);

fn pending_load() -> std::sync::MutexGuard<'static, Option<RunningLoad>> {
    match PENDING_LOAD.lock() {
        Ok(pending) => pending,
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// Whether a load left running in the background is worth another look:
/// it has ended, or it is given up.
fn load_has_ended() -> bool {
    let mut pending = pending_load();
    let Some(running) = pending.take() else {
        return false;
    };
    match running.outcome.try_recv() {
        Err(std::sync::mpsc::TryRecvError::Empty) if running.began.elapsed() < LOAD_ABANDONED => {
            *pending = Some(running);
            false
        }
        Err(std::sync::mpsc::TryRecvError::Empty) => true,
        // Its outcome goes back in the slot for `build_engine` to read.
        outcome => {
            let (sender, receiver) = std::sync::mpsc::channel();
            let _ = sender.send(outcome.unwrap_or_else(|_| Err("the loading thread stopped without an answer".to_string())));
            *pending = Some(RunningLoad { began: running.began, outcome: receiver });
            true
        }
    }
}

/// The embedding engine, or why there is none. Never stops the process: a
/// search falls back on words, a write keeps its memory and embeds it later.
fn build_engine() -> Engine {
    let model = selected_model();
    let unavailable = |error: String| {
        eprintln!(
            "[MemoryPilot] Embedding model {} unavailable: {}. Searches use word matching only; new memories are embedded once it is back.",
            model.label(),
            error
        );
        Engine::Unavailable(error, std::time::Instant::now())
    };
    let mut pending = pending_load();
    let loaded = match pending.take() {
        Some(running) if running.began.elapsed() < LOAD_ABANDONED => match running.outcome.try_recv() {
            Ok(loaded) => loaded,
            Err(std::sync::mpsc::TryRecvError::Empty) => {
                let waiting = running.began.elapsed().as_secs();
                *pending = Some(running);
                Err(format!("still loading after {} s (download in progress, or a network that does not answer)", waiting))
            }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => Err("the loading thread stopped without an answer".to_string()),
        },
        _ => {
            let (sender, receiver) = std::sync::mpsc::channel();
            let began = std::time::Instant::now();
            let spawned = std::thread::Builder::new().name("mp-embed-load".into()).spawn(move || {
                let _ = sender.send(OrtEmbedder::load(model));
            });
            match spawned.map(|_| receiver.recv_timeout(LOAD_DEADLINE)) {
                Ok(Ok(loaded)) => loaded,
                Ok(Err(std::sync::mpsc::RecvTimeoutError::Timeout)) => {
                    *pending = Some(RunningLoad { began, outcome: receiver });
                    Err(format!("still loading after {} s (download in progress, or a network that does not answer)", LOAD_DEADLINE.as_secs()))
                }
                Ok(Err(std::sync::mpsc::RecvTimeoutError::Disconnected)) => Err("the loading thread stopped without an answer".to_string()),
                Err(error) => Err(format!("cannot start the loading thread: {}", error)),
            }
        }
    };
    match loaded {
        Ok(engine) => Engine::Ort(Box::new(engine)),
        Err(error) => unavailable(error),
    }
}

fn embed_pool() -> &'static crate::pool::IdlePool<Engine> {
    let pool = EMBED_POOL.get_or_init(|| {
        let default_pool = 1;
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
/// skipping them costs several points of recall.
fn with_prefix(text: &str, prefix: &str) -> String {
    let mut prefixed = String::with_capacity(prefix.len() + text.len());
    prefixed.push_str(prefix);
    prefixed.push_str(text);
    prefixed
}

/// Whether the last embedding asked for failed.
static MODEL_DOWN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Whether the embedding model was unavailable the last time it was asked:
/// what a caller that only read results (a recall) tells its reader.
pub fn model_is_down() -> bool {
    #[cfg(test)]
    if outage::is_on() {
        return true;
    }
    MODEL_DOWN.load(std::sync::atomic::Ordering::Relaxed)
}

fn embed_prefixed(texts: Vec<String>) -> Result<Vec<Vec<f32>>, String> {
    if texts.is_empty() {
        return Ok(Vec::new());
    }
    // A test of the outage fails its own calls, not the engine every other
    // test shares.
    #[cfg(test)]
    if outage::is_on() {
        return Err("embedding model unavailable (test)".to_string());
    }
    let expected = texts.len();
    let embedded = {
        let mut pooled = embed_pool().acquire();
        pooled.get().embed(texts)
    }
    .and_then(|vectors| {
        if vectors.len() == expected {
            Ok(vectors)
        } else {
            Err(format!("embedding returned {} vectors for {} texts", vectors.len(), expected))
        }
    });
    MODEL_DOWN.store(embedded.is_err(), std::sync::atomic::Ordering::Relaxed);
    embedded
}

fn embed_one(text: &str, prefix: &str) -> Result<Vec<f32>, String> {
    embed_prefixed(vec![with_prefix(text, prefix)])?
        .pop()
        .ok_or_else(|| "embedding returned no vector".to_string())
}

/// Embed a *search query*. Use this for the text a user is looking
/// with, never for stored content. `Err` when the model is unavailable:
/// the caller searches by words, it does not stop.
pub fn embed_query(text: &str) -> Result<Vec<f32>, String> {
    embed_one(text, JINA_QUERY_PREFIX)
}

/// Embed one stored memory (document side). `Err` when the model is
/// unavailable: the memory is kept without a vector and embedded later.
pub fn embed_text(text: &str) -> Result<Vec<f32>, String> {
    embed_one(text, JINA_DOCUMENT_PREFIX)
}

/// Whether the files of the active model are in the cache: loading it then
/// needs no network and takes under a second.
pub fn is_cached() -> bool {
    let cache = fastembed_cache_dir();
    selected_model().files().iter().all(|file| JINA.is_cached(&cache, file))
}

/// Downloads the active model's files into the cache (Docker image, first
/// boot, a command that needs the model whatever the wait). Does not run
/// inference.
pub fn prefetch() -> Result<(), String> {
    selected_model().fetch(true).map(|_| ())
}

/// Embed a batch of stored memories (document side).
pub fn embed_batch(texts: &[&str]) -> Result<Vec<Vec<f32>>, String> {
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

/// Tests of what happens without the model: an outage for the calls of the
/// current thread only.
#[cfg(test)]
pub(crate) mod outage {
    thread_local! {
        static ON: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }

    pub(crate) fn is_on() -> bool {
        ON.with(|on| on.get())
    }

    /// The model is down for this thread until the guard is dropped.
    pub(crate) struct Outage;

    pub(crate) fn start() -> Outage {
        ON.with(|on| on.set(true));
        Outage
    }

    impl Drop for Outage {
        fn drop(&mut self) {
            ON.with(|on| on.set(false));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn embed_query(text: &str) -> Vec<f32> {
        super::embed_query(text).expect("embedding model")
    }

    fn embed_text(text: &str) -> Vec<f32> {
        super::embed_text(text).expect("embedding model")
    }

    fn embed_batch(texts: &[&str]) -> Vec<Vec<f32>> {
        super::embed_batch(texts).expect("embedding model")
    }

    #[test]
    fn an_engine_that_could_not_load_answers_with_its_reason_until_the_next_try() {
        let mut engine = Engine::Unavailable("model file missing".to_string(), std::time::Instant::now());
        assert_eq!(engine.embed(vec!["a text".to_string()]), Err("model file missing".to_string()));
        // Asked again at once: the same reason, no new load.
        assert_eq!(engine.embed(vec!["another".to_string()]), Err("model file missing".to_string()));
        assert!(matches!(engine, Engine::Unavailable(..)));
    }

    #[test]
    fn an_outage_is_an_error_not_a_stop() {
        let _outage = outage::start();
        assert!(super::embed_query("anything").is_err());
        assert!(super::embed_text("anything").is_err());
        assert!(super::embed_batch(&["a", "b"]).is_err());
        assert_eq!(super::embed_batch(&[]), Ok(Vec::new()), "nothing to embed needs no model");
    }

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
