use std::cmp::Ordering;
use std::collections::HashSet;
use std::sync::OnceLock;

use crate::db::SearchResult;

pub fn rerank_local(query: &str, results: &mut Vec<SearchResult>) {
    if results.len() <= 1 {
        return;
    }

    let query_lower = query.to_ascii_lowercase();
    let query_tokens = significant_tokens(&query_lower);
    let query_entities = entity_values(query, None);
    let intent = LocalIntent::from_query(&query_lower);

    for result in results.iter_mut() {
        let content_lower = result.memory.content.to_ascii_lowercase();
        let mut factor: f64 = 1.0;

        if !query_tokens.is_empty() {
            let token_hits = query_tokens
                .iter()
                .filter(|token| content_lower.contains(token.as_str()))
                .count();
            let coverage = token_hits as f64 / query_tokens.len() as f64;
            if coverage >= 0.5 {
                factor *= 1.0 + (coverage * 0.10);
            }
        }

        if query_lower.len() >= 12 && content_lower.contains(&query_lower) {
            factor *= 1.12;
        }

        if !query_entities.is_empty() {
            let content_entities =
                entity_values(&result.memory.content, result.memory.project.as_deref());
            let overlap = query_entities.intersection(&content_entities).count();
            if overlap > 0 {
                factor *= 1.0 + (overlap as f64 * 0.05).min(0.15);
            }
        }

        if intent.preference && preference_signal(&result.memory.kind, &content_lower) {
            factor *= 1.08;
        }
        if intent.temporal && temporal_signal(&result.memory.id, &content_lower) {
            factor *= 1.06;
        }
        if intent.user_turn && content_lower.trim_start().starts_with("user:") {
            factor *= 1.07;
        }
        if intent.assistant_turn && content_lower.trim_start().starts_with("assistant:") {
            factor *= 1.07;
        }
        if intent.update && update_signal(&content_lower) {
            factor *= 1.07;
        }
        result.score = (result.score * factor.min(1.35) * 10000.0).round() / 10000.0;
    }

    results.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.memory.id.cmp(&right.memory.id))
    });

    for index in 1..results.len() {
        let previous = results[index - 1].score;
        if results[index].score > previous {
            results[index].score = (previous * 0.999 * 10000.0).round() / 10000.0;
        }
    }
}

/// Returns the best raw cross-encoder logit among the reranked window
/// when the lane ran — a calibrated relevance signal the caller folds
/// into its confidence verdict — and `None` when it was skipped.
pub fn rerank_cross_encoder_if_enabled(
    query: &str,
    results: &mut Vec<SearchResult>,
) -> Option<f32> {
    if results.len() <= 1 || !should_run_cross_encoder(query) {
        return None;
    }

    // Confidence gate: when the primary lane already separates the top
    // hit from the rest by a wide score margin, the cross-encoder
    // rarely changes the order — and paying ~150-200 ms per query for
    // a no-op decision is the wrong trade. The gate measures the
    // relative gap between the top-1 and the top-3 score; if the top
    // is confidently ahead, we skip rerank. Tuning: a gap >= 25% of
    // the top-1 score has been observed to leave R@5 unchanged on
    // memorypilot-fr and LongMemEval while halving the average rerank
    // workload on the easy English tail. Lower thresholds were tried
    // in v4.3 (0.12, 0.18) but cost R@5 on FR without producing a
    // real latency win, so 0.25 was retained.
    //
    // The gate is intentionally OFF when the user forces rerank with
    // MEMORYPILOT_CROSS_RERANK=1: in that mode they explicitly want
    // every query reranked (typically for ranking ablations).
    if !is_force_enabled() && is_confident_top(results) {
        return None;
    }

    let top_k = std::env::var("MEMORYPILOT_CROSS_RERANK_TOP_K")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(12)
        .clamp(2, 64)
        .min(results.len());

    let documents = results
        .iter()
        .take(top_k)
        .map(|result| truncate_document(&result.memory.content))
        .collect::<Vec<_>>();

    let reranked = {
        // Acquire one model from the rerank pool. The pool serializes
        // access only at the level of an individual model — multiple
        // workers can rerank in parallel as long as the pool has free
        // models. Without it, every concurrent search funneled through
        // a single Mutex<TextRerank> and the throughput collapsed
        // (-18% qps when forcing rerank on every query in the
        // concurrency bench).
        match acquire_pooled_reranker() {
            Some(mut handle) => handle
                .with_model(|model| model.scores(query, &documents))
                .flatten(),
            None => None,
        }
    };

    let Some(scores) = reranked else {
        return None;
    };
    if scores.len() != top_k {
        return None;
    }
    let best_raw = scores
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let reranked: Vec<CrossScore> = scores
        .into_iter()
        .enumerate()
        .map(|(index, score)| CrossScore { index, score })
        .collect();

    let min_cross = reranked
        .iter()
        .map(|result| result.score)
        .fold(f32::INFINITY, f32::min);
    let max_cross = reranked
        .iter()
        .map(|result| result.score)
        .fold(f32::NEG_INFINITY, f32::max);
    let cross_span = (max_cross - min_cross).max(0.0001);

    let original_scores = results
        .iter()
        .take(top_k)
        .map(|result| result.score)
        .collect::<Vec<_>>();
    let min_original = original_scores
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let max_original = original_scores
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let original_span = (max_original - min_original).max(0.0001);

    // Fusion ratio between the original RRF score and the cross
    // encoder's relevance signal. Tunable via env var so callers can
    // sweep without recompiling. Default 0.55 / 0.45 was the best
    // operating point in the in-house sweep on memorypilot-fr-v2 and
    // LongMemEval-S — the previous 0.70 / 0.30 was too conservative
    // and left several preference / temporal cases on the table where
    // the cross encoder was clearly more confident than RRF.
    let cross_weight = std::env::var("MEMORYPILOT_CROSS_RERANK_WEIGHT")
        .ok()
        .and_then(|raw| raw.parse::<f64>().ok())
        .unwrap_or(0.45)
        .clamp(0.0, 1.0);
    let original_weight = 1.0 - cross_weight;

    let mut fused = results.drain(..top_k).collect::<Vec<_>>();
    for cross in reranked {
        if let Some(result) = fused.get_mut(cross.index) {
            let original_norm = (result.score - min_original) / original_span;
            let cross_norm = (cross.score - min_cross) as f64 / cross_span as f64;
            let fused_score = (original_norm * original_weight) + (cross_norm * cross_weight);
            result.score = (fused_score * 10000.0).round() / 10000.0;
        }
    }

    fused.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.memory.id.cmp(&right.memory.id))
    });
    fused.extend(results.drain(..));
    *results = fused;
    Some(best_raw)
}

struct CrossScore {
    index: usize,
    score: f32,
}

enum CrossRerankerState {
    Ready(Reranker),
    Unavailable(String),
}

/// A loaded cross-encoder. The default mmarco model is driven through
/// `ort` directly; the legacy fastembed rerankers stay available behind
/// `MEMORYPILOT_RERANKER_MODEL`.
enum Reranker {
    Ort(OrtReranker),
    Fastembed(fastembed::TextRerank),
}

impl Reranker {
    /// One relevance score per document, in input order.
    fn scores(&mut self, query: &str, documents: &[String]) -> Option<Vec<f32>> {
        match self {
            Reranker::Ort(model) => match model.scores(query, documents) {
                Ok(scores) => Some(scores),
                Err(error) => {
                    eprintln!("[reranker] cross-encoder run failed: {}", error);
                    None
                }
            },
            Reranker::Fastembed(model) => {
                let results = model
                    .rerank(query.to_string(), documents, false, Some(16))
                    .ok()?;
                let mut scores = vec![0.0f32; documents.len()];
                for result in results {
                    if let Some(slot) = scores.get_mut(result.index) {
                        *slot = result.score;
                    }
                }
                Some(scores)
            }
        }
    }
}

/// Padded tokens per ONNX run for the cross-encoder (batch × longest
/// pair). 12 memory snippets of ~100 tokens fit in one run; 512-token
/// pairs go two at a time. Small on purpose — see the embedder.
const RERANK_TOKEN_BUDGET: usize = 1024;
const RERANK_MAX_TOKENS: usize = 512;

struct OrtReranker {
    session: ort::session::Session,
    tokenizer: crate::tokenizer::Unigram,
    run_options: ort::session::RunOptions,
    needs_token_type_ids: bool,
}

/// An XLM-R–family cross-encoder served through `ort`: Hugging Face repo
/// plus the int8 graph inside it. All three share the same Unigram
/// tokenizer and `<s> q </s></s> d </s>` pair template.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct OrtRerankerSpec {
    pub repo: &'static str,
    pub onnx_file: &'static str,
    pub label: &'static str,
}

impl OrtReranker {
    fn load(spec: OrtRerankerSpec) -> Result<Self, String> {
        // The repositories embed the weights in the protobuf; ONNX Runtime
        // 1.28 loads that layout with ~3 copies resident. Convert once to
        // external data so the weights are memory-mapped instead.
        let graph_path = crate::onnx_external::externalized(spec.repo, spec.onnx_file, false)?;
        let tokenizer_path = crate::embedding::hf_file(spec.repo, "tokenizer.json", false)?;
        let tokenizer = crate::tokenizer::Unigram::from_file(&tokenizer_path)
            .map_err(|error| format!("reranker tokenizer load: {}", error))?;

        use ort::session::builder::GraphOptimizationLevel;
        let session = ort::session::Session::builder()
            .map_err(|error| format!("ort builder: {}", error))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|error| format!("ort optimization level: {}", error))?
            .with_intra_threads(rerank_intra_threads())
            .map_err(|error| format!("ort intra threads: {}", error))?
            .with_memory_pattern(false)
            .map_err(|error| format!("ort memory pattern: {}", error))?
            .with_config_entry("session.use_device_allocator_for_initializers", "1")
            .map_err(|error| format!("ort config entry: {}", error))?
            .commit_from_file(&graph_path)
            .map_err(|error| format!("ort session load {}: {}", graph_path.display(), error))?;
        let needs_token_type_ids = session
            .inputs()
            .iter()
            .any(|input| input.name() == "token_type_ids");

        let mut run_options = ort::session::RunOptions::new()
            .map_err(|error| format!("ort run options: {}", error))?;
        run_options
            .set("memory.enable_memory_arena_shrinkage", "cpu:0")
            .map_err(|error| format!("ort arena shrinkage: {}", error))?;

        Ok(Self {
            session,
            tokenizer,
            run_options,
            needs_token_type_ids,
        })
    }

    fn scores(&mut self, query: &str, documents: &[String]) -> Result<Vec<f32>, String> {
        let encodings: Vec<Vec<u32>> = documents
            .iter()
            .map(|document| self.tokenizer.encode_pair(query, document, RERANK_MAX_TOKENS))
            .collect();

        let mut order: Vec<usize> = (0..encodings.len()).collect();
        order.sort_by_key(|&index| std::cmp::Reverse(encodings[index].len()));

        let mut scores = vec![0.0f32; documents.len()];
        let mut start = 0;
        while start < order.len() {
            let width = encodings[order[start]].len().max(1);
            let rows = (RERANK_TOKEN_BUDGET / width).max(1);
            let end = (start + rows).min(order.len());
            let pack = &order[start..end];
            let logits = self.run_pack(&encodings, pack, width)?;
            for (&index, logit) in pack.iter().zip(logits) {
                scores[index] = logit;
            }
            start = end;
        }
        Ok(scores)
    }

    fn run_pack(
        &mut self,
        encodings: &[Vec<u32>],
        pack: &[usize],
        width: usize,
    ) -> Result<Vec<f32>, String> {
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
        let mut inputs = ort::inputs![
            "input_ids" => ort::value::Tensor::from_array((shape, ids))
                .map_err(|error| format!("input_ids tensor: {}", error))?,
            "attention_mask" => ort::value::Tensor::from_array((shape, mask))
                .map_err(|error| format!("attention_mask tensor: {}", error))?,
        ];
        if self.needs_token_type_ids {
            let zeros = vec![0i64; batch * width];
            inputs.push((
                "token_type_ids".into(),
                ort::value::Tensor::from_array((shape, zeros))
                    .map_err(|error| format!("token_type_ids tensor: {}", error))?
                    .into(),
            ));
        }
        let outputs = self
            .session
            .run_with_options(inputs, &self.run_options)
            .map_err(|error| format!("ort run: {}", error))?;
        let (out_shape, data) = outputs["logits"]
            .try_extract_tensor::<f32>()
            .map_err(|error| format!("logits extract: {}", error))?;
        let dims: Vec<i64> = out_shape.iter().copied().collect();
        // [batch, 1] for the single-label regression head.
        if dims.first().copied() != Some(batch as i64) || data.len() != batch {
            return Err(format!("logits shape {:?}, expected [{}, 1]", dims, batch));
        }
        Ok(data.to_vec())
    }
}

/// True when the top-1 score sits far enough above the top-3 score
/// that a rerank is unlikely to change the ordering. Returns `false`
/// for very short result lists where the gap is meaningless. The
/// gate ratio is tunable via `MEMORYPILOT_GATE_RATIO` (default 0.25
/// — empirically the best operating point on the FR bench, lower
/// ratios cost R@5 without meaningfully improving latency because
/// most non-English queries have naturally low gap/top ratios and
/// fail the gate either way).
fn is_confident_top(results: &[SearchResult]) -> bool {
    if results.len() < 3 {
        return false;
    }
    let top = results[0].score;
    let third = results[2].score;
    if top <= 0.0 {
        return false;
    }
    let ratio = std::env::var("MEMORYPILOT_GATE_RATIO")
        .ok()
        .and_then(|raw| raw.parse::<f64>().ok())
        .unwrap_or(0.25)
        .clamp(0.0, 1.0);
    let gap = top - third;
    gap / top >= ratio
}

fn is_force_enabled() -> bool {
    matches!(
        std::env::var("MEMORYPILOT_CROSS_RERANK").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE") | Ok("always") | Ok("fastembed") | Ok("onnx")
    )
}

fn should_run_cross_encoder(query: &str) -> bool {
    // Default: `adaptive`. Empirically the right operating point —
    // `1`/`true` blanket-on inflates p50 latency from ~50ms to ~2s on
    // LongMemEval (43x) for a +0.5pp R@5 gain, while `adaptive` keeps
    // latency near baseline on the easy English long tail and still
    // captures the high-value gains on hard queries (preference,
    // temporal, "did I mention", multilingual). On the FR bench the
    // adaptive lane catches every French query (R@5 +6.6pp, R@10
    // +6.6pp, MRR +5.9pp at ~480ms/query).
    //
    // Tested in v4.3: dropping the heuristic entirely and relying
    // only on the score-based confidence gate cost -2.8 pp R@5 on
    // memorypilot-fr-v2 because the gate signal alone misclassifies
    // a long tail of ambiguous French queries with naturally wide
    // gaps. The surface heuristic + gate combo is strictly better.
    match std::env::var("MEMORYPILOT_CROSS_RERANK") {
        Ok(value)
            if matches!(
                value.as_str(),
                "1" | "true" | "TRUE" | "always" | "fastembed" | "onnx"
            ) =>
        {
            true
        }
        Ok(value) if matches!(value.as_str(), "0" | "false" | "FALSE" | "off") => false,
        // Default + explicit "adaptive": run only when the query
        // looks hard or non-English (where the multilingual cross
        // encoder shines).
        _ => is_hard_query(query),
    }
}

fn is_hard_query(query: &str) -> bool {
    let query_lower = query.to_ascii_lowercase();
    let intent = LocalIntent::from_query(&query_lower);
    if intent.temporal || intent.preference || intent.update {
        return true;
    }
    if contains_any(
        &query_lower,
        &[
            "what did i",
            "who did i",
            "where did i",
            "how long",
            "do you think",
            "should i",
            "did i mention",
            "i mentioned",
            "what was the",
            "who was",
        ],
    ) {
        return true;
    }
    looks_non_english(query)
}

/// Cheap heuristic to detect non-English queries. Triggers on French
/// accented characters, common French/Spanish/German function words,
/// and CJK ranges. Designed to be conservative: false positives just
/// add latency, but missing a non-English query loses the +6pp R@5
/// the cross encoder buys us on the FR bench.
fn looks_non_english(query: &str) -> bool {
    if query
        .chars()
        .any(|c| matches!(c, 'é' | 'è' | 'ê' | 'à' | 'â' | 'ç' | 'ô' | 'û' | 'ù' | 'î' | 'ï' | 'ü' | 'ö' | 'ä' | 'ñ' | 'É' | 'È' | 'Ê' | 'À'))
    {
        return true;
    }
    if query.chars().any(|c| {
        let cp = c as u32;
        // CJK Unified Ideographs, Hiragana, Katakana, Hangul.
        (0x3040..=0x30FF).contains(&cp)
            || (0x4E00..=0x9FFF).contains(&cp)
            || (0xAC00..=0xD7AF).contains(&cp)
    }) {
        return true;
    }
    let lower = query.to_ascii_lowercase();
    let padded = format!(" {} ", lower);
    // Strong French markers: articles, prepositions, common verbs and
    // question words that essentially never appear in English. The
    // false-positive cost is just extra latency on a misclassified
    // query; the false-negative cost is a -6pp R@5 hit on the FR
    // bench, so we err on the side of recall.
    let french_markers: &[&str] = &[
        " un ", " une ", " des ", " les ", " du ", " au ", " aux ", " et ", " ou ",
        " avec ", " sans ", " pour ", " dans ", " entre ", " chez ", " sur ", " sous ",
        " vers ", " contre ", " selon ", " parmi ", " depuis ", " pendant ", " malgré ",
        " comment ", " pourquoi ", " quand ", " quel ", " quelle ", " quels ", " quelles ",
        " que ", " qui ", " quoi ", " où ", " ce ", " cette ", " ces ", " cet ",
        " mais ", " donc ", " car ", " parce ", " ainsi ",
        " est ", " sont ", " être ", " avoir ", " faire ", " aller ",
        " je ", " tu ", " nous ", " vous ", " ils ", " elles ", " on ",
        " mon ", " ton ", " son ", " ma ", " ta ", " sa ", " mes ", " tes ", " ses ",
        " notre ", " votre ", " leur ", " leurs ",
        " plus ", " moins ", " très ", " bien ", " trop ", " aussi ",
        "qu'", "n'", "d'", "l'", "j'", "s'", "c'", "m'", "t'",
    ];
    if french_markers.iter().any(|m| padded.contains(m)) {
        return true;
    }
    let other_romance_markers: &[&str] = &[
        " hola ", " gracias ", " porque ", " usted ", " también ", " pero ", " muy ",
        " danke ", " bitte ", " nicht ", " sehr ", " auch ", " ist ", " mit ", " der ",
        " die ", " das ", " und ", " oder ", " aber ",
    ];
    other_romance_markers.iter().any(|m| padded.contains(m))
}

/// Pool of cross-encoder model instances. Each instance owns its own
/// ONNX session (~120 MB resident for the default int8
/// mmarco-mMiniLMv2, ~1.1 GB for the legacy jina-v2-multilingual-base),
/// so the default pool size is intentionally small (1). Throughput-bound
/// workloads can opt in to 2 via `MEMORYPILOT_RERANK_POOL_SIZE=2`.
/// Sessions are built on first use and released after
/// `MEMORYPILOT_MODEL_IDLE_SECS` idle (see `crate::pool`).
static RERANK_POOL: OnceLock<crate::pool::IdlePool<CrossRerankerState>> = OnceLock::new();

fn rerank_pool() -> &'static crate::pool::IdlePool<CrossRerankerState> {
    let pool = RERANK_POOL.get_or_init(|| {
        let pool_size = std::env::var("MEMORYPILOT_RERANK_POOL_SIZE")
            .ok()
            .and_then(|raw| raw.parse::<usize>().ok())
            .unwrap_or(1)
            .clamp(1, 4);
        crate::pool::IdlePool::new("reranker", pool_size, init_cross_reranker)
    });
    static EVICTOR: OnceLock<()> = OnceLock::new();
    EVICTOR.get_or_init(|| pool.spawn_evictor());
    pool
}

struct PooledReranker {
    inner: crate::pool::PoolGuard<CrossRerankerState>,
}

impl PooledReranker {
    fn with_model<R>(&mut self, f: impl FnOnce(&mut Reranker) -> R) -> Option<R> {
        match self.inner.get() {
            CrossRerankerState::Ready(model) => Some(f(model)),
            CrossRerankerState::Unavailable(error) => {
                eprintln!("[reranker] cross-encoder unavailable: {}", error);
                None
            }
        }
    }
}

fn acquire_pooled_reranker() -> Option<PooledReranker> {
    Some(PooledReranker {
        inner: rerank_pool().acquire(),
    })
}

/// Pre-load every cross-encoder model in the pool and run a single
/// throwaway query against a tiny document so the first real call
/// from a benchmark or HTTP handler does not pay the ONNX hydration
/// cost. Safe to call multiple times.
pub fn warmup_cross_reranker() {
    let pool = rerank_pool();
    let mut warmed: Vec<PooledReranker> = Vec::with_capacity(pool.capacity());
    for _ in 0..pool.capacity() {
        let Some(mut handle) = acquire_pooled_reranker() else {
            break;
        };
        let _ = handle.with_model(|model| model.scores("warmup", &["warmup".to_string()]));
        warmed.push(handle);
    }
    drop(warmed);
}

/// Default cross-encoder: `mmarco-mMiniLMv2-L12-H384-v1` (Apache-2.0,
/// 14 languages incl. French, trained on mMARCO), dynamically quantized
/// to int8 — 118 MB on disk, ~250 MB resident. It replaces
/// jina-reranker-v2-base-multilingual (1.1 GB) as the default, for a
/// rerank of 12 candidates in ~100 ms instead of ~200 ms. Its quality on
/// short memory snippets is on par: the cross-encoder only refines the
/// top of an already good hybrid list.
///
/// Why this export and not the official `cross-encoder/…/onnx/*qint8*`
/// files: those are QDQ-format graphs whose 250k-row embedding table
/// ONNX Runtime materialises in fp32 at load — measured +306 MB after
/// load and +552 MB after a few runs, versus +118 MB / +249 MB for the
/// dynamic-quantization export below, which keeps the table int8.
const MMARCO: OrtRerankerSpec = OrtRerankerSpec {
    repo: "SugoLabs/mmarco-mMiniLMv2-L12-H384-v1",
    onnx_file: "onnx/model_quantized.onnx",
    label: "mmarco-mMiniLMv2-L12-H384-v1 int8",
};

/// `jina-reranker-v2-base-multilingual` (278M params, CC-BY-NC-4.0) as
/// the int8 export shipped in the official repository: 280 MB on disk,
/// memory-mapped. Opt in with `MEMORYPILOT_RERANKER_MODEL=jina-v2`.
const JINA_V2_INT8: OrtRerankerSpec = OrtRerankerSpec {
    repo: "jinaai/jina-reranker-v2-base-multilingual",
    onnx_file: "onnx/model_int8.onnx",
    label: "jina-reranker-v2-base-multilingual int8",
};

/// `gte-multilingual-reranker-base` (306M params, Apache-2.0), int8
/// export from onnx-community: 341 MB on disk, memory-mapped. Opt in
/// with `MEMORYPILOT_RERANKER_MODEL=gte-multilingual`.
const GTE_MULTILINGUAL_INT8: OrtRerankerSpec = OrtRerankerSpec {
    repo: "onnx-community/gte-multilingual-reranker-base",
    onnx_file: "onnx/model_int8.onnx",
    label: "gte-multilingual-reranker-base int8",
};

enum RerankerChoice {
    Ort(OrtRerankerSpec),
    Fastembed(fastembed::RerankerModel),
}

fn init_cross_reranker() -> CrossRerankerState {
    let result = match cross_reranker_model() {
        RerankerChoice::Ort(spec) => OrtReranker::load(spec)
            .map(Reranker::Ort)
            .map_err(|error| format!("{} init: {}", spec.label, error)),
        RerankerChoice::Fastembed(model) => {
            let cache_dir = dirs::home_dir()
                .unwrap_or_else(std::env::temp_dir)
                .join(".fastembed_cache")
                .join("rerank");
            let _ = std::fs::create_dir_all(&cache_dir);
            let options = fastembed::RerankInitOptions::new(model)
                .with_show_download_progress(false)
                .with_cache_dir(cache_dir)
                .with_intra_threads(rerank_intra_threads());
            fastembed::TextRerank::try_new(options)
                .map(Reranker::Fastembed)
                .map_err(|error| error.to_string())
        }
    };
    match result {
        Ok(model) => CrossRerankerState::Ready(model),
        Err(error) => CrossRerankerState::Unavailable(error),
    }
}

/// Rerank runs on 12 short query/document pairs; two threads saturate
/// it. `MEMORYPILOT_RERANK_THREADS` overrides.
fn rerank_intra_threads() -> usize {
    let available = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(2);
    std::env::var("MEMORYPILOT_RERANK_THREADS")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .unwrap_or_else(|| available.min(2))
        .clamp(1, available.max(1))
}

fn cross_reranker_model() -> RerankerChoice {
    match std::env::var("MEMORYPILOT_RERANKER_MODEL")
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "bge-base" | "bge-reranker-base" | "baai/bge-reranker-base" => {
            RerankerChoice::Fastembed(fastembed::RerankerModel::BGERerankerBase)
        }
        "bge-v2-m3" | "rozgo/bge-reranker-v2-m3" => {
            RerankerChoice::Fastembed(fastembed::RerankerModel::BGERerankerV2M3)
        }
        "jina-v1" | "jinaai/jina-reranker-v1-turbo-en" => {
            RerankerChoice::Fastembed(fastembed::RerankerModel::JINARerankerV1TurboEn)
        }
        "jina-v2-multilingual-fp32" => {
            RerankerChoice::Fastembed(fastembed::RerankerModel::JINARerankerV2BaseMultiligual)
        }
        "jina-v2" | "jina-v2-multilingual" | "jinaai/jina-reranker-v2-base-multilingual" => {
            RerankerChoice::Ort(JINA_V2_INT8)
        }
        "gte" | "gte-multilingual" | "alibaba-nlp/gte-multilingual-reranker-base" => {
            RerankerChoice::Ort(GTE_MULTILINGUAL_INT8)
        }
        _ => RerankerChoice::Ort(MMARCO),
    }
}

fn truncate_document(content: &str) -> String {
    const MAX_CHARS: usize = 1400;
    if content.len() <= MAX_CHARS {
        return content.to_string();
    }
    let boundary = content
        .char_indices()
        .take_while(|(index, _)| *index <= MAX_CHARS)
        .map(|(index, _)| index)
        .last()
        .unwrap_or(MAX_CHARS);
    content[..boundary].to_string()
}

#[derive(Debug, Default)]
struct LocalIntent {
    preference: bool,
    temporal: bool,
    user_turn: bool,
    assistant_turn: bool,
    update: bool,
}

impl LocalIntent {
    fn from_query(query_lower: &str) -> Self {
        Self {
            preference: contains_any(
                query_lower,
                &[
                    "prefer",
                    "preference",
                    "favorite",
                    "favourite",
                    "like",
                    "préf",
                    "aime",
                    "favori",
                ],
            ),
            temporal: contains_any(
                query_lower,
                &[
                    "when", "before", "after", "last", "recent", "ago", "week", "month", "avant",
                    "après", "dernier", "semaine", "mois",
                ],
            ),
            user_turn: contains_any(query_lower, &["i said", "i told", "user", "j'ai dit"]),
            assistant_turn: contains_any(
                query_lower,
                &[
                    "you said",
                    "assistant",
                    "claude",
                    "cursor",
                    "chatgpt",
                    "tu as dit",
                ],
            ),
            update: contains_any(
                query_lower,
                &[
                    "changed",
                    "updated",
                    "actually",
                    "instead",
                    "switch",
                    "correction",
                    "modifié",
                    "en fait",
                    "plutôt",
                ],
            ),
        }
    }
}

fn significant_tokens(text: &str) -> Vec<String> {
    text.split(|character: char| !character.is_alphanumeric())
        .filter(|token| token.len() >= 4)
        .filter(|token| !STOPWORDS.contains(token))
        .map(ToOwned::to_owned)
        .collect()
}

fn entity_values(text: &str, project: Option<&str>) -> HashSet<String> {
    crate::graph::extract_entities(text, project)
        .into_iter()
        .filter(|entity| crate::graph::is_reliable_link_entity(entity))
        .map(|entity| entity.value.to_ascii_lowercase())
        .collect()
}

fn preference_signal(kind: &str, content_lower: &str) -> bool {
    kind == "preference"
        || contains_any(
            content_lower,
            &[
                "prefer",
                "favorite",
                "favourite",
                "like",
                "would rather",
                "i usually",
                "i often",
                "i enjoy",
                "i love",
                "i tend",
                "i think",
                "i recently",
                "i've been",
                "my ",
                "préf",
                "aime",
                "favori",
            ],
        )
}

fn temporal_signal(memory_id: &str, content_lower: &str) -> bool {
    memory_id.contains("__t")
        || contains_any(
            content_lower,
            &[
                "today",
                "yesterday",
                "last ",
                "ago",
                "week",
                "month",
                "maintenant",
                "hier",
                "semaine",
            ],
        )
}

fn update_signal(content_lower: &str) -> bool {
    contains_any(
        content_lower,
        &[
            "changed",
            "updated",
            "actually",
            "instead",
            "switched",
            "correction",
            "modifié",
            "désormais",
            "en fait",
        ],
    )
}

fn contains_any(text: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| text.contains(needle))
}

const STOPWORDS: &[&str] = &[
    "what", "when", "where", "which", "with", "that", "this", "then", "than", "have", "about",
    "your", "pour", "dans", "avec", "quoi", "quel", "quelle", "est-ce",
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::Memory;

    fn rss_mb() -> f64 {
        let out = std::process::Command::new("ps")
            .args(["-o", "rss=", "-p", &std::process::id().to_string()])
            .output()
            .expect("ps");
        String::from_utf8_lossy(&out.stdout)
            .trim()
            .parse::<f64>()
            .unwrap_or(0.0)
            / 1024.0
    }

    /// Memory probe for the default cross-encoder: load + a realistic
    /// rerank of 12 snippets. Prints RSS deltas; asserts the model ranks
    /// the relevant snippet first. Run with `--nocapture` to read the
    /// numbers.
    #[test]
    fn mmarco_ort_reranker_scores_and_footprint() {
        let before = rss_mb();
        let mut model = OrtReranker::load(MMARCO).expect("mmarco load");
        let loaded = rss_mb();
        let query = "comment publier un post Instagram depuis Sociomator";
        let mut docs: Vec<String> = (0..11)
            .map(|i| format!("Mémoire {} : configuration WAL SQLite et pool de lecture pour la concurrence.", i))
            .collect();
        docs.push("Publier une publication Instagram depuis l'app Sociomator : onglet Planifier, bouton Publier.".to_string());
        let mut scores = Vec::new();
        for _ in 0..5 {
            scores = model.scores(query, &docs).expect("scores");
        }
        let after = rss_mb();
        // Worst case the search path produces: 12 snippets at the
        // 1400-char truncation limit (~400 tokens each).
        let long_docs: Vec<String> = (0..12)
            .map(|i| format!("Mémoire {} : ", i) + &"configuration WAL SQLite, pool de lecture, index ANN usearch et migration du modèle d'embedding. ".repeat(14))
            .collect();
        let started = std::time::Instant::now();
        for _ in 0..3 {
            model.scores(query, &long_docs).expect("long scores");
        }
        let long_ms = started.elapsed().as_millis() / 3;
        let after_long = rss_mb();
        eprintln!(
            "[mmarco probe] load +{:.0} MB, after 5 short reranks +{:.0} MB, after 3 long reranks +{:.0} MB ({} ms each, total {:.0} MB)",
            loaded - before,
            after - before,
            after_long - before,
            long_ms,
            after_long
        );
        let best = scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(best, 11, "relevant snippet must rank first: {:?}", scores);
    }

    #[test]
    fn confidence_gate_skips_when_top_is_clear() {
        let results = vec![
            result("1", "doc one", 1.00),
            result("2", "doc two", 0.85),
            result("3", "doc three", 0.60),
        ];
        // gap = 0.40, top = 1.0, ratio = 0.40 >= 0.25 → confident
        assert!(is_confident_top(&results));
    }

    #[test]
    fn confidence_gate_runs_when_top_is_close() {
        let results = vec![
            result("1", "doc one", 1.00),
            result("2", "doc two", 0.96),
            result("3", "doc three", 0.92),
        ];
        // gap = 0.08, ratio = 0.08 < 0.25 → not confident
        assert!(!is_confident_top(&results));
    }

    #[test]
    fn confidence_gate_runs_on_short_list() {
        let results = vec![
            result("1", "doc one", 1.00),
            result("2", "doc two", 0.10),
        ];
        assert!(!is_confident_top(&results));
    }

    #[test]
    fn local_rerank_boosts_exact_relevance() {
        let mut results = vec![
            result("1", "user: unrelated cooking note", 1.0),
            result("2", "user: I prefer dark mode dashboards", 0.96),
        ];
        rerank_local("prefer dark mode", &mut results);
        assert_eq!(results[0].memory.id, "2");
    }

    #[test]
    fn french_query_marked_hard() {
        assert!(looks_non_english("comment pondérer les colonnes BM25"));
        assert!(looks_non_english("où trouver le fichier"));
        assert!(looks_non_english("démarrage non bloquant avec warm-up"));
        // No accents, but unmistakably French via articles/prepositions.
        assert!(looks_non_english(
            "ajouter un secret sur Cloudflare Pages en ligne de commande"
        ));
        assert!(looks_non_english(
            "convention de nommage des outils MCP"
        ));
        assert!(looks_non_english("strategie de fusion des doublons"));
    }

    #[test]
    fn english_query_not_marked_hard_by_language() {
        assert!(!looks_non_english("what is the configuration value"));
        assert!(!looks_non_english("how to configure BM25 weights"));
    }

    #[test]
    fn cjk_query_marked_hard() {
        assert!(looks_non_english("中文 query"));
        assert!(looks_non_english("こんにちは world"));
    }

    #[test]
    fn romance_marker_detection() {
        assert!(looks_non_english("hola que tal"));
        assert!(looks_non_english("danke schön"));
    }

    fn result(id: &str, content: &str, score: f64) -> SearchResult {
        SearchResult {
            memory: Memory {
                id: id.to_string(),
                content: content.to_string(),
                kind: "note".to_string(),
                project: None,
                tags: Vec::new(),
                source: "test".to_string(),
                importance: 3,
                expires_at: None,
                created_at: String::new(),
                updated_at: String::new(),
                metadata: None,
                last_accessed_at: None,
                access_count: 0,
            },
            score,
            sources: Vec::new(),
        }
    }
}
