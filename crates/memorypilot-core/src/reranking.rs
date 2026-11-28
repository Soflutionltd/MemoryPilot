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

    let top_k = rerank_width(
        std::env::var("MEMORYPILOT_CROSS_RERANK_TOP_K").ok().as_deref(),
        WIDE_SCOPE.with(|c| c.get()),
    )
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

    let cross_weight = cross_weight_from(
        std::env::var("MEMORYPILOT_CROSS_RERANK_WEIGHT").ok().as_deref(),
    );
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

/// Fusion ratio between the original RRF score and the cross encoder's
/// relevance signal, both min-max normalised over the reranked window.
/// At 1.0 the window is ordered by the cross-encoder alone. Measured with
/// multilingual-reranker-small: `--benchmark-fr` R@1 98.2% at 1.0 and
/// 96.3% at 0.6; LongMemEval-S coverage with the first 50 turns, rerank
/// over 64, 97.0% of the questions complete at 1.0.
const DEFAULT_CROSS_WEIGHT: f64 = 1.0;

/// `MEMORYPILOT_CROSS_RERANK_WEIGHT`, bounded to [0, 1]. A value that does
/// not parse, or is not finite, falls back to the default: a NaN weight
/// would make every fused score NaN.
fn cross_weight_from(raw: Option<&str>) -> f64 {
    raw.and_then(|value| value.trim().parse::<f64>().ok())
        .filter(|value| value.is_finite())
        .unwrap_or(DEFAULT_CROSS_WEIGHT)
        .clamp(0.0, 1.0)
}

struct CrossScore {
    index: usize,
    score: f32,
}

enum CrossRerankerState {
    Ready(Reranker),
    /// The model could not be loaded, and when that was tried.
    Unavailable(String, std::time::Instant),
}

/// A failed load (no network at the first search) is tried again after
/// this long, instead of leaving the search without rerank until the
/// session is evicted.
const UNAVAILABLE_RETRY: std::time::Duration = std::time::Duration::from_secs(60);

/// A loaded cross-encoder, driven through `ort` directly.
enum Reranker {
    Ort(Box<OrtReranker>),
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
    tokenizer: crate::tokenizer::SentencePieceBpe,
    run_options: ort::session::RunOptions,
    needs_token_type_ids: bool,
}

/// A cross-encoder served through `ort`: its int8 graph and its
/// `tokenizer.json`, both pinned files of a release of this repository.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct OrtRerankerSpec {
    pub label: &'static str,
    /// Release of the MemoryPilot repository that carries the files.
    pub mirror_tag: &'static str,
    pub graph: crate::model_mirror::MirroredFile,
    pub tokenizer: crate::model_mirror::MirroredFile,
}

impl OrtReranker {
    fn load(spec: OrtRerankerSpec) -> Result<Self, String> {
        // The repositories embed the weights in the protobuf; ONNX Runtime
        // 1.28 loads that layout with ~3 copies resident. Convert once to
        // external data so the weights are memory-mapped instead.
        let source = crate::model_mirror::mirrored_file(spec.mirror_tag, &spec.graph)?;
        let graph_path = crate::onnx_external::externalized_source(&source, spec.mirror_tag)?;
        let tokenizer_path = crate::model_mirror::mirrored_file(spec.mirror_tag, &spec.tokenizer)?;
        let tokenizer = crate::tokenizer::SentencePieceBpe::from_file(&tokenizer_path)
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
        let logits = outputs.get("logits").ok_or("the graph has no logits output")?;
        let (out_shape, data) = logits
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

thread_local! {
    static FORCE_SCOPE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static WIDE_SCOPE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Candidates the cross-encoder rescores by default, and in a wide rerank.
const DEFAULT_RERANK_WIDTH: usize = 12;
pub const WIDE_RERANK_WIDTH: usize = 64;

/// Run `f` with the cross-encoder forced on and rescoring the first
/// [`WIDE_RERANK_WIDTH`] candidates of every search it makes. Used by the
/// compact search: the turns that hold the answer then sit high enough for
/// a short context to keep them (LongMemEval-S, first 50 turns: every
/// evidence turn present for 97.0% of the questions, 95.7% without).
pub fn with_wide_rerank<T>(f: impl FnOnce() -> T) -> T {
    let previous = WIDE_SCOPE.with(|c| c.replace(true));
    let out = with_forced_rerank(f);
    WIDE_SCOPE.with(|c| c.set(previous));
    out
}

/// True inside [`with_wide_rerank`].
pub(crate) fn wide_scope() -> bool {
    WIDE_SCOPE.with(|c| c.get())
}

/// How many candidates the cross-encoder rescores: the wide width inside
/// [`with_wide_rerank`], else `MEMORYPILOT_CROSS_RERANK_TOP_K` (2 to 64).
fn rerank_width(raw: Option<&str>, wide: bool) -> usize {
    if wide {
        return WIDE_RERANK_WIDTH;
    }
    raw.and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(DEFAULT_RERANK_WIDTH)
        .clamp(2, WIDE_RERANK_WIDTH)
}

/// Run `f` with the cross-encoder forced on for every search it makes
/// (unless MEMORYPILOT_CROSS_RERANK=off). Used by `recall`, which runs
/// once per conversation: precision of the first result matters more
/// there than the ~0.2 s the rerank costs.
pub fn with_forced_rerank<T>(f: impl FnOnce() -> T) -> T {
    let previous = FORCE_SCOPE.with(|c| c.replace(true));
    let out = f();
    FORCE_SCOPE.with(|c| c.set(previous));
    out
}

fn forced_scope() -> bool {
    FORCE_SCOPE.with(|c| c.get())
}

fn is_force_enabled() -> bool {
    forced_scope() || matches!(
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
        _ if forced_scope() => true,
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
/// ONNX session (the int8 graph is memory-mapped, 268 MB on disk),
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
        let state = self.inner.get();
        if matches!(state, CrossRerankerState::Unavailable(_, tried) if tried.elapsed() >= UNAVAILABLE_RETRY) {
            *state = init_cross_reranker();
        }
        match state {
            CrossRerankerState::Ready(model) => Some(f(model)),
            CrossRerankerState::Unavailable(error, _) => {
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

/// The cross-encoder: Horizon Labs' `multilingual-reranker-small` (141M
/// params, 47 languages, Apache-2.0, fine-tuned from mmBERT-small) as the
/// int8 export of its repository at revision `23139fec`: 268 MB on disk,
/// memory-mapped. The files are mirrored unmodified, with their licence
/// and notice, in the `model-horizon-small-v1` release of this repository.
pub(crate) const HORIZON_SMALL_INT8: OrtRerankerSpec = OrtRerankerSpec {
    label: "multilingual-reranker-small int8",
    mirror_tag: "model-horizon-small-v1",
    graph: crate::model_mirror::MirroredFile {
        name: "model_quantized.onnx",
        bytes: 268_407_694,
        sha256: "f913154682468bfe92d6a7e02ed4e994e548c112fddc74452b573b17c949046e",
        upstream: "https://huggingface.co/Horizon-Labs/multilingual-reranker-small/resolve/23139fec6f1f448cdf89a5ac2d2c6bf5e8c7b398/onnx/model_quantized.onnx",
    },
    tokenizer: crate::model_mirror::MirroredFile {
        name: "tokenizer.json",
        bytes: 34_363_439,
        sha256: "7925f638b5444897efe3b4608eed56c1652f0ed39c2662c8302de1bc264c6088",
        upstream: "https://huggingface.co/Horizon-Labs/multilingual-reranker-small/resolve/23139fec6f1f448cdf89a5ac2d2c6bf5e8c7b398/tokenizer.json",
    },
};

enum RerankerChoice {
    Ort(OrtRerankerSpec),
}

fn init_cross_reranker() -> CrossRerankerState {
    let result = match cross_reranker_model() {
        RerankerChoice::Ort(spec) => OrtReranker::load(spec)
            .map(|model| Reranker::Ort(Box::new(model)))
            .map_err(|error| format!("{} init: {}{}", spec.label, error, discard_corrupt_files(spec))),
    };
    match result {
        Ok(model) => CrossRerankerState::Ready(model),
        Err(error) => CrossRerankerState::Unavailable(error, std::time::Instant::now()),
    }
}

/// After a failed load: a cached file whose print no longer matches is
/// removed, with the copy converted from it, so that the next attempt
/// fetches it again. Returns what to add to the error message.
fn discard_corrupt_files(spec: OrtRerankerSpec) -> String {
    let mut notes = String::new();
    for file in [&spec.graph, &spec.tokenizer] {
        match crate::model_mirror::remove_if_corrupt(spec.mirror_tag, file) {
            Ok(true) => notes.push_str(&format!("; cached {} was corrupt and is removed, it will be fetched again", file.name)),
            Ok(false) => {}
            Err(error) => notes.push_str(&format!("; cached {} could not be checked: {}", file.name, error)),
        }
    }
    if !notes.is_empty() {
        if let Err(error) = crate::onnx_external::remove_externalized(spec.mirror_tag) {
            notes.push_str(&format!("; {}", error));
        }
    }
    notes
}

/// Threads of one rerank: half the cores, between 2 and 4. Memories are
/// often long (pairs of up to 512 tokens), and on 12 of them four threads
/// answer in two thirds of the time two take (measured on an M2 Max: 3.2 s
/// with 2, 2.2 s with 4, 1.8 s with 8, model load included); beyond four,
/// concurrent searches would fight for the cores.
/// `MEMORYPILOT_RERANK_THREADS` overrides.
fn rerank_intra_threads() -> usize {
    let available = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(2);
    std::env::var("MEMORYPILOT_RERANK_THREADS")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .unwrap_or_else(|| default_rerank_threads(available))
        .clamp(1, available.max(1))
}

fn default_rerank_threads(available: usize) -> usize {
    (available / 2).clamp(2, 4)
}

fn cross_reranker_model() -> RerankerChoice {
    let requested = std::env::var("MEMORYPILOT_RERANKER_MODEL").unwrap_or_default();
    if !names_the_reranker(&requested) {
        eprintln!(
            "[MemoryPilot] MEMORYPILOT_RERANKER_MODEL={} is no longer supported: only multilingual-reranker-small is.",
            requested.trim()
        );
    }
    RerankerChoice::Ort(HORIZON_SMALL_INT8)
}

/// True for an empty `MEMORYPILOT_RERANKER_MODEL` or one that names the
/// only cross-encoder.
fn names_the_reranker(requested: &str) -> bool {
    matches!(
        requested.trim().to_ascii_lowercase().as_str(),
        "" | "horizon-small" | "multilingual-reranker-small" | "horizon-labs/multilingual-reranker-small"
    )
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
    #[test]
    fn rerank_threads_follow_the_machine_between_two_and_four() {
        for (cores, threads) in [(1, 2), (2, 2), (4, 2), (6, 3), (8, 4), (12, 4), (32, 4)] {
            assert_eq!(super::default_rerank_threads(cores), threads, "{cores} cores");
        }
    }

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
    fn reranker_scores_and_footprint() {
        let before = rss_mb();
        let mut model = OrtReranker::load(HORIZON_SMALL_INT8).expect("reranker load");
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
            "[reranker probe] load +{:.0} MB, after 5 short reranks +{:.0} MB, after 3 long reranks +{:.0} MB ({} ms each, total {:.0} MB)",
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
    fn only_the_shipped_reranker_is_a_known_name() {
        assert!(names_the_reranker(""));
        assert!(names_the_reranker(" Horizon-Small "));
        assert!(names_the_reranker("Horizon-Labs/multilingual-reranker-small"));
        assert!(!names_the_reranker("jina-v2"));
        assert!(!names_the_reranker("mmarco"));
    }

    /// A relevant document in another language than the query still comes
    /// first. Needs the model files (network on first run).
    #[test]
    fn reranker_ranks_the_relevant_document_first_across_languages() {
        let mut model = OrtReranker::load(HORIZON_SMALL_INT8).expect("reranker load");
        let documents = vec![
            "L'équipe est partie en randonnée en montagne le week-end dernier.".to_string(),
            "Le projet stocke ses données dans une base PostgreSQL.".to_string(),
            "Le déjeuner est servi à midi à la cafétéria.".to_string(),
        ];
        for query in ["which database does the project use", "quelle base de données utilise le projet"] {
            let scores = model.scores(query, &documents).expect("scores");
            assert!(scores[1] > scores[0] && scores[1] > scores[2], "{}: {:?}", query, scores);
        }
    }

    #[test]
    fn rerank_width_follows_the_setting_unless_wide() {
        assert_eq!(rerank_width(None, false), 12);
        assert_eq!(rerank_width(Some("24"), false), 24);
        assert_eq!(rerank_width(Some("500"), false), 64);
        assert_eq!(rerank_width(Some("0"), false), 2);
        assert_eq!(rerank_width(Some("abc"), false), 12);
        assert_eq!(rerank_width(Some("8"), true), 64);
        assert!(!WIDE_SCOPE.with(|c| c.get()));
        with_wide_rerank(|| {
            assert!(WIDE_SCOPE.with(|c| c.get()) && forced_scope());
        });
        assert!(!WIDE_SCOPE.with(|c| c.get()) && !forced_scope());
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

    #[test]
    fn cross_weight_default_and_bounds() {
        assert_eq!(cross_weight_from(None), 1.0);
        assert_eq!(cross_weight_from(None), DEFAULT_CROSS_WEIGHT);
        assert_eq!(cross_weight_from(Some("")), DEFAULT_CROSS_WEIGHT);
        assert_eq!(cross_weight_from(Some("abc")), DEFAULT_CROSS_WEIGHT);
        assert_eq!(cross_weight_from(Some("NaN")), DEFAULT_CROSS_WEIGHT);
        assert_eq!(cross_weight_from(Some("inf")), DEFAULT_CROSS_WEIGHT);
        assert_eq!(cross_weight_from(Some(" 0.6 ")), 0.6);
        assert_eq!(cross_weight_from(Some("0")), 0.0);
        assert_eq!(cross_weight_from(Some("1")), 1.0);
        assert_eq!(cross_weight_from(Some("-0.3")), 0.0);
        assert_eq!(cross_weight_from(Some("1.7")), 1.0);
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
