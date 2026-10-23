//! Extractive reader: a span-prediction head on top of the retriever.
//!
//! Given a question and the top passages, returns the shortest text span
//! that answers it — or nothing, when every passage scores below the
//! model's own "no answer" logit. This is *not* a language model: it
//! cannot count across passages, do arithmetic or paraphrase. It exists
//! so the retriever's quality can be measured end-to-end (question →
//! answer string) with the same local-only constraints as the rest of
//! the daemon, and so an agent can ask for the literal value instead of
//! re-reading five passages.
//!
//! Model: `deepset/roberta-base-squad2` (SQuAD 2.0, F1 ≈ 83 / EM ≈ 80 on
//! the dev set), served as the int8 ONNX export from
//! `onnx-community/roberta-base-squad2-ONNX` — 125 MB on disk, English.
//! Loaded on first use through the same idle pool as the cross-encoder,
//! so it costs nothing while unused.

use std::sync::OnceLock;

use crate::tokenizer::MergeBpe;

pub const READER_REPO: &str = "onnx-community/roberta-base-squad2-ONNX";
const READER_ONNX: &str = "onnx/model_quantized.onnx";

/// RoBERTa positional limit (514 positions, 2 reserved).
const MAX_SEQ: usize = 512;
const MAX_QUESTION_TOKENS: usize = 64;
/// Overlap between consecutive windows over a long passage.
const WINDOW_STRIDE: usize = 128;
/// Windows per passage; beyond this the tail of a very long passage is
/// not read. 3 × ~440 context tokens covers ~1 300 tokens.
const MAX_WINDOWS_PER_PASSAGE: usize = 3;
const MAX_ANSWER_TOKENS: usize = 30;
/// Padded tokens per ONNX run (rows × longest row).
const TOKEN_BUDGET: usize = 4096;

#[derive(Debug, Clone, serde::Serialize)]
pub struct ReaderAnswer {
    pub text: String,
    /// `span_logit − null_logit` of the winning window: the model's own
    /// margin for "there is an answer here". Higher is more certain.
    pub score: f64,
    pub span_score: f64,
    pub null_score: f64,
    pub passage_index: usize,
}

/// `MEMORYPILOT_READER_NULL_MARGIN` — the span must beat the no-answer
/// logit by at least this much. 0 is the SQuAD 2.0 default operating
/// point; raise it to trade recall for precision.
fn null_margin() -> f64 {
    std::env::var("MEMORYPILOT_READER_NULL_MARGIN")
        .ok()
        .and_then(|raw| raw.trim().parse::<f64>().ok())
        .unwrap_or(0.0)
}

/// Best answer span across `passages`, or `None` when the model finds
/// no answer in any of them. `Err` when the model cannot be loaded.
pub fn answer(question: &str, passages: &[&str]) -> Result<Option<ReaderAnswer>, String> {
    if passages.is_empty() || question.trim().is_empty() {
        return Ok(None);
    }
    let mut handle = reader_pool().acquire();
    match handle.get() {
        ReaderState::Ready(model) => model.answer(question, passages),
        ReaderState::Unavailable(error) => Err(error.clone()),
    }
}

/// Pre-load the reader so a benchmark's first question does not pay
/// the download + ONNX hydration.
pub fn warmup() -> Result<(), String> {
    let mut handle = reader_pool().acquire();
    match handle.get() {
        ReaderState::Ready(model) => model.answer("warmup", &["warmup passage"]).map(|_| ()),
        ReaderState::Unavailable(error) => Err(error.clone()),
    }
}

enum ReaderState {
    Ready(Reader),
    Unavailable(String),
}

static READER_POOL: OnceLock<crate::pool::IdlePool<ReaderState>> = OnceLock::new();

fn reader_pool() -> &'static crate::pool::IdlePool<ReaderState> {
    let pool = READER_POOL.get_or_init(|| crate::pool::IdlePool::new("reader", 1, init_reader));
    static EVICTOR: OnceLock<()> = OnceLock::new();
    EVICTOR.get_or_init(|| pool.spawn_evictor());
    pool
}

fn init_reader() -> ReaderState {
    match Reader::load() {
        Ok(reader) => ReaderState::Ready(reader),
        Err(error) => {
            eprintln!("[reader] unavailable: {}", error);
            ReaderState::Unavailable(error)
        }
    }
}

struct Reader {
    session: ort::session::Session,
    tokenizer: MergeBpe,
    run_options: ort::session::RunOptions,
    needs_token_type_ids: bool,
}

/// One `<s> q </s></s> ctx </s>` sequence and what it covers.
struct Window {
    ids: Vec<u32>,
    /// Token positions [context_start, context_end) inside `ids`.
    context_start: usize,
    context_end: usize,
    /// Byte range of the passage covered by each context token.
    offsets: Vec<(usize, usize)>,
    passage_index: usize,
}

impl Reader {
    fn load() -> Result<Self, String> {
        let graph_path = crate::onnx_external::externalized(READER_REPO, READER_ONNX, false)?;
        let tokenizer_path = crate::embedding::hf_file(READER_REPO, "tokenizer.json", false)?;
        let tokenizer = MergeBpe::from_file(&tokenizer_path)
            .map_err(|error| format!("reader tokenizer load: {}", error))?;

        use ort::session::builder::GraphOptimizationLevel;
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(2)
            .clamp(1, 4);
        let session = ort::session::Session::builder()
            .map_err(|error| format!("ort builder: {}", error))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|error| format!("ort optimization level: {}", error))?
            .with_intra_threads(threads)
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

    fn windows(&self, question: &str, passages: &[&str]) -> Vec<Window> {
        let mut question_ids = self.tokenizer.encode(question.trim());
        question_ids.truncate(MAX_QUESTION_TOKENS);
        // <s> q </s></s> … </s>
        let overhead = question_ids.len() + 4;
        let capacity = MAX_SEQ.saturating_sub(overhead).max(16);

        let mut windows = Vec::new();
        for (passage_index, passage) in passages.iter().enumerate() {
            let (context_ids, offsets) = self.tokenizer.encode_with_offsets(passage);
            if context_ids.is_empty() {
                continue;
            }
            let mut start = 0usize;
            let mut produced = 0usize;
            while start < context_ids.len() && produced < MAX_WINDOWS_PER_PASSAGE {
                let end = (start + capacity).min(context_ids.len());
                let mut ids = Vec::with_capacity(overhead + (end - start));
                ids.push(self.tokenizer.bos_id());
                ids.extend_from_slice(&question_ids);
                ids.push(self.tokenizer.eos_id());
                ids.push(self.tokenizer.eos_id());
                let context_start = ids.len();
                ids.extend_from_slice(&context_ids[start..end]);
                let context_end = ids.len();
                ids.push(self.tokenizer.eos_id());
                windows.push(Window {
                    ids,
                    context_start,
                    context_end,
                    offsets: offsets[start..end].to_vec(),
                    passage_index,
                });
                produced += 1;
                if end == context_ids.len() {
                    break;
                }
                start = end.saturating_sub(WINDOW_STRIDE).max(start + 1);
            }
        }
        windows
    }

    fn answer(&mut self, question: &str, passages: &[&str]) -> Result<Option<ReaderAnswer>, String> {
        let windows = self.windows(question, passages);
        if windows.is_empty() {
            return Ok(None);
        }

        // Pack rows by descending length so padding stays small.
        let mut order: Vec<usize> = (0..windows.len()).collect();
        order.sort_by_key(|&index| std::cmp::Reverse(windows[index].ids.len()));

        let mut best: Option<ReaderAnswer> = None;
        let mut start = 0;
        while start < order.len() {
            let width = windows[order[start]].ids.len().max(1);
            let rows = (TOKEN_BUDGET / width).clamp(1, 16);
            let end = (start + rows).min(order.len());
            let pack = &order[start..end];
            let (start_logits, end_logits) = self.run_pack(&windows, pack, width)?;
            for (row, &window_index) in pack.iter().enumerate() {
                let window = &windows[window_index];
                let starts = &start_logits[row * width..(row + 1) * width];
                let ends = &end_logits[row * width..(row + 1) * width];
                if let Some(candidate) = best_span(window, passages[window.passage_index], starts, ends)
                {
                    let replace = best
                        .as_ref()
                        .map(|current| candidate.score > current.score)
                        .unwrap_or(true);
                    if replace {
                        best = Some(candidate);
                    }
                }
            }
            start = end;
        }

        Ok(best.filter(|answer| answer.score >= null_margin()))
    }

    fn run_pack(
        &mut self,
        windows: &[Window],
        pack: &[usize],
        width: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let batch = pack.len();
        let pad_id = self.tokenizer.pad_id() as i64;
        let mut ids = Vec::with_capacity(batch * width);
        let mut mask = Vec::with_capacity(batch * width);
        for &index in pack {
            let row = &windows[index].ids;
            let len = row.len().min(width);
            ids.extend(row[..len].iter().map(|&id| id as i64));
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
        let extract = |name: &str| -> Result<Vec<f32>, String> {
            let (shape, data) = outputs[name]
                .try_extract_tensor::<f32>()
                .map_err(|error| format!("{} extract: {}", name, error))?;
            let dims: Vec<i64> = shape.iter().copied().collect();
            if dims != [batch as i64, width as i64] {
                return Err(format!("{} shape {:?}, expected [{}, {}]", name, dims, batch, width));
            }
            Ok(data.to_vec())
        };
        Ok((extract("start_logits")?, extract("end_logits")?))
    }
}

/// Highest `start + end` span inside the context region of one window,
/// at most [`MAX_ANSWER_TOKENS`] long, scored against the window's
/// no-answer logit (position 0).
fn best_span(window: &Window, passage: &str, starts: &[f32], ends: &[f32]) -> Option<ReaderAnswer> {
    let null_score = (starts[0] + ends[0]) as f64;
    let mut best_value = f32::NEG_INFINITY;
    let mut best_pair: Option<(usize, usize)> = None;
    for start in window.context_start..window.context_end {
        let end_limit = (start + MAX_ANSWER_TOKENS).min(window.context_end);
        for end in start..end_limit {
            let value = starts[start] + ends[end];
            if value > best_value {
                best_value = value;
                best_pair = Some((start, end));
            }
        }
    }
    let (start, end) = best_pair?;
    let first = window.offsets[start - window.context_start];
    let last = window.offsets[end - window.context_start];
    let text = passage
        .get(first.0..last.1)
        .map(|raw| clean_span(raw))
        .filter(|cleaned| !cleaned.is_empty())?;
    Some(ReaderAnswer {
        text,
        score: best_value as f64 - null_score,
        span_score: best_value as f64,
        null_score,
        passage_index: window.passage_index,
    })
}

/// Trim whitespace and dangling punctuation off a predicted span.
fn clean_span(raw: &str) -> String {
    raw.trim()
        .trim_matches(|character: char| matches!(character, ',' | ';' | ':' | '.' | '!' | '?' | '"' | '\'' | '(' | ')' | '[' | ']'))
        .trim()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// End-to-end on the real model; skipped when the model cannot be
    /// fetched (offline CI).
    #[test]
    fn extracts_a_span_and_abstains_on_unrelated_text() {
        if std::env::var("MEMORYPILOT_READER_TESTS").is_err() {
            eprintln!("set MEMORYPILOT_READER_TESTS=1 to run the reader test");
            return;
        }
        let passages = [
            "user: I finally graduated last spring with a degree in Business Administration from a small college upstate.",
            "assistant: Congratulations on finishing your marathon training plan!",
        ];
        let found = answer("What degree did I graduate with?", &passages)
            .expect("model")
            .expect("an answer");
        assert!(
            found.text.to_lowercase().contains("business administration"),
            "got {:?}",
            found.text
        );
        assert_eq!(found.passage_index, 0);

        let none = answer("What is the capital of Mongolia?", &passages).expect("model");
        assert!(none.is_none(), "unrelated passages must yield no answer, got {:?}", none);
    }
}
