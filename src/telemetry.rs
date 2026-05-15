//! Structured retrieval telemetry.
//!
//! When the environment variable `MEMORYPILOT_TELEMETRY` is set to `1`
//! or `stderr`, each `Database::search()` call emits a single line of
//! JSON describing the query, the candidate counts from each retrieval
//! source, the timing breakdown and the top result. The output is
//! safe to pipe directly into Datadog, Loki, Vector or any JSONL log
//! pipeline.
//!
//! Setting `MEMORYPILOT_TELEMETRY=<path>` writes the same JSONL records
//! to that file (append mode). When the variable is unset, telemetry
//! is a no-op and the cost is a single OnceLock load.
//!
//! The implementation is deliberately allocation-light when telemetry
//! is disabled: every call site builds a `RetrievalTrace`, but the
//! serialization / write only runs when telemetry is enabled.

use std::fs::OpenOptions;
use std::io::Write;
use std::sync::{Mutex, OnceLock};

use serde::Serialize;
use serde_json::json;

#[derive(Debug, Clone, Default, Serialize)]
pub struct RetrievalTrace {
    pub query_truncated: String,
    pub query_chars: usize,
    pub project: Option<String>,
    pub kind: Option<String>,
    pub tags_count: usize,
    pub limit: usize,

    pub candidates_bm25: usize,
    pub candidates_vector: usize,
    pub candidates_ann: usize,
    pub candidates_total_unique: usize,

    pub fts_variants: usize,
    pub kg_expansion_terms: usize,

    pub timing_ms_total: f64,
    pub timing_ms_embed_query: f64,
    pub timing_ms_bm25: f64,
    pub timing_ms_vector: f64,
    pub timing_ms_fusion: f64,

    pub results_returned: usize,
    pub top_score: f64,
    pub top_sources: Vec<String>,
}

#[derive(Clone, Debug)]
enum Sink {
    Disabled,
    Stderr,
    File(std::path::PathBuf),
}

static SINK: OnceLock<Sink> = OnceLock::new();
static FILE_HANDLE: OnceLock<Mutex<std::fs::File>> = OnceLock::new();

fn sink() -> &'static Sink {
    SINK.get_or_init(|| match std::env::var("MEMORYPILOT_TELEMETRY") {
        Ok(value) => {
            let trimmed = value.trim();
            if trimmed.is_empty() || trimmed == "0" || trimmed.eq_ignore_ascii_case("off") {
                Sink::Disabled
            } else if trimmed == "1" || trimmed.eq_ignore_ascii_case("stderr") {
                Sink::Stderr
            } else {
                Sink::File(std::path::PathBuf::from(trimmed))
            }
        }
        Err(_) => Sink::Disabled,
    })
}

pub fn is_enabled() -> bool {
    !matches!(sink(), Sink::Disabled)
}

pub fn emit(trace: &RetrievalTrace) {
    match sink() {
        Sink::Disabled => {}
        Sink::Stderr => {
            if let Ok(line) = serde_json::to_string(&json!({
                "kind": "memorypilot.retrieval",
                "ts_ms": chrono::Utc::now().timestamp_millis(),
                "trace": trace,
            })) {
                let _ = writeln!(std::io::stderr(), "{}", line);
            }
        }
        Sink::File(path) => {
            let handle = FILE_HANDLE.get_or_init(|| {
                let file = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)
                    .unwrap_or_else(|_| {
                        // Fall back to a temp file rather than crashing the
                        // server if the configured path is unwritable.
                        OpenOptions::new()
                            .create(true)
                            .append(true)
                            .open(std::env::temp_dir().join("memorypilot-telemetry.jsonl"))
                            .expect("telemetry temp file fallback")
                    });
                Mutex::new(file)
            });
            if let Ok(line) = serde_json::to_string(&json!({
                "kind": "memorypilot.retrieval",
                "ts_ms": chrono::Utc::now().timestamp_millis(),
                "trace": trace,
            })) {
                if let Ok(mut file) = handle.lock() {
                    let _ = writeln!(file, "{}", line);
                }
            }
        }
    }
}

pub fn truncate_query(query: &str) -> String {
    let max = 200;
    if query.chars().count() <= max {
        query.to_string()
    } else {
        query.chars().take(max).collect::<String>() + "…"
    }
}
