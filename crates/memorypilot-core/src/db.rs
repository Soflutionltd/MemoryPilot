use chrono::Utc;
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
/// MemoryPilot v4.0 Database Engine — SQLite + FTS5.
/// Features: dedup, importance, TTL, bulk ops, export, auto-prompt, lazy embedding, content hash.
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use uuid::Uuid;

#[path = "db/benchmark.rs"]
mod benchmark;
#[path = "db/benchmark_fr.rs"]
mod benchmark_fr;
#[path = "db/benchmark_episodic.rs"]
mod benchmark_episodic;
#[path = "db/benchmark_longmemeval.rs"]
mod benchmark_longmemeval;
#[path = "db/compaction.rs"]
mod compaction;
#[path = "db/consolidate.rs"]
mod consolidate;
#[path = "db/embed_worker.rs"]
mod embed_worker;
#[path = "db/episodic.rs"]
mod episodic;
#[path = "db/export.rs"]
mod export;
#[path = "db/schema.rs"]
mod schema;
#[path = "db/skills.rs"]
mod skills;
#[path = "db/profile.rs"]
mod profile;
#[path = "db/vault.rs"]
mod vault;
#[path = "db/conflicts.rs"]
mod conflicts;
#[path = "db/supersede.rs"]
mod supersede;
#[path = "db/verification.rs"]
mod verification;

/// Whether sealed credentials are decrypted on read (default no:
/// `MEMORYPILOT_SECRETS=reveal`, or `MEMORYPILOT_HTTP_SECRETS=on` over HTTP).
pub fn set_secret_reveal(on: bool) {
    vault::set_reveal(on);
}

/// The HTTP server: reveal_secret is refused unless reveal is on.
pub fn set_remote_server(on: bool) {
    vault::set_remote(on);
}

pub use vault::RevealConfirm;

/// Put back a vault key (base64) after checking it opens every sealed credential.
#[cfg(target_os = "macos")]
pub fn restore_vault_key(encoded: &str, db_path: &std::path::Path) -> Result<(usize, usize), String> {
    vault::restore_key(encoded, db_path)
}

/// Seal the credentials of an old backup file in place (no schema change).
pub fn seal_backup_file(path: &std::path::Path) -> Result<usize, String> {
    vault::seal_file(path)
}

pub(crate) use vault::contains_secret_value;
#[path = "db/transcript.rs"]
mod transcript;
#[path = "db/lifecycle.rs"]
mod lifecycle;
#[path = "db/projects.rs"]
mod projects;
#[path = "db/scope.rs"]
mod scope;
#[path = "db/ranking.rs"]
mod ranking;
#[path = "db/setup.rs"]
mod setup;
#[path = "db/memories.rs"]
mod memories;
#[path = "db/knowledge.rs"]
mod knowledge;
#[path = "db/search.rs"]
mod search;
#[path = "db/maintenance.rs"]
mod maintenance;
#[path = "db/recall.rs"]
mod recall;
#[path = "db/store.rs"]
mod store;
#[path = "db/sqlite_store.rs"]
mod sqlite_store;
pub use store::MemoryStore;

use embed_worker::{
    cached_embed_text, embed_db_path, queue_access_update, queue_embedding_job,
    queue_startup_backfill, set_embed_ann_index, set_embed_db_path,
};

#[allow(unused_imports)]
pub use episodic::{Episode, EpisodeHit, RollupReport};
pub use consolidate::AUTO_THRESHOLD as AUTO_CONSOLIDATE_THRESHOLD;
pub use benchmark_longmemeval::LongMemEvalOptions;

const DB_DIR: &str = ".MemoryPilot";
const DB_FILE: &str = "memory.db";
const PROMPT_FILE: &str = "GLOBAL_PROMPT.md";
const DEDUP_THRESHOLD: f64 = 0.85;
/// `config` key holding the label of the encoder that produced the
/// stored vectors. See `Database::sync_embedding_model`.
const EMBEDDING_MODEL_KEY: &str = "embedding_model";
/// Startup backfills larger than this run in the background worker.
const SYNC_BACKFILL_MAX: usize = 32;

pub(crate) fn content_hash(text: &str) -> String {
    let mut h: u64 = 14695981039346656037;
    for b in text.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(1099511628211);
    }
    format!("{:016x}", h)
}

/// Kinds whose text is a slice of something larger and carries no
/// context of its own — a transcript chunk does not say which project
/// or which day it belongs to, so the vector cannot either.
const CONTEXT_POOR_KINDS: &[&str] = &["transcript", "transcript_chunk", "session"];

impl Database {
    /// The text handed to the encoder for a memory. Context-poor kinds
    /// are prefixed with `[project; date]` — the same idea as
    /// contextual retrieval, sourced from metadata instead of a model —
    /// so "what did we do on Sociomator in March" lands on the right
    /// chunk. Every other kind is embedded verbatim; the stored content
    /// and its hash are never touched.
    pub(crate) fn embedding_input(
        kind: &str,
        project: Option<&str>,
        created_at: &str,
        content: &str,
    ) -> String {
        if !CONTEXT_POOR_KINDS.contains(&kind) || content.trim_start().starts_with('[') {
            return content.to_string();
        }
        let mut header: Vec<String> = Vec::with_capacity(2);
        if let Some(project) = project.filter(|p| !p.trim().is_empty()) {
            header.push(format!("project: {}", project.trim()));
        }
        if let Some(day) = crate::temporal::day_of_timestamp(created_at) {
            header.push(format!("date: {}", day.format("%Y-%m-%d")));
        }
        if header.is_empty() {
            return content.to_string();
        }
        format!("[{}] {}", header.join("; "), content)
    }
}

#[derive(Debug, Clone, Default)]
struct QueryIntent {
    preference: bool,
    temporal: bool,
    user_turn: bool,
    assistant_turn: bool,
    update_or_correction: bool,
    technical: bool,
}

/// Serialized by hand (db/vault.rs): credentials go out masked, with a pointer.
#[derive(Debug, Clone, Deserialize)]
pub struct Memory {
    pub id: String,
    pub content: String,
    pub kind: String,
    pub project: Option<String>,
    pub tags: Vec<String>,
    pub source: String,
    pub importance: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<String>,
    pub created_at: String,
    pub updated_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_accessed_at: Option<String>,
    pub access_count: i32,
}
#[derive(Debug, Clone, Serialize)]
pub struct SearchResult {
    pub memory: Memory,
    pub score: f64,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub sources: Vec<String>,
}

/// How much the ranker trusts its own top hit. Built from signals that
/// exist for every query — the raw cosine between the query and the
/// top-1 vector, the relative gap to the runner-up — plus the raw
/// cross-encoder logit when that lane ran. `abstain` is the calibrated
/// verdict: nothing in the store answers this query, callers should say
/// so rather than present the least-bad candidate.
#[derive(Debug, Clone, Serialize, Default)]
pub struct SearchConfidence {
    pub top_cosine: f32,
    /// How far the top-1 cosine stands above the mean of the other
    /// finalists: a flat profile means nothing really matched.
    pub peak: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cross_score: Option<f32>,
    pub margin: f64,
    pub level: &'static str,
    pub abstain: bool,
}

impl SearchConfidence {
    /// `MEMORYPILOT_ABSTAIN_COSINE` — below this query·top-1 cosine the
    /// search abstains unless the cross-encoder vouches for the hit.
    ///
    /// Calibrated on LongMemEval-S (100 answerable + 7 abstention): the
    /// answerable top-1 cosine sits at p10 = 0.42, p50 = 0.54, and the
    /// abstention questions — built from the *same user's* other sessions —
    /// overlap it almost entirely (p50 = 0.46). No cheap signal separates
    /// them, so the hard `abstain` is deliberately conservative (0.35 loses
    /// 4 % of answerable questions) and is meant for genuinely off-topic
    /// queries; `level` carries the graded signal for everything else.
    pub fn abstain_cosine() -> f32 {
        std::env::var("MEMORYPILOT_ABSTAIN_COSINE")
            .ok()
            .and_then(|raw| raw.trim().parse::<f32>().ok())
            .unwrap_or(0.35)
    }

    /// Cross-encoder logit above which the hit is trusted whatever the
    /// cosine says (mmarco logits are positive for relevant pairs).
    pub fn trusted_cross_score() -> f32 {
        std::env::var("MEMORYPILOT_TRUST_CROSS")
            .ok()
            .and_then(|raw| raw.trim().parse::<f32>().ok())
            .unwrap_or(1.0)
    }

    fn none() -> Self {
        Self {
            level: "none",
            abstain: true,
            ..Self::default()
        }
    }

    fn assess(top_cosine: f32, peak: f32, cross_score: Option<f32>, margin: f64) -> Self {
        let floor = Self::abstain_cosine();
        let cross_vouches = cross_score
            .map(|score| score >= Self::trusted_cross_score())
            .unwrap_or(false);
        // mmarco logits below −4 on the top hit: 1 in 8 answerable LME
        // questions still land there, so this only bites with a weak cosine.
        let cross_denies = cross_score.map(|score| score < -4.0).unwrap_or(false);
        let abstain = (top_cosine < floor && !cross_vouches) || (cross_denies && top_cosine < floor + 0.05);
        let level = if abstain {
            "low"
        } else if top_cosine >= floor + 0.2 || cross_vouches || margin >= 0.25 {
            "high"
        } else {
            "medium"
        };
        Self {
            top_cosine,
            peak,
            cross_score,
            margin,
            level,
            abstain,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Project {
    pub name: String,
    pub path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub created_at: String,
    pub memory_count: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecallMode {
    Safe,
    Default,
    Full,
}

impl RecallMode {
    pub fn parse(value: Option<&str>) -> Result<Self, String> {
        match value.unwrap_or("safe").trim().to_ascii_lowercase().as_str() {
            "safe" => Ok(Self::Safe),
            "default" => Ok(Self::Default),
            "full" => Ok(Self::Full),
            other => Err(format!(
                "Invalid recall mode '{}'. Use safe, default, or full.",
                other
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Safe => "safe",
            Self::Default => "default",
            Self::Full => "full",
        }
    }

    pub fn includes_credentials(self) -> bool {
        matches!(self, Self::Full)
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryScope {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thread_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub window_id: Option<String>,
}

impl MemoryScope {
    pub fn is_empty(&self) -> bool {
        self.session_id.is_none() && self.thread_id.is_none() && self.window_id.is_none()
    }
}

/// Number of read-only SQLite connections per Database handle. Each handle
/// (one per HTTP worker / MCP client) keeps this many readers warm so that
/// concurrent searches no longer serialize on a single Mutex<Connection>.
/// SQLite WAL allows N readers + 1 writer concurrently, so a larger pool
/// only costs file descriptors and ~80 KB of cache per connection.
const READ_POOL_SIZE: usize = 16;

pub struct Database {
    conn: Connection,
    read_pool: Vec<Mutex<Connection>>,
    ann: Option<Arc<crate::ann::AnnIndex>>,
    /// Set to `true` once the detached `spawn_ann_warmup` thread has
    /// finished hydrating the in-memory ANN index from SQLite. Lets
    /// callers opt into a deterministic search path via
    /// [`Database::wait_for_ann_warm`] / [`Database::open_at_warm`]
    /// without paying that cost on the default `open_at`.
    ann_warm_complete: Arc<std::sync::atomic::AtomicBool>,
}

/// RAII helper used by the ANN warm-up thread: ensures the
/// completion flag is published exactly once, no matter which exit
/// path the worker takes (success, prepare failure, query failure).
struct WarmupGuard {
    flag: Arc<std::sync::atomic::AtomicBool>,
}

impl Drop for WarmupGuard {
    fn drop(&mut self) {
        self.flag
            .store(true, std::sync::atomic::Ordering::Release);
    }
}

impl Database {

    // ─── EXPORT ───────────────────────────────────────

    pub fn export_memories(&self, project: Option<&str>, format: &str) -> Result<String, String> {
        export::export_memories(self, project, format)
    }

    pub fn export_session_markdown(
        &self,
        session_id: Option<&str>,
        thread_id: Option<&str>,
        window_id: Option<&str>,
        project: Option<&str>,
    ) -> Result<String, String> {
        export::export_session_markdown(self, session_id, thread_id, window_id, project)
    }
} // end impl Database

// ─── Supporting types ─────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct TranscriptAddReport {
    pub transcript_id: String,
    pub chunks_total: usize,
    pub chunk_added: usize,
    pub chunk_merged: usize,
    pub chunk_skipped: usize,
    pub distilled_candidates: usize,
    pub distilled_added: usize,
    pub distilled_merged: usize,
    pub distilled_skipped: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BulkItem {
    pub content: String,
    #[serde(default = "default_kind")]
    pub kind: String,
    pub project: Option<String>,
    pub tags: Option<Vec<String>>,
    #[serde(default = "default_source")]
    pub source: String,
    pub importance: Option<i32>,
    pub expires_at: Option<String>,
    pub metadata: Option<serde_json::Value>,
    pub session_id: Option<String>,
    pub thread_id: Option<String>,
    pub window_id: Option<String>,
}
fn default_kind() -> String {
    "fact".into()
}
fn default_source() -> String {
    "cursor".into()
}

impl BulkItem {
    fn scope(&self) -> MemoryScope {
        MemoryScope {
            session_id: self.session_id.clone(),
            thread_id: self.thread_id.clone(),
            window_id: self.window_id.clone(),
        }
    }
}

// ─── Row helper ───────────────────────────────────

fn row_to_memory(row: &rusqlite::Row) -> Memory {
    let tags_str: String = row.get(4).unwrap_or_default();
    let tags: Vec<String> = serde_json::from_str(&tags_str).unwrap_or_default();
    let meta_str: Option<String> = row.get(8).unwrap_or(None);
    let metadata: Option<serde_json::Value> = meta_str.and_then(|s| serde_json::from_str(&s).ok());
    let raw_content: String = row.get(1).unwrap_or_default();
    let kind: String = row.get(2).unwrap_or_default();
    // Credentials are read masked (see db/vault.rs).
    let content = vault::read_content(&kind, raw_content, metadata.as_ref());
    Memory {
        id: row.get(0).unwrap_or_default(),
        content,
        kind,
        project: row.get(3).unwrap_or(None),
        tags,
        source: row.get(5).unwrap_or_default(),
        importance: row.get(6).unwrap_or(3),
        expires_at: row.get(7).unwrap_or(None),
        metadata,
        created_at: row.get(9).unwrap_or_default(),
        updated_at: row.get(10).unwrap_or_default(),
        last_accessed_at: row.get(11).unwrap_or(None),
        access_count: row.get(12).unwrap_or(0),
    }
}

fn parse_v1_memory(
    m: &serde_json::Value,
    project: Option<String>,
    batch: &mut Vec<(String, String, Option<String>, Vec<String>, String)>,
) {
    let c = m
        .get("content")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    if c.is_empty() {
        return;
    }
    let k = m
        .get("kind")
        .or(m.get("type"))
        .and_then(|v| v.as_str())
        .unwrap_or("fact");
    let kind = match k {
        "context" => "fact",
        "architecture" => "decision",
        "component" | "workflow" => "pattern",
        o => o,
    }
    .to_string();
    let tags: Vec<String> = m
        .get("tags")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    let source = m
        .get("source")
        .and_then(|v| v.as_str())
        .unwrap_or("v1-import")
        .to_string();
    batch.push((c, kind, project, tags, source));
}

#[cfg(test)]
#[path = "db/tests.rs"]
mod tests;
