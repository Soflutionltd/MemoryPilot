//! Conversation index: finds a past conversation (Cursor, Claude Code,
//! Claude desktop agent sessions) from a vague, spoken description, and
//! the commits, files and projects it touched.
//!
//! Sources are read in place and read-only. Messages are indexed as
//! said; nothing is summarised into memories, which is what polluted
//! recall when transcripts were distilled automatically.

pub mod claude;
pub mod cursor;
pub mod search;
pub mod store;
pub mod text;

pub use search::{ConversationHit, ConversationMessage, SearchOptions};
pub use store::ConversationStore;

use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
}

#[derive(Debug, Clone)]
pub struct Message {
    pub key: String,
    pub role: Role,
    pub at: Option<String>,
    pub order: Option<usize>,
    pub text: String,
}

/// One conversation as a source describes it. `origin_id` is the unit
/// the source stores (a Cursor composer, a Claude session file);
/// `root_id` is the conversation the user had, which differs for Cursor
/// sub-agents.
#[derive(Debug, Clone)]
pub struct SourceConversation {
    pub origin_id: String,
    pub root_id: String,
    pub tool: &'static str,
    pub title: Option<String>,
    pub workspace: Option<String>,
    pub created_at: Option<String>,
    pub version: String,
    pub source_path: Option<String>,
}

/// Normalises a source timestamp to RFC 3339 UTC with milliseconds, so
/// that string order is time order.
pub fn fold_timestamp(raw: &str) -> Option<String> {
    chrono::DateTime::parse_from_rfc3339(raw)
        .ok()
        .map(|at| at.with_timezone(&chrono::Utc).to_rfc3339_opts(chrono::SecondsFormat::Millis, true))
}

#[derive(Debug, Default, serde::Serialize)]
pub struct SyncReport {
    pub origins_checked: usize,
    pub origins_updated: usize,
    pub messages_written: usize,
    pub errors: Vec<String>,
    pub timed_out: bool,
    pub elapsed_ms: u128,
}

pub struct SyncOptions {
    pub budget: Option<Duration>,
}

/// Brings the index up to date with every source. Unchanged origins cost
/// one version comparison; changed ones only read their new messages.
pub fn sync(store: &mut ConversationStore, options: &SyncOptions) -> SyncReport {
    let started = Instant::now();
    let mut report = SyncReport::default();
    let over_budget = |report: &mut SyncReport| -> bool {
        let over = options.budget.is_some_and(|budget| started.elapsed() > budget);
        report.timed_out |= over;
        over
    };

    if let Some(path) = cursor::default_store_path() {
        if let Err(error) = sync_cursor(store, &path, &mut report, &over_budget) {
            report.errors.push(error);
        }
    }
    if !over_budget(&mut report) {
        sync_claude(store, &mut report, &over_budget);
    }
    report.elapsed_ms = started.elapsed().as_millis();
    report
}

fn hash_key(key: &str) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in key.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn sync_cursor(
    store: &mut ConversationStore,
    path: &std::path::Path,
    report: &mut SyncReport,
    over_budget: &dyn Fn(&mut SyncReport) -> bool,
) -> Result<(), String> {
    let conn = cursor::open(path)?;
    let mut composers = cursor::list_composers(&conn)?;
    // Main conversations first, so sub-agents attach to a titled row.
    composers.sort_by_key(|c| c.conversation.origin_id != c.conversation.root_id);
    for composer in &composers {
        if over_budget(report) {
            return Ok(());
        }
        report.origins_checked += 1;
        let source = &composer.conversation;
        if store.origin_version(&source.origin_id)?.as_deref() == Some(source.version.as_str()) {
            continue;
        }
        let ids = cursor::bubble_ids(&conn, &source.origin_id)?;
        let mut seen = store.seen_hashes(&source.origin_id)?;
        seen.sort_unstable();
        let fresh: Vec<String> = ids
            .iter()
            .filter(|id| seen.binary_search(&hash_key(id)).is_err())
            .cloned()
            .collect();
        let messages = cursor::read_messages(&conn, composer, &fresh)?;
        // The last turn may still be streaming: its messages stay unseen
        // so the next sync reads their final text. Without a request in
        // the batch, the whole batch continues the stored last turn.
        let turn_start = messages.iter().rposition(|m| m.role == Role::User).unwrap_or(0);
        let open_turn: Vec<u64> = messages[turn_start..].iter().map(|m| hash_key(&m.key)).collect();
        let mut all_seen: Vec<u64> = ids
            .iter()
            .map(|id| hash_key(id))
            .filter(|hash| !open_turn.contains(hash))
            .collect();
        all_seen.sort_unstable();
        all_seen.dedup();
        let is_main = source.origin_id == source.root_id;
        report.messages_written += store.apply(source, is_main, &messages, &all_seen)?;
        report.origins_updated += 1;
    }
    Ok(())
}

fn sync_claude(
    store: &mut ConversationStore,
    report: &mut SyncReport,
    over_budget: &dyn Fn(&mut SyncReport) -> bool,
) {
    let roots = claude::default_roots();
    if roots.is_empty() {
        return;
    }
    let titles = claude::desktop_titles(&roots);
    for file in claude::list_sessions(&roots) {
        if over_budget(report) {
            return;
        }
        report.origins_checked += 1;
        let session = match claude::read_session(&file, &titles) {
            Ok(session) => session,
            Err(error) => {
                report.errors.push(error);
                continue;
            }
        };
        let source = &session.conversation;
        match store.origin_version(&source.origin_id) {
            Ok(Some(version)) if version == source.version => continue,
            Ok(_) => {}
            Err(error) => {
                report.errors.push(error);
                continue;
            }
        }
        let mut seen: Vec<u64> = session.messages.iter().map(|m| hash_key(&m.key)).collect();
        seen.sort_unstable();
        match store.apply(source, true, &session.messages, &seen) {
            Ok(written) => {
                report.messages_written += written;
                report.origins_updated += 1;
            }
            Err(error) => report.errors.push(error),
        }
    }
}

/// Embeds pending messages in batches until none are left or the budget
/// runs out. Returns how many were embedded.
pub fn embed_pending(store: &mut ConversationStore, budget: Option<Duration>) -> Result<usize, String> {
    let started = Instant::now();
    let mut done = 0;
    loop {
        if budget.is_some_and(|b| started.elapsed() > b) {
            return Ok(done);
        }
        let batch = store.pending_embeddings(32)?;
        if batch.is_empty() {
            return Ok(done);
        }
        let texts: Vec<&str> = batch.iter().map(|(_, text)| text.as_str()).collect();
        let vectors = crate::embedding::embed_batch(&texts);
        if vectors.len() != batch.len() {
            return Err(format!("Embedding batch returned {} vectors for {} texts", vectors.len(), batch.len()));
        }
        let rows: Vec<(i64, Vec<u8>)> = batch
            .iter()
            .zip(vectors.iter())
            .map(|((id, _), vector)| (*id, crate::embedding::quantize_to_blob(vector)))
            .collect();
        store.store_embeddings(&rows)?;
        done += rows.len();
    }
}
