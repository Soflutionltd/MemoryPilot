//! Conversation index: finds a past conversation (Cursor, Claude Code,
//! Claude desktop agent sessions, and claude.ai or Gemini conversations an
//! importer pushes) from a vague, spoken description, and
//! the commits, files and projects it touched.
//!
//! Sources are read in place and read-only. Messages are indexed as
//! said; nothing is summarised into memories, which is what polluted
//! recall when transcripts were distilled automatically.

pub mod actions;
pub mod archive;
pub mod claude;
pub mod coverage;
pub mod cursor;
pub mod imported;
pub mod search;
pub mod store;
pub mod submissions;
pub mod text;

pub use search::{ConversationHit, ConversationMessage, SearchOptions};
pub use store::ConversationStore;

use std::time::{Duration, Instant};
#[cfg(test)]
use std::collections::HashMap;

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
    /// What the agent did in this message, one line per tool call that
    /// reaches outside or changes something (see `actions`).
    pub actions: Vec<String>,
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
    /// Files, folders or lines left out of the index, and why.
    pub skipped: Vec<String>,
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
    // A stored credential whose value cannot be read (vault key out of
    // reach) is a value the filter does not know: a conversation quoting it
    // would be indexed in clear. Nothing is read until it can be.
    let unreadable = crate::secrets::unreadable_credentials();
    if unreadable > 0 {
        report.errors.push(format!(
            "{} stored credential(s) cannot be read (vault key out of reach): their values would not be masked. Nothing was indexed.",
            unreadable
        ));
        // Recorded: the coverage warnings then say the index is not kept up.
        store.record_sync(&report, chrono::Utc::now());
        return report;
    }
    let over_budget = |report: &mut SyncReport| -> bool {
        let over = options.budget.is_some_and(|budget| started.elapsed() > budget);
        report.timed_out |= over;
        over
    };

    match cursor::default_store_path() {
        Some(path) => {
            if let Err(error) = sync_cursor(store, &path, &mut report, &over_budget) {
                report.errors.push(error);
            }
        }
        // Cursor's store is gone: its conversations come from their archive.
        None => {
            if let Some(dir) = archive::default_dir().filter(|dir| dir.is_dir()) {
                sync_archive(store, &dir, &mut report, &over_budget);
            }
        }
    }
    if !over_budget(&mut report) {
        sync_claude(store, &mut report, &over_budget);
    }
    report.elapsed_ms = started.elapsed().as_millis();
    store.record_sync(&report, chrono::Utc::now());
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
    let (mut composers, unreadable) = cursor::list_composers(&conn)?;
    if unreadable > 0 {
        report.skipped.push(format!("{} Cursor conversations could not be read and were left out", unreadable));
    }
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
        let (messages, unreadable) = cursor::read_messages(&conn, composer, &fresh)?;
        if !unreadable.is_empty() {
            report.skipped.push(format!(
                "Cursor conversation {}: {} messages could not be read and were left out",
                source.origin_id,
                unreadable.len()
            ));
        }
        // The last turn may still be streaming: its messages stay unseen
        // so the next sync reads their final text. Without a request in
        // the batch, the whole batch continues the stored last turn.
        let turn_start = messages.iter().rposition(|m| m.role == Role::User).unwrap_or(0);
        let open_turn: Vec<u64> = messages[turn_start..].iter().map(|m| hash_key(&m.key)).collect();
        let mut all_seen: Vec<u64> = ids
            .iter()
            .map(|id| hash_key(id))
            .filter(|hash| !open_turn.contains(hash))
            // Not marked as read: a later reader may know their form.
            .filter(|hash| !unreadable.iter().any(|id| hash_key(id) == *hash))
            .collect();
        all_seen.sort_unstable();
        all_seen.dedup();
        let is_main = source.origin_id == source.root_id;
        report.messages_written += store.apply(source, is_main, &messages, &all_seen)?;
        report.origins_updated += 1;
    }
    Ok(())
}

/// Indexes the archived Cursor conversations under the ids and message
/// keys the live store gave them, so nothing already indexed is doubled.
pub fn sync_archive(
    store: &mut ConversationStore,
    dir: &std::path::Path,
    report: &mut SyncReport,
    over_budget: &dyn Fn(&mut SyncReport) -> bool,
) {
    let (conversations, errors) = archive::read_archive(dir);
    report.errors.extend(errors);
    for conversation in &conversations {
        let mut read = false;
        for (source, messages) in &conversation.threads {
            if over_budget(report) {
                return;
            }
            report.origins_checked += 1;
            match store.origin_version(&source.origin_id) {
                Ok(Some(version)) if version == source.version => continue,
                Ok(_) => {}
                Err(error) => {
                    report.errors.push(error);
                    continue;
                }
            }
            let mut seen: Vec<u64> = messages.iter().map(|message| hash_key(&message.key)).collect();
            seen.sort_unstable();
            match store.apply(source, source.origin_id == source.root_id, messages, &seen) {
                Ok(written) => {
                    report.messages_written += written;
                    report.origins_updated += 1;
                    read = true;
                }
                Err(error) => report.errors.push(error),
            }
        }
        // Once, when the conversation enters the index: a project given to
        // it since (a rename, a merge) is never overwritten by the archive.
        if read {
            if let Err(error) = store.set_identity(&conversation.id, conversation.project.as_deref(), conversation.workspace.as_deref(), true) {
                report.errors.push(error);
            }
        }
    }
}

fn sync_claude(
    store: &mut ConversationStore,
    report: &mut SyncReport,
    over_budget: &dyn Fn(&mut SyncReport) -> bool,
) {
    sync_claude_roots(store, &claude::default_roots(), report, over_budget);
}

fn sync_claude_roots(
    store: &mut ConversationStore,
    roots: &[std::path::PathBuf],
    report: &mut SyncReport,
    over_budget: &dyn Fn(&mut SyncReport) -> bool,
) {
    if roots.is_empty() {
        return;
    }
    let titles = claude::desktop_titles(roots);
    let (files, skipped) = claude::list_sessions(roots);
    report.skipped.extend(skipped);
    // A session resumed from another starts with the lines of that one. They
    // are left to it when it is known: its file is there, or it was indexed
    // before its file was cleaned up (they would be indexed twice otherwise).
    let mut known: std::collections::HashSet<String> = files.iter().map(|file| file.id.clone()).collect();
    match store.origin_ids("claude-code") {
        Ok(indexed) => known.extend(indexed),
        Err(error) => {
            report.errors.push(error);
            return;
        }
    }
    let today = chrono::Utc::now().format("%Y-%m-%d").to_string();
    for group in claude::same_session(&files) {
        if over_budget(report) {
            return;
        }
        report.origins_checked += group.len();
        let version = claude::group_version(&group);
        let origin = group[0].id.as_str();
        // Files this session was last read from: when one has gone since,
        // what it alone held stays in the index.
        let mut files_before = 0;
        if !origin.is_empty() {
            match store.origin_version(origin) {
                Ok(Some(stored)) if stored == version => continue,
                Ok(Some(stored)) => files_before = claude::files_in_version(&stored),
                Ok(None) => {}
                Err(error) => {
                    report.errors.push(error);
                    continue;
                }
            }
        }
        let session = match claude::read_sessions(&group, &titles, &known) {
            Ok(session) => session,
            Err(error) => {
                report.errors.push(error);
                continue;
            }
        };
        report.skipped.extend(session.unreadable.iter().map(|(path, lines)| {
            format!("Claude session {}: {} lines are not JSON and were left out", path.display(), lines)
        }));
        let source = &session.conversation;
        if origin.is_empty() {
            match store.origin_version(&source.origin_id) {
                Ok(Some(stored)) if stored == source.version => continue,
                Ok(Some(stored)) => files_before = claude::files_in_version(&stored),
                Ok(None) => {}
                Err(error) => {
                    report.errors.push(error);
                    continue;
                }
            }
        }
        let mut seen: Vec<u64> = session.messages.iter().map(|m| hash_key(&m.key)).collect();
        seen.sort_unstable();
        // The file was read to its end: what the index holds of it and it
        // no longer does is dropped, and a day it says nothing on was a
        // session opened and left, not a conversation the index lost.
        let drop_what_is_gone = group.len() >= files_before;
        match store.apply_whole(source, &session.messages, &seen, &session.days, &today, drop_what_is_gone) {
            Ok(written) => {
                report.messages_written += written;
                report.origins_updated += group.len();
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
        let vectors = crate::embedding::embed_batch(&texts)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    fn line(kind: &str, uuid: &str, session: &str, at: &str, text: &str) -> String {
        let content = if kind == "user" {
            serde_json::json!(text)
        } else {
            serde_json::json!([{ "type": "text", "text": text }])
        };
        format!(
            "{}\n",
            serde_json::json!({
                "type": kind, "uuid": uuid, "sessionId": session, "cwd": "/work/courly",
                "timestamp": at, "message": { "content": content }
            })
        )
    }

    fn fresh_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("mp-{}-{}", name, std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("projects")).unwrap();
        dir
    }

    fn keys(store: &ConversationStore, conversation: &str) -> Vec<String> {
        let mut statement = store
            .conn
            .prepare("SELECT source_key FROM segments WHERE conversation_id = ?1 AND role IN ('user', 'assistant') ORDER BY at, id")
            .unwrap();
        statement.query_map([conversation], |row| row.get(0)).unwrap().map(Result::unwrap).collect()
    }

    fn text_of(store: &ConversationStore, key: &str) -> Option<String> {
        use rusqlite::OptionalExtension;
        store.conn.query_row("SELECT text FROM segments WHERE source_key = ?1", [key], |row| row.get(0)).optional().unwrap()
    }

    const TARIFS: &str = "La page des tarifs est prête avec les trois forfaits, le mensuel, l'annuel et l'offre équipe.";
    const SEPA: &str = "Le virement SEPA est branché sur le tunnel de paiement, avec le mandat signé à la commande.";

    /// The file of the first session is cleaned up, the resumed one goes
    /// on: its copied lines used to be indexed a second time, under it.
    #[test]
    fn a_resumed_session_whose_original_is_gone_does_not_double_its_messages() {
        let dir = fresh_dir("orphan");
        let projects = dir.join("projects/-work-courly");
        std::fs::create_dir_all(&projects).unwrap();
        let earlier = [
            line("user", "u1", "first", "2026-11-16T17:29:03Z", "Prépare la page des tarifs de Courly"),
            line("assistant", "a1", "first", "2026-11-16T17:30:00Z", TARIFS),
        ]
        .concat();
        let own = [
            line("user", "u2", "resumed", "2026-11-19T09:00:00Z", "Ajoute le paiement par virement SEPA"),
            line("assistant", "a2", "resumed", "2026-11-19T09:02:00Z", SEPA),
        ]
        .concat();
        std::fs::write(projects.join("first.jsonl"), &earlier).unwrap();
        std::fs::write(projects.join("resumed.jsonl"), format!("{}{}", earlier, own)).unwrap();
        let mut store = ConversationStore::open(&dir.join("conversations.db")).unwrap();
        let sync = |store: &mut ConversationStore| {
            let mut report = SyncReport::default();
            sync_claude_roots(store, &[dir.join("projects")], &mut report, &|_| false);
            assert!(report.errors.is_empty(), "{:?}", report.errors);
        };
        sync(&mut store);
        std::fs::remove_file(projects.join("first.jsonl")).unwrap();
        let more = line("user", "u3", "resumed", "2026-11-20T08:00:00Z", "Ajoute aussi le prélèvement mensuel automatique");
        std::fs::write(projects.join("resumed.jsonl"), format!("{}{}{}", earlier, own, more)).unwrap();
        sync(&mut store);

        assert_eq!(keys(&store, "first"), ["claude-code:first:u1", "claude-code:first:u1#answer"]);
        assert_eq!(
            keys(&store, "resumed"),
            ["claude-code:resumed:u2", "claude-code:resumed:u2#answer", "claude-code:resumed:u3"]
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The same session file name in two folders: each took the origin
    /// from the other, only the last one read stayed, at every sync.
    #[test]
    fn two_files_of_one_session_are_read_as_one() {
        let dir = fresh_dir("same-name");
        let (here, there) = (dir.join("projects/-work-a"), dir.join("projects/-work-b"));
        std::fs::create_dir_all(&here).unwrap();
        std::fs::create_dir_all(&there).unwrap();
        let first = [
            line("user", "u1", "same", "2026-11-16T17:29:03Z", "Prépare la page des tarifs de Courly"),
            line("assistant", "a1", "same", "2026-11-16T17:30:00Z", TARIFS),
        ]
        .concat();
        let second = [
            line("user", "u2", "same", "2026-11-19T09:00:00Z", "Ajoute le paiement par virement SEPA"),
            line("assistant", "a2", "same", "2026-11-19T09:02:00Z", SEPA),
            line("user", "u3", "same", "2026-11-19T09:05:00Z", "Ajoute aussi le prélèvement mensuel automatique"),
        ]
        .concat();
        std::fs::write(here.join("same.jsonl"), &first).unwrap();
        // The second file starts with a copy of the first.
        std::fs::write(there.join("same.jsonl"), format!("{}{}", first, second)).unwrap();
        let mut store = ConversationStore::open(&dir.join("conversations.db")).unwrap();
        let mut report = SyncReport::default();
        sync_claude_roots(&mut store, &[dir.join("projects")], &mut report, &|_| false);
        assert!(report.errors.is_empty(), "{:?}", report.errors);
        assert_eq!(
            keys(&store, "same"),
            [
                "claude-code:same:u1",
                "claude-code:same:u1#answer",
                "claude-code:same:u2",
                "claude-code:same:u2#answer",
                "claude-code:same:u3"
            ]
        );
        // Nothing changed: nothing is read or written again.
        let mut again = SyncReport::default();
        sync_claude_roots(&mut store, &[dir.join("projects")], &mut again, &|_| false);
        assert_eq!((again.origins_checked, again.origins_updated, again.messages_written), (2, 0, 0));

        // One of the two files is cleaned up: what it alone held stays.
        std::fs::remove_file(there.join("same.jsonl")).unwrap();
        let all = keys(&store, "same");
        for _ in 0..2 {
            let mut after = SyncReport::default();
            sync_claude_roots(&mut store, &[dir.join("projects")], &mut after, &|_| false);
            assert!(after.errors.is_empty(), "{:?}", after.errors);
            assert_eq!(keys(&store, "same"), all);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// An index written by the reader that cut action lines before masking
    /// them: the cut lines go when the index is opened, and a session read
    /// again writes its actions anew instead of adding to the old ones.
    #[test]
    fn action_lines_cut_before_masking_leave_the_index() {
        let dir = fresh_dir("cut-lines");
        let projects = dir.join("projects/-work-courly");
        std::fs::create_dir_all(&projects).unwrap();
        let call = |uuid: &str, at: &str, command: &str| {
            format!(
                "{}\n",
                serde_json::json!({
                    "type": "assistant", "uuid": uuid, "sessionId": "s", "timestamp": at,
                    "message": { "content": [{ "type": "tool_use", "name": "Bash", "input": { "command": command } }] }
                })
            )
        };
        let file = [
            line("user", "u1", "s", "2026-11-16T17:29:03Z", "Ouvre un ticket chez le fournisseur"),
            call("a1", "2026-11-16T17:29:10Z", "gh issue create --repo vendor/sdk --title 'Audio'"),
        ]
        .concat();
        std::fs::write(projects.join("s.jsonl"), &file).unwrap();
        let path = dir.join("conversations.db");
        let fragment = format!("ghp_{}", crate::secrets::tests::fake(20, 82));
        {
            let mut store = ConversationStore::open(&path).unwrap();
            let mut report = SyncReport::default();
            sync_claude_roots(&mut store, &[dir.join("projects")], &mut report, &|_| false);
            assert!(report.errors.is_empty(), "{:?}", report.errors);
            // What the former reader left: a line cut across a token, on
            // this turn and on a turn of a source that is not read again.
            let cut = format!("Bash: {} {}…", "x".repeat(260), fragment);
            store.conn.execute("UPDATE segments SET text = text || char(10) || ?1 WHERE source_key = 'claude-code:s:u1#actions'", [&cut]).unwrap();
            store
                .conn
                .execute(
                    "INSERT INTO segments (conversation_id, origin_id, source_key, role, at, text, text_hash)
                     SELECT conversation_id, origin_id, 'cursor:c:b1#actions', 'action', at, ?1, 'old' FROM segments LIMIT 1",
                    [&cut],
                )
                .unwrap();
            store.conn.execute("DELETE FROM upkeep", []).unwrap();
        }
        let mut store = ConversationStore::open(&path).unwrap();
        let holding: i64 = store
            .conn
            .query_row("SELECT COUNT(*) FROM segments WHERE instr(text, ?1) > 0", [&fragment], |row| row.get(0))
            .unwrap();
        assert_eq!(holding, 0, "the cut lines are gone as soon as the index opens");

        // The session grows: its actions are written again, whole.
        let more = call("a2", "2026-11-16T17:30:00Z", "gh issue comment 42 --repo vendor/sdk --body 'Logs joints'");
        std::fs::write(projects.join("s.jsonl"), format!("{}{}", file, more)).unwrap();
        let mut report = SyncReport::default();
        sync_claude_roots(&mut store, &[dir.join("projects")], &mut report, &|_| false);
        assert!(report.errors.is_empty(), "{:?}", report.errors);
        assert_eq!(
            text_of(&store, "claude-code:s:u1#actions").as_deref(),
            Some("Bash: gh issue create --repo vendor/sdk --title 'Audio'\nBash: gh issue comment 42 --repo vendor/sdk --body 'Logs joints'")
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A request made of an image alone opens a turn, and what the agent
    /// writes before any request stays before it when the file is read again.
    #[test]
    fn a_wordless_request_opens_a_turn_and_a_reread_keeps_the_turns() {
        let dir = fresh_dir("turns");
        let projects = dir.join("projects/-work-courly");
        std::fs::create_dir_all(&projects).unwrap();
        let image = format!(
            "{}\n",
            serde_json::json!({
                "type": "user", "uuid": "u2", "sessionId": "s", "cwd": "/work/courly", "timestamp": "2026-11-16T17:31:00Z",
                "message": { "content": [{ "type": "image", "source": { "type": "base64", "media_type": "image/png", "data": "AAAA" } }] }
            })
        );
        let tool_result = format!(
            "{}\n",
            serde_json::json!({
                "type": "user", "uuid": "r1", "sessionId": "s", "timestamp": "2026-11-16T17:29:30Z",
                "message": { "content": [{ "type": "tool_result", "content": "ls" }] }
            })
        );
        let opening = "Voici le point de départ : le dépôt est cloné, les dépendances sont installées et les tests passent.";
        let file = [
            line("assistant", "a0", "s", "2026-11-16T17:28:00Z", opening),
            line("user", "u1", "s", "2026-11-16T17:29:03Z", "Prépare la page des tarifs de Courly"),
            tool_result,
            line("assistant", "a1", "s", "2026-11-16T17:30:00Z", TARIFS),
            image,
            line("assistant", "a2", "s", "2026-11-16T17:32:00Z", SEPA),
        ]
        .concat();
        std::fs::write(projects.join("s.jsonl"), &file).unwrap();
        let mut store = ConversationStore::open(&dir.join("conversations.db")).unwrap();
        let sync = |store: &mut ConversationStore| {
            let mut report = SyncReport::default();
            sync_claude_roots(store, &[dir.join("projects")], &mut report, &|_| false);
            assert!(report.errors.is_empty(), "{:?}", report.errors);
        };
        let check = |store: &ConversationStore| {
            assert_eq!(text_of(store, "claude-code:s:start#answer").as_deref(), Some(opening));
            assert_eq!(text_of(store, "claude-code:s:u1#answer").as_deref(), Some(TARIFS), "a tool result opens no turn");
            assert_eq!(text_of(store, "claude-code:s:u2#answer").as_deref(), Some(SEPA));
        };
        sync(&mut store);
        check(&store);
        let more = line("user", "u3", "s", "2026-11-16T17:40:00Z", "Ajoute aussi le prélèvement mensuel automatique");
        std::fs::write(projects.join("s.jsonl"), format!("{}{}", file, more)).unwrap();
        sync(&mut store);
        check(&store);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A session resumed from another starts with the lines of that one,
    /// under its id. It used to be indexed under that id: its own id
    /// stayed unknown, and both files took turns on one origin.
    #[test]
    fn a_resumed_claude_session_is_indexed_under_its_own_id() {
        let dir = std::env::temp_dir().join(format!("mp-resumed-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let first_dir = dir.join("projects/-work-courly-worktree");
        let resumed_dir = dir.join("projects/-work-courly");
        std::fs::create_dir_all(&first_dir).unwrap();
        std::fs::create_dir_all(&resumed_dir).unwrap();
        let earlier = [
            line("user", "u1", "first", "2026-11-16T17:29:03Z", "Prépare la page des tarifs de Courly"),
            line("assistant", "a1", "first", "2026-11-16T17:30:00Z", "La page des tarifs est prête avec les trois forfaits, le mensuel, l'annuel et l'offre équipe."),
        ]
        .concat();
        std::fs::write(first_dir.join("first.jsonl"), &earlier).unwrap();
        let own = [
            line("user", "u2", "resumed", "2026-11-19T09:00:00Z", "Ajoute le paiement par virement SEPA"),
            line("assistant", "a2", "resumed", "2026-11-19T09:02:00Z", "Le virement SEPA est branché sur le tunnel de paiement, avec le mandat signé à la commande."),
        ]
        .concat();
        // Copied history, one line that is not JSON, then its own turns;
        // the last line is still being written.
        let resumed = format!("{}{{not json\n{}{{\"type\":\"user\"", earlier, own);
        std::fs::write(resumed_dir.join("resumed.jsonl"), resumed).unwrap();

        let mut store = ConversationStore::open(&dir.join("conversations.db")).unwrap();
        // The index as the former reader left it: the resumed session's
        // messages filed under the session it resumed.
        let (files, _) = claude::list_sessions(&[dir.join("projects")]);
        let resumed_file = files.iter().find(|file| file.id == "resumed").unwrap();
        let mut misfiled = claude::read_session(resumed_file, &HashMap::new(), &Default::default()).unwrap();
        misfiled.conversation.origin_id = "first".into();
        misfiled.conversation.root_id = "first".into();
        misfiled.conversation.version = "old".into();
        store.apply(&misfiled.conversation, true, &misfiled.messages, &[]).unwrap();
        // Seen running on a day it said something, and on one it did not.
        let day = |raw: &str| chrono::DateTime::parse_from_rfc3339(raw).unwrap().with_timezone(&chrono::Utc);
        store.record_activity("resumed", "claude", None, day("2026-11-19T09:00:00Z")).unwrap();
        store.record_activity("resumed", "claude", None, day("2026-11-21T10:00:00Z")).unwrap();

        let mut report = SyncReport::default();
        sync_claude_roots(&mut store, &[dir.join("projects")], &mut report, &|_| false);
        assert!(report.errors.is_empty(), "{:?}", report.errors);
        assert_eq!(report.skipped.len(), 1, "{:?}", report.skipped);
        assert!(report.skipped[0].contains("resumed.jsonl: 1 lines are not JSON"), "{}", report.skipped[0]);

        let segments = |conversation: &str| -> Vec<String> {
            let mut statement = store
                .conn
                .prepare("SELECT source_key FROM segments WHERE conversation_id = ?1 AND role IN ('user', 'assistant') ORDER BY at")
                .unwrap();
            statement.query_map([conversation], |row| row.get(0)).unwrap().map(Result::unwrap).collect()
        };
        assert_eq!(segments("first"), ["claude-code:first:u1", "claude-code:first:u1#answer"]);
        assert_eq!(segments("resumed"), ["claude-code:resumed:u2", "claude-code:resumed:u2#answer"]);
        let count = |conversation: &str| -> i64 {
            store.conn.query_row("SELECT message_count FROM conversations WHERE id = ?1", [conversation], |row| row.get(0)).unwrap()
        };
        assert_eq!((count("first"), count("resumed")), (2, 2));
        let found = store.search_within("resumed", "virement SEPA", 5, false).unwrap().expect("the resumed session is known");
        assert!(!found.is_empty());

        let days: Vec<String> = {
            let mut statement = store.conn.prepare("SELECT day FROM activity WHERE conversation_id = 'resumed'").unwrap();
            statement.query_map([], |row| row.get(0)).unwrap().map(Result::unwrap).collect()
        };
        assert_eq!(days, ["2026-11-19"], "a day the session said nothing on is not reported as lost");

        // Nothing changed: no file is read again.
        let mut again = SyncReport::default();
        sync_claude_roots(&mut store, &[dir.join("projects")], &mut again, &|_| false);
        assert_eq!((again.origins_checked, again.origins_updated), (2, 0));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
