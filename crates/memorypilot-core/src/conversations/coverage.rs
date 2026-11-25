//! What the index may be missing.
//!
//! A search that finds nothing is read as "it never happened". Often the
//! session simply was not indexed: the sync failed, the window was never
//! read, or a long conversation has a hole of several weeks. The index
//! therefore keeps two traces, the outcome of each sync and the days an
//! agent session was seen running, and says where they disagree with
//! what it holds.

use super::store::ConversationStore;
use super::SyncReport;
use rusqlite::{params, OptionalExtension};

/// Traces older than this are dropped.
const KEPT_DAYS: i64 = 90;
/// A session seen running is expected in the index after this long: it
/// is indexed when it ends, or at the next search.
const INDEXING_DELAY_HOURS: i64 = 6;
/// A sync is due at every session end: none for this long means the
/// hooks are not installed or keep failing.
const STALE_SYNC_DAYS: i64 = 3;
/// A conversation that goes silent this long between two messages is
/// reported when it is searched: either nothing happened, or the weeks
/// in between were not read.
const SILENT_DAYS: i64 = 7;
const MAX_REPORTED: usize = 5;

pub(super) const SCHEMA: &str = "
    CREATE TABLE IF NOT EXISTS sync_log (
        id INTEGER PRIMARY KEY,
        at TEXT NOT NULL,
        origins_checked INTEGER NOT NULL,
        origins_updated INTEGER NOT NULL,
        messages_written INTEGER NOT NULL,
        errors TEXT,
        timed_out INTEGER NOT NULL,
        elapsed_ms INTEGER NOT NULL
    );
    CREATE TABLE IF NOT EXISTS activity (
        conversation_id TEXT NOT NULL,
        day TEXT NOT NULL,
        tool TEXT NOT NULL,
        workspace TEXT,
        events INTEGER NOT NULL DEFAULT 1,
        last_at TEXT NOT NULL,
        PRIMARY KEY (conversation_id, day)
    ) WITHOUT ROWID;
";

type Now = chrono::DateTime<chrono::Utc>;

fn days_between(earlier: &str, later: &str) -> Option<i64> {
    let day = |raw: &str| chrono::NaiveDate::parse_from_str(raw.get(..10)?, "%Y-%m-%d").ok();
    Some((day(later)? - day(earlier)?).num_days())
}

impl ConversationStore {
    /// Keeps the outcome of a sync. Best effort: a lost line only makes
    /// the index look less recently synced than it is.
    pub fn record_sync(&self, report: &SyncReport, now: Now) {
        let left_out: Vec<&String> = report.errors.iter().chain(report.skipped.iter()).collect();
        let errors = (!left_out.is_empty()).then(|| left_out.iter().map(|line| line.as_str()).collect::<Vec<_>>().join(" | "));
        let _ = self.conn.execute(
            "INSERT INTO sync_log (at, origins_checked, origins_updated, messages_written, errors, timed_out, elapsed_ms)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                now.to_rfc3339(),
                report.origins_checked as i64,
                report.origins_updated as i64,
                report.messages_written as i64,
                errors,
                report.timed_out,
                report.elapsed_ms as i64
            ],
        );
        let cutoff = (now - chrono::Duration::days(KEPT_DAYS)).to_rfc3339();
        let _ = self.conn.execute("DELETE FROM sync_log WHERE at < ?1", [&cutoff]);
        let _ = self.conn.execute("DELETE FROM activity WHERE last_at < ?1", [&cutoff]);
    }

    /// Notes that a session of `tool` was running in `conversation_id`
    /// (called by the session hooks).
    pub fn record_activity(&self, conversation_id: &str, tool: &str, workspace: Option<&str>, now: Now) -> Result<(), String> {
        self.conn
            .execute(
                "INSERT INTO activity (conversation_id, day, tool, workspace, last_at) VALUES (?1, ?2, ?3, ?4, ?5)
                 ON CONFLICT(conversation_id, day) DO UPDATE SET
                    events = events + 1, last_at = excluded.last_at, workspace = COALESCE(excluded.workspace, workspace)",
                params![conversation_id, now.format("%Y-%m-%d").to_string(), tool, workspace, now.to_rfc3339()],
            )
            .map(|_| ())
            .map_err(|error| format!("Activity: {}", error))
    }

    /// Drops the days a session was seen running on which its source,
    /// read to the end, holds no message: it was opened and nothing was
    /// said. `today` is kept, the session may still be speaking.
    pub fn forget_silent_days(&self, conversation_id: &str, days_said: &std::collections::BTreeSet<String>, today: &str) -> Result<(), String> {
        let mut statement = self
            .conn
            .prepare("SELECT day FROM activity WHERE conversation_id = ?1 AND day < ?2")
            .map_err(|error| format!("Activity days: {}", error))?;
        let days: Vec<String> = statement
            .query_map(params![conversation_id, today], |row| row.get(0))
            .and_then(|rows| rows.collect())
            .map_err(|error| format!("Activity days: {}", error))?;
        for day in days.iter().filter(|day| !days_said.contains(*day)) {
            self.conn
                .execute("DELETE FROM activity WHERE conversation_id = ?1 AND day = ?2", params![conversation_id, day])
                .map_err(|error| format!("Activity forget: {}", error))?;
        }
        Ok(())
    }

    /// What the index may be missing, in sentences an agent can relay:
    /// for the whole index, or for one conversation.
    pub fn coverage_warnings(&self, conversation: Option<&str>, now: Now) -> Vec<String> {
        let mut warnings = Vec::new();
        if conversation.is_none() {
            warnings.extend(self.sync_warnings(now));
        }
        warnings.extend(self.unindexed_sessions(conversation, now));
        if let Some(conversation) = conversation {
            warnings.extend(self.silent_periods(conversation));
        }
        warnings
    }

    fn sync_warnings(&self, now: Now) -> Vec<String> {
        let last: Option<(String, Option<String>, bool)> = self
            .conn
            .query_row("SELECT at, errors, timed_out FROM sync_log ORDER BY id DESC LIMIT 1", [], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?))
            })
            .optional()
            .ok()
            .flatten();
        let Some((at, errors, timed_out)) = last else { return Vec::new() };
        let mut warnings = Vec::new();
        if let Some(days) = days_between(&at, &now.to_rfc3339()).filter(|days| *days >= STALE_SYNC_DAYS) {
            warnings.push(format!(
                "The conversation index was last synced {} days ago ({}): what was said since is not in it. Run `MemoryPilot --index-conversations`, and check the session hooks are installed.",
                days,
                &at[..10]
            ));
        }
        if let Some(errors) = errors {
            warnings.push(format!("The last sync ({}) reported errors, some conversations were not read: {}", &at[..10], errors));
        }
        // A full sync has no budget: only a quick catch-up before a search stops early.
        let full_since: Option<String> = self
            .conn
            .query_row("SELECT MAX(at) FROM sync_log WHERE timed_out = 0", [], |row| row.get(0))
            .ok()
            .flatten();
        if timed_out && full_since.as_deref().and_then(|full| days_between(full, &at)).is_none_or(|days| days >= STALE_SYNC_DAYS) {
            warnings.push(
                "Recent syncs stopped at their time budget and none ran to the end lately: some conversations are behind. Run `MemoryPilot --index-conversations`.".to_string(),
            );
        }
        warnings
    }

    /// Days a session was seen running with nothing indexed for that day.
    fn unindexed_sessions(&self, conversation: Option<&str>, now: Now) -> Vec<String> {
        let settled = (now - chrono::Duration::hours(INDEXING_DELAY_HOURS)).to_rfc3339();
        let Ok(mut statement) = self.conn.prepare(
            "SELECT a.conversation_id, a.day, a.tool, a.workspace FROM activity a
             WHERE a.last_at < ?1 AND (?2 IS NULL OR a.conversation_id = ?2)
               AND NOT EXISTS (
                   SELECT 1 FROM segments s WHERE s.conversation_id = a.conversation_id AND substr(s.at, 1, 10) = a.day)
             ORDER BY a.day DESC LIMIT ?3",
        ) else {
            return Vec::new();
        };
        let rows = statement.query_map(params![settled, conversation, MAX_REPORTED as i64], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?, row.get::<_, String>(2)?, row.get::<_, Option<String>>(3)?))
        });
        let Ok(rows) = rows else { return Vec::new() };
        rows.filter_map(Result::ok)
            .map(|(id, day, tool, workspace)| {
                format!(
                    "A {} session ran on {}{} (conversation {}) and nothing of that day is indexed: what was said then cannot be found here.",
                    tool,
                    day,
                    workspace.map(|folder| format!(" in {}", folder)).unwrap_or_default(),
                    id
                )
            })
            .collect()
    }

    /// Stretches of a conversation with no message, longest first.
    fn silent_periods(&self, conversation: &str) -> Vec<String> {
        let Ok(mut statement) = self.conn.prepare(
            "SELECT DISTINCT substr(at, 1, 10) FROM segments
             WHERE conversation_id = ?1 AND at IS NOT NULL AND role != 'title' ORDER BY 1",
        ) else {
            return Vec::new();
        };
        let days: Vec<String> = statement
            .query_map([conversation], |row| row.get::<_, String>(0))
            .map(|rows| rows.filter_map(Result::ok).collect())
            .unwrap_or_default();
        let mut silent: Vec<(i64, &str, &str)> = days
            .windows(2)
            .filter_map(|pair| Some((days_between(&pair[0], &pair[1])?, pair[0].as_str(), pair[1].as_str())))
            .filter(|(length, _, _)| *length >= SILENT_DAYS)
            .collect();
        silent.sort_by(|a, b| b.0.cmp(&a.0));
        silent
            .into_iter()
            .take(MAX_REPORTED)
            .map(|(length, from, to)| {
                format!(
                    "No message of this conversation is indexed between {} and {} ({} days): either nothing was said, or that period was not read. Do not conclude that nothing happened then.",
                    from, to, length
                )
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::super::{Message, Role, SourceConversation};
    use super::*;

    fn store(name: &str) -> (ConversationStore, std::path::PathBuf) {
        let dir = std::env::temp_dir().join(format!("mp-coverage-{}-{}", name, std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        (ConversationStore::open(&dir.join("c.db")).unwrap(), dir)
    }

    fn said(store: &mut ConversationStore, conversation: &str, days: &[&str]) {
        let source = SourceConversation {
            origin_id: conversation.to_string(),
            root_id: conversation.to_string(),
            tool: "cursor",
            title: None,
            workspace: None,
            created_at: None,
            version: days.len().to_string(),
            source_path: None,
        };
        let messages: Vec<Message> = days
            .iter()
            .enumerate()
            .map(|(index, day)| Message {
                key: format!("m{}", index),
                role: Role::User,
                at: Some(format!("{}T10:00:00.000Z", day)),
                order: None,
                text: format!("Message numéro {} de la conversation sur la publication", index),
                actions: Vec::new(),
            })
            .collect();
        store.apply(&source, true, &messages, &[]).unwrap();
    }

    fn at(day: &str) -> Now {
        chrono::DateTime::parse_from_rfc3339(&format!("{}T12:00:00Z", day)).unwrap().with_timezone(&chrono::Utc)
    }

    #[test]
    fn a_hole_of_weeks_in_a_conversation_is_said_when_it_is_searched() {
        let (mut store, dir) = store("hole");
        said(&mut store, "long", &["2026-10-19", "2026-10-20", "2026-10-21", "2026-11-07", "2026-11-08"]);
        said(&mut store, "steady", &["2026-11-01", "2026-11-03", "2026-11-06"]);
        let warnings = store.coverage_warnings(Some("long"), at("2026-11-18"));
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert!(warnings[0].contains("between 2026-10-21 and 2026-11-07 (17 days)"), "{}", warnings[0]);
        assert!(store.coverage_warnings(Some("steady"), at("2026-11-18")).is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_session_seen_running_but_never_indexed_is_reported() {
        let (mut store, dir) = store("unindexed");
        said(&mut store, "known", &["2026-11-10"]);
        store.record_activity("known", "cursor", Some("/work/app"), at("2026-11-10")).unwrap();
        store.record_activity("known", "cursor", Some("/work/app"), at("2026-11-12")).unwrap();
        store.record_activity("never-read", "claude", None, at("2026-11-13")).unwrap();
        // A session of this morning is not expected in the index yet.
        store.record_activity("live", "claude", None, at("2026-11-18") - chrono::Duration::hours(2)).unwrap();

        let all = store.coverage_warnings(None, at("2026-11-18"));
        assert_eq!(all.len(), 2, "{all:?}");
        assert!(all[0].contains("A claude session ran on 2026-11-13") && all[0].contains("never-read"));
        assert!(all[1].contains("A cursor session ran on 2026-11-12 in /work/app"));
        let one = store.coverage_warnings(Some("known"), at("2026-11-18"));
        assert_eq!(one.len(), 1);
        // Once that day is read, nothing is left to report for it.
        said(&mut store, "known", &["2026-11-10", "2026-11-12"]);
        assert!(store.coverage_warnings(Some("known"), at("2026-11-18")).is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_stale_or_failing_sync_is_reported_and_old_traces_are_dropped() {
        let (store, dir) = store("sync");
        assert!(store.coverage_warnings(None, at("2026-11-18")).is_empty(), "no trace yet, nothing to say");
        store.record_activity("old", "cursor", None, at("2026-06-01")).unwrap();
        let failed = SyncReport { errors: vec!["Cursor store locked".to_string()], ..SyncReport::default() };
        store.record_sync(&failed, at("2026-11-10"));
        let warnings = store.coverage_warnings(None, at("2026-11-18"));
        assert_eq!(warnings.len(), 2, "{warnings:?}");
        assert!(warnings[0].contains("last synced 8 days ago (2026-11-10)"));
        assert!(warnings[1].contains("Cursor store locked"));
        let kept: i64 = store.conn.query_row("SELECT COUNT(*) FROM activity", [], |row| row.get(0)).unwrap();
        assert_eq!(kept, 0, "a trace older than the retention is dropped");

        store.record_sync(&SyncReport::default(), at("2026-11-18"));
        assert!(store.coverage_warnings(None, at("2026-11-18")).is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
