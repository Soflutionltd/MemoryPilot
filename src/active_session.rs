//! Which conversation is the agent calling this server in right now.
//!
//! The MCP server is told nothing about its caller, so the hooks leave a
//! trace: each session start and each prompt notes the conversation id,
//! its folder and the processes the hook descends from. The server, a
//! child of the same agent process, finds its own conversation by the
//! nearest process they share. `find_conversation` uses it to leave the
//! current conversation out: the question just typed in it would
//! otherwise always be the best match.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

const FILE: &str = "active-conversations.json";
/// A session that has not sent a prompt for this long is no longer the
/// one calling.
const TTL_HOURS: i64 = 24 * 7;
/// Without a shared process, only a session seen this recently in the
/// same folder is taken for the current one.
const FOLDER_ONLY_HOURS: i64 = 12;
const MAX_ANCESTORS: usize = 12;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct Entry {
    conversation_id: String,
    workspace: Option<String>,
    agent: String,
    at: String,
    ancestors: Vec<u32>,
}

pub fn state_path() -> PathBuf {
    crate::hooks::hook_path(memorypilot_core::layout::Area::State, FILE)
}

fn folder(path: &str) -> String {
    path.trim_end_matches('/').to_string()
}

fn read(path: &Path) -> Vec<Entry> {
    std::fs::read_to_string(path).ok().and_then(|raw| serde_json::from_str(&raw).ok()).unwrap_or_default()
}

fn age_hours(entry: &Entry, now: chrono::DateTime<chrono::Utc>) -> Option<i64> {
    let at = chrono::DateTime::parse_from_rfc3339(&entry.at).ok()?;
    Some(now.signed_duration_since(at).num_hours())
}

/// Notes that `conversation_id` is live. Best effort: a lost trace only
/// means the current conversation shows up in its own search.
pub fn record(
    path: &Path,
    conversation_id: &str,
    workspace: Option<&str>,
    agent: &str,
    ancestors: &[u32],
    now: chrono::DateTime<chrono::Utc>,
) {
    let mut entries = read(path);
    entries.retain(|entry| {
        entry.conversation_id != conversation_id && age_hours(entry, now).is_some_and(|hours| hours < TTL_HOURS)
    });
    entries.push(Entry {
        conversation_id: conversation_id.to_string(),
        workspace: workspace.map(folder),
        agent: agent.to_string(),
        at: now.to_rfc3339(),
        ancestors: ancestors.to_vec(),
    });
    let tmp = path.with_extension("json.tmp");
    if std::fs::write(&tmp, serde_json::to_string(&entries).unwrap_or_default()).is_ok() {
        let _ = std::fs::rename(tmp, path);
    }
}

/// The conversation the caller is in: the live session sharing the
/// nearest process with `own_ancestors` (nearest first); between equals,
/// the one in `workspace`, then the latest. With no shared process, the
/// latest recent session of `workspace`.
pub fn current(
    path: &Path,
    own_ancestors: &[u32],
    workspace: Option<&str>,
    now: chrono::DateTime<chrono::Utc>,
) -> Option<String> {
    let workspace = workspace.map(folder);
    let live: Vec<(Entry, i64)> = read(path)
        .into_iter()
        .filter_map(|entry| {
            let hours = age_hours(&entry, now)?;
            (hours < TTL_HOURS).then_some((entry, hours))
        })
        .collect();
    let in_folder = |entry: &Entry| workspace.is_some() && entry.workspace == workspace;
    let depth = |entry: &Entry| own_ancestors.iter().position(|pid| *pid > 1 && entry.ancestors.contains(pid));

    let nearest = live.iter().filter_map(|(entry, _)| depth(entry)).min();
    let candidates: Vec<&Entry> = match nearest {
        Some(nearest) => live.iter().map(|(entry, _)| entry).filter(|entry| depth(entry) == Some(nearest)).collect(),
        None => live
            .iter()
            .filter(|(entry, hours)| in_folder(entry) && *hours < FOLDER_ONLY_HOURS)
            .map(|(entry, _)| entry)
            .collect(),
    };
    candidates
        .into_iter()
        .max_by(|a, b| in_folder(a).cmp(&in_folder(b)).then(a.at.cmp(&b.at)))
        .map(|entry| entry.conversation_id.clone())
}

/// The conversation of the agent calling this server, from the trace its
/// hooks leave. `None` when no hook ran (an agent without hooks).
pub fn calling_conversation() -> Option<String> {
    static ANCESTORS: std::sync::OnceLock<Vec<u32>> = std::sync::OnceLock::new();
    let workspace = std::env::current_dir().ok().map(|dir| dir.to_string_lossy().into_owned());
    current(&state_path(), ANCESTORS.get_or_init(own_ancestors), workspace.as_deref(), chrono::Utc::now())
}

/// `metadata` with where a memory comes from: the conversation it is
/// written in and when, so that a fact can be traced back to what was
/// said (`read_conversation` with `at`). An origin already given is kept.
pub fn with_origin(metadata: Option<&serde_json::Value>, conversation: Option<&str>, now: chrono::DateTime<chrono::Utc>) -> Option<serde_json::Value> {
    let conversation = conversation?;
    if metadata.and_then(|m| m.get("origin")).is_some() {
        return None;
    }
    let mut object = match metadata {
        Some(serde_json::Value::Object(object)) => object.clone(),
        _ => serde_json::Map::new(),
    };
    object.insert("origin".into(), serde_json::json!({ "conversation_id": conversation, "at": now.to_rfc3339() }));
    Some(serde_json::Value::Object(object))
}

/// The chain of parents of `pid` in a `pid ppid` table, nearest first.
fn chain(table: &str, pid: u32) -> Vec<u32> {
    let parents: std::collections::HashMap<u32, u32> = table
        .lines()
        .filter_map(|line| {
            let mut fields = line.split_whitespace();
            Some((fields.next()?.parse().ok()?, fields.next()?.parse().ok()?))
        })
        .collect();
    let mut ancestors = Vec::new();
    let mut current = pid;
    while let Some(parent) = parents.get(&current).copied() {
        if parent <= 1 || ancestors.contains(&parent) || ancestors.len() == MAX_ANCESTORS {
            break;
        }
        ancestors.push(parent);
        current = parent;
    }
    ancestors
}

/// The processes this one descends from, nearest first. Empty where the
/// process table cannot be read (Windows): the folder alone decides then.
pub fn own_ancestors() -> Vec<u32> {
    if cfg!(windows) {
        return Vec::new();
    }
    std::process::Command::new("ps")
        .args(["-axo", "pid=,ppid="])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| chain(&String::from_utf8_lossy(&output.stdout), std::process::id()))
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("mp-active-{}-{}", name, std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(FILE)
    }

    #[test]
    fn the_chain_of_parents_stops_at_the_first_process() {
        let table = "  1     0\n 40     1\n 77    40\n 90    77\n 91    77\n";
        assert_eq!(chain(table, 90), vec![77, 40]);
        assert_eq!(chain(table, 500), Vec::<u32>::new());
    }

    #[test]
    fn the_session_sharing_the_nearest_process_is_the_current_one() {
        let path = temp("nearest");
        let now = chrono::Utc::now();
        // Two sessions of one app (process 10): each has its own agent process.
        record(&path, "mine", Some("/work/app/"), "claude", &[300, 200, 10], now - chrono::Duration::hours(3));
        record(&path, "other", Some("/work/app"), "claude", &[301, 201, 10], now);
        assert_eq!(current(&path, &[200, 10], Some("/work/app"), now), Some("mine".to_string()));
        assert_eq!(current(&path, &[201, 10], Some("/work/app"), now), Some("other".to_string()));
        // Sharing only the app: the session of the caller's folder, then the latest.
        record(&path, "elsewhere", Some("/work/site"), "cursor", &[400, 10], now - chrono::Duration::minutes(5));
        assert_eq!(current(&path, &[999, 10], Some("/work/site"), now), Some("elsewhere".to_string()));
        assert_eq!(current(&path, &[999, 10], None, now), Some("other".to_string()));
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn without_a_shared_process_only_a_recent_session_of_the_folder_counts() {
        let path = temp("folder");
        let now = chrono::Utc::now();
        record(&path, "old", Some("/work/app"), "cursor", &[50], now - chrono::Duration::hours(FOLDER_ONLY_HOURS + 1));
        assert_eq!(current(&path, &[60], Some("/work/app"), now), None);
        record(&path, "recent", Some("/work/app"), "cursor", &[51], now - chrono::Duration::hours(1));
        assert_eq!(current(&path, &[60], Some("/work/app"), now), Some("recent".to_string()));
        assert_eq!(current(&path, &[60], Some("/work/other"), now), None);
        assert_eq!(current(&path, &[], None, now), None);
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn a_memory_is_stamped_with_the_conversation_it_is_written_in() {
        let now = chrono::Utc::now();
        let stamped = with_origin(Some(&serde_json::json!({ "location": "src/main.rs" })), Some("conv-1"), now).expect("stamped");
        assert_eq!(stamped["origin"]["conversation_id"], "conv-1");
        assert_eq!(stamped["origin"]["at"], now.to_rfc3339());
        assert_eq!(stamped["location"], "src/main.rs");
        assert_eq!(with_origin(None, None, now), None, "no known conversation, nothing added");
        let given = serde_json::json!({ "origin": { "conversation_id": "other" } });
        assert_eq!(with_origin(Some(&given), Some("conv-1"), now), None, "an origin given by the caller is kept");
    }

    #[test]
    fn a_session_is_noted_once_and_forgotten_after_a_week() {
        let path = temp("ttl");
        let now = chrono::Utc::now();
        record(&path, "stale", None, "claude", &[7], now - chrono::Duration::hours(TTL_HOURS + 1));
        record(&path, "live", None, "claude", &[8], now - chrono::Duration::hours(2));
        record(&path, "live", None, "claude", &[9], now);
        let entries = read(&path);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].ancestors, vec![9]);
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }
}
