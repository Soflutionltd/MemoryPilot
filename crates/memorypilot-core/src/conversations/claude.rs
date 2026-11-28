//! Reads Claude Code session logs (`~/.claude/projects/**/<session>.jsonl`)
//! and the Claude desktop agent sessions, which use the same format under
//! `Application Support/Claude/local-agent-mode-sessions`. Conversations
//! held on claude.ai are not on disk and cannot be read here.

use super::{fold_timestamp, Message, Role, SourceConversation};
use serde::Deserialize;
use serde_json::Value;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};

pub fn default_roots() -> Vec<PathBuf> {
    if let Ok(paths) = std::env::var("MEMORYPILOT_CLAUDE_ROOTS") {
        return std::env::split_paths(&paths).collect();
    }
    let Some(home) = dirs::home_dir() else { return Vec::new() };
    let mut roots = vec![home.join(".claude/projects")];
    if cfg!(target_os = "macos") {
        roots.push(home.join("Library/Application Support/Claude/local-agent-mode-sessions"));
    }
    roots.into_iter().filter(|root| root.exists()).collect()
}

pub struct SessionFile {
    pub path: PathBuf,
    /// The session the file holds: its name. Claude Code names the file
    /// after the session, and its hooks report that id.
    pub id: String,
    pub version: String,
}

/// Bumped when the reader changes what it takes from a file, so that
/// every session is read again once.
const READER_VERSION: u32 = 3;

/// Session logs under `roots`, and what could not be listed (a folder or
/// file that cannot be read), so nothing is left out without a word.
/// `audit.jsonl` files from the desktop app are event logs, not
/// conversations.
pub fn list_sessions(roots: &[PathBuf]) -> (Vec<SessionFile>, Vec<String>) {
    let mut files = Vec::new();
    let mut skipped = Vec::new();
    for root in roots {
        collect(root, 0, &mut files, &mut skipped);
    }
    (files, skipped)
}

const MAX_DEPTH: usize = 9;

fn collect(dir: &Path, depth: usize, out: &mut Vec<SessionFile>, skipped: &mut Vec<String>) {
    if depth > MAX_DEPTH {
        // Plugin and skill folders go deeper with no session in them:
        // only a folder that holds session logs is worth a word.
        let logs = count_logs(dir);
        if logs > 0 {
            skipped.push(format!("{} Claude session logs under {} not read: deeper than {} folders", logs, dir.display(), MAX_DEPTH));
        }
        return;
    }
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(error) => {
            skipped.push(format!("Claude sessions under {} not read: {}", dir.display(), error));
            return;
        }
    };
    for entry in entries {
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) => {
                skipped.push(format!("An entry of {} not read: {}", dir.display(), error));
                continue;
            }
        };
        let path = entry.path();
        let file_type = match entry.file_type() {
            Ok(file_type) => file_type,
            Err(error) => {
                skipped.push(format!("{} not read: {}", path.display(), error));
                continue;
            }
        };
        if file_type.is_dir() {
            if entry.file_name() != "subagents" {
                collect(&path, depth + 1, out, skipped);
            }
            continue;
        }
        let is_session = path.extension().is_some_and(|e| e == "jsonl")
            && path.file_name().is_some_and(|n| n != "audit.jsonl");
        if !is_session {
            continue;
        }
        let meta = match entry.metadata() {
            Ok(meta) => meta,
            Err(error) => {
                skipped.push(format!("Claude session {} not read: {}", path.display(), error));
                continue;
            }
        };
        let modified = meta
            .modified()
            .ok()
            .and_then(|m| m.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_millis())
            .unwrap_or(0);
        let id = path.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
        out.push(SessionFile { path, id, version: format!("{}:{}:{}", READER_VERSION, meta.len(), modified) });
    }
}

/// Session logs anywhere under `dir`.
fn count_logs(dir: &Path) -> usize {
    let Ok(entries) = std::fs::read_dir(dir) else { return 0 };
    entries
        .flatten()
        .map(|entry| {
            let path = entry.path();
            match entry.file_type() {
                Ok(kind) if kind.is_dir() => count_logs(&path),
                Ok(_) if path.extension().is_some_and(|e| e == "jsonl") && path.file_name().is_some_and(|n| n != "audit.jsonl") => 1,
                _ => 0,
            }
        })
        .sum()
}

/// Desktop session titles, keyed by the Claude Code session id they wrap.
pub fn desktop_titles(roots: &[PathBuf]) -> HashMap<String, String> {
    let mut titles = HashMap::new();
    for root in roots {
        let mut stack = vec![(root.clone(), 0usize)];
        while let Some((dir, depth)) = stack.pop() {
            let Ok(entries) = std::fs::read_dir(&dir) else { continue };
            for entry in entries.flatten() {
                let path = entry.path();
                let name = entry.file_name().to_string_lossy().to_string();
                if path.is_dir() && depth < 3 && !name.starts_with('.') {
                    stack.push((path, depth + 1));
                } else if name.starts_with("local_") && name.ends_with(".json") {
                    let Ok(raw) = std::fs::read(&path) else { continue };
                    let Ok(meta) = serde_json::from_slice::<DesktopMeta>(&raw) else { continue };
                    if let (Some(id), Some(title)) = (meta.cli_session_id, meta.title) {
                        if !title.trim().is_empty() {
                            titles.insert(id, title);
                        }
                    }
                }
            }
        }
    }
    titles
}

#[derive(Deserialize)]
struct DesktopMeta {
    #[serde(rename = "cliSessionId")]
    cli_session_id: Option<String>,
    title: Option<String>,
}

#[derive(Deserialize)]
struct Line {
    #[serde(rename = "type")]
    kind: Option<String>,
    uuid: Option<String>,
    #[serde(rename = "sessionId")]
    session_id: Option<String>,
    timestamp: Option<String>,
    cwd: Option<String>,
    #[serde(rename = "isSidechain", default)]
    is_sidechain: bool,
    #[serde(rename = "isMeta", default)]
    is_meta: bool,
    message: Option<LineMessage>,
    summary: Option<String>,
}

#[derive(Deserialize)]
struct LineMessage {
    content: Option<Value>,
}

pub struct Session {
    pub conversation: SourceConversation,
    pub messages: Vec<Message>,
    /// Days (UTC, `YYYY-MM-DD`) on which the file holds a request or an
    /// answer, tool results included.
    pub days: BTreeSet<String>,
    /// Lines that are not JSON, the last one excepted while it is being
    /// written.
    pub unreadable_lines: usize,
}

/// The files of one session, together. Two files carry the same session id
/// when a session went on from another folder: read apart, each took the
/// origin from the other at every sync, and only the messages of the last
/// one read stayed. Files without an id in their name stay alone.
pub fn same_session(files: &[SessionFile]) -> Vec<Vec<&SessionFile>> {
    let mut groups: Vec<Vec<&SessionFile>> = Vec::new();
    let mut place: HashMap<&str, usize> = HashMap::new();
    for file in files {
        match place.get(file.id.as_str()).filter(|_| !file.id.is_empty()) {
            Some(index) => groups[*index].push(file),
            None => {
                place.insert(file.id.as_str(), groups.len());
                groups.push(vec![file]);
            }
        }
    }
    for group in &mut groups {
        group.sort_by(|left, right| left.path.cmp(&right.path));
    }
    groups
}

/// The version of a session: that of its file, or of all its files.
pub fn group_version(group: &[&SessionFile]) -> String {
    group.iter().map(|file| file.version.as_str()).collect::<Vec<_>>().join("+")
}

/// A session read from all its files.
pub struct MergedSession {
    pub conversation: SourceConversation,
    pub messages: Vec<Message>,
    pub days: BTreeSet<String>,
    /// Files holding lines that are not JSON, and how many.
    pub unreadable: Vec<(PathBuf, usize)>,
}

/// Reads the files of one session (see `same_session`) as one: each message
/// once, in the order it was said.
pub fn read_sessions(
    group: &[&SessionFile],
    titles: &HashMap<String, String>,
    known_sessions: &HashSet<String>,
) -> Result<MergedSession, String> {
    let mut merged: Option<MergedSession> = None;
    for file in group {
        let session = read_session(file, titles, known_sessions)?;
        let unreadable = (session.unreadable_lines > 0).then(|| (file.path.clone(), session.unreadable_lines));
        match merged.as_mut() {
            None => {
                merged = Some(MergedSession {
                    conversation: session.conversation,
                    messages: session.messages,
                    days: session.days,
                    unreadable: unreadable.into_iter().collect(),
                });
            }
            Some(whole) => {
                let held: HashSet<String> = whole.messages.iter().map(|message| message.key.clone()).collect();
                whole.messages.extend(session.messages.into_iter().filter(|message| !held.contains(&message.key)));
                whole.days.extend(session.days);
                whole.unreadable.extend(unreadable);
                let earlier = match (&session.conversation.created_at, &whole.conversation.created_at) {
                    (Some(this), Some(first)) => this < first,
                    (Some(_), None) => true,
                    _ => false,
                };
                if earlier {
                    whole.conversation = session.conversation;
                }
            }
        }
    }
    let mut whole = merged.ok_or("no session file to read")?;
    if group.len() > 1 {
        // In the order said. A message without a time stays right after
        // the one before it in its file.
        let mut last = String::new();
        let mut timed: Vec<(String, Message)> = Vec::with_capacity(whole.messages.len());
        for message in std::mem::take(&mut whole.messages) {
            if let Some(at) = &message.at {
                last = at.clone();
            }
            timed.push((last.clone(), message));
        }
        timed.sort_by(|left, right| left.0.cmp(&right.0));
        whole.messages = timed.into_iter().map(|(_, message)| message).collect();
    }
    whole.conversation.version = group_version(group);
    Ok(whole)
}

/// Reads one session file. A session resumed from another one starts
/// with the lines of that session, under its id: those are left to the
/// file of that session when it is among `known_sessions`, so that each
/// session is indexed under its own id and its messages once.
pub fn read_session(
    file: &SessionFile,
    titles: &HashMap<String, String>,
    known_sessions: &HashSet<String>,
) -> Result<Session, String> {
    let raw = std::fs::read_to_string(&file.path)
        .map_err(|error| format!("Claude session {}: {}", file.path.display(), error))?;
    let mut first_session_id = None;
    let mut cwd = None;
    let mut created_at = None;
    let mut summary = None;
    let mut messages = Vec::new();
    let mut days = BTreeSet::new();
    let mut unreadable_lines = 0;
    let own_id = (!file.id.is_empty()).then_some(file.id.as_str());
    let complete = raw.ends_with('\n');
    let line_count = raw.lines().count();

    for (index, line) in raw.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let line = match serde_json::from_str::<Line>(line) {
            Ok(line) => line,
            Err(_) => {
                if complete || index + 1 < line_count {
                    unreadable_lines += 1;
                }
                continue;
            }
        };
        let foreign = match (own_id, line.session_id.as_deref()) {
            (Some(own), Some(other)) => other != own && known_sessions.contains(other),
            _ => false,
        };
        if foreign {
            continue;
        }
        if first_session_id.is_none() {
            first_session_id = line.session_id.clone();
        }
        if cwd.is_none() {
            cwd = line.cwd.clone();
        }
        let at = line.timestamp.as_deref().and_then(fold_timestamp);
        if created_at.is_none() {
            created_at = at.clone();
        }
        let role = match line.kind.as_deref() {
            Some("user") => Role::User,
            Some("assistant") => Role::Assistant,
            Some("summary") => {
                summary = line.summary.clone().or(summary);
                continue;
            }
            _ => continue,
        };
        if let Some(day) = at.as_deref().and_then(|at| at.get(..10)) {
            days.insert(day.to_string());
        }
        if line.is_meta || line.is_sidechain {
            continue;
        }
        let Some(content) = line.message.and_then(|m| m.content) else { continue };
        let text = text_of(&content);
        let actions = if role == Role::Assistant { actions_of(&content) } else { Vec::new() };
        // A request made of an image alone has no text to index, but it
        // opens a turn: the answer that follows is the answer to it, not
        // to the request before.
        let wordless_request = role == Role::User && has_attachment(&content);
        if text.trim().is_empty() && actions.is_empty() && !wordless_request {
            continue;
        }
        let Some(key) = line.uuid else { continue };
        messages.push(Message { key, role, at, order: None, text, actions });
    }

    let id = own_id.map(str::to_string).or(first_session_id).unwrap_or_default();
    if id.is_empty() {
        return Err(format!("Claude session {}: no session id", file.path.display()));
    }
    let title = titles.get(&id).cloned().or(summary);
    Ok(Session {
        conversation: SourceConversation {
            origin_id: id.clone(),
            root_id: id,
            tool: "claude-code",
            title,
            workspace: cwd,
            created_at,
            version: file.version.clone(),
            source_path: Some(file.path.to_string_lossy().to_string()),
        },
        messages,
        days,
        unreadable_lines,
    })
}

/// Text blocks of a message. Tool calls, tool results and thinking are
/// left out: a hit should point at what was said, not at a file the
/// agent happened to read.
fn text_of(content: &Value) -> String {
    match content {
        Value::String(text) => text.clone(),
        Value::Array(blocks) => blocks
            .iter()
            .filter(|block| block.get("type").and_then(Value::as_str) == Some("text"))
            .filter_map(|block| block.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

/// Whether the user attached something to a message (an image, a document).
/// Tool results, which also come as user lines, are not attachments.
fn has_attachment(content: &Value) -> bool {
    let Value::Array(blocks) = content else { return false };
    blocks
        .iter()
        .any(|block| matches!(block.get("type").and_then(Value::as_str), Some("image") | Some("document")))
}

/// The tool calls of a message, as action lines.
fn actions_of(content: &Value) -> Vec<String> {
    let Value::Array(blocks) = content else { return Vec::new() };
    blocks
        .iter()
        .filter(|block| block.get("type").and_then(Value::as_str) == Some("tool_use"))
        .filter_map(|block| {
            super::actions::action_line(block.get("name")?.as_str()?, block.get("input").unwrap_or(&Value::Null))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_text_turns_and_skips_tool_results() {
        let dir = std::env::temp_dir().join(format!("mp-claude-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("abc.jsonl");
        std::fs::write(
            &path,
            concat!(
                r#"{"type":"user","uuid":"u1","sessionId":"s1","cwd":"/Users/x/Cursor/App/Konta","timestamp":"2026-10-28T10:00:00Z","message":{"content":"Retrouve la facture 141"}}"#, "\n",
                r#"{"type":"assistant","uuid":"a1","sessionId":"s1","timestamp":"2026-10-28T10:00:05Z","message":{"content":[{"type":"thinking","thinking":"x"},{"type":"text","text":"Elle est payée sur Revolut."}]}}"#, "\n",
                r#"{"type":"user","uuid":"u2","sessionId":"s1","timestamp":"2026-10-28T10:00:06Z","message":{"content":[{"type":"tool_result","content":"ls"}]}}"#, "\n",
            ),
        )
        .unwrap();
        let file = SessionFile { path, id: "s1".into(), version: "1".into() };
        let session = read_session(&file, &HashMap::new(), &HashSet::new()).unwrap();
        assert_eq!(session.conversation.origin_id, "s1");
        assert_eq!(session.messages.len(), 2);
        assert_eq!(session.messages[1].text, "Elle est payée sur Revolut.");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_too_deep_folder_is_reported_only_when_it_holds_sessions() {
        let dir = std::env::temp_dir().join(format!("mp-claude-deep-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let deep = (0..=MAX_DEPTH + 1).fold(dir.clone(), |path, level| path.join(format!("d{}", level)));
        std::fs::create_dir_all(&deep).unwrap();
        std::fs::write(deep.join("schema.xsd"), "").unwrap();
        let (files, skipped) = list_sessions(std::slice::from_ref(&dir));
        assert!(files.is_empty() && skipped.is_empty(), "{skipped:?}");
        std::fs::write(deep.join("lost.jsonl"), "").unwrap();
        let (_, skipped) = list_sessions(std::slice::from_ref(&dir));
        assert_eq!(skipped.len(), 1);
        assert!(skipped[0].starts_with("1 Claude session logs under"), "{}", skipped[0]);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn glued_token_is_masked_when_indexed() {
        let dir = std::env::temp_dir().join(format!("mp-claude-secret-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("def.jsonl");
        let token = format!("ghp_{}", crate::secrets::tests::fake(36, 70));
        let line = serde_json::json!({
            "type": "user", "uuid": "u1", "sessionId": "s2", "cwd": "/Users/x/Cursor/App/Konta",
            "timestamp": "2026-10-28T10:00:00Z",
            "message": { "content": format!("Lance le push avec GITHUB_TOKEN=ghp_{}", &token[4..]) }
        });
        std::fs::write(&path, format!("{}\n", line)).unwrap();
        let session = read_session(&SessionFile { path, id: "s2".into(), version: "1".into() }, &HashMap::new(), &HashSet::new()).unwrap();
        let mut store = super::super::ConversationStore::open(&dir.join("conversations.db")).unwrap();
        store.apply(&session.conversation, true, &session.messages, &[]).unwrap();
        let text: String = store
            .conn
            .query_row("SELECT text FROM segments WHERE role = 'user'", [], |row| row.get(0))
            .unwrap();
        assert!(!text.contains(&token));
        assert!(text.ends_with(&format!("GITHUB_TOKEN={}", crate::secrets::mask("github_token", &token))));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
