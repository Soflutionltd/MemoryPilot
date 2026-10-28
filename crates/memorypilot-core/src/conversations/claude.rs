//! Reads Claude Code session logs (`~/.claude/projects/**/<session>.jsonl`)
//! and the Claude desktop agent sessions, which use the same format under
//! `Application Support/Claude/local-agent-mode-sessions`. Conversations
//! held on claude.ai are not on disk and cannot be read here.

use super::{fold_timestamp, Message, Role, SourceConversation};
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;
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
    pub version: String,
}

/// Session logs under `roots`. `audit.jsonl` files from the desktop app
/// are event logs, not conversations.
pub fn list_sessions(roots: &[PathBuf]) -> Vec<SessionFile> {
    let mut files = Vec::new();
    for root in roots {
        collect(root, 0, &mut files);
    }
    files
}

fn collect(dir: &Path, depth: usize, out: &mut Vec<SessionFile>) {
    if depth > 9 {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(file_type) = entry.file_type() else { continue };
        if file_type.is_dir() {
            if entry.file_name() != "subagents" {
                collect(&path, depth + 1, out);
            }
            continue;
        }
        let is_session = path.extension().is_some_and(|e| e == "jsonl")
            && path.file_name().is_some_and(|n| n != "audit.jsonl");
        if !is_session {
            continue;
        }
        let Ok(meta) = entry.metadata() else { continue };
        let modified = meta
            .modified()
            .ok()
            .and_then(|m| m.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_millis())
            .unwrap_or(0);
        out.push(SessionFile { path, version: format!("{}:{}", meta.len(), modified) });
    }
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
}

pub fn read_session(file: &SessionFile, titles: &HashMap<String, String>) -> Result<Session, String> {
    let raw = std::fs::read_to_string(&file.path)
        .map_err(|error| format!("Claude session {}: {}", file.path.display(), error))?;
    let mut session_id = None;
    let mut cwd = None;
    let mut created_at = None;
    let mut summary = None;
    let mut messages = Vec::new();

    for line in raw.lines() {
        let Ok(line) = serde_json::from_str::<Line>(line) else { continue };
        if session_id.is_none() {
            session_id = line.session_id.clone();
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
        if line.is_meta || line.is_sidechain {
            continue;
        }
        let Some(text) = line.message.and_then(|m| m.content).map(|c| text_of(&c)) else { continue };
        if text.trim().is_empty() {
            continue;
        }
        let Some(key) = line.uuid else { continue };
        messages.push(Message { key, role, at, order: None, text });
    }

    let fallback_id = file
        .path
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();
    let id = session_id.unwrap_or(fallback_id);
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
        let file = SessionFile { path, version: "1".into() };
        let session = read_session(&file, &HashMap::new()).unwrap();
        assert_eq!(session.conversation.origin_id, "s1");
        assert_eq!(session.messages.len(), 2);
        assert_eq!(session.messages[1].text, "Elle est payée sur Revolut.");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
