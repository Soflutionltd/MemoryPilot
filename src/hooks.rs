//! Cursor Agent hooks (https://cursor.com/docs/agent/hooks).
//!
//! `MemoryPilot --hook session-start` : reads the hook payload on stdin and
//! answers `{"additional_context": "<compact recall>"}` so every new Cursor
//! conversation starts with the project's memories already loaded.
//!
//! `MemoryPilot --hook stop` : answers `{}` immediately, then hands the
//! transcript to a detached `MemoryPilot --hook-ingest` child that distills
//! only the lines added since the previous turn (anti-pollution, see
//! `ingest_session_transcript`).
//!
//! `MemoryPilot --install-cursor-hooks` : merges both entries into
//! `~/.cursor/hooks.json`, keeping any hook the user already has.
//!
//! Hooks must never break Cursor: every failure path prints `{}` and exits 0.

use crate::db::{Database, MemoryScope, RecallMode};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};

/// Hard cap on the context injected at session start (~2k tokens).
const MAX_CONTEXT_CHARS: usize = 8000;
/// Hard cap on the text sent to the distiller per turn.
const MAX_INGEST_BYTES: usize = 400_000;
const STATE_FILE: &str = "cursor-hooks-state.json";
const LOG_FILE: &str = "cursor-hooks.log";

fn open_db() -> Result<Database, String> {
    match std::env::var("MEMORYPILOT_DB") {
        Ok(path) if !path.trim().is_empty() => Database::open_at(Path::new(&path)),
        _ => Database::open(),
    }
}

fn data_dir() -> PathBuf {
    match std::env::var("MEMORYPILOT_DB") {
        Ok(path) if !path.trim().is_empty() => Path::new(&path)
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from(".")),
        _ => dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".MemoryPilot"),
    }
}

fn log(message: &str) {
    use std::io::Write;
    let path = data_dir().join(LOG_FILE);
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        let _ = writeln!(
            file,
            "{} {}",
            chrono::Local::now().format("%Y-%m-%d %H:%M:%S"),
            message
        );
    }
}

fn read_stdin_json() -> Value {
    let mut input = String::new();
    let _ = std::io::stdin().read_to_string(&mut input);
    serde_json::from_str(&input).unwrap_or_else(|_| json!({}))
}

fn first_workspace_root(payload: &Value) -> Option<String> {
    payload
        .get("workspace_roots")
        .and_then(|v| v.as_array())
        .and_then(|roots| roots.iter().find_map(|r| r.as_str()))
        .map(String::from)
}

fn str_field(payload: &Value, key: &str) -> Option<String> {
    payload
        .get(key)
        .and_then(|v| v.as_str())
        .filter(|s| !s.trim().is_empty())
        .map(String::from)
}

/// Entry point for `--hook <event>`.
pub fn run_hook(event: &str) {
    let payload = read_stdin_json();
    let output = match event {
        "session-start" | "sessionStart" => session_start(&payload),
        "stop" => {
            spawn_ingest(&payload);
            json!({})
        }
        other => {
            log(&format!("unknown hook event '{}'", other));
            json!({})
        }
    };
    println!("{}", output);
}

fn session_start(payload: &Value) -> Value {
    let working_dir = first_workspace_root(payload);
    let db = match open_db() {
        Ok(db) => db,
        Err(error) => {
            log(&format!("session-start: db open failed: {}", error));
            return json!({});
        }
    };
    let scope = MemoryScope {
        session_id: str_field(payload, "session_id"),
        thread_id: str_field(payload, "conversation_id"),
        window_id: None,
    };
    let recalled = db.recall(
        None,
        working_dir.as_deref(),
        None,
        RecallMode::Safe,
        false,
        true,
        &scope,
    );
    match recalled {
        Ok(ctx) => {
            let text = ctx
                .get("compact")
                .and_then(|v| v.as_str())
                .map(String::from)
                .unwrap_or_else(|| ctx.to_string());
            let context = format!(
                "MemoryPilot (mémoire persistante d'Antoine, chargée automatiquement au démarrage) :\n{}",
                truncate_chars(&text, MAX_CONTEXT_CHARS)
            );
            json!({ "additional_context": context })
        }
        Err(error) => {
            log(&format!("session-start: recall failed: {}", error));
            json!({})
        }
    }
}

fn spawn_ingest(payload: &Value) {
    let Some(transcript) = str_field(payload, "transcript_path") else {
        log("stop: no transcript_path in payload");
        return;
    };
    let conversation = str_field(payload, "conversation_id").unwrap_or_default();
    let root = first_workspace_root(payload).unwrap_or_default();
    let exe = match std::env::current_exe() {
        Ok(exe) => exe,
        Err(error) => {
            log(&format!("stop: current_exe failed: {}", error));
            return;
        }
    };
    // Detached child: Cursor gets its answer immediately, the distillation
    // (embedding model load + dedup) runs in the background.
    let spawned = std::process::Command::new(exe)
        .arg("--hook-ingest")
        .arg(&transcript)
        .arg(&conversation)
        .arg(&root)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn();
    if let Err(error) = spawned {
        log(&format!("stop: spawn failed: {}", error));
    }
}

/// Entry point for `--hook-ingest <transcript> <conversation_id> <workspace_root>`.
pub fn run_ingest(args: &[String]) {
    let Some(pos) = args.iter().position(|a| a == "--hook-ingest") else {
        return;
    };
    let transcript = args.get(pos + 1).cloned().unwrap_or_default();
    let conversation = args.get(pos + 2).cloned().unwrap_or_default();
    let root = args.get(pos + 3).cloned().unwrap_or_default();
    if transcript.is_empty() {
        return;
    }
    match ingest_new_lines(&transcript, &conversation, &root) {
        Ok(summary) => log(&format!("ingest {}: {}", conversation, summary)),
        Err(error) => log(&format!("ingest {} failed: {}", conversation, error)),
    }
}

fn state_key(transcript: &str) -> String {
    transcript.to_string()
}

fn load_state() -> HashMap<String, usize> {
    std::fs::read_to_string(data_dir().join(STATE_FILE))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

fn save_state(state: &HashMap<String, usize>) {
    if let Ok(serialized) = serde_json::to_string_pretty(state) {
        let path = data_dir().join(STATE_FILE);
        let tmp = path.with_extension("json.tmp");
        if std::fs::write(&tmp, serialized).is_ok() {
            let _ = std::fs::rename(tmp, path);
        }
    }
}

fn ingest_new_lines(transcript: &str, conversation: &str, root: &str) -> Result<String, String> {
    let raw = std::fs::read_to_string(transcript)
        .map_err(|e| format!("read '{}': {}", transcript, e))?;
    let lines: Vec<&str> = raw.lines().collect();

    let mut state = load_state();
    let key = state_key(transcript);
    let already = state.get(&key).copied().unwrap_or(0).min(lines.len());
    if already == lines.len() {
        return Ok("nothing new".into());
    }

    let text = cursor_jsonl_to_transcript(&lines[already..].join("\n"));
    let text = truncate_bytes(&text, MAX_INGEST_BYTES);
    if text.trim().is_empty() {
        state.insert(key, lines.len());
        save_state(&state);
        return Ok("no text in new lines".into());
    }

    let db = open_db()?;
    let project = if root.is_empty() {
        None
    } else {
        db.detect_project(root).ok().flatten()
    };
    let scope = MemoryScope {
        session_id: None,
        thread_id: (!conversation.is_empty()).then(|| conversation.to_string()),
        window_id: None,
    };
    let tags = vec![
        "session-ingest".to_string(),
        "distilled".to_string(),
        "cursor-hook".to_string(),
    ];
    let source = format!("cursor-hook:{}", conversation);
    let report = db.ingest_session_transcript(
        &text,
        project.as_deref(),
        &tags,
        &source,
        &scope,
        false,
    )?;

    state.insert(key, lines.len());
    save_state(&state);
    Ok(format!(
        "project={} lines={}..{} candidates={} added={} merged={} skipped={}",
        project.as_deref().unwrap_or("global"),
        already,
        lines.len(),
        report.distilled_candidates,
        report.distilled_added,
        report.distilled_merged,
        report.distilled_skipped
    ))
}

/// Converts Cursor's agent-transcript JSONL into the `USER: … / ASSISTANT: …`
/// one-line-per-message format the distiller segments on. Tool calls and
/// non-text blocks are dropped, `<user_query>` wrappers stripped, obvious
/// secrets redacted.
pub fn cursor_jsonl_to_transcript(jsonl: &str) -> String {
    let mut out = String::new();
    for line in jsonl.lines() {
        let Ok(obj) = serde_json::from_str::<Value>(line.trim()) else {
            continue;
        };
        let role = match obj.get("role").and_then(|v| v.as_str()) {
            Some("user") => "USER",
            Some("assistant") => "ASSISTANT",
            _ => continue,
        };
        let content = obj
            .get("message")
            .and_then(|m| m.get("content"))
            .cloned()
            .unwrap_or(Value::Null);
        let mut text = String::new();
        match content {
            Value::String(s) => text.push_str(&s),
            Value::Array(blocks) => {
                for block in blocks {
                    if block.get("type").and_then(|v| v.as_str()) == Some("text") {
                        if let Some(t) = block.get("text").and_then(|v| v.as_str()) {
                            if !text.is_empty() {
                                text.push(' ');
                            }
                            text.push_str(t);
                        }
                    }
                }
            }
            _ => {}
        }
        // Cursor injects system blocks (<mcp_meta_tools>, rules, reminders)
        // as "user" messages: from users keep only what they typed.
        let cleaned = if role == "USER" {
            let queries = extract_tag_contents(&text, "user_query");
            if !queries.is_empty() {
                queries.join(" ")
            } else if text.trim_start().starts_with('<') {
                continue;
            } else {
                text
            }
        } else {
            text
        };
        let flattened = cleaned.split_whitespace().collect::<Vec<_>>().join(" ");
        if flattened.is_empty() {
            continue;
        }
        out.push_str(role);
        out.push_str(": ");
        out.push_str(&redact_secrets(&flattened));
        out.push('\n');
    }
    out
}

fn extract_tag_contents(text: &str, tag: &str) -> Vec<String> {
    let open = format!("<{}>", tag);
    let close = format!("</{}>", tag);
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find(&open) {
        let after = &rest[start + open.len()..];
        match after.find(&close) {
            Some(end) => {
                out.push(after[..end].trim().to_string());
                rest = &after[end + close.len()..];
            }
            None => {
                out.push(after.trim().to_string());
                break;
            }
        }
    }
    out
}

/// Redacts tokens that look like credentials before anything reaches the
/// store: known API-key prefixes, JWTs, IBANs and long high-entropy strings.
pub fn redact_secrets(text: &str) -> String {
    text.split(' ')
        .map(|word| {
            let core = word.trim_matches(|c: char| !c.is_alphanumeric() && c != '_' && c != '-');
            if looks_secret(core) {
                word.replace(core, "[REDACTED]")
            } else {
                word.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn looks_secret(token: &str) -> bool {
    if token.len() < 16 {
        return false;
    }
    const PREFIXES: [&str; 12] = [
        "sk-", "sk_live_", "sk_test_", "rk_live_", "ghp_", "gho_", "github_pat_", "xoxb-",
        "xoxp-", "AKIA", "AIza", "re_",
    ];
    if PREFIXES.iter().any(|p| token.starts_with(p)) {
        return true;
    }
    if token.starts_with("eyJ") && token.matches('.').count() >= 2 {
        return true;
    }
    // IBAN: 2 letters, 2 digits, then 11+ alphanumerics.
    let bytes = token.as_bytes();
    if token.len() >= 15
        && token.len() <= 34
        && bytes[0].is_ascii_uppercase()
        && bytes[1].is_ascii_uppercase()
        && bytes[2].is_ascii_digit()
        && bytes[3].is_ascii_digit()
        && token[4..].chars().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit())
    {
        return true;
    }
    // Long mixed token (letters + digits, no spaces) typical of keys.
    if token.len() >= 32 && !token.contains('/') && !token.contains('.') {
        let has_digit = token.chars().any(|c| c.is_ascii_digit());
        let has_upper = token.chars().any(|c| c.is_ascii_uppercase());
        let has_lower = token.chars().any(|c| c.is_ascii_lowercase());
        let alnum = token
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-');
        if alnum && has_digit && has_upper && has_lower {
            return true;
        }
    }
    false
}

fn truncate_chars(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    let mut cut: String = text.chars().take(max).collect();
    cut.push_str("\n…");
    cut
}

/// Keeps the most recent part of the text when it exceeds `max` bytes.
fn truncate_bytes(text: &str, max: usize) -> String {
    if text.len() <= max {
        return text.to_string();
    }
    let mut start = text.len() - max;
    while !text.is_char_boundary(start) {
        start += 1;
    }
    text[start..].to_string()
}

/// Entry point for `--install-cursor-hooks`.
pub fn install_cursor_hooks() {
    let exe = match std::env::current_exe().and_then(|p| p.canonicalize()) {
        Ok(exe) => exe.to_string_lossy().to_string(),
        Err(error) => {
            eprintln!("Cannot resolve MemoryPilot path: {}", error);
            std::process::exit(1);
        }
    };
    let Some(home) = dirs::home_dir() else {
        eprintln!("Cannot find home directory");
        std::process::exit(1);
    };
    let path = home.join(".cursor").join("hooks.json");
    let mut config: Value = std::fs::read_to_string(&path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_else(|| json!({ "version": 1, "hooks": {} }));
    if config.get("version").is_none() {
        config["version"] = json!(1);
    }
    if !config.get("hooks").map(|h| h.is_object()).unwrap_or(false) {
        config["hooks"] = json!({});
    }
    let entries = [
        (
            "sessionStart",
            json!({ "command": format!("\"{}\" --hook session-start", exe), "timeout": 20 }),
        ),
        (
            "stop",
            json!({ "command": format!("\"{}\" --hook stop", exe), "timeout": 10 }),
        ),
    ];
    for (event, entry) in entries {
        let list = config["hooks"]
            .as_object_mut()
            .expect("hooks is an object")
            .entry(event)
            .or_insert_with(|| json!([]));
        if !list.is_array() {
            *list = json!([]);
        }
        let arr = list.as_array_mut().expect("array");
        // Replace any previous MemoryPilot entry, keep the user's other hooks.
        arr.retain(|h| {
            !h.get("command")
                .and_then(|c| c.as_str())
                .map(|c| c.contains("MemoryPilot") && c.contains("--hook"))
                .unwrap_or(false)
        });
        arr.push(entry);
    }
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if path.exists() {
        let _ = std::fs::copy(&path, path.with_extension("json.bak"));
    }
    match std::fs::write(&path, serde_json::to_string_pretty(&config).unwrap()) {
        Ok(()) => println!("Cursor hooks installed in {}", path.display()),
        Err(error) => {
            eprintln!("Cannot write {}: {}", path.display(), error);
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_cursor_jsonl() {
        let jsonl = r#"{"role":"user","message":{"content":[{"type":"text","text":"<user_query>\nOn garde Postgres pour la prod.\n</user_query>"}]}}
{"role":"assistant","message":{"content":[{"type":"text","text":"D'accord.\nJe note."},{"type":"tool_use","name":"x"}]}}
not json
{"role":"tool","message":{"content":"ignored"}}
{"role":"user","message":{"content":[{"type":"text","text":"<mcp_meta_tools>\nDo not call mcp_auth in parallel.\n</mcp_meta_tools>"}]}}"#;
        let out = cursor_jsonl_to_transcript(jsonl);
        assert_eq!(
            out,
            "USER: On garde Postgres pour la prod.\nASSISTANT: D'accord. Je note.\n"
        );
    }

    #[test]
    fn redacts_obvious_secrets() {
        let out = redact_secrets("clé sk-abcdefghijklmnop1234 et IBAN GB53REVO23012063435124 ok");
        assert!(!out.contains("sk-abcdef"));
        assert!(!out.contains("GB53REVO"));
        assert!(out.contains("ok"));
        assert_eq!(redact_secrets("MemoryPilot est un serveur MCP"), "MemoryPilot est un serveur MCP");
    }

    #[test]
    fn truncate_bytes_keeps_tail_on_char_boundary() {
        let out = truncate_bytes("ééééé", 5);
        assert!(out.len() <= 5 && out.chars().all(|c| c == 'é'));
    }
}
