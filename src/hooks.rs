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

/// Cursor sends `workspace_roots`, Claude Code and Codex send `cwd`.
fn first_workspace_root(payload: &Value) -> Option<String> {
    payload
        .get("workspace_roots")
        .and_then(|v| v.as_array())
        .and_then(|roots| roots.iter().find_map(|r| r.as_str()))
        .map(String::from)
        .or_else(|| str_field(payload, "cwd"))
}

/// Cursor: `conversation_id`; Claude Code and Codex: `session_id`.
fn conversation_id(payload: &Value) -> Option<String> {
    str_field(payload, "conversation_id").or_else(|| str_field(payload, "session_id"))
}

/// Which agent called the hook: explicit `--agent`, else inferred from the
/// payload (only Cursor sends `workspace_roots` / `cursor_version`).
fn detect_agent(args: &[String], payload: &Value) -> String {
    if let Some(pos) = args.iter().position(|a| a == "--agent") {
        if let Some(agent) = args.get(pos + 1) {
            return agent.to_ascii_lowercase();
        }
    }
    if payload.get("cursor_version").is_some() || payload.get("workspace_roots").is_some() {
        "cursor".into()
    } else {
        "claude".into()
    }
}

fn str_field(payload: &Value, key: &str) -> Option<String> {
    payload
        .get(key)
        .and_then(|v| v.as_str())
        .filter(|s| !s.trim().is_empty())
        .map(String::from)
}

/// Cursor also runs the hooks declared in `~/.claude/settings.json`
/// (third-party imports, on by default) next to its own `~/.cursor/hooks.json`:
/// the Claude Code entry must stand down there, or every Cursor conversation
/// gets the recall twice and is ingested twice.
fn is_cursor_running_foreign_hook(agent: &str, payload: &Value) -> bool {
    agent != "cursor"
        && (payload.get("cursor_version").is_some() || payload.get("workspace_roots").is_some())
}

/// Entry point for `--hook <event> [--agent cursor|claude|codex]`.
pub fn run_hook(event: &str, args: &[String]) {
    let payload = read_stdin_json();
    let agent = detect_agent(args, &payload);
    if is_cursor_running_foreign_hook(&agent, &payload) {
        println!("{}", json!({}));
        return;
    }
    let output = match event {
        "session-start" | "sessionStart" | "SessionStart" => {
            match session_start_context(&payload) {
                // Cursor: top-level `additional_context`. Claude Code and
                // Codex: `hookSpecificOutput.additionalContext`.
                Some(context) if agent == "cursor" => json!({ "additional_context": context }),
                Some(context) => json!({
                    "hookSpecificOutput": {
                        "hookEventName": "SessionStart",
                        "additionalContext": context
                    }
                }),
                None => json!({}),
            }
        }
        "stop" | "Stop" => {
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

fn session_start_context(payload: &Value) -> Option<String> {
    let working_dir = first_workspace_root(payload);
    let db = match open_db() {
        Ok(db) => db,
        Err(error) => {
            log(&format!("session-start: db open failed: {}", error));
            return None;
        }
    };
    let scope = MemoryScope {
        session_id: str_field(payload, "session_id"),
        thread_id: conversation_id(payload),
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
            Some(format!(
                "MemoryPilot (mémoire persistante d'Antoine, chargée automatiquement au démarrage) :\n{}",
                truncate_chars(&text, MAX_CONTEXT_CHARS)
            ))
        }
        Err(error) => {
            log(&format!("session-start: recall failed: {}", error));
            None
        }
    }
}

fn spawn_ingest(payload: &Value) {
    let Some(transcript) = str_field(payload, "transcript_path") else {
        log("stop: no transcript_path in payload");
        return;
    };
    let conversation = conversation_id(payload).unwrap_or_default();
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

/// Chunks ingested per hook run. A long backlog (first run on an old
/// conversation, missed hooks) continues on the next run: nothing is
/// skipped, the cursor only moves past what was ingested.
const MAX_CHUNKS_PER_RUN: usize = 8;

/// Splits `lines` into consecutive ranges of at most `max` bytes each
/// (a single longer line is its own range).
fn chunk_ranges(lines: &[&str], max: usize) -> Vec<std::ops::Range<usize>> {
    let mut ranges = Vec::new();
    let mut start = 0usize;
    let mut size = 0usize;
    for (index, line) in lines.iter().enumerate() {
        let len = line.len() + 1;
        if size > 0 && size + len > max {
            ranges.push(start..index);
            start = index;
            size = 0;
        }
        size += len;
    }
    if start < lines.len() {
        ranges.push(start..lines.len());
    }
    ranges
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

    let (mut candidates, mut added, mut merged, mut skipped) = (0, 0, 0, 0);
    let mut cursor = already;
    for range in chunk_ranges(&lines[already..], MAX_INGEST_BYTES).into_iter().take(MAX_CHUNKS_PER_RUN) {
        let end = already + range.end;
        let text = cursor_jsonl_to_transcript(&lines[already + range.start..end].join("\n"));
        // Only a single line longer than the budget is cut, and it is said.
        if text.len() > MAX_INGEST_BYTES {
            eprintln!("[MemoryPilot] Transcript line over {} bytes: only its end is ingested.", MAX_INGEST_BYTES);
        }
        let text = truncate_bytes(&text, MAX_INGEST_BYTES);
        if !text.trim().is_empty() {
            let report = db.ingest_session_transcript(&text, project.as_deref(), &tags, &source, &scope, false)?;
            candidates += report.distilled_candidates;
            added += report.distilled_added;
            merged += report.distilled_merged;
            skipped += report.distilled_skipped;
        }
        cursor = end;
        state.insert(key.clone(), cursor);
        save_state(&state);
    }
    Ok(format!(
        "project={} lines={}..{}{} candidates={} added={} merged={} skipped={}",
        project.as_deref().unwrap_or("global"),
        already,
        cursor,
        if cursor < lines.len() { format!(" (continues next run, {} lines left)", lines.len() - cursor) } else { String::new() },
        candidates,
        added,
        merged,
        skipped
    ))
}

/// Converts Cursor's agent-transcript JSONL into the `USER: … / ASSISTANT: …`
/// one-line-per-message format the distiller segments on. Tool calls and
/// non-text blocks are dropped, `<user_query>` wrappers stripped, obvious
/// secrets redacted.
///
/// Only the agent's FINAL answer of each turn is kept: every assistant text
/// written between tool calls is working narration ("I'm checking…",
/// "Since he wants…") and polluted the store in production. The last
/// assistant text before the next user message (or the end of the file,
/// the stop hook fires once the turn is over) is the answer.
pub fn cursor_jsonl_to_transcript(jsonl: &str) -> String {
    let mut out = String::new();
    let mut pending_answer: Option<String> = None;
    let flush = |out: &mut String, answer: &mut Option<String>| {
        if let Some(text) = answer.take() {
            out.push_str("ASSISTANT: ");
            out.push_str(&text);
            out.push('\n');
        }
    };
    for line in jsonl.lines() {
        let Ok(obj) = serde_json::from_str::<Value>(line.trim()) else {
            continue;
        };
        // Claude Code marks injected/meta lines (command output, caveats).
        if obj.get("isMeta").and_then(|v| v.as_bool()) == Some(true)
            || obj.get("isSidechain").and_then(|v| v.as_bool()) == Some(true)
        {
            continue;
        }
        // Cursor: {role, message:{content}}. Claude Code: {type, message:{role,
        // content}}. Codex rollout: {type:"response_item", payload:{type:
        // "message", role, content}}.
        let message = obj
            .get("payload")
            .filter(|p| p.get("type").and_then(|t| t.as_str()) == Some("message"))
            .or_else(|| obj.get("message"))
            .cloned()
            .unwrap_or(Value::Null);
        let raw_role = obj
            .get("role")
            .and_then(|v| v.as_str())
            .or_else(|| message.get("role").and_then(|v| v.as_str()));
        let role = match raw_role {
            Some("user") => "USER",
            Some("assistant") => "ASSISTANT",
            _ => continue,
        };
        let content = message.get("content").cloned().unwrap_or(Value::Null);
        let mut text = String::new();
        match content {
            Value::String(s) => text.push_str(&s),
            Value::Array(blocks) => {
                for block in blocks {
                    let kind = block.get("type").and_then(|v| v.as_str());
                    if matches!(kind, Some("text") | Some("input_text") | Some("output_text")) {
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
        let redacted = redact_secrets(&flattened);
        if role == "ASSISTANT" {
            pending_answer = Some(redacted);
            continue;
        }
        flush(&mut out, &mut pending_answer);
        out.push_str("USER: ");
        out.push_str(&redacted);
        out.push('\n');
    }
    flush(&mut out, &mut pending_answer);
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

fn is_memorypilot_command(command: Option<&str>) -> bool {
    command
        .map(|c| c.contains("MemoryPilot") && c.contains("--hook"))
        .unwrap_or(false)
}

fn read_json_or(path: &Path, default: Value) -> Value {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or(default)
}

fn write_json_with_backup(path: &Path, value: &Value) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    if path.exists() {
        let _ = std::fs::copy(path, path.with_extension("json.bak"));
    }
    let rendered = serde_json::to_string_pretty(value).map_err(|e| e.to_string())?;
    std::fs::write(path, rendered).map_err(|e| e.to_string())
}

fn ensure_object<'a>(value: &'a mut Value, key: &str) -> &'a mut serde_json::Map<String, Value> {
    if !value.get(key).map(|v| v.is_object()).unwrap_or(false) {
        value[key] = json!({});
    }
    value[key].as_object_mut().expect("object")
}

/// Cursor format: `hooks.<event> = [{command, timeout}]`.
fn install_cursor(exe: &str, home: &Path) -> Result<PathBuf, String> {
    let path = home.join(".cursor").join("hooks.json");
    let mut config = read_json_or(&path, json!({ "version": 1, "hooks": {} }));
    if config.get("version").is_none() {
        config["version"] = json!(1);
    }
    let hooks = ensure_object(&mut config, "hooks");
    for (event, arg, timeout) in [
        ("sessionStart", "session-start", 20),
        ("stop", "stop", 10),
    ] {
        let list = hooks.entry(event).or_insert_with(|| json!([]));
        if !list.is_array() {
            *list = json!([]);
        }
        let arr = list.as_array_mut().expect("array");
        arr.retain(|h| !is_memorypilot_command(h.get("command").and_then(|c| c.as_str())));
        arr.push(json!({
            "command": format!("\"{}\" --hook {} --agent cursor", exe, arg),
            "timeout": timeout
        }));
    }
    write_json_with_backup(&path, &config)?;
    Ok(path)
}

/// Claude Code and Codex format: `hooks.<Event> = [{matcher, hooks:[{type,
/// command, timeout}]}]`. Other settings and hooks are preserved.
fn install_matcher_style(exe: &str, path: &Path, agent: &str) -> Result<PathBuf, String> {
    let mut config = read_json_or(path, json!({}));
    if !config.is_object() {
        config = json!({});
    }
    let hooks = ensure_object(&mut config, "hooks");
    for (event, arg, matcher, timeout) in [
        ("SessionStart", "session-start", "startup|resume|clear|compact", 20),
        ("Stop", "stop", "*", 10),
    ] {
        let list = hooks.entry(event).or_insert_with(|| json!([]));
        if !list.is_array() {
            *list = json!([]);
        }
        let groups = list.as_array_mut().expect("array");
        // Drop our previous entries, then groups left empty.
        for group in groups.iter_mut() {
            if let Some(inner) = group.get_mut("hooks").and_then(|h| h.as_array_mut()) {
                inner.retain(|h| !is_memorypilot_command(h.get("command").and_then(|c| c.as_str())));
            }
        }
        groups.retain(|g| {
            g.get("hooks")
                .and_then(|h| h.as_array())
                .map(|h| !h.is_empty())
                .unwrap_or(true)
        });
        groups.push(json!({
            "matcher": matcher,
            "hooks": [{
                "type": "command",
                "command": format!("\"{}\" --hook {} --agent {}", exe, arg, agent),
                "timeout": timeout
            }]
        }));
    }
    write_json_with_backup(path, &config)?;
    Ok(path.to_path_buf())
}

/// Entry point for `--install-hooks` (alias `--install-cursor-hooks`):
/// Cursor always, Claude Code and Codex when their config folder exists.
pub fn install_hooks() {
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
    let mut results = vec![("Cursor", install_cursor(&exe, &home))];
    if home.join(".claude").is_dir() {
        let path = home.join(".claude").join("settings.json");
        results.push(("Claude Code", install_matcher_style(&exe, &path, "claude")));
    }
    if home.join(".codex").is_dir() {
        let path = home.join(".codex").join("hooks.json");
        results.push(("Codex", install_matcher_style(&exe, &path, "codex")));
    }
    let mut failed = false;
    for (agent, result) in results {
        match result {
            Ok(path) => println!("{} hooks installed in {}", agent, path.display()),
            Err(error) => {
                failed = true;
                eprintln!("{} hooks failed: {}", agent, error);
            }
        }
    }
    if failed {
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_ranges_cover_every_line_in_order() {
        let lines: Vec<String> = (0..1000).map(|i| format!("line {i} {}", "x".repeat(i % 50))).collect();
        let refs: Vec<&str> = lines.iter().map(String::as_str).collect();
        let ranges = chunk_ranges(&refs, 4_000);
        assert!(ranges.len() > 1);
        let mut next = 0;
        for range in &ranges {
            assert_eq!(range.start, next, "gap or overlap");
            assert!(range.end > range.start);
            let bytes: usize = refs[range.clone()].iter().map(|l| l.len() + 1).sum();
            assert!(bytes <= 4_000 || range.len() == 1);
            next = range.end;
        }
        assert_eq!(next, refs.len());
        let long = "y".repeat(10_000);
        assert_eq!(chunk_ranges(&["a", &long, "b"], 4_000), vec![0..1, 1..2, 2..3]);
    }

    #[test]
    fn converts_cursor_jsonl() {
        let jsonl = r#"{"role":"user","message":{"content":[{"type":"text","text":"<user_query>\nOn garde Postgres pour la prod.\n</user_query>"}]}}
{"role":"assistant","message":{"content":[{"type":"text","text":"I'm checking the config first."},{"type":"tool_use","name":"x"}]}}
{"role":"assistant","message":{"content":[{"type":"text","text":"D'accord.\nJe note."}]}}
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
    fn converts_claude_code_and_codex_jsonl() {
        let jsonl = r#"{"type":"user","message":{"role":"user","content":"On passe sur Resend pour les emails."}}
{"type":"user","isMeta":true,"message":{"role":"user","content":"Caveat: ignored"}}
{"type":"user","message":{"role":"user","content":[{"type":"tool_result","content":"x"}]}}
{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"Noté."},{"type":"tool_use","name":"Bash"}]}}
{"type":"response_item","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Codex ok."}]}}"#;
        assert_eq!(
            cursor_jsonl_to_transcript(jsonl),
            // "Noté." is mid-turn narration: only the turn's last answer stays.
            "USER: On passe sur Resend pour les emails.\nASSISTANT: Codex ok.\n"
        );
    }

    #[test]
    fn agent_narration_is_filtered() {
        use crate::db::Database;
        assert!(Database::is_agent_narration("checking licenses confirms no gpl in the runtime."));
        assert!(Database::is_agent_narration("that leaves 4 remaining high-severity issues."));
        assert!(Database::is_agent_narration("je vais vérifier le build maintenant."));
        assert!(!Database::is_agent_narration("better-auth 1.7 keeps identical string keys, so 2fa data stays readable."));
        assert!(!Database::is_agent_narration("le déploiement passe par launchctl kickstart."));
    }

    #[test]
    fn detects_agent_from_payload() {
        assert_eq!(detect_agent(&[], &json!({"workspace_roots": ["/x"]})), "cursor");
        assert_eq!(detect_agent(&[], &json!({"cwd": "/x"})), "claude");
        let args = vec!["--agent".to_string(), "codex".to_string()];
        assert_eq!(detect_agent(&args, &json!({})), "codex");
    }

    #[test]
    fn claude_hook_stands_down_inside_cursor() {
        let cursor_payload = json!({"cursor_version": "3.20.1", "workspace_roots": ["/x"]});
        assert!(is_cursor_running_foreign_hook("claude", &cursor_payload));
        assert!(!is_cursor_running_foreign_hook("cursor", &cursor_payload));
        assert!(!is_cursor_running_foreign_hook("claude", &json!({"session_id": "s", "cwd": "/x"})));
    }

    #[test]
    fn truncate_bytes_keeps_tail_on_char_boundary() {
        let out = truncate_bytes("ééééé", 5);
        assert!(out.len() <= 5 && out.chars().all(|c| c == 'é'));
    }
}
