//! Cursor Agent hooks (https://cursor.com/docs/agent/hooks).
//!
//! `MemoryPilot --hook session-start` : reads the hook payload on stdin and
//! answers `{"additional_context": "<compact recall>"}` so every new Cursor
//! conversation starts with the project's memories already loaded.
//!
//! `MemoryPilot --hook user-prompt` (Claude Code `UserPromptSubmit`) :
//! searches the store with the prompt the user just sent and injects the few
//! memories that answer it, in full, unless the ranker abstains. A memory
//! is injected once per session.
//!
//! `MemoryPilot --hook session-index` : the session is over; brings the
//! conversation index up to date in a detached `--hook-index` child, so
//! `find_conversation` sees it. Nothing is distilled into memories.
//!
//! `MemoryPilot --hook stop` : answers `{}` immediately, then hands the
//! transcript to a detached `MemoryPilot --hook-ingest` child that distills
//! only the lines added since the previous turn (anti-pollution, see
//! `ingest_session_transcript`). Opt-in: `--install-hooks --with-ingest`.
//!
//! `MemoryPilot --hook session-end` : the session is over; ingests the lines
//! the last `stop` did not cover (same incremental ingest), then indexes.
//!
//! `MemoryPilot --install-hooks [--with-ingest]` : merges the entries into
//! Cursor, Claude Code and Codex settings, keeping any hook the user already
//! has. Without `--with-ingest` the distilling hooks are removed.
//!
//! Hooks must never break Cursor: every failure path prints `{}` and exits 0.

use crate::db::{Database, MemoryScope, RecallMode, SearchResult};
use serde_json::{json, Value};
use sha2::Digest;
use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};

/// Hard cap on the context injected at session start (~2k tokens).
const MAX_CONTEXT_CHARS: usize = 8000;
/// Hard cap on the memories injected for one prompt (~1k tokens).
const MAX_PROMPT_CONTEXT_CHARS: usize = 4000;
const MAX_PROMPT_MEMORY_CHARS: usize = 1200;
const MAX_PROMPT_MEMORIES: usize = 3;
/// Memories injected per session, all prompts together: past it the hook
/// stops searching. Bounds what the recall adds to a long conversation
/// (10 x 1,200 characters, about 3,000 tokens at most).
const MAX_SESSION_MEMORIES: usize = 10;
/// A hit is injected when it scores at least this share of the top hit.
/// Only the cross-encoder rerank (the default search mode) spreads the
/// scores enough for this to select: without it the fused ranks are flat
/// (0.04, 0.03, 0.03), so only the top hit is a candidate then.
const PROMPT_RELATIVE_SCORE: f64 = 0.6;
/// Shorter prompts ("ok", "vas-y") carry no topic to search for.
const PROMPT_MIN_WORDS: usize = 4;
const PROMPT_QUERY_CHARS: usize = 600;
const PROMPT_SEARCH_LIMIT: usize = 12;
/// One file per session: ids of the memories already injected in it.
const PROMPT_SEEN_DIR: &str = "prompt-injections";
const INDEX_LOCK_KEY: &str = "index-conversations";
/// Hard cap on the text sent to the distiller per turn.
const MAX_INGEST_BYTES: usize = 400_000;
const STATE_FILE: &str = "cursor-hooks-state.json";
/// Workspace → time of the last successful session-start injection. McpHub
/// reads it to skip a second full recall in the same session.
const INJECTIONS_FILE: &str = "session-hook-injections.json";
/// Injections older than this are dropped when the file is rewritten.
const INJECTION_TRACE_TTL_HOURS: i64 = 24;
const LOG_FILE: &str = "cursor-hooks.log";
/// One marker per session id: Cursor also runs the Claude Code hooks, the
/// second session-start of the same session stands down.
const SESSION_MARKS_DIR: &str = "session-start-marks";
/// A same-session start after this delay (resume, compaction) is served again.
const SESSION_MARK_WINDOW_SECS: u64 = 120;
/// Marks older than this are swept.
const SESSION_MARK_TTL_SECS: u64 = 24 * 3600;
/// Stop and SessionEnd can fire together: ingests of one transcript queue up.
const INGEST_LOCK_WAIT_SECS: u64 = 120;
const INGEST_LOCK_STALE_SECS: u64 = 600;
/// A pid lock file found empty is being written by its owner for this long.
const LOCK_PID_GRACE_SECS: u64 = 5;

fn open_db() -> Result<Database, String> {
    match std::env::var("MEMORYPILOT_DB") {
        Ok(path) if !path.trim().is_empty() => Database::open_at(Path::new(&path)),
        _ => Database::open(),
    }
}

/// Where a hook file lives: next to `MEMORYPILOT_DB` when it is set (tests),
/// else in `area` of the layout (state/ or logs/), whose folder is created.
pub(crate) fn hook_path(area: memorypilot_core::layout::Area, name: &str) -> PathBuf {
    let path = match std::env::var("MEMORYPILOT_DB") {
        Ok(db) if !db.trim().is_empty() => Path::new(&db)
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."))
            .join(name),
        _ => memorypilot_core::layout::path(area, name).unwrap_or_else(|| PathBuf::from(".").join(name)),
    };
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    path
}

fn log(message: &str) {
    use std::io::Write;
    let path = hook_path(memorypilot_core::layout::Area::Logs, LOG_FILE);
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
    if matches!(event, "session-start" | "sessionStart" | "SessionStart" | "user-prompt" | "userPromptSubmit" | "UserPromptSubmit") {
        note_active_conversation(&agent, &payload);
    }
    let output = match event {
        "session-start" | "sessionStart" | "SessionStart" => {
            if !claim_session_start(&hook_path(memorypilot_core::layout::Area::State, SESSION_MARKS_DIR), &payload) {
                println!("{}", json!({}));
                return;
            }
            spawn_detached(&["--sync-agents"], "session-start: sync-agents");
            if crate::auto_update::due() {
                spawn_detached(&["--auto-update"], "session-start: auto-update");
            }
            // What the background updates did since the last session, said once.
            let (for_model, for_user) = crate::auto_update::take_report().unzip();
            let context = match (session_start_context(&payload), for_model) {
                (Some(context), Some(updates)) => Some(format!("{}\n\n{}", context, updates)),
                (context, updates) => context.or(updates),
            };
            let mut output = match context {
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
            };
            // Claude Code shows `systemMessage` to the user, whatever the model does with its context.
            if let Some(updates) = for_user.filter(|_| agent == "claude") {
                output["systemMessage"] = json!(updates);
            }
            output
        }
        "user-prompt" | "userPromptSubmit" | "UserPromptSubmit" => {
            // Cursor has no per-prompt context injection.
            match (agent != "cursor").then(|| prompt_context(&payload)).flatten() {
                Some(context) => json!({
                    "hookSpecificOutput": {
                        "hookEventName": "UserPromptSubmit",
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
        "session-end" | "sessionEnd" | "SessionEnd" => {
            spawn_ingest(&payload);
            spawn_detached(&["--hook-index"], "session-end: index");
            json!({})
        }
        "session-index" => {
            spawn_detached(&["--hook-index"], "session-index");
            json!({})
        }
        other => {
            log(&format!("unknown hook event '{}'", other));
            json!({})
        }
    };
    println!("{}", output);
}

/// Leaves the trace `find_conversation` reads to know which conversation
/// its caller is in.
fn note_active_conversation(agent: &str, payload: &Value) {
    let Some(conversation) = conversation_id(payload) else { return };
    crate::active_session::record(
        &crate::active_session::state_path(),
        &conversation,
        first_workspace_root(payload).as_deref(),
        agent,
        &crate::active_session::own_ancestors(),
        chrono::Utc::now(),
    );
    // The index is told a session ran today, to say so if it never reads it.
    if let Ok(store) = memorypilot_core::conversations::ConversationStore::open_default() {
        if let Err(error) = store.record_activity(&conversation, agent, first_workspace_root(payload).as_deref(), chrono::Utc::now()) {
            log(&format!("activity: {}", error));
        }
    }
}

fn age_secs(path: &Path) -> Option<u64> {
    let modified = std::fs::metadata(path).and_then(|m| m.modified()).ok()?;
    Some(modified.elapsed().map(|d| d.as_secs()).unwrap_or(0))
}

/// True when this hook should serve the session start: no session id, or
/// the first claim of `session_id` + `source` within the window. The
/// marker is created atomically, so two hooks firing together cannot both win.
fn claim_session_start(dir: &Path, payload: &Value) -> bool {
    let Some(session) = conversation_id(payload) else {
        return true;
    };
    if std::fs::create_dir_all(dir).is_err() {
        return true;
    }
    let source = str_field(payload, "source").unwrap_or_default();
    let digest = sha2::Sha256::digest(format!("{}|{}", session, source).as_bytes());
    let name: String = digest.iter().take(16).map(|b| format!("{:02x}", b)).collect();
    let mark = dir.join(name);
    let create = || std::fs::OpenOptions::new().write(true).create_new(true).open(&mark).is_ok();
    if create() {
        sweep_old_marks(dir);
        return true;
    }
    match age_secs(&mark) {
        Some(age) if age < SESSION_MARK_WINDOW_SECS => false,
        _ => std::fs::remove_file(&mark).is_ok() && create(),
    }
}

fn sweep_old_marks(dir: &Path) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        if age_secs(&entry.path()).is_some_and(|age| age > SESSION_MARK_TTL_SECS) {
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

/// Runs `MemoryPilot <args>` detached, without waiting or failing the hook.
fn spawn_detached(args: &[&str], label: &str) {
    let exe = match std::env::current_exe() {
        Ok(exe) => exe,
        Err(error) => {
            log(&format!("{}: current_exe failed: {}", label, error));
            return;
        }
    };
    let spawned = std::process::Command::new(exe)
        .args(args)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn();
    if let Err(error) = spawned {
        log(&format!("{}: spawn failed: {}", label, error));
    }
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
            if let Some(workspace) = working_dir.as_deref() {
                record_injection(workspace);
            }
            Some(format!(
                "MemoryPilot (your persistent memory, loaded at session start):\n{}",
                truncate_chars(&text, MAX_CONTEXT_CHARS)
            ))
        }
        Err(error) => {
            log(&format!("session-start: recall failed: {}", error));
            None
        }
    }
}

/// The search query for a prompt, or `None` when it carries no topic:
/// a slash command, or fewer than `PROMPT_MIN_WORDS` words.
fn prompt_query(prompt: &str) -> Option<String> {
    let prompt = prompt.trim();
    if prompt.starts_with('/') || prompt.split_whitespace().count() < PROMPT_MIN_WORDS {
        return None;
    }
    Some(prompt.chars().take(PROMPT_QUERY_CHARS).collect())
}

/// The hits worth injecting, best first: of the opened project or global,
/// not yet injected in this session, close enough to the top hit. Sealed
/// credentials and raw transcripts never go in.
fn pick_prompt_memories<'a>(results: &'a [SearchResult], project: Option<&str>, seen: &[String]) -> Vec<&'a SearchResult> {
    let Some(top_score) = results.first().map(|hit| hit.score) else {
        return Vec::new();
    };
    let in_scope = |hit: &SearchResult| match (hit.memory.project.as_deref(), project) {
        (None, _) => true,
        (Some(of_memory), Some(opened)) => of_memory.eq_ignore_ascii_case(opened),
        (Some(_), None) => false,
    };
    results
        .iter()
        .filter(|hit| hit.score >= top_score * PROMPT_RELATIVE_SCORE)
        .filter(|hit| in_scope(hit))
        .filter(|hit| !matches!(hit.memory.kind.as_str(), "credential" | "transcript"))
        .filter(|hit| !seen.contains(&hit.memory.id))
        .take(MAX_PROMPT_MEMORIES)
        .collect()
}

fn render_prompt_memories(picked: &[&SearchResult]) -> String {
    let mut text = String::from("MemoryPilot: memories related to this message (full text; get_memory(id) for the metadata):");
    for hit in picked {
        let memory = &hit.memory;
        text.push_str(&format!(
            "\n\n[{} · {} · {} · id:{}]\n{}",
            memory.kind,
            memory.project.as_deref().unwrap_or("global"),
            memory.updated_at.chars().take(10).collect::<String>(),
            memory.id.chars().take(8).collect::<String>(),
            truncate_chars(memory.content.trim(), MAX_PROMPT_MEMORY_CHARS),
        ));
    }
    truncate_chars(&text, MAX_PROMPT_CONTEXT_CHARS)
}

/// Where the ids injected in `session` are kept.
fn prompt_seen_path(session: &str) -> PathBuf {
    let digest = sha2::Sha256::digest(session.as_bytes());
    let name: String = digest.iter().take(16).map(|b| format!("{:02x}", b)).collect();
    hook_path(memorypilot_core::layout::Area::State, PROMPT_SEEN_DIR).join(format!("{}.json", name))
}

/// How many more memories this session may get, `already` being injected.
fn session_budget_left(already: usize) -> usize {
    MAX_SESSION_MEMORIES.saturating_sub(already)
}

fn load_seen(path: &Path) -> Vec<String> {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|raw| serde_json::from_str(&raw).ok())
        .unwrap_or_default()
}

/// Memories that answer the prompt the user just sent, or `None`: no topic
/// in the prompt, nothing the ranker trusts, or all of it already injected.
fn prompt_context(payload: &Value) -> Option<String> {
    let query = prompt_query(&str_field(payload, "prompt")?)?;
    let seen_path = conversation_id(payload).map(|session| prompt_seen_path(&session));
    let mut seen = seen_path.as_deref().map(load_seen).unwrap_or_default();
    let budget = session_budget_left(seen.len());
    if budget == 0 {
        return None;
    }
    let db = match open_db() {
        Ok(db) => db,
        Err(error) => {
            log(&format!("user-prompt: db open failed: {}", error));
            return None;
        }
    };
    let project = first_workspace_root(payload).and_then(|dir| db.detect_project(&dir).ok().flatten());
    let (results, confidence) = match db.search_at(chrono::Utc::now(), &query, PROMPT_SEARCH_LIMIT, None, None, None, None) {
        Ok(found) => found,
        Err(error) => {
            log(&format!("user-prompt: search failed: {}", error));
            return None;
        }
    };
    // The ranker skips the cross-encoder when its top hit is already sure:
    // the scores are then flat fused ranks, which vouch for that hit only.
    let reranked = confidence.cross_score.is_some();
    let candidates = if reranked { &results[..] } else { &results[..results.len().min(1)] };
    let mut picked = if confidence.abstain { Vec::new() } else { pick_prompt_memories(candidates, project.as_deref(), &seen) };
    picked.truncate(budget);
    // One line per searched prompt, without its text: what the gate decided.
    log(&format!(
        "user-prompt: project={} confidence={} cosine={:.2} margin={:.2} hits={} injected={}",
        project.as_deref().unwrap_or("-"),
        confidence.level,
        confidence.top_cosine,
        confidence.margin,
        results.len(),
        picked.len()
    ));
    if picked.is_empty() {
        return None;
    }
    if let Some(path) = seen_path {
        seen.extend(picked.iter().map(|hit| hit.memory.id.clone()));
        if let Some(dir) = path.parent() {
            if std::fs::create_dir_all(dir).is_ok() {
                sweep_old_marks(dir);
            }
        }
        if let Err(error) = std::fs::write(&path, serde_json::to_string(&seen).unwrap_or_default()) {
            log(&format!("user-prompt: cannot record injected ids: {}", error));
        }
    }
    Some(render_prompt_memories(&picked))
}

/// Entry point for `--hook-index`: full catch-up of the conversation index,
/// one run at a time (a run already in progress covers this session too, or
/// the next session end does).
pub fn run_index() {
    let Some(_lock) = IngestLock::try_acquire(INDEX_LOCK_KEY) else {
        log("index: another run in progress, skipped");
        return;
    };
    memorypilot_core::secrets::ensure_known_secrets();
    crate::conversations_cli::run_index(&[]);
}

/// Runs `action` (an update run or an undo of the third-party skills and
/// plugins) alone: both rewrite the same state.
pub fn alone_on_agents_updates(action: impl FnOnce() -> i32) -> i32 {
    let Some(_lock) = IngestLock::try_acquire(crate::auto_update::LOCK_KEY) else {
        println!("An update run is in progress: try again in a moment.");
        return 1;
    };
    action()
}

/// Notes that `workspace` just got the recall. Best effort: a lost trace
/// only means McpHub recalls once more.
fn record_injection(workspace: &str) {
    let path = hook_path(memorypilot_core::layout::Area::State, INJECTIONS_FILE);
    let now = chrono::Utc::now();
    let mut traces = match read_json_or(&path, json!({})) {
        Value::Object(map) => map,
        _ => serde_json::Map::new(),
    };
    traces.retain(|_, at| {
        at.as_str()
            .and_then(|at| chrono::DateTime::parse_from_rfc3339(at).ok())
            .is_some_and(|at| now.signed_duration_since(at) < chrono::Duration::hours(INJECTION_TRACE_TTL_HOURS))
    });
    traces.insert(
        workspace.trim_end_matches('/').to_string(),
        json!(now.to_rfc3339()),
    );
    let tmp = path.with_extension("json.tmp");
    if std::fs::write(&tmp, Value::Object(traces).to_string()).is_ok() {
        let _ = std::fs::rename(tmp, path);
    }
}

fn spawn_ingest(payload: &Value) {
    let Some(transcript) = str_field(payload, "transcript_path") else {
        log("stop: no transcript_path in payload");
        return;
    };
    let conversation = conversation_id(payload).unwrap_or_default();
    let root = first_workspace_root(payload).unwrap_or_default();
    // Detached child: the agent gets its answer immediately, the distillation
    // (embedding model load + dedup) runs in the background.
    spawn_detached(&["--hook-ingest", &transcript, &conversation, &root], "stop");
}

/// Exclusive lock file, removed on drop.
struct IngestLock(PathBuf);

impl IngestLock {
    fn path(key: &str) -> PathBuf {
        let digest = sha2::Sha256::digest(key.as_bytes());
        let name: String = digest.iter().take(16).map(|b| format!("{:02x}", b)).collect();
        hook_path(memorypilot_core::layout::Area::State, &format!("ingest-{}.lock", name))
    }

    /// The lock, or `None` at once when a live run holds it. The file holds
    /// the owner's pid: a run that exited or was killed without dropping the
    /// lock is replaced at once, a long one is never taken for stale.
    fn try_acquire(key: &str) -> Option<IngestLock> {
        Self::try_acquire_at(Self::path(key))
    }

    fn try_acquire_at(path: PathBuf) -> Option<IngestLock> {
        use std::io::Write;
        for _ in 0..2 {
            if let Ok(mut file) = std::fs::OpenOptions::new().write(true).create_new(true).open(&path) {
                let _ = write!(file, "{}", std::process::id());
                return Some(IngestLock(path));
            }
            let owner = std::fs::read_to_string(&path).ok().and_then(|raw| raw.trim().parse::<u32>().ok());
            let held = match owner {
                Some(pid) => crate::update::process_alive(pid),
                // Created, pid not written yet: the owner is starting.
                None => age_secs(&path).is_some_and(|age| age < LOCK_PID_GRACE_SECS),
            };
            if held {
                return None;
            }
            let _ = std::fs::remove_file(&path);
        }
        None
    }

    fn acquire(transcript: &str) -> Option<IngestLock> {
        let path = Self::path(transcript);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(INGEST_LOCK_WAIT_SECS);
        loop {
            if std::fs::OpenOptions::new().write(true).create_new(true).open(&path).is_ok() {
                return Some(IngestLock(path));
            }
            if age_secs(&path).is_some_and(|age| age > INGEST_LOCK_STALE_SECS) {
                let _ = std::fs::remove_file(&path);
                continue;
            }
            if std::time::Instant::now() >= deadline {
                return None;
            }
            std::thread::sleep(std::time::Duration::from_millis(250));
        }
    }
}

impl Drop for IngestLock {
    fn drop(&mut self) {
        // A pid lock taken over by another run is no longer ours to remove.
        let owner = std::fs::read_to_string(&self.0).ok().and_then(|raw| raw.trim().parse::<u32>().ok());
        if owner.is_none_or(|pid| pid == std::process::id()) {
            let _ = std::fs::remove_file(&self.0);
        }
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
    let Some(_lock) = IngestLock::acquire(&transcript) else {
        log(&format!("ingest {}: another ingest still running, skipped", conversation));
        return;
    };
    memorypilot_core::secrets::ensure_known_secrets();
    match ingest_new_lines(&transcript, &conversation, &root) {
        Ok(summary) => log(&format!("ingest {}: {}", conversation, summary)),
        Err(error) => log(&format!("ingest {} failed: {}", conversation, error)),
    }
}

fn state_key(transcript: &str) -> String {
    transcript.to_string()
}

fn load_state() -> HashMap<String, usize> {
    std::fs::read_to_string(hook_path(memorypilot_core::layout::Area::State, STATE_FILE))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

fn save_state(state: &HashMap<String, usize>) {
    if let Ok(serialized) = serde_json::to_string_pretty(state) {
        let path = hook_path(memorypilot_core::layout::Area::State, STATE_FILE);
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

/// Redacts credentials before anything reaches the store: the same
/// filter as the conversation index (`memorypilot_core::secrets`).
pub fn redact_secrets(text: &str) -> String {
    memorypilot_core::secrets::redact(text).into_owned()
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

pub(crate) fn write_json_with_backup(path: &Path, value: &Value) -> Result<(), String> {
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

/// The hooks one agent gets: `(event, --hook argument, matcher, timeout)`.
/// Every event MemoryPilot ever installed is listed, with `None` for the
/// ones this agent does not get, so a reinstall also removes them.
/// The distilling hooks (`stop`, `session-end`) come only with `with_ingest`.
/// The conversation index reads Cursor and Claude Code sessions, not Codex.
fn wanted_hooks(agent: &str, with_ingest: bool) -> Vec<(&'static str, Option<&'static str>, &'static str, u64)> {
    let indexed = agent != "codex";
    let session_end = match (with_ingest, indexed) {
        (true, _) => Some("session-end"),
        (false, true) => Some("session-index"),
        (false, false) => None,
    };
    let stop = with_ingest.then_some("stop");
    if agent == "cursor" {
        return vec![
            ("sessionStart", Some("session-start"), "", 20),
            ("stop", stop, "", 10),
            ("sessionEnd", session_end, "", 10),
        ];
    }
    vec![
        ("SessionStart", Some("session-start"), "startup|resume|clear|compact", 20),
        ("UserPromptSubmit", (agent == "claude").then_some("user-prompt"), "", 10),
        ("Stop", stop, "*", 10),
        ("SessionEnd", session_end, "*", 10),
    ]
}

/// Cursor format: `hooks.<event> = [{command, timeout}]`.
fn install_cursor(exe: &str, home: &Path, with_ingest: bool) -> Result<PathBuf, String> {
    let path = home.join(".cursor").join("hooks.json");
    let mut config = read_json_or(&path, json!({ "version": 1, "hooks": {} }));
    if config.get("version").is_none() {
        config["version"] = json!(1);
    }
    let hooks = ensure_object(&mut config, "hooks");
    for (event, arg, _, timeout) in wanted_hooks("cursor", with_ingest) {
        let mut entries: Vec<Value> = hooks.get(event).and_then(|list| list.as_array()).cloned().unwrap_or_default();
        entries.retain(|h| !is_memorypilot_command(h.get("command").and_then(|c| c.as_str())));
        if let Some(arg) = arg {
            entries.push(json!({
                "command": format!("\"{}\" --hook {} --agent cursor", exe, arg),
                "timeout": timeout
            }));
        }
        if entries.is_empty() {
            hooks.remove(event);
        } else {
            hooks.insert(event.into(), Value::Array(entries));
        }
    }
    write_json_with_backup(&path, &config)?;
    Ok(path)
}

/// Claude Code and Codex format: `hooks.<Event> = [{matcher, hooks:[{type,
/// command, timeout}]}]`. Other settings and hooks are preserved.
fn install_matcher_style(exe: &str, path: &Path, agent: &str, with_ingest: bool) -> Result<PathBuf, String> {
    let mut config = read_json_or(path, json!({}));
    if !config.is_object() {
        config = json!({});
    }
    let hooks = ensure_object(&mut config, "hooks");
    for (event, arg, matcher, timeout) in wanted_hooks(agent, with_ingest) {
        let mut groups: Vec<Value> = hooks.get(event).and_then(|list| list.as_array()).cloned().unwrap_or_default();
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
        if let Some(arg) = arg {
            let mut group = json!({
                "hooks": [{
                    "type": "command",
                    "command": format!("\"{}\" --hook {} --agent {}", exe, arg, agent),
                    "timeout": timeout
                }]
            });
            // UserPromptSubmit takes no matcher.
            if !matcher.is_empty() {
                group["matcher"] = json!(matcher);
            }
            groups.push(group);
        }
        if groups.is_empty() {
            hooks.remove(event);
        } else {
            hooks.insert(event.into(), Value::Array(groups));
        }
    }
    write_json_with_backup(path, &config)?;
    Ok(path.to_path_buf())
}

/// Entry point for `--install-hooks [--with-ingest]` (alias
/// `--install-cursor-hooks`): Cursor always, Claude Code and Codex when their
/// config folder exists.
pub fn install_hooks(args: &[String]) {
    let with_ingest = args.iter().any(|a| a == "--with-ingest");
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
    let mut results = vec![("Cursor", install_cursor(&exe, &home, with_ingest))];
    if home.join(".claude").is_dir() {
        let path = home.join(".claude").join("settings.json");
        results.push(("Claude Code", install_matcher_style(&exe, &path, "claude", with_ingest)));
    }
    if home.join(".codex").is_dir() {
        let path = home.join(".codex").join("hooks.json");
        results.push(("Codex", install_matcher_style(&exe, &path, "codex", with_ingest)));
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

    fn hit(id: &str, kind: &str, project: Option<&str>, score: f64) -> SearchResult {
        SearchResult {
            memory: crate::db::Memory {
                id: id.into(),
                content: format!("contenu de {id}"),
                kind: kind.into(),
                project: project.map(String::from),
                tags: Vec::new(),
                source: "test".into(),
                importance: 3,
                expires_at: None,
                created_at: "2026-11-01T10:00:00+00:00".into(),
                updated_at: "2026-11-02T10:00:00+00:00".into(),
                metadata: None,
                last_accessed_at: None,
                access_count: 0,
            },
            score,
            sources: Vec::new(),
        }
    }

    #[test]
    fn index_lock_is_held_by_a_live_owner_only() {
        let path = std::env::temp_dir().join(format!("mp-index-lock-{}.lock", std::process::id()));
        let _ = std::fs::remove_file(&path);

        let lock = IngestLock::try_acquire_at(path.clone()).expect("free lock");
        assert_eq!(std::fs::read_to_string(&path).expect("pid"), std::process::id().to_string());
        assert!(IngestLock::try_acquire_at(path.clone()).is_none(), "held by this live process");
        drop(lock);
        assert!(!path.exists(), "released on drop");

        // An owner that exited without dropping the lock (process::exit, kill).
        std::fs::write(&path, u32::MAX.to_string()).expect("dead owner");
        let lock = IngestLock::try_acquire_at(path.clone()).expect("a dead owner's lock is replaced at once");

        // Taken over by another run meanwhile: not ours to remove any more.
        std::fs::write(&path, "1").expect("other owner");
        drop(lock);
        assert!(path.exists(), "another owner's lock is left alone");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn prompt_without_a_topic_is_not_searched() {
        assert_eq!(prompt_query("ok vas-y"), None);
        assert_eq!(prompt_query("   "), None);
        assert_eq!(prompt_query("/code-review ultra sur la branche courante"), None);
        assert_eq!(
            prompt_query("  comment est déployé le site en production ?  ").as_deref(),
            Some("comment est déployé le site en production ?")
        );
        let long = "mot ".repeat(1000);
        assert_eq!(prompt_query(&long).expect("query").chars().count(), PROMPT_QUERY_CHARS);
    }

    #[test]
    fn prompt_memories_stay_in_scope_and_are_injected_once() {
        let results = vec![
            hit("a", "decision", Some("MonApp"), 1.0),
            hit("b", "fact", Some("autre-projet"), 0.95),
            hit("c", "credential", Some("monapp"), 0.9),
            hit("d", "preference", None, 0.8),
            hit("e", "transcript", None, 0.75),
            hit("f", "fact", Some("monapp"), 0.7),
            hit("g", "fact", Some("monapp"), 0.65),
            hit("h", "fact", Some("monapp"), 0.3),
        ];
        let ids = |picked: Vec<&SearchResult>| picked.iter().map(|hit| hit.memory.id.clone()).collect::<Vec<_>>();

        assert_eq!(ids(pick_prompt_memories(&results, Some("monapp"), &[])), ["a", "d", "f"], "project or global, three at most");
        assert_eq!(ids(pick_prompt_memories(&results, Some("monapp"), &["a".into(), "d".into()])), ["f", "g"], "already injected ones are skipped");
        assert_eq!(ids(pick_prompt_memories(&results, None, &[])), ["d"], "no project detected: global only");
        assert!(pick_prompt_memories(&[], Some("monapp"), &[]).is_empty());
        assert_eq!(session_budget_left(0), MAX_SESSION_MEMORIES);
        assert_eq!(session_budget_left(MAX_SESSION_MEMORIES - 1), 1);
        assert_eq!(session_budget_left(MAX_SESSION_MEMORIES), 0);
        assert_eq!(session_budget_left(MAX_SESSION_MEMORIES + 5), 0, "a session past the cap gets nothing more");

        let text = render_prompt_memories(&pick_prompt_memories(&results, Some("monapp"), &[]));
        assert!(text.contains("[decision · MonApp · 2026-11-02 · id:a]\ncontenu de a"));
        assert!(text.contains("[preference · global · 2026-11-02 · id:d]"));
    }

    #[test]
    fn install_drops_the_distilling_hooks_unless_asked() {
        let dir = std::env::temp_dir().join(format!("mp-install-hooks-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("dir");
        let path = dir.join("settings.json");
        let commands = |path: &Path, event: &str| -> Vec<String> {
            read_json_or(path, json!({}))["hooks"][event]
                .as_array()
                .map(|groups| {
                    groups
                        .iter()
                        .flat_map(|g| g["hooks"].as_array().cloned().unwrap_or_default())
                        .filter_map(|h| h["command"].as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default()
        };

        std::fs::write(&path, r#"{"hooks":{"Stop":[{"matcher":"*","hooks":[{"type":"command","command":"other-tool --notify"}]}]}}"#).expect("write");
        install_matcher_style("/bin/MemoryPilot", &path, "claude", true).expect("install with ingest");
        assert_eq!(commands(&path, "Stop"), ["other-tool --notify", "\"/bin/MemoryPilot\" --hook stop --agent claude"]);
        assert_eq!(commands(&path, "SessionEnd"), ["\"/bin/MemoryPilot\" --hook session-end --agent claude"]);

        install_matcher_style("/bin/MemoryPilot", &path, "claude", false).expect("install");
        assert_eq!(commands(&path, "Stop"), ["other-tool --notify"], "ours is removed, the user's stays");
        assert_eq!(commands(&path, "SessionEnd"), ["\"/bin/MemoryPilot\" --hook session-index --agent claude"]);
        assert_eq!(commands(&path, "UserPromptSubmit"), ["\"/bin/MemoryPilot\" --hook user-prompt --agent claude"]);
        assert_eq!(commands(&path, "SessionStart").len(), 1);
        let config = read_json_or(&path, json!({}));
        assert!(config["hooks"]["UserPromptSubmit"][0].get("matcher").is_none());

        let codex = dir.join("hooks.json");
        install_matcher_style("/bin/MemoryPilot", &codex, "codex", false).expect("codex");
        let config = read_json_or(&codex, json!({}));
        let events: Vec<&String> = config["hooks"].as_object().expect("hooks").keys().collect();
        assert_eq!(events, ["SessionStart"], "Codex sessions are not indexed and get no per-prompt recall");
        let _ = std::fs::remove_dir_all(&dir);
    }

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
        let out = redact_secrets("clé sk-aB3xY9kL2mN8pQ4rS7tU1v et IBAN GB53REVO23012063435124 ok");
        assert!(!out.contains("sk-aB3x"));
        assert!(!out.contains("GB53REVO"));
        assert!(out.contains("[secret masqué : openai "));
        assert!(out.contains("[secret masqué : iban "));
        assert!(out.ends_with(" ok"));
        // Glued to `token=` or inside JSON quotes: the old space-only split missed both.
        let glued = redact_secrets(r#"token=ghp_aB3xY9kL2mN8pQ4rS7tU1vW5zA6bC0dE2fGh {"key":"ghp_aB3xY9kL2mN8pQ4rS7tU1vW5zA6bC0dE2fGh"}"#);
        assert!(!glued.contains("ghp_aB3x"), "{}", glued);
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
    fn session_start_is_served_once_per_session() {
        let dir = std::env::temp_dir().join(format!("mp-session-marks-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let payload = json!({"session_id": "abc", "source": "startup"});
        assert!(claim_session_start(&dir, &payload));
        assert!(!claim_session_start(&dir, &payload), "same session twice");
        assert!(!claim_session_start(&dir, &json!({"conversation_id": "abc", "source": "startup"})));
        assert!(claim_session_start(&dir, &json!({"session_id": "abc", "source": "compact"})));
        assert!(claim_session_start(&dir, &json!({"session_id": "other"})));
        assert!(claim_session_start(&dir, &json!({})), "no id: always served");
        assert!(claim_session_start(&dir, &json!({})));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn truncate_bytes_keeps_tail_on_char_boundary() {
        let out = truncate_bytes("ééééé", 5);
        assert!(out.len() <= 5 && out.chars().all(|c| c == 'é'));
    }
}
