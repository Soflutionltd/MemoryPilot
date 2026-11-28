//! What the agent did in a turn, next to what it said.
//!
//! A turn is indexed by its request and its answers. What happened in
//! between (a command run, a page opened, a form filled on a vendor's
//! site) is often the only trace that something was sent somewhere. Each
//! tool call that reaches outside or changes something becomes one short
//! line; calls that only read the workspace are left out, they are nine
//! out of ten and name nothing a later question asks about.

use serde_json::Value;

/// A line longer than this is cut: the arguments of a call are a few
/// hundred bytes, a pasted file is not an action.
pub const MAX_ACTION_CHARS: usize = 300;

/// Tools that only read or edit the workspace, by the last part of their
/// name. A call carrying a URL is kept whatever its name.
const QUIET: &[&str] = &[
    "read", "read_file", "grep", "grep_search", "glob", "glob_file_search", "ls", "list_dir", "codebase_search",
    "file_search", "edit", "edit_file", "write", "multiedit", "search_replace", "notebookedit", "delete_file",
    "todowrite", "todo_write", "update_plan", "toolsearch", "agent", "task", "skill", "read_lints", "wrap_up",
];
const QUIET_PREFIXES: &[&str] =
    &["read_", "get_", "list_", "search_", "find_", "recall", "screenshot", "snapshot", "tabs_", "mark_", "show_"];
/// Arguments that say what a call did, most telling first.
const TELLING: &[&str] = &["command", "url", "query", "q", "text", "message", "body", "title", "prompt"];

/// The last part of a tool name: `mcp__server__tool` and `server.tool`
/// are named after their tool.
fn short_name(name: &str) -> &str {
    name.rsplit("__").next().and_then(|part| part.rsplit('.').next()).unwrap_or(name)
}

fn is_quiet(name: &str) -> bool {
    let short = short_name(name).to_lowercase();
    QUIET.contains(&short.as_str()) || QUIET_PREFIXES.iter().any(|prefix| short.starts_with(prefix))
}

/// The text on one line, secrets masked, cut at `max` characters. Masked
/// before the cut: a secret that straddles it would otherwise stay in clear,
/// too short for the filter to recognise once the line is stored.
fn one_line(text: &str, max: usize) -> String {
    let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let flat = crate::secrets::redact(&flat);
    match flat.char_indices().nth(max) {
        Some((index, _)) => format!("{}…", &flat[..index]),
        None => flat.into_owned(),
    }
}

/// Addresses found in a text, without their query string or fragment.
pub fn urls(text: &str) -> Vec<String> {
    let mut found = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find("http") {
        let candidate = &rest[start..];
        let scheme = if candidate.starts_with("https://") { 8 } else if candidate.starts_with("http://") { 7 } else { 0 };
        if scheme == 0 {
            rest = &candidate[4..];
            continue;
        }
        let end = candidate
            .find(|c: char| c.is_whitespace() || "\"'<>)]}`\\,".contains(c))
            .unwrap_or(candidate.len());
        let address = candidate[..end].split(['?', '#']).next().unwrap_or_default().trim_end_matches(['.', '/', ':', ';']);
        if address.len() > scheme && address[scheme..].contains('.') && !found.iter().any(|known| known == address) {
            found.push(address.to_string());
        }
        rest = &candidate[end.max(scheme)..];
    }
    found
}

/// One line for a tool call, or `None` for a call that only reads or
/// edits the workspace.
pub fn action_line(name: &str, arguments: &Value) -> Option<String> {
    let name = name.trim();
    if name.is_empty() {
        return None;
    }
    let raw = match arguments {
        Value::String(text) => text.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    };
    if is_quiet(name) && urls(&raw).is_empty() {
        return None;
    }
    // Arguments stored as a JSON string (Cursor) are read as JSON too.
    let parsed = match arguments {
        Value::String(text) => serde_json::from_str::<Value>(text).ok(),
        other => Some(other.clone()),
    };
    let telling: Vec<String> = parsed
        .as_ref()
        .and_then(Value::as_object)
        .map(|object| {
            TELLING
                .iter()
                .filter_map(|key| object.get(*key).and_then(Value::as_str))
                .filter(|value| !value.trim().is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    let said = if telling.is_empty() { raw } else { telling.join(" | ") };
    let said = one_line(&said, MAX_ACTION_CHARS);
    Some(if said.is_empty() { name.to_string() } else { format!("{}: {}", name, said) })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn a_call_that_reaches_outside_is_one_line_and_a_read_is_none() {
        assert_eq!(
            action_line("Bash", &json!({ "command": "gh issue create --repo vendor/sdk --title 'Audio on scheduled posts'" })).as_deref(),
            Some("Bash: gh issue create --repo vendor/sdk --title 'Audio on scheduled posts'")
        );
        assert_eq!(
            action_line("mcp__browser__navigate", &json!({ "url": "https://developers.example.com/community/threads/42/?ref=nav" })).as_deref(),
            Some("mcp__browser__navigate: https://developers.example.com/community/threads/42/?ref=nav")
        );
        // Cursor stores the arguments as a JSON string.
        assert_eq!(
            action_line("run_terminal_cmd", &json!("{\"command\":\"curl -X POST https://api.example.com/tickets\"}")).as_deref(),
            Some("run_terminal_cmd: curl -X POST https://api.example.com/tickets")
        );
        assert_eq!(action_line("Read", &json!({ "file_path": "/work/app/src/main.rs" })), None);
        assert_eq!(action_line("mcp__memory__search_memory", &json!({ "query": "carousel" })), None);
        // A read of a page is where the agent went: kept.
        assert!(action_line("mcp__browser__get_page_text", &json!({ "url": "https://status.example.com/" })).is_some());
        let long = action_line("Bash", &json!({ "command": "echo ".repeat(200) })).expect("line");
        assert!(long.chars().count() <= MAX_ACTION_CHARS + "Bash: …".chars().count());
    }

    #[test]
    fn a_secret_across_the_cut_is_masked_whole() {
        let token = format!("ghp_{}", crate::secrets::tests::fake(36, 81));
        for before in [MAX_ACTION_CHARS - 40, MAX_ACTION_CHARS - 12, MAX_ACTION_CHARS - 3] {
            let command = format!("{} {}", "x".repeat(before - 1), token);
            let line = action_line("Bash", &json!({ "command": command })).expect("line");
            for kept in 8..=token.len() {
                assert!(!line.contains(&token[..kept]), "{} characters of the token left at {}", kept, before);
            }
        }
        let pushed = format!("git push https://x-access-token:{}@github.com/vendor/sdk.git {}", token, "y".repeat(300));
        let line = action_line("Bash", &json!({ "command": pushed })).expect("line");
        assert!(!line.contains(&token[..8]), "{}", line);
    }

    #[test]
    fn addresses_are_read_without_their_query() {
        assert_eq!(
            urls("see https://developers.example.com/community/threads/42/?ref=nav#top, then (http://localhost:3000/x) and https://a.b."),
            vec!["https://developers.example.com/community/threads/42".to_string(), "https://a.b".to_string()]
        );
        assert!(urls("no address here, not even httpd or https://").is_empty());
    }
}
