use crate::db::Memory;

pub fn export_session_markdown(
    memories: &[Memory],
    session_id: Option<&str>,
    thread_id: Option<&str>,
    window_id: Option<&str>,
) -> String {
    let title = session_id
        .or(thread_id)
        .or(window_id)
        .unwrap_or("memorypilot-session");
    let mut output = String::new();

    output.push_str("---\n");
    output.push_str("source: MemoryPilot\n");
    output.push_str("format: session_export\n");
    output.push_str(&format!("title: \"{}\"\n", escape_yaml(title)));
    if let Some(value) = session_id {
        output.push_str(&format!("session_id: \"{}\"\n", escape_yaml(value)));
    }
    if let Some(value) = thread_id {
        output.push_str(&format!("thread_id: \"{}\"\n", escape_yaml(value)));
    }
    if let Some(value) = window_id {
        output.push_str(&format!("window_id: \"{}\"\n", escape_yaml(value)));
    }
    output.push_str(&format!("memory_count: {}\n", memories.len()));
    output.push_str("---\n\n");
    output.push_str(&format!("# MemoryPilot Session Export: {}\n\n", title));

    if let Some(capsule) = crate::session_capsule::build_extractve_capsule(memories) {
        output.push_str(&capsule);
        output.push('\n');
    }

    for memory in memories {
        output.push_str(&format!(
            "## {} · {} · importance {}\n\n",
            memory.kind, memory.id, memory.importance
        ));
        if let Some(project) = &memory.project {
            output.push_str(&format!("Project: `{}`\n\n", project));
        }
        if !memory.tags.is_empty() {
            output.push_str(&format!("Tags: `{}`\n\n", memory.tags.join("` `")));
        }
        output.push_str(&sanitize_content(&memory.content));
        output.push_str("\n\n");
    }

    output
}

pub fn scope_matches(
    memory: &Memory,
    session_id: Option<&str>,
    thread_id: Option<&str>,
    window_id: Option<&str>,
) -> bool {
    let Some(metadata) = memory.metadata.as_ref().and_then(|value| value.as_object()) else {
        return false;
    };

    matches_scope(metadata, "session_id", session_id)
        && matches_scope(metadata, "thread_id", thread_id)
        && matches_scope(metadata, "window_id", window_id)
}

fn matches_scope(
    metadata: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    expected: Option<&str>,
) -> bool {
    match expected {
        Some(expected_value) => metadata
            .get(key)
            .and_then(|value| value.as_str())
            .map(|actual| actual == expected_value)
            .unwrap_or(false),
        None => true,
    }
}

fn sanitize_content(content: &str) -> String {
    let mut cleaned = String::new();
    let mut kept_lines = 0usize;

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            if !cleaned.ends_with("\n\n") {
                cleaned.push('\n');
            }
            continue;
        }
        if looks_like_tool_noise(trimmed) {
            continue;
        }
        if kept_lines >= 80 {
            cleaned.push_str("\n[truncated]\n");
            break;
        }
        cleaned.push_str(trimmed);
        cleaned.push('\n');
        kept_lines += 1;
    }

    let trimmed = cleaned.trim();
    if trimmed.len() > 12_000 {
        format!(
            "{}\n[truncated]",
            &trimmed[..safe_boundary(trimmed, 12_000)]
        )
    } else {
        trimmed.to_string()
    }
}

fn looks_like_tool_noise(line: &str) -> bool {
    let lower = line.to_ascii_lowercase();
    lower.starts_with("{\"tool")
        || lower.starts_with("{\"role\":\"tool")
        || lower.starts_with("<system")
        || lower.starts_with("</system")
        || lower.starts_with("[tool")
        || lower.starts_with("tool_result")
        || lower.starts_with("tool:")
}

fn safe_boundary(text: &str, max: usize) -> usize {
    let mut boundary = max.min(text.len());
    while boundary > 0 && !text.is_char_boundary(boundary) {
        boundary -= 1;
    }
    boundary
}

fn escape_yaml(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn export_filters_tool_noise() {
        let memory = Memory {
            id: "m1".to_string(),
            content: "User: keep this\n{\"tool\":\"drop\"}".to_string(),
            kind: "transcript".to_string(),
            project: None,
            tags: Vec::new(),
            source: "test".to_string(),
            importance: 3,
            expires_at: None,
            created_at: String::new(),
            updated_at: String::new(),
            metadata: None,
            last_accessed_at: None,
            access_count: 0,
        };
        let output = export_session_markdown(&[memory], Some("s1"), None, None);
        assert!(output.contains("keep this"));
        assert!(!output.contains("{\"tool\""));
    }
}
