use super::{Database, Memory};

/// Every memory of the project (or of all projects), page by page: an
/// export is complete whatever the store's size.
fn all_memories(db: &Database, project: Option<&str>) -> Result<Vec<Memory>, String> {
    const PAGE: usize = 5_000;
    let mut all = Vec::new();
    let mut offset = 0usize;
    loop {
        let (page, _) = db.list_memories(project, None, None, PAGE, offset)?;
        let got = page.len();
        all.extend(page);
        if got < PAGE {
            return Ok(all);
        }
        offset += got;
    }
}

pub(super) fn export_memories(
    db: &Database,
    project: Option<&str>,
    format: &str,
) -> Result<String, String> {
    let canonical_project = Database::canonical_project(project);
    let memories = all_memories(db, canonical_project.as_deref())?;
    match format {
        "json" => {
            serde_json::to_string_pretty(&memories).map_err(|error| format!("JSON: {}", error))
        }
        "markdown" | "md" => export_memories_markdown(canonical_project.as_deref(), &memories),
        _ => Err(format!(
            "Unknown format '{}'. Use 'json' or 'markdown'.",
            format
        )),
    }
}

pub(super) fn export_session_markdown(
    db: &Database,
    session_id: Option<&str>,
    thread_id: Option<&str>,
    window_id: Option<&str>,
    project: Option<&str>,
) -> Result<String, String> {
    if session_id.is_none() && thread_id.is_none() && window_id.is_none() {
        return Err("Provide at least one of session_id, thread_id, or window_id.".to_string());
    }

    let canonical_project = Database::canonical_project(project);
    let mut memories = all_memories(db, canonical_project.as_deref())?;
    memories.retain(|memory| {
        crate::session_export::scope_matches(memory, session_id, thread_id, window_id)
    });
    memories.sort_by(|left, right| left.created_at.cmp(&right.created_at));

    Ok(crate::session_export::export_session_markdown(
        &memories, session_id, thread_id, window_id,
    ))
}

fn export_memories_markdown(project: Option<&str>, memories: &[Memory]) -> Result<String, String> {
    let mut markdown = String::new();
    let title = project.unwrap_or("All Memories");
    markdown.push_str(&format!("# MemoryPilot Export: {}\n\n", title));
    markdown.push_str(&format!("Total: {} memories\n\n", memories.len()));

    let mut by_kind: std::collections::BTreeMap<String, Vec<&Memory>> =
        std::collections::BTreeMap::new();
    for memory in memories {
        by_kind.entry(memory.kind.clone()).or_default().push(memory);
    }

    for (kind, grouped_memories) in &by_kind {
        markdown.push_str(&format!("## {} ({})\n\n", kind, grouped_memories.len()));
        for memory in grouped_memories {
            let tags = if memory.tags.is_empty() {
                String::new()
            } else {
                format!(" `{}`", memory.tags.join("` `"))
            };
            let importance = "*".repeat(memory.importance as usize);
            markdown.push_str(&format!("- [{}] {}{}\n", importance, memory.content, tags));
        }
        markdown.push('\n');
    }

    Ok(markdown)
}
