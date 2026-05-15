use std::collections::VecDeque;
use std::sync::{LazyLock, Mutex};

use serde::Serialize;
use uuid::Uuid;

const MAX_WORKING_ITEMS: usize = 256;
const MAX_CONTENT_CHARS: usize = 4_000;

static WORKING_MEMORY: LazyLock<Mutex<VecDeque<WorkingMemoryItem>>> =
    LazyLock::new(|| Mutex::new(VecDeque::with_capacity(MAX_WORKING_ITEMS)));

#[derive(Debug, Clone, Default)]
pub struct WorkingMemoryFilter {
    pub project: Option<String>,
    pub session_id: Option<String>,
    pub thread_id: Option<String>,
    pub window_id: Option<String>,
    pub query: Option<String>,
    pub limit: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct WorkingMemoryItem {
    pub id: String,
    pub content: String,
    pub project: Option<String>,
    pub tags: Vec<String>,
    pub importance: i32,
    pub session_id: Option<String>,
    pub thread_id: Option<String>,
    pub window_id: Option<String>,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct WorkingMemoryClearReport {
    pub removed: usize,
    pub remaining: usize,
}

pub fn remember(
    content: &str,
    project: Option<String>,
    tags: Vec<String>,
    importance: i32,
    session_id: Option<String>,
    thread_id: Option<String>,
    window_id: Option<String>,
) -> Result<WorkingMemoryItem, String> {
    let content = content.trim();
    if content.is_empty() {
        return Err("content is required".into());
    }
    let content = truncate_chars(content, MAX_CONTENT_CHARS);
    let item = WorkingMemoryItem {
        id: Uuid::new_v4().to_string(),
        content,
        project: project.and_then(normalize_optional),
        tags,
        importance: importance.clamp(1, 5),
        session_id: session_id.and_then(normalize_optional),
        thread_id: thread_id.and_then(normalize_optional),
        window_id: window_id.and_then(normalize_optional),
        created_at: chrono::Utc::now().to_rfc3339(),
    };

    let mut items = WORKING_MEMORY
        .lock()
        .map_err(|_| "working memory lock poisoned".to_string())?;
    while items.len() >= MAX_WORKING_ITEMS {
        items.pop_front();
    }
    items.push_back(item.clone());
    Ok(item)
}

pub fn recall(filter: &WorkingMemoryFilter) -> Vec<WorkingMemoryItem> {
    let items = match WORKING_MEMORY.lock() {
        Ok(items) => items,
        Err(_) => return Vec::new(),
    };
    let query_terms = filter
        .query
        .as_deref()
        .map(significant_terms)
        .unwrap_or_default();

    let mut scored = items
        .iter()
        .rev()
        .filter(|item| matches_project(item, filter.project.as_deref()))
        .filter(|item| matches_scope(item, filter))
        .filter_map(|item| {
            let query_score = query_match_score(item, &query_terms);
            if !query_terms.is_empty() && query_score == 0 {
                return None;
            }
            let scope_score = scope_score(item, filter);
            Some((query_score, scope_score, item.importance, item.clone()))
        })
        .collect::<Vec<_>>();

    scored.sort_by(|left, right| {
        right
            .0
            .cmp(&left.0)
            .then_with(|| right.1.cmp(&left.1))
            .then_with(|| right.2.cmp(&left.2))
            .then_with(|| right.3.created_at.cmp(&left.3.created_at))
    });

    scored
        .into_iter()
        .map(|(_, _, _, item)| item)
        .take(filter.limit.max(1).min(50))
        .collect()
}

pub fn clear(filter: &WorkingMemoryFilter, all: bool) -> WorkingMemoryClearReport {
    let mut items = match WORKING_MEMORY.lock() {
        Ok(items) => items,
        Err(_) => {
            return WorkingMemoryClearReport {
                removed: 0,
                remaining: 0,
            }
        }
    };

    let before = items.len();
    if all {
        items.clear();
    } else {
        items.retain(|item| {
            !(matches_project(item, filter.project.as_deref()) && matches_scope(item, filter))
        });
    }

    WorkingMemoryClearReport {
        removed: before.saturating_sub(items.len()),
        remaining: items.len(),
    }
}

fn normalize_optional(value: String) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_ascii_lowercase())
    }
}

fn truncate_chars(content: &str, max_chars: usize) -> String {
    let mut output = String::new();
    for character in content.chars().take(max_chars) {
        output.push(character);
    }
    output
}

fn matches_project(item: &WorkingMemoryItem, project: Option<&str>) -> bool {
    let Some(project) = project else {
        return true;
    };
    let project = project.trim().to_ascii_lowercase();
    item.project
        .as_deref()
        .map(|item_project| item_project == project)
        .unwrap_or(true)
}

fn matches_scope(item: &WorkingMemoryItem, filter: &WorkingMemoryFilter) -> bool {
    field_matches(filter.session_id.as_deref(), item.session_id.as_deref())
        && field_matches(filter.thread_id.as_deref(), item.thread_id.as_deref())
        && field_matches(filter.window_id.as_deref(), item.window_id.as_deref())
}

fn field_matches(expected: Option<&str>, actual: Option<&str>) -> bool {
    expected
        .map(|expected| {
            actual
                .map(|actual| actual.eq_ignore_ascii_case(expected.trim()))
                .unwrap_or(false)
        })
        .unwrap_or(true)
}

fn scope_score(item: &WorkingMemoryItem, filter: &WorkingMemoryFilter) -> i32 {
    let mut score = 0;
    if filter.thread_id.is_some() && filter.thread_id == item.thread_id {
        score += 4;
    }
    if filter.window_id.is_some() && filter.window_id == item.window_id {
        score += 2;
    }
    if filter.session_id.is_some() && filter.session_id == item.session_id {
        score += 1;
    }
    score
}

fn query_match_score(item: &WorkingMemoryItem, query_terms: &[String]) -> usize {
    if query_terms.is_empty() {
        return 0;
    }
    let haystack = format!(
        "{} {}",
        item.content.to_ascii_lowercase(),
        item.tags.join(" ").to_ascii_lowercase()
    );
    query_terms
        .iter()
        .filter(|term| haystack.contains(term.as_str()))
        .count()
}

fn significant_terms(query: &str) -> Vec<String> {
    query
        .split_whitespace()
        .map(|term| {
            term.trim_matches(|character: char| {
                !character.is_alphanumeric() && character != '-' && character != '_'
            })
            .to_ascii_lowercase()
        })
        .filter(|term| term.len() >= 3)
        .filter(|term| {
            !matches!(
                term.as_str(),
                "the" | "and" | "for" | "with" | "pour" | "les"
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clear_all() {
        let _ = clear(&WorkingMemoryFilter::default(), true);
    }

    #[test]
    fn remembers_and_recalls_scoped_items() {
        clear_all();
        let _ = remember(
            "Current task is refactoring the MCP working memory.",
            Some("MemoryPilot".into()),
            vec!["scratchpad".into()],
            4,
            Some("session-a".into()),
            Some("thread-a".into()),
            None,
        )
        .expect("remember");
        let _ = remember(
            "Other project note",
            Some("Other".into()),
            Vec::new(),
            3,
            Some("session-b".into()),
            None,
            None,
        )
        .expect("remember other");

        let recalled = recall(&WorkingMemoryFilter {
            project: Some("memorypilot".into()),
            session_id: Some("session-a".into()),
            query: Some("working memory".into()),
            limit: 5,
            ..WorkingMemoryFilter::default()
        });

        assert_eq!(recalled.len(), 1);
        assert!(recalled[0].content.contains("working memory"));
    }

    #[test]
    fn clear_removes_only_matching_scope() {
        clear_all();
        let _ = remember(
            "Keep me",
            Some("memorypilot".into()),
            Vec::new(),
            3,
            Some("keep".into()),
            None,
            None,
        )
        .expect("remember keep");
        let _ = remember(
            "Remove me",
            Some("memorypilot".into()),
            Vec::new(),
            3,
            Some("remove".into()),
            None,
            None,
        )
        .expect("remember remove");

        let report = clear(
            &WorkingMemoryFilter {
                project: Some("memorypilot".into()),
                session_id: Some("remove".into()),
                ..WorkingMemoryFilter::default()
            },
            false,
        );

        assert_eq!(report.removed, 1);
        assert_eq!(
            recall(&WorkingMemoryFilter {
                project: Some("memorypilot".into()),
                limit: 10,
                ..WorkingMemoryFilter::default()
            })
            .len(),
            1
        );
    }
}
