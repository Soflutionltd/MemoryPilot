//! Memory scope (session, thread, window, working dir) stored in metadata.

use super::*;

impl Database {
    pub(super) fn should_include_in_context(memory: &Memory, mode: RecallMode) -> bool {
        (mode.includes_credentials() || memory.kind != "credential")
            && !supersede::is_superseded(&memory.tags)
    }

    pub(super) fn metadata_object(
        metadata: Option<&serde_json::Value>,
    ) -> serde_json::Map<String, serde_json::Value> {
        match metadata.cloned() {
            Some(serde_json::Value::Object(object)) => object,
            Some(other) => {
                let mut object = serde_json::Map::new();
                object.insert("value".into(), other);
                object
            }
            None => serde_json::Map::new(),
        }
    }

    pub(super) fn merge_metadata(
        base: Option<&serde_json::Value>,
        overlay: Option<&serde_json::Value>,
    ) -> Option<serde_json::Value> {
        let mut object = Self::metadata_object(base);
        for (key, value) in Self::metadata_object(overlay) {
            object.insert(key, value);
        }
        if object.is_empty() {
            None
        } else {
            Some(serde_json::Value::Object(object))
        }
    }

    pub(super) fn apply_scope_to_metadata(
        metadata: Option<&serde_json::Value>,
        scope: &MemoryScope,
    ) -> Option<serde_json::Value> {
        let mut object = Self::metadata_object(metadata);
        if let Some(session_id) = &scope.session_id {
            object.insert(
                "session_id".into(),
                serde_json::Value::String(session_id.clone()),
            );
        }
        if let Some(thread_id) = &scope.thread_id {
            object.insert(
                "thread_id".into(),
                serde_json::Value::String(thread_id.clone()),
            );
        }
        if let Some(window_id) = &scope.window_id {
            object.insert(
                "window_id".into(),
                serde_json::Value::String(window_id.clone()),
            );
        }
        if object.is_empty() {
            None
        } else {
            Some(serde_json::Value::Object(object))
        }
    }

    pub(super) fn metadata_scope(metadata: Option<&serde_json::Value>) -> MemoryScope {
        let Some(serde_json::Value::Object(object)) = metadata else {
            return MemoryScope::default();
        };
        MemoryScope {
            session_id: object
                .get("session_id")
                .and_then(|value| value.as_str())
                .map(String::from),
            thread_id: object
                .get("thread_id")
                .and_then(|value| value.as_str())
                .map(String::from),
            window_id: object
                .get("window_id")
                .and_then(|value| value.as_str())
                .map(String::from),
        }
    }

    pub(super) fn scope_match_score(memory: &Memory, scope: &MemoryScope) -> i32 {
        if scope.is_empty() {
            return 0;
        }

        let memory_scope = Self::metadata_scope(memory.metadata.as_ref());
        let mut score = 0;
        if scope.thread_id.is_some() && scope.thread_id == memory_scope.thread_id {
            score += 4;
        }
        if scope.window_id.is_some() && scope.window_id == memory_scope.window_id {
            score += 2;
        }
        if scope.session_id.is_some() && scope.session_id == memory_scope.session_id {
            score += 1;
        }
        score
    }

    pub(super) fn list_scope_memories(
        &self,
        project: Option<&str>,
        scope: &MemoryScope,
        limit: usize,
    ) -> Result<Vec<Memory>, String> {
        if scope.is_empty() {
            return Ok(Vec::new());
        }

        let candidate_limit = limit.max(50).min(200);
        let (memories, _) =
            self.list_memories(project, None, Some("transcript"), candidate_limit, 0)?;
        let mut scoped: Vec<(i32, Memory)> = memories
            .into_iter()
            .filter_map(|memory| {
                let score = Self::scope_match_score(&memory, scope);
                if score > 0 {
                    Some((score, memory))
                } else {
                    None
                }
            })
            .collect();

        scoped.sort_by(|left, right| {
            right
                .0
                .cmp(&left.0)
                .then_with(|| right.1.updated_at.cmp(&left.1.updated_at))
        });

        Ok(scoped
            .into_iter()
            .take(limit)
            .map(|(_, memory)| memory)
            .collect())
    }
}
