//! Recall, project brain and project context, global prompt.

use super::*;

impl Database {
    pub(super) fn preview_snippet(content: &str) -> String {
        let trimmed = content.trim().replace('\n', " ");
        if trimmed.len() <= 120 {
            trimmed
        } else {
            format!("{}...", trimmed.chars().take(117).collect::<String>())
        }
    }

    pub(super) fn recall_explanation(
        &self,
        memory: &Memory,
        selection_source: &str,
        project: Option<&str>,
        mode: RecallMode,
        search_score: Option<f64>,
        link_boosts: &std::collections::HashMap<String, f64>,
    ) -> serde_json::Value {
        let graph_boost = link_boosts.get(&memory.id).copied().unwrap_or(0.0);
        serde_json::json!({
            "id": memory.id,
            "kind": memory.kind,
            "project": memory.project,
            "selection_source": selection_source,
            "importance": memory.importance,
            "access_count": memory.access_count,
            "updated_at": memory.updated_at,
            "search_score": search_score,
            "reason": {
                "mode": mode.as_str(),
                "project_match": project.is_some() && memory.project.as_deref() == project,
                "importance_weight": ((memory.importance as f64 / 3.0) * 100.0).round() / 100.0,
                "recency_boost": Self::recency_boost(memory),
                "access_boost": Self::access_boost(memory),
                "graph_boost": (graph_boost * 100.0).round() / 100.0,
                "age_days": Self::memory_age_days(memory),
            }
        })
    }

    // ─── GLOBAL PROMPT (auto-scan) ────────────────────

    pub fn get_global_prompt(
        &self,
        project: Option<&str>,
        working_dir: Option<&str>,
    ) -> Option<String> {
        let canonical_project = Self::canonical_project(project);
        let mut prompts: Vec<String> = Vec::new();

        // Helper to read file if modified since last cache, or use cache
        fn get_cached_prompt(path: &std::path::Path) -> Option<String> {
            if !path.exists() {
                return None;
            }
            let metadata = std::fs::metadata(path).ok()?;
            let modified = metadata.modified().ok()?;

            let mut cache = crate::PROMPT_CACHE.lock().unwrap();
            let path_str = path.to_string_lossy().to_string();

            if let Some((last_mod, content)) = cache.get(&path_str) {
                if last_mod == &modified {
                    return Some(content.clone());
                }
            }

            if let Ok(content) = std::fs::read_to_string(path) {
                cache.insert(path_str, (modified, content.clone()));
                Some(content)
            } else {
                None
            }
        }

        // 1. Check configured path
        if let Some(path_str) = self.get_config("global_prompt_path") {
            let path = std::path::Path::new(&path_str);
            if let Some(content) = get_cached_prompt(path) {
                prompts.push(content);
            }
        }

        // 2. Auto-scan ~/.MemoryPilot/GLOBAL_PROMPT.md
        let home_prompt = dirs::home_dir().map(|h| h.join(DB_DIR).join(PROMPT_FILE));
        if let Some(path) = &home_prompt {
            if let Some(content) = get_cached_prompt(path) {
                if !prompts.iter().any(|p| p == &content) {
                    prompts.push(content);
                }
            }
        }

        // 3. Auto-scan project root GLOBAL_PROMPT.md
        let proj_dir: Option<String> = working_dir.map(Self::infer_project_root).or_else(|| {
            let proj_name = canonical_project.as_deref()?;
            let mut stmt = self
                .conn
                .prepare("SELECT path FROM projects WHERE name=?1")
                .ok()?;
            stmt.query_row(params![proj_name], |r| r.get::<_, String>(0))
                .ok()
        });

        if let Some(dir) = proj_dir {
            let proj_prompt = std::path::Path::new(&dir).join(PROMPT_FILE);
            if let Some(content) = get_cached_prompt(&proj_prompt) {
                if !prompts.iter().any(|p| p == &content) {
                    prompts.push(content);
                }
            }
        }

        if prompts.is_empty() {
            None
        } else {
            Some(prompts.join("\n\n---\n\n"))
        }
    }

    // ─── AAAK-STYLE COMPRESSION ─────────────────────────

    pub(super) fn compress_memory(mem: &Memory) -> String {
        let kind_short = match mem.kind.as_str() {
            "decision" => "DEC",
            "preference" => "PREF",
            "bug" => "BUG",
            "pattern" => "PAT",
            "skill" => "SKILL",
            "credential" => "CRED",
            "snippet" => "SNIP",
            "todo" => "TODO",
            "note" => "NOTE",
            "fact" => "FACT",
            "transcript" => "TXN",
            "architecture" => "ARCH",
            "milestone" => "MILE",
            "problem" => "PROB",
            _ => "MEM",
        };
        let truncated = crate::text::preview(&mem.content, 200);
        let tags_str = if mem.tags.is_empty() {
            String::new()
        } else {
            format!(" | tags:{}", mem.tags.join(","))
        };
        let proj_str = mem
            .project
            .as_ref()
            .map(|p| format!(" | proj:{}", p))
            .unwrap_or_default();
        format!(
            "[{}:{}] {}{}{}",
            kind_short,
            mem.importance,
            truncated.replace('\n', " "),
            tags_str,
            proj_str
        )
    }

    pub(super) fn compress_memories(mems: &[Memory]) -> String {
        mems.iter()
            .map(Self::compress_memory)
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub(super) fn compress_strings(kind: &str, items: &[String]) -> String {
        if items.is_empty() {
            return String::new();
        }
        let tag = kind.to_uppercase();
        items
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let truncated = crate::text::preview(s, 150);
                format!("[{}:{}] {}", tag, i + 1, truncated.replace('\n', " "))
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub fn get_project_brain(
        &self,
        project: &str,
        max_tokens: Option<usize>,
        compact: bool,
    ) -> Result<serde_json::Value, String> {
        let canonical_project =
            Self::canonical_project_name(project).ok_or("Project name cannot be empty")?;
        let max_t = max_tokens.unwrap_or(1500);
        let max_chars = max_t * 4;
        let mut current_chars = 0;

        let mut tech_stack = Vec::new();
        if let Ok(mut stmt) = self.conn.prepare("SELECT DISTINCT entity_value FROM memory_entities e JOIN memories m ON e.memory_id = m.id WHERE m.project = ?1 AND e.entity_kind = 'tech' LIMIT 15") {
            if let Ok(rows) = stmt.query_map(params![&canonical_project], |r| r.get::<_, String>(0)) {
                for tech in rows.flatten() {
                    let len = tech.len();
                    if current_chars + len > max_chars { break; }
                    current_chars += len;
                    tech_stack.push(tech);
                }
            }
        }

        let (core_arch, _) =
            self.list_memories(Some(&canonical_project), Some("architecture"), None, 10, 0)?;
        let mut arch_content = Vec::new();
        for m in core_arch {
            if current_chars + m.content.len() > max_chars {
                break;
            }
            current_chars += m.content.len();
            arch_content.push(m.content);
        }

        let (decisions, _) =
            self.list_memories(Some(&canonical_project), Some("decision"), None, 10, 0)?;
        let mut dec_content = Vec::new();
        for m in decisions {
            if current_chars + m.content.len() > max_chars {
                break;
            }
            current_chars += m.content.len();
            dec_content.push(m.content);
        }

        let (bugs, _) = self.list_memories(Some(&canonical_project), Some("bug"), None, 10, 0)?;
        let mut bug_content = Vec::new();
        for m in bugs {
            if verification::is_closed(&m) {
                continue;
            }
            let line = match self.verification(&m, false) {
                Some(found) if found["status"] == "likely_resolved" || found["status"] == "possibly_resolved" => {
                    let commit = &found["commits"][0];
                    format!(
                        "[{} by {} {}] {}",
                        found["status"].as_str().unwrap_or_default(),
                        commit["sha"].as_str().unwrap_or_default(),
                        commit["subject"].as_str().unwrap_or_default(),
                        m.content
                    )
                }
                _ => m.content,
            };
            if current_chars + line.len() > max_chars {
                break;
            }
            current_chars += line.len();
            bug_content.push(line);
        }

        let mut recent_content = Vec::new();
        if let Ok(mut stmt) = self.conn.prepare("SELECT content, kind, metadata FROM memories WHERE project = ?1 AND updated_at > datetime('now','-7 days') ORDER BY updated_at DESC LIMIT 10") {
            if let Ok(rows) = stmt.query_map(params![&canonical_project], |r| {
                let metadata: Option<String> = r.get(2)?;
                let metadata: Option<serde_json::Value> = metadata.and_then(|m| serde_json::from_str(&m).ok());
                Ok(vault::read_content(&r.get::<_, String>(1)?, r.get(0)?, metadata.as_ref()))
            }) {
                for content in rows.flatten() {
                    if current_chars + content.len() > max_chars { break; }
                    current_chars += content.len();
                    recent_content.push(content);
                }
            }
        }

        let mut key_components = Vec::new();
        if let Ok(mut stmt) = self.conn.prepare("SELECT DISTINCT entity_value FROM memory_entities e JOIN memories m ON e.memory_id = m.id WHERE m.project = ?1 AND e.entity_kind IN ('component', 'file') LIMIT 15") {
            if let Ok(rows) = stmt.query_map(params![&canonical_project], |r| r.get::<_, String>(0)) {
                for comp in rows.flatten() {
                    let len = comp.len();
                    if current_chars + len > max_chars { break; }
                    current_chars += len;
                    key_components.push(comp);
                }
            }
        }

        // Team members (person entities)
        let mut team_members = Vec::new();
        if let Ok(mut stmt) = self.conn.prepare("SELECT DISTINCT entity_value FROM memory_entities e JOIN memories m ON e.memory_id = m.id WHERE m.project = ?1 AND e.entity_kind = 'person' LIMIT 20") {
            if let Ok(rows) = stmt.query_map(params![&canonical_project], |r| r.get::<_, String>(0)) {
                for person in rows.flatten() {
                    team_members.push(person);
                }
            }
        }

        let derived_facts = self.compact_project_facts(&canonical_project);

        if compact {
            let mut lines = Vec::new();
            lines.push(format!("# {} brain", canonical_project));
            if let Some(facts) = derived_facts.as_ref().and_then(|f| f.as_object()) {
                for (key, value) in facts {
                    let text = match value {
                        serde_json::Value::Array(items) => items
                            .iter()
                            .filter_map(|item| item.as_str())
                            .collect::<Vec<_>>()
                            .join("; "),
                        other => other.as_str().unwrap_or_default().to_string(),
                    };
                    lines.push(format!("REPO {}: {}", key.to_uppercase(), text));
                }
            }
            if !tech_stack.is_empty() {
                lines.push(format!("STACK: {}", tech_stack.join(", ")));
            }
            if !key_components.is_empty() {
                lines.push(format!("COMPONENTS: {}", key_components.join(", ")));
            }
            if !team_members.is_empty() {
                lines.push(format!("TEAM: {}", team_members.join(", ")));
            }
            if !arch_content.is_empty() {
                lines.push(Self::compress_strings("ARCH", &arch_content));
            }
            if !dec_content.is_empty() {
                lines.push(Self::compress_strings("DEC", &dec_content));
            }
            if !bug_content.is_empty() {
                lines.push(Self::compress_strings("BUG", &bug_content));
            }
            if !recent_content.is_empty() {
                lines.push(Self::compress_strings("RECENT", &recent_content));
            }
            return Ok(
                serde_json::json!({ "compact": lines.join("\n"), "approx_tokens_used": lines.join("\n").len() / 4 }),
            );
        }

        let mut brain = serde_json::json!({
            "project": canonical_project,
            "tech_stack": tech_stack,
            "core_architecture": arch_content,
            "current_critical_decisions": dec_content,
            "active_bugs_known": bug_content,
            "recent_changes": recent_content,
            "key_components": key_components,
            "team_members": team_members,
            "approx_tokens_used": current_chars / 4
        });
        if let Some(facts) = derived_facts {
            brain["derived_facts"] = facts;
        }
        Ok(brain)
    }

    pub fn get_project_context(
        &self,
        project: Option<&str>,
        working_dir: Option<&str>,
        mode: RecallMode,
        scope: &MemoryScope,
    ) -> Result<serde_json::Value, String> {
        let proj_name = match Self::canonical_project(project) {
            Some(p) => Some(p),
            None => match working_dir {
                Some(wd) => self.detect_project(wd)?,
                None => None,
            },
        };
        let proj_ref = proj_name.as_deref();
        let (proj_memories, proj_total) = if let Some(p) = proj_ref {
            let (memories, total) =
                self.list_memories(Some(p), None, Some("transcript"), 100, 0)?;
            (
                memories
                    .into_iter()
                    .filter(|memory| Self::should_include_in_context(memory, mode))
                    .collect::<Vec<_>>(),
                total,
            )
        } else {
            (vec![], 0)
        };
        let (prefs, _) = self.list_memories(None, Some("preference"), None, 50, 0)?;
        let prefs = prefs
            .into_iter()
            .filter(|memory| memory.project.is_none())
            .collect::<Vec<_>>();
        let (patterns, _) = self.list_memories(None, Some("pattern"), None, 50, 0)?;
        let patterns = patterns
            .into_iter()
            .filter(|memory| memory.project.is_none())
            .collect::<Vec<_>>();
        let (snippets, _) = self.list_memories(None, Some("snippet"), None, 20, 0)?;
        let scope_memories = self.list_scope_memories(proj_ref, scope, 20)?;

        Ok(serde_json::json!({
            "mode": mode.as_str(),
            "project": proj_ref.unwrap_or("none"),
            "project_memories": proj_total,
            "global_preferences": prefs.len(),
            "global_patterns": patterns.len(),
            "context": {
                "scope": scope_memories.iter().map(|m| serde_json::json!({"kind":m.kind,"content":m.content,"tags":m.tags,"importance":m.importance})).collect::<Vec<_>>(),
                "project": proj_memories.iter().map(|m| serde_json::json!({"kind":m.kind,"content":m.content,"tags":m.tags,"importance":m.importance})).collect::<Vec<_>>(),
                "preferences": prefs.iter().map(|m| &m.content).collect::<Vec<_>>(),
                "patterns": patterns.iter().map(|m| serde_json::json!({"content":m.content,"tags":m.tags})).collect::<Vec<_>>(),
                "snippets": snippets.iter().map(|m| serde_json::json!({"content":m.content,"tags":m.tags})).collect::<Vec<_>>(),
            }
        }))
    }

    // ─── RECALL (auto-context loader) ─────────────────

    /// One-shot context loader for new conversations.
    /// Combines: project context, global prompt, critical memories, and optional hint search.
    /// Whether a memory may appear in a recall scoped to `project`.
    pub(super) fn recall_project_compatible(memory: &Memory, project: Option<&str>) -> bool {
        let Some(current) = project else { return true };
        match memory.project.as_deref() {
            None => true,
            Some(own) if own == current => true,
            Some(_) => {
                // The other project's memory must be ABOUT the current one:
                // named in its opening (the subject) or mentioned repeatedly,
                // not a passing reference at the end.
                let needle = current.to_lowercase();
                if needle.len() < 3 {
                    return false;
                }
                let lower = memory.content.to_lowercase();
                let head: String = lower.chars().take(200).collect();
                head.contains(&needle) || lower.matches(needle.as_str()).count() >= 2
            }
        }
    }

    pub fn recall(
        &self,
        project: Option<&str>,
        working_dir: Option<&str>,
        hints: Option<&str>,
        mode: RecallMode,
        explain: bool,
        compact: bool,
        scope: &MemoryScope,
    ) -> Result<serde_json::Value, String> {
        // ~4 chars per token — 800 token budget = 3200 chars for memories
        const TOKEN_BUDGET_CHARS: usize = 3200;

        // Auto-detect project
        let proj_name = match Self::canonical_project(project) {
            Some(p) => Some(p),
            None => match working_dir {
                Some(wd) => self.detect_project(wd)?,
                None => None,
            },
        };
        let proj_ref = proj_name.as_deref();
        let hint_terms: Vec<String> = hints
            .unwrap_or_default()
            .split_whitespace()
            .map(|term: &str| {
                term.trim_matches(|character: char| {
                    !character.is_alphanumeric() && character != '-' && character != '_'
                })
                .to_ascii_lowercase()
            })
            .filter(|term: &String| term.len() > 2)
            .collect();
        let hint_entity_keys = Self::entity_overlap_keys(hints.unwrap_or_default(), proj_ref);

        let mut seen_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut used_chars: usize = 0;
        let link_boosts = if explain {
            self.build_link_boosts()
        } else {
            std::collections::HashMap::new()
        };
        let mut selected_memories_for_explain: Vec<serde_json::Value> = Vec::new();
        let credentials_hidden = if mode.includes_credentials() {
            0
        } else {
            self.conn
                .query_row(
                    "SELECT COUNT(*) FROM memories WHERE kind = 'credential'",
                    [],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap_or(0)
        };

        // Helper: check budget and dedup, returns true if item should be included
        macro_rules! budget_check {
            ($id:expr, $content:expr) => {{
                if seen_ids.contains($id) {
                    false
                } else if used_chars + $content.len() > TOKEN_BUDGET_CHARS {
                    false
                } else {
                    seen_ids.insert($id.to_string());
                    used_chars += $content.len();
                    true
                }
            }};
        }

        // 1. Hint-based search first: the memories that answer the current task
        // claim their place (and budget) before the generic critical ones.
        let hint_results: Vec<SearchResult> = if let Some(h) = hints {
            if !h.trim().is_empty() {
                crate::reranking::with_forced_rerank(|| self.search(h, 10, proj_ref, None, None, None))
                    .unwrap_or_default()
                    .into_iter()
                    .filter(|result| result.memory.kind != "transcript")
                    // Project isolation: when the project is known, a memory
                    // from ANOTHER project only gets in if it explicitly
                    // mentions the current one (shared infra, sister apps).
                    // Global memories (no project) stay eligible.
                    .filter(|result| Self::recall_project_compatible(&result.memory, proj_ref))
                    .filter(|result| Self::should_include_in_context(&result.memory, mode))
                    .filter(|r| budget_check!(&r.memory.id, &r.memory.content))
                    .collect()
            } else {
                vec![]
            }
        } else {
            vec![]
        };

        // 2. Critical memories next, but keep them project-safe and hint-aware.
        let critical: Vec<Memory> = {
            let has_contextual_hint = !hint_terms.is_empty();
            let mut stmt = self.conn.prepare(
                "SELECT id,content,kind,project,tags,source,importance,expires_at,metadata,created_at,updated_at,last_accessed_at,access_count \
                 FROM memories WHERE importance >= 4 \
                 AND (expires_at IS NULL OR expires_at > datetime('now')) \
                 ORDER BY importance DESC, updated_at DESC LIMIT 30"
            ).map_err(|e| format!("Recall critical: {}", e))?;
            let rows = stmt
                .query_map([], |r| Ok(row_to_memory(r)))
                .map_err(|e| format!("Recall critical: {}", e))?;
            let mut candidates: Vec<(i32, Memory)> = rows
                .flatten()
                .filter(|memory| Self::should_include_in_context(memory, mode))
                .map(|memory| {
                    let lowered_content = memory.content.to_ascii_lowercase();
                    let tag_text = memory.tags.join(" ").to_ascii_lowercase();
                    let hint_overlap = hint_terms
                        .iter()
                        .filter(|term| {
                            lowered_content.contains(term.as_str())
                                || tag_text.contains(term.as_str())
                        })
                        .count() as i32;
                    (memory, hint_overlap)
                })
                .filter(|(memory, hint_overlap)| {
                    if let Some(project_name) = proj_ref {
                        if memory.project.as_deref() == Some(project_name) {
                            return true;
                        }
                        memory.project.is_none() && *hint_overlap > 0
                    } else {
                        true
                    }
                })
                .map(|(memory, hint_overlap)| {
                    let project_match =
                        if proj_ref.is_some() && memory.project.as_deref() == proj_ref {
                            8
                        } else if memory.project.is_none() && hint_overlap > 0 {
                            2
                        } else {
                            0
                        };
                    let score = project_match
                        + (hint_overlap * 4)
                        + (memory.importance * 2)
                        + memory.access_count.min(5);
                    (score, memory)
                })
                .collect();

            candidates.sort_by(|left, right| {
                right
                    .0
                    .cmp(&left.0)
                    .then_with(|| right.1.updated_at.cmp(&left.1.updated_at))
            });

            let critical_limit = if has_contextual_hint {
                3
            } else if proj_ref.is_some() {
                6
            } else {
                12
            };
            candidates
                .into_iter()
                .filter(|(score, _)| *score > 0 || (proj_ref.is_none() && !has_contextual_hint))
                .map(|(_, memory)| memory)
                .take(critical_limit)
                .filter(|memory| budget_check!(&memory.id, &memory.content))
                .collect()
        };

        // 1b. Pinned memories — always included regardless of score/hints
        let pinned_memories: Vec<Memory> = {
            let mut stmt = self.conn.prepare(
                "SELECT id,content,kind,project,tags,source,importance,expires_at,metadata,created_at,updated_at,last_accessed_at,access_count \
                 FROM memories WHERE tags LIKE '%pinned%' ORDER BY importance DESC, updated_at DESC"
            ).map_err(|e| format!("Recall pinned: {}", e))?;
            let result: Vec<Memory> = stmt
                .query_map([], |r| Ok(row_to_memory(r)))
                .map_err(|e| format!("Recall pinned: {}", e))?
                .flatten()
                .filter(|m| Self::should_include_in_context(m, mode))
                .filter(|m| budget_check!(&m.id, &m.content))
                .collect();
            result
        };

        // 3. Scope memories from the same session / thread / window
        let scope_memories = self
            .list_scope_memories(proj_ref, scope, 15)?
            .into_iter()
            .filter(|memory| Self::should_include_in_context(memory, mode))
            .filter(|memory| budget_check!(&memory.id, &memory.content))
            .collect::<Vec<_>>();

        // 4. Project memories (excluding transcripts — too verbose)
        let (proj_memories, proj_total) = if let Some(p) = proj_ref {
            let (all, total) = self.list_memories(Some(p), None, Some("transcript"), 50, 0)?;
            let filtered: Vec<Memory> = all
                .into_iter()
                .filter(|memory| Self::should_include_in_context(memory, mode))
                .filter(|m| budget_check!(&m.id, &m.content))
                .collect();
            (filtered, total)
        } else {
            (vec![], 0)
        };

        // 5. Global preferences + patterns + decisions (with remaining budget)
        let (prefs, _) = self.list_memories(None, Some("preference"), None, 20, 0)?;
        let prefs: Vec<Memory> = Self::select_global_context_memories(
            prefs
                .into_iter()
                .filter(|memory| memory.project.is_none())
                .collect(),
            &hint_terms,
            &hint_entity_keys,
            proj_ref,
            12,
        )
        .into_iter()
        .filter(|m| budget_check!(&m.id, &m.content))
        .collect();

        let (patterns, _) = self.list_memories(None, Some("pattern"), None, 15, 0)?;
        let patterns: Vec<Memory> = Self::select_global_context_memories(
            patterns
                .into_iter()
                .filter(|memory| memory.project.is_none())
                .collect(),
            &hint_terms,
            &hint_entity_keys,
            proj_ref,
            10,
        )
        .into_iter()
        .filter(|m| budget_check!(&m.id, &m.content))
        .collect();

        let (decisions, _) = self.list_memories(None, Some("decision"), None, 15, 0)?;
        let decisions: Vec<Memory> = Self::select_global_context_memories(
            decisions
                .into_iter()
                .filter(|memory| memory.project.is_none())
                .collect(),
            &hint_terms,
            &hint_entity_keys,
            proj_ref,
            10,
        )
        .into_iter()
        .filter(|m| budget_check!(&m.id, &m.content))
        .collect();

        // 6. Global prompt — lazy: only load if budget allows and file exists
        let global_prompt = if used_chars < TOKEN_BUDGET_CHARS {
            self.get_global_prompt(proj_ref, working_dir)
        } else {
            None
        };

        // 7. Stats (cheap, always included)
        let total: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))
            .unwrap_or(0);
        let projects_count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM projects", [], |r| r.get(0))
            .unwrap_or(0);
        let project_context_memories_count = proj_memories.len();

        if explain {
            for result in &hint_results {
                let mut explanation = self.recall_explanation(
                    &result.memory,
                    "hint",
                    proj_ref,
                    mode,
                    Some(result.score),
                    &link_boosts,
                );
                if let Some(object) = explanation.as_object_mut() {
                    object.insert(
                        "retrieval_sources".into(),
                        serde_json::json!(result.sources),
                    );
                }
                selected_memories_for_explain.push(explanation);
            }
            for memory in &scope_memories {
                selected_memories_for_explain.push(self.recall_explanation(
                    memory,
                    "scope",
                    proj_ref,
                    mode,
                    None,
                    &link_boosts,
                ));
            }
            for memory in &proj_memories {
                selected_memories_for_explain.push(self.recall_explanation(
                    memory,
                    "project",
                    proj_ref,
                    mode,
                    None,
                    &link_boosts,
                ));
            }
            for memory in &critical {
                selected_memories_for_explain.push(self.recall_explanation(
                    memory,
                    "critical",
                    proj_ref,
                    mode,
                    None,
                    &link_boosts,
                ));
            }
            for memory in &prefs {
                selected_memories_for_explain.push(self.recall_explanation(
                    memory,
                    "preference",
                    proj_ref,
                    mode,
                    None,
                    &link_boosts,
                ));
            }
            for memory in &patterns {
                selected_memories_for_explain.push(self.recall_explanation(
                    memory,
                    "pattern",
                    proj_ref,
                    mode,
                    None,
                    &link_boosts,
                ));
            }
            for memory in &decisions {
                selected_memories_for_explain.push(self.recall_explanation(
                    memory,
                    "decision",
                    proj_ref,
                    mode,
                    None,
                    &link_boosts,
                ));
            }
        }

        let secrets = vault::secret_references(
            hint_results
                .iter()
                .map(|r| &r.memory)
                .chain(&critical)
                .chain(&pinned_memories)
                .chain(&scope_memories)
                .chain(&proj_memories)
                .chain(&prefs)
                .chain(&patterns)
                .chain(&decisions),
        );
        if compact {
            let mut lines = Vec::new();
            lines.push(format!(
                "# recall | proj:{} | mode:{}",
                proj_ref.unwrap_or("none"),
                mode.as_str()
            ));
            if !critical.is_empty() {
                lines.push(format!(
                    "--- critical ---\n{}",
                    Self::compress_memories(&critical)
                ));
            }
            let hint_mems: Vec<Memory> = hint_results.iter().map(|r| r.memory.clone()).collect();
            if !hint_mems.is_empty() {
                lines.push(format!(
                    "--- hints ---\n{}",
                    Self::compress_memories(&hint_mems)
                ));
            }
            if !scope_memories.is_empty() {
                lines.push(format!(
                    "--- scope ---\n{}",
                    Self::compress_memories(&scope_memories)
                ));
            }
            if !proj_memories.is_empty() {
                lines.push(format!(
                    "--- project ---\n{}",
                    Self::compress_memories(&proj_memories)
                ));
            }
            if !prefs.is_empty() {
                lines.push(format!(
                    "--- prefs ---\n{}",
                    Self::compress_memories(&prefs)
                ));
            }
            if !patterns.is_empty() {
                lines.push(format!(
                    "--- patterns ---\n{}",
                    Self::compress_memories(&patterns)
                ));
            }
            if !decisions.is_empty() {
                lines.push(format!(
                    "--- decisions ---\n{}",
                    Self::compress_memories(&decisions)
                ));
            }
            if let Some(gp) = &global_prompt {
                let gp_short = crate::text::preview(gp, 500);
                lines.push(format!("--- global_prompt ---\n{}", gp_short));
            }
            if let Some(listed) = secrets.as_ref().and_then(|s| s["items"].as_array()) {
                let refs: Vec<String> = listed
                    .iter()
                    .map(|item| format!("{} → {} (reveal_secret id={})", item["label"].as_str().unwrap_or(""), item["where"].as_str().unwrap_or(""), item["id"].as_str().unwrap_or("")))
                    .collect();
                lines.push(format!("--- secrets (values never shown) ---\n{}", refs.join("\n")));
            }
            return Ok(serde_json::json!({
                "status": "recalled",
                "mode": mode.as_str(),
                "project": proj_ref.unwrap_or("none"),
                "compact": lines.join("\n"),
                "stats": { "total_memories": total, "projects": projects_count },
            }));
        }

        let mut response = serde_json::json!({
            "status": "recalled",
            "mode": mode.as_str(),
            "project": proj_ref.unwrap_or("none"),
            "stats": { "total_memories": total, "projects": projects_count, "project_memories": proj_total },
            "hint_results": hint_results.iter().map(|r| serde_json::json!({
                "content": r.memory.content, "kind": r.memory.kind, "project": r.memory.project
            })).collect::<Vec<_>>(),
            "critical_memories": critical.iter().map(|m| serde_json::json!({
                "content": m.content, "kind": m.kind, "project": m.project,
                "importance": m.importance
            })).collect::<Vec<_>>(),
            "pinned_memories": pinned_memories.iter().map(|m| serde_json::json!({
                "id": m.id, "content": m.content, "kind": m.kind, "project": m.project,
                "importance": m.importance
            })).collect::<Vec<_>>(),
            "scope_context": scope_memories.iter().map(|m| serde_json::json!({
                "content": m.content, "kind": m.kind, "importance": m.importance
            })).collect::<Vec<_>>(),
            "project_context": proj_memories.iter().map(|m| serde_json::json!({
                "content": m.content, "kind": m.kind, "importance": m.importance
            })).collect::<Vec<_>>(),
            "preferences": prefs.iter().map(|m| m.content.as_str()).collect::<Vec<_>>(),
            "patterns": patterns.iter().map(|m| m.content.as_str()).collect::<Vec<_>>(),
            "decisions": decisions.iter().map(|m| m.content.as_str()).collect::<Vec<_>>(),
            "global_prompt": global_prompt.as_deref().unwrap_or(""),
        });
        if let (Some(secrets), Some(object)) = (secrets, response.as_object_mut()) {
            object.insert("secrets".into(), secrets);
        }

        if explain {
            response.as_object_mut().map(|object| {
                object.insert(
                    "explain".into(),
                    serde_json::json!({
                        "budget_chars_used": used_chars,
                        "budget_chars_limit": TOKEN_BUDGET_CHARS,
                        "selected_count": selected_memories_for_explain.len(),
                        "project_detected": proj_ref,
                        "credentials_hidden_in_safe_mode": credentials_hidden,
                        "category_counts": {
                            "critical": critical.len(),
                            "pinned": pinned_memories.len(),
                            "hint": hint_results.len(),
                            "scope": scope_memories.len(),
                            "project": project_context_memories_count,
                            "preferences": prefs.len(),
                            "patterns": patterns.len(),
                            "decisions": decisions.len(),
                        },
                        "selected_memories": selected_memories_for_explain,
                    }),
                );
            });
        }

        Ok(response)
    }
}
