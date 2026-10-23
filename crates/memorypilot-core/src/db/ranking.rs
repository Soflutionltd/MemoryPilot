//! Ranking signals shared by search and recall: intent, recency, links, graph expansion.

use super::*;

impl Database {
    pub(super) fn memory_age_days(memory: &Memory) -> Option<i64> {
        let updated_at = chrono::DateTime::parse_from_rfc3339(&memory.updated_at).ok()?;
        Some((Utc::now() - updated_at.with_timezone(&Utc)).num_days())
    }

    pub(super) fn memory_haystack(memory: &Memory) -> String {
        format!(
            "{} {} {}",
            memory.content.to_ascii_lowercase(),
            memory.tags.join(" ").to_ascii_lowercase(),
            memory.source.to_ascii_lowercase()
        )
    }

    pub(super) fn entity_overlap_keys(
        content: &str,
        project: Option<&str>,
    ) -> std::collections::HashSet<String> {
        let mut keys = std::collections::HashSet::new();
        for entity in crate::graph::extract_entities(content, project) {
            keys.insert(entity.value.to_ascii_lowercase());
        }
        keys
    }

    pub(super) fn global_context_score(
        memory: &Memory,
        hint_terms: &[String],
        hint_entity_keys: &std::collections::HashSet<String>,
        project: Option<&str>,
    ) -> i32 {
        let haystack = Self::memory_haystack(memory);
        let hint_overlap = hint_terms
            .iter()
            .filter(|term| haystack.contains(term.as_str()))
            .count() as i32;
        let entity_overlap =
            crate::graph::extract_entities(&memory.content, memory.project.as_deref())
                .into_iter()
                .filter(|entity| hint_entity_keys.contains(&entity.value.to_ascii_lowercase()))
                .count() as i32;
        let recency_bonus = match Self::memory_age_days(memory) {
            Some(age_days) if age_days <= 14 => 2,
            Some(age_days) if age_days <= 45 => 1,
            Some(age_days) if age_days >= 180 => -1,
            _ => 0,
        };
        let access_bonus = memory.access_count.min(3);
        let kind_bias = match memory.kind.as_str() {
            "preference" => 3,
            "pattern" => 2,
            "skill" => 2,
            "decision" => 1,
            _ => 0,
        };
        let project_bonus = project
            .map(|project_name| haystack.contains(project_name) as i32)
            .unwrap_or(0);
        let contextual = project.is_some() || !hint_terms.is_empty();
        let generic_penalty = if contextual && hint_overlap == 0 && entity_overlap == 0 {
            if project.is_some() && hint_terms.len() >= 2 {
                8
            } else {
                5
            }
        } else {
            0
        };

        kind_bias
            + (hint_overlap * 4)
            + (entity_overlap * 3)
            + access_bonus
            + recency_bonus
            + project_bonus
            - generic_penalty
    }

    pub(super) fn select_global_context_memories(
        memories: Vec<Memory>,
        hint_terms: &[String],
        hint_entity_keys: &std::collections::HashSet<String>,
        project: Option<&str>,
        limit: usize,
    ) -> Vec<Memory> {
        let contextual = project.is_some() || !hint_terms.is_empty();
        let mut ranked: Vec<(i32, Memory)> = memories
            .into_iter()
            .map(|memory| {
                (
                    Self::global_context_score(&memory, hint_terms, hint_entity_keys, project),
                    memory,
                )
            })
            .collect();

        ranked.sort_by(|left, right| {
            right
                .0
                .cmp(&left.0)
                .then_with(|| right.1.importance.cmp(&left.1.importance))
                .then_with(|| right.1.updated_at.cmp(&left.1.updated_at))
        });

        ranked
            .into_iter()
            .filter(|(score, _)| !contextual || *score > 0)
            .take(limit)
            .map(|(_, memory)| memory)
            .collect()
    }

    pub(super) fn recency_boost(memory: &Memory) -> f64 {
        let age_days = Self::memory_age_days(memory).unwrap_or(365).max(0) as f64;
        ((30.0 - age_days).max(0.0) / 30.0 * 0.25 * 100.0).round() / 100.0
    }

    pub(super) fn access_boost(memory: &Memory) -> f64 {
        ((memory.access_count.min(10) as f64) / 10.0 * 0.2 * 100.0).round() / 100.0
    }

    pub(super) fn build_link_boosts(&self) -> std::collections::HashMap<String, f64> {
        self.build_link_boosts_for(&[])
    }

    pub(super) fn build_link_boosts_for(
        &self,
        candidate_ids: &[&String],
    ) -> std::collections::HashMap<String, f64> {
        let mut link_boosts = std::collections::HashMap::new();
        let rows_data = if candidate_ids.is_empty() || candidate_ids.len() > 200 {
            self.link_relations(None)
        } else {
            self.link_relations(Some(candidate_ids))
        };

        for (target_id, relation_type) in rows_data {
            let boost: f64 = match relation_type.as_str() {
                "deprecates" => -0.6,
                // A newer memory restates this one with a different
                // value (see `link_superseded_memories`): keep it
                // retrievable, but below its replacement.
                "supersedes" => -0.35,
                "depends_on" | "implements" | "resolves" | "resolved_by" | "fixed_by" | "fixes" => {
                    0.08
                }
                "shares_topic" => 0.06,
                "same_agent" => 0.05,
                "same_origin" => 0.02,
                _ => 0.03,
            };
            let total = link_boosts.entry(target_id).or_insert(0.0);
            *total = (*total + boost).clamp(-0.8_f64, 0.25_f64);
        }
        link_boosts
    }

    pub(super) fn get_kg_expansion_terms(&self, query: &str) -> Vec<String> {
        let words: Vec<&str> = query.split_whitespace().collect();
        if words.len() > 15 {
            return Vec::new();
        }

        let mut terms: Vec<String> = Vec::new();
        let query_lower = query.to_lowercase();

        for word in &words {
            if word.len() < 3 {
                continue;
            }
            let lower = word.to_lowercase();

            // KG triples (related subjects/objects), then entity co-occurrence
            for r in self.related_terms(&lower) {
                if !query_lower.contains(&r.to_lowercase()) && r.len() >= 2 {
                    terms.push(r.to_lowercase());
                }
            }
        }

        terms.truncate(12);
        terms
    }

    pub(super) fn build_adjacency_set(&self, ids: &[String]) -> std::collections::HashSet<(String, String)> {
        self.adjacency_pairs(ids)
    }

    pub(super) fn batch_triple_counts(
        &self,
        candidate_ids: &[&String],
    ) -> std::collections::HashMap<String, (i64, i64)> {
        self.triple_counts(candidate_ids)
    }

    // ─── SEARCH (FTS5 BM25 × importance) ──────────────

    pub(super) fn infer_query_intent(query: &str) -> QueryIntent {
        let lower = query.to_ascii_lowercase();
        let contains_any = |needles: &[&str]| needles.iter().any(|needle| lower.contains(needle));
        QueryIntent {
            preference: contains_any(&[
                "prefer",
                "preference",
                "like",
                "favorite",
                "favourite",
                "would rather",
                "préfère",
                "préférence",
                "aime",
                "favori",
                "plutôt",
            ]),
            temporal: contains_any(&[
                "when",
                "before",
                "after",
                "first",
                "last",
                "latest",
                "recent",
                "currently",
                "now",
                "timeline",
                "previously",
                "earlier",
                "ensuite",
                "avant",
                "après",
                "dernier",
                "récemment",
                "maintenant",
                "actuellement",
            ]),
            user_turn: contains_any(&[
                "user",
                "human",
                "i said",
                "i told",
                "j'ai dit",
                "je voulais",
                "utilisateur",
            ]),
            assistant_turn: contains_any(&[
                "assistant",
                "you said",
                "claude",
                "cursor",
                "chatgpt",
                "tu as dit",
            ]),
            update_or_correction: contains_any(&[
                "changed",
                "updated",
                "correction",
                "actually",
                "instead",
                "not anymore",
                "now uses",
                "switch",
                "switched",
                "remplace",
                "corrige",
                "modifié",
                "désormais",
                "maintenant",
                "en fait",
                "plutôt",
            ]),
            technical: contains_any(&[
                "bug",
                "error",
                "stack",
                "architecture",
                "api",
                "database",
                "file",
                "function",
                "component",
                "deploy",
                "build",
                "benchmark",
                "embedding",
                "reranker",
            ]),
        }
    }

    pub(super) fn query_intent_factor(intent: &QueryIntent, memory: &Memory) -> f64 {
        let content = memory.content.to_ascii_lowercase();
        let mut factor: f64 = 1.0;

        if intent.preference {
            let pref_signal = memory.kind == "preference"
                || content.contains("prefer")
                || content.contains("préf")
                || content.contains("like")
                || content.contains("aime")
                || content.contains("favorite")
                || content.contains("plutôt");
            if pref_signal {
                factor *= 1.16;
            }
        }

        if intent.user_turn && content.trim_start().starts_with("user:") {
            factor *= 1.12;
        }
        if intent.assistant_turn && content.trim_start().starts_with("assistant:") {
            factor *= 1.12;
        }

        if intent.update_or_correction {
            let update_signal = content.contains("changed")
                || content.contains("updated")
                || content.contains("actually")
                || content.contains("instead")
                || content.contains("correction")
                || content.contains("switch")
                || content.contains("maintenant")
                || content.contains("désormais")
                || content.contains("en fait");
            if update_signal || matches!(memory.kind.as_str(), "decision" | "architecture" | "fact")
            {
                factor *= 1.1;
            }
        }

        if intent.temporal {
            if let Some(turn) = memory
                .id
                .split("__t")
                .last()
                .and_then(|value| value.parse::<usize>().ok())
            {
                if turn >= 20 {
                    factor *= 1.08;
                } else if turn <= 2
                    && (content.contains("first")
                        || content.contains("initial")
                        || content.contains("started"))
                {
                    factor *= 1.05;
                }
            }
            if content.contains("today")
                || content.contains("yesterday")
                || content.contains("now")
                || content.contains("maintenant")
            {
                factor *= 1.05;
            }
        }

        if intent.technical
            && matches!(
                memory.kind.as_str(),
                "architecture" | "decision" | "bug" | "snippet" | "pattern" | "skill"
            )
        {
            factor *= 1.07;
        }

        factor.min(1.35)
    }

    pub(super) fn cognitive_activation_factor(memory: &Memory, now_ts: f64) -> f64 {
        let frequency = ((memory.access_count.max(0) as f64 + 1.0).ln() / 21.0_f64.ln()).min(1.0);
        let last_touch = memory
            .last_accessed_at
            .as_deref()
            .unwrap_or(memory.updated_at.as_str());
        let recency = chrono::DateTime::parse_from_rfc3339(last_touch)
            .ok()
            .map(|timestamp| {
                let age_days = ((now_ts - timestamp.timestamp() as f64) / 86400.0).max(0.0);
                (1.0 / (1.0 + age_days / 14.0)).min(1.0)
            })
            .unwrap_or(0.0);

        1.0 + ((frequency * 0.07) + (recency * 0.05)).min(0.12)
    }

    /// Stored vectors of the given memories, decoded. Rows without an
    /// embedding (still queued) are absent from the map.
    pub(super) fn batch_memory_vectors(
        &self,
        ids: &[&String],
    ) -> std::collections::HashMap<String, Vec<f32>> {
        if ids.is_empty() || ids.len() > 200 {
            return std::collections::HashMap::new();
        }
        self.memory_vectors(ids)
    }

    /// `date` entities of the given memories, as calendar days.
    pub(super) fn batch_memory_dates(
        &self,
        ids: &[&String],
    ) -> std::collections::HashMap<String, Vec<chrono::NaiveDate>> {
        let mut dates: std::collections::HashMap<String, Vec<chrono::NaiveDate>> =
            std::collections::HashMap::new();
        for (id, values) in self.memory_date_values(ids) {
            for value in values {
                if let Some(day) = crate::temporal::day_of_timestamp(&value) {
                    dates.entry(id.clone()).or_default().push(day);
                }
            }
        }
        dates
    }
}
