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
        let mut rows_data: Vec<(String, String)> = Vec::new();

        if candidate_ids.is_empty() || candidate_ids.len() > 200 {
            if let Ok(mut stmt) = self
                .conn
                .prepare("SELECT target_id, relation_type FROM memory_links")
            {
                if let Ok(rows) = stmt.query_map([], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                }) {
                    for r in rows.flatten() {
                        rows_data.push(r);
                    }
                }
            }
        } else {
            let placeholders: Vec<String> = (1..=candidate_ids.len())
                .map(|i| format!("?{}", i))
                .collect();
            let sql = format!(
                "SELECT target_id, relation_type FROM memory_links WHERE target_id IN ({})",
                placeholders.join(",")
            );
            if let Ok(mut stmt) = self.conn.prepare(&sql) {
                let param_refs: Vec<&dyn rusqlite::types::ToSql> = candidate_ids
                    .iter()
                    .map(|id| id as &dyn rusqlite::types::ToSql)
                    .collect();
                if let Ok(rows) = stmt.query_map(param_refs.as_slice(), |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                }) {
                    for r in rows.flatten() {
                        rows_data.push(r);
                    }
                }
            }
        }

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

            // KG triples: related subjects/objects
            if let Ok(mut stmt) = self.conn.prepare(
                "SELECT DISTINCT object FROM knowledge_triples WHERE lower(subject) = ?1 AND valid_to IS NULL LIMIT 3"
            ) {
                if let Ok(rows) = stmt.query_map(params![&lower], |r| r.get::<_, String>(0)) {
                    for r in rows.flatten() {
                        if !query_lower.contains(&r.to_lowercase()) && r.len() >= 2 { terms.push(r.to_lowercase()); }
                    }
                }
            }
            if let Ok(mut stmt) = self.conn.prepare(
                "SELECT DISTINCT subject FROM knowledge_triples WHERE lower(object) = ?1 AND valid_to IS NULL LIMIT 3"
            ) {
                if let Ok(rows) = stmt.query_map(params![&lower], |r| r.get::<_, String>(0)) {
                    for r in rows.flatten() {
                        if !query_lower.contains(&r.to_lowercase()) && r.len() >= 2 { terms.push(r.to_lowercase()); }
                    }
                }
            }

            // Entity co-occurrence
            if let Ok(mut stmt) = self.conn.prepare(
                "SELECT DISTINCT b.entity_value FROM memory_entities a \
                 JOIN memory_entities b ON a.memory_id = b.memory_id AND a.entity_value != b.entity_value \
                 WHERE lower(a.entity_value) = ?1 LIMIT 4"
            ) {
                if let Ok(rows) = stmt.query_map(params![lower], |r| r.get::<_, String>(0)) {
                    for r in rows.flatten() {
                        if !query_lower.contains(&r.to_lowercase()) && r.len() >= 2 { terms.push(r.to_lowercase()); }
                    }
                }
            }
        }

        terms.truncate(12);
        terms
    }

    pub(super) fn build_adjacency_set(&self, ids: &[String]) -> std::collections::HashSet<(String, String)> {
        let mut adj = std::collections::HashSet::new();
        if ids.is_empty() {
            return adj;
        }
        let placeholders: Vec<String> = (1..=ids.len()).map(|i| format!("?{}", i)).collect();
        let sql = format!(
            "SELECT source_id, target_id FROM memory_links WHERE source_id IN ({0}) AND target_id IN ({0})",
            placeholders.join(",")
        );
        if let Ok(mut stmt) = self.conn.prepare(&sql) {
            // Bind each id twice (once for source_id IN, once for target_id IN)
            let mut params: Vec<&dyn rusqlite::types::ToSql> = Vec::with_capacity(ids.len());
            for id in ids {
                params.push(id as &dyn rusqlite::types::ToSql);
            }
            if let Ok(rows) = stmt.query_map(params.as_slice(), |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
            }) {
                for row in rows.flatten() {
                    adj.insert(row);
                }
            }
        }

        // Also connect via shared entities
        let ent_sql = format!(
            "SELECT DISTINCT a.memory_id, b.memory_id FROM memory_entities a \
             JOIN memory_entities b ON a.entity_value = b.entity_value AND a.entity_kind = b.entity_kind AND a.memory_id != b.memory_id \
             WHERE a.entity_kind != 'date' AND a.memory_id IN ({0}) AND b.memory_id IN ({0})",
            placeholders.join(",")
        );
        if let Ok(mut stmt) = self.conn.prepare(&ent_sql) {
            let mut params: Vec<&dyn rusqlite::types::ToSql> = Vec::with_capacity(ids.len());
            for id in ids {
                params.push(id as &dyn rusqlite::types::ToSql);
            }
            if let Ok(rows) = stmt.query_map(params.as_slice(), |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
            }) {
                for row in rows.flatten() {
                    adj.insert(row);
                }
            }
        }
        adj
    }

    pub(super) fn batch_triple_counts(
        &self,
        candidate_ids: &[&String],
    ) -> std::collections::HashMap<String, (i64, i64)> {
        let mut counts: std::collections::HashMap<String, (i64, i64)> =
            std::collections::HashMap::new();
        if candidate_ids.is_empty() {
            return counts;
        }
        let placeholders: Vec<String> = (1..=candidate_ids.len())
            .map(|i| format!("?{}", i))
            .collect();
        let sql = format!(
            "SELECT source_memory_id, \
             SUM(CASE WHEN valid_to IS NULL THEN 1 ELSE 0 END), \
             SUM(CASE WHEN valid_to IS NOT NULL THEN 1 ELSE 0 END) \
             FROM knowledge_triples WHERE source_memory_id IN ({}) GROUP BY source_memory_id",
            placeholders.join(",")
        );
        if let Ok(mut stmt) = self.conn.prepare(&sql) {
            let param_refs: Vec<&dyn rusqlite::types::ToSql> = candidate_ids
                .iter()
                .map(|id| id as &dyn rusqlite::types::ToSql)
                .collect();
            if let Ok(rows) = stmt.query_map(param_refs.as_slice(), |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, i64>(1)?,
                    r.get::<_, i64>(2)?,
                ))
            }) {
                for row in rows.flatten() {
                    counts.insert(row.0, (row.1, row.2));
                }
            }
        }
        counts
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
        let mut vectors = std::collections::HashMap::new();
        if ids.is_empty() || ids.len() > 200 {
            return vectors;
        }
        let placeholders: Vec<String> = (1..=ids.len()).map(|i| format!("?{}", i)).collect();
        let sql = format!(
            "SELECT id, embedding FROM memories WHERE embedding IS NOT NULL AND id IN ({})",
            placeholders.join(",")
        );
        let rconn = self.read_conn();
        let Ok(mut stmt) = rconn.prepare(&sql) else {
            return vectors;
        };
        let params: Vec<&dyn rusqlite::types::ToSql> =
            ids.iter().map(|id| *id as &dyn rusqlite::types::ToSql).collect();
        let expected_dim = crate::embedding::vector_dim();
        if let Ok(rows) = stmt.query_map(params.as_slice(), |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, Vec<u8>>(1)?))
        }) {
            for (id, blob) in rows.flatten() {
                let vector = crate::embedding::blob_to_vec(&blob);
                if vector.len() == expected_dim {
                    vectors.insert(id, vector);
                }
            }
        }
        vectors
    }

    /// `date` entities of the given memories, as calendar days.
    pub(super) fn batch_memory_dates(
        &self,
        ids: &[&String],
    ) -> std::collections::HashMap<String, Vec<chrono::NaiveDate>> {
        let mut dates: std::collections::HashMap<String, Vec<chrono::NaiveDate>> =
            std::collections::HashMap::new();
        if ids.is_empty() {
            return dates;
        }
        // Candidate pools are a few hundred ids at most; chunk to stay
        // under SQLite's default variable limit.
        for chunk in ids.chunks(400) {
            let placeholders: Vec<String> =
                (1..=chunk.len()).map(|i| format!("?{}", i)).collect();
            let sql = format!(
                "SELECT memory_id, entity_value FROM memory_entities WHERE entity_kind = 'date' AND memory_id IN ({})",
                placeholders.join(",")
            );
            let Ok(mut stmt) = self.conn.prepare(&sql) else {
                continue;
            };
            let params: Vec<&dyn rusqlite::types::ToSql> =
                chunk.iter().map(|id| *id as &dyn rusqlite::types::ToSql).collect();
            let rows: Vec<(String, String)> = match stmt.query_map(params.as_slice(), |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            }) {
                Ok(rows) => rows.flatten().collect(),
                Err(_) => continue,
            };
            for (id, value) in rows {
                if let Some(day) = crate::temporal::day_of_timestamp(&value) {
                    dates.entry(id).or_default().push(day);
                }
            }
        }
        dates
    }
}
