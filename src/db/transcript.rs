use uuid::Uuid;

use super::{BulkItem, Database, MemoryScope, TranscriptAddReport};

#[derive(Debug, Clone)]
struct DistilledTranscriptCandidate {
    score: i32,
    normalized_key: String,
    item: BulkItem,
}

impl Database {
    fn split_transcript_chunks(content: &str, chunk_size: usize) -> Vec<String> {
        crate::splitter::split_memory_text(content, chunk_size)
    }

    fn transcript_segments(content: &str) -> Vec<(Option<&'static str>, String)> {
        let mut segments = Vec::new();

        for raw_line in content.lines() {
            let trimmed_line = raw_line.trim();
            if trimmed_line.is_empty() {
                continue;
            }

            let lower = trimmed_line.to_ascii_lowercase();
            let (role, body) = if lower.starts_with("user:") {
                (Some("user"), trimmed_line[5..].trim())
            } else if lower.starts_with("assistant:") {
                (Some("assistant"), trimmed_line[10..].trim())
            } else if lower.starts_with("system:") {
                (Some("system"), trimmed_line[7..].trim())
            } else if lower.starts_with("developer:") {
                (Some("developer"), trimmed_line[10..].trim())
            } else {
                (None, trimmed_line)
            };

            let mut current = String::new();
            let mut chars = body.chars().peekable();
            while let Some(character) = chars.next() {
                current.push(character);
                if matches!(character, '.' | '!' | '?') {
                    let next_is_space = chars
                        .peek()
                        .map(|next| next.is_whitespace())
                        .unwrap_or(true);
                    if next_is_space {
                        let candidate = current.trim();
                        if !candidate.is_empty() {
                            segments.push((role, candidate.to_string()));
                        }
                        current.clear();
                    }
                }
            }

            let remaining = current.trim();
            if !remaining.is_empty() {
                segments.push((role, remaining.to_string()));
            }
        }

        segments
    }

    fn normalized_transcript_segment(segment: &str) -> Option<String> {
        let cleaned = segment
            .trim()
            .trim_matches(|character: char| {
                matches!(character, '"' | '\'' | '`' | '-' | '*' | ':' | ';')
            })
            .replace("  ", " ");
        if cleaned.len() < 24 || cleaned.len() > 260 {
            return None;
        }

        let lowered = cleaned.to_ascii_lowercase();
        if lowered.starts_with("http://")
            || lowered.starts_with("https://")
            || lowered.starts_with("tool:")
            || lowered.starts_with("[tool")
            || lowered.starts_with("[thinking")
            || lowered.starts_with("```")
            || cleaned
                .chars()
                .filter(|character| !character.is_alphanumeric() && !character.is_whitespace())
                .count()
                > cleaned.len() / 3
        {
            return None;
        }

        Some(cleaned)
    }

    fn transcript_candidate_limit(kind: &str) -> usize {
        match kind {
            "preference" => 3,
            "decision" => 3,
            "milestone" => 2,
            "bug" => 2,
            "todo" => 2,
            "fact" => 3,
            _ => 1,
        }
    }

    fn build_transcript_candidate(
        segment: &str,
        role: Option<&str>,
        project: Option<&str>,
        tags: &[String],
        source: &str,
        transcript_id: &str,
        scope: &MemoryScope,
    ) -> Option<DistilledTranscriptCandidate> {
        let content = Self::normalized_transcript_segment(segment)?;
        let lowered = content.to_ascii_lowercase();
        if lowered.ends_with('?') {
            return None;
        }

        let has_file = content.contains('/') && content.contains('.');
        let entities = crate::graph::extract_entities(&content, project);
        let entity_bonus = entities.len() as i32;

        let preference_markers = [
            "always",
            "never",
            "prefer",
            "do not",
            "don't",
            "must",
            "should",
            "need to",
            "my rule",
            "i like",
            "i hate",
            "convention",
            "toujours",
            "ne jamais",
            "je veux",
            "il faut",
            "pas de",
            "je préfère",
        ];
        let decision_markers = [
            "decided",
            "chose",
            "switched to",
            "instead of",
            "trade-off",
            "because we",
            "we will",
            "we'll",
            "will use",
            "use ",
            "uses ",
            "keep ",
            "switch ",
            "migrate ",
            "supports ",
            "store ",
            "now supports",
            "on va",
            "garder",
            "utiliser",
            "choisir",
            "on a décidé",
        ];
        let todo_markers = [
            "todo",
            "next step",
            "follow up",
            "remaining",
            "pending",
            "should add",
            "prochaine étape",
            "reste à",
            "à faire",
        ];
        let bug_markers = [
            "bug",
            "error",
            "failed",
            "fails",
            "failing",
            "panic",
            "crash",
            "not found",
            "cannot",
            "doesn't work",
            "does not work",
            "broken",
            "root cause",
            "workaround",
            "ne marche",
            "erreur",
            "échoue",
            "cassé",
        ];
        let milestone_markers = [
            "it works",
            "breakthrough",
            "figured out",
            "shipped",
            "deployed",
            "released",
            "completed",
            "done",
            "finally",
            "working now",
            "launched",
            "resolved",
            "ça marche",
            "terminé",
            "fini",
            "déployé",
            "livré",
        ];
        let resolution_markers = [
            "fixed",
            "solved",
            "resolved",
            "patched",
            "corrected",
            "réglé",
            "corrigé",
        ];
        let personal_fact_markers = [
            "i am ",
            "i'm ",
            "i work",
            "i currently",
            "my current role",
            "my previous",
            "i bought",
            "i received",
            "i mentioned",
            "i have been",
            "i've been",
            "my sister",
            "my brother",
            "my mother",
            "my mom",
            "my father",
            "my dad",
            "my friend",
            "my partner",
            "my job",
            "mon travail",
            "ma soeur",
            "mon frère",
            "ma mère",
            "mon père",
        ];

        let contains_any = |markers: &[&str]| markers.iter().any(|marker| lowered.contains(marker));
        let count_matches = |markers: &[&str]| {
            markers
                .iter()
                .filter(|marker| lowered.contains(**marker))
                .count() as i32
        };

        let pref_score = count_matches(&preference_markers);
        let dec_score = count_matches(&decision_markers);
        let bug_score = count_matches(&bug_markers);
        let todo_score = count_matches(&todo_markers);
        let mile_score = count_matches(&milestone_markers);
        let resolution_hits = count_matches(&resolution_markers);
        let personal_fact_score = count_matches(&personal_fact_markers);

        let is_resolved_problem = bug_score > 0 && resolution_hits > 0;

        let (kind, importance, score) =
            if contains_any(&preference_markers) && role == Some("user") && pref_score >= 1 {
                ("preference", 5, 16 + entity_bonus + pref_score)
            } else if is_resolved_problem || (mile_score >= 1 && bug_score == 0) {
                (
                    "milestone",
                    4,
                    15 + entity_bonus + mile_score + resolution_hits,
                )
            } else if bug_score >= 1 && !is_resolved_problem {
                (
                    "bug",
                    4,
                    14 + entity_bonus + bug_score + if has_file { 2 } else { 0 },
                )
            } else if dec_score >= 1 && (has_file || entity_bonus > 0 || role.is_some()) {
                (
                    "decision",
                    4,
                    13 + entity_bonus + dec_score + if role == Some("user") { 1 } else { 0 },
                )
            } else if role == Some("user") && personal_fact_score >= 1 {
                ("fact", 4, 12 + entity_bonus + personal_fact_score)
            } else if todo_score >= 1 || (role == Some("assistant") && lowered.contains("next")) {
                ("todo", 3, 11 + entity_bonus + todo_score)
            } else if (entity_bonus >= 2 || has_file) && lowered.contains("safe")
                || lowered.contains("mode")
                || lowered.contains("benchmark")
                || lowered.contains("scope")
                || lowered.contains("project")
                || lowered.contains("credential")
            {
                ("fact", 3, 9 + entity_bonus + if has_file { 2 } else { 0 })
            } else {
                return None;
            };

        if score < 9 {
            return None;
        }

        let mut candidate_tags = tags.to_vec();
        for tag in ["transcript", "distilled", kind] {
            if !candidate_tags.iter().any(|existing| existing == tag) {
                candidate_tags.push(tag.to_string());
            }
        }

        let metadata = serde_json::json!({
            "transcript_id": transcript_id,
            "distilled_from": "transcript",
            "speaker_role": role,
            "distillation_score": score,
        });

        Some(DistilledTranscriptCandidate {
            score,
            normalized_key: content.to_ascii_lowercase(),
            item: BulkItem {
                content,
                kind: kind.to_string(),
                project: project.map(String::from),
                tags: Some(candidate_tags),
                source: format!("{}:transcript-distilled", source),
                importance: Some(importance),
                expires_at: None,
                metadata: Some(metadata),
                session_id: scope.session_id.clone(),
                thread_id: scope.thread_id.clone(),
                window_id: scope.window_id.clone(),
            },
        })
    }

    fn distill_transcript_memories(
        content: &str,
        project: Option<&str>,
        tags: &[String],
        source: &str,
        transcript_id: &str,
        scope: &MemoryScope,
    ) -> Vec<BulkItem> {
        let mut candidates = Vec::new();
        for (role, segment) in Self::transcript_segments(content) {
            if let Some(candidate) = Self::build_transcript_candidate(
                &segment,
                role,
                project,
                tags,
                source,
                transcript_id,
                scope,
            ) {
                candidates.push(candidate);
            }
        }

        candidates.sort_by(|left, right| {
            right
                .score
                .cmp(&left.score)
                .then_with(|| right.item.importance.cmp(&left.item.importance))
                .then_with(|| right.item.content.len().cmp(&left.item.content.len()))
        });

        let mut seen = std::collections::HashSet::new();
        let mut kind_counts: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        let mut selected = Vec::new();

        for candidate in candidates {
            if selected.len() >= 10 {
                break;
            }
            if !seen.insert(candidate.normalized_key.clone()) {
                continue;
            }

            let kind_limit = Self::transcript_candidate_limit(&candidate.item.kind);
            let count = kind_counts.entry(candidate.item.kind.clone()).or_insert(0);
            if *count >= kind_limit {
                continue;
            }

            *count += 1;
            selected.push(candidate.item);
        }

        selected
    }

    pub fn add_transcript(
        &self,
        content: &str,
        project: Option<&str>,
        tags: &[String],
        source: &str,
        scope: &MemoryScope,
        distill: bool,
    ) -> Result<TranscriptAddReport, String> {
        let transcript_id = Uuid::new_v4().to_string();
        let chunks = Self::split_transcript_chunks(content, 2000);

        let transcript_items = chunks
            .iter()
            .enumerate()
            .map(|(index, chunk)| BulkItem {
                content: chunk.clone(),
                kind: "transcript".to_string(),
                project: project.map(String::from),
                tags: Some(tags.to_vec()),
                source: source.to_string(),
                importance: Some(3),
                expires_at: None,
                metadata: Some(serde_json::json!({
                    "transcript_id": transcript_id,
                    "chunk_index": index,
                    "total_chunks": chunks.len()
                })),
                session_id: scope.session_id.clone(),
                thread_id: scope.thread_id.clone(),
                window_id: scope.window_id.clone(),
            })
            .collect::<Vec<_>>();

        let (chunk_added, chunk_merged, chunk_skipped) =
            match self.add_memories_bulk(&transcript_items) {
                Ok((added, merged, skipped)) => (added.len(), merged, skipped),
                Err(error) => return Err(error),
            };

        let distilled_items = if distill {
            Self::distill_transcript_memories(content, project, tags, source, &transcript_id, scope)
        } else {
            Vec::new()
        };
        let distilled_candidates = distilled_items.len();
        let (distilled_added, distilled_merged, distilled_skipped) = if distilled_items.is_empty() {
            (0, 0, 0)
        } else {
            match self.add_memories_bulk(&distilled_items) {
                Ok((added, merged, skipped)) => (added.len(), merged, skipped),
                Err(error) => return Err(error),
            }
        };

        Ok(TranscriptAddReport {
            transcript_id,
            chunks_total: chunks.len(),
            chunk_added,
            chunk_merged,
            chunk_skipped,
            distilled_candidates,
            distilled_added,
            distilled_merged,
            distilled_skipped,
        })
    }

    pub fn ingest_session_transcript(
        &self,
        content: &str,
        project: Option<&str>,
        tags: &[String],
        source: &str,
        scope: &MemoryScope,
        store_raw_transcript: bool,
    ) -> Result<TranscriptAddReport, String> {
        if store_raw_transcript {
            return self.add_transcript(content, project, tags, source, scope, true);
        }

        let transcript_id = Uuid::new_v4().to_string();
        let distilled_items = Self::distill_transcript_memories(
            content,
            project,
            tags,
            source,
            &transcript_id,
            scope,
        );
        let distilled_candidates = distilled_items.len();
        let (distilled_added, distilled_merged, distilled_skipped) = if distilled_items.is_empty() {
            (0, 0, 0)
        } else {
            match self.add_memories_bulk(&distilled_items) {
                Ok((added, merged, skipped)) => (added.len(), merged, skipped),
                Err(error) => return Err(error),
            }
        };

        Ok(TranscriptAddReport {
            transcript_id,
            chunks_total: 0,
            chunk_added: 0,
            chunk_merged: 0,
            chunk_skipped: 0,
            distilled_candidates,
            distilled_added,
            distilled_merged,
            distilled_skipped,
        })
    }
}
