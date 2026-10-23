use uuid::Uuid;

use super::{BulkItem, Database, MemoryScope, TranscriptAddReport};

#[derive(Debug, Clone)]
struct DistilledTranscriptCandidate {
    score: i32,
    normalized_key: String,
    item: BulkItem,
}


/// Long fenced code blocks and agent self-narration are the bulk of a raw
/// transcript but almost never what a later recall needs (the distilled
/// memories keep the decisions; the repo keeps the code). They are trimmed
/// before chunking: a code block keeps its first ~220 characters plus a
/// marker, narration sentences ("Let me check…", "Je vais regarder…") are
/// dropped. Line structure ("user:" / "assistant:") is preserved.
/// `MEMORYPILOT_TRANSCRIPT_COMPACT=off` disables it.
pub(crate) fn compact_raw_transcript(content: &str) -> String {
    if std::env::var("MEMORYPILOT_TRANSCRIPT_COMPACT")
        .map(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "off" | "0" | "false"))
        .unwrap_or(false)
    {
        return content.to_string();
    }
    const CODE_KEEP: usize = 220;
    const CODE_MAX: usize = 420;

    // 1. Fenced code blocks (may span lines).
    let mut out = String::with_capacity(content.len());
    let mut rest = content;
    while let Some(start) = rest.find("```") {
        out.push_str(&rest[..start]);
        let after = &rest[start + 3..];
        let (block, tail, closed) = match after.find("```") {
            Some(end) => (&rest[start..start + 3 + end + 3], &after[end + 3..], true),
            None => (&rest[start..], "", false),
        };
        let chars = block.chars().count();
        if chars > CODE_MAX {
            let kept: String = block.chars().take(CODE_KEEP).collect();
            out.push_str(kept.trim_end());
            out.push_str(&format!(" … [code omis : {} caractères]", chars - CODE_KEEP));
            if closed {
                out.push_str(" ```");
            }
        } else {
            out.push_str(block);
        }
        rest = tail;
    }
    out.push_str(rest);

    // 2. Agent narration sentences, line by line.
    const NARRATION: &[&str] = &[
        "let me ", "now i ", "now let", "i'm going to", "i am going to", "i'm realizing",
        "i'm thinking", "i'll ", "i will ", "i need to ", "i should ", "okay, so", "okay, let",
        "alright", "let's check", "let's see", "let's look", "je vais ", "voyons", "maintenant je",
    ];
    let mut result = String::with_capacity(out.len());
    for (index, line) in out.split('\n').enumerate() {
        if index > 0 {
            result.push('\n');
        }
        let mut kept: Vec<&str> = Vec::new();
        let mut start = 0;
        let bytes = line.as_bytes();
        for i in 0..bytes.len() {
            let end_of_sentence = matches!(bytes[i], b'.' | b'!' | b'?')
                && (i + 1 == bytes.len() || bytes[i + 1] == b' ');
            if end_of_sentence || i + 1 == bytes.len() {
                let sentence = &line[start..=i];
                let trimmed = sentence.trim_start();
                // Keep the role prefix ("assistant: Let me...") but drop the narration after it.
                let (prefix, body) = match trimmed.find(": ") {
                    Some(p) if p <= 10 && trimmed[..p].chars().all(|c| c.is_ascii_alphabetic()) => {
                        (&trimmed[..p + 2], &trimmed[p + 2..])
                    }
                    _ => ("", trimmed),
                };
                let lower = body.trim_start().to_lowercase();
                if NARRATION.iter().any(|n| lower.starts_with(n)) && !body.contains('`') {
                    if !prefix.is_empty() {
                        kept.push(prefix.trim_end());
                    }
                } else {
                    kept.push(sentence.trim());
                }
                start = i + 1;
            }
        }
        result.push_str(&kept.join(" "));
    }
    result
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

    /// Agent process chatter ("Checking…", "Let me…", "Je vais vérifier…"),
    /// sentences that only make sense in context ("It…", "That leaves…") and
    /// markdown scaffolding are not durable knowledge. `lowered` must be
    /// lowercase. User sentences never go through this filter.
    pub fn is_agent_narration(lowered: &str) -> bool {
        const OPENERS: [&str; 58] = [
            "checking", "let me", "let's", "i'll", "i will", "i'm going", "i am going",
            "i'm now", "i'm checking", "i'm looking", "i'm holding", "i need to", "i see",
            "i found", "i've ", "i have ", "i can", "i think", "now i", "now let", "now,",
            "first,", "next,", "then,", "looking at", "reading", "searching", "running",
            "okay", "ok,", "ok ", "great", "perfect", "good,", "alright", "sure",
            "it ", "it's", "this ", "that ", "these ", "those ", "they ", "there ",
            "je vais", "je regarde", "je vérifie", "je lance", "je relance", "je commence",
            "maintenant", "d'abord", "ensuite", "voyons", "parfait", "d'accord", "ça ",
            "cela ",
        ];
        if OPENERS.iter().any(|opener| lowered.starts_with(opener)) {
            return true;
        }
        lowered.contains("##") || lowered.starts_with('#') || lowered.contains("](http")
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
        let lowered = content.to_lowercase();
        if lowered.ends_with('?') {
            return None;
        }
        if role == Some("assistant") && Self::is_agent_narration(&lowered) {
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
        let compacted = compact_raw_transcript(content);
        let chunks = Self::split_transcript_chunks(&compacted, 2000);

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

impl Database {
    /// Archive old raw transcripts. The raw chunks are written to a JSONL
    /// file under `~/.MemoryPilot/archive/` (nothing is lost, re-importable)
    /// and removed from the live index, where they are noise for recall.
    /// Optional `distill` first extracts rule-based memories (off by default:
    /// on long agent logs the heuristics mostly pick up tool chatter).
    /// Dry run by default.
    pub fn archive_transcripts(
        &self,
        min_age_days: i64,
        distill: bool,
        apply: bool,
    ) -> Result<serde_json::Value, String> {
        let age = min_age_days.max(30);
        let cutoff = (chrono::Utc::now() - chrono::Duration::days(age)).to_rfc3339();
        type Row = (String, String, Option<String>, String, String, i64, Option<String>, String);
        let rows: Vec<Row> = {
            let mut stmt = self
                .conn
                .prepare(
                    "SELECT id, content, project, tags, source, importance, metadata, created_at
                     FROM memories
                     WHERE kind = 'transcript' AND created_at < ?1
                       AND tags NOT LIKE '%pinned%'
                     ORDER BY created_at",
                )
                .map_err(|e| format!("archive prepare: {}", e))?;
            let rows = stmt
                .query_map(rusqlite::params![cutoff], |r| {
                    Ok((
                        r.get(0)?,
                        r.get(1)?,
                        r.get(2)?,
                        r.get(3)?,
                        r.get::<_, Option<String>>(4)?.unwrap_or_default(),
                        r.get(5)?,
                        r.get(6)?,
                        r.get(7)?,
                    ))
                })
                .map_err(|e| format!("archive query: {}", e))?;
            rows.flatten().collect()
        };

        let mut groups: std::collections::BTreeMap<String, Vec<(i64, usize)>> =
            std::collections::BTreeMap::new();
        for (i, row) in rows.iter().enumerate() {
            let meta: serde_json::Value = row
                .6
                .as_deref()
                .and_then(|m| serde_json::from_str(m).ok())
                .unwrap_or(serde_json::Value::Null);
            let tid = meta
                .get("transcript_id")
                .and_then(|v| v.as_str())
                .map(String::from)
                .unwrap_or_else(|| row.0.clone());
            let index = meta.get("chunk_index").and_then(|v| v.as_i64()).unwrap_or(0);
            groups.entry(tid).or_default().push((index, i));
        }

        let chars: usize = rows.iter().map(|r| r.1.len()).sum();
        let mut distilled = 0usize;
        let mut added = 0usize;
        if distill {
            for (tid, parts) in groups.iter_mut() {
                parts.sort_by_key(|p| p.0);
                let first = &rows[parts[0].1];
                let tags: Vec<String> = serde_json::from_str::<Vec<String>>(&first.3)
                    .unwrap_or_default()
                    .into_iter()
                    .filter(|t| t != "pinned")
                    .collect();
                let text = parts.iter().map(|p| rows[p.1].1.as_str()).collect::<Vec<_>>().join(" ");
                let items = Self::distill_transcript_memories(
                    &text,
                    first.2.as_deref(),
                    &tags,
                    "transcript-archive",
                    tid,
                    &MemoryScope::default(),
                );
                distilled += items.len();
                if apply && !items.is_empty() {
                    added += self.add_memories_bulk(&items)?.0.len();
                }
            }
        }

        let mut archive_file: Option<String> = None;
        if apply && !rows.is_empty() {
            let dir = dirs::home_dir()
                .ok_or("No home")?
                .join(".MemoryPilot")
                .join("archive");
            std::fs::create_dir_all(&dir).map_err(|e| format!("archive dir: {}", e))?;
            let path = dir.join(format!(
                "transcripts-{}.jsonl",
                chrono::Utc::now().format("%Y%m%d-%H%M%S")
            ));
            let mut out = String::new();
            for r in &rows {
                out.push_str(
                    &serde_json::json!({
                        "id": r.0, "content": r.1, "project": r.2, "tags": r.3,
                        "source": r.4, "importance": r.5, "metadata": r.6, "created_at": r.7,
                    })
                    .to_string(),
                );
                out.push('\n');
            }
            std::fs::write(&path, out).map_err(|e| format!("archive write: {}", e))?;
            for r in &rows {
                self.delete_memory(&r.0)?;
            }
            archive_file = Some(path.display().to_string());
        }

        Ok(serde_json::json!({
            "applied": apply,
            "older_than_days": age,
            "transcripts": groups.len(),
            "chunks": rows.len(),
            "chars": chars,
            "distilled_candidates": distilled,
            "distilled_added": added,
            "archive_file": archive_file,
        }))
    }
}

#[cfg(test)]
mod compact_tests {
    use super::compact_raw_transcript;

    #[test]
    fn long_code_block_is_trimmed_short_one_kept() {
        let long = format!("Voici : ```rust {} ``` fin.", "let x = 1; ".repeat(80));
        let out = compact_raw_transcript(&long);
        assert!(out.contains("[code omis"));
        assert!(out.len() < long.len() / 2);
        let short = "Commande : ```npm run build``` ok.";
        assert_eq!(compact_raw_transcript(short), short);
    }

    #[test]
    fn narration_dropped_facts_kept() {
        let text = "assistant: Let me check the build. The build uses `cargo build --release`. Je vais regarder les logs. Le binaire est dans target/release.";
        let out = compact_raw_transcript(text);
        assert!(!out.contains("Let me check"));
        assert!(!out.contains("Je vais regarder"));
        assert!(out.contains("cargo build --release"));
        assert!(out.contains("Le binaire est dans target/release."));
        assert!(out.starts_with("assistant:"));
    }

    #[test]
    fn line_structure_preserved() {
        let text = "user: Salut.\nassistant: Let me look. Voici la réponse.";
        let out = compact_raw_transcript(text);
        assert_eq!(out.lines().count(), 2);
        assert!(out.lines().nth(1).unwrap().contains("Voici la réponse."));
    }
}
