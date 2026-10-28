//! Finding a conversation from a vague description.
//!
//! Three lanes are fused per message with reciprocal rank fusion, then
//! grouped per conversation:
//! - exact: the query's words and their stems (BM25);
//! - fuzzy: vocabulary words one or two edits away, and adjacent words
//!   glued together, for speech-to-text slips ("Gina" for Jina, "Memory
//!   Pilot" for MemoryPilot);
//! - vector: the multilingual encoder MemoryPilot already ships.

use super::store::{project_name, ConversationStore};
use super::text;
use rusqlite::params;
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::sync::{Mutex, OnceLock};

const RRF_K: f64 = 60.0;
const LANE_DEPTH: usize = 300;
const FUZZY_WEIGHT: f64 = 0.6;
const VECTOR_WEIGHT: f64 = 1.0;
const EXACT_WEIGHT: f64 = 1.0;
/// A sub-agent's messages are the agent briefing itself: they describe
/// the topic well but should not outrank what the user said.
const SUBAGENT_WEIGHT: f64 = 0.5;
/// Bonus for a message holding all the query words rather than one.
const COVERAGE_WEIGHT: f64 = 2.0;
const SIZE_DAMPING: f64 = 0.25;


#[derive(Debug, Clone, Default)]
pub struct SearchOptions {
    pub query: String,
    pub limit: usize,
    pub tool: Option<String>,
    pub project: Option<String>,
    pub after: Option<String>,
    pub before: Option<String>,
    pub use_vectors: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct MatchedMessage {
    pub message_id: i64,
    pub role: String,
    pub at: Option<String>,
    pub snippet: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ConversationHit {
    pub conversation_id: String,
    pub tool: String,
    pub title: Option<String>,
    pub project: Option<String>,
    pub workspace: Option<String>,
    pub created_at: Option<String>,
    pub updated_at: Option<String>,
    pub message_count: i64,
    pub score: f64,
    pub best_match: MatchedMessage,
    pub other_matches: usize,
    pub commits: Vec<String>,
    pub files: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub matched_through: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ConversationMessage {
    pub message_id: i64,
    pub role: String,
    pub at: Option<String>,
    pub text: String,
}

struct SegmentRow {
    conversation_id: String,
    role: String,
    at: Option<String>,
    text: String,
    is_main: bool,
    message_count: i64,
}

/// Accepted spellings of each query word: itself, its stem as a prefix,
/// and the close vocabulary words found for speech-to-text slips.
struct TermForms {
    exact: Vec<String>,
    prefixes: Vec<String>,
}

fn term_forms(terms: &[String], fuzzy: &[(String, String)]) -> Vec<TermForms> {
    terms
        .iter()
        .map(|term| {
            let mut exact = vec![term.clone()];
            exact.extend(fuzzy.iter().filter(|(source, _)| source == term).map(|(_, alt)| alt.clone()));
            let prefixes = text::word_tokens(&crate::stemming::stem_query(term))
                .into_iter()
                .filter(|stem| stem.chars().count() >= 4)
                .collect();
            TermForms { exact, prefixes }
        })
        .collect()
}

/// Share of the query words present in one message.
fn coverage(forms: &[TermForms], body: &str) -> f64 {
    if forms.is_empty() {
        return 0.0;
    }
    let tokens: HashSet<String> = text::word_tokens(&text::fold(body)).into_iter().collect();
    let covered = forms
        .iter()
        .filter(|form| {
            form.exact.iter().any(|word| tokens.contains(word))
                || form.prefixes.iter().any(|prefix| tokens.iter().any(|token| token.starts_with(prefix.as_str())))
        })
        .count();
    covered as f64 / forms.len() as f64
}

impl ConversationStore {
    pub fn search(&self, options: &SearchOptions) -> Result<Vec<ConversationHit>, String> {
        let limit = options.limit.clamp(1, 50);
        let terms = text::query_terms(&options.query);
        if terms.is_empty() && options.query.trim().is_empty() {
            return Ok(Vec::new());
        }

        let mut scores: HashMap<i64, f64> = HashMap::new();
        let mut lanes: HashMap<i64, HashSet<&'static str>> = HashMap::new();
        let mut add_lane = |ranked: Vec<i64>, weight: f64, name: &'static str| {
            for (rank, id) in ranked.into_iter().enumerate() {
                *scores.entry(id).or_default() += weight / (RRF_K + rank as f64 + 1.0);
                lanes.entry(id).or_default().insert(name);
            }
        };

        let exact = exact_expression(&terms);
        if let Some(expression) = exact {
            add_lane(self.fts_ranked(&expression)?, EXACT_WEIGHT, "exact");
        }
        let fuzzy_terms = self.fuzzy_expansions(&terms)?;
        let forms = term_forms(&terms, &fuzzy_terms);
        if !fuzzy_terms.is_empty() {
            let expression = or_expression(fuzzy_terms.iter().map(|(_, alt)| alt.as_str()));
            add_lane(self.fts_ranked(&expression)?, FUZZY_WEIGHT, "fuzzy");
        }
        if options.use_vectors {
            let query_vector = crate::embedding::embed_query(&options.query);
            add_lane(self.vector_ranked(&query_vector)?, VECTOR_WEIGHT, "vector");
        }
        if scores.is_empty() {
            return Ok(Vec::new());
        }

        let ids: Vec<i64> = scores.keys().copied().collect();
        let rows = self.segment_rows(&ids)?;
        let mut per_conversation: HashMap<String, Vec<(f64, i64)>> = HashMap::new();
        let mut sizes: HashMap<String, i64> = HashMap::new();
        for (id, score) in &scores {
            let Some(row) = rows.get(id) else { continue };
            if !within(row.at.as_deref(), options.after.as_deref(), options.before.as_deref()) {
                continue;
            }
            let role_weight = match row.role.as_str() {
                "title" => 1.15,
                "user" => 1.0,
                _ => 0.85,
            };
            let origin_weight = if row.is_main { 1.0 } else { SUBAGENT_WEIGHT };
            let coverage_weight = 1.0 + COVERAGE_WEIGHT * coverage(&forms, &row.text);
            per_conversation
                .entry(row.conversation_id.clone())
                .or_default()
                .push((score * role_weight * origin_weight * coverage_weight, *id));
            sizes.insert(row.conversation_id.clone(), row.message_count);
        }

        let mut ranked: Vec<(String, f64, Vec<(f64, i64)>)> = per_conversation
            .into_iter()
            .map(|(conversation, mut matches)| {
                matches.sort_by(|a, b| b.0.total_cmp(&a.0));
                // Best message, plus a little for corroborating ones: a
                // long conversation must not win on volume alone.
                let total = matches
                    .iter()
                    .take(3)
                    .enumerate()
                    .map(|(i, (s, _))| s * [1.0, 0.35, 0.15][i])
                    .sum::<f64>();
                // A conversation with thousands of messages lands a good
                // match on almost any query; damp that by its size.
                let size = sizes.get(&conversation).copied().unwrap_or(0).max(0) as f64;
                let total = total / (1.0 + SIZE_DAMPING * (1.0 + size / 50.0).ln());
                (conversation, total, matches)
            })
            .collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));

        let mut hits = Vec::new();
        for (conversation_id, score, matches) in ranked {
            let Some(mut hit) = self.conversation_hit(&conversation_id, score, &matches, &rows, &terms)? else {
                continue;
            };
            if let Some(tool) = options.tool.as_deref() {
                if !hit.tool.eq_ignore_ascii_case(tool) {
                    continue;
                }
            }
            if let Some(project) = options.project.as_deref() {
                let wanted = project.to_lowercase();
                let matches_project = [hit.project.as_deref(), hit.workspace.as_deref()]
                    .into_iter()
                    .flatten()
                    .any(|p| p.to_lowercase().contains(&wanted));
                if !matches_project {
                    continue;
                }
            }
            let best_id = hit.best_match.message_id;
            let mut through: Vec<String> = lanes
                .get(&best_id)
                .map(|set| set.iter().map(|s| s.to_string()).collect())
                .unwrap_or_default();
            through.sort();
            hit.matched_through = through;
            hits.push(hit);
            if hits.len() == limit {
                break;
            }
        }
        Ok(hits)
    }

    fn fts_ranked(&self, expression: &str) -> Result<Vec<i64>, String> {
        let mut statement = self
            .conn
            .prepare_cached("SELECT rowid FROM segments_fts WHERE segments_fts MATCH ?1 ORDER BY rank LIMIT ?2")
            .map_err(|error| format!("FTS search: {}", error))?;
        let rows = statement
            .query_map(params![expression, LANE_DEPTH as i64], |row| row.get::<_, i64>(0))
            .map_err(|error| format!("FTS search: {}", error))?;
        Ok(rows.filter_map(Result::ok).collect())
    }

    /// Vocabulary words close to a query word the index barely knows.
    /// Returns (query term, replacement) pairs.
    fn fuzzy_expansions(&self, terms: &[String]) -> Result<Vec<(String, String)>, String> {
        let vocabulary = self.vocabulary()?;
        let glued = text::glued_pairs(terms);
        let mut expansions = Vec::new();
        for term in terms.iter().chain(glued.iter()) {
            let own = vocabulary.frequency(term);
            let is_glued = glued.contains(term);
            if is_glued && own > 0 {
                expansions.push((term.clone(), term.clone()));
                continue;
            }
            // A word the index knows well is not a transcription slip.
            if own >= 5 {
                continue;
            }
            let max = text::allowed_edits(term);
            if max == 0 {
                continue;
            }
            let mut close: Vec<(usize, u32, &str)> = vocabulary
                .near_length(term, max)
                .filter(|(word, _)| word != term)
                .filter_map(|(word, frequency)| {
                    text::edit_distance_within(term, word, max).map(|distance| (distance, frequency, word))
                })
                .collect();
            close.sort_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)));
            for (_, _, word) in close.into_iter().take(3) {
                expansions.push((term.clone(), word.to_string()));
            }
        }
        expansions.sort();
        expansions.dedup();
        Ok(expansions)
    }

    fn vocabulary(&self) -> Result<std::sync::Arc<Vocabulary>, String> {
        static CACHE: OnceLock<Mutex<Option<(i64, std::sync::Arc<Vocabulary>)>>> = OnceLock::new();
        let generation: i64 = self
            .conn
            .query_row("SELECT COALESCE(MAX(id), 0) + COUNT(*) FROM segments", [], |row| row.get(0))
            .map_err(|error| format!("Vocabulary generation: {}", error))?;
        let cache = CACHE.get_or_init(|| Mutex::new(None));
        if let Some((cached, vocabulary)) = cache.lock().map_err(|_| "Vocabulary lock")?.as_ref() {
            if *cached == generation {
                return Ok(vocabulary.clone());
            }
        }
        let mut statement = self
            .conn
            .prepare("SELECT term, doc FROM segments_vocab WHERE doc >= 2")
            .map_err(|error| format!("Vocabulary: {}", error))?;
        let rows = statement
            .query_map([], |row| Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?)))
            .map_err(|error| format!("Vocabulary: {}", error))?;
        let mut by_length: HashMap<usize, Vec<(String, u32)>> = HashMap::new();
        let mut frequency: HashMap<String, u32> = HashMap::new();
        for (term, doc) in rows.filter_map(Result::ok) {
            let length = term.chars().count();
            if !(3..=32).contains(&length) || term.chars().any(|c| c.is_ascii_digit()) {
                continue;
            }
            let doc = doc.min(u32::MAX as i64) as u32;
            frequency.insert(term.clone(), doc);
            by_length.entry(length).or_default().push((term, doc));
        }
        let vocabulary = std::sync::Arc::new(Vocabulary { by_length, frequency });
        *cache.lock().map_err(|_| "Vocabulary lock")? = Some((generation, vocabulary.clone()));
        Ok(vocabulary)
    }

    fn vector_ranked(&self, query: &[f32]) -> Result<Vec<i64>, String> {
        let mut statement = self
            .conn
            .prepare_cached("SELECT id, embedding FROM segments WHERE embed_state = 2")
            .map_err(|error| format!("Vector scan: {}", error))?;
        let rows = statement
            .query_map([], |row| Ok((row.get::<_, i64>(0)?, row.get::<_, Vec<u8>>(1)?)))
            .map_err(|error| format!("Vector scan: {}", error))?;
        let mut scored: Vec<(f32, i64)> = rows
            .filter_map(Result::ok)
            .map(|(id, blob)| (crate::embedding::similarity_with_blob(query, &blob), id))
            .collect();
        let depth = LANE_DEPTH.min(scored.len());
        if depth == 0 {
            return Ok(Vec::new());
        }
        scored.select_nth_unstable_by(depth - 1, |a, b| b.0.total_cmp(&a.0));
        scored.truncate(depth);
        scored.sort_by(|a, b| b.0.total_cmp(&a.0));
        Ok(scored.into_iter().map(|(_, id)| id).collect())
    }

    fn segment_rows(&self, ids: &[i64]) -> Result<HashMap<i64, SegmentRow>, String> {
        let mut rows = HashMap::new();
        let mut statement = self
            .conn
            .prepare_cached(
                "SELECT s.conversation_id, s.role, s.at, s.text, COALESCE(o.is_main, 1), c.message_count
                 FROM segments s LEFT JOIN origins o ON o.id = s.origin_id
                 JOIN conversations c ON c.id = s.conversation_id WHERE s.id = ?1",
            )
            .map_err(|error| format!("Segment rows: {}", error))?;
        for id in ids {
            let row = statement.query_row([id], |row| {
                Ok(SegmentRow {
                    conversation_id: row.get(0)?,
                    role: row.get(1)?,
                    at: row.get(2)?,
                    text: row.get(3)?,
                    is_main: row.get::<_, i64>(4)? == 1,
                    message_count: row.get(5)?,
                })
            });
            if let Ok(row) = row {
                rows.insert(*id, row);
            }
        }
        Ok(rows)
    }

    fn conversation_hit(
        &self,
        conversation_id: &str,
        score: f64,
        matches: &[(f64, i64)],
        rows: &HashMap<i64, SegmentRow>,
        terms: &[String],
    ) -> Result<Option<ConversationHit>, String> {
        let meta = self
            .conn
            .query_row(
                "SELECT tool, title, project, workspace, created_at, updated_at, message_count
                 FROM conversations WHERE id = ?1",
                [conversation_id],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, Option<String>>(1)?,
                        row.get::<_, Option<String>>(2)?,
                        row.get::<_, Option<String>>(3)?,
                        row.get::<_, Option<String>>(4)?,
                        row.get::<_, Option<String>>(5)?,
                        row.get::<_, i64>(6)?,
                    ))
                },
            )
            .ok();
        let Some((tool, title, project, workspace, created_at, updated_at, message_count)) = meta else {
            return Ok(None);
        };
        let Some((_, best_id)) = matches.first() else { return Ok(None) };
        let Some(best) = rows.get(best_id) else { return Ok(None) };
        let refs = |kind: &str| -> Result<Vec<String>, String> {
            let mut statement = self
                .conn
                .prepare_cached(
                    "SELECT value FROM refs WHERE conversation_id = ?1 AND kind = ?2
                     GROUP BY value ORDER BY MAX(segment_id = ?3) DESC, COUNT(*) DESC LIMIT 5",
                )
                .map_err(|error| format!("Refs: {}", error))?;
            let values = statement
                .query_map(params![conversation_id, kind, best_id], |row| row.get::<_, String>(0))
                .map_err(|error| format!("Refs: {}", error))?
                .filter_map(Result::ok)
                .collect();
            Ok(values)
        };
        Ok(Some(ConversationHit {
            conversation_id: conversation_id.to_string(),
            tool,
            title,
            project: project.or_else(|| workspace.as_deref().map(project_name)),
            workspace,
            created_at,
            updated_at,
            message_count,
            score: (score * 1000.0).round() / 1000.0,
            best_match: MatchedMessage {
                message_id: *best_id,
                role: best.role.clone(),
                at: best.at.clone(),
                snippet: snippet(&best.text, terms, 320),
            },
            other_matches: matches.len().saturating_sub(1),
            commits: refs("commit")?,
            files: refs("file")?,
            matched_through: Vec::new(),
        }))
    }

    /// Messages of a conversation, oldest first. With `around` and no
    /// offset, the page is centred on that message. `next_offset` is an
    /// absolute position to pass back as `offset`.
    pub fn read(
        &self,
        conversation_id: &str,
        around: Option<i64>,
        offset: usize,
        limit: usize,
    ) -> Result<(Vec<ConversationMessage>, Option<usize>), String> {
        let limit = limit.clamp(1, 100);
        let start = match around.filter(|_| offset == 0) {
            Some(message_id) => {
                let position: i64 = self
                    .conn
                    .query_row(
                        "SELECT COUNT(*) FROM segments s, segments t
                         WHERE t.id = ?2 AND s.conversation_id = ?1 AND s.role != 'title'
                           AND (COALESCE(s.at, '') < COALESCE(t.at, '') OR (COALESCE(s.at, '') = COALESCE(t.at, '') AND s.id < t.id))",
                        params![conversation_id, message_id],
                        |row| row.get(0),
                    )
                    .map_err(|error| format!("Read position: {}", error))?;
                (position as usize).saturating_sub(limit / 2)
            }
            None => offset,
        };
        let mut statement = self
            .conn
            .prepare_cached(
                "SELECT id, role, at, text FROM segments WHERE conversation_id = ?1 AND role != 'title'
                 ORDER BY COALESCE(at, ''), id LIMIT ?2 OFFSET ?3",
            )
            .map_err(|error| format!("Read: {}", error))?;
        let messages: Vec<ConversationMessage> = statement
            .query_map(params![conversation_id, (limit + 1) as i64, start as i64], |row| {
                Ok(ConversationMessage {
                    message_id: row.get(0)?,
                    role: row.get(1)?,
                    at: row.get(2)?,
                    text: row.get(3)?,
                })
            })
            .map_err(|error| format!("Read: {}", error))?
            .filter_map(Result::ok)
            .collect();
        let has_more = messages.len() > limit;
        let messages: Vec<ConversationMessage> = messages.into_iter().take(limit).collect();
        Ok((messages, has_more.then_some(start + limit)))
    }

    /// Conversations that mention a commit or a file.
    pub fn by_reference(&self, kind: &str, value: &str, limit: usize) -> Result<Vec<(String, i64)>, String> {
        let mut statement = self
            .conn
            .prepare_cached(
                "SELECT conversation_id, COUNT(*) FROM refs WHERE kind = ?1 AND (value = ?2 OR value LIKE '%' || ?2)
                 GROUP BY conversation_id ORDER BY COUNT(*) DESC LIMIT ?3",
            )
            .map_err(|error| format!("References: {}", error))?;
        let rows = statement
            .query_map(params![kind, value, limit as i64], |row| Ok((row.get(0)?, row.get(1)?)))
            .map_err(|error| format!("References: {}", error))?;
        Ok(rows.filter_map(Result::ok).collect())
    }
}

pub(super) struct Vocabulary {
    by_length: HashMap<usize, Vec<(String, u32)>>,
    frequency: HashMap<String, u32>,
}

impl Vocabulary {
    fn frequency(&self, term: &str) -> u32 {
        self.frequency.get(term).copied().unwrap_or(0)
    }

    fn near_length<'a>(&'a self, term: &str, max: usize) -> impl Iterator<Item = (&'a str, u32)> + 'a {
        let length = term.chars().count();
        (length.saturating_sub(max)..=length + max)
            .filter_map(move |l| self.by_length.get(&l))
            .flatten()
            .map(|(word, frequency)| (word.as_str(), *frequency))
    }
}

fn quote(term: &str) -> String {
    format!("\"{}\"", term.replace('"', "\"\""))
}

fn or_expression<'a>(terms: impl Iterator<Item = &'a str>) -> String {
    let mut seen = HashSet::new();
    terms
        .filter(|t| seen.insert(t.to_string()))
        .map(quote)
        .collect::<Vec<_>>()
        .join(" OR ")
}

/// Every query word and its stem, OR-ed: a vague question rarely
/// contains all the words of the message it is looking for.
fn exact_expression(terms: &[String]) -> Option<String> {
    if terms.is_empty() {
        return None;
    }
    let mut alternatives: Vec<String> = terms.to_vec();
    for term in terms {
        for stem in text::word_tokens(&crate::stemming::stem_query(term)) {
            if stem.chars().count() >= 3 {
                alternatives.push(stem);
            }
        }
    }
    Some(or_expression(alternatives.iter().map(String::as_str)))
}

fn within(at: Option<&str>, after: Option<&str>, before: Option<&str>) -> bool {
    let Some(at) = at else { return after.is_none() && before.is_none() };
    after.is_none_or(|after| at >= after) && before.is_none_or(|before| at < before)
}

/// The part of a message around the first query word it contains.
fn snippet(body: &str, terms: &[String], max: usize) -> String {
    let folded: Vec<(usize, String)> = body
        .char_indices()
        .map(|(i, c)| (i, text::fold(&c.to_string())))
        .collect();
    let flat: String = folded.iter().map(|(_, c)| c.as_str()).collect();
    let first = terms
        .iter()
        .filter_map(|term| flat.find(term.as_str()))
        .min()
        .unwrap_or(0);
    let mut char_position = 0;
    let mut consumed = 0;
    for (index, (_, folded_char)) in folded.iter().enumerate() {
        if consumed >= first {
            char_position = index;
            break;
        }
        consumed += folded_char.len();
    }
    let start = char_position.saturating_sub(max / 4);
    let text: String = body.chars().skip(start).take(max).collect();
    let text = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let prefix = if start > 0 { "…" } else { "" };
    let suffix = if body.chars().count() > start + max { "…" } else { "" };
    format!("{}{}{}", prefix, text, suffix)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snippet_centres_on_the_first_match() {
        let body = format!("{} la comparaison Jina pour le tri {}", "x ".repeat(200), "y ".repeat(200));
        let cut = snippet(&body, &["jina".to_string()], 80);
        assert!(cut.contains("Jina"));
        assert!(cut.starts_with('…'));
    }

    #[test]
    fn coverage_counts_stems_and_misheard_words() {
        let terms = vec!["gina".to_string(), "comparaison".to_string(), "tri".to_string()];
        let forms = term_forms(&terms, &[("gina".to_string(), "jina".to_string())]);
        assert_eq!(coverage(&forms, "Les comparaisons Jina et BGE pour le tri des souvenirs"), 1.0);
        assert!((coverage(&forms, "Le tri des factures") - 1.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn exact_expression_ors_words_and_stems() {
        let expression = exact_expression(&["factures".to_string(), "revolut".to_string()]).unwrap();
        assert!(expression.contains("\"factures\""));
        assert!(expression.contains(" OR "));
    }
}
