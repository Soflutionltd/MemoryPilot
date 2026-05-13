use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use crate::db::SearchResult;

pub fn should_expand_candidates(query: &str) -> bool {
    let lower = query.to_ascii_lowercase();
    contains_any(
        &lower,
        &[
            "when",
            "before",
            "after",
            "last",
            "ago",
            "week",
            "month",
            "prefer",
            "favorite",
            "favourite",
            "like",
            "suggest",
            "recommend",
            "tips",
            "should",
            "what did",
            "who did",
            "what was",
        ],
    )
}

pub fn fuse_sessions(
    query: &str,
    candidates: Vec<SearchResult>,
    limit: usize,
) -> Vec<SearchResult> {
    if candidates.len() <= limit || !is_session_like(&candidates) {
        return candidates.into_iter().take(limit).collect();
    }

    let query_terms = significant_terms(query);
    let mut groups: HashMap<String, Vec<(usize, SearchResult)>> = HashMap::new();
    let mut passthrough = Vec::new();

    for (rank, candidate) in candidates.into_iter().enumerate() {
        match session_key(&candidate.memory.id) {
            Some(key) => groups.entry(key).or_default().push((rank, candidate)),
            None => passthrough.push((rank, candidate)),
        }
    }

    if groups.len() < limit.min(5) {
        let mut flattened: Vec<(usize, SearchResult)> =
            groups.into_values().flatten().chain(passthrough).collect();
        flattened.sort_by_key(|(rank, _)| *rank);
        return flattened
            .into_iter()
            .map(|(_, result)| result)
            .take(limit)
            .collect();
    }

    let mut sessions: Vec<SessionBucket> = groups
        .into_iter()
        .filter_map(|(key, mut entries)| {
            entries.sort_by(|left, right| {
                right
                    .1
                    .score
                    .partial_cmp(&left.1.score)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| left.0.cmp(&right.0))
            });
            let best_rank = entries
                .iter()
                .map(|(rank, _)| *rank)
                .min()
                .unwrap_or(usize::MAX);
            let best_score = entries
                .first()
                .map(|(_, result)| result.score)
                .unwrap_or(0.0);
            let top3_avg = entries
                .iter()
                .take(3)
                .map(|(_, result)| result.score)
                .sum::<f64>()
                / entries.len().min(3) as f64;
            let lexical_hits = entries
                .iter()
                .take(3)
                .map(|(_, result)| lexical_overlap(&query_terms, &result.memory.content))
                .max()
                .unwrap_or(0);
            let density_bonus = (entries.len().min(4) as f64 - 1.0).max(0.0) * 0.012;
            let lexical_bonus = lexical_hits as f64 * 0.008;
            let session_score =
                (best_score * 0.82) + (top3_avg * 0.18) + density_bonus + lexical_bonus;

            Some(SessionBucket {
                key,
                best_rank,
                score: session_score,
                entries,
            })
        })
        .collect();

    sessions.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.best_rank.cmp(&right.best_rank))
            .then_with(|| left.key.cmp(&right.key))
    });

    let mut fused = Vec::with_capacity(limit);
    let mut used_ids = HashSet::new();

    for session in &mut sessions {
        if fused.len() >= limit {
            break;
        }
        if let Some((_, result)) = session.entries.first() {
            if used_ids.insert(result.memory.id.clone()) {
                fused.push(result.clone());
            }
        }
    }

    let mut leftovers: Vec<(usize, SearchResult)> = sessions
        .into_iter()
        .flat_map(|session| session.entries.into_iter().skip(1))
        .chain(passthrough)
        .collect();
    leftovers.sort_by(|left, right| {
        right
            .1
            .score
            .partial_cmp(&left.1.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.0.cmp(&right.0))
    });

    for (_, result) in leftovers {
        if fused.len() >= limit {
            break;
        }
        if used_ids.insert(result.memory.id.clone()) {
            fused.push(result);
        }
    }

    fused.truncate(limit);
    fused
}

#[derive(Debug)]
struct SessionBucket {
    key: String,
    best_rank: usize,
    score: f64,
    entries: Vec<(usize, SearchResult)>,
}

fn is_session_like(candidates: &[SearchResult]) -> bool {
    candidates
        .iter()
        .filter(|candidate| session_key(&candidate.memory.id).is_some())
        .count()
        >= 8
}

fn session_key(id: &str) -> Option<String> {
    id.split_once("__t")
        .map(|(session, _)| session.to_string())
        .filter(|session| !session.is_empty())
}

fn lexical_overlap(query_terms: &[String], content: &str) -> usize {
    if query_terms.is_empty() {
        return 0;
    }
    let lower = content.to_ascii_lowercase();
    query_terms
        .iter()
        .filter(|term| lower.contains(term.as_str()))
        .count()
}

fn significant_terms(query: &str) -> Vec<String> {
    let mut terms: Vec<String> = query
        .split(|character: char| !character.is_alphanumeric())
        .filter(|term| term.len() >= 4)
        .filter(|term| !STOPWORDS.contains(&term.to_ascii_lowercase().as_str()))
        .map(|term| term.to_ascii_lowercase())
        .collect();

    let lower = query.to_ascii_lowercase();
    for (word, short) in WEEKDAY_ALIASES {
        if lower.contains(word) {
            terms.push((*short).to_string());
        }
    }
    for (word, digit) in NUMBER_ALIASES {
        if lower.contains(word) {
            terms.push((*digit).to_string());
        }
    }
    terms.sort();
    terms.dedup();
    terms
}

fn contains_any(text: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| text.contains(needle))
}

const STOPWORDS: &[&str] = &[
    "what", "when", "where", "which", "with", "that", "this", "then", "than", "have", "about",
    "your", "from", "during", "there", "some",
];

const WEEKDAY_ALIASES: &[(&str, &str)] = &[
    ("monday", "mon"),
    ("tuesday", "tue"),
    ("wednesday", "wed"),
    ("thursday", "thu"),
    ("friday", "fri"),
    ("saturday", "sat"),
    ("sunday", "sun"),
];

const NUMBER_ALIASES: &[(&str, &str)] = &[
    ("one", "1"),
    ("two", "2"),
    ("three", "3"),
    ("four", "4"),
    ("five", "5"),
    ("six", "6"),
    ("seven", "7"),
    ("eight", "8"),
    ("nine", "9"),
    ("ten", "10"),
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::Memory;

    #[test]
    fn diversifies_repeated_sessions() {
        let candidates = vec![
            result("a__t0", 1.0),
            result("a__t1", 0.99),
            result("a__t2", 0.98),
            result("b__t0", 0.97),
            result("c__t0", 0.96),
            result("d__t0", 0.95),
            result("e__t0", 0.94),
            result("f__t0", 0.93),
            result("g__t0", 0.92),
        ];
        let fused = fuse_sessions("what did I do last week", candidates, 5);
        let sessions: HashSet<String> = fused
            .iter()
            .filter_map(|result| session_key(&result.memory.id))
            .collect();
        assert_eq!(fused.len(), 5);
        assert!(sessions.len() >= 5);
    }

    #[test]
    fn expands_temporal_aliases() {
        let terms = significant_terms("what happened four weeks ago last Tuesday");
        assert!(terms.contains(&"4".to_string()));
        assert!(terms.contains(&"tue".to_string()));
    }

    fn result(id: &str, score: f64) -> SearchResult {
        SearchResult {
            memory: Memory {
                id: id.to_string(),
                content: "user: sample memory".to_string(),
                kind: "transcript_chunk".to_string(),
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
            },
            score,
        }
    }
}
