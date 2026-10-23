//! Maximal Marginal Relevance over the final candidate pool.
//!
//! The ranker scores every candidate on its own, so five restatements of
//! one fact — or five turns of one conversation — can fill the whole
//! top-k while a second, equally relevant source sits at rank 6. For a
//! question that spans several memories ("how many projects did I run
//! this year?") that is a miss the reader cannot recover from.
//!
//! MMR re-orders the pool greedily: each step picks the candidate that
//! maximises `λ·relevance − (1−λ)·max_similarity_to_already_picked`.
//! Similarity is the cosine between stored vectors, re-scaled so that
//! anything under the redundancy floor (default 0.55) counts as zero: two memories on
//! different topics must not pay a penalty for merely sharing a domain.
//! Relevance is min-max normalised inside the pool so λ means the same
//! thing whatever the ranker's dynamic range is.
//!
//! Scores are preserved (the first pick keeps its score, later picks are
//! clamped to stay monotonically non-increasing) so downstream consumers
//! keep seeing a sorted list.

use std::cmp::Ordering;
use std::collections::HashMap;

use crate::db::SearchResult;

/// Cosine below which two candidates are considered unrelated.
/// `MEMORYPILOT_MMR_FLOOR` overrides it for calibration runs.
fn redundancy_floor() -> f32 {
    std::env::var("MEMORYPILOT_MMR_FLOOR")
        .ok()
        .and_then(|raw| raw.trim().parse::<f32>().ok())
        .unwrap_or(0.55)
        .clamp(0.0, 0.99)
}

/// `MEMORYPILOT_MMR_POOL` — pool multiplier over `limit` handed to the
/// rerankers when MMR is on (default 2).
pub fn pool_multiplier() -> usize {
    std::env::var("MEMORYPILOT_MMR_POOL")
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .unwrap_or(2)
        .clamp(1, 4)
}

/// `MEMORYPILOT_MMR_LAMBDA` — 1.0 disables the diversity term, 0.0 is
/// pure diversity. Default 0.4, chosen on LongMemEval-S@100 and
/// memorypilot-fr-v2: the top hit is never moved, so MRR is untouched,
/// and 0.4 is where multi-session gold coverage peaks (0.6 only lifts
/// one extra session, 0.3 starts trading distinct sessions for
/// redundant turns).
pub fn lambda() -> f64 {
    std::env::var("MEMORYPILOT_MMR_LAMBDA")
        .ok()
        .and_then(|raw| raw.trim().parse::<f64>().ok())
        .unwrap_or(0.4)
        .clamp(0.0, 1.0)
}

pub fn is_enabled() -> bool {
    !matches!(
        std::env::var("MEMORYPILOT_MMR").as_deref(),
        Ok("0") | Ok("false") | Ok("off")
    )
}

/// Re-order `results` in place. `vectors` maps memory id → stored
/// embedding; candidates without a vector never incur a penalty (and
/// never cause one). No-op for fewer than three candidates.
pub fn rerank_mmr(results: &mut Vec<SearchResult>, vectors: &HashMap<String, Vec<f32>>) {
    if results.len() < 3 || !is_enabled() {
        return;
    }
    let lambda = lambda();
    if lambda >= 1.0 {
        return;
    }
    let floor = redundancy_floor();

    let (min_score, max_score) = results.iter().fold(
        (f64::INFINITY, f64::NEG_INFINITY),
        |(low, high), result| (low.min(result.score), high.max(result.score)),
    );
    let span = (max_score - min_score).max(1e-9);
    let relevance: Vec<f64> = results
        .iter()
        .map(|result| (result.score - min_score) / span)
        .collect();
    let embeddings: Vec<Option<&Vec<f32>>> = results
        .iter()
        .map(|result| vectors.get(&result.memory.id))
        .collect();

    let mut remaining: Vec<usize> = (1..results.len()).collect();
    let mut order: Vec<usize> = vec![0];
    // Highest similarity to any already-picked candidate, per remaining one.
    let mut max_similarity: Vec<f32> = vec![0.0; results.len()];
    update_similarities(0, &embeddings, &mut max_similarity);

    while !remaining.is_empty() {
        let mut best_position = 0;
        let mut best_value = f64::NEG_INFINITY;
        for (position, &candidate) in remaining.iter().enumerate() {
            let redundancy = redundancy(max_similarity[candidate], floor);
            let value = lambda * relevance[candidate] - (1.0 - lambda) * redundancy;
            let better = match value.partial_cmp(&best_value) {
                Some(Ordering::Greater) => true,
                // Deterministic tie-break on the id, as the ranker does.
                Some(Ordering::Equal) => {
                    results[candidate].memory.id < results[remaining[best_position]].memory.id
                }
                _ => false,
            };
            if better {
                best_value = value;
                best_position = position;
            }
        }
        let picked = remaining.remove(best_position);
        order.push(picked);
        update_similarities(picked, &embeddings, &mut max_similarity);
    }

    let reordered: Vec<SearchResult> = {
        let mut slots: Vec<Option<SearchResult>> = results.drain(..).map(Some).collect();
        order
            .iter()
            .map(|&index| slots[index].take().expect("each index is used once"))
            .collect()
    };
    *results = reordered;

    for index in 1..results.len() {
        let previous = results[index - 1].score;
        if results[index].score > previous {
            results[index].score = (previous * 0.999 * 10000.0).round() / 10000.0;
        }
    }
}

fn redundancy(similarity: f32, floor: f32) -> f64 {
    if similarity <= floor {
        0.0
    } else {
        ((similarity - floor) / (1.0 - floor)) as f64
    }
}

fn update_similarities(
    picked: usize,
    embeddings: &[Option<&Vec<f32>>],
    max_similarity: &mut [f32],
) {
    let Some(picked_vector) = embeddings[picked] else {
        return;
    };
    for (index, embedding) in embeddings.iter().enumerate() {
        if index == picked {
            continue;
        }
        if let Some(vector) = embedding {
            let similarity = crate::embedding::cosine_similarity(picked_vector, vector);
            if similarity > max_similarity[index] {
                max_similarity[index] = similarity;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::Memory;

    fn result(id: &str, score: f64) -> SearchResult {
        SearchResult {
            memory: Memory {
                id: id.to_string(),
                content: String::new(),
                kind: "fact".to_string(),
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
            sources: Vec::new(),
        }
    }

    fn unit(values: &[f32]) -> Vec<f32> {
        let norm = values.iter().map(|v| v * v).sum::<f32>().sqrt();
        values.iter().map(|v| v / norm).collect()
    }

    #[test]
    fn near_duplicates_are_pushed_down() {
        std::env::remove_var("MEMORYPILOT_MMR");
        std::env::remove_var("MEMORYPILOT_MMR_LAMBDA");
        let mut results = vec![
            result("a", 1.0),
            result("a2", 0.98),
            result("a3", 0.97),
            result("b", 0.9),
            result("c", 0.85),
        ];
        let mut vectors = HashMap::new();
        vectors.insert("a".into(), unit(&[1.0, 0.0, 0.0]));
        vectors.insert("a2".into(), unit(&[0.99, 0.1, 0.0]));
        vectors.insert("a3".into(), unit(&[0.98, 0.15, 0.0]));
        vectors.insert("b".into(), unit(&[0.0, 1.0, 0.0]));
        vectors.insert("c".into(), unit(&[0.0, 0.0, 1.0]));
        rerank_mmr(&mut results, &vectors);
        let ids: Vec<&str> = results.iter().map(|r| r.memory.id.as_str()).collect();
        assert_eq!(ids[0], "a");
        assert_eq!(ids[1], "b", "the first distinct source moves up: {:?}", ids);
        for pair in results.windows(2) {
            assert!(pair[0].score >= pair[1].score);
        }
    }

    #[test]
    fn unrelated_candidates_keep_their_order() {
        std::env::remove_var("MEMORYPILOT_MMR");
        std::env::remove_var("MEMORYPILOT_MMR_LAMBDA");
        let mut results = vec![result("a", 1.0), result("b", 0.9), result("c", 0.8)];
        let mut vectors = HashMap::new();
        vectors.insert("a".into(), unit(&[1.0, 0.0, 0.0]));
        vectors.insert("b".into(), unit(&[0.0, 1.0, 0.0]));
        vectors.insert("c".into(), unit(&[0.0, 0.0, 1.0]));
        rerank_mmr(&mut results, &vectors);
        let ids: Vec<&str> = results.iter().map(|r| r.memory.id.as_str()).collect();
        assert_eq!(ids, vec!["a", "b", "c"]);
    }

    #[test]
    fn missing_vectors_are_neutral() {
        std::env::remove_var("MEMORYPILOT_MMR");
        let mut results = vec![result("a", 1.0), result("b", 0.9), result("c", 0.8)];
        rerank_mmr(&mut results, &HashMap::new());
        let ids: Vec<&str> = results.iter().map(|r| r.memory.id.as_str()).collect();
        assert_eq!(ids, vec!["a", "b", "c"]);
    }
}
