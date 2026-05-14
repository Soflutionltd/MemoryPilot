use std::cmp::Ordering;
use std::collections::HashSet;
use std::sync::{Mutex, OnceLock};

use crate::db::SearchResult;

pub fn rerank_local(query: &str, results: &mut Vec<SearchResult>) {
    if results.len() <= 1 {
        return;
    }

    let query_lower = query.to_ascii_lowercase();
    let query_tokens = significant_tokens(&query_lower);
    let query_entities = entity_values(query, None);
    let intent = LocalIntent::from_query(&query_lower);

    for result in results.iter_mut() {
        let content_lower = result.memory.content.to_ascii_lowercase();
        let mut factor: f64 = 1.0;

        if !query_tokens.is_empty() {
            let token_hits = query_tokens
                .iter()
                .filter(|token| content_lower.contains(token.as_str()))
                .count();
            let coverage = token_hits as f64 / query_tokens.len() as f64;
            if coverage >= 0.5 {
                factor *= 1.0 + (coverage * 0.10);
            }
        }

        if query_lower.len() >= 12 && content_lower.contains(&query_lower) {
            factor *= 1.12;
        }

        if !query_entities.is_empty() {
            let content_entities =
                entity_values(&result.memory.content, result.memory.project.as_deref());
            let overlap = query_entities.intersection(&content_entities).count();
            if overlap > 0 {
                factor *= 1.0 + (overlap as f64 * 0.05).min(0.15);
            }
        }

        if intent.preference && preference_signal(&result.memory.kind, &content_lower) {
            factor *= 1.08;
        }
        if intent.temporal && temporal_signal(&result.memory.id, &content_lower) {
            factor *= 1.06;
        }
        if intent.user_turn && content_lower.trim_start().starts_with("user:") {
            factor *= 1.07;
        }
        if intent.assistant_turn && content_lower.trim_start().starts_with("assistant:") {
            factor *= 1.07;
        }
        if intent.update && update_signal(&content_lower) {
            factor *= 1.07;
        }
        result.score = (result.score * factor.min(1.35) * 10000.0).round() / 10000.0;
    }

    results.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.memory.id.cmp(&right.memory.id))
    });

    for index in 1..results.len() {
        let previous = results[index - 1].score;
        if results[index].score > previous {
            results[index].score = (previous * 0.999 * 10000.0).round() / 10000.0;
        }
    }
}

pub fn rerank_cross_encoder_if_enabled(query: &str, results: &mut Vec<SearchResult>) {
    if results.len() <= 1 || !should_run_cross_encoder(query) {
        return;
    }

    let top_k = std::env::var("MEMORYPILOT_CROSS_RERANK_TOP_K")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(12)
        .clamp(2, 64)
        .min(results.len());

    let documents = results
        .iter()
        .take(top_k)
        .map(|result| truncate_document(&result.memory.content))
        .collect::<Vec<_>>();

    let reranked = {
        let mut state = cross_reranker().lock().ok();
        match state.as_deref_mut() {
            Some(CrossRerankerState::Ready(model)) => model
                .rerank(query.to_string(), &documents, false, Some(16))
                .ok(),
            Some(CrossRerankerState::Unavailable(error)) => {
                eprintln!("[reranker] cross-encoder unavailable: {}", error);
                None
            }
            None => None,
        }
    };

    let Some(reranked) = reranked else {
        return;
    };
    if reranked.len() != top_k {
        return;
    }

    let min_cross = reranked
        .iter()
        .map(|result| result.score)
        .fold(f32::INFINITY, f32::min);
    let max_cross = reranked
        .iter()
        .map(|result| result.score)
        .fold(f32::NEG_INFINITY, f32::max);
    let cross_span = (max_cross - min_cross).max(0.0001);

    let original_scores = results
        .iter()
        .take(top_k)
        .map(|result| result.score)
        .collect::<Vec<_>>();
    let min_original = original_scores
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let max_original = original_scores
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let original_span = (max_original - min_original).max(0.0001);

    let mut fused = results.drain(..top_k).collect::<Vec<_>>();
    for cross in reranked {
        if let Some(result) = fused.get_mut(cross.index) {
            let original_norm = (result.score - min_original) / original_span;
            let cross_norm = (cross.score - min_cross) as f64 / cross_span as f64;
            let fused_score = (original_norm * 0.70) + (cross_norm * 0.30);
            result.score = (fused_score * 10000.0).round() / 10000.0;
        }
    }

    fused.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.memory.id.cmp(&right.memory.id))
    });
    fused.extend(results.drain(..));
    *results = fused;
}

enum CrossRerankerState {
    Ready(fastembed::TextRerank),
    Unavailable(String),
}

fn should_run_cross_encoder(query: &str) -> bool {
    // Default: `adaptive`. Empirically the right operating point —
    // `1`/`true` blanket-on inflates p50 latency from ~50ms to ~2s on
    // LongMemEval (43x) for a +0.5pp R@5 gain, while `adaptive` keeps
    // latency near baseline on the easy English long tail and still
    // captures the high-value gains on hard queries (preference,
    // temporal, "did I mention", multilingual). On the FR bench the
    // adaptive lane catches every French query (R@5 +6.6pp, R@10
    // +6.6pp, MRR +5.9pp at ~480ms/query).
    match std::env::var("MEMORYPILOT_CROSS_RERANK") {
        Ok(value)
            if matches!(
                value.as_str(),
                "1" | "true" | "TRUE" | "always" | "fastembed" | "onnx"
            ) =>
        {
            true
        }
        Ok(value) if matches!(value.as_str(), "0" | "false" | "FALSE" | "off") => false,
        // Default + explicit "adaptive": run only when the query
        // looks hard or non-English (where the multilingual cross
        // encoder shines).
        _ => is_hard_query(query),
    }
}

fn is_hard_query(query: &str) -> bool {
    let query_lower = query.to_ascii_lowercase();
    let intent = LocalIntent::from_query(&query_lower);
    if intent.temporal || intent.preference || intent.update {
        return true;
    }
    if contains_any(
        &query_lower,
        &[
            "what did i",
            "who did i",
            "where did i",
            "how long",
            "do you think",
            "should i",
            "did i mention",
            "i mentioned",
            "what was the",
            "who was",
        ],
    ) {
        return true;
    }
    looks_non_english(query)
}

/// Cheap heuristic to detect non-English queries. Triggers on French
/// accented characters, common French/Spanish/German function words,
/// and CJK ranges. Designed to be conservative: false positives just
/// add latency, but missing a non-English query loses the +6pp R@5
/// the cross encoder buys us on the FR bench.
fn looks_non_english(query: &str) -> bool {
    if query
        .chars()
        .any(|c| matches!(c, 'é' | 'è' | 'ê' | 'à' | 'â' | 'ç' | 'ô' | 'û' | 'ù' | 'î' | 'ï' | 'ü' | 'ö' | 'ä' | 'ñ' | 'É' | 'È' | 'Ê' | 'À'))
    {
        return true;
    }
    if query.chars().any(|c| {
        let cp = c as u32;
        // CJK Unified Ideographs, Hiragana, Katakana, Hangul.
        (0x3040..=0x30FF).contains(&cp)
            || (0x4E00..=0x9FFF).contains(&cp)
            || (0xAC00..=0xD7AF).contains(&cp)
    }) {
        return true;
    }
    let lower = query.to_ascii_lowercase();
    let padded = format!(" {} ", lower);
    // Strong French markers: articles, prepositions, common verbs and
    // question words that essentially never appear in English. The
    // false-positive cost is just extra latency on a misclassified
    // query; the false-negative cost is a -6pp R@5 hit on the FR
    // bench, so we err on the side of recall.
    let french_markers: &[&str] = &[
        " un ", " une ", " des ", " les ", " du ", " au ", " aux ", " et ", " ou ",
        " avec ", " sans ", " pour ", " dans ", " entre ", " chez ", " sur ", " sous ",
        " vers ", " contre ", " selon ", " parmi ", " depuis ", " pendant ", " malgré ",
        " comment ", " pourquoi ", " quand ", " quel ", " quelle ", " quels ", " quelles ",
        " que ", " qui ", " quoi ", " où ", " ce ", " cette ", " ces ", " cet ",
        " mais ", " donc ", " car ", " parce ", " ainsi ",
        " est ", " sont ", " être ", " avoir ", " faire ", " aller ",
        " je ", " tu ", " nous ", " vous ", " ils ", " elles ", " on ",
        " mon ", " ton ", " son ", " ma ", " ta ", " sa ", " mes ", " tes ", " ses ",
        " notre ", " votre ", " leur ", " leurs ",
        " plus ", " moins ", " très ", " bien ", " trop ", " aussi ",
        "qu'", "n'", "d'", "l'", "j'", "s'", "c'", "m'", "t'",
    ];
    if french_markers.iter().any(|m| padded.contains(m)) {
        return true;
    }
    let other_romance_markers: &[&str] = &[
        " hola ", " gracias ", " porque ", " usted ", " también ", " pero ", " muy ",
        " danke ", " bitte ", " nicht ", " sehr ", " auch ", " ist ", " mit ", " der ",
        " die ", " das ", " und ", " oder ", " aber ",
    ];
    other_romance_markers.iter().any(|m| padded.contains(m))
}

fn cross_reranker() -> &'static Mutex<CrossRerankerState> {
    static RERANKER: OnceLock<Mutex<CrossRerankerState>> = OnceLock::new();
    RERANKER.get_or_init(|| Mutex::new(init_cross_reranker()))
}

fn init_cross_reranker() -> CrossRerankerState {
    let model = cross_reranker_model();
    let cache_dir = dirs::home_dir()
        .unwrap_or_else(std::env::temp_dir)
        .join(".fastembed_cache")
        .join("rerank");
    let _ = std::fs::create_dir_all(&cache_dir);
    let options = fastembed::RerankInitOptions::new(model)
        .with_show_download_progress(false)
        .with_cache_dir(cache_dir);

    match fastembed::TextRerank::try_new(options) {
        Ok(model) => CrossRerankerState::Ready(model),
        Err(error) => CrossRerankerState::Unavailable(error.to_string()),
    }
}

fn cross_reranker_model() -> fastembed::RerankerModel {
    match std::env::var("MEMORYPILOT_RERANKER_MODEL")
        .unwrap_or_else(|_| "jina-v2-multilingual".to_string())
        .to_ascii_lowercase()
        .as_str()
    {
        "bge-base" | "bge-reranker-base" | "baai/bge-reranker-base" => {
            fastembed::RerankerModel::BGERerankerBase
        }
        "bge-v2-m3" | "rozgo/bge-reranker-v2-m3" => fastembed::RerankerModel::BGERerankerV2M3,
        "jina-v1" | "jinaai/jina-reranker-v1-turbo-en" => {
            fastembed::RerankerModel::JINARerankerV1TurboEn
        }
        _ => fastembed::RerankerModel::JINARerankerV2BaseMultiligual,
    }
}

fn truncate_document(content: &str) -> String {
    const MAX_CHARS: usize = 1400;
    if content.len() <= MAX_CHARS {
        return content.to_string();
    }
    let boundary = content
        .char_indices()
        .take_while(|(index, _)| *index <= MAX_CHARS)
        .map(|(index, _)| index)
        .last()
        .unwrap_or(MAX_CHARS);
    content[..boundary].to_string()
}

#[derive(Debug, Default)]
struct LocalIntent {
    preference: bool,
    temporal: bool,
    user_turn: bool,
    assistant_turn: bool,
    update: bool,
}

impl LocalIntent {
    fn from_query(query_lower: &str) -> Self {
        Self {
            preference: contains_any(
                query_lower,
                &[
                    "prefer",
                    "preference",
                    "favorite",
                    "favourite",
                    "like",
                    "préf",
                    "aime",
                    "favori",
                ],
            ),
            temporal: contains_any(
                query_lower,
                &[
                    "when", "before", "after", "last", "recent", "ago", "week", "month", "avant",
                    "après", "dernier", "semaine", "mois",
                ],
            ),
            user_turn: contains_any(query_lower, &["i said", "i told", "user", "j'ai dit"]),
            assistant_turn: contains_any(
                query_lower,
                &[
                    "you said",
                    "assistant",
                    "claude",
                    "cursor",
                    "chatgpt",
                    "tu as dit",
                ],
            ),
            update: contains_any(
                query_lower,
                &[
                    "changed",
                    "updated",
                    "actually",
                    "instead",
                    "switch",
                    "correction",
                    "modifié",
                    "en fait",
                    "plutôt",
                ],
            ),
        }
    }
}

fn significant_tokens(text: &str) -> Vec<String> {
    text.split(|character: char| !character.is_alphanumeric())
        .filter(|token| token.len() >= 4)
        .filter(|token| !STOPWORDS.contains(token))
        .map(ToOwned::to_owned)
        .collect()
}

fn entity_values(text: &str, project: Option<&str>) -> HashSet<String> {
    crate::graph::extract_entities(text, project)
        .into_iter()
        .filter(|entity| crate::graph::is_reliable_link_entity(entity))
        .map(|entity| entity.value.to_ascii_lowercase())
        .collect()
}

fn preference_signal(kind: &str, content_lower: &str) -> bool {
    kind == "preference"
        || contains_any(
            content_lower,
            &[
                "prefer",
                "favorite",
                "favourite",
                "like",
                "would rather",
                "i usually",
                "i often",
                "i enjoy",
                "i love",
                "i tend",
                "i think",
                "i recently",
                "i've been",
                "my ",
                "préf",
                "aime",
                "favori",
            ],
        )
}

fn temporal_signal(memory_id: &str, content_lower: &str) -> bool {
    memory_id.contains("__t")
        || contains_any(
            content_lower,
            &[
                "today",
                "yesterday",
                "last ",
                "ago",
                "week",
                "month",
                "maintenant",
                "hier",
                "semaine",
            ],
        )
}

fn update_signal(content_lower: &str) -> bool {
    contains_any(
        content_lower,
        &[
            "changed",
            "updated",
            "actually",
            "instead",
            "switched",
            "correction",
            "modifié",
            "désormais",
            "en fait",
        ],
    )
}

fn contains_any(text: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| text.contains(needle))
}

const STOPWORDS: &[&str] = &[
    "what", "when", "where", "which", "with", "that", "this", "then", "than", "have", "about",
    "your", "pour", "dans", "avec", "quoi", "quel", "quelle", "est-ce",
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::Memory;

    #[test]
    fn local_rerank_boosts_exact_relevance() {
        let mut results = vec![
            result("1", "user: unrelated cooking note", 1.0),
            result("2", "user: I prefer dark mode dashboards", 0.96),
        ];
        rerank_local("prefer dark mode", &mut results);
        assert_eq!(results[0].memory.id, "2");
    }

    #[test]
    fn french_query_marked_hard() {
        assert!(looks_non_english("comment pondérer les colonnes BM25"));
        assert!(looks_non_english("où trouver le fichier"));
        assert!(looks_non_english("démarrage non bloquant avec warm-up"));
        // No accents, but unmistakably French via articles/prepositions.
        assert!(looks_non_english(
            "ajouter un secret sur Cloudflare Pages en ligne de commande"
        ));
        assert!(looks_non_english(
            "convention de nommage des outils MCP"
        ));
        assert!(looks_non_english("strategie de fusion des doublons"));
    }

    #[test]
    fn english_query_not_marked_hard_by_language() {
        assert!(!looks_non_english("what is the configuration value"));
        assert!(!looks_non_english("how to configure BM25 weights"));
    }

    #[test]
    fn cjk_query_marked_hard() {
        assert!(looks_non_english("中文 query"));
        assert!(looks_non_english("こんにちは world"));
    }

    #[test]
    fn romance_marker_detection() {
        assert!(looks_non_english("hola que tal"));
        assert!(looks_non_english("danke schön"));
    }

    fn result(id: &str, content: &str, score: f64) -> SearchResult {
        SearchResult {
            memory: Memory {
                id: id.to_string(),
                content: content.to_string(),
                kind: "note".to_string(),
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
}
