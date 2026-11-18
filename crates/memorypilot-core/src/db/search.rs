//! Hybrid search: BM25 + vectors, RRF fusion, cross-encoder rerank.

use super::*;

impl Database {
    pub fn search(
        &self,
        query: &str,
        limit: usize,
        project: Option<&str>,
        kind: Option<&str>,
        tags: Option<&[String]>,
        watcher_keywords: Option<&[String]>,
    ) -> Result<Vec<SearchResult>, String> {
        self.search_at(Utc::now(), query, limit, project, kind, tags, watcher_keywords)
            .map(|(results, _)| results)
    }

    /// [`Self::search`] plus the confidence verdict, evaluated as if the
    /// current instant were `now` — temporal phrases in the query
    /// ("last week") and the recency prior are resolved against it. The
    /// benchmarks pass the question's own date; callers otherwise use
    /// `Utc::now()`.
    #[allow(clippy::too_many_arguments)]
    pub fn search_at(
        &self,
        now: chrono::DateTime<Utc>,
        query: &str,
        limit: usize,
        project: Option<&str>,
        kind: Option<&str>,
        tags: Option<&[String]>,
        watcher_keywords: Option<&[String]>,
    ) -> Result<(Vec<SearchResult>, SearchConfidence), String> {
        hybrid_search(self, now, query, limit, project, kind, tags, watcher_keywords)
    }
}

/// Tag of a memory that holds what the assistant said in a conversation.
pub const ROLE_ASSISTANT_TAG: &str = "role:assistant";

/// Appended to a text that was cut short.
pub const CUT_MARK: &str = " [...]";

/// How a compact search shortens its results. What the assistant said is
/// most of the retrieved text and seldom holds the answer below the first
/// ranks: on LongMemEval-S, 70 turns cut this way answered 103 of 106
/// sampled questions with 8 440 prompt tokens each, against 100 of 106 and
/// 15 650 tokens for 50 turns in full (Claude Opus 5.5, gpt-4o judge).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CompactOptions {
    /// Assistant memories ranked within the first `full_top` stay whole.
    pub full_top: usize,
    /// Characters kept of the others; zero leaves them out.
    pub chars: usize,
}

impl Default for CompactOptions {
    fn default() -> Self {
        Self { full_top: 3, chars: 150 }
    }
}

/// Ranks within which assistant memories stay whole when the query asks
/// what the assistant said. On LongMemEval-S (first 50 turns, wide rerank)
/// such a reply ranked 4 to 10 for 5 of the 14 questions left incomplete.
const ASSISTANT_QUERY_FULL_TOP: usize = 10;

/// Phrases of a query about what the assistant said, in English and French,
/// lowercased with a straight apostrophe.
const ASSISTANT_QUERY_MARKERS: &[&str] = &[
    "you said", "you told", "you suggested", "you recommended", "you mentioned",
    "you gave", "you provided", "you listed", "you shared", "you explained",
    "you described", "you wrote", "you came up", "you helped", "you offered",
    "you proposed", "you made", "did you", "your suggestion", "your recommendation",
    "your answer", "your response", "your advice", "your list", "remind me what",
    "remind me of", "remind me which", "remind me the", "we discussed", "we talked",
    "we chatted", "we spoke", "our conversation", "our chat", "our discussion",
    "our previous", "previous conversation", "last conversation", "earlier conversation",
    "previous chat", "last chat", "earlier chat",
    "tu m'as", "m'as-tu", "m'avez-vous", "tu as dit", "tu as mentionné", "tu as suggéré", "tu as conseillé",
    "tu as recommandé", "tu as proposé", "tu as donné", "tu as écrit", "tu as expliqué",
    "vous m'avez", "vous avez dit", "vous avez mentionné", "vous avez suggéré",
    "vous avez conseillé", "vous avez recommandé", "vous avez proposé", "ta réponse",
    "ta suggestion", "ta recommandation", "ton conseil", "tes conseils", "votre réponse",
    "votre suggestion", "votre recommandation", "vos conseils", "rappelle-moi ce que",
    "rappelez-moi ce que", "notre conversation", "notre discussion", "notre échange",
    "on a parlé", "on a discuté", "nous avons parlé", "nous avons discuté",
    "conversation précédente", "dernière conversation",
];

/// Whether `query` asks what the assistant said ("what did you suggest",
/// "rappelle-moi ce que tu m'as conseillé").
pub fn asks_about_assistant(query: &str) -> bool {
    let lower = query.to_lowercase().replace(['\u{2019}', '\u{02BC}'], "'");
    ASSISTANT_QUERY_MARKERS.iter().any(|marker| contains_phrase(&lower, marker))
}

/// Whether `phrase` occurs in `text` as whole words: "did you" is in "what
/// did you say" but not in "did your team ship".
fn contains_phrase(text: &str, phrase: &str) -> bool {
    let is_word = |c: char| c.is_alphanumeric();
    text.match_indices(phrase).any(|(start, _)| {
        let before = text[..start].chars().next_back();
        let after = text[start + phrase.len()..].chars().next();
        !before.is_some_and(is_word) && !after.is_some_and(is_word)
    })
}

impl CompactOptions {
    /// These options for `query`: when it asks what the assistant said, the
    /// reply that holds the answer is kept whole down to the tenth rank.
    pub fn for_query(self, query: &str) -> Self {
        if asks_about_assistant(query) {
            Self { full_top: self.full_top.max(ASSISTANT_QUERY_FULL_TOP), ..self }
        } else {
            self
        }
    }
}

/// For each result, in rank order, whether a compact context shortens it:
/// an assistant memory ranked after the first `full_top` and longer than
/// `chars` characters.
pub fn compact_cuts(results: &[SearchResult], options: CompactOptions) -> Vec<bool> {
    results
        .iter()
        .enumerate()
        .map(|(rank, result)| {
            rank >= options.full_top
                && result.memory.tags.iter().any(|tag| tag == ROLE_ASSISTANT_TAG)
                && result.memory.content.trim().chars().count() > options.chars
        })
        .collect()
}

/// The first `chars` characters of `text` and the cut mark; `None` at
/// zero, where the text is left out altogether.
pub fn cut_text(text: &str, chars: usize) -> Option<String> {
    if chars == 0 {
        return None;
    }
    let kept: String = text.trim().chars().take(chars).collect();
    Some(format!("{}{}", kept.trim_end(), CUT_MARK))
}

impl Database {
    /// [`Self::search_at`] for a caller that builds a context out of many
    /// results: the cross-encoder rescores the first 64 candidates, so the
    /// order can be trusted further down the list.
    #[allow(clippy::too_many_arguments)]
    pub fn search_wide_at(
        &self,
        now: chrono::DateTime<Utc>,
        query: &str,
        limit: usize,
        project: Option<&str>,
        kind: Option<&str>,
        tags: Option<&[String]>,
        watcher_keywords: Option<&[String]>,
    ) -> Result<(Vec<SearchResult>, SearchConfidence), String> {
        crate::reranking::with_wide_rerank(|| self.search_at(now, query, limit, project, kind, tags, watcher_keywords))
    }
}

/// The ranked finalists cut back to `limit`. A wide list is read whole, in
/// the reranker's order: diversity and session fusion push the turn that
/// holds the answer, often the assistant reply right after the matching
/// question, below the ranks a compact context keeps whole, and gain
/// nothing at 40 or 50 results (LongMemEval-S, first 50 turns: every
/// evidence turn present and whole for 97.0% of the questions, 93.8% with
/// them). A default search still spreads its head over distinct sources.
fn final_order(
    query: &str,
    mut results: Vec<SearchResult>,
    vectors: &std::collections::HashMap<String, Vec<f32>>,
    limit: usize,
    wide: bool,
) -> Vec<SearchResult> {
    if wide {
        results.truncate(limit);
        return results;
    }
    crate::diversity::rerank_mmr(&mut results, vectors);
    // The MMR reserve must not leak into session fusion: fusion only
    // sees a pool wider than `limit` when the query asked for it.
    if !crate::session_fusion::should_expand_candidates(query) {
        results.truncate(limit);
    }
    crate::session_fusion::fuse_sessions(query, results, limit)
}

/// Finalists a search keeps before the cross-encoder: `limit`, or at least
/// the wide rerank width when a caller asked for a wide rerank.
fn candidate_pool(limit: usize, wide: bool) -> usize {
    if wide {
        limit.clamp(crate::reranking::WIDE_RERANK_WIDTH, MAX_SEARCH_LIMIT)
    } else {
        limit
    }
}


/// Vectors in the approximate index from which a search scores only the
/// approximate neighbours and the keyword hits, instead of every memory.
pub const ANN_BYPASS_THRESHOLD: usize = 5_000;

/// Largest result count a search returns. Larger requests are capped here,
/// whatever the interface, so no caller can push the pool sizes out of range.
pub const MAX_SEARCH_LIMIT: usize = 100;

/// The search pipeline of [`Database::search_at`] on any store: the SQLite
/// database or the server mode's Postgres store.
#[allow(clippy::too_many_arguments)]
pub fn hybrid_search<S: MemoryStore + RecallStore + ?Sized>(
    store: &S,
    now: chrono::DateTime<Utc>,
    query: &str,
    limit: usize,
    project: Option<&str>,
    kind: Option<&str>,
    tags: Option<&[String]>,
    watcher_keywords: Option<&[String]>,
) -> Result<(Vec<SearchResult>, SearchConfidence), String> {
    if limit == 0 {
        return Ok((Vec::new(), SearchConfidence::none()));
    }
    let limit = limit.min(MAX_SEARCH_LIMIT);
    let canonical_project = Database::canonical_project(project);

    let telemetry_enabled = crate::telemetry::is_enabled();
    let search_start = std::time::Instant::now();
    let mut trace = if telemetry_enabled {
        let mut t = crate::telemetry::RetrievalTrace::default();
        t.query_truncated = crate::telemetry::truncate_query(query);
        t.query_chars = query.chars().count();
        t.project = canonical_project.clone();
        t.kind = kind.map(|k| k.to_string());
        t.tags_count = tags.map(|t| t.len()).unwrap_or(0);
        t.limit = limit;
        Some(t)
    } else {
        None
    };

    let fts_variants = crate::fts::fts5_query_variants(query);
    if fts_variants.is_empty() {
        return Ok((Vec::new(), SearchConfidence::none()));
    }
    if let Some(t) = trace.as_mut() {
        t.fts_variants = fts_variants.len();
    }

    store.expire_memories();

    let embed_start = std::time::Instant::now();
    let query_emb = cached_embed_text(query);
    if let Some(t) = trace.as_mut() {
        t.timing_ms_embed_query = embed_start.elapsed().as_secs_f64() * 1000.0;
    }

    // Pre-compute KG expansion terms for post-retrieval scoring boost
    let kg_expansion = store.get_kg_expansion_terms(query);

    let mut bm25_results = std::collections::HashMap::new();
    let mut all_memories = std::collections::HashMap::new();
    let mut candidate_sources: std::collections::HashMap<
        String,
        std::collections::HashSet<String>,
    > = std::collections::HashMap::new();

    let bm25_start = std::time::Instant::now();
    // 1. FTS5 search. Run the normal prefix query plus exact phrase/proximity variants.
    let run_fts_variant =
        |fts_query: &str,
         source: &'static str,
         rank_offset: usize,
         limit_rows: usize,
         return_err_on_failure: bool,
         bm25_results: &mut std::collections::HashMap<String, usize>,
         all_memories: &mut std::collections::HashMap<String, Memory>,
         candidate_sources: &mut std::collections::HashMap<
            String,
            std::collections::HashSet<String>,
         >|
         -> Result<usize, String> {
            let rows = match store.full_text_candidates(fts_query, canonical_project.as_deref(), kind, limit_rows) {
                Ok(rows) => rows,
                Err(error) if return_err_on_failure => return Err(error),
                Err(_) => return Ok(0),
            };

            let mut produced = 0usize;
            for (rank_index, mem) in rows.into_iter().enumerate() {
                let rank = rank_index + 1 + rank_offset;
                let entry = bm25_results.entry(mem.id.clone()).or_insert(rank);
                *entry = (*entry).min(rank);
                candidate_sources
                    .entry(mem.id.clone())
                    .or_default()
                    .insert(source.to_string());
                all_memories.entry(mem.id.clone()).or_insert(mem);
                produced += 1;
            }
            Ok(produced)
        };

    for (variant_index, (fts_query, source)) in fts_variants.iter().enumerate() {
        let limit_rows = if variant_index == 0 { 150 } else { 75 };
        let return_err_on_failure = variant_index == 0;
        run_fts_variant(
            fts_query,
            *source,
            0,
            limit_rows,
            return_err_on_failure,
            &mut bm25_results,
            &mut all_memories,
            &mut candidate_sources,
        )?;
    }

    // NOTE: lexical synonym expansion (`fts5_synonym_variants`) is
    // intentionally NOT invoked here. Three feeding strategies were
    // measured on memorypilot-fr-30:
    //
    //   - Eager OR into BM25 pool       : R@5 73.3% → 63.3% (-10)
    //   - Penalised rank, capped pool   : R@5 73.3% → 70.0% (-3)
    //   - Candidate-only (vector judge) : R@5 73.3% → 63.3% (-10)
    //
    // All three regress precision because the curated thesaurus is
    // too generic for a small corpus and pulls in long-tail noise.
    // The right fix for cross-lingual / synonym retrieval is the
    // dense cross-encoder reranker, not BM25 widening. The
    // dictionary remains in `query_expansion.rs` for callers that
    // want it (CLI tools, HTTP debugging) and may resurface later
    // for very large corpora where the long tail dilutes naturally.

    // 2a. Optional ANN pre-filter. Marks candidates with `vector_ann` so the
    // explain output exposes which retrieval stage surfaced them. The full
    // scan below stays authoritative for ranking — ANN only annotates here.
    let ann_hits: Vec<(String, f32)> = store.approximate_neighbours(&query_emb, 200);
    for (id, _) in &ann_hits {
        candidate_sources
            .entry(id.clone())
            .or_default()
            .insert("vector_ann".to_string());
    }

    // 2b. Vector Search (read pool connection, scoped). When the ANN index is
    // populated past `ANN_BYPASS_THRESHOLD`, restrict the SQL scan to the union
    // of ANN top-K and BM25 hits — avoids loading every embedding blob just to
    // score it. Below the threshold the full scan is kept (provably no recall
    // regression).
    // Only for unfiltered searches: the ANN top-K is global, so a project,
    // kind or tag filter applied after it could drop the best matches of a
    // small project. A filtered search scans its own subset instead.
    let unfiltered = canonical_project.is_none() && kind.is_none() && tags.map(|t| t.is_empty()).unwrap_or(true);
    let ann_active_for_bypass =
        unfiltered && store.approximate_index_len() >= ANN_BYPASS_THRESHOLD && !ann_hits.is_empty();
    let restricted_ids: Option<Vec<String>> = if ann_active_for_bypass {
        let mut ids: std::collections::HashSet<String> = bm25_results.keys().cloned().collect();
        for (id, _) in &ann_hits {
            ids.insert(id.clone());
        }
        Some(ids.into_iter().collect())
    } else {
        None
    };

    if let Some(t) = trace.as_mut() {
        t.timing_ms_bm25 = bm25_start.elapsed().as_secs_f64() * 1000.0;
        t.candidates_bm25 = bm25_results.len();
    }

    let vector_start = std::time::Instant::now();
    let vector_results = {
        let rows = store.vector_candidates(
            &query_emb,
            canonical_project.as_deref(),
            kind,
            restricted_ids.as_deref(),
        )?;
        let mut vector_scores: Vec<(String, f32)> = Vec::new();
        for (mem, similarity) in rows {
            all_memories
                .entry(mem.id.clone())
                .or_insert_with(|| mem.clone());
            if let Some(score) = similarity {
                candidate_sources
                    .entry(mem.id.clone())
                    .or_default()
                    .insert("vector".to_string());
                vector_scores.push((mem.id, score));
            } else {
                candidate_sources
                    .entry(mem.id.clone())
                    .or_default()
                    .insert("vector_pending".to_string());
                vector_scores.push((mem.id, 0.0));
            }
        }

        vector_scores
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut vr = std::collections::HashMap::new();
        for (i, (id, _)) in vector_scores.iter().take(100).enumerate() {
            vr.insert(id.clone(), i + 1);
        }
        vr
    };

    if let Some(t) = trace.as_mut() {
        t.timing_ms_vector = vector_start.elapsed().as_secs_f64() * 1000.0;
        t.candidates_vector = vector_results.len();
        t.candidates_total_unique = all_memories.len();
        t.candidates_ann = candidate_sources
            .values()
            .filter(|set| set.iter().any(|src| src == "ann"))
            .count();
        t.kg_expansion_terms = kg_expansion.len();
    }
    let fusion_start = std::time::Instant::now();

    // 3. RRF Fusion
    let mut rrf_scores: Vec<(String, f64)> = Vec::new();

    // Fetch graph links for PageRank-like boost (scoped to candidates)
    let candidate_ids: Vec<&String> = all_memories.keys().collect();
    let link_boosts = store.build_link_boosts_for(&candidate_ids);

    // Batch-query knowledge triple counts (avoids N+1)
    let triple_counts = store.batch_triple_counts(&candidate_ids);

    // Temporal grounding: when the query names a period, load the
    // dates attached to the candidates once and let the window
    // factor re-weight them. `None` for non-temporal queries.
    let today = now.date_naive();
    let query_window = if crate::temporal::is_enabled() {
        crate::temporal::parse_query_window(query, today)
    } else {
        None
    };
    let candidate_dates = if query_window.is_some() {
        store.batch_memory_dates(&candidate_ids)
    } else {
        std::collections::HashMap::new()
    };

    let now_ts = now.timestamp() as f64;

    let query_tokens: Vec<String> = query
        .split_whitespace()
        .map(|w| w.to_lowercase())
        .filter(|w| w.len() >= 3)
        .collect();
    let query_intent = Database::infer_query_intent(query);

    for (id, mem) in &all_memories {
        let bm25_rank = bm25_results.get(id).copied().unwrap_or(1000);
        let vec_rank = vector_results.get(id).copied().unwrap_or(1000);
        let mut score = crate::embedding::rrf_score(bm25_rank, vec_rank);

        // Exact term coverage: boost if a high fraction of query terms appear in the memory content
        if !query_tokens.is_empty() {
            let content_lower = mem.content.to_lowercase();
            let match_frac = query_tokens
                .iter()
                .filter(|t| content_lower.contains(t.as_str()))
                .count() as f64
                / query_tokens.len() as f64;
            if match_frac >= 0.8 {
                score *= 1.0 + (match_frac - 0.8) * 0.5; // up to +10% for 100% match
            }
        }

        score *= Database::query_intent_factor(&query_intent, mem);
        score *= Database::cognitive_activation_factor(mem, now_ts);

        // Importance tiebreaker: subtle nudge, never overrides relevance
        // imp 5 → 1.06x, imp 4 → 1.03x, imp 3 → 1.0x, imp 2 → 0.98x, imp 1 → 0.96x
        let imp_factor = 1.0 + (mem.importance as f64 - 3.0) * 0.03;
        score *= imp_factor;

        // Temporal recency: gentle boost, decaying over 30 days
        if let Ok(updated) = chrono::DateTime::parse_from_rfc3339(&mem.updated_at) {
            let age_days = (now_ts - updated.timestamp() as f64) / 86400.0;
            let recency = if age_days <= 3.0 {
                1.05
            } else if age_days <= 30.0 {
                1.0 + 0.05 * ((30.0 - age_days) / 27.0)
            } else {
                1.0
            };
            score *= recency;
        }

        // Temporal window: the query asks about a period. Dates come
        // from the `date` entities extracted at write time plus the
        // memory's own day. Inside → ×1.25, clearly outside → ×0.85.
        if let Some(window) = query_window.as_ref() {
            let mut dates: Vec<chrono::NaiveDate> =
                candidate_dates.get(id.as_str()).cloned().unwrap_or_default();
            if let Some(own_day) = crate::temporal::day_of_timestamp(&mem.created_at) {
                if !dates.contains(&own_day) {
                    dates.push(own_day);
                }
            }
            score *= crate::temporal::window_factor(window, &dates);
        }

        // KG expansion: subtle relevance signal from entity co-occurrence
        if !kg_expansion.is_empty() {
            let content_lower = mem.content.to_lowercase();
            let kg_hits = kg_expansion
                .iter()
                .filter(|t| content_lower.contains(t.as_str()))
                .count();
            if kg_hits > 0 {
                score *= 1.0 + (kg_hits as f64 * 0.04).min(0.15);
            }
        }

        // PageRank-like link boost
        if let Some(lb) = link_boosts.get(id) {
            if *lb < 0.0 {
                score *= 1.0 + lb; // penalty (e.g. 1.0 - 0.9 = 0.1x score)
            } else {
                score *= 1.0 + lb; // boost
            }
        }

        // Watcher boost (dynamic context)
        if let Some(keywords) = watcher_keywords {
            let content_lower = mem.content.to_lowercase();
            let match_count = keywords
                .iter()
                .filter(|w| content_lower.contains(w.to_lowercase().as_str()))
                .count();
            if match_count > 0 {
                score *= 1.0 + (match_count as f64 * 0.2); // +20% per matching keyword
            }
        }

        // Also boost if tag match
        if let Some(filter_tags) = tags {
            let filter_set: std::collections::HashSet<String> =
                filter_tags.iter().map(|t| t.to_lowercase()).collect();
            if mem
                .tags
                .iter()
                .any(|t| filter_set.contains(&t.to_lowercase()))
            {
                score *= 1.5;
            } else {
                score *= 0.1; // penalize if tags are requested but don't match
            }
        }

        // Penalize memories linked to expired knowledge triples (batch-preloaded)
        if let Some(&(active, expired)) = triple_counts.get(id.as_str()) {
            if expired > 0 {
                if active == 0 {
                    score *= 0.3;
                } else {
                    let ratio = active as f64 / (active + expired) as f64;
                    score *= 0.5 + (ratio * 0.5);
                }
            }
        }

        rrf_scores.push((id.clone(), score));
    }

    // Deterministic tie-break on the memory id. Without it, two
    // candidates that happen to score identically (a common case
    // when the cross-encoder normalises into a small dynamic
    // range) would be ordered by the unstable HashMap iteration
    // order, producing different top-K from one process to the
    // next on the same input. The id-based break costs nothing
    // and turns the bench output into a reproducible artefact.
    rrf_scores.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });

    let mut results: Vec<SearchResult> = Vec::new();
    // Session fusion needs a wider pool so it has something to
    // promote. MMR does *not* widen this pool: the graph, cluster and
    // cross-encoder stages are calibrated on a `limit`-sized pool and
    // widening it here cost 4–6 points of R@1 on memorypilot-fr-v2.
    // Instead the RRF ranks just below the pool are kept aside and
    // appended right before MMR, below every genuine finalist, so a
    // distinct candidate can still displace a near-duplicate.
    // A wide rerank needs that many finalists to rescore, whatever the
    // limit: the list is cut back to `limit` once they are ordered.
    let wide = crate::reranking::wide_scope();
    let pool = candidate_pool(limit, wide);
    let session_candidate_limit = if crate::session_fusion::should_expand_candidates(query) {
        (pool * 3).min(30).max(pool)
    } else {
        pool
    };
    let mmr_reserve_limit = if !wide && crate::diversity::is_enabled() {
        (pool * crate::diversity::pool_multiplier()).min(24).max(pool)
    } else {
        0
    };
    let mut mmr_reserve: Vec<SearchResult> = Vec::new();
    for (rank, (id, score)) in rrf_scores.into_iter().enumerate() {
        if rank >= session_candidate_limit.max(mmr_reserve_limit) {
            break;
        }
        if let Some(mem) = all_memories.remove(&id) {
            let mut sources = candidate_sources
                .remove(&id)
                .map(|items| items.into_iter().collect::<Vec<_>>())
                .unwrap_or_default();
            sources.sort();
            let result = SearchResult {
                memory: mem,
                score: (score * 10000.0).round() / 10000.0,
                sources,
            };
            if rank < session_candidate_limit {
                results.push(result);
            } else {
                mmr_reserve.push(result);
            }
        }
    }

    // 4. GraphRAG Traversal (Expand context based on top matches)
    let top_ids: Vec<String> = results
        .iter()
        .take(3)
        .map(|r| r.memory.id.clone())
        .collect();
    // Graph neighbours are context, not matches: they must rank below
    // every candidate the query itself retrieved. RRF scores live in
    // 0.002–0.049 (k = 40), so the former hard-coded 0.1 put every
    // neighbour *above* the genuine top-10, the combinatorial pass
    // then kept them there, and the cross-encoder's top-12 window
    // was spent on them while real hits fell past `limit` — R@10 on
    // memorypilot-fr-v2 was capped at ~70% by this alone. Neighbours
    // now enter at half the weakest genuine score, capped in number,
    // and only the cross-encoder can promote one.
    const GRAPH_EXPANSION_MAX: usize = 4;
    let graph_score = results
        .last()
        .map(|weakest| weakest.score * 0.5)
        .unwrap_or(0.0);
    if let Ok(related_ids) = store.graph_neighbours(&top_ids, 1) {
        // A set iterates in a different order every process: sorted, the
        // capped expansion picks the same neighbours on every run.
        let mut related_ids: Vec<String> = related_ids.into_iter().collect();
        related_ids.sort();
        let mut added = 0usize;
        for rel_id in related_ids {
            if added >= GRAPH_EXPANSION_MAX {
                break;
            }
            // If it's not already in results, fetch it and add it
            if !results.iter().any(|r| r.memory.id == rel_id) {
                if let Ok(Some(mem)) = store.fetch_memory(&rel_id) {
                    results.push(SearchResult {
                        memory: mem,
                        score: graph_score,
                        sources: vec!["graph".to_string()],
                    });
                    added += 1;
                }
            }
        }
    }

    // 5. Combinatorial Reranker — boost connected clusters
    // Greedy subgraph selection: prefer memories that are connected to other selected memories
    if results.len() > 2 {
        let result_ids: Vec<String> = results.iter().map(|r| r.memory.id.clone()).collect();
        let adjacency = store.build_adjacency_set(&result_ids);

        let mut selected: Vec<usize> = Vec::with_capacity(results.len());
        let mut remaining: Vec<usize> = (0..results.len()).collect();

        // Seed: pick the highest-scoring result
        selected.push(remaining.remove(0));

        while !remaining.is_empty() && selected.len() < results.len() {
            let mut best_idx = 0;
            let mut best_combined = f64::NEG_INFINITY;

            for (ri, &cand) in remaining.iter().enumerate() {
                let base_score = results[cand].score;
                let cand_id = &results[cand].memory.id;

                let conn_count = selected
                    .iter()
                    .filter(|&&si| {
                        let sel_id = &results[si].memory.id;
                        adjacency.contains(&(cand_id.clone(), sel_id.clone()))
                            || adjacency.contains(&(sel_id.clone(), cand_id.clone()))
                    })
                    .count();

                // connectivity_bonus: tiebreaker only — 5% per connected memory, capped at 15%
                let connectivity_bonus = (conn_count as f64 * 0.05).min(0.15);
                let combined = base_score * (1.0 + connectivity_bonus);

                if combined > best_combined {
                    best_combined = combined;
                    best_idx = ri;
                }
            }

            selected.push(remaining.remove(best_idx));
        }

        let mut reranked: Vec<SearchResult> = selected
            .into_iter()
            .map(|i| {
                std::mem::replace(
                    &mut results[i],
                    SearchResult {
                        memory: Memory {
                            id: String::new(),
                            content: String::new(),
                            kind: String::new(),
                            project: None,
                            tags: vec![],
                            source: String::new(),
                            importance: 0,
                            expires_at: None,
                            metadata: None,
                            created_at: String::new(),
                            updated_at: String::new(),
                            last_accessed_at: None,
                            access_count: 0,
                        },
                        score: 0.0,
                        sources: Vec::new(),
                    },
                )
            })
            .collect();

        // Ensure monotonically decreasing scores
        for i in 1..reranked.len() {
            let prev = reranked[i - 1].score;
            if reranked[i].score > prev {
                reranked[i].score = prev * 0.99;
            }
        }

        results = reranked;
    }

    crate::reranking::rerank_local(query, &mut results);
    let cross_score = crate::reranking::rerank_cross_encoder_if_enabled(query, &mut results);

    // Reserve candidates enter below the weakest finalist, in RRF
    // order; only MMR can lift one into the top-`limit`.
    if !mmr_reserve.is_empty() {
        let weakest = results.last().map(|r| r.score).unwrap_or(0.0);
        let mut reserve_score = weakest * 0.9;
        for mut reserve in mmr_reserve.drain(..) {
            if results.iter().any(|r| r.memory.id == reserve.memory.id) {
                continue;
            }
            reserve.score = (reserve_score * 10000.0).round() / 10000.0;
            reserve_score *= 0.98;
            results.push(reserve);
        }
    }

    // Superseded memories (replaced by a newer one) stay readable by id
    // but never surface in search.
    results.retain(|r| !supersede::is_superseded(&r.memory.tags));

    // Finalist vectors: one small IN query (≤ ~34 ids). They feed the
    // MMR redundancy term and the confidence cosine.
    let finalist_ids: Vec<&String> = results.iter().map(|r| &r.memory.id).collect();
    let finalist_vectors = store.batch_memory_vectors(&finalist_ids);
    let finalist_cosines: Vec<f32> = results
        .iter()
        .filter_map(|result| finalist_vectors.get(&result.memory.id))
        .map(|vector| crate::embedding::cosine_similarity(&query_emb, vector))
        .collect();
    let top_cosine = results
        .first()
        .and_then(|top| finalist_vectors.get(&top.memory.id))
        .map(|vector| crate::embedding::cosine_similarity(&query_emb, vector))
        .unwrap_or(0.0);
    let peak = if finalist_cosines.len() >= 3 {
        let others: Vec<f32> = finalist_cosines.iter().skip(1).copied().collect();
        top_cosine - others.iter().sum::<f32>() / others.len() as f32
    } else {
        0.0
    };
    let margin = match (results.first(), results.get(1)) {
        (Some(first), Some(second)) if first.score > 0.0 => {
            ((first.score - second.score) / first.score).clamp(0.0, 1.0)
        }
        (Some(_), None) => 1.0,
        _ => 0.0,
    };

    results = final_order(query, results, &finalist_vectors, limit, wide);

    let confidence = if results.is_empty() {
        SearchConfidence::none()
    } else {
        SearchConfidence::assess(top_cosine, peak, cross_score, margin)
    };

    if let Some(mut t) = trace.take() {
        t.timing_ms_fusion = fusion_start.elapsed().as_secs_f64() * 1000.0;
        t.timing_ms_total = search_start.elapsed().as_secs_f64() * 1000.0;
        t.results_returned = results.len();
        if let Some(top) = results.first() {
            t.top_score = top.score;
            t.top_sources = top.sources.clone();
        }
        crate::telemetry::emit(&t);
    }

    let returned: Vec<String> = results.iter().map(|res| res.memory.id.clone()).collect();
    store.record_access(&returned);

    Ok((results, confidence))
}

#[cfg(test)]
mod pool_tests {
    use super::{asks_about_assistant, candidate_pool, final_order, CompactOptions};
    use crate::db::{Memory, SearchResult};
    use std::collections::HashMap;

    fn turn(id: &str, score: f64) -> SearchResult {
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

    #[test]
    fn a_wide_list_keeps_the_reranker_order() {
        std::env::remove_var("MEMORYPILOT_MMR");
        // Three turns of one session first, then one turn of six others.
        let ids = ["a__t0", "a__t1", "a__t2", "b__t0", "c__t0", "d__t0", "e__t0", "f__t0", "g__t0"];
        let results: Vec<SearchResult> =
            ids.iter().enumerate().map(|(index, id)| turn(id, 1.0 - index as f64 * 0.01)).collect();
        // "what did" makes session fusion run on a default search.
        let query = "what did I do last week";
        let wide = final_order(query, results.clone(), &HashMap::new(), 5, true);
        let order: Vec<&str> = wide.iter().map(|r| r.memory.id.as_str()).collect();
        assert_eq!(order, ids[..5].to_vec(), "the session's turns stay where the reranker put them");
        let default = final_order(query, results, &HashMap::new(), 5, false);
        let order: Vec<&str> = default.iter().map(|r| r.memory.id.as_str()).collect();
        assert_eq!(order.len(), 5);
        assert_ne!(order, ids[..5].to_vec(), "a default search still spreads over sessions");
    }

    #[test]
    fn a_query_about_the_assistant_keeps_its_replies_whole_further_down() {
        assert!(asks_about_assistant("Can you remind me what you suggested for the trip?"));
        assert!(asks_about_assistant("Rappelle-moi ce que tu m\u{2019}as conseillé pour le voyage"));
        assert!(!asks_about_assistant("How many projects did I lead this year?"));
        assert!(!asks_about_assistant("Combien de projets ai-je menés cette année ?"));
        // Whole words only.
        assert!(asks_about_assistant("What did you say about the lease?"));
        assert!(!asks_about_assistant("When did your team ship the release?"));
        assert!(!asks_about_assistant("Did yourself or Sam book the hotel?"));
        assert!(!asks_about_assistant("Is the bayou saidto be flooded?"));
        // Other apostrophes and capitals.
        assert!(asks_about_assistant("Qu\u{02BC}est-ce que TU M\u{02BC}AS conseillé ?"));
        assert!(asks_about_assistant("RAPPELLE-MOI CE QUE tu as dit"));
        let defaults = CompactOptions::default();
        assert_eq!(defaults.for_query("What did you recommend for dinner?").full_top, 10);
        assert_eq!(defaults.for_query("Where did I park?"), defaults);
        let wider = CompactOptions { full_top: 20, chars: 150 };
        assert_eq!(wider.for_query("What did you recommend?").full_top, 20);
    }

    #[test]
    fn a_wide_rerank_rescores_its_full_width_whatever_the_limit() {
        assert_eq!(candidate_pool(10, false), 10);
        assert_eq!(candidate_pool(50, false), 50);
        assert_eq!(candidate_pool(30, true), 64);
        assert_eq!(candidate_pool(50, true), 64);
        assert_eq!(candidate_pool(80, true), 80);
    }
}
