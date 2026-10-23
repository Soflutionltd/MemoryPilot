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
        let canonical_project = Self::canonical_project(project);

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

        let _ = self.cleanup_expired();

        let embed_start = std::time::Instant::now();
        let query_emb = cached_embed_text(query);
        if let Some(t) = trace.as_mut() {
            t.timing_ms_embed_query = embed_start.elapsed().as_secs_f64() * 1000.0;
        }

        // Pre-compute KG expansion terms for post-retrieval scoring boost
        let kg_expansion = self.get_kg_expansion_terms(query);

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
                let mut conditions = vec!["memories_fts MATCH ?1".to_string()];
                let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> =
                    vec![Box::new(fts_query.to_string())];

                if let Some(p) = canonical_project.as_deref() {
                    conditions.push(format!("m.project = ?{}", param_values.len() + 1));
                    param_values.push(Box::new(p.to_string()));
                }
                if let Some(k) = kind {
                    conditions.push(format!("m.kind = ?{}", param_values.len() + 1));
                    param_values.push(Box::new(k.to_string()));
                }

                let where_clause = conditions.join(" AND ");
                let sql = format!(
                    "SELECT m.id,m.content,m.kind,m.project,m.tags,m.source,m.importance,m.expires_at,m.metadata,m.created_at,m.updated_at,m.last_accessed_at,m.access_count,
                            bm25(memories_fts, 8.0, 5.0, 2.0, 4.0) AS bm25_score
                     FROM memories_fts f
                     JOIN memories m ON m.rowid = f.rowid
                     WHERE {}
                     ORDER BY bm25_score ASC
                     LIMIT {}", where_clause, limit_rows);

                let mut stmt = match self.conn.prepare(&sql) {
                    Ok(stmt) => stmt,
                    Err(error) if return_err_on_failure => {
                        return Err(format!("Search prepare: {}", error));
                    }
                    Err(_) => return Ok(0),
                };
                let param_refs: Vec<&dyn rusqlite::types::ToSql> =
                    param_values.iter().map(|p| p.as_ref()).collect();
                let rows = match stmt.query_map(param_refs.as_slice(), |row| {
                    let mem = row_to_memory(row);
                    let bm25: f64 = row.get(13)?;
                    Ok((mem, bm25))
                }) {
                    Ok(rows) => rows,
                    Err(error) if return_err_on_failure => {
                        return Err(format!("Search: {}", error));
                    }
                    Err(_) => return Ok(0),
                };

                let mut produced = 0usize;
                for (rank_index, row) in rows.flatten().enumerate() {
                    let (mem, _) = row;
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
        let ann_hits: Vec<(String, f32)> = self
            .ann
            .as_ref()
            .map(|index| index.search(&query_emb, 200))
            .unwrap_or_default();
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
        const ANN_BYPASS_THRESHOLD: usize = 5_000;
        let ann_active_for_bypass = self
            .ann
            .as_ref()
            .map(|index| index.len() >= ANN_BYPASS_THRESHOLD && !ann_hits.is_empty())
            .unwrap_or(false);
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
            let mut vec_conditions = Vec::new();
            let mut vec_params: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
            if let Some(p) = canonical_project.as_deref() {
                vec_conditions.push(format!("project = ?{}", vec_params.len() + 1));
                vec_params.push(Box::new(p.to_string()));
            }
            if let Some(k) = kind {
                vec_conditions.push(format!("kind = ?{}", vec_params.len() + 1));
                vec_params.push(Box::new(k.to_string()));
            }
            if let Some(ids) = restricted_ids.as_ref() {
                let placeholders = (1..=ids.len())
                    .map(|i| format!("?{}", vec_params.len() + i))
                    .collect::<Vec<_>>()
                    .join(",");
                vec_conditions.push(format!("id IN ({})", placeholders));
                for id in ids {
                    vec_params.push(Box::new(id.clone()));
                }
            }
            let vec_where = if vec_conditions.is_empty() {
                String::new()
            } else {
                format!("WHERE {}", vec_conditions.join(" AND "))
            };
            let vec_sql = format!("SELECT id,content,kind,project,tags,source,importance,expires_at,metadata,created_at,updated_at,last_accessed_at,access_count,embedding FROM memories {}", vec_where);
            let rconn = self.read_conn();
            let mut stmt2 = rconn
                .prepare(&vec_sql)
                .map_err(|e| format!("Vector Search: {}", e))?;
            let vec_refs: Vec<&dyn rusqlite::types::ToSql> =
                vec_params.iter().map(|p| p.as_ref()).collect();

            let mut vector_scores: Vec<(String, f32)> = Vec::new();
            let rows2 = stmt2
                .query_map(vec_refs.as_slice(), |row| {
                    let mem = row_to_memory(row);
                    let blob: Option<Vec<u8>> = row.get(13)?;
                    Ok((mem, blob))
                })
                .map_err(|e| format!("Vector Search error: {}", e))?;

            for r in rows2.flatten() {
                let (mem, blob) = r;
                all_memories
                    .entry(mem.id.clone())
                    .or_insert_with(|| mem.clone());
                if let Some(b) = blob {
                    candidate_sources
                        .entry(mem.id.clone())
                        .or_default()
                        .insert("vector".to_string());
                    let score = crate::embedding::similarity_with_blob(&query_emb, &b);
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
        let link_boosts = self.build_link_boosts_for(&candidate_ids);

        // Batch-query knowledge triple counts (avoids N+1)
        let triple_counts = self.batch_triple_counts(&candidate_ids);

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
            self.batch_memory_dates(&candidate_ids)
        } else {
            std::collections::HashMap::new()
        };

        let now_ts = now.timestamp() as f64;

        let query_tokens: Vec<String> = query
            .split_whitespace()
            .map(|w| w.to_lowercase())
            .filter(|w| w.len() >= 3)
            .collect();
        let query_intent = Self::infer_query_intent(query);

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

            score *= Self::query_intent_factor(&query_intent, mem);
            score *= Self::cognitive_activation_factor(mem, now_ts);

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
        let session_candidate_limit = if crate::session_fusion::should_expand_candidates(query) {
            (limit * 3).clamp(limit, 30)
        } else {
            limit
        };
        let mmr_reserve_limit = if crate::diversity::is_enabled() {
            (limit * crate::diversity::pool_multiplier()).clamp(limit, 24)
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
        if let Ok(related_ids) = crate::graph::traverse_graph(&self.conn, &top_ids, 1) {
            let mut added = 0usize;
            for rel_id in related_ids {
                if added >= GRAPH_EXPANSION_MAX {
                    break;
                }
                // If it's not already in results, fetch it and add it
                if !results.iter().any(|r| r.memory.id == rel_id) {
                    if let Ok(Some(mem)) = self.get_memory(&rel_id) {
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
            let adjacency = self.build_adjacency_set(&result_ids);

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
        let finalist_vectors = self.batch_memory_vectors(&finalist_ids);
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

        crate::diversity::rerank_mmr(&mut results, &finalist_vectors);
        // The MMR reserve must not leak into session fusion: fusion only
        // sees a pool wider than `limit` when the query asked for it.
        if !crate::session_fusion::should_expand_candidates(query) {
            results.truncate(limit);
        }
        results = crate::session_fusion::fuse_sessions(query, results, limit);

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

        // Defer access-count + last_accessed updates to the embed worker
        // so the hot search path never holds the WAL writer lock. The
        // worker drains the queue every ~2 s in a single batch UPDATE.
        for res in &results {
            queue_access_update(res.memory.id.clone());
        }

        Ok((results, confidence))
    }
}
