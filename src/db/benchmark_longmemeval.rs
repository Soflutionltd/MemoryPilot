use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use chrono::{NaiveDateTime, TimeZone, Utc};
use rusqlite::{params, Connection};
use serde_json::json;

use super::{content_hash, Database};

/// Options for [`Database::benchmark_longmemeval`].
#[derive(Debug, Clone, Default)]
pub struct LongMemEvalOptions {
    /// Evaluate at most this many answerable questions.
    pub limit: Option<usize>,
    /// Run the extractive reader on the top-5 passages and score the
    /// answer string against the gold answer.
    pub reader: bool,
}

/// Per-question outcome, accumulated into the report.
struct Outcome {
    question_type: String,
    in_top5: bool,
    in_top10: bool,
    best_rank: Option<usize>,
    /// Multi-session only: gold sessions present in the top-10 / total.
    coverage: Option<(usize, usize)>,
    top_cosine: f32,
    abstained: bool,
    qa: Option<QaScore>,
}

#[derive(Debug, Clone, Copy, Default)]
struct QaScore {
    exact: bool,
    f1: f64,
    contains: bool,
    answered: bool,
}

impl Database {
    pub fn benchmark_longmemeval(
        dataset_path: &str,
        options: LongMemEvalOptions,
    ) -> Result<serde_json::Value, String> {
        eprintln!("[LongMemEval] Loading dataset: {}", dataset_path);
        let raw = std::fs::read_to_string(dataset_path)
            .map_err(|e| format!("Cannot read dataset: {}", e))?;

        let entries: Vec<serde_json::Value> =
            serde_json::from_str(&raw).map_err(|e| format!("Invalid JSON: {}", e))?;

        let total = entries.len();
        eprintln!("[LongMemEval] {} questions loaded", total);

        let is_abstention = |entry: &serde_json::Value| {
            entry
                .get("question_id")
                .and_then(|v| v.as_str())
                .map(|qid| qid.ends_with("_abs"))
                .unwrap_or(false)
        };
        let answerable: Vec<&serde_json::Value> =
            entries.iter().filter(|e| !is_abstention(e)).collect();
        let abstention: Vec<&serde_json::Value> =
            entries.iter().filter(|e| is_abstention(e)).collect();

        let eval_count = options
            .limit
            .map(|lim| lim.min(answerable.len()))
            .unwrap_or(answerable.len());
        // Abstention questions are evaluated in proportion to the limit
        // so a quick `--limit 100` still exercises the verdict.
        let abs_count = match options.limit {
            Some(lim) if lim < answerable.len() => {
                ((lim as f64 / answerable.len() as f64) * abstention.len() as f64).ceil() as usize
            }
            _ => abstention.len(),
        }
        .min(abstention.len());
        eprintln!(
            "[LongMemEval] Evaluating {} answerable + {} abstention questions",
            eval_count, abs_count
        );

        let selected: Vec<(&serde_json::Value, bool)> = answerable
            .iter()
            .take(eval_count)
            .map(|e| (*e, false))
            .chain(abstention.iter().take(abs_count).map(|e| (*e, true)))
            .collect();

        eprintln!("[LongMemEval] Phase 1: Pre-computing turn embeddings (global cache)...");
        let mut embedding_cache: HashMap<String, Vec<f32>> = HashMap::new();
        let mut unique_turns: Vec<(String, String)> = Vec::new();
        let mut seen_turns: HashSet<String> = HashSet::new();

        for (entry, _) in &selected {
            for turn in haystack_turns(entry) {
                if seen_turns.insert(turn.key.clone()) {
                    unique_turns.push((turn.key, turn.text));
                }
            }
        }

        let cache_conn = Self::open_lme_embedding_cache(dataset_path).ok();
        if let Some(conn) = cache_conn.as_ref() {
            for (key, text) in &unique_turns {
                let cache_key = format!("{}:{}", key, content_hash(text));
                let cached_blob = conn
                    .query_row(
                        "SELECT embedding FROM embeddings WHERE key = ?1",
                        params![cache_key],
                        |row| row.get::<_, Vec<u8>>(0),
                    )
                    .or_else(|_| {
                        conn.query_row(
                            "SELECT embedding FROM embeddings WHERE key = ?1",
                            params![key],
                            |row| row.get::<_, Vec<u8>>(0),
                        )
                    });
                if let Ok(blob) = cached_blob {
                    let vec = crate::embedding::blob_to_vec(&blob);
                    // Defence in depth: skip cached vectors whose dim
                    // doesn't match the active model. The cache key
                    // already includes the dim, but this guard means
                    // a half-migrated cache file never poisons a run.
                    if vec.len() == crate::embedding::vector_dim() {
                        embedding_cache.insert(key.clone(), vec);
                    }
                }
            }
        }

        let turns_to_embed: Vec<(String, String)> = unique_turns
            .into_iter()
            .filter(|(key, _)| !embedding_cache.contains_key(key))
            .collect();

        eprintln!(
            "[LongMemEval] {} unique turns ({} cached, {} to embed)",
            embedding_cache.len() + turns_to_embed.len(),
            embedding_cache.len(),
            turns_to_embed.len()
        );
        let batch_size = 256;
        for chunk_start in (0..turns_to_embed.len()).step_by(batch_size) {
            let chunk_end = (chunk_start + batch_size).min(turns_to_embed.len());
            let chunk_texts: Vec<&str> = turns_to_embed[chunk_start..chunk_end]
                .iter()
                .map(|(_, t)| t.as_str())
                .collect();
            let chunk_embeddings = crate::embedding::embed_batch(&chunk_texts);
            for (ci, emb) in chunk_embeddings.into_iter().enumerate() {
                let key = &turns_to_embed[chunk_start + ci].0;
                if let Some(conn) = cache_conn.as_ref() {
                    let text = &turns_to_embed[chunk_start + ci].1;
                    let cache_key = format!("{}:{}", key, content_hash(text));
                    let blob = crate::embedding::vec_to_blob(&emb);
                    let _ = conn.execute(
                        "INSERT OR REPLACE INTO embeddings (key, embedding) VALUES (?1, ?2)",
                        params![cache_key, blob],
                    );
                }
                embedding_cache.insert(key.clone(), emb);
            }
            if (chunk_start / batch_size) % 10 == 0 {
                eprintln!(
                    "[LongMemEval] Embedded {}/{} turns...",
                    chunk_end,
                    turns_to_embed.len()
                );
            }
        }
        eprintln!(
            "[LongMemEval] Phase 1 complete: {} embeddings cached ✓",
            embedding_cache.len()
        );

        if options.reader {
            eprintln!("[LongMemEval] Warming up the extractive reader...");
            crate::reader::warmup()?;
        }
        let supersede_pass = !matches!(
            std::env::var("MEMORYPILOT_LME_SUPERSEDE").as_deref(),
            Ok("0") | Ok("false") | Ok("off")
        );

        let mut outcomes: Vec<Outcome> = Vec::new();
        let mut abstention_outcomes: Vec<(f32, bool, Option<bool>)> = Vec::new();
        // Raw confidence signals per question, for offline calibration.
        let mut confidence_samples: Vec<serde_json::Value> = Vec::new();
        let mut total_search_ms = 0.0f64;
        let mut total_reader_ms = 0.0f64;
        let mut misses: Vec<serde_json::Value> = Vec::new();
        let mut qa_failures: Vec<serde_json::Value> = Vec::new();

        for (qi, (entry, is_abs)) in selected.iter().enumerate() {
            let question = entry.get("question").and_then(|v| v.as_str()).unwrap_or("");
            let question_type = entry
                .get("question_type")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            let gold_answer = entry
                .get("answer")
                .map(|v| match v {
                    serde_json::Value::String(s) => s.clone(),
                    other => other.to_string(),
                })
                .unwrap_or_default();
            let answer_session_ids: Vec<String> = entry
                .get("answer_session_ids")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default();
            if question.is_empty() {
                eprintln!("[LongMemEval] Skipping question {} (missing fields)", qi);
                continue;
            }
            let question_date = entry.get("question_date").and_then(|v| v.as_str());
            let question_now = question_date
                .and_then(parse_lme_date)
                .map(|naive| Utc.from_utc_datetime(&naive))
                .unwrap_or_else(Utc::now);

            let tmp_path = std::env::temp_dir().join(format!("memorypilot_lme_{}.db", qi));
            let mut db = match Self::open_lme_memory_db(qi) {
                Ok(db) => db,
                Err(memory_error) => {
                    let _ = std::fs::remove_file(&tmp_path);
                    let _ = std::fs::remove_file(tmp_path.with_extension("db-wal"));
                    let _ = std::fs::remove_file(tmp_path.with_extension("db-shm"));
                    match Self::open_lme_db(&tmp_path) {
                        Ok(d) => d,
                        Err(e) => {
                            eprintln!(
                                "[LongMemEval] Q{}: DB open failed: {}; memory fallback failed: {}",
                                qi, e, memory_error
                            );
                            continue;
                        }
                    }
                }
            };

            let mut turn_count = 0usize;
            {
                let tx = match db.conn.transaction() {
                    Ok(tx) => tx,
                    Err(e) => {
                        eprintln!("[LongMemEval] Q{}: transaction failed: {}", qi, e);
                        continue;
                    }
                };

                for turn in haystack_turns(entry) {
                    let emb = match embedding_cache.get(&turn.key) {
                        Some(e) => e,
                        None => continue,
                    };
                    let blob = crate::embedding::vec_to_blob(emb);
                    let hash = content_hash(&turn.text);
                    // The row is dated at the session, not at benchmark
                    // time: that is what the temporal window and the
                    // recency prior key on, exactly as in production.
                    let created_at = turn
                        .session_day
                        .map(|naive| Utc.from_utc_datetime(&naive).to_rfc3339())
                        .unwrap_or_else(|| question_now.to_rfc3339());
                    let _ = tx.execute(
                        "INSERT INTO memories (id, content, kind, project, tags, source, importance, embedding, content_hash, created_at, updated_at)
                         VALUES (?1, ?2, 'session', NULL, '[]', 'longmemeval', 3, ?3, ?4, ?5, ?5)",
                        params![turn.key, turn.text, blob, hash, created_at],
                    );
                    let _ = tx.execute(
                        "INSERT INTO memories_fts (rowid, content, tags, kind, project) VALUES ((SELECT rowid FROM memories WHERE id=?1), ?2, '[]', 'session', '')",
                        params![turn.key, turn.text],
                    );
                    let own_day = turn.session_day.map(|naive| naive.date());
                    for date in crate::temporal::extract_dates(&turn.text, own_day) {
                        let _ = tx.execute(
                            "INSERT INTO memory_entities (memory_id, entity_kind, entity_value) VALUES (?1, 'date', ?2)",
                            params![turn.key, date.format("%Y-%m-%d").to_string()],
                        );
                    }
                    turn_count += 1;
                }

                if let Err(e) = tx.commit() {
                    eprintln!("[LongMemEval] Q{}: commit failed: {}", qi, e);
                    continue;
                }
            }
            if supersede_pass {
                let _ = db.link_superseded_memories(None);
            }

            if qi % 50 == 0 {
                eprintln!(
                    "[LongMemEval] Q{}/{} ({}{}): {} turns indexed",
                    qi + 1,
                    selected.len(),
                    question_type,
                    if *is_abs { ", abstention" } else { "" },
                    turn_count
                );
            }

            let start = std::time::Instant::now();
            let (results, confidence) =
                match db.search_at(question_now, question, 10, None, None, None, None) {
                    Ok(r) => r,
                    Err(e) => {
                        eprintln!("[LongMemEval] Q{}: search failed: {}", qi, e);
                        let _ = std::fs::remove_file(&tmp_path);
                        continue;
                    }
                };
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            total_search_ms += elapsed_ms;
            confidence_samples.push(json!({
                "abs": *is_abs,
                "type": question_type,
                "cosine": (confidence.top_cosine as f64 * 10000.0).round() / 10000.0,
                "peak": (confidence.peak as f64 * 10000.0).round() / 10000.0,
                "cross": confidence.cross_score.map(|c| (c as f64 * 1000.0).round() / 1000.0),
                "margin": (confidence.margin * 10000.0).round() / 10000.0,
                "abstain": confidence.abstain,
            }));

            let reader_verdict = if options.reader {
                let passages: Vec<&str> = results
                    .iter()
                    .take(5)
                    .map(|r| r.memory.content.as_str())
                    .collect();
                let reader_start = std::time::Instant::now();
                let verdict = crate::reader::answer(question, &passages)
                    .map_err(|e| eprintln!("[LongMemEval] Q{}: reader failed: {}", qi, e))
                    .ok()
                    .flatten();
                total_reader_ms += reader_start.elapsed().as_secs_f64() * 1000.0;
                Some(verdict)
            } else {
                None
            };

            if *is_abs {
                let reader_abstained = reader_verdict.as_ref().map(|verdict| verdict.is_none());
                abstention_outcomes.push((confidence.top_cosine, confidence.abstain, reader_abstained));
                cleanup_lme_db(&tmp_path);
                continue;
            }

            let result_ids: Vec<&str> = results.iter().map(|r| r.memory.id.as_str()).collect();
            let matches_gold =
                |id: &&str| -> bool { answer_session_ids.iter().any(|gold| id.starts_with(gold)) };

            let in_top5 = result_ids.iter().take(5).any(matches_gold);
            let in_top10 = result_ids.iter().take(10).any(matches_gold);
            let best_rank = result_ids.iter().take(10).position(matches_gold);
            let coverage = if answer_session_ids.len() > 1 {
                let covered = answer_session_ids
                    .iter()
                    .filter(|gold| result_ids.iter().take(10).any(|id| id.starts_with(gold.as_str())))
                    .count();
                Some((covered, answer_session_ids.len()))
            } else {
                None
            };

            if !in_top5 {
                misses.push(json!({
                    "index": qi + 1,
                    "question_type": question_type,
                    "question": question,
                    "answer_session_ids": answer_session_ids,
                    "top10_ids": result_ids.iter().take(10).copied().collect::<Vec<_>>(),
                    "top5_hit": in_top5,
                    "top10_hit": in_top10,
                    "top_results": results.iter().take(5).map(|result| json!({
                        "id": result.memory.id,
                        "score": result.score,
                        "preview": result.memory.content.chars().take(220).collect::<String>(),
                    })).collect::<Vec<_>>(),
                }));
            }

            let qa = reader_verdict.map(|verdict| {
                let score = match verdict.as_ref() {
                    Some(found) => score_answer(&found.text, &gold_answer),
                    None => QaScore::default(),
                };
                if !score.contains && qa_failures.len() < 60 {
                    qa_failures.push(json!({
                        "index": qi + 1,
                        "question_type": question_type,
                        "question": question,
                        "gold": gold_answer,
                        "predicted": verdict.as_ref().map(|found| found.text.clone()),
                        "reader_score": verdict.as_ref().map(|found| (found.score * 100.0).round() / 100.0),
                        "gold_in_top5": in_top5,
                    }));
                }
                score
            });

            outcomes.push(Outcome {
                question_type: question_type.to_string(),
                in_top5,
                in_top10,
                best_rank,
                coverage,
                top_cosine: confidence.top_cosine,
                abstained: confidence.abstain,
                qa,
            });

            let done = outcomes.len();
            if done % 10 == 0 || done == eval_count {
                let hits5 = outcomes.iter().filter(|o| o.in_top5).count();
                let hits10 = outcomes.iter().filter(|o| o.in_top10).count();
                eprintln!(
                    "[LongMemEval] {}/{} — R@5: {:.1}% R@10: {:.1}% ({} turns indexed)",
                    done,
                    eval_count,
                    hits5 as f64 / done as f64 * 100.0,
                    hits10 as f64 / done as f64 * 100.0,
                    turn_count
                );
            }

            cleanup_lme_db(&tmp_path);
        }

        let actual_count = outcomes.len().max(1) as f64;
        let pct = |value: f64| (value * 1000.0).round() / 10.0;
        let hits_r5 = outcomes.iter().filter(|o| o.in_top5).count();
        let hits_r10 = outcomes.iter().filter(|o| o.in_top10).count();
        let ndcg_sum: f64 = outcomes
            .iter()
            .map(|o| o.best_rank.map(|pos| 1.0 / (pos as f64 + 2.0).log2()).unwrap_or(0.0))
            .sum();
        let mrr_sum: f64 = outcomes
            .iter()
            .map(|o| o.best_rank.map(|pos| 1.0 / (pos as f64 + 1.0)).unwrap_or(0.0))
            .sum();
        let searched = (outcomes.len() + abstention_outcomes.len()).max(1) as f64;
        let avg_ms = (total_search_ms / searched * 100.0).round() / 100.0;

        let coverage_values: Vec<f64> = outcomes
            .iter()
            .filter_map(|o| o.coverage.map(|(covered, total)| covered as f64 / total as f64))
            .collect();
        let coverage_all = outcomes
            .iter()
            .filter(|o| matches!(o.coverage, Some((covered, total)) if covered == total))
            .count();

        let mut by_category = serde_json::Map::new();
        let mut categories: Vec<&str> = outcomes.iter().map(|o| o.question_type.as_str()).collect();
        categories.sort();
        categories.dedup();
        for category in categories {
            let items: Vec<&Outcome> = outcomes
                .iter()
                .filter(|o| o.question_type == category)
                .collect();
            let c = items.len() as f64;
            let mut entry = json!({
                "count": items.len(),
                "recall_at_5": format!("{:.1}%", items.iter().filter(|o| o.in_top5).count() as f64 / c * 100.0),
                "recall_at_10": format!("{:.1}%", items.iter().filter(|o| o.in_top10).count() as f64 / c * 100.0),
                "ndcg_at_10": format!("{:.1}%", items.iter().map(|o| o.best_rank.map(|pos| 1.0 / (pos as f64 + 2.0).log2()).unwrap_or(0.0)).sum::<f64>() / c * 100.0),
                "mrr": format!("{:.1}%", items.iter().map(|o| o.best_rank.map(|pos| 1.0 / (pos as f64 + 1.0)).unwrap_or(0.0)).sum::<f64>() / c * 100.0),
            });
            let category_coverage: Vec<f64> = items
                .iter()
                .filter_map(|o| o.coverage.map(|(covered, total)| covered as f64 / total as f64))
                .collect();
            if !category_coverage.is_empty() {
                entry["gold_coverage_at_10"] = json!(format!(
                    "{:.1}%",
                    category_coverage.iter().sum::<f64>() / category_coverage.len() as f64 * 100.0
                ));
            }
            if options.reader {
                let scored: Vec<QaScore> = items.iter().filter_map(|o| o.qa).collect();
                if !scored.is_empty() {
                    entry["qa"] = qa_summary(&scored);
                }
            }
            by_category.insert(category.to_string(), entry);
        }

        // Abstention: the verdict should fire on `_abs` questions and stay
        // quiet on answerable ones. Cosine percentiles of both groups are
        // reported so the threshold can be re-calibrated from the data.
        let abs_total = abstention_outcomes.len();
        let abs_correct = abstention_outcomes.iter().filter(|(_, abstained, _)| *abstained).count();
        let false_abstain = outcomes.iter().filter(|o| o.abstained).count();
        let mut abs_cosines: Vec<f32> = abstention_outcomes.iter().map(|(c, _, _)| *c).collect();
        let mut ans_cosines: Vec<f32> = outcomes.iter().map(|o| o.top_cosine).collect();
        abs_cosines.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        ans_cosines.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let sweep: Vec<serde_json::Value> = [0.30, 0.35, 0.40, 0.42, 0.45, 0.50, 0.55]
            .iter()
            .map(|threshold| {
                let caught = abs_cosines.iter().filter(|c| **c < *threshold).count();
                let lost = ans_cosines.iter().filter(|c| **c < *threshold).count();
                json!({
                    "cosine_below": threshold,
                    "abstention_caught": format!("{}/{}", caught, abs_total),
                    "answerable_lost": format!("{}/{}", lost, outcomes.len()),
                })
            })
            .collect();
        let abstention_report = json!({
            "questions": abs_total,
            "threshold_cosine": super::SearchConfidence::abstain_cosine(),
            "abstained_correctly": abs_correct,
            "abstention_accuracy": if abs_total > 0 { format!("{:.1}%", abs_correct as f64 / abs_total as f64 * 100.0) } else { "n/a".into() },
            "false_abstention_on_answerable": format!("{}/{} ({:.1}%)", false_abstain, outcomes.len(), false_abstain as f64 / actual_count * 100.0),
            "reader_abstained_on_abstention": abstention_outcomes.iter().filter(|(_, _, reader)| *reader == Some(true)).count(),
            "top_cosine_percentiles": {
                "abstention": percentiles(&abs_cosines),
                "answerable": percentiles(&ans_cosines),
            },
            "threshold_sweep": sweep,
            "samples": confidence_samples,
        });

        let mut report = json!({
            "benchmark": "LongMemEval-S (ICLR 2025)",
            "dataset": dataset_path,
            "questions_total": total,
            "questions_evaluated": outcomes.len(),
            "questions_abstention_evaluated": abs_total,
            "granularity": "turn",
            "embedding_model": crate::embedding::model_label(),
            "search_engine": "BM25 + cosine RRF (k=40) → cross-encoder → MMR → session fusion; temporal window; supersede links",
            "supersede_pass": supersede_pass,
            "mmr_lambda": crate::diversity::lambda(),
            "metrics": {
                "recall_at_5": format!("{}%", pct(hits_r5 as f64 / actual_count)),
                "recall_at_10": format!("{}%", pct(hits_r10 as f64 / actual_count)),
                "ndcg_at_10": format!("{}%", pct(ndcg_sum / actual_count)),
                "mrr": format!("{}%", pct(mrr_sum / actual_count)),
                "multi_session_gold_coverage_at_10": if coverage_values.is_empty() { "n/a".to_string() } else {
                    format!("{}%", pct(coverage_values.iter().sum::<f64>() / coverage_values.len() as f64))
                },
                "multi_session_all_gold_at_10": if coverage_values.is_empty() { "n/a".to_string() } else {
                    format!("{}/{} ({}%)", coverage_all, coverage_values.len(), pct(coverage_all as f64 / coverage_values.len() as f64))
                },
                "avg_search_latency_ms": avg_ms,
            },
            "abstention": abstention_report,
            "misses": misses,
            "gold_ranks": outcomes.iter().map(|o| o.best_rank).collect::<Vec<_>>(),
            "by_category": by_category,
        });

        if options.reader {
            let scored: Vec<QaScore> = outcomes.iter().filter_map(|o| o.qa).collect();
            let mut qa = qa_summary(&scored);
            qa["reader_model"] = json!(crate::reader::READER_REPO);
            qa["judge"] = json!("SQuAD-style normalised exact-match / token-F1 / containment against the gold answer — not the official GPT-4o judge");
            qa["avg_reader_latency_ms"] = json!((total_reader_ms / searched * 100.0).round() / 100.0);
            qa["failures_sample"] = json!(qa_failures);
            report["qa"] = qa;
        }

        Ok(report)
    }

    fn lme_temporal_prefix(session_date: Option<&str>, question_date: Option<&str>) -> String {
        let session_date = match session_date {
            Some(value) if !value.trim().is_empty() => value.trim(),
            _ => return String::new(),
        };
        let mut parts = vec![format!("date: {}", session_date)];

        if let Some(day_start) = session_date.find('(') {
            if let Some(day_end) = session_date[day_start + 1..].find(')') {
                let weekday = &session_date[day_start + 1..day_start + 1 + day_end];
                if !weekday.is_empty() {
                    parts.push(format!("weekday: {}", weekday));
                }
            }
        }

        if let (Some(session_dt), Some(question_dt)) =
            (parse_lme_date(session_date), question_date.and_then(parse_lme_date))
        {
            let days_ago = (question_dt.date() - session_dt.date()).num_days();
            if days_ago >= 0 {
                parts.push(format!("{} days ago", days_ago));
                let weeks = ((days_ago as f64) / 7.0).round() as i64;
                if weeks >= 1 {
                    parts.push(format!("{} weeks ago", weeks));
                    if weeks == 1 {
                        parts.push("last week".to_string());
                    }
                }
                let months = ((days_ago as f64) / 30.0).round() as i64;
                if months >= 1 {
                    parts.push(format!("{} months ago", months));
                }
                if (1..=7).contains(&days_ago) {
                    if let Some(day_start) = session_date.find('(') {
                        if let Some(day_end) = session_date[day_start + 1..].find(')') {
                            let weekday = &session_date[day_start + 1..day_start + 1 + day_end];
                            parts.push(format!("last {}", weekday));
                        }
                    }
                }
            }
        }

        format!("[{}] ", parts.join("; "))
    }

    fn open_lme_db(path: &Path) -> Result<Self, String> {
        let conn = Connection::open(path).map_err(|e| format!("SQLite open: {}", e))?;
        conn.execute_batch(
            "
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = NORMAL;
            PRAGMA cache_size = -4000;
            PRAGMA foreign_keys = ON;
        ",
        )
        .map_err(|e| format!("Pragma: {}", e))?;

        let mut read_pool = Vec::with_capacity(1);
        let rc = Connection::open_with_flags(
            path,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )
        .map_err(|e| format!("Read pool: {}", e))?;
        let _ = rc.execute_batch("PRAGMA cache_size = -2000;");
        read_pool.push(Mutex::new(rc));

        let db = Self {
            conn,
            read_pool,
            ann: None,
            ann_warm_complete: Arc::new(std::sync::atomic::AtomicBool::new(true)),
        };
        db.init_schema()?;
        Ok(db)
    }

    fn open_lme_memory_db(index: usize) -> Result<Self, String> {
        let uri = format!("file:memorypilot_lme_{}?mode=memory&cache=shared", index);
        let flags = rusqlite::OpenFlags::SQLITE_OPEN_READ_WRITE
            | rusqlite::OpenFlags::SQLITE_OPEN_CREATE
            | rusqlite::OpenFlags::SQLITE_OPEN_URI
            | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX;
        let conn = Connection::open_with_flags(&uri, flags)
            .map_err(|e| format!("SQLite memory open: {}", e))?;
        conn.execute_batch(
            "
            PRAGMA journal_mode = MEMORY;
            PRAGMA synchronous = OFF;
            PRAGMA cache_size = -4000;
            PRAGMA foreign_keys = ON;
        ",
        )
        .map_err(|e| format!("Memory pragma: {}", e))?;

        let mut read_pool = Vec::with_capacity(1);
        let rc = Connection::open_with_flags(&uri, flags)
            .map_err(|e| format!("Memory read pool: {}", e))?;
        let _ = rc.execute_batch("PRAGMA cache_size = -2000;");
        read_pool.push(Mutex::new(rc));

        let db = Self {
            conn,
            read_pool,
            ann: None,
            ann_warm_complete: Arc::new(std::sync::atomic::AtomicBool::new(true)),
        };
        db.init_schema()?;
        Ok(db)
    }

    fn open_lme_embedding_cache(dataset_path: &str) -> Result<Connection, String> {
        let metadata =
            std::fs::metadata(dataset_path).map_err(|e| format!("Cache metadata: {}", e))?;
        let modified = metadata
            .modified()
            .ok()
            .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|duration| duration.as_secs())
            .unwrap_or(0);
        // Include the active embedding dim in the cache key so a model
        // swap (e.g. small → large) automatically picks up a fresh
        // cache file. Without this guard the bench loads 384-dim blobs
        // produced by a previous run and feeds them to a 1024-dim
        // query — every cosine similarity collapses to 0 and recall
        // drops to single-digits. Cheap to do, catches the foot-gun.
        let model_dim = crate::embedding::vector_dim();
        let cache_key = content_hash(&format!(
            "{}:{}:{}:dim{}",
            dataset_path,
            metadata.len(),
            modified,
            model_dim
        ));
        let cache_dir = std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join("target")
            .join("longmemeval-cache");
        std::fs::create_dir_all(&cache_dir).map_err(|e| format!("Cache dir: {}", e))?;
        let cache_path = cache_dir.join(format!("embeddings_{}.sqlite", cache_key));
        let conn = Connection::open(cache_path).map_err(|e| format!("Cache open: {}", e))?;
        conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA synchronous = NORMAL;
             CREATE TABLE IF NOT EXISTS embeddings (
                 key TEXT PRIMARY KEY,
                 embedding BLOB NOT NULL
             );",
        )
        .map_err(|e| format!("Cache schema: {}", e))?;
        Ok(conn)
    }
}

/// One haystack turn ready for indexing.
struct HaystackTurn {
    key: String,
    text: String,
    session_day: Option<NaiveDateTime>,
}

/// Every non-empty turn of a question's haystack, with the same
/// `[date …] role: content` text the embedding cache is keyed on.
fn haystack_turns(entry: &serde_json::Value) -> Vec<HaystackTurn> {
    let question_date = entry.get("question_date").and_then(|v| v.as_str());
    let Some(sessions) = entry.get("haystack_sessions").and_then(|v| v.as_array()) else {
        return Vec::new();
    };
    let Some(session_ids) = entry.get("haystack_session_ids").and_then(|v| v.as_array()) else {
        return Vec::new();
    };
    let session_dates = entry.get("haystack_dates").and_then(|v| v.as_array());
    let mut turns = Vec::new();
    for (si, (session, sid_val)) in sessions.iter().zip(session_ids.iter()).enumerate() {
        let fallback_sid = format!("session_{}", si);
        let sid = sid_val.as_str().unwrap_or(&fallback_sid);
        let session_date = session_dates
            .and_then(|dates| dates.get(si))
            .and_then(|value| value.as_str());
        let temporal_prefix = Database::lme_temporal_prefix(session_date, question_date);
        let session_day = session_date.and_then(parse_lme_date);
        let Some(session_turns) = session.as_array() else {
            continue;
        };
        for (ti, turn) in session_turns.iter().enumerate() {
            let content = turn.get("content").and_then(|v| v.as_str()).unwrap_or("");
            if content.is_empty() {
                continue;
            }
            let role = turn.get("role").and_then(|v| v.as_str()).unwrap_or("user");
            turns.push(HaystackTurn {
                key: format!("{}__t{}", sid, ti),
                text: format!("{}{}: {}", temporal_prefix, role, content),
                session_day,
            });
        }
    }
    turns
}

fn parse_lme_date(value: &str) -> Option<NaiveDateTime> {
    NaiveDateTime::parse_from_str(value.trim(), "%Y/%m/%d (%a) %H:%M").ok()
}

fn cleanup_lme_db(tmp_path: &Path) {
    let _ = std::fs::remove_file(tmp_path);
    let _ = std::fs::remove_file(tmp_path.with_extension("db-wal"));
    let _ = std::fs::remove_file(tmp_path.with_extension("db-shm"));
}

fn percentiles(sorted: &[f32]) -> serde_json::Value {
    if sorted.is_empty() {
        return json!(null);
    }
    let at = |fraction: f64| {
        let index = ((sorted.len() - 1) as f64 * fraction).round() as usize;
        (sorted[index] as f64 * 1000.0).round() / 1000.0
    };
    json!({ "p10": at(0.10), "p25": at(0.25), "p50": at(0.50), "p75": at(0.75), "p90": at(0.90) })
}

// ─── SQuAD-style answer scoring ─────────────────────────────────────

const ARTICLES: &[&str] = &["a", "an", "the"];

fn normalize_answer(text: &str) -> Vec<String> {
    text.to_lowercase()
        .chars()
        .map(|character| if character.is_alphanumeric() || character.is_whitespace() { character } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .filter(|token| !ARTICLES.contains(token))
        .map(String::from)
        .collect()
}

fn score_answer(predicted: &str, gold: &str) -> QaScore {
    let predicted_tokens = normalize_answer(predicted);
    let gold_tokens = normalize_answer(gold);
    if predicted_tokens.is_empty() || gold_tokens.is_empty() {
        return QaScore { answered: !predicted_tokens.is_empty(), ..QaScore::default() };
    }
    let exact = predicted_tokens == gold_tokens;
    let mut gold_counts: HashMap<&str, usize> = HashMap::new();
    for token in &gold_tokens {
        *gold_counts.entry(token.as_str()).or_default() += 1;
    }
    let mut common = 0usize;
    for token in &predicted_tokens {
        if let Some(count) = gold_counts.get_mut(token.as_str()) {
            if *count > 0 {
                *count -= 1;
                common += 1;
            }
        }
    }
    let f1 = if common == 0 {
        0.0
    } else {
        let precision = common as f64 / predicted_tokens.len() as f64;
        let recall = common as f64 / gold_tokens.len() as f64;
        2.0 * precision * recall / (precision + recall)
    };
    let predicted_joined = predicted_tokens.join(" ");
    let gold_joined = gold_tokens.join(" ");
    let contains = predicted_joined.contains(&gold_joined) || gold_joined.contains(&predicted_joined);
    QaScore { exact, f1, contains, answered: true }
}

fn qa_summary(scored: &[QaScore]) -> serde_json::Value {
    let count = scored.len().max(1) as f64;
    json!({
        "evaluated": scored.len(),
        "answered": scored.iter().filter(|s| s.answered).count(),
        "exact_match": format!("{:.1}%", scored.iter().filter(|s| s.exact).count() as f64 / count * 100.0),
        "token_f1": format!("{:.1}%", scored.iter().map(|s| s.f1).sum::<f64>() / count * 100.0),
        "contains": format!("{:.1}%", scored.iter().filter(|s| s.contains).count() as f64 / count * 100.0),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn squad_scoring_normalises_articles_and_punctuation() {
        let exact = score_answer("the Business Administration.", "Business Administration");
        assert!(exact.exact && exact.contains && (exact.f1 - 1.0).abs() < 1e-9);
        let partial = score_answer("a degree in Business Administration", "Business Administration");
        assert!(!partial.exact && partial.contains && partial.f1 > 0.6);
        let wrong = score_answer("marathon", "Business Administration");
        assert!(!wrong.contains && wrong.f1 == 0.0);
        let empty = score_answer("", "42");
        assert!(!empty.answered);
    }

    #[test]
    fn lme_dates_parse() {
        let parsed = parse_lme_date("2023/05/30 (Tue) 23:40").unwrap();
        assert_eq!(parsed.date().to_string(), "2023-05-30");
    }
}
