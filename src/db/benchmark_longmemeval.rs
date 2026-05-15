use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use chrono::Utc;
use rusqlite::{params, Connection};
use serde_json::json;

use super::{content_hash, Database};

impl Database {
    pub fn benchmark_longmemeval(
        dataset_path: &str,
        limit: Option<usize>,
    ) -> Result<serde_json::Value, String> {
        eprintln!("[LongMemEval] Loading dataset: {}", dataset_path);
        let raw = std::fs::read_to_string(dataset_path)
            .map_err(|e| format!("Cannot read dataset: {}", e))?;

        let entries: Vec<serde_json::Value> =
            serde_json::from_str(&raw).map_err(|e| format!("Invalid JSON: {}", e))?;

        let total = entries.len();
        eprintln!("[LongMemEval] {} questions loaded", total);

        let questions: Vec<&serde_json::Value> = entries
            .iter()
            .filter(|e| {
                let qid = e.get("question_id").and_then(|v| v.as_str()).unwrap_or("");
                !qid.ends_with("_abs")
            })
            .collect();

        let abstention_count = total - questions.len();
        let eval_count = if let Some(lim) = limit {
            lim.min(questions.len())
        } else {
            questions.len()
        };
        eprintln!(
            "[LongMemEval] Evaluating {} questions ({} abstention skipped)",
            eval_count, abstention_count
        );

        eprintln!("[LongMemEval] Phase 1: Pre-computing turn embeddings (global cache)...");
        let mut embedding_cache: std::collections::HashMap<String, Vec<f32>> =
            std::collections::HashMap::new();
        let mut unique_turns: Vec<(String, String)> = Vec::new();
        let mut seen_turns: std::collections::HashSet<String> = std::collections::HashSet::new();

        for entry in questions.iter().take(eval_count) {
            let question_date = entry.get("question_date").and_then(|v| v.as_str());
            let sessions = match entry.get("haystack_sessions").and_then(|v| v.as_array()) {
                Some(s) => s,
                None => continue,
            };
            let session_ids = match entry.get("haystack_session_ids").and_then(|v| v.as_array()) {
                Some(s) => s,
                None => continue,
            };
            let session_dates = entry.get("haystack_dates").and_then(|v| v.as_array());
            for (si, (session, sid_val)) in sessions.iter().zip(session_ids.iter()).enumerate() {
                let fallback_sid = format!("session_{}", si);
                let sid = sid_val.as_str().unwrap_or(&fallback_sid);
                let session_date = session_dates
                    .and_then(|dates| dates.get(si))
                    .and_then(|value| value.as_str());
                let temporal_prefix = Self::lme_temporal_prefix(session_date, question_date);
                let turns = match session.as_array() {
                    Some(t) => t,
                    None => continue,
                };
                for (ti, turn) in turns.iter().enumerate() {
                    let role = turn.get("role").and_then(|v| v.as_str()).unwrap_or("user");
                    let content = turn.get("content").and_then(|v| v.as_str()).unwrap_or("");
                    if content.is_empty() {
                        continue;
                    }
                    let key = format!("{}__t{}", sid, ti);
                    if seen_turns.insert(key.clone()) {
                        unique_turns
                            .push((key, format!("{}{}: {}", temporal_prefix, role, content)));
                    }
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

        let mut hits_r5 = 0usize;
        let mut hits_r10 = 0usize;
        let mut ndcg_sum = 0.0f64;
        let mut mrr_sum = 0.0f64;
        let mut total_search_ms = 0.0f64;
        let mut category_stats: std::collections::HashMap<String, (usize, usize, usize, f64, f64)> =
            std::collections::HashMap::new();
        let mut misses: Vec<serde_json::Value> = Vec::new();

        for (qi, entry) in questions.iter().take(eval_count).enumerate() {
            let question = entry.get("question").and_then(|v| v.as_str()).unwrap_or("");
            let question_type = entry
                .get("question_type")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            let answer_session_ids: Vec<String> = entry
                .get("answer_session_ids")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default();
            let sessions = entry.get("haystack_sessions").and_then(|v| v.as_array());
            let session_ids = entry.get("haystack_session_ids").and_then(|v| v.as_array());

            if sessions.is_none() || session_ids.is_none() || question.is_empty() {
                eprintln!("[LongMemEval] Skipping question {} (missing fields)", qi);
                continue;
            }
            let sessions = sessions.unwrap();
            let session_ids = session_ids.unwrap();
            let question_date = entry.get("question_date").and_then(|v| v.as_str());
            let session_dates = entry.get("haystack_dates").and_then(|v| v.as_array());

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

            let now = Utc::now().to_rfc3339();
            let mut turn_count = 0usize;

            {
                let tx = match db.conn.transaction() {
                    Ok(tx) => tx,
                    Err(e) => {
                        eprintln!("[LongMemEval] Q{}: transaction failed: {}", qi, e);
                        continue;
                    }
                };

                for (si, (session, sid_val)) in sessions.iter().zip(session_ids.iter()).enumerate()
                {
                    let fallback_sid = format!("session_{}", si);
                    let sid = sid_val.as_str().unwrap_or(&fallback_sid);
                    let session_date = session_dates
                        .and_then(|dates| dates.get(si))
                        .and_then(|value| value.as_str());
                    let temporal_prefix = Self::lme_temporal_prefix(session_date, question_date);
                    let turns = match session.as_array() {
                        Some(t) => t,
                        None => continue,
                    };
                    for (ti, turn) in turns.iter().enumerate() {
                        let content = turn.get("content").and_then(|v| v.as_str()).unwrap_or("");
                        if content.is_empty() {
                            continue;
                        }
                        let role = turn.get("role").and_then(|v| v.as_str()).unwrap_or("user");
                        let key = format!("{}__t{}", sid, ti);
                        let text = format!("{}{}: {}", temporal_prefix, role, content);
                        let emb = match embedding_cache.get(&key) {
                            Some(e) => e,
                            None => continue,
                        };
                        let blob = crate::embedding::vec_to_blob(emb);
                        let hash = content_hash(&text);
                        let _ = tx.execute(
                            "INSERT INTO memories (id, content, kind, project, tags, source, importance, embedding, content_hash, created_at, updated_at)
                             VALUES (?1, ?2, 'session', NULL, '[]', 'longmemeval', 3, ?3, ?4, ?5, ?5)",
                            params![key, text, blob, hash, now],
                        );
                        let _ = tx.execute(
                            "INSERT INTO memories_fts (rowid, content, tags, kind, project) VALUES ((SELECT rowid FROM memories WHERE id=?1), ?2, '[]', 'session', '')",
                            params![key, text],
                        );
                        turn_count += 1;
                    }
                }

                if let Err(e) = tx.commit() {
                    eprintln!("[LongMemEval] Q{}: commit failed: {}", qi, e);
                    continue;
                }
            }

            if qi % 50 == 0 {
                eprintln!(
                    "[LongMemEval] Q{}/{} ({}): {} turns indexed",
                    qi + 1,
                    eval_count,
                    question_type,
                    turn_count
                );
            }

            let start = std::time::Instant::now();
            let results = match db.search(question, 10, None, None, None, None) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("[LongMemEval] Q{}: search failed: {}", qi, e);
                    let _ = std::fs::remove_file(&tmp_path);
                    continue;
                }
            };
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            total_search_ms += elapsed_ms;

            let result_ids: Vec<&str> = results.iter().map(|r| r.memory.id.as_str()).collect();
            let matches_gold =
                |id: &&str| -> bool { answer_session_ids.iter().any(|gold| id.starts_with(gold)) };

            let in_top5 = result_ids.iter().take(5).any(matches_gold);
            let in_top10 = result_ids.iter().take(10).any(matches_gold);

            if in_top5 {
                hits_r5 += 1;
            }
            if in_top10 {
                hits_r10 += 1;
            }
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

            let best_rank = result_ids.iter().take(10).position(matches_gold);
            let ndcg = match best_rank {
                Some(pos) => 1.0 / (pos as f64 + 2.0).log2(),
                None => 0.0,
            };
            ndcg_sum += ndcg;

            let rr = match best_rank {
                Some(pos) => 1.0 / (pos as f64 + 1.0),
                None => 0.0,
            };
            mrr_sum += rr;

            let cat = category_stats
                .entry(question_type.to_string())
                .or_insert((0, 0, 0, 0.0, 0.0));
            cat.0 += 1;
            if in_top5 {
                cat.1 += 1;
            }
            if in_top10 {
                cat.2 += 1;
            }
            cat.3 += ndcg;
            cat.4 += rr;

            if (qi + 1) % 10 == 0 || qi + 1 == eval_count {
                let running_r5 = hits_r5 as f64 / (qi + 1) as f64 * 100.0;
                let running_r10 = hits_r10 as f64 / (qi + 1) as f64 * 100.0;
                eprintln!(
                    "[LongMemEval] {}/{} — R@5: {:.1}% R@10: {:.1}% ({} sessions indexed)",
                    qi + 1,
                    eval_count,
                    running_r5,
                    running_r10,
                    sessions.len()
                );
            }

            let _ = std::fs::remove_file(&tmp_path);
            let _ = std::fs::remove_file(tmp_path.with_extension("db-wal"));
            let _ = std::fs::remove_file(tmp_path.with_extension("db-shm"));
        }

        let actual_count = eval_count.max(1) as f64;
        let r5 = (hits_r5 as f64 / actual_count * 1000.0).round() / 10.0;
        let r10 = (hits_r10 as f64 / actual_count * 1000.0).round() / 10.0;
        let ndcg10 = (ndcg_sum / actual_count * 1000.0).round() / 10.0;
        let mrr = (mrr_sum / actual_count * 1000.0).round() / 10.0;
        let avg_ms = (total_search_ms / actual_count * 100.0).round() / 100.0;

        let mut by_category = serde_json::Map::new();
        for (cat, (count, r5h, r10h, ndcg_s, mrr_s)) in &category_stats {
            let c = *count as f64;
            by_category.insert(
                cat.clone(),
                json!({
                    "count": count,
                    "recall_at_5": format!("{:.1}%", *r5h as f64 / c * 100.0),
                    "recall_at_10": format!("{:.1}%", *r10h as f64 / c * 100.0),
                    "ndcg_at_10": format!("{:.1}%", ndcg_s / c * 100.0),
                    "mrr": format!("{:.1}%", mrr_s / c * 100.0),
                }),
            );
        }

        Ok(json!({
            "benchmark": "LongMemEval-S (ICLR 2025)",
            "dataset": dataset_path,
            "questions_total": total,
            "questions_evaluated": eval_count,
            "questions_abstention_skipped": abstention_count,
            "granularity": "turn",
            "embedding_model": "multilingual-e5-small (384-dim)",
            "search_engine": "BM25 + cosine RRF (k=40)",
            "metrics": {
                "recall_at_5": format!("{}%", r5),
                "recall_at_10": format!("{}%", r10),
                "ndcg_at_10": format!("{}%", ndcg10),
                "mrr": format!("{}%", mrr),
                "avg_search_latency_ms": avg_ms,
            },
            "misses": misses,
            "by_category": by_category,
        }))
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

        let parse =
            |value: &str| chrono::NaiveDateTime::parse_from_str(value, "%Y/%m/%d (%a) %H:%M").ok();
        if let (Some(session_dt), Some(question_dt)) =
            (parse(session_date), question_date.and_then(parse))
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
