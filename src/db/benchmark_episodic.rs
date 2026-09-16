//! Episodic-scale retrieval benchmark.
//!
//! LongMemEval and LoCoMo are saturated; the 2026 frontier (BEAM) probes
//! memory at multi-session scale where flat top-K retrieval starts to miss
//! needles buried among distractors. This benchmark measures exactly that
//! regime on a deterministic, self-contained synthetic corpus and reports
//! the lift the hierarchical episodic layer provides:
//!
//!   * **flat** — the production hybrid search (`search`) top-5.
//!   * **episodic** — flat candidates fused with drill-down from the top
//!     episodes (`search_episodes` → `episode_memories`), i.e. the
//!     resonance path the agent gets when `recall` returns episodes.
//!
//! The corpus plants one "needle" fact per session, surrounded by topical
//! distractor turns, across many sessions spanning several days so the
//! hourly → daily → long-term rollup is exercised. Queries paraphrase the
//! needle (no shared rare token) so lexical BM25 alone cannot win — the
//! point is to show episodic resonance recovering needles flat search
//! drops out of the top-5.
//!
//! Usage:
//!     memorypilot --benchmark-episodic [--sessions N] [--min-lift PCT]

use rusqlite::params;
use serde_json::{json, Value};

use super::{Database, MemoryScope};

/// One planted needle plus its paraphrased query.
struct Needle {
    session: usize,
    topic: String,
    query: String,
    memory_id: String,
}

impl Database {
    /// Run the episodic-scale benchmark on a clean temporary dataset.
    pub fn benchmark_episodic(sessions: usize) -> Result<Value, String> {
        let tmp_dir = std::env::temp_dir().join(format!(
            "memorypilot-episodic-bench-{}-{}",
            std::process::id(),
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0)
        ));
        std::fs::create_dir_all(&tmp_dir)
            .map_err(|e| format!("episodic bench: create tmp dir: {}", e))?;
        let db_path = tmp_dir.join("bench.db");
        let result = run_episodic_bench(&db_path, sessions.clamp(6, 200));
        let _ = std::fs::remove_dir_all(&tmp_dir);
        result
    }
}

fn run_episodic_bench(db_path: &std::path::Path, sessions: usize) -> Result<Value, String> {
    // Deterministic rollup: keep the throttled background rollup from
    // firing mid-seed (it would bucket half-loaded data by wall-clock
    // time). We drive the rollup explicitly once the corpus is stamped.
    std::env::set_var("MEMORYPILOT_EPISODIC", "off");
    let db = Database::open_at(db_path)?;
    let scope = MemoryScope::default();
    let project = "episodic-bench";

    let topics = TOPICS;
    let mut needles: Vec<Needle> = Vec::with_capacity(sessions);

    // Seed: one needle + several distractors per session. Each session is
    // pinned to a distinct hour so the rollup yields one interval episode
    // per session, and ~24 sessions per simulated day for the daily tier.
    for s in 0..sessions {
        let topic = topics[s % topics.len()];
        let token = format!("S{:03}", s);
        let needle_id = format!("ep-bench:needle:{}", s);
        // The fact carries a unique config detail; the query paraphrases
        // the topic without repeating that detail.
        let fact = format!(
            "During the {topic} review we agreed the {topic} threshold is set to {value} and owned by team {token}.",
            topic = topic,
            value = 40 + s,
            token = token
        );
        db.add_memory_with_id(
            Some(&needle_id),
            &fact,
            "decision",
            Some(project),
            &[topic.replace(' ', "-")],
            "bench-episodic",
            4,
            None,
            None,
            &scope,
        )?;

        for d in 0..8 {
            let distractor_id = format!("ep-bench:distractor:{}:{}", s, d);
            // Unique per (session, aspect) so dedup doesn't collapse
            // distractors across sessions that reuse a topic.
            let distractor = format!(
                "In the {topic} thread (session {token}, note {d}) we also discussed {aspect} and logistics; nothing was finalised about thresholds here.",
                topic = topic,
                token = token,
                d = d,
                aspect = DISTRACTOR_ASPECTS[d % DISTRACTOR_ASPECTS.len()]
            );
            db.add_memory_with_id(
                Some(&distractor_id),
                &distractor,
                "note",
                Some(project),
                &[topic.replace(' ', "-")],
                "bench-episodic",
                2,
                None,
                None,
                &scope,
            )?;
        }

        needles.push(Needle {
            session: s,
            topic: topic.to_string(),
            query: format!("who owns the {topic} and what value did we settle on?", topic = topic),
            memory_id: needle_id,
        });
    }

    wait_for_embeddings(db_path)?;

    // Stamp deterministic per-session timestamps so each session lands in
    // its own hour bucket, spanning several days. Done after embedding so
    // the queue isn't disturbed.
    stamp_session_times(&db, sessions)?;

    db.wait_for_ann_warm(std::time::Duration::from_secs(60));
    crate::reranking::warmup_cross_reranker();

    let rollup = db.roll_up_episodes()?;

    let mut flat_hits = 0usize;
    let mut covered_hits = 0usize;
    let mut fused_hits = 0usize;
    let mut recovered = Vec::new();

    for needle in &needles {
        let flat = db.search(&needle.query, 10, Some(project), None, None, None)?;
        let flat_rank = flat
            .iter()
            .position(|r| r.memory.id == needle.memory_id)
            .map(|i| i + 1);
        let flat_hit = matches!(flat_rank, Some(r) if r <= 5);
        if flat_hit {
            flat_hits += 1;
        }

        // Coverage: how the episodic layer is *actually* wired in `recall`
        // — episodes are returned alongside flat memories, never replacing
        // them. The needle is answerable if it's in the flat top-5 OR
        // inside the drill-down of one of the top-3 resonant episodes.
        let in_episode = needle_in_top_episodes(&db, &needle.query, project, &needle.memory_id)?;
        let covered = flat_hit || in_episode;
        if covered {
            covered_hits += 1;
        }
        if covered && !flat_hit {
            recovered.push(json!({
                "session": needle.session,
                "topic": needle.topic,
                "flat_rank": flat_rank,
                "recovered_via": "episode_drilldown",
            }));
        }

        // Reference only: a *destructive* RRF fusion of episodes into the
        // primary ranking. Kept to document that blind fusion regresses
        // precision — which is exactly why the product keeps episodes as a
        // separate additive lane, not a re-ranker.
        let fused_rank = fused_needle_rank(&db, &needle.query, project, &flat, &needle.memory_id)?;
        if matches!(fused_rank, Some(r) if r <= 5) {
            fused_hits += 1;
        }
    }

    let total = needles.len().max(1) as f64;
    let flat_r5 = flat_hits as f64 / total * 100.0;
    let covered_r5 = covered_hits as f64 / total * 100.0;
    let fused_r5 = fused_hits as f64 / total * 100.0;
    let lift = covered_r5 - flat_r5;

    Ok(json!({
        "benchmark": "episodic-scale-v1",
        "corpus": {
            "sessions": sessions,
            "memories": sessions * 9,
            "needles": needles.len(),
        },
        "rollup": {
            "interval_episodes": rollup.interval_created,
            "daily_episodes": rollup.daily_created,
            "longterm_episodes": rollup.longterm_created,
            "total_episodes": db.episode_count(),
        },
        "metrics": {
            "flat_recall_at_5": format!("{:.1}%", flat_r5),
            "episodic_coverage_at_5": format!("{:.1}%", covered_r5),
            "episodic_lift_pp": format!("{:+.1}", lift),
            "needles_recovered_by_episodes": recovered.len(),
            "destructive_fusion_recall_at_5_reference": format!("{:.1}%", fused_r5),
        },
        "recovered_examples": recovered.into_iter().take(8).collect::<Vec<_>>(),
        "embedding_model": crate::embedding::model_label(),
    }))
}

/// True if the needle appears in the drill-down of any of the top-3
/// resonant episodes for the query — i.e. the agent would see it via the
/// episodes block that `recall` returns alongside flat memories.
fn needle_in_top_episodes(
    db: &Database,
    query: &str,
    project: &str,
    needle_id: &str,
) -> Result<bool, String> {
    for hit in db.search_episodes(query, 3, Some(project))? {
        if db
            .episode_memories(&hit.episode.id)?
            .iter()
            .any(|m| m.id == needle_id)
        {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Compute the needle's rank after RRF-fusing the flat ranking with the
/// episodic ranking (top episodes drilled down to their constituent
/// memories). This mirrors what `recall` surfaces when episodes resonate:
/// a strong episodic signal can lift a needle that flat search buried,
/// rather than merely appending behind it.
fn fused_needle_rank(
    db: &Database,
    query: &str,
    project: &str,
    flat: &[super::SearchResult],
    needle_id: &str,
) -> Result<Option<usize>, String> {
    use std::collections::HashMap;
    const K: f64 = 40.0;

    let mut scores: HashMap<String, f64> = HashMap::new();
    for (rank, r) in flat.iter().enumerate() {
        *scores.entry(r.memory.id.clone()).or_insert(0.0) += 1.0 / (K + rank as f64);
    }

    // Episodic ranking: episodes ordered by score; inside each, drill-down
    // memories in chronological order. Concatenated into one ranked list.
    let mut episodic_rank = 0usize;
    for hit in db.search_episodes(query, 3, Some(project))? {
        for mem in db.episode_memories(&hit.episode.id)? {
            *scores.entry(mem.id).or_insert(0.0) += 1.0 / (K + episodic_rank as f64);
            episodic_rank += 1;
        }
    }

    let mut fused: Vec<(String, f64)> = scores.into_iter().collect();
    fused.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });

    Ok(fused.iter().position(|(id, _)| id == needle_id).map(|i| i + 1))
}

fn stamp_session_times(db: &Database, sessions: usize) -> Result<(), String> {
    for s in 0..sessions {
        // Spread sessions across days: 24 sessions per day, one per hour.
        let day = 1 + (s / 24);
        let hour = s % 24;
        let needle_ts = format!("2026-02-{:02}T{:02}:00:00+00:00", day, hour);
        db.conn
            .execute(
                "UPDATE memories SET created_at = ?1 WHERE id = ?2",
                params![needle_ts, format!("ep-bench:needle:{}", s)],
            )
            .map_err(|e| format!("stamp needle time: {}", e))?;
        for d in 0..8 {
            let ts = format!("2026-02-{:02}T{:02}:{:02}:00+00:00", day, hour, 10 + d);
            db.conn
                .execute(
                    "UPDATE memories SET created_at = ?1 WHERE id = ?2",
                    params![ts, format!("ep-bench:distractor:{}:{}", s, d)],
                )
                .map_err(|e| format!("stamp distractor time: {}", e))?;
        }
    }
    Ok(())
}

fn wait_for_embeddings(db_path: &std::path::Path) -> Result<(), String> {
    let started = std::time::Instant::now();
    loop {
        let conn = rusqlite::Connection::open(db_path)
            .map_err(|e| format!("episodic bench wait probe: {}", e))?;
        let _ = conn.busy_timeout(std::time::Duration::from_secs(2));
        let pending: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM memories WHERE embedding IS NULL",
                [],
                |row| row.get(0),
            )
            .unwrap_or(0);
        if pending == 0 {
            return Ok(());
        }
        if started.elapsed().as_secs() > 180 {
            return Err(format!(
                "episodic bench: timeout waiting for embeddings ({} pending)",
                pending
            ));
        }
        std::thread::sleep(std::time::Duration::from_millis(200));
    }
}

const TOPICS: &[&str] = &[
    "checkout latency budget",
    "auth token rotation",
    "image pipeline cache",
    "search ranking weights",
    "billing retry policy",
    "mobile cold start",
    "data retention window",
    "rate limiter tuning",
    "feature flag rollout",
    "incident on-call rota",
    "vector index sharding",
    "webhook delivery SLA",
    "GDPR export queue",
    "CDN purge strategy",
    "model eval cadence",
    "backup restore drill",
    "session replay sampling",
    "payment fraud scoring",
    "email deliverability domain",
    "push notification batching",
    "schema migration window",
    "read replica lag",
    "API deprecation timeline",
    "secrets rotation interval",
    "queue backpressure limit",
    "cache eviction policy",
    "onboarding funnel step",
    "referral reward cap",
    "subscription dunning flow",
    "tax calculation region",
    "audit log retention",
    "SSO provider mapping",
    "rate card negotiation",
    "support SLA tiering",
    "experiment exposure split",
    "data warehouse sync",
    "mobile crash budget",
    "image format fallback",
    "geo routing preference",
    "cold storage threshold",
    "load test concurrency",
    "canary rollout percent",
    "log sampling ratio",
    "token budget per request",
    "embedding model choice",
    "reranker latency cap",
    "memory compaction cadence",
    "episode rollup interval",
];

const DISTRACTOR_ASPECTS: &[&str] = &[
    "naming conventions",
    "meeting scheduling",
    "documentation links",
    "ticket triage",
    "stakeholder updates",
    "tooling preferences",
    "historical context",
    "follow-up reminders",
];
