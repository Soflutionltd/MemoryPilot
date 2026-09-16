//! Hierarchical episodic memory (MemoryPilot v4.3).
//!
//! Raw memories live in the `memories` table and answer "what exactly was
//! said". Episodes sit *above* them and answer "what happened, and when":
//! they are time-bounded, summarised, embedded rollups of the underlying
//! memories. Three levels form a tree:
//!
//! ```text
//!   longterm (monthly)   ── child_ids ──▶  daily  ── child_ids ──▶  interval (hourly)
//!        ▲                                                              │ source_ids
//!        └──────────────────── drill-down ─────────────────────────────┘ ▶ raw memories
//! ```
//!
//! This is the architecture the 2026 SOTA systems (LIGHT, TiMem's Temporal
//! Memory Tree, EverMemOS) converged on: a small set of dense, salient
//! episodes that resonate at retrieval time, with O(1) drill-down to the
//! exact raw memories when the agent needs the detail. Episodes are few
//! (hundreds, not millions), so the episodic retrieval lane scores them in
//! memory — no FTS triggers, no second ANN index to keep warm.
//!
//! The rollup is fully deterministic and idempotent: an episode's
//! `content_hash` is derived from its sorted `source_ids`, and a UNIQUE
//! index makes re-running the rollup a no-op.

use super::{content_hash, Database, Memory};
use crate::embedding;
use rusqlite::params;
use serde::Serialize;
use uuid::Uuid;

/// Hourly interval bucket key length: "YYYY-MM-DDTHH" (RFC3339 / ISO 8601).
const INTERVAL_KEY_LEN: usize = 13;
/// Daily bucket key length: "YYYY-MM-DD".
const DAILY_KEY_LEN: usize = 10;
/// Monthly (long-term) bucket key length: "YYYY-MM".
const MONTHLY_KEY_LEN: usize = 7;

/// Minimum raw memories required before an interval episode is worth
/// creating. A single memory is already its own best summary.
const MIN_MEMORIES_PER_INTERVAL: usize = 2;
/// Minimum interval episodes in a day before a daily rollup is created.
const MIN_INTERVALS_PER_DAY: usize = 2;
/// Minimum daily episodes in a month before a long-term rollup is created.
const MIN_DAILIES_PER_MONTH: usize = 2;

#[derive(Debug, Clone, Serialize)]
pub struct Episode {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project: Option<String>,
    pub level: String,
    pub title: String,
    pub summary: String,
    pub start_at: String,
    pub end_at: String,
    pub salience: f64,
    pub memory_count: i64,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub source_ids: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub entities: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub child_ids: Vec<String>,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct EpisodeHit {
    pub episode: Episode,
    pub score: f64,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub sources: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct RollupReport {
    pub memories_scanned: usize,
    pub interval_created: usize,
    pub daily_created: usize,
    pub longterm_created: usize,
}

/// A raw memory reduced to the fields the rollup needs.
struct RawRow {
    id: String,
    content: String,
    kind: String,
    project: Option<String>,
    importance: i32,
    created_at: String,
}

impl Database {
    /// Build / refresh the episodic memory tree from raw memories.
    ///
    /// Idempotent: episodes already materialised (same sorted source set)
    /// are skipped via the `content_hash` UNIQUE index. Safe to call from a
    /// background thread; it only writes to the `episodes` table.
    pub fn roll_up_episodes(&self) -> Result<RollupReport, String> {
        let mut report = RollupReport::default();

        // ── Level 1: raw memories → hourly interval episodes ──────────
        let raw_rows = self.load_rollup_rows()?;
        report.memories_scanned = raw_rows.len();
        let interval_buckets = bucket_rows(&raw_rows, INTERVAL_KEY_LEN);
        for (_, rows) in interval_buckets {
            if rows.len() < MIN_MEMORIES_PER_INTERVAL {
                continue;
            }
            if self.insert_interval_episode(&rows)? {
                report.interval_created += 1;
            }
        }

        // ── Level 2: interval episodes → daily episodes ───────────────
        let intervals = self.load_episodes_by_level("interval")?;
        for (_, group) in group_episodes(&intervals, DAILY_KEY_LEN) {
            if group.len() < MIN_INTERVALS_PER_DAY {
                continue;
            }
            if self.insert_parent_episode(&group, "daily")? {
                report.daily_created += 1;
            }
        }

        // ── Level 3: daily episodes → monthly long-term episodes ──────
        let dailies = self.load_episodes_by_level("daily")?;
        for (_, group) in group_episodes(&dailies, MONTHLY_KEY_LEN) {
            if group.len() < MIN_DAILIES_PER_MONTH {
                continue;
            }
            if self.insert_parent_episode(&group, "longterm")? {
                report.longterm_created += 1;
            }
        }

        Ok(report)
    }

    /// Episodic retrieval lane. Scores every episode (project-scoped if
    /// given) against the query with a fused dense + lexical signal and
    /// returns the top `limit`. Cheap because episodes are few.
    pub fn search_episodes(
        &self,
        query: &str,
        limit: usize,
        project: Option<&str>,
    ) -> Result<Vec<EpisodeHit>, String> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let canonical = Self::canonical_project(project);
        let query_emb = super::cached_embed_text(query);
        let query_terms = significant_terms(query);

        let sql = "SELECT id, project, level, title, summary, embedding, start_at, end_at, \
                          salience, memory_count, source_ids, entities, child_ids, created_at, updated_at \
                   FROM episodes";
        let mut stmt = self
            .conn
            .prepare(sql)
            .map_err(|e| format!("search_episodes prepare: {}", e))?;
        let rows = stmt
            .query_map([], |row| {
                let blob: Option<Vec<u8>> = row.get(5)?;
                Ok((row_to_episode(row)?, blob))
            })
            .map_err(|e| format!("search_episodes query: {}", e))?;

        let mut hits: Vec<EpisodeHit> = Vec::new();
        for row in rows.flatten() {
            let (episode, blob) = row;
            if let Some(p) = canonical.as_deref() {
                if episode.project.as_deref() != Some(p) {
                    continue;
                }
            }
            let dense = blob
                .as_ref()
                .map(|b| embedding::similarity_with_blob(&query_emb, b))
                .unwrap_or(0.0)
                .max(0.0) as f64;
            let haystack = format!(
                "{} {} {}",
                episode.title.to_ascii_lowercase(),
                episode.summary.to_ascii_lowercase(),
                episode.entities.join(" ").to_ascii_lowercase()
            );
            let lexical = if query_terms.is_empty() {
                0.0
            } else {
                let hits = query_terms
                    .iter()
                    .filter(|t| haystack.contains(t.as_str()))
                    .count();
                hits as f64 / query_terms.len() as f64
            };
            // Salience nudges denser, more important episodes up without
            // overpowering relevance. Capped so a huge episode can't win
            // on size alone.
            let salience_bonus = (episode.salience / 10.0).min(0.15);
            let score = dense * 0.7 + lexical * 0.3 + salience_bonus;
            if score <= 0.0 {
                continue;
            }
            let mut sources = vec!["episodic".to_string()];
            if dense > 0.0 {
                sources.push("episodic_vector".to_string());
            }
            if lexical > 0.0 {
                sources.push("episodic_lexical".to_string());
            }
            hits.push(EpisodeHit {
                episode,
                score,
                sources,
            });
        }

        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.episode.id.cmp(&b.episode.id))
        });
        hits.truncate(limit);
        Ok(hits)
    }

    /// Fetch a single episode by id.
    pub fn get_episode(&self, id: &str) -> Result<Option<Episode>, String> {
        let sql = "SELECT id, project, level, title, summary, embedding, start_at, end_at, \
                          salience, memory_count, source_ids, entities, child_ids, created_at, updated_at \
                   FROM episodes WHERE id = ?1";
        let mut stmt = self
            .conn
            .prepare(sql)
            .map_err(|e| format!("get_episode prepare: {}", e))?;
        let mut rows = stmt
            .query_map(params![id], |row| row_to_episode(row))
            .map_err(|e| format!("get_episode query: {}", e))?;
        match rows.next() {
            Some(Ok(ep)) => Ok(Some(ep)),
            _ => Ok(None),
        }
    }

    /// Drill down from an episode to its constituent raw memories. For
    /// interval episodes that is `source_ids`; for daily/longterm episodes
    /// it recursively resolves children down to the raw memories.
    pub fn episode_memories(&self, id: &str) -> Result<Vec<Memory>, String> {
        let leaf_ids = self.resolve_leaf_source_ids(id, 0)?;
        if leaf_ids.is_empty() {
            return Ok(Vec::new());
        }
        let placeholders = leaf_ids
            .iter()
            .enumerate()
            .map(|(i, _)| format!("?{}", i + 1))
            .collect::<Vec<_>>()
            .join(",");
        let sql = format!(
            "SELECT id,content,kind,project,tags,source,importance,expires_at,metadata,created_at,updated_at,last_accessed_at,access_count \
             FROM memories WHERE id IN ({}) ORDER BY created_at ASC",
            placeholders
        );
        let mut stmt = self
            .conn
            .prepare(&sql)
            .map_err(|e| format!("episode_memories prepare: {}", e))?;
        let param_refs: Vec<&dyn rusqlite::types::ToSql> = leaf_ids
            .iter()
            .map(|s| s as &dyn rusqlite::types::ToSql)
            .collect();
        let rows = stmt
            .query_map(param_refs.as_slice(), |row| Ok(super::row_to_memory(row)))
            .map_err(|e| format!("episode_memories query: {}", e))?;
        Ok(rows.flatten().collect())
    }

    /// List episodes for inspection / the health dashboard.
    pub fn list_episodes(
        &self,
        project: Option<&str>,
        level: Option<&str>,
        limit: usize,
    ) -> Result<Vec<Episode>, String> {
        let canonical = Self::canonical_project(project);
        let mut sql = String::from(
            "SELECT id, project, level, title, summary, embedding, start_at, end_at, \
                    salience, memory_count, source_ids, entities, child_ids, created_at, updated_at \
             FROM episodes",
        );
        let mut conditions = Vec::new();
        let mut binds: Vec<String> = Vec::new();
        if let Some(p) = canonical.as_deref() {
            binds.push(p.to_string());
            conditions.push(format!("project = ?{}", binds.len()));
        }
        if let Some(l) = level {
            binds.push(l.to_string());
            conditions.push(format!("level = ?{}", binds.len()));
        }
        if !conditions.is_empty() {
            sql.push_str(" WHERE ");
            sql.push_str(&conditions.join(" AND "));
        }
        sql.push_str(" ORDER BY end_at DESC LIMIT ");
        sql.push_str(&limit.clamp(1, 500).to_string());

        let mut stmt = self
            .conn
            .prepare(&sql)
            .map_err(|e| format!("list_episodes prepare: {}", e))?;
        let param_refs: Vec<&dyn rusqlite::types::ToSql> = binds
            .iter()
            .map(|s| s as &dyn rusqlite::types::ToSql)
            .collect();
        let rows = stmt
            .query_map(param_refs.as_slice(), |row| row_to_episode(row))
            .map_err(|e| format!("list_episodes query: {}", e))?;
        Ok(rows.flatten().collect())
    }

    pub fn episode_count(&self) -> i64 {
        self.conn
            .query_row("SELECT COUNT(*) FROM episodes", [], |r| r.get(0))
            .unwrap_or(0)
    }

    // ── internals ────────────────────────────────────────────────────

    fn load_rollup_rows(&self) -> Result<Vec<RawRow>, String> {
        // Episodes summarise primary content. Skip rows that are
        // themselves summaries (capsules) to avoid summarising summaries.
        let sql = "SELECT id, content, kind, project, importance, created_at \
                   FROM memories \
                   WHERE kind NOT IN ('capsule', 'session_capsule') \
                   ORDER BY created_at ASC";
        let mut stmt = self
            .conn
            .prepare(sql)
            .map_err(|e| format!("load_rollup_rows prepare: {}", e))?;
        let rows = stmt
            .query_map([], |row| {
                Ok(RawRow {
                    id: row.get(0)?,
                    content: row.get(1)?,
                    kind: row.get(2)?,
                    project: row.get(3)?,
                    importance: row.get(4)?,
                    created_at: row.get(5)?,
                })
            })
            .map_err(|e| format!("load_rollup_rows query: {}", e))?;
        Ok(rows.flatten().collect())
    }

    fn load_episodes_by_level(&self, level: &str) -> Result<Vec<Episode>, String> {
        let sql = "SELECT id, project, level, title, summary, embedding, start_at, end_at, \
                          salience, memory_count, source_ids, entities, child_ids, created_at, updated_at \
                   FROM episodes WHERE level = ?1 ORDER BY start_at ASC";
        let mut stmt = self
            .conn
            .prepare(sql)
            .map_err(|e| format!("load_episodes_by_level prepare: {}", e))?;
        let rows = stmt
            .query_map(params![level], |row| row_to_episode(row))
            .map_err(|e| format!("load_episodes_by_level query: {}", e))?;
        Ok(rows.flatten().collect())
    }

    /// Materialise one interval episode from raw memories. Returns true if
    /// a new row was inserted (false if it already existed by hash).
    fn insert_interval_episode(&self, rows: &[&RawRow]) -> Result<bool, String> {
        let mut source_ids: Vec<String> = rows.iter().map(|r| r.id.clone()).collect();
        source_ids.sort();
        let hash = content_hash(&source_ids.join("|"));
        // Idempotency check *before* summarising and embedding: a full
        // pass over an already-rolled-up corpus must cost a few index
        // lookups, not one ONNX call per hour of history.
        if self.episode_exists(&hash)? {
            return Ok(false);
        }

        let memories: Vec<Memory> = rows.iter().map(|r| raw_to_memory(r)).collect();
        let summary = build_summary(&memories);
        let title = build_title(&memories);
        let start_at = rows.first().map(|r| r.created_at.clone()).unwrap_or_default();
        let end_at = rows.last().map(|r| r.created_at.clone()).unwrap_or_default();
        let avg_importance =
            rows.iter().map(|r| r.importance as f64).sum::<f64>() / rows.len().max(1) as f64;
        let salience = avg_importance + (rows.len() as f64).ln_1p();
        let project = dominant_project(rows.iter().map(|r| r.project.as_deref()));
        let entities = self.entities_for_memories(&source_ids);

        let embed_text = format!("{}\n{}", title, summary);
        let blob = embedding::vec_to_blob(&embedding::embed_text(&embed_text));

        self.write_episode(
            "interval",
            &title,
            &summary,
            &blob,
            &start_at,
            &end_at,
            salience,
            rows.len() as i64,
            &source_ids,
            &entities,
            &[],
            project.as_deref(),
            &hash,
        )
    }

    /// Materialise one parent (daily / longterm) episode from child episodes.
    fn insert_parent_episode(&self, children: &[&Episode], level: &str) -> Result<bool, String> {
        let mut child_ids: Vec<String> = children.iter().map(|c| c.id.clone()).collect();
        child_ids.sort();
        let hash = content_hash(&format!("{}:{}", level, child_ids.join("|")));
        if self.episode_exists(&hash)? {
            return Ok(false);
        }

        let mut bullets: Vec<String> = children
            .iter()
            .map(|c| format!("- {} ({}→{})", c.title, short_time(&c.start_at), short_time(&c.end_at)))
            .collect();
        bullets.truncate(12);
        let summary = bullets.join("\n");
        let start_at = children
            .iter()
            .map(|c| c.start_at.clone())
            .min()
            .unwrap_or_default();
        let end_at = children
            .iter()
            .map(|c| c.end_at.clone())
            .max()
            .unwrap_or_default();
        let memory_count: i64 = children.iter().map(|c| c.memory_count).sum();
        let salience =
            children.iter().map(|c| c.salience).sum::<f64>() / children.len().max(1) as f64;
        let project = dominant_project(children.iter().map(|c| c.project.as_deref()));
        let title = format!(
            "{} · {} episodes · {}",
            if level == "longterm" { "Long-term" } else { "Daily" },
            children.len(),
            short_time(&start_at)
        );
        let mut entities: Vec<String> = children
            .iter()
            .flat_map(|c| c.entities.iter().cloned())
            .collect();
        entities.sort();
        entities.dedup();
        entities.truncate(24);

        let embed_text = format!("{}\n{}", title, summary);
        let blob = embedding::vec_to_blob(&embedding::embed_text(&embed_text));

        self.write_episode(
            level,
            &title,
            &summary,
            &blob,
            &start_at,
            &end_at,
            salience,
            memory_count,
            &[],
            &entities,
            &child_ids,
            project.as_deref(),
            &hash,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn episode_exists(&self, hash: &str) -> Result<bool, String> {
        self.conn
            .query_row(
                "SELECT 1 FROM episodes WHERE content_hash = ?1 LIMIT 1",
                params![hash],
                |_| Ok(()),
            )
            .map(|_| true)
            .or_else(|error| match error {
                rusqlite::Error::QueryReturnedNoRows => Ok(false),
                other => Err(format!("episode lookup: {}", other)),
            })
    }

    fn write_episode(
        &self,
        level: &str,
        title: &str,
        summary: &str,
        embedding: &[u8],
        start_at: &str,
        end_at: &str,
        salience: f64,
        memory_count: i64,
        source_ids: &[String],
        entities: &[String],
        child_ids: &[String],
        project: Option<&str>,
        hash: &str,
    ) -> Result<bool, String> {
        let now = chrono::Utc::now().to_rfc3339();
        let id = Uuid::new_v4().to_string();
        let source_json = serde_json::to_string(source_ids).unwrap_or_else(|_| "[]".into());
        let entities_json = serde_json::to_string(entities).unwrap_or_else(|_| "[]".into());
        let child_json = serde_json::to_string(child_ids).unwrap_or_else(|_| "[]".into());

        let affected = self
            .conn
            .execute(
                "INSERT OR IGNORE INTO episodes \
                 (id, project, level, title, summary, embedding, start_at, end_at, salience, \
                  memory_count, source_ids, entities, child_ids, content_hash, created_at, updated_at) \
                 VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?15)",
                params![
                    id,
                    project,
                    level,
                    title,
                    summary,
                    embedding,
                    start_at,
                    end_at,
                    salience,
                    memory_count,
                    source_json,
                    entities_json,
                    child_json,
                    hash,
                    now,
                ],
            )
            .map_err(|e| format!("write_episode insert: {}", e))?;
        Ok(affected > 0)
    }

    /// Recursively resolve an episode down to leaf raw-memory ids.
    fn resolve_leaf_source_ids(&self, id: &str, depth: usize) -> Result<Vec<String>, String> {
        if depth > 4 {
            return Ok(Vec::new());
        }
        let Some(ep) = self.get_episode(id)? else {
            return Ok(Vec::new());
        };
        if !ep.source_ids.is_empty() {
            return Ok(ep.source_ids);
        }
        let mut out = Vec::new();
        for child in ep.child_ids {
            out.extend(self.resolve_leaf_source_ids(&child, depth + 1)?);
        }
        out.sort();
        out.dedup();
        Ok(out)
    }

    fn entities_for_memories(&self, ids: &[String]) -> Vec<String> {
        if ids.is_empty() {
            return Vec::new();
        }
        let placeholders = ids
            .iter()
            .enumerate()
            .map(|(i, _)| format!("?{}", i + 1))
            .collect::<Vec<_>>()
            .join(",");
        let sql = format!(
            "SELECT DISTINCT entity_value FROM memory_entities WHERE memory_id IN ({}) LIMIT 24",
            placeholders
        );
        let Ok(mut stmt) = self.conn.prepare(&sql) else {
            return Vec::new();
        };
        let param_refs: Vec<&dyn rusqlite::types::ToSql> =
            ids.iter().map(|s| s as &dyn rusqlite::types::ToSql).collect();
        let Ok(rows) = stmt.query_map(param_refs.as_slice(), |row| row.get::<_, String>(0)) else {
            return Vec::new();
        };
        rows.flatten().collect()
    }
}

// ── free helpers ──────────────────────────────────────────────────────

fn row_to_episode(row: &rusqlite::Row) -> rusqlite::Result<Episode> {
    let source_ids: String = row.get(10)?;
    let entities: String = row.get(11)?;
    let child_ids: String = row.get(12)?;
    Ok(Episode {
        id: row.get(0)?,
        project: row.get(1)?,
        level: row.get(2)?,
        title: row.get(3)?,
        summary: row.get(4)?,
        start_at: row.get(6)?,
        end_at: row.get(7)?,
        salience: row.get(8)?,
        memory_count: row.get(9)?,
        source_ids: serde_json::from_str(&source_ids).unwrap_or_default(),
        entities: serde_json::from_str(&entities).unwrap_or_default(),
        child_ids: serde_json::from_str(&child_ids).unwrap_or_default(),
        created_at: row.get(13)?,
        updated_at: row.get(14)?,
    })
}

fn raw_to_memory(r: &RawRow) -> Memory {
    Memory {
        id: r.id.clone(),
        content: r.content.clone(),
        kind: r.kind.clone(),
        project: r.project.clone(),
        tags: Vec::new(),
        source: String::new(),
        importance: r.importance,
        expires_at: None,
        created_at: r.created_at.clone(),
        updated_at: r.created_at.clone(),
        metadata: None,
        last_accessed_at: None,
        access_count: 0,
    }
}

/// Bucket raw rows by a timestamp-prefix key of the given length, keeping
/// chronological order inside each bucket. Buckets are returned sorted by
/// key for deterministic rollup.
fn bucket_rows<'a>(rows: &'a [RawRow], key_len: usize) -> Vec<(String, Vec<&'a RawRow>)> {
    use std::collections::BTreeMap;
    let mut map: BTreeMap<String, Vec<&RawRow>> = BTreeMap::new();
    for r in rows {
        let key = time_key(&r.created_at, key_len);
        map.entry(key).or_default().push(r);
    }
    map.into_iter().collect()
}

fn group_episodes<'a>(eps: &'a [Episode], key_len: usize) -> Vec<(String, Vec<&'a Episode>)> {
    use std::collections::BTreeMap;
    let mut map: BTreeMap<String, Vec<&Episode>> = BTreeMap::new();
    for e in eps {
        let key = time_key(&e.start_at, key_len);
        map.entry(key).or_default().push(e);
    }
    map.into_iter().collect()
}

fn time_key(ts: &str, key_len: usize) -> String {
    let trimmed: String = ts.chars().take(key_len).collect();
    if trimmed.is_empty() {
        "unknown".to_string()
    } else {
        trimmed
    }
}

fn short_time(ts: &str) -> String {
    ts.chars().take(DAILY_KEY_LEN).collect()
}

fn dominant_project<'a>(projects: impl Iterator<Item = Option<&'a str>>) -> Option<String> {
    use std::collections::HashMap;
    let mut counts: HashMap<String, usize> = HashMap::new();
    for p in projects.flatten() {
        *counts.entry(p.to_string()).or_default() += 1;
    }
    counts
        .into_iter()
        .max_by_key(|(_, c)| *c)
        .map(|(p, _)| p)
}

/// Extractive summary from constituent memories. Reuses the session-capsule
/// scorer; falls back to a simple bullet list of leading sentences.
fn build_summary(memories: &[Memory]) -> String {
    if let Some(capsule) = crate::session_capsule::build_extractve_capsule(memories) {
        return capsule
            .trim_start_matches("## Session Capsule")
            .trim()
            .to_string();
    }
    memories
        .iter()
        .take(6)
        .map(|m| {
            let line: String = m.content.chars().take(160).collect();
            format!("- {}", line.trim())
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn build_title(memories: &[Memory]) -> String {
    let top = memories
        .iter()
        .max_by_key(|m| m.importance)
        .or_else(|| memories.first());
    match top {
        Some(m) => {
            let cleaned = m
                .content
                .trim()
                .trim_start_matches("user:")
                .trim_start_matches("assistant:")
                .trim();
            cleaned.chars().take(90).collect()
        }
        None => "Episode".to_string(),
    }
}

fn significant_terms(query: &str) -> Vec<String> {
    query
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= 3)
        .map(|t| t.to_ascii_lowercase())
        .filter(|t| {
            !matches!(
                t.as_str(),
                "the" | "and" | "for" | "with" | "what" | "when" | "pour" | "les" | "des" | "une"
            )
        })
        .collect()
}
