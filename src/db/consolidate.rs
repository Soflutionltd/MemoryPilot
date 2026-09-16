//! Embedding-based consolidation of near-duplicate memories.
//!
//! The write path already merges exact and word-Jaccard ≥ 0.85
//! duplicates, but agents restate the same fact with different words
//! ("app relancée, PID 339" / "app rebuild + relancée, process actif PID
//! 348"), and Jaccard misses those. Measured on a 2 166-memory corpus:
//! 107 memories sit within cosine 0.95 of another one in the same
//! project — almost all of them ephemeral status facts that only dilute
//! the candidate list the cross-encoder gets to see.
//!
//! Consolidation clusters memories of the same project whose stored
//! vectors are within `threshold`, keeps the most recently updated one
//! (status facts supersede each other), folds tags / importance /
//! access counts / links into it, and deletes the rest. A word-overlap
//! floor keeps translations apart (see `MIN_LEXICAL_OVERLAP`). It never touches
//! durable kinds (`decision`, `preference`, `architecture`, `pattern`,
//! `credential`) nor pinned memories: two decisions can be close in
//! embedding space and still both matter. Dry-run by default.

use std::collections::HashMap;

use rusqlite::params;
use serde_json::{json, Value};

use super::Database;

/// Cosine threshold used by the automatic daily pass. 0.95 was chosen
/// from the corpus histogram: above it the pairs are restatements of
/// one event; between 0.90 and 0.95 genuinely distinct items appear
/// (two slogans, two migrations).
pub const AUTO_THRESHOLD: f32 = 0.95;
/// Second guard: word-Jaccard the two texts must also reach. The
/// embedder is multilingual, so a French sentence and its English
/// translation sit at cosine ≈ 0.98 — they are two memories, not one.
/// Restatements of one event ("PID 339" / "PID 348") share most words.
const MIN_LEXICAL_OVERLAP: f64 = 0.6;
const AUTO_INTERVAL_SECS: i64 = 24 * 3600;
const AUTO_LAST_RUN_KEY: &str = "last_consolidate_at";

/// Kinds whose instances describe a state at a point in time and are
/// safely superseded by a newer restatement.
const EPHEMERAL_KINDS: &[&str] = &[
    "fact",
    "milestone",
    "note",
    "todo",
    "bug",
    "transcript",
    "resolved",
    "problem",
    "technical",
    "snippet",
];

struct Candidate {
    id: String,
    project: String,
    kind: String,
    tags: Vec<String>,
    importance: i64,
    access_count: i64,
    created_at: String,
    updated_at: String,
    content: String,
    normalized: String,
    vector: Vec<f32>,
}

/// Union-find over candidate indices.
fn root(parent: &mut [usize], mut index: usize) -> usize {
    while parent[index] != index {
        parent[index] = parent[parent[index]];
        index = parent[index];
    }
    index
}

fn unite(parent: &mut [usize], a: usize, b: usize) {
    let (ra, rb) = (root(parent, a), root(parent, b));
    if ra != rb {
        parent[ra.max(rb)] = ra.min(rb);
    }
}

impl Database {
    fn consolidation_candidates(&self, project: Option<&str>) -> Result<Vec<Candidate>, String> {
        let canonical_project = Self::canonical_project(project);
        let kinds = EPHEMERAL_KINDS
            .iter()
            .map(|kind| format!("'{}'", kind))
            .collect::<Vec<_>>()
            .join(",");
        let mut sql = format!(
            "SELECT id, project, kind, tags, importance, access_count, created_at, updated_at, content, embedding
             FROM memories
             WHERE embedding IS NOT NULL AND kind IN ({}) AND tags NOT LIKE '%pinned%'",
            kinds
        );
        let mut params_values: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
        if let Some(p) = canonical_project.as_deref() {
            sql.push_str(" AND project = ?1");
            params_values.push(Box::new(p.to_string()));
        }
        let mut stmt = self
            .conn
            .prepare(&sql)
            .map_err(|error| format!("consolidate prepare: {}", error))?;
        let refs: Vec<&dyn rusqlite::types::ToSql> = params_values.iter().map(|p| p.as_ref()).collect();
        let rows = stmt
            .query_map(refs.as_slice(), |row| {
                let tags: String = row.get(3)?;
                let blob: Vec<u8> = row.get(9)?;
                let content: String = row.get(8)?;
                Ok(Candidate {
                    id: row.get(0)?,
                    project: row.get::<_, Option<String>>(1)?.unwrap_or_default(),
                    kind: row.get(2)?,
                    tags: serde_json::from_str(&tags).unwrap_or_default(),
                    importance: row.get(4)?,
                    access_count: row.get(5)?,
                    created_at: row.get(6)?,
                    updated_at: row.get(7)?,
                    normalized: Self::normalize(&content),
                    content,
                    vector: crate::embedding::blob_to_vec(&blob),
                })
            })
            .map_err(|error| format!("consolidate query: {}", error))?;
        let expected_dim = crate::embedding::vector_dim();
        Ok(rows
            .flatten()
            .filter(|candidate| candidate.vector.len() == expected_dim)
            .collect())
    }

    /// Cluster near-duplicate ephemeral memories per project and, when
    /// `apply` is set, fold every cluster into its newest member.
    pub fn consolidate_memories(
        &self,
        threshold: f32,
        apply: bool,
        project: Option<&str>,
    ) -> Result<Value, String> {
        let threshold = threshold.clamp(0.80, 0.999);
        let candidates = self.consolidation_candidates(project)?;

        // Pairwise cosine inside each project. Vectors are unit-norm, so
        // the dot product is the cosine. 2 000 memories → ~2 M dot
        // products of 768 floats, well under a second.
        let mut by_project: HashMap<&str, Vec<usize>> = HashMap::new();
        for (index, candidate) in candidates.iter().enumerate() {
            by_project
                .entry(candidate.project.as_str())
                .or_default()
                .push(index);
        }
        let mut parent: Vec<usize> = (0..candidates.len()).collect();
        for members in by_project.values() {
            for (position, &a) in members.iter().enumerate() {
                for &b in &members[position + 1..] {
                    let cosine =
                        crate::embedding::cosine_similarity(&candidates[a].vector, &candidates[b].vector);
                    if cosine >= threshold
                        && Self::similarity(&candidates[a].normalized, &candidates[b].normalized)
                            >= MIN_LEXICAL_OVERLAP
                    {
                        unite(&mut parent, a, b);
                    }
                }
            }
        }

        let mut clusters: HashMap<usize, Vec<usize>> = HashMap::new();
        for index in 0..candidates.len() {
            let cluster_root = root(&mut parent, index);
            clusters.entry(cluster_root).or_default().push(index);
        }
        let mut clusters: Vec<Vec<usize>> = clusters
            .into_values()
            .filter(|members| members.len() > 1)
            .collect();
        // Newest member first: it becomes the canonical memory.
        for members in &mut clusters {
            members.sort_by(|&a, &b| candidates[b].updated_at.cmp(&candidates[a].updated_at));
        }
        clusters.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| candidates[a[0]].id.cmp(&candidates[b[0]].id)));

        let mut removed = 0usize;
        let mut examples = Vec::new();
        for members in &clusters {
            let canonical = &candidates[members[0]];
            let duplicates: Vec<&Candidate> = members[1..].iter().map(|&index| &candidates[index]).collect();
            if examples.len() < 12 {
                examples.push(json!({
                    "keep": { "id": canonical.id, "kind": canonical.kind, "project": canonical.project,
                              "updated_at": canonical.updated_at, "content": snippet(&canonical.content) },
                    "remove": duplicates.iter().map(|duplicate| json!({
                        "id": duplicate.id,
                        "updated_at": duplicate.updated_at,
                        "content": snippet(&duplicate.content),
                    })).collect::<Vec<_>>(),
                }));
            }
            if apply {
                self.fold_cluster(canonical, &duplicates)?;
            }
            removed += duplicates.len();
        }

        Ok(json!({
            "threshold": threshold,
            "applied": apply,
            "scanned": candidates.len(),
            "clusters": clusters.len(),
            "memories_removed": removed,
            "examples": examples,
        }))
    }

    /// Merge `duplicates` into `canonical`: union of tags, max importance,
    /// summed access counts, earliest creation date, links re-pointed,
    /// then the duplicate rows (and their FTS / ANN entries) are deleted.
    fn fold_cluster(&self, canonical: &Candidate, duplicates: &[&Candidate]) -> Result<(), String> {
        let mut tags = canonical.tags.clone();
        let mut importance = canonical.importance;
        let mut access_count = canonical.access_count;
        let mut created_at = canonical.created_at.clone();
        for duplicate in duplicates {
            for tag in &duplicate.tags {
                if !tags.contains(tag) {
                    tags.push(tag.clone());
                }
            }
            importance = importance.max(duplicate.importance);
            access_count += duplicate.access_count;
            if duplicate.created_at < created_at {
                created_at = duplicate.created_at.clone();
            }
        }
        let tags_json = serde_json::to_string(&tags).unwrap_or_else(|_| "[]".into());
        self.conn
            .execute(
                "UPDATE memories SET tags=?1, importance=?2, access_count=?3, created_at=?4 WHERE id=?5",
                params![tags_json, importance, access_count, created_at, canonical.id],
            )
            .map_err(|error| format!("consolidate update: {}", error))?;

        for duplicate in duplicates {
            // Re-point graph edges so context gathered around the duplicate
            // survives; edges that would collide or self-loop are dropped.
            let _ = self.conn.execute(
                "UPDATE OR IGNORE memory_links SET source_id=?1 WHERE source_id=?2 AND target_id<>?1",
                params![canonical.id, duplicate.id],
            );
            let _ = self.conn.execute(
                "UPDATE OR IGNORE memory_links SET target_id=?1 WHERE target_id=?2 AND source_id<>?1",
                params![canonical.id, duplicate.id],
            );
            let _ = self.conn.execute(
                "DELETE FROM memory_links WHERE source_id=?1 OR target_id=?1",
                params![duplicate.id],
            );
            let _ = self.conn.execute(
                "INSERT OR IGNORE INTO memory_entities (memory_id, entity_kind, entity_value, valid_from, valid_to)
                 SELECT ?1, entity_kind, entity_value, valid_from, valid_to FROM memory_entities WHERE memory_id=?2",
                params![canonical.id, duplicate.id],
            );
            self.delete_memory(&duplicate.id)?;
        }
        Ok(())
    }

    /// Link every memory that a newer one restates with a different value
    /// (`supersedes` edge, newer → older). The ranker then keeps the older
    /// statement retrievable but below its replacement — the local answer
    /// to "what is my current X?" when the store holds two Xs.
    ///
    /// A pair qualifies when both live in the same project, sit within
    /// [`SUPERSEDE_COSINE`] of each other, share enough words to be about
    /// the same thing yet not so many that they are the same statement
    /// (that band belongs to consolidation), and are at least
    /// [`SUPERSEDE_MIN_GAP_SECS`] apart. Idempotent; returns the number
    /// of edges written.
    pub fn link_superseded_memories(&self, project: Option<&str>) -> Result<Value, String> {
        let candidates = self.supersede_candidates(project)?;
        let mut by_project: HashMap<&str, Vec<usize>> = HashMap::new();
        for (index, candidate) in candidates.iter().enumerate() {
            by_project
                .entry(candidate.project.as_str())
                .or_default()
                .push(index);
        }

        let mut pairs: Vec<(usize, usize, f32)> = Vec::new();
        for members in by_project.values() {
            for (position, &a) in members.iter().enumerate() {
                for &b in &members[position + 1..] {
                    let (older, newer) = if candidates[a].created_at <= candidates[b].created_at {
                        (a, b)
                    } else {
                        (b, a)
                    };
                    if seconds_between(&candidates[older].created_at, &candidates[newer].created_at)
                        < SUPERSEDE_MIN_GAP_SECS
                    {
                        continue;
                    }
                    let cosine = crate::embedding::cosine_similarity(
                        &candidates[a].vector,
                        &candidates[b].vector,
                    );
                    if cosine < SUPERSEDE_COSINE {
                        continue;
                    }
                    let overlap =
                        Self::similarity(&candidates[a].normalized, &candidates[b].normalized);
                    if !(SUPERSEDE_MIN_OVERLAP..SUPERSEDE_MAX_OVERLAP).contains(&overlap) {
                        continue;
                    }
                    pairs.push((newer, older, cosine));
                }
            }
        }

        let now = chrono::Utc::now().to_rfc3339();
        let mut written = 0usize;
        let mut examples = Vec::new();
        for (newer, older, cosine) in &pairs {
            let changed = self
                .conn
                .execute(
                    "INSERT INTO memory_links (source_id, target_id, relation_type, confidence, created_at)
                     VALUES (?1, ?2, 'supersedes', ?3, ?4)
                     ON CONFLICT(source_id, target_id) DO UPDATE SET relation_type = 'supersedes', confidence = ?3
                     WHERE memory_links.relation_type IN ('relates_to', 'shares_topic', 'same_agent', 'same_origin')",
                    params![candidates[*newer].id, candidates[*older].id, *cosine as f64, now],
                )
                .map_err(|error| format!("supersede link: {}", error))?;
            written += changed;
            if examples.len() < 12 && changed > 0 {
                examples.push(json!({
                    "newer": { "id": candidates[*newer].id, "content": snippet(&candidates[*newer].content) },
                    "older": { "id": candidates[*older].id, "content": snippet(&candidates[*older].content) },
                    "cosine": ((*cosine as f64) * 1000.0).round() / 1000.0,
                }));
            }
        }

        Ok(json!({
            "scanned": candidates.len(),
            "pairs": pairs.len(),
            "links_written": written,
            "examples": examples,
        }))
    }

    fn supersede_candidates(&self, project: Option<&str>) -> Result<Vec<Candidate>, String> {
        let canonical_project = Self::canonical_project(project);
        let mut sql = String::from(
            "SELECT id, project, kind, tags, importance, access_count, created_at, updated_at, content, embedding
             FROM memories
             WHERE embedding IS NOT NULL AND kind NOT IN ('credential') AND tags NOT LIKE '%pinned%'",
        );
        let mut params_values: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
        if let Some(p) = canonical_project.as_deref() {
            sql.push_str(" AND project = ?1");
            params_values.push(Box::new(p.to_string()));
        }
        let mut stmt = self
            .conn
            .prepare(&sql)
            .map_err(|error| format!("supersede prepare: {}", error))?;
        let refs: Vec<&dyn rusqlite::types::ToSql> =
            params_values.iter().map(|p| p.as_ref()).collect();
        let rows = stmt
            .query_map(refs.as_slice(), |row| {
                let tags: String = row.get(3)?;
                let blob: Vec<u8> = row.get(9)?;
                let content: String = row.get(8)?;
                Ok(Candidate {
                    id: row.get(0)?,
                    project: row.get::<_, Option<String>>(1)?.unwrap_or_default(),
                    kind: row.get(2)?,
                    tags: serde_json::from_str(&tags).unwrap_or_default(),
                    importance: row.get(4)?,
                    access_count: row.get(5)?,
                    created_at: row.get(6)?,
                    updated_at: row.get(7)?,
                    normalized: Self::normalize(&content),
                    content,
                    vector: crate::embedding::blob_to_vec(&blob),
                })
            })
            .map_err(|error| format!("supersede query: {}", error))?;
        let expected_dim = crate::embedding::vector_dim();
        Ok(rows
            .flatten()
            .filter(|candidate| candidate.vector.len() == expected_dim)
            .collect())
    }

    /// Daily automatic pass at `AUTO_THRESHOLD`, throttled through the
    /// `config` table so it survives restarts. Called from the write path.
    pub(super) fn maybe_consolidate(&self) {
        if std::env::var("MEMORYPILOT_CONSOLIDATE")
            .map(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "off" | "0" | "false"))
            .unwrap_or(false)
        {
            return;
        }
        let now = chrono::Utc::now().timestamp();
        let last: i64 = self
            .conn
            .query_row(
                "SELECT value FROM config WHERE key=?1",
                params![AUTO_LAST_RUN_KEY],
                |row| row.get::<_, String>(0),
            )
            .ok()
            .and_then(|raw| raw.parse().ok())
            .unwrap_or(0);
        if now - last < AUTO_INTERVAL_SECS {
            return;
        }
        let _ = self.conn.execute(
            "INSERT OR REPLACE INTO config (key, value) VALUES (?1, ?2)",
            params![AUTO_LAST_RUN_KEY, now.to_string()],
        );
        match self.consolidate_memories(AUTO_THRESHOLD, true, None) {
            Ok(report) => {
                let removed = report["memories_removed"].as_u64().unwrap_or(0);
                if removed > 0 {
                    eprintln!(
                        "[MemoryPilot] Consolidated {} near-duplicate memories into {} newer ones (cosine ≥ {}).",
                        removed,
                        report["clusters"].as_u64().unwrap_or(0),
                        AUTO_THRESHOLD
                    );
                }
            }
            Err(error) => eprintln!("[MemoryPilot] consolidation skipped: {}", error),
        }
        match self.link_superseded_memories(None) {
            Ok(report) => {
                let written = report["links_written"].as_u64().unwrap_or(0);
                if written > 0 {
                    eprintln!(
                        "[MemoryPilot] Linked {} superseded memories below their newer restatement.",
                        written
                    );
                }
            }
            Err(error) => eprintln!("[MemoryPilot] supersede pass skipped: {}", error),
        }
    }
}

/// Cosine from which two memories are considered to be about the same
/// thing. Lower than the consolidation threshold on purpose: a changed
/// value ("Honda Civic" → "Tesla Model 3") moves the vector more than a
/// restatement does.
const SUPERSEDE_COSINE: f32 = 0.86;
/// Word-Jaccard band: enough shared words to be the same subject, not
/// so many that the pair is one statement said twice.
const SUPERSEDE_MIN_OVERLAP: f64 = 0.30;
const SUPERSEDE_MAX_OVERLAP: f64 = 0.85;
/// Two memories written within this gap are one conversation, not an
/// update.
const SUPERSEDE_MIN_GAP_SECS: i64 = 10 * 60;

fn seconds_between(earlier: &str, later: &str) -> i64 {
    let parse = |value: &str| {
        chrono::DateTime::parse_from_rfc3339(value)
            .map(|stamp| stamp.timestamp())
            .ok()
    };
    match (parse(earlier), parse(later)) {
        (Some(a), Some(b)) => b - a,
        _ => i64::MAX,
    }
}

fn snippet(content: &str) -> String {
    let mut out: String = content.chars().take(120).collect();
    if content.chars().count() > 120 {
        out.push('…');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_db() -> (Database, std::path::PathBuf) {
        let dir = std::env::temp_dir().join(format!(
            "mp-consolidate-{}-{}",
            std::process::id(),
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("memory.db");
        (Database::open_at(&path).unwrap(), dir)
    }

    /// Insert straight into the table so the background embed worker
    /// never overwrites the synthetic vector the test relies on.
    fn add(db: &Database, content: &str, kind: &str, tags: &[&str], importance: i32, vector: &[f32]) -> String {
        let id = uuid::Uuid::new_v4().to_string();
        let now = chrono::Utc::now().to_rfc3339();
        let mut unit = vector.to_vec();
        let norm = unit.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in &mut unit {
            *x /= norm;
        }
        unit.resize(crate::embedding::vector_dim(), 0.0);
        let blob = crate::embedding::quantize_to_blob(&unit);
        let tags_json = serde_json::to_string(tags).unwrap();
        db.conn
            .execute(
                "INSERT INTO memories (id,content,kind,project,tags,source,importance,embedding,content_hash,created_at,updated_at,access_count)
                 VALUES (?1,?2,?3,'proj',?4,'test',?5,?6,?7,?8,?8,0)",
                params![id, content, kind, tags_json, importance, blob, crate::db::content_hash(content), now],
            )
            .unwrap();
        id
    }

    #[test]
    fn folds_near_duplicates_into_newest_and_spares_durable_kinds() {
        let (db, dir) = temp_db();
        // Nearly identical vectors for the two status facts and the two decisions.
        let old = add(&db, "App relancée, process actif PID 339", "fact", &["build"], 3, &[1.0, 0.02, 0.0]);
        std::thread::sleep(std::time::Duration::from_millis(5));
        let new = add(&db, "App rebuild + relancée, process actif PID 348", "fact", &["pid"], 4, &[1.0, 0.03, 0.0]);
        let other = add(&db, "Le slogan retenu est « Organisez, planifiez »", "fact", &[], 3, &[0.0, 1.0, 0.0]);
        let decision_a = add(&db, "Décision: on garde Turborepo", "decision", &[], 4, &[0.0, 0.0, 1.0]);
        let decision_b = add(&db, "Décision: Turborepo est conservé", "decision", &[], 4, &[0.0, 0.01, 1.0]);
        // A translation pair: identical meaning, near-identical vector, no shared words.
        let french = add(&db, "Email support est support@docix.io", "fact", &[], 3, &[0.5, 0.5, 0.5]);
        let english = add(&db, "Support address: contact the team at docix", "fact", &[], 3, &[0.5, 0.5, 0.51]);

        let dry = db.consolidate_memories(0.95, false, None).unwrap();
        assert_eq!(dry["clusters"], 1);
        assert_eq!(dry["memories_removed"], 1);
        assert!(db.get_memory(&old).unwrap().is_some(), "dry run deletes nothing");

        let applied = db.consolidate_memories(0.95, true, None).unwrap();
        assert_eq!(applied["memories_removed"], 1);
        assert!(db.get_memory(&old).unwrap().is_none(), "older duplicate removed");
        let kept = db.get_memory(&new).unwrap().expect("newest kept");
        assert_eq!(kept.importance, 4);
        assert!(kept.tags.contains(&"build".to_string()) && kept.tags.contains(&"pid".to_string()));
        assert!(db.get_memory(&decision_a).unwrap().is_some() && db.get_memory(&decision_b).unwrap().is_some());
        assert!(db.get_memory(&other).unwrap().is_some());
        assert!(
            db.get_memory(&french).unwrap().is_some() && db.get_memory(&english).unwrap().is_some(),
            "translations share a vector but not words: both kept"
        );

        let again = db.consolidate_memories(0.95, true, None).unwrap();
        assert_eq!(again["memories_removed"], 0, "idempotent");
        let _ = std::fs::remove_dir_all(dir);
    }

    fn add_at(db: &Database, content: &str, kind: &str, created_at: &str, vector: &[f32]) -> String {
        let id = uuid::Uuid::new_v4().to_string();
        let mut unit = vector.to_vec();
        let norm = unit.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in &mut unit {
            *x /= norm;
        }
        unit.resize(crate::embedding::vector_dim(), 0.0);
        let blob = crate::embedding::quantize_to_blob(&unit);
        db.conn
            .execute(
                "INSERT INTO memories (id,content,kind,project,tags,source,importance,embedding,content_hash,created_at,updated_at,access_count)
                 VALUES (?1,?2,?3,'proj','[]','test',3,?4,?5,?6,?6,0)",
                params![id, content, kind, blob, crate::db::content_hash(content), created_at],
            )
            .unwrap();
        id
    }

    #[test]
    fn links_a_changed_value_below_its_newer_restatement() {
        let (db, dir) = temp_db();
        let old_car = add_at(&db, "I drive a Honda Civic to work", "fact", "2026-01-10T09:00:00+00:00", &[1.0, 0.3, 0.0]);
        let new_car = add_at(&db, "I now drive a Tesla Model 3 to work", "fact", "2026-06-10T09:00:00+00:00", &[1.0, 0.0, 0.3]);
        // Same conversation, minutes apart: not an update.
        let same_a = add_at(&db, "Deploy uses the docix Cloudflare project", "fact", "2026-03-01T10:00:00+00:00", &[0.0, 1.0, 0.2]);
        let same_b = add_at(&db, "Deploy target: docix Cloudflare project only", "fact", "2026-03-01T10:04:00+00:00", &[0.0, 1.0, 0.25]);
        // Unrelated words, close vector: a translation, not an update.
        let french = add_at(&db, "Le support répond en moins d'une heure", "fact", "2026-02-01T10:00:00+00:00", &[0.3, 0.3, 1.0]);
        let english = add_at(&db, "Support replies within one hour", "fact", "2026-04-01T10:00:00+00:00", &[0.3, 0.3, 1.05]);

        let report = db.link_superseded_memories(None).unwrap();
        assert_eq!(report["links_written"], 1, "{}", report);

        let relation: String = db
            .conn
            .query_row(
                "SELECT relation_type FROM memory_links WHERE source_id=?1 AND target_id=?2",
                params![new_car, old_car],
                |row| row.get(0),
            )
            .expect("newer supersedes older");
        assert_eq!(relation, "supersedes");
        let count = |a: &str, b: &str| -> i64 {
            db.conn
                .query_row(
                    "SELECT COUNT(*) FROM memory_links WHERE (source_id=?1 AND target_id=?2) OR (source_id=?2 AND target_id=?1)",
                    params![a, b],
                    |row| row.get(0),
                )
                .unwrap()
        };
        assert_eq!(count(&same_a, &same_b), 0, "minutes apart is one conversation");
        assert_eq!(count(&french, &english), 0, "translations share no words");

        let again = db.link_superseded_memories(None).unwrap();
        assert_eq!(again["links_written"], 0, "idempotent");

        // The ranker demotes the superseded row.
        let boosts = db.build_link_boosts_for(&[&old_car, &new_car]);
        assert!(boosts.get(&old_car).copied().unwrap_or(0.0) < 0.0);
        assert!(boosts.get(&new_car).is_none());
        let _ = std::fs::remove_dir_all(dir);
    }
}
