//! Automatic supersession: when a newer memory replaces an older one, the
//! older one is tagged `superseded` (metadata.superseded_by = newer id,
//! importance 1). Nothing is deleted: `get_memory` still returns it, and
//! `unsupersede` undoes it, but recall and search stop surfacing it.
//!
//! Precision first (it runs unattended), two rules on the same project:
//! A. explicit replacement: cosine >= 0.86 and the newer one says
//!    "X instead of Y" / "au lieu de Y" where Y is in the older one;
//!    or a bare change marker ("désormais", "no longer"...) on a
//!    near-identical pair (cosine >= 0.95);
//! B. status update: same kind (milestone / fact), cosine >= 0.93, same
//!    subject words, but the values (build numbers, versions, sizes)
//!    are all different.

use std::collections::{BTreeMap, HashSet};

use rusqlite::params;
use serde_json::{json, Value};

use super::conflicts::{clip, lower_padded, values};
use super::Database;

pub(crate) const SUPERSEDED_TAG: &str = "superseded";
const AUTO_LAST_RUN_KEY: &str = "supersede_last_run";
const AUTO_INTERVAL_SECS: i64 = 6 * 3600;

const SKIP_KINDS: &[&str] = &["transcript", "credential", "skill", "snippet", "todo"];
const STATUS_KINDS: &[&str] = &["milestone", "fact"];
const DURABLE_KINDS: &[&str] = &[
    "decision",
    "preference",
    "fact",
    "architecture",
    "pattern",
    "note",
    "technical",
];

/// Markers naming WHAT was replaced ("X au lieu de Y"): the old memory
/// must contain Y for the rule to fire.
const OBJECT_MARKERS: &[&str] = &[
    "au lieu de ",
    "au lieu du ",
    "au lieu des ",
    "au lieu d'",
    "instead of ",
];

/// Bare "this changed" markers: only trusted on near-identical pairs.
const CHANGE_MARKERS: &[&str] = &[
    "désormais",
    "dorénavant",
    "n'est plus",
    " ne plus ",
    "no longer",
    "remplacé par",
    "replaced by",
    "deprecated",
    "obsolète",
    "obsolete",
    "abandonné",
    "switched to",
    "migré vers",
    "migrated to",
    "previously",
    "auparavant",
    "remis comme avant",
    "reverted",
    "changed to",
    "passé à",
    "passe à",
];
const BARE_MARKER_MIN_COSINE: f32 = 0.95;

/// The thing the newer memory says was replaced, e.g. "*0.60" in
/// "low=0.30 au lieu de *0.60".
fn replaced_objects(lower: &str) -> Vec<String> {
    let mut out = Vec::new();
    for marker in OBJECT_MARKERS {
        let mut rest = lower;
        while let Some(pos) = rest.find(marker) {
            let after = &rest[pos + marker.len()..];
            let token: String = after
                .split_whitespace()
                .map(|w| {
                    w.trim_matches(|c: char| {
                        matches!(c, ',' | '.' | ';' | ':' | ')' | '(' | '"' | '\'' | '`')
                    })
                })
                .filter(|w| {
                    !matches!(
                        *w,
                        "la" | "le"
                            | "les"
                            | "l"
                            | "un"
                            | "une"
                            | "the"
                            | "a"
                            | "an"
                            | "ce"
                            | "cet"
                            | "cette"
                    )
                })
                .find(|w| w.chars().count() >= 3)
                .unwrap_or_default()
                .to_string();
            // Only distinctive objects ("*0.60", "garde-fou", "iscalculatingsize"):
            // a plain word like "rester" or "qualité" matches too much.
            let distinctive =
                token.chars().any(|c| !c.is_alphabetic()) || token.chars().count() >= 8;
            if !token.is_empty() && distinctive {
                out.push(token);
            }
            rest = after;
        }
    }
    out
}

struct Item {
    id: String,
    kind: String,
    content: String,
    created_at: String,
    tags: String,
    vector: Vec<f32>,
}

fn subject_words(lower: &str) -> HashSet<String> {
    lower
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.chars().count() >= 4 && !w.chars().any(|c| c.is_ascii_digit()))
        .take(24)
        .map(String::from)
        .collect()
}

fn jaccard(a: &HashSet<String>, b: &HashSet<String>) -> f64 {
    let union = a.union(b).count();
    if union == 0 {
        return 0.0;
    }
    a.intersection(b).count() as f64 / union as f64
}

fn compatible_kinds(old: &str, new: &str) -> bool {
    old == new || (DURABLE_KINDS.contains(&old) && DURABLE_KINDS.contains(&new))
}

impl Database {
    /// Find (and with `apply`, tag) memories replaced by a newer one.
    pub fn supersede_outdated(
        &self,
        project: Option<&str>,
        apply: bool,
        limit: usize,
    ) -> Result<Value, String> {
        let now = chrono::Utc::now().to_rfc3339();
        let skip = SKIP_KINDS
            .iter()
            .map(|k| format!("'{}'", k))
            .collect::<Vec<_>>()
            .join(",");
        let mut sql = format!(
            "SELECT id, kind, ifnull(project,''), content, created_at, tags, embedding FROM memories
             WHERE embedding IS NOT NULL AND kind NOT IN ({})
               AND (expires_at IS NULL OR expires_at > ?1)",
            skip
        );
        let canonical = project.and_then(Self::canonical_project_name);
        if canonical.is_some() {
            sql.push_str(" AND project = ?2");
        }
        let mut stmt = self
            .conn
            .prepare(&sql)
            .map_err(|e| format!("supersede prepare: {}", e))?;
        let map_row = |r: &rusqlite::Row| -> rusqlite::Result<(String, Item)> {
            let blob: Vec<u8> = r.get(6)?;
            Ok((
                r.get(2)?,
                Item {
                    id: r.get(0)?,
                    kind: r.get(1)?,
                    content: r.get(3)?,
                    created_at: r.get(4)?,
                    tags: r.get(5)?,
                    vector: crate::embedding::blob_to_vec(&blob),
                },
            ))
        };
        let rows: Vec<(String, Item)> = match canonical.as_deref() {
            Some(p) => stmt
                .query_map(params![now, p], map_row)
                .map_err(|e| e.to_string())?
                .flatten()
                .collect(),
            None => stmt
                .query_map(params![now], map_row)
                .map_err(|e| e.to_string())?
                .flatten()
                .collect(),
        };
        drop(stmt);
        let dim = crate::embedding::vector_dim();
        let mut by_project: BTreeMap<String, Vec<Item>> = BTreeMap::new();
        for (p, item) in rows {
            if item.vector.len() == dim && !item.tags.contains(SUPERSEDED_TAG) {
                by_project.entry(p).or_default().push(item);
            }
        }

        // For each old memory keep the best (newest strong) replacement.
        let mut found: Vec<(String, String, &'static str, f32, String, String, String)> =
            Vec::new();
        for items in by_project.values() {
            let lowers: Vec<String> = items.iter().map(|i| lower_padded(&i.content)).collect();
            let vals: Vec<HashSet<String>> = items.iter().map(|i| values(&i.content)).collect();
            let subj: Vec<HashSet<String>> = lowers.iter().map(|l| subject_words(l)).collect();
            let change: Vec<bool> = lowers
                .iter()
                .map(|l| CHANGE_MARKERS.iter().any(|m| l.contains(m)))
                .collect();
            let objects: Vec<Vec<String>> = lowers.iter().map(|l| replaced_objects(l)).collect();
            let mut taken: HashSet<usize> = HashSet::new();
            for old in 0..items.len() {
                if items[old].tags.contains("pinned") {
                    continue;
                }
                let mut best: Option<(usize, &'static str, f32)> = None;
                for new in 0..items.len() {
                    if new == old || items[new].created_at <= items[old].created_at {
                        continue;
                    }
                    if !compatible_kinds(&items[old].kind, &items[new].kind) {
                        continue;
                    }
                    let cos =
                        crate::embedding::cosine_similarity(&items[old].vector, &items[new].vector);
                    if cos < 0.86 || cos > 0.995 {
                        continue;
                    }
                    let names_old = objects[new]
                        .iter()
                        .any(|o| lowers[old].contains(o.as_str()))
                        && !objects[new]
                            .iter()
                            .any(|o| lowers[new].matches(o.as_str()).count() > 1);
                    let rule = if names_old {
                        Some("explicit replacement")
                    } else if change[new] && !change[old] && cos >= BARE_MARKER_MIN_COSINE {
                        Some("explicit change")
                    } else if cos >= 0.93
                        && items[old].kind == items[new].kind
                        && STATUS_KINDS.contains(&items[old].kind.as_str())
                        && !vals[old].is_empty()
                        && !vals[new].is_empty()
                        && vals[old].is_disjoint(&vals[new])
                        && jaccard(&subj[old], &subj[new]) >= 0.5
                    {
                        Some("status update")
                    } else {
                        None
                    };
                    if let Some(rule) = rule {
                        let better = match best {
                            None => true,
                            Some((b, _, bcos)) => {
                                cos > bcos + 0.02
                                    || (cos + 0.02 >= bcos
                                        && items[new].created_at > items[b].created_at)
                            }
                        };
                        if better {
                            best = Some((new, rule, cos));
                        }
                    }
                }
                if let Some((new, rule, cos)) = best {
                    if taken.insert(old) {
                        found.push((
                            items[old].id.clone(),
                            items[new].id.clone(),
                            rule,
                            cos,
                            clip(&items[old].content, 200),
                            clip(&items[new].content, 200),
                            items[old].kind.clone(),
                        ));
                    }
                }
            }
        }

        let total = found.len();
        if apply {
            for (old_id, new_id, rule, _, _, _, _) in &found {
                self.mark_superseded(old_id, new_id, rule)?;
            }
        }
        let samples: Vec<Value> = found
            .iter()
            .take(limit.max(1))
            .map(|(o, n, rule, cos, oc, nc, kind)| {
                json!({
                    "old": { "id": o, "kind": kind, "content": oc },
                    "new": { "id": n, "content": nc },
                    "rule": rule,
                    "similarity": (cos * 1000.0).round() / 1000.0,
                })
            })
            .collect();
        Ok(json!({ "applied": apply, "superseded": total, "pairs": samples }))
    }

    fn mark_superseded(&self, old_id: &str, new_id: &str, rule: &str) -> Result<(), String> {
        let Some(memory) = self.get_memory(old_id)? else {
            return Ok(());
        };
        let mut tags = memory.tags.clone();
        if !tags.iter().any(|t| t == SUPERSEDED_TAG) {
            tags.push(SUPERSEDED_TAG.to_string());
        }
        let mut meta = memory.metadata.clone().unwrap_or_else(|| json!({}));
        if !meta.is_object() {
            meta = json!({ "value": meta });
        }
        meta["superseded_by"] = json!(new_id);
        meta["superseded_rule"] = json!(rule);
        meta["superseded_at"] = json!(chrono::Utc::now().to_rfc3339());
        meta["importance_before_supersede"] = json!(memory.importance);
        self.update_memory_full(old_id, None, None, Some(&tags), Some(1), None, Some(&meta))?;
        Ok(())
    }

    /// Undo a supersession (restores tags and importance).
    pub fn unsupersede(&self, id: &str) -> Result<bool, String> {
        let Some(memory) = self.get_memory(id)? else {
            return Ok(false);
        };
        if !memory.tags.iter().any(|t| t == SUPERSEDED_TAG) {
            return Ok(false);
        }
        let tags: Vec<String> = memory
            .tags
            .iter()
            .filter(|t| t.as_str() != SUPERSEDED_TAG)
            .cloned()
            .collect();
        let mut meta = memory.metadata.clone().unwrap_or_else(|| json!({}));
        let importance = meta
            .get("importance_before_supersede")
            .and_then(|v| v.as_i64())
            .unwrap_or(3) as i32;
        if let Some(obj) = meta.as_object_mut() {
            for key in [
                "superseded_by",
                "superseded_rule",
                "superseded_at",
                "importance_before_supersede",
            ] {
                obj.remove(key);
            }
            obj.insert("supersede_rejected".into(), json!(true));
        }
        self.update_memory_full(
            id,
            None,
            None,
            Some(&tags),
            Some(importance),
            None,
            Some(&meta),
        )?;
        Ok(true)
    }

    /// Throttled automatic pass (every 6 h, MEMORYPILOT_SUPERSEDE=off).
    pub(super) fn maybe_supersede(&self) {
        if std::env::var("MEMORYPILOT_SUPERSEDE")
            .map(|v| {
                matches!(
                    v.trim().to_ascii_lowercase().as_str(),
                    "off" | "0" | "false"
                )
            })
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
        if let Ok(report) = self.supersede_outdated(None, true, 1) {
            let n = report["superseded"].as_u64().unwrap_or(0);
            if n > 0 {
                eprintln!(
                    "[MemoryPilot] Marked {} outdated memories as superseded.",
                    n
                );
            }
        }
    }

    /// Keep the FTS index in step with `memories`: drop rows whose memory
    /// is gone, index memories that have no row.
    pub fn repair_fts(&self) -> Result<(usize, usize), String> {
        let removed = self
            .conn
            .execute(
                "DELETE FROM memories_fts WHERE rowid NOT IN (SELECT rowid FROM memories)",
                [],
            )
            .map_err(|e| format!("fts orphans: {}", e))?;
        let missing: Vec<(i64, String, String, String, Option<String>)> = {
            let mut stmt = self
                .conn
                .prepare(
                    "SELECT rowid, content, tags, kind, project FROM memories
                     WHERE rowid NOT IN (SELECT rowid FROM memories_fts)",
                )
                .map_err(|e| format!("fts missing: {}", e))?;
            let rows = stmt
                .query_map([], |r| {
                    Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?))
                })
                .map_err(|e| format!("fts missing: {}", e))?;
            rows.flatten().collect()
        };
        for (rowid, content, tags, kind, project) in &missing {
            self.conn
                .execute(
                    "INSERT INTO memories_fts (rowid,content,tags,kind,project) VALUES (?1,?2,?3,?4,?5)",
                    params![rowid, Self::fts_index_content(content), tags, kind, project.as_deref().unwrap_or("")],
                )
                .map_err(|e| format!("fts insert: {}", e))?;
        }
        Ok((removed, missing.len()))
    }
}

pub(crate) fn is_superseded(tags: &[String]) -> bool {
    tags.iter().any(|t| t == SUPERSEDED_TAG)
}
