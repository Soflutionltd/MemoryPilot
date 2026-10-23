//! Contradiction finder: memories of the same project that talk about the
//! same thing (high cosine) but disagree (negation on one side only,
//! different values, or an explicit "changed / instead / no longer").
//!
//! Local and deterministic. It only reports: the user decides which one
//! wins, then `delete_memory` / `update_memory` (or `kg_invalidate`).

use std::collections::{BTreeMap, HashSet};

use rusqlite::params;
use serde_json::{json, Value};

use super::Database;

const SKIP_KINDS: &[&str] = &["transcript", "credential", "skill", "snippet"];

/// Strong negations only: "ne ... aucune erreur" / "0 erreur" must not
/// count as disagreement.
const NEGATIONS: &[&str] = &[
    " pas ",
    " jamais ",
    " not ",
    " never ",
    "n't ",
    "faux",
    "wrong",
    "incorrect",
    "interdit",
    "forbidden",
    "à éviter",
    " avoid ",
    " ne plus ",
    " no longer ",
];

/// Durable statements. Milestones / bugs / todos are logs where a newer
/// entry naturally follows an older one; `all_kinds` includes them.
const DURABLE_KINDS: &[&str] = &[
    "decision",
    "preference",
    "fact",
    "architecture",
    "pattern",
    "note",
    "technical",
];

const CHANGE_MARKERS: &[&str] = &[
    "désormais",
    "dorénavant",
    "n'est plus",
    "ne plus",
    "plus jamais",
    "no longer",
    "instead",
    "au lieu",
    "remplac",
    "replaced",
    "deprecated",
    "obsolète",
    "obsolete",
    "abandonn",
    "switched to",
    "migré",
    "migrated",
    "previously",
    "auparavant",
    "avant c'était",
    "changed to",
    "passé à",
    "passe à",
    "now uses",
    "utilise maintenant",
];

struct Item {
    id: String,
    kind: String,
    content: String,
    created_at: String,
    vector: Vec<f32>,
}

pub(super) fn lower_padded(text: &str) -> String {
    format!(
        " {} ",
        text.to_lowercase()
            .replace(['\n', '\t', ',', '.', ';', ':', '(', ')'], " ")
    )
}

fn negation_count(lower: &str) -> usize {
    NEGATIONS.iter().filter(|m| lower.contains(*m)).count()
}

fn has_change_marker(lower: &str) -> bool {
    CHANGE_MARKERS.iter().any(|m| lower.contains(*m))
}

/// Value-like tokens: numbers (ports, versions, sizes, prices), ignoring
/// dates, years and single digits which differ for unrelated reasons.
pub(super) fn values(text: &str) -> HashSet<String> {
    text.split(|c: char| c.is_whitespace() || matches!(c, ',' | ';' | '(' | ')' | '"' | '\'' | '`'))
        .map(|t| t.trim_matches(|c: char| !c.is_alphanumeric()))
        .filter(|t| t.chars().any(|c| c.is_ascii_digit()))
        .filter(|t| {
            let digits: String = t.chars().filter(|c| c.is_ascii_digit()).collect();
            if digits.len() < 2 || t.len() > 16 {
                return false;
            }
            // dates / years / times
            if t.contains('-') && digits.len() >= 6 {
                return false;
            }
            if let Ok(n) = t.parse::<u32>() {
                if (2000..=2099).contains(&n) {
                    return false;
                }
            }
            !t.contains('h')
                || !t
                    .chars()
                    .next()
                    .map(|c| c.is_ascii_digit())
                    .unwrap_or(false)
        })
        .map(|t| t.to_lowercase())
        .collect()
}

pub(super) fn clip(text: &str, max: usize) -> String {
    let t: String = text.chars().take(max).collect();
    if text.chars().count() > max {
        format!("{}…", t)
    } else {
        t
    }
}

impl Database {
    /// Report likely contradictions. `threshold` is the minimum cosine
    /// (default 0.82); pairs above 0.985 are left to `dedupe_report`.
    pub fn find_conflicts(
        &self,
        project: Option<&str>,
        threshold: f32,
        limit: usize,
        all_kinds: bool,
    ) -> Result<Value, String> {
        let now = chrono::Utc::now().to_rfc3339();
        let skip = SKIP_KINDS
            .iter()
            .map(|k| format!("'{}'", k))
            .collect::<Vec<_>>()
            .join(",");
        let mut sql = format!(
            "SELECT id, kind, ifnull(project,''), content, created_at, embedding FROM memories
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
            .map_err(|e| format!("conflicts prepare: {}", e))?;
        let map_row = |r: &rusqlite::Row| -> rusqlite::Result<(String, Item)> {
            let blob: Vec<u8> = r.get(5)?;
            Ok((
                r.get(2)?,
                Item {
                    id: r.get(0)?,
                    kind: r.get(1)?,
                    content: r.get(3)?,
                    created_at: r.get(4)?,
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
        let dim = crate::embedding::vector_dim();
        let mut by_project: BTreeMap<String, Vec<Item>> = BTreeMap::new();
        for (p, item) in rows {
            if item.vector.len() == dim
                && (all_kinds || DURABLE_KINDS.contains(&item.kind.as_str()))
            {
                by_project.entry(p).or_default().push(item);
            }
        }

        let threshold = threshold.clamp(0.5, 0.98);
        let mut pairs: Vec<(f32, Value)> = Vec::new();
        let mut scanned = 0usize;
        for (proj, items) in &by_project {
            let lowers: Vec<String> = items.iter().map(|i| lower_padded(&i.content)).collect();
            let negs: Vec<usize> = lowers.iter().map(|l| negation_count(l)).collect();
            let vals: Vec<HashSet<String>> = items.iter().map(|i| values(&i.content)).collect();
            scanned += items.len();
            for a in 0..items.len() {
                for b in (a + 1)..items.len() {
                    let cos =
                        crate::embedding::cosine_similarity(&items[a].vector, &items[b].vector);
                    if cos < threshold || cos > 0.985 {
                        continue;
                    }
                    let mut signals: Vec<&str> = Vec::new();
                    if (negs[a] == 0) != (negs[b] == 0) {
                        signals.push("negation on one side");
                    }
                    if !vals[a].is_empty() && !vals[b].is_empty() && vals[a].is_disjoint(&vals[b]) {
                        signals.push("different values");
                    }
                    let (old, new) = if items[a].created_at <= items[b].created_at {
                        (a, b)
                    } else {
                        (b, a)
                    };
                    if has_change_marker(&lowers[new]) {
                        signals.push("newer one says it changed");
                    }
                    if signals.is_empty() {
                        continue;
                    }
                    let weight = |s: &&str| match *s {
                        "newer one says it changed" => 0.06,
                        "negation on one side" => 0.05,
                        _ => 0.02,
                    };
                    let score = cos + signals.iter().map(weight).sum::<f32>();
                    pairs.push((
                        score,
                        json!({
                            "project": if proj.is_empty() { None } else { Some(proj) },
                            "similarity": (cos * 1000.0).round() / 1000.0,
                            "signals": signals,
                            "older": {
                                "id": items[old].id, "kind": items[old].kind,
                                "created_at": items[old].created_at,
                                "content": clip(&items[old].content, 300),
                            },
                            "newer": {
                                "id": items[new].id, "kind": items[new].kind,
                                "created_at": items[new].created_at,
                                "content": clip(&items[new].content, 300),
                            },
                            "hint": "If the newer one is right: delete_memory(older.id). Otherwise delete or update the newer one.",
                        }),
                    ));
                }
            }
        }
        pairs.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap_or(std::cmp::Ordering::Equal));
        let total = pairs.len();
        let conflicts: Vec<Value> = pairs.into_iter().take(limit.max(1)).map(|p| p.1).collect();
        Ok(json!({
            "scanned": scanned,
            "threshold": threshold,
            "total_pairs": total,
            "conflicts": conflicts,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values_skip_dates_and_years() {
        let v = values("port 3000 depuis le 2026-05-12 en 2026, v4.8");
        assert!(v.contains("3000"));
        assert!(!v.contains("2026"));
        assert!(!v.contains("2026-05-12"));
        assert!(v.contains("v4.8") || v.contains("4.8"));
    }

    #[test]
    fn negation_detection() {
        assert!(negation_count(&lower_padded("Ne pas utiliser Homebrew")) > 0);
        assert_eq!(negation_count(&lower_padded("Utiliser Homebrew")), 0);
    }
}
