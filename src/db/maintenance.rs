//! Maintenance: expiry, GC, capsules, bulk delete, health and dedupe reports, embedding backfill, stats and config.

use super::*;

impl Database {
    pub(super) fn hygiene_report(&self) -> crate::gc::HygieneReport {
        let stale_threshold = (Utc::now() - chrono::Duration::days(30)).to_rfc3339();
        crate::gc::HygieneReport {
            projects_missing_path: self.conn.query_row(
                "SELECT COUNT(*) FROM projects WHERE trim(path) = ''",
                [],
                |row| row.get(0),
            ).unwrap_or(0),
            memory_project_mismatches: self.conn.query_row(
                "SELECT COUNT(*) FROM memories m WHERE m.project IS NOT NULL AND NOT EXISTS (SELECT 1 FROM projects p WHERE p.name = m.project)",
                [],
                |row| row.get(0),
            ).unwrap_or(0),
            never_accessed_memories: self.conn.query_row(
                "SELECT COUNT(*) FROM memories WHERE access_count = 0",
                [],
                |row| row.get(0),
            ).unwrap_or(0),
            stale_low_value_memories: self.conn.query_row(
                "SELECT COUNT(*) FROM memories WHERE importance <= 2 AND access_count = 0 AND updated_at < ?1",
                params![stale_threshold],
                |row| row.get(0),
            ).unwrap_or(0),
            orphan_entities: self.conn.query_row(
                "SELECT COUNT(*) FROM memory_entities WHERE memory_id NOT IN (SELECT id FROM memories)",
                [],
                |row| row.get(0),
            ).unwrap_or(0),
            orphan_links: self.conn.query_row(
                "SELECT COUNT(*) FROM memory_links WHERE source_id NOT IN (SELECT id FROM memories) OR target_id NOT IN (SELECT id FROM memories)",
                [],
                |row| row.get(0),
            ).unwrap_or(0),
            credential_memories: self.conn.query_row(
                "SELECT COUNT(*) FROM memories WHERE kind = 'credential'",
                [],
                |row| row.get(0),
            ).unwrap_or(0),
            global_memories: self.conn.query_row(
                "SELECT COUNT(*) FROM memories WHERE project IS NULL",
                [],
                |row| row.get(0),
            ).unwrap_or(0),
        }
    }

    // ─── TTL / EXPIRATION ─────────────────────────────

    pub fn cleanup_expired(&self) -> Result<usize, String> {
        static LAST_CLEANUP: OnceLock<Mutex<std::time::Instant>> = OnceLock::new();
        let last = LAST_CLEANUP.get_or_init(|| {
            Mutex::new(std::time::Instant::now() - std::time::Duration::from_secs(120))
        });
        if let Ok(mut ts) = last.lock() {
            if ts.elapsed() < std::time::Duration::from_secs(60) {
                return Ok(0);
            }
            *ts = std::time::Instant::now();
        }
        let now = Utc::now().to_rfc3339();
        let _ = self.conn.execute(
            "DELETE FROM memories_fts WHERE rowid IN (SELECT rowid FROM memories WHERE expires_at IS NOT NULL AND expires_at < ?1)",
            params![now]);
        let affected = self
            .conn
            .execute(
                "DELETE FROM memories WHERE expires_at IS NOT NULL AND expires_at < ?1",
                params![now],
            )
            .map_err(|e| format!("Cleanup: {}", e))?;
        Ok(affected)
    }

    // ─── GC & COMPRESSION ─────────────────────────────

    pub fn run_gc(
        &self,
        config: &crate::gc::GcConfig,
        dry_run: bool,
    ) -> Result<crate::gc::GcReport, String> {
        #[derive(Clone)]
        struct GcCandidate {
            id: String,
            content: String,
            project: Option<String>,
            importance: i32,
            age_days: i64,
            gc_score: f64,
        }

        let db_path = dirs::home_dir()
            .unwrap_or_default()
            .join(DB_DIR)
            .join(DB_FILE);
        let size_before = std::fs::metadata(&db_path).map(|m| m.len()).unwrap_or(0);

        let mut expired_removed = 0;
        if !dry_run {
            expired_removed = self.cleanup_expired()?;
        }

        // Find mergeable candidates
        let now = chrono::Utc::now();
        let mut groups_merged = 0;
        let mut memories_compressed = 0;
        let mut preview_candidates: Vec<crate::gc::GcPreviewCandidate> = Vec::new();

        for kind in &config.compressible_kinds {
            let sql = "SELECT id, content, project, importance, updated_at FROM memories WHERE kind = ?1 AND tags NOT LIKE '%pinned%'";
            if let Ok(mut stmt) = self.conn.prepare(&sql) {
                if let Ok(rows) = stmt.query_map(params![kind], |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, Option<String>>(2)?,
                        r.get::<_, i32>(3)?,
                        r.get::<_, String>(4)?,
                    ))
                }) {
                    let mut by_project: std::collections::HashMap<
                        Option<String>,
                        Vec<GcCandidate>,
                    > = std::collections::HashMap::new();
                    for row in rows.flatten() {
                        let updated_at = chrono::DateTime::parse_from_rfc3339(&row.4)
                            .unwrap_or_else(|_| chrono::Utc::now().into());
                        let age_days = (now - updated_at.with_timezone(&chrono::Utc)).num_days();

                        let score = crate::gc::gc_score(row.3, age_days, kind, config);
                        if score > 0.6
                            && row.3 < config.importance_threshold
                            && age_days >= config.age_days
                        {
                            by_project
                                .entry(row.2.clone())
                                .or_default()
                                .push(GcCandidate {
                                    id: row.0,
                                    content: row.1,
                                    project: row.2,
                                    importance: row.3,
                                    age_days,
                                    gc_score: score,
                                });
                        }
                    }

                    for (proj, mut items) in by_project {
                        if items.len() > 1 {
                            items.truncate(config.max_merge_group);
                            let contents: Vec<String> =
                                items.iter().map(|item| item.content.clone()).collect();
                            let merged_content =
                                crate::gc::merge_memories(&contents, kind, proj.as_deref());

                            let ids_to_delete: Vec<String> =
                                items.iter().map(|item| item.id.clone()).collect();
                            let gc_score_avg = items.iter().map(|item| item.gc_score).sum::<f64>()
                                / items.len() as f64;
                            let age_days_min =
                                items.iter().map(|item| item.age_days).min().unwrap_or(0);
                            let age_days_max =
                                items.iter().map(|item| item.age_days).max().unwrap_or(0);
                            let importance_min =
                                items.iter().map(|item| item.importance).min().unwrap_or(0);
                            let importance_max =
                                items.iter().map(|item| item.importance).max().unwrap_or(0);

                            preview_candidates.push(crate::gc::GcPreviewCandidate {
                                kind: kind.clone(),
                                project: proj
                                    .clone()
                                    .or_else(|| items.iter().find_map(|item| item.project.clone())),
                                memory_ids: ids_to_delete.clone(),
                                sample_contents: items
                                    .iter()
                                    .take(3)
                                    .map(|item| Self::preview_snippet(&item.content))
                                    .collect(),
                                confidence_score: (gc_score_avg * 100.0).round() / 100.0,
                                gc_score_avg: (gc_score_avg * 100.0).round() / 100.0,
                                age_days_min,
                                age_days_max,
                                importance_min,
                                importance_max,
                            });

                            if !dry_run {
                                if self
                                    .add_memory(
                                        &merged_content,
                                        kind,
                                        proj.as_deref(),
                                        &["merged".to_string()],
                                        "gc_compressor",
                                        3,
                                        None,
                                        None,
                                        &MemoryScope::default(),
                                    )
                                    .is_ok()
                                {
                                    for id in ids_to_delete {
                                        let _ = self.delete_memory(&id);
                                        memories_compressed += 1;
                                    }
                                    groups_merged += 1;
                                }
                            } else {
                                memories_compressed += ids_to_delete.len();
                                groups_merged += 1;
                            }
                        }
                    }
                }
            }
        }

        let mut orphan_links_removed = 0;
        if !dry_run {
            orphan_links_removed += self
                .conn
                .execute(
                    "DELETE FROM memory_entities WHERE memory_id NOT IN (SELECT id FROM memories)",
                    [],
                )
                .unwrap_or(0);

            orphan_links_removed += self.conn.execute(
                "DELETE FROM memory_links WHERE source_id NOT IN (SELECT id FROM memories) OR target_id NOT IN (SELECT id FROM memories)",
                []
            ).unwrap_or(0);
        }

        let size_after = std::fs::metadata(&db_path).map(|m| m.len()).unwrap_or(0);
        let hygiene = self.hygiene_report();

        Ok(crate::gc::GcReport {
            expired_removed,
            groups_merged,
            memories_compressed,
            orphan_links_removed: orphan_links_removed as usize,
            db_size_before: size_before,
            db_size_after: size_after,
            preview_mode: dry_run,
            preview_candidates,
            hygiene,
        })
    }

    // ─── MEMORY CAPSULES ──────────────────────────────

    /// Compress old low-importance memories into dense capsules.
    /// Groups by project + age, merges into ~100-200 token summaries.
    /// Preserves Knowledge Graph links. Returns (capsules_created, memories_compressed).
    pub fn compact_to_capsules(
        &self,
        age_days: i64,
        importance_max: i32,
    ) -> Result<(usize, usize), String> {
        let now = chrono::Utc::now();
        let sql = "SELECT id, content, kind, project, importance, updated_at FROM memories WHERE kind NOT IN ('credential', 'architecture', 'transcript_chunk') AND importance <= ?1";
        let mut stmt = self
            .conn
            .prepare(sql)
            .map_err(|e| format!("Capsule prepare: {}", e))?;
        let all_rows: Vec<(String, String, String, Option<String>, i32, String)> = stmt
            .query_map(params![importance_max], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                    r.get::<_, Option<String>>(3)?,
                    r.get::<_, i32>(4)?,
                    r.get::<_, String>(5)?,
                ))
            })
            .map_err(|e| format!("Capsule query: {}", e))?
            .filter_map(|r| r.ok())
            .collect();

        let rows: Vec<&(String, String, String, Option<String>, i32, String)> = all_rows
            .iter()
            .filter(|(_id, _content, _kind, _proj, _imp, updated_at)| {
                chrono::DateTime::parse_from_rfc3339(updated_at)
                    .map(|dt| (now - dt.with_timezone(&chrono::Utc)).num_days() >= age_days)
                    .unwrap_or(false)
            })
            .collect();

        if rows.len() < 3 {
            return Ok((0, 0));
        }

        // Group by project
        let mut by_project: std::collections::HashMap<
            Option<String>,
            Vec<(String, String, String)>,
        > = std::collections::HashMap::new();
        for (id, content, kind, project, _imp, _updated) in rows {
            by_project.entry(project.clone()).or_default().push((
                id.clone(),
                content.clone(),
                kind.clone(),
            ));
        }

        let mut capsules_created = 0usize;
        let mut memories_compressed = 0usize;

        for (project, items) in by_project {
            // Process in chunks of 10
            for chunk in items.chunks(10) {
                if chunk.len() < 2 {
                    continue;
                }
                let contents: Vec<String> = chunk.iter().map(|(_, c, _)| c.clone()).collect();
                let kinds: Vec<String> = chunk.iter().map(|(_, _, k)| k.clone()).collect();
                let capsule = crate::gc::capsule_summary(&contents, &kinds, project.as_deref());

                if self
                    .add_memory(
                        &capsule,
                        "note",
                        project.as_deref(),
                        &["capsule".to_string(), "compressed".to_string()],
                        "auto_capsule",
                        3,
                        None,
                        None,
                        &MemoryScope::default(),
                    )
                    .is_ok()
                {
                    for (id, _, _) in chunk {
                        let _ = self.delete_memory(id);
                        memories_compressed += 1;
                    }
                    capsules_created += 1;
                }
            }
        }

        Ok((capsules_created, memories_compressed))
    }

    // ─── BULK DELETE ──────────────────────────────────

    pub fn bulk_delete(
        &self,
        kind: Option<&str>,
        project: Option<&str>,
        tag: Option<&str>,
        older_than_days: Option<i64>,
        importance_max: Option<i32>,
    ) -> Result<usize, String> {
        let canonical_project = Self::canonical_project(project);
        let mut conditions: Vec<String> = Vec::new();
        let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();

        if let Some(k) = kind {
            conditions.push(format!("kind = ?{}", param_values.len() + 1));
            param_values.push(Box::new(k.to_string()));
        }
        if let Some(p) = canonical_project.as_deref() {
            conditions.push(format!("project = ?{}", param_values.len() + 1));
            param_values.push(Box::new(p.to_string()));
        }
        if let Some(imp) = importance_max {
            conditions.push(format!("importance <= ?{}", param_values.len() + 1));
            param_values.push(Box::new(imp));
        }
        if let Some(days) = older_than_days {
            let cutoff = (chrono::Utc::now() - chrono::Duration::days(days)).to_rfc3339();
            conditions.push(format!("updated_at < ?{}", param_values.len() + 1));
            param_values.push(Box::new(cutoff));
        }

        // Never bulk-delete pinned memories
        conditions.push("tags NOT LIKE '%pinned%'".to_string());

        if conditions.is_empty() {
            return Err("At least one filter required".to_string());
        }

        // First collect IDs to delete (for cascading entity/link cleanup)
        let select_sql = format!("SELECT id FROM memories WHERE {}", conditions.join(" AND "));
        let mut stmt = self
            .conn
            .prepare(&select_sql)
            .map_err(|e| format!("Bulk select: {}", e))?;
        let params_ref: Vec<&dyn rusqlite::types::ToSql> =
            param_values.iter().map(|p| p.as_ref()).collect();
        let ids: Vec<String> = stmt
            .query_map(params_ref.as_slice(), |r| r.get::<_, String>(0))
            .map_err(|e| format!("Bulk query: {}", e))?
            .filter_map(|r| r.ok())
            .collect();

        // Filter by tag if specified (needs JSON parsing)
        let ids_to_delete: Vec<String> = if let Some(tag_filter) = tag {
            ids.into_iter()
                .filter(|id| {
                    self.get_memory(id)
                        .ok()
                        .flatten()
                        .map(|m| m.tags.iter().any(|t| t == tag_filter))
                        .unwrap_or(false)
                })
                .collect()
        } else {
            ids
        };

        let mut deleted = 0;
        for id in &ids_to_delete {
            if self.delete_memory(id).unwrap_or(false) {
                deleted += 1;
            }
        }
        Ok(deleted)
    }

    // ─── MEMORY HEALTH REPORT ────────────────────────

    pub fn memory_health_report(&self) -> Result<serde_json::Value, String> {
        use serde_json::json;
        let now = chrono::Utc::now();

        let total: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))
            .unwrap_or(0);

        // By kind
        let mut by_kind: Vec<(String, i64)> = Vec::new();
        if let Ok(mut stmt) = self
            .conn
            .prepare("SELECT kind, COUNT(*) FROM memories GROUP BY kind ORDER BY COUNT(*) DESC")
        {
            if let Ok(rows) =
                stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)))
            {
                by_kind = rows.filter_map(|r| r.ok()).collect();
            }
        }

        // By project
        let mut by_project: Vec<(String, i64)> = Vec::new();
        if let Ok(mut stmt) = self.conn.prepare("SELECT COALESCE(project, '(global)'), COUNT(*) FROM memories GROUP BY project ORDER BY COUNT(*) DESC") {
            if let Ok(rows) = stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?))) {
                by_project = rows.filter_map(|r| r.ok()).collect();
            }
        }

        // By importance
        let mut by_importance: Vec<(i32, i64)> = Vec::new();
        if let Ok(mut stmt) = self.conn.prepare("SELECT importance, COUNT(*) FROM memories GROUP BY importance ORDER BY importance DESC") {
            if let Ok(rows) = stmt.query_map([], |r| Ok((r.get::<_, i32>(0)?, r.get::<_, i64>(1)?))) {
                by_importance = rows.filter_map(|r| r.ok()).collect();
            }
        }

        // Pinned count
        let pinned: i64 = self
            .conn
            .query_row(
                "SELECT COUNT(*) FROM memories WHERE tags LIKE '%pinned%'",
                [],
                |r| r.get(0),
            )
            .unwrap_or(0);

        // Stale memories (> 30 days, importance <= 2)
        let cutoff_30 = (now - chrono::Duration::days(30)).to_rfc3339();
        let stale: i64 = self
            .conn
            .query_row(
                "SELECT COUNT(*) FROM memories WHERE updated_at < ?1 AND importance <= 2",
                params![cutoff_30],
                |r| r.get(0),
            )
            .unwrap_or(0);

        // Compression potential (compressible old memories)
        let compressible: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM memories WHERE updated_at < ?1 AND importance <= 3 AND kind NOT IN ('credential', 'architecture', 'transcript_chunk') AND tags NOT LIKE '%pinned%'",
            params![cutoff_30], |r| r.get(0),
        ).unwrap_or(0);

        // Average age in days
        let avg_age: f64 = self
            .conn
            .query_row(
                "SELECT AVG(julianday('now') - julianday(created_at)) FROM memories",
                [],
                |r| r.get(0),
            )
            .unwrap_or(0.0);

        // Orphan entities and links
        let orphan_entities: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM memory_entities WHERE memory_id NOT IN (SELECT id FROM memories)", [], |r| r.get(0),
        ).unwrap_or(0);
        let orphan_links: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM memory_links WHERE source_id NOT IN (SELECT id FROM memories) OR target_id NOT IN (SELECT id FROM memories)", [], |r| r.get(0),
        ).unwrap_or(0);

        // DB size
        let db_path = dirs::home_dir()
            .unwrap_or_default()
            .join(DB_DIR)
            .join(DB_FILE);
        let db_size = std::fs::metadata(&db_path).map(|m| m.len()).unwrap_or(0);

        // Capsule count
        let capsules: i64 = self
            .conn
            .query_row(
                "SELECT COUNT(*) FROM memories WHERE tags LIKE '%capsule%'",
                [],
                |r| r.get(0),
            )
            .unwrap_or(0);

        Ok(json!({
            "total_memories": total,
            "pinned": pinned,
            "capsules": capsules,
            "stale_low_value": stale,
            "compression_potential": compressible,
            "orphan_entities": orphan_entities,
            "orphan_links": orphan_links,
            "avg_age_days": (avg_age * 10.0).round() / 10.0,
            "db_size_bytes": db_size,
            "db_size_mb": (db_size as f64 / 1_048_576.0 * 100.0).round() / 100.0,
            "by_kind": by_kind.into_iter().map(|(k, c)| json!({"kind": k, "count": c})).collect::<Vec<_>>(),
            "by_project": by_project.into_iter().map(|(p, c)| json!({"project": p, "count": c})).collect::<Vec<_>>(),
            "by_importance": by_importance.into_iter().map(|(i, c)| json!({"importance": i, "count": c})).collect::<Vec<_>>(),
        }))
    }

    // ─── DEDUPE REPORT ───────────────────────────────

    pub fn dedupe_report(
        &self,
        project: Option<&str>,
        threshold: f64,
    ) -> Result<serde_json::Value, String> {
        use serde_json::json;
        let canonical_project = Self::canonical_project(project);
        let memories: Vec<(String, String, String, i32)> = if let Some(proj) =
            canonical_project.as_deref()
        {
            let mut stmt = self.conn.prepare("SELECT id, content, kind, importance FROM memories WHERE project = ?1 AND kind != 'credential' ORDER BY created_at DESC LIMIT 500")
                .map_err(|e| format!("Dedupe prepare: {}", e))?;
            let result: Vec<(String, String, String, i32)> = stmt
                .query_map(params![proj], |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, String>(2)?,
                        r.get::<_, i32>(3)?,
                    ))
                })
                .map_err(|e| format!("Dedupe query: {}", e))?
                .filter_map(|r| r.ok())
                .collect();
            result
        } else {
            let mut stmt = self.conn.prepare("SELECT id, content, kind, importance FROM memories WHERE kind != 'credential' ORDER BY created_at DESC LIMIT 500")
                .map_err(|e| format!("Dedupe prepare: {}", e))?;
            let result: Vec<(String, String, String, i32)> = stmt
                .query_map([], |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, String>(2)?,
                        r.get::<_, i32>(3)?,
                    ))
                })
                .map_err(|e| format!("Dedupe query: {}", e))?
                .filter_map(|r| r.ok())
                .collect();
            result
        };

        let mut groups: Vec<serde_json::Value> = Vec::new();
        let mut seen: std::collections::HashSet<usize> = std::collections::HashSet::new();

        for i in 0..memories.len() {
            if seen.contains(&i) {
                continue;
            }
            let mut group: Vec<serde_json::Value> = Vec::new();
            let words_i: std::collections::HashSet<String> = memories[i]
                .1
                .split_whitespace()
                .map(|w: &str| {
                    w.trim_matches(|c: char| !c.is_alphanumeric())
                        .to_lowercase()
                })
                .filter(|w: &String| w.len() > 2)
                .collect();
            if words_i.is_empty() {
                continue;
            }

            for j in (i + 1)..memories.len() {
                if seen.contains(&j) {
                    continue;
                }
                let words_j: std::collections::HashSet<String> = memories[j]
                    .1
                    .split_whitespace()
                    .map(|w: &str| {
                        w.trim_matches(|c: char| !c.is_alphanumeric())
                            .to_lowercase()
                    })
                    .filter(|w: &String| w.len() > 2)
                    .collect();
                if words_j.is_empty() {
                    continue;
                }

                let intersection = words_i.intersection(&words_j).count();
                let union = words_i.union(&words_j).count();
                let jaccard = if union > 0 {
                    intersection as f64 / union as f64
                } else {
                    0.0
                };

                if jaccard >= threshold {
                    if group.is_empty() {
                        group.push(json!({
                            "id": memories[i].0,
                            "kind": memories[i].2,
                            "importance": memories[i].3,
                            "preview": crate::text::preview(&memories[i].1, 120),
                        }));
                    }
                    group.push(json!({
                        "id": memories[j].0,
                        "kind": memories[j].2,
                        "importance": memories[j].3,
                        "similarity": (jaccard * 100.0).round() / 100.0,
                        "preview": crate::text::preview(&memories[j].1, 120),
                    }));
                    seen.insert(j);
                }
            }

            if !group.is_empty() {
                seen.insert(i);
                groups.push(json!({
                    "group_size": group.len(),
                    "memories": group,
                }));
            }
        }

        Ok(json!({
            "threshold": threshold,
            "duplicate_groups": groups.len(),
            "total_duplicates": groups.iter().map(|g| g["group_size"].as_u64().unwrap_or(0)).sum::<u64>(),
            "groups": groups,
        }))
    }

    // ─── STATS ────────────────────────────────────────

    pub fn stats(&self) -> Result<serde_json::Value, String> {
        let total: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))
            .unwrap_or(0);
        let global: i64 = self
            .conn
            .query_row(
                "SELECT COUNT(*) FROM memories WHERE project IS NULL",
                [],
                |r| r.get(0),
            )
            .unwrap_or(0);
        let projects: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM projects", [], |r| r.get(0))
            .unwrap_or(0);
        let expired: i64 = self
            .conn
            .query_row(
                "SELECT COUNT(*) FROM memories WHERE expires_at IS NOT NULL AND expires_at < ?1",
                params![Utc::now().to_rfc3339()],
                |r| r.get(0),
            )
            .unwrap_or(0);

        let mut by_kind = serde_json::Map::new();
        if let Ok(mut stmt) = self
            .conn
            .prepare("SELECT kind, COUNT(*) FROM memories GROUP BY kind")
        {
            if let Ok(rows) =
                stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)))
            {
                for row in rows.flatten() {
                    by_kind.insert(row.0, serde_json::json!(row.1));
                }
            }
        }
        let mut by_project = serde_json::Map::new();
        if let Ok(mut stmt) = self.conn.prepare(
            "SELECT COALESCE(project,'__global__'), COUNT(*) FROM memories GROUP BY project",
        ) {
            if let Ok(rows) =
                stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)))
            {
                for row in rows.flatten() {
                    by_project.insert(row.0, serde_json::json!(row.1));
                }
            }
        }
        let db_path = dirs::home_dir()
            .unwrap_or_default()
            .join(DB_DIR)
            .join(DB_FILE);
        let size = std::fs::metadata(&db_path).map(|m| m.len()).unwrap_or(0);
        let size_str = if size < 1024 {
            format!("{} B", size)
        } else if size < 1048576 {
            format!("{} KB", size / 1024)
        } else {
            format!("{:.1} MB", size as f64 / 1048576.0)
        };
        let hygiene = self.hygiene_report();

        Ok(
            serde_json::json!({ "total_memories": total, "global_memories": global, "projects": projects,
            "expired_pending": expired, "by_kind": by_kind, "by_project": by_project, "db_size": size_str,
            "hygiene": hygiene }),
        )
    }

    // ─── CONFIG ───────────────────────────────────────

    pub fn get_config(&self, key: &str) -> Option<String> {
        self.conn
            .query_row("SELECT value FROM config WHERE key=?1", params![key], |r| {
                r.get(0)
            })
            .ok()
    }

    pub fn set_config(&self, key: &str, value: &str) -> Result<(), String> {
        self.conn.execute("INSERT INTO config (key,value) VALUES (?1,?2) ON CONFLICT(key) DO UPDATE SET value=?2",
            params![key, value]).map_err(|e| format!("Config: {}", e))?;
        Ok(())
    }

    // ─── PROJECT CONTEXT ──────────────────────────────

    pub fn backfill_embeddings(&self) -> Result<usize, String> {
        self.backfill_embeddings_inner(false)
    }

    pub fn backfill_embeddings_force(&self) -> Result<usize, String> {
        self.backfill_embeddings_inner(true)
    }

    pub(super) fn backfill_embeddings_inner(&self, force: bool) -> Result<usize, String> {
        let sql = if force {
            "SELECT id, content, content_hash, kind, project, created_at FROM memories"
        } else {
            "SELECT id, content, content_hash, kind, project, created_at FROM memories WHERE embedding IS NULL"
        };
        let mut stmt = self
            .conn
            .prepare(sql)
            .map_err(|e| format!("Backfill prepare: {}", e))?;

        let rows = stmt
            .query_map([], |row| {
                let kind: String = row.get(3)?;
                let project: Option<String> = row.get(4)?;
                let created_at: String = row.get(5)?;
                let content: String = row.get(1)?;
                let embed_text =
                    Self::embedding_input(&kind, project.as_deref(), &created_at, &content);
                Ok((
                    row.get::<_, String>(0)?,
                    content,
                    row.get::<_, Option<String>>(2)?,
                    embed_text,
                ))
            })
            .map_err(|e| format!("Backfill query: {}", e))?;

        // (id, raw content for the hash, text handed to the encoder)
        let mut to_embed: Vec<(String, String, String)> = Vec::new();
        let mut skipped = 0usize;
        for r in rows.flatten() {
            let (id, content, existing_hash, embed_text) = r;
            if !force {
                let new_hash = content_hash(&content);
                if existing_hash.as_deref() == Some(&new_hash) {
                    let has_emb: bool = self
                        .conn
                        .query_row(
                            "SELECT embedding IS NOT NULL FROM memories WHERE id = ?1",
                            params![&id],
                            |r| r.get(0),
                        )
                        .unwrap_or(false);
                    if has_emb {
                        skipped += 1;
                        continue;
                    }
                }
            }
            to_embed.push((id, content, embed_text));
        }

        if to_embed.is_empty() {
            if skipped > 0 {
                eprintln!(
                    "  Skipped {} memories (content unchanged, embedding exists)",
                    skipped
                );
            }
            return Ok(0);
        }

        eprintln!(
            "  Computing embeddings for {} memories (skipped {} unchanged)...",
            to_embed.len(),
            skipped
        );

        let texts: Vec<&str> = to_embed.iter().map(|(_, _, t)| t.as_str()).collect();
        let embeddings = crate::embedding::embed_batch(&texts);
        let mut count = 0;
        for ((id, content, _), emb) in to_embed.iter().zip(embeddings.iter()) {
            let blob = crate::embedding::vec_to_blob(emb);
            let hash = content_hash(content);
            let _ = self.conn.execute(
                "UPDATE memories SET embedding = ?1, content_hash = ?2 WHERE id = ?3",
                params![blob, &hash, id],
            );
            if let Some(ann) = self.ann.as_ref() {
                let _ = ann.add(id, emb);
            }
            count += 1;
        }
        if count > 0 {
            if let Some(ann) = self.ann.as_ref() {
                let _ = ann.persist();
            }
        }
        if force {
            // Every vector now comes from the active model: record it so
            // the next `open_at` does not re-invalidate the corpus.
            self.set_config(EMBEDDING_MODEL_KEY, crate::embedding::model_label())?;
        }
        Ok(count)
    }
}
