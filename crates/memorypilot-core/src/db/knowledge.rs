//! Links between memories and the temporal knowledge graph.

use super::*;

/// Re-extracts a memory's entities (and the calendar dates it names), then
/// links it, both ways, to the memories sharing a reliable entity with it.
pub fn rebuild_memory_links<S: GraphStore + ?Sized>(store: &S, memory: &Memory) -> Result<(), String> {
    // A credential's entities come from its masked label, never from the secret.
    let masked;
    let text = if memory.kind == "credential" && !memory.content.starts_with(vault::SEAL_PREFIX) {
        masked = vault::masked_label(&memory.content);
        masked.as_str()
    } else {
        memory.content.as_str()
    };
    let entities = crate::graph::extract_entities(text, memory.project.as_deref());
    let mut entity_rows: Vec<(String, String)> = entities
        .iter()
        .map(|entity| (entity.kind.to_string(), entity.value.clone()))
        .collect();
    // Calendar dates named in the text, resolved against the memory's
    // own day. Not a link entity (two memories sharing a date are not
    // related); read by the temporal window in `search`.
    let own_day = crate::temporal::day_of_timestamp(&memory.created_at);
    for date in crate::temporal::extract_dates(text, own_day) {
        entity_rows.push(("date".into(), date.format("%Y-%m-%d").to_string()));
    }
    store.replace_entities(&memory.id, &entity_rows)?;

    // In entity order: when a memory is reached through several entities,
    // the first one gives the link its relation, the same on every rebuild.
    let mut targets = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for entity in &entities {
        if !crate::graph::is_reliable_link_entity(entity) {
            continue;
        }
        let relation_hint = crate::graph::relation_for_entity_kind(entity.kind);
        let limit = match entity.kind {
            "topic" => 8,
            "agent" => 6,
            "platform" | "origin" => 4,
            _ => 10,
        };
        let related = store.memories_with_entity(
            entity.kind,
            &entity.value,
            &memory.id,
            memory.project.as_deref(),
            entity.kind == "topic",
            limit,
        )?;
        for (target_id, target_kind) in related {
            if seen.insert(target_id.clone()) {
                targets.push((target_id, target_kind, relation_hint));
            }
        }
    }

    let mut links = Vec::with_capacity(targets.len() * 2);
    for (target_id, target_kind, relation_hint) in targets {
        let (relation, reverse) = if relation_hint == "relates_to" {
            (
                crate::graph::infer_relation(&memory.kind, &target_kind),
                crate::graph::infer_relation(&target_kind, &memory.kind),
            )
        } else {
            (relation_hint, relation_hint)
        };
        links.push((memory.id.clone(), target_id.clone(), relation.to_string()));
        links.push((target_id, memory.id.clone(), reverse.to_string()));
    }
    store.replace_links(&memory.id, &links)
}

impl Database {
    // ─── KNOWLEDGE GRAPH ──────────────────────────────

    pub fn rebuild_links(&self, memory: &Memory) -> Result<(), String> {
        rebuild_memory_links(self, memory)
    }

    // ─── KNOWLEDGE TRIPLES ─────────────────────────────

    pub fn add_triple(
        &self,
        subject: &str,
        predicate: &str,
        object: &str,
        valid_from: Option<&str>,
        valid_to: Option<&str>,
        confidence: Option<f64>,
        source_memory_id: Option<&str>,
    ) -> Result<serde_json::Value, String> {
        let sub = subject.to_lowercase().replace(' ', "_");
        let pred = predicate.to_lowercase().replace(' ', "_");
        let obj = object.to_lowercase().replace(' ', "_");

        let existing: Option<String> = self.conn.prepare(
            "SELECT id FROM knowledge_triples WHERE subject=?1 AND predicate=?2 AND object=?3 AND valid_to IS NULL"
        ).ok().and_then(|mut s| s.query_row(params![&sub, &pred, &obj], |r| r.get(0)).ok());

        if let Some(id) = existing {
            return Ok(
                serde_json::json!({"triple_id": id, "already_exists": true, "fact": format!("{} -> {} -> {}", subject, predicate, object)}),
            );
        }

        let id = format!(
            "t_{}_{}_{}_{}",
            &sub,
            &pred,
            &obj,
            &Uuid::new_v4().to_string()[..8]
        );
        let conf = confidence.unwrap_or(1.0);
        let now = Utc::now().to_rfc3339();
        self.conn.execute(
            "INSERT INTO knowledge_triples (id, subject, predicate, object, valid_from, valid_to, confidence, source_memory_id, created_at) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9)",
            params![&id, &sub, &pred, &obj, valid_from, valid_to, conf, source_memory_id, &now]
        ).map_err(|e| format!("add_triple: {}", e))?;
        Ok(
            serde_json::json!({"triple_id": id, "fact": format!("{} -> {} -> {}", subject, predicate, object)}),
        )
    }

    pub fn invalidate_triple(
        &self,
        subject: &str,
        predicate: &str,
        object: &str,
        ended: Option<&str>,
    ) -> Result<serde_json::Value, String> {
        let sub = subject.to_lowercase().replace(' ', "_");
        let pred = predicate.to_lowercase().replace(' ', "_");
        let obj = object.to_lowercase().replace(' ', "_");
        let end_date = ended
            .unwrap_or(&Utc::now().format("%Y-%m-%d").to_string())
            .to_string();
        let changed = self.conn.execute(
            "UPDATE knowledge_triples SET valid_to=?1 WHERE subject=?2 AND predicate=?3 AND object=?4 AND valid_to IS NULL",
            params![&end_date, &sub, &pred, &obj]
        ).map_err(|e| format!("invalidate_triple: {}", e))?;
        Ok(
            serde_json::json!({"invalidated": changed, "fact": format!("{} -> {} -> {}", subject, predicate, object), "ended": end_date}),
        )
    }

    pub fn query_kg_entity(
        &self,
        name: &str,
        as_of: Option<&str>,
        direction: &str,
    ) -> Result<serde_json::Value, String> {
        let eid = name.to_lowercase().replace(' ', "_");
        let mut facts = Vec::new();

        if direction == "outgoing" || direction == "both" {
            let mut sql = "SELECT subject, predicate, object, valid_from, valid_to, confidence, source_memory_id FROM knowledge_triples WHERE subject = ?1".to_string();
            let mut param_vals: Vec<Box<dyn rusqlite::types::ToSql>> = vec![Box::new(eid.clone())];
            if let Some(date) = as_of {
                sql += &format!(" AND (valid_from IS NULL OR valid_from <= ?{n}) AND (valid_to IS NULL OR valid_to >= ?{n})", n=param_vals.len()+1);
                param_vals.push(Box::new(date.to_string()));
            }
            if let Ok(mut stmt) = self.conn.prepare(&sql) {
                let refs: Vec<&dyn rusqlite::types::ToSql> =
                    param_vals.iter().map(|p| p.as_ref()).collect();
                if let Ok(rows) = stmt.query_map(refs.as_slice(), |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, String>(2)?,
                        r.get::<_, Option<String>>(3)?,
                        r.get::<_, Option<String>>(4)?,
                        r.get::<_, f64>(5)?,
                    ))
                }) {
                    for row in rows.flatten() {
                        facts.push(serde_json::json!({"direction":"outgoing","subject":row.0,"predicate":row.1,"object":row.2,"valid_from":row.3,"valid_to":row.4,"confidence":row.5,"current":row.4.is_none()}));
                    }
                }
            }
        }
        if direction == "incoming" || direction == "both" {
            let mut sql = "SELECT subject, predicate, object, valid_from, valid_to, confidence FROM knowledge_triples WHERE object = ?1".to_string();
            let mut param_vals: Vec<Box<dyn rusqlite::types::ToSql>> = vec![Box::new(eid.clone())];
            if let Some(date) = as_of {
                sql += &format!(" AND (valid_from IS NULL OR valid_from <= ?{n}) AND (valid_to IS NULL OR valid_to >= ?{n})", n=param_vals.len()+1);
                param_vals.push(Box::new(date.to_string()));
            }
            if let Ok(mut stmt) = self.conn.prepare(&sql) {
                let refs: Vec<&dyn rusqlite::types::ToSql> =
                    param_vals.iter().map(|p| p.as_ref()).collect();
                if let Ok(rows) = stmt.query_map(refs.as_slice(), |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, String>(2)?,
                        r.get::<_, Option<String>>(3)?,
                        r.get::<_, Option<String>>(4)?,
                        r.get::<_, f64>(5)?,
                    ))
                }) {
                    for row in rows.flatten() {
                        facts.push(serde_json::json!({"direction":"incoming","subject":row.0,"predicate":row.1,"object":row.2,"valid_from":row.3,"valid_to":row.4,"confidence":row.5,"current":row.4.is_none()}));
                    }
                }
            }
        }
        Ok(
            serde_json::json!({"entity": name, "as_of": as_of, "facts": facts, "count": facts.len()}),
        )
    }

    pub fn kg_timeline(&self, entity: Option<&str>) -> Result<serde_json::Value, String> {
        let mut results = Vec::new();
        let sql = if let Some(name) = entity {
            let eid = name.to_lowercase().replace(' ', "_");
            let mut stmt = self.conn.prepare(
                "SELECT subject, predicate, object, valid_from, valid_to, confidence FROM knowledge_triples WHERE subject = ?1 OR object = ?1 ORDER BY valid_from ASC NULLS LAST"
            ).map_err(|e| format!("kg_timeline: {}", e))?;
            let rows = stmt
                .query_map(params![&eid], |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, String>(2)?,
                        r.get::<_, Option<String>>(3)?,
                        r.get::<_, Option<String>>(4)?,
                    ))
                })
                .map_err(|e| format!("kg_timeline: {}", e))?;
            for r in rows.flatten() {
                results.push(serde_json::json!({"subject":r.0,"predicate":r.1,"object":r.2,"valid_from":r.3,"valid_to":r.4,"current":r.4.is_none()}));
            }
            return Ok(
                serde_json::json!({"entity": name, "timeline": results, "count": results.len()}),
            );
        } else {
            "SELECT subject, predicate, object, valid_from, valid_to FROM knowledge_triples ORDER BY valid_from ASC NULLS LAST LIMIT 100"
        };
        if entity.is_none() {
            let mut stmt = self
                .conn
                .prepare(sql)
                .map_err(|e| format!("kg_timeline: {}", e))?;
            let rows = stmt
                .query_map([], |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, String>(2)?,
                        r.get::<_, Option<String>>(3)?,
                        r.get::<_, Option<String>>(4)?,
                    ))
                })
                .map_err(|e| format!("kg_timeline: {}", e))?;
            for r in rows.flatten() {
                results.push(serde_json::json!({"subject":r.0,"predicate":r.1,"object":r.2,"valid_from":r.3,"valid_to":r.4,"current":r.4.is_none()}));
            }
        }
        Ok(serde_json::json!({"entity": "all", "timeline": results, "count": results.len()}))
    }

    pub fn kg_stats(&self) -> Result<serde_json::Value, String> {
        let total: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM knowledge_triples", [], |r| r.get(0))
            .unwrap_or(0);
        let current: i64 = self
            .conn
            .query_row(
                "SELECT COUNT(*) FROM knowledge_triples WHERE valid_to IS NULL",
                [],
                |r| r.get(0),
            )
            .unwrap_or(0);
        let expired = total - current;
        let mut predicates = Vec::new();
        if let Ok(mut stmt) = self
            .conn
            .prepare("SELECT DISTINCT predicate FROM knowledge_triples ORDER BY predicate")
        {
            if let Ok(rows) = stmt.query_map([], |r| r.get::<_, String>(0)) {
                for p in rows.flatten() {
                    predicates.push(p);
                }
            }
        }
        let mut entities = std::collections::HashSet::new();
        if let Ok(mut stmt) = self.conn.prepare("SELECT DISTINCT subject FROM knowledge_triples UNION SELECT DISTINCT object FROM knowledge_triples") {
            if let Ok(rows) = stmt.query_map([], |r| r.get::<_, String>(0)) {
                for e in rows.flatten() { entities.insert(e); }
            }
        }
        Ok(serde_json::json!({
            "entities": entities.len(),
            "triples": total,
            "current_facts": current,
            "expired_facts": expired,
            "relationship_types": predicates,
        }))
    }

    // ─── FIND RELATED ─────────────────────────────────

    pub fn find_related(&self, id: &str, depth: u32) -> Result<serde_json::Value, String> {
        use serde_json::json;
        let related_ids = crate::graph::traverse_graph(&self.conn, &[id.to_string()], depth)?;
        let mut results: Vec<serde_json::Value> = Vec::new();
        for rid in &related_ids {
            if rid == id {
                continue;
            }
            if let Ok(Some(mem)) = self.get_memory(rid) {
                results.push(json!({
                    "id": mem.id,
                    "kind": mem.kind,
                    "project": mem.project,
                    "importance": mem.importance,
                    "preview": crate::text::preview(&mem.content, 150),
                    "tags": mem.tags,
                }));
            }
        }
        Ok(json!({
            "source_id": id,
            "depth": depth,
            "related_count": results.len(),
            "related": results,
        }))
    }
}
