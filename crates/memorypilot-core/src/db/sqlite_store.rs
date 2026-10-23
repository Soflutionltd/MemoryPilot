//! SQLite implementation of [`MemoryStore`] and [`RecallStore`]: `memories`
//! rows, their `memories_fts` entries, the in-process ANN index, links,
//! knowledge triples and entities.

use super::*;
use std::collections::{HashMap, HashSet};

const MEMORY_COLUMNS: &str = "id,content,kind,project,tags,source,importance,expires_at,metadata,created_at,updated_at,last_accessed_at,access_count";

impl MemoryStore for Database {
    fn insert_memory(&self, memory: &Memory, content_hash: &str) -> Result<(), String> {
        let tags_json = serde_json::to_string(&memory.tags).unwrap_or_else(|_| "[]".into());
        let meta_json = memory
            .metadata
            .as_ref()
            .map(|m| serde_json::to_string(m).unwrap_or_default());
        self.conn.execute(
            "INSERT INTO memories (id,content,kind,project,tags,source,importance,expires_at,metadata,embedding,content_hash,created_at,updated_at,access_count)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,NULL,?10,?11,?12,0)",
            params![memory.id, memory.content, memory.kind, memory.project.as_deref(), tags_json, memory.source, memory.importance, memory.expires_at.as_deref(), meta_json, content_hash, memory.created_at, memory.updated_at],
        ).map_err(|e| format!("Insert: {}", e))?;
        let rowid = self.conn.last_insert_rowid();
        let fts_content = Self::fts_index_content(&memory.content);
        self.conn.execute(
            "INSERT INTO memories_fts (rowid,content,tags,kind,project) VALUES (?1,?2,?3,?4,?5)",
            params![rowid, fts_content, tags_json, memory.kind, memory.project.as_deref().unwrap_or("")],
        ).map_err(|e| format!("FTS insert: {}", e))?;
        Ok(())
    }

    fn fetch_memory(&self, id: &str) -> Result<Option<Memory>, String> {
        let mut stmt = self
            .conn
            .prepare(&format!("SELECT {} FROM memories WHERE id=?1", MEMORY_COLUMNS))
            .map_err(|e| format!("Prepare: {}", e))?;
        let mut rows = stmt
            .query(params![id])
            .map_err(|e| format!("Query: {}", e))?;
        match rows.next().map_err(|e| format!("Next: {}", e))? {
            Some(row) => Ok(Some(row_to_memory(row))),
            None => Ok(None),
        }
    }

    fn stored_content(&self, id: &str) -> Result<String, String> {
        self.conn
            .query_row("SELECT content FROM memories WHERE id=?1", params![id], |r| r.get(0))
            .map_err(|e| format!("Update read: {}", e))
    }

    fn write_memory(&self, memory: &Memory, content_hash: &str, content_changed: bool) -> Result<(), String> {
        let tags_json = serde_json::to_string(&memory.tags).unwrap_or_else(|_| "[]".into());
        let metadata_json = memory
            .metadata
            .as_ref()
            .map(|value| serde_json::to_string(value).unwrap_or_default());
        if content_changed {
            self.conn.execute(
                "UPDATE memories SET content=?1,kind=?2,tags=?3,importance=?4,expires_at=?5,metadata=?6,updated_at=?7,embedding=NULL,content_hash=?8 WHERE id=?9",
                params![memory.content, memory.kind, tags_json, memory.importance, memory.expires_at, metadata_json, memory.updated_at, content_hash, memory.id],
            ).map_err(|e| format!("Update: {}", e))?;
        } else {
            self.conn.execute(
                "UPDATE memories SET content=?1,kind=?2,tags=?3,importance=?4,expires_at=?5,metadata=?6,updated_at=?7 WHERE id=?8",
                params![memory.content, memory.kind, tags_json, memory.importance, memory.expires_at, metadata_json, memory.updated_at, memory.id],
            ).map_err(|e| format!("Update: {}", e))?;
        }
        if let Ok(rowid) = self.conn.query_row::<i64, _, _>(
            "SELECT rowid FROM memories WHERE id=?1",
            params![memory.id],
            |r| r.get(0),
        ) {
            let _ = self
                .conn
                .execute("DELETE FROM memories_fts WHERE rowid=?1", params![rowid]);
            let proj = memory.project.as_deref().unwrap_or("");
            let fts_content = Self::fts_index_content(&memory.content);
            let _ = self.conn.execute(
                "INSERT INTO memories_fts (rowid,content,tags,kind,project) VALUES (?1,?2,?3,?4,?5)",
                params![rowid, fts_content, tags_json, memory.kind, proj]);
        }
        Ok(())
    }

    fn remove_memory(&self, id: &str) -> Result<bool, String> {
        if let Ok(rowid) = self.conn.query_row::<i64, _, _>(
            "SELECT rowid FROM memories WHERE id=?1",
            params![id],
            |r| r.get(0),
        ) {
            let _ = self
                .conn
                .execute("DELETE FROM memories_fts WHERE rowid=?1", params![rowid]);
        }
        let affected = self
            .conn
            .execute("DELETE FROM memories WHERE id=?1", params![id])
            .map_err(|e| format!("Delete: {}", e))?;
        if let Some(ann) = self.ann.as_ref() {
            let _ = ann.remove(id);
        }
        Ok(affected > 0)
    }

    fn list_memory_rows(
        &self,
        project: Option<&str>,
        kind: Option<&str>,
        exclude_kind: Option<&str>,
        limit: usize,
        offset: usize,
    ) -> Result<(Vec<Memory>, i64), String> {
        let mut conditions: Vec<String> = Vec::new();
        let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();

        if let Some(p) = project {
            conditions.push(format!("project = ?{}", param_values.len() + 1));
            param_values.push(Box::new(p.to_string()));
        }
        if let Some(k) = kind {
            conditions.push(format!("kind = ?{}", param_values.len() + 1));
            param_values.push(Box::new(k.to_string()));
        }
        if let Some(ek) = exclude_kind {
            conditions.push(format!("kind != ?{}", param_values.len() + 1));
            param_values.push(Box::new(ek.to_string()));
        }

        let where_clause = if conditions.is_empty() {
            String::new()
        } else {
            format!(" WHERE {}", conditions.join(" AND "))
        };

        let count_sql = format!("SELECT COUNT(*) FROM memories{}", where_clause);
        let param_refs: Vec<&dyn rusqlite::types::ToSql> =
            param_values.iter().map(|p| p.as_ref()).collect();
        let total: i64 = self
            .conn
            .query_row(&count_sql, param_refs.as_slice(), |r| r.get(0))
            .map_err(|e| format!("Count: {}", e))?;

        let data_sql = format!(
            "SELECT {} FROM memories{} ORDER BY updated_at DESC LIMIT ?{} OFFSET ?{}",
            MEMORY_COLUMNS,
            where_clause,
            param_values.len() + 1,
            param_values.len() + 2
        );
        param_values.push(Box::new(limit as i64));
        param_values.push(Box::new(offset as i64));
        let param_refs2: Vec<&dyn rusqlite::types::ToSql> =
            param_values.iter().map(|p| p.as_ref()).collect();

        let mut stmt = self
            .conn
            .prepare(&data_sql)
            .map_err(|e| format!("List: {}", e))?;
        let memories: Vec<Memory> = stmt
            .query_map(param_refs2.as_slice(), |r| Ok(row_to_memory(r)))
            .map_err(|e| format!("List query: {}", e))?
            .filter_map(|r| r.ok())
            .collect();
        Ok((memories, total))
    }

    fn find_by_content_hash(&self, content_hash: &str, project: Option<&str>) -> Option<Memory> {
        if let Some(p) = project {
            self.conn
                .prepare(&format!("SELECT {} FROM memories WHERE content_hash=?1 AND project=?2 LIMIT 1", MEMORY_COLUMNS))
                .ok()
                .and_then(|mut s| s.query_row(params![content_hash, p], |r| Ok(row_to_memory(r))).ok())
        } else {
            self.conn
                .prepare(&format!("SELECT {} FROM memories WHERE content_hash=?1 AND project IS NULL LIMIT 1", MEMORY_COLUMNS))
                .ok()
                .and_then(|mut s| s.query_row(params![content_hash], |r| Ok(row_to_memory(r))).ok())
        }
    }

    fn recent_memories(&self, project: Option<&str>, limit: usize) -> Result<Vec<Memory>, String> {
        if let Some(p) = project {
            let mut stmt = self
                .conn
                .prepare(&format!("SELECT {} FROM memories WHERE project=?1 ORDER BY updated_at DESC LIMIT ?2", MEMORY_COLUMNS))
                .map_err(|e| format!("Dedup: {}", e))?;
            let rows = stmt
                .query_map(params![p, limit as i64], |r| Ok(row_to_memory(r)))
                .map_err(|e| format!("Dedup: {}", e))?;
            Ok(rows.flatten().collect())
        } else {
            let mut stmt = self
                .conn
                .prepare(&format!("SELECT {} FROM memories WHERE project IS NULL ORDER BY updated_at DESC LIMIT ?1", MEMORY_COLUMNS))
                .map_err(|e| format!("Dedup: {}", e))?;
            let rows = stmt
                .query_map(params![limit as i64], |r| Ok(row_to_memory(r)))
                .map_err(|e| format!("Dedup: {}", e))?;
            Ok(rows.flatten().collect())
        }
    }
}

fn placeholders(count: usize) -> String {
    (1..=count).map(|i| format!("?{}", i)).collect::<Vec<_>>().join(",")
}

impl RecallStore for Database {
    fn full_text_candidates(
        &self,
        fts_query: &str,
        project: Option<&str>,
        kind: Option<&str>,
        limit: usize,
    ) -> Result<Vec<Memory>, String> {
        let mut conditions = vec!["memories_fts MATCH ?1".to_string()];
        let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> = vec![Box::new(fts_query.to_string())];
        if let Some(p) = project {
            conditions.push(format!("m.project = ?{}", param_values.len() + 1));
            param_values.push(Box::new(p.to_string()));
        }
        if let Some(k) = kind {
            conditions.push(format!("m.kind = ?{}", param_values.len() + 1));
            param_values.push(Box::new(k.to_string()));
        }
        let sql = format!(
            "SELECT m.id,m.content,m.kind,m.project,m.tags,m.source,m.importance,m.expires_at,m.metadata,m.created_at,m.updated_at,m.last_accessed_at,m.access_count,
                    bm25(memories_fts, 8.0, 5.0, 2.0, 4.0) AS bm25_score
             FROM memories_fts f
             JOIN memories m ON m.rowid = f.rowid
             WHERE {}
             ORDER BY bm25_score ASC
             LIMIT {}",
            conditions.join(" AND "),
            limit
        );
        let mut stmt = self.conn.prepare(&sql).map_err(|e| format!("Search prepare: {}", e))?;
        let param_refs: Vec<&dyn rusqlite::types::ToSql> = param_values.iter().map(|p| p.as_ref()).collect();
        let rows = stmt
            .query_map(param_refs.as_slice(), |row| Ok(row_to_memory(row)))
            .map_err(|e| format!("Search: {}", e))?;
        Ok(rows.flatten().collect())
    }

    fn vector_candidates(
        &self,
        query_embedding: &[f32],
        project: Option<&str>,
        kind: Option<&str>,
        only_ids: Option<&[String]>,
    ) -> Result<Vec<(Memory, Option<f32>)>, String> {
        let mut vec_conditions = Vec::new();
        let mut vec_params: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
        if let Some(p) = project {
            vec_conditions.push(format!("project = ?{}", vec_params.len() + 1));
            vec_params.push(Box::new(p.to_string()));
        }
        if let Some(k) = kind {
            vec_conditions.push(format!("kind = ?{}", vec_params.len() + 1));
            vec_params.push(Box::new(k.to_string()));
        }
        if let Some(ids) = only_ids {
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
        let vec_sql = format!("SELECT {},embedding FROM memories {}", MEMORY_COLUMNS, vec_where);
        let rconn = self.read_conn();
        let mut stmt = rconn.prepare(&vec_sql).map_err(|e| format!("Vector Search: {}", e))?;
        let vec_refs: Vec<&dyn rusqlite::types::ToSql> = vec_params.iter().map(|p| p.as_ref()).collect();
        let rows = stmt
            .query_map(vec_refs.as_slice(), |row| {
                let mem = row_to_memory(row);
                let blob: Option<Vec<u8>> = row.get(13)?;
                Ok((mem, blob))
            })
            .map_err(|e| format!("Vector Search error: {}", e))?;
        Ok(rows
            .flatten()
            .map(|(mem, blob)| {
                let score = blob.map(|b| crate::embedding::similarity_with_blob(query_embedding, &b));
                (mem, score)
            })
            .collect())
    }

    fn approximate_neighbours(&self, query_embedding: &[f32], limit: usize) -> Vec<(String, f32)> {
        self.ann
            .as_ref()
            .map(|index| index.search(query_embedding, limit))
            .unwrap_or_default()
    }

    fn approximate_index_len(&self) -> usize {
        self.ann.as_ref().map(|index| index.len()).unwrap_or(0)
    }

    fn link_relations(&self, target_ids: Option<&[&String]>) -> Vec<(String, String)> {
        let mut rows_data: Vec<(String, String)> = Vec::new();
        match target_ids {
            None => {
                if let Ok(mut stmt) = self.conn.prepare("SELECT target_id, relation_type FROM memory_links") {
                    if let Ok(rows) = stmt.query_map([], |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))) {
                        rows_data.extend(rows.flatten());
                    }
                }
            }
            Some(ids) => {
                if ids.is_empty() {
                    return rows_data;
                }
                let sql = format!(
                    "SELECT target_id, relation_type FROM memory_links WHERE target_id IN ({})",
                    placeholders(ids.len())
                );
                if let Ok(mut stmt) = self.conn.prepare(&sql) {
                    let param_refs: Vec<&dyn rusqlite::types::ToSql> =
                        ids.iter().map(|id| id as &dyn rusqlite::types::ToSql).collect();
                    if let Ok(rows) = stmt.query_map(param_refs.as_slice(), |row| {
                        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                    }) {
                        rows_data.extend(rows.flatten());
                    }
                }
            }
        }
        rows_data
    }

    fn related_terms(&self, word: &str) -> Vec<String> {
        let mut terms = Vec::new();
        for sql in [
            "SELECT DISTINCT object FROM knowledge_triples WHERE lower(subject) = ?1 AND valid_to IS NULL LIMIT 3",
            "SELECT DISTINCT subject FROM knowledge_triples WHERE lower(object) = ?1 AND valid_to IS NULL LIMIT 3",
            "SELECT DISTINCT b.entity_value FROM memory_entities a \
             JOIN memory_entities b ON a.memory_id = b.memory_id AND a.entity_value != b.entity_value \
             WHERE lower(a.entity_value) = ?1 LIMIT 4",
        ] {
            if let Ok(mut stmt) = self.conn.prepare(sql) {
                if let Ok(rows) = stmt.query_map(params![word], |r| r.get::<_, String>(0)) {
                    terms.extend(rows.flatten());
                }
            }
        }
        terms
    }

    fn adjacency_pairs(&self, ids: &[String]) -> HashSet<(String, String)> {
        let mut adj = HashSet::new();
        if ids.is_empty() {
            return adj;
        }
        let marks = placeholders(ids.len());
        let link_sql = format!(
            "SELECT source_id, target_id FROM memory_links WHERE source_id IN ({0}) AND target_id IN ({0})",
            marks
        );
        let entity_sql = format!(
            "SELECT DISTINCT a.memory_id, b.memory_id FROM memory_entities a \
             JOIN memory_entities b ON a.entity_value = b.entity_value AND a.entity_kind = b.entity_kind AND a.memory_id != b.memory_id \
             WHERE a.entity_kind != 'date' AND a.memory_id IN ({0}) AND b.memory_id IN ({0})",
            marks
        );
        let params: Vec<&dyn rusqlite::types::ToSql> = ids.iter().map(|id| id as &dyn rusqlite::types::ToSql).collect();
        for sql in [link_sql, entity_sql] {
            if let Ok(mut stmt) = self.conn.prepare(&sql) {
                if let Ok(rows) = stmt.query_map(params.as_slice(), |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))) {
                    adj.extend(rows.flatten());
                }
            }
        }
        adj
    }

    fn triple_counts(&self, ids: &[&String]) -> HashMap<String, (i64, i64)> {
        let mut counts = HashMap::new();
        if ids.is_empty() {
            return counts;
        }
        let sql = format!(
            "SELECT source_memory_id, \
             SUM(CASE WHEN valid_to IS NULL THEN 1 ELSE 0 END), \
             SUM(CASE WHEN valid_to IS NOT NULL THEN 1 ELSE 0 END) \
             FROM knowledge_triples WHERE source_memory_id IN ({}) GROUP BY source_memory_id",
            placeholders(ids.len())
        );
        if let Ok(mut stmt) = self.conn.prepare(&sql) {
            let param_refs: Vec<&dyn rusqlite::types::ToSql> =
                ids.iter().map(|id| id as &dyn rusqlite::types::ToSql).collect();
            if let Ok(rows) = stmt.query_map(param_refs.as_slice(), |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?, r.get::<_, i64>(2)?))
            }) {
                for row in rows.flatten() {
                    counts.insert(row.0, (row.1, row.2));
                }
            }
        }
        counts
    }

    fn memory_vectors(&self, ids: &[&String]) -> HashMap<String, Vec<f32>> {
        let mut vectors = HashMap::new();
        if ids.is_empty() {
            return vectors;
        }
        let sql = format!(
            "SELECT id, embedding FROM memories WHERE embedding IS NOT NULL AND id IN ({})",
            placeholders(ids.len())
        );
        let rconn = self.read_conn();
        let Ok(mut stmt) = rconn.prepare(&sql) else {
            return vectors;
        };
        let params: Vec<&dyn rusqlite::types::ToSql> =
            ids.iter().map(|id| *id as &dyn rusqlite::types::ToSql).collect();
        let expected_dim = crate::embedding::vector_dim();
        if let Ok(rows) = stmt.query_map(params.as_slice(), |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, Vec<u8>>(1)?))
        }) {
            for (id, blob) in rows.flatten() {
                let vector = crate::embedding::blob_to_vec(&blob);
                if vector.len() == expected_dim {
                    vectors.insert(id, vector);
                }
            }
        }
        vectors
    }

    fn memory_date_values(&self, ids: &[&String]) -> HashMap<String, Vec<String>> {
        let mut values: HashMap<String, Vec<String>> = HashMap::new();
        // Chunked to stay under SQLite's default variable limit.
        for chunk in ids.chunks(400) {
            let sql = format!(
                "SELECT memory_id, entity_value FROM memory_entities WHERE entity_kind = 'date' AND memory_id IN ({})",
                placeholders(chunk.len())
            );
            let Ok(mut stmt) = self.conn.prepare(&sql) else {
                continue;
            };
            let params: Vec<&dyn rusqlite::types::ToSql> =
                chunk.iter().map(|id| *id as &dyn rusqlite::types::ToSql).collect();
            let rows: Vec<(String, String)> = match stmt.query_map(params.as_slice(), |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            }) {
                Ok(rows) => rows.flatten().collect(),
                Err(_) => continue,
            };
            for (id, value) in rows {
                values.entry(id).or_default().push(value);
            }
        }
        values
    }

    fn graph_neighbours(&self, ids: &[String], depth: u32) -> Result<HashSet<String>, String> {
        crate::graph::traverse_graph(&self.conn, ids, depth)
    }

    fn project_path(&self, project: &str) -> Option<String> {
        let mut stmt = self.conn.prepare("SELECT path FROM projects WHERE name=?1").ok()?;
        stmt.query_row(params![project], |r| r.get::<_, String>(0)).ok()
    }

    fn project_entities(&self, project: &str, kinds: &[&str], limit: usize) -> Vec<String> {
        if kinds.is_empty() {
            return Vec::new();
        }
        let kind_filter = if kinds.len() == 1 {
            "e.entity_kind = ?2".to_string()
        } else {
            format!(
                "e.entity_kind IN ({})",
                (0..kinds.len()).map(|i| format!("?{}", i + 2)).collect::<Vec<_>>().join(", ")
            )
        };
        let sql = format!(
            "SELECT DISTINCT entity_value FROM memory_entities e JOIN memories m ON e.memory_id = m.id WHERE m.project = ?1 AND {} LIMIT {}",
            kind_filter, limit
        );
        let mut params: Vec<&dyn rusqlite::types::ToSql> = vec![&project as &dyn rusqlite::types::ToSql];
        params.extend(kinds.iter().map(|kind| kind as &dyn rusqlite::types::ToSql));
        let Ok(mut stmt) = self.conn.prepare(&sql) else {
            return Vec::new();
        };
        let values = match stmt.query_map(params.as_slice(), |r| r.get::<_, String>(0)) {
            Ok(rows) => rows.flatten().collect(),
            Err(_) => Vec::new(),
        };
        values
    }

    fn recent_project_contents(&self, project: &str, days: u32, limit: usize) -> Vec<String> {
        let sql = format!(
            "SELECT content, kind, metadata FROM memories WHERE project = ?1 AND updated_at > datetime('now', ?2) ORDER BY updated_at DESC LIMIT {}",
            limit
        );
        let Ok(mut stmt) = self.conn.prepare(&sql) else {
            return Vec::new();
        };
        let window = format!("-{} days", days);
        let rows = stmt.query_map(params![project, window], |r| {
            let metadata: Option<String> = r.get(2)?;
            let metadata: Option<serde_json::Value> = metadata.and_then(|m| serde_json::from_str(&m).ok());
            Ok(vault::read_content(&r.get::<_, String>(1)?, r.get(0)?, metadata.as_ref()))
        });
        match rows {
            Ok(rows) => rows.flatten().collect(),
            Err(_) => Vec::new(),
        }
    }

    fn important_memories(&self, min_importance: i32, limit: usize) -> Result<Vec<Memory>, String> {
        let sql = format!(
            "SELECT {} FROM memories WHERE importance >= ?1 \
             AND (expires_at IS NULL OR expires_at > datetime('now')) \
             ORDER BY importance DESC, updated_at DESC LIMIT {}",
            MEMORY_COLUMNS, limit
        );
        let mut stmt = self.conn.prepare(&sql).map_err(|e| format!("Recall critical: {}", e))?;
        let rows = stmt
            .query_map(params![min_importance], |r| Ok(row_to_memory(r)))
            .map_err(|e| format!("Recall critical: {}", e))?;
        Ok(rows.flatten().collect())
    }

    fn pinned_memories(&self) -> Result<Vec<Memory>, String> {
        let sql = format!(
            "SELECT {} FROM memories WHERE tags LIKE '%pinned%' ORDER BY importance DESC, updated_at DESC",
            MEMORY_COLUMNS
        );
        let mut stmt = self.conn.prepare(&sql).map_err(|e| format!("Recall pinned: {}", e))?;
        let rows = stmt
            .query_map([], |r| Ok(row_to_memory(r)))
            .map_err(|e| format!("Recall pinned: {}", e))?;
        Ok(rows.flatten().collect())
    }

    fn memory_count(&self, kind: Option<&str>) -> i64 {
        match kind {
            Some(k) => self
                .conn
                .query_row("SELECT COUNT(*) FROM memories WHERE kind = ?1", params![k], |r| r.get(0))
                .unwrap_or(0),
            None => self
                .conn
                .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))
                .unwrap_or(0),
        }
    }

    fn project_count(&self) -> i64 {
        self.conn
            .query_row("SELECT COUNT(*) FROM projects", [], |r| r.get(0))
            .unwrap_or(0)
    }

    fn expire_memories(&self) {
        let _ = self.cleanup_expired();
    }

    fn record_access(&self, ids: &[String]) {
        // Deferred to the embed worker so the hot search path never holds
        // the WAL writer lock; it drains the queue every ~2 s in one UPDATE.
        for id in ids {
            queue_access_update(id.clone());
        }
    }
}
