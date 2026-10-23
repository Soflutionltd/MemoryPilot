//! SQLite implementation of [`MemoryStore`]: `memories` rows, the
//! `memories_fts` entry of each row, and the in-process ANN index.

use super::*;

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
