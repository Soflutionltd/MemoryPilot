//! Postgres implementation of `MemoryStore` and `RecallStore`, bound to one
//! scope. Every statement runs in a transaction that first sets
//! `app.tenant_id` (row level security) and filters by tenant and space
//! itself; links and entities are reached through memories the scope can
//! see, so nothing of another tenant, space or end user leaks through them.

use chrono::{DateTime, Utc};
use memorypilot_core::db::{Memory, MemoryStore, RecallStore};
use pgvector::Vector;
use postgres::types::{Json, ToSql};
use postgres::{Client, NoTls, Row, Transaction};
use std::collections::{HashMap, HashSet};
use std::sync::Mutex;
use uuid::Uuid;

use crate::pg_error;
use crate::text::{fts5_to_tsquery, index_text};

/// Where a store reads and writes. With a subject it sees that end user's
/// memories plus the ones shared by the space, and writes as that end
/// user; without one it sees and writes only the space's shared memories.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Scope {
    pub tenant: Uuid,
    pub space: Uuid,
    pub subject: Option<Uuid>,
}

const COLUMNS: &str = "m.id, m.content, m.kind, m.project, m.tags, m.source, m.importance, m.expires_at, m.metadata, m.created_at, m.updated_at, m.last_accessed_at, m.access_count";

/// BM25 weights of the SQLite index (content 8, tags 5, project 4, kind 2)
/// scaled to `ts_rank`'s {D, C, B, A} order.
const RANK_WEIGHTS: &str = "'{0.25, 0.5, 0.625, 1.0}'";

type Param = Box<dyn ToSql + Sync>;

/// Rows of `alias` the scope reads. Parameters: $1 tenant, $2 space, $3 subject.
fn visible(alias: &str) -> String {
    format!("{a}.tenant_id = $1 AND {a}.space_id = $2 AND ({a}.subject_id IS NULL OR {a}.subject_id = $3)", a = alias)
}

/// Rows of `alias` the scope may change.
fn owned(alias: &str) -> String {
    format!("{a}.tenant_id = $1 AND {a}.space_id = $2 AND {a}.subject_id IS NOT DISTINCT FROM $3", a = alias)
}

fn refs(params: &[Param]) -> Vec<&(dyn ToSql + Sync)> {
    params.iter().map(|p| p.as_ref()).collect()
}

fn timestamp(value: &str, field: &str) -> Result<DateTime<Utc>, String> {
    DateTime::parse_from_rfc3339(value)
        .map(|t| t.with_timezone(&Utc))
        .map_err(|e| format!("{} is not an RFC 3339 timestamp ({}): {}", field, value, e))
}

fn optional_timestamp(value: Option<&str>, field: &str) -> Result<Option<DateTime<Utc>>, String> {
    value.map(|v| timestamp(v, field)).transpose()
}

fn refuse_credential(memory: &Memory) -> Result<(), String> {
    if memory.kind == "credential" {
        return Err("Credentials are not stored in server mode yet: they need the per-tenant encryption key".into());
    }
    Ok(())
}

fn row_to_memory(row: &Row) -> Result<Memory, postgres::Error> {
    let tags: Json<Vec<String>> = row.try_get(4)?;
    let expires_at: Option<DateTime<Utc>> = row.try_get(7)?;
    let created_at: DateTime<Utc> = row.try_get(9)?;
    let updated_at: DateTime<Utc> = row.try_get(10)?;
    let last_accessed_at: Option<DateTime<Utc>> = row.try_get(11)?;
    Ok(Memory {
        id: row.try_get(0)?,
        content: row.try_get(1)?,
        kind: row.try_get(2)?,
        project: row.try_get(3)?,
        tags: tags.0,
        source: row.try_get(5)?,
        importance: row.try_get(6)?,
        expires_at: expires_at.map(|t| t.to_rfc3339()),
        created_at: created_at.to_rfc3339(),
        updated_at: updated_at.to_rfc3339(),
        metadata: row.try_get(8)?,
        last_accessed_at: last_accessed_at.map(|t| t.to_rfc3339()),
        access_count: row.try_get(12)?,
    })
}

/// For the trait methods that cannot return an error (as in SQLite, an
/// empty answer): the failure is still printed.
fn reported<T: Default>(answer: Result<T, String>, what: &str) -> T {
    answer.unwrap_or_else(|error| {
        eprintln!("[memorypilot-server] {}: {}", what, error);
        T::default()
    })
}

/// Owner connection: migrations and the tenant hierarchy. Not subject to
/// row level security (it owns the tables), so never handed to request code.
pub struct PgAdmin {
    client: Client,
}

impl PgAdmin {
    pub fn connect(url: &str) -> Result<Self, String> {
        Ok(Self { client: Client::connect(url, NoTls).map_err(pg_error)? })
    }

    /// Applies the missing migrations; returns their versions.
    pub fn migrate(&mut self) -> Result<Vec<i32>, String> {
        crate::schema::migrate(&mut self.client)
    }

    pub fn create_tenant(&mut self, name: &str) -> Result<Uuid, String> {
        let row = self
            .client
            .query_one("INSERT INTO tenants (name) VALUES ($1) RETURNING id", &[&name])
            .map_err(pg_error)?;
        Ok(row.get(0))
    }

    pub fn create_space(&mut self, tenant: Uuid, name: &str) -> Result<Uuid, String> {
        let row = self
            .client
            .query_one("INSERT INTO spaces (tenant_id, name) VALUES ($1, $2) RETURNING id", &[&tenant, &name])
            .map_err(pg_error)?;
        Ok(row.get(0))
    }

    pub fn create_subject(&mut self, tenant: Uuid, space: Uuid, external_id: &str) -> Result<Uuid, String> {
        let row = self
            .client
            .query_one(
                "INSERT INTO subjects (tenant_id, space_id, external_id) VALUES ($1, $2, $3) RETURNING id",
                &[&tenant, &space, &external_id],
            )
            .map_err(pg_error)?;
        Ok(row.get(0))
    }
}

pub struct PgStore {
    client: Mutex<Client>,
    scope: Scope,
}

impl PgStore {
    /// Connects as `memorypilot_app` (the login role must be a member) and
    /// checks that the space, and the subject when given, belong to the tenant.
    pub fn connect(url: &str, scope: Scope) -> Result<Self, String> {
        let mut client = Client::connect(url, NoTls).map_err(pg_error)?;
        client.batch_execute("SET ROLE memorypilot_app").map_err(pg_error)?;
        let store = Self { client: Mutex::new(client), scope };
        let known: bool = store
            .query(
                "SELECT EXISTS(SELECT 1 FROM spaces WHERE tenant_id = $1 AND id = $2)
                    AND ($3::uuid IS NULL OR EXISTS(SELECT 1 FROM subjects WHERE tenant_id = $1 AND space_id = $2 AND id = $3))",
                Vec::new(),
            )?
            .first()
            .map(|row| row.get(0))
            .unwrap_or(false);
        if !known {
            return Err("Unknown space or subject for this tenant".into());
        }
        Ok(store)
    }

    pub fn scope(&self) -> Scope {
        self.scope
    }

    fn run<T>(&self, work: impl FnOnce(&mut Transaction<'_>) -> Result<T, postgres::Error>) -> Result<T, String> {
        let mut client = self.client.lock().map_err(|_| "Postgres connection lock poisoned".to_string())?;
        let mut tx = client.transaction().map_err(pg_error)?;
        tx.execute("SELECT set_config('app.tenant_id', $1, true)", &[&self.scope.tenant.to_string()])
            .map_err(pg_error)?;
        let value = work(&mut tx).map_err(pg_error)?;
        tx.commit().map_err(pg_error)?;
        Ok(value)
    }

    fn scope_params(&self, extra: Vec<Param>) -> Vec<Param> {
        let mut params: Vec<Param> = vec![Box::new(self.scope.tenant), Box::new(self.scope.space), Box::new(self.scope.subject)];
        params.extend(extra);
        params
    }

    /// `sql` with $1-$3 bound to the scope, then `extra` from $4.
    fn query(&self, sql: &str, extra: Vec<Param>) -> Result<Vec<Row>, String> {
        let params = self.scope_params(extra);
        self.run(|tx| tx.query(sql, &refs(&params)))
    }

    fn execute(&self, sql: &str, extra: Vec<Param>) -> Result<u64, String> {
        let params = self.scope_params(extra);
        self.run(|tx| tx.execute(sql, &refs(&params)))
    }

    fn memories(&self, sql: &str, extra: Vec<Param>) -> Result<Vec<Memory>, String> {
        self.query(sql, extra)?.iter().map(row_to_memory).collect::<Result<_, _>>().map_err(pg_error)
    }

    fn strings(&self, sql: &str, extra: Vec<Param>) -> Result<Vec<String>, String> {
        Ok(self.query(sql, extra)?.iter().map(|row| row.get(0)).collect())
    }

    /// Stores the embedding the server computed for one of the scope's memories.
    pub fn set_embedding(&self, id: &str, vector: &[f32]) -> Result<(), String> {
        let changed = self.execute(
            &format!("UPDATE memories AS m SET embedding = $5 WHERE {} AND m.id = $4", owned("m")),
            vec![Box::new(id.to_string()), Box::new(Vector::from(vector.to_vec()))],
        )?;
        if changed == 0 {
            return Err(format!("Embedding: no memory {} this scope can change", id));
        }
        Ok(())
    }

    /// Links one of the scope's memories to a memory it can see.
    pub fn add_link(&self, source_id: &str, target_id: &str, relation: &str) -> Result<(), String> {
        let added = self.execute(
            &format!(
                "INSERT INTO memory_links (tenant_id, source_id, target_id, relation_type)
                 SELECT $1, s.id, t.id, $6 FROM memories s, memories t
                 WHERE {} AND {} AND s.id = $4 AND t.id = $5
                 ON CONFLICT (tenant_id, source_id, target_id) DO UPDATE SET relation_type = EXCLUDED.relation_type",
                owned("s"),
                visible("t")
            ),
            vec![Box::new(source_id.to_string()), Box::new(target_id.to_string()), Box::new(relation.to_string())],
        )?;
        if added == 0 {
            return Err(format!("Link: {} or {} is not in this scope", source_id, target_id));
        }
        Ok(())
    }

    pub fn add_entity(&self, memory_id: &str, kind: &str, value: &str) -> Result<(), String> {
        let added = self.execute(
            &format!(
                "INSERT INTO memory_entities (tenant_id, memory_id, entity_kind, entity_value)
                 SELECT $1, m.id, $5, $6 FROM memories m WHERE {} AND m.id = $4",
                owned("m")
            ),
            vec![Box::new(memory_id.to_string()), Box::new(kind.to_string()), Box::new(value.to_string())],
        )?;
        if added == 0 {
            return Err(format!("Entity: no memory {} this scope can change", memory_id));
        }
        Ok(())
    }

    /// A current knowledge triple owned by the scope, optionally sourced
    /// from a memory it can see.
    pub fn add_triple(&self, subject: &str, predicate: &str, object: &str, source_memory_id: Option<&str>) -> Result<(), String> {
        if let Some(id) = source_memory_id {
            if self.fetch_memory(id)?.is_none() {
                return Err(format!("Triple: no memory {} in this scope", id));
            }
        }
        self.execute(
            "INSERT INTO knowledge_triples (tenant_id, space_id, subject_id, subject, predicate, object, source_memory_id)
             VALUES ($1, $2, $3, $4, $5, $6, $7)",
            vec![
                Box::new(subject.to_string()),
                Box::new(predicate.to_string()),
                Box::new(object.to_string()),
                Box::new(source_memory_id.map(String::from)),
            ],
        )?;
        Ok(())
    }

    pub fn set_project_path(&self, name: &str, path: &str) -> Result<(), String> {
        let params: Vec<Param> = vec![
            Box::new(self.scope.tenant),
            Box::new(self.scope.space),
            Box::new(name.to_string()),
            Box::new(path.to_string()),
        ];
        self.run(|tx| {
            tx.execute(
                "INSERT INTO projects (tenant_id, space_id, name, path) VALUES ($1, $2, $3, $4)
                 ON CONFLICT (tenant_id, space_id, name) DO UPDATE SET path = EXCLUDED.path",
                &refs(&params),
            )
        })?;
        Ok(())
    }
}

impl MemoryStore for PgStore {
    fn insert_memory(&self, memory: &Memory, content_hash: &str) -> Result<(), String> {
        refuse_credential(memory)?;
        let created_at = timestamp(&memory.created_at, "created_at")?;
        let updated_at = timestamp(&memory.updated_at, "updated_at")?;
        let expires_at = optional_timestamp(memory.expires_at.as_deref(), "expires_at")?;
        self.execute(
            "INSERT INTO memories (tenant_id, space_id, subject_id, id, content, fts_content, kind, project, tags, source,
                                   importance, expires_at, metadata, content_hash, created_at, updated_at)
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)",
            vec![
                Box::new(memory.id.clone()),
                Box::new(memory.content.clone()),
                Box::new(index_text(&memory.content)),
                Box::new(memory.kind.clone()),
                Box::new(memory.project.clone()),
                Box::new(Json(memory.tags.clone())),
                Box::new(memory.source.clone()),
                Box::new(memory.importance),
                Box::new(expires_at),
                Box::new(memory.metadata.clone()),
                Box::new(content_hash.to_string()),
                Box::new(created_at),
                Box::new(updated_at),
            ],
        )
        .map_err(|e| format!("Insert: {}", e))?;
        Ok(())
    }

    fn fetch_memory(&self, id: &str) -> Result<Option<Memory>, String> {
        let mut found = self.memories(
            &format!("SELECT {} FROM memories m WHERE {} AND m.id = $4", COLUMNS, visible("m")),
            vec![Box::new(id.to_string())],
        )?;
        Ok(found.pop())
    }

    fn stored_content(&self, id: &str) -> Result<String, String> {
        self.strings(
            &format!("SELECT m.content FROM memories m WHERE {} AND m.id = $4", visible("m")),
            vec![Box::new(id.to_string())],
        )?
        .pop()
        .ok_or_else(|| format!("Update read: no memory {} in this scope", id))
    }

    fn write_memory(&self, memory: &Memory, content_hash: &str, content_changed: bool) -> Result<(), String> {
        refuse_credential(memory)?;
        let updated_at = timestamp(&memory.updated_at, "updated_at")?;
        let expires_at = optional_timestamp(memory.expires_at.as_deref(), "expires_at")?;
        let mut params: Vec<Param> = vec![
            Box::new(memory.id.clone()),
            Box::new(memory.content.clone()),
            Box::new(index_text(&memory.content)),
            Box::new(memory.kind.clone()),
            Box::new(Json(memory.tags.clone())),
            Box::new(memory.importance),
            Box::new(expires_at),
            Box::new(memory.metadata.clone()),
            Box::new(updated_at),
        ];
        let reset_embedding = if content_changed {
            params.push(Box::new(content_hash.to_string()));
            ", embedding = NULL, content_hash = $13"
        } else {
            ""
        };
        let changed = self
            .execute(
                &format!(
                    "UPDATE memories AS m SET content = $5, fts_content = $6, kind = $7, tags = $8, importance = $9,
                         expires_at = $10, metadata = $11, updated_at = $12{}
                     WHERE {} AND m.id = $4",
                    reset_embedding,
                    owned("m")
                ),
                params,
            )
            .map_err(|e| format!("Update: {}", e))?;
        if changed == 0 {
            return Err(format!("Update: no memory {} this scope can change", memory.id));
        }
        Ok(())
    }

    fn remove_memory(&self, id: &str) -> Result<bool, String> {
        let removed = self
            .execute(
                &format!("DELETE FROM memories AS m WHERE {} AND m.id = $4", owned("m")),
                vec![Box::new(id.to_string())],
            )
            .map_err(|e| format!("Delete: {}", e))?;
        Ok(removed > 0)
    }

    fn list_memory_rows(
        &self,
        project: Option<&str>,
        kind: Option<&str>,
        exclude_kind: Option<&str>,
        limit: usize,
        offset: usize,
    ) -> Result<(Vec<Memory>, i64), String> {
        let mut conditions = vec![visible("m")];
        let mut extra: Vec<Param> = Vec::new();
        for (filter, value) in [("m.project =", project), ("m.kind =", kind), ("m.kind <>", exclude_kind)] {
            if let Some(value) = value {
                extra.push(Box::new(value.to_string()));
                conditions.push(format!("{} ${}", filter, extra.len() + 3));
            }
        }
        let where_clause = conditions.join(" AND ");
        let filters = extra.len();
        let count_params = self.scope_params(extra);
        let (rows, total) = {
            let total: i64 = self
                .run(|tx| tx.query_one(&format!("SELECT count(*) FROM memories m WHERE {}", where_clause), &refs(&count_params)))
                .map_err(|e| format!("Count: {}", e))?
                .get(0);
            let mut page_params = count_params;
            page_params.push(Box::new(limit as i64));
            page_params.push(Box::new(offset as i64));
            let sql = format!(
                "SELECT {} FROM memories m WHERE {} ORDER BY m.updated_at DESC, m.id LIMIT ${} OFFSET ${}",
                COLUMNS,
                where_clause,
                filters + 4,
                filters + 5
            );
            let rows = self.run(|tx| tx.query(&sql, &refs(&page_params))).map_err(|e| format!("List: {}", e))?;
            (rows, total)
        };
        let memories = rows.iter().map(row_to_memory).collect::<Result<_, _>>().map_err(pg_error)?;
        Ok((memories, total))
    }

    fn find_by_content_hash(&self, content_hash: &str, project: Option<&str>) -> Option<Memory> {
        let mut extra: Vec<Param> = vec![Box::new(content_hash.to_string())];
        let project_filter = match project {
            Some(p) => {
                extra.push(Box::new(p.to_string()));
                "m.project = $5"
            }
            None => "m.project IS NULL",
        };
        let sql = format!(
            "SELECT {} FROM memories m WHERE {} AND m.content_hash = $4 AND {} ORDER BY m.created_at, m.id LIMIT 1",
            COLUMNS,
            owned("m"),
            project_filter
        );
        reported(self.memories(&sql, extra), "content hash lookup").pop()
    }

    fn recent_memories(&self, project: Option<&str>, limit: usize) -> Result<Vec<Memory>, String> {
        let mut extra: Vec<Param> = vec![Box::new(limit as i64)];
        let project_filter = match project {
            Some(p) => {
                extra.push(Box::new(p.to_string()));
                "m.project = $5"
            }
            None => "m.project IS NULL",
        };
        self.memories(
            &format!(
                "SELECT {} FROM memories m WHERE {} AND {} ORDER BY m.updated_at DESC, m.id LIMIT $4",
                COLUMNS,
                owned("m"),
                project_filter
            ),
            extra,
        )
        .map_err(|e| format!("Dedup: {}", e))
    }
}

impl RecallStore for PgStore {
    fn full_text_candidates(&self, fts_query: &str, project: Option<&str>, kind: Option<&str>, limit: usize) -> Result<Vec<Memory>, String> {
        let Some(tsquery) = fts5_to_tsquery(fts_query) else {
            return Ok(Vec::new());
        };
        let mut conditions = vec![visible("m"), "m.fts @@ to_tsquery('simple', $4)".to_string()];
        let mut extra: Vec<Param> = vec![Box::new(tsquery)];
        for (column, value) in [("m.project", project), ("m.kind", kind)] {
            if let Some(value) = value {
                extra.push(Box::new(value.to_string()));
                conditions.push(format!("{} = ${}", column, extra.len() + 3));
            }
        }
        self.memories(
            &format!(
                "SELECT {} FROM memories m WHERE {}
                 ORDER BY ts_rank({}, m.fts, to_tsquery('simple', $4)) DESC, m.id LIMIT {}",
                COLUMNS,
                conditions.join(" AND "),
                RANK_WEIGHTS,
                limit
            ),
            extra,
        )
        .map_err(|e| format!("Search: {}", e))
    }

    fn vector_candidates(
        &self,
        query_embedding: &[f32],
        project: Option<&str>,
        kind: Option<&str>,
        only_ids: Option<&[String]>,
    ) -> Result<Vec<(Memory, Option<f32>)>, String> {
        let mut conditions = vec![visible("m")];
        let mut extra: Vec<Param> = vec![Box::new(Vector::from(query_embedding.to_vec())), Box::new(query_embedding.len() as i32)];
        for (column, value) in [("m.project", project), ("m.kind", kind)] {
            if let Some(value) = value {
                extra.push(Box::new(value.to_string()));
                conditions.push(format!("{} = ${}", column, extra.len() + 3));
            }
        }
        if let Some(ids) = only_ids {
            extra.push(Box::new(ids.to_vec()));
            conditions.push(format!("m.id = ANY(${})", extra.len() + 3));
        }
        let rows = self
            .query(
                &format!(
                    "SELECT {}, CASE WHEN m.embedding IS NULL OR vector_dims(m.embedding) <> $5 THEN NULL
                                  ELSE (1 - (m.embedding <=> $4))::real END
                     FROM memories m WHERE {} ORDER BY m.created_at, m.id",
                    COLUMNS,
                    conditions.join(" AND ")
                ),
                extra,
            )
            .map_err(|e| format!("Vector Search: {}", e))?;
        rows.iter()
            .map(|row| Ok((row_to_memory(row)?, row.try_get::<_, Option<f32>>(13)?)))
            .collect::<Result<_, postgres::Error>>()
            .map_err(|e| format!("Vector Search error: {}", pg_error(e)))
    }

    fn approximate_neighbours(&self, _query_embedding: &[f32], _limit: usize) -> Vec<(String, f32)> {
        Vec::new()
    }

    fn approximate_index_len(&self) -> usize {
        0
    }

    fn link_relations(&self, target_ids: Option<&[&String]>) -> Vec<(String, String)> {
        let mut sql = format!(
            "SELECT l.target_id, l.relation_type FROM memory_links l
             JOIN memories s ON s.tenant_id = l.tenant_id AND s.id = l.source_id
             JOIN memories m ON m.tenant_id = l.tenant_id AND m.id = l.target_id
             WHERE {} AND {}",
            visible("s"),
            visible("m")
        );
        let mut extra: Vec<Param> = Vec::new();
        if let Some(ids) = target_ids {
            if ids.is_empty() {
                return Vec::new();
            }
            sql.push_str(" AND l.target_id = ANY($4)");
            extra.push(Box::new(ids.iter().map(|id| id.to_string()).collect::<Vec<_>>()));
        }
        let rows = reported(self.query(&sql, extra), "link relations");
        rows.iter().map(|row| (row.get(0), row.get(1))).collect()
    }

    fn related_terms(&self, word: &str) -> Vec<String> {
        let mut terms = Vec::new();
        for sql in [
            format!("SELECT DISTINCT t.object FROM knowledge_triples t WHERE {} AND lower(t.subject) = $4 AND t.valid_to IS NULL LIMIT 3", visible("t")),
            format!("SELECT DISTINCT t.subject FROM knowledge_triples t WHERE {} AND lower(t.object) = $4 AND t.valid_to IS NULL LIMIT 3", visible("t")),
            format!(
                "SELECT DISTINCT b.entity_value FROM memory_entities a
                 JOIN memory_entities b ON b.tenant_id = a.tenant_id AND b.memory_id = a.memory_id AND b.entity_value <> a.entity_value
                 JOIN memories m ON m.tenant_id = a.tenant_id AND m.id = a.memory_id
                 WHERE {} AND lower(a.entity_value) = $4 LIMIT 4",
                visible("m")
            ),
        ] {
            terms.extend(reported(self.strings(&sql, vec![Box::new(word.to_string())]), "related terms"));
        }
        terms
    }

    fn adjacency_pairs(&self, ids: &[String]) -> HashSet<(String, String)> {
        let mut pairs = HashSet::new();
        if ids.is_empty() {
            return pairs;
        }
        for sql in [
            format!(
                "SELECT l.source_id, l.target_id FROM memory_links l
                 JOIN memories s ON s.tenant_id = l.tenant_id AND s.id = l.source_id
                 JOIN memories m ON m.tenant_id = l.tenant_id AND m.id = l.target_id
                 WHERE {} AND {} AND l.source_id = ANY($4) AND l.target_id = ANY($4)",
                visible("s"),
                visible("m")
            ),
            format!(
                "SELECT DISTINCT a.memory_id, b.memory_id FROM memory_entities a
                 JOIN memory_entities b ON b.tenant_id = a.tenant_id AND b.entity_value = a.entity_value
                      AND b.entity_kind = a.entity_kind AND b.memory_id <> a.memory_id
                 JOIN memories s ON s.tenant_id = a.tenant_id AND s.id = a.memory_id
                 JOIN memories m ON m.tenant_id = b.tenant_id AND m.id = b.memory_id
                 WHERE {} AND {} AND a.entity_kind <> 'date' AND a.memory_id = ANY($4) AND b.memory_id = ANY($4)",
                visible("s"),
                visible("m")
            ),
        ] {
            let rows = reported(self.query(&sql, vec![Box::new(ids.to_vec())]), "adjacency");
            pairs.extend(rows.iter().map(|row| (row.get(0), row.get(1))));
        }
        pairs
    }

    fn triple_counts(&self, ids: &[&String]) -> HashMap<String, (i64, i64)> {
        if ids.is_empty() {
            return HashMap::new();
        }
        let sql = format!(
            "SELECT t.source_memory_id, count(*) FILTER (WHERE t.valid_to IS NULL), count(*) FILTER (WHERE t.valid_to IS NOT NULL)
             FROM knowledge_triples t WHERE {} AND t.source_memory_id = ANY($4) GROUP BY t.source_memory_id",
            visible("t")
        );
        let ids: Vec<String> = ids.iter().map(|id| id.to_string()).collect();
        let rows = reported(self.query(&sql, vec![Box::new(ids)]), "triple counts");
        rows.iter().map(|row| (row.get(0), (row.get(1), row.get(2)))).collect()
    }

    fn memory_vectors(&self, ids: &[&String]) -> HashMap<String, Vec<f32>> {
        if ids.is_empty() {
            return HashMap::new();
        }
        let sql = format!(
            "SELECT m.id, m.embedding FROM memories m WHERE {} AND m.embedding IS NOT NULL AND m.id = ANY($4)",
            visible("m")
        );
        let ids: Vec<String> = ids.iter().map(|id| id.to_string()).collect();
        let expected_dim = memorypilot_core::embedding::vector_dim();
        reported(self.query(&sql, vec![Box::new(ids)]), "memory vectors")
            .iter()
            .map(|row| (row.get::<_, String>(0), row.get::<_, Vector>(1).to_vec()))
            .filter(|(_, vector)| vector.len() == expected_dim)
            .collect()
    }

    fn memory_date_values(&self, ids: &[&String]) -> HashMap<String, Vec<String>> {
        let mut values: HashMap<String, Vec<String>> = HashMap::new();
        if ids.is_empty() {
            return values;
        }
        let sql = format!(
            "SELECT e.memory_id, e.entity_value FROM memory_entities e
             JOIN memories m ON m.tenant_id = e.tenant_id AND m.id = e.memory_id
             WHERE {} AND e.entity_kind = 'date' AND e.memory_id = ANY($4)",
            visible("m")
        );
        let ids: Vec<String> = ids.iter().map(|id| id.to_string()).collect();
        for row in reported(self.query(&sql, vec![Box::new(ids)]), "memory dates") {
            values.entry(row.get(0)).or_default().push(row.get(1));
        }
        values
    }

    fn graph_neighbours(&self, ids: &[String], depth: u32) -> Result<HashSet<String>, String> {
        if ids.is_empty() || depth == 0 {
            return Ok(HashSet::new());
        }
        let sql = format!(
            "SELECT l.target_id FROM memory_links l JOIN memories m ON m.tenant_id = l.tenant_id AND m.id = l.target_id
             WHERE {0} AND l.source_id = ANY($4)
             UNION
             SELECT l.source_id FROM memory_links l JOIN memories m ON m.tenant_id = l.tenant_id AND m.id = l.source_id
             WHERE {0} AND l.target_id = ANY($4)",
            visible("m")
        );
        let mut visited: HashSet<String> = ids.iter().cloned().collect();
        let mut frontier: Vec<String> = ids.to_vec();
        for _ in 0..depth {
            if frontier.is_empty() {
                break;
            }
            frontier = self
                .strings(&sql, vec![Box::new(frontier)])
                .map_err(|e| format!("Graph: {}", e))?
                .into_iter()
                .filter(|id| visited.insert(id.clone()))
                .collect();
        }
        Ok(visited)
    }

    fn project_path(&self, project: &str) -> Option<String> {
        let params: Vec<Param> = vec![Box::new(self.scope.tenant), Box::new(self.scope.space), Box::new(project.to_string())];
        let rows = reported(
            self.run(|tx| tx.query("SELECT path FROM projects WHERE tenant_id = $1 AND space_id = $2 AND name = $3", &refs(&params))),
            "project path",
        );
        rows.first().map(|row| row.get(0))
    }

    fn project_entities(&self, project: &str, kinds: &[&str], limit: usize) -> Vec<String> {
        if kinds.is_empty() {
            return Vec::new();
        }
        let sql = format!(
            "SELECT DISTINCT e.entity_value FROM memory_entities e
             JOIN memories m ON m.tenant_id = e.tenant_id AND m.id = e.memory_id
             WHERE {} AND m.project = $4 AND e.entity_kind = ANY($5) ORDER BY e.entity_value LIMIT {}",
            visible("m"),
            limit
        );
        let kinds: Vec<String> = kinds.iter().map(|kind| kind.to_string()).collect();
        reported(self.strings(&sql, vec![Box::new(project.to_string()), Box::new(kinds)]), "project entities")
    }

    fn recent_project_contents(&self, project: &str, days: u32, limit: usize) -> Vec<String> {
        let sql = format!(
            "SELECT m.content FROM memories m
             WHERE {} AND m.project = $4 AND m.updated_at > now() - make_interval(days => $5)
             ORDER BY m.updated_at DESC LIMIT {}",
            visible("m"),
            limit
        );
        let days = i32::try_from(days).unwrap_or(i32::MAX);
        reported(self.strings(&sql, vec![Box::new(project.to_string()), Box::new(days)]), "recent project contents")
    }

    fn important_memories(&self, min_importance: i32, limit: usize) -> Result<Vec<Memory>, String> {
        self.memories(
            &format!(
                "SELECT {} FROM memories m WHERE {} AND m.importance >= $4 AND (m.expires_at IS NULL OR m.expires_at > now())
                 ORDER BY m.importance DESC, m.updated_at DESC LIMIT {}",
                COLUMNS,
                visible("m"),
                limit
            ),
            vec![Box::new(min_importance)],
        )
        .map_err(|e| format!("Recall critical: {}", e))
    }

    fn pinned_memories(&self) -> Result<Vec<Memory>, String> {
        self.memories(
            &format!(
                "SELECT {} FROM memories m WHERE {} AND m.tags ? 'pinned' ORDER BY m.importance DESC, m.updated_at DESC",
                COLUMNS,
                visible("m")
            ),
            Vec::new(),
        )
        .map_err(|e| format!("Recall pinned: {}", e))
    }

    fn memory_count(&self, kind: Option<&str>) -> i64 {
        let (sql, extra): (String, Vec<Param>) = match kind {
            Some(k) => (format!("SELECT count(*) FROM memories m WHERE {} AND m.kind = $4", visible("m")), vec![Box::new(k.to_string())]),
            None => (format!("SELECT count(*) FROM memories m WHERE {}", visible("m")), Vec::new()),
        };
        reported(self.query(&sql, extra), "memory count").first().map(|row| row.get(0)).unwrap_or(0)
    }

    fn project_count(&self) -> i64 {
        let params: Vec<Param> = vec![Box::new(self.scope.tenant), Box::new(self.scope.space)];
        reported(
            self.run(|tx| tx.query("SELECT count(*) FROM projects WHERE tenant_id = $1 AND space_id = $2", &refs(&params))),
            "project count",
        )
        .first()
        .map(|row| row.get(0))
        .unwrap_or(0)
    }
}
