//! Postgres implementation of `MemoryStore`, `GraphStore` and `RecallStore`, bound to one
//! scope. Every statement runs in a transaction that first sets
//! `app.tenant_id` (row level security) and filters by tenant and space
//! itself; links and entities are reached through memories the scope can
//! see, so nothing of another tenant, space or end user leaks through them.

use chrono::{DateTime, Utc};
use memorypilot_core::db::{GraphStore, Memory, MemoryStore, RecallStore};
use pgvector::Vector;
use postgres::types::{Json, ToSql};
use postgres::{Client, Row, Transaction};
use std::collections::{HashMap, HashSet};
use std::sync::Mutex;
use uuid::Uuid;

use crate::auth::{generate_key, info_from_row, permission_names};
use crate::crypto::random_bytes;
use crate::pool::{app_session, ExpiryClock, PooledClient};
use crate::text::{fts5_to_tsquery, index_text};
use crate::{pg_error, MasterKey, PgConnection, PgPool};
use crate::{ApiKeyInfo, DailyUsage, NewApiKey, Permission, TenantLimits};
use memorypilot_core::db::{seal_text, unwrap_key, wrap_key, SEAL_PREFIX};

/// Where a store reads and writes. With a subject it sees that end user's
/// memories plus the ones shared by the space, and writes as that end
/// user; without one it sees and writes only the space's shared memories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
/// pgvector's ceiling for `hnsw.ef_search`, which bounds the neighbours one
/// index scan returns.
const MAX_EF_SEARCH: usize = 1000;

fn visible(alias: &str) -> String {
    format!("{a}.tenant_id = $1 AND {a}.space_id = $2 AND ({a}.subject_id IS NULL OR {a}.subject_id = $3)", a = alias)
}

/// Rows of `alias` the scope may change.
fn owned(alias: &str) -> String {
    format!(
        "{a}.tenant_id = $1 AND {a}.space_id = $2 AND {a}.subject_id IS NOT DISTINCT FROM $3",
        a = alias
    )
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

fn public_metadata(metadata: Option<serde_json::Value>) -> Option<serde_json::Value> {
    let mut value = metadata?;
    let Some(object) = value.as_object_mut() else {
        return Some(value);
    };
    object.remove("enc");
    if object.is_empty() {
        None
    } else {
        Some(value)
    }
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
        metadata: public_metadata(row.try_get(8)?),
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

/// One change, by id only.
#[derive(Debug, Clone)]
pub struct AuditEntry {
    pub at: DateTime<Utc>,
    pub space: Option<Uuid>,
    pub subject: Option<Uuid>,
    pub key: Option<Uuid>,
    /// `key` (a request) or `admin` (a command).
    pub actor: String,
    pub action: String,
    pub target: Option<String>,
}

/// Owner connection: migrations and the tenant hierarchy. Not subject to
/// row level security (it owns the tables), so never handed to request code.
pub struct PgAdmin {
    client: Client,
    master: Option<MasterKey>,
}

impl PgAdmin {
    pub fn connect(connection: &PgConnection) -> Result<Self, String> {
        Ok(Self {
            client: connection.connect()?,
            master: None,
        })
    }

    pub fn with_master_key(mut self, key: MasterKey) -> Self {
        self.master = Some(key);
        self
    }

    pub fn set_master_key(&mut self, key: MasterKey) {
        self.master = Some(key);
    }

    /// Whether this session runs over TLS, as the server sees it.
    pub fn session_encrypted(&mut self) -> Result<bool, String> {
        let row = self
            .client
            .query_one("SELECT coalesce((SELECT ssl FROM pg_stat_ssl WHERE pid = pg_backend_pid()), false)", &[])
            .map_err(pg_error)?;
        Ok(row.get(0))
    }

    /// Applies the missing migrations; returns their versions.
    pub fn migrate(&mut self) -> Result<Vec<i32>, String> {
        crate::schema::migrate(&mut self.client)
    }

    pub fn create_tenant(&mut self, name: &str) -> Result<Uuid, String> {
        let row = self
            .client
            .query_one(
                "INSERT INTO tenants (name) VALUES ($1) RETURNING id",
                &[&name],
            )
            .map_err(pg_error)?;
        let id: Uuid = row.get(0);
        if self.master.is_some() {
            self.provision_key(id)?;
        }
        Ok(id)
    }

    /// Wraps a new data key for `tenant` when it has none. False if it
    /// already has one, or if this admin has no master key.
    pub fn provision_key(&mut self, tenant: Uuid) -> Result<bool, String> {
        let Some(master) = &self.master else {
            return Ok(false);
        };
        let dek = random_bytes::<32>()?;
        let wrapped = wrap_key(master.bytes(), &dek)
            .ok_or_else(|| "Could not wrap the tenant key".to_string())?;
        let changed = self
            .client
            .execute(
                "UPDATE tenants SET wrapped_dek = $2 WHERE id = $1 AND wrapped_dek IS NULL",
                &[&tenant, &wrapped],
            )
            .map_err(pg_error)?;
        Ok(changed == 1)
    }

    pub fn has_wrapped_key(&mut self, tenant: Uuid) -> Result<bool, String> {
        let row = self
            .client
            .query_opt(
                "SELECT wrapped_dek IS NOT NULL FROM tenants WHERE id = $1",
                &[&tenant],
            )
            .map_err(pg_error)?;
        Ok(row.map(|row| row.get(0)).unwrap_or(false))
    }

    pub fn create_space(&mut self, tenant: Uuid, name: &str) -> Result<Uuid, String> {
        let row = self
            .client
            .query_one(
                "INSERT INTO spaces (tenant_id, name) VALUES ($1, $2) RETURNING id",
                &[&tenant, &name],
            )
            .map_err(pg_error)?;
        Ok(row.get(0))
    }

    pub fn create_subject(
        &mut self,
        tenant: Uuid,
        space: Uuid,
        external_id: &str,
    ) -> Result<Uuid, String> {
        let row = self
            .client
            .query_one(
                "INSERT INTO subjects (tenant_id, space_id, external_id) VALUES ($1, $2, $3) RETURNING id",
                &[&tenant, &space, &external_id],
            )
            .map_err(pg_error)?;
        Ok(row.get(0))
    }

    fn audit(
        &mut self,
        tenant: Uuid,
        space: Option<Uuid>,
        action: &str,
        target: &str,
    ) -> Result<(), String> {
        self.client
            .execute(
                "INSERT INTO audit_log (tenant_id, space_id, actor, action, target) VALUES ($1, $2, 'admin', $3, $4)",
                &[&tenant, &space, &action, &target],
            )
            .map(|_| ())
            .map_err(pg_error)
    }

    /// The tenant's most recent changes, newest first.
    pub fn audit_log(&mut self, tenant: Uuid, limit: usize) -> Result<Vec<AuditEntry>, String> {
        Ok(self
            .client
            .query(
                "SELECT at, space_id, subject_id, key_id, actor, action, target FROM audit_log
                 WHERE tenant_id = $1 ORDER BY at DESC, id DESC LIMIT $2",
                &[&tenant, &(limit as i64)],
            )
            .map_err(pg_error)?
            .iter()
            .map(|row| AuditEntry {
                at: row.get(0),
                space: row.get(1),
                subject: row.get(2),
                key: row.get(3),
                actor: row.get(4),
                action: row.get(5),
                target: row.get(6),
            })
            .collect())
    }

    /// Adds counts to the daily totals, in one transaction. Rows of a space
    /// deleted meanwhile are dropped.
    pub fn record_usage(&mut self, rows: &[DailyUsage]) -> Result<(), String> {
        let to_db = |n: u64| i64::try_from(n).map_err(|_| "usage count too large".to_string());
        let mut tx = self.client.transaction().map_err(pg_error)?;
        for row in rows {
            tx.execute(
                "INSERT INTO usage_daily (tenant_id, space_id, day, requests, writes, searches)
                 SELECT $1, $2, $3, $4, $5, $6 WHERE EXISTS (SELECT 1 FROM spaces WHERE tenant_id = $1 AND id = $2)
                 ON CONFLICT (tenant_id, space_id, day) DO UPDATE SET
                     requests = usage_daily.requests + EXCLUDED.requests,
                     writes = usage_daily.writes + EXCLUDED.writes,
                     searches = usage_daily.searches + EXCLUDED.searches",
                &[&row.tenant, &row.space, &row.day, &to_db(row.requests)?, &to_db(row.writes)?, &to_db(row.searches)?],
            )
            .map_err(pg_error)?;
        }
        tx.commit().map_err(pg_error)
    }

    /// Daily usage of the tenant's spaces over the last `days` days, oldest first.
    pub fn usage(&mut self, tenant: Uuid, days: u32) -> Result<Vec<DailyUsage>, String> {
        Ok(self
            .client
            .query(
                "SELECT space_id, day, requests, writes, searches FROM usage_daily
                 WHERE tenant_id = $1 AND day > current_date - $2::integer ORDER BY day, space_id",
                &[&tenant, &(days as i32)],
            )
            .map_err(pg_error)?
            .iter()
            .map(|row| DailyUsage {
                tenant,
                space: row.get(0),
                day: row.get(1),
                requests: row.get::<_, i64>(2).max(0) as u64,
                writes: row.get::<_, i64>(3).max(0) as u64,
                searches: row.get::<_, i64>(4).max(0) as u64,
            })
            .collect())
    }

    /// Writes every memory of the tenant as one JSON object per line
    /// (portability when a customer leaves); returns how many.
    pub fn export_tenant(
        &mut self,
        tenant: Uuid,
        out: &mut dyn std::io::Write,
    ) -> Result<usize, String> {
        use postgres::fallible_iterator::FallibleIterator;
        let sql = format!(
            "SELECT {}, sp.id, sp.name, su.external_id FROM memories m
             JOIN spaces sp ON sp.tenant_id = m.tenant_id AND sp.id = m.space_id
             LEFT JOIN subjects su ON su.tenant_id = m.tenant_id AND su.id = m.subject_id
             WHERE m.tenant_id = $1 ORDER BY m.created_at, m.id",
            COLUMNS
        );
        let mut rows = self
            .client
            .query_raw(
                sql.as_str(),
                [&tenant as &(dyn postgres::types::ToSql + Sync)],
            )
            .map_err(pg_error)?;
        let mut written = 0;
        while let Some(row) = rows.next().map_err(pg_error)? {
            let memory = row_to_memory(&row).map_err(pg_error)?;
            let line = serde_json::json!({
                "space": row.get::<_, Uuid>(13).to_string(),
                "space_name": row.get::<_, String>(14),
                "end_user": row.get::<_, Option<String>>(15),
                "memory": memory,
            });
            writeln!(out, "{}", line).map_err(|e| format!("Export: {}", e))?;
            written += 1;
        }
        Ok(written)
    }

    /// Deletes a tenant and everything it owns, irreversibly. `name` must be
    /// the tenant's exact name, as a guard against a wrong id.
    pub fn delete_tenant(&mut self, tenant: Uuid, name: &str) -> Result<(), String> {
        let deleted = self
            .client
            .execute(
                "DELETE FROM tenants WHERE id = $1 AND name = $2",
                &[&tenant, &name],
            )
            .map_err(pg_error)?;
        if deleted != 1 {
            return Err("No tenant with this id and this exact name".into());
        }
        Ok(())
    }

    /// Sets a tenant's limits (`None`: the server default); false for an unknown tenant.
    pub fn set_limits(&mut self, tenant: Uuid, limits: TenantLimits) -> Result<bool, String> {
        let requests_per_minute = limits
            .requests_per_minute
            .map(|n| i32::try_from(n).map_err(|_| "requests per minute too large".to_string()))
            .transpose()?;
        let max_memories = limits
            .max_memories
            .map(|n| i64::try_from(n).map_err(|_| "memory quota too large".to_string()))
            .transpose()?;
        let changed = self
            .client
            .execute(
                "UPDATE tenants SET requests_per_minute = $2, max_memories = $3 WHERE id = $1",
                &[&tenant, &requests_per_minute, &max_memories],
            )
            .map_err(pg_error)?;
        if changed == 1 {
            let target = format!(
                "requests_per_minute={:?} max_memories={:?}",
                requests_per_minute, max_memories
            );
            self.audit(tenant, None, "limits.set", &target)?;
        }
        Ok(changed == 1)
    }

    /// Scopes (any tenant) holding memories without a vector yet.
    pub fn pending_embedding_scopes(&mut self, limit: usize) -> Result<Vec<Scope>, String> {
        Ok(self
            .client
            .query(
                "SELECT DISTINCT tenant_id, space_id, subject_id FROM memories
                 WHERE embedding IS NULL LIMIT $1",
                &[&(limit as i64)],
            )
            .map_err(pg_error)?
            .iter()
            .map(|row| Scope {
                tenant: row.get(0),
                space: row.get(1),
                subject: row.get(2),
            })
            .collect())
    }

    /// A new key for one space of `tenant`. The secret is in the answer only.
    pub fn create_api_key(
        &mut self,
        tenant: Uuid,
        space: Uuid,
        name: &str,
        permissions: &[Permission],
    ) -> Result<NewApiKey, String> {
        let names = permission_names(permissions)?;
        let key = generate_key()?;
        let row = self
            .client
            .query_opt(
                "INSERT INTO api_keys (tenant_id, space_id, name, prefix, key_hash, permissions)
                 SELECT $1, $2, $3, $4, $5, $6 FROM spaces WHERE tenant_id = $1 AND id = $2
                 RETURNING id",
                &[&tenant, &space, &name, &key.prefix, &key.hash, &names],
            )
            .map_err(pg_error)?
            .ok_or("Unknown space for this tenant")?;
        let id: Uuid = row.get(0);
        self.audit(tenant, Some(space), "key.create", &id.to_string())?;
        Ok(NewApiKey {
            id,
            secret: key.secret,
        })
    }

    /// Revokes a key of `tenant`; false when it has no such live key.
    pub fn revoke_api_key(&mut self, tenant: Uuid, key: Uuid) -> Result<bool, String> {
        let changed = self
            .client
            .execute(
                "UPDATE api_keys SET revoked_at = now()
                 WHERE tenant_id = $1 AND id = $2 AND revoked_at IS NULL",
                &[&tenant, &key],
            )
            .map_err(pg_error)?;
        if changed == 1 {
            self.audit(tenant, None, "key.revoke", &key.to_string())?;
        }
        Ok(changed == 1)
    }

    /// Index names on `table` (used to check the HNSW migration).
    pub fn indexes_on(&mut self, table: &str) -> Result<Vec<String>, String> {
        if table.is_empty() || !table.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            return Err("Not a table name".into());
        }
        self.client
            .query(
                "SELECT indexname FROM pg_indexes WHERE schemaname = current_schema() AND tablename = $1 ORDER BY indexname",
                &[&table],
            )
            .map_err(pg_error)?
            .iter()
            .map(|row| Ok(row.get(0)))
            .collect()
    }

    pub fn list_api_keys(&mut self, tenant: Uuid) -> Result<Vec<ApiKeyInfo>, String> {
        self.client
            .query(
                "SELECT id, space_id, name, prefix, permissions, created_at, last_used_at, revoked_at
                 FROM api_keys WHERE tenant_id = $1 ORDER BY created_at, id",
                &[&tenant],
            )
            .map_err(pg_error)?
            .iter()
            .map(info_from_row)
            .collect()
    }
}

pub struct PgStore {
    client: Mutex<PooledClient>,
    scope: Scope,
    expiries: ExpiryClock,
    dek: Option<[u8; 32]>,
}

/// Expired memories are dropped at most this often per store.
const EXPIRY_INTERVAL: std::time::Duration = std::time::Duration::from_secs(60);

impl PgStore {
    /// Opens its own session as `memorypilot_app` (the login role must be
    /// a member) and checks that the space, and the subject when given,
    /// belong to the tenant.
    pub fn connect(connection: &PgConnection, scope: Scope) -> Result<Self, String> {
        Self::connect_with_master(connection, scope, None)
    }

    pub fn connect_with_master(
        connection: &PgConnection,
        scope: Scope,
        master: Option<&MasterKey>,
    ) -> Result<Self, String> {
        let client = PooledClient::standalone(app_session(connection)?);
        Self::bind(client, scope, ExpiryClock::default(), master)
    }

    /// Same, on a session of `pool` that goes back to it when the store is dropped.
    pub fn from_pool(pool: &PgPool, scope: Scope) -> Result<Self, String> {
        Self::bind(pool.checkout()?, scope, pool.expiries(), pool.master_key())
    }

    fn bind(
        client: PooledClient,
        scope: Scope,
        expiries: ExpiryClock,
        master: Option<&MasterKey>,
    ) -> Result<Self, String> {
        let store = Self {
            client: Mutex::new(client),
            scope,
            expiries,
            dek: None,
        };
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
        let dek = store.load_dek(master)?;
        Ok(Self {
            client: store.client,
            scope: store.scope,
            expiries: store.expiries,
            dek,
        })
    }

    fn load_dek(&self, master: Option<&MasterKey>) -> Result<Option<[u8; 32]>, String> {
        let Some(master) = master else {
            return Ok(None);
        };
        let tenant = self.scope.tenant;
        let wrapped: Option<Vec<u8>> = self
            .run(|tx| tx.query_opt("SELECT wrapped_dek FROM tenants WHERE id = $1", &[&tenant]))?
            .and_then(|row| row.get(0));
        match wrapped {
            Some(wrapped) => unwrap_key(master.bytes(), &wrapped)
                .map(Some)
                .ok_or_else(|| "Could not unwrap the tenant key (wrong master key)".into()),
            None => Ok(None),
        }
    }

    /// Seals a credential with the tenant key: searchable masked label,
    /// ciphertext in `metadata.enc`. Already sealed rows are left as they are.
    fn seal(&self, memory: &Memory) -> Result<Memory, String> {
        if memory.kind != "credential" || memory.content.starts_with(SEAL_PREFIX) {
            return Ok(memory.clone());
        }
        let dek = self.dek.as_ref().ok_or_else(|| {
            "Credentials need the per-tenant encryption key (set MEMORYPILOT_MASTER_KEY)"
                .to_string()
        })?;
        let (masked, enc) = seal_text(dek, &memory.content)
            .ok_or_else(|| "Could not seal the credential".to_string())?;
        let mut sealed = memory.clone();
        sealed.content = masked;
        let mut meta = sealed
            .metadata
            .take()
            .unwrap_or_else(|| serde_json::json!({}));
        if let Some(object) = meta.as_object_mut() {
            object.insert("enc".into(), serde_json::json!(enc));
        }
        sealed.metadata = Some(meta);
        Ok(sealed)
    }

    pub fn scope(&self) -> Scope {
        self.scope
    }

    fn run<T>(
        &self,
        work: impl FnOnce(&mut Transaction<'_>) -> Result<T, postgres::Error>,
    ) -> Result<T, String> {
        let mut client = self
            .client
            .lock()
            .map_err(|_| "Postgres connection lock poisoned".to_string())?;
        let mut tx = client.transaction().map_err(pg_error)?;
        tx.execute(
            "SELECT set_config('app.tenant_id', $1, true)",
            &[&self.scope.tenant.to_string()],
        )
        .map_err(pg_error)?;
        let value = work(&mut tx).map_err(pg_error)?;
        tx.commit().map_err(pg_error)?;
        Ok(value)
    }

    fn scope_params(&self, extra: Vec<Param>) -> Vec<Param> {
        let mut params: Vec<Param> = vec![
            Box::new(self.scope.tenant),
            Box::new(self.scope.space),
            Box::new(self.scope.subject),
        ];
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
        self.query(sql, extra)?
            .iter()
            .map(row_to_memory)
            .collect::<Result<_, _>>()
            .map_err(pg_error)
    }

    fn strings(&self, sql: &str, extra: Vec<Param>) -> Result<Vec<String>, String> {
        Ok(self
            .query(sql, extra)?
            .iter()
            .map(|row| row.get(0))
            .collect())
    }

    fn end_user(&self) -> Result<Uuid, String> {
        self.scope
            .subject
            .ok_or_else(|| "This needs an end user (X-MemoryPilot-Subject)".to_string())
    }

    /// Everything stored for this scope's end user (GDPR access and
    /// portability): their own memories and facts, not the shared ones.
    pub fn export_subject(&self) -> Result<serde_json::Value, String> {
        self.end_user()?;
        let memories = self.memories(
            &format!(
                "SELECT {} FROM memories m WHERE {} ORDER BY m.created_at, m.id",
                COLUMNS,
                owned("m")
            ),
            Vec::new(),
        )?;
        let facts: Vec<serde_json::Value> = self
            .query(
                "SELECT t.subject, t.predicate, t.object, t.valid_to, t.created_at FROM knowledge_triples t
                 WHERE t.tenant_id = $1 AND t.space_id = $2 AND t.subject_id = $3 ORDER BY t.created_at, t.id",
                Vec::new(),
            )?
            .iter()
            .map(|row| {
                let valid_to: Option<DateTime<Utc>> = row.get(3);
                let created_at: DateTime<Utc> = row.get(4);
                serde_json::json!({
                    "subject": row.get::<_, String>(0),
                    "predicate": row.get::<_, String>(1),
                    "object": row.get::<_, String>(2),
                    "valid_to": valid_to.map(|at| at.to_rfc3339()),
                    "created_at": created_at.to_rfc3339(),
                })
            })
            .collect();
        Ok(serde_json::json!({ "memories": memories, "facts": facts }))
    }

    /// Erases this scope's end user and everything they own (GDPR); returns
    /// how many memories went with them.
    pub fn delete_subject(&self) -> Result<i64, String> {
        self.end_user()?;
        let owned_memories = format!("SELECT count(*) FROM memories m WHERE {}", owned("m"));
        self.run(|tx| {
            let params: Vec<Param> = vec![
                Box::new(self.scope.tenant),
                Box::new(self.scope.space),
                Box::new(self.scope.subject),
            ];
            let count: i64 = tx.query_one(&owned_memories, &refs(&params))?.get(0);
            tx.execute(
                "DELETE FROM subjects WHERE tenant_id = $1 AND space_id = $2 AND id = $3",
                &refs(&params),
            )?;
            Ok(count)
        })
    }

    /// Records a change made through API key `key` in this scope.
    pub fn record_audit(&self, key: Uuid, action: &str, target: &str) -> Result<(), String> {
        self.execute(
            "INSERT INTO audit_log (tenant_id, space_id, subject_id, key_id, actor, action, target)
             VALUES ($1, $2, $3, $4, 'key', $5, $6)",
            vec![
                Box::new(key),
                Box::new(action.to_string()),
                Box::new(target.to_string()),
            ],
        )
        .map(|_| ())
    }

    /// Memories of the whole tenant, every space and end user included.
    pub fn tenant_memory_count(&self) -> Result<i64, String> {
        self.run(|tx| {
            tx.query_one(
                "SELECT count(*) FROM memories WHERE tenant_id = $1",
                &[&self.scope.tenant],
            )
            .map(|row| row.get(0))
        })
    }

    /// Whether this scope may change or delete memory `id`: an end user
    /// sees the space's shared memories but changes only their own.
    pub fn can_change(&self, id: &str) -> Result<bool, String> {
        let rows = self.query(
            &format!(
                "SELECT 1 FROM memories m WHERE {} AND m.id = $4",
                owned("m")
            ),
            vec![Box::new(id.to_string())],
        )?;
        Ok(!rows.is_empty())
    }

    /// Computes the embeddings of up to `batch` memories of the scope that
    /// have none (new or edited), with the same encoder input as SQLite.
    /// Returns how many were stored; 0 once everything is embedded.
    pub fn embed_pending(&self, batch: usize) -> Result<usize, String> {
        let rows = self.query(
            &format!(
                "SELECT m.id, m.kind, m.project, m.created_at, m.content, m.content_hash FROM memories m
                 WHERE {} AND m.embedding IS NULL ORDER BY m.created_at, m.id LIMIT $4",
                visible("m")
            ),
            vec![Box::new(batch as i64)],
        )?;
        if rows.is_empty() {
            return Ok(0);
        }
        let inputs: Vec<String> = rows
            .iter()
            .map(|row| {
                let created_at: DateTime<Utc> = row.get(3);
                memorypilot_core::db::Database::embedding_input(
                    row.get(1),
                    row.get(2),
                    &created_at.to_rfc3339(),
                    row.get(4),
                )
            })
            .collect();
        let texts: Vec<&str> = inputs.iter().map(String::as_str).collect();
        let vectors = memorypilot_core::embedding::embed_batch(&texts);
        if vectors.len() != rows.len() {
            return Err(format!(
                "Embedding: {} vectors for {} memories",
                vectors.len(),
                rows.len()
            ));
        }
        let mut stored = 0usize;
        for (row, vector) in rows.iter().zip(vectors) {
            let id: String = row.get(0);
            let hash: String = row.get(5);
            // Skipped when the content changed meanwhile: the edit queued it again.
            stored += self.execute(
                &format!(
                    "UPDATE memories AS m SET embedding = $6
                     WHERE {} AND m.id = $4 AND m.content_hash = $5 AND m.embedding IS NULL",
                    visible("m")
                ),
                vec![Box::new(id), Box::new(hash), Box::new(Vector::from(vector))],
            )? as usize;
        }
        Ok(stored)
    }

    /// Stores the embedding the server computed for one of the scope's memories.
    pub fn set_embedding(&self, id: &str, vector: &[f32]) -> Result<(), String> {
        let changed = self.execute(
            &format!(
                "UPDATE memories AS m SET embedding = $5 WHERE {} AND m.id = $4",
                owned("m")
            ),
            vec![
                Box::new(id.to_string()),
                Box::new(Vector::from(vector.to_vec())),
            ],
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
            return Err(format!(
                "Link: {} or {} is not in this scope",
                source_id, target_id
            ));
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
            vec![
                Box::new(memory_id.to_string()),
                Box::new(kind.to_string()),
                Box::new(value.to_string()),
            ],
        )?;
        if added == 0 {
            return Err(format!(
                "Entity: no memory {} this scope can change",
                memory_id
            ));
        }
        Ok(())
    }

    /// A current knowledge triple owned by the scope, optionally sourced
    /// from a memory it can see.
    pub fn add_triple(
        &self,
        subject: &str,
        predicate: &str,
        object: &str,
        source_memory_id: Option<&str>,
    ) -> Result<(), String> {
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

    /// Nearest ids in this scope through the HNSW index, ignoring how many
    /// vectors the scope holds. Search uses [`MemoryStore::approximate_neighbours`].
    pub fn hnsw_search(&self, query_embedding: &[f32], limit: usize) -> Vec<(String, f32)> {
        let dim = memorypilot_core::embedding::vector_dim();
        if query_embedding.len() != dim || limit == 0 {
            return Vec::new();
        }
        let limit = limit.min(MAX_EF_SEARCH);
        let sql = format!(
            "SELECT m.id, (1 - (m.embedding::vector({d}) <=> $4))::real FROM memories m
             WHERE {} AND vector_dims(m.embedding) = {d}
             ORDER BY m.embedding::vector({d}) <=> $4 LIMIT $5",
            visible("m"),
            d = dim
        );
        let params = self.scope_params(vec![
            Box::new(Vector::from(query_embedding.to_vec())),
            Box::new(limit as i64),
        ]);
        let found = self.run(|tx| {
            tx.execute(
                "SELECT set_config('hnsw.ef_search', $1, true),
                        set_config('hnsw.iterative_scan', 'relaxed_order', true)",
                &[&limit.to_string()],
            )?;
            tx.query(&sql, &refs(&params))
        });
        let mut neighbours: Vec<(String, f32)> = reported(
            found.map(|rows| rows.iter().map(|row| (row.get(0), row.get(1))).collect()),
            "approximate neighbours",
        );
        neighbours.sort_by(|a, b| b.1.total_cmp(&a.1));
        neighbours
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

impl PgStore {
    /// Runs `work` only when the scope may change memory `id`.
    fn on_owned_memory(
        &self,
        id: &str,
        what: &str,
        work: impl FnOnce(&mut Transaction<'_>) -> Result<(), postgres::Error>,
    ) -> Result<(), String> {
        let scope = self.scope;
        let check = format!(
            "SELECT EXISTS(SELECT 1 FROM memories m WHERE {} AND m.id = $4)",
            owned("m")
        );
        let done = self.run(|tx| {
            let owned: bool = tx
                .query_one(&check, &[&scope.tenant, &scope.space, &scope.subject, &id])?
                .get(0);
            if owned {
                work(tx)?;
            }
            Ok(owned)
        })?;
        if !done {
            return Err(format!("{}: no memory {} this scope can change", what, id));
        }
        Ok(())
    }
}

impl GraphStore for PgStore {
    fn ensure_project(&self, name: &str) -> Result<(), String> {
        let params: Vec<Param> = vec![
            Box::new(self.scope.tenant),
            Box::new(self.scope.space),
            Box::new(name.to_string()),
        ];
        self.run(|tx| {
            tx.execute(
                "INSERT INTO projects (tenant_id, space_id, name) VALUES ($1, $2, $3) ON CONFLICT DO NOTHING",
                &refs(&params),
            )
        })?;
        Ok(())
    }

    fn replace_entities(
        &self,
        memory_id: &str,
        entities: &[(String, String)],
    ) -> Result<(), String> {
        let mut seen = HashSet::new();
        let (kinds, values): (Vec<String>, Vec<String>) = entities
            .iter()
            .filter(|entity| seen.insert(*entity))
            .cloned()
            .unzip();
        let tenant = self.scope.tenant;
        self.on_owned_memory(memory_id, "Entities", |tx| {
            tx.execute(
                "DELETE FROM memory_entities WHERE tenant_id = $1 AND memory_id = $2",
                &[&tenant, &memory_id],
            )?;
            tx.execute(
                "INSERT INTO memory_entities (tenant_id, memory_id, entity_kind, entity_value)
                 SELECT $1, $2, n.kind, n.value FROM unnest($3::text[], $4::text[]) AS n(kind, value)",
                &[&tenant, &memory_id, &kinds, &values],
            )?;
            Ok(())
        })
    }

    fn memories_with_entity(
        &self,
        kind: &str,
        value: &str,
        exclude_id: &str,
        project: Option<&str>,
        any_project: bool,
        limit: usize,
    ) -> Result<Vec<(String, String)>, String> {
        let mut extra: Vec<Param> = vec![
            Box::new(kind.to_string()),
            Box::new(value.to_string()),
            Box::new(exclude_id.to_string()),
            Box::new(limit as i64),
        ];
        let project_filter = match project {
            Some(_) if any_project => "",
            Some(name) => {
                extra.push(Box::new(name.to_string()));
                " AND (m.project = $8 OR m.project IS NULL)"
            }
            None => " AND m.project IS NULL",
        };
        let rows = self
            .query(
                &format!(
                    // Oldest first, as SQLite's scan of its entity index returns them.
                    "SELECT m.id, m.kind FROM memory_entities e
                     JOIN memories m ON m.tenant_id = e.tenant_id AND m.id = e.memory_id
                     WHERE {} AND lower(e.entity_value) = lower($5) AND e.entity_kind = $4
                       AND e.entity_value = $5 AND e.memory_id <> $6{}
                     GROUP BY m.id, m.kind, m.created_at ORDER BY m.created_at, m.id LIMIT $7",
                    visible("m"),
                    project_filter
                ),
                extra,
            )
            .map_err(|e| format!("Related memories: {}", e))?;
        Ok(rows.iter().map(|row| (row.get(0), row.get(1))).collect())
    }

    /// Only the links whose other end the scope sees are replaced: the
    /// space rewriting a shared memory keeps its end users' links to it.
    fn replace_links(
        &self,
        memory_id: &str,
        links: &[(String, String, String)],
    ) -> Result<(), String> {
        let mut sources = Vec::with_capacity(links.len());
        let mut targets = Vec::with_capacity(links.len());
        let mut relations = Vec::with_capacity(links.len());
        for (source, target, relation) in links {
            sources.push(source.clone());
            targets.push(target.clone());
            relations.push(relation.clone());
        }
        let scope = self.scope;
        let delete = format!(
            "DELETE FROM memory_links AS l WHERE l.tenant_id = $1 AND (
                 (l.source_id = $4 AND EXISTS (SELECT 1 FROM memories m WHERE {0} AND m.id = l.target_id))
              OR (l.target_id = $4 AND EXISTS (SELECT 1 FROM memories m WHERE {0} AND m.id = l.source_id)))",
            visible("m")
        );
        let insert = format!(
            "INSERT INTO memory_links (tenant_id, source_id, target_id, relation_type)
             SELECT $1, s.id, t.id, n.relation
             FROM unnest($5::text[], $6::text[], $7::text[]) WITH ORDINALITY AS n(source, target, relation, position)
             JOIN memories s ON s.tenant_id = $1 AND s.id = n.source
             JOIN memories t ON t.tenant_id = $1 AND t.id = n.target
             WHERE {} AND {} AND (n.source = $4 OR n.target = $4)
             ORDER BY n.position
             ON CONFLICT (tenant_id, source_id, target_id) DO NOTHING",
            visible("s"),
            visible("t")
        );
        self.on_owned_memory(memory_id, "Links", |tx| {
            tx.execute(
                &delete,
                &[&scope.tenant, &scope.space, &scope.subject, &memory_id],
            )?;
            tx.execute(
                &insert,
                &[
                    &scope.tenant,
                    &scope.space,
                    &scope.subject,
                    &memory_id,
                    &sources,
                    &targets,
                    &relations,
                ],
            )?;
            Ok(())
        })
    }
}

impl MemoryStore for PgStore {
    fn insert_memory(&self, memory: &Memory, content_hash: &str) -> Result<(), String> {
        let sealed = self.seal(memory)?;
        let memory = &sealed;
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
            &format!(
                "SELECT {} FROM memories m WHERE {} AND m.id = $4",
                COLUMNS,
                visible("m")
            ),
            vec![Box::new(id.to_string())],
        )?;
        Ok(found.pop())
    }

    fn stored_content(&self, id: &str) -> Result<String, String> {
        self.strings(
            &format!(
                "SELECT m.content FROM memories m WHERE {} AND m.id = $4",
                visible("m")
            ),
            vec![Box::new(id.to_string())],
        )?
        .pop()
        .ok_or_else(|| format!("Update read: no memory {} in this scope", id))
    }

    fn write_memory(
        &self,
        memory: &Memory,
        content_hash: &str,
        content_changed: bool,
    ) -> Result<(), String> {
        let sealed = self.seal(memory)?;
        let memory = &sealed;
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
            return Err(format!(
                "Update: no memory {} this scope can change",
                memory.id
            ));
        }
        Ok(())
    }

    fn remove_memory(&self, id: &str) -> Result<bool, String> {
        let removed = self
            .execute(
                &format!(
                    "DELETE FROM memories AS m WHERE {} AND m.id = $4",
                    owned("m")
                ),
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
        for (filter, value) in [
            ("m.project =", project),
            ("m.kind =", kind),
            ("m.kind <>", exclude_kind),
        ] {
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
                .run(|tx| {
                    tx.query_one(
                        &format!("SELECT count(*) FROM memories m WHERE {}", where_clause),
                        &refs(&count_params),
                    )
                })
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
            let rows = self
                .run(|tx| tx.query(&sql, &refs(&page_params)))
                .map_err(|e| format!("List: {}", e))?;
            (rows, total)
        };
        let memories = rows
            .iter()
            .map(row_to_memory)
            .collect::<Result<_, _>>()
            .map_err(pg_error)?;
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
    fn full_text_candidates(
        &self,
        fts_query: &str,
        project: Option<&str>,
        kind: Option<&str>,
        limit: usize,
    ) -> Result<Vec<Memory>, String> {
        let Some(tsquery) = fts5_to_tsquery(fts_query) else {
            return Ok(Vec::new());
        };
        let mut conditions = vec![
            visible("m"),
            "m.fts @@ to_tsquery('simple', $4)".to_string(),
        ];
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
        let mut extra: Vec<Param> = vec![
            Box::new(Vector::from(query_embedding.to_vec())),
            Box::new(query_embedding.len() as i32),
        ];
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

    /// Through the HNSW index of the model's dimension (migration 0007),
    /// only for scopes large enough for search to use it: below, a small
    /// tenant would walk the index of the whole table for nothing.
    fn approximate_neighbours(&self, query_embedding: &[f32], limit: usize) -> Vec<(String, f32)> {
        if self.approximate_index_len() < memorypilot_core::db::ANN_BYPASS_THRESHOLD {
            return Vec::new();
        }
        self.hnsw_search(query_embedding, limit)
    }

    fn approximate_index_len(&self) -> usize {
        let dim = memorypilot_core::embedding::vector_dim();
        let sql = format!(
            "SELECT count(*) FROM (SELECT 1 FROM memories m
             WHERE {} AND vector_dims(m.embedding) = {} LIMIT {}) capped",
            visible("m"),
            dim,
            memorypilot_core::db::ANN_BYPASS_THRESHOLD
        );
        let count = self
            .query(&sql, Vec::new())
            .map(|rows| rows[0].get::<_, i64>(0) as usize);
        reported(count, "approximate index size")
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
            extra.push(Box::new(
                ids.iter().map(|id| id.to_string()).collect::<Vec<_>>(),
            ));
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
        rows.iter()
            .map(|row| (row.get(0), (row.get(1), row.get(2))))
            .collect()
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
        let params: Vec<Param> = vec![
            Box::new(self.scope.tenant),
            Box::new(self.scope.space),
            Box::new(project.to_string()),
        ];
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
        reported(
            self.strings(&sql, vec![Box::new(project.to_string()), Box::new(kinds)]),
            "project entities",
        )
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
        reported(
            self.strings(&sql, vec![Box::new(project.to_string()), Box::new(days)]),
            "recent project contents",
        )
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
            Some(k) => (
                format!(
                    "SELECT count(*) FROM memories m WHERE {} AND m.kind = $4",
                    visible("m")
                ),
                vec![Box::new(k.to_string())],
            ),
            None => (
                format!("SELECT count(*) FROM memories m WHERE {}", visible("m")),
                Vec::new(),
            ),
        };
        reported(self.query(&sql, extra), "memory count")
            .first()
            .map(|row| row.get(0))
            .unwrap_or(0)
    }

    fn expire_memories(&self) {
        {
            let Ok(mut clock) = self.expiries.lock() else {
                return;
            };
            if clock
                .get(&self.scope)
                .is_some_and(|at| at.elapsed() < EXPIRY_INTERVAL)
            {
                return;
            }
            clock.insert(self.scope, std::time::Instant::now());
        }
        let expired = self.execute(
            // Visible, not only owned: an expired shared memory must not
            // surface in an end user's search until the space's own pass.
            &format!("DELETE FROM memories AS m WHERE {} AND m.expires_at IS NOT NULL AND m.expires_at < now()", visible("m")),
            Vec::new(),
        );
        reported(expired, "expiry");
    }

    fn record_access(&self, ids: &[String]) {
        if ids.is_empty() {
            return;
        }
        let recorded = self.execute(
            &format!(
                "UPDATE memories AS m SET access_count = m.access_count + 1, last_accessed_at = now()
                 WHERE {} AND m.id = ANY($4)",
                visible("m")
            ),
            vec![Box::new(ids.to_vec())],
        );
        reported(recorded, "access count");
    }

    fn project_count(&self) -> i64 {
        let params: Vec<Param> = vec![Box::new(self.scope.tenant), Box::new(self.scope.space)];
        reported(
            self.run(|tx| {
                tx.query(
                    "SELECT count(*) FROM projects WHERE tenant_id = $1 AND space_id = $2",
                    &refs(&params),
                )
            }),
            "project count",
        )
        .first()
        .map(|row| row.get(0))
        .unwrap_or(0)
    }
}
