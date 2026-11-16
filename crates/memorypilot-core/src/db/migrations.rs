//! Versioned schema migrations, tracked in `PRAGMA user_version`.
//!
//! Version 1 is the schema `init_schema` and `upgrade_schema` produce (they
//! stay idempotent and run first). Every later change is one entry here,
//! applied once, in order, each in its own transaction.

use super::*;
use rusqlite::TransactionBehavior;

const MIGRATIONS: &[(i32, &str)] = &[
    (1, ""),
    (
        2,
        "ALTER TABLE projects ADD COLUMN remote TEXT;
         CREATE INDEX idx_projects_remote ON projects(remote) WHERE remote IS NOT NULL;
         CREATE TABLE project_aliases (
             alias TEXT NOT NULL,
             kind TEXT NOT NULL CHECK (kind IN ('name', 'path')),
             project TEXT NOT NULL,
             created_at TEXT NOT NULL,
             PRIMARY KEY (kind, alias)
         );
         CREATE INDEX idx_project_aliases_project ON project_aliases(project);",
    ),
];

/// The schema version this binary writes.
pub const SCHEMA_VERSION: i32 = MIGRATIONS[MIGRATIONS.len() - 1].0;

impl Database {
    pub(super) fn schema_version(&self) -> Result<i32, String> {
        self.conn
            .query_row("PRAGMA user_version", [], |row| row.get(0))
            .map_err(|error| format!("Read schema version: {}", error))
    }

    /// Applies the missing migrations and returns their versions.
    pub(super) fn apply_migrations(&self) -> Result<Vec<i32>, String> {
        let mut applied = Vec::new();
        for (version, sql) in MIGRATIONS {
            if self.schema_version()? >= *version {
                continue;
            }
            let transaction = rusqlite::Transaction::new_unchecked(&self.conn, TransactionBehavior::Immediate)
                .map_err(|error| format!("Migration {} begin: {}", version, error))?;
            // Another process may have applied it while this one waited for the lock.
            let current: i32 = transaction
                .query_row("PRAGMA user_version", [], |row| row.get(0))
                .map_err(|error| format!("Migration {} version: {}", version, error))?;
            if current >= *version {
                continue;
            }
            transaction
                .execute_batch(sql)
                .map_err(|error| format!("Migration {}: {}", version, error))?;
            transaction
                .execute_batch(&format!("PRAGMA user_version = {}", version))
                .map_err(|error| format!("Migration {} stamp: {}", version, error))?;
            transaction
                .commit()
                .map_err(|error| format!("Migration {} commit: {}", version, error))?;
            applied.push(*version);
        }
        if applied.contains(&2) {
            self.refresh_project_remotes();
        }
        Ok(applied)
    }
}
