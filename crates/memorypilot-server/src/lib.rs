//! MemoryPilot server mode: one Postgres database for many customers.
//!
//! A [`PgStore`] is bound to a [`Scope`] (tenant, space, optional subject)
//! and implements the core's `MemoryStore` and `RecallStore`, so search and
//! recall run unchanged on top of it. Isolation is enforced twice: every
//! query filters by tenant and space, and row level security (as the
//! `memorypilot_app` role, `app.tenant_id` set per transaction) rejects
//! any row of another tenant.

mod auth;
mod connection;
mod pg_store;
mod pool;
mod schema;
pub mod text;

pub use auth::{ApiKey, ApiKeyInfo, NewApiKey, Permission};
pub use connection::PgConnection;
pub use pg_store::{PgAdmin, PgStore, Scope};
pub use pool::PgPool;

pub(crate) fn pg_error(error: postgres::Error) -> String {
    pg_message(&error)
}

pub(crate) fn pg_message(error: &postgres::Error) -> String {
    match error.as_db_error() {
        Some(db) => format!("Postgres: {}", db.message()),
        None => format!("Postgres: {}", error),
    }
}
