//! Versioned migrations, applied in order inside one transaction under an
//! advisory lock so two servers starting together do not race.

use postgres::Client;

const MIGRATIONS: &[(i32, &str)] = &[
    (1, include_str!("../migrations/0001_tenancy.sql")),
    (2, include_str!("../migrations/0002_api_keys.sql")),
    (3, include_str!("../migrations/0003_pending_embeddings.sql")),
    (4, include_str!("../migrations/0004_limits.sql")),
    (5, include_str!("../migrations/0005_usage_audit.sql")),
    (6, include_str!("../migrations/0006_erasure.sql")),
    (7, include_str!("../migrations/0007_hnsw.sql")),
    (8, include_str!("../migrations/0008_tenant_keys.sql")),
    (9, include_str!("../migrations/0009_oauth.sql")),
];

pub fn migrate(client: &mut Client) -> Result<Vec<i32>, String> {
    let mut tx = client.transaction().map_err(crate::pg_error)?;
    tx.execute(
        "SELECT pg_advisory_xact_lock(hashtext('memorypilot_schema'))",
        &[],
    )
    .map_err(crate::pg_error)?;
    tx.batch_execute(
        "CREATE TABLE IF NOT EXISTS memorypilot_schema (
             version integer PRIMARY KEY,
             applied_at timestamptz NOT NULL DEFAULT now()
         )",
    )
    .map_err(crate::pg_error)?;
    let mut applied = Vec::new();
    for (version, sql) in MIGRATIONS {
        let done: bool = tx
            .query_one(
                "SELECT EXISTS(SELECT 1 FROM memorypilot_schema WHERE version = $1)",
                &[version],
            )
            .map_err(crate::pg_error)?
            .get(0);
        if done {
            continue;
        }
        tx.batch_execute(sql)
            .map_err(|e| format!("Migration {}: {}", version, crate::pg_error(e)))?;
        tx.execute(
            "INSERT INTO memorypilot_schema (version) VALUES ($1)",
            &[version],
        )
        .map_err(crate::pg_error)?;
        applied.push(*version);
    }
    tx.commit().map_err(crate::pg_error)?;
    Ok(applied)
}
