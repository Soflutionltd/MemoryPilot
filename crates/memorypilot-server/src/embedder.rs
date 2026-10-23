//! Vectors are computed after the write, off the request path: new and
//! edited memories are searchable by keywords at once, by meaning once
//! embedded.

use crate::{PgPool, PgStore, Scope};

/// One batch of up to `batch` pending memories per scope, so a tenant that
/// imports thousands of memories does not hold back the others. Returns how
/// many vectors were stored; a scope that fails is reported and skipped.
pub fn embed_scopes(pool: &PgPool, scopes: &[Scope], batch: usize) -> usize {
    scopes
        .iter()
        .map(|scope| {
            PgStore::from_pool(pool, *scope)
                .and_then(|store| store.embed_pending(batch))
                .unwrap_or_else(|error| {
                    eprintln!(
                        "[memorypilot-server] embedding for tenant {} space {}: {}",
                        scope.tenant, scope.space, error
                    );
                    0
                })
        })
        .sum()
}
