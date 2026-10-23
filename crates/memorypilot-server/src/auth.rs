//! API keys: `mpk_` followed by 32 random bytes in URL-safe base64. Only
//! their SHA-256 is stored (a random 256-bit key needs no slow hash).

use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use chrono::{DateTime, Utc};
use ring::rand::{SecureRandom, SystemRandom};
use uuid::Uuid;

use crate::{pg_error, PgPool, Scope, TenantLimits};

const KEY_PREFIX: &str = "mpk_";
const KEY_BYTES: usize = 32;
/// Shown in listings to tell keys apart: the prefix and 8 characters.
const DISPLAY_LENGTH: usize = KEY_PREFIX.len() + 8;
const MAX_EXTERNAL_ID: usize = 200;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Permission {
    Read,
    Write,
    Subjects,
}

impl Permission {
    pub const ALL: [Permission; 3] = [Permission::Read, Permission::Write, Permission::Subjects];

    pub fn as_str(self) -> &'static str {
        match self {
            Permission::Read => "read",
            Permission::Write => "write",
            Permission::Subjects => "subjects",
        }
    }

    pub fn parse(name: &str) -> Result<Self, String> {
        Self::ALL
            .into_iter()
            .find(|permission| permission.as_str() == name)
            .ok_or_else(|| format!("Unknown permission '{}' (read, write, subjects)", name))
    }
}

fn parse_all(names: Vec<String>) -> Result<Vec<Permission>, String> {
    names.iter().map(|name| Permission::parse(name)).collect()
}

/// A live key, as resolved from a request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApiKey {
    pub id: Uuid,
    pub tenant: Uuid,
    pub space: Uuid,
    pub permissions: Vec<Permission>,
    pub limits: TenantLimits,
}

impl ApiKey {
    pub fn allows(&self, permission: Permission) -> bool {
        self.permissions.contains(&permission)
    }

    /// The space itself, or one of its end users.
    pub fn scope(&self, subject: Option<Uuid>) -> Scope {
        Scope {
            tenant: self.tenant,
            space: self.space,
            subject,
        }
    }
}

/// Returned once, at creation: the only time the full key exists outside
/// the client that holds it.
pub struct NewApiKey {
    pub id: Uuid,
    pub secret: String,
}

impl std::fmt::Debug for NewApiKey {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("NewApiKey")
            .field("id", &self.id)
            .field("secret", &"<redacted>")
            .finish()
    }
}

#[derive(Debug, Clone)]
pub struct ApiKeyInfo {
    pub id: Uuid,
    pub space: Uuid,
    pub name: String,
    pub prefix: String,
    pub permissions: Vec<Permission>,
    pub created_at: DateTime<Utc>,
    pub last_used_at: Option<DateTime<Utc>>,
    pub revoked_at: Option<DateTime<Utc>>,
}

pub(crate) struct GeneratedKey {
    pub secret: String,
    pub hash: Vec<u8>,
    pub prefix: String,
}

pub(crate) fn generate_key() -> Result<GeneratedKey, String> {
    let mut bytes = [0u8; KEY_BYTES];
    SystemRandom::new()
        .fill(&mut bytes)
        .map_err(|_| "The system random generator failed".to_string())?;
    let secret = format!("{}{}", KEY_PREFIX, URL_SAFE_NO_PAD.encode(bytes));
    Ok(GeneratedKey {
        hash: key_hash(&secret),
        prefix: secret[..DISPLAY_LENGTH].to_string(),
        secret,
    })
}

fn key_hash(secret: &str) -> Vec<u8> {
    ring::digest::digest(&ring::digest::SHA256, secret.as_bytes())
        .as_ref()
        .to_vec()
}

fn well_formed(presented: &str) -> bool {
    presented.strip_prefix(KEY_PREFIX).is_some_and(|body| {
        body.len() == URL_SAFE_NO_PAD.encode([0u8; KEY_BYTES]).len()
            && body
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_')
    })
}

pub(crate) fn info_from_row(row: &postgres::Row) -> Result<ApiKeyInfo, String> {
    Ok(ApiKeyInfo {
        id: row.get("id"),
        space: row.get("space_id"),
        name: row.get("name"),
        prefix: row.get("prefix"),
        permissions: parse_all(row.get("permissions"))?,
        created_at: row.get("created_at"),
        last_used_at: row.get("last_used_at"),
        revoked_at: row.get("revoked_at"),
    })
}

pub(crate) fn permission_names(permissions: &[Permission]) -> Result<Vec<&'static str>, String> {
    if permissions.is_empty() {
        return Err("A key needs at least one permission".into());
    }
    let mut names: Vec<&'static str> = permissions.iter().map(|p| p.as_str()).collect();
    names.sort_unstable();
    names.dedup();
    Ok(names)
}

impl PgPool {
    /// The live key matching `presented`, if any. A malformed or unknown
    /// key and a revoked one are all `None`.
    pub fn authenticate(&self, presented: &str) -> Result<Option<ApiKey>, String> {
        if !well_formed(presented) {
            return Ok(None);
        }
        let mut client = self.checkout()?;
        let rows = client
            .query(
                "SELECT tenant_id, space_id, key_id, permissions, requests_per_minute, max_memories
                 FROM memorypilot_resolve_key($1)",
                &[&key_hash(presented)],
            )
            .map_err(pg_error)?;
        rows.first()
            .map(|row| {
                Ok(ApiKey {
                    id: row.get("key_id"),
                    tenant: row.get("tenant_id"),
                    space: row.get("space_id"),
                    permissions: parse_all(row.get("permissions"))?,
                    limits: TenantLimits {
                        requests_per_minute: row
                            .get::<_, Option<i32>>("requests_per_minute")
                            .map(|n| n.max(1) as u32),
                        max_memories: row
                            .get::<_, Option<i64>>("max_memories")
                            .map(|n| n.max(0) as u64),
                    },
                })
            })
            .transpose()
    }

    /// The end user known to the app as `external_id`, created on first use.
    /// Needs the subjects permission.
    pub fn subject(&self, key: &ApiKey, external_id: &str) -> Result<Uuid, String> {
        if !key.allows(Permission::Subjects) {
            return Err("This key cannot act for end users".into());
        }
        if external_id.trim().is_empty()
            || external_id.len() > MAX_EXTERNAL_ID
            || external_id.chars().any(char::is_control)
        {
            return Err(format!(
                "An end user id is 1 to {} characters, not only spaces, without control characters",
                MAX_EXTERNAL_ID
            ));
        }
        let mut client = self.checkout()?;
        let mut tx = client.transaction().map_err(pg_error)?;
        tx.execute(
            "SELECT set_config('app.tenant_id', $1, true)",
            &[&key.tenant.to_string()],
        )
        .map_err(pg_error)?;
        tx.execute(
            "INSERT INTO subjects (tenant_id, space_id, external_id) VALUES ($1, $2, $3)
             ON CONFLICT (tenant_id, space_id, external_id) DO NOTHING",
            &[&key.tenant, &key.space, &external_id],
        )
        .map_err(pg_error)?;
        let id: Uuid = tx
            .query_one(
                "SELECT id FROM subjects WHERE tenant_id = $1 AND space_id = $2 AND external_id = $3",
                &[&key.tenant, &key.space, &external_id],
            )
            .map_err(pg_error)?
            .get(0);
        tx.commit().map_err(pg_error)?;
        Ok(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_keys_are_well_formed_and_distinct() {
        let first = generate_key().expect("key");
        let second = generate_key().expect("key");
        assert!(well_formed(&first.secret));
        assert_ne!(first.secret, second.secret);
        assert_eq!(first.hash, key_hash(&first.secret));
        assert_eq!(first.hash.len(), 32);
        assert!(first.secret.starts_with(&first.prefix));
        assert_eq!(first.prefix.len(), DISPLAY_LENGTH);
    }

    #[test]
    fn malformed_keys_are_rejected_before_any_lookup() {
        let key = generate_key().expect("key").secret;
        for bad in [
            "",
            "mpk_",
            &key[..key.len() - 1],
            &format!("{}A", key),
            &key.replacen("mpk_", "mpx_", 1),
            &format!("{}!", &key[..key.len() - 1]),
        ] {
            assert!(!well_formed(bad), "{:?}", bad);
        }
    }

    #[test]
    fn the_secret_never_shows_in_debug_output() {
        let key = NewApiKey {
            id: Uuid::nil(),
            secret: generate_key().expect("key").secret,
        };
        assert!(!format!("{:?}", key).contains(&key.secret));
    }

    #[test]
    fn permissions_round_trip_and_reject_unknown_names() {
        for permission in Permission::ALL {
            assert_eq!(Permission::parse(permission.as_str()), Ok(permission));
        }
        assert!(Permission::parse("admin").is_err());
        assert!(permission_names(&[]).is_err());
        assert_eq!(
            permission_names(&[Permission::Write, Permission::Read, Permission::Write]),
            Ok(vec!["read", "write"])
        );
    }
}
