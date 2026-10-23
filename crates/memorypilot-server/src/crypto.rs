//! Server master key and per-tenant data keys (envelope encryption).
//!
//! Credentials are sealed with a 32-byte key unique to the tenant. That key
//! is stored only wrapped with `MEMORYPILOT_MASTER_KEY`. The master key never
//! goes in the database.

use base64::engine::general_purpose::STANDARD as B64;
use base64::Engine;
use ring::rand::{SecureRandom, SystemRandom};

/// 32-byte XChaCha20-Poly1305 key that wraps every tenant's data key.
#[derive(Clone)]
pub struct MasterKey([u8; 32]);

impl MasterKey {
    pub fn generate() -> Result<Self, String> {
        Ok(Self(random_bytes()?))
    }

    pub fn from_base64(text: &str) -> Result<Self, String> {
        let bytes = B64
            .decode(text.trim())
            .map_err(|_| "MEMORYPILOT_MASTER_KEY is not valid base64".to_string())?;
        let key: [u8; 32] = bytes.try_into().map_err(|_| {
            "MEMORYPILOT_MASTER_KEY must be 32 bytes (base64 of 32 bytes)".to_string()
        })?;
        Ok(Self(key))
    }

    /// `None` when the variable is unset or empty; an error when it is present
    /// but not a 32-byte key.
    pub fn from_env() -> Result<Option<Self>, String> {
        match std::env::var("MEMORYPILOT_MASTER_KEY") {
            Err(std::env::VarError::NotPresent) => Ok(None),
            Err(_) => Err("MEMORYPILOT_MASTER_KEY is not valid UTF-8".into()),
            Ok(value) if value.trim().is_empty() => Ok(None),
            Ok(value) => Self::from_base64(&value).map(Some),
        }
    }

    pub fn to_base64(&self) -> String {
        B64.encode(self.0)
    }

    pub(crate) fn bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl std::fmt::Debug for MasterKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("MasterKey(*)")
    }
}

pub(crate) fn random_bytes<const N: usize>() -> Result<[u8; N], String> {
    let mut bytes = [0u8; N];
    SystemRandom::new()
        .fill(&mut bytes)
        .map_err(|_| "Could not generate random bytes".to_string())?;
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_generated_key_round_trips_through_base64_and_hides_itself() {
        let key = MasterKey::generate().expect("rng");
        let again = MasterKey::from_base64(&key.to_base64()).expect("parse");
        assert_eq!(key.bytes(), again.bytes());
        assert_eq!(format!("{:?}", key), "MasterKey(*)");
        assert!(!format!("{:?}", key).contains(&key.to_base64()));
    }
}
