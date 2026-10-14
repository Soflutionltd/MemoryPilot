//! Encryption at rest for `credential` memories.
//!
//! The SQLite file used to hold logins, keys and tokens in clear. A
//! sealed credential now stores:
//!   - `content`  = "🔒 " + a masked label (secrets replaced by ••••), so
//!     search, FTS and embeddings still find "login admin La Villart";
//!   - `metadata.enc` = base64(nonce ‖ XChaCha20-Poly1305(original)).
//! Reads go through `row_to_memory`, which transparently decrypts, so
//! tools behave exactly as before (safe mode still hides credentials).
//!
//! The 256-bit key lives in the macOS login Keychain (service
//! "MemoryPilot", account "credentials-key-v2"), created on first use through the
//! Keychain Services API. On other systems it falls back to a 0600 key file
//! next to the database. If no key can be obtained, nothing is sealed
//! (never lose data because of a missing key).

use std::sync::OnceLock;

use base64::engine::general_purpose::STANDARD as B64;
use base64::Engine;
use chacha20poly1305::aead::{Aead, AeadCore, KeyInit, OsRng};
use chacha20poly1305::{XChaCha20Poly1305, XNonce};
use rusqlite::params;

use super::{queue_embedding_job, Database};

pub(crate) const SEAL_PREFIX: &str = "🔒 ";
const KEYCHAIN_SERVICE: &str = "MemoryPilot";
const KEYCHAIN_ACCOUNT: &str = "credentials-key-v2";

static KEY: OnceLock<Option<[u8; 32]>> = OnceLock::new();

fn decode_key(text: &str) -> Option<[u8; 32]> {
    let bytes = B64.decode(text.trim()).ok()?;
    bytes.try_into().ok()
}

#[cfg(target_os = "macos")]
fn load_or_create_key() -> Option<[u8; 32]> {
    // Keychain Services API directly: the key never appears on a command
    // line (visible to `ps`) as it would with /usr/bin/security -w.
    use security_framework::passwords::{get_generic_password, set_generic_password};
    if std::env::var("MEMORYPILOT_VAULT").map(|v| v == "off").unwrap_or(false) {
        return None;
    }
    if let Ok(found) = get_generic_password(KEYCHAIN_SERVICE, KEYCHAIN_ACCOUNT) {
        return decode_key(&String::from_utf8_lossy(&found));
    }
    let key = XChaCha20Poly1305::generate_key(&mut OsRng);
    let encoded = B64.encode(key);
    set_generic_password(KEYCHAIN_SERVICE, KEYCHAIN_ACCOUNT, encoded.as_bytes()).ok()?;
    // Read back: only trust a key the Keychain actually returns.
    let check = get_generic_password(KEYCHAIN_SERVICE, KEYCHAIN_ACCOUNT).ok()?;
    decode_key(&String::from_utf8_lossy(&check))
}

#[cfg(not(target_os = "macos"))]
fn load_or_create_key() -> Option<[u8; 32]> {
    if std::env::var("MEMORYPILOT_VAULT").map(|v| v == "off").unwrap_or(false) {
        return None;
    }
    let path = dirs::home_dir()?.join(".MemoryPilot").join("credentials.key");
    if let Ok(text) = std::fs::read_to_string(&path) {
        return decode_key(&text);
    }
    let key = XChaCha20Poly1305::generate_key(&mut OsRng);
    std::fs::write(&path, B64.encode(key)).ok()?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600));
    }
    decode_key(&std::fs::read_to_string(&path).ok()?)
}

fn key() -> Option<&'static [u8; 32]> {
    KEY.get_or_init(load_or_create_key).as_ref()
}

fn encrypt(plain: &str) -> Option<String> {
    let cipher = XChaCha20Poly1305::new(key()?.into());
    let nonce = XChaCha20Poly1305::generate_nonce(&mut OsRng);
    let ct = cipher.encrypt(&nonce, plain.as_bytes()).ok()?;
    let mut blob = nonce.to_vec();
    blob.extend_from_slice(&ct);
    Some(B64.encode(blob))
}

pub(crate) fn decrypt(encoded: &str) -> Option<String> {
    let blob = B64.decode(encoded).ok()?;
    if blob.len() < 24 {
        return None;
    }
    let (nonce, ct) = blob.split_at(24);
    let cipher = XChaCha20Poly1305::new(key()?.into());
    let plain = cipher.decrypt(XNonce::from_slice(nonce), ct).ok()?;
    String::from_utf8(plain).ok()
}

static REVEAL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// Remote (HTTP) mode keeps credentials masked unless explicitly allowed.
pub(crate) fn set_reveal(on: bool) {
    REVEAL.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Transparent read path: sealed row → original text.
pub(crate) fn unseal_content(content: &str, metadata: Option<&serde_json::Value>) -> Option<String> {
    if !content.starts_with(SEAL_PREFIX) || !REVEAL.load(std::sync::atomic::Ordering::Relaxed) {
        return None;
    }
    let enc = metadata?.get("enc")?.as_str()?;
    decrypt(enc)
}

const SECRET_KEYWORDS: &[&str] = &[
    "password", "passwd", "pwd", "mdp", "passe", "token", "secret", "key", "clé", "cle",
];

fn looks_secret(token: &str) -> bool {
    let core = token.trim_matches(|c: char| !c.is_alphanumeric());
    if core.contains('@') {
        return false; // keep emails: they are what you search for
    }
    let len = core.chars().count();
    let digits = core.chars().filter(|c| c.is_ascii_digit()).count();
    let letters = core.chars().filter(|c| c.is_alphabetic()).count();
    (len >= 16 && digits > 0 && letters > 0)
        || core.starts_with("sk-")
        || core.starts_with("sk_")
        || core.starts_with("ghp_")
        || core.starts_with("xox")
        || core.starts_with("eyJ")
}

/// Searchable label with secret-looking tokens masked.
pub(crate) fn masked_label(content: &str) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut mask_next = 0usize;
    for token in content.split_whitespace() {
        let lower = token.to_lowercase();
        let keyword = SECRET_KEYWORDS
            .iter()
            .any(|k| lower.trim_end_matches([':', '=']).ends_with(k));
        if mask_next > 0 && !keyword {
            out.push("••••".to_string());
            mask_next -= 1;
            continue;
        }
        if looks_secret(token) {
            out.push("••••".to_string());
            continue;
        }
        // "password: xyz", "mdp=xyz", "token xyz"
        if keyword {
            if let Some(pos) = token.find([':', '=']) {
                let (head, tail) = token.split_at(pos + 1);
                if tail.trim().is_empty() {
                    out.push(head.to_string());
                    mask_next = 1;
                } else {
                    out.push(format!("{}••••", head));
                }
            } else {
                out.push(token.to_string());
                mask_next = 1;
            }
            continue;
        }
        out.push(token.to_string());
    }
    let label: String = out.join(" ").chars().take(220).collect();
    format!("{}{}", SEAL_PREFIX, label)
}

impl Database {
    /// Encrypt every credential still stored in clear. Idempotent.
    pub fn seal_credentials(&self) -> Result<usize, String> {
        if key().is_none() {
            eprintln!("[MemoryPilot] Vault: no encryption key available, credentials left as is.");
            return Ok(0);
        }
        let rows: Vec<(String, String, Option<String>, Option<String>, String)> = {
            let mut stmt = self
                .conn
                .prepare(
                    "SELECT id, content, metadata, project, created_at FROM memories
                     WHERE kind = 'credential' AND substr(content, 1, 2) != ?1",
                )
                .map_err(|e| format!("seal prepare: {}", e))?;
            let prefix: String = SEAL_PREFIX.chars().take(1).collect();
            let rows = stmt
                .query_map(params![prefix], |r| {
                    Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?))
                })
                .map_err(|e| format!("seal query: {}", e))?;
            rows.flatten().collect()
        };
        let mut sealed = 0usize;
        for (id, content, metadata, project, created_at) in rows {
            if content.starts_with(SEAL_PREFIX) {
                continue;
            }
            let Some(enc) = encrypt(&content) else { continue };
            // Never write a ciphertext we cannot read back.
            if decrypt(&enc).as_deref() != Some(content.as_str()) {
                continue;
            }
            let mut meta: serde_json::Value = metadata
                .as_deref()
                .and_then(|m| serde_json::from_str(m).ok())
                .unwrap_or_else(|| serde_json::json!({}));
            if !meta.is_object() {
                meta = serde_json::json!({ "previous_metadata": meta });
            }
            meta["enc"] = serde_json::Value::String(enc);
            meta["sealed"] = serde_json::Value::Bool(true);
            let label = masked_label(&content);
            let meta_json = serde_json::to_string(&meta).unwrap_or_default();
            self.conn
                .execute(
                    "UPDATE memories SET content=?1, metadata=?2, embedding=NULL, content_hash=?3 WHERE id=?4",
                    params![label, meta_json, super::content_hash(&label), id],
                )
                .map_err(|e| format!("seal update: {}", e))?;
            if let Ok(rowid) = self.conn.query_row::<i64, _, _>(
                "SELECT rowid FROM memories WHERE id=?1",
                params![id],
                |r| r.get(0),
            ) {
                let tags: String = self
                    .conn
                    .query_row("SELECT tags FROM memories WHERE id=?1", params![id], |r| r.get(0))
                    .unwrap_or_else(|_| "[]".into());
                let _ = self
                    .conn
                    .execute("DELETE FROM memories_fts WHERE rowid=?1", params![rowid]);
                let _ = self.conn.execute(
                    "INSERT INTO memories_fts (rowid,content,tags,kind,project) VALUES (?1,?2,?3,'credential',?4)",
                    params![rowid, Self::fts_index_content(&label), tags, project.clone().unwrap_or_default()],
                );
            }
            queue_embedding_job(
                &id,
                &label,
                &Self::embedding_input("credential", project.as_deref(), &created_at, &label),
            );
            sealed += 1;
        }
        Ok(sealed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn masks_secrets_keeps_labels_and_emails() {
        let label = masked_label(
            "Login admin La Villart: email contact@soflution.com / password: Hx9!kLm2pQ token sk-abc123def456ghi789",
        );
        assert!(label.starts_with(SEAL_PREFIX));
        assert!(label.contains("contact@soflution.com"));
        assert!(label.contains("Login admin La Villart"));
        assert!(!label.contains("Hx9!kLm2pQ"));
        assert!(!label.contains("sk-abc123def456ghi789"));
    }
}
