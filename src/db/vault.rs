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
    // Tests never touch the Keychain: an unsigned test binary reading the
    // item would make macOS show an access prompt and block.
    #[cfg(test)]
    {
        KEY.get_or_init(|| {
            let generated = XChaCha20Poly1305::generate_key(&mut OsRng);
            let mut key = [0u8; 32];
            key.copy_from_slice(&generated);
            Some(key)
        })
        .as_ref()
    }
    #[cfg(not(test))]
    {
        KEY.get_or_init(load_or_create_key).as_ref()
    }
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
    "password", "passwd", "pwd", "mdp", "pass", "passe", "token", "secret", "key", "clé", "cle",
];

/// Words announcing a password; the value usually follows within two tokens.
const PASSWORD_KEYWORDS: &[&str] = &["password", "passwd", "pwd", "mdp", "pass", "passe"];

/// After a password keyword, these say where the secret lives, not what it is.
const SECRET_REFERENCES: &[&str] = &[
    "1password", "op-sof", "trousseau", "keychain", "stocké", "stockée", "stored", "dans", "in", "via",
    "voir", "see", "env", "variable", "secret", "manager", "••••", "redacted", "masqué", "oublié",
    "reset", "réinitialisation", "requis", "required", "de", "du", "à", "a", "le", "la", "the", "is",
    "est", "no", "non", "aucun", "sans", "changé", "changed", "et", "and",
];

const PASSWORD_SYMBOLS: &[char] = &['!', '@', '#', '$', '%', '^', '&', '*', '?', '+', '=', '~', '§', '€', '£'];

fn is_email(core: &str) -> bool {
    match core.split_once('@') {
        Some((user, domain)) => {
            !user.is_empty()
                && domain.contains('.')
                && !domain.ends_with('.')
                && user.chars().chain(domain.chars()).all(|c| c.is_alphanumeric() || "._+-".contains(c))
        }
        None => false,
    }
}

pub(crate) fn looks_secret(token: &str) -> bool {
    let core = token.trim_matches(|c: char| !c.is_alphanumeric() && !PASSWORD_SYMBOLS.contains(&c));
    if is_email(core) || core.contains("://") {
        return false; // keep emails and URLs: they are what you search for
    }
    let len = core.chars().count();
    let digits = core.chars().filter(|c| c.is_ascii_digit()).count();
    let letters = core.chars().filter(|c| c.is_alphabetic()).count();
    let has_upper = core.chars().any(|c| c.is_uppercase());
    let has_lower = core.chars().any(|c| c.is_lowercase());
    let has_symbol = core.chars().any(|c| PASSWORD_SYMBOLS.contains(&c));
    let classes = [has_upper, has_lower, digits > 0, has_symbol].iter().filter(|b| **b).count();
    // Typical password ("Hx9!kLm2pQ", "0550…aA@!"): short but mixes a symbol with 2 other classes.
    let password_like = len >= 8 && has_symbol && classes >= 3;
    (len >= 16 && digits > 0 && letters > 0 && !core.contains('/'))
        || password_like
        || core.starts_with("sk-")
        || core.starts_with("sk_")
        || core.starts_with("ghp_")
        || core.starts_with("github_pat_")
        || core.starts_with("xox")
        || (core.starts_with("eyJ") && core.matches('.').count() >= 2)
}

/// True when the text holds a secret VALUE (key, token, password), not
/// merely talks about secrets ("secrets Cloudflare", "clé dans 1Password").
pub(crate) fn contains_secret_value(content: &str) -> bool {
    let tokens: Vec<&str> = content.split_whitespace().collect();
    for (index, token) in tokens.iter().enumerate() {
        if looks_secret(token) {
            return true;
        }
        let lower = token.to_lowercase();
        let word = lower.trim_matches(|c: char| !c.is_alphanumeric());
        let announces_password = PASSWORD_KEYWORDS.contains(&word)
            || (word == "passe" && index > 0 && tokens[index - 1].to_lowercase() == "de");
        if !announces_password {
            continue;
        }
        let mut assigned = token.ends_with(':') || token.ends_with('=');
        let mut value: Option<&str> = None;
        for next in tokens[index + 1..].iter().take(3) {
            let trimmed = next.trim_matches(|c: char| c == ',' || c == ';');
            if matches!(trimmed, ":" | "=" | "/" | "-" | "→") {
                assigned = true;
                continue;
            }
            let cleaned = trimmed.trim_start_matches([':', '=']);
            if cleaned.len() != trimmed.len() {
                assigned = true;
            }
            if !cleaned.is_empty() {
                value = Some(cleaned);
            }
            break;
        }
        let Some(value) = value else { continue };
        let value_lower = value.to_lowercase();
        let bare = value_lower.trim_matches(|c: char| !c.is_alphanumeric() && c != '•');
        let is_reference = SECRET_REFERENCES.contains(&bare)
            || value_lower.starts_with("op://")
            || value_lower.starts_with('$')
            || value_lower.starts_with('<')
            || value_lower.starts_with('(')
            || value_lower.starts_with('•')
            || value_lower.starts_with('[');
        let core = value.trim_end_matches(['.', ',', ')']);
        let mixed = core.chars().any(|c| c.is_ascii_digit() || PASSWORD_SYMBOLS.contains(&c))
            || (core.chars().any(|c| c.is_uppercase()) && core.chars().skip(1).any(|c| c.is_uppercase()) && core.chars().any(|c| c.is_lowercase()));
        if !is_reference && core.chars().count() >= 6 && (assigned || mixed) {
            return true;
        }
    }
    false
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

    /// Repairs credentials sealed by older versions: re-masks labels that kept a
    /// password in clear, and gives back their real kind to memories that hold no
    /// secret value (they were hidden from safe recall). Dry run unless `apply`.
    pub fn repair_credentials(&self, apply: bool) -> Result<CredentialRepair, String> {
        if key().is_none() {
            return Err("no encryption key available: nothing can be checked".into());
        }
        let ids: Vec<String> = {
            let mut stmt = self
                .conn
                .prepare("SELECT id FROM memories WHERE kind = 'credential' ORDER BY created_at")
                .map_err(|e| format!("repair prepare: {}", e))?;
            let rows = stmt.query_map([], |r| r.get(0)).map_err(|e| format!("repair query: {}", e))?;
            rows.flatten().collect()
        };
        let mut report = CredentialRepair::default();
        for id in ids {
            let Some(memory) = self.get_memory(&id)? else { continue };
            if memory.content.starts_with(SEAL_PREFIX) {
                report.undecryptable += 1;
                continue;
            }
            if !contains_secret_value(&memory.content) {
                let kind = crate::gc::auto_classify(&memory.content).1;
                report.reclassified.push((id.clone(), kind.to_string(), masked_label(&memory.content)));
                if apply {
                    self.update_memory_full(&id, None, Some(kind), None, None, None, None)?;
                }
                continue;
            }
            let stored: String = self
                .conn
                .query_row("SELECT content FROM memories WHERE id=?1", params![id], |r| r.get(0))
                .map_err(|e| format!("repair read: {}", e))?;
            let label = masked_label(&memory.content);
            if stored != label {
                report.relabelled += 1;
                if apply {
                    self.replace_sealed_label(&id, &label, memory.project.as_deref(), &memory.created_at)?;
                }
            }
        }
        if apply && (report.relabelled > 0 || !report.reclassified.is_empty()) {
            // Old labels may survive in free pages of the file.
            self.conn.execute_batch("VACUUM;").map_err(|e| format!("vacuum: {}", e))?;
        }
        Ok(report)
    }

    fn replace_sealed_label(&self, id: &str, label: &str, project: Option<&str>, created_at: &str) -> Result<(), String> {
        self.conn
            .execute(
                "UPDATE memories SET content=?1, embedding=NULL, content_hash=?2 WHERE id=?3",
                params![label, super::content_hash(label), id],
            )
            .map_err(|e| format!("relabel update: {}", e))?;
        let (rowid, tags): (i64, String) = self
            .conn
            .query_row("SELECT rowid, tags FROM memories WHERE id=?1", params![id], |r| Ok((r.get(0)?, r.get(1)?)))
            .map_err(|e| format!("relabel read: {}", e))?;
        self.conn
            .execute("DELETE FROM memories_fts WHERE rowid=?1", params![rowid])
            .map_err(|e| format!("relabel fts delete: {}", e))?;
        self.conn
            .execute(
                "INSERT INTO memories_fts (rowid,content,tags,kind,project) VALUES (?1,?2,?3,'credential',?4)",
                params![rowid, Database::fts_index_content(label), tags, project.unwrap_or_default()],
            )
            .map_err(|e| format!("relabel fts insert: {}", e))?;
        queue_embedding_job(id, label, &Database::embedding_input("credential", project, created_at, label));
        Ok(())
    }
}

#[derive(Debug, Default)]
pub struct CredentialRepair {
    /// (id, new kind, masked label) of memories that held no secret value.
    pub reclassified: Vec<(String, String, String)>,
    pub relabelled: usize,
    pub undecryptable: usize,
}

/// Seal the credentials of a standalone SQLite file (an old backup)
/// without migrating its schema, then VACUUM so no clear page remains.
pub(crate) fn seal_file(path: &std::path::Path) -> Result<usize, String> {
    if key().is_none() {
        return Err("no encryption key available".into());
    }
    let conn = rusqlite::Connection::open(path).map_err(|e| format!("open {}: {}", path.display(), e))?;
    let columns: Vec<String> = {
        let mut stmt = conn.prepare("PRAGMA table_info(memories)").map_err(|e| e.to_string())?;
        let cols = stmt.query_map([], |r| r.get::<_, String>(1)).map_err(|e| e.to_string())?;
        cols.flatten().collect()
    };
    if !columns.iter().any(|c| c == "content") {
        return Err("not a MemoryPilot database".into());
    }
    let has = |name: &str| columns.iter().any(|c| c == name);
    let has_fts: bool = conn
        .query_row("SELECT count(*) FROM sqlite_master WHERE name='memories_fts'", [], |r| r.get::<_, i64>(0))
        .map(|n| n > 0)
        .unwrap_or(false);
    let rows: Vec<(i64, String, Option<String>, String, Option<String>)> = {
        let mut stmt = conn
            .prepare("SELECT rowid, content, metadata, tags, project FROM memories WHERE kind = 'credential'")
            .map_err(|e| e.to_string())?;
        let rows = stmt
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)))
            .map_err(|e| e.to_string())?;
        rows.flatten().collect()
    };
    let mut sealed = 0usize;
    for (rowid, content, metadata, tags, project) in rows {
        if content.starts_with(SEAL_PREFIX) {
            continue;
        }
        let Some(enc) = encrypt(&content) else { continue };
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
        conn.execute(
            "UPDATE memories SET content=?1, metadata=?2 WHERE rowid=?3",
            params![label, meta_json, rowid],
        )
        .map_err(|e| format!("seal update: {}", e))?;
        if has("embedding") {
            let _ = conn.execute("UPDATE memories SET embedding=NULL WHERE rowid=?1", params![rowid]);
        }
        if has("content_hash") {
            let _ = conn.execute(
                "UPDATE memories SET content_hash=?1 WHERE rowid=?2",
                params![super::content_hash(&label), rowid],
            );
        }
        if has_fts {
            let _ = conn.execute("DELETE FROM memories_fts WHERE rowid=?1", params![rowid]);
            let _ = conn.execute(
                "INSERT INTO memories_fts (rowid,content,tags,kind,project) VALUES (?1,?2,?3,'credential',?4)",
                params![rowid, Database::fts_index_content(&label), tags, project.unwrap_or_default()],
            );
        }
        sealed += 1;
    }
    conn.execute_batch("VACUUM;").map_err(|e| format!("vacuum: {}", e))?;
    Ok(sealed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encrypt_decrypt_roundtrip() {
        let sealed = encrypt("password: Hx9!kLm2pQ").unwrap();
        assert_eq!(decrypt(&sealed).as_deref(), Some("password: Hx9!kLm2pQ"));
    }

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

    // Real leaks found on 2026-10-21 (values replaced): short passwords after
    // "pass", after "email /", or containing '@' survived the mask in clear.
    #[test]
    fn masks_short_passwords_with_symbols() {
        for content in [
            "Compte comptable de TEST : email test@example.fr / pass 12345678aA@! auth ok",
            "Compte admin : contact@example.com / 0987654321aA@! user_id 42",
            "Compte gratuit Agency : agency@example.com / X7q!mR2z-Kpa9. workspace Motion",
        ] {
            let label = masked_label(content);
            for secret in ["12345678aA@!", "0987654321aA@!", "X7q!mR2z-Kpa9"] {
                assert!(!label.contains(secret), "{} leaked in {}", secret, label);
            }
            assert!(label.contains("@example."), "emails stay searchable: {}", label);
        }
    }

    #[test]
    fn secret_value_detection_ignores_mere_mentions() {
        for prose in [
            "Secrets : jamais dans git, valeur de référence dans 1Password via op-sof (coffre Soflution).",
            "Clé Anthropic : op://Soflution/yquxzdeotwopqmgf5iyzpoqgry/credential",
            "Le catalogue d'outils pèse ~90k tokens par tour, coût 0,05 $ par run.",
            "Mot de passe oublié : flux de réinitialisation par e-mail.",
            "Le mot de passe actuel est exigé avant de reprendre une place.",
            "Secrets Cloudflare posés sur le projet docix (RESEND_API_KEY, POWENS_DOMAIN).",
            "Mews démo: login antoine@example.fr. MDP stocké 1Password.",
            "Team ID = TB8CC687M3, Bundle ID = com.soflution.prive, commit 6052ff30",
            "Docix MCP déployé (commit b9c6836), projet Supabase lpspvwjybttpueaydwmp.",
        ] {
            assert!(!contains_secret_value(prose), "false positive: {}", prose);
        }
    }

    #[test]
    fn secret_value_detection_finds_real_values() {
        for secret in [
            "password: villacalvi",
            "mot de passe : Calvi2026",
            "Compte démo : demo@example.com / pass 12345678aA@!",
            "Resend key re_AbCdEf123456GhIjKl789",
            "GitHub token ghp_abcdefghijklmnopqrstuvwxyz0123",
            "MDP = soleil-corse",
        ] {
            assert!(contains_secret_value(secret), "missed: {}", secret);
        }
    }
}
