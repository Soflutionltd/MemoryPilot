//! Encryption at rest for `credential` memories.
//!
//! The SQLite file used to hold logins, keys and tokens in clear. A
//! sealed credential now stores:
//!   - `content`  = "🔒 " + a masked label (secrets replaced by ••••), so
//!     search, FTS and embeddings still find "login admin La Villart";
//!   - `metadata.enc` = base64(nonce ‖ XChaCha20-Poly1305(original)).
//!
//! Reads never decrypt: every tool sees the masked label, and a serialized
//! credential carries a `secret` pointer (where the value lives, how to ask
//! for it) instead of the value. `reveal_secret` is the only way out, after
//! the user confirms in a native dialog; each request is audited without
//! the value. `MEMORYPILOT_SECRETS=reveal` restores transparent decryption.
//!
//! The 256-bit key lives in the macOS login Keychain (service
//! "MemoryPilot", account "credentials-key-v2"), created on first use through the
//! Keychain Services API. On other systems it falls back to a 0600 key file
//! next to the database. If no key can be obtained, nothing is sealed
//! (never lose data because of a missing key).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

use base64::engine::general_purpose::STANDARD as B64;
use base64::Engine;
use chacha20poly1305::aead::{Aead, AeadCore, KeyInit, OsRng};
use chacha20poly1305::{XChaCha20Poly1305, XNonce};
use rusqlite::params;

use super::{queue_embedding_job, Database, Memory};

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

static REVEAL: AtomicBool = AtomicBool::new(false);
static REMOTE: AtomicBool = AtomicBool::new(false);

/// Transparent decryption on read (off by default).
pub(crate) fn set_reveal(on: bool) {
    REVEAL.store(on, Ordering::Relaxed);
}

/// The HTTP server: nobody sits in front of the machine to confirm a reveal.
pub(crate) fn set_remote(on: bool) {
    REMOTE.store(on, Ordering::Relaxed);
}

pub(crate) fn reveal_enabled() -> bool {
    REVEAL.load(Ordering::Relaxed)
}

/// What a read returns for a stored credential: the masked label unless
/// transparent decryption was explicitly turned on.
pub(crate) fn read_content(kind: &str, raw: String, metadata: Option<&serde_json::Value>) -> String {
    if kind != "credential" {
        return raw;
    }
    if reveal_enabled() {
        return unseal(&raw, metadata).unwrap_or(raw);
    }
    if raw.starts_with(SEAL_PREFIX) {
        raw
    } else {
        masked_label(&raw)
    }
}

/// Sealed row → original text, whatever the read mode.
pub(crate) fn unseal(content: &str, metadata: Option<&serde_json::Value>) -> Option<String> {
    if !content.starts_with(SEAL_PREFIX) {
        return None;
    }
    decrypt(metadata?.get("enc")?.as_str()?)
}

/// Metadata as shown to clients: the ciphertext stays inside.
fn public_metadata(metadata: &serde_json::Value) -> serde_json::Value {
    let mut shown = metadata.clone();
    if let Some(object) = shown.as_object_mut() {
        object.remove("enc");
    }
    shown
}

/// Where the value of a credential lives, from what its label says.
fn secret_location(label: &str, metadata: Option<&serde_json::Value>) -> String {
    if let Some(location) = metadata.and_then(|m| m.get("location")).and_then(|v| v.as_str()) {
        return location.to_string();
    }
    if let Some(reference) = label.split_whitespace().find(|token| token.starts_with("op://")) {
        return reference.trim_end_matches(['.', ',', ';', ')']).to_string();
    }
    if label.to_lowercase().contains("1password") {
        return "1Password (see the label)".into();
    }
    if metadata.and_then(|m| m.get("sealed")).and_then(|v| v.as_bool()) == Some(true) {
        "MemoryPilot vault (encrypted)".into()
    } else {
        "MemoryPilot (not encrypted: no vault key)".into()
    }
}

fn secret_pointer(id: &str, label: &str, metadata: Option<&serde_json::Value>) -> serde_json::Value {
    serde_json::json!({
        "where": secret_location(label, metadata),
        "reveal": { "tool": "memory_admin", "action": "reveal_secret", "args": { "id": id, "reason": "why the value is needed" } },
        "note": "Search, recall and get never return the value. Use the location when you can; reveal_secret asks the user to confirm on screen.",
    })
}

/// The credentials among some memories, as pointers (recall lists carry no id).
pub(crate) fn secret_references<'a>(memories: impl Iterator<Item = &'a Memory>) -> Option<serde_json::Value> {
    let mut seen = std::collections::HashSet::new();
    let items: Vec<serde_json::Value> = memories
        .filter(|memory| memory.kind == "credential" && !reveal_enabled() && seen.insert(memory.id.clone()))
        .map(|memory| {
            let label = if memory.content.starts_with(SEAL_PREFIX) {
                memory.content.clone()
            } else {
                masked_label(&memory.content)
            };
            serde_json::json!({ "id": memory.id, "where": secret_location(&label, memory.metadata.as_ref()), "label": label })
        })
        .collect();
    (!items.is_empty()).then(|| {
        serde_json::json!({
            "items": items,
            "reveal": "memory_admin reveal_secret {id, reason}: the user confirms on screen. Prefer the location.",
        })
    })
}

impl serde::Serialize for Memory {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let masked = self.kind == "credential" && !reveal_enabled();
        let content = if masked && !self.content.starts_with(SEAL_PREFIX) {
            std::borrow::Cow::Owned(masked_label(&self.content))
        } else {
            std::borrow::Cow::Borrowed(self.content.as_str())
        };
        let mut state = serializer.serialize_struct("Memory", 14)?;
        state.serialize_field("id", &self.id)?;
        state.serialize_field("content", content.as_ref())?;
        state.serialize_field("kind", &self.kind)?;
        state.serialize_field("project", &self.project)?;
        state.serialize_field("tags", &self.tags)?;
        state.serialize_field("source", &self.source)?;
        state.serialize_field("importance", &self.importance)?;
        match &self.expires_at {
            Some(expires_at) => state.serialize_field("expires_at", expires_at)?,
            None => state.skip_field("expires_at")?,
        }
        state.serialize_field("created_at", &self.created_at)?;
        state.serialize_field("updated_at", &self.updated_at)?;
        match &self.metadata {
            Some(metadata) => state.serialize_field("metadata", &public_metadata(metadata))?,
            None => state.skip_field("metadata")?,
        }
        if masked {
            state.serialize_field("secret", &secret_pointer(&self.id, &content, self.metadata.as_ref()))?;
        } else {
            state.skip_field("secret")?;
        }
        match &self.last_accessed_at {
            Some(at) => state.serialize_field("last_accessed_at", at)?,
            None => state.skip_field("last_accessed_at")?,
        }
        state.serialize_field("access_count", &self.access_count)?;
        state.end()
    }
}

/// How a reveal is confirmed: `dialog` (default on macOS) asks the user in a
/// native dialog an agent cannot click; `none` trusts the caller (audit only).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RevealConfirm {
    Dialog,
    None,
}

impl RevealConfirm {
    pub fn from_env() -> Result<Self, String> {
        match std::env::var("MEMORYPILOT_REVEAL_CONFIRM").ok().as_deref().map(str::trim) {
            None | Some("") | Some("dialog") => Ok(Self::Dialog),
            Some("none") => Ok(Self::None),
            Some(other) => Err(format!("MEMORYPILOT_REVEAL_CONFIRM={} is not valid: use dialog or none", other)),
        }
    }
}

const DIALOG_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);

#[cfg(target_os = "macos")]
fn ask_user(message: &str) -> Result<bool, String> {
    let osascript = std::path::Path::new("/usr/bin/osascript");
    if !osascript.exists() {
        return Err("osascript not found: no way to ask for confirmation".into());
    }
    // The text goes through argv, never into the script: no AppleScript injection.
    let mut command = std::process::Command::new(osascript);
    command.args([
        "-e",
        "on run argv",
        "-e",
        "display dialog (item 1 of argv) with title \"MemoryPilot\" buttons {\"Refuser\", \"Révéler\"} default button \"Refuser\" cancel button \"Refuser\" with icon caution giving up after 55",
        "-e",
        "end run",
        "--",
        message,
    ]);
    let answer = crate::connectors::run_with_timeout(command, None, DIALOG_TIMEOUT).unwrap_or_default();
    Ok(answer.contains("button returned:Révéler") && !answer.contains("gave up:true"))
}

#[cfg(not(target_os = "macos"))]
fn ask_user(_message: &str) -> Result<bool, String> {
    Err("no confirmation dialog on this system: set MEMORYPILOT_REVEAL_CONFIRM=none to allow reveals with audit only".into())
}

fn audit(memory: &Memory, reason: &str, outcome: &str) {
    let Some(dir) = dirs::home_dir().map(|home| home.join(".MemoryPilot").join("logs")) else {
        return;
    };
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }
    let line = serde_json::json!({
        "at": chrono::Utc::now().to_rfc3339(),
        "id": memory.id,
        "project": memory.project,
        "label": memory.content,
        "reason": reason,
        "outcome": outcome,
    });
    use std::io::Write;
    if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open(dir.join("reveal-audit.log")) {
        let _ = writeln!(file, "{}", line);
    }
}

impl Database {
    /// Clear text of a credential, read straight from the row (no confirmation:
    /// callers are internal, `reveal_secret` is the tool entry point).
    pub(crate) fn credential_clear_text(&self, id: &str) -> Result<Option<String>, String> {
        let row: Option<(String, Option<String>)> = self
            .conn
            .query_row(
                "SELECT content, metadata FROM memories WHERE id=?1 AND kind='credential'",
                params![id],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .ok();
        let Some((content, metadata)) = row else {
            return Ok(None);
        };
        if !content.starts_with(SEAL_PREFIX) {
            return Ok(Some(content));
        }
        let metadata: Option<serde_json::Value> = metadata.and_then(|m| serde_json::from_str(&m).ok());
        unseal(&content, metadata.as_ref())
            .map(Some)
            .ok_or_else(|| format!("Credential {} cannot be decrypted with this machine's key", id))
    }

    /// The value of a credential, after the user said yes. Every request is
    /// written to ~/.MemoryPilot/logs/reveal-audit.log, without the value.
    pub fn reveal_secret(&self, id: &str, reason: &str, confirm: RevealConfirm) -> Result<(Memory, String), String> {
        let reason = reason.trim();
        if reason.is_empty() {
            return Err("reason required: say why the value is needed".into());
        }
        let memory = self.get_memory(id)?.ok_or_else(|| format!("Not found: {}", id))?;
        if memory.kind != "credential" {
            return Err(format!("{} is a {}, not a credential: get_memory returns it in full", id, memory.kind));
        }
        if REMOTE.load(Ordering::Relaxed) && !reveal_enabled() {
            audit(&memory, reason, "refused_remote");
            return Err("Reveals are off on the HTTP server (MEMORYPILOT_HTTP_SECRETS=on allows them)".into());
        }
        if confirm == RevealConfirm::Dialog {
            let label = memory.content.trim_start_matches(SEAL_PREFIX);
            let message = format!(
                "Un agent demande la valeur de ce secret :\n\n{}\n\nProjet : {}\nMotif : {}",
                label.chars().take(200).collect::<String>(),
                memory.project.as_deref().unwrap_or("aucun"),
                reason.chars().take(300).collect::<String>(),
            );
            match ask_user(&message) {
                Ok(true) => {}
                Ok(false) => {
                    audit(&memory, reason, "refused");
                    return Err("The user did not confirm (refused or no answer within 55 s)".into());
                }
                Err(error) => {
                    audit(&memory, reason, "no_dialog");
                    return Err(error);
                }
            }
        }
        let value = self
            .credential_clear_text(id)?
            .ok_or_else(|| format!("Not found: {}", id))?;
        audit(&memory, reason, if confirm == RevealConfirm::Dialog { "revealed" } else { "revealed_unconfirmed" });
        Ok((memory, value))
    }
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
            let Ok(Some(clear)) = self.credential_clear_text(&id) else {
                report.undecryptable += 1;
                continue;
            };
            if !contains_secret_value(&clear) {
                let kind = crate::gc::auto_classify(&clear).1;
                report.reclassified.push((id.clone(), kind.to_string(), masked_label(&clear)));
                if apply {
                    self.update_memory_full(&id, Some(&clear), Some(kind), None, None, None, None)?;
                }
                continue;
            }
            let stored: String = self
                .conn
                .query_row("SELECT content FROM memories WHERE id=?1", params![id], |r| r.get(0))
                .map_err(|e| format!("repair read: {}", e))?;
            let label = masked_label(&clear);
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

    fn credential(content: &str, metadata: Option<serde_json::Value>) -> Memory {
        Memory {
            id: "c1".into(),
            content: content.into(),
            kind: "credential".into(),
            project: Some("demo".into()),
            tags: Vec::new(),
            source: "test".into(),
            importance: 5,
            expires_at: None,
            created_at: "2026-10-22T10:00:00+00:00".into(),
            updated_at: "2026-10-22T10:00:00+00:00".into(),
            metadata,
            last_accessed_at: None,
            access_count: 0,
        }
    }

    #[test]
    fn serialized_credentials_are_masked_with_a_pointer() {
        // Unsealed (no key) and sealed credentials alike.
        let clear = serde_json::to_value(credential("Admin demo : password: Hx9!kLm2pQ op://Soflution/demo", None)).unwrap();
        assert!(!clear.to_string().contains("Hx9!kLm2pQ"));
        assert_eq!(clear["secret"]["where"], "op://Soflution/demo");
        assert_eq!(clear["secret"]["reveal"]["action"], "reveal_secret");

        let sealed = serde_json::to_value(credential(
            &masked_label("Admin demo : password: Hx9!kLm2pQ"),
            Some(serde_json::json!({ "enc": encrypt("Admin demo : password: Hx9!kLm2pQ").unwrap(), "sealed": true, "location": "1Password, Demo" })),
        ))
        .unwrap();
        assert!(sealed["metadata"].get("enc").is_none(), "the ciphertext stays inside");
        assert_eq!(sealed["metadata"]["sealed"], true);
        assert_eq!(sealed["secret"]["where"], "1Password, Demo");

        let mut fact = credential("Planify uses SvelteKit", None);
        fact.kind = "fact".into();
        assert!(serde_json::to_value(fact).unwrap().get("secret").is_none());
    }

    #[test]
    fn reads_return_the_label_and_unseal_needs_the_ciphertext() {
        let plain = "password: Hx9!kLm2pQ";
        let label = masked_label(plain);
        let metadata = serde_json::json!({ "enc": encrypt(plain).unwrap() });
        assert_eq!(read_content("credential", label.clone(), Some(&metadata)), label);
        assert!(!read_content("credential", plain.into(), None).contains("Hx9!kLm2pQ"));
        assert_eq!(read_content("fact", plain.into(), None), plain);
        assert_eq!(unseal(&label, Some(&metadata)).as_deref(), Some(plain));
    }

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
