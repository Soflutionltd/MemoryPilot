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

pub const SEAL_PREFIX: &str = "🔒 ";
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
    // errSecItemNotFound: the only case where a new key may be created. A
    // refused or failed read must never replace the key that sealed the data.
    const ITEM_NOT_FOUND: i32 = -25300;
    match get_generic_password(KEYCHAIN_SERVICE, KEYCHAIN_ACCOUNT) {
        Ok(found) => return decode_key(&String::from_utf8_lossy(&found)),
        Err(error) if error.code() == ITEM_NOT_FOUND => {}
        Err(error) => {
            eprintln!("[MemoryPilot] Vault: Keychain read failed ({}), key left untouched.", error.code());
            return None;
        }
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
    if path.exists() {
        // Never replace an existing key file, even an unreadable one.
        return decode_key(&std::fs::read_to_string(&path).ok()?);
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
    Some(B64.encode(encrypt_bytes(key()?, plain.as_bytes())?))
}

pub(crate) fn decrypt(encoded: &str) -> Option<String> {
    decrypt_with(key()?, encoded)
}

fn decrypt_with(key: &[u8; 32], encoded: &str) -> Option<String> {
    String::from_utf8(decrypt_bytes(key, &B64.decode(encoded).ok()?)?).ok()
}

/// XChaCha20-Poly1305 of `plain` (24-byte nonce then ciphertext).
pub fn encrypt_bytes(key: &[u8; 32], plain: &[u8]) -> Option<Vec<u8>> {
    let cipher = XChaCha20Poly1305::new(key.into());
    let nonce = XChaCha20Poly1305::generate_nonce(&mut OsRng);
    let ct = cipher.encrypt(&nonce, plain).ok()?;
    let mut blob = nonce.to_vec();
    blob.extend_from_slice(&ct);
    Some(blob)
}

pub fn decrypt_bytes(key: &[u8; 32], blob: &[u8]) -> Option<Vec<u8>> {
    if blob.len() < 24 {
        return None;
    }
    let (nonce, ct) = blob.split_at(24);
    let cipher = XChaCha20Poly1305::new(key.into());
    cipher.decrypt(XNonce::from_slice(nonce), ct).ok()
}

/// Searchable masked label plus the ciphertext (base64) for `metadata.enc`.
pub fn seal_text(key: &[u8; 32], plain: &str) -> Option<(String, String)> {
    Some((masked_label(plain), B64.encode(encrypt_bytes(key, plain.as_bytes())?)))
}

/// Wraps a tenant data key with the server master key.
pub fn wrap_key(master: &[u8; 32], dek: &[u8; 32]) -> Option<Vec<u8>> {
    encrypt_bytes(master, dek)
}

pub fn unwrap_key(master: &[u8; 32], wrapped: &[u8]) -> Option<[u8; 32]> {
    decrypt_bytes(master, wrapped)?.try_into().ok()
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
    shown_label(&raw)
}

/// A stored credential as shown: the label is masked again on every read, so
/// a label written by an older, looser mask never shows its secret.
fn shown_label(stored: &str) -> String {
    masked_label(stored.trim_start_matches(SEAL_PREFIX))
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
            let label = shown_label(&memory.content);
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
        let content = if masked {
            std::borrow::Cow::Owned(shown_label(&self.content))
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
    /// The value comes from 1Password when the credential points there, else
    /// from the local encrypted copy; the source is returned with it.
    pub fn reveal_secret(&self, id: &str, reason: &str, confirm: RevealConfirm) -> Result<(Memory, String, &'static str), String> {
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
        let location = memory.metadata.as_ref().and_then(|m| m.get("location")).and_then(|v| v.as_str());
        let from_onepassword = location
            .filter(|l| l.starts_with("op://"))
            .and_then(|reference| crate::onepassword::read(reference).ok());
        let (value, source) = match from_onepassword {
            Some(value) => (value, "1password"),
            None => (
                self.credential_clear_text(id)?.ok_or_else(|| format!("Not found: {}", id))?,
                "memorypilot",
            ),
        };
        audit(&memory, reason, if confirm == RevealConfirm::Dialog { "revealed" } else { "revealed_unconfirmed" });
        Ok((memory, value, source))
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

/// Words naming a public identifier: the value next to them is looked up, not secret.
const IDENTIFIER_WORDS: &[&str] = &[
    "id", "ids", "sid", "uuid", "guid", "account", "zone", "tag", "team", "project", "projet", "folder",
    "dossier", "parent", "doc", "kv", "d1", "namespace", "bucket", "database", "commit", "sha", "hash",
    "ref", "item", "build", "workspace", "org", "ruleset", "rule",
];

/// Words that make a nearby identifier word irrelevant: the value is a secret.
const SECRET_CONTEXT_WORDS: &[&str] = &["bearer", "credential", "credentials", "apikey", "signing"];

/// Public object ids that only point at a resource (Stripe, Vercel).
const PUBLIC_ID_PREFIXES: &[&str] = &["acct_", "prod_", "price_", "plink_", "cus_", "sub_", "evt_", "we_", "prj_", "team_"];

/// Prefixes of known credential formats: secret whatever the surrounding words say.
const SECRET_PREFIXES: &[&str] = &[
    "sk-", "sk_", "rk_live_", "rk_test_", "whsec_", "ghp_", "gho_", "github_pat_", "glpat-", "sbp_", "xox", "AKIA", "AIza",
];

const FILE_EXTENSIONS: &[&str] = &[
    "p8", "p12", "pem", "pdf", "json", "rs", "ts", "js", "py", "dart", "swift", "svelte", "sql", "sh", "md", "txt",
    "png", "jpg", "jpeg", "webp", "svg", "zip", "dmg", "app", "csv", "yml", "yaml", "toml", "html", "css", "mp4",
    "mov", "log", "plist", "ipa", "apk", "aab", "tar", "gz", "icc", "icm",
];

/// A known prefix followed by a value: the prefix alone (`sk_live`, « AIza ») is a mention.
fn has_secret_prefix(core: &str) -> bool {
    SECRET_PREFIXES
        .iter()
        .any(|prefix| core.strip_prefix(prefix).is_some_and(|rest| rest.chars().filter(|c| c.is_alphanumeric()).count() >= 12))
        || (core.starts_with("eyJ") && core.matches('.').count() >= 2)
}

/// An amount or a measure (`19,99€/mois`, `~$20/mo`, `£500`, `~50-60ms`).
fn is_quantity(core: &str) -> bool {
    let core = core.trim_start_matches(['~', '≈', '<', '>', '+', '±']);
    let (amount, unit) = core.split_once('/').unwrap_or((core, ""));
    if !unit.chars().all(char::is_alphabetic) {
        return false;
    }
    let amount = amount.trim_matches(['$', '€', '£', '¥', '%']);
    let number = amount.trim_end_matches(char::is_alphabetic);
    amount.chars().count() - number.chars().count() <= 3
        && number.starts_with(|c: char| c.is_ascii_digit())
        && number.chars().all(|c| c.is_ascii_digit() || ".,-".contains(c))
}

/// A benchmark result or a conversion (`R@5=97.4%`, `R@5=62.4%->68.8%`, `80%=12MB`).
fn is_metric(core: &str) -> bool {
    let operators = ['=', '<', '>', ':'];
    let mut parts = core.split(operators).filter(|part| !part.is_empty());
    let first = parts.next().unwrap_or_default();
    core.contains(operators)
        && core.rsplit(operators).next().is_some_and(is_quantity)
        && (core.ends_with('%') || is_quantity(first))
}

/// An Apple app identifier, Team ID then bundle (`AB12CD345E.com.soflution.app`).
fn is_apple_app_id(core: &str) -> bool {
    let core = core.rsplit(':').next().unwrap_or(core); // `applinks:…`, `webcredentials:…`
    core.split_once('.').is_some_and(|(team, bundle)| {
        team.len() == 10
            && team.chars().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit())
            && bundle.contains('.')
            && bundle.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || ".-".contains(c))
    })
}

/// A package with its version (`mcp-remote@0.15.1`, `@scope/pkg@1.2.0`, `actions/checkout@v4`).
fn is_package_spec(core: &str) -> bool {
    let Some((name, version)) = core.rsplit_once('@') else { return false };
    let name = name.strip_prefix('@').unwrap_or(name);
    let version = version.trim_start_matches(['^', '~', 'v']);
    !name.is_empty()
        && name.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || "-_./".contains(c))
        && version.starts_with(|c: char| c.is_ascii_digit())
        && version.chars().all(|c| c.is_ascii_alphanumeric() || ".-+".contains(c))
}

fn is_file_name(core: &str) -> bool {
    core.rsplit_once('.')
        .is_some_and(|(stem, extension)| !stem.is_empty() && FILE_EXTENSIONS.contains(&extension.to_ascii_lowercase().as_str()))
}

/// Splits capitalised words, each optionally followed by digits (`OnlystParrains`,
/// `Qwen3Attention`, `buildStripeCheckoutUrl2024`), into (words, letters); a number
/// gives (0, 0). None when the part does not follow this pattern, as random keys
/// do within a few characters.
fn camel_words(part: &str) -> Option<(usize, usize)> {
    let chars: Vec<char> = part.chars().collect();
    if chars.is_empty() {
        return None;
    }
    if chars.iter().all(|c| c.is_ascii_digit()) {
        return Some((0, 0));
    }
    let (mut words, mut letters, mut index) = (0, 0, 0);
    while index < chars.len() {
        if chars[index].is_ascii_uppercase() {
            index += 1;
        } else if index > 0 {
            return None;
        }
        let start = index;
        while index < chars.len() && chars[index].is_ascii_lowercase() {
            index += 1;
        }
        if index == start {
            return None;
        }
        words += 1;
        letters += index - start + usize::from(start > 0 && chars[start - 1].is_ascii_uppercase());
        while index < chars.len() && chars[index].is_ascii_digit() {
            index += 1;
        }
    }
    Some((words, letters))
}

/// Real words average three letters or more; `AbCdEf12GhIj` does not.
fn made_of_words<'a>(parts: impl IntoIterator<Item = &'a str>) -> bool {
    let mut total = (0, 0);
    for part in parts {
        let Some((words, letters)) = camel_words(part) else { return false };
        total = (total.0 + words, total.1 + letters);
    }
    total.1 >= 3 * total.0
}

/// Lowercase words of a label, split on punctuation and camelCase (`(zoneId` -> zone, id).
fn label_words(token: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut current = String::new();
    let mut previous_lower = false;
    for c in token.chars() {
        if !c.is_alphanumeric() {
            if !current.is_empty() {
                words.push(std::mem::take(&mut current));
            }
            previous_lower = false;
            continue;
        }
        if c.is_uppercase() && previous_lower && !current.is_empty() {
            words.push(std::mem::take(&mut current));
        }
        current.extend(c.to_lowercase());
        previous_lower = c.is_lowercase() || c.is_ascii_digit();
    }
    if !current.is_empty() {
        words.push(current);
    }
    words
}

/// True when the words just before `index` (same clause, three tokens at most)
/// announce an identifier ("Account ID CF:", "(zoneId", "KV FLIGHTS_KV id")
/// and none of them announces a secret.
fn names_identifier(tokens: &[&str], index: usize) -> bool {
    let mut words = Vec::new();
    for token in tokens[..index].iter().rev().take(3) {
        if !words.is_empty() && token.ends_with(['.', ';', ',']) {
            break;
        }
        words.extend(label_words(token));
    }
    let secret = words.iter().any(|word| {
        SECRET_KEYWORDS.contains(&word.as_str()) || SECRET_CONTEXT_WORDS.contains(&word.as_str())
    });
    !secret && words.iter().any(|word| IDENTIFIER_WORDS.contains(&word.as_str()))
}

/// Words joined by '+' or '/' (`BM25+vector`, `iOS+Android`): every part is
/// letters then at most digits, and one is a lowercase word of 4+ letters.
fn joined_words(core: &str) -> bool {
    let parts: Vec<&str> = core.split(['+', '/']).filter(|part| !part.is_empty()).collect();
    parts.len() >= 2
        && parts.iter().all(|part| {
            let letters = part.trim_end_matches(|c: char| c.is_ascii_digit());
            !letters.is_empty() && letters.chars().all(|c| c.is_ascii_alphabetic())
        })
        && parts
            .iter()
            .any(|part| part.len() >= 4 && part.chars().all(|c| c.is_ascii_lowercase()))
}

pub(crate) fn looks_secret(token: &str) -> bool {
    let core = token.trim_matches(|c: char| !c.is_alphanumeric() && !PASSWORD_SYMBOLS.contains(&c));
    let is_path = core.starts_with("~/")
        || core.starts_with("./")
        || core.starts_with("../")
        || (core.starts_with('/') && core[1..].contains('/'));
    if is_email(core) || core.contains("://") || is_path {
        return false; // keep emails, URLs and file paths: they are what you search for
    }
    // A setting (`SameSite=Lax`, `max-age=60`, `API_KEY=…`): only its value can be secret.
    if let Some((key, value)) = core.split_once('=') {
        let setting_name = key.chars().count() >= 2 && key.chars().all(|c| c.is_alphabetic() || c == '-' || c == '_');
        let value = value.trim_matches(['"', '\'', '`']);
        if setting_name && !value.is_empty() {
            let name = key.to_lowercase();
            if ["id", "sid", "uuid", "guid"].iter().any(|word| name.ends_with(word)) {
                return has_secret_prefix(value); // `projectId=prj_…`, `CLOUDFLARE_ACCOUNT_ID=…`
            }
            // `license_key=<uuid>`, `client_secret=…`: the name says it, whatever the value looks
            // like, except a plain word (`key=roompilot` names a project).
            let names_password = ["password", "passwd", "pwd"].iter().any(|word| name.ends_with(word));
            let names_secret = ["key", "secret", "token"].iter().any(|word| name.ends_with(word));
            let plain_word = value.chars().all(|c| c.is_lowercase() || "-_:.".contains(c));
            let reference = value.starts_with(['$', '<', '(', '[', '•'])
                || ["env:", "op://", "secret:"].iter().any(|scheme| value.starts_with(scheme));
            if (names_password || (names_secret && !plain_word)) && !reference && value.chars().count() >= 8 {
                return true;
            }
            return looks_secret(value);
        }
    }
    if has_secret_prefix(core) {
        return true;
    }
    // Code, query strings, globs, prices, measures, metrics, package versions and file names.
    if core.contains(['(', ')', '[', ']', '{', '}', '<', '>', ',', '\'', '"', '`', '→'])
        || (core.contains('?') && core.contains('='))
        || core.starts_with("*.")
        || is_apple_app_id(core)
        || is_quantity(core)
        || is_metric(core)
        || is_package_spec(core)
        || is_file_name(core)
        || PUBLIC_ID_PREFIXES.iter().any(|prefix| core.starts_with(prefix))
    {
        return false;
    }
    let len = core.chars().count();
    let digits = core.chars().filter(|c| c.is_ascii_digit()).count();
    let letters = core.chars().filter(|c| c.is_alphabetic()).count();
    let has_upper = core.chars().any(|c| c.is_uppercase());
    let has_lower = core.chars().any(|c| c.is_lowercase());
    let has_symbol = core.chars().any(|c| PASSWORD_SYMBOLS.contains(&c));
    let classes = [has_upper, has_lower, digits > 0, has_symbol].iter().filter(|b| **b).count();
    // Typical password ("Hx9!kLm2pQ", "0550…aA@!"): short but mixes a symbol, a digit
    // and another class. Words joined by '+' or '/' (`BM25+vector`), paths, decimals
    // (`targetQuality*0.15`) and long hyphenated names are not one.
    let chars: Vec<char> = core.chars().collect();
    let decimal = chars
        .windows(3)
        .any(|w| w[0].is_ascii_digit() && (w[1] == '.' || w[1] == ',') && w[2].is_ascii_digit());
    let password_like = len >= 8
        && has_symbol
        && digits > 0
        && letters >= 2
        && classes >= 3
        && !core.contains('/')
        && !decimal
        && core.matches('-').count() < 3
        && !joined_words(core);
    let separators = ['-', '_', '.', ':'];
    let word_like = made_of_words(core.split(separators).filter(|part| !part.is_empty()));
    // Long random token. Lowercase words joined by separators
    // (`multilingual-e5-small`, `x86_64-unknown-linux-musl`) and camelCase names are not keys.
    let separated = core.contains(separators);
    let random_token = len >= 16
        && digits > 0
        && letters > 0
        && !core.contains('/')
        && !decimal
        && !word_like
        && (!separated || (has_upper && has_lower));
    // A long random part behind a prefix, even in lowercase (`sbp_<40 hex>`, `glpat-…`).
    let random_part = !core.contains('/') && core.split(separators).any(|part| {
        part.chars().count() >= 20
            && part.chars().any(|c| c.is_ascii_digit())
            && part.chars().any(|c| c.is_alphabetic())
            && !made_of_words([part])
    });
    random_token || random_part || password_like
}

/// True when the text holds a secret VALUE (key, token, password), not
/// merely talks about secrets ("secrets Cloudflare", "clé dans 1Password").
pub(crate) fn contains_secret_value(content: &str) -> bool {
    find_secret_value(content).is_some()
}

/// The first secret value of the text (best effort: the full text stays the reference).
pub(crate) fn find_secret_value(content: &str) -> Option<String> {
    let tokens: Vec<&str> = content.split_whitespace().collect();
    for (index, token) in tokens.iter().enumerate() {
        if looks_secret(token) {
            let core = token.trim_matches(|c: char| !c.is_alphanumeric() && !PASSWORD_SYMBOLS.contains(&c));
            if !has_secret_prefix(core) && !core.contains('=') && names_identifier(&tokens, index) {
                continue;
            }
            return Some(core.to_string());
        }
        let lower = token.to_lowercase();
        let word = lower.trim_matches(|c: char| !c.is_alphanumeric());
        if !PASSWORD_KEYWORDS.contains(&word) {
            continue;
        }
        // "pass"/"passe" alone are also verbs ("le ViewModel passe targetQuality").
        let after_de = index > 0 && tokens[index - 1].to_lowercase() == "de";
        let verb_form = matches!(word, "pass" | "passe") && !after_de;
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
        if core.contains(',') && !core.chars().any(char::is_alphabetic) {
            continue; // screen coordinates ("mot de passe 402,519")
        }
        let mixed = core.chars().any(|c| c.is_ascii_digit() || PASSWORD_SYMBOLS.contains(&c))
            || (core.chars().any(|c| c.is_uppercase()) && core.chars().skip(1).any(|c| c.is_uppercase()) && core.chars().any(|c| c.is_lowercase()));
        if !is_reference && core.chars().count() >= 6 && (assigned || (mixed && !verb_form)) {
            return Some(core.to_string());
        }
    }
    None
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
        // A value may end like a keyword (`sk_live_secret`): only the bare word stops the mask.
        let bare_keyword = SECRET_KEYWORDS.contains(&lower.trim_matches(|c: char| !c.is_alphanumeric()));
        if mask_next > 0 && !bare_keyword {
            if matches!(lower.as_str(), ":" | "=" | "/" | "-" | "→" | "is" | "est" | "was" | "vaut") {
                out.push(token.to_string());
                continue;
            }
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
    let mut label = out.join(" ");
    if let Some(value) = find_secret_value(content) {
        label = label.replace(&value, "••••");
    }
    let label: String = label.chars().take(220).collect();
    format!("{}{}", SEAL_PREFIX, label)
}

impl Database {
    /// Reads the vault key on a background thread when the database holds
    /// credentials, so a Keychain prompt shows at startup without blocking
    /// requests. Without credentials the Keychain is never read.
    pub fn warm_vault_key(&self) {
        let has_credentials = self
            .conn
            .query_row("SELECT EXISTS(SELECT 1 FROM memories WHERE kind = 'credential')", [], |r| r.get::<_, bool>(0))
            .unwrap_or(false);
        if has_credentials {
            std::thread::spawn(|| {
                let _ = key();
            });
        }
    }

    /// Encrypt every credential still stored in clear. Idempotent. Waits
    /// for the vault key: for a request that just stored a credential.
    pub fn seal_credentials(&self) -> Result<usize, String> {
        self.seal_in_clear(true)
    }

    /// Same, but only once the key is loaded: startup and background passes
    /// never wait on a Keychain prompt. What is left is sealed by the next
    /// pass after `warm_vault_key` has loaded the key.
    pub fn seal_pending_credentials(&self) -> Result<usize, String> {
        self.seal_in_clear(false)
    }

    fn seal_in_clear(&self, wait_for_key: bool) -> Result<usize, String> {
        let rows: Vec<(String, String, Option<String>, Option<String>, String)> = {
            let mut stmt = self
                .conn
                .prepare(
                    "SELECT id, content, metadata, project, created_at FROM memories
                     WHERE kind = 'credential' AND substr(content, 1, 1) != ?1",
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
        // Nothing in clear: the Keychain is not read, so no access prompt.
        if rows.is_empty() {
            return Ok(0);
        }
        if !wait_for_key && KEY.get().is_none() {
            return Ok(0);
        }
        if key().is_none() {
            eprintln!("[MemoryPilot] Vault: no encryption key available, credentials left as is.");
            return Ok(0);
        }
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

/// Puts back a vault key read on stdin (base64), after checking it opens
/// every sealed credential. Returns (opened, sealed).
#[cfg(target_os = "macos")]
pub(crate) fn restore_key(encoded: &str, db_path: &std::path::Path) -> Result<(usize, usize), String> {
    let key = decode_key(encoded).ok_or("not a base64 32-byte key")?;
    let conn = rusqlite::Connection::open_with_flags(db_path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
        .map_err(|e| format!("open {}: {}", db_path.display(), e))?;
    let mut stmt = conn
        .prepare("SELECT metadata FROM memories WHERE kind = 'credential' AND metadata LIKE '%\"enc\"%'")
        .map_err(|e| e.to_string())?;
    let encs: Vec<String> = stmt
        .query_map([], |r| r.get::<_, Option<String>>(0))
        .map_err(|e| e.to_string())?
        .flatten()
        .flatten()
        .filter_map(|m| serde_json::from_str::<serde_json::Value>(&m).ok())
        .filter_map(|m| m.get("enc").and_then(|v| v.as_str()).map(String::from))
        .collect();
    let opened = encs.iter().filter(|enc| decrypt_with(&key, enc).is_some()).count();
    if encs.is_empty() || opened != encs.len() {
        return Err(format!("this key opens {} of {} sealed credentials: Keychain left untouched", opened, encs.len()));
    }
    security_framework::passwords::set_generic_password(KEYCHAIN_SERVICE, KEYCHAIN_ACCOUNT, B64.encode(key).as_bytes())
        .map_err(|e| format!("Keychain write failed ({})", e.code()))?;
    let check = security_framework::passwords::get_generic_password(KEYCHAIN_SERVICE, KEYCHAIN_ACCOUNT)
        .map_err(|e| format!("Keychain read-back failed ({})", e.code()))?;
    if decode_key(&String::from_utf8_lossy(&check)) != Some(key) {
        return Err("Keychain read-back differs".into());
    }
    Ok((opened, encs.len()))
}

/// A recovered text belongs to a sealed row when it gives back the stored
/// label: exactly, or with every visible word of an older label in order.
/// A label cut at 220 characters may end with a truncated word.
fn label_matches(stored: &str, clear: &str) -> bool {
    if masked_label(clear) == stored {
        return true;
    }
    let body = stored.trim_start_matches(SEAL_PREFIX);
    let truncated = body.chars().count() >= 220;
    let mut words = clear.split_whitespace();
    let visible: Vec<&str> = body.split_whitespace().filter(|word| !word.contains('•')).collect();
    let last = visible.len().saturating_sub(1);
    !visible.is_empty()
        && visible.iter().enumerate().all(|(index, word)| {
            words.any(|w| w == *word || (truncated && index == last && w.starts_with(word)))
        })
}

#[derive(Debug, Default)]
pub struct Reseal {
    pub resealed: usize,
    /// Ids whose recovered text does not give back the stored label.
    pub mismatched: Vec<String>,
    /// Sealed credentials absent from the recovered set.
    pub missing: Vec<String>,
    /// Recovered ids that are not credentials of this database.
    pub unknown: Vec<String>,
}

impl Database {
    /// Re-encrypts with the current key the credentials recovered in clear
    /// (id → text) after a key loss. Labels are kept; only `metadata.enc`
    /// changes, and nothing is written unless every text matches its label.
    /// Dry run unless `apply`.
    pub fn reseal_recovered(
        &self,
        recovered: &std::collections::HashMap<String, String>,
        apply: bool,
    ) -> Result<Reseal, String> {
        let rows: Vec<(String, String, Option<String>)> = {
            let mut stmt = self
                .conn
                .prepare("SELECT id, content, metadata FROM memories WHERE kind = 'credential'")
                .map_err(|e| format!("reseal prepare: {}", e))?;
            let rows = stmt
                .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
                .map_err(|e| format!("reseal query: {}", e))?;
            rows.flatten().collect()
        };
        let mut report = Reseal::default();
        let mut updates: Vec<(String, String)> = Vec::new();
        for (id, stored, metadata) in &rows {
            let Some(clear) = recovered.get(id) else {
                report.missing.push(id.clone());
                continue;
            };
            if !label_matches(stored, clear) {
                report.mismatched.push(id.clone());
                continue;
            }
            if !apply {
                report.resealed += 1;
                continue;
            }
            let enc = encrypt(clear).ok_or("no encryption key available")?;
            if decrypt(&enc).as_deref() != Some(clear.as_str()) {
                return Err(format!("{}: ciphertext does not read back", id));
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
            updates.push((id.clone(), serde_json::to_string(&meta).map_err(|e| e.to_string())?));
        }
        report.unknown = recovered
            .keys()
            .filter(|id| !rows.iter().any(|(row_id, _, _)| row_id == *id))
            .cloned()
            .collect();
        if !apply {
            return Ok(report);
        }
        if !report.mismatched.is_empty() || !report.unknown.is_empty() {
            return Err(format!(
                "{} text(s) do not match their label and {} id(s) are unknown: nothing written",
                report.mismatched.len(),
                report.unknown.len()
            ));
        }
        let tx = self.conn.unchecked_transaction().map_err(|e| format!("reseal begin: {}", e))?;
        for (id, meta_json) in &updates {
            tx.execute("UPDATE memories SET metadata=?1 WHERE id=?2", params![meta_json, id])
                .map_err(|e| format!("reseal update: {}", e))?;
        }
        tx.commit().map_err(|e| format!("reseal commit: {}", e))?;
        report.resealed = updates.len();
        Ok(report)
    }
}

#[derive(Debug, Default)]
pub struct OnePasswordSync {
    /// "present", "created" or "to_create".
    pub key: &'static str,
    pub already: usize,
    /// (id, op:// reference) of credentials whose value was already in 1Password.
    pub linked: Vec<(String, String)>,
    /// (id, masked label) of credentials copied into a new 1Password item.
    pub created: Vec<(String, String)>,
    pub failed: Vec<(String, String)>,
}

impl Database {
    /// Points every credential to 1Password: an item about the same account
    /// (title or URL host in the text) already holding its value is linked,
    /// otherwise a new item (tag memorypilot-import) is created. The
    /// vault key is backed up first. Nothing is deleted; dry run unless `apply`.
    pub fn sync_to_onepassword(&self, apply: bool) -> Result<OnePasswordSync, String> {
        use crate::onepassword as op;
        let key = key().ok_or("no encryption key available: credentials cannot be read")?;
        let items = op::vault_items()?;
        let mut report = OnePasswordSync {
            key: if op::find_title(&items, op::KEY_ITEM_TITLE).is_some() { "present" } else { "to_create" },
            ..Default::default()
        };
        if apply && report.key == "to_create" {
            let template = serde_json::json!({
                "title": op::KEY_ITEM_TITLE,
                "category": "PASSWORD",
                "tags": ["memorypilot"],
                "fields": [
                    { "id": "password", "type": "CONCEALED", "purpose": "PASSWORD", "label": "password", "value": B64.encode(key) },
                    { "id": "notesPlain", "type": "STRING", "purpose": "NOTES", "label": "notesPlain",
                      "value": format!("Clé XChaCha20-Poly1305 (base64) des identifiants MemoryPilot. Sur un nouveau Mac, la remettre dans le Trousseau : service {}, compte {}.", KEYCHAIN_SERVICE, KEYCHAIN_ACCOUNT) }
                ]
            });
            op::create_item(&template)?;
            report.key = "created";
        }
        let ids: Vec<String> = {
            let mut stmt = self
                .conn
                .prepare("SELECT id FROM memories WHERE kind = 'credential' ORDER BY created_at")
                .map_err(|e| format!("sync prepare: {}", e))?;
            let rows = stmt.query_map([], |r| r.get(0)).map_err(|e| format!("sync query: {}", e))?;
            rows.flatten().collect()
        };
        for id in ids {
            let Some(memory) = self.get_memory(&id)? else { continue };
            let location = memory.metadata.as_ref().and_then(|m| m.get("location")).and_then(|v| v.as_str());
            if location.is_some_and(|l| l.starts_with("op://")) {
                report.already += 1;
                continue;
            }
            let clear = match self.credential_clear_text(&id) {
                Ok(Some(clear)) => clear,
                Ok(None) => continue,
                Err(error) => {
                    report.failed.push((id, error));
                    continue;
                }
            };
            let secret = find_secret_value(&clear);
            let label = memory.content.trim_start_matches(SEAL_PREFIX).to_string();
            let known = secret.as_deref().and_then(|value| op::find_value(&items, value, &clear));
            let reference = match known {
                Some(reference) => {
                    report.linked.push((id.clone(), reference.clone()));
                    reference
                }
                None => {
                    report.created.push((id.clone(), label.clone()));
                    if !apply {
                        continue;
                    }
                    let title = format!(
                        "{} · {}",
                        memory.project.as_deref().unwrap_or("global"),
                        label.chars().take(70).collect::<String>()
                    );
                    let (template, field) = op::item_template(&title, memory.project.as_deref(), &clear, secret.as_deref());
                    match op::create_item(&template) {
                        Ok(item_id) => op::reference(&item_id, field),
                        Err(error) => {
                            report.created.pop();
                            report.failed.push((id, error));
                            continue;
                        }
                    }
                }
            };
            if apply {
                let mut metadata = memory.metadata.clone().unwrap_or_else(|| serde_json::json!({}));
                if let Some(object) = metadata.as_object_mut() {
                    object.insert("location".into(), serde_json::json!(reference));
                }
                self.update_memory_full(&id, None, None, None, None, None, Some(&metadata))?;
            }
        }
        Ok(report)
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
    fn recovered_text_must_give_back_its_label() {
        let clear = "Admin demo : login admin@demo.fr password: Hx9!kLm2pQ";
        assert!(label_matches(&masked_label(clear), clear));
        assert!(label_matches("🔒 Admin demo : login •••• password: ••••", clear), "older label");
        assert!(!label_matches(&masked_label(clear), "Admin autre : password: Hx9!kLm2pQ"));
        assert!(!label_matches("🔒 •••• ••••", clear), "a label without visible words proves nothing");

        let long = format!("{}VILLART", "mot ".repeat(54));
        let cut = format!("{}{}VILLAR", SEAL_PREFIX, "mot ".repeat(54));
        assert!(label_matches(&cut, &long), "word cut by the 220-character cap");
        assert!(!label_matches("🔒 Admin demo : login VILLAR", "Admin demo : login VILLART"), "short label: no prefix match");
    }

    #[test]
    fn reseal_recovered_rewrites_only_matching_ciphertexts() {
        let dir = std::env::temp_dir().join(format!("mp-reseal-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let db = Database::open_at(&dir.join("memory.db")).unwrap();
        let clear = "Admin demo : password: Hx9!kLm2pQ";
        let (memory, _) = db
            .add_memory(clear, "credential", Some("demo"), &[], "test", 5, None, None, &super::super::MemoryScope::default())
            .unwrap();
        let id = memory.id;
        let enc_of = |db: &Database| -> String {
            let meta: String = db
                .conn
                .query_row("SELECT metadata FROM memories WHERE id=?1", params![id], |r| r.get(0))
                .unwrap();
            serde_json::from_str::<serde_json::Value>(&meta).unwrap()["enc"].as_str().unwrap().to_string()
        };
        let before = enc_of(&db);

        let wrong = std::collections::HashMap::from([(id.clone(), "Admin autre : password: x".to_string())]);
        assert!(db.reseal_recovered(&wrong, true).is_err());
        assert_eq!(enc_of(&db), before, "a mismatch writes nothing");

        let right = std::collections::HashMap::from([(id.clone(), clear.to_string())]);
        let dry = db.reseal_recovered(&right, false).unwrap();
        assert_eq!((dry.resealed, dry.mismatched.len(), dry.missing.len()), (1, 0, 0));
        assert_eq!(enc_of(&db), before, "dry run");

        assert_eq!(db.reseal_recovered(&right, true).unwrap().resealed, 1);
        let after = enc_of(&db);
        assert_ne!(after, before);
        assert_eq!(decrypt(&after).as_deref(), Some(clear));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn pending_credentials_are_sealed_once_the_key_is_loaded() {
        use super::super::MemoryStore;
        let dir = std::env::temp_dir().join(format!("mp-pending-seal-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let db = Database::open_at(&dir.join("memory.db")).unwrap();
        db.insert_memory(&credential("Admin demo : password: Hx9!kLm2pQ", None), "hash").unwrap();
        assert!(key().is_some());
        assert_eq!(db.seal_pending_credentials().unwrap(), 1);
        assert_eq!(db.seal_pending_credentials().unwrap(), 0, "idempotent");
        let stored = db.stored_content("c1").unwrap();
        assert!(stored.starts_with(SEAL_PREFIX) && !stored.contains("Hx9!kLm2pQ"));
        let _ = std::fs::remove_dir_all(&dir);
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
            // Technical notes the detector used to seal (2026-10-23).
            "Modèle multilingual-e5-small via fastembed, cible x86_64-unknown-linux-musl ou wasm32-unknown-unknown.",
            "Rerank avec jina-reranker-v2 après BM25+vector, cartes avec hover:-translate-y-1.",
            "Cookie SameSite=Lax, header stale-while-revalidate=60, FCM pour iOS+Android.",
            "FAISS pour C++/Python, charger via load() plutôt que onMount/$effect.",
            "D1 uuid bd40952d-15fc-434c-89af-f274846c5d25, migration 20260930120000_admin_guard.sql.",
            "Appel avec max_tokens=4096, clé lue par op-sof.",
        ] {
            assert!(!contains_secret_value(prose), "false positive: {}", prose);
        }
    }

    // Notes the detector sealed on write (2026-10-25): prices, public ids named
    // by their label, metrics, code, package versions, file and topic names.
    #[test]
    fn secret_value_detection_ignores_ids_prices_and_code() {
        let hex = "9f8e7d6c5b4a39281706f5e4d3c2b1a0";
        let notes = [
            "[Soflution_Pricing] Starter: 19€/mois".to_string(),
            "Forfait Pro: 19,99€/mois, GPU €120/month, max £500/mois, Browserbase ~$20/mois".to_string(),
            format!("Cloudflare account ID: {hex}"),
            format!("Account ID CF: {hex} (zone sociomator.com (zoneId {hex}), KV FLIGHTS_KV id {hex})"),
            format!("Twilio Account SID: AC{hex}"),
            "PV (Doc ID 1AbCdEfGhIjKlMnOpQrStUvWxYz0123456789) parent folder 1ZyXwVuTsRqPoNmLkJiHgFeDcBa98".to_string(),
            "Stripe acct_1AbCdEfGhIjKlMnO, product prod_AbCdEfGhIjKl12, tarif price_1AbCdEfGhIjKlMnOpQ".to_string(),
            format!("Vercel projectId=prj_AbCdEfGhIjKlMnOpQrSt12, CLOUDFLARE_ACCOUNT_ID={hex}"),
            "Resend domaine roompilot.io (id=0a1b2c3d-4e5f-4a6b-8c7d-9e0f1a2b3c4d) VERIFIED".to_string(),
            "LongMemEval R@5=97.4%, FR R@5=62.4%->68.8%, latence ~50-60ms".to_string(),
            "Deps: mcp-remote@0.15.1, @modelcontextprotocol/sdk@1.20.0, npx wrangler@4, actions/checkout@v4".to_string(),
            "[OnlystParrains_Ui_Oct2026] Écran parrains : fetch membres actifs".to_string(),
            "Helper buildStripeCheckoutUrl2024, modèle Qwen3Attention".to_string(),
            "Ghostscript low=max(0.08, targetQuality*0.15)".to_string(),
            "APNs AuthKey_AB12CD34EF.p8 posé en secret Cloudflare".to_string(),
            "RoomPilot déployé via cf-orchestrator project key=roompilot".to_string(),
            "et le ViewModel passe targetQuality au service".to_string(),
            "L'ancienne STRIPE_API_KEY sk_live, format des clés Google « AIza »".to_string(),
            "Estimation (ex 80%=12MB), étapes A1→A2→A3, lien header-anchor?param=1".to_string(),
            "Stripe lookup_key=sociomator_pro, Défaut=0, Sauvegardes *.bak-20261025-client".to_string(),
            format!("Profil Adobe CoatedFOGRA39.icc, webcredentials applinks:{}.com.soflution.app", "AB12CD345E"),
            format!("Bindings: AUTH_KV (id {hex}), WAF (ruleset {hex})"),
            "Coords : email [402,519] · mot de passe 402,519 · bouton [402,600]".to_string(),
            "Header x-apikey='env:cegid_loop' ou x-apikey='key:cegid_loop' lu au runtime".to_string(),
        ];
        for note in &notes {
            assert!(!contains_secret_value(note), "false positive: {}", note);
        }
    }

    #[test]
    fn secret_words_win_over_identifier_words() {
        let hex = "9f8e7d6c5b4a39281706f5e4d3c2b1a0";
        let stripe = format!("Stripe account {}_live_{}", "sk", "AbCdEf0123456789GhIjKl");
        for secret in [
            format!("Cloudflare account token: {hex}"),
            format!("Supabase id 11112222-3333-4444-8555-666677778888, token {hex}"),
            "workspace aaaabbbb-cccc-4ddd-8eee-ffff00001111 pwd Ab12Cd34!x".to_string(),
            stripe,
            format!("POWENS_CLIENT_ID=37447765. POWENS_CLIENT_{}=AbCdEf123456GhIjKl789", "SECRET"),
            "Secret posé : CF_API_KEY=\"AbCdEf123456GhIjKl789\"".to_string(),
            "Compte admin@example.com / Lavillart2026!".to_string(),
        ] {
            assert!(contains_secret_value(&secret), "missed: {}", secret);
            let value = find_secret_value(&secret).expect("value");
            assert!(!masked_label(&secret).contains(&value), "{} shown in the label of {}", value, secret);
        }
    }

    #[test]
    fn secret_value_detection_finds_real_values() {
        // Built at run time: a literal would trip secret scanners on the repository.
        let supabase = format!("Supabase access token: sbp_{}01234567", "0123456789abcdef".repeat(2));
        let gitlab = format!("GitLab {}-ab12cd34ef56gh78ij90", "glpat");
        let license = format!("plan=ultra, license_{}=1f2e3d4c-5b6a-4978-8695-a4b3c2d1e0f9", "key");
        for secret in [
            "password: villacalvi",
            "mot de passe : Calvi2026",
            "Compte démo : demo@example.com / pass 12345678aA@!",
            "Resend key re_AbCdEf123456GhIjKl789",
            "GitHub token ghp_abcdefghijklmnopqrstuvwxyz0123",
            "MDP = soleil-corse",
            "Accès staging Zq8!vT3#pLm2",
            "Code Kx7+mP2q",
            "Jeton tok_51HxYzAbCdEfGh12345678",
            "Clé QWERTYZXCV7ASDFGHJKL",
            "env token=4f9c2a8e1b7d6c5f3a2e",
            "API_KEY=sk-abc123def456ghi789",
            supabase.as_str(),
            gitlab.as_str(),
            license.as_str(),
        ] {
            assert!(contains_secret_value(secret), "missed: {}", secret);
            let value = find_secret_value(secret).expect("value");
            assert!(!masked_label(secret).contains(&value), "{} shown in the label of {}", value, secret);
        }
    }
}
