//! Secret redaction for text stored as said (conversations, hook
//! transcripts). One filter: known credential formats, private key
//! blocks, passwords in URIs, values after a password or token name,
//! bearer tokens, IBANs and long random tokens, and the values listed as
//! known secrets (SHA-256 prints only) whatever they look like.
//!
//! Each value becomes `[secret masqué : <kind> <print>]`, the print being
//! the first 4 hex digits of its SHA-256: two mentions of the same secret
//! stay recognisable without revealing it. Masks already in the text are
//! left alone, so redacting twice changes nothing.

use sha2::{Digest, Sha256};
use std::borrow::Cow;
use std::collections::HashSet;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock, RwLock};

const MASK_OPEN: &str = "[secret masqué : ";

/// One SHA-256 print (64 hex digits) per line, next to conversations.db.
pub const KNOWN_SECRETS_FILE: &str = "known-secrets.sha256";
/// Shortest value matched against the known prints.
const KNOWN_MIN_CHARS: usize = 8;
/// Shortest token of a credential's text taken as a known secret.
const CREDENTIAL_TOKEN_MIN_CHARS: usize = 12;

/// Credential prefixes, the minimum length of what follows, and the kind
/// shown in the mask. Longer prefixes come before the ones they extend.
const PREFIXES: &[(&str, usize, &str)] = &[
    ("github_pat_", 22, "github_pat"),
    ("ghp_", 36, "github_token"),
    ("gho_", 36, "github_token"),
    ("ghu_", 36, "github_token"),
    ("ghs_", 36, "github_token"),
    ("ghr_", 36, "github_token"),
    ("sk-ant-", 20, "anthropic"),
    ("sk_live_", 20, "stripe_secret"),
    ("sk_test_", 20, "stripe_secret"),
    ("rk_live_", 20, "stripe_secret"),
    ("rk_test_", 20, "stripe_secret"),
    ("sk-", 20, "openai"),
    ("whsec_", 24, "stripe_webhook"),
    ("re_", 20, "resend"),
    ("cfut_", 20, "cloudflare"),
    ("glpat-", 20, "gitlab"),
    ("npm_", 36, "npm"),
    ("xoxb-", 20, "slack"),
    ("xoxp-", 20, "slack"),
    ("xoxa-", 20, "slack"),
    ("xoxe-", 20, "slack"),
    ("xoxs-", 20, "slack"),
    ("AIza", 30, "google_api"),
    ("sbp_", 40, "generic"),
    ("sb_secret_", 20, "supabase_secret"),
];

/// Public ids that only point at a resource: never masked, however random.
const PUBLIC_PREFIXES: &[&str] = &[
    "pk_", "sb_publishable_", "acct_", "prod_", "price_", "plink_", "cus_", "sub_", "evt_", "we_", "prj_", "team_",
];

/// Names whose value is a password (`password:`, `DB_PASSWD=`, `mdp :`).
const PASSWORD_NAMES: &[&str] = &["password", "passwd", "pwd", "mdp", "motdepasse", "mot_de_passe"];

/// Names whose value is a token or a key (`token=`, `client_secret:`, `RESEND_API_KEY=`).
const TOKEN_NAMES: &[&str] = &["token", "secret", "api_key", "apikey", "api-key", "secret_key", "access_key", "private_key"];

/// Words between a password name and its value ("le mot de passe est …").
const CONNECTORS: &[&str] = &["est", "is", "vaut", "was", "était"];

/// After a password name, these say where the secret lives, not what it is.
const REFERENCES: &[&str] = &[
    "1password", "op-sof", "trousseau", "keychain", "stocké", "stockée", "stored", "dans", "in", "via", "voir",
    "see", "env", "variable", "manager", "redacted", "masqué", "oublié", "reset", "réinitialisation", "requis",
    "required", "changé", "modifié", "expiré", "incorrect", "invalide", "invalid", "wrong", "faux", "correct",
    "changed", "expired", "missing", "manquant", "vide", "empty",
];

/// Type names and placeholders written where a password value would go.
const CODE_WORDS: &[&str] = &[
    "string", "str", "text", "number", "boolean", "bool", "null", "none", "undefined", "optional", "password",
    "secret", "hidden", "true", "false", "example", "exemple", "placeholder", "xxxxxx", "xxxxxxxx",
];

/// Labels of public ids in a credential's text: the value after them is
/// not a secret (`Account ID a4b5…`, `client_id=…`, `login: …`).
const PUBLIC_LABELS: &[&str] = &[
    "account id", "account_id", "accountid", "account", "zone id", "zone_id", "app id", "app_id", "team id",
    "team_id", "key id", "key_id", "issuer id", "issuer_id", "client id", "client_id", "clientid", "project id",
    "project_id", "tenant id", "tenant_id", "org id", "org_id", "user id", "user_id", "user", "username", "login",
    "email", "e-mail", "identifiant", "utilisateur", "compte", "id", "issuer", "apiissuer", "workspace", "build",
    "bundle", "hash", "hashes", "hachage",
];

const PASSWORD_SYMBOLS: &[char] = &['!', '@', '#', '$', '%', '^', '&', '*', '?', '+', '=', '~', '§', '€', '£'];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finding {
    pub range: Range<usize>,
    pub kind: &'static str,
}

/// SHA-256 prints of secret values that have no recognisable format.
/// Only prints are held, never the values.
#[derive(Debug, Default, Clone)]
pub struct KnownSecrets {
    prints: HashSet<[u8; 32]>,
}

impl KnownSecrets {
    pub fn len(&self) -> usize {
        self.prints.len()
    }

    pub fn is_empty(&self) -> bool {
        self.prints.is_empty()
    }

    pub fn insert(&mut self, value: &str) {
        self.prints.insert(Sha256::digest(value.as_bytes()).into());
    }

    pub fn contains(&self, value: &str) -> bool {
        !self.prints.is_empty() && self.prints.contains(&<[u8; 32]>::from(Sha256::digest(value.as_bytes())))
    }

    /// A print written as 64 hex digits; false when the line is not one.
    pub fn insert_print(&mut self, hex: &str) -> bool {
        let hex = hex.trim();
        if hex.len() != 64 || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
            return false;
        }
        let mut print = [0u8; 32];
        for (index, byte) in print.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&hex[2 * index..2 * index + 2], 16).unwrap_or(0);
        }
        self.prints.insert(print)
    }

    /// Prints of a file of known secrets; a missing file adds nothing.
    pub fn read_file(&mut self, path: &Path) -> usize {
        let Ok(content) = std::fs::read_to_string(path) else { return 0 };
        content.lines().filter(|line| self.insert_print(line)).count()
    }

    /// Every token of the credentials' clear texts that may be a secret
    /// value: no logins, hosts, emails, file names or plain words, and no
    /// value written after a public-id label in any of them.
    pub fn insert_credential_texts<S: AsRef<str>>(&mut self, texts: &[S]) -> usize {
        let mut public = HashSet::new();
        let mut secret = Vec::new();
        for clear in texts.iter().map(AsRef::as_ref) {
            for token in whitespace_tokens(clear) {
                let url = clear[token.clone()].contains("://");
                for range in candidates(clear, token) {
                    let value = &clear[range.clone()];
                    if value.chars().count() < CREDENTIAL_TOKEN_MIN_CHARS || !credential_secret(value) {
                        continue;
                    }
                    // In a URL only a query value or the password before `@`.
                    if url && !(clear[..range.start].ends_with('=') || clear[range.end..].starts_with('@')) {
                        continue;
                    }
                    if public_label_before(clear, range.start) {
                        public.insert(value);
                    } else {
                        secret.push(value);
                    }
                }
            }
        }
        let before = self.len();
        for value in secret.into_iter().filter(|value| !public.contains(value)) {
            self.insert(value);
        }
        self.len() - before
    }
}

static KNOWN: RwLock<Option<Arc<KnownSecrets>>> = RwLock::new(None);
static KNOWN_LOADED: OnceLock<()> = OnceLock::new();

/// Replaces the known secrets used by `redact` and `findings`.
pub fn set_known_secrets(known: KnownSecrets) {
    if let Ok(mut slot) = KNOWN.write() {
        *slot = Some(Arc::new(known));
    }
}

fn known_secrets() -> Option<Arc<KnownSecrets>> {
    KNOWN.read().ok().and_then(|slot| slot.clone())
}

/// Loads the known secrets once per process (see `reload_known_secrets`).
pub fn ensure_known_secrets() {
    KNOWN_LOADED.get_or_init(|| {
        reload_known_secrets();
    });
}

/// Loads the known secrets again from both sources: the prints file and
/// the credentials of memory.db. A missing file, database or vault key
/// only leaves that source out. Returns how many prints are held.
pub fn reload_known_secrets() -> usize {
    let known = load_known_secrets_from(known_secrets_path().as_deref(), memory_db_path().as_deref());
    let count = known.len();
    set_known_secrets(known);
    count
}

pub fn load_known_secrets_from(file: Option<&Path>, memory_db: Option<&Path>) -> KnownSecrets {
    let mut known = KnownSecrets::default();
    if let Some(file) = file {
        known.read_file(file);
    }
    if let Some(db) = memory_db.filter(|db| db.exists()) {
        known.insert_credential_texts(&crate::db::credential_clear_texts(db).unwrap_or_default());
    }
    known
}

/// The folder of conversations.db / memory.db (`MEMORYPILOT_CONVERSATIONS_DB`,
/// then `MEMORYPILOT_DB`, else ~/.MemoryPilot).
fn data_dir() -> Option<PathBuf> {
    for variable in ["MEMORYPILOT_CONVERSATIONS_DB", "MEMORYPILOT_DB"] {
        if let Ok(path) = std::env::var(variable) {
            if !path.trim().is_empty() {
                return Path::new(&path).parent().map(Path::to_path_buf);
            }
        }
    }
    Some(dirs::home_dir()?.join(".MemoryPilot"))
}

pub fn known_secrets_path() -> Option<PathBuf> {
    data_dir().map(|dir| dir.join(KNOWN_SECRETS_FILE))
}

fn memory_db_path() -> Option<PathBuf> {
    match std::env::var("MEMORYPILOT_DB") {
        Ok(path) if !path.trim().is_empty() => Some(PathBuf::from(path)),
        _ => Some(dirs::home_dir()?.join(".MemoryPilot").join("memory.db")),
    }
}

/// Adds the print of `value` to the known secrets file (created 0600) and
/// returns its short print. The value itself is never written.
pub fn add_known_secret(value: &str, path: &Path) -> Result<String, String> {
    let value = value.trim();
    if value.is_empty() {
        return Err("empty value: nothing added".into());
    }
    if value.contains(char::is_whitespace) {
        return Err("the value contains spaces: only single tokens can be recognised".into());
    }
    if value.chars().count() < KNOWN_MIN_CHARS {
        return Err(format!("value shorter than {} characters: too likely to appear by chance", KNOWN_MIN_CHARS));
    }
    let digest = Sha256::digest(value.as_bytes());
    let hex: String = digest.iter().map(|byte| format!("{:02x}", byte)).collect();
    let short = hex[..4].to_string();
    if let Some(dir) = path.parent().filter(|dir| !dir.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).map_err(|error| format!("{}: {}", dir.display(), error))?;
    }
    let existing = std::fs::read_to_string(path).unwrap_or_default();
    if !existing.lines().any(|line| line.trim().eq_ignore_ascii_case(&hex)) {
        use std::io::Write;
        let mut options = std::fs::OpenOptions::new();
        options.create(true).append(true);
        #[cfg(unix)]
        std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
        let mut file = options.open(path).map_err(|error| format!("{}: {}", path.display(), error))?;
        let separator = if existing.is_empty() || existing.ends_with('\n') { "" } else { "\n" };
        writeln!(file, "{}{}", separator, hex).map_err(|error| format!("{}: {}", path.display(), error))?;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
            .map_err(|error| format!("{}: {}", path.display(), error))?;
    }
    Ok(short)
}

/// The text with every secret value replaced by its mask.
pub fn redact(text: &str) -> Cow<'_, str> {
    match known_secrets() {
        Some(known) => redact_with(text, &known),
        None => redact_with(text, &KnownSecrets::default()),
    }
}

/// `redact` with an explicit set of known secrets.
pub fn redact_with<'a>(text: &'a str, known: &KnownSecrets) -> Cow<'a, str> {
    let found = findings_with(text, known);
    if found.is_empty() {
        return Cow::Borrowed(text);
    }
    let mut out = String::with_capacity(text.len());
    let mut last = 0;
    for finding in &found {
        out.push_str(&text[last..finding.range.start]);
        out.push_str(&mask(finding.kind, &text[finding.range.clone()]));
        last = finding.range.end;
    }
    out.push_str(&text[last..]);
    Cow::Owned(out)
}

/// `[secret masqué : <kind> <first 4 hex digits of the value's SHA-256>]`.
pub fn mask(kind: &str, value: &str) -> String {
    let digest = Sha256::digest(value.as_bytes());
    format!("{}{} {:02x}{:02x}]", MASK_OPEN, kind, digest[0], digest[1])
}

/// Secret values of the text, in order, without overlaps.
pub fn findings(text: &str) -> Vec<Finding> {
    match known_secrets() {
        Some(known) => findings_with(text, &known),
        None => findings_with(text, &KnownSecrets::default()),
    }
}

pub fn findings_with(text: &str, known: &KnownSecrets) -> Vec<Finding> {
    let mut found = Found { taken: masks(text), out: Vec::new() };
    private_keys(text, &mut found);
    uri_passwords(text, &mut found);
    let words = words(text);
    for word in &words {
        bearer_token(text, word.clone(), &mut found);
    }
    for (index, word) in words.iter().enumerate() {
        if let Some((range, kind)) = word_secret(text, word.clone()) {
            found.add(range, kind);
        }
        spaced_iban(text, &words, index, &mut found);
    }
    for index in 0..words.len() {
        named_value(text, &words, index, &mut found);
    }
    known_values(text, known, &mut found);
    let mut out = found.out;
    out.retain(|finding| !public_value(&text[finding.range.clone()]));
    out.sort_by_key(|finding| finding.range.start);
    out
}

struct Found {
    taken: Vec<Range<usize>>,
    out: Vec<Finding>,
}

impl Found {
    fn add(&mut self, range: Range<usize>, kind: &'static str) {
        let overlaps = |other: &Range<usize>| range.start < other.end && other.start < range.end;
        if range.is_empty() || self.taken.iter().any(overlaps) || self.out.iter().any(|f| overlaps(&f.range)) {
            return;
        }
        self.out.push(Finding { range, kind });
    }
}

/// Masks written by an earlier pass: never redacted again.
fn masks(text: &str) -> Vec<Range<usize>> {
    let mut ranges = Vec::new();
    let mut from = 0;
    while let Some(start) = text[from..].find(MASK_OPEN).map(|i| from + i) {
        let end = text[start..].find(']').map(|i| start + i + 1).unwrap_or(text.len());
        ranges.push(start..end);
        from = end;
    }
    ranges
}

fn is_separator(c: char) -> bool {
    c.is_whitespace() || "=:\"'`,()[]{}<>;\\".contains(c)
}

/// Byte ranges of the runs between separators.
fn words(text: &str) -> Vec<Range<usize>> {
    let mut words = Vec::new();
    let mut start = None;
    for (index, c) in text.char_indices() {
        if is_separator(c) {
            if let Some(start) = start.take() {
                words.push(start..index);
            }
        } else if start.is_none() {
            start = Some(index);
        }
    }
    if let Some(start) = start {
        words.push(start..text.len());
    }
    words
}

/// The word without the punctuation around it (`_` and `-` belong to keys).
fn core_range(text: &str, word: Range<usize>) -> Range<usize> {
    let keep = |c: char| c.is_alphanumeric() || c == '_' || c == '-';
    let slice = &text[word.clone()];
    let start = word.start + (slice.len() - slice.trim_start_matches(|c: char| !keep(c)).len());
    let end = word.end - (slice.len() - slice.trim_end_matches(|c: char| !keep(c)).len());
    start..end.max(start)
}

fn word_secret(text: &str, word: Range<usize>) -> Option<(Range<usize>, &'static str)> {
    let core = core_range(text, word);
    if text[..core.start].ends_with("base64,") {
        return None; // inline image or file data
    }
    let value = &text[core.clone()];
    let lead = value.len() - value.trim_start_matches(['-', '_']).len();
    if let Some((len, kind)) = known_format(&value[lead..]) {
        return Some((core.start + lead..core.start + lead + len, kind));
    }
    // A JSON string escape glued to the value (`\nghp_…`).
    if text[..core.start].ends_with('\\') && value.starts_with(['n', 'r', 't']) {
        if let Some((len, kind)) = known_format(&value[1..]) {
            return Some((core.start + 1..core.start + 1 + len, kind));
        }
    }
    if is_iban(value) {
        return Some((core, "iban"));
    }
    long_random_token(value).then_some((core, "generic"))
}

/// Length and kind of the credential the text starts with, if any.
fn known_format(value: &str) -> Option<(usize, &'static str)> {
    for (prefix, min, kind) in PREFIXES {
        let Some(rest) = value.strip_prefix(prefix) else { continue };
        let len = rest
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-')
            .count();
        let body = rest[..len].trim_end_matches(['-', '_']);
        if body.len() >= *min
            && body.chars().any(|c| c.is_ascii_digit())
            && !identifier_like(body.split(['-', '_']))
        {
            return Some((prefix.len() + body.len(), kind));
        }
    }
    // AWS access key id: AKIA (or ASIA for temporary ones) and 16 more.
    if value.starts_with("AKIA") || value.starts_with("ASIA") {
        let body: Vec<char> = value.chars().skip(4).take(17).collect();
        if body.len() >= 16
            && body[..16].iter().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit())
            && body.get(16).is_none_or(|c| !c.is_ascii_alphanumeric())
        {
            return Some((20, "aws"));
        }
    }
    if value.starts_with("eyJ") {
        let len = value
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.'))
            .count();
        let parts: Vec<&str> = value[..len].trim_end_matches('.').split('.').collect();
        if parts.len() == 3 && parts.iter().all(|part| part.len() >= 10) {
            return Some((parts.iter().map(|part| part.len()).sum::<usize>() + 2, "jwt"));
        }
    }
    None
}

/// Values meant to be public, whoever found them: publishable keys, Twilio
/// Account SIDs, AWS documentation examples, Supabase anon JWTs and JWT parts
/// without their signature.
fn public_value(value: &str) -> bool {
    PUBLIC_PREFIXES.iter().any(|prefix| value.starts_with(prefix))
        || twilio_account_sid(value)
        || aws_documentation_example(value)
        || public_jwt(value)
}

/// The access key ids of the AWS documentation, such as `AKIAIOSFODNN7EXAMPLE`.
fn aws_documentation_example(value: &str) -> bool {
    value.len() == 20 && (value.starts_with("AKIA") || value.starts_with("ASIA")) && value.ends_with("EXAMPLE")
}

/// `AC` and 32 lowercase hex digits.
fn twilio_account_sid(value: &str) -> bool {
    value.len() == 34
        && value.starts_with("AC")
        && value[2..].bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// A JWT whose payload has the `anon` role, or a header and payload alone.
fn public_jwt(value: &str) -> bool {
    use base64::Engine;
    if !value.starts_with("eyJ") {
        return false;
    }
    let claims = |part: &str| -> Option<serde_json::Map<String, serde_json::Value>> {
        let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(part.trim_end_matches('=')).ok()?;
        serde_json::from_slice(&bytes).ok()
    };
    let parts: Vec<&str> = value.split('.').collect();
    match parts.len() {
        1 | 2 => parts.iter().all(|part| claims(part).is_some()),
        3 => claims(parts[1]).is_some_and(|payload| payload.get("role").and_then(|role| role.as_str()) == Some("anon")),
        _ => false,
    }
}

/// 2 letters, 2 check digits, 11 to 30 letters or digits, valid checksum.
fn is_iban(value: &str) -> bool {
    let bytes = value.as_bytes();
    (15..=34).contains(&value.len())
        && bytes[..2].iter().all(u8::is_ascii_uppercase)
        && bytes[2..4].iter().all(u8::is_ascii_digit)
        && bytes[4..].iter().all(|b| b.is_ascii_uppercase() || b.is_ascii_digit())
        && iban_checksum(value)
}

fn iban_checksum(iban: &str) -> bool {
    let mut remainder = 0u32;
    for c in iban[4..].chars().chain(iban[..4].chars()) {
        remainder = match c.to_digit(36) {
            Some(digit) if digit < 10 => (remainder * 10 + digit) % 97,
            Some(value) => (remainder * 100 + value) % 97,
            None => return false,
        };
    }
    remainder == 1
}

/// An IBAN written in groups of four (`FR76 3000 6000 …`).
fn spaced_iban(text: &str, words: &[Range<usize>], index: usize, found: &mut Found) {
    let first = &text[words[index].clone()];
    let head = first.as_bytes();
    if head.len() != 4 || !head[..2].iter().all(u8::is_ascii_uppercase) || !head[2..].iter().all(u8::is_ascii_digit) {
        return;
    }
    let mut compact = first.to_string();
    let mut end = words[index].end;
    for word in &words[index + 1..] {
        let group = text[word.clone()].trim_end_matches('.');
        if &text[end..word.start] != " "
            || group.len() > 4
            || !group.bytes().all(|b| b.is_ascii_uppercase() || b.is_ascii_digit())
            || compact.len() + group.len() > 34
        {
            break;
        }
        compact.push_str(group);
        end = word.start + group.len();
    }
    if is_iban(&compact) {
        found.add(words[index].start..end, "iban");
    }
}

/// Long token mixing upper case, lower case and digits, typical of keys.
/// Paths, dotted names and identifiers made of words are not keys.
fn long_random_token(value: &str) -> bool {
    value.len() >= 32
        && value.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
        && value.chars().any(|c| c.is_ascii_digit())
        && value.chars().any(|c| c.is_ascii_uppercase())
        && value.chars().any(|c| c.is_ascii_lowercase())
        && !PUBLIC_PREFIXES.iter().any(|prefix| value.starts_with(prefix))
        && !identifier_like(value.split(['-', '_']))
}

/// Words joined by `_`/`-` or in camelCase, short acronyms allowed
/// (`parse_JSON_v2_payload`, `useInfiniteQuery2Options`).
fn identifier_like<'a>(parts: impl IntoIterator<Item = &'a str>) -> bool {
    let parts: Vec<&str> = parts.into_iter().filter(|part| !part.is_empty()).collect();
    let acronym = |part: &&str| part.len() <= 5 && part.chars().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit());
    let words: Vec<&str> = parts.iter().filter(|part| !acronym(part)).copied().collect();
    !parts.is_empty() && made_of_words(words)
}

/// Splits capitalised words, each optionally followed by digits (`OnlystParrains`,
/// `Qwen3Attention`, `buildStripeCheckoutUrl2024`), into (words, letters); a number
/// gives (0, 0). None when the part does not follow this pattern, as random keys
/// do within a few characters.
pub(crate) fn camel_words(part: &str) -> Option<(usize, usize)> {
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
pub(crate) fn made_of_words<'a>(parts: impl IntoIterator<Item = &'a str>) -> bool {
    let mut total = (0, 0);
    for part in parts {
        let Some((words, letters)) = camel_words(part) else { return false };
        total = (total.0 + words, total.1 + letters);
    }
    total.1 >= 3 * total.0
}

fn whitespace_tokens(text: &str) -> Vec<Range<usize>> {
    let mut tokens = Vec::new();
    let mut start = None;
    for (index, c) in text.char_indices() {
        if c.is_whitespace() {
            if let Some(start) = start.take() {
                tokens.push(start..index);
            }
        } else if start.is_none() {
            start = Some(index);
        }
    }
    if let Some(start) = start {
        tokens.push(start..text.len());
    }
    tokens
}

/// Readings of a token that may be a whole value, longest first: as
/// written, without its quotes or brackets, without a final punctuation,
/// and each piece between `=`, `:`, `@`, `/`, `&`, `?` and the like.
fn candidates(text: &str, token: Range<usize>) -> Vec<Range<usize>> {
    // `/` splits URLs only: base64 values contain it.
    let separators = if text[token.clone()].contains("://") { "=:\"'`,()[]{}<>;\\@&?/" } else { "=:\"'`,()[]{}<>;\\@&?" };
    let narrow = |range: &Range<usize>, chars: &str, both: bool| {
        let slice = &text[range.clone()];
        let start = if both { range.start + (slice.len() - slice.trim_start_matches(|c: char| chars.contains(c)).len()) } else { range.start };
        let end = range.end - (slice.len() - slice.trim_end_matches(|c: char| chars.contains(c)).len());
        start..end.max(start)
    };
    let unquoted = narrow(&token, "\"'`()[]{}<>«»", true);
    let unpunctuated = narrow(&unquoted, ".,;:", false);
    let mut out = vec![token.clone(), unquoted.clone(), unpunctuated.clone()];
    let mut start = unquoted.start;
    for (index, c) in text[unquoted.clone()].char_indices() {
        let padding = c == '=' && text[unquoted.start + index..unquoted.end].bytes().all(|b| b == b'=');
        if separators.contains(c) && !padding {
            out.push(start..unquoted.start + index);
            start = unquoted.start + index + c.len_utf8();
        }
    }
    if start > unquoted.start {
        let last = start..unquoted.end;
        out.push(narrow(&last, ".,;:", false));
        out.push(last);
    }
    out.retain(|range| text[range.clone()].chars().count() >= KNOWN_MIN_CHARS);
    out.sort_by_key(|range| std::cmp::Reverse(range.len()));
    out.dedup();
    out
}

/// Values of the text whose print is a known secret, whatever they look like.
fn known_values(text: &str, known: &KnownSecrets, found: &mut Found) {
    if known.is_empty() {
        return;
    }
    for token in whitespace_tokens(text) {
        if let Some(range) = candidates(text, token).into_iter().find(|range| known.contains(&text[range.clone()])) {
            // Symbols glued after a known part belong to the value (`…@!`).
            let rest = &text[range.end..];
            let tail = rest
                .char_indices()
                .find(|&(index, c)| {
                    !PASSWORD_SYMBOLS.contains(&c)
                        || matches!(c, '?' | '=')
                        || (c == '&' && starts_with_assignment(&rest[index + 1..]))
                })
                .map_or(rest.len(), |(index, _)| index);
            found.add(range.start..range.end + tail, "known");
        }
    }
}

/// A token of a credential's text worth remembering as a secret value.
fn credential_secret(value: &str) -> bool {
    let letters = value.chars().filter(|c| c.is_alphabetic() && !matches!(c, 'T' | 'Z')).count();
    let host = value.contains('.')
        && value.split(['.', ':', '/']).all(|part| part.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '*'));
    let file = value.rsplit_once('.').is_some_and(|(stem, extension)| {
        (1..=5).contains(&extension.len())
            && extension.chars().all(|c| c.is_ascii_alphanumeric())
            && extension.chars().any(|c| c.is_ascii_alphabetic())
            && stem.chars().all(|c| c.is_alphanumeric() || "_-.".contains(c))
    });
    // `NAME=value` and `key:value` are split into their pieces; base64 keeps its padding.
    let assignment = value.trim_end_matches('=').contains('=') || value.contains(':');
    // Ids of workspaces, builds and users far outnumber UUID-shaped keys.
    let uuid = value.split('-').map(str::len).eq([8, 4, 4, 4, 12]) && value.chars().all(|c| c.is_ascii_hexdigit() || c == '-');
    value.chars().count() <= 200
        && letters > 0
        && !host
        && !file
        && !uuid
        && !assignment
        && !contains_email(value)
        && !value.contains('•')
        && !value.contains("://")
        && !value.starts_with(['/', '~', '$', '<', '{', '%', '#'])
        && !value.starts_with("op:")
        && !value.starts_with(SEAL_MARK)
        && !is_email(value)
        && !public_value(value)
        && (value.chars().any(|c| c.is_ascii_digit()) || value.chars().any(|c| PASSWORD_SYMBOLS.contains(&c)))
        && !identifier_like(value.split(['-', '_', '.']))
}

const SEAL_MARK: &str = "🔒";

/// True when a public-id label (`Account ID`, `CLOUDFLARE_ACCOUNT_ID=`,
/// `**Issuer** :`) comes right before `start`.
fn public_label_before(text: &str, start: usize) -> bool {
    let before = text[..start].trim_end_matches(|c: char| c.is_whitespace() || "=:\"'`→*|-()".contains(c));
    let lower = before.to_lowercase();
    PUBLIC_LABELS
        .iter()
        .any(|label| lower.ends_with(label) && !lower[..lower.len() - label.len()].ends_with(char::is_alphanumeric))
}

/// `user@domain.tld` anywhere in the value.
fn contains_email(value: &str) -> bool {
    value.match_indices('@').any(|(at, _)| {
        let domain: String = value[at + 1..].chars().take_while(|c| c.is_alphanumeric() || *c == '.' || *c == '-').collect();
        at > 0 && value[..at].ends_with(|c: char| c.is_alphanumeric() || "._+-".contains(c)) && {
            let domain = domain.trim_end_matches('.');
            domain.contains('.') && domain.rsplit('.').next().is_some_and(|tld| tld.len() >= 2 && tld.chars().all(char::is_alphabetic))
        }
    })
}

/// `-----BEGIN … PRIVATE KEY-----` up to its END line, or to the end of
/// the key material when the END line is missing.
fn private_keys(text: &str, found: &mut Found) {
    let mut from = 0;
    while let Some(start) = text[from..].find("-----BEGIN ").map(|i| from + i) {
        let header_start = start + "-----BEGIN ".len();
        let Some(header_len) = text[header_start..].find("-----") else { return };
        let label = &text[header_start..header_start + header_len];
        let body_start = header_start + header_len + 5;
        from = body_start;
        if !label.ends_with("PRIVATE KEY") || label.contains('\n') {
            continue;
        }
        let end = match text[body_start..].find("-----END ").map(|i| body_start + i) {
            Some(end_line) => {
                let after = end_line + "-----END ".len();
                text[after..].find("-----").map(|i| after + i + 5).unwrap_or(text.len())
            }
            None => {
                let len = text[body_start..]
                    .find(|c: char| !(c.is_ascii_alphanumeric() || c.is_whitespace() || "+/=\\".contains(c)))
                    .unwrap_or(text.len() - body_start);
                let material = text[body_start..body_start + len].chars().filter(char::is_ascii_alphanumeric).count();
                if material < 40 {
                    continue; // the header is only mentioned
                }
                body_start + len
            }
        };
        found.add(start..end, "private_key");
        from = end;
    }
}

/// The password of URIs with credentials (`postgres://user:password@host`).
fn uri_passwords(text: &str, found: &mut Found) {
    let mut from = 0;
    while let Some(marker) = text[from..].find("://").map(|i| from + i) {
        from = marker + 3;
        let scheme = text[..marker]
            .chars()
            .rev()
            .take_while(|c| c.is_ascii_alphanumeric() || "+.-".contains(*c))
            .count();
        if scheme == 0 {
            continue;
        }
        let rest = &text[from..];
        let token_end = rest
            .find(|c: char| c.is_whitespace() || "\"'<>`)]".contains(c))
            .unwrap_or(rest.len());
        let authority_end = rest[..token_end].find(['/', '?', '#']).unwrap_or(token_end);
        let Some(at) = rest[..authority_end].rfind('@') else { continue };
        let Some(colon) = rest[..at].find(':') else { continue };
        let password = &rest[colon + 1..at];
        let placeholder = password.starts_with(['$', '<', '{', '[', '%'])
            || password.chars().all(|c| matches!(c, '*' | 'x' | 'X' | '•'));
        if !placeholder {
            found.add(from + colon + 1..from + at, "url_password");
        }
    }
}

/// `Authorization: Bearer <token>`, unless the token has a known format.
fn bearer_token(text: &str, word: Range<usize>, found: &mut Found) {
    if !text[word.clone()].eq_ignore_ascii_case("bearer") {
        return;
    }
    let start = skip(text, word.end, |c| c == ' ' || c == '\t');
    if start == word.end {
        return;
    }
    let value = unquoted_value(text, start);
    if value.len() >= 16
        && value.chars().any(|c| c.is_ascii_digit())
        && !value.starts_with(['$', '<', '{', '[', '%'])
        && known_format(value).is_none()
        && !identifier_like(value.split(['-', '_', '.']))
    {
        found.add(start..start + value.len(), "bearer");
    }
}

/// A value announced by its name: `password: …`, `TOKEN=…`, `"api_key": "…"`,
/// `mot de passe : …`, `le mot de passe est `…``, `MDP `…``.
fn named_value(text: &str, words: &[Range<usize>], index: usize, found: &mut Found) {
    let word = &words[index];
    let name = text[word.clone()].to_lowercase();
    let name = name.trim_matches(|c: char| !c.is_alphanumeric() && c != '_' && c != '-');
    let spelled_out = name == "passe"
        && index >= 2
        && text[words[index - 2].clone()].trim_start_matches(|c: char| !c.is_alphanumeric()).eq_ignore_ascii_case("mot")
        && text[words[index - 1].clone()].eq_ignore_ascii_case("de");
    let password = spelled_out || PASSWORD_NAMES.iter().any(|n| name.ends_with(n));
    if (!password && !TOKEN_NAMES.iter().any(|n| name.ends_with(n))) || name.ends_with("1password") {
        return;
    }
    // `?user=a&password=…`: the value ends at the next `&`.
    let name_end = word.start + text[word.clone()].trim_end_matches(|c: char| !c.is_alphanumeric()).len();
    let name_start = text[..name_end]
        .char_indices()
        .rev()
        .take_while(|(_, c)| c.is_alphanumeric() || *c == '_' || *c == '-')
        .last()
        .map_or(name_end, |(i, _)| i);
    let in_query = text[..name_start].ends_with(['?', '&']);
    // A quote after a space opens the value (`MDP `…``); right after the
    // name it closes a JSON or YAML key.
    let spaced = skip(text, word.end, |c| c == ' ' || c == '\t');
    let direct_quote = password && spaced > word.end && text[spaced..].starts_with(['`', '"', '\'']);
    let mut at = if direct_quote { spaced } else { skip(text, word.end, |c| matches!(c, '"' | '\'' | '`' | '\\')) };
    at = skip(text, at, |c| c == ' ' || c == '\t');
    let mut assigned = direct_quote;
    let mut connected = false;
    let mut arrow = false;
    if !direct_quote {
        if password {
            let lower = text[at..].to_lowercase();
            if let Some(connector) = CONNECTORS
                .iter()
                .find(|w| lower.starts_with(*w) && lower[w.len()..].starts_with([' ', '\t']))
            {
                at = skip(text, at + connector.len(), |c| c == ' ' || c == '\t');
                assigned = true;
                connected = true;
            }
        }
        // `:`, `=`, `=>`, `:=` or `→`, also after a connector (`est : …`).
        let tail = &text[at..];
        if let Some(op) = ["=>", ":=", "=", ":", "→"].into_iter().find(|op| tail.starts_with(op)) {
            at += op.len();
            if text[at..].starts_with('=') {
                return; // a comparison (`password == …`)
            }
            assigned = true;
            arrow = op == "→" || op == "=>";
            at = skip(text, at, |c| c == ' ' || c == '\t');
        }
        if !password && !assigned {
            return;
        }
    }
    let (start, value, quote) = match text[at..].strip_prefix("\\\"").map(|_| "\\\"").or_else(|| {
        ["\"", "'", "`"].into_iter().find(|q| text[at..].starts_with(q))
    }) {
        Some(quote) => {
            let start = at + quote.len();
            let line_end = text[start..].find('\n').map(|i| start + i).unwrap_or(text.len());
            let Some(len) = text[start..line_end].find(quote) else { return };
            let value = &text[start..start + len];
            if value.contains(char::is_whitespace) {
                return;
            }
            (start, value, Some(quote))
        }
        None if password && assigned && !in_query => (at, password_value(text, at), None),
        None => (at, unquoted_value(text, at), None),
    };
    // Without a digit or a symbol, only a value set apart in backticks, or
    // quoted after "est"/"is", reads as a password.
    let set_apart = password && assigned && !arrow && (quote == Some("`") || (connected && quote.is_some()));
    if secret_value(value, password, assigned, set_apart) {
        found.add(start..start + value.len(), if password { "password" } else { "generic" });
    }
}

fn skip(text: &str, from: usize, matches: impl Fn(char) -> bool) -> usize {
    text[from..].find(|c: char| !matches(c)).map(|i| from + i).unwrap_or(text.len())
}

/// The value up to a space, a quote or a closing bracket, without a final period.
fn unquoted_value(text: &str, start: usize) -> &str {
    let rest = &text[start..];
    let end = rest
        .find(|c: char| c.is_whitespace() || "\"'`\\,;)]}><&".contains(c))
        .unwrap_or(rest.len());
    rest[..end].trim_end_matches('.')
}

/// A password runs to the next space or quote: `&`, `,` and `;` may be part
/// of it. Only a following `name=` (`;Database=…`) ends it earlier, and a
/// closing bracket it never opened.
fn password_value(text: &str, start: usize) -> &str {
    let rest = &text[start..];
    let mut end = rest.find(|c: char| c.is_whitespace() || "\"'`\\".contains(c)).unwrap_or(rest.len());
    if let Some(cut) = rest[..end]
        .char_indices()
        .find(|(index, c)| matches!(c, '&' | ';') && starts_with_assignment(&rest[index + 1..end]))
        .map(|(index, _)| index)
    {
        end = cut;
    }
    // Markdown bold around the line (`**mot de passe : …**`).
    let mut value = rest[..end].strip_suffix("**").unwrap_or(&rest[..end]).trim_end_matches('.');
    while let Some(last) = value.chars().last() {
        let open = match last {
            ')' => '(',
            ']' => '[',
            '}' => '{',
            '>' => '<',
            _ => break,
        };
        if value.matches(open).count() >= value.matches(last).count() {
            break;
        }
        value = &value[..value.len() - 1];
    }
    value
}

fn starts_with_assignment(text: &str) -> bool {
    let name = text.find(|c: char| !(c.is_ascii_alphanumeric() || c == '_' || c == '-')).unwrap_or(text.len());
    name > 0 && text[name..].starts_with('=')
}

/// Code rather than a value: a call, a path or a closure (`z.string().min(8)`,
/// `bcrypt::hash(&pw,`, `os.getenv(`).
fn is_code(value: &str) -> bool {
    value.contains("()")
        || value.contains("::")
        || value.contains("=>")
        || value.contains("->")
        || value
            .char_indices()
            .any(|(index, c)| c == '(' && value[..index].ends_with(|p: char| p.is_ascii_alphanumeric() || p == '_'))
}

fn secret_value(value: &str, password: bool, assigned: bool, set_apart: bool) -> bool {
    let len = value.chars().count();
    if len < if password { 5 } else { 8 } || len > 200 {
        return false;
    }
    let lower = value.to_lowercase();
    let bare = lower.trim_matches(|c: char| !c.is_alphanumeric());
    let reference = value.starts_with(['$', '<', '(', '[', '{', ')', ']', '}', '•', '%', '*', '&', '/', '~', '.'])
        || ["op://", "env:", "secret:", "process.env", "import.meta", "os.environ", "env."]
            .iter()
            .any(|scheme| lower.starts_with(scheme))
        || lower.contains("://")
        || REFERENCES.contains(&bare)
        || CODE_WORDS.contains(&bare)
        || is_code(value)
        || is_email(value);
    if reference {
        return false;
    }
    let digit = value.chars().any(|c| c.is_ascii_digit());
    let symbol = value.chars().any(|c| PASSWORD_SYMBOLS.contains(&c));
    if !digit && !symbol {
        // A word or a variable name, unless set apart as the value.
        return set_apart
            && value.chars().all(|c| c.is_alphanumeric() || c == '-' || c == '_')
            && !(value.contains('_') && identifier_like(value.split(['_', '-'])));
    }
    if !assigned && (!digit || !symbol) {
        return false; // unannounced: only a value mixing digits and symbols
    }
    let identifier = value.chars().all(|c| c.is_alphanumeric() || "_-.".contains(c));
    if identifier && !symbol {
        if !password && value.chars().all(|c| c.is_ascii_digit()) {
            return false; // a count or a duration (`token: 4096`)
        }
        // Code: `config.password2`, `token_v2_value`. A password may be words
        // and digits (`Soleil2024`), a token may not.
        let code = !password || value.contains(['_', '.']);
        if code && identifier_like(value.split(['_', '-', '.'])) {
            return false;
        }
    }
    true
}

fn is_email(value: &str) -> bool {
    match value.split_once('@') {
        Some((user, domain)) => {
            !user.is_empty()
                && domain.contains('.')
                && !domain.ends_with('.')
                && user.chars().chain(domain.chars()).all(|c| c.is_alphanumeric() || "._+-".contains(c))
        }
        None => false,
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Deterministic fake key material: mixed case and digits, never a real value.
    pub(crate) fn fake(len: usize, seed: u64) -> String {
        const ALPHABET: &[u8] = b"aB3xY9kL2mN8pQ4rS7tU1vW5zA6bC0dEfGhJ";
        let mut state = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                ALPHABET[(state >> 33) as usize % ALPHABET.len()] as char
            })
            .collect()
    }

    /// A Supabase-shaped JWT with the given role and a fake signature.
    pub(crate) fn supabase_jwt(role: &str, seed: u64) -> String {
        use base64::Engine;
        let encode = |json: String| base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(json);
        format!(
            "{}.{}.{}",
            encode(r#"{"alg":"HS256","typ":"JWT"}"#.to_string()),
            encode(format!(r#"{{"iss":"supabase","ref":"{}","role":"{}","iat":1700000000}}"#, fake(20, seed).to_lowercase(), role)),
            fake(43, seed + 1)
        )
    }

    fn positives() -> Vec<(String, &'static str, String)> {
        let mut cases = Vec::new();
        let mut case = |template: &str, kind: &'static str, value: String| {
            cases.push((template.replace("{}", &value), kind, value));
        };
        case("PAT {} pour le CI", "github_pat", format!("github_pat_{}", fake(59, 1)));
        for (seed, prefix) in ["ghp_", "gho_", "ghu_", "ghs_", "ghr_"].iter().enumerate() {
            case("token: {}", "github_token", format!("{}{}", prefix, fake(36, seed as u64 + 2)));
        }
        case("export OPENAI_API_KEY={}", "openai", format!("sk-proj-{}", fake(48, 10)));
        case("clé {}.", "anthropic", format!("sk-ant-api03-{}", fake(80, 11)));
        case("STRIPE_SECRET_KEY={}", "stripe_secret", format!("sk_live_{}", fake(99, 12)));
        case("test {}", "stripe_secret", format!("sk_test_{}", fake(40, 13)));
        case("restricted {}", "stripe_secret", format!("rk_live_{}", fake(40, 14)));
        case("webhook {}", "stripe_webhook", format!("whsec_{}", fake(32, 15)));
        case("RESEND_API_KEY=\"{}\"", "resend", format!("re_{}", fake(32, 16)));
        case("Cloudflare: {}", "cloudflare", format!("cfut_{}", fake(40, 17)));
        case("gitlab {}", "gitlab", format!("glpat-{}", fake(20, 18)));
        case("//registry.npmjs.org/:_authToken={}", "npm", format!("npm_{}", fake(36, 19)));
        case("slack {}", "slack", format!("xoxb-1234567890-1234567890123-{}", fake(24, 20)));
        case("aws_access_key_id = {}", "aws", "AKIAQ7ZK3M9X2B5N8W4R".to_string());
        case("maps {}", "google_api", format!("AIza{}", fake(35, 21)));
        case("SUPABASE_SECRET_KEY={}", "supabase_secret", format!("sb_secret_{}", fake(32, 34)));
        case("**Anon Key** : `{}`", "jwt", supabase_jwt("service_role", 35));
        case(
            "cookie session={}; Path=/",
            "jwt",
            format!("eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.{}", fake(43, 22)),
        );
        case("virement sur {} demain", "iban", "GB82WEST12345698765432".to_string());
        case("IBAN : {}.", "iban", "FR76 3000 6000 0112 3456 7890 189".to_string());
        case(
            "voici la clé\n{}\nfin",
            "private_key",
            format!(
                "-----BEGIN OPENSSH PRIVATE KEY-----\n{}\n{}\n{}\n-----END OPENSSH PRIVATE KEY-----",
                fake(64, 23),
                fake(64, 24),
                fake(20, 25)
            ),
        );
        case("password: {}", "password", "Hx9!kLm2pQ".to_string());
        case("le mot de passe est {}", "password", "Soleil2024".to_string());
        case("mdp={} pour le NAS", "password", "Tr0ub4dor#3".to_string());
        case(r#"{"user":"antoine","password":"{}"}"#, "password", "Zq8$wV3nLp".to_string());
        case("DATABASE_URL=postgres://admin:{}@db.example.com:5432/app", "url_password", "S3cr3tPassw0rd".to_string());
        case("Authorization: Bearer {}", "bearer", fake(40, 26));
        case("long key {} here", "generic", fake(40, 27));
        case("csrf_token={}&next=/", "generic", fake(24, 28));
        case(r#"{"api_key": "{}"}"#, "generic", fake(28, 29));
        // Glued after `token=`, inside JSON quotes, after a newline or a JSON `\n` escape.
        case("token={}", "github_token", format!("ghp_{}", fake(36, 30)));
        case(r#"{"auth":"{}"}"#, "github_pat", format!("github_pat_{}", fake(40, 31)));
        case("premier\n{}", "github_token", format!("ghp_{}", fake(36, 32)));
        case(r#"{"text":"voici\n{}"}"#, "github_token", format!("ghp_{}", fake(36, 33)));
        cases
    }

    #[test]
    fn masks_every_kind_with_its_print() {
        for (text, kind, value) in positives() {
            let out = redact(&text);
            assert!(!out.contains(&value), "{} leaked in {}", kind, out);
            assert!(out.contains(&mask(kind, &value)), "expected {} in {}", mask(kind, &value), out);
            assert_eq!(redact(&out), out, "second pass changed {}", out);
        }
    }

    #[test]
    fn mask_has_the_exact_format_and_a_stable_print() {
        let value = format!("github_pat_{}", fake(40, 40));
        let masked = mask("github_pat", &value);
        let print = &masked["[secret masqué : github_pat ".len()..masked.len() - 1];
        assert!(masked.starts_with("[secret masqué : github_pat ") && masked.ends_with(']'));
        assert_eq!(print.len(), 4);
        assert!(print.chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
        let first = redact(&format!("a {} b", value)).into_owned();
        let second = redact(&format!("token={}", value)).into_owned();
        assert!(first.contains(&masked) && second.contains(&masked));
        assert_ne!(mask("github_pat", &format!("github_pat_{}", fake(40, 41))), masked);
    }

    #[test]
    fn leaves_public_ids_and_ordinary_text_alone() {
        let image = format!("data:image/png;base64,{}", fake(400, 50));
        let stripe_public = format!("pk_live_{}", fake(99, 51));
        let clerk_public = format!("CLERK_PUBLISHABLE_KEY=pk_test_{}", fake(48, 52));
        let supabase_publishable = format!("PUBLIC_SUPABASE_KEY=`sb_publishable_{}`", fake(31, 53));
        let anon = format!("NEXT_PUBLIC_SUPABASE_ANON_KEY={}", supabase_jwt("anon", 54));
        let anon_token = format!("token: {} et Bearer {}", supabase_jwt("anon", 56), supabase_jwt("anon", 56));
        let truncated = format!("`{}...` (ta clé anon)", supabase_jwt("anon", 58).split('.').next().unwrap());
        let unsigned = supabase_jwt("anon", 60).rsplit_once('.').unwrap().0.to_string();
        let twilio = format!("Account SID : AC{} - api_key: AC{}", fake_hex("twilio"), fake_hex("twilio2"));
        let negatives = [
            "commit 3f9c2a1b7e4d8f0a6c5b9e2d1f3a7c8b9d0e1f2a puis a1b2c3d",
            "uuid 550e8400-e29b-41d4-a716-446655440000",
            "![logo](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==)",
            &image,
            "Account ID Cloudflare a4b52d0892a12d1c1bbd8bf93c70c3f7",
            "App ID 6758370345, Team ID TB8CC687M3, bundle com.soflution.prive",
            "Key ID 77ZLGAGK7W, Issuer ID 69a6de70-03db-47e3-e053-5b8c7c11a4d1",
            "aws_access_key_id = AKIAIOSFODNN7EXAMPLE",
            &stripe_public,
            &clerk_public,
            &supabase_publishable,
            &anon,
            &anon_token,
            &truncated,
            &unsigned,
            &twilio,
            "re_render et re_export, sk-learn (scikit-learn)",
            "/Users/antoinepinelli/Cursor/App/MemoryPilot/src/hooks.rs et ~/Cursor/App/Konta/apps/web",
            "https://docs.example.com/api/v2?page=3#auth postgres://localhost:5432/app git@github.com:Soflutionltd/MemoryPilot.git",
            "op://Soflution/Resend API Key/credential",
            "mot de passe oublié",
            "le mot de passe est dans 1Password",
            "password: required, Password reset link sent",
            "On garde Postgres pour la prod, le déploiement passe par launchctl kickstart.",
            "Le rapprochement est corrigé et testé : les virements Qonto passent par verifyMatch.",
            "let token = std::env::var(\"GITHUB_TOKEN\")?;",
            "fn login(password: &str, token: Option<String>) -> Result<(), Error> { if password == confirm { return Ok(()) } }",
            "pub struct Config { pub api_key: String, pub secret: Vec<u8> }",
            "const apiKey = process.env.RESEND_API_KEY;",
            "type Props = { token: string; password?: string };",
            "password: z.string().min(8),",
            "headers: { Authorization: `Bearer ${token}` }",
            "useInfiniteQueryWithPagination2Options test_redacts_github_pat_v2_in_JSON_payloads",
            "max_tokens: 4096, \"token_type\": \"Bearer\", token: 3600",
            "sha256:9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
        ];
        for text in negatives {
            assert_eq!(redact(text), text);
        }
    }
    fn fake_hex(seed: &str) -> String {
        Sha256::digest(seed.as_bytes())[..16].iter().map(|byte| format!("{:02x}", byte)).collect()
    }

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("mp-secrets-{}-{}-{}", name, std::process::id(), uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn password_keeps_its_last_characters() {
        for value in ["Ab3xQ9k&!", "Zq8wV3nL;", "Tr0ub4dor#,", "Kp4!m&Zr2&!"] {
            for template in ["password: {} et la suite", "mdp={}", "le mot de passe est {}", "DB_PASSWORD={}"] {
                let text = template.replace("{}", value);
                let out = redact_with(&text, &KnownSecrets::default());
                assert_eq!(out, template.replace("{}", &mask("password", value)), "{}", text);
            }
        }
        let url = "https://example.com/login?user=a&password=Xy9!k&next=1";
        assert_eq!(
            redact(url),
            format!("https://example.com/login?user=a&password={}&next=1", mask("password", "Xy9!k"))
        );
        let listed = "Server=db;User Id=sa;Password=Abc123!x;Database=app";
        assert_eq!(redact(listed), format!("Server=db;User Id=sa;Password={};Database=app", mask("password", "Abc123!x")));
    }

    #[test]
    fn password_announced_in_a_sentence() {
        let cases = [
            ("le mot de passe est `Soleil2026@@`", "Soleil2026@@"),
            ("password is Hx9!kLm2", "Hx9!kLm2"),
            ("MDP : `abcdEFGH`", "abcdEFGH"),
            ("MDP `abcdEFGH` pour le NAS", "abcdEFGH"),
            ("le mot de passe est `soleilbleu`", "soleilbleu"),
            ("mot de passe : Rz7#pLq2", "Rz7#pLq2"),
            ("le mot de passe vaut \"Qw3rty!9\"", "Qw3rty!9"),
            ("**Mot de passe : Abc123!x**", "Abc123!x"),
        ];
        for (text, value) in cases {
            let out = redact_with(text, &KnownSecrets::default());
            assert!(!out.contains(value) && out.contains(&mask("password", value)), "{} -> {}", text, out);
        }
        let negatives = [
            "le mot de passe est dans 1Password",
            "password is required",
            "mot de passe oublié",
            "le mot de passe est oublié",
            "le mot de passe est requis",
            "le mot de passe est changé",
            "the password is wrong",
            r#"{"password": "Passwort", "forgotPassword": "Passwort vergessen?"}"#,
            "password: `string`",
            "let hash = bcrypt::hash(&password, 12)?;",
            "password = os.getenv(\"DB_PASSWORD\", \"\")",
            "password: z.string().min(8).max(64),",
            "`app.example.io/signup/set-password` → `app.example.io/signup/set-password`",
            "postgres://postgres.abcdef:[TON-PASSWORD]@aws-0-eu-west-3.pooler.supabase.com:5432/postgres",
            "**1Password est verrouillé.** L'onglet reste ouvert",
            "après MDP → `submit` auto",
            "Fallback `/etc/passwd` : `chemin/vers/fichier.txt`",
        ];
        for text in negatives {
            assert_eq!(redact_with(text, &KnownSecrets::default()), text);
        }
    }

    #[test]
    fn known_secret_from_the_prints_file() {
        let token = fake_hex("known-file");
        let text = format!("export BREVO_KEY_ID={} puis `{}`.", token, token);
        assert_eq!(redact_with(&text, &KnownSecrets::default()), text, "an unknown hex token stays");
        let dir = temp_dir("file");
        let file = dir.join(KNOWN_SECRETS_FILE);
        assert!(add_known_secret("", &file).is_err());
        assert!(add_known_secret(" \n", &file).is_err());
        assert!(!file.exists());
        let short = add_known_secret(&format!("{}\n", token), &file).unwrap();
        assert_eq!(add_known_secret(&token, &file).unwrap(), short);
        let content = std::fs::read_to_string(&file).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 1);
        assert!(lines[0].len() == 64 && lines[0].bytes().all(|b| b.is_ascii_hexdigit()));
        assert!(lines[0].starts_with(&short) && !content.contains(&token));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(std::fs::metadata(&file).unwrap().permissions().mode() & 0o777, 0o600);
        }
        let known = load_known_secrets_from(Some(&file), None);
        let masked = mask("known", &token);
        assert_eq!(masked.len(), mask("generic", &token).len() - "generic".len() + "known".len());
        assert_eq!(redact_with(&text, &known), format!("export BREVO_KEY_ID={} puis `{}`.", masked, masked));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn known_secret_from_a_credential() {
        let token = fake_hex("known-credential");
        let account = fake_hex("public-account");
        let dir = temp_dir("credential");
        let path = dir.join("memory.db");
        {
            let db = crate::db::Database::open_at(&path).unwrap();
            let content = format!("Cloudflare Account ID {} — jeton d'API {} (login admin@example.com)", account, token);
            let (stored, _) = db
                .add_memory(&content, "credential", Some("test"), &[], "test", 3, None, None, &Default::default())
                .unwrap();
            assert!(stored.content.starts_with(crate::db::SEAL_PREFIX));
        }
        let known = load_known_secrets_from(Some(&dir.join(KNOWN_SECRETS_FILE)), Some(&path));
        let text = format!("le token {} du compte {} (admin@example.com)", token, account);
        let out = redact_with(&text, &known);
        assert_eq!(out, format!("le token {} du compte {} (admin@example.com)", mask("known", &token), account));
        assert!(redact_with(&text, &KnownSecrets::default()).contains(&token));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn credential_tokens_skip_public_values() {
        let mut known = KnownSecrets::default();
        let clear = "Prod DB postgres://admin:Pg5ecret!2026x@db.example.com:5432/app, host db.example.com:5432, \
                     user admin@example.com, client_id=Abc123DefGhi456, 2026-10-31T12:00:00Z";
        let account = fake_hex("account-id");
        let issuer = "69a6de70-03db-47e3-e053-5b8c7c11a4d1";
        let client_secret = format!("{}/{}=", fake(24, 90), fake(7, 91));
        let second = format!(
            "CLOUDFLARE_ACCOUNT_ID={} **Issuer** : {} AuthKey_77ZLGAGK7W.p8 NODE_ENV=production \
             550e8400-e29b-41d4-a716-446655440000 \
             https://app.example.com/projects/a93315ed/web/app1b2c3d4e5f6 CLIENT_SECRET={}",
            account, issuer, client_secret
        );
        let anon = supabase_jwt("anon", 92);
        let twilio = format!("AC{}", fake_hex("twilio-sid"));
        let publishable = format!("sb_publishable_{}", fake(31, 93));
        let third = format!("Cloudflare {} et {} Anon Key {} SID {} {}", account, issuer, anon, twilio, publishable);
        known.insert_credential_texts(&[clear, second.as_str(), third.as_str()]);
        assert!(known.contains("Pg5ecret!2026x"));
        assert!(known.contains(&client_secret));
        for public in [
            "db.example.com:5432",
            "admin@example.com",
            "Abc123DefGhi456",
            "2026-10-31T12:00:00Z",
            &account,
            issuer,
            "AuthKey_77ZLGAGK7W.p8",
            "NODE_ENV=production",
            "app1b2c3d4e5f6",
            "550e8400-e29b-41d4-a716-446655440000",
            &anon,
            &twilio,
            &publishable,
        ] {
            assert!(!known.contains(public), "{} taken as a secret", public);
        }
        // Symbols glued after a known part are masked with it.
        let mut known = KnownSecrets::default();
        known.insert("9182736450aZ");
        let out = redact_with("`9182736450aZ@&!` puis 9182736450aZ.", &known);
        assert_eq!(out, format!("`{}` puis {}.", mask("known", "9182736450aZ@&!"), mask("known", "9182736450aZ")));
    }
}
