//! Secret redaction for text stored as said (conversations, hook
//! transcripts). One filter: known credential formats, private key
//! blocks, passwords in URIs, values after a password or token name,
//! bearer tokens, IBANs and long random tokens.
//!
//! Each value becomes `[secret masqué : <kind> <print>]`, the print being
//! the first 4 hex digits of its SHA-256: two mentions of the same secret
//! stay recognisable without revealing it. Masks already in the text are
//! left alone, so redacting twice changes nothing.

use sha2::{Digest, Sha256};
use std::borrow::Cow;
use std::ops::Range;

const MASK_OPEN: &str = "[secret masqué : ";

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
];

/// Public ids that only point at a resource: never masked, however random.
const PUBLIC_PREFIXES: &[&str] = &[
    "pk_live_", "pk_test_", "acct_", "prod_", "price_", "plink_", "cus_", "sub_", "evt_", "we_", "prj_", "team_",
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
    "required",
];

const PASSWORD_SYMBOLS: &[char] = &['!', '@', '#', '$', '%', '^', '&', '*', '?', '+', '=', '~', '§', '€', '£'];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finding {
    pub range: Range<usize>,
    pub kind: &'static str,
}

/// The text with every secret value replaced by its mask.
pub fn redact(text: &str) -> Cow<'_, str> {
    let found = findings(text);
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
    let mut out = found.out;
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
/// `mot de passe : …`.
fn named_value(text: &str, words: &[Range<usize>], index: usize, found: &mut Found) {
    let word = &words[index];
    let name = text[word.clone()].to_lowercase();
    let name = name.trim_matches(|c: char| !c.is_alphanumeric() && c != '_' && c != '-');
    let spelled_out = name == "passe"
        && index >= 2
        && text[words[index - 2].clone()].eq_ignore_ascii_case("mot")
        && text[words[index - 1].clone()].eq_ignore_ascii_case("de");
    let password = spelled_out || PASSWORD_NAMES.iter().any(|n| name.ends_with(n));
    if !password && !TOKEN_NAMES.iter().any(|n| name.ends_with(n)) {
        return;
    }
    // Closing quote of a JSON or YAML key, then `:`, `=`, `=>`, `:=` or `→`.
    let mut at = skip(text, word.end, |c| matches!(c, '"' | '\'' | '`' | '\\'));
    at = skip(text, at, |c| c == ' ' || c == '\t');
    let tail = &text[at..];
    let operator = ["=>", ":=", "=", ":", "→"].into_iter().find(|op| tail.starts_with(op));
    let mut assigned = operator.is_some();
    if let Some(op) = operator {
        at += op.len();
        if text[at..].starts_with('=') {
            return; // a comparison (`password == …`)
        }
    }
    at = skip(text, at, |c| c == ' ' || c == '\t');
    if password {
        let lower = text[at..].to_lowercase();
        if let Some(connector) = CONNECTORS
            .iter()
            .find(|w| lower.starts_with(*w) && lower[w.len()..].starts_with([' ', '\t']))
        {
            at = skip(text, at + connector.len(), |c| c == ' ' || c == '\t');
            assigned = true;
        }
    } else if !assigned {
        return;
    }
    let (start, value) = match text[at..].strip_prefix("\\\"").map(|_| "\\\"").or_else(|| {
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
            (start, value)
        }
        None => (at, unquoted_value(text, at)),
    };
    if secret_value(value, password, assigned) {
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

fn secret_value(value: &str, password: bool, assigned: bool) -> bool {
    let len = value.chars().count();
    if len < if password { 6 } else { 8 } || len > 200 {
        return false;
    }
    let lower = value.to_lowercase();
    let reference = value.starts_with(['$', '<', '(', '[', '{', '•', '%', '*', '&', '/', '~', '.'])
        || ["op://", "env:", "secret:", "process.env", "import.meta", "os.environ", "env."]
            .iter()
            .any(|scheme| lower.starts_with(scheme))
        || lower.contains("://")
        || REFERENCES.contains(&lower.trim_matches(|c: char| !c.is_alphanumeric()))
        || is_email(value);
    if reference {
        return false;
    }
    let digit = value.chars().any(|c| c.is_ascii_digit());
    let symbol = value.chars().any(|c| PASSWORD_SYMBOLS.contains(&c));
    if !digit && !symbol {
        return false; // a word or a variable name
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
        let negatives = [
            "commit 3f9c2a1b7e4d8f0a6c5b9e2d1f3a7c8b9d0e1f2a puis a1b2c3d",
            "uuid 550e8400-e29b-41d4-a716-446655440000",
            "![logo](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==)",
            &image,
            "Account ID Cloudflare a4b52d0892a12d1c1bbd8bf93c70c3f7",
            "App ID 6758370345, Team ID TB8CC687M3, bundle com.soflution.prive",
            "Key ID 77ZLGAGK7W, Issuer ID 69a6de70-03db-47e3-e053-5b8c7c11a4d1",
            &stripe_public,
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
}
