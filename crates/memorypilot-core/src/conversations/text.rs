//! Text handling for indexed conversations: cleaning what the agent
//! harness injects into user turns, pulling commit / file references out
//! of a turn, and the edit-distance helpers behind typo-tolerant search.

use std::collections::BTreeSet;

/// Blocks Cursor and Claude Code wrap around what the user typed. They
/// are harness context, not something the user said.
const INJECTED_BLOCKS: &[&str] = &[
    "system_reminder",
    "system-reminder",
    "system_notification",
    "attached_files",
    "open_and_recently_viewed_files",
    "timestamp",
    "user_info",
    "git_status",
    "rules",
    "agent_transcripts",
    "agent_skills",
    "command-name",
    "command-message",
    "command-args",
    "local-command-stdout",
    "local-command-caveat",
];

pub const MAX_USER_CHARS: usize = 3_000;
pub const MAX_ANSWER_CHARS: usize = 3_000;
pub const MAX_NOTES_CHARS: usize = 1_500;

/// What the user actually typed in a turn.
pub fn clean_user_text(raw: &str) -> String {
    if let Some(inner) = between(raw, "<user_query>", "</user_query>") {
        return truncate_chars(inner.trim(), MAX_USER_CHARS);
    }
    let mut text = raw.to_string();
    for tag in INJECTED_BLOCKS {
        text = strip_block(&text, tag);
    }
    truncate_chars(text.trim(), MAX_USER_CHARS)
}

pub fn truncate_chars(text: &str, max: usize) -> String {
    match text.char_indices().nth(max) {
        Some((index, _)) => text[..index].to_string(),
        None => text.to_string(),
    }
}

fn between<'a>(text: &'a str, open: &str, close: &str) -> Option<&'a str> {
    let start = text.find(open)? + open.len();
    let end = text[start..].find(close).map(|i| start + i).unwrap_or(text.len());
    Some(&text[start..end])
}

fn strip_block(text: &str, tag: &str) -> String {
    let open = format!("<{}", tag);
    let close = format!("</{}>", tag);
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find(&open) {
        let after_name = &rest[start + open.len()..];
        if !after_name.starts_with('>') && !after_name.starts_with(' ') {
            out.push_str(&rest[..start + open.len()]);
            rest = after_name;
            continue;
        }
        out.push_str(&rest[..start]);
        match rest[start..].find(&close) {
            Some(end) => rest = &rest[start + end + close.len()..],
            None => {
                rest = "";
                break;
            }
        }
    }
    out.push_str(rest);
    out
}

/// Lowercased, accent-free word tokens used for the vocabulary and for
/// query terms. Mirrors FTS5 `unicode61 remove_diacritics 2` closely
/// enough for typo matching.
pub fn word_tokens(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(fold)
        .collect()
}

pub fn fold(token: &str) -> String {
    use unicode_normalization::UnicodeNormalization;
    token
        .nfd()
        .filter(|c| !unicode_normalization::char::is_combining_mark(*c))
        .flat_map(char::to_lowercase)
        .collect()
}

const STOPWORDS: &[&str] = &[
    "le", "la", "les", "un", "une", "des", "du", "de", "d", "l", "et", "ou", "a", "au", "aux", "en",
    "dans", "sur", "pour", "par", "avec", "sans", "que", "qui", "quoi", "est", "c", "ce", "ca",
    "cette", "ces", "on", "nous", "vous", "tu", "je", "j", "il", "elle", "ils", "y", "ne", "pas",
    "plus", "se", "sa", "son", "ses", "mon", "ma", "mes", "ton", "ta", "tes", "notre", "nos",
    "leur", "leurs", "lui", "me", "te", "moi", "toi", "avait", "avais", "avoir", "ete", "etait",
    "fait", "faire", "truc", "machin", "chose", "bon", "ben", "bah", "alors", "donc", "voila",
    "comme", "quand", "ou", "the", "of", "and", "to", "in", "is", "it", "for", "on", "with", "that",
    "this", "was", "we", "our", "conversation", "discussion", "parle", "parlait", "bosse",
    "bossait", "travaille", "travaillait", "souviens", "rappelle", "retrouve",
];

pub fn is_stopword(token: &str) -> bool {
    STOPWORDS.contains(&token)
}

/// Query terms worth matching: folded, no stopwords, no single letters.
pub fn query_terms(query: &str) -> Vec<String> {
    let mut seen = BTreeSet::new();
    word_tokens(query)
        .into_iter()
        .filter(|t| t.chars().count() >= 2 && !is_stopword(t))
        .filter(|t| seen.insert(t.clone()))
        .collect()
}

/// Optimal string alignment distance (Damerau-Levenshtein restricted to
/// adjacent transpositions), stopping early once `max` is exceeded.
pub fn edit_distance_within(a: &str, b: &str, max: usize) -> Option<usize> {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    if a.len().abs_diff(b.len()) > max {
        return None;
    }
    let width = b.len() + 1;
    let mut prev2 = vec![0usize; width];
    let mut prev: Vec<usize> = (0..width).collect();
    let mut cur = vec![0usize; width];
    for i in 1..=a.len() {
        cur[0] = i;
        let mut row_min = cur[0];
        for j in 1..=b.len() {
            let cost = usize::from(a[i - 1] != b[j - 1]);
            let mut value = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
            if i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1] {
                value = value.min(prev2[j - 2] + 1);
            }
            cur[j] = value;
            row_min = row_min.min(value);
        }
        if row_min > max {
            return None;
        }
        std::mem::swap(&mut prev2, &mut prev);
        std::mem::swap(&mut prev, &mut cur);
    }
    let distance = prev[b.len()];
    (distance <= max).then_some(distance)
}

/// Edits tolerated for a query term: speech-to-text slips on proper
/// nouns ("Gina" for "Jina") are one edit; short words get none, or
/// every 3-letter word would match a dozen others.
pub fn allowed_edits(term: &str) -> usize {
    match term.chars().count() {
        0..=3 => 0,
        4..=7 => 1,
        _ => 2,
    }
}

/// Speech-to-text often splits or glues proper nouns ("Memory Pilot" for
/// MemoryPilot, "Docix" for "Doc X"): adjacent terms are also tried
/// glued together.
pub fn glued_pairs(terms: &[String]) -> Vec<String> {
    terms
        .windows(2)
        .map(|pair| format!("{}{}", pair[0], pair[1]))
        .filter(|glued| glued.chars().count() >= 6)
        .collect()
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Reference {
    pub kind: &'static str,
    pub value: String,
}

/// Commits and files a turn talks about. Commits must look like a short
/// or full SHA *and* sit next to the word commit or inside backticks, so
/// that hex-looking ids (UUIDs, colours) are not taken for commits.
pub fn extract_references(text: &str) -> Vec<Reference> {
    let mut refs = BTreeSet::new();
    for (index, token) in token_spans(text) {
        let bare = token.trim_matches(|c: char| "`(),.:;".contains(c));
        if is_sha(bare) {
            let preceding: Vec<char> = text[..index].chars().rev().take(24).collect();
            let before: String = preceding.into_iter().rev().collect::<String>().to_lowercase();
            let backticked = token.starts_with('`');
            if backticked || before.contains("commit") {
                refs.insert(Reference { kind: "commit", value: bare.to_lowercase() });
            }
        } else if let Some(file) = file_reference(bare) {
            refs.insert(Reference { kind: "file", value: file });
        }
    }
    refs.into_iter().collect()
}

fn token_spans(text: &str) -> impl Iterator<Item = (usize, &str)> {
    text.split_whitespace().map(move |token| {
        let index = token.as_ptr() as usize - text.as_ptr() as usize;
        (index, token)
    })
}

fn is_sha(token: &str) -> bool {
    let len = token.len();
    (7..=40).contains(&len)
        && token.chars().all(|c| c.is_ascii_hexdigit())
        && token.chars().any(|c| c.is_ascii_digit())
        && token.chars().any(|c| c.is_ascii_alphabetic())
}

const FILE_EXTENSIONS: &[&str] = &[
    "rs", "ts", "tsx", "js", "mjs", "cjs", "svelte", "dart", "swift", "kt", "py", "go", "java",
    "sql", "toml", "json", "yaml", "yml", "md", "mdc", "html", "css", "sh", "rb", "php", "vue",
];

fn file_reference(token: &str) -> Option<String> {
    let token = token.trim_matches(|c: char| c == '\'' || c == '"' || c == '[' || c == ']' || c == '*');
    let token = token.split(':').next().unwrap_or(token);
    let (stem, extension) = token.rsplit_once('.')?;
    if !FILE_EXTENSIONS.contains(&extension) || stem.is_empty() || token.contains("://") {
        return None;
    }
    let name_start = stem.rfind('/').map(|i| i + 1).unwrap_or(0);
    let name = &stem[name_start..];
    if name.is_empty() || !name.chars().all(|c| c.is_alphanumeric() || "-_+.[]".contains(c)) {
        return None;
    }
    if !token.contains('/') && name.chars().count() < 3 {
        return None;
    }
    Some(token.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_only_the_user_query() {
        let raw = "<timestamp>Wednesday</timestamp>\n<user_query>\nLa facture 141 est-elle due ?\n</user_query>";
        assert_eq!(clean_user_text(raw), "La facture 141 est-elle due ?");
    }

    #[test]
    fn strips_injected_blocks_without_user_query() {
        let raw = "<system_reminder>ignore</system_reminder>Relance le build <attached_files a=\"1\">x</attached_files>";
        assert_eq!(clean_user_text(raw), "Relance le build");
    }

    #[test]
    fn one_edit_bridges_a_misheard_name() {
        assert_eq!(edit_distance_within("gina", "jina", 1), Some(1));
        assert_eq!(edit_distance_within("tri", "tir", allowed_edits("tri")), None);
        assert_eq!(edit_distance_within("reranker", "rearnker", 2), Some(1));
        assert_eq!(edit_distance_within("memorypilot", "photopilot", 2), None);
    }

    #[test]
    fn glues_split_proper_nouns() {
        let terms = query_terms("la comparaison Memory Pilot");
        assert!(glued_pairs(&terms).contains(&"memorypilot".to_string()));
    }

    #[test]
    fn drops_spoken_filler_words() {
        assert_eq!(query_terms("Bah, le truc sur la galerie d'art"), vec!["galerie", "art"]);
    }

    #[test]
    fn extracts_commits_only_in_context() {
        let refs = extract_references(
            "Commit 744d524 poussé. Le fichier `apps/web/src/routes/api/reconciliation/invoice/+server.ts` change. id 4aa7ce72-c3ac",
        );
        assert!(refs.contains(&Reference { kind: "commit", value: "744d524".into() }));
        assert!(refs.contains(&Reference {
            kind: "file",
            value: "apps/web/src/routes/api/reconciliation/invoice/+server.ts".into()
        }));
        assert!(!refs.iter().any(|r| r.value.starts_with("4aa7ce72")));
    }
}
