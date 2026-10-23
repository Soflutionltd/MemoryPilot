//! Full-text normalisation shared by indexing and querying, so Postgres
//! matches what SQLite's `unicode61 remove_diacritics 2` tokenizer matches,
//! and the FTS5 queries built by `memorypilot_core::fts` translated to
//! `to_tsquery('simple', …)`.

use unicode_normalization::{char::is_combining_mark, UnicodeNormalization};

/// Lowercased, accents removed.
pub fn fold(text: &str) -> String {
    text.nfd().filter(|c| !is_combining_mark(*c)).collect::<String>().to_lowercase()
}

/// What the `fts` column indexes: the content and its stemmed projection,
/// folded, every other character a space (Postgres' parser would keep
/// `example.com` or an e-mail address as one token; FTS5 splits them).
pub fn index_text(content: &str) -> String {
    let stem = memorypilot_core::stemming::stem_text(content);
    let text = if stem.is_empty() { fold(content) } else { fold(&format!("{} {}", content, stem)) };
    text.chars().map(|c| if c.is_alphanumeric() { c } else { ' ' }).collect()
}

struct Segment {
    text: String,
    prefix: bool,
}

/// Quoted strings (`""` escapes a quote) and bare words, each with whether
/// a `*` follows it.
fn segments(query: &str) -> Vec<Segment> {
    let mut found = Vec::new();
    let mut chars = query.chars().peekable();
    while let Some(&next) = chars.peek() {
        if next.is_whitespace() {
            chars.next();
            continue;
        }
        let mut text = String::new();
        if next == '"' {
            chars.next();
            while let Some(c) = chars.next() {
                if c == '"' {
                    if chars.peek() == Some(&'"') {
                        chars.next();
                        text.push('"');
                    } else {
                        break;
                    }
                } else {
                    text.push(c);
                }
            }
        } else {
            while let Some(&c) = chars.peek() {
                if c.is_whitespace() || c == '"' || c == '*' {
                    break;
                }
                text.push(c);
                chars.next();
            }
        }
        let prefix = chars.peek() == Some(&'*');
        if prefix {
            chars.next();
        }
        found.push(Segment { text, prefix });
    }
    found
}

/// A segment's words as consecutive lexemes, the last one a prefix match
/// when the segment had `*`. None when it has no word.
fn segment_query(segment: &Segment) -> Option<String> {
    let folded = fold(&segment.text);
    let words: Vec<&str> = folded.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).collect();
    let last = words.len().checked_sub(1)?;
    Some(
        words
            .iter()
            .enumerate()
            .map(|(i, word)| if i == last && segment.prefix { format!("'{}':*", word) } else { format!("'{}'", word) })
            .collect::<Vec<_>>()
            .join(" <-> "),
    )
}

/// The `to_tsquery` text for an FTS5 query: every segment required, the
/// words of one segment adjacent (a quoted phrase), `NEAR(…, n)` read as
/// all of its terms anywhere in the memory. None when no word is left.
pub fn fts5_to_tsquery(query: &str) -> Option<String> {
    let trimmed = query.trim();
    let terms = match trimmed.strip_prefix("NEAR(") {
        Some(inner) => inner.rsplit_once(',').map_or(inner, |(terms, _)| terms),
        None => trimmed,
    };
    let parts: Vec<String> = segments(terms).iter().filter_map(segment_query).collect();
    if parts.is_empty() {
        None
    } else {
        Some(parts.join(" & "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn folds_case_and_accents() {
        assert_eq!(fold("Déploiement À Noël"), "deploiement a noel");
    }

    #[test]
    fn translates_the_variants_search_builds() {
        assert_eq!(fts5_to_tsquery("deploy").as_deref(), Some("'deploy'"));
        assert_eq!(fts5_to_tsquery("\"depl\"* \"Fri\"*").as_deref(), Some("'depl':* & 'fri':*"));
        assert_eq!(fts5_to_tsquery("\"friday evenings\"").as_deref(), Some("'friday' <-> 'evenings'"));
        assert_eq!(fts5_to_tsquery("NEAR(\"never\" \"evenings\", 8)").as_deref(), Some("'never' & 'evenings'"));
        assert_eq!(fts5_to_tsquery("\"CGEvent.tapCreate\"*").as_deref(), Some("'cgevent' <-> 'tapcreate':*"));
        assert_eq!(fts5_to_tsquery("\"café\"*").as_deref(), Some("'cafe':*"));
    }

    #[test]
    fn quotes_and_operators_never_reach_tsquery() {
        let translated = fts5_to_tsquery("\"it\"\"s\"* \"a'b\" \"x|y&z!\"*").expect("words");
        assert_eq!(translated, "'it' <-> 's':* & 'a' <-> 'b' & 'x' <-> 'y' <-> 'z':*");
        assert!(fts5_to_tsquery("\"\"* \"...\"").is_none());
        assert!(fts5_to_tsquery("   ").is_none());
    }

    #[test]
    fn indexes_folded_content_with_its_stems() {
        let indexed = index_text("Les Déploiements");
        assert!(indexed.starts_with("les deploiements"));
        assert!(indexed.len() > "les deploiements".len(), "stemmed projection appended");
        assert!(index_text("see docs.example.com").starts_with("see docs example com"));
    }
}
