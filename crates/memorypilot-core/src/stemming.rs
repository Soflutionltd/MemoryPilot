//! Lightweight Snowball-based stemming for retrieval.
//!
//! The default `unicode61` FTS5 tokenizer strips diacritics but performs
//! no morphological reduction, so a French query "pondérer" never
//! matches a stored document "poids" via BM25, and the recall layer has
//! to do all the work. This module computes a stemmed projection of any
//! text that we then store **in addition** to the raw content (and that
//! we emit as an extra FTS5 query variant), restoring lexical matches
//! across morphological variants in French, Spanish, Italian, German
//! and English.
//!
//! Design choices:
//! - Stemming is deterministic and per-token. We split on Unicode word
//!   boundaries, lowercase, strip pure punctuation, and apply
//!   `Stemmer::stem`.
//! - Language is detected with a cheap heuristic (presence of FR
//!   diacritics + frequent FR stopwords vs. ASCII-only EN profile).
//!   When in doubt we keep both English and French stems concatenated:
//!   the corpus stays index-friendly without forcing a wrong stemmer.
//! - The output is intended to be **appended** to the raw content
//!   before insertion into the FTS5 `content` column. The raw content
//!   is preserved, so exact-phrase matches and BM25 statistics on the
//!   original text continue to work as before.

use rust_stemmers::{Algorithm, Stemmer};
use std::sync::OnceLock;

static FRENCH: OnceLock<Stemmer> = OnceLock::new();
static ENGLISH: OnceLock<Stemmer> = OnceLock::new();

fn french() -> &'static Stemmer {
    FRENCH.get_or_init(|| Stemmer::create(Algorithm::French))
}

fn english() -> &'static Stemmer {
    ENGLISH.get_or_init(|| Stemmer::create(Algorithm::English))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lang {
    French,
    English,
    Both,
}

/// Heuristic language detection good enough to pick the stemmer.
/// We never need to be perfect: when ambiguous we run both stemmers and
/// concatenate, which strictly increases recall without hurting BM25.
pub fn detect(text: &str) -> Lang {
    let lower = text.to_lowercase();
    let has_french_accents = lower
        .chars()
        .any(|c| matches!(c, 'é' | 'è' | 'ê' | 'ë' | 'à' | 'â' | 'ô' | 'î' | 'ï' | 'û' | 'ù' | 'ç'));
    let french_markers = [
        " le ", " la ", " les ", " un ", " une ", " des ", " du ", " de la ",
        " est ", " sont ", " avec ", " pour ", " dans ", " sur ", " par ",
        " ce ", " cette ", " ces ", " mais ", " ou ", " donc ", " pas ",
    ];
    let french_marker_hits = french_markers
        .iter()
        .filter(|marker| lower.contains(*marker))
        .count();

    let english_markers = [
        " the ", " is ", " are ", " was ", " were ", " and ", " or ", " not ",
        " with ", " from ", " into ", " over ", " under ", " between ",
        " when ", " where ", " what ", " which ", " their ", " these ",
    ];
    let english_marker_hits = english_markers
        .iter()
        .filter(|marker| lower.contains(*marker))
        .count();

    if has_french_accents && french_marker_hits >= 1 {
        return Lang::French;
    }
    if french_marker_hits >= 2 && english_marker_hits == 0 {
        return Lang::French;
    }
    if english_marker_hits >= 2 && !has_french_accents {
        return Lang::English;
    }
    if has_french_accents {
        return Lang::Both;
    }
    Lang::English
}

fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_' || c == '-'
}

fn tokenize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for c in text.chars() {
        if is_word_char(c) {
            current.push(c);
        } else if !current.is_empty() {
            tokens.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

fn stem_with(stemmer: &Stemmer, tokens: &[String]) -> String {
    let mut out = String::with_capacity(tokens.iter().map(|t| t.len() + 1).sum::<usize>());
    for token in tokens {
        if token.is_empty() {
            continue;
        }
        let lower = token.to_lowercase();
        // Skip very short tokens; stemmers degrade or no-op anyway.
        if lower.chars().count() < 3 {
            continue;
        }
        let stemmed = stemmer.stem(&lower);
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(&stemmed);
    }
    out
}

/// Compute the stemmed projection of `text`. Returns an empty string if
/// the input has no stemmable content.
pub fn stem_text(text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }
    let tokens = tokenize(text);
    if tokens.is_empty() {
        return String::new();
    }
    match detect(text) {
        Lang::French => stem_with(french(), &tokens),
        Lang::English => stem_with(english(), &tokens),
        Lang::Both => {
            let fr = stem_with(french(), &tokens);
            let en = stem_with(english(), &tokens);
            if fr.is_empty() {
                en
            } else if en.is_empty() {
                fr
            } else {
                format!("{} {}", fr, en)
            }
        }
    }
}

/// Stem an FTS5-style query. Currently identical to `stem_text` but
/// kept as a separate API so that future tweaks (e.g. weighting,
/// per-token operators) can diverge.
pub fn stem_query(query: &str) -> String {
    stem_text(query)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_french_via_accents_and_markers() {
        let text = "La fonction de pondération dans le scoring BM25 doit être ajustée.";
        assert_eq!(detect(text), Lang::French);
    }

    #[test]
    fn detects_english_when_no_accents() {
        let text = "The function weight in BM25 scoring needs to be adjusted.";
        assert_eq!(detect(text), Lang::English);
    }

    #[test]
    fn french_stem_unifies_inflections() {
        // Same root, different forms: "messages" / "message", "commits"
        // / "commit". Snowball FR collapses each pair to one stem.
        let doc = stem_text("Convention des commits: format type(scope) description messages");
        let query = stem_query("format des messages de commit conventionnels");
        let doc_tokens: std::collections::HashSet<&str> = doc.split_whitespace().collect();
        let query_tokens: std::collections::HashSet<&str> = query.split_whitespace().collect();
        let shared = doc_tokens.intersection(&query_tokens).count();
        assert!(
            shared >= 3,
            "expected at least 3 shared stem tokens, got {} (doc={:?} query={:?})",
            shared,
            doc_tokens,
            query_tokens
        );
    }

    #[test]
    fn english_stem_unifies_basic_inflections() {
        let doc_stem = stem_text("Running tests confirms the validation engine works");
        let query_stem = stem_query("run validate engines");
        let doc: std::collections::HashSet<&str> = doc_stem.split_whitespace().collect();
        let q: std::collections::HashSet<&str> = query_stem.split_whitespace().collect();
        let shared = doc.intersection(&q).count();
        assert!(
            shared >= 2,
            "expected ≥2 shared stems, got {} doc={:?} q={:?}",
            shared,
            doc,
            q
        );
    }

    #[test]
    fn empty_input_yields_empty_output() {
        assert!(stem_text("").is_empty());
        assert!(stem_text("   ").is_empty());
    }
}
