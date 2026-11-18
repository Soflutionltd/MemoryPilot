//! French and English words that mean the same, for searching across
//! the two languages.
//!
//! A question asked in French about something written in English shares
//! no word with it: "musique d'une publication programmée" never meets
//! "music on a scheduled post". Vectors bridge the two, but a long
//! conversation is found by its words first. The lexicon gives each
//! query word its counterparts in the other language; they are searched
//! in a lane of their own, at a lower weight than the words typed.
//! It is a list of everyday and product vocabulary written for this
//! project, not a dictionary: a word it does not know is left alone.

use std::collections::HashMap;
use std::sync::OnceLock;

const SOURCE: &str = include_str!("lexicon_fr_en.txt");

fn fold(word: &str) -> String {
    crate::conversations::text::fold(word.trim())
}

/// The spellings a word is looked up by: itself and its stem, so that
/// "publications" finds "publication".
fn keys(word: &str) -> Vec<String> {
    let folded = fold(word);
    let stem = fold(&crate::stemming::stem_query(&folded));
    if stem.is_empty() || stem == folded {
        vec![folded]
    } else {
        vec![folded, stem]
    }
}

struct Lexicon {
    /// Folded word or stem to the words that mean the same in the other language.
    by_word: HashMap<String, Vec<String>>,
}

fn lexicon() -> &'static Lexicon {
    static LEXICON: OnceLock<Lexicon> = OnceLock::new();
    LEXICON.get_or_init(|| {
        let mut by_word: HashMap<String, Vec<String>> = HashMap::new();
        let mut link = |from: &str, to: &str| {
            let to = fold(to);
            // Expressions are matched word by word in the index: only single words are keys.
            if from.contains(' ') || to == fold(from) {
                return;
            }
            for key in keys(from) {
                let known = by_word.entry(key).or_default();
                for word in to.split_whitespace() {
                    if !known.iter().any(|existing| existing == word) {
                        known.push(word.to_string());
                    }
                }
            }
        };
        for line in SOURCE.lines().filter(|line| !line.starts_with('#')) {
            let Some((french, english)) = line.split_once('|') else { continue };
            for english in english.split(',').map(str::trim).filter(|word| !word.is_empty()) {
                link(french.trim(), english);
                link(english, french.trim());
            }
        }
        Lexicon { by_word }
    })
}

/// The words of the other language that mean the same as `word` (French
/// or English, any case, with or without accents). Empty for a word the
/// lexicon does not know.
pub fn counterparts(word: &str) -> Vec<String> {
    let lexicon = lexicon();
    let mut found: Vec<String> = Vec::new();
    let own = keys(word);
    for key in &own {
        for counterpart in lexicon.by_word.get(key).into_iter().flatten() {
            if !own.contains(counterpart) && !found.contains(counterpart) {
                found.push(counterpart.clone());
            }
        }
    }
    found
}

/// For each of `terms` that has counterparts: the term and one of them.
pub fn translated(terms: &[String]) -> Vec<(String, String)> {
    terms
        .iter()
        .flat_map(|term| counterparts(term).into_iter().map(move |counterpart| (term.clone(), counterpart)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_word_gives_its_counterparts_in_both_directions() {
        assert_eq!(counterparts("musique"), vec!["music".to_string()]);
        assert_eq!(counterparts("Music"), vec!["musique".to_string()]);
        assert!(counterparts("programmée").contains(&"scheduled".to_string()), "accents and agreement are folded");
        assert!(counterparts("publications").contains(&"post".to_string()), "a plural finds its singular");
        assert!(counterparts("scheduled").contains(&"programme".to_string()));
        assert!(counterparts("réponse").contains(&"reply".to_string()));
    }

    #[test]
    fn an_unknown_or_shared_word_is_left_alone() {
        assert!(counterparts("zorglub").is_empty());
        assert!(counterparts("instagram").is_empty());
        // Spelt the same in both languages: nothing to add.
        assert!(!counterparts("forum").contains(&"forum".to_string()));
        assert!(counterparts("").is_empty());
    }

    #[test]
    fn every_line_of_the_lexicon_is_well_formed() {
        let lines: Vec<&str> = SOURCE.lines().filter(|line| !line.starts_with('#')).collect();
        assert!(lines.len() > 300, "{} lines", lines.len());
        for line in lines {
            let (french, english) = line.split_once('|').unwrap_or_else(|| panic!("no separator: {line}"));
            assert!(!french.trim().is_empty() && english.split(',').all(|word| !word.trim().is_empty()), "{line}");
        }
        let terms = vec!["musique".to_string(), "zorglub".to_string(), "fil".to_string()];
        assert_eq!(
            translated(&terms),
            vec![("musique".to_string(), "music".to_string()), ("fil".to_string(), "thread".to_string())]
        );
    }
}
