//! UTF-8-safe string truncation.
//!
//! Slicing a `&str` at a byte offset panics when the offset falls inside
//! a multi-byte character — and French text hits that on every "é".
//! One such slice in the session-capsule path took the whole MCP server
//! down on `add_memory` (v4.5.0 hotfix). Every "first N bytes" preview
//! goes through here.

/// Longest prefix of `text` that is at most `max_bytes` long and ends on
/// a char boundary.
pub fn prefix(text: &str, max_bytes: usize) -> &str {
    if text.len() <= max_bytes {
        return text;
    }
    let mut end = max_bytes;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

/// `text` cut to `max_bytes` with an ellipsis when something was dropped.
pub fn preview(text: &str, max_bytes: usize) -> String {
    if text.len() <= max_bytes {
        text.to_string()
    } else {
        format!("{}...", prefix(text, max_bytes))
    }
}

/// Like [`preview`], but never ends on half a word or half a number
/// ("pro 89..." read as a full price): cut at the last space, or at a
/// clause end (". ", "; ", ", ") when one sits in the last 15% of the
/// window. A text without any space is cut hard.
pub fn preview_words(text: &str, max_bytes: usize) -> String {
    if text.len() <= max_bytes {
        return text.to_string();
    }
    let cut = prefix(text, max_bytes);
    let clause_floor = cut.len() * 85 / 100;
    let clause = [". ", "; ", ", "]
        .iter()
        .filter_map(|sep| cut.rfind(sep).map(|at| at + 1))
        .filter(|&end| end >= clause_floor)
        .max();
    let end = clause
        .or_else(|| cut.rfind(char::is_whitespace).filter(|&at| at > 0))
        .unwrap_or(cut.len());
    format!("{}...", cut[..end].trim_end_matches([',', ';', '.', ' ']))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn never_splits_a_character() {
        let french = "Procédure appliquée à la build : état WAITING";
        for max in 0..=french.len() {
            let cut = prefix(french, max);
            assert!(cut.len() <= max);
            assert!(french.starts_with(cut));
        }
        assert_eq!(prefix("é", 1), "");
        assert_eq!(preview("abc", 3), "abc");
        assert_eq!(preview("héllo", 2), "h...");
    }

    #[test]
    fn preview_words_ends_on_a_boundary() {
        // A clause end in the last 15% of the window wins.
        let pricing = "free 0€ 1000 emails, starter 15€ 20000, growth 39€ 60000, pro 89€ 150000";
        assert_eq!(preview_words(pricing, 64), "free 0€ 1000 emails, starter 15€ 20000, growth 39€ 60000...");
        // An earlier one does not: the cut falls on the last space.
        assert_eq!(
            preview_words("Début, puis une longue suite de mots", 30),
            "Début, puis une longue suite..."
        );
        assert_eq!(preview_words("un deux trois quatre cinq", 16), "un deux trois...");
        assert_eq!(preview_words("Une phrase. Puis une autre bien longue", 14), "Une phrase...");
        assert_eq!(preview_words("abcdefghij", 5), "abcde...");
        assert_eq!(preview_words("court", 10), "court");
    }
}
