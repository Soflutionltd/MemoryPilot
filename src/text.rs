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
}
