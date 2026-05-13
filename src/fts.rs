pub fn sanitize_fts5_query(query: &str) -> Option<String> {
    let normalized = query.replace('(', " ").replace(')', " ");
    let terms: Vec<String> = normalized
        .split_whitespace()
        .filter_map(sanitize_term)
        .collect();

    if terms.is_empty() {
        None
    } else {
        Some(terms.join(" "))
    }
}

#[allow(dead_code)]
pub fn lexical_terms(query: &str) -> Vec<String> {
    query
        .split(|character: char| !character.is_alphanumeric())
        .map(str::trim)
        .filter(|term| term.len() >= 3)
        .map(|term| term.to_ascii_lowercase())
        .collect()
}

fn sanitize_term(term: &str) -> Option<String> {
    let cleaned = term.replace('"', "\"\"");
    if cleaned.is_empty() {
        return None;
    }
    Some(format!("\"{}\"*", cleaned))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitizes_special_characters() {
        let query = sanitize_fts5_query("CGEvent.tapCreate(options: .defaultTap)").unwrap();
        assert!(query.contains("\"CGEvent.tapCreate\"*"));
        assert!(query.contains("\"options:\"*"));
    }

    #[test]
    fn returns_none_for_empty_query() {
        assert!(sanitize_fts5_query("()").is_none());
    }
}
