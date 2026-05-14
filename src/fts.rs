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

pub fn fts5_query_variants(query: &str) -> Vec<(String, &'static str)> {
    let mut variants = Vec::new();
    if let Some(primary) = sanitize_fts5_query(query) {
        variants.push((primary, "fts_prefix"));
    }

    let terms = lexical_terms(query);
    if terms.len() >= 2 {
        let phrase = terms
            .iter()
            .take(6)
            .map(|term| term.replace('"', "\"\""))
            .collect::<Vec<_>>()
            .join(" ");
        if !phrase.is_empty() {
            variants.push((format!("\"{}\"", phrase), "fts_phrase"));
        }
    }

    if (2..=8).contains(&terms.len()) {
        let near_terms = terms
            .iter()
            .take(5)
            .map(|term| format!("\"{}\"", term.replace('"', "\"\"")))
            .collect::<Vec<_>>()
            .join(" ");
        variants.push((format!("NEAR({}, 8)", near_terms), "fts_near"));
    }

    variants
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

    #[test]
    fn builds_phrase_and_near_variants() {
        let variants = fts5_query_variants("SettingsPanel render bug");
        assert!(variants.iter().any(|(_, source)| *source == "fts_prefix"));
        assert!(variants.iter().any(|(_, source)| *source == "fts_phrase"));
        assert!(variants.iter().any(|(_, source)| *source == "fts_near"));
    }
}
