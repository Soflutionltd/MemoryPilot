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

    // Stemmed prefix variant. The FTS5 `content` column stores both the
    // raw text and the Snowball-stemmed projection of every memory, so a
    // stemmed-query prefix match recovers French/English inflection
    // variants (e.g. "messages" vs "message", "running" vs "run") that
    // unicode61 cannot bridge alone.
    let stemmed_query = crate::stemming::stem_query(query);
    if !stemmed_query.is_empty() {
        if let Some(stem_primary) = sanitize_fts5_query(&stemmed_query) {
            // Only add it if it differs from the raw prefix to avoid
            // double-counting BM25 evidence on non-inflected queries.
            if !variants
                .iter()
                .any(|(existing, _)| existing == &stem_primary)
            {
                variants.push((stem_primary, "fts_stem"));
            }
        }
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
