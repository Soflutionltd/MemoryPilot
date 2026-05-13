use crate::db::Memory;

pub fn build_extractve_capsule(memories: &[Memory]) -> Option<String> {
    if memories.len() < 2 {
        return None;
    }

    let mut scored = memories
        .iter()
        .filter_map(|memory| {
            let sentence = first_sentence(&memory.content)?;
            let score = sentence_score(&sentence, memory);
            Some((score, sentence))
        })
        .collect::<Vec<_>>();

    if scored.is_empty() {
        return None;
    }

    scored.sort_by(|left, right| {
        right
            .0
            .cmp(&left.0)
            .then_with(|| left.1.len().cmp(&right.1.len()))
    });

    let bullets = scored
        .into_iter()
        .map(|(_, sentence)| sentence)
        .take(6)
        .map(|sentence| format!("- {}", sentence))
        .collect::<Vec<_>>();

    if bullets.is_empty() {
        None
    } else {
        Some(format!("## Session Capsule\n\n{}\n", bullets.join("\n")))
    }
}

fn first_sentence(content: &str) -> Option<String> {
    let cleaned = content
        .trim()
        .trim_start_matches("user:")
        .trim_start_matches("assistant:")
        .trim();
    if cleaned.len() < 24 {
        return None;
    }
    let end = cleaned
        .find(". ")
        .map(|index| index + 1)
        .unwrap_or_else(|| cleaned.len().min(180));
    Some(cleaned[..end].trim().to_string())
}

fn sentence_score(sentence: &str, memory: &Memory) -> i32 {
    let lower = sentence.to_ascii_lowercase();
    let mut score = memory.importance;
    if memory.kind == "preference" || lower.contains("prefer") || lower.contains("i usually") {
        score += 5;
    }
    if memory.kind == "decision" || lower.contains("decided") || lower.contains("because") {
        score += 4;
    }
    if memory.kind == "fact" || lower.contains("i bought") || lower.contains("i work") {
        score += 3;
    }
    if lower.contains("today") || lower.contains("yesterday") || lower.contains("last ") {
        score += 2;
    }
    score
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_extractve_capsule() {
        let memories = vec![
            memory("user: I usually prefer quiet restaurants. More text."),
            memory("assistant: Here are some unrelated suggestions."),
        ];
        let capsule = build_extractve_capsule(&memories).unwrap();
        assert!(capsule.contains("Session Capsule"));
        assert!(capsule.contains("quiet restaurants"));
    }

    fn memory(content: &str) -> Memory {
        Memory {
            id: "m".to_string(),
            content: content.to_string(),
            kind: "preference".to_string(),
            project: None,
            tags: Vec::new(),
            source: "test".to_string(),
            importance: 3,
            expires_at: None,
            created_at: String::new(),
            updated_at: String::new(),
            metadata: None,
            last_accessed_at: None,
            access_count: 0,
        }
    }
}
