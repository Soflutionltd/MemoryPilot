pub fn split_memory_text(content: &str, target_chars: usize) -> Vec<String> {
    if let Some(chunks) = crate::code_chunker::split_code_chunks(content, target_chars) {
        return chunks;
    }
    if looks_code_like(content) {
        return split_code_like_text(content, target_chars);
    }
    crate::chunker::split_text_chunks(content, target_chars)
}

pub fn split_code_like_text(content: &str, target_chars: usize) -> Vec<String> {
    split_by_structural_boundaries(content, target_chars.max(512))
}

fn split_by_structural_boundaries(content: &str, target_chars: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current = String::new();

    for line in content.lines() {
        let starts_boundary = line.starts_with("fn ")
            || line.starts_with("pub fn ")
            || line.starts_with("class ")
            || line.starts_with("struct ")
            || line.starts_with("impl ")
            || line.starts_with("def ")
            || line.starts_with("export ");

        if starts_boundary && !current.trim().is_empty() && current.len() > target_chars / 2 {
            chunks.push(current.trim().to_string());
            current.clear();
        } else if current.len() + line.len() + 1 > target_chars && !current.trim().is_empty() {
            chunks.push(current.trim().to_string());
            current.clear();
        }

        if !current.is_empty() {
            current.push('\n');
        }
        current.push_str(line);
    }

    if !current.trim().is_empty() {
        chunks.push(current.trim().to_string());
    }

    if chunks.is_empty() {
        crate::chunker::split_text_chunks(content, target_chars)
    } else {
        chunks
    }
}

fn looks_code_like(content: &str) -> bool {
    let mut signals = 0usize;
    for line in content.lines().take(80) {
        let trimmed = line.trim_start();
        if trimmed.starts_with("fn ")
            || trimmed.starts_with("pub fn ")
            || trimmed.starts_with("class ")
            || trimmed.starts_with("struct ")
            || trimmed.starts_with("impl ")
            || trimmed.starts_with("def ")
            || trimmed.starts_with("export ")
            || trimmed.starts_with("import ")
            || trimmed.starts_with("use ")
        {
            signals += 1;
        }
    }
    signals >= 2
}
