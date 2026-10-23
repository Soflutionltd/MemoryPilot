#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextChunk {
    pub text: String,
    pub start_line: usize,
    pub end_line: usize,
}

pub fn split_text_chunks(content: &str, target_chars: usize) -> Vec<String> {
    chunk_text(content, target_chars, default_overlap(target_chars))
        .into_iter()
        .map(|chunk| chunk.text)
        .collect()
}

pub fn chunk_text(content: &str, target_chars: usize, overlap_chars: usize) -> Vec<TextChunk> {
    let target = target_chars.max(512);
    let overlap = overlap_chars.min(target / 3);
    let blocks = paragraph_blocks(content);

    if blocks.is_empty() {
        return Vec::new();
    }

    let mut chunks = Vec::new();
    let mut current = String::new();
    let mut start_line = blocks[0].start_line;
    let mut end_line = blocks[0].end_line;

    for block in blocks {
        let separator_len = if current.is_empty() { 0 } else { 2 };
        let would_exceed = current.len() + separator_len + block.text.len() > target;

        if would_exceed && !current.is_empty() {
            push_chunk(&mut chunks, &current, start_line, end_line);
            let carry = overlap_tail(&current, overlap);
            current = carry;
            start_line = block.start_line;
        }

        if !current.is_empty() {
            current.push_str("\n\n");
        }
        current.push_str(&block.text);
        end_line = block.end_line;
    }

    push_chunk(&mut chunks, &current, start_line, end_line);
    chunks
}

fn default_overlap(target_chars: usize) -> usize {
    (target_chars / 6).clamp(120, 400)
}

fn push_chunk(chunks: &mut Vec<TextChunk>, content: &str, start_line: usize, end_line: usize) {
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return;
    }
    chunks.push(TextChunk {
        text: trimmed.to_string(),
        start_line,
        end_line,
    });
}

#[derive(Debug)]
struct ParagraphBlock {
    text: String,
    start_line: usize,
    end_line: usize,
}

fn paragraph_blocks(content: &str) -> Vec<ParagraphBlock> {
    let mut blocks = Vec::new();
    let mut current = String::new();
    let mut start_line = 1usize;
    let mut last_line = 1usize;
    let mut in_code_block = false;

    for (index, line) in content.lines().enumerate() {
        let line_number = index + 1;
        let trimmed = line.trim();
        let starts_code_fence = trimmed.starts_with("```") || trimmed.starts_with("~~~");
        let starts_markdown_boundary = !in_code_block
            && (trimmed.starts_with('#')
                || trimmed.starts_with("- ")
                || trimmed.starts_with("* ")
                || trimmed.starts_with("> "));

        if trimmed.is_empty() {
            if !current.trim().is_empty() {
                blocks.push(ParagraphBlock {
                    text: current.trim().to_string(),
                    start_line,
                    end_line: last_line,
                });
                current.clear();
            }
            start_line = line_number + 1;
            continue;
        }

        if starts_markdown_boundary && !current.trim().is_empty() {
            blocks.push(ParagraphBlock {
                text: current.trim().to_string(),
                start_line,
                end_line: last_line,
            });
            current.clear();
            start_line = line_number;
        }

        if current.is_empty() {
            start_line = line_number;
        } else {
            current.push('\n');
        }
        current.push_str(trimmed);
        last_line = line_number;

        if starts_code_fence {
            if in_code_block {
                blocks.push(ParagraphBlock {
                    text: current.trim().to_string(),
                    start_line,
                    end_line: last_line,
                });
                current.clear();
                start_line = line_number + 1;
            }
            in_code_block = !in_code_block;
        }
    }

    if !current.trim().is_empty() {
        blocks.push(ParagraphBlock {
            text: current.trim().to_string(),
            start_line,
            end_line: last_line,
        });
    }

    if blocks.len() == 1 && blocks[0].text.len() > 4096 {
        split_large_block(&blocks[0])
    } else {
        blocks
    }
}

fn split_large_block(block: &ParagraphBlock) -> Vec<ParagraphBlock> {
    let mut blocks = Vec::new();
    let mut current = String::new();
    for sentence in sentence_like_segments(&block.text) {
        if current.len() + sentence.len() + 1 > 1200 && !current.is_empty() {
            blocks.push(ParagraphBlock {
                text: current.trim().to_string(),
                start_line: block.start_line,
                end_line: block.end_line,
            });
            current.clear();
        }
        if !current.is_empty() {
            current.push(' ');
        }
        current.push_str(sentence);
    }
    if !current.trim().is_empty() {
        blocks.push(ParagraphBlock {
            text: current.trim().to_string(),
            start_line: block.start_line,
            end_line: block.end_line,
        });
    }
    blocks
}

fn sentence_like_segments(text: &str) -> Vec<&str> {
    let mut segments = Vec::new();
    let mut start = 0usize;
    for (index, character) in text.char_indices() {
        if matches!(character, '.' | '!' | '?' | '\n') {
            let end = index + character.len_utf8();
            let candidate = text[start..end].trim();
            if !candidate.is_empty() {
                segments.push(candidate);
            }
            start = end;
        }
    }
    let tail = text[start..].trim();
    if !tail.is_empty() {
        segments.push(tail);
    }
    segments
}

fn overlap_tail(content: &str, max_chars: usize) -> String {
    if max_chars == 0 || content.len() <= max_chars {
        return String::new();
    }

    let mut start = content.len().saturating_sub(max_chars);
    while start < content.len() && !content.is_char_boundary(start) {
        start += 1;
    }

    let tail = content[start..].trim_start();
    match tail.find(char::is_whitespace) {
        Some(first_space) => tail[first_space..].trim_start().to_string(),
        None => tail.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_preserve_short_text() {
        let chunks = split_text_chunks("hello world", 2000);
        assert_eq!(chunks, vec!["hello world"]);
    }

    #[test]
    fn chunks_long_paragraphs() {
        let input = "One sentence. ".repeat(400);
        let chunks = split_text_chunks(&input, 1000);
        assert!(chunks.len() > 1);
        assert!(chunks.iter().all(|chunk| !chunk.trim().is_empty()));
    }

    #[test]
    fn chunks_respect_markdown_boundaries() {
        let input = "# Title\nIntro\n\n```rust\nfn main() {}\n```\n\n- item one\n- item two";
        let chunks = chunk_text(input, 80, 0);
        assert!(chunks.iter().any(|chunk| chunk.text.contains("fn main")));
        assert!(chunks.iter().any(|chunk| chunk.text.contains("- item one")));
    }
}
