#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CodeLanguage {
    Rust,
    Python,
    TypeScript,
    Tsx,
    JavaScript,
    Svelte,
}

#[derive(Debug, Clone)]
#[cfg(feature = "code-aware-chunking")]
struct CodeUnit {
    start_byte: usize,
    end_byte: usize,
    start_line: usize,
    end_line: usize,
}

pub fn split_code_chunks(content: &str, target_chars: usize) -> Option<Vec<String>> {
    let language = detect_language(content)?;
    if language == CodeLanguage::Svelte {
        return split_svelte_chunks(content, target_chars);
    }

    #[cfg(feature = "code-aware-chunking")]
    {
        split_with_tree_sitter(content, target_chars.max(512), language)
    }

    #[cfg(not(feature = "code-aware-chunking"))]
    {
        let _ = (content, target_chars);
        None
    }
}

fn detect_language(content: &str) -> Option<CodeLanguage> {
    let sample = content.lines().take(120).collect::<Vec<_>>().join("\n");
    let lower = sample.to_ascii_lowercase();

    if lower.contains("<script") && lower.contains("</script>") {
        return Some(CodeLanguage::Svelte);
    }

    let rust_score = score_prefixes(
        &sample,
        &[
            "fn ", "pub fn ", "impl ", "struct ", "enum ", "trait ", "use ", "mod ",
        ],
    );
    if rust_score >= 2 || (sample.contains("->") && sample.contains("::")) {
        return Some(CodeLanguage::Rust);
    }

    let python_score = score_prefixes(
        &sample,
        &["def ", "class ", "async def ", "from ", "import "],
    );
    if python_score >= 2 && sample.contains(':') {
        return Some(CodeLanguage::Python);
    }

    if lower.contains("interface ")
        || lower.contains("type ")
        || lower.contains(": string")
        || lower.contains(": number")
        || lower.contains(": boolean")
    {
        return Some(CodeLanguage::TypeScript);
    }

    if lower.contains("tsx") || lower.contains("react") || sample.contains("</") {
        return Some(CodeLanguage::Tsx);
    }

    let js_score = score_prefixes(
        &sample,
        &[
            "function ",
            "export ",
            "import ",
            "const ",
            "let ",
            "class ",
            "async function ",
        ],
    );
    if js_score >= 2 {
        return Some(CodeLanguage::JavaScript);
    }

    None
}

fn score_prefixes(sample: &str, prefixes: &[&str]) -> usize {
    sample
        .lines()
        .filter(|line| {
            let trimmed = line.trim_start();
            prefixes.iter().any(|prefix| trimmed.starts_with(prefix))
        })
        .count()
}

#[cfg(feature = "code-aware-chunking")]
fn split_with_tree_sitter(
    content: &str,
    target_chars: usize,
    language: CodeLanguage,
) -> Option<Vec<String>> {
    let mut parser = tree_sitter::Parser::new();
    parser.set_language(&tree_sitter_language(language)).ok()?;
    let tree = parser.parse(content, None)?;
    if tree.root_node().has_error() {
        return None;
    }

    let mut units = Vec::new();
    collect_code_units(tree.root_node(), content.len(), language, &mut units);
    if units.is_empty() {
        return None;
    }

    units.sort_by_key(|unit| unit.start_byte);
    let line_starts = line_start_offsets(content);
    let mut expanded = Vec::new();
    for unit in merge_small_units(units, target_chars) {
        let start = line_starts
            .get(unit.start_line.saturating_sub(1))
            .copied()
            .unwrap_or(unit.start_byte);
        let end = line_starts
            .get(unit.end_line)
            .copied()
            .unwrap_or(content.len())
            .min(content.len());
        if start < end {
            expanded.push(content[start..end].trim().to_string());
        }
    }

    let chunks = expanded
        .into_iter()
        .filter(|chunk| !chunk.is_empty())
        .collect::<Vec<_>>();
    if chunks.is_empty() {
        None
    } else {
        Some(chunks)
    }
}

#[cfg(feature = "code-aware-chunking")]
fn tree_sitter_language(language: CodeLanguage) -> tree_sitter::Language {
    match language {
        CodeLanguage::Rust => tree_sitter_rust::LANGUAGE.into(),
        CodeLanguage::Python => tree_sitter_python::LANGUAGE.into(),
        CodeLanguage::TypeScript => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
        CodeLanguage::Tsx => tree_sitter_typescript::LANGUAGE_TSX.into(),
        CodeLanguage::JavaScript => tree_sitter_javascript::LANGUAGE.into(),
        CodeLanguage::Svelte => unreachable!("svelte is handled before tree-sitter parsing"),
    }
}

#[cfg(feature = "code-aware-chunking")]
fn collect_code_units(
    node: tree_sitter::Node,
    content_len: usize,
    language: CodeLanguage,
    units: &mut Vec<CodeUnit>,
) {
    if is_boundary_node(node.kind(), language) {
        let start = node.start_position();
        let end = node.end_position();
        units.push(CodeUnit {
            start_byte: node.start_byte(),
            end_byte: node.end_byte().min(content_len),
            start_line: start.row + 1,
            end_line: end.row + 1,
        });
        return;
    }

    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        collect_code_units(child, content_len, language, units);
    }
}

#[cfg(feature = "code-aware-chunking")]
fn is_boundary_node(kind: &str, language: CodeLanguage) -> bool {
    match language {
        CodeLanguage::Rust => matches!(
            kind,
            "function_item"
                | "impl_item"
                | "struct_item"
                | "enum_item"
                | "trait_item"
                | "mod_item"
                | "macro_definition"
        ),
        CodeLanguage::Python => matches!(
            kind,
            "function_definition" | "class_definition" | "decorated_definition"
        ),
        CodeLanguage::TypeScript | CodeLanguage::Tsx => matches!(
            kind,
            "function_declaration"
                | "method_definition"
                | "class_declaration"
                | "interface_declaration"
                | "type_alias_declaration"
                | "lexical_declaration"
                | "export_statement"
        ),
        CodeLanguage::JavaScript => matches!(
            kind,
            "function_declaration"
                | "method_definition"
                | "class_declaration"
                | "lexical_declaration"
                | "export_statement"
        ),
        CodeLanguage::Svelte => false,
    }
}

#[cfg(feature = "code-aware-chunking")]
fn merge_small_units(units: Vec<CodeUnit>, target_chars: usize) -> Vec<CodeUnit> {
    let mut merged = Vec::new();
    let mut current: Option<CodeUnit> = None;

    for unit in units {
        match current.as_mut() {
            Some(active)
                if unit.end_byte.saturating_sub(active.start_byte) <= target_chars
                    && unit.start_byte >= active.end_byte =>
            {
                active.end_byte = unit.end_byte;
                active.end_line = unit.end_line;
            }
            Some(_) => {
                if let Some(done) = current.replace(unit) {
                    merged.push(done);
                }
            }
            None => current = Some(unit),
        }
    }

    if let Some(done) = current {
        merged.push(done);
    }

    merged
}

#[cfg(feature = "code-aware-chunking")]
fn line_start_offsets(content: &str) -> Vec<usize> {
    let mut starts = vec![0usize];
    for (index, character) in content.char_indices() {
        if character == '\n' {
            starts.push(index + 1);
        }
    }
    starts
}

fn split_svelte_chunks(content: &str, target_chars: usize) -> Option<Vec<String>> {
    let mut chunks = Vec::new();
    let mut cursor = 0usize;

    while let Some(relative_start) = content[cursor..].find("<script") {
        let script_start = cursor + relative_start;
        if script_start > cursor {
            push_markup_chunks(&content[cursor..script_start], target_chars, &mut chunks);
        }

        let Some(relative_tag_end) = content[script_start..].find('>') else {
            break;
        };
        let body_start = script_start + relative_tag_end + 1;
        let Some(relative_end) = content[body_start..].find("</script>") else {
            break;
        };
        let body_end = body_start + relative_end;
        let open_tag = &content[script_start..body_start];
        let body = &content[body_start..body_end];
        let close_tag_end = body_end + "</script>".len();

        if let Some(script_chunks) = split_code_chunks(body, target_chars) {
            for chunk in script_chunks {
                chunks.push(format!("{}{}\n</script>", open_tag.trim_end(), chunk));
            }
        } else {
            chunks.push(content[script_start..close_tag_end].trim().to_string());
        }
        cursor = close_tag_end;
    }

    if cursor < content.len() {
        push_markup_chunks(&content[cursor..], target_chars, &mut chunks);
    }

    let chunks = chunks
        .into_iter()
        .filter(|chunk| !chunk.trim().is_empty())
        .collect::<Vec<_>>();
    if chunks.is_empty() {
        None
    } else {
        Some(chunks)
    }
}

fn push_markup_chunks(markup: &str, target_chars: usize, chunks: &mut Vec<String>) {
    chunks.extend(crate::chunker::split_text_chunks(markup, target_chars));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_rust_on_function_boundaries() {
        let source = r#"
use std::fmt;

pub struct User {
    name: String,
}

impl User {
    pub fn new(name: String) -> Self {
        Self { name }
    }

    pub fn label(&self) -> String {
        format!("user: {}", self.name)
    }
}

pub fn helper() -> usize {
    42
}
"#;
        let chunks = split_code_chunks(source, 260).expect("rust chunks");
        assert!(chunks.iter().any(|chunk| chunk.contains("impl User")));
        assert!(chunks.iter().any(|chunk| chunk.contains("pub fn helper")));
        assert!(chunks
            .iter()
            .all(|chunk| !chunk.contains("format!(\"user") || chunk.contains("pub fn label")));
    }

    #[test]
    fn chunks_python_classes_and_functions() {
        let source = r#"
import os

class Worker:
    def run(self):
        return os.getcwd()

def build_worker():
    return Worker()
"#;
        let chunks = split_code_chunks(source, 160).expect("python chunks");
        assert!(chunks.iter().any(|chunk| chunk.contains("class Worker")));
        assert!(chunks
            .iter()
            .any(|chunk| chunk.contains("def build_worker")));
    }

    #[test]
    fn chunks_typescript_exports() {
        let source = r#"
import type { User } from './types';

export interface Session {
    id: string;
}

export function createSession(user: User): Session {
    return { id: user.id };
}
"#;
        let chunks = split_code_chunks(source, 160).expect("ts chunks");
        assert!(chunks
            .iter()
            .any(|chunk| chunk.contains("export interface Session")));
        assert!(chunks.iter().any(|chunk| chunk.contains("createSession")));
    }

    #[test]
    fn chunks_svelte_script_without_losing_markup() {
        let source = r#"
<script lang="ts">
    export let name: string;
    function greet() {
        return `hello ${name}`;
    }
</script>

<main>
    <h1>{greet()}</h1>
</main>
"#;
        let chunks = split_code_chunks(source, 180).expect("svelte chunks");
        assert!(chunks
            .iter()
            .any(|chunk| chunk.contains("<script lang=\"ts\">")));
        assert!(chunks.iter().any(|chunk| chunk.contains("<main>")));
    }
}
