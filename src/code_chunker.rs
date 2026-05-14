#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CodeLanguage {
    Rust,
    Python,
    TypeScript,
    Tsx,
    JavaScript,
    Svelte,
    Go,
    Java,
    Kotlin,
    Swift,
}

#[derive(Debug, Clone)]
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

    split_with_tree_sitter(content, target_chars.max(512), language)
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

    // Kotlin must be checked before Python because both have "class" and
    // "import", and before Java because both share "package" / "class".
    let kotlin_score = score_prefixes(
        &sample,
        &[
            "fun ",
            "object ",
            "data class ",
            "sealed class ",
            "@Composable",
            "suspend fun ",
            "companion object",
        ],
    );
    if kotlin_score >= 2
        || (sample.contains("fun ")
            && (sample.contains("val ") || sample.contains("var "))
            && (sample.contains("data class ")
                || sample.contains("suspend fun")
                || sample.contains("@Composable")
                || sample.contains("override fun")))
    {
        return Some(CodeLanguage::Kotlin);
    }

    // Swift before Python (shares "class") and before Java (shares "class").
    let swift_score = score_prefixes(
        &sample,
        &[
            "import Foundation",
            "import SwiftUI",
            "import UIKit",
            "import Combine",
            "func ",
            "@State",
            "@Binding",
            "@MainActor",
            "@objc",
        ],
    );
    if swift_score >= 2
        && (sample.contains("func ")
            || sample.contains("@State")
            || sample.contains("@Binding"))
        && (sample.contains("import SwiftUI")
            || sample.contains("import Foundation")
            || sample.contains("import UIKit")
            || sample.contains("import Combine")
            || sample.contains("@State")
            || sample.contains("@Binding")
            || sample.contains("@MainActor"))
    {
        return Some(CodeLanguage::Swift);
    }

    // Go before Python: "package" and "func " uniquely identify Go.
    let go_score = score_prefixes(
        &sample,
        &[
            "package ",
            "func ",
            "import (",
            "type ",
            "var ",
            "const ",
        ],
    );
    if (sample.contains("package main") || sample.contains("package "))
        && go_score >= 2
        && (sample.contains("func ") || sample.contains("import ("))
    {
        return Some(CodeLanguage::Go);
    }

    // Java before Python (shares "class" and "import").
    let java_score = score_prefixes(
        &sample,
        &[
            "package ",
            "import ",
            "public class ",
            "public interface ",
            "public enum ",
            "private ",
            "protected ",
            "@Override",
            "@Service",
            "@Component",
        ],
    );
    if java_score >= 2
        && (sample.contains("public ") || sample.contains("private ") || sample.contains("class "))
        && (sample.contains("public static")
            || sample.contains("@Override")
            || sample.contains("System.out")
            || sample.contains("extends ")
            || sample.contains("implements ")
            || sample.contains("public class")
            || sample.contains("public interface"))
    {
        return Some(CodeLanguage::Java);
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

fn tree_sitter_language(language: CodeLanguage) -> tree_sitter::Language {
    match language {
        CodeLanguage::Rust => tree_sitter_rust::LANGUAGE.into(),
        CodeLanguage::Python => tree_sitter_python::LANGUAGE.into(),
        CodeLanguage::TypeScript => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
        CodeLanguage::Tsx => tree_sitter_typescript::LANGUAGE_TSX.into(),
        CodeLanguage::JavaScript => tree_sitter_javascript::LANGUAGE.into(),
        CodeLanguage::Go => tree_sitter_go::LANGUAGE.into(),
        CodeLanguage::Java => tree_sitter_java::LANGUAGE.into(),
        CodeLanguage::Kotlin => tree_sitter_kotlin_ng::LANGUAGE.into(),
        CodeLanguage::Swift => tree_sitter_swift::LANGUAGE.into(),
        CodeLanguage::Svelte => unreachable!("svelte is handled before tree-sitter parsing"),
    }
}

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
        CodeLanguage::Go => matches!(
            kind,
            "function_declaration"
                | "method_declaration"
                | "type_declaration"
                | "var_declaration"
                | "const_declaration"
        ),
        CodeLanguage::Java => matches!(
            kind,
            "method_declaration"
                | "constructor_declaration"
                | "class_declaration"
                | "interface_declaration"
                | "enum_declaration"
                | "annotation_type_declaration"
        ),
        CodeLanguage::Kotlin => matches!(
            kind,
            "function_declaration"
                | "class_declaration"
                | "object_declaration"
                | "property_declaration"
                | "secondary_constructor"
                | "anonymous_initializer"
        ),
        CodeLanguage::Swift => matches!(
            kind,
            "function_declaration"
                | "class_declaration"
                | "protocol_declaration"
                | "init_declaration"
                | "deinit_declaration"
                | "subscript_declaration"
                | "property_declaration"
                | "extension_declaration"
        ),
        CodeLanguage::Svelte => false,
    }
}

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
    fn chunks_go_on_function_boundaries() {
        let source = r#"
package main

import "fmt"

type User struct {
    Name string
}

func (u *User) Greet() string {
    return fmt.Sprintf("hello %s", u.Name)
}

func NewUser(name string) *User {
    return &User{Name: name}
}
"#;
        let chunks = split_code_chunks(source, 200).expect("go chunks");
        assert!(chunks.iter().any(|chunk| chunk.contains("type User struct")));
        assert!(chunks.iter().any(|chunk| chunk.contains("func NewUser")));
    }

    #[test]
    fn chunks_java_on_class_boundaries() {
        let source = r#"
package com.example.bench;

import java.util.List;

public class UserService {
    public User findById(String id) {
        return new User(id, "anon");
    }

    @Override
    public String toString() {
        return "UserService";
    }
}

public interface UserRepository {
    User load(String id);
}
"#;
        let chunks = split_code_chunks(source, 220).expect("java chunks");
        assert!(chunks
            .iter()
            .any(|chunk| chunk.contains("public class UserService")));
        assert!(chunks
            .iter()
            .any(|chunk| chunk.contains("public interface UserRepository")));
    }

    #[test]
    fn chunks_kotlin_on_function_boundaries() {
        let source = r#"
package com.example.bench

import kotlinx.coroutines.flow.Flow

data class User(val id: String, val name: String)

class UserRepository {
    suspend fun loadUser(id: String): User {
        return User(id, "anon")
    }

    @Composable
    fun renderHeader(name: String) {
        println("hello $name")
    }
}
"#;
        let chunks = split_code_chunks(source, 220).expect("kotlin chunks");
        assert!(chunks
            .iter()
            .any(|chunk| chunk.contains("class UserRepository")));
        assert!(chunks
            .iter()
            .any(|chunk| chunk.contains("data class User")));
    }

    #[test]
    fn chunks_swift_on_function_boundaries() {
        let source = r#"
import Foundation
import SwiftUI

struct User: Identifiable {
    let id: String
    var name: String
}

class UserStore {
    @State private var users: [User] = []

    func loadUser(id: String) -> User {
        return User(id: id, name: "anon")
    }
}

extension UserStore {
    func count() -> Int {
        return users.count
    }
}
"#;
        let chunks = split_code_chunks(source, 220).expect("swift chunks");
        assert!(chunks.iter().any(|chunk| chunk.contains("struct User")));
        assert!(chunks.iter().any(|chunk| chunk.contains("class UserStore")));
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
