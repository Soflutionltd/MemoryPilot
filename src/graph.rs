/// MemoryPilot v3.0 — Knowledge Graph Engine.
/// Automatic entity extraction, relationship management, and graph traversal.
use std::collections::HashSet;

/// Extracted entity from memory content.
#[derive(Debug, Clone)]
pub struct Entity {
    pub kind: &'static str, // "project", "tech", "component", "file", "person", "agent", "origin", "topic"
    pub value: String,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct CorpusAnalysis {
    pub origin: &'static str,
    pub platform: Option<&'static str>,
    pub agents: Vec<String>,
    pub personas: Vec<String>,
    pub topics: Vec<String>,
    pub confidence: f64,
}

pub fn is_reliable_link_entity(entity: &Entity) -> bool {
    matches!(
        entity.kind,
        "file" | "component" | "person" | "agent" | "topic" | "platform" | "origin"
    )
}

pub fn relation_for_entity_kind(kind: &str) -> &'static str {
    match kind {
        "topic" => "shares_topic",
        "agent" => "same_agent",
        "origin" | "platform" => "same_origin",
        _ => "relates_to",
    }
}

const PERSON_VERBS: &[&str] = &[
    "said",
    "asked",
    "decided",
    "recommended",
    "pushed",
    "wrote",
    "built",
    "created",
    "designed",
    "fixed",
    "reported",
    "suggested",
    "mentioned",
    "reviewed",
    "approved",
    "deployed",
    "assigned",
    "committed",
    "merged",
    "a dit",
    "a demandé",
    "a décidé",
    "a recommandé",
    "a écrit",
    "a construit",
];

const PERSON_ROLES: &[&str] = &[
    "lead",
    "manager",
    "developer",
    "dev",
    "designer",
    "architect",
    "engineer",
    "cto",
    "ceo",
    "founder",
    "cofounder",
    "intern",
    "qa",
    "frontend",
    "backend",
    "fullstack",
    "devops",
    "pm",
    "product",
];

const PERSON_STOPWORDS: &[&str] = &[
    "the",
    "this",
    "that",
    "then",
    "they",
    "there",
    "their",
    "these",
    "not",
    "you",
    "your",
    "has",
    "have",
    "had",
    "was",
    "were",
    "are",
    "will",
    "can",
    "could",
    "would",
    "should",
    "may",
    "might",
    "all",
    "any",
    "some",
    "for",
    "from",
    "with",
    "into",
    "over",
    "true",
    "false",
    "null",
    "none",
    "self",
    "todo",
    "note",
    "error",
    "warning",
    "info",
    "debug",
    "fatal",
    "panic",
    "ok",
    "err",
    "result",
    "option",
    "vec",
    "string",
    "type",
    "struct",
    "enum",
    "impl",
    "pub",
    "mod",
    "use",
    "let",
    "mut",
    "return",
    "match",
    "loop",
    "while",
    "break",
    "continue",
    "async",
    "await",
    "trait",
    "where",
    "super",
    "crate",
    "class",
    "function",
    "import",
    "export",
    "default",
    "extends",
    "interface",
    "const",
    "var",
    "new",
    "delete",
    "typeof",
    "instanceof",
    "first",
    "second",
    "third",
    "next",
    "last",
    "final",
    "set",
    "get",
    "add",
    "put",
    "run",
    "end",
    "key",
    "map",
    "les",
    "des",
    "une",
    "est",
    "pas",
    "par",
    "sur",
    "dans",
    "qui",
    "que",
    "pour",
    "avec",
    "mais",
    "plus",
    "fait",
    "très",
];

const AGENT_NAMES: &[&str] = &[
    "claude",
    "cursor",
    "chatgpt",
    "gpt",
    "codex",
    "gemini",
    "copilot",
    "assistant",
    "anthropic",
    "openai",
    "sonnet",
    "opus",
    "haiku",
    "windsurf",
    "cline",
    "roo",
    "aider",
];

const TRANSCRIPT_MARKERS: &[&str] = &[
    "user:",
    "assistant:",
    "human:",
    "system:",
    "\"role\":\"user\"",
    "\"role\":\"assistant\"",
    "<user_query>",
    "<assistant",
    "claude code",
    "cursor",
    "chatgpt",
    "conversation",
];

/// Known tech patterns for auto-detection.
const TECH_PATTERNS: &[&str] = &[
    "svelte",
    "sveltekit",
    "svelte 5",
    "react",
    "vue",
    "next",
    "nuxt",
    "astro",
    "supabase",
    "firebase",
    "postgresql",
    "sqlite",
    "redis",
    "mongodb",
    "tailwind",
    "css",
    "sass",
    "bootstrap",
    "rust",
    "typescript",
    "javascript",
    "python",
    "swift",
    "go",
    "java",
    "cloudflare",
    "vercel",
    "netlify",
    "aws",
    "hetzner",
    "docker",
    "stripe",
    "auth",
    "jwt",
    "oauth",
    "better-auth",
    "onnx",
    "bert",
    "openai",
    "claude",
    "llm",
    "mcp",
    "tauri",
    "electron",
    "flutter",
    "xcode",
    "git",
    "github",
    "npm",
    "cargo",
    "pnpm",
];

const TOPIC_PATTERNS: &[&str] = &[
    "authentication",
    "auth",
    "oauth",
    "jwt",
    "database",
    "sqlite",
    "postgresql",
    "supabase",
    "stripe",
    "payments",
    "billing",
    "cloudflare",
    "deployment",
    "benchmark",
    "longmemeval",
    "locomo",
    "embedding",
    "reranker",
    "memory",
    "knowledge graph",
    "graphrag",
    "compression",
    "garbage collection",
    "sveltekit",
    "svelte",
    "tailwind",
    "rust",
    "mcp",
    "http",
    "security",
    "license",
    "installer",
    "readme",
    "performance",
    "temporal",
];

/// Known component patterns (file-like).
const COMPONENT_HINTS: &[&str] = &[
    "component",
    "page",
    "layout",
    "modal",
    "button",
    "form",
    "input",
    "header",
    "footer",
    "sidebar",
    "nav",
    "card",
    "table",
    "dialog",
    "dashboard",
    "settings",
    "profile",
    "auth",
    "login",
    "signup",
];

fn push_entity(
    entities: &mut Vec<Entity>,
    seen: &mut HashSet<String>,
    kind: &'static str,
    value: String,
) {
    let value = value.trim();
    if value.is_empty() {
        return;
    }
    if seen.insert(format!("{}:{}", kind, value.to_ascii_lowercase())) {
        entities.push(Entity {
            kind,
            value: value.to_string(),
        });
    }
}

fn normalize_topic(topic: &str) -> String {
    topic
        .trim()
        .to_ascii_lowercase()
        .replace(' ', "-")
        .replace('_', "-")
}

pub fn analyze_corpus(content: &str, source: Option<&str>) -> CorpusAnalysis {
    let lower = content.to_ascii_lowercase();
    let source_lower = source.unwrap_or_default().to_ascii_lowercase();
    let mut agents: Vec<String> = Vec::new();
    let mut personas: Vec<String> = Vec::new();
    let mut topics: Vec<String> = Vec::new();

    let mut score: f64 = 0.0;
    for marker in TRANSCRIPT_MARKERS {
        if lower.contains(marker) {
            score += 0.12;
        }
    }

    let platform = if lower.contains("claude code") || source_lower.contains("claude") {
        Some("claude")
    } else if lower.contains("cursor") || source_lower.contains("cursor") {
        Some("cursor")
    } else if lower.contains("chatgpt") || source_lower.contains("chatgpt") {
        Some("chatgpt")
    } else if lower.contains("codex") || source_lower.contains("codex") {
        Some("codex")
    } else if lower.contains("gemini") || source_lower.contains("gemini") {
        Some("gemini")
    } else {
        None
    };
    if platform.is_some() {
        score += 0.2;
    }

    for agent in AGENT_NAMES {
        if lower.contains(agent) || source_lower.contains(agent) {
            let agent_name = agent.to_string();
            if !agents.contains(&agent_name) {
                agents.push(agent_name);
            }
        }
    }

    for line in content.lines().take(300) {
        let trimmed = line.trim();
        if let Some((speaker, _)) = trimmed.split_once(':') {
            let name = speaker
                .trim()
                .trim_matches(|c: char| !c.is_alphanumeric() && c != '-' && c != '_');
            let lower_name = name.to_ascii_lowercase();
            if name.len() >= 2
                && name.len() <= 24
                && AGENT_NAMES.iter().any(|agent| lower_name.contains(agent))
            {
                if !personas.contains(&name.to_string()) {
                    personas.push(name.to_string());
                }
            }
        }
    }

    for topic in TOPIC_PATTERNS {
        if lower.contains(topic) {
            let normalized = normalize_topic(topic);
            if !topics.contains(&normalized) {
                topics.push(normalized);
            }
        }
    }

    let origin = if score >= 0.25
        || lower.matches("user:").count() + lower.matches("assistant:").count() >= 2
    {
        "ai_transcript"
    } else if lower.contains("cargo.toml")
        || lower.contains("package.json")
        || lower.contains("src/")
        || lower.contains("fn ")
    {
        "codebase"
    } else if lower.contains("# ") || lower.contains("## ") || lower.contains("- ") {
        "notes"
    } else {
        "memory"
    };

    CorpusAnalysis {
        origin,
        platform,
        agents,
        personas,
        topics,
        confidence: (score
            + if origin != "memory" {
                0.25_f64
            } else {
                0.0_f64
            })
        .min(1.0_f64),
    }
}

/// Extract entities from memory content automatically.
/// Detects: projects, technologies, components, file paths, people.
pub fn extract_entities(content: &str, project: Option<&str>) -> Vec<Entity> {
    let lower = content.to_lowercase();
    let mut entities: Vec<Entity> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    let analysis = analyze_corpus(content, None);

    // 1. Project (from parameter or content)
    if let Some(p) = project {
        push_entity(&mut entities, &mut seen, "project", p.to_string());
    }

    push_entity(
        &mut entities,
        &mut seen,
        "origin",
        analysis.origin.to_string(),
    );
    if let Some(platform) = analysis.platform {
        push_entity(&mut entities, &mut seen, "platform", platform.to_string());
    }
    for agent in &analysis.agents {
        push_entity(&mut entities, &mut seen, "agent", agent.clone());
    }
    for persona in &analysis.personas {
        push_entity(&mut entities, &mut seen, "agent", persona.clone());
    }
    for topic in &analysis.topics {
        push_entity(&mut entities, &mut seen, "topic", topic.clone());
    }

    // 2. Technologies
    for tech in TECH_PATTERNS {
        if lower.contains(tech) {
            push_entity(&mut entities, &mut seen, "tech", tech.to_string());
            push_entity(&mut entities, &mut seen, "topic", normalize_topic(tech));
        }
    }

    // 3. File paths (detect patterns like src/foo/bar.ts, lib/components/X.svelte)
    for word in content.split_whitespace() {
        let w = word.trim_matches(|c: char| {
            !c.is_alphanumeric() && c != '/' && c != '.' && c != '_' && c != '-'
        });
        if w.contains('/') && w.contains('.') && w.len() > 4 {
            push_entity(&mut entities, &mut seen, "file", w.to_string());
        }
        // Also detect .svelte, .ts, .rs files without path
        if (w.ends_with(".svelte")
            || w.ends_with(".ts")
            || w.ends_with(".tsx")
            || w.ends_with(".rs")
            || w.ends_with(".py")
            || w.ends_with(".js"))
            && w.len() > 4
            && !w.starts_with('.')
        {
            push_entity(&mut entities, &mut seen, "file", w.to_string());
        }
    }

    // 4. Components (UI component names)
    for hint in COMPONENT_HINTS {
        if lower.contains(hint) {
            for word in content.split_whitespace() {
                let w = word.trim_matches(|c: char| !c.is_alphanumeric() && c != '-' && c != '_');
                if w.len() > 2
                    && (w.chars().next().map(|c| c.is_uppercase()).unwrap_or(false)
                        || w.contains('-')
                        || w.contains('_'))
                    && lower_contains_near(&lower, hint, &w.to_lowercase(), 50)
                {
                    push_entity(&mut entities, &mut seen, "component", w.to_string());
                }
            }
        }
    }

    // 5. Person detection
    let words: Vec<&str> = content.split_whitespace().collect();
    for (i, word) in words.iter().enumerate() {
        let w = word.trim_matches(|c: char| !c.is_alphanumeric());
        if w.len() < 2 || w.len() > 20 {
            continue;
        }
        let first_char = match w.chars().next() {
            Some(c) => c,
            None => continue,
        };
        if !first_char.is_uppercase() {
            continue;
        }
        let w_lower = w.to_lowercase();
        if PERSON_STOPWORDS.iter().any(|sw| *sw == w_lower) {
            continue;
        }
        if TECH_PATTERNS.iter().any(|tp| *tp == w_lower) {
            continue;
        }
        if COMPONENT_HINTS.iter().any(|ch| *ch == w_lower) {
            continue;
        }
        if AGENT_NAMES.iter().any(|agent| w_lower.contains(agent)) {
            continue;
        }

        // Check for "@Name" pattern
        let is_at_mention = word.starts_with('@');

        // Check for "hey Name" / "thanks Name"
        let is_direct_address = if i > 0 {
            matches!(
                words[i - 1].to_lowercase().as_str(),
                "hey" | "hi" | "thanks" | "merci" | "salut" | "cc" | "bonjour"
            )
        } else {
            false
        };

        // Check for "Name verb" pattern (Name said/asked/decided...)
        let followed_by_person_verb = if i + 1 < words.len() {
            let next_lower = words[i + 1].to_lowercase();
            PERSON_VERBS.iter().any(|v| next_lower.starts_with(v))
        } else {
            false
        };

        // Check for role keyword nearby (within 5 words)
        let has_role_nearby = words
            .iter()
            .skip(i.saturating_sub(3))
            .take(7)
            .any(|nearby| {
                PERSON_ROLES
                    .iter()
                    .any(|r| nearby.to_lowercase().contains(r))
            });

        if is_at_mention || is_direct_address || followed_by_person_verb || has_role_nearby {
            let name = if is_at_mention {
                w.trim_start_matches('@')
            } else {
                w
            };
            if name.len() >= 2 && seen.insert(format!("person:{}", name.to_lowercase())) {
                entities.push(Entity {
                    kind: "person",
                    value: name.to_string(),
                });
            }
        }
    }

    entities
}

pub fn traverse_graph(
    conn: &rusqlite::Connection,
    root_ids: &[String],
    max_depth: u32,
) -> Result<HashSet<String>, String> {
    if root_ids.is_empty() || max_depth == 0 {
        return Ok(HashSet::new());
    }

    let mut stmt_fwd = conn
        .prepare("SELECT target_id FROM memory_links WHERE source_id = ?1")
        .map_err(|e| format!("traverse prepare fwd: {}", e))?;
    let mut stmt_rev = conn
        .prepare("SELECT source_id FROM memory_links WHERE target_id = ?1")
        .map_err(|e| format!("traverse prepare rev: {}", e))?;

    let mut current_level: HashSet<String> = root_ids.iter().cloned().collect();
    let mut all_visited: HashSet<String> = current_level.clone();

    for _ in 0..max_depth {
        let mut next_level = HashSet::new();

        for id in &current_level {
            if let Ok(rows) =
                stmt_fwd.query_map(rusqlite::params![id], |row| row.get::<_, String>(0))
            {
                for r in rows.flatten() {
                    if all_visited.insert(r.clone()) {
                        next_level.insert(r);
                    }
                }
            }

            if let Ok(rows) =
                stmt_rev.query_map(rusqlite::params![id], |row| row.get::<_, String>(0))
            {
                for r in rows.flatten() {
                    if all_visited.insert(r.clone()) {
                        next_level.insert(r);
                    }
                }
            }
        }

        if next_level.is_empty() {
            break;
        }
        current_level = next_level;
    }

    Ok(all_visited)
}

/// Infer relationship type between two memories based on their kinds.
pub fn infer_relation(source_kind: &str, target_kind: &str) -> &'static str {
    match (source_kind, target_kind) {
        ("bug", "decision") | ("bug", "architecture") => "resolved_by",
        ("decision", "bug") => "resolves",
        ("bug", "snippet") => "fixed_by",
        ("snippet", "bug") => "fixes",
        ("decision", "architecture") | ("decision", "pattern") => "implements",
        ("architecture", "decision") => "decided_by",
        ("todo", _) => "depends_on",
        (_, "todo") => "blocks",
        _ => "relates_to",
    }
}

/// Check if two substrings appear within `distance` chars of each other.
fn lower_contains_near(text: &str, a: &str, b: &str, distance: usize) -> bool {
    if let Some(pos_a) = text.find(a) {
        if let Some(pos_b) = text.find(b) {
            let diff = if pos_a > pos_b {
                pos_a - pos_b
            } else {
                pos_b - pos_a
            };
            return diff <= distance;
        }
    }
    false
}
