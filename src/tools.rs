/// MCP Tool definitions and handlers for MemoryPilot v2.1.
use serde_json::{json, Value};
use crate::db::{Database, BulkItem, MemoryScope, RecallMode};
use crate::protocol::{tool_result, tool_error};

const VALID_KINDS: &[&str] = &[
    "fact", "preference", "decision", "pattern", "snippet",
    "bug", "credential", "todo", "note", "transcript",
    "milestone", "architecture", "problem",
];

pub fn tool_definitions() -> Value {
    json!({ "tools": [
        {
            "name": "recall",
            "description": "⚡ START HERE — Call this at the beginning of EVERY new conversation. Loads all relevant context in one shot: project memories, global preferences, critical facts, patterns, decisions, and GLOBAL_PROMPT. Optionally pass hints about the current task for targeted search.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "project": { "type": ["string","null"], "description": "Project name (or null for auto-detect)" },
                    "working_dir": { "type": ["string","null"], "description": "Current working directory for project auto-detection" },
                    "hints": { "type": ["string","null"], "description": "Keywords about current task for targeted memory search" },
                    "mode": { "type": ["string","null"], "enum": ["safe", "default", "full"], "description": "Recall mode. `safe` is the default and excludes credentials unless `full` is explicitly requested." },
                    "explain": { "type": ["boolean","null"], "description": "When true, include why each memory was selected: source, search score, recency, access boost, graph boost, and project match." },
                    "compact": { "type": ["boolean","null"], "description": "When true, output uses compressed shorthand (~3x fewer tokens). Default false." },
                    "session_id": { "type": ["string","null"], "description": "Optional session scope for prioritizing same-session memories." },
                    "thread_id": { "type": ["string","null"], "description": "Optional thread scope for prioritizing the same conversation." },
                    "window_id": { "type": ["string","null"], "description": "Optional window scope for prioritizing memories from the same Cursor window." }
                }
            }
        },
        {
            "name": "add_memory",
            "description": "Store a new memory with dedup. If near-duplicate exists, merges instead of creating. Kinds: fact, preference, decision, pattern, snippet, bug, credential, todo, note.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "content": { "type": "string", "description": "The memory content" },
                    "kind": { "type": "string", "enum": VALID_KINDS, "default": "fact" },
                    "project": { "type": ["string","null"], "description": "Project name or null for global" },
                    "tags": { "type": "array", "items": { "type": "string" }, "default": [] },
                    "source": { "type": "string", "default": "cursor" },
                    "importance": { "type": "integer", "minimum": 1, "maximum": 5, "default": 3, "description": "1=trivial, 3=normal, 5=critical" },
                    "expires_at": { "type": ["string","null"], "description": "ISO date after which memory auto-deletes (e.g. 2025-06-01T00:00:00Z)" },
                    "metadata": { "type": ["object","null"] },
                    "session_id": { "type": ["string","null"] },
                    "thread_id": { "type": ["string","null"] },
                    "window_id": { "type": ["string","null"] }
                },
                "required": ["content"]
            }
        },        {
            "name": "add_memories",
            "description": "Bulk add multiple memories in one call. Each item supports dedup. Saves context window by batching 5-20 memories.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "memories": { "type": "array", "items": {
                        "type": "object",
                        "properties": {
                            "content": { "type": "string" },
                            "kind": { "type": "string", "default": "fact" },
                            "project": { "type": ["string","null"] },
                            "tags": { "type": ["array","null"], "items": { "type": "string" } },
                            "source": { "type": "string", "default": "cursor" },
                            "importance": { "type": ["integer","null"] },
                            "expires_at": { "type": ["string","null"] },
                            "metadata": { "type": ["object","null"] },
                            "session_id": { "type": ["string","null"] },
                            "thread_id": { "type": ["string","null"] },
                            "window_id": { "type": ["string","null"] }
                        },
                        "required": ["content"]
                    }}
                },
                "required": ["memories"]
            }
        },
        {
            "name": "add_transcript",
            "description": "Store a long conversation transcript by automatically chunking it, then distill a few high-value memories (decisions, preferences, todos, bugs, facts) without polluting the context window.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "content": { "type": "string", "description": "The full transcript text to chunk and store" },
                    "project": { "type": ["string","null"], "description": "Project name or null for global" },
                    "tags": { "type": "array", "items": { "type": "string" }, "default": [] },
                    "source": { "type": "string", "default": "cursor" },
                    "distill": { "type": ["boolean","null"], "description": "When true (default), extract a few high-value structured memories from the transcript." },
                    "session_id": { "type": ["string","null"] },
                    "thread_id": { "type": ["string","null"] },
                    "window_id": { "type": ["string","null"] }
                },
                "required": ["content"]
            }
        },
        {
            "name": "search_memory",
            "description": "FTS5 BM25 full-text search weighted by importance. Supports prefix (svelt*) and multi-word queries. Auto-cleans expired.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string" },
                    "limit": { "type": "integer", "default": 10 },
                    "project": { "type": ["string","null"] },
                    "kind": { "type": ["string","null"] },
                    "tags": { "type": ["array","null"], "items": { "type": "string" } }
                },
                "required": ["query"]
            }
        },        {
            "name": "get_memory",
            "description": "Retrieve a single memory by ID.",
            "inputSchema": { "type": "object", "properties": { "id": { "type": "string" } }, "required": ["id"] }
        },
        {
            "name": "update_memory",
            "description": "Update memory content, kind, tags, importance, or expiration.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "id": { "type": "string" },
                    "content": { "type": ["string","null"] },
                    "kind": { "type": ["string","null"] },
                    "tags": { "type": ["array","null"], "items": { "type": "string" } },
                    "importance": { "type": ["integer","null"], "minimum": 1, "maximum": 5 },
                    "expires_at": { "type": ["string","null"] }
                },
                "required": ["id"]
            }
        },
        {
            "name": "delete_memory",
            "description": "Delete a memory by ID.",
            "inputSchema": { "type": "object", "properties": { "id": { "type": "string" } }, "required": ["id"] }
        },
        {
            "name": "list_memories",
            "description": "List memories with optional filters and pagination.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "project": { "type": ["string","null"] },
                    "kind": { "type": ["string","null"] },
                    "limit": { "type": "integer", "default": 20 },
                    "offset": { "type": "integer", "default": 0 }
                }
            }
        },        {
            "name": "get_project_context",
            "description": "Load full project context in ONE call: project memories + global preferences + patterns + snippets. Auto-detects project from working_dir.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "project": { "type": ["string","null"] },
                    "working_dir": { "type": ["string","null"], "description": "Current directory for auto-detection" },
                    "mode": { "type": ["string","null"], "enum": ["safe", "default", "full"], "description": "Context mode. `safe` is the default and excludes credentials unless `full` is explicitly requested." },
                    "session_id": { "type": ["string","null"] },
                    "thread_id": { "type": ["string","null"] },
                    "window_id": { "type": ["string","null"] }
                }
            }
        },
        {
            "name": "get_project_brain",
            "description": "INSTANT PROJECT BRAIN — Dense JSON summary (<1500 tokens): tech stack, architecture, active bugs, recent changes, preferences, key components. Use at start of focused work.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "project": { "type": ["string","null"], "description": "Project name (or null for auto-detect)" },
                    "working_dir": { "type": ["string","null"], "description": "Auto-detect project from path" },
                    "max_tokens": { "type": "integer", "description": "Dynamic budget. Default is 1500" },
                    "compact": { "type": ["boolean","null"], "description": "Compressed shorthand output (~3x fewer tokens). Default false." }
                }
            }
        },
        {
            "name": "register_project",
            "description": "Register project with filesystem path for auto-detection.",
            "inputSchema": {
                "type": "object",
                "properties": { "name": { "type": "string" }, "path": { "type": "string" }, "description": { "type": ["string","null"] } },
                "required": ["name", "path"]
            }
        },
        { "name": "list_projects", "description": "List all projects with memory counts.", "inputSchema": { "type": "object", "properties": {} } },
        { "name": "get_stats", "description": "Database statistics: totals, by kind, by project, expired count, db size, plus hygiene signals (missing paths, stale low-value memories, orphan records, etc.).", "inputSchema": { "type": "object", "properties": {} } },
        {
            "name": "benchmark_recall",
            "description": "Run a local recall benchmark on the current memory base. Uses a fixed golden scenario set first, then generated fallback scenarios if coverage is missing. Measures top-1/top-5 hit rate, cross-project leakage, credential leakage in safe mode, and explain consistency.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "scenario_limit": { "type": "integer", "default": 12 }
                }
            }
        },
        {
            "name": "benchmark_search",
            "description": "Search quality benchmark. Measures R@5, R@10, NDCG@10, cluster coherence (combinatorial reranker quality), and average search latency. Uses real memories as targets.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "scenario_limit": { "type": "integer", "default": 20 }
                }
            }
        },
        {
            "name": "get_global_prompt",
            "description": "Load GLOBAL_PROMPT.md. Auto-scans: 1) configured path, 2) ~/.MemoryPilot/GLOBAL_PROMPT.md, 3) project root GLOBAL_PROMPT.md.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "project": { "type": ["string","null"] },
                    "working_dir": { "type": ["string","null"] }
                }
            }
        },
        {
            "name": "export_memories",
            "description": "Export memories as JSON or Markdown. Useful for backup, sharing, or injecting into Claude.ai.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "project": { "type": ["string","null"], "description": "Filter by project (null=all)" },
                    "format": { "type": "string", "enum": ["json", "markdown"], "default": "markdown" }
                }
            }
        },
        {
            "name": "set_config",
            "description": "Set a config value (e.g. global_prompt_path).",
            "inputSchema": { "type": "object", "properties": { "key": { "type": "string" }, "value": { "type": "string" } }, "required": ["key", "value"] }
        },
        { "name": "migrate_v1", "description": "Import from v1 JSON files. Skips duplicates.", "inputSchema": { "type": "object", "properties": {} } },
        { "name": "cleanup_expired", "description": "Manually remove all expired memories.", "inputSchema": { "type": "object", "properties": {} } },
        { 
            "name": "run_gc", 
            "description": "Trigger Garbage Collection manually. Compresses old bugs/snippets and deletes expired. `preview`/`dry_run` returns exact candidate groups with confidence and hygiene signals before mutating anything.", 
            "inputSchema": { 
                "type": "object", 
                "properties": {
                    "age_days": { "type": "integer", "default": 30 },
                    "importance_threshold": { "type": "integer", "default": 3 },
                    "dry_run": { "type": "boolean", "default": false },
                    "preview": { "type": "boolean", "default": false }
                } 
            } 
        },
        {
            "name": "toggle_auto_lint",
            "description": "Enable or disable background auto-linting/self-healing for the current project.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "enabled": { "type": "boolean" }
                },
                "required": ["enabled"]
            }
        },
        {
            "name": "get_file_context",
            "description": "Get memories related to recently modified files in the working directory. Uses the file watcher to know what you're working on.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "working_dir": { "type": "string" }
                },
                "required": ["working_dir"]
            }
        },
        {
            "name": "kg_add",
            "description": "Add a fact triple to the knowledge graph (subject -> predicate -> object). Deduplicates active triples. Use valid_from/valid_to for temporal facts.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "subject": { "type": "string", "description": "Entity or concept (e.g. 'MemoryPilot')" },
                    "predicate": { "type": "string", "description": "Relationship type (e.g. 'uses', 'built_with', 'depends_on')" },
                    "object": { "type": "string", "description": "Target entity (e.g. 'SQLite')" },
                    "valid_from": { "type": ["string","null"], "description": "When this fact became true (ISO date)" },
                    "valid_to": { "type": ["string","null"], "description": "When this fact stopped being true (ISO date)" },
                    "confidence": { "type": ["number","null"], "description": "Confidence 0.0-1.0 (default 1.0)" },
                    "source_memory_id": { "type": ["string","null"], "description": "Link to source memory" }
                },
                "required": ["subject", "predicate", "object"]
            }
        },
        {
            "name": "kg_invalidate",
            "description": "Mark a knowledge triple as ended/expired. The fact is preserved but marked as no longer current.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "subject": { "type": "string" },
                    "predicate": { "type": "string" },
                    "object": { "type": "string" },
                    "ended": { "type": ["string","null"], "description": "End date (defaults to today)" }
                },
                "required": ["subject", "predicate", "object"]
            }
        },
        {
            "name": "kg_query",
            "description": "Query all relationships for an entity. Optionally filter by time window.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "entity": { "type": "string", "description": "Entity name to query" },
                    "as_of": { "type": ["string","null"], "description": "Show facts valid at this date (ISO date)" },
                    "direction": { "type": "string", "default": "both", "enum": ["outgoing", "incoming", "both"] }
                },
                "required": ["entity"]
            }
        },
        {
            "name": "kg_timeline",
            "description": "Chronological timeline of all facts for an entity, showing when things became true and expired.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "entity": { "type": ["string","null"], "description": "Entity name (null for all entities, limited to 100)" }
                }
            }
        },
        {
            "name": "kg_stats",
            "description": "Knowledge graph overview: entity count, triple count, current vs expired facts, relationship types.",
            "inputSchema": { "type": "object", "properties": {} }
        }
    ]})
}
/// Handle a tools/call request.
pub fn handle_tool_call(db: &Database, name: &str, args: &Value) -> Value {
    match name {
        "recall" => handle_recall(db, args),
        "add_memory" => handle_add(db, args),
        "add_memories" => handle_add_bulk(db, args),
        "add_transcript" => handle_add_transcript(db, args),
        "search_memory" => handle_search(db, args),
        "get_memory" => handle_get(db, args),
        "update_memory" => handle_update(db, args),
        "delete_memory" => handle_delete(db, args),
        "list_memories" => handle_list(db, args),
        "get_project_context" => handle_project_context(db, args),
        "get_project_brain" => handle_get_project_brain(db, args),
        "register_project" => handle_register_project(db, args),
        "list_projects" => handle_list_projects(db),
        "get_stats" => handle_stats(db),
        "benchmark_recall" => handle_benchmark_recall(db, args),
        "benchmark_search" => handle_benchmark_search(db, args),
        "get_global_prompt" => handle_global_prompt(db, args),
        "export_memories" => handle_export(db, args),
        "set_config" => handle_set_config(db, args),
        "migrate_v1" => handle_migrate(db),
        "cleanup_expired" => handle_cleanup(db),
        "run_gc" => handle_run_gc(db, args),
        "toggle_auto_lint" => handle_toggle_lint(args),
        "get_file_context" => handle_get_file_context(db, args),
        "kg_add" => handle_kg_add(db, args),
        "kg_invalidate" => handle_kg_invalidate(db, args),
        "kg_query" => handle_kg_query(db, args),
        "kg_timeline" => handle_kg_timeline(db, args),
        "kg_stats" => handle_kg_stats(db),
        _ => tool_error(&format!("Unknown tool: {}", name)),
    }
}

fn scope_from_args(args: &Value) -> MemoryScope {
    MemoryScope {
        session_id: args.get("session_id").and_then(|v| v.as_str()).map(String::from),
        thread_id: args.get("thread_id").and_then(|v| v.as_str()).map(String::from),
        window_id: args.get("window_id").and_then(|v| v.as_str()).map(String::from),
    }
}

fn handle_recall(db: &Database, args: &Value) -> Value {
    let project = args.get("project").and_then(|v| v.as_str());
    let working_dir = args.get("working_dir").and_then(|v| v.as_str());
    let hints = args.get("hints").and_then(|v| v.as_str());
    let explain = args.get("explain").and_then(|v| v.as_bool()).unwrap_or(false);
    let compact = args.get("compact").and_then(|v| v.as_bool()).unwrap_or(false);
    let scope = scope_from_args(args);
    let mode = match RecallMode::from_str(args.get("mode").and_then(|v| v.as_str())) {
        Ok(mode) => mode,
        Err(error) => return tool_error(&error),
    };
    match db.recall(project, working_dir, hints, mode, explain, compact, &scope) {
        Ok(ctx) => tool_result(&serde_json::to_string_pretty(&ctx).unwrap()),
        Err(e) => tool_error(&e),
    }
}

fn handle_add(db: &Database, args: &Value) -> Value {
    let content = match args.get("content").and_then(|v| v.as_str()) {
        Some(c) if !c.trim().is_empty() => c,
        _ => return tool_error("content is required"),
    };
    let kind = args.get("kind").and_then(|v| v.as_str()).unwrap_or("fact");
    if !VALID_KINDS.contains(&kind) { return tool_error(&format!("Invalid kind '{}'. Valid: {:?}", kind, VALID_KINDS)); }
    let project = args.get("project").and_then(|v| v.as_str());
    let tags: Vec<String> = args.get("tags").and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect()).unwrap_or_default();
    let source = args.get("source").and_then(|v| v.as_str()).unwrap_or("cursor");
    let importance = args.get("importance").and_then(|v| v.as_i64()).unwrap_or(3) as i32;
    let expires_at = args.get("expires_at").and_then(|v| v.as_str());
    let metadata = args.get("metadata").filter(|v| !v.is_null());
    let scope = scope_from_args(args);

    match db.add_memory(content, kind, project, &tags, source, importance, expires_at, metadata, &scope) {
        Ok((mem, was_merged)) => {
            let mut result = serde_json::to_value(&mem).unwrap_or(json!({}));
            if was_merged { result.as_object_mut().map(|o| o.insert("_merged".into(), json!(true))); }
            tool_result(&serde_json::to_string_pretty(&result).unwrap())
        }
        Err(e) => tool_error(&e),
    }
}
fn handle_add_bulk(db: &Database, args: &Value) -> Value {
    let items: Vec<BulkItem> = match args.get("memories").and_then(|v| serde_json::from_value::<Vec<BulkItem>>(v.clone()).ok()) {
        Some(items) if !items.is_empty() => items,
        _ => return tool_error("memories array is required and cannot be empty"),
    };
    match db.add_memories_bulk(&items) {
        Ok((added, merged, skipped)) => {
            tool_result(&format!("Bulk complete: {} added, {} merged (dedup), {} skipped. Total processed: {}.",
                added.len(), merged, skipped, items.len()))
        }
        Err(e) => tool_error(&e),
    }
}

fn handle_add_transcript(db: &Database, args: &Value) -> Value {
    let content = match args.get("content").and_then(|v| v.as_str()) {
        Some(c) if !c.trim().is_empty() => c,
        _ => return tool_error("content is required"),
    };
    let project = args.get("project").and_then(|v| v.as_str()).map(String::from);
    let tags: Vec<String> = args.get("tags").and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect()).unwrap_or_default();
    let source = args.get("source").and_then(|v| v.as_str()).unwrap_or("cursor").to_string();
    let distill = args.get("distill").and_then(|v| v.as_bool()).unwrap_or(true);
    let scope = scope_from_args(args);

    match db.add_transcript(content, project.as_deref(), &tags, &source, &scope, distill) {
        Ok(report) => tool_result(&serde_json::to_string_pretty(&report).unwrap()),
        Err(e) => tool_error(&e),
    }
}

fn handle_search(db: &Database, args: &Value) -> Value {
    let query = match args.get("query").and_then(|v| v.as_str()) {
        Some(q) if !q.trim().is_empty() => q,
        _ => return tool_error("query is required"),
    };
    let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(10) as usize;
    let project = args.get("project").and_then(|v| v.as_str());
    let kind = args.get("kind").and_then(|v| v.as_str());
    let tags: Option<Vec<String>> = args.get("tags").and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect());
        
    let mut watcher_keywords = Vec::new();
    if let Some(watcher) = crate::WATCHER_STATE.get() {
        if let Ok(state) = watcher.lock() {
            watcher_keywords = state.get_boost_keywords();
        }
    }
    
    let wk_ref = if watcher_keywords.is_empty() { None } else { Some(watcher_keywords.as_slice()) };
    
    match db.search(query, limit, project, kind, tags.as_deref(), wk_ref) {
        Ok(results) => {
            let output = json!({ "query": query, "count": results.len(),
                "results": results.iter().map(|r| json!({
                    "id": r.memory.id, "content": r.memory.content, "kind": r.memory.kind,
                    "project": r.memory.project, "tags": r.memory.tags, "score": r.score, "importance": r.memory.importance,
                })).collect::<Vec<_>>()
            });
            tool_result(&serde_json::to_string_pretty(&output).unwrap())
        }
        Err(e) => tool_error(&e),
    }
}

fn handle_get(db: &Database, args: &Value) -> Value {
    let id = match args.get("id").and_then(|v| v.as_str()) { Some(i) => i, _ => return tool_error("id required") };
    match db.get_memory(id) {
        Ok(Some(mem)) => tool_result(&serde_json::to_string_pretty(&mem).unwrap()),
        Ok(None) => tool_error(&format!("Not found: {}", id)),
        Err(e) => tool_error(&e),
    }
}
fn handle_update(db: &Database, args: &Value) -> Value {
    let id = match args.get("id").and_then(|v| v.as_str()) { Some(i) => i, _ => return tool_error("id required") };
    let content = args.get("content").and_then(|v| v.as_str());
    let kind = args.get("kind").and_then(|v| v.as_str());
    let tags: Option<Vec<String>> = args.get("tags").and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect());
    let importance = args.get("importance").and_then(|v| v.as_i64()).map(|i| i as i32);
    let expires_at = args.get("expires_at").and_then(|v| v.as_str());
    match db.update_memory_full(id, content, kind, tags.as_deref(), importance, expires_at, None) {
        Ok(Some(mem)) => tool_result(&serde_json::to_string_pretty(&mem).unwrap()),
        Ok(None) => tool_error(&format!("Not found: {}", id)),
        Err(e) => tool_error(&e),
    }
}

fn handle_delete(db: &Database, args: &Value) -> Value {
    let id = match args.get("id").and_then(|v| v.as_str()) { Some(i) => i, _ => return tool_error("id required") };
    match db.delete_memory(id) {
        Ok(true) => tool_result(&format!("Deleted: {}", id)),
        Ok(false) => tool_error(&format!("Not found: {}", id)),
        Err(e) => tool_error(&e),
    }
}

fn handle_list(db: &Database, args: &Value) -> Value {
    let project = args.get("project").and_then(|v| v.as_str());
    let kind = args.get("kind").and_then(|v| v.as_str());
    let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(20) as usize;
    let offset = args.get("offset").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    match db.list_memories(project, kind, None, limit, offset) {
        Ok((memories, total)) => {
            tool_result(&serde_json::to_string_pretty(&json!({"total":total,"count":memories.len(),"offset":offset,"memories":memories})).unwrap())
        }
        Err(e) => tool_error(&e),
    }
}
fn handle_project_context(db: &Database, args: &Value) -> Value {
    let project = args.get("project").and_then(|v| v.as_str());
    let working_dir = args.get("working_dir").and_then(|v| v.as_str());
    let scope = scope_from_args(args);
    let mode = match RecallMode::from_str(args.get("mode").and_then(|v| v.as_str())) {
        Ok(mode) => mode,
        Err(error) => return tool_error(&error),
    };
    match db.get_project_context(project, working_dir, mode, &scope) {
        Ok(ctx) => tool_result(&serde_json::to_string_pretty(&ctx).unwrap()),
        Err(e) => tool_error(&e),
    }
}

fn handle_get_project_brain(db: &Database, args: &Value) -> Value {
    let proj_detect = args.get("working_dir").and_then(|v| v.as_str()).and_then(|wd| db.detect_project(wd).ok().flatten());

    let project = match args.get("project").and_then(|v| v.as_str()).or_else(|| proj_detect.as_deref()) {
        Some(p) => p,
        None => return tool_error("project or working_dir is required, and project must be found"),
    };

    let max_tokens = args.get("max_tokens").and_then(|v| v.as_u64()).map(|v| v as usize);
    let compact = args.get("compact").and_then(|v| v.as_bool()).unwrap_or(false);

    match db.get_project_brain(project, max_tokens, compact) {
        Ok(brain) => tool_result(&serde_json::to_string_pretty(&brain).unwrap()),
        Err(e) => tool_error(&e),
    }
}

fn handle_register_project(db: &Database, args: &Value) -> Value {
    let name = match args.get("name").and_then(|v| v.as_str()) { Some(n) => n, _ => return tool_error("name required") };
    let path = match args.get("path").and_then(|v| v.as_str()) { Some(p) => p, _ => return tool_error("path required") };
    let desc = args.get("description").and_then(|v| v.as_str());
    match db.register_project(name, path, desc) {
        Ok(proj) => tool_result(&serde_json::to_string_pretty(&proj).unwrap()),
        Err(e) => tool_error(&e),
    }
}

fn handle_list_projects(db: &Database) -> Value {
    match db.list_projects() {
        Ok(p) => tool_result(&serde_json::to_string_pretty(&p).unwrap()),
        Err(e) => tool_error(&e),
    }
}

fn handle_stats(db: &Database) -> Value {
    match db.stats() {
        Ok(s) => tool_result(&serde_json::to_string_pretty(&s).unwrap()),
        Err(e) => tool_error(&e),
    }
}

fn handle_benchmark_recall(db: &Database, args: &Value) -> Value {
    let scenario_limit = args.get("scenario_limit").and_then(|v| v.as_u64()).unwrap_or(12) as usize;
    match db.benchmark_recall(scenario_limit) {
        Ok(report) => tool_result(&serde_json::to_string_pretty(&report).unwrap()),
        Err(error) => tool_error(&error),
    }
}

fn handle_benchmark_search(db: &Database, args: &Value) -> Value {
    let scenario_limit = args.get("scenario_limit").and_then(|v| v.as_u64()).unwrap_or(20) as usize;
    match db.benchmark_search(scenario_limit) {
        Ok(report) => tool_result(&serde_json::to_string_pretty(&report).unwrap()),
        Err(error) => tool_error(&error),
    }
}

fn handle_global_prompt(db: &Database, args: &Value) -> Value {
    let project = args.get("project").and_then(|v| v.as_str());
    let working_dir = args.get("working_dir").and_then(|v| v.as_str());
    match db.get_global_prompt(project, working_dir) {
        Some(prompt) => tool_result(&prompt),
        None => tool_error("No GLOBAL_PROMPT.md found. Place it in ~/.MemoryPilot/ or project root, or use set_config(key='global_prompt_path')."),
    }
}

fn handle_export(db: &Database, args: &Value) -> Value {
    let project = args.get("project").and_then(|v| v.as_str());
    let format = args.get("format").and_then(|v| v.as_str()).unwrap_or("markdown");
    match db.export_memories(project, format) {
        Ok(output) => tool_result(&output),
        Err(e) => tool_error(&e),
    }
}

fn handle_set_config(db: &Database, args: &Value) -> Value {
    const ALLOWED_KEYS: &[&str] = &["global_prompt_path", "auto_lint", "compact_default"];
    let key = match args.get("key").and_then(|v| v.as_str()) { Some(k) => k, _ => return tool_error("key required") };
    let value = match args.get("value").and_then(|v| v.as_str()) { Some(v) => v, _ => return tool_error("value required") };
    if !ALLOWED_KEYS.contains(&key) {
        return tool_error(&format!("Unknown config key '{}'. Allowed: {}", key, ALLOWED_KEYS.join(", ")));
    }
    if key == "global_prompt_path" {
        let path = std::path::Path::new(value);
        if let Ok(canonical) = std::fs::canonicalize(path) {
            let home = dirs::home_dir().unwrap_or_default();
            if !canonical.starts_with(&home) {
                return tool_error("global_prompt_path must be within your home directory");
            }
            if !canonical.to_string_lossy().ends_with(".md") {
                return tool_error("global_prompt_path must point to a .md file");
            }
        } else if !value.starts_with("~/") && !value.starts_with(&dirs::home_dir().unwrap_or_default().to_string_lossy().to_string()) {
            return tool_error("global_prompt_path must be within your home directory");
        }
    }
    match db.set_config(key, value) {
        Ok(()) => tool_result(&format!("Config '{}' = '{}'", key, value)),
        Err(e) => tool_error(&e),
    }
}

fn handle_migrate(db: &Database) -> Value {
    match db.migrate_from_v1() {
        Ok(count) => tool_result(&format!("Migrated {} memories from v1 to SQLite.", count)),
        Err(e) => tool_error(&format!("Migration failed: {}", e)),
    }
}

fn handle_cleanup(db: &Database) -> Value {
    match db.cleanup_expired() {
        Ok(count) => tool_result(&format!("Cleaned up {} expired memories.", count)),
        Err(e) => tool_error(&e),
    }
}

fn handle_run_gc(db: &Database, args: &Value) -> Value {
    let mut config = crate::gc::GcConfig::default();
    if let Some(age) = args.get("age_days").and_then(|v| v.as_i64()) { config.age_days = age; }
    if let Some(imp) = args.get("importance_threshold").and_then(|v| v.as_i64()) { config.importance_threshold = imp as i32; }
    let dry_run = args.get("dry_run").and_then(|v| v.as_bool()).unwrap_or(false)
        || args.get("preview").and_then(|v| v.as_bool()).unwrap_or(false);
    
    match db.run_gc(&config, dry_run) {
        Ok(report) => tool_result(&serde_json::to_string_pretty(&report).unwrap()),
        Err(e) => tool_error(&e),
    }
}

fn handle_toggle_lint(args: &Value) -> Value {
    let enabled = match args.get("enabled").and_then(|v| v.as_bool()) {
        Some(b) => b,
        None => return tool_error("enabled boolean is required"),
    };
    
    if let Some(watcher) = crate::WATCHER_STATE.get() {
        if let Ok(mut state) = watcher.lock() {
            state.auto_lint = enabled;
            if enabled {
                tool_result("Auto-linting (Self-Healing) has been ENABLED for this project.")
            } else {
                state.active_lint_error = None;
                tool_result("Auto-linting has been DISABLED.")
            }
        } else {
            tool_error("Could not acquire watcher lock")
        }
    } else {
        tool_error("Watcher is not running")
    }
}

fn handle_get_file_context(db: &Database, args: &Value) -> Value {
    let _wd = match args.get("working_dir").and_then(|v| v.as_str()) {
        Some(w) => w,
        None => return tool_error("working_dir required"),
    };
    
    let mut keywords = Vec::new();
    if let Some(watcher) = crate::WATCHER_STATE.get() {
        if let Ok(state) = watcher.lock() {
            keywords = state.get_boost_keywords();
        }
    }
    
    if keywords.is_empty() {
        return tool_result("No recent file changes detected by watcher.");
    }
    
    let query = keywords.join(" ");
    match db.search(&query, 10, None, None, None, Some(&keywords)) {
        Ok(results) => {
            let output = json!({ 
                "recent_file_keywords": keywords, 
                "count": results.len(),
                "results": results.iter().map(|r| json!({
                    "id": r.memory.id, "content": r.memory.content, "kind": r.memory.kind,
                    "project": r.memory.project, "tags": r.memory.tags, "score": r.score, "importance": r.memory.importance,
                })).collect::<Vec<_>>()
            });
            tool_result(&serde_json::to_string_pretty(&output).unwrap())
        }
        Err(e) => tool_error(&e),
    }
}

// ─── KNOWLEDGE GRAPH TOOLS ───────────────────────────

fn handle_kg_add(db: &Database, args: &Value) -> Value {
    let subject = match args.get("subject").and_then(|v| v.as_str()) {
        Some(s) => s, None => return tool_error("subject required"),
    };
    let predicate = match args.get("predicate").and_then(|v| v.as_str()) {
        Some(s) => s, None => return tool_error("predicate required"),
    };
    let object = match args.get("object").and_then(|v| v.as_str()) {
        Some(s) => s, None => return tool_error("object required"),
    };
    let valid_from = args.get("valid_from").and_then(|v| v.as_str());
    let valid_to = args.get("valid_to").and_then(|v| v.as_str());
    let confidence = args.get("confidence").and_then(|v| v.as_f64());
    let source_id = args.get("source_memory_id").and_then(|v| v.as_str());
    match db.add_triple(subject, predicate, object, valid_from, valid_to, confidence, source_id) {
        Ok(result) => tool_result(&serde_json::to_string_pretty(&result).unwrap()),
        Err(e) => tool_error(&e),
    }
}

fn handle_kg_invalidate(db: &Database, args: &Value) -> Value {
    let subject = match args.get("subject").and_then(|v| v.as_str()) {
        Some(s) => s, None => return tool_error("subject required"),
    };
    let predicate = match args.get("predicate").and_then(|v| v.as_str()) {
        Some(s) => s, None => return tool_error("predicate required"),
    };
    let object = match args.get("object").and_then(|v| v.as_str()) {
        Some(s) => s, None => return tool_error("object required"),
    };
    let ended = args.get("ended").and_then(|v| v.as_str());
    match db.invalidate_triple(subject, predicate, object, ended) {
        Ok(result) => tool_result(&serde_json::to_string_pretty(&result).unwrap()),
        Err(e) => tool_error(&e),
    }
}

fn handle_kg_query(db: &Database, args: &Value) -> Value {
    let entity = match args.get("entity").and_then(|v| v.as_str()) {
        Some(s) => s, None => return tool_error("entity required"),
    };
    let as_of = args.get("as_of").and_then(|v| v.as_str());
    let direction = args.get("direction").and_then(|v| v.as_str()).unwrap_or("both");
    match db.query_kg_entity(entity, as_of, direction) {
        Ok(result) => tool_result(&serde_json::to_string_pretty(&result).unwrap()),
        Err(e) => tool_error(&e),
    }
}

fn handle_kg_timeline(db: &Database, args: &Value) -> Value {
    let entity = args.get("entity").and_then(|v| v.as_str());
    match db.kg_timeline(entity) {
        Ok(result) => tool_result(&serde_json::to_string_pretty(&result).unwrap()),
        Err(e) => tool_error(&e),
    }
}

fn handle_kg_stats(db: &Database) -> Value {
    match db.kg_stats() {
        Ok(result) => tool_result(&serde_json::to_string_pretty(&result).unwrap()),
        Err(e) => tool_error(&e),
    }
}