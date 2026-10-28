//! Command-line entry points for the conversation index.

use crate::protocol::{tool_error, tool_result};
use memorypilot_core::conversations::{self, ConversationStore, SearchOptions, SyncOptions};
use memorypilot_core::db::Database;
use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::Mutex;
use std::time::{Duration, Instant};

fn open_store() -> ConversationStore {
    match ConversationStore::open_default() {
        Ok(store) => store,
        Err(error) => {
            eprintln!("✗ {}", error);
            std::process::exit(1);
        }
    }
}

/// `--index-conversations [--no-embed]`: full catch-up, then embeddings.
pub fn run_index(args: &[String]) {
    let mut store = open_store();
    let report = conversations::sync(&mut store, &SyncOptions { budget: None });
    println!("{}", serde_json::to_string_pretty(&report).unwrap_or_default());
    if !args.iter().any(|a| a == "--no-embed") {
        let started = Instant::now();
        match conversations::embed_pending(&mut store, None) {
            Ok(count) => eprintln!("embedded {} messages in {:.1}s", count, started.elapsed().as_secs_f64()),
            Err(error) => {
                eprintln!("✗ {}", error);
                std::process::exit(1);
            }
        }
    }
    match store.stats() {
        Ok(stats) => println!("{}", serde_json::to_string_pretty(&stats).unwrap_or_default()),
        Err(error) => eprintln!("✗ {}", error),
    }
}

/// `--find-conversation "<query>" [--limit N]`
pub fn run_find(args: &[String]) {
    let Some(query) = value_after(args, "--find-conversation") else {
        eprintln!("✗ --find-conversation needs a query");
        std::process::exit(2);
    };
    let limit = value_after(args, "--limit").and_then(|v| v.parse().ok()).unwrap_or(5);
    let store = open_store();
    let options = SearchOptions { query, limit, use_vectors: true, ..Default::default() };
    match store.search(&options) {
        Ok(hits) => println!("{}", serde_json::to_string_pretty(&hits).unwrap_or_default()),
        Err(error) => {
            eprintln!("✗ {}", error);
            std::process::exit(1);
        }
    }
}

#[derive(Deserialize)]
struct Question {
    q: String,
    accept: Vec<String>,
}

/// `--benchmark-conversations <questions.json> [--lexical-only]`: each
/// question lists the conversation id prefixes that count as right.
pub fn run_benchmark(args: &[String]) {
    let Some(path) = value_after(args, "--benchmark-conversations") else {
        eprintln!("✗ --benchmark-conversations needs a questions file");
        std::process::exit(2);
    };
    let questions: Vec<Question> = match std::fs::read_to_string(&path)
        .map_err(|e| e.to_string())
        .and_then(|raw| serde_json::from_str(&raw).map_err(|e| e.to_string()))
    {
        Ok(questions) => questions,
        Err(error) => {
            eprintln!("✗ {}: {}", path, error);
            std::process::exit(2);
        }
    };
    let use_vectors = !args.iter().any(|a| a == "--lexical-only");
    let store = open_store();
    let mut ranks = Vec::new();
    let mut latencies = Vec::new();
    let mut rows = Vec::new();
    for question in &questions {
        let started = Instant::now();
        let options = SearchOptions { query: question.q.clone(), limit: 10, use_vectors, ..Default::default() };
        let hits = store.search(&options).unwrap_or_default();
        latencies.push(started.elapsed());
        let rank = hits
            .iter()
            .position(|hit| question.accept.iter().any(|a| hit.conversation_id.starts_with(a.as_str())))
            .map(|r| r + 1);
        ranks.push(rank);
        rows.push(json!({
            "q": question.q,
            "rank": rank,
            "top": hits.iter().take(3).map(|h| format!("{} {}", &h.conversation_id[..8.min(h.conversation_id.len())], h.title.clone().unwrap_or_default())).collect::<Vec<_>>(),
        }));
    }
    let total = questions.len().max(1) as f64;
    let at = |k: usize| ranks.iter().filter(|r| r.is_some_and(|r| r <= k)).count() as f64 / total;
    let mrr = ranks.iter().map(|r| r.map(|r| 1.0 / r as f64).unwrap_or(0.0)).sum::<f64>() / total;
    latencies.sort();
    let median = latencies.get(latencies.len() / 2).copied().unwrap_or(Duration::ZERO);
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "questions": questions.len(),
            "vectors": use_vectors,
            "recall_at_1": at(1),
            "recall_at_3": at(3),
            "recall_at_5": at(5),
            "mrr": (mrr * 1000.0).round() / 1000.0,
            "median_ms": median.as_millis(),
            "details": rows,
        }))
        .unwrap_or_default()
    );
}

/// Catch-up before a search, at most once a minute and within a small
/// budget: the conversation the user is in right now keeps changing,
/// and a full catch-up belongs to `--index-conversations`.
fn refresh(store: &mut ConversationStore) {
    static LAST: Mutex<Option<Instant>> = Mutex::new(None);
    let Ok(mut last) = LAST.lock() else { return };
    if last.is_some_and(|at| at.elapsed() < Duration::from_secs(60)) {
        return;
    }
    *last = Some(Instant::now());
    drop(last);
    conversations::sync(store, &SyncOptions { budget: Some(Duration::from_secs(3)) });
    let _ = conversations::embed_pending(store, Some(Duration::from_millis(1500)));
}

pub fn handle_find_conversation(db: &Database, args: &Value) -> Value {
    let text = |key: &str| {
        args.get(key)
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|v| !v.is_empty())
            .map(str::to_string)
    };
    let query = text("query");
    let reference = text("reference");
    if query.is_none() && reference.is_none() {
        return tool_error("query or reference required");
    }
    let limit = args.get("limit").and_then(Value::as_u64).unwrap_or(5).clamp(1, 20) as usize;
    let offset = args.get("offset").and_then(Value::as_u64).unwrap_or(0).min(200) as usize;
    let mut store = match ConversationStore::open_default() {
        Ok(store) => store,
        Err(error) => return tool_error(&error),
    };
    refresh(&mut store);

    let mut hits = Vec::new();
    if let Some(reference) = reference.as_deref() {
        let kind = if reference.len() >= 7 && reference.chars().all(|c| c.is_ascii_hexdigit()) {
            "commit"
        } else {
            "file"
        };
        let value = if kind == "commit" { reference.to_lowercase() } else { reference.to_string() };
        let found = match store.by_reference(kind, &value, offset + limit + 1) {
            Ok(found) => found,
            Err(error) => return tool_error(&error),
        };
        for (conversation_id, mentions) in found {
            hits.push(json!({ "conversation_id": conversation_id, "mentions": mentions }));
        }
    }
    let mut related_memories = Vec::new();
    if let Some(query) = query.as_deref() {
        let options = SearchOptions {
            query: query.to_string(),
            limit: offset + limit + 1,
            tool: text("tool"),
            project: text("project"),
            after: text("after"),
            before: text("before"),
            use_vectors: true,
        };
        match store.search(&options) {
            Ok(found) => hits.extend(found.into_iter().map(|hit| json!(hit))),
            Err(error) => return tool_error(&error),
        }
        if let Ok(memories) = db.search(query, 3, text("project").as_deref(), None, None, None) {
            related_memories = memories
                .into_iter()
                .map(|r| {
                    json!({
                        "id": r.memory.id,
                        "kind": r.memory.kind,
                        "project": r.memory.project,
                        "created_at": r.memory.created_at,
                        "content": memorypilot_core::text::preview(&r.memory.content, 280),
                    })
                })
                .collect();
        }
    }
    let has_more = hits.len() > offset + limit;
    let page: Vec<Value> = hits.into_iter().skip(offset).take(limit).collect();
    let body = json!({
        "conversations": page,
        "next_offset": has_more.then_some(offset + limit),
        "related_memories": related_memories,
    });
    tool_result(&serde_json::to_string_pretty(&body).unwrap_or_default())
}

pub fn handle_read_conversation(args: &Value) -> Value {
    let Some(conversation_id) = args.get("conversation_id").and_then(Value::as_str) else {
        return tool_error("conversation_id required");
    };
    let message_id = args.get("message_id").and_then(Value::as_i64);
    let offset = args.get("offset").and_then(Value::as_u64).unwrap_or(0) as usize;
    let limit = args.get("limit").and_then(Value::as_u64).unwrap_or(20).clamp(1, 100) as usize;
    let store = match ConversationStore::open_default() {
        Ok(store) => store,
        Err(error) => return tool_error(&error),
    };
    match store.read(conversation_id, message_id, offset, limit) {
        Ok((messages, next_offset)) => tool_result(
            &serde_json::to_string_pretty(&json!({ "messages": messages, "next_offset": next_offset }))
                .unwrap_or_default(),
        ),
        Err(error) => tool_error(&error),
    }
}

fn value_after(args: &[String], flag: &str) -> Option<String> {
    args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).cloned()
}
