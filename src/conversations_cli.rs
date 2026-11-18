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

/// `--index-conversations [--no-embed] [--rebuild]`: full catch-up, then
/// embeddings. `--rebuild` reads every conversation again from its source.
pub fn run_index(args: &[String]) {
    let mut store = open_store();
    if args.iter().any(|a| a == "--rebuild") {
        match store.forget_what_was_read() {
            Ok(origins) => eprintln!("rebuild: {} conversations and sub-agents will be read again", origins),
            Err(error) => {
                eprintln!("✗ {}", error);
                std::process::exit(1);
            }
        }
    }
    let report = conversations::sync(&mut store, &SyncOptions { budget: None });
    println!("{}", serde_json::to_string_pretty(&report).unwrap_or_default());
    // A conversation belongs to the project of its folder or repository,
    // whatever the folder is called today.
    match Database::open() {
        Ok(db) => {
            match store.assign_known_projects(&|folder| db.known_project_for_path(folder)) {
                Ok(changed) => eprintln!("filed {} conversations under the project of their folder", changed),
                Err(error) => eprintln!("⚠ {}", error),
            }
            match db.record_submissions(&store) {
                Ok(added) => eprintln!("recorded {} requests typed on third-party sites, to confirm", added),
                Err(error) => eprintln!("⚠ {}", error),
            }
        }
        Err(error) => eprintln!("⚠ projects not assigned: {}", error),
    }
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

/// `--redact-conversations [--apply]`: masks the secrets already in the
/// index. Without `--apply` only counts; no value is ever printed.
pub fn run_redact(args: &[String]) {
    let apply = args.iter().any(|a| a == "--apply");
    let mut store = open_store();
    match store.redact_stored_secrets(apply) {
        Ok(report) => {
            println!("{}", serde_json::to_string_pretty(&report).unwrap_or_default());
            println!(
                "{} : {} segment(s) et {} titre(s) dans {} conversation(s){}.",
                if apply { "Appliqué" } else { "À blanc (--apply pour écrire)" },
                report.segments,
                report.titles,
                report.conversations,
                if apply && report.segments > 0 { ", segments remis en file pour les embeddings" } else { "" }
            );
        }
        Err(error) => {
            eprintln!("✗ {}", error);
            std::process::exit(1);
        }
    }
}

/// `--archive-cursor [--apply] [--out DIR] [--store PATH]`: one Markdown
/// file per Cursor conversation, at full length, secrets masked. Without
/// `--apply` only counts. Cursor's store is opened immutable: Cursor must
/// be closed, and nothing is ever written to it. No message is printed.
///
/// `--archive-cursor --check-secrets`: says whether every stored
/// credential can be read (exit 4 otherwise); `--apply` refuses to write
/// when one cannot, since its value would not be masked.
///
/// `--archive-cursor --index`: records in the conversation index the
/// project and folder the archive holds for each conversation (run it once
/// the archive has been checked).
pub fn run_archive(args: &[String]) {
    use memorypilot_core::conversations::{archive, cursor};
    let fail = |message: String| -> ! {
        eprintln!("✗ {}", message);
        std::process::exit(1);
    };
    let out_dir = match value_after(args, "--out").map(std::path::PathBuf::from).or_else(archive::default_dir) {
        Some(dir) => dir,
        None => fail("Cannot find home directory".to_string()),
    };
    if args.iter().any(|a| a == "--index") {
        let (conversations, errors) = archive::read_archive(&out_dir);
        let mut store = open_store();
        let mut updated = 0;
        let mut absent = 0;
        for conversation in &conversations {
            match store.set_identity(&conversation.id, conversation.project.as_deref(), conversation.workspace.as_deref(), true) {
                Ok(true) => updated += 1,
                Ok(false) => absent += 1,
                Err(error) => fail(error),
            }
        }
        println!(
            "{}",
            json!({ "archived": conversations.len(), "index_updated": updated, "not_in_index": absent, "unreadable_files": errors })
        );
        return;
    }
    let store_path = match value_after(args, "--store").map(std::path::PathBuf::from).or_else(cursor::default_store_path) {
        Some(path) => path,
        None => fail("Cursor store not found (--store PATH)".to_string()),
    };
    let db = match Database::open() {
        Ok(db) => db,
        Err(error) => fail(error),
    };
    // Both sources of known values: the prints file and the credentials.
    let known_prints = memorypilot_core::secrets::reload_known_secrets();
    // A credential the vault cannot open is a value the filter does not
    // know: it would be archived in clear. Nothing is written then.
    let (credentials, readable) = match memorypilot_core::layout::memory_db().map(|path| memorypilot_core::db::credentials_readable(&path)) {
        Some(Ok(counts)) => counts,
        Some(Err(error)) => fail(error),
        None => fail("Cannot find home directory".to_string()),
    };
    let apply = args.iter().any(|a| a == "--apply");
    if args.iter().any(|a| a == "--check-secrets") {
        println!("{}", json!({ "credentials": credentials, "readable": readable, "known_secret_prints": known_prints }));
        std::process::exit(if readable < credentials { 4 } else { 0 });
    }
    if readable < credentials {
        let message = format!(
            "{} des {} identifiants enregistrés sont illisibles (clé du coffre hors d'atteinte : Trousseau refusé ou sans réponse) : leurs valeurs ne seraient pas masquées.",
            credentials - readable,
            credentials
        );
        if apply {
            fail(format!("{} Rien n'est écrit.", message));
        }
        eprintln!("⚠ {}", message);
    }
    let options = archive::ArchiveOptions {
        store: &store_path,
        out_dir: &out_dir,
        apply,
        home: dirs::home_dir().map(|home| home.to_string_lossy().into_owned()),
        known_project: &|folder| db.known_project_for_path(folder),
        remote_of: &|project| db.project_remote(project),
        slug: &|name| Database::project_slug(name),
        is_project: &|name| db.is_known_project(name),
    };
    match archive::archive_cursor(&options) {
        Ok(report) => {
            let incomplete = !report.unreadable.is_empty();
            let mut value = serde_json::to_value(&report).unwrap_or_default();
            value["known_secret_prints"] = json!(known_prints);
            value["credentials"] = json!({ "stored": credentials, "readable": readable });
            println!("{}", serde_json::to_string_pretty(&value).unwrap_or_default());
            eprintln!(
                "{} : {} conversation(s), {} sans projet, {} secret(s) masqué(s).",
                if report.applied { "Écrit" } else { "À blanc (--apply pour écrire)" },
                report.conversations,
                report.without_project,
                report.secrets_masked
            );
            if incomplete {
                eprintln!("✗ {} conversation(s) illisible(s), voir \"unreadable\".", report.unreadable.len());
                std::process::exit(3);
            }
        }
        Err(error) => fail(error),
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
    // The conversation the caller is in holds the question just asked:
    // it is left out unless asked for.
    let include_current = args.get("include_current").and_then(Value::as_bool).unwrap_or(false);
    let current = (!include_current).then(|| text("current_conversation_id").or_else(crate::active_session::calling_conversation)).flatten();
    let is_current = |hit: &Value| current.is_some() && hit.get("conversation_id").and_then(Value::as_str) == current.as_deref();
    let spare = usize::from(current.is_some());

    let mut hits = Vec::new();
    if let Some(reference) = reference.as_deref() {
        let is_commit = reference.len() >= 7 && reference.chars().all(|c| c.is_ascii_hexdigit());
        let is_address = reference.starts_with("http://") || reference.starts_with("https://");
        let lookups: Vec<(&str, String)> = if is_commit {
            vec![("commit", reference.to_lowercase())]
        } else if is_address {
            let address = reference.split(['?', '#']).next().unwrap_or(reference).trim_end_matches('/');
            vec![("url", address.to_string())]
        } else {
            // A file path first; a bare site name ("developers.example.com") is no file.
            vec![("file", reference.to_string()), ("url", reference.to_string())]
        };
        for (kind, value) in lookups {
            let found = match store.by_reference(kind, &value, offset + limit + 1 + spare) {
                Ok(found) => found,
                Err(error) => return tool_error(&error),
            };
            if found.is_empty() {
                continue;
            }
            for (conversation_id, mentions) in found {
                hits.push(json!({ "conversation_id": conversation_id, "mentions": mentions }));
            }
            break;
        }
    }
    let mut related_memories = Vec::new();
    if let Some(query) = query.as_deref() {
        let options = SearchOptions {
            query: query.to_string(),
            limit: offset + limit + 1 + spare,
            tool: text("tool"),
            project_aliases: text("project").map(|project| db.name_aliases(&project)).unwrap_or_default(),
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
    let found = hits.len();
    hits.retain(|hit| !is_current(hit));
    let left_out = (hits.len() < found).then(|| current.clone()).flatten();
    let has_more = hits.len() > offset + limit;
    let page: Vec<Value> = hits.into_iter().skip(offset).take(limit).collect();
    // An empty answer is read as "it never happened" unless told otherwise.
    let note = page.is_empty().then_some(
        "Nothing found. This does not prove it never happened: a session may be missing from the index, or written in another language or with other words. Say it was not found, not that it does not exist.",
    );
    let coverage_warnings = store.coverage_warnings(None, chrono::Utc::now());
    let body = json!({
        "conversations": page,
        "note": note,
        "coverage_warnings": coverage_warnings,
        "next_offset": has_more.then_some(offset + limit),
        "current_conversation_left_out": left_out,
        "related_memories": related_memories,
    });
    tool_result(&compact_json(body))
}

/// Results go straight into the agent's context: no indentation, no
/// empty fields.
fn compact_json(mut value: Value) -> String {
    fn prune(value: &mut Value) {
        match value {
            Value::Object(map) => {
                map.retain(|k, v| k == "next_offset" || !(v.is_null() || v.as_array().is_some_and(Vec::is_empty)));
                map.values_mut().for_each(prune);
            }
            Value::Array(items) => items.iter_mut().for_each(prune),
            _ => {}
        }
    }
    prune(&mut value);
    serde_json::to_string(&value).unwrap_or_default()
}

/// `ingest_conversation`: stores a claude.ai or Gemini conversation an
/// importer pushes. Only messages not received before are written.
pub fn handle_ingest_conversation(args: &Value) -> Value {
    let conversation = match conversations::imported::parse(args) {
        Ok(conversation) => conversation,
        Err(error) => return tool_error(&error),
    };
    let mut store = match ConversationStore::open_default() {
        Ok(store) => store,
        Err(error) => return tool_error(&error),
    };
    match conversations::imported::ingest(&mut store, &conversation) {
        Ok(report) => tool_result(&compact_json(json!(report))),
        Err(error) => tool_error(&error),
    }
}

pub fn handle_read_conversation(args: &Value) -> Value {
    let Some(conversation_id) = args.get("conversation_id").and_then(Value::as_str) else {
        return tool_error("conversation_id required");
    };
    let message_id = args.get("message_id").and_then(Value::as_i64);
    let offset = args.get("offset").and_then(Value::as_u64).unwrap_or(0) as usize;
    let limit = args.get("limit").and_then(Value::as_u64).unwrap_or(6).clamp(1, 50) as usize;
    let store = match ConversationStore::open_default() {
        Ok(store) => store,
        Err(error) => return tool_error(&error),
    };
    // `at`: the page around a moment, as recorded in the origin of a memory.
    let message_id = match (message_id, args.get("at").and_then(Value::as_str)) {
        (None, Some(at)) => match store.message_at(conversation_id, at) {
            Ok(found) => found,
            Err(error) => return tool_error(&error),
        },
        (message_id, _) => message_id,
    };
    match store.read(conversation_id, message_id, offset, limit) {
        Ok((messages, _)) if messages.is_empty() && offset == 0 => tool_error(&format!(
            "No conversation {}: pass the full conversation_id returned by find_conversation",
            conversation_id
        )),
        Ok((messages, next_offset)) => tool_result(&compact_json(json!({ "messages": messages, "next_offset": next_offset }))),
        Err(error) => tool_error(&error),
    }
}

/// `search_in_conversation`: the messages of one conversation that
/// match a query, so a long conversation is searched, not paged through.
pub fn handle_search_in_conversation(args: &Value) -> Value {
    let Some(conversation_id) = args.get("conversation_id").and_then(Value::as_str) else {
        return tool_error("conversation_id required");
    };
    let Some(query) = args.get("query").and_then(Value::as_str).map(str::trim).filter(|query| !query.is_empty()) else {
        return tool_error("query required");
    };
    let limit = args.get("limit").and_then(Value::as_u64).unwrap_or(10).clamp(1, 50) as usize;
    let mut store = match ConversationStore::open_default() {
        Ok(store) => store,
        Err(error) => return tool_error(&error),
    };
    refresh(&mut store);
    match store.search_within(conversation_id, query, limit, true) {
        Ok(None) => tool_error(&format!(
            "No conversation {}: pass the full conversation_id returned by find_conversation",
            conversation_id
        )),
        Ok(Some(messages)) => {
            let note = messages.is_empty().then_some(
                "No message matches. The index keeps each request, the answers and longer drafts of the agent and its actions: try other words or the other language before concluding it is not there.",
            );
            let coverage_warnings = store.coverage_warnings(Some(conversation_id), chrono::Utc::now());
            tool_result(&compact_json(json!({
                "conversation_id": conversation_id,
                "messages": messages,
                "note": note,
                "coverage_warnings": coverage_warnings,
            })))
        }
        Err(error) => tool_error(&error),
    }
}

fn value_after(args: &[String], flag: &str) -> Option<String> {
    args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).cloned()
}
