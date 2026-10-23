//! Checks every store must pass, written against the traits so the SQLite
//! store and the server mode's Postgres store run the same assertions.
//! Each check expects an empty store and leaves its own rows behind.

use super::{add_memory_to, content_hash, update_memory_in, GraphStore, Memory, MemoryScope, MemoryStore, RecallStore};

fn contract_memory(id: &str, content: &str, kind: &str, project: Option<&str>, updated_at: &str) -> Memory {
    Memory {
        id: id.into(),
        content: content.into(),
        kind: kind.into(),
        project: project.map(String::from),
        tags: vec!["contract".into()],
        source: "test".into(),
        importance: 3,
        expires_at: None,
        created_at: "2026-01-01T00:00:00+00:00".into(),
        updated_at: updated_at.into(),
        metadata: Some(serde_json::json!({ "origin": "contract" })),
        last_accessed_at: None,
        access_count: 0,
    }
}

/// Row storage: insert, fetch, hash lookup, listing, rewrite, removal.
pub fn assert_memory_store_contract(store: &dyn MemoryStore) {
    let older = contract_memory("m-old", "Deploys go through cf-orchestrator", "decision", Some("acme"), "2026-01-02T00:00:00+00:00");
    let newer = contract_memory("m-new", "Invoices are numbered per year", "fact", Some("acme"), "2026-01-03T00:00:00+00:00");
    let global = contract_memory("m-global", "Answers in French", "preference", None, "2026-01-04T00:00:00+00:00");
    for memory in [&older, &newer, &global] {
        store.insert_memory(memory, &content_hash(&memory.content)).expect("insert");
    }

    let fetched = store.fetch_memory("m-old").expect("fetch").expect("stored");
    assert_eq!(fetched.content, older.content);
    assert_eq!(fetched.kind, "decision");
    assert_eq!(fetched.project.as_deref(), Some("acme"));
    assert_eq!(fetched.tags, older.tags);
    assert_eq!(fetched.metadata, older.metadata);
    assert_eq!(store.stored_content("m-old").expect("content"), older.content);
    assert!(store.fetch_memory("missing").expect("fetch").is_none());

    let hash = content_hash(&older.content);
    assert_eq!(store.find_by_content_hash(&hash, Some("acme")).map(|m| m.id), Some("m-old".into()));
    assert!(store.find_by_content_hash(&hash, None).is_none(), "hash lookup stays inside the project");

    let recent: Vec<String> = store.recent_memories(Some("acme"), 10).expect("recent").into_iter().map(|m| m.id).collect();
    assert_eq!(recent, vec!["m-new", "m-old"]);
    let unscoped: Vec<String> = store.recent_memories(None, 10).expect("recent").into_iter().map(|m| m.id).collect();
    assert_eq!(unscoped, vec!["m-global"]);

    let (all, total) = store.list_memory_rows(None, None, None, 10, 0).expect("list");
    assert_eq!(total, 3);
    assert_eq!(all.iter().map(|m| m.id.as_str()).collect::<Vec<_>>(), vec!["m-global", "m-new", "m-old"]);
    let (page, total) = store.list_memory_rows(Some("acme"), None, None, 1, 1).expect("list");
    assert_eq!((total, page.len(), page[0].id.as_str()), (2, 1, "m-old"));
    let (facts, _) = store.list_memory_rows(Some("acme"), Some("fact"), None, 10, 0).expect("list");
    assert_eq!(facts.iter().map(|m| m.id.as_str()).collect::<Vec<_>>(), vec!["m-new"]);
    let (others, total) = store.list_memory_rows(None, None, Some("fact"), 10, 0).expect("list");
    assert_eq!((total, others.len()), (2, 2));

    let mut edited = older.clone();
    edited.content = "Deploys go through cf-orchestrator only".into();
    edited.tags = vec!["contract".into(), "deploy".into()];
    edited.importance = 5;
    edited.updated_at = "2026-01-05T00:00:00+00:00".into();
    store.write_memory(&edited, &content_hash(&edited.content), true).expect("write");
    let reread = store.fetch_memory("m-old").expect("fetch").expect("stored");
    assert_eq!((reread.content.as_str(), reread.importance), (edited.content.as_str(), 5));
    assert_eq!(reread.tags, edited.tags);
    assert_eq!(reread.updated_at, edited.updated_at);
    assert_eq!(store.find_by_content_hash(&content_hash(&edited.content), Some("acme")).map(|m| m.id), Some("m-old".into()));
    assert!(store.find_by_content_hash(&hash, Some("acme")).is_none(), "the old hash is replaced");
    assert_eq!(store.recent_memories(Some("acme"), 1).expect("recent")[0].id, "m-old");

    assert!(store.remove_memory("m-old").expect("remove"));
    assert!(!store.remove_memory("m-old").expect("remove again"));
    assert!(store.fetch_memory("m-old").expect("fetch").is_none());
    assert_eq!(store.list_memory_rows(None, None, None, 10, 0).expect("list").1, 2);
}

/// What search and recall need from any store, on memories it stored itself.
pub fn assert_recall_store_contract<S: MemoryStore + RecallStore>(store: &S) {
    let mut pinned = contract_memory("r-pinned", "Release notes are written in English", "convention", Some("acme"), "2026-02-01T00:00:00+00:00");
    pinned.tags = vec!["pinned".into()];
    pinned.importance = 2;
    let mut critical = contract_memory("r-critical", "Never deploy on Friday evenings", "decision", Some("acme"), "2026-02-02T00:00:00+00:00");
    critical.importance = 5;
    let other = contract_memory("r-other", "Staging database lives in Frankfurt", "fact", Some("globex"), "2026-02-03T00:00:00+00:00");
    for memory in [&pinned, &critical, &other] {
        store.insert_memory(memory, &content_hash(&memory.content)).expect("insert");
    }

    let full_text = |query: &str, project: Option<&str>, kind: Option<&str>| -> Vec<String> {
        store.full_text_candidates(query, project, kind, 10).expect("full text").into_iter().map(|m| m.id).collect()
    };
    assert_eq!(full_text("deploy", Some("acme"), None), vec!["r-critical"]);
    assert!(full_text("deploy", Some("globex"), None).is_empty(), "full text stays inside the project");
    assert!(full_text("deploy", Some("acme"), Some("fact")).is_empty(), "kind filter applies");
    assert_eq!(full_text("\"depl\"* \"frid\"*", Some("acme"), None), vec!["r-critical"], "prefix terms, all required");
    assert!(full_text("\"depl\"* \"frankfurt\"*", None, None).is_empty(), "prefix terms, all required");
    assert_eq!(full_text("\"friday evenings\"", None, None), vec!["r-critical"], "phrase");
    assert!(full_text("\"evenings friday\"", None, None).is_empty(), "phrase keeps word order");
    assert_eq!(full_text("NEAR(\"never\" \"evenings\", 8)", None, None), vec!["r-critical"], "near");

    let query = vec![0.0f32; crate::embedding::vector_dim()];
    let mut lane: Vec<(String, Option<f32>)> = store
        .vector_candidates(&query, Some("acme"), None, None)
        .expect("vector")
        .into_iter()
        .map(|(m, score)| (m.id, score))
        .collect();
    lane.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(lane, vec![("r-critical".to_string(), None), ("r-pinned".to_string(), None)], "embeddings not computed yet: no score");
    let only: Vec<String> = store
        .vector_candidates(&query, None, None, Some(&["r-other".to_string()]))
        .expect("vector")
        .into_iter()
        .map(|(m, _)| m.id)
        .collect();
    assert_eq!(only, vec!["r-other"]);

    assert_eq!(store.important_memories(4, 10).expect("important").iter().map(|m| m.id.as_str()).collect::<Vec<_>>(), vec!["r-critical"]);
    assert_eq!(store.pinned_memories().expect("pinned").iter().map(|m| m.id.as_str()).collect::<Vec<_>>(), vec!["r-pinned"]);
    assert_eq!(store.memory_count(None), 3);
    assert_eq!(store.memory_count(Some("fact")), 1);
    assert!(store.link_relations(None).is_empty());
    let reached = store.graph_neighbours(&["r-critical".to_string()], 1).expect("graph");
    assert_eq!(reached.into_iter().collect::<Vec<_>>(), vec!["r-critical"], "no link: only the root");
    assert!(store.triple_counts(&[&"r-critical".to_string()]).is_empty());
}

/// The write path on any store: entities and dates extracted, links to
/// memories sharing an entity (never across projects for a file), near-
/// duplicates merged, entities replaced on an edit, projects registered.
pub fn assert_graph_store_contract<S: MemoryStore + GraphStore + RecallStore>(store: &S) {
    let add = |id: &str, content: &str, project: &str, tags: &[&str], importance: i32| {
        let tags: Vec<String> = tags.iter().map(|tag| tag.to_string()).collect();
        add_memory_to(store, Some(id), content, "fact", Some(project), &tags, "test", importance, None, None, &MemoryScope::default())
            .expect("add")
    };
    let neighbours = |id: &str| -> Vec<String> {
        let mut reached: Vec<String> = store.graph_neighbours(&[id.to_string()], 1).expect("graph").into_iter().collect();
        reached.sort();
        reached
    };
    let dates = |id: &str| store.memory_date_values(&[&id.to_string()]).remove(id).unwrap_or_default();

    let first = add("g-one", "Ledger rounding lives in notes/ledger/rounding.md since 2026-03-14", "Acme", &["ledger"], 3);
    assert!(!first.merged && first.content_changed);
    assert_eq!(first.memory.project.as_deref(), Some("acme"), "project name canonical");
    add("g-two", "Rounding rules were rewritten in notes/ledger/rounding.md", "acme", &[], 3);
    add("g-other", "Globex copied notes/ledger/rounding.md too", "globex", &[], 3);
    assert_eq!(neighbours("g-one"), vec!["g-one", "g-two"], "linked within the project, not across");
    assert_eq!(neighbours("g-two"), vec!["g-one", "g-two"], "links go both ways");
    assert_eq!(dates("g-one"), vec!["2026-03-14".to_string()]);
    assert_eq!(store.project_count(), 2);

    let again = add("g-dup", "Ledger rounding lives in notes/ledger/rounding.md since 2026-03-14", "acme", &["bug"], 5);
    assert!(again.merged && !again.content_changed, "a duplicate is merged, its content unchanged");
    assert_eq!(again.memory.id, "g-one");
    assert_eq!((again.memory.tags.clone(), again.memory.importance), (vec!["ledger".to_string(), "bug".to_string()], 5));
    assert_eq!(store.memory_count(None), 3);
    assert_eq!(neighbours("g-one"), vec!["g-one", "g-two"], "a merge keeps the links");

    let edited = update_memory_in(store, "g-one", Some("Ledger rounding lives in notes/ledger/rounding.md"), None, None, None, None, None)
        .expect("update")
        .expect("exists");
    assert!(edited.content_changed && !edited.merged);
    assert!(dates("g-one").is_empty(), "an edit replaces the entities");
    assert_eq!(neighbours("g-one"), vec!["g-one", "g-two"], "and rebuilds the links");
    assert!(update_memory_in(store, "missing", Some("text"), None, None, None, None, None).expect("update").is_none());
}
