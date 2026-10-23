//! Against a real Postgres with pgvector (`scripts/test-postgres.sh` starts
//! one and runs these). Every test works in tenants it creates, so they run
//! in parallel on a shared database.

use memorypilot_core::db::store_contract::{
    assert_graph_store_contract, assert_memory_store_contract, assert_recall_store_contract,
};
use memorypilot_core::db::{
    add_memory_to, fr_scenarios, hybrid_search, score_fr_queries, update_memory_in, Memory,
    MemoryScope, MemoryStore, RecallStore, WriteOutcome,
};
use memorypilot_server::{PgAdmin, PgStore, Scope};
use uuid::Uuid;

const NEEDS_POSTGRES: &str = "needs Postgres with pgvector: scripts/test-postgres.sh";

/// Measured 100% through the full write path (SQLite: 100%); one near-tied query may slip.
const MIN_FR_RECALL_AT_10: f64 = 99.0;

fn url() -> String {
    std::env::var("MEMORYPILOT_TEST_PG_URL")
        .unwrap_or_else(|_| panic!("MEMORYPILOT_TEST_PG_URL is not set ({})", NEEDS_POSTGRES))
}

fn admin() -> PgAdmin {
    let mut admin = PgAdmin::connect(&url()).expect("admin connection");
    admin.migrate().expect("migrations");
    admin
}

/// A new tenant with one space named "app" (the same name in every tenant).
fn new_space(admin: &mut PgAdmin) -> Scope {
    let tenant = admin
        .create_tenant(&format!("test-{}", Uuid::new_v4()))
        .expect("tenant");
    let space = admin.create_space(tenant, "app").expect("space");
    Scope {
        tenant,
        space,
        subject: None,
    }
}

fn store(scope: Scope) -> PgStore {
    PgStore::connect(&url(), scope).expect("store connection")
}

fn memory(id: &str, content: &str) -> Memory {
    Memory {
        id: id.into(),
        content: content.into(),
        kind: "fact".into(),
        project: Some("acme".into()),
        tags: vec!["pinned".into()],
        source: "test".into(),
        importance: 5,
        expires_at: None,
        created_at: chrono::Utc::now().to_rfc3339(),
        updated_at: chrono::Utc::now().to_rfc3339(),
        metadata: None,
        last_accessed_at: None,
        access_count: 0,
    }
}

fn unit_vector(axis: usize) -> Vec<f32> {
    let mut vector = vec![0.0f32; memorypilot_core::embedding::vector_dim()];
    vector[axis] = 1.0;
    vector
}

fn ids(memories: Vec<Memory>) -> Vec<String> {
    memories.into_iter().map(|m| m.id).collect()
}

/// One memory with everything search and recall read: embedding, link,
/// entities, triple, project path.
fn populate(store: &PgStore) {
    store
        .insert_memory(
            &memory("a-1", "Invoices are sent by Stripe on the first day"),
            "hash-a1",
        )
        .expect("insert a-1");
    store
        .insert_memory(
            &memory("a-2", "Stripe webhooks land on the billing worker"),
            "hash-a2",
        )
        .expect("insert a-2");
    store
        .set_embedding("a-1", &unit_vector(0))
        .expect("embedding a-1");
    store
        .set_embedding("a-2", &unit_vector(1))
        .expect("embedding a-2");
    store.add_link("a-1", "a-2", "relates_to").expect("link");
    for id in ["a-1", "a-2"] {
        store.add_entity(id, "tech", "Stripe").expect("entity");
    }
    store.add_entity("a-1", "tech", "Invoices").expect("entity");
    store.add_entity("a-1", "date", "2026-10-01").expect("date");
    store
        .add_triple("billing", "uses", "stripe", Some("a-1"))
        .expect("triple");
    store
        .set_project_path("acme", "/srv/acme")
        .expect("project");
}

#[test]
#[ignore = "needs Postgres with pgvector: scripts/test-postgres.sh"]
fn postgres_store_meets_the_memory_store_contract() {
    let mut admin = admin();
    assert_memory_store_contract(&store(new_space(&mut admin)));
}

#[test]
#[ignore = "needs Postgres with pgvector: scripts/test-postgres.sh"]
fn postgres_store_meets_the_memory_store_contract_as_an_end_user() {
    let mut admin = admin();
    let mut scope = new_space(&mut admin);
    scope.subject = Some(
        admin
            .create_subject(scope.tenant, scope.space, "user-1")
            .expect("subject"),
    );
    assert_memory_store_contract(&store(scope));
}

fn end_user(admin: &mut PgAdmin, space: Scope, name: &str) -> Scope {
    Scope {
        subject: Some(
            admin
                .create_subject(space.tenant, space.space, name)
                .expect("subject"),
        ),
        ..space
    }
}

fn add(store: &PgStore, id: &str, content: &str) -> WriteOutcome {
    add_memory_to(
        store,
        Some(id),
        content,
        "fact",
        Some("acme"),
        &[],
        "test",
        3,
        None,
        None,
        &MemoryScope::default(),
    )
    .expect("add")
}

fn neighbours(store: &PgStore, id: &str) -> Vec<String> {
    let mut reached: Vec<String> = store
        .graph_neighbours(&[id.to_string()], 1)
        .expect("graph")
        .into_iter()
        .collect();
    reached.sort();
    reached
}

#[test]
#[ignore = "needs Postgres with pgvector: scripts/test-postgres.sh"]
fn postgres_store_meets_the_graph_store_contract() {
    let mut admin = admin();
    assert_graph_store_contract(&store(new_space(&mut admin)));
}

#[test]
#[ignore = "needs Postgres with pgvector: scripts/test-postgres.sh"]
fn postgres_store_meets_the_graph_store_contract_as_an_end_user() {
    let mut admin = admin();
    let space = new_space(&mut admin);
    assert_graph_store_contract(&store(end_user(&mut admin, space, "user-1")));
}

/// End users link their memories to the shared ones, never to each other's;
/// the space rewriting a shared memory keeps the links it cannot see; a
/// duplicate of a shared memory becomes the end user's own.
#[test]
#[ignore = "needs Postgres with pgvector: scripts/test-postgres.sh"]
fn end_users_link_only_to_what_they_see() {
    let mut admin = admin();
    let space = new_space(&mut admin);
    let (alice_scope, bob_scope) = (
        end_user(&mut admin, space, "alice"),
        end_user(&mut admin, space, "bob"),
    );
    let (shared, alice, bob) = (store(space), store(alice_scope), store(bob_scope));

    add(
        &shared,
        "s-rules",
        "Ledger rounding lives in notes/ledger/rounding.md",
    );
    add(&alice, "a-note", "Alice reviewed notes/ledger/rounding.md");
    add(&bob, "b-note", "Bob annotated notes/ledger/rounding.md");
    assert_eq!(neighbours(&alice, "a-note"), vec!["a-note", "s-rules"]);
    assert_eq!(neighbours(&bob, "b-note"), vec!["b-note", "s-rules"]);
    assert_eq!(
        neighbours(&alice, "s-rules"),
        vec!["a-note", "s-rules"],
        "Bob's memory stays hidden from Alice"
    );
    assert_eq!(
        neighbours(&shared, "s-rules"),
        vec!["s-rules"],
        "the space sees no end user's memory"
    );

    update_memory_in(
        &shared,
        "s-rules",
        Some("Ledger rounding lives in notes/ledger/rounding.md, half up"),
        None,
        None,
        None,
        None,
        None,
    )
    .expect("update")
    .expect("exists");
    assert_eq!(
        neighbours(&alice, "s-rules"),
        vec!["a-note", "s-rules"],
        "the space's rewrite keeps Alice's link"
    );
    assert_eq!(neighbours(&bob, "s-rules"), vec!["b-note", "s-rules"]);

    let copy = add(
        &alice,
        "a-copy",
        "Ledger rounding lives in notes/ledger/rounding.md, half up",
    );
    assert!(
        !copy.merged,
        "an end user never merges into a shared memory"
    );
    assert!(shared.fetch_memory("a-copy").expect("fetch").is_none());
    assert!(
        update_memory_in(
            &alice,
            "s-rules",
            Some("Alice's version"),
            None,
            None,
            None,
            None,
            None
        )
        .is_err(),
        "nor rewrites one"
    );
}

#[test]
#[ignore = "needs Postgres with pgvector: scripts/test-postgres.sh"]
fn a_secret_is_refused_by_the_write_path() {
    let mut admin = admin();
    let store = store(new_space(&mut admin));
    let refused = add_memory_to(
        &store,
        None,
        "Staging password: Zq8!vT3#pLm2",
        "fact",
        None,
        &[],
        "test",
        3,
        None,
        None,
        &MemoryScope::default(),
    );
    assert!(
        refused.is_err(),
        "detected as a credential, refused until the per-tenant key exists"
    );
    assert_eq!(store.memory_count(None), 0);
}

#[test]
#[ignore = "needs Postgres with pgvector: scripts/test-postgres.sh"]
fn postgres_store_meets_the_recall_store_contract() {
    let mut admin = admin();
    assert_recall_store_contract(&store(new_space(&mut admin)));
}

#[test]
#[ignore = "needs Postgres with pgvector: scripts/test-postgres.sh"]
fn search_and_recall_read_vectors_links_entities_and_triples() {
    let mut admin = admin();
    let store = store(new_space(&mut admin));
    populate(&store);

    let mut lane: Vec<(String, Option<f32>)> = store
        .vector_candidates(&unit_vector(0), Some("acme"), None, None)
        .expect("vector")
        .into_iter()
        .map(|(m, score)| (m.id, score))
        .collect();
    lane.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(lane[0].0, "a-1");
    assert!((lane[0].1.expect("a-1 scored") - 1.0).abs() < 1e-6);
    assert!(lane[1].1.expect("a-2 scored").abs() < 1e-6);
    let (a1, a2) = ("a-1".to_string(), "a-2".to_string());
    assert_eq!(
        store.memory_vectors(&[&a1]).get("a-1"),
        Some(&unit_vector(0))
    );

    assert_eq!(
        store.link_relations(Some(&[&a2])),
        vec![("a-2".to_string(), "relates_to".to_string())]
    );
    let reached = store
        .graph_neighbours(std::slice::from_ref(&a1), 1)
        .expect("graph");
    assert_eq!(reached.len(), 2, "a-1 and its linked a-2: {:?}", reached);
    let pairs = store.adjacency_pairs(&[a1.clone(), a2.clone()]);
    assert!(pairs.contains(&(a1.clone(), a2.clone())), "link");
    assert!(
        pairs.contains(&(a2.clone(), a1.clone())),
        "shared Stripe entity"
    );

    let mut related = store.related_terms("billing");
    related.extend(store.related_terms("invoices"));
    assert!(
        related.contains(&"stripe".to_string()),
        "triple object: {:?}",
        related
    );
    assert!(
        related.contains(&"Stripe".to_string()),
        "co-occurring entity: {:?}",
        related
    );
    assert_eq!(store.triple_counts(&[&a1]).get("a-1"), Some(&(1, 0)));
    assert_eq!(
        store.memory_date_values(&[&a1]).get("a-1"),
        Some(&vec!["2026-10-01".to_string()])
    );
    assert_eq!(store.project_path("acme").as_deref(), Some("/srv/acme"));
    assert_eq!(
        store.project_entities("acme", &["tech"], 10),
        vec!["Invoices", "Stripe"]
    );
    assert_eq!(store.recent_project_contents("acme", 7, 10).len(), 2);
    assert_eq!(store.project_count(), 1);

    let hits = ids(store
        .full_text_candidates("\"stripe\"* \"webhook\"*", None, None, 10)
        .expect("full text"));
    assert_eq!(hits, vec!["a-2"]);
    store.remove_memory("a-2").expect("remove");
    assert!(
        store.link_relations(None).is_empty(),
        "links go with their memory"
    );
}

fn embed_everything(store: &PgStore) {
    while store.embed_pending(64).expect("embed") > 0 {}
}

fn search_ids(store: &PgStore, query: &str, limit: usize, project: Option<&str>) -> Vec<String> {
    hybrid_search(
        store,
        chrono::Utc::now(),
        query,
        limit,
        project,
        None,
        None,
        None,
    )
    .expect("search")
    .0
    .into_iter()
    .map(|result| result.memory.id)
    .collect()
}

#[test]
#[ignore = "needs Postgres with pgvector: scripts/test-postgres.sh"]
fn the_search_pipeline_runs_on_postgres() {
    let mut admin = admin();
    let store = store(new_space(&mut admin));
    let mut expired = memory(
        "expired",
        "Stripe webhooks used to be verified by the legacy proxy",
    );
    expired.expires_at = Some("2020-01-01T00:00:00+00:00".into());
    for (id, content) in [
        (
            "webhook",
            "Stripe webhook signatures are checked against the raw request body",
        ),
        (
            "deploy",
            "Deploys go through cf-orchestrator, never wrangler directly",
        ),
        ("french", "Les factures sont numérotées par année civile"),
    ] {
        store
            .insert_memory(&memory(id, content), id)
            .expect("insert");
    }
    store.insert_memory(&expired, "expired").expect("insert");
    embed_everything(&store);
    assert!(
        store
            .memory_vectors(&[&"webhook".to_string()])
            .contains_key("webhook"),
        "embedded"
    );

    assert_eq!(
        search_ids(
            &store,
            "how do we verify Stripe webhook signatures",
            3,
            None
        )
        .first()
        .map(String::as_str),
        Some("webhook")
    );
    assert_eq!(
        search_ids(&store, "numérotation des factures", 3, None)
            .first()
            .map(String::as_str),
        Some("french")
    );
    assert!(
        !search_ids(&store, "legacy proxy webhooks", 5, None).contains(&"expired".to_string()),
        "expired memories never surface"
    );
    assert!(
        store.fetch_memory("expired").expect("fetch").is_none(),
        "and are dropped"
    );
    assert!(
        store
            .fetch_memory("webhook")
            .expect("fetch")
            .expect("kept")
            .access_count
            >= 1,
        "access counted"
    );

    let mut edited = memory("deploy", "Deploys go through cf-orchestrator only");
    edited.updated_at = chrono::Utc::now().to_rfc3339();
    store
        .write_memory(&edited, "deploy-v2", true)
        .expect("edit");
    assert!(
        store.memory_vectors(&[&"deploy".to_string()]).is_empty(),
        "an edit drops the stale vector"
    );
    assert_eq!(store.embed_pending(64).expect("embed"), 1);
}

/// The French benchmark replayed on Postgres through the same write path
/// as SQLite (dedup, entities, links).
#[test]
#[ignore = "needs Postgres with pgvector: scripts/test-postgres.sh"]
fn the_french_benchmark_runs_on_postgres() {
    let mut admin = admin();
    let store = store(new_space(&mut admin));
    let mut id_map = std::collections::HashMap::new();
    for scenario in fr_scenarios() {
        let tags: Vec<String> = scenario.tags.iter().map(|tag| tag.to_string()).collect();
        let outcome = add_memory_to(
            &store,
            Some(&format!("fr-bench:{}", scenario.id)),
            scenario.content,
            scenario.kind,
            Some(scenario.project),
            &tags,
            "bench-fr",
            3,
            None,
            None,
            &MemoryScope::default(),
        )
        .expect("add");
        id_map.insert(scenario.id.to_string(), outcome.memory.id);
    }
    embed_everything(&store);
    memorypilot_core::reranking::warmup_cross_reranker();
    let report = score_fr_queries(&id_map, |query, project| {
        Ok(hybrid_search(
            &store,
            chrono::Utc::now(),
            query,
            10,
            project,
            None,
            None,
            None,
        )?
        .0
        .into_iter()
        .map(|result| result.memory.id)
        .collect())
    })
    .expect("benchmark");
    println!("postgres fr benchmark: {}", report["metrics"]);
    for query in report["per_query"].as_array().into_iter().flatten() {
        if query["rank"] != 1 {
            println!(
                "  rank {} for {} ({})",
                query["rank"], query["query_id"], query["target"]
            );
        }
    }
    let percent = |key: &str| -> f64 {
        report["metrics"][key]
            .as_str()
            .unwrap_or("0")
            .trim_end_matches('%')
            .parse()
            .unwrap_or(0.0)
    };
    assert!(
        percent("recall_at_10") >= MIN_FR_RECALL_AT_10,
        "recall@10 {} below {}",
        percent("recall_at_10"),
        MIN_FR_RECALL_AT_10
    );
}

#[test]
#[ignore = "needs Postgres with pgvector: scripts/test-postgres.sh"]
fn another_tenant_reads_and_changes_nothing() {
    let mut admin = admin();
    let owner = store(new_space(&mut admin));
    populate(&owner);
    let intruder = store(new_space(&mut admin));
    let (a1, a2) = ("a-1".to_string(), "a-2".to_string());

    assert!(intruder.fetch_memory("a-1").expect("fetch").is_none());
    assert!(intruder.stored_content("a-1").is_err());
    assert_eq!(
        intruder
            .list_memory_rows(None, None, None, 10, 0)
            .expect("list")
            .1,
        0
    );
    assert!(intruder
        .find_by_content_hash("hash-a1", Some("acme"))
        .is_none());
    assert!(intruder
        .recent_memories(Some("acme"), 10)
        .expect("recent")
        .is_empty());
    assert!(intruder
        .full_text_candidates("stripe", None, None, 10)
        .expect("full text")
        .is_empty());
    assert!(intruder
        .vector_candidates(&unit_vector(0), None, None, None)
        .expect("vector")
        .is_empty());
    assert!(intruder
        .vector_candidates(&unit_vector(0), None, None, Some(std::slice::from_ref(&a1)))
        .expect("vector")
        .is_empty());
    assert!(intruder.memory_vectors(&[&a1]).is_empty());
    assert!(intruder.memory_date_values(&[&a1]).is_empty());
    assert!(intruder.link_relations(None).is_empty());
    assert!(intruder.link_relations(Some(&[&a2])).is_empty());
    assert!(intruder.related_terms("billing").is_empty());
    assert!(intruder.related_terms("invoices").is_empty());
    assert!(intruder
        .adjacency_pairs(&[a1.clone(), a2.clone()])
        .is_empty());
    assert!(intruder.triple_counts(&[&a1]).is_empty());
    assert_eq!(
        intruder
            .graph_neighbours(std::slice::from_ref(&a1), 2)
            .expect("graph")
            .len(),
        1,
        "only the root it asked for"
    );
    assert!(intruder.project_path("acme").is_none());
    assert!(intruder.project_entities("acme", &["tech"], 10).is_empty());
    assert!(intruder.recent_project_contents("acme", 7, 10).is_empty());
    assert!(intruder
        .important_memories(1, 10)
        .expect("important")
        .is_empty());
    assert!(intruder.pinned_memories().expect("pinned").is_empty());
    assert_eq!(
        (intruder.memory_count(None), intruder.project_count()),
        (0, 0)
    );

    let mut forged = memory("a-1", "Overwritten by another tenant");
    assert!(intruder.write_memory(&forged, "hash-x", true).is_err());
    assert!(!intruder.remove_memory("a-1").expect("remove"));
    assert!(intruder.set_embedding("a-1", &unit_vector(2)).is_err());
    assert!(intruder.add_entity("a-1", "tech", "Leak").is_err());
    assert!(intruder.add_triple("x", "y", "z", Some("a-1")).is_err());

    forged.content = "The intruder's own memory".into();
    intruder
        .insert_memory(&forged, "hash-x")
        .expect("same id in another tenant");
    assert!(
        intruder.add_link("a-1", "a-2", "relates_to").is_err(),
        "a-2 belongs to the other tenant"
    );
    let kept = owner
        .fetch_memory("a-1")
        .expect("fetch")
        .expect("still there");
    assert_eq!(kept.content, "Invoices are sent by Stripe on the first day");
    assert_eq!(
        owner.memory_vectors(&[&a1]).get("a-1"),
        Some(&unit_vector(0))
    );
    assert_eq!(owner.memory_count(None), 2);
}

#[test]
#[ignore = "needs Postgres with pgvector: scripts/test-postgres.sh"]
fn a_store_cannot_bind_to_another_tenants_space() {
    let mut admin = admin();
    let theirs = new_space(&mut admin);
    let mine = new_space(&mut admin);
    assert!(PgStore::connect(
        &url(),
        Scope {
            tenant: mine.tenant,
            space: theirs.space,
            subject: None
        }
    )
    .is_err());
    let their_user = admin
        .create_subject(theirs.tenant, theirs.space, "user-1")
        .expect("subject");
    assert!(PgStore::connect(
        &url(),
        Scope {
            subject: Some(their_user),
            ..mine
        }
    )
    .is_err());
}

#[test]
#[ignore = "needs Postgres with pgvector: scripts/test-postgres.sh"]
fn end_users_see_their_memories_and_the_shared_ones() {
    let mut admin = admin();
    let space = new_space(&mut admin);
    let alice = Scope {
        subject: Some(
            admin
                .create_subject(space.tenant, space.space, "alice")
                .expect("alice"),
        ),
        ..space
    };
    let bob = Scope {
        subject: Some(
            admin
                .create_subject(space.tenant, space.space, "bob")
                .expect("bob"),
        ),
        ..space
    };
    let (shared, alice, bob) = (store(space), store(alice), store(bob));

    shared
        .insert_memory(
            &memory("shared-1", "Support answers within a day"),
            "hash-s",
        )
        .expect("shared");
    alice
        .insert_memory(
            &memory("alice-1", "Alice prefers invoices in English"),
            "hash-a",
        )
        .expect("alice");

    assert_eq!(
        ids(alice
            .list_memory_rows(None, None, None, 10, 0)
            .expect("list")
            .0)
        .len(),
        2
    );
    assert_eq!(
        ids(bob
            .list_memory_rows(None, None, None, 10, 0)
            .expect("list")
            .0),
        vec!["shared-1"]
    );
    assert_eq!(
        ids(shared
            .list_memory_rows(None, None, None, 10, 0)
            .expect("list")
            .0),
        vec!["shared-1"]
    );
    assert!(bob.fetch_memory("alice-1").expect("fetch").is_none());
    assert!(bob
        .full_text_candidates("invoices", None, None, 10)
        .expect("full text")
        .is_empty());

    let edited = memory("shared-1", "Changed by an end user");
    assert!(
        alice.write_memory(&edited, "hash-e", true).is_err(),
        "end users cannot change shared memories"
    );
    assert!(!alice.remove_memory("shared-1").expect("remove"));
    assert!(!bob.remove_memory("alice-1").expect("remove"));
    assert!(
        alice.find_by_content_hash("hash-s", Some("acme")).is_none(),
        "dedup only against what it can change"
    );
    assert_eq!(
        shared
            .fetch_memory("shared-1")
            .expect("fetch")
            .expect("kept")
            .content,
        "Support answers within a day"
    );
}

#[test]
#[ignore = "needs Postgres with pgvector: scripts/test-postgres.sh"]
fn row_level_security_alone_keeps_tenants_apart() {
    let mut admin = admin();
    let owner_scope = new_space(&mut admin);
    let owner = store(owner_scope);
    populate(&owner);
    let other = new_space(&mut admin);

    let mut client = postgres::Client::connect(&url(), postgres::NoTls).expect("raw connection");
    client
        .batch_execute("SET ROLE memorypilot_app")
        .expect("app role");
    let count = |client: &mut postgres::Client, tenant: Option<Uuid>, table: &str| -> i64 {
        let mut tx = client.transaction().expect("tx");
        if let Some(tenant) = tenant {
            tx.execute(
                "SELECT set_config('app.tenant_id', $1, true)",
                &[&tenant.to_string()],
            )
            .expect("tenant");
        }
        let n: i64 = tx
            .query_one(&format!("SELECT count(*) FROM {}", table), &[])
            .expect("count")
            .get(0);
        tx.commit().expect("commit");
        n
    };
    for table in [
        "memories",
        "memory_links",
        "memory_entities",
        "knowledge_triples",
        "projects",
        "spaces",
        "tenants",
    ] {
        assert_eq!(
            count(&mut client, Some(other.tenant), table),
            if matches!(table, "spaces" | "tenants") {
                1
            } else {
                0
            },
            "{} as the other tenant",
            table
        );
        assert_eq!(
            count(&mut client, None, table),
            0,
            "{} without a tenant",
            table
        );
    }
    assert!(count(&mut client, Some(owner_scope.tenant), "memories") >= 2);

    let mut tx = client.transaction().expect("tx");
    tx.execute(
        "SELECT set_config('app.tenant_id', $1, true)",
        &[&other.tenant.to_string()],
    )
    .expect("tenant");
    let changed = tx
        .execute("UPDATE memories SET content = 'leak' WHERE id = 'a-1'", &[])
        .expect("update");
    assert_eq!(changed, 0, "update cannot reach another tenant's row");
    let forged = tx.execute(
        "INSERT INTO projects (tenant_id, space_id, name) VALUES ($1, $2, 'forged')",
        &[&owner_scope.tenant, &owner_scope.space],
    );
    assert!(forged.is_err(), "insert into another tenant is rejected");
}
