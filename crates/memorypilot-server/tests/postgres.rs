//! Against a real Postgres with pgvector (`scripts/test-postgres.sh` starts
//! one and runs these). Every test works in tenants it creates, so they run
//! in parallel on a shared database.

use memorypilot_core::db::store_contract::{assert_memory_store_contract, assert_recall_store_contract};
use memorypilot_core::db::{Memory, MemoryStore, RecallStore};
use memorypilot_server::{PgAdmin, PgStore, Scope};
use uuid::Uuid;

const NEEDS_POSTGRES: &str = "needs Postgres with pgvector: scripts/test-postgres.sh";

fn url() -> String {
    std::env::var("MEMORYPILOT_TEST_PG_URL").unwrap_or_else(|_| panic!("MEMORYPILOT_TEST_PG_URL is not set ({})", NEEDS_POSTGRES))
}

fn admin() -> PgAdmin {
    let mut admin = PgAdmin::connect(&url()).expect("admin connection");
    admin.migrate().expect("migrations");
    admin
}

/// A new tenant with one space named "app" (the same name in every tenant).
fn new_space(admin: &mut PgAdmin) -> Scope {
    let tenant = admin.create_tenant(&format!("test-{}", Uuid::new_v4())).expect("tenant");
    let space = admin.create_space(tenant, "app").expect("space");
    Scope { tenant, space, subject: None }
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
    store.insert_memory(&memory("a-1", "Invoices are sent by Stripe on the first day"), "hash-a1").expect("insert a-1");
    store.insert_memory(&memory("a-2", "Stripe webhooks land on the billing worker"), "hash-a2").expect("insert a-2");
    store.set_embedding("a-1", &unit_vector(0)).expect("embedding a-1");
    store.set_embedding("a-2", &unit_vector(1)).expect("embedding a-2");
    store.add_link("a-1", "a-2", "relates_to").expect("link");
    for id in ["a-1", "a-2"] {
        store.add_entity(id, "tech", "Stripe").expect("entity");
    }
    store.add_entity("a-1", "tech", "Invoices").expect("entity");
    store.add_entity("a-1", "date", "2026-10-01").expect("date");
    store.add_triple("billing", "uses", "stripe", Some("a-1")).expect("triple");
    store.set_project_path("acme", "/srv/acme").expect("project");
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
    scope.subject = Some(admin.create_subject(scope.tenant, scope.space, "user-1").expect("subject"));
    assert_memory_store_contract(&store(scope));
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
    assert_eq!(store.memory_vectors(&[&a1]).get("a-1"), Some(&unit_vector(0)));

    assert_eq!(store.link_relations(Some(&[&a2])), vec![("a-2".to_string(), "relates_to".to_string())]);
    let reached = store.graph_neighbours(std::slice::from_ref(&a1), 1).expect("graph");
    assert_eq!(reached.len(), 2, "a-1 and its linked a-2: {:?}", reached);
    let pairs = store.adjacency_pairs(&[a1.clone(), a2.clone()]);
    assert!(pairs.contains(&(a1.clone(), a2.clone())), "link");
    assert!(pairs.contains(&(a2.clone(), a1.clone())), "shared Stripe entity");

    let mut related = store.related_terms("billing");
    related.extend(store.related_terms("invoices"));
    assert!(related.contains(&"stripe".to_string()), "triple object: {:?}", related);
    assert!(related.contains(&"Stripe".to_string()), "co-occurring entity: {:?}", related);
    assert_eq!(store.triple_counts(&[&a1]).get("a-1"), Some(&(1, 0)));
    assert_eq!(store.memory_date_values(&[&a1]).get("a-1"), Some(&vec!["2026-10-01".to_string()]));
    assert_eq!(store.project_path("acme").as_deref(), Some("/srv/acme"));
    assert_eq!(store.project_entities("acme", &["tech"], 10), vec!["Invoices", "Stripe"]);
    assert_eq!(store.recent_project_contents("acme", 7, 10).len(), 2);
    assert_eq!(store.project_count(), 1);

    let hits = ids(store.full_text_candidates("\"stripe\"* \"webhook\"*", None, None, 10).expect("full text"));
    assert_eq!(hits, vec!["a-2"]);
    store.remove_memory("a-2").expect("remove");
    assert!(store.link_relations(None).is_empty(), "links go with their memory");
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
    assert_eq!(intruder.list_memory_rows(None, None, None, 10, 0).expect("list").1, 0);
    assert!(intruder.find_by_content_hash("hash-a1", Some("acme")).is_none());
    assert!(intruder.recent_memories(Some("acme"), 10).expect("recent").is_empty());
    assert!(intruder.full_text_candidates("stripe", None, None, 10).expect("full text").is_empty());
    assert!(intruder.vector_candidates(&unit_vector(0), None, None, None).expect("vector").is_empty());
    assert!(intruder.vector_candidates(&unit_vector(0), None, None, Some(std::slice::from_ref(&a1))).expect("vector").is_empty());
    assert!(intruder.memory_vectors(&[&a1]).is_empty());
    assert!(intruder.memory_date_values(&[&a1]).is_empty());
    assert!(intruder.link_relations(None).is_empty());
    assert!(intruder.link_relations(Some(&[&a2])).is_empty());
    assert!(intruder.related_terms("billing").is_empty());
    assert!(intruder.related_terms("invoices").is_empty());
    assert!(intruder.adjacency_pairs(&[a1.clone(), a2.clone()]).is_empty());
    assert!(intruder.triple_counts(&[&a1]).is_empty());
    assert_eq!(intruder.graph_neighbours(std::slice::from_ref(&a1), 2).expect("graph").len(), 1, "only the root it asked for");
    assert!(intruder.project_path("acme").is_none());
    assert!(intruder.project_entities("acme", &["tech"], 10).is_empty());
    assert!(intruder.recent_project_contents("acme", 7, 10).is_empty());
    assert!(intruder.important_memories(1, 10).expect("important").is_empty());
    assert!(intruder.pinned_memories().expect("pinned").is_empty());
    assert_eq!((intruder.memory_count(None), intruder.project_count()), (0, 0));

    let mut forged = memory("a-1", "Overwritten by another tenant");
    assert!(intruder.write_memory(&forged, "hash-x", true).is_err());
    assert!(!intruder.remove_memory("a-1").expect("remove"));
    assert!(intruder.set_embedding("a-1", &unit_vector(2)).is_err());
    assert!(intruder.add_entity("a-1", "tech", "Leak").is_err());
    assert!(intruder.add_triple("x", "y", "z", Some("a-1")).is_err());

    forged.content = "The intruder's own memory".into();
    intruder.insert_memory(&forged, "hash-x").expect("same id in another tenant");
    assert!(intruder.add_link("a-1", "a-2", "relates_to").is_err(), "a-2 belongs to the other tenant");
    let kept = owner.fetch_memory("a-1").expect("fetch").expect("still there");
    assert_eq!(kept.content, "Invoices are sent by Stripe on the first day");
    assert_eq!(owner.memory_vectors(&[&a1]).get("a-1"), Some(&unit_vector(0)));
    assert_eq!(owner.memory_count(None), 2);
}

#[test]
#[ignore = "needs Postgres with pgvector: scripts/test-postgres.sh"]
fn a_store_cannot_bind_to_another_tenants_space() {
    let mut admin = admin();
    let theirs = new_space(&mut admin);
    let mine = new_space(&mut admin);
    assert!(PgStore::connect(&url(), Scope { tenant: mine.tenant, space: theirs.space, subject: None }).is_err());
    let their_user = admin.create_subject(theirs.tenant, theirs.space, "user-1").expect("subject");
    assert!(PgStore::connect(&url(), Scope { subject: Some(their_user), ..mine }).is_err());
}

#[test]
#[ignore = "needs Postgres with pgvector: scripts/test-postgres.sh"]
fn end_users_see_their_memories_and_the_shared_ones() {
    let mut admin = admin();
    let space = new_space(&mut admin);
    let alice = Scope { subject: Some(admin.create_subject(space.tenant, space.space, "alice").expect("alice")), ..space };
    let bob = Scope { subject: Some(admin.create_subject(space.tenant, space.space, "bob").expect("bob")), ..space };
    let (shared, alice, bob) = (store(space), store(alice), store(bob));

    shared.insert_memory(&memory("shared-1", "Support answers within a day"), "hash-s").expect("shared");
    alice.insert_memory(&memory("alice-1", "Alice prefers invoices in English"), "hash-a").expect("alice");

    assert_eq!(ids(alice.list_memory_rows(None, None, None, 10, 0).expect("list").0).len(), 2);
    assert_eq!(ids(bob.list_memory_rows(None, None, None, 10, 0).expect("list").0), vec!["shared-1"]);
    assert_eq!(ids(shared.list_memory_rows(None, None, None, 10, 0).expect("list").0), vec!["shared-1"]);
    assert!(bob.fetch_memory("alice-1").expect("fetch").is_none());
    assert!(bob.full_text_candidates("invoices", None, None, 10).expect("full text").is_empty());

    let edited = memory("shared-1", "Changed by an end user");
    assert!(alice.write_memory(&edited, "hash-e", true).is_err(), "end users cannot change shared memories");
    assert!(!alice.remove_memory("shared-1").expect("remove"));
    assert!(!bob.remove_memory("alice-1").expect("remove"));
    assert!(alice.find_by_content_hash("hash-s", Some("acme")).is_none(), "dedup only against what it can change");
    assert_eq!(shared.fetch_memory("shared-1").expect("fetch").expect("kept").content, "Support answers within a day");
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
    client.batch_execute("SET ROLE memorypilot_app").expect("app role");
    let count = |client: &mut postgres::Client, tenant: Option<Uuid>, table: &str| -> i64 {
        let mut tx = client.transaction().expect("tx");
        if let Some(tenant) = tenant {
            tx.execute("SELECT set_config('app.tenant_id', $1, true)", &[&tenant.to_string()]).expect("tenant");
        }
        let n: i64 = tx.query_one(&format!("SELECT count(*) FROM {}", table), &[]).expect("count").get(0);
        tx.commit().expect("commit");
        n
    };
    for table in ["memories", "memory_links", "memory_entities", "knowledge_triples", "projects", "spaces", "tenants"] {
        assert_eq!(count(&mut client, Some(other.tenant), table), if matches!(table, "spaces" | "tenants") { 1 } else { 0 }, "{} as the other tenant", table);
        assert_eq!(count(&mut client, None, table), 0, "{} without a tenant", table);
    }
    assert!(count(&mut client, Some(owner_scope.tenant), "memories") >= 2);

    let mut tx = client.transaction().expect("tx");
    tx.execute("SELECT set_config('app.tenant_id', $1, true)", &[&other.tenant.to_string()]).expect("tenant");
    let changed = tx.execute("UPDATE memories SET content = 'leak' WHERE id = 'a-1'", &[]).expect("update");
    assert_eq!(changed, 0, "update cannot reach another tenant's row");
    let forged = tx.execute(
        "INSERT INTO projects (tenant_id, space_id, name) VALUES ($1, $2, 'forged')",
        &[&owner_scope.tenant, &owner_scope.space],
    );
    assert!(forged.is_err(), "insert into another tenant is rejected");
}
