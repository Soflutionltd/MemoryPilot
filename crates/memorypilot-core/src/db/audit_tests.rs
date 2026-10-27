//! Regression tests for the defects found in the October 2026 audit.
//! Each test drives the public path the MCP tools use.

use super::{Database, MemoryScope};
use std::path::PathBuf;
use uuid::Uuid;

fn temp_db_path(test_name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("memory-pilot-audit-{test_name}-{}.db", Uuid::new_v4()))
}

fn cleanup(path: &PathBuf) {
    let _ = std::fs::remove_file(path);
    let _ = std::fs::remove_file(format!("{}-wal", path.display()));
    let _ = std::fs::remove_file(format!("{}-shm", path.display()));
}

fn add(db: &Database, content: &str, kind: &str, project: &str) -> (super::Memory, bool) {
    db.add_memory(content, kind, Some(project), &[], "test", 3, None, None, &MemoryScope::default())
        .expect("add memory")
}

/// A limit above the internal pool caps used to panic in `clamp` (min > max),
/// which aborts the whole process in release builds.
#[test]
fn search_accepts_large_limits_without_panicking() {
    let path = temp_db_path("large-limit");
    {
        let db = Database::open_at(&path).expect("db");
        for i in 0..5 {
            add(&db, &format!("Deployment note {i} about the ingestion worker"), "note", "proj");
        }
        for limit in [1usize, 24, 25, 31, 50, 100, 10_000] {
            db.search("deployment ingestion worker", limit, Some("proj"), None, None, None)
                .expect("plain search");
            db.search("when did we deploy last week", limit, Some("proj"), None, None, None)
                .expect("temporal search");
        }
        assert!(db.search("deployment", 0, None, None, None, None).expect("zero").is_empty());
    }
    cleanup(&path);
}

const URI: &str = "postgres://alice:synthetic_password@db.example.com/app";

fn file_bytes_contain(path: &PathBuf, needle: &str) -> bool {
    let mut hit = false;
    for file in [path.display().to_string(), format!("{}-wal", path.display())] {
        if let Ok(bytes) = std::fs::read(&file) {
            hit |= bytes.windows(needle.len()).any(|w| w == needle.as_bytes());
        }
    }
    hit
}

/// A connection URI with a password: masked label, no clear text in the
/// table, FTS, entities, project brain or database files.
#[test]
fn credential_uri_never_stored_or_returned_in_clear() {
    let path = temp_db_path("cred-uri");
    {
        let db = Database::open_at(&path).expect("db");
        let (stored, _) = add(&db, &format!("Prod database: {URI}"), "credential", "proj");
        assert!(!stored.content.contains("synthetic_password"), "label: {}", stored.content);
        assert!(stored.content.contains("postgres://alice:••••@db.example.com/app"), "label: {}", stored.content);

        // The same URI in a note is detected as a credential.
        let (note, _) = add(&db, &format!("Staging uses DATABASE_URL={URI} for now"), "note", "proj");
        assert_eq!(note.kind, "credential");
        assert!(!note.content.contains("synthetic_password"));

        let raw: i64 = db
            .conn
            .query_row("SELECT COUNT(*) FROM memories WHERE content LIKE '%synthetic_password%'", [], |r| r.get(0))
            .unwrap();
        assert_eq!(raw, 0);
        let fts: i64 = db
            .conn
            .query_row("SELECT COUNT(*) FROM memories_fts WHERE memories_fts MATCH 'synthetic_password'", [], |r| r.get(0))
            .unwrap();
        assert_eq!(fts, 0);
        let entities: i64 = db
            .conn
            .query_row("SELECT COUNT(*) FROM memory_entities WHERE entity_value LIKE '%synthetic_password%'", [], |r| r.get(0))
            .unwrap();
        assert_eq!(entities, 0);
        let brain = db.get_project_brain("proj", Some(1500), false).expect("brain");
        assert!(!brain.to_string().contains("synthetic_password"), "brain leaked: {brain}");
        let got = db.get_memory(&stored.id).unwrap().unwrap();
        assert!(!got.content.contains("synthetic_password"));
        // Still recoverable from the vault.
        assert_eq!(db.credential_clear_text(&stored.id).unwrap().as_deref(), Some(format!("Prod database: {URI}").as_str()));
    }
    assert!(!file_bytes_contain(&path, "synthetic_password"), "secret found in the database files");
    cleanup(&path);
}

/// Databases written by older versions: the startup migration re-masks
/// leaked labels, reclassifies notes holding a URI password and rebuilds
/// entities from the masked text.
#[test]
fn startup_migration_cleans_credentials_left_by_older_versions() {
    let path = temp_db_path("cred-migration");
    {
        let db = Database::open_at(&path).expect("db");
        let now = chrono::Utc::now().to_rfc3339();
        for (id, kind, content) in [
            ("legacy-cred", "credential", format!("🔒 Prod database: {URI}")),
            ("legacy-note", "note", format!("Staging DATABASE_URL={URI}")),
        ] {
            db.conn
                .execute(
                    "INSERT INTO memories (id,content,kind,project,tags,source,importance,expires_at,metadata,embedding,content_hash,created_at,updated_at,access_count)
                     VALUES (?1,?2,?3,'proj','[]','test',3,NULL,NULL,NULL,'h',?4,?4,0)",
                    rusqlite::params![id, content, kind, now],
                )
                .unwrap();
            let rowid = db.conn.last_insert_rowid();
            db.conn
                .execute(
                    "INSERT INTO memories_fts (rowid,content,tags,kind,project) VALUES (?1,?2,'[]',?3,'proj')",
                    rusqlite::params![rowid, content, kind],
                )
                .unwrap();
            db.conn
                .execute(
                    "INSERT INTO memory_entities (memory_id, entity_kind, entity_value) VALUES (?1,'file',?2)",
                    rusqlite::params![id, URI],
                )
                .unwrap();
        }
        db.conn.execute("DELETE FROM config WHERE key='credential_mask_version'", []).unwrap();
    }
    {
        let db = Database::open_at(&path).expect("reopen");
        for sql in [
            "SELECT COUNT(*) FROM memories WHERE content LIKE '%synthetic_password%'",
            "SELECT COUNT(*) FROM memory_entities WHERE entity_value LIKE '%synthetic_password%'",
            "SELECT COUNT(*) FROM memories_fts WHERE memories_fts MATCH 'synthetic_password'",
        ] {
            let n: i64 = db.conn.query_row(sql, [], |r| r.get(0)).unwrap();
            assert_eq!(n, 0, "{sql}");
        }
        let kind: String = db.conn.query_row("SELECT kind FROM memories WHERE id='legacy-note'", [], |r| r.get(0)).unwrap();
        assert_eq!(kind, "credential");
    }
    cleanup(&path);
}

/// A same-length correction of a near-duplicate used to be dropped, the old
/// figure kept and its date refreshed.
#[test]
fn dedup_keeps_the_correction_and_the_previous_version() {
    let path = temp_db_path("dedup-correction");
    {
        let db = Database::open_at(&path).expect("db");
        let old = "Le délai de règlement contractuel applicable à tous les clients de notre entreprise est fixé à 30 jours.";
        let new = "Le délai de règlement contractuel applicable à tous les clients de notre entreprise est fixé à 60 jours.";
        let (first, _) = add(&db, old, "fact", "proj");
        let (second, merged) = add(&db, new, "fact", "proj");
        assert!(merged, "still one memory");
        assert_eq!(second.id, first.id);
        let stored = db.get_memory(&first.id).unwrap().unwrap();
        assert!(stored.content.contains("60 jours"), "{}", stored.content);
        let versions = stored.metadata.as_ref().and_then(|m| m.get("previous_versions")).and_then(|v| v.as_array()).cloned().unwrap_or_default();
        assert_eq!(versions.len(), 1);
        assert!(versions[0]["content"].as_str().unwrap().contains("30 jours"));

        // A repeat that adds nothing keeps the stored text.
        let (_, merged) = add(&db, "Le délai de règlement contractuel applicable à tous les clients de notre entreprise est fixé à 60 jours", "fact", "proj");
        assert!(merged);
        let stored = db.get_memory(&first.id).unwrap().unwrap();
        assert!(stored.content.ends_with("60 jours."));
        let versions = stored.metadata.as_ref().and_then(|m| m.get("previous_versions")).and_then(|v| v.as_array()).map(|v| v.len()).unwrap_or(0);
        assert_eq!(versions, 1, "no version added for a repeat");
    }
    cleanup(&path);
}
