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

fn file_bytes_contain(path: &std::path::Path, needle: &str) -> bool {
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

fn seed_old(db: &Database, id: &str, content: &str, kind: &str, importance: i32, tags: &str) {
    let stale = (chrono::Utc::now() - chrono::Duration::days(60)).to_rfc3339();
    db.conn
        .execute(
            "INSERT INTO memories (id,content,kind,project,tags,source,importance,expires_at,metadata,embedding,content_hash,created_at,updated_at,access_count)
             VALUES (?1,?2,?3,'proj',?4,'test',?5,NULL,NULL,NULL,?6,?7,?7,0)",
            rusqlite::params![id, content, kind, tags, importance, super::content_hash(content), stale],
        )
        .unwrap();
    let rowid = db.conn.last_insert_rowid();
    db.conn
        .execute(
            "INSERT INTO memories_fts (rowid,content,tags,kind,project) VALUES (?1,?2,?3,?4,'proj')",
            rusqlite::params![rowid, content, tags, kind],
        )
        .unwrap();
}

/// Capsules used to delete their originals (pinned ones included) with no
/// way back. Pinned memories are now skipped and every original is archived.
#[test]
fn compaction_skips_pinned_and_archives_every_original() {
    let path = temp_db_path("capsule-archive");
    {
        let db = Database::open_at(&path).expect("db");
        for i in 0..6 {
            seed_old(&db, &format!("n{i}"), &format!("Point {i}. Root cause detail {i}: the cache key ignored the tenant id {i}."), "note", 2, "[]");
        }
        seed_old(&db, "pinned-decision", "We decided to keep Postgres 16. Reason: logical replication.", "decision", 2, "[\"pinned\"]");
        let (capsules, compressed) = db.compact_to_capsules(14, 3).expect("compact");
        assert!(capsules >= 1 && compressed == 6, "capsules {capsules}, compressed {compressed}");
        assert!(db.get_memory("pinned-decision").unwrap().is_some(), "pinned memory compacted");
        let archived = db.list_archived(None, 50).unwrap();
        assert_eq!(archived.len(), 6);
        assert!(archived.iter().all(|a| a["replaced_by"].is_string()));
        // Full text recoverable.
        let restored = db.restore_archived("n3").unwrap().expect("restored");
        assert!(restored.content.contains("Root cause detail 3"));
        assert_eq!(db.list_archived(None, 50).unwrap().len(), 5);
    }
    cleanup(&path);
}

#[test]
fn gc_merge_archives_its_sources() {
    let path = temp_db_path("gc-archive");
    {
        let db = Database::open_at(&path).expect("db");
        for i in 0..10 {
            seed_old(&db, &format!("t{i}"), &format!("Todo {i}: fix the export. Detail {i} explains why the cursor skips lines."), "todo", 1, "[]");
        }
        db.run_gc(&crate::gc::GcConfig::default(), false).expect("gc");
        let remaining: i64 = db.conn.query_row("SELECT COUNT(*) FROM memories WHERE id LIKE 't%'", [], |r| r.get(0)).unwrap();
        let archived = db.list_archived(None, 50).unwrap().len() as i64;
        assert!(archived > 0, "gc merged nothing");
        assert_eq!(remaining + archived, 10, "an original vanished: {remaining} left, {archived} archived");
    }
    cleanup(&path);
}

/// A vector computed for an old text must not be written once the memory
/// was edited while its batch was embedding.
#[test]
fn stale_embedding_is_not_written_for_edited_content() {
    let path = temp_db_path("stale-embedding");
    {
        let db = Database::open_at(&path).expect("db");
        let (mem, _) = add(&db, "The ingestion worker runs every five minutes on the cron host", "fact", "proj");
        db.conn.execute("UPDATE memories SET embedding=NULL WHERE id=?1", rusqlite::params![mem.id]).unwrap();
        // Job taken from the queue with the old text, then the memory is edited.
        let old_text = mem.content.clone();
        db.update_memory_full(&mem.id, Some("The ingestion worker now runs every hour on the queue host"), None, None, None, None, None)
            .unwrap();
        db.conn.execute("UPDATE memories SET embedding=NULL WHERE id=?1", rusqlite::params![mem.id]).unwrap();
        let blob = crate::embedding::vec_to_blob(&[0.5f32; 8]);
        let written = super::embed_worker::store_embedding(&db.conn, &mem.id, &old_text, &blob, &super::content_hash(&old_text));
        assert_eq!(written, 0, "stale vector written");
        let current = db.get_memory(&mem.id).unwrap().unwrap().content;
        let written = super::embed_worker::store_embedding(&db.conn, &mem.id, &current, &blob, &super::content_hash(&current));
        assert_eq!(written, 1);
    }
    cleanup(&path);
}

/// Episodes used to group every project of an hour together, and were never
/// refreshed when a source memory changed or was deleted.
#[test]
fn episodes_stay_within_one_project_and_follow_their_sources() {
    let path = temp_db_path("episodes");
    {
        let db = Database::open_at(&path).expect("db");
        let hour = "2026-09-01T10:15:00+00:00";
        for (id, project, content) in [
            ("a1", "alpha", "Alpha billing export moved to the nightly job"),
            ("a2", "alpha", "Alpha invoices now carry the VAT number"),
            ("b1", "beta", "Beta mobile app switched to the new login screen"),
            ("b2", "beta", "Beta push notifications use the EU endpoint"),
        ] {
            db.conn
                .execute(
                    "INSERT INTO memories (id,content,kind,project,tags,source,importance,expires_at,metadata,embedding,content_hash,created_at,updated_at,access_count)
                     VALUES (?1,?2,'fact',?3,'[]','test',3,NULL,NULL,NULL,'h',?4,?4,0)",
                    rusqlite::params![id, content, project, hour],
                )
                .unwrap();
        }
        db.roll_up_episodes().expect("rollup");
        let episodes = db.list_episodes(None, Some("interval"), 50).unwrap();
        assert_eq!(episodes.len(), 2, "one episode per project");
        for e in &episodes {
            let ids = db.resolve_episode_sources(&e.id);
            let prefix = if e.project.as_deref() == Some("alpha") { 'a' } else { 'b' };
            assert!(ids.iter().all(|id| id.starts_with(prefix)), "mixed projects: {ids:?}");
        }
        // Source edited: the episode is rebuilt from the new text.
        db.conn.execute("UPDATE memories SET content='Alpha billing export moved to the hourly job' WHERE id='a1'", []).unwrap();
        db.roll_up_episodes().expect("rollup 2");
        let alpha = db.list_episodes(Some("alpha"), Some("interval"), 50).unwrap();
        assert_eq!(alpha.len(), 1, "stale episode kept");
        assert!(alpha[0].summary.contains("hourly") || alpha[0].title.contains("hourly") || !alpha[0].summary.contains("nightly"), "summary not refreshed: {}", alpha[0].summary);
        // Source deleted: below the minimum, the episode goes.
        db.delete_memory("b2").unwrap();
        db.roll_up_episodes().expect("rollup 3");
        assert!(db.list_episodes(Some("beta"), Some("interval"), 50).unwrap().is_empty());
    }
    cleanup(&path);
}

/// Exports stopped at 10,000 memories without saying so.
#[test]
fn export_includes_every_memory_past_ten_thousand() {
    let path = temp_db_path("export-all");
    {
        let db = Database::open_at(&path).expect("db");
        let now = chrono::Utc::now().to_rfc3339();
        let tx = db.conn.unchecked_transaction().unwrap();
        for i in 0..10_050 {
            tx.execute(
                "INSERT INTO memories (id,content,kind,project,tags,source,importance,expires_at,metadata,embedding,content_hash,created_at,updated_at,access_count)
                 VALUES (?1,?2,'note','proj','[]','test',3,NULL,NULL,NULL,'h',?3,?3,0)",
                rusqlite::params![format!("m{i:05}"), format!("note {i}"), now],
            )
            .unwrap();
        }
        tx.commit().unwrap();
        let json = db.export_memories(Some("proj"), "json").expect("export");
        let rows: Vec<serde_json::Value> = serde_json::from_str(&json).unwrap();
        assert_eq!(rows.len(), 10_050);
        let ids: std::collections::HashSet<&str> = rows.iter().map(|r| r["id"].as_str().unwrap()).collect();
        assert_eq!(ids.len(), 10_050, "duplicates across pages");
    }
    cleanup(&path);
}
