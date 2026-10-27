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
