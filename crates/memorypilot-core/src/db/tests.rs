use super::{Database, MemoryScope, RecallMode};
use rusqlite::params;
use std::path::PathBuf;
use uuid::Uuid;

fn temp_db_path(test_name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("memory-pilot-{test_name}-{}.db", Uuid::new_v4()))
}

fn cleanup_db_files(path: &PathBuf) {
    let _ = std::fs::remove_file(path);
    let _ = std::fs::remove_file(format!("{}-wal", path.display()));
    let _ = std::fs::remove_file(format!("{}-shm", path.display()));
}

/// Regression: auto-compaction used to hold `LAST_COMPACT` across
/// `run_gc` / `compact_to_capsules`, whose merged rows go through
/// `add_memory` → `maybe_auto_compact` → the same `Mutex`. With a
/// corpus over `AUTO_COMPACT_THRESHOLD` the server froze on its first
/// write. The test drives exactly that path and fails if the write
/// does not come back within a generous bound.
#[test]
fn auto_compaction_reenters_add_memory_without_deadlock() {
    let path = temp_db_path("compaction-reentrancy");
    {
        let db = Database::open_at(&path).expect("db");
        let stale = (chrono::Utc::now() - chrono::Duration::days(40)).to_rfc3339();
        for index in 0..crate::gc::AUTO_COMPACT_THRESHOLD {
            let content = format!("Old low-value note number {index} about the ingestion worker.");
            db.conn
                .execute(
                    "INSERT INTO memories (id,content,kind,project,tags,source,importance,expires_at,metadata,embedding,content_hash,created_at,updated_at,access_count)
                     VALUES (?1,?2,'note','memorypilot','[]','test',1,NULL,NULL,NULL,?3,?4,?4,0)",
                    params![Uuid::new_v4().to_string(), &content, super::content_hash(&content), &stale],
                )
                .expect("seed memory");
        }
        super::compaction::arm_compaction_for_tests();

        let (done_tx, done_rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let outcome = db.add_memory(
                "Fresh decision that trips auto-compaction.",
                "decision",
                Some("memorypilot"),
                &[],
                "test",
                4,
                None,
                None,
                &MemoryScope::default(),
            );
            // Both GC merges (`gc_compressor`) and capsule compaction
            // (`auto_capsule`) write their rows through `add_memory`.
            let reentrant_writes: i64 = db
                .conn
                .query_row(
                    "SELECT COUNT(*) FROM memories WHERE source IN ('auto_capsule', 'gc_compressor')",
                    [],
                    |row| row.get(0),
                )
                .unwrap_or(0);
            let _ = done_tx.send((outcome.map(|_| ()), reentrant_writes));
        });

        let (outcome, reentrant_writes) = done_rx
            .recv_timeout(std::time::Duration::from_secs(60))
            .expect("add_memory deadlocked inside auto-compaction");
        outcome.expect("add_memory");
        assert!(
            reentrant_writes > 0,
            "compaction path was not exercised: no GC merge or capsule written"
        );
    }
    cleanup_db_files(&path);
}

#[test]
fn episodic_rollup_search_and_drilldown() {
    let path = temp_db_path("episodic-rollup");
    {
        let db = Database::open_at(&path).expect("db");
        let empty_tags: Vec<String> = Vec::new();
        let contents = [
            "Decided to use Tokio for the async runtime in the ingestion worker.",
            "Switched the embedding model to BGE-M3 for better multilingual recall.",
            "Fixed a deadlock between two tokio mutexes in the rollup path.",
            "Benchmarked French retrieval: R@5 improved after the model swap.",
        ];
        for content in contents {
            db.add_memory(
                content,
                "decision",
                Some("MemoryPilot"),
                &empty_tags,
                "test",
                4,
                None,
                None,
                &MemoryScope::default(),
            )
            .expect("add memory");
        }

        let report = db.roll_up_episodes().expect("rollup");
        assert!(
            report.interval_created >= 1,
            "expected at least one interval episode, got {:?}",
            report
        );
        assert!(db.episode_count() >= 1, "episodes table should be populated");

        let hits = db
            .search_episodes("tokio async runtime deadlock", 5, Some("MemoryPilot"))
            .expect("search episodes");
        assert!(!hits.is_empty(), "episodic search should return a hit");

        let top = &hits[0].episode;
        let memories = db.episode_memories(&top.id).expect("drill down");
        assert!(
            !memories.is_empty(),
            "drill-down should resolve raw memories"
        );

        // Idempotent: a second rollup creates nothing new.
        let second = db.roll_up_episodes().expect("rollup again");
        assert_eq!(
            second.interval_created, 0,
            "rollup must be idempotent on unchanged data"
        );
    }
    cleanup_db_files(&path);
}

#[test]
fn normalize_project_identity_merges_case_variants() {
    let path = temp_db_path("normalize-project-identity");
    {
        let db = Database::open_at(&path).expect("db");
        let now = chrono::Utc::now().to_rfc3339();

        db.conn
            .execute(
                "INSERT INTO projects (name, path, created_at) VALUES (?1, '', ?2)",
                params!["Planify", &now],
            )
            .expect("insert project");
        db.conn
            .execute(
                "INSERT INTO memories (id, content, kind, project, tags, source, importance, created_at, updated_at, access_count)
                 VALUES (?1, ?2, 'fact', 'Planify', '[]', 'test', 3, ?3, ?3, 0)",
                params![Uuid::new_v4().to_string(), "Uppercase project", &now],
            )
            .expect("insert memory");
        db.conn
            .execute(
                "INSERT INTO memories_fts (rowid, content, tags, kind, project)
                 SELECT rowid, content, tags, kind, project FROM memories WHERE project = 'Planify'",
                [],
            )
            .expect("insert fts");

        db.normalize_project_identities().expect("normalize");

        let projects = db.list_projects().expect("list projects");
        assert_eq!(projects.len(), 1);
        assert_eq!(projects[0].name, "planify");

        let count: i64 = db
            .conn
            .query_row(
                "SELECT COUNT(*) FROM memories WHERE project = 'planify'",
                [],
                |row| row.get(0),
            )
            .expect("count memories");
        assert_eq!(count, 1);
    }
    cleanup_db_files(&path);
}

#[test]
fn detect_project_backfills_missing_path_from_root() {
    let path = temp_db_path("detect-project-root");
    let project_root = std::env::temp_dir().join(format!("planify-root-{}", Uuid::new_v4()));
    let nested_dir = project_root.join("apps").join("web");

    std::fs::create_dir_all(&nested_dir).expect("mkdir");
    std::fs::write(
        project_root.join("Cargo.toml"),
        "[package]\nname = \"planify\"\nversion = \"0.1.0\"\n",
    )
    .expect("write cargo");

    {
        let db = Database::open_at(&path).expect("db");
        let empty_tags: Vec<String> = Vec::new();
        let _ = db
            .add_memory(
                "Planify architecture note",
                "fact",
                Some("Planify"),
                &empty_tags,
                "test",
                3,
                None,
                None,
                &MemoryScope::default(),
            )
            .expect("add memory");

        let detected = db
            .detect_project(&nested_dir.to_string_lossy())
            .expect("detect");
        assert_eq!(detected.as_deref(), Some("planify"));

        let stored_path: String = db
            .conn
            .query_row(
                "SELECT path FROM projects WHERE name = 'planify'",
                [],
                |row| row.get(0),
            )
            .expect("stored path");
        assert_eq!(
            stored_path,
            Database::normalize_path(&project_root.to_string_lossy())
        );
    }
    let _ = std::fs::remove_dir_all(&project_root);
    cleanup_db_files(&path);
}

#[test]
fn secret_values_are_sealed_at_once_and_stay_sealed_after_update() {
    let path = temp_db_path("seal-on-write");
    {
        let db = Database::open_at(&path).expect("db");
        let tags: Vec<String> = Vec::new();
        let secret = "Hx9!kLm2pQ";
        let (mem, _) = db
            .add_memory(
                &format!("Compte démo : demo@example.com / password: {}", secret),
                "fact",
                Some("demo"),
                &tags,
                "test",
                3,
                None,
                None,
                &MemoryScope::default(),
            )
            .expect("add");
        assert_eq!(mem.kind, "credential");
        let stored = |db: &Database| -> String {
            db.conn
                .query_row("SELECT content FROM memories WHERE id=?1", params![mem.id], |r| r.get(0))
                .expect("row")
        };
        assert!(stored(&db).starts_with(super::vault::SEAL_PREFIX));
        assert!(!stored(&db).contains(secret));

        db.update_memory_full(&mem.id, None, None, Some(&["compte".to_string()]), Some(4), None, None)
            .expect("update");
        assert!(!stored(&db).contains(secret), "an update must not write the secret back in clear");
        assert!(!db.get_memory(&mem.id).expect("get").expect("mem").content.contains(secret), "reads stay masked");
        let reveal = |db: &Database| db.reveal_secret(&mem.id, "test", super::RevealConfirm::None).expect("reveal").1;
        assert!(reveal(&db).contains(secret));

        // Replacing the metadata keeps the ciphertext.
        db.update_memory_full(&mem.id, None, None, None, None, None, Some(&serde_json::json!({ "location": "1Password, Démo" })))
            .expect("metadata update");
        assert!(reveal(&db).contains(secret), "a metadata update must not lose the secret");
        // Editing the label (🔒 kept) keeps the secret.
        let label = format!("{}Compte démo, rotation prévue : demo@example.com / password: ••••", super::vault::SEAL_PREFIX);
        db.update_memory_full(&mem.id, Some(&label), None, None, None, None, None).expect("label edit");
        assert!(reveal(&db).contains(secret));
        // Writing the masked text back without the 🔒 would replace the secret by the mask.
        let masked_back = "Compte démo : demo@example.com / password: ••••";
        assert!(db.update_memory_full(&mem.id, Some(masked_back), None, None, None, None, None).is_err());
        assert!(reveal(&db).contains(secret));
        // Changing the kind alone would let reads return the value.
        assert!(db.update_memory_full(&mem.id, None, Some("fact"), None, None, None, None).is_err());
        assert!(!db.get_memory(&mem.id).expect("get").expect("mem").content.contains(secret));
    }
    let _ = std::fs::remove_file(&path);
}

#[test]
fn recall_safe_excludes_credentials_but_full_keeps_them() {
    let path = temp_db_path("recall-safe");
    {
        let db = Database::open_at(&path).expect("db");
        let empty_tags: Vec<String> = Vec::new();

        let _ = db
            .add_memory(
                "API token is sk_live_secret",
                "credential",
                Some("Planify"),
                &empty_tags,
                "test",
                5,
                None,
                None,
                &MemoryScope::default(),
            )
            .expect("add credential");
        let _ = db
            .add_memory(
                "Planify uses SvelteKit",
                "decision",
                Some("planify"),
                &empty_tags,
                "test",
                4,
                None,
                None,
                &MemoryScope::default(),
            )
            .expect("add decision");

        let safe = db
            .recall(
                Some("Planify"),
                None,
                Some("token"),
                RecallMode::Safe,
                false,
                false,
                &MemoryScope::default(),
            )
            .expect("safe recall");
        let full = db
            .recall(
                Some("Planify"),
                None,
                Some("token"),
                RecallMode::Full,
                false,
                false,
                &MemoryScope::default(),
            )
            .expect("full recall");

        let safe_text = serde_json::to_string(&safe).expect("safe json");
        let full_text = serde_json::to_string(&full).expect("full json");
        assert!(!safe_text.contains("sk_live_secret"));
        assert!(!full_text.contains("sk_live_secret"), "full mode shows the credential, never its value");
        assert!(full_text.contains("reveal_secret"), "full mode points to the value");
    }
    cleanup_db_files(&path);
}

#[test]
fn recall_explain_includes_selection_reasons() {
    let path = temp_db_path("recall-explain");
    {
        let db = Database::open_at(&path).expect("db");
        let empty_tags: Vec<String> = Vec::new();

        let _ = db
            .add_memory(
                "Planify uses SvelteKit and Rust",
                "decision",
                Some("Planify"),
                &empty_tags,
                "test",
                5,
                None,
                None,
                &MemoryScope::default(),
            )
            .expect("add decision");

        let explained = db
            .recall(
                Some("Planify"),
                None,
                Some("SvelteKit"),
                RecallMode::Safe,
                true,
                false,
                &MemoryScope::default(),
            )
            .expect("explained recall");

        let explain = explained.get("explain").expect("explain block");
        assert!(explain
            .get("selected_memories")
            .and_then(|value| value.as_array())
            .map(|items| !items.is_empty())
            .unwrap_or(false));
        let rendered = serde_json::to_string(explain).expect("render explain");
        assert!(rendered.contains("selection_source"));
        assert!(rendered.contains("graph_boost"));
        assert!(rendered.contains("importance_weight"));
    }
    cleanup_db_files(&path);
}

#[test]
fn gc_preview_returns_candidates_and_hygiene() {
    let path = temp_db_path("gc-preview");
    {
        let db = Database::open_at(&path).expect("db");
        let now = (chrono::Utc::now() - chrono::Duration::days(90)).to_rfc3339();

        for idx in 0..2 {
            let id = Uuid::new_v4().to_string();
            db.conn
                .execute(
                    "INSERT INTO memories (id, content, kind, project, tags, source, importance, created_at, updated_at, access_count)
                     VALUES (?1, ?2, 'note', 'planify', '[]', 'test', 1, ?3, ?3, 0)",
                    params![id, format!("Old note {}", idx), &now],
                )
                .expect("insert gc memory");
            db.conn
                .execute(
                    "INSERT INTO memories_fts (rowid, content, tags, kind, project)
                     SELECT rowid, content, tags, kind, project FROM memories WHERE id = ?1",
                    params![id],
                )
                .expect("insert gc fts");
        }

        let report = db
            .run_gc(&crate::gc::GcConfig::default(), true)
            .expect("gc preview");
        assert!(report.preview_mode);
        assert!(!report.preview_candidates.is_empty());
        assert!(report.hygiene.never_accessed_memories >= 2);
    }
    cleanup_db_files(&path);
}

#[test]
fn graph_links_stay_within_project_scope() {
    let path = temp_db_path("graph-scope");
    {
        let db = Database::open_at(&path).expect("db");
        let empty_tags: Vec<String> = Vec::new();

        let (planify_a, _) = db
            .add_memory(
                "Planify bug in src/routes/settings.ts for SettingsPanel component",
                "bug",
                Some("Planify"),
                &empty_tags,
                "test",
                4,
                None,
                None,
                &MemoryScope::default(),
            )
            .expect("add planify a");
        let (onlyst, _) = db
            .add_memory(
                "Onlyst bug in src/routes/settings.ts for SettingsPanel component",
                "bug",
                Some("Onlyst"),
                &empty_tags,
                "test",
                4,
                None,
                None,
                &MemoryScope::default(),
            )
            .expect("add onlyst");
        let (planify_b, _) = db
            .add_memory(
                "Planify fix touches src/routes/settings.ts and SettingsPanel rendering",
                "decision",
                Some("planify"),
                &empty_tags,
                "test",
                4,
                None,
                None,
                &MemoryScope::default(),
            )
            .expect("add planify b");

        let same_project_links: i64 = db
            .conn
            .query_row(
                "SELECT COUNT(*) FROM memory_links WHERE source_id = ?1 AND target_id = ?2",
                params![&planify_a.id, &planify_b.id],
                |row| row.get(0),
            )
            .expect("same project links");
        let cross_project_links: i64 = db
            .conn
            .query_row(
                "SELECT COUNT(*) FROM memory_links WHERE source_id = ?1 AND target_id = ?2",
                params![&planify_a.id, &onlyst.id],
                |row| row.get(0),
            )
            .expect("cross project links");

        assert!(same_project_links > 0);
        assert_eq!(cross_project_links, 0);
    }
    cleanup_db_files(&path);
}

#[test]
fn recall_scope_prioritizes_same_thread_memories() {
    let path = temp_db_path("scope-recall");
    {
        let db = Database::open_at(&path).expect("db");
        let empty_tags: Vec<String> = Vec::new();
        let scope = MemoryScope {
            session_id: Some("session-a".into()),
            thread_id: Some("thread-a".into()),
            window_id: Some("window-a".into()),
        };

        let _ = db
            .add_memory(
                "Planify thread-specific architecture note",
                "decision",
                Some("planify"),
                &empty_tags,
                "test",
                3,
                None,
                None,
                &scope,
            )
            .expect("add scoped memory");
        let _ = db
            .add_memory(
                "Planify generic architecture note",
                "decision",
                Some("planify"),
                &empty_tags,
                "test",
                5,
                None,
                None,
                &MemoryScope::default(),
            )
            .expect("add generic memory");

        let recall = db
            .recall(
                Some("planify"),
                None,
                None,
                RecallMode::Safe,
                true,
                false,
                &scope,
            )
            .expect("scoped recall");

        let scope_context = recall
            .get("scope_context")
            .and_then(|value| value.as_array())
            .expect("scope context");
        assert!(!scope_context.is_empty());

        let explain_text = serde_json::to_string(recall.get("explain").expect("explain"))
            .expect("render explain");
        assert!(explain_text.contains("\"selection_source\":\"scope\""));
    }
    cleanup_db_files(&path);
}

#[test]
fn benchmark_recall_reports_core_metrics() {
    let path = temp_db_path("benchmark-recall");
    {
        let db = Database::open_at(&path).expect("db");
        let empty_tags: Vec<String> = Vec::new();

        let _ = db
            .add_memory(
                "Planify uses SvelteKit and Rust for the settings dashboard",
                "decision",
                Some("planify"),
                &empty_tags,
                "test",
                5,
                None,
                None,
                &MemoryScope::default(),
            )
            .expect("add planify");
        let _ = db
            .add_memory(
                "Onlyst uses Swift and Supabase for member onboarding",
                "decision",
                Some("onlyst"),
                &empty_tags,
                "test",
                5,
                None,
                None,
                &MemoryScope::default(),
            )
            .expect("add onlyst");
        let _ = db
            .add_memory(
                "MemoryPilot keeps safe mode by default and hides credentials unless full mode is explicitly requested.",
                "decision",
                Some("memorypilot"),
                &empty_tags,
                "test",
                5,
                None,
                None,
                &MemoryScope::default(),
            )
            .expect("add memorypilot preference");
        let _ = db
            .add_memory(
                "MemoryPilot adds a benchmark_recall CLI command to measure top1 and top5 hit rate on recall quality.",
                "decision",
                Some("memorypilot"),
                &empty_tags,
                "test",
                5,
                None,
                None,
                &MemoryScope::default(),
            )
            .expect("add memorypilot benchmark decision");

        let report = db.benchmark_recall(10).expect("benchmark");
        assert!(
            report
                .get("scenario_count")
                .and_then(|value| value.as_u64())
                .unwrap_or(0)
                >= 2
        );
        assert!(
            report
                .get("golden_defined_count")
                .and_then(|value| value.as_u64())
                .unwrap_or(0)
                >= 10
        );
        assert!(report
            .get("top5_hit_rate")
            .and_then(|value| value.as_f64())
            .is_some());
        assert!(report
            .get("cross_project_leak_rate")
            .and_then(|value| value.as_f64())
            .is_some());
        assert!(report
            .get("credential_leak_rate_safe")
            .and_then(|value| value.as_f64())
            .is_some());
        assert!(
            report
                .get("scenario_source_counts")
                .and_then(|value| value.get("golden"))
                .and_then(|value| value.as_u64())
                .unwrap_or(0)
                >= 1
        );
    }
    cleanup_db_files(&path);
}

#[test]
fn benchmark_recall_uses_golden_scenarios_when_available() {
    let path = temp_db_path("benchmark-golden");
    {
        let db = Database::open_at(&path).expect("db");
        let empty_tags: Vec<String> = Vec::new();

        let _ = db
            .add_memory(
                "MemoryPilot keeps safe mode by default and hides credentials unless full mode is explicitly requested.",
                "decision",
                Some("memorypilot"),
                &empty_tags,
                "test",
                5,
                None,
                None,
                &MemoryScope::default(),
            )
            .expect("add safe mode preference");
        let _ = db
            .add_memory(
                "MemoryPilot adds a benchmark_recall CLI command to measure top1 and top5 hit rate on recall quality.",
                "decision",
                Some("memorypilot"),
                &empty_tags,
                "test",
                5,
                None,
                None,
                &MemoryScope::default(),
            )
            .expect("add benchmark decision");

        let report = db.benchmark_recall(12).expect("benchmark");
        let golden_count = report
            .get("scenario_source_counts")
            .and_then(|value| value.get("golden"))
            .and_then(|value| value.as_u64())
            .unwrap_or(0);
        assert!(golden_count >= 2);
        assert!(
            report
                .get("golden_executed_count")
                .and_then(|value| value.as_u64())
                .unwrap_or(0)
                >= 2
        );
        assert!(report
            .get("scenarios")
            .and_then(|value| value.as_array())
            .map(|items| items.iter().any(|item| item
                .get("scenario_source")
                .and_then(|value| value.as_str())
                == Some("golden")))
            .unwrap_or(false));
    }
    cleanup_db_files(&path);
}

#[test]
fn add_transcript_distills_structured_memories() {
    let path = temp_db_path("transcript-distill");
    {
        let db = Database::open_at(&path).expect("db");
        let scope = MemoryScope {
            session_id: Some("session-1".into()),
            thread_id: Some("thread-1".into()),
            window_id: Some("window-1".into()),
        };
        let tags = vec!["chat".to_string()];
        let transcript = "\
USER: We should always keep safe mode by default and never expose credentials in recall.\n\
ASSISTANT: MemoryPilot will keep safe mode by default and only expose credentials in full mode.\n\
USER: Add a benchmark_recall command to measure top1 and top5 hit rate.\n\
ASSISTANT: We'll add a benchmark_recall tool and a --benchmark-recall CLI command.\n\
USER: There is a bug in src/db.rs where cross-project leakage pollutes recall results.\n";

        let report = db
            .add_transcript(transcript, Some("MemoryPilot"), &tags, "test", &scope, true)
            .expect("ingest transcript");

        assert!(report.chunks_total >= 1);
        assert!(report.distilled_candidates >= 2);
        assert!(report.distilled_added >= 2);

        let (memories, _) = db
            .list_memories(Some("memorypilot"), None, Some("transcript"), 20, 0)
            .expect("list memories");

        assert!(memories
            .iter()
            .any(|memory| memory.kind == "preference" && memory.content.contains("safe mode")));
        assert!(memories
            .iter()
            .any(|memory| memory.kind == "decision"
                && memory.content.contains("benchmark_recall")));
        assert!(memories
            .iter()
            .any(|memory| memory.kind == "bug"
                && memory.content.contains("cross-project leakage")));
        assert!(memories.iter().all(|memory| memory.kind != "transcript"));
    }
    cleanup_db_files(&path);
}

#[test]
fn distilled_transcript_memories_improve_recall_without_loading_chunks() {
    let path = temp_db_path("transcript-recall");
    {
        let db = Database::open_at(&path).expect("db");
        let scope = MemoryScope {
            session_id: Some("session-2".into()),
            thread_id: Some("thread-2".into()),
            window_id: Some("window-2".into()),
        };
        let transcript = "\
USER: We should always keep safe mode by default and never expose credentials in recall.\n\
ASSISTANT: MemoryPilot will keep safe mode by default and only expose credentials in full mode.\n\
USER: Add a benchmark_recall command to measure top1 and top5 hit rate.\n\
ASSISTANT: We'll add a benchmark_recall tool and a --benchmark-recall CLI command.\n";

        let report = db
            .add_transcript(transcript, Some("MemoryPilot"), &[], "test", &scope, true)
            .expect("ingest transcript");
        assert!(report.distilled_added >= 1);

        let recall = db
            .recall(
                Some("memorypilot"),
                None,
                Some("benchmark_recall safe mode credentials"),
                RecallMode::Safe,
                true,
                false,
                &scope,
            )
            .expect("recall");

        let rendered = serde_json::to_string(&recall).expect("render recall");
        assert!(rendered.contains("benchmark_recall") || rendered.contains("safe mode"));
        assert!(!rendered.contains("\"kind\":\"transcript\""));
    }
    cleanup_db_files(&path);
}

#[test]
fn ingest_session_distills_without_raw_chunks_by_default() {
    let path = temp_db_path("session-ingest");
    {
        let db = Database::open_at(&path).expect("db");
        let scope = MemoryScope {
            session_id: Some("session-ingest-1".into()),
            thread_id: Some("thread-ingest-1".into()),
            window_id: None,
        };
        let transcript = "\
USER: We should always keep MemoryPilot as the only MCP server through MCP Hub.\n\
ASSISTANT: MemoryPilot will stay the single MCP server and expose new capabilities as tools.\n\
USER: There is a bug in src/tools.rs where duplicate MCP servers would confuse setup.\n";

        let report = db
            .ingest_session_transcript(
                transcript,
                Some("MemoryPilot"),
                &["session-ingest".to_string()],
                "test-session",
                &scope,
                false,
            )
            .expect("ingest session");

        assert_eq!(report.chunks_total, 0);
        assert_eq!(report.chunk_added, 0);
        assert!(report.distilled_added >= 1);

        let (memories, _) = db
            .list_memories(Some("memorypilot"), None, None, 20, 0)
            .expect("list memories");
        assert!(memories.iter().all(|memory| memory.kind != "transcript"));
        assert!(memories.iter().any(|memory| {
            memory.content.contains("only MCP server")
                || memory.content.contains("duplicate MCP servers")
        }));
    }
    cleanup_db_files(&path);
}

#[test]
fn search_reports_lexical_candidate_sources() {
    let path = temp_db_path("search-sources");
    {
        let db = Database::open_at(&path).expect("db");
        let tags = vec!["SettingsPanel".to_string(), "render".to_string()];
        let _ = db
            .add_memory(
                "SettingsPanel render bug happens in src/routes/settings.ts when the modal opens.",
                "bug",
                Some("MemoryPilot"),
                &tags,
                "test",
                4,
                None,
                None,
                &MemoryScope::default(),
            )
            .expect("add memory");

        let results = db
            .search(
                "SettingsPanel render bug",
                5,
                Some("memorypilot"),
                Some("bug"),
                None,
                None,
            )
            .expect("search");

        assert!(!results.is_empty());
        assert!(results[0]
            .sources
            .iter()
            .any(|source| source.starts_with("fts_")));
    }
    cleanup_db_files(&path);
}

fn contract_memory(id: &str, content: &str, kind: &str, project: Option<&str>, updated_at: &str) -> super::Memory {
    super::Memory {
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

/// What any `MemoryStore` must do, written against the trait so the
/// server mode's store runs the same checks.
fn assert_memory_store_contract(store: &dyn super::MemoryStore) {
    use super::content_hash;
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

#[test]
fn sqlite_store_meets_the_memory_store_contract() {
    let path = temp_db_path("store-contract");
    {
        let db = Database::open_at(&path).expect("db");
        assert_memory_store_contract(&db);
        let fts_rows = |term: &str| -> i64 {
            db.conn
                .query_row("SELECT COUNT(*) FROM memories_fts WHERE memories_fts MATCH ?1", [term], |r| r.get(0))
                .expect("fts")
        };
        assert_eq!(fts_rows("invoices"), 1, "a stored memory has its full-text entry");
        assert_eq!(fts_rows("orchestrator"), 0, "a removed memory leaves no full-text entry");
    }
    cleanup_db_files(&path);
}
