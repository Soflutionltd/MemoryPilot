//! Merging, renaming and identifying projects.

use super::{Database, MemoryScope, SCHEMA_VERSION};
use rusqlite::params;
use std::path::{Path, PathBuf};
use uuid::Uuid;

/// A temporary folder holding one database and the repositories of a test.
struct Workspace {
    root: PathBuf,
}

impl Workspace {
    fn new(test_name: &str) -> Self {
        let root = std::env::temp_dir().join(format!("memory-pilot-{test_name}-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&root).expect("workspace");
        Self { root: std::fs::canonicalize(&root).expect("canonical workspace") }
    }

    fn db_path(&self) -> PathBuf {
        self.root.join("memory.db")
    }

    fn open(&self) -> Database {
        Database::open_at(&self.db_path()).expect("db")
    }

    /// A repository with an `origin` remote, as `git init` + `git remote add` leave it.
    fn repository(&self, folder: &str, remote: &str) -> PathBuf {
        let root = self.root.join(folder);
        std::fs::create_dir_all(root.join(".git")).expect("repository");
        std::fs::write(
            root.join(".git/config"),
            format!("[core]\n\tbare = false\n[remote \"origin\"]\n\turl = {remote}\n\tfetch = +refs/heads/*:refs/remotes/origin/*\n"),
        )
        .expect("git config");
        root
    }

    /// A linked worktree of `main`, as `git worktree add` leaves it.
    fn worktree(&self, folder: &str, main: &Path) -> PathBuf {
        let root = self.root.join(folder);
        let name = root.file_name().expect("worktree name").to_string_lossy().into_owned();
        let git_dir = main.join(".git/worktrees").join(&name);
        std::fs::create_dir_all(&root).expect("worktree");
        std::fs::create_dir_all(&git_dir).expect("worktree git dir");
        std::fs::write(git_dir.join("commondir"), "../..\n").expect("commondir");
        std::fs::write(root.join(".git"), format!("gitdir: {}\n", git_dir.display())).expect("gitdir pointer");
        root
    }

    fn backups(&self) -> Vec<PathBuf> {
        std::fs::read_dir(self.root.join("backups"))
            .map(|entries| entries.flatten().map(|entry| entry.path()).collect())
            .unwrap_or_default()
    }
}

impl Drop for Workspace {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

fn add(db: &Database, content: &str, kind: &str, project: &str) -> String {
    db.add_memory(content, kind, Some(project), &[], "test", 3, None, None, &MemoryScope::default())
        .expect("add memory")
        .0
        .id
}

fn count(db: &Database, sql: &str, project: &str) -> i64 {
    db.conn.query_row(sql, params![project], |row| row.get(0)).expect("count")
}

/// Rows per table for one project name: memories, full-text index,
/// episodes, archive, project entities, triples, project row.
fn footprint(db: &Database, project: &str) -> [i64; 7] {
    [
        count(db, "SELECT COUNT(*) FROM memories WHERE project = ?1", project),
        count(db, "SELECT COUNT(*) FROM memories_fts WHERE project = ?1", project),
        count(db, "SELECT COUNT(*) FROM episodes WHERE project = ?1", project),
        count(db, "SELECT COUNT(*) FROM memories_archive WHERE project = ?1", project),
        count(db, "SELECT COUNT(*) FROM memory_entities WHERE entity_kind = 'project' AND entity_value = ?1", project),
        count(db, "SELECT COUNT(*) FROM knowledge_triples WHERE subject = ?1 OR object = ?1", project),
        count(db, "SELECT COUNT(*) FROM projects WHERE name = ?1", project),
    ]
}

/// Two memories, one episode, one archived original and one triple
/// extracted from the first memory, under `project`.
fn seed_project(db: &Database, project: &str) -> Vec<String> {
    let kept = add(db, &format!("The {project} mail client syncs folders over IMAP IDLE."), "fact", project);
    let other = add(db, &format!("Decision for {project}: drafts are saved locally before upload."), "decision", project);
    let archived = add(db, &format!("Old note about the {project} splash screen colours."), "note", project);
    db.archive_and_delete(&archived, "test", None).expect("archive");
    let now = chrono::Utc::now().to_rfc3339();
    db.conn
        .execute(
            "INSERT INTO episodes (id, project, summary, start_at, end_at, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?4, ?4, ?4)",
            params![Uuid::new_v4().to_string(), project, format!("Work session on {project}"), now],
        )
        .expect("episode");
    db.add_triple(project, "written_in", "rust", None, None, Some(1.0), Some(&kept)).expect("triple");
    vec![kept, other]
}

#[test]
fn schema_is_stamped_with_the_current_version() {
    let workspace = Workspace::new("schema-version");
    let db = workspace.open();
    assert_eq!(db.schema_version().expect("version"), SCHEMA_VERSION);
    assert!(db.apply_migrations().expect("second run").is_empty());
}

#[test]
fn a_database_from_before_versioning_is_migrated_without_losing_projects() {
    let workspace = Workspace::new("schema-upgrade");
    {
        let conn = rusqlite::Connection::open(workspace.db_path()).expect("old database");
        conn.execute_batch(
            "CREATE TABLE projects (name TEXT PRIMARY KEY, path TEXT NOT NULL DEFAULT '', description TEXT, created_at TEXT NOT NULL);
             INSERT INTO projects VALUES ('widget', '', 'kept', '2026-01-01T00:00:00+00:00');",
        )
        .expect("old schema");
        let version: i32 = conn.query_row("PRAGMA user_version", [], |row| row.get(0)).expect("version");
        assert_eq!(version, 0);
    }
    let db = workspace.open();
    assert_eq!(db.schema_version().expect("version"), SCHEMA_VERSION);
    let description: Option<String> = db
        .conn
        .query_row("SELECT description FROM projects WHERE name = 'widget' AND remote IS NULL", [], |row| row.get(0))
        .expect("project kept");
    assert_eq!(description.as_deref(), Some("kept"));
    db.conn.prepare("SELECT alias, kind, project, created_at FROM project_aliases").expect("alias table");
    drop(db);
    assert_eq!(workspace.open().schema_version().expect("reopen"), SCHEMA_VERSION);
}

#[test]
fn merge_dry_run_counts_every_table_and_changes_nothing() {
    let workspace = Workspace::new("merge-dry-run");
    let db = workspace.open();
    seed_project(&db, "ratatoskr");
    seed_project(&db, "courly");
    let before = (footprint(&db, "ratatoskr"), footprint(&db, "courly"));

    let report = db.merge_projects("ratatoskr", "courly", true).expect("dry run");

    assert!(report.dry_run);
    assert_eq!(
        [report.memories, report.memories_fts, report.episodes, report.memories_archive, report.project_entities, report.knowledge_triples],
        [2, 2, 1, 1, 2, 1]
    );
    assert_eq!(report.knowledge_triples_left, 0);
    assert_eq!(report.backup, None);
    assert_eq!((footprint(&db, "ratatoskr"), footprint(&db, "courly")), before);
    assert!(workspace.backups().is_empty(), "a dry run takes no backup");
}

#[test]
fn merge_moves_the_five_tables_and_the_graph_then_deletes_the_source() {
    let workspace = Workspace::new("merge");
    let db = workspace.open();
    let moved = seed_project(&db, "ratatoskr");
    seed_project(&db, "courly");
    let links_before: i64 = db.conn.query_row("SELECT COUNT(*) FROM memory_links", [], |row| row.get(0)).expect("links");
    let vectors_before = db.ann_index().map(|index| index.len());

    let report = db.merge_projects("Ratatoskr", "courly", false).expect("merge");

    assert_eq!(footprint(&db, "ratatoskr"), [0; 7]);
    assert_eq!(footprint(&db, "courly"), [4, 4, 2, 2, 4, 2, 1]);
    assert_eq!(report.memories, 2);
    let links_after: i64 = db.conn.query_row("SELECT COUNT(*) FROM memory_links", [], |row| row.get(0)).expect("links");
    assert_eq!(links_after, links_before, "links are keyed by memory id");
    assert_eq!(db.ann_index().map(|index| index.len()), vectors_before, "the vector index stores no project");

    // The moved memories are found through the full-text index under the target.
    let found = db.search("IMAP IDLE ratatoskr", 5, Some("courly"), None, None, None).expect("search");
    assert!(found.iter().any(|result| result.memory.id == moved[0]));
    assert!(db.search("IMAP IDLE", 5, Some("ratatoskr"), None, None, None).expect("search").is_empty());

    // The old name still leads to the project.
    assert_eq!(db.project_for_alias("name", "ratatoskr").as_deref(), Some("courly"));

    // The copy taken first still holds the source project.
    let backup = PathBuf::from(report.backup.expect("backup path"));
    assert_eq!(workspace.backups(), vec![backup.clone()]);
    let copy = rusqlite::Connection::open(&backup).expect("open backup");
    let kept: i64 = copy
        .query_row("SELECT COUNT(*) FROM memories WHERE project = 'ratatoskr'", [], |row| row.get(0))
        .expect("backup content");
    assert_eq!(kept, 2);
}

#[test]
fn triples_naming_the_source_from_elsewhere_are_left_alone() {
    let workspace = Workspace::new("merge-triples");
    let db = workspace.open();
    // A project called like a language: the triple below is about the language.
    let own = add(&db, "The swift project ships the iOS share extension.", "fact", "swift");
    db.add_triple("swift", "has_target", "share extension", None, None, Some(1.0), Some(&own)).expect("own triple");
    db.add_triple("swift", "is_a", "programming language", None, None, Some(1.0), None).expect("general triple");
    seed_project(&db, "courly");

    let dry = db.merge_projects("swift", "courly", true).expect("dry run");
    assert_eq!((dry.knowledge_triples, dry.knowledge_triples_left), (1, 1));
    db.merge_projects("swift", "courly", false).expect("merge");

    let subject = |predicate: &str| -> String {
        db.conn
            .query_row("SELECT subject FROM knowledge_triples WHERE predicate = ?1", params![predicate], |row| row.get(0))
            .expect("triple")
    };
    assert_eq!(subject("has_target"), "courly");
    assert_eq!(subject("is_a"), "swift");
}

#[test]
fn the_backup_of_the_latest_change_survives_the_retention() {
    let workspace = Workspace::new("backup-retention");
    let db = workspace.open();
    seed_project(&db, "p0");
    seed_project(&db, "courly");
    // More renames than copies are kept, then a merge: its label sorts
    // before "rename", its copy must still be the one that stays.
    for step in 0..11 {
        db.rename_project(&format!("p{step}"), &format!("p{}", step + 1), None, false).expect("rename");
    }
    let report = db.merge_projects("p11", "courly", false).expect("merge");

    let backup = PathBuf::from(report.backup.expect("backup path"));
    assert!(backup.is_file(), "the copy taken before the merge was pruned");
    let mut kept = workspace.backups();
    kept.sort();
    assert_eq!(kept.len(), 10);
    assert_eq!(kept.last(), Some(&backup));
    assert!(kept.iter().all(|path| !path.to_string_lossy().contains("rename-p0-into-p1-") && !path.to_string_lossy().contains("rename-p1-into-p2-")), "the oldest copies go first");
}

#[test]
fn vectors_carrying_the_project_name_are_queued_for_a_new_embedding() {
    let workspace = Workspace::new("merge-reembed");
    let db = workspace.open();
    let chunk = add(&db, "user: the sync still drops the second folder after a reconnect.", "transcript", "ratatoskr");
    add(&db, "Folders are synced one account at a time.", "fact", "ratatoskr");

    let report = db.merge_projects("ratatoskr", "courly", false).expect("merge");

    assert_eq!(report.reembed_queued, 1, "only the transcript is embedded under a [project; date] header");
    assert_eq!(count(&db, "SELECT COUNT(*) FROM memories WHERE id = ?1 AND project = 'courly'", &chunk), 1);
}

#[test]
fn the_remote_of_a_merged_project_still_leads_to_the_target() {
    let workspace = Workspace::new("merge-remote");
    let db = workspace.open();
    let old_repository = workspace.repository("App/ratatoskr", "git@github.com:Acme/ratatoskr.git");
    let new_repository = workspace.repository("App/courly", "git@github.com:Acme/courly.git");
    db.register_project("ratatoskr", &old_repository.to_string_lossy(), None).expect("register");
    db.register_project("courly", &new_repository.to_string_lossy(), None).expect("register");
    add(&db, "The outbox retries a send three times.", "fact", "ratatoskr");
    db.merge_projects("ratatoskr", "courly", false).expect("merge");
    // A fresh clone of the old repository, in a folder named after neither.
    let clone = workspace.repository("Elsewhere/mail-archive", "https://github.com/acme/ratatoskr");

    assert_eq!(db.detect_project(&clone.to_string_lossy()).expect("detect").as_deref(), Some("courly"));
    assert_eq!(db.list_projects().expect("projects").len(), 1);
}

#[test]
fn merge_keeps_both_copies_of_a_text_present_on_each_side() {
    let workspace = Workspace::new("merge-duplicates");
    let db = workspace.open();
    add(&db, "Releases are signed with the Developer ID certificate.", "fact", "ratatoskr");
    add(&db, "Releases are signed with the Developer ID certificate.", "fact", "courly");

    assert_eq!(db.merge_projects("ratatoskr", "courly", true).expect("dry run").duplicate_contents, 1);
    db.merge_projects("ratatoskr", "courly", false).expect("merge");

    assert_eq!(count(&db, "SELECT COUNT(*) FROM memories WHERE project = ?1", "courly"), 2);
}

#[test]
fn a_failed_merge_leaves_every_table_as_it_was() {
    let workspace = Workspace::new("merge-rollback");
    let db = workspace.open();
    seed_project(&db, "ratatoskr");
    seed_project(&db, "courly");
    let before = (footprint(&db, "ratatoskr"), footprint(&db, "courly"));
    // Memories, the full-text index and episodes are already updated when
    // the archive step fails.
    db.conn
        .execute_batch(
            "CREATE TRIGGER refuse_archive_update BEFORE UPDATE ON memories_archive
             BEGIN SELECT RAISE(ABORT, 'archive is read-only'); END;",
        )
        .expect("trigger");

    let error = db.merge_projects("ratatoskr", "courly", false).expect_err("merge must fail");

    assert!(error.contains("archive is read-only"), "{error}");
    assert_eq!((footprint(&db, "ratatoskr"), footprint(&db, "courly")), before);
    assert_eq!(db.project_for_alias("name", "ratatoskr"), None);
    // The connection is usable again: no transaction was left open.
    db.conn.execute_batch("DROP TRIGGER refuse_archive_update").expect("drop trigger");
    db.merge_projects("ratatoskr", "courly", false).expect("merge after the cause is gone");
    assert_eq!(footprint(&db, "ratatoskr"), [0; 7]);
}

#[test]
fn merge_refuses_unknown_and_identical_projects() {
    let workspace = Workspace::new("merge-refusals");
    let db = workspace.open();
    seed_project(&db, "courly");
    assert!(db.merge_projects("nowhere", "courly", false).expect_err("unknown").contains("Unknown project"));
    assert!(db.merge_projects("Courly", "courly", false).expect_err("same").contains("same project"));
    assert!(workspace.backups().is_empty(), "a refused merge takes no backup");
}

#[test]
fn rename_moves_everything_and_keeps_the_old_name_and_folder_as_aliases() {
    let workspace = Workspace::new("rename");
    let db = workspace.open();
    let old_folder = workspace.repository("Suite/ratatoskr", "git@github.com:Acme/courly.git");
    db.register_project("ratatoskr", &old_folder.to_string_lossy(), Some("Mail client")).expect("register");
    seed_project(&db, "ratatoskr");
    let new_folder = workspace.root.join("Suite/courly");
    std::fs::rename(&old_folder, &new_folder).expect("rename folder");

    let dry = db.rename_project("ratatoskr", "courly", Some(&new_folder.to_string_lossy()), true).expect("dry run");
    assert_eq!(footprint(&db, "courly"), [0; 7]);
    assert_eq!(dry.memories, 2);

    let report = db.rename_project("ratatoskr", "courly", Some(&new_folder.to_string_lossy()), false).expect("rename");

    assert_eq!(footprint(&db, "ratatoskr"), [0; 7]);
    assert_eq!(footprint(&db, "courly"), [2, 2, 1, 1, 2, 1, 1]);
    assert_eq!(report.path, new_folder.to_string_lossy());
    let (path, description, remote): (String, Option<String>, Option<String>) = db
        .conn
        .query_row("SELECT path, description, remote FROM projects WHERE name = 'courly'", [], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?))
        })
        .expect("project row");
    assert_eq!(path, new_folder.to_string_lossy());
    assert_eq!(description.as_deref(), Some("Mail client"));
    assert_eq!(remote.as_deref(), Some("github.com/acme/courly"));
    assert_eq!(db.project_for_alias("name", "ratatoskr").as_deref(), Some("courly"));
    assert_eq!(db.project_for_alias("path", &old_folder.to_string_lossy()).as_deref(), Some("courly"));
    assert_eq!(db.detect_project(&new_folder.to_string_lossy()).expect("detect").as_deref(), Some("courly"));
    assert_eq!(workspace.backups().len(), 1);
}

#[test]
fn rename_to_an_existing_project_is_refused_and_points_to_merge() {
    let workspace = Workspace::new("rename-refused");
    let db = workspace.open();
    seed_project(&db, "ratatoskr");
    seed_project(&db, "courly");
    let before = (footprint(&db, "ratatoskr"), footprint(&db, "courly"));

    let error = db.rename_project("ratatoskr", "courly", None, false).expect_err("must refuse");

    assert!(error.contains("merge_projects"), "{error}");
    assert_eq!((footprint(&db, "ratatoskr"), footprint(&db, "courly")), before);
}

#[test]
fn a_project_can_change_folder_without_changing_name() {
    let workspace = Workspace::new("move-folder");
    let db = workspace.open();
    let old_folder = workspace.repository("App/MemoryPilot", "git@github.com:Acme/MemoryPilot.git");
    db.register_project("memorypilot", &old_folder.to_string_lossy(), None).expect("register");
    add(&db, "The recall hook runs at session start.", "fact", "memorypilot");
    let new_folder = workspace.root.join("App/MCP/MemoryPilot");
    std::fs::create_dir_all(new_folder.parent().expect("parent")).expect("MCP");
    std::fs::rename(&old_folder, &new_folder).expect("move folder");

    let report = db
        .rename_project("memorypilot", "memorypilot", Some(&new_folder.to_string_lossy()), false)
        .expect("new folder");

    assert_eq!(report.path, new_folder.to_string_lossy());
    assert_eq!(count(&db, "SELECT COUNT(*) FROM memories WHERE project = ?1", "memorypilot"), 1);
    assert_eq!(db.project_for_alias("path", &old_folder.to_string_lossy()).as_deref(), Some("memorypilot"));
    assert_eq!(db.detect_project(&new_folder.join("src").to_string_lossy()).expect("detect").as_deref(), Some("memorypilot"));
}

#[test]
fn a_single_memory_moves_with_its_index_row_and_project_entity() {
    let workspace = Workspace::new("move-memory");
    let db = workspace.open();
    let id = add(&db, "Attachments above 25 MB are sent as download links.", "decision", "ratatoskr");

    let moved = db.move_memory_to_project(&id, Some("Courly")).expect("move").expect("memory");

    assert_eq!(moved.project.as_deref(), Some("courly"));
    assert_eq!(footprint(&db, "ratatoskr")[..2], [0, 0]);
    assert_eq!(footprint(&db, "ratatoskr")[4], 0);
    assert_eq!(footprint(&db, "courly")[..2], [1, 1]);
    assert_eq!(footprint(&db, "courly")[4], 1);
    assert_eq!(footprint(&db, "courly")[6], 1, "the target project is created");

    let global = db.move_memory_to_project(&id, Some("global")).expect("move").expect("memory");
    assert_eq!(global.project, None);
    assert_eq!(count(&db, "SELECT COUNT(*) FROM memory_entities WHERE memory_id = ?1 AND entity_kind = 'project'", &id), 0);
    assert_eq!(count(&db, "SELECT COUNT(*) FROM memories_fts WHERE project = '' AND rowid = (SELECT rowid FROM memories WHERE id = ?1)", &id), 1);
    assert!(db.move_memory_to_project("missing-id", Some("courly")).expect("move").is_none());

    // An empty or unusable name never sends a memory to the global scope.
    let back = db.move_memory_to_project(&id, Some("courly")).expect("move").expect("memory");
    assert_eq!(back.project.as_deref(), Some("courly"));
    for name in ["", "   ", "!!!", "none", "null"] {
        let error = db.move_memory_to_project(&id, Some(name)).expect_err("must refuse");
        assert!(error.contains("Invalid project name"), "{error}");
    }
    assert_eq!(count(&db, "SELECT COUNT(*) FROM memories WHERE id = ?1 AND project = 'courly'", &id), 1);
}

#[test]
fn a_worktree_belongs_to_the_project_of_its_main_repository() {
    let workspace = Workspace::new("worktree");
    let db = workspace.open();
    let main = workspace.repository("App/Suite/courly", "https://github.com/Acme/courly.git");
    db.register_project("courly", &main.to_string_lossy(), None).expect("register");
    let worktree = workspace.worktree("Worktrees/courly/inbox-zero", &main);

    assert_eq!(db.detect_project(&worktree.to_string_lossy()).expect("detect").as_deref(), Some("courly"));
    assert_eq!(db.detect_project(&worktree.join("src").to_string_lossy()).expect("detect").as_deref(), Some("courly"));

    // Registering the worktree itself records the main repository.
    let registered = db.register_project("courly", &worktree.to_string_lossy(), None).expect("register worktree");
    assert_eq!(registered.path, main.to_string_lossy());
    assert_eq!(db.list_projects().expect("projects").len(), 1);
}

#[test]
fn a_worktree_of_an_unregistered_repository_is_named_after_the_repository() {
    let workspace = Workspace::new("worktree-unregistered");
    let db = workspace.open();
    let main = workspace.repository("App/widget", "git@github.com:Acme/widget.git");
    let worktree = workspace.worktree("Worktrees/widget/fix-login", &main);

    assert_eq!(db.detect_project(&worktree.to_string_lossy()).expect("detect").as_deref(), Some("widget"));
}

#[test]
fn two_folders_with_the_same_remote_are_the_same_project() {
    let workspace = Workspace::new("same-remote");
    let db = workspace.open();
    let first = workspace.repository("App/courly", "git@github.com:Acme/courly.git");
    db.register_project("courly", &first.to_string_lossy(), None).expect("register");
    add(&db, "The sidebar lists accounts by unread count.", "fact", "courly");
    let clone = workspace.repository("Elsewhere/mail-client-copy", "https://github.com/acme/courly");

    assert_eq!(db.detect_project(&clone.to_string_lossy()).expect("detect").as_deref(), Some("courly"));
    assert_eq!(db.list_projects().expect("projects").len(), 1);
}

#[test]
fn an_old_folder_name_follows_its_alias() {
    let workspace = Workspace::new("name-alias");
    let db = workspace.open();
    seed_project(&db, "ratatoskr");
    db.rename_project("ratatoskr", "courly", None, false).expect("rename");
    // A checkout still named after the old project, without any remote.
    let folder = workspace.root.join("Old/ratatoskr");
    std::fs::create_dir_all(folder.join(".git")).expect("folder");

    assert_eq!(db.detect_project(&folder.to_string_lossy()).expect("detect").as_deref(), Some("courly"));
}

#[test]
fn a_known_project_is_found_for_a_folder_without_ever_making_one_up() {
    let workspace = Workspace::new("known-project");
    let db = workspace.open();
    let repository = workspace.repository("App/courly", "git@github.com:Acme/courly.git");
    db.register_project("courly", &repository.to_string_lossy(), None).expect("register");
    add(&db, "The sidebar lists accounts by unread count.", "fact", "courly");
    let known = |path: &Path| db.known_project_for_path(&path.to_string_lossy());

    assert_eq!(known(&repository.join("src/sync")).as_deref(), Some("courly"));
    // Another clone of the same remote, and a worktree of the repository.
    let clone = workspace.repository("Elsewhere/mail-copy", "https://github.com/acme/courly");
    assert_eq!(known(&clone).as_deref(), Some("courly"));
    let worktree = workspace.worktree("Worktrees/courly/fix", &repository);
    assert_eq!(known(&worktree.join("src")).as_deref(), Some("courly"));
    // A folder that exists but belongs to nothing known, and one that does not exist.
    let stranger = workspace.repository("App/stranger", "git@github.com:Acme/stranger.git");
    assert_eq!(known(&stranger), None);
    assert_eq!(known(&workspace.root.join("App/never-existed/src")), None);
    assert_eq!(db.list_projects().expect("projects").len(), 1, "nothing was registered on the way");

    // The folder is deleted: its recorded path still answers.
    std::fs::remove_dir_all(&repository).expect("delete");
    assert_eq!(known(&repository.join("src/sync")).as_deref(), Some("courly"));
    assert_eq!(db.project_remote("courly").as_deref(), Some("github.com/acme/courly"));
}

#[test]
fn health_reports_missing_folders_duplicates_and_stale_entities_with_their_fix() {
    let workspace = Workspace::new("hygiene");
    let db = workspace.open();
    let first = workspace.repository("App/courly", "git@github.com:Acme/courly.git");
    let second = workspace.repository("App/ratatoskr", "https://github.com/acme/courly.git");
    db.register_project("courly", &first.to_string_lossy(), None).expect("register");
    db.register_project("ratatoskr", &second.to_string_lossy(), None).expect("register");
    add(&db, "Threads are grouped by the References header.", "fact", "courly");
    add(&db, "Search runs on the local index first.", "fact", "courly");
    add(&db, "The compose window autosaves every five seconds.", "fact", "ratatoskr");
    let gone = workspace.root.join("App/gone");
    db.conn
        .execute(
            "INSERT INTO projects (name, path, created_at) VALUES ('gone', ?1, '2026-01-01T00:00:00+00:00')",
            params![gone.to_string_lossy()],
        )
        .expect("project without folder");

    let report = db.memory_health_report().expect("health");
    let projects = &report["projects"];

    assert_eq!(projects["missing_paths"].as_array().expect("missing").len(), 1);
    assert_eq!(projects["missing_paths"][0]["project"], "gone");
    let same = projects["same_project"].as_array().expect("same");
    assert_eq!(same.len(), 1);
    assert_eq!(same[0]["reason"], "same_remote");
    assert_eq!(same[0]["fix"]["action"], "merge_projects");
    assert_eq!(same[0]["fix"]["args"]["source"], "ratatoskr");
    assert_eq!(same[0]["fix"]["args"]["target"], "courly", "the project with the most memories is kept");
    assert_eq!(projects["issues"], 2);

    // The reported merge, then a manual SQL move that forgot the graph.
    db.merge_projects("ratatoskr", "courly", false).expect("merge");
    db.conn
        .execute("UPDATE memory_entities SET entity_value = 'ratatoskr' WHERE entity_kind = 'project' AND rowid = (SELECT MIN(rowid) FROM memory_entities WHERE entity_kind = 'project')", [])
        .expect("stale entity");
    let report = db.memory_health_report().expect("health");
    assert_eq!(report["projects"]["same_project"].as_array().expect("same").len(), 0);
    assert_eq!(report["projects"]["stale_project_entities"][0]["entity"], "ratatoskr");
    assert_eq!(report["projects"]["stale_project_entities"][0]["project"], "courly");
    assert_eq!(report["projects"]["missing_project_entities"][0]["project"], "courly");
    assert_eq!(report["projects"]["project_entities_fix"]["action"], "repair_index");

    assert_eq!(db.repair_project_entities().expect("repair"), (1, 1));
    let report = db.memory_health_report().expect("health");
    assert_eq!(report["projects"]["stale_project_entities"].as_array().expect("stale").len(), 0);
    assert_eq!(report["projects"]["missing_project_entities"].as_array().expect("missing").len(), 0);
    assert_eq!(report["projects"]["issues"], 1, "only the missing folder is left");
}

#[test]
fn a_project_recreated_under_an_alias_is_reported_for_merge() {
    let workspace = Workspace::new("alias-duplicate");
    let db = workspace.open();
    seed_project(&db, "ratatoskr");
    db.rename_project("ratatoskr", "courly", None, false).expect("rename");
    // An old client writes under the former name again.
    add(&db, "Signatures are stored per account.", "fact", "ratatoskr");

    let report = db.project_hygiene_report().expect("hygiene");

    let same = report["same_project"].as_array().expect("same");
    assert_eq!(same.len(), 1);
    assert_eq!(same[0]["reason"], "shared_name_alias");
    assert_eq!(same[0]["fix"]["args"]["source"], "ratatoskr");
    assert_eq!(same[0]["fix"]["args"]["target"], "courly");
}
