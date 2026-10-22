//! Verified memory, database side: which repository a memory belongs to,
//! what its Git history says about it, and the explicit lifecycle
//! (`metadata.status` = open / resolved / verified) that overrides any guess.
//!
//! Search, recall and get results are annotated with a `verification`
//! object only when there is something to say, within a time budget.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use chrono::DateTime;
use rusqlite::params;
use serde_json::{json, Map, Value};

use super::{Database, Memory};
use crate::verify::{self, Commit};

const ACTIONABLE_KINDS: &[&str] = &["bug", "problem", "todo"];
/// Kinds that state how things are now; milestones and bugs describe the past.
const CURRENT_STATE_KINDS: &[&str] = &[
    "fact",
    "decision",
    "architecture",
    "pattern",
    "note",
    "skill",
    "snippet",
];
/// Kinds that may describe an open problem in prose ("faille non corrigée").
const PROSE_KINDS: &[&str] = &["fact", "note", "decision", "pattern"];
const SKIP_KINDS: &[&str] = &["credential", "transcript"];
const OPEN_MARKERS: &[&str] = &[
    "non corrigé",
    "pas corrigé",
    "pas encore corrigé",
    "not fixed",
    "still broken",
    "toujours cassé",
];
/// A memory that records its own fix is a log of the past, not an open problem.
const RESOLVED_MARKERS: &[&str] = &[
    "corrigé",
    "corrigée",
    "corrigés",
    "corrigées",
    "correctif",
    "correction",
    "résolu",
    "fixed",
    "fixes:",
    "fix:",
    "resolved",
    "patched",
];
pub(crate) const STATUSES: &[&str] = &["open", "resolved", "verified"];
const CLOSED_STATUSES: &[&str] = &["resolved", "verified"];
const REPO_CACHE_TTL: Duration = Duration::from_secs(60);
const ANNOTATION_BUDGET: Duration = Duration::from_millis(1500);
const MAX_ANNOTATIONS: usize = 20;
const MAX_LISTED_ANCHORS: usize = 5;
const HEADLINE_CHARS: usize = 300;

type RepoCache = HashMap<String, (Instant, Option<PathBuf>)>;

fn repo_cache() -> &'static Mutex<RepoCache> {
    static CACHE: OnceLock<Mutex<RepoCache>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// `metadata.status` when it is one of the lifecycle values.
pub(crate) fn stored_status(memory: &Memory) -> Option<&str> {
    memory
        .metadata
        .as_ref()?
        .get("status")?
        .as_str()
        .filter(|status| STATUSES.contains(status))
}

pub(crate) fn is_closed(memory: &Memory) -> bool {
    stored_status(memory).is_some_and(|status| CLOSED_STATUSES.contains(&status))
}

fn is_open_problem(memory: &Memory) -> bool {
    let lower = memory.content.to_lowercase();
    let says_open = OPEN_MARKERS.iter().any(|marker| lower.contains(marker));
    let kind = memory.kind.as_str();
    let candidate = ACTIONABLE_KINDS.contains(&kind) || (PROSE_KINDS.contains(&kind) && says_open);
    if !candidate {
        return false;
    }
    let negated = [
        "non corrigé",
        "pas corrigé",
        "pas encore corrigé",
        "not fixed",
        "not resolved",
        "non résolu",
    ];
    let mut stripped = lower.clone();
    for phrase in negated {
        stripped = stripped.replace(phrase, " ");
    }
    !RESOLVED_MARKERS
        .iter()
        .any(|marker| stripped.contains(marker))
}

fn commit_json(commit: &Commit) -> Value {
    json!({
        "sha": commit.short_sha(),
        "date": commit.date.to_rfc3339(),
        "subject": commit.subject,
    })
}

fn report(
    status: &str,
    since: &str,
    anchors: &[verify::Anchor],
    commits: &[&Commit],
    limit: usize,
    hint: &str,
) -> Value {
    let anchors: Vec<&str> = anchors
        .iter()
        .take(MAX_LISTED_ANCHORS)
        .map(|a| a.label())
        .collect();
    let listed: Vec<Value> = commits.iter().take(limit).map(|c| commit_json(c)).collect();
    let mut value = json!({
        "status": status,
        "since": since,
        "commits": listed,
        "commit_count": commits.len(),
        "hint": hint,
    });
    if !anchors.is_empty() {
        value["anchors"] = json!(anchors);
    }
    value
}

impl Database {
    fn verification_enabled(&self) -> bool {
        !verify::disabled()
            && !matches!(
                self.get_config("verify_memory").as_deref().map(str::trim),
                Some("off" | "0" | "false")
            )
    }

    /// Git work tree of `project`, from its registered path.
    pub(crate) fn project_repo(&self, project: &str) -> Option<PathBuf> {
        if let Ok(guard) = repo_cache().lock() {
            if let Some((checked, repo)) = guard.get(project) {
                if checked.elapsed() < REPO_CACHE_TTL {
                    return repo.clone();
                }
            }
        }
        let path: Option<String> = self
            .conn
            .query_row(
                "SELECT path FROM projects WHERE name=?1",
                params![project],
                |row| row.get::<_, Option<String>>(0),
            )
            .ok()
            .flatten();
        let repo = path.as_deref().and_then(verify::repo_root);
        if let Ok(mut guard) = repo_cache().lock() {
            guard.insert(project.to_string(), (Instant::now(), repo.clone()));
        }
        repo
    }

    /// What the repository says about `memory`. `None` when there is
    /// nothing to report (no repository, nothing cited changed). With
    /// `detail`, always answers and lists more commits.
    pub fn verification(&self, memory: &Memory, detail: bool) -> Option<Value> {
        if !detail && !self.verification_enabled() {
            return None;
        }
        if SKIP_KINDS.contains(&memory.kind.as_str()) {
            return detail
                .then(|| json!({ "status": "not_applicable", "reason": "kind not verified" }));
        }
        if let Some(status) = stored_status(memory).filter(|s| CLOSED_STATUSES.contains(s)) {
            let metadata = memory.metadata.as_ref();
            let field = |key: &str| {
                metadata
                    .and_then(|m| m.get(key))
                    .cloned()
                    .unwrap_or(Value::Null)
            };
            return Some(json!({
                "status": status,
                "resolved_by": field("resolved_by"),
                "status_at": field("status_at"),
                "status_note": field("status_note"),
            }));
        }
        let no_repo = || {
            detail.then(|| json!({ "status": "unverifiable", "reason": "no Git repository registered for this project" }))
        };
        let Some(project) = memory.project.as_deref() else {
            return no_repo();
        };
        let Some(repo) = self.project_repo(project) else {
            return no_repo();
        };
        let unreadable =
            |reason: &str| detail.then(|| json!({ "status": "unverifiable", "reason": reason }));
        let Some(snapshot) = verify::snapshot(&repo) else {
            return unreadable("the repository has no readable history");
        };
        let Ok(since) = DateTime::parse_from_rfc3339(&memory.created_at) else {
            return unreadable("the memory has no valid creation date");
        };
        let anchors = verify::anchors(&memory.content, &repo, &snapshot.files);
        let touching: Vec<&Commit> = verify::touching(&snapshot, &anchors, since)
            .into_iter()
            .filter(|commit| commit.kind.is_none() || commit.changes_behaviour())
            .collect();
        let limit = if detail { 10 } else { 3 };

        if is_open_problem(memory) {
            // A prose memory cites files for several topics: only a bug,
            // problem or todo is about the files it cites.
            let fixes: Vec<&Commit> = if ACTIONABLE_KINDS.contains(&memory.kind.as_str()) {
                touching
                    .iter()
                    .copied()
                    .filter(|commit| commit.is_fix())
                    .collect()
            } else {
                Vec::new()
            };
            if !fixes.is_empty() {
                return Some(report(
                    "likely_resolved",
                    &memory.created_at,
                    &anchors,
                    &fixes,
                    limit,
                    "A later fix commit touches the code this memory cites: it is probably resolved. Check, then record it with resolve_memory.",
                ));
            }
            let text = format!("{} {}", headline(&memory.content), memory.tags.join(" "));
            let matches = verify::fix_matches(&snapshot, &text, since, limit);
            if !matches.is_empty() {
                return Some(report(
                    "possibly_resolved",
                    &memory.created_at,
                    &anchors,
                    &matches,
                    limit,
                    "A later fix commit reads like the resolution of this memory: check before treating it as open.",
                ));
            }
        }
        if CURRENT_STATE_KINDS.contains(&memory.kind.as_str()) {
            if let Some(contradicted) = contradiction_report(memory, &repo) {
                return Some(contradicted);
            }
        }
        if !touching.is_empty() {
            return Some(report(
                "changed_since",
                &memory.created_at,
                &anchors,
                &touching,
                limit,
                "Files this memory cites changed after it was written: re-check before relying on it.",
            ));
        }
        detail.then(|| {
            let mut value = report(
                "no_change_found",
                &memory.created_at,
                &anchors,
                &[],
                0,
                "No later commit touches what this memory cites.",
            );
            value["repo"] = json!(repo.to_string_lossy());
            value
        })
    }

    /// Facts computed from the project's repository (full detail).
    pub fn project_facts(&self, project: &str) -> Option<Value> {
        let project = Self::canonical_project_name(project)?;
        let repo = self.project_repo(&project)?;
        crate::facts::project_facts(&repo).map(|facts| facts.full.clone())
    }

    /// The few lines of repository facts shown by recall and the project brain.
    pub fn compact_project_facts(&self, project: &str) -> Option<Value> {
        if !self.verification_enabled() {
            return None;
        }
        let project = Self::canonical_project_name(project)?;
        let repo = self.project_repo(&project)?;
        let facts = crate::facts::project_facts(&repo)?;
        let empty = facts
            .compact
            .as_object()
            .is_none_or(|object| object.is_empty());
        (!empty).then(|| facts.compact.clone())
    }

    /// Adds `verification` to every memory object (`id` + `content` + `kind`)
    /// found in `value`, when there is something to report.
    pub fn annotate_verification(&self, value: &mut Value) {
        if !self.verification_enabled() {
            return;
        }
        let started = Instant::now();
        let mut checked: HashMap<String, Option<Value>> = HashMap::new();
        self.annotate_node(value, started, &mut checked);
    }

    fn annotate_node(
        &self,
        value: &mut Value,
        started: Instant,
        checked: &mut HashMap<String, Option<Value>>,
    ) {
        match value {
            Value::Array(items) => {
                for item in items {
                    self.annotate_node(item, started, checked);
                }
            }
            Value::Object(object) => {
                if let Some(id) = memory_id(object) {
                    if let Some(verification) = self.cached_verification(&id, started, checked) {
                        object.insert("verification".into(), verification);
                    }
                }
                for (key, child) in object.iter_mut() {
                    if key != "verification" && key != "metadata" {
                        self.annotate_node(child, started, checked);
                    }
                }
            }
            _ => {}
        }
    }

    fn cached_verification(
        &self,
        id: &str,
        started: Instant,
        checked: &mut HashMap<String, Option<Value>>,
    ) -> Option<Value> {
        if let Some(known) = checked.get(id) {
            return known.clone();
        }
        if checked.len() >= MAX_ANNOTATIONS || started.elapsed() > ANNOTATION_BUDGET {
            return None;
        }
        let verification = self
            .get_memory(id)
            .ok()
            .flatten()
            .and_then(|memory| self.verification(&memory, false));
        checked.insert(id.to_string(), verification.clone());
        verification
    }

    /// Records the lifecycle of a memory in its metadata. `open` clears the
    /// previous resolution.
    pub fn set_memory_status(
        &self,
        id: &str,
        status: &str,
        commit: Option<&str>,
        note: Option<&str>,
    ) -> Result<Option<Memory>, String> {
        if !STATUSES.contains(&status) {
            return Err(format!("status must be one of: {}", STATUSES.join(", ")));
        }
        let Some(existing) = self.get_memory(id)? else {
            return Ok(None);
        };
        if existing.kind == "credential" {
            return Err("credentials have no lifecycle status".into());
        }
        let mut metadata = match existing.metadata.clone() {
            Some(Value::Object(object)) => object,
            _ => Map::new(),
        };
        metadata.insert("status".into(), json!(status));
        metadata.insert("status_at".into(), json!(chrono::Utc::now().to_rfc3339()));
        if status == "open" {
            metadata.remove("resolved_by");
        } else if let Some(commit) = commit.map(str::trim).filter(|c| !c.is_empty()) {
            metadata.insert("resolved_by".into(), json!(commit));
        }
        match note.map(str::trim).filter(|n| !n.is_empty()) {
            Some(note) => {
                metadata.insert("status_note".into(), json!(note));
            }
            None => {
                metadata.remove("status_note");
            }
        }
        self.update_memory_full(
            id,
            None,
            None,
            None,
            None,
            None,
            Some(&Value::Object(metadata)),
        )
    }
}

/// The opening of a memory, where the problem is stated; long bodies list
/// enough unrelated words to match any commit.
fn headline(content: &str) -> &str {
    match content.char_indices().nth(HEADLINE_CHARS) {
        Some((index, _)) => &content[..index],
        None => content,
    }
}

fn contradiction_report(memory: &Memory, repo: &std::path::Path) -> Option<Value> {
    let facts = crate::facts::project_facts(repo)?;
    let found = crate::facts::contradictions(&memory.content, &facts.dependencies);
    if found.is_empty() {
        return None;
    }
    let claims: Vec<Value> = found
        .iter()
        .map(|c| {
            json!({
                "claim": c.claim,
                "repo": format!("{} {}", c.package, c.actual),
                "file": c.file,
            })
        })
        .collect();
    Some(json!({
        "status": "contradicted",
        "since": memory.created_at,
        "claims": claims,
        "hint": "The repository now uses a different version than this memory states: trust the repository, then update or supersede the memory.",
    }))
}

fn memory_id(object: &Map<String, Value>) -> Option<String> {
    let is_memory = object.get("content").is_some_and(Value::is_string)
        && object.get("kind").is_some_and(Value::is_string);
    if !is_memory {
        return None;
    }
    object.get("id").and_then(Value::as_str).map(str::to_string)
}

#[cfg(test)]
mod tests {
    use crate::db::{Database, MemoryScope};
    use std::path::{Path, PathBuf};
    use std::process::Command;
    use uuid::Uuid;

    struct Fixture {
        db_path: PathBuf,
        repo: PathBuf,
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.db_path);
            let _ = std::fs::remove_file(format!("{}-wal", self.db_path.display()));
            let _ = std::fs::remove_file(format!("{}-shm", self.db_path.display()));
            let _ = std::fs::remove_dir_all(&self.repo);
        }
    }

    fn git(repo: &Path, args: &[&str], date: &str) {
        let status = Command::new("git")
            .arg("-C")
            .arg(repo)
            .args(args)
            .env("GIT_AUTHOR_DATE", date)
            .env("GIT_COMMITTER_DATE", date)
            .env("GIT_AUTHOR_NAME", "Test")
            .env("GIT_AUTHOR_EMAIL", "test@example.com")
            .env("GIT_COMMITTER_NAME", "Test")
            .env("GIT_COMMITTER_EMAIL", "test@example.com")
            .output()
            .expect("git runs");
        assert!(
            status.status.success(),
            "git {args:?}: {}",
            String::from_utf8_lossy(&status.stderr)
        );
    }

    fn commit_file(repo: &Path, file: &str, body: &str, subject: &str, date: &str) {
        let path = repo.join(file);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, body).unwrap();
        git(repo, &["add", "-A"], date);
        git(
            repo,
            &["commit", "-q", "--no-gpg-sign", "-m", subject],
            date,
        );
    }

    fn fixture(name: &str) -> (Fixture, Database, String) {
        let id = Uuid::new_v4();
        let repo = std::env::temp_dir().join(format!("memory-pilot-verify-{name}-{id}"));
        std::fs::create_dir_all(&repo).unwrap();
        git(
            &repo,
            &["init", "-q", "-b", "main"],
            "2026-10-01T10:00:00+00:00",
        );
        commit_file(
            &repo,
            "src/lib/auth.ts",
            "export const auth = open();\n",
            "feat(auth): signup",
            "2026-10-01T10:00:00+00:00",
        );
        commit_file(
            &repo,
            "README.md",
            "# demo\n",
            "docs: readme",
            "2026-10-01T11:00:00+00:00",
        );
        let db_path = std::env::temp_dir().join(format!("memory-pilot-verify-{name}-{id}.db"));
        let db = Database::open_at(&db_path).expect("db");
        let project = db
            .register_project(
                &format!("verifydemo{}", id.simple()),
                repo.to_str().unwrap(),
                None,
            )
            .expect("project")
            .name;
        (Fixture { db_path, repo }, db, project)
    }

    fn add(db: &Database, project: &str, content: &str, kind: &str, created_at: &str) -> String {
        let memory = db
            .add_memory(
                content,
                kind,
                Some(project),
                &[],
                "test",
                3,
                None,
                None,
                &MemoryScope::default(),
            )
            .expect("add")
            .0;
        db.conn
            .execute(
                "UPDATE memories SET created_at=?1 WHERE id=?2",
                rusqlite::params![created_at, memory.id],
            )
            .unwrap();
        memory.id
    }

    #[test]
    fn a_fix_on_a_cited_file_marks_the_bug_likely_resolved_until_confirmed() {
        let (fixture, db, project) = fixture("likely");
        let bug = add(
            &db,
            &project,
            "Inscription ouverte dans src/lib/auth.ts : n'importe qui crée un compte.",
            "bug",
            "2026-10-02T09:00:00+00:00",
        );
        let unrelated = add(
            &db,
            &project,
            "Le logo est en PNG 900x429.",
            "fact",
            "2026-10-02T09:00:00+00:00",
        );
        commit_file(
            &fixture.repo,
            "src/lib/auth.ts",
            "export const auth = closed();\n",
            "fix(auth): close public signup",
            "2026-10-03T10:00:00+00:00",
        );

        let memory = db.get_memory(&bug).unwrap().unwrap();
        let verification = db.verification(&memory, false).expect("reported");
        assert_eq!(verification["status"], "likely_resolved");
        assert_eq!(verification["anchors"][0], "src/lib/auth.ts");
        assert_eq!(
            verification["commits"][0]["subject"],
            "fix(auth): close public signup"
        );

        let unrelated = db.get_memory(&unrelated).unwrap().unwrap();
        assert!(db.verification(&unrelated, false).is_none());

        db.set_memory_status(&bug, "resolved", Some("abc1234"), Some("checked"))
            .unwrap();
        let memory = db.get_memory(&bug).unwrap().unwrap();
        let verification = db.verification(&memory, false).expect("reported");
        assert_eq!(verification["status"], "resolved");
        assert_eq!(verification["resolved_by"], "abc1234");

        db.set_memory_status(&bug, "open", None, None).unwrap();
        let memory = db.get_memory(&bug).unwrap().unwrap();
        assert_eq!(
            db.verification(&memory, false).unwrap()["status"],
            "likely_resolved"
        );
        assert!(memory.metadata.unwrap().get("resolved_by").is_none());
    }

    #[test]
    fn a_fix_written_before_the_memory_never_resolves_it() {
        let (fixture, db, project) = fixture("before");
        commit_file(
            &fixture.repo,
            "src/lib/auth.ts",
            "export const auth = closed();\n",
            "fix(auth): close public signup",
            "2026-10-03T10:00:00+00:00",
        );
        let bug = add(
            &db,
            &project,
            "Inscription ouverte dans src/lib/auth.ts.",
            "bug",
            "2026-10-04T09:00:00+00:00",
        );
        let memory = db.get_memory(&bug).unwrap().unwrap();
        assert!(db.verification(&memory, false).is_none());
        assert_eq!(
            db.verification(&memory, true).unwrap()["status"],
            "no_change_found"
        );
    }

    #[test]
    fn search_style_output_is_annotated_in_place() {
        let (fixture, db, project) = fixture("annotate");
        let decision = add(
            &db,
            &project,
            "La config d'auth vit dans src/lib/auth.ts.",
            "decision",
            "2026-10-02T09:00:00+00:00",
        );
        commit_file(
            &fixture.repo,
            "src/lib/auth.ts",
            "export const auth = v2();\n",
            "refactor(auth): split config",
            "2026-10-03T10:00:00+00:00",
        );
        let mut output = serde_json::json!({
            "results": [{ "id": decision, "content": "La config d'auth vit dans src/lib/auth.ts.", "kind": "decision" }]
        });
        db.annotate_verification(&mut output);
        assert_eq!(
            output["results"][0]["verification"]["status"],
            "changed_since"
        );
    }

    #[test]
    fn an_older_framework_version_than_the_repository_is_contradicted() {
        let (fixture, db, project) = fixture("contradicted");
        commit_file(
            &fixture.repo,
            "package.json",
            r#"{"name":"demo","devDependencies":{"tailwindcss":"^4.1.0","svelte":"^5.2.0"}}"#,
            "feat(web): tailwind 4",
            "2026-10-02T10:00:00+00:00",
        );
        let stale = add(
            &db,
            &project,
            "Le site utilise Tailwind 3 et Svelte 5.",
            "fact",
            "2026-10-03T09:00:00+00:00",
        );
        let plan = add(
            &db,
            &project,
            "Ne pas passer à Tailwind 5 avant la sortie stable.",
            "decision",
            "2026-10-03T09:00:00+00:00",
        );
        let history = add(
            &db,
            &project,
            "Passage à Tailwind 3 terminé.",
            "milestone",
            "2026-10-03T09:00:00+00:00",
        );

        let verification = db
            .verification(&db.get_memory(&stale).unwrap().unwrap(), false)
            .expect("reported");
        assert_eq!(verification["status"], "contradicted");
        assert_eq!(verification["claims"][0]["repo"], "tailwindcss ^4.1.0");
        assert!(db
            .verification(&db.get_memory(&plan).unwrap().unwrap(), false)
            .is_none());
        assert!(db
            .verification(&db.get_memory(&history).unwrap().unwrap(), false)
            .is_none());

        let facts = db.project_facts(&project).expect("facts");
        assert_eq!(
            facts["packages"][0]["dependencies"]["tailwindcss"],
            "^4.1.0"
        );
        assert_eq!(
            db.compact_project_facts(&project).unwrap()["stack"][0],
            "svelte ^5.2.0"
        );
    }

    fn memory(kind: &str, content: &str) -> crate::db::Memory {
        crate::db::Memory {
            id: "m".into(),
            content: content.into(),
            kind: kind.into(),
            project: None,
            tags: Vec::new(),
            source: "test".into(),
            importance: 3,
            expires_at: None,
            created_at: "2026-10-01T00:00:00+00:00".into(),
            updated_at: "2026-10-01T00:00:00+00:00".into(),
            metadata: None,
            last_accessed_at: None,
            access_count: 0,
        }
    }

    #[test]
    fn only_unresolved_problems_are_open() {
        use super::is_open_problem;
        assert!(is_open_problem(&memory(
            "bug",
            "Inscription ouverte dans auth.ts."
        )));
        assert!(!is_open_problem(&memory(
            "bug",
            "Import relevé: duplicate key. Fix: dedup org-wide."
        )));
        assert!(!is_open_problem(&memory(
            "bug",
            "Faille corrigée le 19/10 (commit 9ede2ce)."
        )));
        assert!(is_open_problem(&memory(
            "bug",
            "Rôle lecteur peut écrire : non corrigé."
        )));
        assert!(is_open_problem(&memory(
            "fact",
            "Le lecteur peut écrire : signalé, non corrigé."
        )));
        assert!(!is_open_problem(&memory("fact", "Le logo fait 900x429.")));
        assert!(!is_open_problem(&memory(
            "skill",
            "Non corrigé : procédure de build."
        )));
    }

    #[test]
    fn status_values_are_validated() {
        let (_fixture, db, project) = fixture("credential");
        let note = add(
            &db,
            &project,
            "Une note.",
            "note",
            "2026-10-02T09:00:00+00:00",
        );
        assert!(db.set_memory_status(&note, "closed", None, None).is_err());
        assert!(db
            .set_memory_status(&note, "verified", None, None)
            .unwrap()
            .is_some());
    }
}
