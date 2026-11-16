//! Merging and renaming projects, and moving one memory to another project.
//!
//! A project name is stored in `memories`, `memories_fts`, `episodes`,
//! `memories_archive`, the `project` entities of the knowledge graph,
//! `knowledge_triples` and `projects`: every change here covers them all in
//! one transaction, after a copy of the database has been taken.
//!
//! The vector index (`memory.db.ann.usearch`) is keyed by memory id and the
//! query cache (`memory.db.query_cache.sqlite`) by the hash of the query
//! text: neither stores a project, so neither needs an update. Only the
//! vectors of context-poor kinds, embedded under a `[project; date]` header,
//! depend on the name: those are re-embedded in the background.

use super::*;
use rusqlite::TransactionBehavior;

/// Copies of the database kept before a merge or a rename.
const PROJECT_BACKUPS_KEPT: usize = 10;
const PROJECT_BACKUP_PREFIX: &str = "memory-before-project-";

/// What a merge or a rename moves (dry run) or moved, per table.
#[derive(Debug, Clone, Serialize)]
pub struct ProjectMoveReport {
    pub operation: &'static str,
    pub source: String,
    pub target: String,
    pub dry_run: bool,
    pub memories: i64,
    pub memories_fts: i64,
    pub episodes: i64,
    pub memories_archive: i64,
    pub project_entities: i64,
    pub knowledge_triples: i64,
    pub aliases: i64,
    /// Merge only: memories whose text already exists in the target. Both
    /// copies are kept; `consolidate_memories` folds them on request.
    pub duplicate_contents: i64,
    /// Folder of the target project once the change is applied.
    pub path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub backup: Option<String>,
    /// Memories whose vector carries the project name, queued for re-embedding.
    pub reembed_queued: usize,
}

#[derive(Default)]
struct ProjectRow {
    path: String,
    description: Option<String>,
    remote: Option<String>,
    created_at: Option<String>,
}

impl Database {
    fn project_row(&self, name: &str) -> Option<ProjectRow> {
        self.conn
            .query_row(
                "SELECT path, description, remote, created_at FROM projects WHERE name = ?1",
                params![name],
                |row| {
                    Ok(ProjectRow {
                        path: row.get(0)?,
                        description: row.get(1)?,
                        remote: row.get(2)?,
                        created_at: row.get(3)?,
                    })
                },
            )
            .ok()
    }

    fn count(&self, sql: &str, bindings: &[&dyn rusqlite::ToSql]) -> Result<i64, String> {
        self.conn
            .query_row(sql, bindings, |row| row.get(0))
            .map_err(|error| format!("Project count: {}", error))
    }

    fn project_move_counts(
        &self,
        operation: &'static str,
        source: &str,
        target: &str,
        dry_run: bool,
    ) -> Result<ProjectMoveReport, String> {
        Ok(ProjectMoveReport {
            operation,
            source: source.to_string(),
            target: target.to_string(),
            dry_run,
            memories: self.count("SELECT COUNT(*) FROM memories WHERE project = ?1", &[&source])?,
            memories_fts: self.count(
                "SELECT COUNT(*) FROM memories_fts
                 WHERE rowid IN (SELECT rowid FROM memories WHERE project = ?1) OR project = ?1",
                &[&source],
            )?,
            episodes: self.count("SELECT COUNT(*) FROM episodes WHERE project = ?1", &[&source])?,
            memories_archive: self.count("SELECT COUNT(*) FROM memories_archive WHERE project = ?1", &[&source])?,
            project_entities: self.count(
                "SELECT COUNT(*) FROM memory_entities WHERE entity_kind = 'project' AND lower(entity_value) = ?1",
                &[&source],
            )?,
            knowledge_triples: self.count(
                "SELECT COUNT(*) FROM knowledge_triples
                 WHERE lower(trim(subject)) IN (?1, replace(?1, '-', '_'))
                    OR lower(trim(object)) IN (?1, replace(?1, '-', '_'))",
                &[&source],
            )?,
            aliases: self.count("SELECT COUNT(*) FROM project_aliases WHERE project = ?1", &[&source])?,
            duplicate_contents: self.count(
                "SELECT COUNT(*) FROM memories s
                 WHERE s.project = ?1 AND s.content_hash IS NOT NULL
                   AND EXISTS (SELECT 1 FROM memories t WHERE t.project = ?2 AND t.content_hash = s.content_hash)",
                &[&source, &target],
            )?,
            path: String::new(),
            backup: None,
            reembed_queued: 0,
        })
    }

    /// Whether a project name is in use anywhere a project is stored.
    fn project_has_data(&self, name: &str) -> Result<bool, String> {
        let used = self.count(
            "SELECT (SELECT COUNT(*) FROM projects WHERE name = ?1)
                  + (SELECT COUNT(*) FROM memories WHERE project = ?1)
                  + (SELECT COUNT(*) FROM episodes WHERE project = ?1)
                  + (SELECT COUNT(*) FROM memories_archive WHERE project = ?1)",
            &[&name],
        )?;
        Ok(used > 0)
    }

    /// Where the copies taken before a merge or a rename go: `backups/memory`
    /// for the user's database, a `backups` folder beside any other one.
    fn project_backups_dir(&self) -> Result<PathBuf, String> {
        let db_path = self.conn.path().map(PathBuf::from).ok_or("An in-memory database cannot be backed up")?;
        let is_user_database = crate::layout::memory_db()
            .and_then(|path| std::fs::canonicalize(path).ok())
            .zip(std::fs::canonicalize(&db_path).ok())
            .map(|(user, current)| user == current)
            .unwrap_or(false);
        if is_user_database {
            if let Some(backups) = crate::layout::dir(crate::layout::Area::Backups) {
                return Ok(backups.join("memory"));
            }
        }
        let parent = db_path.parent().ok_or("The database has no parent folder")?;
        Ok(parent.join("backups"))
    }

    /// Full copy of the database (`VACUUM INTO`), readable by its owner only.
    fn backup_before_project_change(&self, label: &str) -> Result<PathBuf, String> {
        let dir = self.project_backups_dir()?;
        std::fs::create_dir_all(&dir).map_err(|error| format!("Cannot create {}: {}", dir.display(), error))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700))
                .map_err(|error| format!("{}: {}", dir.display(), error))?;
        }
        let stamp = Utc::now().format("%Y%m%d-%H%M%S%3f");
        let target = dir.join(format!("{}{}-{}.db", PROJECT_BACKUP_PREFIX, label, stamp));
        self.conn
            .execute("VACUUM INTO ?1", params![target.to_string_lossy()])
            .map_err(|error| format!("Database backup to {}: {}", target.display(), error))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&target, std::fs::Permissions::from_mode(0o600))
                .map_err(|error| format!("{}: {}", target.display(), error))?;
        }
        let mut copies: Vec<PathBuf> = std::fs::read_dir(&dir)
            .map(|entries| {
                entries
                    .flatten()
                    .map(|entry| entry.path())
                    .filter(|path| {
                        path.file_name()
                            .and_then(|name| name.to_str())
                            .map(|name| name.starts_with(PROJECT_BACKUP_PREFIX) && name.ends_with(".db"))
                            .unwrap_or(false)
                    })
                    .collect()
            })
            .unwrap_or_default();
        // The timestamp in the name sorts them by age.
        copies.sort();
        let excess = copies.len().saturating_sub(PROJECT_BACKUPS_KEPT);
        for old in copies.into_iter().take(excess) {
            let _ = std::fs::remove_file(old);
        }
        Ok(target)
    }

    /// Moves everything stored under `source` to `target`, then deletes
    /// `source`, which stays known as an alias of `target`. With `dry_run`,
    /// only counts what would move. A copy of the database is taken first.
    pub fn merge_projects(&self, source: &str, target: &str, dry_run: bool) -> Result<ProjectMoveReport, String> {
        let source = Self::canonical_project_name(source).ok_or("source: project name required")?;
        let target = Self::canonical_project_name(target).ok_or("target: project name required")?;
        if source == target {
            return Err(format!("Source and target are the same project: {}", source));
        }
        if !self.project_has_data(&source)? {
            return Err(format!("Unknown project: {}", source));
        }
        self.relocate_project("merge", &source, &target, None, dry_run)
    }

    /// Gives a project a new name, and a new folder when `new_path` is set.
    /// The old name and the old folder stay known as aliases. Refused when
    /// the new name is already a project: that is a merge.
    pub fn rename_project(
        &self,
        old_name: &str,
        new_name: &str,
        new_path: Option<&str>,
        dry_run: bool,
    ) -> Result<ProjectMoveReport, String> {
        let source = Self::canonical_project_name(old_name).ok_or("old_name: project name required")?;
        let target = Self::canonical_project_name(new_name).ok_or("new_name: project name required")?;
        if !self.project_has_data(&source)? {
            return Err(format!("Unknown project: {}", source));
        }
        let new_path = new_path.map(str::trim).filter(|path| !path.is_empty());
        if source == target {
            let Some(path) = new_path else {
                return Err(format!("{} already has this name: give a new name or a new path", source));
            };
            if dry_run {
                let mut report = self.project_move_counts("rename", &source, &target, true)?;
                report.path = Self::identity_root(&Self::infer_project_root(path));
                return Ok(report);
            }
            let backup = self.backup_before_project_change(&format!("rename-{}", source))?;
            let project = self.register_project(&source, path, None)?;
            let mut report = self.project_move_counts("rename", &source, &target, false)?;
            report.path = project.path;
            report.backup = Some(backup.to_string_lossy().into_owned());
            return Ok(report);
        }
        if self.project_has_data(&target)? {
            return Err(format!(
                "Project {} already exists: use merge_projects (source {}, target {}) to fold one into the other",
                target, source, target
            ));
        }
        self.relocate_project("rename", &source, &target, new_path, dry_run)
    }

    fn relocate_project(
        &self,
        operation: &'static str,
        source: &str,
        target: &str,
        new_path: Option<&str>,
        dry_run: bool,
    ) -> Result<ProjectMoveReport, String> {
        let mut report = self.project_move_counts(operation, source, target, dry_run)?;
        let source_row = self.project_row(source).unwrap_or_default();
        let target_row = self.project_row(target).unwrap_or_default();
        let chosen_path = new_path.map(|path| Self::identity_root(&Self::infer_project_root(path)));
        let final_path = chosen_path
            .clone()
            .or_else(|| Some(target_row.path.clone()).filter(|path| !path.is_empty()))
            .unwrap_or_else(|| source_row.path.clone());
        let final_remote = chosen_path
            .as_deref()
            .and_then(|path| git_identity::origin_remote(Path::new(path)))
            .or(target_row.remote.clone())
            .or(source_row.remote.clone());
        report.path = final_path.clone();
        if dry_run {
            return Ok(report);
        }

        let backup = self.backup_before_project_change(&format!("{}-{}-into-{}", operation, source, target))?;
        report.backup = Some(backup.to_string_lossy().into_owned());

        let now = Utc::now().to_rfc3339();
        let created_at = [source_row.created_at.clone(), target_row.created_at.clone()]
            .into_iter()
            .flatten()
            .min()
            .unwrap_or_else(|| now.clone());
        let description = target_row.description.clone().or(source_row.description.clone());

        let transaction = rusqlite::Transaction::new_unchecked(&self.conn, TransactionBehavior::Immediate)
            .map_err(|error| format!("Project {} begin: {}", operation, error))?;
        let step = |sql: &str, bindings: &[&dyn rusqlite::ToSql]| -> Result<usize, String> {
            transaction
                .execute(sql, bindings)
                .map_err(|error| format!("Project {} ({}): {}", operation, sql.split_whitespace().take(3).collect::<Vec<_>>().join(" "), error))
        };

        // Vectors of context-poor kinds carry the project name in their header.
        let reembed_ids: Vec<String> = {
            let mut statement = transaction
                .prepare(&format!(
                    "SELECT id FROM memories WHERE project = ?1 AND kind IN ({})",
                    CONTEXT_POOR_KINDS.iter().map(|kind| format!("'{}'", kind)).collect::<Vec<_>>().join(",")
                ))
                .map_err(|error| format!("Project {} re-embed list: {}", operation, error))?;
            let ids = statement
                .query_map(params![source], |row| row.get(0))
                .map_err(|error| format!("Project {} re-embed list: {}", operation, error))?
                .flatten()
                .collect();
            ids
        };

        step(
            "UPDATE memories_fts SET project = ?2
             WHERE rowid IN (SELECT rowid FROM memories WHERE project = ?1) OR project = ?1",
            &[&source, &target],
        )?;
        step(
            &format!(
                "UPDATE memories SET embedding = NULL WHERE project = ?1 AND kind IN ({})",
                CONTEXT_POOR_KINDS.iter().map(|kind| format!("'{}'", kind)).collect::<Vec<_>>().join(",")
            ),
            &[&source],
        )?;
        step("UPDATE memories SET project = ?2 WHERE project = ?1", &[&source, &target])?;
        step("UPDATE episodes SET project = ?2 WHERE project = ?1", &[&source, &target])?;
        step("UPDATE memories_archive SET project = ?2 WHERE project = ?1", &[&source, &target])?;
        // A memory that already names the target keeps a single project entity.
        step(
            "DELETE FROM memory_entities
             WHERE entity_kind = 'project' AND lower(entity_value) = ?1
               AND memory_id IN (SELECT memory_id FROM memory_entities WHERE entity_kind = 'project' AND entity_value = ?2)",
            &[&source, &target],
        )?;
        step(
            "UPDATE memory_entities SET entity_value = ?2 WHERE entity_kind = 'project' AND lower(entity_value) = ?1",
            &[&source, &target],
        )?;
        // Triples spell a multi-word name with underscores, projects with hyphens.
        step(
            "UPDATE knowledge_triples SET subject = replace(?2, '-', '_')
             WHERE lower(trim(subject)) IN (?1, replace(?1, '-', '_'))",
            &[&source, &target],
        )?;
        step(
            "UPDATE knowledge_triples SET object = replace(?2, '-', '_')
             WHERE lower(trim(object)) IN (?1, replace(?1, '-', '_'))",
            &[&source, &target],
        )?;

        step(
            "INSERT INTO projects (name, path, description, remote, created_at) VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(name) DO UPDATE SET
                path = excluded.path, description = excluded.description,
                remote = excluded.remote, created_at = excluded.created_at",
            &[&target, &final_path, &description, &final_remote, &created_at],
        )?;
        step("UPDATE project_aliases SET project = ?2 WHERE project = ?1", &[&source, &target])?;
        Self::add_project_alias(&transaction, "name", source, target)?;
        for former_path in [&source_row.path, &target_row.path] {
            if *former_path != final_path {
                Self::add_project_alias(&transaction, "path", former_path, target)?;
            }
        }
        step(
            "DELETE FROM project_aliases WHERE (kind = 'name' AND alias = ?1) OR (kind = 'path' AND alias = ?2)",
            &[&target, &final_path],
        )?;
        step("DELETE FROM projects WHERE name = ?1", &[&source])?;
        transaction
            .commit()
            .map_err(|error| format!("Project {} commit: {}", operation, error))?;

        report.reembed_queued = self.queue_reembedding(&reembed_ids);
        Ok(report)
    }

    /// Queues memories whose vector was cleared because its header named
    /// another project. A memory the worker does not reach before the
    /// process ends is picked up by the next startup backfill.
    fn queue_reembedding(&self, ids: &[String]) -> usize {
        let mut queued = 0;
        for id in ids {
            let row: Option<(String, String, Option<String>, String)> = self
                .conn
                .query_row(
                    "SELECT content, kind, project, created_at FROM memories WHERE id = ?1",
                    params![id],
                    |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
                )
                .ok();
            if let Some((content, kind, project, created_at)) = row {
                queue_embedding_job(id, &content, &Self::embedding_input(&kind, project.as_deref(), &created_at, &content));
                queued += 1;
            }
        }
        queued
    }

    /// Moves one memory to another project (`None`: the global scope),
    /// keeping the full-text index and the project entity in step. Returns
    /// `None` when the memory does not exist.
    pub fn move_memory_to_project(&self, id: &str, project: Option<&str>) -> Result<Option<Memory>, String> {
        let Some(mut memory) = self.fetch_memory(id)? else {
            return Ok(None);
        };
        let target = Self::canonical_project(project);
        if memory.project == target {
            return Ok(Some(memory));
        }
        if let Some(name) = target.as_deref() {
            self.ensure_project(name)?;
        }
        let reembed = CONTEXT_POOR_KINDS.contains(&memory.kind.as_str());
        let now = Utc::now().to_rfc3339();
        let transaction = rusqlite::Transaction::new_unchecked(&self.conn, TransactionBehavior::Immediate)
            .map_err(|error| format!("Move memory begin: {}", error))?;
        transaction
            .execute(
                "UPDATE memories_fts SET project = ?1 WHERE rowid = (SELECT rowid FROM memories WHERE id = ?2)",
                params![target.as_deref().unwrap_or(""), id],
            )
            .map_err(|error| format!("Move memory (full-text index): {}", error))?;
        transaction
            .execute(
                "UPDATE memories SET project = ?1, updated_at = ?2,
                        embedding = CASE WHEN ?3 THEN NULL ELSE embedding END
                 WHERE id = ?4",
                params![target, now, reembed, id],
            )
            .map_err(|error| format!("Move memory: {}", error))?;
        transaction
            .execute(
                "DELETE FROM memory_entities WHERE memory_id = ?1 AND entity_kind = 'project'",
                params![id],
            )
            .map_err(|error| format!("Move memory (entities): {}", error))?;
        if let Some(name) = target.as_deref() {
            transaction
                .execute(
                    "INSERT INTO memory_entities (memory_id, entity_kind, entity_value) VALUES (?1, 'project', ?2)",
                    params![id, name],
                )
                .map_err(|error| format!("Move memory (entities): {}", error))?;
        }
        transaction
            .commit()
            .map_err(|error| format!("Move memory commit: {}", error))?;
        memory.project = target;
        memory.updated_at = now;
        if memory.kind != "credential" {
            // Links follow shared entities, the project among them. A
            // credential is read masked: its links are left as they are.
            rebuild_memory_links(self, &memory)?;
        }
        if reembed {
            self.queue_reembedding(&[id.to_string()]);
        }
        Ok(Some(memory))
    }
}
