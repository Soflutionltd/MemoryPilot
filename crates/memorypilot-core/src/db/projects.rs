//! Project identity: canonical names, roots and paths, registration and detection.

use super::*;

impl Database {
    pub(super) fn canonical_project_name(name: &str) -> Option<String> {
        let trimmed = name.trim();
        if trimmed.is_empty() {
            return None;
        }

        let mut slug = String::new();
        let mut previous_was_separator = false;

        for character in trimmed.chars() {
            if character.is_ascii_alphanumeric() {
                slug.push(character.to_ascii_lowercase());
                previous_was_separator = false;
            } else if !slug.is_empty() && !previous_was_separator {
                slug.push('-');
                previous_was_separator = true;
            }
        }

        while slug.ends_with('-') {
            slug.pop();
        }

        // "global", "__global__", "none"… are how callers spell "no project":
        // they must land in the NULL (global) scope, not in a project of
        // that name. normalize_project_identities() migrates old rows.
        if slug.is_empty() || matches!(slug.as_str(), "global" | "none" | "null") {
            None
        } else {
            Some(slug)
        }
    }

    pub(super) fn canonical_project(project: Option<&str>) -> Option<String> {
        project.and_then(Self::canonical_project_name)
    }

    pub(super) fn normalize_path(path: &str) -> String {
        let trimmed = path.trim();
        if trimmed.is_empty() {
            return String::new();
        }

        let canonical = std::fs::canonicalize(trimmed).unwrap_or_else(|_| PathBuf::from(trimmed));
        let rendered = canonical.to_string_lossy().replace('\\', "/");
        if rendered == "/" {
            rendered
        } else {
            rendered.trim_end_matches('/').to_string()
        }
    }

    pub(super) fn infer_project_root(working_dir: &str) -> String {
        let normalized_dir = Self::normalize_path(working_dir);
        if normalized_dir.is_empty() {
            return normalized_dir;
        }

        let normalized_path = PathBuf::from(&normalized_dir);
        let search_start = if normalized_path.is_file() {
            normalized_path
                .parent()
                .map(Path::to_path_buf)
                .unwrap_or_else(|| normalized_path.clone())
        } else {
            normalized_path.clone()
        };

        for ancestor in search_start.ancestors() {
            if ancestor.join(".git").exists()
                || ancestor.join("Cargo.toml").exists()
                || ancestor.join("package.json").exists()
                || ancestor.join("pyproject.toml").exists()
                || ancestor.join("go.mod").exists()
            {
                return Self::normalize_path(&ancestor.to_string_lossy());
            }
        }

        normalized_dir
    }

    /// Names a root may go by, most specific first: Cargo package, npm
    /// package, folder (`MemoryPilot` in Cargo.toml and on disk).
    pub(super) fn project_slug_candidates(project_root: &str) -> Vec<String> {
        let root = Path::new(project_root);
        let mut candidates: Vec<String> = Vec::new();

        let cargo_manifest = root.join("Cargo.toml");
        if cargo_manifest.exists() {
            if let Ok(content) = std::fs::read_to_string(&cargo_manifest) {
                let mut in_package_section = false;
                for line in content.lines() {
                    let trimmed = line.trim();
                    if trimmed.starts_with('[') {
                        in_package_section = trimmed == "[package]";
                        continue;
                    }
                    if in_package_section && trimmed.starts_with("name") {
                        if let Some((_, raw_value)) = trimmed.split_once('=') {
                            if let Some(slug) =
                                Self::canonical_project_name(raw_value.trim().trim_matches('"'))
                            {
                                candidates.push(slug);
                                break;
                            }
                        }
                    }
                }
            }
        }

        let package_manifest = root.join("package.json");
        if package_manifest.exists() {
            if let Ok(content) = std::fs::read_to_string(&package_manifest) {
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
                    if let Some(name) = json.get("name").and_then(|value| value.as_str()) {
                        if let Some(slug) = Self::canonical_project_name(name) {
                            candidates.push(slug);
                        }
                    }
                }
            }
        }

        if let Some(slug) = root.file_name().and_then(|name| name.to_str()).and_then(Self::canonical_project_name) {
            candidates.push(slug);
        }
        candidates.dedup();
        candidates
    }

    fn project_is_known(&self, project_name: &str) -> bool {
        self.conn
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM projects WHERE name = ?1)
                     OR EXISTS(SELECT 1 FROM memories WHERE project = ?1)",
                params![project_name],
                |row| row.get(0),
            )
            .unwrap_or(false)
    }

    /// The home folder, its parents and the temp folders hold every project:
    /// such a path is never the root of one.
    pub(super) fn is_generic_folder(path: &str) -> bool {
        if path.is_empty() || path == "/" {
            return true;
        }
        if let Some(home) = dirs::home_dir().map(|home| Self::normalize_path(&home.to_string_lossy())) {
            if path == home || Self::path_matches(&home, path) {
                return true;
            }
        }
        let temp = Self::normalize_path(&std::env::temp_dir().to_string_lossy());
        [temp.as_str(), "/tmp", "/private/tmp", "/var/tmp", "/private/var/tmp"].contains(&path)
    }

    pub(super) fn path_matches(working_dir: &str, project_path: &str) -> bool {
        if project_path.is_empty() {
            return false;
        }

        working_dir == project_path || working_dir.starts_with(&format!("{project_path}/"))
    }

    pub(super) fn remember_project_path_if_known(&self, project_name: &str, path: &str) -> Result<(), String> {
        if project_name.is_empty() || path.is_empty() {
            return Ok(());
        }

        if !self.project_is_known(project_name) {
            return Ok(());
        }

        let now = Utc::now().to_rfc3339();
        let remote = git_identity::origin_remote(Path::new(path));
        self.conn
            .execute(
                "INSERT INTO projects (name, path, remote, created_at) VALUES (?1, ?2, ?3, ?4)
                 ON CONFLICT(name) DO UPDATE SET
                    path = CASE
                        WHEN projects.path = '' AND excluded.path != '' THEN excluded.path
                        ELSE projects.path
                    END,
                    remote = COALESCE(projects.remote, excluded.remote)",
                params![project_name, path, remote, now],
            )
            .map_err(|error| format!("Remember path: {}", error))?;
        Ok(())
    }

    /// The folder that names the project `project_root` belongs to: the main
    /// repository for a linked worktree (a worktree is never a project of
    /// its own), the root itself otherwise.
    pub(super) fn identity_root(project_root: &str) -> String {
        match git_identity::main_worktree_root(Path::new(project_root)) {
            Some(main_root) => Self::normalize_path(&main_root.to_string_lossy()),
            None => project_root.to_string(),
        }
    }

    /// The project an old name or an old path now points to.
    pub(super) fn project_for_alias(&self, kind: &str, alias: &str) -> Option<String> {
        self.conn
            .query_row(
                "SELECT project FROM project_aliases WHERE kind = ?1 AND alias = ?2",
                params![kind, alias],
                |row| row.get(0),
            )
            .ok()
    }

    /// The project registered with this remote, or the one a merged project
    /// had it; the one holding the most memories when several share it
    /// (get_memory_health reports those).
    fn project_for_remote(&self, remote: &str) -> Option<String> {
        self.conn
            .query_row(
                "SELECT p.name FROM projects p
                 WHERE p.remote = ?1
                 ORDER BY (SELECT COUNT(*) FROM memories m WHERE m.project = p.name) DESC, p.created_at, p.name
                 LIMIT 1",
                params![remote],
                |row| row.get(0),
            )
            .ok()
            .or_else(|| self.project_for_alias("remote", remote))
    }

    /// Records that `alias` (an old name, an old path or the remote of a
    /// merged project) means `project`.
    /// A name or a path a live project still uses is never an alias.
    pub(super) fn add_project_alias(
        conn: &Connection,
        kind: &str,
        alias: &str,
        project: &str,
    ) -> Result<(), String> {
        if alias.is_empty() || (kind == "name" && alias == project) {
            return Ok(());
        }
        conn.execute(
            "INSERT INTO project_aliases (alias, kind, project, created_at) VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(kind, alias) DO UPDATE SET project = excluded.project",
            params![alias, kind, project, Utc::now().to_rfc3339()],
        )
        .map_err(|error| format!("Project alias: {}", error))?;
        Ok(())
    }

    /// Fills the remote of every project whose folder exists and has one.
    /// A remote already stored is kept: it identifies the project even after
    /// its folder moved.
    pub(super) fn refresh_project_remotes(&self) {
        let Ok(mut statement) = self
            .conn
            .prepare("SELECT name, path FROM projects WHERE path != '' AND remote IS NULL")
        else {
            return;
        };
        let rows: Vec<(String, String)> = statement
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
            .map(|rows| rows.flatten().collect())
            .unwrap_or_default();
        for (name, path) in rows {
            if let Some(remote) = git_identity::origin_remote(Path::new(&path)) {
                let _ = self
                    .conn
                    .execute("UPDATE projects SET remote = ?1 WHERE name = ?2", params![remote, name]);
            }
        }
    }

    pub(super) fn normalize_project_identities(&self) -> Result<(), String> {
        #[derive(Clone)]
        struct CanonicalProjectRecord {
            path: String,
            description: Option<String>,
            created_at: String,
        }

        let now = Utc::now().to_rfc3339();
        let mut canonical_projects: std::collections::BTreeMap<String, CanonicalProjectRecord> =
            std::collections::BTreeMap::new();
        let mut project_names_to_delete: std::collections::BTreeSet<String> =
            std::collections::BTreeSet::new();

        {
            let mut statement = self
                .conn
                .prepare("SELECT name, path, description, created_at FROM projects")
                .map_err(|error| format!("Normalize projects: {}", error))?;
            let rows = statement
                .query_map([], |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, Option<String>>(2)?,
                        row.get::<_, String>(3)?,
                    ))
                })
                .map_err(|error| format!("Normalize projects: {}", error))?;

            for row in rows.flatten() {
                let (name, path, description, created_at) = row;
                match Self::canonical_project_name(&name) {
                    Some(slug) => {
                        let normalized_path = Self::normalize_path(&path);
                        let entry = canonical_projects.entry(slug.clone()).or_insert_with(|| {
                            CanonicalProjectRecord {
                                path: normalized_path.clone(),
                                description: description
                                    .clone()
                                    .filter(|value| !value.trim().is_empty()),
                                created_at: created_at.clone(),
                            }
                        });

                        if entry.path.is_empty() && !normalized_path.is_empty() {
                            entry.path = normalized_path.clone();
                        }
                        if entry.description.is_none() {
                            entry.description =
                                description.clone().filter(|value| !value.trim().is_empty());
                        }
                        if created_at < entry.created_at {
                            entry.created_at = created_at.clone();
                        }

                        if slug != name || normalized_path != path {
                            project_names_to_delete.insert(name);
                        }
                    }
                    None => {
                        project_names_to_delete.insert(name);
                    }
                }
            }
        }

        let mut distinct_memory_projects: Vec<String> = Vec::new();
        {
            let mut statement = self
                .conn
                .prepare("SELECT DISTINCT project FROM memories WHERE project IS NOT NULL AND trim(project) != ''")
                .map_err(|error| format!("Normalize memory projects: {}", error))?;
            let rows = statement
                .query_map([], |row| row.get::<_, String>(0))
                .map_err(|error| format!("Normalize memory projects: {}", error))?;
            for row in rows.flatten() {
                distinct_memory_projects.push(row);
            }
        }

        for raw_project in &distinct_memory_projects {
            if let Some(slug) = Self::canonical_project_name(raw_project) {
                canonical_projects
                    .entry(slug.clone())
                    .or_insert_with(|| CanonicalProjectRecord {
                        path: String::new(),
                        description: None,
                        created_at: now.clone(),
                    });
                if slug != *raw_project {
                    project_names_to_delete.insert(raw_project.clone());
                }
            }
        }

        if canonical_projects.is_empty() && project_names_to_delete.is_empty() {
            return Ok(());
        }

        let transaction = self
            .conn
            .unchecked_transaction()
            .map_err(|error| format!("Normalize transaction: {}", error))?;

        for raw_project in &distinct_memory_projects {
            match Self::canonical_project_name(raw_project) {
                Some(slug) if slug != *raw_project => {
                    transaction
                        .execute(
                            "UPDATE memories SET project = ?1 WHERE project = ?2",
                            params![slug, raw_project],
                        )
                        .map_err(|error| format!("Normalize memories: {}", error))?;
                    transaction
                        .execute(
                            "UPDATE memories_fts SET project = ?1 WHERE project = ?2",
                            params![slug, raw_project],
                        )
                        .map_err(|error| format!("Normalize memories_fts: {}", error))?;
                }
                Some(_) => {}
                None => {
                    transaction
                        .execute(
                            "UPDATE memories SET project = NULL WHERE project = ?1",
                            params![raw_project],
                        )
                        .map_err(|error| format!("Nullify memories project: {}", error))?;
                    transaction
                        .execute(
                            "UPDATE memories_fts SET project = '' WHERE project = ?1",
                            params![raw_project],
                        )
                        .map_err(|error| format!("Nullify memories_fts project: {}", error))?;
                }
            }
        }

        for (slug, record) in &canonical_projects {
            transaction
                .execute(
                    "INSERT INTO projects (name, path, description, created_at) VALUES (?1, ?2, ?3, ?4)
                     ON CONFLICT(name) DO UPDATE SET
                        path = CASE
                            WHEN excluded.path != '' THEN excluded.path
                            ELSE projects.path
                        END,
                        description = COALESCE(projects.description, excluded.description)",
                    params![slug, record.path, record.description, record.created_at],
                )
                .map_err(|error| format!("Upsert canonical project: {}", error))?;
        }

        for legacy_name in project_names_to_delete {
            if Self::canonical_project_name(&legacy_name).as_deref() != Some(legacy_name.as_str()) {
                transaction
                    .execute("DELETE FROM projects WHERE name = ?1", params![legacy_name])
                    .map_err(|error| format!("Delete legacy project: {}", error))?;
            }
        }

        transaction
            .commit()
            .map_err(|error| format!("Normalize commit: {}", error))?;
        Ok(())
    }

    // ─── PROJECTS ─────────────────────────────────────

    pub(super) fn ensure_project(&self, name: &str) -> Result<(), String> {
        let Some(canonical_name) = Self::canonical_project_name(name) else {
            return Ok(());
        };
        let now = Utc::now().to_rfc3339();
        self.conn
            .execute(
                "INSERT OR IGNORE INTO projects (name,path,created_at) VALUES (?1,'',?2)",
                params![canonical_name, now],
            )
            .map_err(|e| format!("Ensure: {}", e))?;
        Ok(())
    }

    /// Creates the project or updates its folder. A linked worktree is
    /// registered as its main repository; the `origin` remote is stored, and
    /// a folder the project leaves stays known as an alias.
    pub fn register_project(
        &self,
        name: &str,
        path: &str,
        description: Option<&str>,
    ) -> Result<Project, String> {
        let canonical_name =
            Self::canonical_project_name(name).ok_or("Project name cannot be empty")?;
        let project_root = Self::identity_root(&Self::infer_project_root(path));
        let remote = git_identity::origin_remote(Path::new(&project_root));
        let now = Utc::now().to_rfc3339();
        let previous_path: Option<String> = self
            .conn
            .query_row("SELECT path FROM projects WHERE name = ?1", params![&canonical_name], |row| row.get(0))
            .ok();
        self.conn
            .execute(
                "INSERT INTO projects (name,path,description,remote,created_at) VALUES (?1,?2,?3,?4,?5)
             ON CONFLICT(name) DO UPDATE SET path=?2, description=COALESCE(?3,description), remote=COALESCE(?4,remote)",
                params![&canonical_name, &project_root, description, remote, &now],
            )
            .map_err(|e| format!("Register: {}", e))?;
        if let Some(previous_path) = previous_path.filter(|previous| *previous != project_root) {
            Self::add_project_alias(&self.conn, "path", &previous_path, &canonical_name)?;
        }
        self.conn
            .execute(
                "DELETE FROM project_aliases WHERE (kind = 'path' AND alias = ?1) OR (kind = 'name' AND alias = ?2)",
                params![&project_root, &canonical_name],
            )
            .map_err(|e| format!("Register aliases: {}", e))?;
        let count: i64 = self
            .conn
            .query_row(
                "SELECT COUNT(*) FROM memories WHERE project=?1",
                params![&canonical_name],
                |r| r.get(0),
            )
            .unwrap_or(0);
        Ok(Project {
            name: canonical_name,
            path: project_root,
            description: description.map(String::from),
            created_at: now,
            memory_count: count,
        })
    }

    pub fn list_projects(&self) -> Result<Vec<Project>, String> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT p.name, p.path, p.description, p.created_at, COUNT(m.id) as cnt
             FROM projects p LEFT JOIN memories m ON m.project = p.name
             GROUP BY p.name ORDER BY cnt DESC",
            )
            .map_err(|e| format!("List projects: {}", e))?;
        let projects = stmt
            .query_map([], |row| {
                Ok(Project {
                    name: row.get(0)?,
                    path: row.get(1)?,
                    description: row.get(2)?,
                    created_at: row.get(3)?,
                    memory_count: row.get(4)?,
                })
            })
            .map_err(|e| format!("Projects: {}", e))?
            .filter_map(|r| r.ok())
            .collect();
        Ok(projects)
    }

    /// The project a working directory belongs to, in this order: a
    /// registered folder, a folder the project used to live in, the `origin`
    /// remote of the repository, then the name the folder goes by. An old
    /// name follows its alias even when a project was created under it
    /// again: that keeps a duplicate from growing, and get_memory_health
    /// reports the pair. A linked worktree answers for its main repository.
    pub fn detect_project(&self, working_dir: &str) -> Result<Option<String>, String> {
        let normalized_dir = Self::normalize_path(working_dir);
        if normalized_dir.is_empty() {
            return Ok(None);
        }
        let project_root = Self::infer_project_root(&normalized_dir);
        let identity_root = Self::identity_root(&project_root);
        let matches_folder = |folder: &str| {
            [&project_root, &normalized_dir, &identity_root]
                .iter()
                .any(|candidate| Self::path_matches(candidate, folder))
        };

        let folders = |sql: &str| -> Result<Vec<(String, String)>, String> {
            let mut statement = self.conn.prepare(sql).map_err(|e| format!("Detect: {}", e))?;
            let rows = statement
                .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
                .map_err(|e| format!("Detect2: {}", e))?
                .filter_map(|r| r.ok())
                .collect();
            Ok(rows)
        };
        let registered = folders("SELECT name, path FROM projects WHERE path != '' ORDER BY length(path) DESC")?;
        let former = folders("SELECT project, alias FROM project_aliases WHERE kind = 'path' ORDER BY length(alias) DESC")?;
        for (name, path) in registered.iter().chain(former.iter()) {
            let normalized_path = Self::normalize_path(path);
            if Self::is_generic_folder(&normalized_path) {
                continue;
            }
            if matches_folder(&normalized_path) {
                return Ok(Some(name.clone()));
            }
        }
        if Self::is_generic_folder(&identity_root) {
            return Ok(None);
        }
        if let Some(project_name) = git_identity::origin_remote(Path::new(&identity_root))
            .and_then(|remote| self.project_for_remote(&remote))
        {
            return Ok(Some(project_name));
        }
        let candidates: Vec<String> = Self::project_slug_candidates(&identity_root)
            .into_iter()
            .map(|candidate| self.project_for_alias("name", &candidate).unwrap_or(candidate))
            .collect();
        let project_name = candidates
            .iter()
            .find(|candidate| self.project_is_known(candidate))
            .or(candidates.first())
            .cloned();
        if let Some(ref project_name) = project_name {
            let _ = self.remember_project_path_if_known(project_name, &identity_root);
        }
        Ok(project_name)
    }
}
