//! Where MemoryPilot keeps its files under `~/.MemoryPilot/`.
//!
//! ```text
//! data/      memory.db, conversations.db and their WAL, index and cache files
//! state/     what changes at every session (hook state, injections, marks, update)
//! config/    http.token, credentials.key, known-secrets.sha256, GLOBAL_PROMPT.md
//! logs/      every log file
//! backups/   memory/ (database copies before an update), and the user's own
//! archive/   archived transcripts, v1 files, old copies kept out of the way
//! agents/    rules, skills and plugins (see agents.rs)
//! ```
//!
//! Until `MemoryPilot --migrate-layout --apply` has run, files are still at
//! the root: every lookup takes the new path when it exists, else the old
//! one when only it exists, so an old and a new layout both work. The
//! migration leaves symbolic links at the old paths for older binaries and
//! for McpHub.

use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Area {
    Data,
    State,
    Config,
    Logs,
    Backups,
    Archive,
}

impl Area {
    pub fn folder(self) -> &'static str {
        match self {
            Area::Data => "data",
            Area::State => "state",
            Area::Config => "config",
            Area::Logs => "logs",
            Area::Backups => "backups",
            Area::Archive => "archive",
        }
    }
}

/// `~/.MemoryPilot`, or `MEMORYPILOT_HOME` when set (tests, other machines).
pub fn root() -> Option<PathBuf> {
    match std::env::var("MEMORYPILOT_HOME") {
        Ok(path) if !path.trim().is_empty() => Some(PathBuf::from(path)),
        _ => dirs::home_dir().map(|home| home.join(".MemoryPilot")),
    }
}

pub fn dir_in(root: &Path, area: Area) -> PathBuf {
    root.join(area.folder())
}

/// The path of `name` in `area`: the new one, unless only the old root-level
/// file (or folder) exists yet.
pub fn path_in(root: &Path, area: Area, name: &str) -> PathBuf {
    let new = dir_in(root, area).join(name);
    let legacy = root.join(name);
    let exists = |p: &Path| std::fs::symlink_metadata(p).is_ok();
    if !exists(&new) && exists(&legacy) && !is_link(&legacy) {
        return legacy;
    }
    new
}

fn is_link(path: &Path) -> bool {
    std::fs::symlink_metadata(path).map(|m| m.file_type().is_symlink()).unwrap_or(false)
}

pub fn dir(area: Area) -> Option<PathBuf> {
    root().map(|root| dir_in(&root, area))
}

pub fn path(area: Area, name: &str) -> Option<PathBuf> {
    root().map(|root| path_in(&root, area, name))
}

/// `MEMORYPILOT_DB`, else `data/memory.db`.
pub fn memory_db() -> Option<PathBuf> {
    match std::env::var("MEMORYPILOT_DB") {
        Ok(path) if !path.trim().is_empty() => Some(PathBuf::from(path)),
        _ => path(Area::Data, "memory.db"),
    }
}

/// `MEMORYPILOT_CONVERSATIONS_DB`, else `data/conversations.db`.
pub fn conversations_db() -> Option<PathBuf> {
    match std::env::var("MEMORYPILOT_CONVERSATIONS_DB") {
        Ok(path) if !path.trim().is_empty() => Some(PathBuf::from(path)),
        _ => path(Area::Data, "conversations.db"),
    }
}

/// Folder for files that live next to a database chosen by environment
/// variable (tests point `MEMORYPILOT_DB` at a temporary file), else `area`.
pub fn beside_db_or(area: Area) -> Option<PathBuf> {
    for variable in ["MEMORYPILOT_CONVERSATIONS_DB", "MEMORYPILOT_DB"] {
        if let Ok(path) = std::env::var(variable) {
            if !path.trim().is_empty() {
                return Path::new(&path).parent().map(Path::to_path_buf);
            }
        }
    }
    dir(area)
}

/// One move of `--migrate-layout`: `from` (root level) to `to`.
#[derive(Debug, Clone, PartialEq)]
pub struct Move {
    pub from: PathBuf,
    pub to: PathBuf,
    /// Leave a symbolic link at `from` (read by older binaries or McpHub).
    pub link_back: bool,
}

/// What `--migrate-layout` does with each root entry. Unknown entries stay.
pub fn plan(root: &Path) -> Vec<Move> {
    let mut moves = Vec::new();
    let Ok(entries) = std::fs::read_dir(root) else { return moves };
    let mut names: Vec<String> = entries
        .flatten()
        .filter(|e| !is_link(&e.path()))
        .filter_map(|e| e.file_name().into_string().ok())
        .collect();
    names.sort();
    for name in names {
        let target = |area: Area, sub: Option<&str>| {
            let base = dir_in(root, area);
            match sub {
                Some(sub) => base.join(sub).join(&name),
                None => base.join(&name),
            }
        };
        let (to, link_back) = if is_db_file(&name, "memory.db") || is_db_file(&name, "conversations.db") {
            (target(Area::Data, None), true)
        } else if name.starts_with("memory.db.bak-") || name.starts_with("memory.db.bak") {
            (target(Area::Backups, Some("memory")), false)
        } else {
            match name.as_str() {
                "cursor-hooks-state.json" | "session-hook-injections.json" | "session-start-marks" | "update" => {
                    (target(Area::State, None), true)
                }
                "http.token" | "credentials.key" | "known-secrets.sha256" | "GLOBAL_PROMPT.md" | "remote.host" => {
                    (target(Area::Config, None), true)
                }
                "cursor-hooks.log" | "agents-sync.log" => (target(Area::Logs, None), false),
                "global.json" | "projects.json" | "projects" | "config.json" => (target(Area::Archive, Some("v1")), false),
                _ if name.ends_with(".bak") || name.contains(".archive-") => (target(Area::Archive, None), false),
                _ => continue,
            }
        };
        moves.push(Move { from: root.join(&name), to, link_back });
    }
    moves
}

/// `memory.db`, `memory.db-wal`, `memory.db-shm`, `memory.db.ann.usearch`,
/// `memory.db.ann.usearch.keys.json`, `memory.db.query_cache.sqlite`.
fn is_db_file(name: &str, db: &str) -> bool {
    name == db
        || name == format!("{}-wal", db)
        || name == format!("{}-shm", db)
        || (name.starts_with(&format!("{}.", db)) && !name.starts_with(&format!("{}.bak", db)))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TempRoot(PathBuf);

    impl Drop for TempRoot {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn temp_root() -> TempRoot {
        static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("memorypilot-layout-{}-{}", std::process::id(), n));
        std::fs::create_dir_all(&dir).unwrap();
        TempRoot(dir)
    }

    #[test]
    fn new_path_unless_only_the_old_one_exists() {
        let root = temp_root();
        let r = &root.0;
        assert_eq!(path_in(r, Area::Config, "http.token"), r.join("config/http.token"));
        std::fs::write(r.join("http.token"), "t").unwrap();
        assert_eq!(path_in(r, Area::Config, "http.token"), r.join("http.token"), "not migrated yet");
        std::fs::create_dir_all(r.join("config")).unwrap();
        std::fs::rename(r.join("http.token"), r.join("config/http.token")).unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink(r.join("config/http.token"), r.join("http.token")).unwrap();
        assert_eq!(path_in(r, Area::Config, "http.token"), r.join("config/http.token"), "migrated, old path is a link");
    }

    #[test]
    fn plan_sorts_every_known_entry_and_leaves_the_rest() {
        let root = temp_root();
        let r = &root.0;
        for name in [
            "memory.db", "memory.db-wal", "memory.db-shm", "memory.db.ann.usearch", "memory.db.ann.usearch.keys.json",
            "memory.db.query_cache.sqlite", "memory.db.bak-20261025-1444-prebuild", "memory.db.bak-v4.10-20261028-0123-shm",
            "conversations.db", "conversations.db-wal", "http.token", "cursor-hooks-state.json", "cursor-hooks-state.json.bak",
            "session-hook-injections.json", "cursor-hooks.log", "global.json", "projects.json", "config.json",
            "GLOBAL_PROMPT.md.archive-2026-10-31", "unknown.txt",
        ] {
            std::fs::write(r.join(name), "x").unwrap();
        }
        for dir in ["projects", "session-start-marks", "update", "agents", "logs", "backups", "archive"] {
            std::fs::create_dir_all(r.join(dir)).unwrap();
        }
        let moves = plan(r);
        let find = |name: &str| moves.iter().find(|m| m.from == r.join(name)).map(|m| (m.to.strip_prefix(r).unwrap().to_path_buf(), m.link_back));
        assert_eq!(find("memory.db"), Some((PathBuf::from("data/memory.db"), true)));
        assert_eq!(find("memory.db-wal"), Some((PathBuf::from("data/memory.db-wal"), true)));
        assert_eq!(find("memory.db.query_cache.sqlite"), Some((PathBuf::from("data/memory.db.query_cache.sqlite"), true)));
        assert_eq!(find("memory.db.bak-v4.10-20261028-0123-shm"), Some((PathBuf::from("backups/memory/memory.db.bak-v4.10-20261028-0123-shm"), false)));
        assert_eq!(find("conversations.db-wal"), Some((PathBuf::from("data/conversations.db-wal"), true)));
        assert_eq!(find("http.token"), Some((PathBuf::from("config/http.token"), true)));
        assert_eq!(find("session-hook-injections.json"), Some((PathBuf::from("state/session-hook-injections.json"), true)));
        assert_eq!(find("session-start-marks"), Some((PathBuf::from("state/session-start-marks"), true)));
        assert_eq!(find("update"), Some((PathBuf::from("state/update"), true)));
        assert_eq!(find("cursor-hooks.log"), Some((PathBuf::from("logs/cursor-hooks.log"), false)));
        assert_eq!(find("projects"), Some((PathBuf::from("archive/v1/projects"), false)));
        assert_eq!(find("cursor-hooks-state.json.bak"), Some((PathBuf::from("archive/cursor-hooks-state.json.bak"), false)));
        assert_eq!(find("GLOBAL_PROMPT.md.archive-2026-10-31"), Some((PathBuf::from("archive/GLOBAL_PROMPT.md.archive-2026-10-31"), false)));
        for stays in ["unknown.txt", "agents", "logs", "backups", "archive"] {
            assert!(find(stays).is_none(), "{} stays", stays);
        }
    }
}
