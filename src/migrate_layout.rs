//! `MemoryPilot --migrate-layout [--apply]`: sorts the root of `~/.MemoryPilot`
//! into data/, state/, config/, logs/, backups/ and archive/ (see
//! `memorypilot_core::layout`).
//!
//! Without `--apply`, prints the plan. With it:
//! 1. refuses while another process holds memory.db or conversations.db open
//!    (moving a SQLite file under an open connection risks the WAL); the
//!    MemoryPilot HTTP LaunchAgent is stopped and restarted by the command;
//! 2. copies memory.db to backups/memory/memory-before-layout.db;
//! 3. moves each entry, leaving a symbolic link at the old path when older
//!    binaries or McpHub still read it there;
//! 4. checks the moved database with `PRAGMA quick_check`.
//!
//! Nothing is deleted.

use memorypilot_core::layout::{self, Area, Move};
use std::path::{Path, PathBuf};
use std::process::Command;

const SERVICE: &str = "com.soflution.memorypilot-http";

fn uid() -> String {
    Command::new("id").arg("-u").output().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).unwrap_or_default()
}

fn service_plist() -> Option<PathBuf> {
    let path = dirs::home_dir()?.join("Library/LaunchAgents").join(format!("{}.plist", SERVICE));
    path.is_file().then_some(path)
}

fn service_running() -> bool {
    Command::new("launchctl")
        .args(["print", &format!("gui/{}/{}", uid(), SERVICE)])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Processes other than this one holding one of `files` open (`lsof`).
fn holders(files: &[PathBuf]) -> Vec<String> {
    let existing: Vec<String> = files.iter().filter(|f| f.exists()).map(|f| f.to_string_lossy().to_string()).collect();
    if existing.is_empty() {
        return Vec::new();
    }
    let Ok(out) = Command::new("lsof").arg("-Fpc").args(&existing).output() else {
        return vec!["lsof unavailable: cannot check that no process holds the databases".into()];
    };
    let me = std::process::id().to_string();
    let mut found = Vec::new();
    let mut pid = String::new();
    for line in String::from_utf8_lossy(&out.stdout).lines() {
        if let Some(p) = line.strip_prefix('p') {
            pid = p.to_string();
        } else if let Some(command) = line.strip_prefix('c') {
            if pid != me {
                found.push(format!("{} (pid {})", command, pid));
            }
        }
    }
    found.sort();
    found.dedup();
    found
}

fn shown(root: &Path, path: &Path) -> String {
    path.strip_prefix(root).map(|p| p.display().to_string()).unwrap_or_else(|_| path.display().to_string())
}

fn private_dir(path: &Path) -> Result<(), String> {
    std::fs::create_dir_all(path).map_err(|e| format!("{}: {}", path.display(), e))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700)).map_err(|e| format!("{}: {}", path.display(), e))?;
    }
    Ok(())
}

/// `Ok(false)` when the source is gone since the plan (SQLite removes -wal
/// and -shm when the last connection closes, as the backup does).
fn apply_move(root: &Path, m: &Move) -> Result<bool, String> {
    if std::fs::symlink_metadata(&m.from).is_err() {
        return Ok(false);
    }
    if std::fs::symlink_metadata(&m.to).is_ok() {
        return Err(format!("{} already exists: {} left in place", shown(root, &m.to), shown(root, &m.from)));
    }
    if let Some(dir) = m.to.parent() {
        private_dir(dir)?;
    }
    std::fs::rename(&m.from, &m.to).map_err(|e| format!("{} → {}: {}", shown(root, &m.from), shown(root, &m.to), e))?;
    #[cfg(unix)]
    if m.link_back {
        std::os::unix::fs::symlink(&m.to, &m.from).map_err(|e| format!("link {}: {}", shown(root, &m.from), e))?;
    }
    Ok(true)
}

pub fn run(args: &[String]) -> i32 {
    let Some(root) = layout::root() else {
        eprintln!("Cannot find home directory");
        return 1;
    };
    let apply = args.iter().any(|a| a == "--apply");
    let moves = layout::plan(&root);
    if moves.is_empty() {
        println!("{}: already sorted, nothing to move.", root.display());
        return 0;
    }
    for m in &moves {
        println!("  {} → {}{}", shown(&root, &m.from), shown(&root, &m.to), if m.link_back { "  (+ link at the old path)" } else { "" });
    }
    if !apply {
        println!("Plan only. Run with --apply to move ({} entries). Nothing is deleted.", moves.len());
        return 0;
    }

    let dbs: Vec<PathBuf> = ["memory.db", "conversations.db"].iter().map(|n| root.join(n)).collect();
    let service = service_running();
    if service {
        let status = Command::new("launchctl").args(["bootout", &format!("gui/{}/{}", uid(), SERVICE)]).status();
        if !status.map(|s| s.success()).unwrap_or(false) {
            eprintln!("✗ cannot stop {}: nothing moved", SERVICE);
            return 1;
        }
        println!("▸ {} stopped", SERVICE);
        std::thread::sleep(std::time::Duration::from_secs(2));
    }
    let restart = |service: bool| {
        if let (true, Some(plist)) = (service, service_plist()) {
            let ok = Command::new("launchctl").args(["bootstrap", &format!("gui/{}", uid())]).arg(&plist).status().map(|s| s.success()).unwrap_or(false);
            println!("{} {} restarted", if ok { "▸" } else { "✗" }, SERVICE);
        }
    };
    let busy = holders(&dbs);
    if !busy.is_empty() {
        eprintln!("✗ these processes hold the databases open, nothing moved:");
        for b in &busy {
            eprintln!("    {}", b);
        }
        eprintln!("  Quit them (Claude Desktop, other MCP clients), then run again.");
        restart(service);
        return 1;
    }

    let live = root.join("memory.db");
    if live.is_file() {
        let backups = layout::dir_in(&root, Area::Backups).join("memory");
        if let Err(e) = private_dir(&backups) {
            eprintln!("✗ {}: nothing moved", e);
            restart(service);
            return 1;
        }
        let target = backups.join("memory-before-layout.db");
        let _ = std::fs::remove_file(&target);
        let copied = rusqlite::Connection::open(&live)
            .and_then(|conn| conn.execute("VACUUM INTO ?1", [target.to_string_lossy().as_ref()]).map(|_| ()));
        if let Err(e) = copied {
            eprintln!("✗ backup of memory.db failed ({}): nothing moved", e);
            restart(service);
            return 1;
        }
        // VACUUM INTO follows the umask (0644): the copy holds every memory.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if let Err(e) = std::fs::set_permissions(&target, std::fs::Permissions::from_mode(0o600)) {
                eprintln!("✗ {}: {}: nothing moved", target.display(), e);
                restart(service);
                return 1;
            }
        }
        println!("▸ backup: {}", shown(&root, &target));
    }

    let mut failed = 0;
    let mut moved = 0;
    let mut gone = 0;
    for m in &moves {
        match apply_move(&root, m) {
            Ok(true) => moved += 1,
            Ok(false) => gone += 1,
            Err(e) => {
                failed += 1;
                eprintln!("  ✗ {}", e);
            }
        }
    }
    let moved_db = layout::dir_in(&root, Area::Data).join("memory.db");
    if moved_db.is_file() {
        match rusqlite::Connection::open(&moved_db).and_then(|c| c.query_row("PRAGMA quick_check", [], |r| r.get::<_, String>(0))) {
            Ok(check) if check == "ok" => println!("▸ data/memory.db: quick_check ok"),
            Ok(check) => {
                failed += 1;
                eprintln!("  ✗ data/memory.db quick_check: {}", check);
            }
            Err(e) => {
                failed += 1;
                eprintln!("  ✗ data/memory.db: {}", e);
            }
        }
    }
    restart(service);
    println!("{} of {} entries moved, {} gone since the plan, {} problem(s). Nothing deleted.", moved, moves.len(), gone, failed);
    if failed == 0 { 0 } else { 1 }
}
