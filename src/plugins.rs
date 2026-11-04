//! Whole plugins kept in `~/.MemoryPilot/agents/plugins/<name>/`.
//!
//! A plugin folder is a Claude Code plugin (`.claude-plugin/plugin.json`,
//! `skills/`, `agents/`, `hooks/`...). It may also hold:
//! - `cursor/`: the same plugin in Cursor's format (`.cursor-plugin/plugin.json`),
//!   linked as is into `~/.cursor/plugins/local/<name>`;
//! - `plugin.source.json`: where a vendor plugin comes from, read by
//!   `--update-plugins` to show the upstream diff and apply it on request.
//!
//! Claude Code installs plugins from a marketplace: MemoryPilot generates
//! `plugins/.claude-plugin/marketplace.json` (marketplace `memorypilot`) and
//! drives `claude plugin install / update / uninstall`.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

pub const MARKETPLACE: &str = "memorypilot";
pub const SOURCE_FILE: &str = "plugin.source.json";
pub const CURSOR_DIR: &str = "cursor";
const UPDATES_DIR: &str = ".updates";
const BACKUPS_DIR: &str = ".backups";
/// Previous versions kept in `plugins/.backups/` per plugin after an update.
const BACKUPS_KEPT: usize = 3;

#[derive(Debug, Clone, PartialEq)]
pub struct Plugin {
    pub name: String,
    pub dir: PathBuf,
    pub version: String,
    pub description: String,
    /// `cursor/` when it holds `.cursor-plugin/plugin.json`.
    pub cursor_dir: Option<PathBuf>,
}

/// `plugin.source.json` of a vendor plugin.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Source {
    /// Git URL of the upstream repository.
    pub repo: String,
    /// Commit the stored copy comes from.
    pub commit: String,
    /// Branch followed by `--update-plugins` (default branch when absent).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub branch: Option<String>,
    /// Folder of the Claude Code plugin inside the repository.
    pub claude: String,
    /// Folder of the Cursor plugin inside the repository, when it has one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cursor: Option<String>,
}

fn read_json(path: &Path) -> Option<Value> {
    std::fs::read_to_string(path).ok().and_then(|t| serde_json::from_str(&t).ok())
}

/// Plugins of `root` (the `plugins/` folder), sorted by name.
pub fn list(root: &Path) -> Vec<Plugin> {
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(root)
        .map(|entries| {
            entries
                .flatten()
                .filter(|e| !e.file_name().to_string_lossy().starts_with('.'))
                .map(|e| e.path())
                .filter(|p| p.join(".claude-plugin").join("plugin.json").is_file())
                .collect()
        })
        .unwrap_or_default();
    dirs.sort();
    dirs.into_iter()
        .map(|dir| {
            let manifest = read_json(&dir.join(".claude-plugin").join("plugin.json")).unwrap_or(Value::Null);
            let folder = dir.file_name().unwrap_or_default().to_string_lossy().to_string();
            let cursor = dir.join(CURSOR_DIR);
            Plugin {
                name: manifest["name"].as_str().unwrap_or(&folder).to_string(),
                version: manifest["version"].as_str().unwrap_or("").to_string(),
                description: manifest["description"].as_str().unwrap_or("").to_string(),
                cursor_dir: cursor.join(".cursor-plugin").join("plugin.json").is_file().then_some(cursor),
                dir,
            }
        })
        .collect()
}

pub fn read_source(dir: &Path) -> Option<Source> {
    std::fs::read_to_string(dir.join(SOURCE_FILE)).ok().and_then(|t| serde_json::from_str(&t).ok())
}

/// Files under `dir`, relative and sorted, skipping `skip` (top-level names).
fn files(dir: &Path, skip: &[&str]) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(current) = stack.pop() {
        for entry in std::fs::read_dir(&current).into_iter().flatten().flatten() {
            let path = entry.path();
            let Ok(rel) = path.strip_prefix(dir) else { continue };
            if current == dir && skip.iter().any(|s| rel == Path::new(s)) {
                continue;
            }
            let Ok(kind) = entry.file_type() else { continue };
            if kind.is_dir() {
                stack.push(path);
            } else {
                out.push(rel.to_path_buf());
            }
        }
    }
    out.sort();
    out
}

/// Content fingerprint of the Claude Code part of a plugin (the `cursor/`
/// copy and the source file are left out: they never need a Claude update).
pub fn claude_fingerprint(dir: &Path) -> String {
    let mut hasher = Sha256::new();
    for rel in files(dir, &[CURSOR_DIR, SOURCE_FILE]) {
        hasher.update(rel.to_string_lossy().as_bytes());
        hasher.update([0]);
        hasher.update(std::fs::read(dir.join(&rel)).unwrap_or_default());
        hasher.update([0]);
    }
    hasher.finalize().iter().map(|b| format!("{:02x}", b)).collect()
}

pub fn marketplace_path(root: &Path) -> PathBuf {
    root.join(".claude-plugin").join("marketplace.json")
}

/// The generated `marketplace.json` listing every plugin of `root`.
pub fn render_marketplace(root: &Path) -> String {
    let plugins: Vec<Value> = list(root)
        .iter()
        .map(|p| {
            let folder = p.dir.file_name().unwrap_or_default().to_string_lossy().to_string();
            let mut entry = json!({ "name": p.name, "source": format!("./{}", folder) });
            if !p.version.is_empty() {
                entry["version"] = json!(p.version);
            }
            if !p.description.is_empty() {
                entry["description"] = json!(p.description);
            }
            entry
        })
        .collect();
    let doc = json!({
        "$schema": "https://anthropic.com/claude-code/marketplace.schema.json",
        "name": MARKETPLACE,
        "owner": { "name": "MemoryPilot" },
        "metadata": { "description": "Generated by MemoryPilot from ~/.MemoryPilot/agents/plugins. Do not edit: rewritten at each sync." },
        "plugins": plugins,
    });
    serde_json::to_string_pretty(&doc).unwrap_or_default() + "\n"
}

// ─── Claude Code CLI ─────────────────────────────────────────────────────

/// `claude` executable: `MEMORYPILOT_CLAUDE_BIN`, then PATH, then the
/// installer's usual places.
fn claude_bin(home: &Path) -> Option<PathBuf> {
    if let Some(bin) = std::env::var_os("MEMORYPILOT_CLAUDE_BIN").filter(|v| !v.is_empty()) {
        return Some(PathBuf::from(bin));
    }
    if let Some(path) = std::env::var_os("PATH") {
        if let Some(dir) = std::env::split_paths(&path).find(|d| d.join("claude").is_file()) {
            return Some(dir.join("claude"));
        }
    }
    [home.join(".local/bin/claude"), home.join(".claude/local/claude")].into_iter().find(|p| p.is_file())
}

pub fn claude_cli(home: &Path, args: &[&str]) -> Result<(), String> {
    let bin = claude_bin(home).ok_or("claude command not found (set MEMORYPILOT_CLAUDE_BIN)")?;
    let out = std::process::Command::new(&bin)
        .args(args)
        .stdin(std::process::Stdio::null())
        .output()
        .map_err(|e| format!("claude {}: {}", args.join(" "), e))?;
    if out.status.success() {
        return Ok(());
    }
    let mut msg = String::from_utf8_lossy(&out.stderr).trim().to_string();
    if msg.is_empty() {
        msg = String::from_utf8_lossy(&out.stdout).trim().to_string();
    }
    Err(format!("claude {}: {}", args.join(" "), msg.chars().take(300).collect::<String>()))
}

/// Is the `memorypilot` marketplace registered in Claude Code?
pub fn marketplace_known(claude_dir: &Path) -> bool {
    read_json(&claude_dir.join("plugins/known_marketplaces.json")).is_some_and(|v| v.get(MARKETPLACE).is_some())
}

/// Is `<name>@memorypilot` installed in Claude Code?
pub fn plugin_installed(claude_dir: &Path, name: &str) -> bool {
    read_json(&claude_dir.join("plugins/installed_plugins.json"))
        .is_some_and(|v| v["plugins"].get(format!("{}@{}", name, MARKETPLACE)).is_some())
}

// ─── Vendor updates ──────────────────────────────────────────────────────

fn run(cmd: &str, args: &[&str], cwd: Option<&Path>) -> Result<String, String> {
    let mut command = std::process::Command::new(cmd);
    command.args(args).stdin(std::process::Stdio::null());
    if let Some(cwd) = cwd {
        command.current_dir(cwd);
    }
    let out = command.output().map_err(|e| format!("{} {}: {}", cmd, args.join(" "), e))?;
    if !out.status.success() {
        return Err(format!("{} {}: {}", cmd, args.join(" "), String::from_utf8_lossy(&out.stderr).trim()));
    }
    Ok(String::from_utf8_lossy(&out.stdout).to_string())
}

fn copy_dir(from: &Path, to: &Path) -> Result<(), String> {
    for rel in files(from, &[]) {
        let dest = to.join(&rel);
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent).map_err(|e| format!("{}: {}", parent.display(), e))?;
        }
        std::fs::copy(from.join(&rel), &dest).map_err(|e| format!("{}: {}", dest.display(), e))?;
    }
    Ok(())
}

/// Builds `plugins/.updates/<folder>/` from the upstream head: the plugin as
/// it would be stored, with `plugin.source.json` on the new commit.
fn stage(root: &Path, folder: &str, source: &Source) -> Result<(PathBuf, Source), String> {
    let work = root.join(UPDATES_DIR);
    let clone = work.join(format!("{}.clone", folder));
    let staged = work.join(folder);
    let _ = std::fs::remove_dir_all(&clone);
    let _ = std::fs::remove_dir_all(&staged);
    std::fs::create_dir_all(&work).map_err(|e| format!("{}: {}", work.display(), e))?;
    let clone_str = clone.to_string_lossy().to_string();
    let mut args = vec!["clone", "--quiet", "--depth", "1"];
    if let Some(branch) = &source.branch {
        args.extend(["--branch", branch.as_str()]);
    }
    args.extend([source.repo.as_str(), clone_str.as_str()]);
    run("git", &args, None)?;
    let commit = run("git", &["rev-parse", "HEAD"], Some(&clone))?.trim().to_string();
    copy_dir(&clone.join(&source.claude), &staged)?;
    if let Some(cursor) = &source.cursor {
        copy_dir(&clone.join(cursor), &staged.join(CURSOR_DIR))?;
    }
    let _ = std::fs::remove_dir_all(&clone);
    let updated = Source { commit, ..source.clone() };
    let text = serde_json::to_string_pretty(&updated).map_err(|e| e.to_string())? + "\n";
    std::fs::write(staged.join(SOURCE_FILE), text).map_err(|e| e.to_string())?;
    Ok((staged, updated))
}

/// Unified diff between the stored and the staged plugin (`git diff --no-index`
/// exits 1 when they differ, which is not an error here).
fn diff(old: &Path, new: &Path, stat: bool) -> Result<String, String> {
    let mut args = vec!["diff", "--no-index", "--no-color"];
    if stat {
        args.push("--stat=120");
    }
    let old_s = old.to_string_lossy().to_string();
    let new_s = new.to_string_lossy().to_string();
    args.extend([old_s.as_str(), new_s.as_str()]);
    let out = std::process::Command::new("git")
        .args(&args)
        .output()
        .map_err(|e| format!("git diff: {}", e))?;
    match out.status.code() {
        Some(0) | Some(1) => Ok(String::from_utf8_lossy(&out.stdout).to_string()),
        _ => Err(format!("git diff: {}", String::from_utf8_lossy(&out.stderr).trim())),
    }
}

fn version_of(dir: &Path) -> String {
    read_json(&dir.join(".claude-plugin/plugin.json"))
        .and_then(|v| v["version"].as_str().map(str::to_string))
        .unwrap_or_else(|| "?".into())
}

/// Moves the current copy to `.backups/<folder>-<version>-<date>` and keeps
/// the `BACKUPS_KEPT` most recent ones of this plugin.
fn backup(root: &Path, folder: &str, dir: &Path) -> Result<PathBuf, String> {
    let backups = root.join(BACKUPS_DIR);
    std::fs::create_dir_all(&backups).map_err(|e| format!("{}: {}", backups.display(), e))?;
    let name = format!("{}-{}-{}", folder, version_of(dir), chrono::Local::now().format("%Y%m%d-%H%M%S"));
    let dest = backups.join(&name);
    std::fs::rename(dir, &dest).map_err(|e| format!("{}: {}", dest.display(), e))?;
    let prefix = format!("{}-", folder);
    let mut olds: Vec<PathBuf> = std::fs::read_dir(&backups)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| e.file_name().to_string_lossy().starts_with(&prefix))
        .map(|e| e.path())
        .collect();
    olds.sort_by_key(|p| p.metadata().and_then(|m| m.modified()).ok());
    while olds.len() > BACKUPS_KEPT {
        let _ = std::fs::remove_dir_all(olds.remove(0));
    }
    Ok(dest)
}

/// Entry point for `--update-plugins [name] [--apply]`. Without `--apply`:
/// fetches each vendor plugin's upstream head, prints the diff summary and
/// writes the full diff next to the staged copy. With `--apply`: replaces
/// the stored copy by the staged one (previous copy kept in `.backups/`),
/// then the usual sync updates the tools.
pub fn update_plugins(args: &[String]) -> i32 {
    let Some(home) = dirs::home_dir() else {
        eprintln!("Cannot find home directory");
        return 1;
    };
    let root = home.join(".MemoryPilot/agents/plugins");
    let apply = args.iter().any(|a| a == "--apply");
    let only: Option<&String> = args
        .iter()
        .position(|a| a == "--update-plugins")
        .and_then(|i| args.get(i + 1))
        .filter(|a| !a.starts_with("--"));
    let plugins: Vec<Plugin> = list(&root).into_iter().filter(|p| only.is_none_or(|n| *n == p.name)).collect();
    if plugins.is_empty() {
        eprintln!("No plugin{} in {}", only.map(|n| format!(" named {}", n)).unwrap_or_default(), root.display());
        return 2;
    }
    let mut status = 0;
    for plugin in plugins {
        let folder = plugin.dir.file_name().unwrap_or_default().to_string_lossy().to_string();
        let Some(source) = read_source(&plugin.dir) else {
            println!("{}: no {} (own plugin), nothing to update", plugin.name, SOURCE_FILE);
            continue;
        };
        let staged = root.join(UPDATES_DIR).join(&folder);
        if apply {
            let Some(staged_source) = read_source(&staged) else {
                println!("{}: nothing staged, run --update-plugins {} first and read the diff", plugin.name, plugin.name);
                status = 1;
                continue;
            };
            match backup(&root, &folder, &plugin.dir).and_then(|kept| {
                std::fs::rename(&staged, &plugin.dir).map_err(|e| e.to_string())?;
                Ok(kept)
            }) {
                Ok(kept) => println!(
                    "{}: {} → {} applied (commit {}), previous copy in {}",
                    plugin.name,
                    plugin.version,
                    version_of(&plugin.dir),
                    &staged_source.commit[..staged_source.commit.len().min(7)],
                    kept.display()
                ),
                Err(e) => {
                    eprintln!("{}: apply failed: {}", plugin.name, e);
                    status = 1;
                }
            }
            continue;
        }
        match stage(&root, &folder, &source) {
            Ok((staged, upstream)) if upstream.commit == source.commit => {
                let _ = std::fs::remove_dir_all(&staged);
                println!("{} {}: up to date (commit {})", plugin.name, plugin.version, &source.commit[..source.commit.len().min(7)]);
            }
            Ok((staged, upstream)) => {
                let full = diff(&plugin.dir, &staged, false).unwrap_or_default();
                let patch = root.join(UPDATES_DIR).join(format!("{}.diff", folder));
                let _ = std::fs::write(&patch, &full);
                println!(
                    "{}: {} → {} (commit {} → {})",
                    plugin.name,
                    plugin.version,
                    version_of(&staged),
                    &source.commit[..source.commit.len().min(7)],
                    &upstream.commit[..upstream.commit.len().min(7)]
                );
                match diff(&plugin.dir, &staged, true) {
                    Ok(stat) => print!("{}", stat),
                    Err(e) => eprintln!("  {}", e),
                }
                println!("  Full diff: {}", patch.display());
                println!("  Apply after review: MemoryPilot --update-plugins {} --apply", plugin.name);
            }
            Err(e) => {
                eprintln!("{}: {}", plugin.name, e);
                status = 1;
            }
        }
    }
    status
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
        TempRoot(std::env::temp_dir().join(format!("memorypilot-plugins-{}-{}", std::process::id(), n)))
    }

    fn write(path: &Path, text: &str) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
    }

    #[test]
    fn lists_plugins_with_their_cursor_variant() {
        let root = temp_root();
        let r = &root.0;
        write(&r.join("kit/.claude-plugin/plugin.json"), r#"{"name":"kit","version":"1.2.0","description":"Kit"}"#);
        write(&r.join("kit/cursor/.cursor-plugin/plugin.json"), "{}");
        write(&r.join("solo/.claude-plugin/plugin.json"), r#"{"name":"solo"}"#);
        write(&r.join("not-a-plugin/README.md"), "x");
        write(&r.join(".updates/kit/.claude-plugin/plugin.json"), r#"{"name":"kit"}"#);
        let plugins = list(r);
        assert_eq!(plugins.iter().map(|p| p.name.as_str()).collect::<Vec<_>>(), ["kit", "solo"]);
        assert_eq!(plugins[0].version, "1.2.0");
        assert_eq!(plugins[0].cursor_dir.as_deref(), Some(r.join("kit/cursor").as_path()));
        assert!(plugins[1].cursor_dir.is_none());
        let market: Value = serde_json::from_str(&render_marketplace(r)).unwrap();
        assert_eq!(market["name"], MARKETPLACE);
        assert_eq!(market["plugins"][0], json!({"name": "kit", "source": "./kit", "version": "1.2.0", "description": "Kit"}));
        assert_eq!(market["plugins"][1], json!({"name": "solo", "source": "./solo"}));
    }

    #[test]
    fn fingerprint_ignores_the_cursor_copy_and_source_file() {
        let root = temp_root();
        let dir = root.0.join("kit");
        write(&dir.join(".claude-plugin/plugin.json"), "{}");
        write(&dir.join("skills/a/SKILL.md"), "one");
        let before = claude_fingerprint(&dir);
        write(&dir.join("cursor/.cursor-plugin/plugin.json"), "{}");
        write(&dir.join(SOURCE_FILE), "{}");
        assert_eq!(claude_fingerprint(&dir), before);
        write(&dir.join("skills/a/SKILL.md"), "two");
        assert_ne!(claude_fingerprint(&dir), before);
    }

    #[test]
    fn backups_keep_the_three_latest() {
        let root = temp_root();
        let r = &root.0;
        for i in 0..5 {
            let dir = r.join("kit");
            write(&dir.join(".claude-plugin/plugin.json"), &format!(r#"{{"version":"1.{}"}}"#, i));
            backup(r, "kit", &dir).unwrap();
            std::thread::sleep(std::time::Duration::from_millis(1100));
        }
        let left: Vec<String> = std::fs::read_dir(r.join(BACKUPS_DIR)).unwrap().flatten().map(|e| e.file_name().to_string_lossy().to_string()).collect();
        assert_eq!(left.len(), BACKUPS_KEPT);
        assert!(left.iter().all(|n| !n.starts_with("kit-1.0-") && !n.starts_with("kit-1.1-")), "{:?}", left);
    }
}
