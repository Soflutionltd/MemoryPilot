//! Third-party skills kept in `~/.MemoryPilot/agents/vendor/<name>/`.
//!
//! A vendor skill is a copy of a skill folder (`SKILL.md` and its files) of
//! someone else's Git repository. `skill.source.json` next to `SKILL.md`
//! records where it comes from, so it can be updated later. MemoryPilot
//! deploys it to every wired tool like the user's own skills.
//!
//! `MemoryPilot --add-skill <owner/repo[@skill]> [--skill NAME] [--path DIR]
//! [--branch B] [--apply]` : without `--apply`, fetches the skill into
//! `vendor/.updates/<name>/` and shows what it holds (or the diff with the
//! copy already stored); nothing is installed. With `--apply`, the staged
//! copy that was just reviewed replaces the stored one (previous copy kept
//! in `vendor/.backups/`) and the tools are synced.
//!
//! `MemoryPilot --update-skills [NAME] [--apply]` : the same two steps for
//! every vendor skill that has a `skill.source.json`.
//!
//! Nothing a skill ships is ever run: files are copied, symbolic links are
//! left out, and the copy is only read by the agents once the user applied it.

use crate::plugins::{diff, files, run};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

pub const SOURCE_FILE: &str = "skill.source.json";
const UPDATES_DIR: &str = ".updates";
const BACKUPS_DIR: &str = ".backups";
/// Previous copies kept in `vendor/.backups/<name>/` after an update.
const BACKUPS_KEPT: usize = 3;
/// How deep a skill folder is looked for inside a repository.
const MAX_SEARCH_DEPTH: usize = 6;

/// `skill.source.json` of a vendor skill.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Source {
    /// Git URL of the upstream repository.
    pub repo: String,
    /// Commit the stored copy comes from.
    pub commit: String,
    /// Branch followed by `--update-skills` (default branch when absent).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub branch: Option<String>,
    /// Folder of the skill inside the repository (`.` for its root).
    pub path: String,
}

pub fn read_source(dir: &Path) -> Option<Source> {
    std::fs::read_to_string(dir.join(SOURCE_FILE)).ok().and_then(|t| serde_json::from_str(&t).ok())
}

/// A skill name is a single folder name: letters, digits, `-` and `_`.
fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 64
        && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

/// Splits `owner/repo`, `owner/repo@skill`, a Git URL or a local path (each
/// with an optional `@skill`) into the URL to clone and the skill asked for.
fn parse_spec(spec: &str) -> Result<(String, Option<String>), String> {
    let spec = spec.trim();
    let (location, skill) = match spec.rsplit_once('@') {
        Some((location, skill)) if !location.is_empty() && valid_name(skill) => (location, Some(skill.to_string())),
        _ => (spec, None),
    };
    let is_url_or_path = location.contains("://") || location.starts_with("git@") || location.starts_with(['/', '.', '~']);
    if is_url_or_path {
        return Ok((location.to_string(), skill));
    }
    let part = |text: &str| !text.is_empty() && text.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'));
    match location.split_once('/') {
        Some((owner, repo)) if part(owner) && part(repo) && !owner.starts_with('.') && !repo.starts_with('.') => {
            Ok((format!("https://github.com/{}/{}.git", owner, repo), skill))
        }
        _ => Err(format!("'{}' is not owner/repo, owner/repo@skill or a Git URL", spec)),
    }
}

/// The `name:` of a `SKILL.md` front matter, when it has one.
fn declared_name(skill_md: &Path) -> Option<String> {
    let text = std::fs::read_to_string(skill_md).ok()?;
    let front_matter = text.strip_prefix("---")?.split("\n---").next()?;
    front_matter
        .lines()
        .find_map(|line| line.strip_prefix("name:"))
        .map(|name| name.trim().trim_matches(['"', '\'']).to_string())
        .filter(|name| valid_name(name))
}

/// The repository's own name, for a skill sitting at its root: the last
/// segment of its URL, without `.git`.
fn repository_name(repo: &str) -> Option<String> {
    let last = repo.trim_end_matches('/').rsplit(['/', ':']).next()?;
    let name = last.strip_suffix(".git").unwrap_or(last);
    valid_name(name).then(|| name.to_string())
}

/// Skill folders of a cloned repository: `(name, folder relative to the clone)`.
/// The name is the one `SKILL.md` declares, else the folder's (`root_name`
/// for a skill at the root, whose folder is only the clone's).
fn find_skills(clone: &Path, root_name: Option<&str>) -> Vec<(String, PathBuf)> {
    let mut found = Vec::new();
    let mut stack = vec![(clone.to_path_buf(), 0usize)];
    while let Some((dir, depth)) = stack.pop() {
        let skill_md = dir.join("SKILL.md");
        if skill_md.is_file() {
            let relative = dir.strip_prefix(clone).unwrap_or(Path::new("")).to_path_buf();
            let folder_name = if relative.as_os_str().is_empty() {
                root_name.map(String::from)
            } else {
                dir.file_name().map(|n| n.to_string_lossy().to_string())
            };
            if let Some(name) = declared_name(&skill_md).or(folder_name).filter(|name| valid_name(name)) {
                found.push((name, relative));
            }
            continue;
        }
        if depth >= MAX_SEARCH_DEPTH {
            continue;
        }
        for entry in std::fs::read_dir(&dir).into_iter().flatten().flatten() {
            let name = entry.file_name();
            let is_real_dir = entry.file_type().map(|kind| kind.is_dir()).unwrap_or(false);
            if is_real_dir && name != ".git" && name != "node_modules" {
                stack.push((entry.path(), depth + 1));
            }
        }
    }
    found.sort();
    found
}

/// Copies a skill folder, leaving symbolic links out (a link could point
/// anywhere on the machine that clones). Returns the links left out.
fn copy_skill(from: &Path, to: &Path) -> Result<Vec<PathBuf>, String> {
    let mut links = Vec::new();
    for relative in files(from, &[]) {
        let source = from.join(&relative);
        let is_link = std::fs::symlink_metadata(&source).map(|m| m.file_type().is_symlink()).unwrap_or(true);
        if is_link {
            links.push(relative);
            continue;
        }
        let dest = to.join(&relative);
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent).map_err(|e| format!("{}: {}", parent.display(), e))?;
        }
        std::fs::copy(&source, &dest).map_err(|e| format!("{}: {}", dest.display(), e))?;
    }
    Ok(links)
}

/// What `stage` fetched.
#[derive(Debug)]
struct Staged {
    name: String,
    dir: PathBuf,
    source: Source,
    /// Symbolic links of the upstream folder, not copied.
    links: Vec<PathBuf>,
}

/// Why nothing was staged.
#[derive(Debug, PartialEq)]
enum StageError {
    /// The repository holds several skills and none was asked for.
    Choose(Vec<String>),
    Other(String),
}

impl From<String> for StageError {
    fn from(message: String) -> Self {
        StageError::Other(message)
    }
}

/// Fetches the upstream head of `repo` and builds `vendor/.updates/<name>/`:
/// the skill as it would be stored, with its `skill.source.json`. The skill
/// is the folder `path` when given, else the one named `skill`, else the
/// only one of the repository.
fn stage(root: &Path, repo: &str, branch: Option<&str>, skill: Option<&str>, path: Option<&str>) -> Result<Staged, StageError> {
    let work = root.join(UPDATES_DIR);
    std::fs::create_dir_all(&work).map_err(|e| format!("{}: {}", work.display(), e))?;
    // A dot name can never be a skill's: the clone cannot collide with a staged one.
    let clone = work.join(format!(".clone-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&clone);
    let clone_text = clone.to_string_lossy().to_string();
    let mut args = vec!["clone", "--quiet", "--depth", "1"];
    if let Some(branch) = branch {
        args.extend(["--branch", branch]);
    }
    args.extend(["--", repo, clone_text.as_str()]);
    let staged = run("git", &args, None).map_err(StageError::from).and_then(|_| {
        let commit = run("git", &["rev-parse", "HEAD"], Some(&clone))?.trim().to_string();
        let (name, relative) = locate(&clone, repository_name(repo).as_deref(), skill, path)?;
        let dir = work.join(&name);
        let _ = std::fs::remove_dir_all(&dir);
        let links = copy_skill(&clone.join(&relative), &dir)?;
        let relative_text = relative.to_string_lossy().to_string();
        let source = Source {
            repo: repo.to_string(),
            commit,
            branch: branch.map(String::from),
            path: if relative_text.is_empty() { ".".into() } else { relative_text },
        };
        let text = serde_json::to_string_pretty(&source).map_err(|e| e.to_string())? + "\n";
        std::fs::write(dir.join(SOURCE_FILE), text).map_err(|e| e.to_string())?;
        Ok(Staged { name, dir, source, links })
    });
    let _ = std::fs::remove_dir_all(&clone);
    staged
}

/// The skill folder to take in a clone, and the name it is stored under.
fn locate(clone: &Path, root_name: Option<&str>, skill: Option<&str>, path: Option<&str>) -> Result<(String, PathBuf), StageError> {
    if let Some(path) = path {
        let relative = PathBuf::from(path.trim_matches('/'));
        let stays_inside = relative.components().all(|part| matches!(part, std::path::Component::Normal(_) | std::path::Component::CurDir));
        if !stays_inside {
            return Err(format!("--path {} must stay inside the repository", path).into());
        }
        let mut walked = clone.to_path_buf();
        for part in relative.components() {
            walked.push(part);
            if std::fs::symlink_metadata(&walked).is_ok_and(|m| m.file_type().is_symlink()) {
                return Err(format!("--path {} goes through a symbolic link of the repository", path).into());
            }
        }
        let skill_md = clone.join(&relative).join("SKILL.md");
        if !skill_md.is_file() {
            return Err(format!("no SKILL.md in {} of the repository", path).into());
        }
        let folder_name = relative.file_name().map(|n| n.to_string_lossy().to_string());
        let name = skill
            .map(String::from)
            .or_else(|| declared_name(&skill_md))
            .or(folder_name)
            .filter(|name| valid_name(name))
            .ok_or_else(|| "cannot name this skill: pass --skill NAME".to_string())?;
        return Ok((name, relative));
    }
    let skills = find_skills(clone, root_name);
    let Some(wanted) = skill else {
        let mut names: Vec<String> = skills.iter().map(|(name, _)| name.clone()).collect();
        names.dedup();
        return match names.len() {
            0 => Err("no SKILL.md found in the repository (a skill at its root that declares no name needs --path . --skill NAME)".to_string().into()),
            1 => locate(clone, root_name, Some(&names[0]), None),
            _ => Err(StageError::Choose(names)),
        };
    };
    let folders: Vec<PathBuf> = skills
        .into_iter()
        .filter(|(name, relative)| name == wanted || relative.file_name().is_some_and(|folder| folder == wanted))
        .map(|(_, relative)| relative)
        .collect();
    let Some(first) = folders.first() else {
        return Err(format!("no skill named {} in the repository", wanted).into());
    };
    // A repository may ship the same skill in two places (one per agent).
    if folders.iter().any(|other| !same_content(&clone.join(first), &clone.join(other))) {
        let listed: Vec<String> = folders.iter().map(|folder| folder.display().to_string()).collect();
        return Err(format!("several different skills are named {}: pass --path with one of {}", wanted, listed.join(", ")).into());
    }
    Ok((wanted.to_string(), first.clone()))
}

/// True when two skill folders hold the same files (the source file aside).
fn same_content(a: &Path, b: &Path) -> bool {
    let listed = files(a, &[SOURCE_FILE]);
    listed == files(b, &[SOURCE_FILE]) && listed.iter().all(|relative| std::fs::read(a.join(relative)).ok() == std::fs::read(b.join(relative)).ok())
}

fn short(commit: &str) -> &str {
    &commit[..commit.len().min(7)]
}

/// Moves the stored copy to `.backups/<name>/<commit>-<date>` and keeps the
/// `BACKUPS_KEPT` most recent ones of this skill.
fn backup(root: &Path, name: &str) -> Result<PathBuf, String> {
    let dir = root.join(name);
    let backups = root.join(BACKUPS_DIR).join(name);
    std::fs::create_dir_all(&backups).map_err(|e| format!("{}: {}", backups.display(), e))?;
    let label = read_source(&dir).map(|source| short(&source.commit).to_string()).unwrap_or_else(|| "manual".into());
    let dest = backups.join(format!("{}-{}", label, chrono::Local::now().format("%Y%m%d-%H%M%S")));
    std::fs::rename(&dir, &dest).map_err(|e| format!("{}: {}", dest.display(), e))?;
    let mut olds: Vec<PathBuf> = std::fs::read_dir(&backups).into_iter().flatten().flatten().map(|e| e.path()).collect();
    olds.sort_by_key(|p| p.metadata().and_then(|m| m.modified()).ok());
    while olds.len() > BACKUPS_KEPT {
        let _ = std::fs::remove_dir_all(olds.remove(0));
    }
    Ok(dest)
}

/// Replaces `vendor/<name>` by the staged copy. Returns the source applied
/// and where the previous copy went, when there was one.
fn apply_staged(root: &Path, name: &str, expected_repo: Option<&str>) -> Result<(Source, Option<PathBuf>), String> {
    let staged = root.join(UPDATES_DIR).join(name);
    let source = read_source(&staged).ok_or_else(|| format!("nothing staged for {}: run the command without --apply first and read what it shows", name))?;
    if let Some(expected) = expected_repo.filter(|expected| *expected != source.repo) {
        return Err(format!("the staged copy of {} comes from {}, not from {}: stage it again from the repository you mean", name, source.repo, expected));
    }
    if !staged.join("SKILL.md").is_file() {
        return Err(format!("the staged copy of {} has no SKILL.md", name));
    }
    let dir = root.join(name);
    let kept = if dir.exists() { Some(backup(root, name)?) } else { None };
    if let Err(error) = std::fs::rename(&staged, &dir) {
        // Put the previous copy back: a stored skill must never be left missing.
        let restored = kept.as_ref().is_some_and(|kept| std::fs::rename(kept, &dir).is_ok());
        return Err(format!("{} ({})", error, if restored { "previous copy restored" } else { "no previous copy restored" }));
    }
    let _ = std::fs::remove_file(root.join(UPDATES_DIR).join(format!("{}.diff", name)));
    Ok((source, kept))
}

/// Files of a staged skill, with a mark on what is not plain text guidance.
fn describe(dir: &Path) -> Vec<String> {
    files(dir, &[SOURCE_FILE])
        .into_iter()
        .map(|relative| {
            let size = std::fs::metadata(dir.join(&relative)).map(|m| m.len()).unwrap_or(0);
            let is_text = relative
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| matches!(e.to_ascii_lowercase().as_str(), "md" | "txt" | "json" | "yaml" | "yml" | "toml" | "csv"));
            format!("  {:>8} bytes  {}{}", size, relative.display(), if is_text { "" } else { "   <- script or binary: read it before applying" })
        })
        .collect()
}

/// Prints what was staged against the stored copy, and how to apply it.
/// Returns false when the stored copy already matches upstream.
fn report(root: &Path, staged: &Staged, apply_command: &str) -> bool {
    let stored = root.join(&staged.name);
    let stored_source = read_source(&stored);
    if stored.exists() {
        // The source file changes with every upstream commit: compare the skill itself.
        let unchanged = same_content(&stored, &staged.dir);
        if unchanged && stored_source.is_some() {
            let _ = std::fs::remove_dir_all(&staged.dir);
            println!("{}: up to date (commit {})", staged.name, short(&staged.source.commit));
            return false;
        }
        println!(
            "{}: {} → {}",
            staged.name,
            stored_source.as_ref().map(|s| format!("commit {}", short(&s.commit))).unwrap_or_else(|| "copy imported by hand".into()),
            format_args!("commit {} of {}", short(&staged.source.commit), staged.source.repo)
        );
        if unchanged {
            println!("  Same files as the stored copy: applying only records where it comes from.");
        }
        match diff(&stored, &staged.dir, true) {
            Ok(stat) => print!("{}", stat),
            Err(error) => eprintln!("  {}", error),
        }
        let patch = root.join(UPDATES_DIR).join(format!("{}.diff", staged.name));
        if std::fs::write(&patch, diff(&stored, &staged.dir, false).unwrap_or_default()).is_ok() {
            println!("  Full diff: {}", patch.display());
        }
    } else {
        println!("{}: new skill from {} (commit {}, folder {})", staged.name, staged.source.repo, short(&staged.source.commit), staged.source.path);
        for line in describe(&staged.dir) {
            println!("{}", line);
        }
        println!("  Read it: {}", staged.dir.display());
    }
    for link in &staged.links {
        println!("  Left out (symbolic link): {}", link.display());
    }
    println!("  Apply after review: {}", apply_command);
    true
}

fn vendor_root() -> Result<PathBuf, String> {
    let home = dirs::home_dir().ok_or("Cannot find home directory")?;
    let agents = home.join(".MemoryPilot/agents");
    if !agents.is_dir() {
        return Err(format!("{} does not exist: run MemoryPilot --setup-agents first", agents.display()));
    }
    Ok(agents.join("vendor"))
}

/// The value following `flag` in `args`.
fn option<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).map(String::as_str).filter(|v| !v.starts_with("--"))
}

/// An own skill of the same name would hide the vendor one.
fn own_skill_exists(root: &Path, name: &str) -> bool {
    root.parent().is_some_and(|agents| agents.join("skills").join(name).exists())
}

/// Entry point for `--add-skill <spec> [--skill NAME] [--path DIR] [--branch B] [--apply]`.
pub fn add_skill(args: &[String]) -> i32 {
    let Some(spec) = option(args, "--add-skill") else {
        eprintln!("Usage: MemoryPilot --add-skill <owner/repo[@skill]> [--skill NAME] [--path DIR] [--branch B] [--apply]");
        return 2;
    };
    let root = match vendor_root() {
        Ok(root) => root,
        Err(error) => {
            eprintln!("{}", error);
            return 1;
        }
    };
    let (repo, from_spec) = match parse_spec(spec) {
        Ok(parsed) => parsed,
        Err(error) => {
            eprintln!("{}", error);
            return 2;
        }
    };
    let skill = option(args, "--skill").map(String::from).or(from_spec);
    if let Some(name) = skill.as_deref().filter(|name| !valid_name(name)) {
        eprintln!("'{}' is not a skill name (letters, digits, - and _)", name);
        return 2;
    }
    if args.iter().any(|a| a == "--apply") {
        let Some(name) = skill else {
            eprintln!("--apply needs the skill name: --add-skill {}@NAME --apply", spec);
            return 2;
        };
        if own_skill_exists(&root, &name) {
            eprintln!("{}: a skill of yours has this name in agents/skills, it would hide the imported one", name);
            return 1;
        }
        return match apply_staged(&root, &name, Some(&repo)) {
            Ok((source, kept)) => {
                println!(
                    "{}: commit {} applied{}",
                    name,
                    short(&source.commit),
                    kept.map(|kept| format!(", previous copy in {}", kept.display())).unwrap_or_default()
                );
                crate::agents::sync_agents();
                0
            }
            Err(error) => {
                eprintln!("{}: apply failed: {}", name, error);
                1
            }
        };
    }
    match stage(&root, &repo, option(args, "--branch"), skill.as_deref(), option(args, "--path")) {
        Ok(staged) => {
            if own_skill_exists(&root, &staged.name) {
                println!("  Note: agents/skills/{} exists and would hide this one.", staged.name);
            }
            let location = spec.rsplit_once('@').filter(|(_, tail)| valid_name(tail)).map(|(head, _)| head).unwrap_or(spec);
            report(&root, &staged, &format!("MemoryPilot --add-skill {}@{} --apply", location, staged.name));
            0
        }
        Err(StageError::Choose(names)) => {
            println!("This repository holds {} skills, pick one with @NAME:", names.len());
            for name in names {
                println!("  {}", name);
            }
            2
        }
        Err(StageError::Other(error)) => {
            eprintln!("{}", error);
            1
        }
    }
}

/// Entry point for `--update-skills [NAME] [--apply]`.
pub fn update_skills(args: &[String]) -> i32 {
    let root = match vendor_root() {
        Ok(root) => root,
        Err(error) => {
            eprintln!("{}", error);
            return 1;
        }
    };
    let apply = args.iter().any(|a| a == "--apply");
    // args[0] is the binary: the skill name is the first other argument that is not an option.
    let only = args.iter().skip(1).find(|a| !a.starts_with("--"));
    let mut names: Vec<String> = std::fs::read_dir(&root)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|entry| entry.file_name().into_string().ok())
        .filter(|name| !name.starts_with('.') && root.join(name).join("SKILL.md").is_file())
        .filter(|name| only.is_none_or(|wanted| wanted == name))
        .collect();
    names.sort();
    if names.is_empty() {
        eprintln!("No vendor skill{} in {}", only.map(|n| format!(" named {}", n)).unwrap_or_default(), root.display());
        return 2;
    }
    let mut status = 0;
    for name in names {
        let Some(source) = read_source(&root.join(&name)) else {
            println!("{}: imported by hand (no {}): adopt it with --add-skill <owner/repo>@{}", name, SOURCE_FILE, name);
            continue;
        };
        if apply {
            if !root.join(UPDATES_DIR).join(&name).is_dir() {
                continue;
            }
            match apply_staged(&root, &name, Some(&source.repo)) {
                Ok((applied, kept)) => println!(
                    "{}: commit {} applied{}",
                    name,
                    short(&applied.commit),
                    kept.map(|kept| format!(", previous copy in {}", kept.display())).unwrap_or_default()
                ),
                Err(error) => {
                    eprintln!("{}: apply failed: {}", name, error);
                    status = 1;
                }
            }
            continue;
        }
        let path = (source.path != ".").then_some(source.path.as_str());
        match stage(&root, &source.repo, source.branch.as_deref(), Some(&name), path) {
            Ok(staged) => {
                report(&root, &staged, &format!("MemoryPilot --update-skills {} --apply", name));
            }
            Err(StageError::Choose(_)) => {
                eprintln!("{}: its repository now holds several skills at its root, re-add it with --add-skill", name);
                status = 1;
            }
            Err(StageError::Other(error)) => {
                eprintln!("{}: {}", name, error);
                status = 1;
            }
        }
    }
    if apply {
        crate::agents::sync_agents();
    }
    status
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TempDir(PathBuf);

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn temp_dir(label: &str) -> TempDir {
        static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("memorypilot-vendor-{}-{}-{}", label, std::process::id(), n));
        std::fs::create_dir_all(&dir).unwrap();
        TempDir(dir)
    }

    fn write(path: &Path, text: &str) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
    }

    fn git(dir: &Path, args: &[&str]) {
        let mut all = vec!["-c", "user.name=test", "-c", "user.email=test@example.com", "-c", "commit.gpgsign=false"];
        all.extend_from_slice(args);
        run("git", &all, Some(dir)).unwrap();
    }

    /// A local repository standing for upstream, with two skills.
    fn upstream() -> TempDir {
        let repo = temp_dir("upstream");
        git(&repo.0, &["init", "--quiet", "--initial-branch", "main"]);
        write(&repo.0.join("skills/alpha/SKILL.md"), "---\nname: alpha\ndescription: First.\n---\nAlpha v1\n");
        write(&repo.0.join("skills/alpha/references/guide.md"), "guide v1\n");
        write(&repo.0.join("skills/beta-folder/SKILL.md"), "---\nname: \"beta\"\ndescription: Second.\n---\nBeta\n");
        write(&repo.0.join("README.md"), "not a skill\n");
        git(&repo.0, &["add", "-A"]);
        git(&repo.0, &["commit", "--quiet", "-m", "init"]);
        repo
    }

    #[test]
    fn specs_name_a_repository_and_maybe_a_skill() {
        assert_eq!(parse_spec("vercel-labs/skills"), Ok(("https://github.com/vercel-labs/skills.git".into(), None)));
        assert_eq!(
            parse_spec("vercel-labs/skills@find-skills"),
            Ok(("https://github.com/vercel-labs/skills.git".into(), Some("find-skills".into())))
        );
        assert_eq!(parse_spec("git@github.com:owner/repo.git"), Ok(("git@github.com:owner/repo.git".into(), None)));
        assert_eq!(
            parse_spec("https://example.com/owner/repo.git@alpha"),
            Ok(("https://example.com/owner/repo.git".into(), Some("alpha".into())))
        );
        assert_eq!(parse_spec("/tmp/repo@alpha"), Ok(("/tmp/repo".into(), Some("alpha".into()))));
        assert_eq!(parse_spec("../repo"), Ok(("../repo".into(), None)), "a local path is cloned as is");
        for bad in ["", "repo", "owner/repo/extra", "owner/..", "owner/re po"] {
            assert!(parse_spec(bad).is_err(), "{bad}");
        }
        assert!(valid_name("find-skills") && valid_name("a_b1"));
        for bad in ["", "..", "a/b", ".hidden", "a b", "é"] {
            assert!(!valid_name(bad), "{bad}");
        }
    }

    #[test]
    fn a_skill_is_staged_then_applied_then_updated() {
        let repo = upstream();
        let agents = temp_dir("agents");
        let root = agents.0.join("vendor");
        let url = repo.0.to_string_lossy().to_string();

        // Several skills and none asked for: the user picks.
        assert_eq!(stage(&root, &url, None, None, None).unwrap_err(), StageError::Choose(vec!["alpha".into(), "beta".into()]));
        assert!(matches!(stage(&root, &url, None, Some("gamma"), None), Err(StageError::Other(_))));

        // Staging installs nothing.
        let staged = stage(&root, &url, None, Some("alpha"), None).expect("stage");
        assert_eq!(staged.name, "alpha");
        assert_eq!(staged.source.path, "skills/alpha");
        assert!(staged.dir.join("references/guide.md").is_file());
        assert!(!root.join("alpha").exists(), "nothing stored before --apply");
        assert!(!root.join(UPDATES_DIR).join(format!(".clone-{}", std::process::id())).exists(), "the clone is removed");

        let other_repo = apply_staged(&root, "alpha", Some("https://example.com/other/repo.git")).unwrap_err();
        assert!(other_repo.contains("comes from"), "{other_repo}");
        assert!(!root.join("alpha").exists(), "a copy staged from another repository is not installed");
        let (applied, kept) = apply_staged(&root, "alpha", Some(&url)).expect("apply");
        assert_eq!(applied, staged.source);
        assert!(kept.is_none());
        assert_eq!(read_source(&root.join("alpha")), Some(staged.source.clone()));
        assert!(apply_staged(&root, "alpha", None).is_err(), "nothing staged any more");

        // The same skill shipped twice is one skill; two different ones need --path.
        write(&repo.0.join(".claude/skills/alpha/SKILL.md"), "---\nname: alpha\ndescription: First.\n---\nAlpha v1\n");
        write(&repo.0.join(".claude/skills/alpha/references/guide.md"), "guide v1\n");
        write(&repo.0.join("other/beta/SKILL.md"), "---\nname: beta\ndescription: Another one.\n---\nNot the same\n");
        git(&repo.0, &["add", "-A"]);
        git(&repo.0, &["commit", "--quiet", "-m", "mirror"]);
        assert_eq!(stage(&root, &url, None, None, None).unwrap_err(), StageError::Choose(vec!["alpha".into(), "beta".into()]));
        assert!(stage(&root, &url, None, Some("alpha"), None).is_ok(), "identical copies are one skill");
        let _ = std::fs::remove_dir_all(root.join(UPDATES_DIR).join("alpha"));
        let ambiguous = stage(&root, &url, None, Some("beta"), None).unwrap_err();
        assert!(matches!(&ambiguous, StageError::Other(message) if message.contains("--path")), "{ambiguous:?}");
        git(&repo.0, &["rm", "-r", "--quiet", "other", ".claude"]);
        git(&repo.0, &["commit", "--quiet", "-m", "one copy again"]);

        // The declared name wins over the folder name.
        let beta = stage(&root, &url, None, Some("beta"), None).expect("beta");
        assert_eq!(beta.source.path, "skills/beta-folder");

        // Upstream moves: the update is staged, the stored copy untouched until applied.
        write(&repo.0.join("skills/alpha/SKILL.md"), "---\nname: alpha\ndescription: First.\n---\nAlpha v2\n");
        git(&repo.0, &["commit", "--quiet", "-am", "v2"]);
        let update = stage(&root, &url, None, Some("alpha"), Some("skills/alpha")).expect("update");
        assert_ne!(update.source.commit, staged.source.commit);
        assert!(std::fs::read_to_string(root.join("alpha/SKILL.md")).unwrap().contains("Alpha v1"));
        assert!(report(&root, &update, "apply"), "a change is reported");
        let (_, kept) = apply_staged(&root, "alpha", Some(&url)).expect("apply update");
        let kept = kept.expect("previous copy kept");
        assert!(std::fs::read_to_string(kept.join("SKILL.md")).unwrap().contains("Alpha v1"));
        assert!(std::fs::read_to_string(root.join("alpha/SKILL.md")).unwrap().contains("Alpha v2"));

        // Same content upstream: reported up to date, nothing left staged.
        let again = stage(&root, &url, None, Some("alpha"), None).expect("again");
        assert!(!report(&root, &again, "apply"));
        assert!(!root.join(UPDATES_DIR).join("alpha").exists());
    }

    #[test]
    fn a_skill_at_the_root_takes_the_repository_name() {
        assert_eq!(repository_name("https://github.com/owner/my-skill.git").as_deref(), Some("my-skill"));
        assert_eq!(repository_name("git@github.com:owner/my_skill.git").as_deref(), Some("my_skill"));
        assert_eq!(repository_name("/tmp/some dir/"), None);

        let parent = temp_dir("root-skill");
        let repo = parent.0.join("solo-skill");
        std::fs::create_dir_all(&repo).unwrap();
        git(&repo, &["init", "--quiet", "--initial-branch", "main"]);
        write(&repo.join("SKILL.md"), "---\ndescription: No name declared.\n---\nSolo\n");
        git(&repo, &["add", "-A"]);
        git(&repo, &["commit", "--quiet", "-m", "init"]);
        let agents = temp_dir("agents");
        let root = agents.0.join("vendor");

        let staged = stage(&root, &repo.to_string_lossy(), None, None, None).expect("stage");
        assert_eq!(staged.name, "solo-skill", "not the clone folder's name");
        assert_eq!(staged.source.path, ".");
        assert!(staged.dir.join("SKILL.md").is_file(), "the staged copy is not the removed clone");
        assert!(!staged.dir.join(".git").exists());
    }

    #[test]
    fn a_path_cannot_leave_the_repository_and_links_are_left_out() {
        let repo = upstream();
        let agents = temp_dir("agents");
        let root = agents.0.join("vendor");
        let url = repo.0.to_string_lossy().to_string();
        for bad in ["../outside", "skills/../../etc", "/etc"] {
            let error = stage(&root, &url, None, Some("x"), Some(bad)).unwrap_err();
            assert!(matches!(&error, StageError::Other(message) if message.contains("inside the repository") || message.contains("no SKILL.md")), "{bad}: {error:?}");
        }
        assert!(!root.join(UPDATES_DIR).join("x").exists());

        #[cfg(unix)]
        {
            // A folder of the repository that is a link to somewhere else on the machine.
            let outside = temp_dir("outside");
            write(&outside.0.join("SKILL.md"), "---\nname: stolen\n---\nprivate\n");
            std::os::unix::fs::symlink(&outside.0, repo.0.join("linked")).unwrap();
            git(&repo.0, &["add", "-A"]);
            git(&repo.0, &["commit", "--quiet", "-m", "linked folder"]);
            let through_link = stage(&root, &url, None, Some("stolen"), Some("linked")).unwrap_err();
            assert!(matches!(&through_link, StageError::Other(message) if message.contains("symbolic link")), "{through_link:?}");
            assert!(!root.join(UPDATES_DIR).join("stolen").exists());

            std::os::unix::fs::symlink("/etc/hosts", repo.0.join("skills/alpha/hosts-link")).unwrap();
            git(&repo.0, &["add", "-A"]);
            git(&repo.0, &["commit", "--quiet", "-m", "link"]);
            let staged = stage(&root, &url, None, Some("alpha"), None).expect("stage");
            assert_eq!(staged.links, vec![PathBuf::from("hosts-link")]);
            assert!(!staged.dir.join("hosts-link").exists());
        }
    }
}
