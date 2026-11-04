//! Read-only access to the skills of `~/.MemoryPilot/agents/` over MCP, so a
//! client that does not load them from disk (Claude app, a remote agent) can
//! still find and read them.
//!
//! Three sources, in priority order when two skills share a name:
//! `skills/<name>/SKILL.md` (perso), `vendor/<name>/SKILL.md` (vendor) and
//! `plugins/<plugin>/skills/<name>/SKILL.md`, exposed as `<plugin>:<name>`
//! like Claude Code names plugin skills.

use serde_json::{json, Value};
use std::path::{Path, PathBuf};

/// Largest file `get_agent_skill` returns, to keep one call in a model context.
const MAX_FILE_BYTES: u64 = 256 * 1024;

#[derive(Debug, Clone, PartialEq)]
pub struct AgentSkill {
    /// Name to pass to `get_agent_skill`: `<name>` or `<plugin>:<name>`.
    pub name: String,
    pub description: String,
    /// `perso`, `vendor` or `plugin:<plugin>`.
    pub origin: String,
    pub dir: PathBuf,
}

pub fn agents_dir() -> Option<PathBuf> {
    dirs::home_dir().map(|home| home.join(".MemoryPilot").join("agents"))
}

fn sorted_dirs(dir: &Path) -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(dir)
        .map(|entries| {
            entries
                .flatten()
                .filter(|e| !e.file_name().to_string_lossy().starts_with('.'))
                .map(|e| e.path())
                .filter(|p| p.is_dir())
                .collect()
        })
        .unwrap_or_default();
    dirs.sort();
    dirs
}

/// Unquotes a YAML scalar written on one line: `"a \"b\""`, `'it''s'` or bare.
fn unquote(value: &str) -> String {
    let value = value.trim();
    if value.len() >= 2 && value.starts_with('"') && value.ends_with('"') {
        return value[1..value.len() - 1].replace("\\\"", "\"").replace("\\\\", "\\");
    }
    if value.len() >= 2 && value.starts_with('\'') && value.ends_with('\'') {
        return value[1..value.len() - 1].replace("''", "'");
    }
    value.to_string()
}

/// `name` and `description` of a SKILL.md frontmatter. Handles one-line
/// scalars and the `>` / `|` block scalars plugins often use.
pub fn parse_frontmatter(text: &str) -> (Option<String>, Option<String>) {
    let mut lines = text.lines();
    if lines.next().map(str::trim) != Some("---") {
        return (None, None);
    }
    let body: Vec<&str> = lines.take_while(|l| l.trim() != "---").collect();
    let field = |key: &str| -> Option<String> {
        let prefix = format!("{}:", key);
        let start = body.iter().position(|l| l.starts_with(&prefix))?;
        let rest = body[start][prefix.len()..].trim();
        if rest.starts_with('>') || rest.starts_with('|') {
            let block: Vec<&str> = body[start + 1..]
                .iter()
                .take_while(|l| l.trim().is_empty() || l.starts_with(' ') || l.starts_with('\t'))
                .map(|l| l.trim())
                .collect();
            let joiner = if rest.starts_with('>') { " " } else { "\n" };
            let text = block.join(joiner).trim().to_string();
            return (!text.is_empty()).then_some(text);
        }
        let value = unquote(rest);
        (!value.is_empty()).then_some(value)
    };
    (field("name"), field("description"))
}

fn read_skill(dir: &Path, name: String, origin: String) -> Option<AgentSkill> {
    let text = std::fs::read_to_string(dir.join("SKILL.md")).ok()?;
    let (_, description) = parse_frontmatter(&text);
    Some(AgentSkill { name, description: description.unwrap_or_default(), origin, dir: dir.to_path_buf() })
}

/// Every skill of the agents folder, perso first, then vendor, then plugins.
/// A vendor skill with the name of a perso skill is left out.
pub fn list(root: &Path) -> Vec<AgentSkill> {
    let mut skills: Vec<AgentSkill> = Vec::new();
    for (folder, origin) in [("skills", "perso"), ("vendor", "vendor")] {
        for dir in sorted_dirs(&root.join(folder)) {
            let name = dir.file_name().unwrap_or_default().to_string_lossy().to_string();
            if skills.iter().any(|s| s.name == name) {
                continue;
            }
            if let Some(skill) = read_skill(&dir, name, origin.to_string()) {
                skills.push(skill);
            }
        }
    }
    for plugin in sorted_dirs(&root.join("plugins")) {
        let plugin_name = plugin.file_name().unwrap_or_default().to_string_lossy().to_string();
        for dir in sorted_dirs(&plugin.join("skills")) {
            let name = format!("{}:{}", plugin_name, dir.file_name().unwrap_or_default().to_string_lossy());
            if let Some(skill) = read_skill(&dir, name, format!("plugin:{}", plugin_name)) {
                skills.push(skill);
            }
        }
    }
    skills
}

/// Finds a skill by its exact name, or by the bare name of a plugin skill
/// when only one plugin has it.
pub fn find<'a>(skills: &'a [AgentSkill], name: &str) -> Result<&'a AgentSkill, String> {
    if let Some(skill) = skills.iter().find(|s| s.name == name) {
        return Ok(skill);
    }
    let suffix = format!(":{}", name);
    let matches: Vec<&AgentSkill> = skills.iter().filter(|s| s.name.ends_with(&suffix)).collect();
    match matches.as_slice() {
        [one] => Ok(one),
        [] => Err(format!("No agent skill named '{}'. Call list_agent_skills for the names.", name)),
        many => Err(format!(
            "'{}' exists in several plugins: {}. Pass the full name.",
            name,
            many.iter().map(|s| s.name.as_str()).collect::<Vec<_>>().join(", ")
        )),
    }
}

/// Files of the skill folder other than SKILL.md, relative and sorted.
fn skill_files(dir: &Path) -> Vec<String> {
    let mut files = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(current) = stack.pop() {
        for entry in std::fs::read_dir(&current).into_iter().flatten().flatten() {
            let path = entry.path();
            if entry.file_name().to_string_lossy().starts_with('.') {
                continue;
            }
            if path.is_dir() {
                stack.push(path);
            } else if let Ok(rel) = path.strip_prefix(dir) {
                let rel = rel.to_string_lossy().to_string();
                if rel != "SKILL.md" {
                    files.push(rel);
                }
            }
        }
    }
    files.sort();
    files
}

/// Reads a file of the skill folder. `file` is relative to the folder and
/// may not leave it (no `..`, no absolute path, no symlink pointing outside).
fn read_skill_file(dir: &Path, file: &str) -> Result<String, String> {
    let rel = Path::new(file);
    if rel.is_absolute() || rel.components().any(|c| !matches!(c, std::path::Component::Normal(_))) {
        return Err(format!("'{}' must be a path inside the skill folder, e.g. references/guide.md.", file));
    }
    let root = dir.canonicalize().map_err(|e| format!("Skill folder unreadable: {}", e))?;
    let path = dir
        .join(rel)
        .canonicalize()
        .map_err(|_| format!("No file '{}' in this skill. Its files are listed by get_agent_skill without 'file'.", file))?;
    if !path.starts_with(&root) || !path.is_file() {
        return Err(format!("'{}' is not a file inside the skill folder.", file));
    }
    let size = path.metadata().map(|m| m.len()).unwrap_or(0);
    if size > MAX_FILE_BYTES {
        return Err(format!("'{}' is {} KB, above the {} KB limit of one call.", file, size / 1024, MAX_FILE_BYTES / 1024));
    }
    std::fs::read_to_string(&path).map_err(|_| format!("'{}' is not a text file.", file))
}

pub fn list_json(root: &Path) -> Value {
    let skills = list(root);
    json!({
        "count": skills.len(),
        "skills": skills.iter().map(|s| json!({
            "name": s.name,
            "description": s.description,
            "origin": s.origin,
        })).collect::<Vec<_>>(),
    })
}

pub fn get_json(root: &Path, name: &str, file: Option<&str>) -> Result<Value, String> {
    let skills = list(root);
    let skill = find(&skills, name)?;
    match file {
        Some(file) => Ok(json!({
            "name": skill.name,
            "file": file,
            "content": read_skill_file(&skill.dir, file)?,
        })),
        None => Ok(json!({
            "name": skill.name,
            "origin": skill.origin,
            "content": read_skill_file(&skill.dir, "SKILL.md")?,
            "files": skill_files(&skill.dir),
        })),
    }
}

/// One line for the MCP `instructions`: the skill names, so the client knows
/// they exist without paying for every description.
pub fn instructions_line(root: &Path) -> Option<String> {
    let skills = list(root);
    if skills.is_empty() {
        return None;
    }
    let names: Vec<&str> = skills.iter().map(|s| s.name.as_str()).collect();
    Some(format!(
        "Agent skills (the user's playbooks; before a task one of them covers, read it with get_agent_skill(name), descriptions in list_agent_skills): {}.",
        names.join(", ")
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(path: &Path, text: &str) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
    }

    /// Temporary folder removed when the test ends.
    struct TempRoot(PathBuf);

    impl TempRoot {
        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempRoot {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn fixture() -> TempRoot {
        static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let root = TempRoot(std::env::temp_dir().join(format!("memorypilot-agent-skills-{}-{}", std::process::id(), n)));
        let r = root.path();
        write(&r.join("skills/deploy/SKILL.md"), "---\nname: deploy\ndescription: \"Déploie \\\"vite\\\"\"\n---\n# Deploy\n");
        write(&r.join("skills/deploy/references/steps.md"), "étapes");
        write(&r.join("skills/.hidden/SKILL.md"), "---\nname: hidden\n---\n");
        write(&r.join("skills/empty/README.md"), "no SKILL.md");
        write(&r.join("vendor/deploy/SKILL.md"), "---\nname: deploy\ndescription: vendor copy\n---\n");
        write(&r.join("vendor/stripe/SKILL.md"), "---\nname: stripe\ndescription: 'Stripe, it''s here'\n---\n");
        write(&r.join("plugins/kit/skills/audit/SKILL.md"), "---\nname: audit\ndescription: >\n  Audits a page\n  in depth.\n---\n");
        write(&r.join("plugins/kit/skills/audit/reference/a.md"), "A");
        write(&r.join("plugins/other/skills/audit/SKILL.md"), "---\nname: audit\ndescription: other\n---\n");
        write(&r.join("plugins/other/skills/polish/SKILL.md"), "---\nname: polish\ndescription: |\n  line one\n  line two\n---\n");
        root
    }

    #[test]
    fn lists_perso_vendor_and_plugins_in_priority_order() {
        let root = fixture();
        let skills = list(root.path());
        let names: Vec<(&str, &str)> = skills.iter().map(|s| (s.name.as_str(), s.origin.as_str())).collect();
        assert_eq!(
            names,
            vec![
                ("deploy", "perso"),
                ("stripe", "vendor"),
                ("kit:audit", "plugin:kit"),
                ("other:audit", "plugin:other"),
                ("other:polish", "plugin:other"),
            ]
        );
        assert_eq!(skills[0].description, "Déploie \"vite\"");
        assert_eq!(skills[1].description, "Stripe, it's here");
        assert_eq!(skills[2].description, "Audits a page in depth.");
        assert_eq!(skills[4].description, "line one\nline two");
    }

    #[test]
    fn finds_bare_plugin_names_only_when_unique() {
        let root = fixture();
        let skills = list(root.path());
        assert_eq!(find(&skills, "polish").unwrap().name, "other:polish");
        assert!(find(&skills, "audit").unwrap_err().contains("kit:audit, other:audit"));
        assert!(find(&skills, "missing").is_err());
    }

    #[test]
    fn returns_skill_with_its_files_and_reads_one_file() {
        let root = fixture();
        let skill = get_json(root.path(), "deploy", None).unwrap();
        assert_eq!(skill["origin"], "perso");
        assert!(skill["content"].as_str().unwrap().contains("# Deploy"));
        assert_eq!(skill["files"], json!(["references/steps.md"]));
        let file = get_json(root.path(), "deploy", Some("references/steps.md")).unwrap();
        assert_eq!(file["content"], "étapes");
    }

    #[test]
    fn refuses_paths_outside_the_skill_folder() {
        let root = fixture();
        for bad in ["../../vendor/stripe/SKILL.md", "/etc/hosts", "references/../../x"] {
            assert!(get_json(root.path(), "deploy", Some(bad)).is_err(), "{}", bad);
        }
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(root.path().join("vendor/stripe/SKILL.md"), root.path().join("skills/deploy/link.md")).unwrap();
            assert!(get_json(root.path(), "deploy", Some("link.md")).is_err());
        }
    }

    #[test]
    fn instructions_line_lists_names_or_nothing() {
        let root = fixture();
        let line = instructions_line(root.path()).unwrap();
        assert!(line.ends_with("deploy, stripe, kit:audit, other:audit, other:polish."), "{}", line);
        assert!(instructions_line(&root.path().join("absent")).is_none());
    }
}
