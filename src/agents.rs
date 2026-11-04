//! The user's own rules and skills for every coding agent.
//!
//! `~/.MemoryPilot/agents/` holds `rules/AGENTS.md`, `skills/<name>/SKILL.md`
//! (Agent Skills format), `vendor/` and `plugins/<name>/` (whole plugins, see
//! `plugins.rs`): Claude Code installs them as plugins of the generated
//! `memorypilot` marketplace, Cursor links their `cursor/` variant into
//! `~/.cursor/plugins/local/`, and without one the other tools get their skills. MemoryPilot ships none of them: it
//! only creates the empty folder and wires it into the tools the user picked.
//!
//! `MemoryPilot --setup-agents [--tools a,b] [--dry-run]` : creates the folder
//! when missing (never touches an existing one), wires the tools, writes
//! `agents/.manifest.json`.
//!
//! `MemoryPilot --sync-agents` : run at every session start. Compares the
//! inputs (AGENTS.md, skill names, tools) with the manifest and writes
//! nothing when they match. Never fails: exit 0, errors go to a log.
//!
//! `MemoryPilot --uninstall-agents [--tools a,b]` : removes exactly what the
//! manifest lists and is still unmodified, never the folder itself.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

const MANIFEST_FILE: &str = ".manifest.json";
const SYNC_LOG: &str = "agents-sync.log";
const MANIFEST_VERSION: u32 = 1;
/// Cursor reading `~/.cursor/rules/agents.mdc` is not documented. Single
/// switch: set to false and the next sync removes the generated rule.
const CURSOR_RULE_FROM_AGENTS_MD: bool = true;
const BLOCK_BEGIN: &str = "<!-- MemoryPilot agents: begin -->";
const BLOCK_END: &str = "<!-- MemoryPilot agents: end -->";
const IMPORT_MARKER: &str = "<!-- MemoryPilot agents: your rules, imported from MemoryPilot -->";
const LINKED_RULE_NAME: &str = "memorypilot-agents.md";
const RULES_TEMPLATE: &str = "<!--
Your personal rules for every coding agent, in plain Markdown.
MemoryPilot wires this file into the tools you picked (Claude Code, Codex,
Cursor, Gemini CLI, ...). Nothing here comes from MemoryPilot: write your own.

Example:
- Answer in English, keep explanations short.
- Run the tests before saying a change is done.

Skills go in ../skills/<name>/SKILL.md (Agent Skills format).
-->
";

/// (id, display name). Tools reading `~/.agents/skills` are every one but Claude Code.
const TOOLS: &[(&str, &str)] = &[
    ("claude", "Claude Code"),
    ("cursor", "Cursor"),
    ("codex", "Codex"),
    ("gemini", "Gemini CLI"),
    ("copilot", "VS Code Copilot"),
    ("windsurf", "Windsurf / Devin"),
    ("opencode", "OpenCode"),
    ("cline", "Cline"),
    ("roo", "Roo Code"),
];
const SHARED_SKILLS: &str = "shared";

fn tool_name(id: &str) -> &str {
    TOOLS.iter().find(|(t, _)| *t == id).map(|(_, n)| *n).unwrap_or(id)
}

struct Paths {
    home: PathBuf,
    agents: PathBuf,
    claude: PathBuf,
    codex: PathBuf,
    gemini: PathBuf,
    opencode: PathBuf,
}

fn env_dir(key: &str) -> Option<PathBuf> {
    std::env::var(key).ok().filter(|v| !v.trim().is_empty()).map(PathBuf::from)
}

impl Paths {
    fn from_env() -> Option<Paths> {
        let home = dirs::home_dir()?;
        Some(Paths {
            agents: home.join(".MemoryPilot").join("agents"),
            claude: env_dir("CLAUDE_CONFIG_DIR").unwrap_or_else(|| home.join(".claude")),
            codex: env_dir("CODEX_HOME").unwrap_or_else(|| home.join(".codex")),
            // GEMINI_CLI_HOME replaces the home directory, `.gemini` stays.
            gemini: env_dir("GEMINI_CLI_HOME").unwrap_or_else(|| home.clone()).join(".gemini"),
            opencode: env_dir("OPENCODE_CONFIG_DIR").unwrap_or_else(|| home.join(".config").join("opencode")),
            home,
        })
    }

    fn rules(&self) -> PathBuf {
        self.agents.join("rules").join("AGENTS.md")
    }

    fn skills(&self) -> PathBuf {
        self.agents.join("skills")
    }

    fn plugins(&self) -> PathBuf {
        self.agents.join("plugins")
    }

    fn manifest(&self) -> PathBuf {
        self.agents.join(MANIFEST_FILE)
    }

    fn vscode_storage(&self, extension: &str) -> [PathBuf; 2] {
        [
            self.home.join("Library/Application Support/Code/User/globalStorage").join(extension),
            self.home.join(".config/Code/User/globalStorage").join(extension),
        ]
    }
}

fn on_path(command: &str) -> bool {
    std::env::var_os("PATH")
        .map(|path| std::env::split_paths(&path).any(|dir| dir.join(command).is_file()))
        .unwrap_or(false)
}

/// A tool is offered only when its config folder or its command exists.
fn detected(paths: &Paths, id: &str) -> bool {
    let home = &paths.home;
    match id {
        "claude" => paths.claude.is_dir() || on_path("claude"),
        "cursor" => home.join(".cursor").is_dir() || on_path("cursor"),
        "codex" => paths.codex.is_dir() || on_path("codex"),
        "gemini" => paths.gemini.is_dir() || on_path("gemini"),
        "copilot" => home.join(".copilot").is_dir() || home.join(".vscode").is_dir() || on_path("code"),
        "windsurf" => home.join(".codeium/windsurf").is_dir() || on_path("windsurf"),
        "opencode" => paths.opencode.is_dir() || on_path("opencode"),
        "cline" => {
            home.join("Documents/Cline").is_dir()
                || paths.vscode_storage("saoudrizwan.claude-dev").iter().any(|p| p.is_dir())
        }
        "roo" => {
            home.join(".roo").is_dir()
                || paths.vscode_storage("rooveterinaryinc.roo-cline").iter().any(|p| p.is_dir())
        }
        _ => false,
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Entry {
    /// Symbolic link `path` → `target`.
    Link {
        tool: String,
        path: PathBuf,
        target: PathBuf,
        #[serde(default)]
        adopted: bool,
    },
    /// Marker comment + `@import` line appended to `path`.
    Import {
        tool: String,
        path: PathBuf,
        text: String,
        #[serde(default)]
        created_file: bool,
    },
    /// Generated block between markers; `sha256` of the whole block.
    Block {
        tool: String,
        path: PathBuf,
        sha256: String,
        #[serde(default)]
        created_file: bool,
    },
    /// Whole generated file.
    File { tool: String, path: PathBuf, sha256: String },
    /// `value` added to the JSON array `key` of `path`.
    JsonArray {
        tool: String,
        path: PathBuf,
        key: String,
        value: String,
        #[serde(default)]
        created_file: bool,
    },
    /// Plugin folder `path` installed in Claude Code as `<name>@memorypilot`;
    /// `sha256` of its content when last installed or updated.
    ClaudePlugin { tool: String, path: PathBuf, name: String, sha256: String },
}

impl Entry {
    fn tool(&self) -> &str {
        match self {
            Entry::Link { tool, .. }
            | Entry::Import { tool, .. }
            | Entry::Block { tool, .. }
            | Entry::File { tool, .. }
            | Entry::JsonArray { tool, .. }
            | Entry::ClaudePlugin { tool, .. } => tool,
        }
    }

    fn path(&self) -> &Path {
        match self {
            Entry::Link { path, .. }
            | Entry::Import { path, .. }
            | Entry::Block { path, .. }
            | Entry::File { path, .. }
            | Entry::JsonArray { path, .. }
            | Entry::ClaudePlugin { path, .. } => path,
        }
    }
}

#[derive(Serialize, Deserialize, Default, Debug)]
struct Manifest {
    version: u32,
    agents_dir: PathBuf,
    tools: Vec<String>,
    /// Fingerprint of everything the wiring derives from: `--sync-agents`
    /// does nothing while it matches.
    inputs: String,
    entries: Vec<Entry>,
}

#[derive(Default)]
struct Report {
    done: Vec<String>,
    skipped: Vec<String>,
    errors: Vec<String>,
}

struct Ctx<'a> {
    paths: &'a Paths,
    dry_run: bool,
    report: Report,
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|byte| format!("{:02x}", byte)).collect()
}

fn display(paths: &Paths, path: &Path) -> String {
    match path.strip_prefix(&paths.home) {
        Ok(rest) => format!("~/{}", rest.display()),
        Err(_) => path.display().to_string(),
    }
}

/// Skill folders holding a `SKILL.md`, sorted.
fn list_skills(paths: &Paths) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(paths.skills())
        .map(|entries| {
            entries
                .flatten()
                .filter_map(|e| e.file_name().into_string().ok())
                .filter(|name| !name.starts_with('.'))
                .filter(|name| paths.skills().join(name).join("SKILL.md").is_file())
                .collect()
        })
        .unwrap_or_default();
    names.sort();
    names
}

fn inputs_fingerprint(paths: &Paths, tools: &[String]) -> String {
    let rules = std::fs::read(paths.rules()).map(|b| sha256_hex(&b)).unwrap_or_else(|_| "none".into());
    let plugins: Vec<String> = crate::plugins::list(&paths.plugins())
        .iter()
        .map(|p| format!("{}:{}:{}", p.name, crate::plugins::claude_fingerprint(&p.dir), p.cursor_dir.is_some()))
        .collect();
    let cursor_skills: Vec<String> = cursor_skills_dir(paths)
        .and_then(|dir| std::fs::read_dir(dir).ok())
        .map(|entries| entries.flatten().map(|e| e.file_name().to_string_lossy().to_string()).collect())
        .unwrap_or_default();
    let mut cursor_skills = cursor_skills;
    cursor_skills.sort();
    let text = format!(
        "v{}|{}|cursor_rule={}|{}|{}|{}|cursor_skills={}|{}|{}|{}|{}|{}",
        MANIFEST_VERSION,
        tools.join(","),
        CURSOR_RULE_FROM_AGENTS_MD,
        rules,
        list_skills(paths).join("/"),
        plugins.join("/"),
        cursor_skills.join("/"),
        paths.agents.display(),
        paths.claude.display(),
        paths.codex.display(),
        paths.gemini.display(),
        paths.opencode.display()
    );
    sha256_hex(text.as_bytes())
}

fn load_manifest(paths: &Paths) -> Result<Option<Manifest>, String> {
    match std::fs::read_to_string(paths.manifest()) {
        Ok(text) => serde_json::from_str(&text)
            .map(Some)
            .map_err(|e| format!("{}: {}", paths.manifest().display(), e)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(format!("{}: {}", paths.manifest().display(), e)),
    }
}

/// Atomic write, skipped when the content is already there.
fn write_if_changed(path: &Path, content: &str) -> Result<bool, String> {
    if std::fs::read_to_string(path).ok().as_deref() == Some(content) {
        return Ok(false);
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("{}: {}", parent.display(), e))?;
    }
    let tmp = path.with_extension("mp-tmp");
    std::fs::write(&tmp, content).map_err(|e| format!("{}: {}", tmp.display(), e))?;
    std::fs::rename(&tmp, path).map_err(|e| format!("{}: {}", path.display(), e))?;
    Ok(true)
}

/// Rewrites `agents/catalog.json` when its content changes (see catalog.rs).
fn write_catalog(paths: &Paths, tools: &[String]) -> Result<bool, String> {
    let ctx = crate::catalog::Ctx { home: &paths.home, agents: &paths.agents, claude: &paths.claude, tools };
    write_if_changed(&paths.agents.join("catalog.json"), &ctx.render())
}

fn save_manifest(paths: &Paths, manifest: &Manifest) -> Result<bool, String> {
    let rendered = serde_json::to_string_pretty(manifest).map_err(|e| e.to_string())? + "\n";
    write_if_changed(&paths.manifest(), &rendered)
}

fn same_target(link: &Path, target: &Path) -> bool {
    match std::fs::read_link(link) {
        Ok(points_to) if points_to == target => true,
        Ok(_) => match (link.canonicalize(), target.canonicalize()) {
            (Ok(a), Ok(b)) => a == b,
            _ => false,
        },
        Err(_) => false,
    }
}

fn is_symlink(path: &Path) -> bool {
    std::fs::symlink_metadata(path).map(|m| m.file_type().is_symlink()).unwrap_or(false)
}

#[cfg(unix)]
fn make_symlink(target: &Path, link: &Path) -> std::io::Result<()> {
    std::os::unix::fs::symlink(target, link)
}

#[cfg(not(unix))]
fn make_symlink(_target: &Path, _link: &Path) -> std::io::Result<()> {
    Err(std::io::Error::other("symbolic links are only created on macOS and Linux"))
}

/// Appends `text` after the existing content, one blank line apart.
fn append_text(content: &str, text: &str) -> String {
    if content.is_empty() {
        text.to_string()
    } else if content.ends_with("\n\n") {
        format!("{}{}", content, text)
    } else if content.ends_with('\n') {
        format!("{}\n{}", content, text)
    } else {
        format!("{}\n\n{}", content, text)
    }
}

/// Removes `start..end` and the blank line `append_text` put before it.
fn remove_span(content: &str, start: usize, end: usize) -> String {
    let before = &content[..start];
    let after = &content[end..];
    if after.trim().is_empty() {
        let kept = before.trim_end_matches('\n');
        if kept.is_empty() {
            String::new()
        } else {
            format!("{}\n", kept)
        }
    } else if before.ends_with("\n\n") && after.starts_with('\n') {
        format!("{}{}", before, &after[1..])
    } else {
        format!("{}{}", before, after)
    }
}

/// Byte span of the generated block (markers included, with the newline after the end marker).
fn find_block(content: &str) -> Option<(usize, usize)> {
    let start = content.find(BLOCK_BEGIN)?;
    let end_marker = start + content[start..].find(BLOCK_END)?;
    let mut end = end_marker + BLOCK_END.len();
    if content[end..].starts_with('\n') {
        end += 1;
    }
    Some((start, end))
}

fn render_block(rules: &str) -> String {
    let mut body = rules.to_string();
    if !body.ends_with('\n') {
        body.push('\n');
    }
    format!(
        "{}\n<!-- Generated from ~/.MemoryPilot/agents/rules/AGENTS.md by MemoryPilot: edit that file, not this block. -->\n{}{}\n",
        BLOCK_BEGIN, body, BLOCK_END
    )
}

fn render_cursor_rule(rules: &str) -> String {
    format!(
        "---\ndescription: Personal rules from ~/.MemoryPilot/agents/rules/AGENTS.md (generated by MemoryPilot, edit that file instead)\nglobs:\nalwaysApply: true\n---\n{}",
        rules
    )
}

fn import_text(rules: &Path) -> String {
    format!("{}\n@{}\n", IMPORT_MARKER, rules.display())
}

/// What the wiring of the given tools should be.
fn desired(paths: &Paths, tools: &[String]) -> Vec<Entry> {
    let has = |id: &str| tools.iter().any(|t| t == id);
    let rules_path = paths.rules();
    let rules = std::fs::read_to_string(&rules_path).ok();
    let skills = list_skills(paths);
    let mut out = Vec::new();
    let skill_links = |out: &mut Vec<Entry>, tool: &str, dir: PathBuf| {
        for name in &skills {
            out.push(Entry::Link {
                tool: tool.into(),
                path: dir.join(name),
                target: paths.skills().join(name),
                adopted: false,
            });
        }
    };
    if has("claude") {
        if rules.is_some() {
            out.push(Entry::Import {
                tool: "claude".into(),
                path: paths.claude.join("CLAUDE.md"),
                text: import_text(&rules_path),
                created_file: false,
            });
        }
        skill_links(&mut out, "claude", paths.claude.join("skills"));
    }
    if tools.iter().any(|t| t != "claude") {
        skill_links(&mut out, SHARED_SKILLS, paths.home.join(".agents").join("skills"));
    }
    let only_cursor_reads_shared = tools.iter().all(|t| t == "claude" || t == "cursor");
    if let Some(cursor_skills) = cursor_skills_dir(paths).filter(|_| has("cursor") && only_cursor_reads_shared) {
        // Cursor also reads ~/.cursor/skills: a skill already there would be
        // seen twice, so it gets no second link in ~/.agents/skills. Kept when
        // another tool reads ~/.agents/skills: it must not lose the skill.
        out.retain(|e| match e {
            Entry::Link { tool, path, .. } if tool == SHARED_SKILLS => path
                .file_name()
                .is_none_or(|name| std::fs::symlink_metadata(cursor_skills.join(name)).is_err()),
            _ => true,
        });
    }
    for plugin in crate::plugins::list(&paths.plugins()) {
        if has("claude") {
            out.push(Entry::ClaudePlugin {
                tool: "claude".into(),
                sha256: crate::plugins::claude_fingerprint(&plugin.dir),
                path: plugin.dir.clone(),
                name: plugin.name.clone(),
            });
        }
        match &plugin.cursor_dir {
            // Cursor gets the plugin in its own format; its skills are not
            // linked again into ~/.agents/skills, which Cursor reads too.
            Some(cursor_dir) if has("cursor") => out.push(Entry::Link {
                tool: "cursor".into(),
                path: paths.home.join(".cursor/plugins/local").join(&plugin.name),
                target: cursor_dir.clone(),
                adopted: false,
            }),
            Some(_) => {}
            None if tools.iter().any(|t| t != "claude") => {
                for dir in list_plugin_skills(&plugin.dir) {
                    let name = dir.file_name().unwrap_or_default().to_os_string();
                    out.push(Entry::Link {
                        tool: SHARED_SKILLS.into(),
                        path: paths.home.join(".agents/skills").join(name),
                        target: dir,
                        adopted: false,
                    });
                }
            }
            None => {}
        }
    }
    let Some(rules) = rules else {
        return out;
    };
    if has("cursor") && CURSOR_RULE_FROM_AGENTS_MD {
        out.push(Entry::File {
            tool: "cursor".into(),
            path: paths.home.join(".cursor/rules/agents.mdc"),
            sha256: sha256_hex(render_cursor_rule(&rules).as_bytes()),
        });
    }
    let block_sha = sha256_hex(render_block(&rules).as_bytes());
    if has("codex") {
        out.push(Entry::Block {
            tool: "codex".into(),
            path: paths.codex.join("AGENTS.md"),
            sha256: block_sha.clone(),
            created_file: false,
        });
    }
    if has("gemini") {
        out.push(Entry::Import {
            tool: "gemini".into(),
            path: paths.gemini.join("GEMINI.md"),
            text: import_text(&rules_path),
            created_file: false,
        });
    }
    if has("copilot") {
        out.push(Entry::Block {
            tool: "copilot".into(),
            path: paths.home.join(".copilot/copilot-instructions.md"),
            sha256: block_sha,
            created_file: false,
        });
    }
    if has("opencode") {
        out.push(Entry::JsonArray {
            tool: "opencode".into(),
            path: paths.opencode.join("opencode.json"),
            key: "instructions".into(),
            value: rules_path.display().to_string(),
            created_file: false,
        });
    }
    for (tool, dir) in [
        ("cline", paths.home.join("Documents/Cline/Rules")),
        ("roo", paths.home.join(".roo/rules")),
    ] {
        if has(tool) {
            out.push(Entry::Link {
                tool: tool.into(),
                path: dir.join(LINKED_RULE_NAME),
                target: rules_path.clone(),
                adopted: false,
            });
        }
    }
    out
}

/// Skill folders of a plugin (`skills/<name>/SKILL.md`), sorted.
fn list_plugin_skills(plugin: &Path) -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(plugin.join("skills"))
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.join("SKILL.md").is_file())
        .collect();
    dirs.sort();
    dirs
}

fn same_slot(a: &Entry, b: &Entry) -> bool {
    if let (Entry::ClaudePlugin { name: a_name, .. }, Entry::ClaudePlugin { name: b_name, .. }) = (a, b) {
        // A renamed plugin is another plugin for Claude Code: the old one is uninstalled.
        return a_name == b_name && a.path() == b.path();
    }
    std::mem::discriminant(a) == std::mem::discriminant(b) && a.path() == b.path()
}

/// `~/.cursor/skills` when it is a real folder of its own. `None` when it is
/// missing or resolves into the agents folder (then its entries are ours, not
/// Cursor's, and must neither be cleaned nor taken as taken names).
fn cursor_skills_dir(paths: &Paths) -> Option<PathBuf> {
    let dir = paths.home.join(".cursor/skills");
    let real = dir.canonicalize().ok()?;
    let agents = paths.agents.canonicalize().unwrap_or_else(|_| paths.agents.clone());
    (!is_symlink(&dir) && !real.starts_with(&agents)).then_some(dir)
}

impl Ctx<'_> {
    fn did(&mut self, what: String) {
        self.report.done.push(if self.dry_run { format!("would {}", what) } else { what });
    }

    fn skip(&mut self, what: String) {
        self.report.skipped.push(what);
    }

    fn fail(&mut self, what: String) {
        self.report.errors.push(what);
    }

    fn write(&mut self, path: &Path, content: &str, what: String) -> bool {
        if std::fs::read_to_string(path).ok().as_deref() == Some(content) {
            return true;
        }
        if self.dry_run {
            self.did(what);
            return true;
        }
        match write_if_changed(path, content) {
            Ok(changed) => {
                if changed {
                    self.did(what);
                }
                true
            }
            Err(e) => {
                self.fail(e);
                false
            }
        }
    }

    fn remove_file(&mut self, path: &Path, what: String) {
        if self.dry_run {
            self.did(what);
            return;
        }
        match std::fs::remove_file(path) {
            Ok(()) => self.did(what),
            Err(e) => self.fail(format!("{}: {}", path.display(), e)),
        }
    }

    /// Brings one desired entry about. Returns what the manifest should
    /// record, or None when the slot is someone else's.
    fn ensure(&mut self, want: Entry, old: Option<&Entry>) -> Option<Entry> {
        let shown = display(self.paths, want.path());
        match want {
            Entry::Link { tool, path, target, .. } => {
                if std::fs::symlink_metadata(&path).is_ok() {
                    if same_target(&path, &target) {
                        let adopted = match old {
                            Some(Entry::Link { adopted, .. }) => *adopted,
                            _ => true,
                        };
                        return Some(Entry::Link { tool, path, target, adopted });
                    }
                    self.skip(format!("{} (name already taken)", shown));
                    return None;
                }
                if !self.dry_run {
                    if let Some(parent) = path.parent() {
                        if let Err(e) = std::fs::create_dir_all(parent) {
                            self.fail(format!("{}: {}", parent.display(), e));
                            return None;
                        }
                    }
                    if let Err(e) = make_symlink(&target, &path) {
                        self.fail(format!("{}: {}", path.display(), e));
                        return None;
                    }
                }
                self.did(format!("link {}", shown));
                Some(Entry::Link { tool, path, target, adopted: false })
            }
            Entry::Import { tool, path, text, .. } => {
                if is_symlink(&path) {
                    self.skip(format!(
                        "{} is a symbolic link: not modified (Claude Cowork ignores a linked CLAUDE.md). Add this line to the real file yourself: {}",
                        shown,
                        text.lines().nth(1).unwrap_or_default()
                    ));
                    return None;
                }
                let existing = std::fs::read_to_string(&path).ok();
                let created_before = matches!(old, Some(Entry::Import { created_file: true, .. }));
                let content = existing.clone().unwrap_or_default();
                if content.contains(&text) {
                    return Some(Entry::Import { tool, path, text, created_file: created_before });
                }
                let import_line = text.lines().nth(1).unwrap_or_default().to_string();
                if content.lines().any(|line| line.trim() == import_line) {
                    // The user imports it already: theirs, not recorded.
                    return None;
                }
                let updated = append_text(&content, &text);
                let ok = self.write(&path, &updated, format!("import rules in {}", shown));
                ok.then_some(Entry::Import {
                    tool,
                    path,
                    text,
                    created_file: created_before || existing.is_none(),
                })
            }
            Entry::Block { tool, path, sha256, .. } => {
                let rules = std::fs::read_to_string(self.paths.rules()).unwrap_or_default();
                let block = render_block(&rules);
                let existing = std::fs::read_to_string(&path).ok();
                let created_before = matches!(old, Some(Entry::Block { created_file: true, .. }));
                let content = existing.clone().unwrap_or_default();
                let updated = match find_block(&content) {
                    Some((start, end)) => {
                        let current = &content[start..end];
                        if current == block {
                            return Some(Entry::Block { tool, path, sha256, created_file: created_before });
                        }
                        let ours = matches!(old, Some(Entry::Block { sha256: old_sha, .. }) if *old_sha == sha256_hex(current.as_bytes()));
                        if !ours {
                            self.skip(format!("{}: MemoryPilot block edited by hand, not regenerated", shown));
                            return None;
                        }
                        format!("{}{}{}", &content[..start], block, &content[end..])
                    }
                    None => append_text(&content, &block),
                };
                let ok = self.write(&path, &updated, format!("write rules block in {}", shown));
                ok.then_some(Entry::Block {
                    tool,
                    path,
                    sha256,
                    created_file: created_before || existing.is_none(),
                })
            }
            Entry::File { tool, path, sha256 } => {
                let rules = std::fs::read_to_string(self.paths.rules()).unwrap_or_default();
                let wanted = render_cursor_rule(&rules);
                if let Ok(current) = std::fs::read_to_string(&path) {
                    if current == wanted {
                        return Some(Entry::File { tool, path, sha256 });
                    }
                    let ours = matches!(old, Some(Entry::File { sha256: old_sha, .. }) if *old_sha == sha256_hex(current.as_bytes()));
                    if !ours {
                        self.skip(format!("{} (file already exists or edited by hand)", shown));
                        return None;
                    }
                } else if path.exists() || is_symlink(&path) {
                    self.skip(format!("{} (cannot be read)", shown));
                    return None;
                }
                let ok = self.write(&path, &wanted, format!("generate {}", shown));
                ok.then_some(Entry::File { tool, path, sha256 })
            }
            Entry::JsonArray { tool, path, key, value, .. } => {
                let created_before = matches!(old, Some(Entry::JsonArray { created_file: true, .. }));
                let existing = std::fs::read_to_string(&path).ok();
                let text = existing.clone().unwrap_or_else(|| "{}\n".to_string());
                let config = match serde_json::from_str::<Value>(&text) {
                    Ok(v) if v.is_object() => v,
                    _ => {
                        self.skip(format!("{}: not plain JSON, add \"{}\" to \"{}\" yourself", shown, value, key));
                        return None;
                    }
                };
                match config.get(key.as_str()) {
                    Some(Value::Array(items)) if items.iter().any(|v| v.as_str() == Some(value.as_str())) => {
                        // Already listed: ours only if recorded before.
                        return old.is_some().then_some(Entry::JsonArray { tool, path, key, value, created_file: created_before });
                    }
                    Some(Value::Array(_)) | None => {}
                    Some(_) => {
                        self.skip(format!("{}: \"{}\" is not an array", shown, key));
                        return None;
                    }
                }
                let Some(updated) = json_array_add(&text, &key, &value) else {
                    self.skip(format!("{}: could not be edited safely, add \"{}\" to \"{}\" yourself", shown, value, key));
                    return None;
                };
                let ok = self.write(&path, &updated, format!("add rules to {}", shown));
                ok.then_some(Entry::JsonArray { tool, path, key, value, created_file: created_before || existing.is_none() })
            }
            Entry::ClaudePlugin { tool, path, name, sha256 } => {
                let id = format!("{}@{}", name, crate::plugins::MARKETPLACE);
                let root = self.paths.plugins();
                let market = crate::plugins::marketplace_path(&root);
                if !self.write(&market, &crate::plugins::render_marketplace(&root), format!("generate {}", display(self.paths, &market))) {
                    return None;
                }
                let claude = self.paths.claude.clone();
                let home = self.paths.home.clone();
                if !crate::plugins::marketplace_known(&claude) {
                    if self.dry_run {
                        self.did(format!("add marketplace {} to Claude Code", crate::plugins::MARKETPLACE));
                    } else if let Err(e) = crate::plugins::claude_cli(&home, &["plugin", "marketplace", "add", &root.to_string_lossy()]) {
                        self.fail(e);
                        return old.cloned();
                    } else {
                        self.did(format!("add marketplace {} to Claude Code", crate::plugins::MARKETPLACE));
                    }
                }
                let installed = crate::plugins::plugin_installed(&claude, &name);
                let changed = !matches!(old, Some(Entry::ClaudePlugin { sha256: old_sha, .. }) if *old_sha == sha256);
                let steps: Vec<Vec<&str>> = if !installed {
                    vec![vec!["plugin", "install", &id, "--scope", "user"]]
                } else if changed {
                    vec![vec!["plugin", "marketplace", "update", crate::plugins::MARKETPLACE], vec!["plugin", "update", &id, "--scope", "user"]]
                } else {
                    return Some(Entry::ClaudePlugin { tool, path, name, sha256 });
                };
                let what = if installed { format!("update Claude Code plugin {}", id) } else { format!("install Claude Code plugin {}", id) };
                if !self.dry_run {
                    for step in &steps {
                        if let Err(e) = crate::plugins::claude_cli(&home, step) {
                            self.fail(e);
                            // Keep the previous print so the next sync tries again.
                            return old.cloned().filter(|_| installed);
                        }
                    }
                }
                self.did(what);
                Some(Entry::ClaudePlugin { tool, path, name, sha256 })
            }
        }
    }

    /// Undoes one recorded entry if it is still exactly ours. False only when
    /// a Claude Code plugin could not be uninstalled: the entry must stay recorded.
    fn remove(&mut self, entry: &Entry) -> bool {
        if let Entry::ClaudePlugin { name, .. } = entry {
            return self.uninstall_claude_plugin(name);
        }
        self.remove_wiring(entry);
        true
    }

    fn uninstall_claude_plugin(&mut self, name: &str) -> bool {
        if !crate::plugins::plugin_installed(&self.paths.claude, name) {
            return true;
        }
        let id = format!("{}@{}", name, crate::plugins::MARKETPLACE);
        if !self.dry_run {
            if let Err(e) = crate::plugins::claude_cli(&self.paths.home, &["plugin", "uninstall", &id, "--scope", "user"]) {
                self.fail(e);
                return false;
            }
        }
        self.did(format!("uninstall Claude Code plugin {}", id));
        true
    }

    fn remove_wiring(&mut self, entry: &Entry) {
        let shown = display(self.paths, entry.path());
        match entry {
            Entry::Link { path, target, .. } => {
                if is_symlink(path) && same_target(path, target) {
                    self.remove_file(path, format!("unlink {}", shown));
                }
            }
            Entry::Import { path, text, created_file, .. } => {
                if is_symlink(path) {
                    return;
                }
                let Ok(content) = std::fs::read_to_string(path) else { return };
                let Some(start) = content.find(text.as_str()) else { return };
                let updated = remove_span(&content, start, start + text.len());
                self.drop_or_write(path, &updated, *created_file, format!("remove rules import from {}", shown));
            }
            Entry::Block { path, sha256, created_file, .. } => {
                let Ok(content) = std::fs::read_to_string(path) else { return };
                let Some((start, end)) = find_block(&content) else { return };
                if sha256_hex(&content.as_bytes()[start..end]) != *sha256 {
                    self.skip(format!("{}: MemoryPilot block edited by hand, left in place", shown));
                    return;
                }
                let updated = remove_span(&content, start, end);
                self.drop_or_write(path, &updated, *created_file, format!("remove rules block from {}", shown));
            }
            Entry::File { path, sha256, .. } => {
                match std::fs::read(path) {
                    Ok(bytes) if sha256_hex(&bytes) == *sha256 => self.remove_file(path, format!("remove {}", shown)),
                    Ok(_) => self.skip(format!("{}: edited by hand, left in place", shown)),
                    Err(_) => {}
                }
            }
            Entry::JsonArray { path, key, value, created_file, .. } => {
                let Ok(text) = std::fs::read_to_string(path) else { return };
                let Some(updated) = json_array_remove(&text, key, value) else { return };
                let left = serde_json::from_str::<Value>(&updated).ok();
                let only_ours = left.as_ref().and_then(|v| v.as_object()).is_some_and(|map| {
                    map.len() == 1 && map.get(key.as_str()).and_then(|v| v.as_array()).is_some_and(|a| a.is_empty())
                });
                if *created_file && only_ours {
                    self.remove_file(path, format!("remove {}", shown));
                } else {
                    self.write(path, &updated, format!("remove rules from {}", shown));
                }
            }
            Entry::ClaudePlugin { .. } => {}
        }
    }

    /// Once no plugin is installed through it any more, removes the
    /// `memorypilot` marketplace from Claude Code and its generated file.
    fn drop_marketplace_if_unused(&mut self, entries: &[Entry]) {
        if entries.iter().any(|e| matches!(e, Entry::ClaudePlugin { .. })) {
            return;
        }
        if crate::plugins::marketplace_known(&self.paths.claude) {
            if !self.dry_run {
                if let Err(e) = crate::plugins::claude_cli(&self.paths.home, &["plugin", "marketplace", "remove", crate::plugins::MARKETPLACE]) {
                    self.fail(e);
                    return;
                }
            }
            self.did(format!("remove marketplace {} from Claude Code", crate::plugins::MARKETPLACE));
        }
        let market = crate::plugins::marketplace_path(&self.paths.plugins());
        if market.is_file() {
            self.remove_file(&market, format!("remove {}", display(self.paths, &market)));
        }
    }

    fn drop_or_write(&mut self, path: &Path, updated: &str, created_file: bool, what: String) {
        if created_file && updated.trim().is_empty() {
            self.remove_file(path, what);
        } else {
            self.write(path, updated, what);
        }
    }

    /// Makes the wiring match `tools`: removes recorded entries no longer
    /// wanted, creates or regenerates the wanted ones. Writes only what differs.
    /// Links in `~/.cursor/skills` that point into the agents folder: the
    /// wiring made by hand before MemoryPilot. Cursor gets the same skills
    /// from `~/.agents/skills`, so they would make every skill appear twice.
    fn remove_legacy_cursor_links(&mut self) {
        let Some(dir) = cursor_skills_dir(self.paths) else { return };
        let Ok(skills) = self.paths.skills().canonicalize() else { return };
        let relinked = list_skills(self.paths);
        let Ok(entries) = std::fs::read_dir(&dir) else { return };
        // Only links to a skill MemoryPilot links again in ~/.agents/skills:
        // anything else (vendor, plugins, other folders) is left for Cursor.
        let mut links: Vec<PathBuf> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| is_symlink(p))
            .filter(|p| {
                p.canonicalize().is_ok_and(|target| {
                    target.parent() == Some(skills.as_path())
                        && target.file_name().is_some_and(|n| relinked.iter().any(|r| n == r.as_str()))
                })
            })
            .collect();
        links.sort();
        for link in links {
            let shown = display(self.paths, &link);
            self.remove_file(&link, format!("remove legacy link {} (Cursor reads ~/.agents/skills)", shown));
        }
    }

    fn reconcile(&mut self, old: &[Entry], tools: &[String]) -> Vec<Entry> {
        if tools.iter().any(|t| t == "cursor") {
            self.remove_legacy_cursor_links();
        }
        let wanted = desired(self.paths, tools);
        let mut kept = Vec::new();
        for entry in old {
            if !wanted.iter().any(|w| same_slot(w, entry) && w.tool() == entry.tool()) && !self.remove(entry) {
                kept.push(entry.clone());
            }
        }
        for want in wanted {
            let previous = old.iter().find(|e| same_slot(e, &want));
            if let Some(entry) = self.ensure(want, previous) {
                kept.push(entry);
            }
        }
        if old.iter().any(|e| matches!(e, Entry::ClaudePlugin { .. })) {
            self.drop_marketplace_if_unused(&kept);
        }
        kept
    }
}

/// End (exclusive) of the JSON string starting at `start` (a `"`).
fn json_string_end(bytes: &[u8], start: usize) -> Option<usize> {
    let mut i = start + 1;
    while i < bytes.len() {
        match bytes[i] {
            b'\\' => i += 2,
            b'"' => return Some(i + 1),
            _ => i += 1,
        }
    }
    None
}

/// End (exclusive) of the JSON value starting at `start`.
fn json_value_end(bytes: &[u8], start: usize) -> Option<usize> {
    let mut depth = 0usize;
    let mut i = start;
    while i < bytes.len() {
        match bytes[i] {
            b'"' => {
                i = json_string_end(bytes, i)?;
                if depth == 0 {
                    return Some(i);
                }
                continue;
            }
            b'[' | b'{' => depth += 1,
            b']' | b'}' => {
                if depth == 0 {
                    return Some(i);
                }
                depth -= 1;
                if depth == 0 {
                    return Some(i + 1);
                }
            }
            b',' if depth == 0 => return Some(i),
            _ => {}
        }
        i += 1;
    }
    Some(i)
}

fn skip_ws(bytes: &[u8], mut i: usize) -> usize {
    while i < bytes.len() && bytes[i].is_ascii_whitespace() {
        i += 1;
    }
    i
}

/// Spans of the members of the object or array opening at `open`: for an
/// object `(key, value_start, value_end)`, for an array `("", start, end)`.
fn json_members(text: &str, open: usize) -> Option<Vec<(String, usize, usize)>> {
    let bytes = text.as_bytes();
    let is_object = bytes.get(open) == Some(&b'{');
    let mut out = Vec::new();
    let mut i = skip_ws(bytes, open + 1);
    loop {
        match bytes.get(i)? {
            b']' | b'}' => return Some(out),
            b',' => i = skip_ws(bytes, i + 1),
            _ => {
                let mut name = String::new();
                if is_object {
                    let end = json_string_end(bytes, i)?;
                    name = serde_json::from_str(&text[i..end]).ok()?;
                    i = skip_ws(bytes, end);
                    if bytes.get(i) != Some(&b':') {
                        return None;
                    }
                    i = skip_ws(bytes, i + 1);
                }
                let end = json_value_end(bytes, i)?;
                out.push((name, i, end));
                i = skip_ws(bytes, end);
            }
        }
    }
}

/// Adds the string `value` to the top-level array `key`, editing the text
/// in place so the rest of the file (order, layout) is kept. The result
/// is checked against the same edit done on parsed JSON.
fn json_array_add(text: &str, key: &str, value: &str) -> Option<String> {
    let mut expected: Value = serde_json::from_str(text).ok()?;
    let list = expected.as_object_mut()?.entry(key.to_string()).or_insert_with(|| json!([]));
    list.as_array_mut()?.push(json!(value));
    let item = serde_json::to_string(value).ok()?;
    let bytes = text.as_bytes();
    let root = skip_ws(bytes, 0);
    let members = json_members(text, root)?;
    let updated = match members.iter().find(|(name, _, _)| name == key) {
        Some((_, start, end)) => {
            let items = json_members(text, *start)?;
            match items.last() {
                Some((_, _, last_end)) => format!("{}, {}{}", &text[..*last_end], item, &text[*last_end..]),
                None => format!("{}[{}]{}", &text[..*start], item, &text[*end..]),
            }
        }
        None if members.is_empty() => {
            let close = text[root..].find('}')? + root;
            format!("{}{{\n  {}: [{}]\n{}", &text[..root], serde_json::to_string(key).ok()?, item, &text[close..])
        }
        None => format!(
            "{}\n  {}: [{}],{}",
            &text[..root + 1],
            serde_json::to_string(key).ok()?,
            item,
            &text[root + 1..]
        ),
    };
    let updated = if updated.ends_with('\n') { updated } else { updated + "\n" };
    (serde_json::from_str::<Value>(&updated).ok()? == expected).then_some(updated)
}

/// Removes the string `value` from the top-level array `key`, in place.
fn json_array_remove(text: &str, key: &str, value: &str) -> Option<String> {
    let mut expected: Value = serde_json::from_str(text).ok()?;
    let list = expected.get_mut(key)?.as_array_mut()?;
    let before = list.len();
    list.retain(|v| v.as_str() != Some(value));
    if list.len() == before {
        return None;
    }
    let bytes = text.as_bytes();
    let root = skip_ws(bytes, 0);
    let (_, open, _) = json_members(text, root)?.into_iter().find(|(name, _, _)| name == key)?;
    let items = json_members(text, open)?;
    let matches: Vec<usize> = (0..items.len())
        .filter(|&i| serde_json::from_str::<Value>(&text[items[i].1..items[i].2]).ok().and_then(|v| v.as_str().map(|s| s == value)) == Some(true))
        .collect();
    let [index] = matches[..] else { return None };
    let (cut_start, cut_end) = if index > 0 {
        (items[index - 1].2, items[index].2)
    } else if let Some(next) = items.get(1) {
        (items[0].1, next.1)
    } else {
        (items[0].1, items[0].2)
    };
    let mut updated = text.to_string();
    updated.replace_range(cut_start..cut_end, "");
    (serde_json::from_str::<Value>(&updated).ok()? == expected).then_some(updated)
}

fn parse_tools(args: &[String]) -> Result<Option<Vec<String>>, String> {
    let value = args
        .iter()
        .position(|a| a == "--tools")
        .map(|pos| args.get(pos + 1).cloned().unwrap_or_default())
        .or_else(|| args.iter().find_map(|a| a.strip_prefix("--tools=").map(String::from)));
    let Some(value) = value else { return Ok(None) };
    let mut tools = Vec::new();
    for id in value.split(',').map(|s| s.trim().to_ascii_lowercase()).filter(|s| !s.is_empty() && s != "none") {
        if !TOOLS.iter().any(|(t, _)| *t == id) {
            return Err(format!(
                "unknown tool '{}' (known: {})",
                id,
                TOOLS.iter().map(|(t, _)| *t).collect::<Vec<_>>().join(", ")
            ));
        }
        tools.push(id);
    }
    Ok(Some(tools))
}

fn ordered(tools: impl IntoIterator<Item = String>) -> Vec<String> {
    let set: Vec<String> = tools.into_iter().collect();
    TOOLS.iter().map(|(t, _)| t.to_string()).filter(|t| set.contains(t)).collect()
}

/// Creates `agents/` with an example-only AGENTS.md. Never touches an existing folder.
fn create_agents_dir(paths: &Paths, dry_run: bool) -> Result<bool, String> {
    if paths.agents.exists() || is_symlink(&paths.agents) {
        return Ok(false);
    }
    if dry_run {
        return Ok(true);
    }
    let io = |e: std::io::Error| format!("{}: {}", paths.agents.display(), e);
    std::fs::create_dir_all(paths.agents.join("rules")).map_err(io)?;
    std::fs::create_dir_all(paths.skills()).map_err(io)?;
    std::fs::create_dir_all(paths.agents.join("vendor")).map_err(io)?;
    std::fs::write(paths.rules(), RULES_TEMPLATE).map_err(io)?;
    Ok(true)
}

/// Entry point for `--setup-agents [--tools a,b] [--dry-run] [--list-tools]`.
pub fn setup_agents(args: &[String]) -> i32 {
    let Some(paths) = Paths::from_env() else {
        eprintln!("Cannot find home directory");
        return 1;
    };
    if args.iter().any(|a| a == "--list-tools") {
        for (id, name) in TOOLS {
            if detected(&paths, id) {
                println!("{}\t{}", id, name);
            }
        }
        return 0;
    }
    let dry_run = args.iter().any(|a| a == "--dry-run");
    let chosen = match parse_tools(args) {
        Ok(chosen) => chosen,
        Err(e) => {
            eprintln!("{}", e);
            return 2;
        }
    };
    let created = match create_agents_dir(&paths, dry_run) {
        Ok(created) => created,
        Err(e) => {
            eprintln!("Cannot create {}: {}", paths.agents.display(), e);
            return 1;
        }
    };
    let manifest = match load_manifest(&paths) {
        Ok(m) => m.unwrap_or_default(),
        Err(e) => {
            eprintln!("Unreadable manifest, not modifying anything: {}", e);
            return 1;
        }
    };
    let picked = chosen.unwrap_or_else(|| {
        TOOLS.iter().map(|(t, _)| t.to_string()).filter(|t| detected(&paths, t)).collect()
    });
    let tools = ordered(manifest.tools.iter().cloned().chain(picked));
    let mut ctx = Ctx { paths: &paths, dry_run, report: Report::default() };
    let entries = ctx.reconcile(&manifest.entries, &tools);
    let new_manifest = Manifest {
        version: MANIFEST_VERSION,
        agents_dir: paths.agents.clone(),
        inputs: inputs_fingerprint(&paths, &tools),
        tools: tools.clone(),
        entries,
    };
    if !dry_run {
        if let Err(e) = save_manifest(&paths, &new_manifest) {
            ctx.fail(e);
        }
        if let Err(e) = write_catalog(&paths, &tools) {
            ctx.fail(e);
        }
    }
    print_summary(&paths, &new_manifest, &ctx.report, created, dry_run);
    0
}

fn print_summary(paths: &Paths, manifest: &Manifest, report: &Report, created: bool, dry_run: bool) {
    let agents = display(paths, &paths.agents);
    if dry_run {
        println!("Dry run: nothing written.");
        if created {
            println!("would create {}/ (rules/AGENTS.md, skills/, vendor/)", agents);
        }
    } else if created {
        println!("Created {}/ for your own rules and skills.", agents);
    } else {
        println!("Rules and skills folder: {}/ (left as is)", agents);
    }
    if created {
        println!();
        println!("  Rules:  {}/rules/AGENTS.md", agents);
        println!("  Skills: {}/skills/<name>/SKILL.md (Agent Skills format)", agents);
        println!();
        println!("  They are yours: MemoryPilot ships none and never overwrites them.");
        println!("  They are wired into your tools automatically, and new skills or rule");
        println!("  changes are picked up at the start of each session (MemoryPilot --sync-agents).");
        println!("  Claude Desktop (chat) cannot be wired: its rules and skills live in your claude.ai account.");
    }
    println!();
    for line in &report.done {
        println!("  ✓ {}", line);
    }
    let wired: Vec<&str> = manifest.tools.iter().map(|t| tool_name(t)).collect();
    println!(
        "  Wired: {}",
        if wired.is_empty() { "none".to_string() } else { wired.join(", ") }
    );
    let not_wired: Vec<&str> = TOOLS
        .iter()
        .filter(|(t, _)| !manifest.tools.iter().any(|m| m == t) && detected(paths, t))
        .map(|(_, n)| *n)
        .collect();
    if !not_wired.is_empty() {
        println!("  Detected, not wired: {} (MemoryPilot --setup-agents --tools ...)", not_wired.join(", "));
    }
    for line in &report.skipped {
        println!("  - skipped: {}", line);
    }
    for line in &report.errors {
        println!("  ✗ {}", line);
    }
    if created {
        println!();
        println!("  Undo: MemoryPilot --uninstall-agents [--tools ...] (your files stay).");
    }
}

fn sync_log(message: &str) {
    use std::io::Write;
    let Some(root) = memorypilot_core::layout::root().filter(|r| r.is_dir()) else { return };
    let path = memorypilot_core::layout::path_in(&root, memorypilot_core::layout::Area::Logs, SYNC_LOG);
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(file, "{} {}", chrono::Local::now().format("%Y-%m-%d %H:%M:%S"), message);
    }
}

/// Entry point for `--sync-agents`. Always exits 0; writes nothing when
/// the inputs match the manifest.
pub fn sync_agents() {
    let Some(paths) = Paths::from_env() else { return };
    let manifest = match load_manifest(&paths) {
        Ok(Some(m)) => m,
        Ok(None) => return,
        Err(e) => {
            sync_log(&format!("sync: {}", e));
            return;
        }
    };
    // The catalog also follows content edits and deployments, which the
    // fingerprint does not cover: it is refreshed at every sync.
    if let Err(e) = write_catalog(&paths, &manifest.tools) {
        sync_log(&format!("sync error: catalog: {}", e));
    }
    let inputs = inputs_fingerprint(&paths, &manifest.tools);
    if inputs == manifest.inputs {
        return;
    }
    let mut ctx = Ctx { paths: &paths, dry_run: false, report: Report::default() };
    let entries = ctx.reconcile(&manifest.entries, &manifest.tools);
    // After an error (a `claude plugin` command that failed, a locked file),
    // an empty print makes the next session try again.
    let inputs = if ctx.report.errors.is_empty() { inputs } else { String::new() };
    let tools = manifest.tools.clone();
    let updated = Manifest { inputs, entries, ..manifest };
    if let Err(e) = save_manifest(&paths, &updated) {
        ctx.fail(e);
    }
    if let Err(e) = write_catalog(&paths, &tools) {
        ctx.fail(e);
    }
    for line in &ctx.report.done {
        sync_log(&format!("sync: {}", line));
    }
    for line in &ctx.report.skipped {
        sync_log(&format!("sync: skipped {}", line));
    }
    for line in &ctx.report.errors {
        sync_log(&format!("sync error: {}", line));
    }
}

/// Entry point for `--uninstall-agents [--tools a,b]`.
pub fn uninstall_agents(args: &[String]) -> i32 {
    let Some(paths) = Paths::from_env() else {
        eprintln!("Cannot find home directory");
        return 1;
    };
    let dry_run = args.iter().any(|a| a == "--dry-run");
    let manifest = match load_manifest(&paths) {
        Ok(Some(m)) => m,
        Ok(None) => {
            println!("Nothing to remove: no {}", display(&paths, &paths.manifest()));
            return 0;
        }
        Err(e) => {
            eprintln!("Unreadable manifest, not modifying anything: {}", e);
            return 1;
        }
    };
    let removed = match parse_tools(args) {
        Ok(Some(tools)) => tools,
        Ok(None) => manifest.tools.clone(),
        Err(e) => {
            eprintln!("{}", e);
            return 2;
        }
    };
    let remaining: Vec<String> = manifest.tools.iter().filter(|t| !removed.contains(t)).cloned().collect();
    let shared_needed = remaining.iter().any(|t| t != "claude");
    let goes = |e: &Entry| {
        if e.tool() == SHARED_SKILLS {
            !shared_needed
        } else {
            removed.iter().any(|t| t == e.tool())
        }
    };
    let mut ctx = Ctx { paths: &paths, dry_run, report: Report::default() };
    let mut entries = Vec::new();
    for entry in &manifest.entries {
        if !goes(entry) || !ctx.remove(entry) {
            entries.push(entry.clone());
        }
    }
    if manifest.entries.iter().any(|e| matches!(e, Entry::ClaudePlugin { .. })) {
        ctx.drop_marketplace_if_unused(&entries);
    }
    let in_sync = manifest.inputs == inputs_fingerprint(&paths, &manifest.tools);
    let inputs = if in_sync { inputs_fingerprint(&paths, &remaining) } else { manifest.inputs.clone() };
    let updated = Manifest { tools: remaining, inputs, entries, ..manifest };
    if !dry_run {
        if let Err(e) = save_manifest(&paths, &updated) {
            ctx.fail(e);
        }
    }
    for line in &ctx.report.done {
        println!("  ✓ {}", line);
    }
    for line in &ctx.report.skipped {
        println!("  - kept: {}", line);
    }
    for line in &ctx.report.errors {
        println!("  ✗ {}", line);
    }
    let names: Vec<&str> = removed.iter().map(|t| tool_name(t)).collect();
    println!(
        "Unwired: {}. {}/ and its content are kept.",
        if names.is_empty() { "nothing".to_string() } else { names.join(", ") },
        display(&paths, &paths.agents)
    );
    if ctx.report.errors.is_empty() { 0 } else { 1 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn append_then_remove_restores_the_file() {
        let text = import_text(Path::new("/h/.MemoryPilot/agents/rules/AGENTS.md"));
        for original in ["", "# Mine\n", "# Mine\n\n", "# Mine"] {
            let appended = append_text(original, &text);
            let start = appended.find(&text).unwrap();
            let restored = remove_span(&appended, start, start + text.len());
            let expected = if original.is_empty() { String::new() } else { "# Mine\n".to_string() };
            assert_eq!(restored, expected, "original {:?}", original);
        }
    }

    #[test]
    fn block_is_found_and_replaced_in_place() {
        let block = render_block("rule one\n");
        let content = format!("# Top\n\n{}\n# Bottom\n", block);
        let (start, end) = find_block(&content).unwrap();
        assert_eq!(&content[start..end], block);
        assert_eq!(remove_span(&content, start, end), "# Top\n\n# Bottom\n");
    }

    #[test]
    fn json_edits_keep_the_rest_of_the_file() {
        let original = "{\n  \"theme\": \"x\",\n  \"instructions\": [\"a.md\"],\n  \"mcp\": {\"k\": [1, 2]}\n}\n";
        let added = json_array_add(original, "instructions", "/r/AGENTS.md").unwrap();
        assert!(added.starts_with("{\n  \"theme\": \"x\",\n  \"instructions\": [\"a.md\", \"/r/AGENTS.md\"]"), "{}", added);
        assert_eq!(json_array_remove(&added, "instructions", "/r/AGENTS.md").unwrap(), original);
        let no_key = "{\"theme\": \"x\"}\n";
        let added = json_array_add(no_key, "instructions", "/r").unwrap();
        let parsed: Value = serde_json::from_str(&added).unwrap();
        assert_eq!(parsed, json!({"instructions": ["/r"], "theme": "x"}));
        let removed = json_array_remove(&added, "instructions", "/r").unwrap();
        assert_eq!(serde_json::from_str::<Value>(&removed).unwrap(), json!({"instructions": [], "theme": "x"}));
        let first = "{\"instructions\": [\"/r\", \"b\"]}";
        assert_eq!(json_array_remove(first, "instructions", "/r").unwrap(), "{\"instructions\": [\"b\"]}");
        let empty = json_array_add("{}", "instructions", "/r").unwrap();
        assert_eq!(serde_json::from_str::<Value>(&empty).unwrap(), json!({"instructions": ["/r"]}));
        assert!(json_array_add("{ // comment\n}", "instructions", "/r").is_none());
    }

    #[test]
    fn rejects_unknown_tools() {
        let args = vec!["--tools".to_string(), "claude,nope".to_string()];
        assert!(parse_tools(&args).is_err());
        let args = vec!["--tools=cursor,Codex".to_string()];
        assert_eq!(parse_tools(&args).unwrap(), Some(vec!["cursor".to_string(), "codex".to_string()]));
    }
}
