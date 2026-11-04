//! `agents/catalog.json`: everything the agents folder provides and where it
//! is deployed, rewritten by `--setup-agents` and `--sync-agents` (only when
//! it changes). One entry per rule, skill, plugin and MCP server:
//!
//! - `type`: `rule`, `skill`, `plugin` or `mcp`;
//! - `origin`: `perso` (written here) or `vendor` (imported, kept as is);
//! - `version`, `source`, `url`: for vendor items, where they come from;
//! - `targets`: `claude_code`, `cursor`, `claude_app` (true when deployed);
//! - `updated`: date of the most recent file of the item;
//! - `status`: `ok` (deployed to every wired tool), `partial` or `not_deployed`.
//!
//! MCP servers are read from McpHub's config by name only: their arguments
//! and environment can hold secrets and are never copied.

use serde_json::{json, Value};
use std::path::{Path, PathBuf};

pub struct Ctx<'a> {
    pub home: &'a Path,
    pub agents: &'a Path,
    pub claude: &'a Path,
    /// Tools wired by `--setup-agents` (`claude`, `cursor`, ...).
    pub tools: &'a [String],
}

fn read_json(path: &Path) -> Option<Value> {
    std::fs::read_to_string(path).ok().and_then(|t| serde_json::from_str(&t).ok())
}

fn links_to(link: &Path, target: &Path) -> bool {
    match (link.canonicalize(), target.canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    }
}

/// Date (YYYY-MM-DD) of the most recent file under `path`.
fn updated(path: &Path) -> Option<String> {
    let mut newest = None;
    let mut stack = vec![path.to_path_buf()];
    while let Some(p) = stack.pop() {
        let Ok(meta) = std::fs::metadata(&p) else { continue };
        if meta.is_dir() {
            for entry in std::fs::read_dir(&p).into_iter().flatten().flatten() {
                if !entry.file_name().to_string_lossy().starts_with('.') {
                    stack.push(entry.path());
                }
            }
        } else if let Ok(modified) = meta.modified() {
            newest = Some(newest.map_or(modified, |n: std::time::SystemTime| n.max(modified)));
        }
    }
    newest.map(|t| chrono::DateTime::<chrono::Local>::from(t).format("%Y-%m-%d").to_string())
}

fn sorted_dirs(dir: &Path) -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| !e.file_name().to_string_lossy().starts_with('.'))
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    dirs.sort();
    dirs
}

/// `name@1.2.3` or `@scope/name@1.2.3`, as npx and uvx take them.
fn is_versioned_package(arg: &str) -> bool {
    if arg.starts_with('-') {
        return false;
    }
    let body = match arg.strip_prefix('@') {
        Some(scoped) => match scoped.split_once('/') {
            Some((_, rest)) => rest,
            None => return false,
        },
        None => arg,
    };
    match body.split_once('@') {
        Some((name, version)) => !name.is_empty() && !name.contains('/') && version.chars().next().is_some_and(|c| c.is_ascii_digit()),
        None => false,
    }
}

impl Ctx<'_> {
    fn wired(&self, tool: &str) -> bool {
        self.tools.iter().any(|t| t == tool)
    }

    /// Claude Desktop reads the skills over MCP (get_agent_skill) when it has MemoryPilot.
    fn claude_app_has_memorypilot(&self) -> bool {
        self.claude_app_has_server("MemoryPilot")
    }

    fn claude_app_has_server(&self, name: &str) -> bool {
        read_json(&self.home.join("Library/Application Support/Claude/claude_desktop_config.json"))
            .is_some_and(|v| v["mcpServers"].get(name).is_some())
    }

    fn status(&self, targets: &Value) -> &'static str {
        let expected: Vec<&str> = [("claude", "claude_code"), ("cursor", "cursor")]
            .iter()
            .filter(|(tool, _)| self.wired(tool))
            .map(|(_, key)| *key)
            .collect();
        let deployed = expected.iter().filter(|k| targets[**k] == true).count();
        match (deployed, expected.len()) {
            (_, 0) => "not_deployed",
            (d, n) if d == n => "ok",
            (0, _) => "not_deployed",
            _ => "partial",
        }
    }

    fn entry(&self, kind: &str, name: &str, origin: &str, path: &Path, targets: Value, extra: Value) -> Value {
        let mut item = json!({
            "type": kind,
            "name": name,
            "origin": origin,
            "version": Value::Null,
            "source": Value::Null,
            "url": Value::Null,
            "path": path.strip_prefix(self.agents).map(|p| p.display().to_string()).unwrap_or_else(|_| path.display().to_string()),
            "targets": targets,
            "updated": updated(path),
        });
        if let (Some(item), Some(extra)) = (item.as_object_mut(), extra.as_object()) {
            for (k, v) in extra {
                item.insert(k.clone(), v.clone());
            }
        }
        let status = self.status(&item["targets"]);
        item["status"] = json!(status);
        item
    }

    fn rule(&self) -> Option<Value> {
        let rules = self.agents.join("rules/AGENTS.md");
        if !rules.is_file() {
            return None;
        }
        let claude_md = std::fs::read_to_string(self.claude.join("CLAUDE.md")).unwrap_or_default();
        let targets = json!({
            "claude_code": claude_md.contains(&rules.display().to_string()),
            "cursor": self.home.join(".cursor/rules/agents.mdc").is_file(),
            "claude_app": false,
        });
        Some(self.entry("rule", "AGENTS.md", "perso", &rules, targets, json!({})))
    }

    fn skills(&self, folder: &str, origin: &str, app: bool) -> Vec<Value> {
        sorted_dirs(&self.agents.join(folder))
            .into_iter()
            .filter(|dir| dir.join("SKILL.md").is_file())
            .map(|dir| {
                let name = dir.file_name().unwrap_or_default().to_string_lossy().to_string();
                let claude_code = links_to(&self.claude.join("skills").join(&name), &dir);
                // Cursor reads ~/.claude/skills as well as ~/.agents/skills.
                let cursor = self.wired("cursor") && (claude_code || links_to(&self.home.join(".agents/skills").join(&name), &dir));
                let targets = json!({
                    "claude_code": claude_code,
                    "cursor": cursor,
                    "claude_app": app,
                });
                let extra = if origin == "vendor" {
                    json!({ "source": "import manuel (voir vendor/README.md)" })
                } else {
                    json!({})
                };
                self.entry("skill", &name, origin, &dir, targets, extra)
            })
            .collect()
    }

    fn plugins(&self, app: bool) -> Vec<Value> {
        let installed = read_json(&self.claude.join("plugins/installed_plugins.json")).unwrap_or(Value::Null);
        crate::plugins::list(&self.agents.join("plugins"))
            .into_iter()
            .map(|p| {
                let source = crate::plugins::read_source(&p.dir);
                let cursor = p.cursor_dir.as_ref().is_some_and(|c| links_to(&self.home.join(".cursor/plugins/local").join(&p.name), c));
                let targets = json!({
                    "claude_code": installed["plugins"].get(format!("{}@{}", p.name, crate::plugins::MARKETPLACE)).is_some(),
                    "cursor": cursor,
                    "claude_app": app,
                });
                let extra = json!({
                    "version": (!p.version.is_empty()).then_some(p.version.clone()),
                    "source": source.as_ref().map(|s| format!("git {}", &s.commit[..s.commit.len().min(7)])),
                    "url": source.as_ref().map(|s| s.repo.clone()),
                });
                self.entry("plugin", &p.name, if source.is_some() { "vendor" } else { "perso" }, &p.dir, targets, extra)
            })
            .collect()
    }

    /// MCP servers of McpHub, by name only (arguments and environment are never read out).
    fn mcp_servers(&self) -> Vec<Value> {
        let Some(config) = read_json(&self.home.join(".mcphub/config.json")) else { return Vec::new() };
        let Some(servers) = config.get("servers").and_then(|s| s.as_object()) else { return Vec::new() };
        let own = self.home.join("Cursor").display().to_string();
        let app = self.claude_app_has_server("McpHub");
        let mut out: Vec<Value> = servers
            .iter()
            .map(|(name, server)| {
                let command = server["command"].as_str().unwrap_or("");
                let args: Vec<&str> = server["args"].as_array().map(|a| a.iter().filter_map(|v| v.as_str()).collect()).unwrap_or_default();
                let url = server["url"].as_str().map(|u| u.split('?').next().unwrap_or(u).to_string());
                // A published package written `name@version` among the arguments.
                let package = args.iter().find(|a| is_versioned_package(a));
                let perso = command.starts_with(&own) || args.iter().any(|a| a.starts_with(&own)) || command.contains("/.local/bin/");
                let (version, source) = match package {
                    Some(p) => {
                        let at = p.rfind('@').unwrap_or(p.len());
                        (Some(p[at + 1..].to_string()), Some(p[..at].to_string()))
                    }
                    None => (None, Some(Path::new(command).file_name().map(|f| f.to_string_lossy().to_string()).unwrap_or_default())),
                };
                json!({
                    "type": "mcp",
                    "name": name,
                    "origin": if perso { "perso" } else { "vendor" },
                    "version": version,
                    "source": if url.is_some() { Value::Null } else { json!(source) },
                    "url": url,
                    "path": Value::Null,
                    // Served by McpHub to the clients that have it; nothing when disabled.
                    "targets": {
                        "claude_code": server["disabled"] != true,
                        "cursor": server["disabled"] != true,
                        "claude_app": server["disabled"] != true && app,
                    },
                    "via": "McpHub",
                    "updated": Value::Null,
                    "status": if server["disabled"] == true { "disabled" } else { "ok" },
                })
            })
            .collect();
        out.sort_by(|a, b| a["name"].as_str().cmp(&b["name"].as_str()));
        out
    }

    pub fn render(&self) -> String {
        let app = self.claude_app_has_memorypilot();
        let mut items: Vec<Value> = self.rule().into_iter().collect();
        items.extend(self.skills("skills", "perso", app));
        items.extend(self.skills("vendor", "vendor", app));
        items.extend(self.plugins(app));
        items.extend(self.mcp_servers());
        let doc = json!({
            "generated_by": "MemoryPilot --sync-agents",
            "tools": self.tools,
            "items": items,
        });
        serde_json::to_string_pretty(&doc).unwrap_or_default() + "\n"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn versioned_packages_only() {
        assert!(is_versioned_package("@modelcontextprotocol/server-github@2025.4.8"));
        assert!(is_versioned_package("mcp-remote@0.1.18"));
        assert!(!is_versioned_package("-y"));
        assert!(!is_versioned_package("@scope/name"));
        assert!(!is_versioned_package("https://mcp.example.com/sse"));
        assert!(!is_versioned_package("user@host"));
    }
}
