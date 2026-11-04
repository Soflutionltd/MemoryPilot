//! `--setup-agents`, `--sync-agents`, `--uninstall-agents` and
//! `--install-hooks` run as child processes in a throwaway HOME, with an
//! empty environment (no CLAUDE_CONFIG_DIR, CODEX_HOME, ..., no real PATH).

use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{Duration, Instant, SystemTime};

const BIN: &str = env!("CARGO_BIN_EXE_MemoryPilot");

struct Home(PathBuf);

impl Home {
    fn new(name: &str) -> Home {
        let nanos = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_nanos();
        let dir = std::env::temp_dir().join(format!("mp-agents-{}-{}-{}", name, std::process::id(), nanos));
        std::fs::create_dir_all(dir.join("bin")).unwrap();
        // canonicalize: /var vs /private/var on macOS.
        Home(dir.canonicalize().unwrap())
    }

    fn p(&self, rel: &str) -> PathBuf {
        self.0.join(rel)
    }

    fn run(&self, args: &[&str]) -> Output {
        let out = Command::new(BIN)
            .args(args)
            .env_clear()
            .env("HOME", &self.0)
            .env("PATH", self.p("bin"))
            .env("TMPDIR", std::env::temp_dir())
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{:?} failed: {}\n{}",
            args,
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        out
    }

    fn stdout(&self, args: &[&str]) -> String {
        String::from_utf8_lossy(&self.run(args).stdout).into_owned()
    }

    fn mkdirs(&self, rels: &[&str]) {
        for rel in rels {
            std::fs::create_dir_all(self.p(rel)).unwrap();
        }
    }

    fn write(&self, rel: &str, content: &str) {
        let path = self.p(rel);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, content).unwrap();
    }

    fn read(&self, rel: &str) -> String {
        std::fs::read_to_string(self.p(rel)).unwrap_or_else(|e| panic!("{}: {}", rel, e))
    }

    fn add_skill(&self, name: &str) {
        self.write(
            &format!(".MemoryPilot/agents/skills/{}/SKILL.md", name),
            &format!("---\nname: {}\ndescription: test skill\n---\nDo it.\n", name),
        );
    }

    fn manifest(&self) -> Value {
        serde_json::from_str(&self.read(".MemoryPilot/agents/.manifest.json")).unwrap()
    }

    fn rules(&self) -> PathBuf {
        self.p(".MemoryPilot/agents/rules/AGENTS.md")
    }

    fn all_tools(&self) {
        self.mkdirs(&[".claude", ".cursor", ".codex", ".gemini", ".copilot", ".config/opencode", "Documents/Cline", ".roo", ".codeium/windsurf"]);
    }
}

impl Drop for Home {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn is_link_to(link: &Path, target: &Path) -> bool {
    std::fs::read_link(link).map(|t| t == target).unwrap_or(false)
}

fn files_under(dir: &Path) -> Vec<String> {
    let mut out = Vec::new();
    fn walk(base: &Path, dir: &Path, out: &mut Vec<String>) {
        for entry in std::fs::read_dir(dir).unwrap().flatten() {
            let path = entry.path();
            let rel = path.strip_prefix(base).unwrap().display().to_string();
            if entry.file_type().unwrap().is_dir() {
                out.push(format!("{}/", rel));
                walk(base, &path, out);
            } else {
                out.push(rel);
            }
        }
    }
    walk(dir, dir, &mut out);
    out.sort();
    out
}

fn mtimes(paths: &[PathBuf]) -> Vec<SystemTime> {
    paths
        .iter()
        .map(|p| std::fs::symlink_metadata(p).unwrap().modified().unwrap())
        .collect()
}

#[test]
fn fresh_install_creates_an_empty_folder() {
    let home = Home::new("fresh");
    let out = home.stdout(&["--setup-agents"]);
    let agents = home.p(".MemoryPilot/agents");
    assert_eq!(
        files_under(&agents),
        vec![".manifest.json", "rules/", "rules/AGENTS.md", "skills/", "vendor/"]
    );
    let rules = home.read(".MemoryPilot/agents/rules/AGENTS.md");
    let trimmed = rules.trim();
    assert!(trimmed.starts_with("<!--") && trimmed.ends_with("-->") && trimmed.matches("-->").count() == 1, "{}", rules);
    assert!(out.contains("rules/AGENTS.md") && out.contains("SKILL.md") && out.contains("Wired: none"), "{}", out);
    assert!(out.contains("Claude Desktop"), "{}", out);
    // Nothing detected: nothing else written in HOME.
    assert_eq!(files_under(&home.0), vec![".MemoryPilot/", ".MemoryPilot/agents/", ".MemoryPilot/agents/.manifest.json", ".MemoryPilot/agents/rules/", ".MemoryPilot/agents/rules/AGENTS.md", ".MemoryPilot/agents/skills/", ".MemoryPilot/agents/vendor/", "bin/"]);
}

#[test]
fn list_tools_shows_only_detected_tools() {
    let home = Home::new("detect");
    home.mkdirs(&[".claude", ".codex"]);
    home.write("bin/gemini", "");
    let out = home.stdout(&["--setup-agents", "--list-tools"]);
    assert_eq!(out, "claude\tClaude Code\ncodex\tCodex\ngemini\tGemini CLI\n");
    assert!(!home.p(".MemoryPilot").exists(), "--list-tools must not write");
}

#[test]
fn reruns_create_no_duplicates() {
    let home = Home::new("rerun");
    home.all_tools();
    home.run(&["--setup-agents"]);
    home.add_skill("alpha");
    for _ in 0..2 {
        home.run(&["--setup-agents"]);
        home.run(&["--install-hooks"]);
    }
    let rules = home.rules().display().to_string();
    let import = format!("@{}", rules);
    assert_eq!(home.read(".claude/CLAUDE.md").matches(&import).count(), 1);
    assert_eq!(home.read(".gemini/GEMINI.md").matches(&import).count(), 1);
    assert_eq!(home.read(".codex/AGENTS.md").matches("MemoryPilot agents: begin").count(), 1);
    assert_eq!(home.read(".copilot/copilot-instructions.md").matches("MemoryPilot agents: begin").count(), 1);
    let opencode: Value = serde_json::from_str(&home.read(".config/opencode/opencode.json")).unwrap();
    assert_eq!(opencode["instructions"], serde_json::json!([rules]));
    assert!(is_link_to(&home.p(".claude/skills/alpha"), &home.p(".MemoryPilot/agents/skills/alpha")));
    assert!(is_link_to(&home.p(".agents/skills/alpha"), &home.p(".MemoryPilot/agents/skills/alpha")));
    assert!(is_link_to(&home.p("Documents/Cline/Rules/memorypilot-agents.md"), &home.rules()));
    assert!(is_link_to(&home.p(".roo/rules/memorypilot-agents.md"), &home.rules()));
    assert!(!home.p(".cursor/skills").exists(), "nothing in ~/.cursor/skills");
    let mdc = home.read(".cursor/rules/agents.mdc");
    assert!(mdc.starts_with("---\ndescription: ") && mdc.contains("\nglobs:\nalwaysApply: true\n---\n"), "{}", mdc);
    let manifest = home.manifest();
    let entries = manifest["entries"].as_array().unwrap();
    let mut paths: Vec<&str> = entries.iter().map(|e| e["path"].as_str().unwrap()).collect();
    let total = paths.len();
    paths.sort();
    paths.dedup();
    assert_eq!(paths.len(), total, "duplicate manifest entries");

    // Hooks: one MemoryPilot command per event, SessionEnd included.
    let ours = |cmd: &str| cmd.contains("MemoryPilot") && cmd.contains("--hook");
    for file in [".claude/settings.json", ".codex/hooks.json"] {
        let config: Value = serde_json::from_str(&home.read(file)).unwrap();
        for event in ["SessionStart", "Stop", "SessionEnd"] {
            let count = config["hooks"][event]
                .as_array()
                .unwrap()
                .iter()
                .flat_map(|g| g["hooks"].as_array().unwrap())
                .filter(|h| ours(h["command"].as_str().unwrap()))
                .count();
            assert_eq!(count, 1, "{} {}", file, event);
        }
        assert!(config["hooks"]["SessionEnd"].to_string().contains("--hook session-end"));
    }
    let cursor: Value = serde_json::from_str(&home.read(".cursor/hooks.json")).unwrap();
    for event in ["sessionStart", "stop", "sessionEnd"] {
        let count = cursor["hooks"][event]
            .as_array()
            .unwrap()
            .iter()
            .filter(|h| ours(h["command"].as_str().unwrap()))
            .count();
        assert_eq!(count, 1, "cursor {}", event);
    }
}

#[test]
fn existing_files_and_taken_names_are_kept() {
    let home = Home::new("existing");
    home.mkdirs(&[".claude", ".codex"]);
    home.write(".claude/CLAUDE.md", "# My own rules\n\nBe brief.\n");
    home.write(".codex/AGENTS.md", "# Codex rules of mine\n");
    // A user's folder already in place, with their own rules and skills.
    home.write(".MemoryPilot/agents/rules/AGENTS.md", "Always write tests.\n");
    home.add_skill("deploy");
    home.add_skill("review");
    home.add_skill("adopted");
    // Taken names: a real skill folder for Claude, a foreign link for the others.
    home.write(".claude/skills/deploy/SKILL.md", "user's own deploy skill\n");
    std::fs::create_dir_all(home.p(".agents/skills")).unwrap();
    std::os::unix::fs::symlink("/elsewhere/review", home.p(".agents/skills/review")).unwrap();
    // Already linked to agents/ (as on a machine set up by hand): adopted.
    std::os::unix::fs::symlink(home.p(".MemoryPilot/agents/skills/adopted"), home.p(".claude/skills/adopted")).unwrap();

    let out = home.stdout(&["--setup-agents", "--tools", "claude,codex"]);
    assert!(out.contains("left as is"), "{}", out);
    assert!(out.contains("skipped: ~/.claude/skills/deploy (name already taken)"), "{}", out);
    assert!(out.contains("skipped: ~/.agents/skills/review (name already taken)"), "{}", out);
    assert_eq!(home.read(".MemoryPilot/agents/rules/AGENTS.md"), "Always write tests.\n");
    assert!(!home.p(".MemoryPilot/agents/vendor").exists(), "existing folder untouched");
    assert_eq!(home.read(".claude/skills/deploy/SKILL.md"), "user's own deploy skill\n");
    assert_eq!(std::fs::read_link(home.p(".agents/skills/review")).unwrap(), PathBuf::from("/elsewhere/review"));
    assert!(is_link_to(&home.p(".claude/skills/review"), &home.p(".MemoryPilot/agents/skills/review")));
    assert!(is_link_to(&home.p(".agents/skills/deploy"), &home.p(".MemoryPilot/agents/skills/deploy")));
    let claude = home.read(".claude/CLAUDE.md");
    assert!(claude.starts_with("# My own rules\n\nBe brief.\n\n<!-- MemoryPilot agents"), "{}", claude);
    let codex = home.read(".codex/AGENTS.md");
    assert!(codex.starts_with("# Codex rules of mine\n\n<!-- MemoryPilot agents: begin -->") && codex.contains("Always write tests."), "{}", codex);
    let manifest = home.manifest();
    let adopted = manifest["entries"]
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["path"].as_str().unwrap().ends_with(".claude/skills/adopted"))
        .expect("adopted link recorded");
    assert_eq!(adopted["adopted"], true);
    assert!(!manifest.to_string().contains(".claude/skills/deploy"));

    // Full uninstall: only what we added goes, the adopted link included.
    home.run(&["--uninstall-agents"]);
    assert_eq!(home.read(".claude/CLAUDE.md"), "# My own rules\n\nBe brief.\n");
    assert_eq!(home.read(".codex/AGENTS.md"), "# Codex rules of mine\n");
    assert_eq!(home.read(".claude/skills/deploy/SKILL.md"), "user's own deploy skill\n");
    assert_eq!(std::fs::read_link(home.p(".agents/skills/review")).unwrap(), PathBuf::from("/elsewhere/review"));
    assert!(std::fs::symlink_metadata(home.p(".claude/skills/review")).is_err());
    assert!(std::fs::symlink_metadata(home.p(".claude/skills/adopted")).is_err());
    assert_eq!(home.read(".MemoryPilot/agents/rules/AGENTS.md"), "Always write tests.\n");
    assert!(home.p(".MemoryPilot/agents/skills/deploy/SKILL.md").is_file());
}

#[test]
fn linked_claude_md_is_not_written() {
    let home = Home::new("symlinked");
    home.write("dotfiles/CLAUDE.md", "# kept in dotfiles\n");
    home.mkdirs(&[".claude"]);
    std::os::unix::fs::symlink(home.p("dotfiles/CLAUDE.md"), home.p(".claude/CLAUDE.md")).unwrap();
    let out = home.stdout(&["--setup-agents", "--tools", "claude"]);
    assert!(out.contains("is a symbolic link"), "{}", out);
    assert_eq!(home.read("dotfiles/CLAUDE.md"), "# kept in dotfiles\n");
}

#[test]
fn sync_wires_new_skills_then_writes_nothing() {
    let home = Home::new("sync");
    home.mkdirs(&[".claude", ".codex", ".cursor"]);
    home.run(&["--setup-agents"]);
    home.add_skill("fresh");
    home.run(&["--sync-agents"]);
    let claude_link = home.p(".claude/skills/fresh");
    let shared_link = home.p(".agents/skills/fresh");
    assert!(is_link_to(&claude_link, &home.p(".MemoryPilot/agents/skills/fresh")));
    assert!(is_link_to(&shared_link, &home.p(".MemoryPilot/agents/skills/fresh")));
    assert!(!home.p(".cursor/skills").exists());

    let watched = vec![
        home.p(".MemoryPilot/agents/.manifest.json"),
        home.p(".claude/CLAUDE.md"),
        home.p(".codex/AGENTS.md"),
        home.p(".cursor/rules/agents.mdc"),
        claude_link.clone(),
        shared_link.clone(),
    ];
    let before = mtimes(&watched);
    let files_before = files_under(&home.0);
    std::thread::sleep(Duration::from_millis(30));
    let runs = 20;
    let started = Instant::now();
    for _ in 0..runs {
        home.run(&["--sync-agents"]);
    }
    let per_run = started.elapsed() / runs;
    println!("--sync-agents with nothing to do: {:?} per run (debug build, process spawn included)", per_run);
    assert_eq!(mtimes(&watched), before, "a no-op sync wrote something");
    assert_eq!(files_under(&home.0), files_before);
    assert!(per_run < Duration::from_millis(200), "{:?}", per_run);

    // A removed skill is unwired at the next sync.
    std::fs::remove_dir_all(home.p(".MemoryPilot/agents/skills/fresh")).unwrap();
    home.run(&["--sync-agents"]);
    assert!(std::fs::symlink_metadata(&claude_link).is_err());
    assert!(std::fs::symlink_metadata(&shared_link).is_err());
}

#[test]
fn sync_never_fails() {
    let home = Home::new("sync-errors");
    // No manifest: nothing written.
    home.run(&["--sync-agents"]);
    assert_eq!(files_under(&home.0), vec!["bin/"]);
    home.write(".MemoryPilot/agents/.manifest.json", "{ not json");
    let out = home.run(&["--sync-agents"]);
    assert!(out.stdout.is_empty() && out.stderr.is_empty());
    assert!(home.read(".MemoryPilot/agents-sync.log").contains("sync:"));
}

#[test]
fn rules_change_regenerates_copies() {
    let home = Home::new("regen");
    home.mkdirs(&[".cursor", ".codex", ".copilot"]);
    home.write(".copilot/copilot-instructions.md", "Mine first.\n");
    home.run(&["--setup-agents"]);
    std::fs::write(home.rules(), "Use tabs.\n").unwrap();
    home.run(&["--sync-agents"]);
    let mdc = home.read(".cursor/rules/agents.mdc");
    assert!(mdc.ends_with("---\nUse tabs.\n"), "{}", mdc);
    let codex = home.read(".codex/AGENTS.md");
    assert!(codex.contains("Use tabs.") && !codex.contains("Your personal rules"), "{}", codex);
    let copilot = home.read(".copilot/copilot-instructions.md");
    assert!(copilot.starts_with("Mine first.\n\n<!-- MemoryPilot agents: begin -->") && copilot.contains("Use tabs."), "{}", copilot);
    assert_eq!(copilot.matches("MemoryPilot agents: begin").count(), 1);

    // A hand-edited copy is never overwritten.
    std::fs::write(home.p(".cursor/rules/agents.mdc"), "my edit\n").unwrap();
    std::fs::write(home.rules(), "Use spaces.\n").unwrap();
    home.run(&["--sync-agents"]);
    assert_eq!(home.read(".cursor/rules/agents.mdc"), "my edit\n");
    assert!(home.read(".codex/AGENTS.md").contains("Use spaces."));
}

#[test]
fn uninstall_one_tool_then_all() {
    let home = Home::new("uninstall");
    home.all_tools();
    home.write(".codex/AGENTS.md", "# Codex own\n");
    home.write(".gemini/GEMINI.md", "Gemini own.\n");
    home.write(".config/opencode/opencode.json", "{\n  \"theme\": \"dark\",\n  \"instructions\": [\"CONTRIBUTING.md\"]\n}\n");
    home.write(".agents/skills/mine/SKILL.md", "not ours\n");
    home.run(&["--setup-agents"]);
    home.add_skill("one");
    home.run(&["--sync-agents"]);

    let out = home.stdout(&["--uninstall-agents", "--tools", "codex"]);
    assert!(out.contains("Unwired: Codex"), "{}", out);
    assert_eq!(home.read(".codex/AGENTS.md"), "# Codex own\n");
    assert!(home.read(".gemini/GEMINI.md").contains("MemoryPilot agents"));
    assert!(home.read(".claude/CLAUDE.md").contains("MemoryPilot agents"));
    assert!(is_link_to(&home.p(".agents/skills/one"), &home.p(".MemoryPilot/agents/skills/one")));
    assert_eq!(home.manifest()["tools"].as_array().unwrap().len(), 8);
    // Codex stays unwired at the next sync.
    std::fs::write(home.rules(), "Changed.\n").unwrap();
    home.run(&["--sync-agents"]);
    assert_eq!(home.read(".codex/AGENTS.md"), "# Codex own\n");
    assert!(home.read(".copilot/copilot-instructions.md").contains("Changed."));

    home.run(&["--uninstall-agents"]);
    assert_eq!(home.read(".gemini/GEMINI.md"), "Gemini own.\n");
    assert_eq!(
        home.read(".config/opencode/opencode.json"),
        "{\n  \"theme\": \"dark\",\n  \"instructions\": [\"CONTRIBUTING.md\"]\n}\n"
    );
    assert_eq!(home.read(".agents/skills/mine/SKILL.md"), "not ours\n");
    for gone in [
        ".claude/CLAUDE.md",
        ".copilot/copilot-instructions.md",
        ".cursor/rules/agents.mdc",
        ".claude/skills/one",
        ".agents/skills/one",
        "Documents/Cline/Rules/memorypilot-agents.md",
        ".roo/rules/memorypilot-agents.md",
    ] {
        assert!(std::fs::symlink_metadata(home.p(gone)).is_err(), "{} still there", gone);
    }
    assert_eq!(home.read(".MemoryPilot/agents/rules/AGENTS.md"), "Changed.\n");
    assert!(home.p(".MemoryPilot/agents/skills/one/SKILL.md").is_file());
    assert_eq!(home.manifest()["tools"], serde_json::json!([]));
    // Nothing wired any more: sync stays quiet.
    home.add_skill("two");
    home.run(&["--sync-agents"]);
    assert!(std::fs::symlink_metadata(home.p(".claude/skills/two")).is_err());
}

#[test]
fn dry_run_writes_nothing() {
    let home = Home::new("dry");
    home.mkdirs(&[".claude"]);
    let out = home.stdout(&["--setup-agents", "--dry-run"]);
    assert!(out.contains("Dry run"), "{}", out);
    assert_eq!(files_under(&home.0), vec![".claude/", "bin/"]);
}

/// Fake `claude` in `~/bin`: logs each call and keeps the two files
/// MemoryPilot reads to know what is installed.
fn fake_claude(home: &Home) {
    home.write(
        "bin/claude",
        r#"#!/bin/sh
PATH=/bin:/usr/bin
P="$HOME/.claude/plugins"
mkdir -p "$P"
echo "$*" >> "$HOME/claude-calls.log"
case "$1 $2 $3" in
  "plugin marketplace add") echo '{"memorypilot":{}}' > "$P/known_marketplaces.json" ;;
  "plugin marketplace remove") echo '{}' > "$P/known_marketplaces.json" ;;
esac
case "$1 $2" in
  "plugin install")
    old=$(cat "$P/installed.list" 2>/dev/null)
    printf '%s\n%s\n' "$old" "$3" | grep . > "$P/installed.list" ;;
  "plugin uninstall")
    grep -vx "$3" "$P/installed.list" > "$P/installed.tmp"; mv "$P/installed.tmp" "$P/installed.list" ;;
esac
{ printf '{"plugins":{'; sep=""; while read -r id; do printf '%s"%s":[{}]' "$sep" "$id"; sep=","; done < "$P/installed.list"; printf '}}'; } > "$P/installed_plugins.json" 2>/dev/null
exit 0
"#,
    );
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(home.p("bin/claude"), std::fs::Permissions::from_mode(0o755)).unwrap();
}

fn claude_calls(home: &Home) -> Vec<String> {
    std::fs::read_to_string(home.p("claude-calls.log")).unwrap_or_default().lines().map(str::to_string).collect()
}

#[test]
fn plugins_install_in_claude_code_link_cursor_variant_and_update() {
    let home = Home::new("plugins");
    home.mkdirs(&[".claude", ".cursor"]);
    fake_claude(&home);
    home.run(&["--setup-agents", "--tools", "claude,cursor"]);
    let plugins = ".MemoryPilot/agents/plugins";
    home.write(&format!("{}/kit/.claude-plugin/plugin.json", plugins), r#"{"name":"kit","version":"1.0.0"}"#);
    home.write(&format!("{}/kit/skills/audit/SKILL.md", plugins), "---\nname: audit\n---\n");
    home.write(&format!("{}/kit/cursor/.cursor-plugin/plugin.json", plugins), r#"{"name":"kit"}"#);
    home.write(&format!("{}/solo/.claude-plugin/plugin.json", plugins), r#"{"name":"solo"}"#);
    home.write(&format!("{}/solo/skills/polish/SKILL.md", plugins), "---\nname: polish\n---\n");

    home.run(&["--sync-agents"]);
    let calls = claude_calls(&home);
    assert_eq!(
        calls,
        vec![
            format!("plugin marketplace add {}", home.p(plugins).display()),
            "plugin install kit@memorypilot --scope user".to_string(),
            "plugin install solo@memorypilot --scope user".to_string(),
        ]
    );
    let market: Value = serde_json::from_str(&home.read(&format!("{}/.claude-plugin/marketplace.json", plugins))).unwrap();
    assert_eq!(market["name"], "memorypilot");
    assert_eq!(market["plugins"].as_array().unwrap().len(), 2);
    assert!(is_link_to(&home.p(".cursor/plugins/local/kit"), &home.p(&format!("{}/kit/cursor", plugins))));
    assert!(is_link_to(&home.p(".agents/skills/polish"), &home.p(&format!("{}/solo/skills/polish", plugins))));
    assert!(std::fs::symlink_metadata(home.p(".agents/skills/audit")).is_err(), "kit has a Cursor variant: no shared skill link");

    home.run(&["--sync-agents"]);
    assert_eq!(claude_calls(&home).len(), 3, "nothing changed: no claude call");

    home.write(&format!("{}/kit/skills/audit/SKILL.md", plugins), "---\nname: audit\n---\nv2\n");
    home.run(&["--sync-agents"]);
    assert_eq!(
        claude_calls(&home)[3..],
        ["plugin marketplace update memorypilot".to_string(), "plugin update kit@memorypilot --scope user".to_string()]
    );

    home.write(&format!("{}/kit/cursor/README.md", plugins), "cursor only");
    home.run(&["--sync-agents"]);
    assert_eq!(claude_calls(&home).len(), 5, "a change in cursor/ needs no Claude Code update");

    home.run(&["--uninstall-agents", "--tools", "claude"]);
    assert_eq!(
        claude_calls(&home)[5..],
        [
            "plugin uninstall kit@memorypilot --scope user".to_string(),
            "plugin uninstall solo@memorypilot --scope user".to_string(),
            "plugin marketplace remove memorypilot".to_string(),
        ]
    );
    assert!(!home.p(&format!("{}/.claude-plugin/marketplace.json", plugins)).exists());
    assert!(is_link_to(&home.p(".cursor/plugins/local/kit"), &home.p(&format!("{}/kit/cursor", plugins))), "Cursor still wired");
}

#[test]
fn failed_claude_call_is_retried_at_next_sync() {
    let home = Home::new("plugins-retry");
    home.mkdirs(&[".claude"]);
    home.run(&["--setup-agents", "--tools", "claude"]);
    home.write(".MemoryPilot/agents/plugins/kit/.claude-plugin/plugin.json", r#"{"name":"kit"}"#);
    home.run(&["--sync-agents"]);
    assert!(home.read(".MemoryPilot/agents-sync.log").contains("claude command not found"));
    assert_eq!(home.manifest()["inputs"], "", "empty print: next session retries");
    fake_claude(&home);
    home.run(&["--sync-agents"]);
    assert_eq!(claude_calls(&home).last().unwrap(), "plugin install kit@memorypilot --scope user");
}

#[test]
fn cursor_sees_each_skill_once() {
    let home = Home::new("cursor-once");
    home.mkdirs(&[".cursor/skills"]);
    home.add_skill("mine");
    home.add_skill("theirs");
    // Hand-made link to the agents folder (old wiring) and a third-party
    // skill of the same name as one of ours.
    std::os::unix::fs::symlink(home.p(".MemoryPilot/agents/skills/mine"), home.p(".cursor/skills/mine")).unwrap();
    home.write(".cursor/skills/theirs/SKILL.md", "---\nname: theirs\n---\n");
    home.run(&["--setup-agents", "--tools", "cursor"]);
    assert!(std::fs::symlink_metadata(home.p(".cursor/skills/mine")).is_err(), "legacy link removed");
    assert!(is_link_to(&home.p(".agents/skills/mine"), &home.p(".MemoryPilot/agents/skills/mine")));
    assert!(std::fs::symlink_metadata(home.p(".agents/skills/theirs")).is_err(), "no second copy for Cursor");
    assert!(home.p(".cursor/skills/theirs/SKILL.md").is_file(), "third-party folder untouched");
    home.run(&["--sync-agents"]);
    assert!(std::fs::symlink_metadata(home.p(".agents/skills/theirs")).is_err());
}
