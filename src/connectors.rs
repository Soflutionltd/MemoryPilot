//! Connectors: facts that live outside the repository but belong to it.
//! - GitHub (through the `gh` CLI and its own login): open pull requests,
//!   recently merged ones, open issues, and the state of any referenced one.
//! - cf-orchestrator deploy history: last successful deploy of each target
//!   and how many commits landed after it.
//! - Supabase production: which migration files of the repository are not
//!   applied, and which applied migrations are missing from the repository
//!   (Management API, token from SUPABASE_ACCESS_TOKEN or the Keychain).
//!
//! Network reads are slow, so recall, search and the project brain only
//! read what is cached and trigger a background refresh when it is older
//! than REFRESH_AFTER; the explicit `project_facts` tool waits for it.

use std::collections::{HashMap, HashSet};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use chrono::{DateTime, FixedOffset};
use serde_json::{json, Map, Value};

use crate::facts;
use crate::verify::{self, RepoSnapshot};

const REFRESH_AFTER: Duration = Duration::from_secs(300);
const COMMAND_TIMEOUT: Duration = Duration::from_secs(10);
const GH_LIMIT: &str = "40";
const MAX_LISTED: usize = 8;
const SUPABASE_API: &str = "https://api.supabase.com/v1/projects";
const KEYCHAIN_SERVICE: &str = "MemoryPilot";
const SUPABASE_KEYCHAIN_ACCOUNT: &str = "supabase-access-token";
const ENV_FILES: &[&str] = &[".env", ".env.local", ".env.production", ".env.development"];

struct Entry {
    fetched: Instant,
    data: Arc<Value>,
}

#[derive(Default)]
struct Cache {
    entries: HashMap<PathBuf, Entry>,
    refreshing: HashSet<PathBuf>,
}

fn cache() -> &'static Mutex<Cache> {
    static CACHE: OnceLock<Mutex<Cache>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(Cache::default()))
}

pub fn disabled() -> bool {
    std::env::var("MEMORYPILOT_CONNECTORS")
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "0" | "off" | "false"
            )
        })
        .unwrap_or(false)
}

/// Last known connector data, without waiting. Starts a background refresh
/// when there is none or it is stale.
pub fn cached(repo: &Path) -> Option<Arc<Value>> {
    if disabled() {
        return None;
    }
    let mut guard = cache().lock().ok()?;
    let current = guard
        .entries
        .get(repo)
        .map(|entry| (entry.fetched.elapsed(), Arc::clone(&entry.data)));
    let stale = current.as_ref().is_none_or(|(age, _)| *age > REFRESH_AFTER);
    if stale && guard.refreshing.insert(repo.to_path_buf()) {
        let repo = repo.to_path_buf();
        std::thread::spawn(move || refresh(&repo));
    }
    current.map(|(_, data)| data)
}

/// Fresh connector data, refreshing now when the cache is stale.
pub fn fresh(repo: &Path) -> Option<Arc<Value>> {
    if disabled() {
        return None;
    }
    if let Ok(guard) = cache().lock() {
        if let Some(entry) = guard.entries.get(repo) {
            if entry.fetched.elapsed() <= REFRESH_AFTER {
                return Some(Arc::clone(&entry.data));
            }
        }
    }
    Some(refresh(repo))
}

fn refresh(repo: &Path) -> Arc<Value> {
    let data = Arc::new(collect(repo));
    if let Ok(mut guard) = cache().lock() {
        guard.entries.insert(
            repo.to_path_buf(),
            Entry {
                fetched: Instant::now(),
                data: Arc::clone(&data),
            },
        );
        guard.refreshing.remove(repo);
    }
    data
}

fn collect(repo: &Path) -> Value {
    let mut data = Map::new();
    data.insert("fetched_at".into(), json!(chrono::Utc::now().to_rfc3339()));
    if let Some(github) = github(repo) {
        data.insert("github".into(), github);
    }
    let snapshot = verify::snapshot(repo);
    let facts = facts::project_facts(repo);
    if let (Some(snapshot), Some(facts)) = (snapshot.as_ref(), facts.as_ref()) {
        let deploys = deploys(snapshot, &facts.full);
        if !deploys.is_empty() {
            data.insert("deploys".into(), json!(deploys));
        }
        if facts.full.get("supabase_migrations").is_some() {
            data.insert("supabase".into(), supabase(repo, snapshot));
        }
    }
    Value::Object(data)
}

/// Runs a command with a deadline; stdout on success.
pub fn run_with_timeout(
    mut command: Command,
    stdin: Option<&str>,
    timeout: Duration,
) -> Option<String> {
    command
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .stdin(if stdin.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .env("GIT_OPTIONAL_LOCKS", "0")
        .env("GH_PROMPT_DISABLED", "1")
        .env("NO_COLOR", "1");
    let mut child = command.spawn().ok()?;
    if let Some(input) = stdin {
        let mut pipe = child.stdin.take()?;
        if pipe.write_all(input.as_bytes()).is_err() {
            let _ = child.kill();
            let _ = child.wait();
            return None;
        }
    }
    let mut stdout = child.stdout.take()?;
    let reader = std::thread::spawn(move || {
        let mut text = String::new();
        let _ = stdout.read_to_string(&mut text);
        text
    });
    let deadline = Instant::now() + timeout;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let text = reader.join().ok()?;
                return status.success().then_some(text);
            }
            Ok(None) if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(40)),
            _ => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
        }
    }
}

/// MCP clients start the server with a minimal PATH: look where package
/// managers install binaries too.
fn find_binary(name: &str) -> Option<PathBuf> {
    let mut dirs: Vec<PathBuf> = std::env::var_os("PATH")
        .map(|path| std::env::split_paths(&path).collect())
        .unwrap_or_default();
    dirs.extend(["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin"].map(PathBuf::from));
    if let Some(home) = dirs::home_dir() {
        dirs.push(home.join(".local/bin"));
    }
    dirs.into_iter()
        .map(|dir| dir.join(name))
        .find(|path| path.is_file())
}

fn gh_json(repo: &Path, args: &[&str]) -> Option<Value> {
    let gh = find_binary("gh")?;
    let mut command = Command::new(gh);
    command.current_dir(repo).args(args);
    serde_json::from_str(&run_with_timeout(command, None, COMMAND_TIMEOUT)?).ok()
}

fn github(repo: &Path) -> Option<Value> {
    let pulls = gh_json(
        repo,
        &[
            "pr",
            "list",
            "--state",
            "all",
            "--limit",
            GH_LIMIT,
            "--json",
            "number,title,state,mergedAt,closedAt,updatedAt,headRefName,url",
        ],
    )?;
    let issues = gh_json(
        repo,
        &[
            "issue",
            "list",
            "--state",
            "all",
            "--limit",
            GH_LIMIT,
            "--json",
            "number,title,state,closedAt,updatedAt,url",
        ],
    )
    .unwrap_or_else(|| json!([]));
    let pulls = pulls.as_array().cloned().unwrap_or_default();
    let issues = issues.as_array().cloned().unwrap_or_default();
    let open_pulls: Vec<Value> = pulls
        .iter()
        .filter(|pr| pr["state"] == "OPEN")
        .map(|pr| json!({ "number": pr["number"], "title": pr["title"], "branch": pr["headRefName"], "updated_at": pr["updatedAt"] }))
        .collect();
    let merged: Vec<Value> = pulls
        .iter()
        .filter(|pr| pr["state"] == "MERGED")
        .take(MAX_LISTED)
        .map(|pr| json!({ "number": pr["number"], "title": pr["title"], "merged_at": pr["mergedAt"] }))
        .collect();
    let open_issues: Vec<Value> = issues
        .iter()
        .filter(|issue| issue["state"] == "OPEN")
        .map(|issue| json!({ "number": issue["number"], "title": issue["title"], "updated_at": issue["updatedAt"] }))
        .collect();
    let url = pulls
        .iter()
        .chain(issues.iter())
        .find_map(|item| item["url"].as_str())
        .and_then(|url| {
            url.split("/pull/")
                .next()
                .map(|base| base.split("/issues/").next().unwrap_or(base).to_string())
        });
    let mut states = Map::new();
    for pr in &pulls {
        if let Some(number) = pr["number"].as_u64() {
            states.insert(
                format!("pr:{number}"),
                json!({ "state": pr["state"], "title": pr["title"], "at": pr["mergedAt"].as_str().or(pr["closedAt"].as_str()) }),
            );
        }
    }
    for issue in &issues {
        if let Some(number) = issue["number"].as_u64() {
            states.insert(
                format!("issue:{number}"),
                json!({ "state": issue["state"], "title": issue["title"], "at": issue["closedAt"] }),
            );
        }
    }
    Some(json!({
        "url": url,
        "open_pull_requests": open_pulls,
        "merged_recently": merged,
        "open_issues": open_issues,
        "states": states,
    }))
}

fn orchestrator_jobs() -> Vec<Value> {
    let Some(dir) = dirs::home_dir().map(|home| home.join(".cf-orchestrator").join("jobs")) else {
        return Vec::new();
    };
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    entries
        .flatten()
        .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "json"))
        .filter_map(|entry| std::fs::read_to_string(entry.path()).ok())
        .filter_map(|text| serde_json::from_str::<Value>(&text).ok())
        .filter(|job| job["kind"] == "deploy")
        .collect()
}

fn deploy_url(job_id: &str) -> Option<String> {
    let path = dirs::home_dir()?
        .join(".cf-orchestrator")
        .join("logs")
        .join(format!("{job_id}.log"));
    let log = std::fs::read_to_string(path).ok()?;
    log.split_whitespace()
        .rev()
        .find(|word| {
            word.starts_with("https://")
                && (word.contains(".pages.dev") || word.contains(".workers.dev"))
        })
        .map(|url| url.trim_end_matches(['.', ',', ')']).to_string())
}

fn job_time(job: &Value, key: &str) -> Option<DateTime<FixedOffset>> {
    DateTime::parse_from_rfc3339(job[key].as_str()?).ok()
}

fn deploys(snapshot: &RepoSnapshot, facts: &Value) -> Vec<Value> {
    let Some(targets) = facts.get("deploy").and_then(Value::as_array) else {
        return Vec::new();
    };
    let jobs = orchestrator_jobs();
    let mut out = Vec::new();
    for target in targets {
        let Some(name) = target["name"].as_str() else {
            continue;
        };
        let dir = target["dir"].as_str().unwrap_or(".");
        let mut mine: Vec<&Value> = jobs.iter().filter(|job| job["project"] == name).collect();
        mine.sort_by_key(|job| job["created_at"].as_str().unwrap_or("").to_string());
        let success = mine.iter().rev().find(|job| job["status"] == "succeeded");
        let failure = mine.iter().rev().find(|job| job["status"] == "failed");
        let mut entry = json!({ "name": name, "dir": dir });
        if let Some(job) = success {
            let ended = job_time(job, "ended_at").or_else(|| job_time(job, "created_at"));
            entry["last_success"] = json!(ended.map(|t| t.to_rfc3339()));
            if let Some(id) = job["id"].as_str() {
                entry["url"] = json!(deploy_url(id));
            }
            if let Some(ended) = ended {
                let prefix = if dir == "." {
                    String::new()
                } else {
                    format!("{}/", dir.trim_end_matches('/'))
                };
                let since: Vec<&verify::Commit> = snapshot
                    .commits
                    .iter()
                    .filter(|c| {
                        c.date > ended
                            && (prefix.is_empty() || c.files.iter().any(|f| f.starts_with(&prefix)))
                    })
                    .collect();
                entry["commits_since"] = json!(since.len());
                if let Some(latest) = since.first() {
                    entry["latest_undeployed"] =
                        json!(format!("{} {}", latest.short_sha(), latest.subject));
                }
            }
        }
        if let Some(job) = failure {
            let failed_at = job_time(job, "created_at");
            let after_success = match (failed_at, success.and_then(|s| job_time(s, "created_at"))) {
                (Some(failed), Some(succeeded)) => failed > succeeded,
                (Some(_), None) => true,
                _ => false,
            };
            if after_success {
                entry["last_failure"] = json!(failed_at.map(|t| t.to_rfc3339()));
                entry["failure_summary"] = job["summary"].clone();
            }
        }
        out.push(entry);
    }
    out
}

fn supabase_ref(repo: &Path, snapshot: &RepoSnapshot) -> Option<String> {
    let mut counts: HashMap<String, usize> = HashMap::new();
    let mut count_refs = |text: &str| {
        for (index, _) in text.match_indices(".supabase.co") {
            let start = index.saturating_sub(20);
            let (Some(candidate), Some(before)) = (text.get(start..index), text.get(..start))
            else {
                continue;
            };
            let boundary = before
                .chars()
                .next_back()
                .is_none_or(|c| !c.is_ascii_alphanumeric());
            if candidate.len() == 20
                && boundary
                && candidate.chars().all(|c| c.is_ascii_lowercase())
            {
                *counts.entry(candidate.to_string()).or_insert(0) += 1;
            }
        }
    };
    for file in snapshot.files.iter().filter(|f| f.ends_with("project-ref")) {
        if let Ok(text) = std::fs::read_to_string(repo.join(file)) {
            let reference = text.trim();
            if reference.len() == 20 && reference.chars().all(|c| c.is_ascii_lowercase()) {
                return Some(reference.to_string());
            }
        }
    }
    let mut command = Command::new("git");
    command
        .arg("-C")
        .arg(repo)
        .args(["grep", "-h", "-o", "-I", "-E", "[a-z]{20}\\.supabase\\.co"]);
    if let Some(found) = run_with_timeout(command, None, COMMAND_TIMEOUT) {
        count_refs(&found);
    }
    let mut dirs: HashSet<PathBuf> = HashSet::new();
    dirs.insert(repo.to_path_buf());
    for file in &snapshot.files {
        if file.ends_with("package.json")
            || file.ends_with("pubspec.yaml")
            || file.ends_with("wrangler.toml")
        {
            if let Some((dir, _)) = file.rsplit_once('/') {
                dirs.insert(repo.join(dir));
            }
        }
    }
    for dir in dirs {
        for name in ENV_FILES {
            if let Ok(text) = std::fs::read_to_string(dir.join(name)) {
                count_refs(&text);
            }
        }
    }
    counts
        .into_iter()
        .max_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(&a.0)))
        .map(|(reference, _)| reference)
}

fn supabase_token() -> Option<String> {
    if let Ok(token) = std::env::var("SUPABASE_ACCESS_TOKEN") {
        if !token.trim().is_empty() {
            return Some(token.trim().to_string());
        }
    }
    keychain_token()
}

#[cfg(target_os = "macos")]
fn keychain_token() -> Option<String> {
    let bytes = security_framework::passwords::get_generic_password(
        KEYCHAIN_SERVICE,
        SUPABASE_KEYCHAIN_ACCOUNT,
    )
    .ok()?;
    let token = String::from_utf8(bytes).ok()?;
    (!token.trim().is_empty()).then(|| token.trim().to_string())
}

#[cfg(not(target_os = "macos"))]
fn keychain_token() -> Option<String> {
    None
}

/// Migration name as Supabase records it: the file name without the
/// leading version and the `.sql` extension.
fn migration_name(file: &str) -> (&str, &str) {
    let stem = file.trim_end_matches(".sql");
    match stem.split_once('_') {
        Some((version, name)) if version.chars().all(|c| c.is_ascii_digit()) => (version, name),
        _ => ("", stem),
    }
}

/// Repository migration files missing from production, and production
/// migrations missing from the repository. A file matches by version or,
/// for migrations applied through the API (versioned at apply time), by name.
pub fn compare_migrations(
    files: &[&str],
    applied: &[(String, String)],
) -> (Vec<String>, Vec<String>) {
    let versions: HashSet<&str> = applied
        .iter()
        .map(|(version, _)| version.as_str())
        .collect();
    let names: HashSet<&str> = applied.iter().map(|(_, name)| name.as_str()).collect();
    let pending: Vec<String> = files
        .iter()
        .filter(|file| {
            let (version, name) = migration_name(file);
            let applied_version = !version.is_empty() && versions.contains(version);
            !applied_version && !names.contains(name)
        })
        .map(|file| file.to_string())
        .collect();
    let file_versions: HashSet<&str> = files.iter().map(|f| migration_name(f).0).collect();
    let file_names: HashSet<&str> = files.iter().map(|f| migration_name(f).1).collect();
    let missing: Vec<String> = applied
        .iter()
        .filter(|(version, name)| {
            !file_versions.contains(version.as_str()) && !file_names.contains(name.as_str())
        })
        .map(|(version, name)| format!("{version}_{name}"))
        .collect();
    (pending, missing)
}

fn supabase(repo: &Path, snapshot: &RepoSnapshot) -> Value {
    let Some(reference) = supabase_ref(repo, snapshot) else {
        return json!({ "status": "unknown_project", "hint": "No <ref>.supabase.co found in the repository or its .env files." });
    };
    let Some(token) = supabase_token() else {
        return json!({
            "ref": reference,
            "status": "not_configured",
            "hint": "Set SUPABASE_ACCESS_TOKEN for the server, or store a token (scope database:read) in the Keychain: service MemoryPilot, account supabase-access-token.",
        });
    };
    let Some(curl) = find_binary("curl") else {
        return json!({ "ref": reference, "status": "error", "hint": "curl not found" });
    };
    let mut command = Command::new(curl);
    command.args([
        "-sS",
        "--max-time",
        "8",
        "-H",
        "@-",
        "-w",
        "\n%{http_code}",
        &format!("{SUPABASE_API}/{reference}/database/migrations"),
    ]);
    let header = format!("Authorization: Bearer {token}\n");
    let Some(response) = run_with_timeout(command, Some(&header), COMMAND_TIMEOUT) else {
        return json!({ "ref": reference, "status": "error", "hint": "Supabase API unreachable" });
    };
    let (body, code) = response.rsplit_once('\n').unwrap_or((&response, ""));
    if code.trim() != "200" {
        return json!({ "ref": reference, "status": "error", "http_status": code.trim() });
    }
    let Ok(Value::Array(rows)) = serde_json::from_str::<Value>(body) else {
        return json!({ "ref": reference, "status": "error", "hint": "unexpected response" });
    };
    let applied: Vec<(String, String)> = rows
        .iter()
        .filter_map(|row| {
            Some((
                row["version"].as_str()?.to_string(),
                row["name"].as_str().unwrap_or("").to_string(),
            ))
        })
        .collect();
    let files: Vec<&str> = snapshot
        .files
        .iter()
        .filter(|f| f.ends_with(".sql") && f.contains("supabase/migrations/"))
        .filter_map(|f| f.rsplit('/').next())
        .collect();
    let (pending, missing) = compare_migrations(&files, &applied);
    let latest = applied
        .iter()
        .max_by(|a, b| a.0.cmp(&b.0))
        .map(|(v, n)| format!("{v}_{n}"));
    json!({
        "ref": reference,
        "status": if pending.is_empty() { "up_to_date" } else { "pending" },
        "applied": applied.len(),
        "latest_applied": latest,
        "pending_in_repo": pending.iter().take(MAX_LISTED).collect::<Vec<_>>(),
        "pending_count": pending.len(),
        "applied_not_in_repo": missing.iter().take(MAX_LISTED).collect::<Vec<_>>(),
        "applied_not_in_repo_count": missing.len(),
    })
}

/// A few lines for recall and the project brain.
pub fn compact(data: &Value) -> Value {
    let mut lines = Map::new();
    if let Some(github) = data.get("github") {
        let prs = github["open_pull_requests"]
            .as_array()
            .map(Vec::len)
            .unwrap_or(0);
        let issues = github["open_issues"].as_array().map(Vec::len).unwrap_or(0);
        let listed: Vec<String> = github["open_pull_requests"]
            .as_array()
            .map(|items| {
                items
                    .iter()
                    .take(3)
                    .map(|pr| format!("#{} {}", pr["number"], pr["title"].as_str().unwrap_or("")))
                    .collect()
            })
            .unwrap_or_default();
        let mut line = format!(
            "{} open, {} open",
            counted(prs as u64, "pull request"),
            counted(issues as u64, "issue")
        );
        if !listed.is_empty() {
            line.push_str(&format!(" ({})", listed.join("; ")));
        }
        lines.insert("github".into(), json!(line));
    }
    if let Some(deploys) = data.get("deploys").and_then(Value::as_array) {
        let items: Vec<String> = deploys
            .iter()
            .map(|d| {
                let mut line = format!(
                    "{}: last deploy {}",
                    d["name"].as_str().unwrap_or("?"),
                    d["last_success"]
                        .as_str()
                        .and_then(|t| t.get(..16))
                        .unwrap_or("never")
                );
                if let Some(count) = d["commits_since"].as_u64().filter(|c| *c > 0) {
                    line.push_str(&format!(", {} not deployed", counted(count, "commit")));
                }
                if let Some(failed) = d["last_failure"].as_str() {
                    line.push_str(&format!(
                        ", last failure {}",
                        failed.get(..16).unwrap_or(failed)
                    ));
                }
                line
            })
            .collect();
        lines.insert("deploys".into(), json!(items));
    }
    if let Some(supabase) = data.get("supabase") {
        let line = match supabase["status"].as_str() {
            Some("up_to_date") => format!(
                "production up to date ({} migrations applied)",
                supabase["applied"]
            ),
            Some("pending") => format!(
                "{} migration files not applied in production: {}",
                supabase["pending_count"],
                supabase["pending_in_repo"]
                    .as_array()
                    .map(|items| items
                        .iter()
                        .filter_map(Value::as_str)
                        .collect::<Vec<_>>()
                        .join(", "))
                    .unwrap_or_default()
            ),
            Some(other) => format!("production migrations: {other}"),
            None => String::new(),
        };
        if !line.is_empty() {
            lines.insert("supabase".into(), json!(line));
        }
    }
    Value::Object(lines)
}

/// `PR #12`, `pull/12`, `issue #7`, `issues/7` in a memory: (kind, number).
fn counted(count: u64, noun: &str) -> String {
    if count == 1 {
        format!("1 {noun}")
    } else {
        format!("{count} {noun}s")
    }
}

pub fn references(text: &str) -> Vec<(&'static str, u64)> {
    let lower = text.to_lowercase();
    let mut found = Vec::new();
    let patterns: &[(&str, &'static str)] = &[
        ("pr #", "pr"),
        ("pr#", "pr"),
        ("pull request #", "pr"),
        ("/pull/", "pr"),
        ("issue #", "issue"),
        ("issue#", "issue"),
        ("/issues/", "issue"),
    ];
    for (pattern, kind) in patterns {
        for (index, _) in lower.match_indices(pattern) {
            let boundary = lower[..index]
                .chars()
                .next_back()
                .is_none_or(|c| !c.is_alphanumeric());
            if !boundary && !pattern.starts_with('/') {
                continue;
            }
            let digits: String = lower[index + pattern.len()..]
                .chars()
                .take_while(char::is_ascii_digit)
                .collect();
            if let Ok(number) = digits.parse::<u64>() {
                if !found.contains(&(*kind, number)) {
                    found.push((*kind, number));
                }
            }
        }
    }
    found
}

/// Referenced pull requests merged or issues closed, from cached data.
pub fn closed_references(data: &Value, text: &str) -> Vec<Value> {
    let Some(states) = data.get("github").and_then(|g| g.get("states")) else {
        return Vec::new();
    };
    references(text)
        .into_iter()
        .filter_map(|(kind, number)| {
            let state = states.get(format!("{kind}:{number}"))?;
            let closed = matches!(state["state"].as_str(), Some("MERGED" | "CLOSED"));
            closed.then(|| json!({ "ref": format!("{kind} #{number}"), "state": state["state"], "title": state["title"], "at": state["at"] }))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Needs a logged-in `gh`: `cargo test live_github -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn live_github() {
        let repo = std::env::var("MEMORYPILOT_LIVE_REPO")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")));
        let github = github(&repo).expect("gh returned nothing");
        println!(
            "{}",
            serde_json::to_string_pretty(&compact(&json!({ "github": github }))).unwrap()
        );
        assert!(github.get("states").is_some());
    }

    #[test]
    fn references_are_read_from_prose_and_urls() {
        let text = "Voir PR #12 et https://github.com/a/b/issues/7, pas l'expr#3 ni le ticket 9.";
        assert_eq!(references(text), vec![("pr", 12), ("issue", 7)]);
    }

    #[test]
    fn closed_references_need_a_merged_or_closed_state() {
        let data = json!({ "github": { "states": {
            "pr:12": { "state": "MERGED", "title": "fix", "at": "2026-10-20T10:00:00Z" },
            "issue:7": { "state": "OPEN", "title": "bug", "at": null }
        }}});
        let found = closed_references(&data, "PR #12 et issue #7");
        assert_eq!(found.len(), 1);
        assert_eq!(found[0]["ref"], "pr #12");
    }

    #[test]
    fn migrations_match_by_version_or_by_name() {
        let files = [
            "0083_bank.sql",
            "0084_allow_identical_bank_operations.sql",
            "20261012180000_ai_grants.sql",
            "0085_new.sql",
        ];
        let applied = vec![
            ("0083".to_string(), "bank".to_string()),
            (
                "20261020101010".to_string(),
                "allow_identical_bank_operations".to_string(),
            ),
            ("20261012180000".to_string(), "ai_grants".to_string()),
            (
                "20261021000000".to_string(),
                "hotfix_only_in_prod".to_string(),
            ),
        ];
        let (pending, missing) = compare_migrations(&files, &applied);
        assert_eq!(pending, vec!["0085_new.sql"]);
        assert_eq!(missing, vec!["20261021000000_hotfix_only_in_prod"]);
    }

    #[test]
    fn commands_time_out_instead_of_hanging() {
        let mut command = Command::new("sleep");
        command.arg("5");
        let started = Instant::now();
        assert!(run_with_timeout(command, None, Duration::from_millis(300)).is_none());
        assert!(started.elapsed() < Duration::from_secs(2));
        let mut command = Command::new("cat");
        command.arg("-");
        assert_eq!(
            run_with_timeout(command, Some("hello"), COMMAND_TIMEOUT).as_deref(),
            Some("hello")
        );
    }
}
