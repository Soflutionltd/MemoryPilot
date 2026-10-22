//! Derived facts: what a project's repository says about itself right now
//! (stack and versions, Cloudflare bindings, deploy targets, Supabase
//! migrations, SvelteKit routes, last commit). Computed from the tracked
//! files on demand and never stored, so they cannot go stale; cached until
//! HEAD or a manifest changes.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::SystemTime;

use serde_json::{json, Map, Value};
use yaml_rust2::{Yaml, YamlLoader};

use crate::verify::{self, RepoSnapshot};

const MAX_MANIFEST_DEPTH: usize = 3;
const MAX_MANIFESTS: usize = 24;
const MAX_ENDPOINTS: usize = 40;
const MAX_COMPACT_STACK: usize = 14;
const RECENT_DAYS: i64 = 30;

const MANIFEST_NAMES: &[&str] = &[
    "package.json",
    "Cargo.toml",
    "pubspec.yaml",
    "wrangler.toml",
    "wrangler.json",
    "wrangler.jsonc",
];
const SKIPPED_SEGMENTS: &[&str] = &[
    "node_modules",
    "vendor",
    "example",
    "examples",
    "test",
    "tests",
    "fixtures",
    "build",
    "dist",
    "target",
    "templates",
    ".svelte-kit",
];

/// Packages whose version is part of the stack, with the names prose uses
/// for them ("Tailwind 4", "SvelteKit 2"). Only these are shown in compact
/// facts and checked against version claims in memories.
pub const FRAMEWORKS: &[(&str, &[&str])] = &[
    (
        "@sveltejs/kit",
        &["sveltekit", "svelte kit", "@sveltejs/kit"],
    ),
    ("svelte", &["svelte"]),
    ("vite", &["vite"]),
    ("tailwindcss", &["tailwind", "tailwindcss", "tailwind css"]),
    ("typescript", &["typescript"]),
    ("next", &["next.js", "nextjs"]),
    ("react", &["react"]),
    ("vue", &["vue", "vue.js"]),
    ("nuxt", &["nuxt"]),
    ("astro", &["astro"]),
    ("hono", &["hono"]),
    ("better-auth", &["better auth", "better-auth"]),
    ("@supabase/supabase-js", &["supabase-js"]),
    ("drizzle-orm", &["drizzle"]),
    ("electron", &["electron"]),
    ("@tauri-apps/api", &["tauri"]),
    ("tauri", &["tauri"]),
    ("gsap", &["gsap"]),
    ("wrangler", &["wrangler"]),
    ("zod", &["zod"]),
    ("flutter", &["flutter"]),
    ("dart", &["dart"]),
    ("supabase_flutter", &["supabase_flutter"]),
    ("flutter_riverpod", &["riverpod"]),
    ("go_router", &["go_router"]),
    ("purchases_flutter", &["purchases_flutter"]),
    ("tokio", &["tokio"]),
    ("axum", &["axum"]),
    ("rusqlite", &["rusqlite"]),
    ("fastembed", &["fastembed"]),
];

#[derive(Debug, Clone)]
pub struct Dependency {
    pub name: String,
    pub version: String,
    pub file: String,
}

pub struct ProjectFacts {
    pub full: Value,
    pub compact: Value,
    pub dependencies: Vec<Dependency>,
}

type FactsCache = HashMap<PathBuf, (String, Arc<ProjectFacts>)>;

fn cache() -> &'static Mutex<FactsCache> {
    static CACHE: OnceLock<Mutex<FactsCache>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn depth(path: &str) -> usize {
    path.matches('/').count()
}

fn skipped(path: &str) -> bool {
    path.split('/')
        .any(|segment| SKIPPED_SEGMENTS.contains(&segment))
}

fn file_name(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

fn parent(path: &str) -> &str {
    path.rsplit_once('/').map(|(dir, _)| dir).unwrap_or("")
}

fn manifests(files: &[String]) -> Vec<&String> {
    let mut found: Vec<&String> = files
        .iter()
        .filter(|path| MANIFEST_NAMES.contains(&file_name(path)))
        .filter(|path| depth(path) <= MAX_MANIFEST_DEPTH && !skipped(path))
        .collect();
    found.sort_by(|a, b| depth(a).cmp(&depth(b)).then(a.cmp(b)));
    found.truncate(MAX_MANIFESTS);
    found
}

fn orchestrator_path() -> Option<PathBuf> {
    dirs::home_dir().map(|home| home.join(".cf-orchestrator").join("projects.json"))
}

fn modified(path: &Path) -> String {
    std::fs::metadata(path)
        .and_then(|meta| meta.modified())
        .ok()
        .and_then(|time| time.duration_since(SystemTime::UNIX_EPOCH).ok())
        .map(|age| age.as_millis().to_string())
        .unwrap_or_default()
}

/// Facts of the repository at `repo`, recomputed only when HEAD or a manifest changed.
pub fn project_facts(repo: &Path) -> Option<Arc<ProjectFacts>> {
    let snapshot = verify::snapshot(repo)?;
    let manifests = manifests(&snapshot.files);
    let mut key = snapshot.head.clone();
    for manifest in &manifests {
        key.push('|');
        key.push_str(&modified(&repo.join(manifest)));
    }
    if let Some(path) = orchestrator_path() {
        key.push('|');
        key.push_str(&modified(&path));
    }
    for refs in [".git/packed-refs", ".git/refs/tags"] {
        key.push('|');
        key.push_str(&modified(&repo.join(refs)));
    }
    if let Ok(guard) = cache().lock() {
        if let Some((cached_key, facts)) = guard.get(repo) {
            if *cached_key == key {
                return Some(Arc::clone(facts));
            }
        }
    }
    let facts = Arc::new(compute(repo, &snapshot, &manifests));
    if let Ok(mut guard) = cache().lock() {
        guard.insert(repo.to_path_buf(), (key, Arc::clone(&facts)));
    }
    Some(facts)
}

fn compute(repo: &Path, snapshot: &RepoSnapshot, manifests: &[&String]) -> ProjectFacts {
    let mut packages = Vec::new();
    let mut cloudflare = Vec::new();
    let mut dependencies = Vec::new();
    let mut sveltekit_dirs = Vec::new();
    for manifest in manifests {
        let Ok(text) = std::fs::read_to_string(repo.join(manifest.as_str())) else {
            continue;
        };
        let parsed = match file_name(manifest) {
            "package.json" => parse_package_json(manifest, &text, &mut dependencies),
            "Cargo.toml" => parse_cargo_toml(manifest, &text, &mut dependencies),
            "pubspec.yaml" => parse_pubspec(manifest, &text, &mut dependencies),
            "wrangler.toml" => {
                if let Some(worker) = parse_wrangler_toml(manifest, &text) {
                    cloudflare.push(worker);
                }
                None
            }
            _ => {
                if let Some(worker) = parse_wrangler_json(manifest, &text) {
                    cloudflare.push(worker);
                }
                None
            }
        };
        if let Some(package) = parsed {
            if file_name(manifest) == "package.json"
                && package["dependencies"].get("@sveltejs/kit").is_some()
            {
                sveltekit_dirs.push(parent(manifest).to_string());
            }
            packages.push(package);
        }
    }

    let mut full = Map::new();
    full.insert("repo".into(), json!(repo.to_string_lossy()));
    full.insert(
        "head".into(),
        json!(&snapshot.head[..snapshot.head.len().min(7)]),
    );
    if let Some(last) = snapshot.commits.first() {
        full.insert(
            "last_commit".into(),
            json!({ "sha": last.short_sha(), "date": last.date.to_rfc3339(), "subject": last.subject }),
        );
        let horizon = last.date - chrono::Duration::days(RECENT_DAYS);
        let recent = snapshot.commits.iter().filter(|c| c.date > horizon).count();
        full.insert("commits_last_30_days_of_activity".into(), json!(recent));
    }
    full.insert("tracked_files".into(), json!(snapshot.files.len()));
    if !packages.is_empty() {
        full.insert("packages".into(), json!(packages));
    }
    if !cloudflare.is_empty() {
        full.insert("cloudflare".into(), json!(cloudflare));
    }
    let deploy = orchestrator_targets(repo);
    if !deploy.is_empty() {
        full.insert("deploy".into(), json!(deploy));
    }
    let migrations = supabase_migrations(&snapshot.files);
    if !migrations.is_empty() {
        full.insert("supabase_migrations".into(), json!(migrations));
    }
    if let Some(release) = release(repo, &snapshot.files) {
        full.insert("release".into(), release);
    }
    let routes: Vec<Value> = sveltekit_dirs
        .iter()
        .filter_map(|dir| sveltekit_routes(&snapshot.files, dir))
        .collect();
    if !routes.is_empty() {
        full.insert("routes".into(), json!(routes));
    }
    let full = Value::Object(full);
    let compact = compact(&full, &dependencies);
    ProjectFacts {
        full,
        compact,
        dependencies,
    }
}

/// Latest tag and the commits since it, plus release-please versions.
fn release(repo: &Path, files: &[String]) -> Option<Value> {
    let mut release = Map::new();
    if let Some(tag) = verify::git(repo, &["describe", "--tags", "--abbrev=0"]) {
        let tag = tag.trim().to_string();
        let range = format!("{tag}..HEAD");
        if let Some(count) = verify::git(repo, &["rev-list", "--count", &range]) {
            release.insert(
                "commits_since_tag".into(),
                json!(count.trim().parse::<u64>().ok()),
            );
        }
        release.insert("latest_tag".into(), json!(tag));
    }
    if files.iter().any(|f| f == ".release-please-manifest.json") {
        if let Some(manifest) = std::fs::read_to_string(repo.join(".release-please-manifest.json"))
            .ok()
            .and_then(|text| serde_json::from_str::<Value>(&text).ok())
        {
            release.insert("release_please".into(), manifest);
        }
    }
    (!release.is_empty()).then_some(Value::Object(release))
}

fn record(dependencies: &mut Vec<Dependency>, file: &str, name: &str, version: &str) {
    dependencies.push(Dependency {
        name: name.to_string(),
        version: version.to_string(),
        file: file.to_string(),
    });
}

fn parse_package_json(file: &str, text: &str, dependencies: &mut Vec<Dependency>) -> Option<Value> {
    let package: Value = serde_json::from_str(text).ok()?;
    let mut deps = BTreeMap::new();
    for section in ["dependencies", "devDependencies", "peerDependencies"] {
        if let Some(entries) = package.get(section).and_then(Value::as_object) {
            for (name, version) in entries {
                if let Some(version) = version.as_str() {
                    deps.entry(name.clone())
                        .or_insert_with(|| version.to_string());
                }
            }
        }
    }
    for (name, version) in &deps {
        record(dependencies, file, name, version);
    }
    let scripts: Vec<&String> = package
        .get("scripts")
        .and_then(Value::as_object)
        .map(|scripts| scripts.keys().collect())
        .unwrap_or_default();
    Some(json!({
        "file": file,
        "type": "npm",
        "name": package.get("name"),
        "version": package.get("version"),
        "package_manager": package.get("packageManager"),
        "scripts": scripts,
        "dependencies": deps,
    }))
}

fn toml_dependency_version(value: &toml::Value) -> Option<String> {
    match value {
        toml::Value::String(version) => Some(version.clone()),
        toml::Value::Table(table) => {
            if let Some(version) = table.get("version").and_then(toml::Value::as_str) {
                Some(version.to_string())
            } else if table.contains_key("path") {
                Some("path".into())
            } else if table.contains_key("git") {
                Some("git".into())
            } else if table.get("workspace").and_then(toml::Value::as_bool) == Some(true) {
                Some("workspace".into())
            } else {
                None
            }
        }
        _ => None,
    }
}

fn parse_cargo_toml(file: &str, text: &str, dependencies: &mut Vec<Dependency>) -> Option<Value> {
    let manifest: toml::Table = text.parse().ok()?;
    let package = manifest.get("package").and_then(toml::Value::as_table);
    let field = |key: &str| {
        package
            .and_then(|p| p.get(key))
            .and_then(toml::Value::as_str)
            .map(str::to_string)
    };
    let mut deps = BTreeMap::new();
    for section in ["dependencies", "dev-dependencies", "build-dependencies"] {
        if let Some(entries) = manifest.get(section).and_then(toml::Value::as_table) {
            for (name, value) in entries {
                if let Some(version) = toml_dependency_version(value) {
                    deps.entry(name.clone()).or_insert(version);
                }
            }
        }
    }
    for (name, version) in &deps {
        record(dependencies, file, name, version);
    }
    let members: Vec<&str> = manifest
        .get("workspace")
        .and_then(|w| w.get("members"))
        .and_then(toml::Value::as_array)
        .map(|members| members.iter().filter_map(toml::Value::as_str).collect())
        .unwrap_or_default();
    let mut value = json!({
        "file": file,
        "type": "cargo",
        "name": field("name"),
        "version": field("version"),
        "edition": field("edition"),
        "rust_version": field("rust-version"),
        "dependencies": deps,
    });
    if !members.is_empty() {
        value["workspace_members"] = json!(members);
    }
    Some(value)
}

fn yaml_scalar(value: &Yaml) -> Option<String> {
    match value {
        Yaml::String(text) | Yaml::Real(text) => Some(text.clone()),
        Yaml::Integer(number) => Some(number.to_string()),
        _ => None,
    }
}

fn parse_pubspec(file: &str, text: &str, dependencies: &mut Vec<Dependency>) -> Option<Value> {
    let docs = YamlLoader::load_from_str(text).ok()?;
    let doc = docs.first()?;
    let mut deps = BTreeMap::new();
    for section in ["dependencies", "dev_dependencies"] {
        let Some(entries) = doc[section].as_hash() else {
            continue;
        };
        for (name, value) in entries {
            let Some(name) = name.as_str() else { continue };
            let version = match value {
                Yaml::Hash(spec) => {
                    let get = |key: &str| spec.get(&Yaml::String(key.into()));
                    if let Some(version) = get("version").and_then(yaml_scalar) {
                        version
                    } else if get("sdk").is_some() {
                        "sdk".into()
                    } else if get("path").is_some() {
                        "path".into()
                    } else if get("git").is_some() {
                        "git".into()
                    } else {
                        continue;
                    }
                }
                Yaml::Null => "any".into(),
                other => match yaml_scalar(other) {
                    Some(version) => version,
                    None => continue,
                },
            };
            deps.entry(name.to_string()).or_insert(version);
        }
    }
    let environment = &doc["environment"];
    if let Some(sdk) = yaml_scalar(&environment["sdk"]) {
        record(dependencies, file, "dart", &sdk);
    }
    if let Some(flutter) = yaml_scalar(&environment["flutter"]) {
        record(dependencies, file, "flutter", &flutter);
    }
    for (name, version) in &deps {
        record(dependencies, file, name, version);
    }
    Some(json!({
        "file": file,
        "type": "pub",
        "name": yaml_scalar(&doc["name"]),
        "version": yaml_scalar(&doc["version"]),
        "dart_sdk": yaml_scalar(&environment["sdk"]),
        "flutter_sdk": yaml_scalar(&environment["flutter"]),
        "dependencies": deps,
    }))
}

/// Cloudflare config as JSON, whatever its source format. Variable values
/// are dropped: `[vars]` sometimes holds values that should be secrets.
fn summarize_wrangler(file: &str, config: &Value) -> Option<Value> {
    let config = config.as_object()?;
    let names = |key: &str, field: &str| -> Vec<Value> {
        config
            .get(key)
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(|item| item.get(field).cloned())
                    .collect()
            })
            .unwrap_or_default()
    };
    let mut routes: Vec<Value> = Vec::new();
    if let Some(route) = config.get("route") {
        routes.push(
            route
                .get("pattern")
                .cloned()
                .unwrap_or_else(|| route.clone()),
        );
    }
    if let Some(list) = config.get("routes").and_then(Value::as_array) {
        routes.extend(list.iter().map(|route| {
            route
                .get("pattern")
                .cloned()
                .unwrap_or_else(|| route.clone())
        }));
    }
    let vars: Vec<&String> = config
        .get("vars")
        .and_then(Value::as_object)
        .map(|vars| vars.keys().collect())
        .unwrap_or_default();
    let kind = if config.contains_key("pages_build_output_dir") {
        "pages"
    } else {
        "worker"
    };
    let mut summary = json!({
        "file": file,
        "kind": kind,
        "name": config.get("name"),
        "compatibility_date": config.get("compatibility_date"),
    });
    let mut add = |key: &str, value: Vec<Value>| {
        if !value.is_empty() {
            summary[key] = json!(value);
        }
    };
    add("d1", names("d1_databases", "database_name"));
    add("kv", names("kv_namespaces", "binding"));
    add("r2", names("r2_buckets", "bucket_name"));
    add("queues", names("queues", "binding"));
    add("routes", routes);
    add("vars", vars.into_iter().map(|name| json!(name)).collect());
    Some(summary)
}

fn parse_wrangler_toml(file: &str, text: &str) -> Option<Value> {
    let table: toml::Table = text.parse().ok()?;
    let config = serde_json::to_value(table).ok()?;
    summarize_wrangler(file, &config)
}

fn parse_wrangler_json(file: &str, text: &str) -> Option<Value> {
    let config: Value = serde_json::from_str(&strip_jsonc(text)).ok()?;
    summarize_wrangler(file, &config)
}

/// JSON with comments and trailing commas (wrangler.jsonc) → plain JSON.
pub fn strip_jsonc(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    let mut in_string = false;
    while let Some(c) = chars.next() {
        if in_string {
            out.push(c);
            if c == '\\' {
                if let Some(escaped) = chars.next() {
                    out.push(escaped);
                }
            } else if c == '"' {
                in_string = false;
            }
            continue;
        }
        match (c, chars.peek()) {
            ('"', _) => {
                in_string = true;
                out.push(c);
            }
            ('/', Some('/')) => {
                for skipped in chars.by_ref() {
                    if skipped == '\n' {
                        out.push('\n');
                        break;
                    }
                }
            }
            ('/', Some('*')) => {
                chars.next();
                let mut previous = ' ';
                for skipped in chars.by_ref() {
                    if previous == '*' && skipped == '/' {
                        break;
                    }
                    previous = skipped;
                }
            }
            _ => out.push(c),
        }
    }
    remove_trailing_commas(&out)
}

fn remove_trailing_commas(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut in_string = false;
    let mut index = 0;
    while index < chars.len() {
        let c = chars[index];
        if in_string {
            out.push(c);
            if c == '\\' && index + 1 < chars.len() {
                out.push(chars[index + 1]);
                index += 1;
            } else if c == '"' {
                in_string = false;
            }
        } else if c == '"' {
            in_string = true;
            out.push(c);
        } else if c == ',' {
            let next = chars[index + 1..].iter().find(|n| !n.is_whitespace());
            if !matches!(next, Some('}') | Some(']')) {
                out.push(c);
            }
        } else {
            out.push(c);
        }
        index += 1;
    }
    out
}

/// cf-orchestrator projects deployed from this repository. Only names and
/// targets: the file also holds account identifiers.
fn orchestrator_targets(repo: &Path) -> Vec<Value> {
    let Some(path) = orchestrator_path() else {
        return Vec::new();
    };
    let Ok(text) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    let Ok(config) = serde_json::from_str::<Value>(&text) else {
        return Vec::new();
    };
    let Some(projects) = config.get("projects").and_then(Value::as_object) else {
        return Vec::new();
    };
    let root = repo.to_string_lossy().trim_end_matches('/').to_string();
    let mut targets: Vec<Value> = projects
        .iter()
        .filter_map(|(name, project)| {
            let cwd = project.get("cwd")?.as_str()?.trim_end_matches('/');
            let dir = if cwd == root {
                "."
            } else {
                cwd.strip_prefix(&format!("{}/", root))?
            };
            Some(json!({
                "name": name,
                "type": project.get("type"),
                "cf_project_name": project.get("cf_project_name"),
                "deploy_target": project.get("deploy_target"),
                "dir": dir,
            }))
        })
        .collect();
    targets.sort_by(|a, b| a["dir"].as_str().cmp(&b["dir"].as_str()));
    targets
}

fn supabase_migrations(files: &[String]) -> Vec<Value> {
    let mut by_dir: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for file in files {
        if file.ends_with(".sql") && file.contains("supabase/migrations/") && !skipped(file) {
            by_dir
                .entry(parent(file))
                .or_default()
                .push(file_name(file));
        }
    }
    by_dir
        .into_iter()
        .map(|(dir, mut names)| {
            names.sort_unstable();
            json!({ "dir": dir, "count": names.len(), "latest": names.last() })
        })
        .collect()
}

fn route_path(relative_dir: &str) -> String {
    let segments: Vec<&str> = relative_dir
        .split('/')
        .filter(|segment| {
            let group = segment.starts_with('(') && segment.ends_with(')');
            !segment.is_empty() && !group
        })
        .collect();
    format!("/{}", segments.join("/"))
}

fn sveltekit_routes(files: &[String], app_dir: &str) -> Option<Value> {
    let prefix = if app_dir.is_empty() {
        "src/routes/".to_string()
    } else {
        format!("{}/src/routes/", app_dir)
    };
    let mut pages = 0usize;
    let mut endpoints: Vec<String> = Vec::new();
    for file in files {
        let Some(relative) = file.strip_prefix(&prefix) else {
            continue;
        };
        let name = file_name(relative);
        let dir = parent(relative);
        if name == "+page.svelte" {
            pages += 1;
        } else if name == "+server.ts" || name == "+server.js" {
            endpoints.push(route_path(dir));
        }
    }
    if pages == 0 && endpoints.is_empty() {
        return None;
    }
    endpoints.sort();
    let total = endpoints.len();
    endpoints.truncate(MAX_ENDPOINTS);
    Some(json!({
        "app": if app_dir.is_empty() { "." } else { app_dir },
        "framework": "sveltekit",
        "pages": pages,
        "endpoints": total,
        "endpoint_paths": endpoints,
    }))
}

/// A few lines for recall and the project brain.
fn compact(full: &Value, dependencies: &[Dependency]) -> Value {
    let mut stack: Vec<String> = Vec::new();
    for (package, _) in FRAMEWORKS {
        if stack.len() >= MAX_COMPACT_STACK {
            break;
        }
        let mut versions: Vec<&str> = Vec::new();
        for dependency in dependencies.iter().filter(|d| d.name == *package) {
            if !version_numbers(&dependency.version).is_empty()
                && !versions.contains(&dependency.version.as_str())
            {
                versions.push(&dependency.version);
            }
        }
        if !versions.is_empty() {
            stack.push(format!("{} {}", package, versions.join(" / ")));
        }
    }
    let mut compact = Map::new();
    if !stack.is_empty() {
        compact.insert("stack".into(), json!(stack));
    }
    if let Some(workers) = full.get("cloudflare").and_then(Value::as_array) {
        let lines: Vec<String> = workers
            .iter()
            .map(|worker| {
                let mut line = format!(
                    "{} {} ({})",
                    worker["kind"].as_str().unwrap_or("worker"),
                    worker["name"].as_str().unwrap_or("?"),
                    worker["file"].as_str().unwrap_or("")
                );
                for key in ["d1", "r2", "kv"] {
                    if let Some(items) = worker.get(key).and_then(Value::as_array) {
                        let names: Vec<&str> = items.iter().filter_map(Value::as_str).collect();
                        line.push_str(&format!(", {}: {}", key, names.join(" ")));
                    }
                }
                line
            })
            .collect();
        compact.insert("cloudflare".into(), json!(lines));
    }
    if let Some(targets) = full.get("deploy").and_then(Value::as_array) {
        let lines: Vec<String> = targets
            .iter()
            .map(|t| {
                format!(
                    "cf-orchestrator {} ({}, {})",
                    t["name"].as_str().unwrap_or("?"),
                    t["type"].as_str().unwrap_or("?"),
                    t["dir"].as_str().unwrap_or(".")
                )
            })
            .collect();
        compact.insert("deploy".into(), json!(lines));
    }
    if let Some(migrations) = full.get("supabase_migrations").and_then(Value::as_array) {
        let lines: Vec<String> = migrations
            .iter()
            .map(|m| {
                format!(
                    "{}: {} files, latest {}",
                    m["dir"].as_str().unwrap_or(""),
                    m["count"],
                    m["latest"].as_str().unwrap_or("")
                )
            })
            .collect();
        compact.insert("supabase_migrations".into(), json!(lines));
    }
    if let Some(routes) = full.get("routes").and_then(Value::as_array) {
        let lines: Vec<String> = routes
            .iter()
            .map(|r| {
                format!(
                    "{}: {} pages, {} endpoints",
                    r["app"].as_str().unwrap_or("."),
                    r["pages"],
                    r["endpoints"]
                )
            })
            .collect();
        compact.insert("routes".into(), json!(lines));
    }
    if let Some(release) = full.get("release") {
        if let Some(tag) = release["latest_tag"].as_str() {
            compact.insert(
                "release".into(),
                json!(format!(
                    "{} ({} commits since)",
                    tag, release["commits_since_tag"]
                )),
            );
        }
    }
    if let Some(last) = full.get("last_commit") {
        compact.insert(
            "last_commit".into(),
            json!(format!(
                "{} {} {}",
                last["sha"].as_str().unwrap_or(""),
                last["date"].as_str().unwrap_or("").get(..10).unwrap_or(""),
                last["subject"].as_str().unwrap_or("")
            )),
        );
    }
    Value::Object(compact)
}

/// Leading `major` or `major.minor` of a version or range ("^4.1.0" → [4, 1, 0]).
pub fn version_numbers(text: &str) -> Vec<u64> {
    text.trim_start_matches(|c: char| !c.is_ascii_digit())
        .split(|c: char| !(c.is_ascii_digit() || c == '.'))
        .next()
        .unwrap_or("")
        .split('.')
        .map_while(|part| part.parse().ok())
        .collect()
}

/// Whether a claimed version is an older release line than the repository's
/// (major, or major.minor below 1.0). A newer claim is a plan or a blocked
/// upgrade ("ne pas passer à TypeScript 7"), not a stale fact.
fn older_line(claim: &[u64], actual: &[u64]) -> bool {
    match (claim.first(), actual.first()) {
        (Some(0), Some(0)) => match (claim.get(1), actual.get(1)) {
            (Some(a), Some(b)) => a < b,
            _ => false,
        },
        (Some(a), Some(b)) => a < b,
        _ => false,
    }
}

fn same_line(claim: &[u64], actual: &[u64]) -> bool {
    match (claim.first(), actual.first()) {
        (Some(0), Some(0)) => match (claim.get(1), actual.get(1)) {
            (Some(a), Some(b)) => a == b,
            _ => true,
        },
        (Some(a), Some(b)) => a == b,
        _ => true,
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Contradiction {
    pub claim: String,
    pub package: String,
    pub actual: String,
    pub file: String,
}

/// Versions `text` states for a framework ("Tailwind 3", "svelte@5").
fn claims(lower: &str, alias: &str) -> Vec<(String, Vec<u64>)> {
    let mut found = Vec::new();
    let mut from = 0;
    while let Some(offset) = lower[from..].find(alias) {
        let start = from + offset;
        let end = start + alias.len();
        from = end;
        let boundary_before = lower[..start].chars().next_back().is_none_or(|c| {
            !c.is_alphanumeric() && !matches!(c, '_' | '-' | '/' | '.' | '+' | '@')
        });
        if !boundary_before {
            continue;
        }
        let rest = &lower[end..];
        let after_space = rest.strip_prefix(' ').unwrap_or(rest);
        let after_marker = after_space
            .strip_prefix('@')
            .or_else(|| after_space.strip_prefix('v'))
            .unwrap_or(after_space);
        let digits: String = after_marker
            .chars()
            .take_while(|c| c.is_ascii_digit() || *c == '.')
            .collect();
        let digits = digits.trim_end_matches('.');
        if digits.is_empty() || !after_marker.starts_with(|c: char| c.is_ascii_digit()) {
            continue;
        }
        let tail = &after_marker[digits.len()..];
        let tail = tail.strip_prefix(".x").unwrap_or(tail);
        if tail.starts_with(|c: char| c.is_alphanumeric() || c == '%' || c == '/') {
            continue;
        }
        let consumed = rest.len() - after_marker.len();
        let claimed = &lower[start..end + consumed + digits.len()];
        let numbers = version_numbers(digits);
        if numbers.first().is_some_and(|major| *major < 1000) {
            found.push((claimed.to_string(), numbers));
        }
    }
    found
}

/// Framework versions stated in `text` that the repository no longer uses.
/// A memory that also names the current version (a migration note) is not
/// contradicted.
pub fn contradictions(text: &str, dependencies: &[Dependency]) -> Vec<Contradiction> {
    let lower = text.to_lowercase();
    let mut found = Vec::new();
    for (package, aliases) in FRAMEWORKS {
        let versions: Vec<&Dependency> = dependencies
            .iter()
            .filter(|d| d.name == *package && !version_numbers(&d.version).is_empty())
            .collect();
        if versions.is_empty() {
            continue;
        }
        let stated: Vec<(String, Vec<u64>)> = aliases
            .iter()
            .flat_map(|alias| claims(&lower, alias))
            .collect();
        if stated.is_empty() {
            continue;
        }
        let matches_repo = |numbers: &Vec<u64>| {
            versions
                .iter()
                .any(|d| same_line(numbers, &version_numbers(&d.version)))
        };
        if stated.iter().any(|(_, numbers)| matches_repo(numbers)) {
            continue;
        }
        let stale = stated.iter().find(|(_, numbers)| {
            versions
                .iter()
                .all(|d| older_line(numbers, &version_numbers(&d.version)))
        });
        let Some((claim, _)) = stale else {
            continue;
        };
        let current = versions[0];
        found.push(Contradiction {
            claim: claim.clone(),
            package: current.name.clone(),
            actual: current.version.clone(),
            file: current.file.clone(),
        });
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dep(name: &str, version: &str) -> Dependency {
        Dependency {
            name: name.into(),
            version: version.into(),
            file: "package.json".into(),
        }
    }

    #[test]
    fn version_numbers_read_ranges() {
        assert_eq!(version_numbers("^4.1.0"), vec![4, 1, 0]);
        assert_eq!(version_numbers(">=3.5.0 <4.0.0"), vec![3, 5, 0]);
        assert_eq!(version_numbers("~0.31"), vec![0, 31]);
        assert!(version_numbers("workspace:*").is_empty());
        assert!(version_numbers("latest").is_empty());
    }

    #[test]
    fn stale_version_claims_are_contradicted() {
        let deps = vec![
            dep("tailwindcss", "^4.1.0"),
            dep("svelte", "^5.2.0"),
            dep("@sveltejs/kit", "^2.8.0"),
        ];
        let found = contradictions(
            "Le site utilise Tailwind 3 et Svelte 5 avec SvelteKit 2.",
            &deps,
        );
        assert_eq!(found.len(), 1, "{found:?}");
        assert_eq!(found[0].claim, "tailwind 3");
        assert_eq!(found[0].actual, "^4.1.0");
    }

    #[test]
    fn migration_notes_and_lookalikes_are_not_contradicted() {
        let deps = vec![
            dep("tailwindcss", "^4.1.0"),
            dep("svelte", "^5.2.0"),
            dep("vite", "^8.0.0"),
        ];
        assert!(contradictions("Migration de Tailwind 3 vers Tailwind 4 faite.", &deps).is_empty());
        assert!(contradictions("svelte-check 3 et sveltekit sans version", &deps).is_empty());
        assert!(contradictions("Vite 8.x, svelte@5", &deps).is_empty());
        assert!(contradictions("Vite 7.2 au lieu de", &deps).len() == 1);
        assert!(contradictions("vite 3D", &deps).is_empty());
        assert!(contradictions("NE PAS passer à Vite 9 tant que", &deps).is_empty());
        assert!(contradictions("+page.svelte 6394 lignes", &deps).is_empty());
        assert!(contradictions("La revue 2 du site", &[dep("vue", "^3.4.0")]).is_empty());
    }

    #[test]
    fn zero_major_compares_minor() {
        let deps = vec![dep("rusqlite", "0.31")];
        assert!(contradictions("rusqlite 0.31 bundled", &deps).is_empty());
        assert_eq!(contradictions("rusqlite 0.29 bundled", &deps).len(), 1);
    }

    #[test]
    fn jsonc_comments_and_trailing_commas_are_removed() {
        let text = "{\n // name\n \"name\": \"site\", /* x */\n \"routes\": [\"a.com/*\",],\n \"url\": \"https://x.com//y\",\n}";
        let value: Value = serde_json::from_str(&strip_jsonc(text)).expect("valid json");
        assert_eq!(value["name"], "site");
        assert_eq!(value["routes"][0], "a.com/*");
        assert_eq!(value["url"], "https://x.com//y");
    }

    #[test]
    fn manifests_parse_into_facts() {
        let mut deps = Vec::new();
        let cargo = "[package]\nname = \"demo\"\nversion = \"1.2.0\"\nedition = \"2021\"\n[dependencies]\ntokio = { version = \"1.40\", features = [\"full\"] }\nlocal = { path = \"../local\" }\nserde = \"1\"\n";
        let facts = parse_cargo_toml("Cargo.toml", cargo, &mut deps).unwrap();
        assert_eq!(facts["edition"], "2021");
        assert_eq!(facts["dependencies"]["tokio"], "1.40");
        assert_eq!(facts["dependencies"]["local"], "path");

        let pubspec = "name: app\nversion: 1.0.0+5\nenvironment:\n  sdk: '>=3.5.0 <4.0.0'\n  flutter: '>=3.27.0'\ndependencies:\n  flutter:\n    sdk: flutter\n  go_router: ^14.2.0\n  http: 1.2\n";
        let facts = parse_pubspec("pubspec.yaml", pubspec, &mut deps).unwrap();
        assert_eq!(facts["dependencies"]["flutter"], "sdk");
        assert_eq!(facts["dependencies"]["http"], "1.2");
        assert!(deps
            .iter()
            .any(|d| d.name == "dart" && d.version == ">=3.5.0 <4.0.0"));

        let wrangler = "name = \"site\"\ncompatibility_date = \"2026-01-01\"\n[[d1_databases]]\nbinding = \"DB\"\ndatabase_name = \"site-db\"\ndatabase_id = \"abc\"\n[vars]\nAPI_KEY = \"should-not-appear\"\n";
        let facts = parse_wrangler_toml("wrangler.toml", wrangler).unwrap();
        assert_eq!(facts["d1"][0], "site-db");
        assert_eq!(facts["vars"][0], "API_KEY");
        assert!(!facts.to_string().contains("should-not-appear"));
    }

    #[test]
    fn sveltekit_routes_drop_groups() {
        let files: Vec<String> = [
            "apps/web/src/routes/(app)/companies/[id]/+page.svelte",
            "apps/web/src/routes/api/bank/connect/+server.ts",
            "apps/web/src/routes/+page.svelte",
        ]
        .iter()
        .map(|f| f.to_string())
        .collect();
        let routes = sveltekit_routes(&files, "apps/web").unwrap();
        assert_eq!(routes["pages"], 2);
        assert_eq!(routes["endpoint_paths"][0], "/api/bank/connect");
    }
}
