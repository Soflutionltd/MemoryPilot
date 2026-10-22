//! Verified memory: a memory that cites files of a Git repository is checked
//! against that repository's history before it is handed to an agent.
//!
//! Nothing from Git is copied into the database. The tracked files and the
//! recent commits (with the files each one touched) are read on demand,
//! cached per repository and re-read only when HEAD moves, so a commit made a
//! second ago already counts. Everything here is deterministic: no model.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use chrono::{DateTime, FixedOffset};

const MAX_COMMITS: usize = 3000;
const HEAD_RECHECK: Duration = Duration::from_secs(2);
const MAX_ANCHORS: usize = 12;
/// A directory cited in prose ("api/sites/[siteId]/") only anchors the memory
/// when it is specific: past this many tracked files it says nothing.
const MAX_DIR_FILES: usize = 400;
const MAX_DIR_PREFIXES: usize = 5;
/// Mass commits (formatting, migrations, imports) touch everything and say
/// nothing about a given file.
const MAX_COMMIT_FILES: usize = 40;
const MIN_SHARED_WORDS: usize = 3;
const MIN_SHARED_RATIO: f64 = 0.5;
/// A word that appears in more commit subjects than this share of the
/// history ("site", "date" in their own repo) says nothing about a match.
const COMMON_WORD_SHARE: usize = 30;

const CODE_EXTENSIONS: &[&str] = &[
    "rs", "ts", "tsx", "js", "jsx", "mjs", "cjs", "svelte", "vue", "py", "go", "dart", "swift",
    "kt", "java", "rb", "php", "sql", "toml", "json", "yaml", "yml", "md", "css", "scss", "html",
    "sh",
];

const STOPWORDS: &[&str] = &[
    "pour",
    "dans",
    "avec",
    "sans",
    "sont",
    "plus",
    "tous",
    "toutes",
    "tout",
    "toute",
    "cette",
    "celle",
    "celui",
    "entre",
    "après",
    "avant",
    "aussi",
    "comme",
    "leur",
    "leurs",
    "nous",
    "vous",
    "elle",
    "elles",
    "mais",
    "donc",
    "quand",
    "faire",
    "fait",
    "être",
    "avoir",
    "chaque",
    "depuis",
    "encore",
    "même",
    "très",
    "ainsi",
    "alors",
    "lors",
    "vers",
    "chez",
    "corrige",
    "corrigé",
    "corrigée",
    "correction",
    "ajout",
    "ajoute",
    "ajouté",
    "ajoutée",
    "supprime",
    "supprimé",
    "when",
    "with",
    "from",
    "that",
    "this",
    "into",
    "have",
    "been",
    "were",
    "will",
    "what",
    "which",
    "there",
    "their",
    "than",
    "then",
    "only",
    "also",
    "more",
    "some",
    "each",
    "does",
    "done",
    "make",
    "made",
    "after",
    "before",
    "they",
    "them",
    "your",
    "fixed",
    "fixes",
    "adds",
    "added",
    "update",
    "updated",
    "remove",
    "removed",
    "now",
    "longer",
];

#[derive(Debug, Clone)]
pub struct Commit {
    pub sha: String,
    pub date: DateTime<FixedOffset>,
    pub subject: String,
    /// Conventional Commits type (`fix`, `feat`…), when the subject follows it.
    pub kind: Option<String>,
    pub scope: Option<String>,
    pub files: Vec<String>,
}

impl Commit {
    pub fn short_sha(&self) -> &str {
        &self.sha[..self.sha.len().min(7)]
    }

    pub fn is_fix(&self) -> bool {
        if self.kind.as_deref() == Some("fix") {
            return true;
        }
        let lower = self.subject.to_lowercase();
        ["fix", "corrig", "résou", "resolve", "hotfix"]
            .iter()
            .any(|marker| lower.contains(marker))
    }

    /// Commits that can change behaviour, as opposed to docs, tests, style, CI or chores.
    pub fn changes_behaviour(&self) -> bool {
        matches!(
            self.kind.as_deref(),
            Some("fix" | "feat" | "refactor" | "perf" | "revert")
        )
    }
}

pub struct RepoSnapshot {
    pub head: String,
    pub files: Vec<String>,
    /// Newest first, as `git log` lists them.
    pub commits: Vec<Commit>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Anchor {
    File(String),
    /// Directory prefix, always ending with `/`.
    Dir(String),
}

impl Anchor {
    pub fn label(&self) -> &str {
        match self {
            Anchor::File(path) | Anchor::Dir(path) => path,
        }
    }

    pub fn covers(&self, file: &str) -> bool {
        match self {
            Anchor::File(path) => path == file,
            Anchor::Dir(prefix) => file.starts_with(prefix.as_str()),
        }
    }
}

pub fn disabled() -> bool {
    std::env::var("MEMORYPILOT_VERIFY")
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "0" | "off" | "false"
            )
        })
        .unwrap_or(false)
}

/// `type(scope)!: description` → (type, scope). Anything else → (None, None).
pub fn parse_subject(subject: &str) -> (Option<String>, Option<String>) {
    let Some((head, _)) = subject.split_once(':') else {
        return (None, None);
    };
    let head = head.trim_end_matches('!');
    let (kind, scope) = match head.split_once('(') {
        Some((kind, rest)) => match rest.strip_suffix(')') {
            Some(scope) if !scope.is_empty() => (kind, Some(scope.to_string())),
            _ => return (None, None),
        },
        None => (head, None),
    };
    if kind.is_empty() || kind.len() > 12 || !kind.chars().all(|c| c.is_ascii_lowercase()) {
        return (None, None);
    }
    (Some(kind.to_string()), scope)
}

fn git(repo: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(args)
        .env("GIT_OPTIONAL_LOCKS", "0")
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// Top level of the Git work tree containing `path`, if any.
pub fn repo_root(path: &str) -> Option<PathBuf> {
    let trimmed = path.trim();
    if trimmed.is_empty() || !Path::new(trimmed).is_dir() {
        return None;
    }
    let top = git(Path::new(trimmed), &["rev-parse", "--show-toplevel"])?;
    let top = top.trim();
    (!top.is_empty()).then(|| PathBuf::from(top))
}

type SnapshotCache = HashMap<PathBuf, (Instant, Arc<RepoSnapshot>)>;

fn cache() -> &'static Mutex<SnapshotCache> {
    static CACHE: OnceLock<Mutex<SnapshotCache>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Tracked files and recent history of `repo`, re-read only when HEAD moved.
pub fn snapshot(repo: &Path) -> Option<Arc<RepoSnapshot>> {
    if let Ok(guard) = cache().lock() {
        if let Some((checked, snapshot)) = guard.get(repo) {
            if checked.elapsed() < HEAD_RECHECK {
                return Some(Arc::clone(snapshot));
            }
        }
    }
    let head = git(repo, &["rev-parse", "HEAD"])?.trim().to_string();
    if let Ok(mut guard) = cache().lock() {
        if let Some((checked, snapshot)) = guard.get_mut(repo) {
            if snapshot.head == head {
                *checked = Instant::now();
                return Some(Arc::clone(snapshot));
            }
        }
    }
    let loaded = Arc::new(load(repo, head)?);
    if let Ok(mut guard) = cache().lock() {
        guard.insert(repo.to_path_buf(), (Instant::now(), Arc::clone(&loaded)));
    }
    Some(loaded)
}

fn load(repo: &Path, head: String) -> Option<RepoSnapshot> {
    let listing = git(repo, &["ls-files", "-z"])?;
    let files = listing
        .split('\0')
        .filter(|file| !file.is_empty())
        .map(str::to_string)
        .collect();
    let max_count = format!("--max-count={}", MAX_COMMITS);
    let log = git(
        repo,
        &[
            "log",
            &max_count,
            "--no-color",
            "--format=%x1e%H%x1f%cI%x1f%s",
            "--name-only",
        ],
    )?;
    Some(RepoSnapshot {
        head,
        files,
        commits: parse_log(&log),
    })
}

pub fn parse_log(log: &str) -> Vec<Commit> {
    log.split('\u{1e}')
        .filter_map(|record| {
            let mut lines = record.lines();
            let header = lines.next()?;
            let mut parts = header.split('\u{1f}');
            let sha = parts.next()?.trim().to_string();
            if sha.is_empty() {
                return None;
            }
            let date = DateTime::parse_from_rfc3339(parts.next()?.trim()).ok()?;
            let subject = parts.next().unwrap_or("").trim().to_string();
            let files = lines
                .map(str::trim)
                .filter(|line| !line.is_empty())
                .map(str::to_string)
                .collect();
            let (kind, scope) = parse_subject(&subject);
            Some(Commit {
                sha,
                date,
                subject,
                kind,
                scope,
                files,
            })
        })
        .collect()
}

fn extension(path: &str) -> Option<&str> {
    let name = path.rsplit('/').next()?;
    let (_, ext) = name.rsplit_once('.')?;
    (!ext.is_empty() && ext.len() <= 6).then_some(ext)
}

fn has_code_extension(path: &str) -> bool {
    extension(path)
        .map(|ext| CODE_EXTENSIONS.contains(&ext.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

/// Tokens of `content` that look like a path or a source file name.
pub fn path_candidates(content: &str) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut candidates = Vec::new();
    let separators = |c: char| {
        c.is_whitespace()
            || matches!(
                c,
                '`' | '"' | '\'' | '(' | ')' | ',' | ';' | '<' | '>' | '|'
            )
    };
    for raw in content.split(separators) {
        if raw.contains("://") || raw.starts_with('~') {
            continue;
        }
        let allowed = |c: char| {
            c.is_alphanumeric()
                || matches!(c, '/' | '.' | '_' | '-' | '[' | ']' | '$' | '@' | '+' | '*')
        };
        let mut token = raw.trim_matches(|c: char| !allowed(c));
        loop {
            let trimmed = token.trim_end_matches(['.', '*', '-']);
            let trimmed = trimmed.strip_suffix("/*").unwrap_or(trimmed);
            if trimmed.len() == token.len() {
                break;
            }
            token = trimmed;
        }
        if token.chars().count() < 4 {
            continue;
        }
        let path_like = token.trim_matches('/').contains('/') || has_code_extension(token);
        if path_like && seen.insert(token.to_string()) {
            candidates.push(token.to_string());
        }
    }
    candidates
}

fn resolve(candidate: &str, files: &[String]) -> Vec<Anchor> {
    let path = candidate.trim_start_matches("./").trim_start_matches('/');
    if path.chars().count() < 4 {
        return Vec::new();
    }
    if files.iter().any(|file| file == path) {
        return vec![Anchor::File(path.to_string())];
    }
    if has_code_extension(path) {
        let suffix = format!("/{}", path);
        let hits: Vec<&String> = files
            .iter()
            .filter(|file| file.ends_with(&suffix))
            .take(4)
            .collect();
        return if (1..=3).contains(&hits.len()) {
            hits.into_iter()
                .map(|file| Anchor::File(file.clone()))
                .collect()
        } else {
            Vec::new()
        };
    }
    let dir = path.trim_end_matches('/');
    if !dir.contains('/') {
        return Vec::new();
    }
    let prefix = format!("{}/", dir);
    let needle = format!("/{}/", dir);
    let mut prefixes: Vec<String> = Vec::new();
    let mut matched = 0usize;
    for file in files {
        let start = if file.starts_with(&prefix) {
            Some(0)
        } else {
            file.find(&needle).map(|index| index + 1)
        };
        let Some(start) = start else { continue };
        matched += 1;
        if matched > MAX_DIR_FILES {
            return Vec::new();
        }
        let found = file[..start + prefix.len()].to_string();
        if !prefixes.contains(&found) {
            prefixes.push(found);
        }
    }
    if prefixes.len() > MAX_DIR_PREFIXES {
        return Vec::new();
    }
    prefixes.into_iter().map(Anchor::Dir).collect()
}

/// Files and directories of the repository that `content` cites.
pub fn anchors(content: &str, repo: &Path, files: &[String]) -> Vec<Anchor> {
    let repo_prefix = format!("{}/", repo.to_string_lossy());
    let mut found: Vec<Anchor> = Vec::new();
    for candidate in path_candidates(content) {
        let relative = candidate.strip_prefix(&repo_prefix).unwrap_or(&candidate);
        for anchor in resolve(relative, files) {
            if !found.contains(&anchor) {
                found.push(anchor);
            }
        }
        if found.len() >= MAX_ANCHORS {
            found.truncate(MAX_ANCHORS);
            break;
        }
    }
    found
}

/// Commits made after `since` that touched at least one anchored file.
pub fn touching<'a>(
    snapshot: &'a RepoSnapshot,
    anchors: &[Anchor],
    since: DateTime<FixedOffset>,
) -> Vec<&'a Commit> {
    if anchors.is_empty() {
        return Vec::new();
    }
    snapshot
        .commits
        .iter()
        .filter(|commit| commit.date > since && commit.files.len() <= MAX_COMMIT_FILES)
        .filter(|commit| {
            commit
                .files
                .iter()
                .any(|file| anchors.iter().any(|anchor| anchor.covers(file)))
        })
        .collect()
}

fn words(text: &str) -> Vec<String> {
    let lower = text.to_lowercase();
    let mut seen = HashSet::new();
    lower
        .split(|c: char| !c.is_alphanumeric())
        .filter(|word| word.chars().count() >= 4 && !STOPWORDS.contains(word))
        .filter(|word| seen.insert(word.to_string()))
        .map(str::to_string)
        .collect()
}

fn word_forms(word: &str) -> Vec<String> {
    let mut forms = vec![word.to_string()];
    for stem in crate::stemming::stem_text(word).split_whitespace() {
        if stem.chars().count() >= 3 && !forms.iter().any(|form| form == stem) {
            forms.push(stem.to_string());
        }
    }
    forms
}

/// Raw words and their stems, the vocabulary a commit subject is matched against.
pub fn vocabulary(text: &str) -> HashSet<String> {
    words(text)
        .iter()
        .flat_map(|word| word_forms(word))
        .collect()
}

fn commit_words(commit: &Commit) -> Vec<String> {
    match &commit.scope {
        Some(scope) => words(&format!("{} {}", scope, commit.subject)),
        None => words(&commit.subject),
    }
}

/// Words that appear in too many commit subjects of the repository to identify anything.
fn common_words(snapshot: &RepoSnapshot) -> HashSet<String> {
    let mut frequency: HashMap<String, usize> = HashMap::new();
    for commit in &snapshot.commits {
        for word in commit_words(commit) {
            *frequency.entry(word).or_insert(0) += 1;
        }
    }
    let limit = (snapshot.commits.len() / COMMON_WORD_SHARE).max(3);
    frequency
        .into_iter()
        .filter(|(_, count)| *count > limit)
        .map(|(word, _)| word)
        .collect()
}

/// How many distinctive words of the commit (subject + scope) the memory
/// shares, and which share of the commit's distinctive words that is.
pub fn shared_words(
    vocabulary: &HashSet<String>,
    commit: &Commit,
    common: &HashSet<String>,
) -> (usize, f64) {
    let distinctive: Vec<String> = commit_words(commit)
        .into_iter()
        .filter(|word| !common.contains(word))
        .collect();
    if distinctive.is_empty() {
        return (0, 0.0);
    }
    let shared = distinctive
        .iter()
        .filter(|word| {
            word_forms(word)
                .iter()
                .any(|form| vocabulary.contains(form))
        })
        .count();
    (shared, shared as f64 / distinctive.len() as f64)
}

/// Fix commits made after `since` whose subject reads like a resolution of
/// `text`, best first.
pub fn fix_matches<'a>(
    snapshot: &'a RepoSnapshot,
    text: &str,
    since: DateTime<FixedOffset>,
    limit: usize,
) -> Vec<&'a Commit> {
    let vocabulary = vocabulary(text);
    if vocabulary.is_empty() {
        return Vec::new();
    }
    let common = common_words(snapshot);
    let mut scored: Vec<(usize, f64, &Commit)> = snapshot
        .commits
        .iter()
        .filter(|commit| commit.date > since && commit.is_fix())
        .filter_map(|commit| {
            let (shared, ratio) = shared_words(&vocabulary, commit, &common);
            (shared >= MIN_SHARED_WORDS && ratio >= MIN_SHARED_RATIO)
                .then_some((shared, ratio, commit))
        })
        .collect();
    scored.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.total_cmp(&a.1)));
    scored
        .into_iter()
        .take(limit)
        .map(|(_, _, commit)| commit)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn commit(subject: &str, date: &str, files: &[&str]) -> Commit {
        let (kind, scope) = parse_subject(subject);
        Commit {
            sha: "9ede2ce0000000000000000000000000000000000".into(),
            date: DateTime::parse_from_rfc3339(date).unwrap(),
            subject: subject.into(),
            kind,
            scope,
            files: files.iter().map(|f| f.to_string()).collect(),
        }
    }

    fn snapshot_of(files: &[&str], commits: Vec<Commit>) -> RepoSnapshot {
        RepoSnapshot {
            head: "head".into(),
            files: files.iter().map(|f| f.to_string()).collect(),
            commits,
        }
    }

    #[test]
    fn parses_conventional_subjects() {
        assert_eq!(
            parse_subject("fix(security): rôles par site"),
            (Some("fix".into()), Some("security".into()))
        );
        assert_eq!(parse_subject("feat!: new API"), (Some("feat".into()), None));
        assert_eq!(parse_subject("Merge branch 'main'"), (None, None));
        assert_eq!(parse_subject("WIP: stuff"), (None, None));
        assert_eq!(parse_subject("fix(: broken"), (None, None));
    }

    #[test]
    fn parses_git_log_records() {
        let log = "\u{1e}abc123\u{1f}2026-10-18T10:00:00+02:00\u{1f}fix(auth): close signup\n\nsrc/lib/auth.ts\nsrc/hooks.server.ts\n\u{1e}def456\u{1f}2026-10-17T09:00:00+02:00\u{1f}docs: readme\n\nREADME.md\n";
        let commits = parse_log(log);
        assert_eq!(commits.len(), 2);
        assert_eq!(commits[0].kind.as_deref(), Some("fix"));
        assert_eq!(
            commits[0].files,
            vec!["src/lib/auth.ts", "src/hooks.server.ts"]
        );
        assert_eq!(commits[1].files, vec!["README.md"]);
    }

    #[test]
    fn anchors_resolve_files_dirs_and_drop_prose() {
        let files = [
            "src/lib/auth.ts",
            "src/routes/api/sites/[siteId]/+server.ts",
            "src/routes/api/sites/[siteId]/pages/+server.ts",
            "src/routes/api/pages/[pageId]/+server.ts",
            "README.md",
        ];
        let files: Vec<String> = files.iter().map(|f| f.to_string()).collect();
        let content = "Faille: `auth.ts` sans disableSignUp, routes /api/sites/[siteId]/* et /api/pages/[pageId]/* sans contrôle site.user_id (save/store/remember). Voir https://example.com/a/b.ts";
        let found = anchors(content, Path::new("/repo"), &files);
        assert!(found.contains(&Anchor::File("src/lib/auth.ts".into())));
        assert!(found.contains(&Anchor::Dir("src/routes/api/sites/[siteId]/".into())));
        assert!(found.contains(&Anchor::Dir("src/routes/api/pages/[pageId]/".into())));
        assert_eq!(found.len(), 3, "prose and URLs must not anchor: {found:?}");
    }

    #[test]
    fn absolute_paths_inside_the_repo_are_made_relative() {
        let files = vec!["src/lib/auth.ts".to_string()];
        let found = anchors("Voir /repo/src/lib/auth.ts.", Path::new("/repo"), &files);
        assert_eq!(found, vec![Anchor::File("src/lib/auth.ts".into())]);
    }

    #[test]
    fn touching_keeps_only_later_commits_on_anchored_files() {
        let snapshot = snapshot_of(
            &["src/lib/auth.ts"],
            vec![
                commit(
                    "fix(auth): close signup",
                    "2026-10-18T10:00:00+02:00",
                    &["src/lib/auth.ts"],
                ),
                commit("docs: readme", "2026-10-18T11:00:00+02:00", &["README.md"]),
                commit(
                    "feat(auth): signup",
                    "2026-10-10T10:00:00+02:00",
                    &["src/lib/auth.ts"],
                ),
            ],
        );
        let since = DateTime::parse_from_rfc3339("2026-10-17T12:00:00+02:00").unwrap();
        let hits = touching(&snapshot, &[Anchor::File("src/lib/auth.ts".into())], since);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].subject, "fix(auth): close signup");
    }

    #[test]
    fn a_later_fix_reads_like_the_resolution_of_a_prose_bug() {
        let bug = "LightFrame 17/10/2026 : faille constatée en lecture de code. Inscription Better Auth ouverte + 22 routes session sans contrôle site.user_id. Un inconnu peut modifier/publier le site d'une instance. security";
        let since = DateTime::parse_from_rfc3339("2026-10-17T12:00:00+02:00").unwrap();
        let snapshot = snapshot_of(
            &[],
            vec![
                commit(
                    "fix(security): rôles par site et contrôle d'accès sur toutes les routes de l'admin",
                    "2026-10-19T10:00:00+02:00",
                    &[],
                ),
                commit("fix(editor): moveBlockDown sans effet", "2026-10-19T11:00:00+02:00", &[]),
                commit("feat(security): contrôle des routes du site", "2026-10-19T12:00:00+02:00", &[]),
            ],
        );
        let hits = fix_matches(&snapshot, bug, since, 3);
        assert_eq!(
            hits.len(),
            1,
            "only the security fix matches: {:?}",
            hits.iter().map(|c| &c.subject).collect::<Vec<_>>()
        );
        assert!(hits[0].subject.starts_with("fix(security)"));
    }

    #[test]
    fn earlier_fixes_never_count() {
        let bug = "Les routes du site sans contrôle d'accès, faille de sécurité";
        let since = DateTime::parse_from_rfc3339("2026-10-20T00:00:00+02:00").unwrap();
        let snapshot = snapshot_of(
            &[],
            vec![commit(
                "fix(security): contrôle d'accès sur les routes du site",
                "2026-10-19T10:00:00+02:00",
                &[],
            )],
        );
        assert!(fix_matches(&snapshot, bug, since, 3).is_empty());
    }
}
