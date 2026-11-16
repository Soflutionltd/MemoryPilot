//! A complete, readable copy of the Cursor conversations, one Markdown
//! file each, kept outside Cursor.
//!
//! The index (`store.rs`) keeps what recall needs: each request and the
//! final answer of each turn, cut to a few thousand characters. This
//! archive keeps everything that was said, at full length: every message
//! the user typed and every text reply, sub-agents included. Tool calls,
//! their results, file contents and the model's reasoning are left out.
//! Secrets are masked before anything is written.
//!
//! Cursor's store is opened immutable and never written. Once it is gone,
//! the archive is the source the index is rebuilt from (`read_archive`).

use super::{cursor, text, Message, Role, SourceConversation};
use crate::secrets;
use serde::Serialize;
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

/// Folder of the conversations no project could be found for.
pub const NO_PROJECT_DIR: &str = "_sans-projet";
const MANIFEST: &str = "index.json";
const MESSAGE_MARK: &str = "<!-- message ";
const THREAD_MARK: &str = "<!-- thread ";

/// `MEMORYPILOT_CURSOR_ARCHIVE`, else `~/Archives/cursor-conversations`.
pub fn default_dir() -> Option<PathBuf> {
    match std::env::var("MEMORYPILOT_CURSOR_ARCHIVE") {
        Ok(path) if !path.trim().is_empty() => Some(PathBuf::from(path)),
        _ => dirs::home_dir().map(|home| home.join("Archives/cursor-conversations")),
    }
}

pub struct ArchiveOptions<'a> {
    /// Cursor's `globalStorage/state.vscdb`.
    pub store: &'a Path,
    pub out_dir: &'a Path,
    /// Without it nothing is written: the report only counts.
    pub apply: bool,
    /// The user's home, to recognise the folders named in tool calls.
    pub home: Option<String>,
    /// The known project of a folder (registered path, alias, Git remote).
    pub known_project: &'a dyn Fn(&str) -> Option<String>,
    /// The Git remote recorded for a project.
    pub remote_of: &'a dyn Fn(&str) -> Option<String>,
    /// The canonical spelling of a project name.
    pub slug: &'a dyn Fn(&str) -> Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Unreadable {
    pub id: String,
    pub reason: String,
}

/// The project and folder found for one conversation.
#[derive(Debug, Clone)]
pub struct Identity {
    pub conversation_id: String,
    pub project: Option<String>,
    pub workspace: Option<String>,
}

#[derive(Debug, Default, Serialize)]
pub struct ArchiveReport {
    pub applied: bool,
    pub out_dir: String,
    /// `composerData` rows of the store.
    pub composers: usize,
    /// Composers holding no message: tabs opened and never used.
    pub empty_composers: usize,
    /// Sub-agent composers filed in their parent conversation.
    pub sub_agents: usize,
    pub conversations: usize,
    pub written: usize,
    pub unchanged: usize,
    pub without_project: usize,
    pub user_messages: usize,
    pub assistant_messages: usize,
    pub bubbles_read: usize,
    pub unreadable_bubbles: usize,
    pub unreadable: Vec<Unreadable>,
    /// Size of the Markdown files, written or not.
    pub bytes: u64,
    pub secrets_masked: usize,
    pub secrets_by_kind: BTreeMap<String, usize>,
    /// How each conversation's project was found.
    pub project_found_by: BTreeMap<String, usize>,
    pub conversations_by_project: BTreeMap<String, usize>,
    #[serde(skip)]
    pub identities: Vec<Identity>,
}

struct Rendered {
    id: String,
    relative_path: PathBuf,
    content: String,
}

/// Reads every conversation of the Cursor store and writes the archive.
pub fn archive_cursor(options: &ArchiveOptions) -> Result<ArchiveReport, String> {
    let conn = cursor::open_immutable(options.store)?;
    let composers = cursor::list_composers(&conn)?;
    let window_folders = cursor::workspace_folders(options.store);
    let mut report = ArchiveReport {
        applied: options.apply,
        out_dir: options.out_dir.to_string_lossy().into_owned(),
        composers: composers.len(),
        ..Default::default()
    };

    // Composers of one conversation, the one the user had first.
    let mut groups: BTreeMap<String, Vec<&cursor::Composer>> = BTreeMap::new();
    for composer in &composers {
        groups.entry(composer.conversation.root_id.clone()).or_default().push(composer);
    }
    let mut resolver = ProjectResolver::new(options);
    let mut rendered: Vec<Rendered> = Vec::new();

    for (root_id, mut members) in groups {
        members.sort_by_key(|composer| (composer.conversation.origin_id != root_id, composer.conversation.created_at.clone()));
        let mut threads = Vec::with_capacity(members.len());
        let mut failed = false;
        for composer in &members {
            match cursor::read_thread(&conn, composer) {
                Ok(thread) => {
                    report.bubbles_read += thread.bubbles;
                    report.unreadable_bubbles += thread.unreadable;
                    threads.push(thread);
                }
                Err(error) => {
                    report.unreadable.push(Unreadable { id: composer.conversation.origin_id.clone(), reason: error });
                    failed = true;
                    break;
                }
            }
        }
        if failed {
            continue;
        }
        let said: usize = threads.iter().map(|thread| thread.messages.len()).sum();
        if said == 0 {
            let unreadable: usize = threads.iter().map(|thread| thread.unreadable).sum();
            if unreadable > 0 {
                report.unreadable.push(Unreadable {
                    id: root_id.clone(),
                    reason: format!("{} bulle(s) au format JSON invalide, aucun message lisible", unreadable),
                });
            }
            report.empty_composers += members.len();
            continue;
        }
        report.conversations += 1;
        report.sub_agents += members.iter().filter(|composer| composer.conversation.origin_id != root_id).count();

        let found = resolver.resolve(&members, &threads, &window_folders);
        *report.project_found_by.entry(found.method.to_string()).or_default() += 1;
        match &found.project {
            Some(project) => *report.conversations_by_project.entry(project.clone()).or_default() += 1,
            None => report.without_project += 1,
        }
        report.identities.push(Identity {
            conversation_id: root_id.clone(),
            project: found.project.clone(),
            workspace: found.workspace.clone(),
        });
        rendered.push(render(&root_id, &members, &threads, &found, options, &mut report));
    }

    report.bytes = rendered.iter().map(|file| file.content.len() as u64).sum();
    report.secrets_masked = report.secrets_by_kind.values().sum();
    if options.apply {
        write_files(options.out_dir, &rendered, &mut report)?;
    }
    Ok(report)
}

struct Found {
    project: Option<String>,
    workspace: Option<String>,
    method: &'static str,
}

/// Finds the project of a conversation. A folder Cursor recorded is taken
/// as it is; folders only named in tool calls or in the text count when
/// they belong to a project already known, never to make one up.
struct ProjectResolver<'a> {
    options: &'a ArchiveOptions<'a>,
    cache: HashMap<String, Option<String>>,
}

impl<'a> ProjectResolver<'a> {
    fn new(options: &'a ArchiveOptions<'a>) -> Self {
        Self { options, cache: HashMap::new() }
    }

    fn known(&mut self, folder: &str) -> Option<String> {
        if let Some(found) = self.cache.get(folder) {
            return found.clone();
        }
        let found = (self.options.known_project)(folder);
        self.cache.insert(folder.to_string(), found.clone());
        found
    }

    fn resolve(
        &mut self,
        members: &[&cursor::Composer],
        threads: &[cursor::Thread],
        window_folders: &HashMap<String, String>,
    ) -> Found {
        let recorded = members
            .first()
            .and_then(|root| root.conversation.workspace.clone())
            .map(|folder| (folder, "dossier de la conversation"))
            .or_else(|| {
                members
                    .iter()
                    .find_map(|composer| window_folders.get(&composer.conversation.origin_id).cloned())
                    .map(|folder| (folder, "fenêtre Cursor"))
            })
            .or_else(|| {
                members
                    .iter()
                    .skip(1)
                    .find_map(|composer| composer.conversation.workspace.clone())
                    .map(|folder| (folder, "dossier d'un sous-agent"))
            });
        if let Some((folder, method)) = recorded {
            let project = self
                .known(&folder)
                .or_else(|| (self.options.slug)(&super::store::project_name(&folder)));
            return Found { project, workspace: Some(folder), method };
        }
        let Some(home) = self.options.home.clone() else {
            return Found { project: None, workspace: None, method: "introuvable" };
        };
        let from_tools = threads.iter().flat_map(|thread| thread.tool_arguments.iter().map(String::as_str));
        if let Some(project) = self.most_named(from_tools, &home) {
            return Found { project: Some(project), workspace: None, method: "chemins des appels d'outils" };
        }
        let from_text = threads.iter().flat_map(|thread| thread.messages.iter().map(|message| message.text.as_str()));
        if let Some(project) = self.most_named(from_text, &home) {
            return Found { project: Some(project), workspace: None, method: "chemins cités dans le texte" };
        }
        Found { project: None, workspace: None, method: "introuvable" }
    }

    /// The known project most folders named in `texts` belong to.
    fn most_named<'t>(&mut self, texts: impl Iterator<Item = &'t str>, home: &str) -> Option<String> {
        let mut folders: HashMap<String, usize> = HashMap::new();
        for text in texts {
            for folder in folders_named(text, home) {
                *folders.entry(folder).or_default() += 1;
            }
        }
        let mut votes: BTreeMap<String, usize> = BTreeMap::new();
        for (folder, count) in folders {
            if let Some(project) = self.known(&folder) {
                *votes.entry(project).or_default() += count;
            }
        }
        votes.into_iter().max_by_key(|(name, count)| (*count, std::cmp::Reverse(name.clone()))).map(|(name, _)| name)
    }
}

/// Folders under the user's home named in a text (`/Users/me/…`, `~/…`).
fn folders_named(text: &str, home: &str) -> Vec<String> {
    let home = home.trim_end_matches('/');
    let mut folders = Vec::new();
    for (prefix, absolute) in [(format!("{}/", home), true), ("~/".to_string(), false)] {
        let mut rest = text;
        while let Some(index) = rest.find(&prefix) {
            let tail = &rest[index + prefix.len()..];
            let end = tail
                .find(|c: char| c.is_whitespace() || matches!(c, '"' | '\'' | '\\' | ',' | ')' | ']' | '}' | '>' | '`' | ':' | ';' | '*'))
                .unwrap_or(tail.len());
            let mut path = tail[..end].trim_end_matches(['/', '.']).to_string();
            // `…/src/main.rs` names its folder; `…/MyProject` is one.
            if path.rsplit('/').next().is_some_and(|last| last.contains('.')) {
                path = path.rsplit_once('/').map(|(parent, _)| parent.to_string()).unwrap_or_default();
            }
            if !path.is_empty() && (absolute || index == 0 || !rest[..index].ends_with(|c: char| c.is_alphanumeric())) {
                folders.push(format!("{}/{}", home, path));
            }
            rest = &tail[end..];
        }
    }
    folders
}

fn redact_counted(text: &str, report: &mut ArchiveReport) -> String {
    for finding in secrets::findings(text) {
        *report.secrets_by_kind.entry(finding.kind.to_string()).or_default() += 1;
    }
    secrets::redact(text).into_owned()
}

/// A line of a message that looks like one of the archive's own markers.
fn escape_marks(text: &str) -> String {
    text.lines()
        .map(|line| {
            if line.starts_with(MESSAGE_MARK) || line.starts_with(THREAD_MARK) || line.starts_with('\\') {
                format!("\\{}", line)
            } else {
                line.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn unescape_marks(line: &str) -> &str {
    line.strip_prefix('\\').unwrap_or(line)
}

fn quoted(value: Option<&str>) -> String {
    match value {
        Some(value) => serde_json::Value::String(value.to_string()).to_string(),
        None => "null".to_string(),
    }
}

fn day_and_minute(at: Option<&str>) -> String {
    at.and_then(|at| at.get(..16)).map(|stamp| stamp.replace('T', " ")).unwrap_or_else(|| "date inconnue".to_string())
}

fn file_slug(title: &str) -> String {
    let mut slug = String::new();
    for character in title.chars() {
        if character.is_alphanumeric() {
            slug.extend(character.to_lowercase());
        } else if !slug.is_empty() && !slug.ends_with('-') {
            slug.push('-');
        }
        if slug.chars().count() >= 60 {
            break;
        }
    }
    slug.trim_end_matches('-').to_string()
}

fn render(
    root_id: &str,
    members: &[&cursor::Composer],
    threads: &[cursor::Thread],
    found: &Found,
    options: &ArchiveOptions,
    report: &mut ArchiveReport,
) -> Rendered {
    let root = &members[0].conversation;
    let title = root
        .title
        .as_deref()
        .or_else(|| members.iter().find_map(|composer| composer.conversation.title.as_deref()))
        .map(|title| redact_counted(title.trim(), report));
    let created_at = members.iter().filter_map(|composer| composer.conversation.created_at.clone()).min();
    let updated_at = threads
        .iter()
        .flat_map(|thread| thread.messages.iter().filter_map(|message| message.at.clone()))
        .max();
    let remote = found.project.as_deref().and_then(|project| (options.remote_of)(project));

    let mut body = String::new();
    let mut messages = 0;
    for (composer, thread) in members.iter().zip(threads) {
        let source = &composer.conversation;
        let is_main = source.origin_id == root_id;
        let mut section = String::new();
        for message in &thread.messages {
            let (role, speaker, said) = match message.role {
                Role::User => ("user", "Moi", text::typed_text(&message.text)),
                Role::Assistant => ("assistant", "Cursor", message.text.trim().to_string()),
            };
            if said.is_empty() {
                continue;
            }
            let said = redact_counted(&said, report);
            match message.role {
                Role::User => report.user_messages += 1,
                Role::Assistant => report.assistant_messages += 1,
            }
            messages += 1;
            section.push_str(&format!(
                "{}id={} role={} at={} -->\n**{}** · {}\n\n{}\n\n",
                MESSAGE_MARK,
                message.key,
                role,
                message.at.as_deref().unwrap_or("-"),
                speaker,
                day_and_minute(message.at.as_deref()),
                escape_marks(&said)
            ));
        }
        if section.is_empty() {
            continue;
        }
        body.push_str(&format!(
            "{}id={} main={} version={} -->\n",
            THREAD_MARK,
            source.origin_id,
            is_main as u8,
            source.version
        ));
        if !is_main {
            let name = source.title.as_deref().map(|title| redact_counted(title.trim(), report));
            body.push_str(&format!("## Sous-agent : {}\n\n", name.as_deref().unwrap_or("sans titre")));
        }
        body.push_str(&section);
    }

    let mut content = String::from("---\n");
    for (key, value) in [
        ("id", quoted(Some(root_id))),
        ("tool", quoted(Some("cursor"))),
        ("title", quoted(title.as_deref())),
        ("created_at", quoted(created_at.as_deref())),
        ("updated_at", quoted(updated_at.as_deref())),
        ("version", quoted(Some(&root.version))),
        ("workspace", quoted(found.workspace.as_deref())),
        ("project", quoted(found.project.as_deref())),
        ("project_found_by", quoted(Some(found.method))),
        ("remote", quoted(remote.as_deref())),
        ("messages", messages.to_string()),
        ("sub_agents", (members.len() - 1).to_string()),
    ] {
        content.push_str(&format!("{}: {}\n", key, value));
    }
    content.push_str(&format!("---\n\n# {}\n\n", title.as_deref().unwrap_or("Conversation sans titre")));
    content.push_str(&body);

    let day = created_at.as_deref().and_then(|at| at.get(..10)).unwrap_or("sans-date");
    let slug = title.as_deref().map(file_slug).filter(|slug| !slug.is_empty()).unwrap_or_else(|| "sans-titre".to_string());
    let short_id: String = root_id.chars().take(8).collect();
    let folder = found.project.as_deref().unwrap_or(NO_PROJECT_DIR);
    Rendered {
        id: root_id.to_string(),
        relative_path: Path::new(folder).join(format!("{}-{}-{}.md", day, slug, short_id)),
        content,
    }
}

fn private_dir(path: &Path) -> Result<(), String> {
    std::fs::create_dir_all(path).map_err(|error| format!("Cannot create {}: {}", path.display(), error))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))
            .map_err(|error| format!("{}: {}", path.display(), error))?;
    }
    Ok(())
}

fn private_write(path: &Path, content: &str) -> Result<(), String> {
    let temporary = path.with_extension("tmp");
    std::fs::write(&temporary, content).map_err(|error| format!("Cannot write {}: {}", temporary.display(), error))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&temporary, std::fs::Permissions::from_mode(0o600))
            .map_err(|error| format!("{}: {}", temporary.display(), error))?;
    }
    std::fs::rename(&temporary, path).map_err(|error| format!("Cannot write {}: {}", path.display(), error))
}

/// Writes the files that changed. The manifest (`index.json`: conversation
/// id → file) is what keeps a second run, or a conversation that changed
/// title or project, from leaving a second copy.
fn write_files(out_dir: &Path, rendered: &[Rendered], report: &mut ArchiveReport) -> Result<(), String> {
    private_dir(out_dir)?;
    let manifest_path = out_dir.join(MANIFEST);
    let mut manifest: BTreeMap<String, String> = std::fs::read_to_string(&manifest_path)
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
        .unwrap_or_default();
    for file in rendered {
        let relative = file.relative_path.to_string_lossy().into_owned();
        let target = out_dir.join(&file.relative_path);
        let previous = manifest.get(&file.id).cloned();
        if previous.as_deref() == Some(relative.as_str())
            && std::fs::read_to_string(&target).map(|current| current == file.content).unwrap_or(false)
        {
            report.unchanged += 1;
            continue;
        }
        if let Some(parent) = target.parent() {
            private_dir(parent)?;
        }
        private_write(&target, &file.content)?;
        if let Some(previous) = previous.filter(|previous| *previous != relative) {
            // The archive's own earlier copy, under its old title or project.
            let _ = std::fs::remove_file(out_dir.join(previous));
        }
        manifest.insert(file.id.clone(), relative);
        report.written += 1;
    }
    let manifest_text = serde_json::to_string_pretty(&manifest).map_err(|error| format!("Manifest: {}", error))?;
    private_write(&manifest_path, &manifest_text)
}

/// One archived conversation, as the index needs it back.
pub struct ArchivedConversation {
    pub id: String,
    pub project: Option<String>,
    pub workspace: Option<String>,
    /// One per composer, the main one first: its source and its messages.
    pub threads: Vec<(SourceConversation, Vec<Message>)>,
}

/// Reads every Markdown file of an archive folder.
pub fn read_archive(dir: &Path) -> (Vec<ArchivedConversation>, Vec<String>) {
    let mut conversations = Vec::new();
    let mut errors = Vec::new();
    let mut files: Vec<PathBuf> = Vec::new();
    for entry in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        if entry.path().is_dir() {
            files.extend(
                std::fs::read_dir(entry.path())
                    .into_iter()
                    .flatten()
                    .flatten()
                    .map(|file| file.path())
                    .filter(|path| path.extension().is_some_and(|extension| extension == "md")),
            );
        }
    }
    files.sort();
    for path in files {
        match std::fs::read_to_string(&path).map_err(|error| error.to_string()).and_then(|text| parse(&text, &path)) {
            Ok(conversation) => conversations.push(conversation),
            Err(error) => errors.push(format!("{}: {}", path.display(), error)),
        }
    }
    (conversations, errors)
}

fn attribute<'t>(line: &'t str, name: &str) -> Option<&'t str> {
    let start = line.find(&format!(" {}=", name))? + name.len() + 2;
    let rest = &line[start..];
    Some(&rest[..rest.find(' ').unwrap_or(rest.len())])
}

fn parse(content: &str, path: &Path) -> Result<ArchivedConversation, String> {
    let rest = content.strip_prefix("---\n").ok_or("no front matter")?;
    let (front, body) = rest.split_once("\n---\n").ok_or("front matter not closed")?;
    let mut fields: HashMap<&str, Option<String>> = HashMap::new();
    for line in front.lines() {
        if let Some((key, value)) = line.split_once(": ") {
            let value = serde_json::from_str::<serde_json::Value>(value).ok();
            fields.insert(key, value.and_then(|value| value.as_str().map(str::to_string)));
        }
    }
    let field = |name: &str| fields.get(name).cloned().flatten();
    let id = field("id").ok_or("no conversation id")?;

    let mut threads: Vec<(SourceConversation, Vec<Message>)> = Vec::new();
    let mut open: Option<Message> = None;
    let mut skip_label = false;
    let close = |open: &mut Option<Message>, threads: &mut Vec<(SourceConversation, Vec<Message>)>| {
        if let (Some(mut message), Some((_, messages))) = (open.take(), threads.last_mut()) {
            message.text = message.text.trim().to_string();
            messages.push(message);
        }
    };
    for line in body.lines() {
        if line.starts_with(THREAD_MARK) {
            close(&mut open, &mut threads);
            let origin_id = attribute(line, "id").ok_or("thread without id")?.to_string();
            let is_main = attribute(line, "main") == Some("1");
            threads.push((
                SourceConversation {
                    origin_id,
                    root_id: id.clone(),
                    tool: "cursor",
                    title: if is_main { field("title") } else { None },
                    workspace: field("workspace"),
                    created_at: field("created_at"),
                    version: attribute(line, "version").unwrap_or("0").to_string(),
                    source_path: Some(path.to_string_lossy().into_owned()),
                },
                Vec::new(),
            ));
        } else if line.starts_with(MESSAGE_MARK) {
            close(&mut open, &mut threads);
            let role = match attribute(line, "role") {
                Some("user") => Role::User,
                Some("assistant") => Role::Assistant,
                other => return Err(format!("unknown role {:?}", other)),
            };
            open = Some(Message {
                key: attribute(line, "id").ok_or("message without id")?.to_string(),
                role,
                at: attribute(line, "at").filter(|at| *at != "-").map(str::to_string),
                order: Some(threads.last().map(|(_, messages)| messages.len()).unwrap_or(0)),
                text: String::new(),
            });
            skip_label = true;
        } else if let Some(message) = open.as_mut() {
            if std::mem::take(&mut skip_label) && line.starts_with("**") {
                continue;
            }
            message.text.push_str(unescape_marks(line));
            message.text.push('\n');
        }
    }
    close(&mut open, &mut threads);
    Ok(ArchivedConversation { id, project: field("project"), workspace: field("workspace"), threads })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::{params, Connection};
    use serde_json::json;

    struct Fixture {
        root: PathBuf,
    }

    impl Fixture {
        fn new(name: &str) -> Self {
            let root = std::env::temp_dir().join(format!("memory-pilot-archive-{name}-{}", uuid::Uuid::new_v4()));
            std::fs::create_dir_all(root.join("User/globalStorage")).expect("fixture");
            Self { root: std::fs::canonicalize(root).expect("canonical") }
        }

        fn store(&self) -> PathBuf {
            self.root.join("User/globalStorage/state.vscdb")
        }

        fn out(&self) -> PathBuf {
            self.root.join("archive")
        }

        fn home(&self) -> String {
            self.root.join("home").to_string_lossy().into_owned()
        }

        fn connection(&self) -> Connection {
            let conn = Connection::open(self.store()).expect("store");
            conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS cursorDiskKV (key TEXT UNIQUE ON CONFLICT REPLACE, value BLOB);
                 CREATE TABLE IF NOT EXISTS ItemTable (key TEXT UNIQUE ON CONFLICT REPLACE, value BLOB);",
            )
            .expect("schema");
            conn
        }

        fn composer(&self, id: &str, extra: serde_json::Value) {
            let mut data = json!({ "composerId": id, "createdAt": 1_767_348_000_000_i64, "lastUpdatedAt": 1_767_348_900_000_i64 });
            for (key, value) in extra.as_object().expect("object") {
                data[key] = value.clone();
            }
            self.connection()
                .execute("INSERT INTO cursorDiskKV VALUES (?1, ?2)", params![format!("composerData:{id}"), data.to_string()])
                .expect("composer");
        }

        fn bubble(&self, composer: &str, id: &str, minute: u32, bubble: serde_json::Value) {
            let mut data = bubble;
            data["bubbleId"] = json!(id);
            data["createdAt"] = json!(format!("2026-01-02T10:{minute:02}:00.000Z"));
            self.connection()
                .execute("INSERT INTO cursorDiskKV VALUES (?1, ?2)", params![format!("bubbleId:{composer}:{id}"), data.to_string()])
                .expect("bubble");
        }

        fn run(&self, apply: bool) -> ArchiveReport {
            let home = self.home();
            let known = move |folder: &str| -> Option<String> {
                folder.starts_with(&format!("{}/App/Widget", home)).then(|| "widget".to_string())
            };
            archive_cursor(&ArchiveOptions {
                store: &self.store(),
                out_dir: &self.out(),
                apply,
                home: Some(self.home()),
                known_project: &known,
                remote_of: &|project| (project == "widget").then(|| "github.com/acme/widget".to_string()),
                slug: &|name| Some(name.to_lowercase()),
            })
            .expect("archive")
        }

        fn files(&self) -> Vec<PathBuf> {
            let mut files = Vec::new();
            for folder in std::fs::read_dir(self.out()).into_iter().flatten().flatten() {
                if folder.path().is_dir() {
                    files.extend(std::fs::read_dir(folder.path()).expect("folder").flatten().map(|file| file.path()));
                }
            }
            files.sort();
            files
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.root);
        }
    }

    /// A value the filter knows as a GitHub token, built at run time so
    /// that no token-shaped string sits in the sources.
    fn fake_token() -> String {
        let mut state: u64 = 0x2545_f491_4f6c_dd1d;
        let alphabet: Vec<char> = ('a'..='z').chain('A'..='Z').chain('0'..='9').collect();
        let tail: String = (0..36)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                alphabet[(state >> 33) as usize % alphabet.len()]
            })
            .collect();
        format!("{}{}", ["gh", "p_"].concat(), tail)
    }

    /// One conversation with a folder, a tool call, a reasoning bubble and
    /// a sub-agent; one conversation known only by its tool paths; one tab
    /// never used.
    fn seed(fixture: &Fixture) -> String {
        let home = fixture.home();
        let token = fake_token();
        fixture.composer(
            "main-0001",
            json!({ "name": "Sync des dossiers IMAP", "subagentComposerIds": ["sub-0001"],
                    "workspaceIdentifier": { "uri": { "fsPath": format!("{home}/App/Widget") } } }),
        );
        fixture.bubble("main-0001", "b1", 1, json!({ "type": 1, "text": "<user_query>\nPourquoi le second dossier ne se synchronise pas ?\n</user_query>" }));
        fixture.bubble("main-0001", "b2", 2, json!({ "type": 2, "text": "", "thinking": { "text": "RAISONNEMENT INTERNE" } }));
        fixture.bubble(
            "main-0001",
            "b3",
            3,
            json!({ "type": 2, "text": "", "toolFormerData": { "name": "read_file",
                    "params": format!("{{\"targetFile\":\"{home}/App/Widget/src/sync.rs\"}}"),
                    "result": "CONTENU DU FICHIER" } }),
        );
        fixture.bubble("main-0001", "b4", 4, json!({ "type": 2, "text": format!("Je regarde le fichier de synchronisation, jeton utilisé : {token}") }));
        fixture.bubble("main-0001", "b5", 5, json!({ "type": 2, "text": "Le second dossier est ignoré parce que IDLE n'est relancé qu'une fois.\n<!-- message faux -->" }));
        fixture.composer("sub-0001", json!({ "name": "Explorer le module IMAP" }));
        fixture.bubble("sub-0001", "s1", 3, json!({ "type": 1, "text": "Trouve où IDLE est relancé." }));
        fixture.bubble("sub-0001", "s2", 4, json!({ "type": 2, "text": "IDLE est relancé dans sync.rs, une seule fois." }));

        fixture.composer("lost-0002", json!({ "name": "Refonte du tri" }));
        fixture.bubble("lost-0002", "b1", 1, json!({ "type": 1, "text": "Change l'ordre de tri des comptes." }));
        fixture.bubble(
            "lost-0002",
            "b2",
            2,
            json!({ "type": 2, "text": "", "toolFormerData": { "rawArgs": format!("{{\"path\":\"{home}/App/Widget/src/sort.rs\"}}") } }),
        );
        fixture.bubble("lost-0002", "b3", 3, json!({ "type": 2, "text": "Les comptes sont maintenant triés par nombre de non lus." }));

        fixture.composer("orphan-0003", json!({ "name": "Question générale" }));
        fixture.bubble("orphan-0003", "b1", 1, json!({ "type": 1, "text": "Quelle différence entre IDLE et NOOP ?" }));
        fixture.composer("empty-0004", json!({}));
        token
    }

    #[test]
    fn a_dry_run_counts_and_writes_nothing() {
        let fixture = Fixture::new("dry-run");
        seed(&fixture);

        let report = fixture.run(false);

        assert_eq!((report.composers, report.conversations, report.empty_composers, report.sub_agents), (5, 3, 1, 1));
        assert_eq!((report.user_messages, report.assistant_messages), (4, 4));
        assert_eq!(report.without_project, 1);
        assert_eq!(report.secrets_by_kind.get("github_token"), Some(&1));
        assert!(!fixture.out().exists(), "a dry run creates nothing");
    }

    #[test]
    fn the_archive_keeps_what_was_said_and_nothing_else() {
        let fixture = Fixture::new("content");
        let token = seed(&fixture);
        let store_before = std::fs::read(fixture.store()).expect("store bytes");

        let report = fixture.run(true);

        assert_eq!((report.written, report.unchanged), (3, 0));
        let files = fixture.files();
        let names: Vec<String> = files
            .iter()
            .map(|path| path.strip_prefix(fixture.out()).expect("relative").to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            names,
            vec![
                "_sans-projet/2026-01-02-question-générale-orphan-0.md",
                "widget/2026-01-02-refonte-du-tri-lost-000.md",
                "widget/2026-01-02-sync-des-dossiers-imap-main-000.md",
            ]
        );
        let main = std::fs::read_to_string(&files[2]).expect("main file");
        assert!(main.contains("project: \"widget\"") && main.contains("remote: \"github.com/acme/widget\""));
        assert!(main.contains("project_found_by: \"dossier de la conversation\""));
        assert!(main.contains("Pourquoi le second dossier ne se synchronise pas ?"));
        assert!(!main.contains("user_query"), "the harness wrapper is dropped");
        assert!(main.contains("Je regarde le fichier de synchronisation"), "progress notes are kept");
        assert!(main.contains("## Sous-agent : Explorer le module IMAP") && main.contains("IDLE est relancé dans sync.rs"));
        for absent in ["CONTENU DU FICHIER", "RAISONNEMENT INTERNE", "targetFile", token.as_str()] {
            assert!(!main.contains(absent), "{absent} must not be archived");
        }
        assert!(main.contains("[secret masqué : github_token "));
        let lost = std::fs::read_to_string(&files[1]).expect("lost file");
        assert!(lost.contains("project_found_by: \"chemins des appels d'outils\""));
        assert_eq!(report.project_found_by.get("introuvable"), Some(&1));

        // Cursor's store is byte for byte what it was.
        assert_eq!(std::fs::read(fixture.store()).expect("store bytes"), store_before);
        assert!(!fixture.store().with_extension("vscdb-wal").exists() && !fixture.root.join("User/globalStorage/state.vscdb-shm").exists());
    }

    #[cfg(unix)]
    #[test]
    fn archive_files_are_readable_by_their_owner_only() {
        use std::os::unix::fs::PermissionsExt;
        let fixture = Fixture::new("permissions");
        seed(&fixture);
        fixture.run(true);
        let mode = |path: &Path| std::fs::metadata(path).expect("metadata").permissions().mode() & 0o777;
        assert_eq!(mode(&fixture.out()), 0o700);
        for file in fixture.files() {
            assert_eq!(mode(&file), 0o600);
            assert_eq!(mode(file.parent().expect("folder")), 0o700);
        }
    }

    #[test]
    fn a_second_run_writes_nothing_and_a_renamed_conversation_leaves_no_copy() {
        let fixture = Fixture::new("idempotent");
        seed(&fixture);
        fixture.run(true);

        let again = fixture.run(true);
        assert_eq!((again.written, again.unchanged), (0, 3));
        assert_eq!(fixture.files().len(), 3);

        // Cursor renamed the conversation and it got one more message.
        fixture.composer(
            "orphan-0003",
            json!({ "name": "IDLE contre NOOP", "workspaceIdentifier": { "uri": { "fsPath": format!("{}/App/Widget", fixture.home()) } } }),
        );
        fixture.bubble("orphan-0003", "b2", 2, json!({ "type": 2, "text": "IDLE attend les nouveautés, NOOP garde la connexion ouverte." }));
        let third = fixture.run(true);

        assert_eq!((third.written, third.unchanged), (1, 2));
        let names: Vec<String> = fixture.files().iter().map(|path| path.file_name().expect("name").to_string_lossy().into_owned()).collect();
        assert_eq!(names.len(), 3, "{names:?}");
        assert!(names.contains(&"2026-01-02-idle-contre-noop-orphan-0.md".to_string()));
        assert!(!fixture.out().join(NO_PROJECT_DIR).join("2026-01-02-question-générale-orphan-0.md").exists());
    }

    #[test]
    fn the_archive_reads_back_message_for_message() {
        let fixture = Fixture::new("round-trip");
        seed(&fixture);
        fixture.run(true);

        let (conversations, errors) = read_archive(&fixture.out());

        assert!(errors.is_empty(), "{errors:?}");
        assert_eq!(conversations.len(), 3);
        let main = conversations.iter().find(|conversation| conversation.id == "main-0001").expect("main");
        assert_eq!(main.project.as_deref(), Some("widget"));
        assert_eq!(main.threads.len(), 2);
        let (source, messages) = &main.threads[0];
        assert_eq!((source.origin_id.as_str(), source.root_id.as_str(), source.tool), ("main-0001", "main-0001", "cursor"));
        assert_eq!(source.title.as_deref(), Some("Sync des dossiers IMAP"));
        let keys: Vec<&str> = messages.iter().map(|message| message.key.as_str()).collect();
        assert_eq!(keys, vec!["b1", "b4", "b5"]);
        assert_eq!(messages[0].text, "Pourquoi le second dossier ne se synchronise pas ?");
        assert_eq!(messages[0].at.as_deref(), Some("2026-01-02T10:01:00.000Z"));
        assert!(messages[2].text.ends_with("<!-- message faux -->"), "a marker-like line of a message survives");
        let (sub, sub_messages) = &main.threads[1];
        assert_eq!((sub.origin_id.as_str(), sub.root_id.as_str(), sub.title.as_deref()), ("sub-0001", "main-0001", None));
        assert_eq!(sub_messages.len(), 2);
    }

    #[test]
    fn the_index_is_rebuilt_from_the_archive_under_the_same_keys() {
        let fixture = Fixture::new("index");
        seed(&fixture);
        fixture.run(true);
        let mut store = crate::conversations::ConversationStore::open(&fixture.root.join("conversations.db")).expect("index");
        let never = |_: &mut crate::conversations::SyncReport| false;

        let mut first = crate::conversations::SyncReport::default();
        crate::conversations::sync_archive(&mut store, &fixture.out(), &mut first, &never);

        assert!(first.errors.is_empty(), "{:?}", first.errors);
        assert_eq!(first.origins_updated, 4, "three conversations, one sub-agent");
        let row = |sql: &str| -> (String, i64) { store.conn.query_row(sql, [], |row| Ok((row.get(0)?, row.get(1)?))).expect("row") };
        assert_eq!(row("SELECT project, message_count FROM conversations WHERE id = 'main-0001'"), ("widget".to_string(), 4));
        assert_eq!(row("SELECT tool, 1 FROM conversations WHERE id = 'lost-0002'").0, "cursor");
        // The key the live Cursor store gives the same message.
        assert_eq!(row("SELECT role, 1 FROM segments WHERE source_key = 'cursor:main-0001:b1'").0, "user");
        assert_eq!(row("SELECT conversation_id, 1 FROM segments WHERE source_key = 'cursor:sub-0001:s1'").0, "main-0001");

        let mut second = crate::conversations::SyncReport::default();
        crate::conversations::sync_archive(&mut store, &fixture.out(), &mut second, &never);
        assert_eq!((second.origins_updated, second.messages_written), (0, 0));
    }

    #[test]
    fn a_conversation_without_folder_takes_the_one_of_its_cursor_window() {
        let fixture = Fixture::new("window");
        seed(&fixture);
        // The window state Cursor keeps per folder lists the conversation.
        let window = fixture.root.join("User/workspaceStorage/0123abcd");
        std::fs::create_dir_all(&window).expect("window");
        let folder = format!("{}/App/Mail Client", fixture.home());
        std::fs::write(window.join("workspace.json"), json!({ "folder": format!("file://{}", folder.replace(' ', "%20")) }).to_string())
            .expect("workspace.json");
        let state = Connection::open(window.join("state.vscdb")).expect("window state");
        state
            .execute_batch("CREATE TABLE ItemTable (key TEXT UNIQUE ON CONFLICT REPLACE, value BLOB);")
            .expect("schema");
        state
            .execute(
                "INSERT INTO ItemTable VALUES ('composer.composerData', ?1)",
                params![json!({ "allComposers": [{ "composerId": "orphan-0003" }], "selectedComposerIds": [] }).to_string()],
            )
            .expect("composer list");
        drop(state);

        let report = fixture.run(true);

        assert_eq!(report.without_project, 0);
        assert_eq!(report.project_found_by.get("fenêtre Cursor"), Some(&1));
        let file = fixture.out().join("mail client/2026-01-02-question-générale-orphan-0.md");
        let content = std::fs::read_to_string(&file).expect("file filed under the folder's name");
        assert!(content.contains(&format!("workspace: \"{}\"", folder)));
    }

    #[test]
    fn folders_are_found_in_tool_arguments_and_in_prose() {
        let home = "/Users/someone";
        assert_eq!(
            folders_named("{\"targetFile\":\"/Users/someone/App/Widget/src/main.rs\"}", home),
            vec!["/Users/someone/App/Widget/src"]
        );
        assert_eq!(folders_named("regarde dans ~/App/Widget, puis lance", home), vec!["/Users/someone/App/Widget"]);
        assert!(folders_named("rien ici, ni dans /etc/hosts", home).is_empty());
    }
}
