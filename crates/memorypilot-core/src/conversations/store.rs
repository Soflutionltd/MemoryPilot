//! The conversation index: its own SQLite file next to `memory.db`.
//! Nothing here is a memory. Conversations are indexed as they were
//! said, so nothing is distilled and nothing can pollute recall; deleting
//! the file loses no information (it is rebuilt from the sources).

use super::text::{self, Reference};
use super::{Message, Role, SourceConversation};
use crate::secrets;
use rusqlite::{params, Connection, OptionalExtension};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub const FILE_NAME: &str = "conversations.db";

/// Messages shorter than this carry no recall value ("ok", "vas-y").
const MIN_USER_CHARS: usize = 12;
/// Agent messages shorter than this are progress narration.
const MIN_ASSISTANT_CHARS: usize = 40;
/// Agent messages are only embedded from this length: below it they are
/// "I'm checking X" notes, which the lexical lanes cover.
const EMBED_ASSISTANT_CHARS: usize = 280;
/// What the agent wrote before its final answer is kept from this
/// length on: a draft, a plan or a message it prepared, not "checking X".
const MIN_STEP_CHARS: usize = 200;
/// Only a long intermediate text is worth a vector.
const EMBED_STEP_CHARS: usize = 600;
/// A long text is stored whole, in parts of `text::MAX_*_CHARS`, up to
/// this many: past it, it is a pasted file rather than something said.
const MAX_PARTS: usize = 6;
pub fn default_path() -> Result<PathBuf, String> {
    let path = crate::layout::conversations_db().ok_or("Cannot find home directory")?;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|error| format!("Cannot create dir: {}", error))?;
    }
    Ok(path)
}

pub struct ConversationStore {
    pub(super) conn: Connection,
}

impl ConversationStore {
    pub fn open(path: &Path) -> Result<Self, String> {
        let conn = Connection::open(path).map_err(|error| format!("Conversations open: {}", error))?;
        conn.execute_batch(
            "
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = NORMAL;
            PRAGMA foreign_keys = OFF;
            PRAGMA cache_size = -16000;
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                tool TEXT NOT NULL,
                title TEXT,
                workspace TEXT,
                project TEXT,
                created_at TEXT,
                updated_at TEXT,
                source_path TEXT,
                message_count INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS origins (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                is_main INTEGER NOT NULL,
                version TEXT,
                seen BLOB
            );
            CREATE TABLE IF NOT EXISTS segments (
                id INTEGER PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                origin_id TEXT NOT NULL,
                role TEXT NOT NULL,
                source_key TEXT NOT NULL UNIQUE,
                at TEXT,
                text TEXT NOT NULL,
                text_hash TEXT NOT NULL,
                embed_state INTEGER NOT NULL DEFAULT 0,
                embedding BLOB
            );
            CREATE INDEX IF NOT EXISTS segments_by_conversation ON segments(conversation_id, at);
            CREATE INDEX IF NOT EXISTS segments_by_origin ON segments(origin_id, at);
            CREATE INDEX IF NOT EXISTS segments_pending_embedding ON segments(embed_state) WHERE embed_state = 1;
            CREATE VIRTUAL TABLE IF NOT EXISTS segments_fts USING fts5(
                body,
                content = '',
                contentless_delete = 1,
                tokenize = 'unicode61 remove_diacritics 2'
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS segments_vocab USING fts5vocab(segments_fts, 'row');
            CREATE TABLE IF NOT EXISTS refs (
                conversation_id TEXT NOT NULL,
                segment_id INTEGER NOT NULL,
                kind TEXT NOT NULL,
                value TEXT NOT NULL,
                PRIMARY KEY (kind, value, conversation_id, segment_id)
            ) WITHOUT ROWID;
            CREATE INDEX IF NOT EXISTS refs_by_conversation ON refs(conversation_id, kind);
            ",
        )
        .map_err(|error| format!("Conversations schema: {}", error))?;
        conn.execute_batch(super::coverage::SCHEMA).map_err(|error| format!("Conversations schema (coverage): {}", error))?;
        conn.execute_batch(super::submissions::SCHEMA).map_err(|error| format!("Conversations schema (submissions): {}", error))?;
        // Added after the first release of the index: set when the project
        // comes from the folder or its repository, which the mentions
        // counted in the messages must not overrule.
        if conn.prepare("SELECT project_pinned FROM conversations LIMIT 0").is_err() {
            // Another process opening the index at the same moment may have added it first.
            match conn.execute("ALTER TABLE conversations ADD COLUMN project_pinned INTEGER NOT NULL DEFAULT 0", []) {
                Err(error) if !error.to_string().contains("duplicate column name") => {
                    return Err(format!("Conversations schema (project_pinned): {}", error));
                }
                _ => {}
            }
        }
        Ok(Self { conn })
    }

    /// The index at its usual place, with the known secrets loaded so that
    /// everything written through it is redacted with them.
    pub fn open_default() -> Result<Self, String> {
        let path = default_path()?;
        secrets::ensure_known_secrets();
        Self::open(&path)
    }

    pub(super) fn origin_version(&self, origin_id: &str) -> Result<Option<String>, String> {
        self.conn
            .query_row("SELECT version FROM origins WHERE id = ?1", [origin_id], |row| row.get(0))
            .optional()
            .map(Option::flatten)
            .map_err(|error| format!("Origin version: {}", error))
    }

    /// The origins of `tool` already in the index.
    pub(super) fn origin_ids(&self, tool: &str) -> Result<std::collections::HashSet<String>, String> {
        let mut statement = self
            .conn
            .prepare("SELECT o.id FROM origins o JOIN conversations c ON c.id = o.conversation_id WHERE c.tool = ?1")
            .map_err(|error| format!("Origins: {}", error))?;
        let rows = statement.query_map([tool], |row| row.get(0)).map_err(|error| format!("Origins: {}", error))?;
        rows.collect::<Result<_, _>>().map_err(|error| format!("Origins: {}", error))
    }

    pub(super) fn seen_hashes(&self, origin_id: &str) -> Result<Vec<u64>, String> {
        let blob: Option<Vec<u8>> = self
            .conn
            .query_row("SELECT seen FROM origins WHERE id = ?1", [origin_id], |row| row.get(0))
            .optional()
            .map_err(|error| format!("Origin seen: {}", error))?
            .flatten();
        Ok(blob
            .unwrap_or_default()
            .chunks_exact(8)
            .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap_or([0; 8])))
            .collect())
    }

    /// Writes one origin's new messages and marks `seen` as read, in one
    /// transaction so an interrupted sync never skips a message.
    pub(crate) fn apply(
        &mut self,
        source: &SourceConversation,
        is_main: bool,
        messages: &[Message],
        seen: &[u64],
    ) -> Result<usize, String> {
        self.apply_with(source, is_main, messages, seen, None)
    }

    /// `apply` for an origin read whole, from its first message (a Claude
    /// Code session file): what the index holds of it and the source no
    /// longer does is dropped, and the days it was seen running without
    /// saying anything are forgotten, in the same transaction as the
    /// version that marks the file as read. Done apart, a sync stopped in
    /// between left the stale messages in place for good.
    pub(crate) fn apply_whole(
        &mut self,
        source: &SourceConversation,
        messages: &[Message],
        seen: &[u64],
        days_said: &std::collections::BTreeSet<String>,
        today: &str,
    ) -> Result<usize, String> {
        self.apply_with(source, true, messages, seen, Some(Whole { days_said, today }))
    }

    fn apply_with(
        &mut self,
        source: &SourceConversation,
        is_main: bool,
        messages: &[Message],
        seen: &[u64],
        whole: Option<Whole>,
    ) -> Result<usize, String> {
        // Every source (Cursor, Claude Code, importers) lands here: secrets
        // are masked before anything is hashed, indexed or embedded.
        let redacted = SourceConversation {
            title: source.title.as_deref().map(|title| secrets::redact(title).into_owned()),
            ..source.clone()
        };
        let source = &redacted;
        let tx = self.conn.transaction().map_err(|error| format!("Sync transaction: {}", error))?;
        if is_main {
            tx.execute(
                "INSERT INTO conversations (id, tool, title, workspace, created_at, source_path)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)
                 ON CONFLICT(id) DO UPDATE SET
                    title = COALESCE(excluded.title, conversations.title),
                    workspace = COALESCE(excluded.workspace, conversations.workspace),
                    created_at = COALESCE(conversations.created_at, excluded.created_at),
                    source_path = COALESCE(excluded.source_path, conversations.source_path)",
                params![
                    source.root_id,
                    source.tool,
                    source.title,
                    source.workspace,
                    source.created_at,
                    source.source_path
                ],
            )
            .map_err(|error| format!("Conversation upsert: {}", error))?;
        } else {
            tx.execute(
                "INSERT INTO conversations (id, tool, created_at) VALUES (?1, ?2, ?3) ON CONFLICT(id) DO NOTHING",
                params![source.root_id, source.tool, source.created_at],
            )
            .map_err(|error| format!("Conversation placeholder: {}", error))?;
        }

        let mut written = 0;
        if is_main {
            if let Some(title) = source.title.as_deref() {
                let key = format!("title:{}", source.root_id);
                let body = format!("{}\n{}", title, source.workspace.as_deref().map(project_name).unwrap_or_default());
                written += upsert_segment(&tx, source, "title", &key, source.created_at.as_deref(), &body)?;
            }
        }
        // One segment per user message; for the agent, its final answer,
        // what it wrote at length on the way (a draft, a plan), and the
        // actions of the turn. Its short progress notes ("checking X") are
        // nine messages out of ten and carry no recall value.
        // New messages continue the last turn the index holds; a source
        // read from its first message starts before any turn (what the
        // agent wrote first would otherwise be filed under the last request).
        let mut current_turn: Option<String> = match whole {
            Some(_) => None,
            None => tx
                .query_row(
                    "SELECT source_key FROM segments WHERE origin_id = ?1 AND role = 'user' AND source_key NOT GLOB '*#p[0-9]*'
                     ORDER BY at DESC, id DESC LIMIT 1",
                    [&source.origin_id],
                    |row| row.get(0),
                )
                .optional()
                .map_err(|error| format!("Current turn: {}", error))?,
        };
        let turn_key = |turn: &Option<String>| turn.clone().unwrap_or_else(|| format!("{}:{}:start", source.tool, source.origin_id));
        let mut answer: Option<&Message> = None;
        let mut actions: Vec<&str> = Vec::new();
        let mut actions_at: Option<&str> = None;
        // Whether the text the current answer replaced was kept as a step.
        let mut previous_kept = false;
        // The steps this conversation already holds, by the hashes of
        // their parts. Cursor keeps one message under many ids (a long
        // conversation: 1.5 million bubbles for 68,000 messages), and each
        // copy would otherwise be kept as a step of its own. Answers are
        // left out: the answer of an open turn becomes its step when the
        // turn goes on, and must then be written.
        let kept_steps: std::cell::RefCell<std::collections::HashSet<String>> = {
            let mut statement = tx
                .prepare("SELECT source_key, text_hash FROM segments WHERE conversation_id = ?1 AND role = 'step' ORDER BY source_key")
                .map_err(|error| format!("Steps so far: {}", error))?;
            let rows: Vec<(String, String)> = statement
                .query_map([&source.root_id], |row| Ok((row.get(0)?, row.get(1)?)))
                .map_err(|error| format!("Steps so far: {}", error))?
                .collect::<Result<_, _>>()
                .map_err(|error| format!("Steps so far: {}", error))?;
            // `key`, `key#p2`, `key#p3`… sort together and in order.
            let mut whole: std::collections::BTreeMap<&str, Vec<&str>> = std::collections::BTreeMap::new();
            for (key, hash) in &rows {
                let step = key.split_once("#step").map(|(step, _)| step).unwrap_or(key);
                whole.entry(step).or_default().push(hash);
            }
            std::cell::RefCell::new(whole.into_values().map(|hashes| hashes.join(",")).collect())
        };
        let flush = |tx: &rusqlite::Transaction, turn: &Option<String>, answer: Option<&Message>, previous_kept: bool| -> Result<usize, String> {
            let Some(message) = answer else { return Ok(0) };
            let body = message.text.trim();
            let key = format!("{}#answer", turn_key(turn));
            if body.chars().count() < MIN_ASSISTANT_CHARS {
                // "Done." ends the turn: what stood as its answer was just
                // kept as a step, and must not stay as the answer too.
                if !body.is_empty() && previous_kept {
                    for part in 0..MAX_PARTS {
                        delete_segment(tx, &if part == 0 { key.clone() } else { format!("{}#p{}", key, part + 1) })?;
                    }
                }
                return Ok(0);
            }
            upsert_parts(tx, source, "assistant", &key, message.at.as_deref(), body, text::MAX_ANSWER_CHARS)
        };
        let step = |tx: &rusqlite::Transaction, message: &Message| -> Result<usize, String> {
            let body = message.text.trim();
            if body.chars().count() < MIN_STEP_CHARS {
                return Ok(0);
            }
            let key = format!("{}:{}:{}#step", source.tool, source.origin_id, message.key);
            let hash = parts_signature(body, text::MAX_ANSWER_CHARS);
            let known = !kept_steps.borrow_mut().insert(hash);
            if known && !segment_exists(tx, &key)? {
                return Ok(0);
            }
            upsert_parts(tx, source, "step", &key, message.at.as_deref(), body, text::MAX_ANSWER_CHARS)
        };
        for message in messages {
            match message.role {
                Role::Assistant => {
                    actions.extend(message.actions.iter().map(String::as_str));
                    actions_at = actions_at.or(message.at.as_deref());
                    if message.text.trim().is_empty() {
                        continue;
                    }
                    // The same text again (a copy of the message under
                    // another id) replaces nothing.
                    if answer.is_some_and(|previous| previous.text.trim() == message.text.trim()) {
                        continue;
                    }
                    // What was the latest text of the turn is now a step.
                    previous_kept = match answer.replace(message) {
                        Some(previous) => {
                            written += step(&tx, previous)?;
                            previous.text.trim().chars().count() >= MIN_STEP_CHARS
                        }
                        None => false,
                    };
                }
                Role::User => {
                    written += flush(&tx, &current_turn, answer.take(), std::mem::take(&mut previous_kept))?;
                    written += upsert_actions(&tx, source, &turn_key(&current_turn), actions_at.take(), &std::mem::take(&mut actions))?;
                    let key = format!("{}:{}:{}", source.tool, source.origin_id, message.key);
                    let body = text::typed_text(&message.text);
                    if body.chars().count() >= MIN_USER_CHARS {
                        written += upsert_parts(&tx, source, "user", &key, message.at.as_deref(), &body, text::MAX_USER_CHARS)?;
                    }
                    current_turn = Some(key);
                }
            }
        }
        written += flush(&tx, &current_turn, answer.take(), previous_kept)?;
        written += upsert_actions(&tx, source, &turn_key(&current_turn), actions_at, &actions)?;

        let mut blob = Vec::with_capacity(seen.len() * 8);
        for hash in seen {
            blob.extend_from_slice(&hash.to_le_bytes());
        }
        tx.execute(
            "INSERT INTO origins (id, conversation_id, is_main, version, seen) VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(id) DO UPDATE SET conversation_id = excluded.conversation_id,
                is_main = excluded.is_main, version = excluded.version, seen = excluded.seen",
            params![source.origin_id, source.root_id, is_main as i64, source.version, blob],
        )
        .map_err(|error| format!("Origin upsert: {}", error))?;

        if let Some(whole) = whole {
            let kept: std::collections::HashSet<&str> = messages.iter().map(|message| message.key.as_str()).collect();
            prune_in(&tx, source, &kept)?;
            super::coverage::forget_silent_days_in(&tx, &source.root_id, whole.days_said, whole.today)?;
        }
        refresh_totals(&tx, &source.root_id)?;
        tx.commit().map_err(|error| format!("Sync commit: {}", error))?;
        Ok(written)
    }

    /// Records the project and folder the archive found for a conversation
    /// already indexed. The folder only fills a gap. The project replaces
    /// the one in place with `replace` (the archive was just built and
    /// checked), else only fills a gap too. Returns whether the
    /// conversation is in the index.
    pub fn set_identity(
        &mut self,
        conversation_id: &str,
        project: Option<&str>,
        workspace: Option<&str>,
        replace: bool,
    ) -> Result<bool, String> {
        let changed = self
            .conn
            .execute(
                "UPDATE conversations SET
                    project = CASE WHEN ?4 THEN COALESCE(?2, project) ELSE COALESCE(project, ?2) END,
                    project_pinned = CASE WHEN ?4 AND ?2 IS NOT NULL THEN 1 ELSE project_pinned END,
                    workspace = COALESCE(workspace, ?3)
                 WHERE id = ?1",
                params![conversation_id, project, workspace, replace],
            )
            .map_err(|error| format!("Conversation identity: {}", error))?;
        Ok(changed > 0)
    }

    /// Files every conversation under the project its folder belongs to
    /// (`known` answers from the registered folders, their former paths
    /// and the repository's remote), instead of the name of the folder or
    /// the paths its messages mention. Returns how many changed.
    pub fn assign_known_projects(&mut self, known: &dyn Fn(&str) -> Option<String>) -> Result<usize, String> {
        let folders: Vec<String> = {
            let mut statement = self
                .conn
                .prepare("SELECT DISTINCT workspace FROM conversations WHERE workspace IS NOT NULL AND workspace != ''")
                .map_err(|error| format!("Conversation folders: {}", error))?;
            let rows = statement
                .query_map([], |row| row.get::<_, String>(0))
                .map_err(|error| format!("Conversation folders: {}", error))?;
            rows.filter_map(Result::ok).collect()
        };
        let mut changed = 0;
        for folder in folders {
            let Some(project) = known(&folder) else { continue };
            changed += self
                .conn
                .execute(
                    "UPDATE conversations SET project = ?2, project_pinned = 1
                     WHERE workspace = ?1 AND (project IS NOT ?2 OR project_pinned = 0)",
                    params![folder, project],
                )
                .map_err(|error| format!("Conversation project: {}", error))?;
        }
        Ok(changed)
    }

    pub fn pending_embeddings(&self, limit: usize) -> Result<Vec<(i64, String)>, String> {
        let mut statement = self
            .conn
            .prepare(
                "SELECT s.id, s.text, s.role, s.at, c.title, COALESCE(c.project, c.workspace)
                 FROM segments s JOIN conversations c ON c.id = s.conversation_id
                 WHERE s.embed_state = 1 LIMIT ?1",
            )
            .map_err(|error| format!("Pending embeddings: {}", error))?;
        let rows = statement
            .query_map([limit as i64], |row| {
                let id: i64 = row.get(0)?;
                let body: String = row.get(1)?;
                let role: String = row.get(2)?;
                let at: Option<String> = row.get(3)?;
                let title: Option<String> = row.get(4)?;
                let project: Option<String> = row.get(5)?;
                Ok((id, embedding_input(&role, &body, at.as_deref(), title.as_deref(), project.as_deref())))
            })
            .map_err(|error| format!("Pending embeddings: {}", error))?;
        Ok(rows.filter_map(Result::ok).collect())
    }

    pub fn store_embeddings(&mut self, vectors: &[(i64, Vec<u8>)]) -> Result<(), String> {
        let tx = self.conn.transaction().map_err(|error| format!("Embedding transaction: {}", error))?;
        {
            let mut statement = tx
                .prepare_cached("UPDATE segments SET embedding = ?2, embed_state = 2 WHERE id = ?1")
                .map_err(|error| format!("Embedding update: {}", error))?;
            for (id, blob) in vectors {
                statement
                    .execute(params![id, blob])
                    .map_err(|error| format!("Embedding update: {}", error))?;
            }
        }
        tx.commit().map_err(|error| format!("Embedding commit: {}", error))
    }

    /// Masks the secrets already stored (`--redact-conversations`). Without
    /// `apply` only counts; with it rewrites each touched segment (text,
    /// hash, FTS row, references) and queues it for a new embedding, and
    /// masks the titles. Values never leave this function.
    pub fn redact_stored_secrets(&mut self, apply: bool) -> Result<RedactionReport, String> {
        let mut report = RedactionReport { applied: apply, ..Default::default() };
        let mut touched = std::collections::HashSet::new();
        let count = |report: &mut RedactionReport, text: &str| {
            for finding in secrets::findings(text) {
                *report.secrets_by_kind.entry(finding.kind.to_string()).or_default() += 1;
            }
        };
        let segments: Vec<StoredSegment> = {
            let mut statement = self
                .conn
                .prepare(
                    "SELECT s.id, s.conversation_id, s.role, s.text, c.workspace
                     FROM segments s LEFT JOIN conversations c ON c.id = s.conversation_id",
                )
                .map_err(|error| format!("Redaction scan: {}", error))?;
            let rows = statement
                .query_map([], |row| -> rusqlite::Result<StoredSegment> {
                    Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?))
                })
                .map_err(|error| format!("Redaction scan: {}", error))?;
            rows.filter_map(Result::ok)
                .filter(|(_, _, _, text, _)| !secrets::findings(text).is_empty())
                .collect()
        };
        let titles: Vec<(String, String)> = {
            let mut statement = self
                .conn
                .prepare("SELECT id, title FROM conversations WHERE title IS NOT NULL")
                .map_err(|error| format!("Redaction scan: {}", error))?;
            let rows = statement
                .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
                .map_err(|error| format!("Redaction scan: {}", error))?;
            rows.filter_map(Result::ok)
                .filter(|(_, title): &(String, String)| !secrets::findings(title).is_empty())
                .collect()
        };
        for (_, conversation_id, _, text, _) in &segments {
            count(&mut report, text);
            touched.insert(conversation_id.clone());
        }
        for (conversation_id, title) in &titles {
            count(&mut report, title);
            touched.insert(conversation_id.clone());
        }
        report.segments = segments.len();
        report.titles = titles.len();
        report.conversations = touched.len();
        if !apply {
            return Ok(report);
        }
        let tx = self.conn.transaction().map_err(|error| format!("Redaction transaction: {}", error))?;
        for (id, conversation_id, role, text, workspace) in &segments {
            let body = secrets::redact(text);
            tx.execute("DELETE FROM segments_fts WHERE rowid = ?1", [id])
                .map_err(|error| format!("FTS delete: {}", error))?;
            tx.execute("DELETE FROM refs WHERE segment_id = ?1", [id])
                .map_err(|error| format!("Refs delete: {}", error))?;
            tx.execute(
                "UPDATE segments SET text = ?2, text_hash = ?3, embed_state = ?4, embedding = NULL WHERE id = ?1",
                params![id, body, crate::db::content_hash(&body), embed_state(role, &body)],
            )
            .map_err(|error| format!("Segment update: {}", error))?;
            index_segment(&tx, conversation_id, workspace.as_deref(), *id, role, &body)?;
        }
        for (conversation_id, title) in &titles {
            tx.execute(
                "UPDATE conversations SET title = ?2 WHERE id = ?1",
                params![conversation_id, secrets::redact(title)],
            )
            .map_err(|error| format!("Title update: {}", error))?;
        }
        tx.commit().map_err(|error| format!("Redaction commit: {}", error))?;
        // The old values stay in freed pages until the file is rewritten.
        if report.segments + report.titles > 0 {
            self.conn
                .execute_batch("VACUUM; PRAGMA wal_checkpoint(TRUNCATE);")
                .map_err(|error| format!("Redaction vacuum: {}", error))?;
        }
        Ok(report)
    }

    pub fn stats(&self) -> Result<HashMap<&'static str, i64>, String> {
        let count = |sql: &str| -> Result<i64, String> {
            self.conn.query_row(sql, [], |row| row.get(0)).map_err(|error| format!("Stats: {}", error))
        };
        let mut stats = HashMap::new();
        stats.insert("conversations", count("SELECT COUNT(*) FROM conversations WHERE message_count > 0")?);
        stats.insert("messages", count("SELECT COUNT(*) FROM segments WHERE role != 'title'")?);
        stats.insert("embedded", count("SELECT COUNT(*) FROM segments WHERE embed_state = 2")?);
        stats.insert("pending_embeddings", count("SELECT COUNT(*) FROM segments WHERE embed_state = 1")?);
        stats.insert("references", count("SELECT COUNT(*) FROM refs")?);
        Ok(stats)
    }
}

/// id, conversation id, role, text, workspace of the conversation.
type StoredSegment = (i64, String, String, String, Option<String>);

#[derive(Debug, Default, serde::Serialize)]
pub struct RedactionReport {
    pub applied: bool,
    pub segments: usize,
    pub titles: usize,
    pub conversations: usize,
    pub secrets_by_kind: std::collections::BTreeMap<String, usize>,
}

/// Text given to the encoder: the conversation's title and project in
/// front, so a bare "OK, fais-le" still lands near its topic.
fn embedding_input(role: &str, body: &str, at: Option<&str>, title: Option<&str>, project: Option<&str>) -> String {
    let mut header = Vec::new();
    if let Some(project) = project.filter(|p| !p.is_empty()) {
        header.push(format!("projet: {}", project_name(project)));
    }
    if let Some(day) = at.and_then(|a| a.get(..10)) {
        header.push(format!("date: {}", day));
    }
    if let Some(title) = title.filter(|t| !t.is_empty() && role != "title") {
        header.push(format!("conversation: {}", title));
    }
    let body = text::truncate_chars(body, 1_200);
    if header.is_empty() {
        body
    } else {
        format!("[{}] {}", header.join("; "), body)
    }
}

pub fn project_name(workspace: &str) -> String {
    workspace
        .trim_end_matches('/')
        .rsplit('/')
        .next()
        .unwrap_or(workspace)
        .to_string()
}

/// What `apply_whole` needs beyond the messages.
#[derive(Clone, Copy)]
struct Whole<'a> {
    days_said: &'a std::collections::BTreeSet<String>,
    today: &'a str,
}

/// Drops the segments of an origin read whole whose message is no longer
/// among `kept` (message keys). A Claude Code session resumed from another
/// was once read under the id of that one: its messages stood in the
/// earlier conversation, and leave it once the resumed session is indexed
/// under its own id. Returns how many segments were dropped.
fn prune_in(tx: &rusqlite::Transaction, source: &SourceConversation, kept: &std::collections::HashSet<&str>) -> Result<usize, String> {
    let prefix = format!("{}:{}:", source.tool, source.origin_id);
    let stale: Vec<i64> = {
        let mut statement = tx
            .prepare("SELECT id, source_key FROM segments WHERE origin_id = ?1")
            .map_err(|error| format!("Prune: {}", error))?;
        let rows: Vec<(i64, String)> = statement
            .query_map([&source.origin_id], |row| Ok((row.get(0)?, row.get(1)?)))
            .and_then(|rows| rows.collect())
            .map_err(|error| format!("Prune: {}", error))?;
        rows.into_iter()
            .filter(|(_, key)| {
                let Some(rest) = key.strip_prefix(&prefix) else { return false };
                let message = rest.split('#').next().unwrap_or(rest);
                message != "start" && !kept.contains(message)
            })
            .map(|(id, _)| id)
            .collect()
    };
    for id in &stale {
        tx.execute("DELETE FROM segments_fts WHERE rowid = ?1", [id]).map_err(|error| format!("FTS delete: {}", error))?;
        tx.execute("DELETE FROM refs WHERE segment_id = ?1", [id]).map_err(|error| format!("Refs delete: {}", error))?;
        tx.execute("DELETE FROM segments WHERE id = ?1", [id]).map_err(|error| format!("Segment delete: {}", error))?;
    }
    Ok(stale.len())
}

fn refresh_totals(tx: &rusqlite::Transaction, conversation_id: &str) -> Result<(), String> {
    tx.execute(
        "UPDATE conversations SET
            message_count = (SELECT COUNT(*) FROM segments WHERE conversation_id = ?1 AND role IN ('user', 'assistant')
                             AND source_key NOT GLOB '*#p[0-9]*'),
            updated_at = COALESCE((SELECT MAX(at) FROM segments WHERE conversation_id = ?1), updated_at),
            project = CASE WHEN project_pinned = 1 THEN project ELSE COALESCE(
                (SELECT value FROM refs WHERE conversation_id = ?1 AND kind = 'project'
                 GROUP BY value ORDER BY COUNT(*) DESC LIMIT 1),
                project) END
         WHERE id = ?1",
        [conversation_id],
    )
    .map_err(|error| format!("Conversation totals: {}", error))?;
    Ok(())
}

fn upsert_segment(
    tx: &rusqlite::Transaction,
    source: &SourceConversation,
    role: &str,
    key: &str,
    at: Option<&str>,
    body: &str,
) -> Result<usize, String> {
    let body = &secrets::redact(body);
    let hash = crate::db::content_hash(body);
    let existing: Option<(i64, String)> = tx
        .query_row(
            "SELECT id, text_hash FROM segments WHERE source_key = ?1",
            [key],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .optional()
        .map_err(|error| format!("Segment lookup: {}", error))?;
    if existing.as_ref().is_some_and(|(_, old)| old == &hash) {
        return Ok(0);
    }
    let embed_state = embed_state(role, body);
    let id = match existing {
        Some((id, _)) => {
            tx.execute("DELETE FROM segments_fts WHERE rowid = ?1", [id])
                .map_err(|error| format!("FTS delete: {}", error))?;
            tx.execute("DELETE FROM refs WHERE segment_id = ?1", [id])
                .map_err(|error| format!("Refs delete: {}", error))?;
            tx.execute(
                "UPDATE segments SET text = ?2, text_hash = ?3, at = COALESCE(?4, at), embed_state = ?5, embedding = NULL WHERE id = ?1",
                params![id, body, hash, at, embed_state],
            )
            .map_err(|error| format!("Segment update: {}", error))?;
            id
        }
        None => {
            tx.execute(
                "INSERT INTO segments (conversation_id, origin_id, role, source_key, at, text, text_hash, embed_state)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                params![source.root_id, source.origin_id, role, key, at, body, hash, embed_state],
            )
            .map_err(|error| format!("Segment insert: {}", error))?;
            tx.last_insert_rowid()
        }
    };
    index_segment(tx, &source.root_id, source.workspace.as_deref(), id, role, body)?;
    Ok(1)
}

/// 1: queued for the encoder, 0: never embedded (short agent notes,
/// the actions of a turn, which are found by their words).
fn embed_state(role: &str, body: &str) -> i64 {
    match role {
        "assistant" if body.chars().count() < EMBED_ASSISTANT_CHARS => 0,
        "step" if body.chars().count() < EMBED_STEP_CHARS => 0,
        "action" => 0,
        _ => 1,
    }
}

/// `body` in parts of at most `max` characters, cut at a paragraph, a
/// line or a space when one is near the end of the part.
fn parts(body: &str, max: usize) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut rest = body.trim();
    while !rest.is_empty() && parts.len() < MAX_PARTS {
        let Some((limit, _)) = rest.char_indices().nth(max) else {
            parts.push(rest);
            break;
        };
        let window = &rest[..limit];
        let floor = window.char_indices().nth(max * 2 / 3).map(|(index, _)| index).unwrap_or(0);
        let cut = ["\n\n", "\n", " "]
            .iter()
            .find_map(|separator| window.rfind(separator).filter(|index| *index >= floor))
            .unwrap_or(limit);
        parts.push(rest[..cut].trim_end());
        rest = rest[cut..].trim_start();
    }
    parts
}

/// Stores a text whole: its first part under `key`, the next ones under
/// `key#p2`, `key#p3`… Parts a shorter rewrite no longer has are removed.
fn upsert_parts(
    tx: &rusqlite::Transaction,
    source: &SourceConversation,
    role: &str,
    key: &str,
    at: Option<&str>,
    body: &str,
    max: usize,
) -> Result<usize, String> {
    let body = secrets::redact(body);
    let parts = parts(&body, max);
    let mut written = 0;
    for (index, part) in parts.iter().enumerate() {
        let part_key = if index == 0 { key.to_string() } else { format!("{}#p{}", key, index + 1) };
        written += upsert_segment(tx, source, role, &part_key, at, part)?;
    }
    for index in parts.len().max(1)..MAX_PARTS {
        delete_segment(tx, &format!("{}#p{}", key, index + 1))?;
    }
    Ok(written)
}

/// The hashes `upsert_parts` stores for the parts of `body`, joined: two
/// long texts that begin alike and end differently are not the same.
fn parts_signature(body: &str, max: usize) -> String {
    let body = secrets::redact(body);
    parts(&body, max).iter().map(|part| crate::db::content_hash(&secrets::redact(part))).collect::<Vec<_>>().join(",")
}

fn segment_exists(tx: &rusqlite::Transaction, key: &str) -> Result<bool, String> {
    tx.query_row("SELECT 1 FROM segments WHERE source_key = ?1", [key], |_| Ok(()))
        .optional()
        .map(|found| found.is_some())
        .map_err(|error| format!("Segment lookup: {}", error))
}

fn delete_segment(tx: &rusqlite::Transaction, key: &str) -> Result<(), String> {
    let id: Option<i64> = tx
        .query_row("SELECT id FROM segments WHERE source_key = ?1", [key], |row| row.get(0))
        .optional()
        .map_err(|error| format!("Segment lookup: {}", error))?;
    let Some(id) = id else { return Ok(()) };
    for sql in ["DELETE FROM segments_fts WHERE rowid = ?1", "DELETE FROM refs WHERE segment_id = ?1", "DELETE FROM segments WHERE id = ?1"] {
        tx.execute(sql, [id]).map_err(|error| format!("Segment delete: {}", error))?;
    }
    Ok(())
}

/// Adds the actions of a turn to the ones already stored for it: a turn
/// is read in several syncs while it runs.
fn upsert_actions(
    tx: &rusqlite::Transaction,
    source: &SourceConversation,
    turn_key: &str,
    at: Option<&str>,
    actions: &[&str],
) -> Result<usize, String> {
    if actions.is_empty() {
        return Ok(0);
    }
    let key = format!("{}#actions", turn_key);
    let mut stored = String::new();
    for index in 0..MAX_PARTS {
        let part_key = if index == 0 { key.clone() } else { format!("{}#p{}", key, index + 1) };
        let part: Option<String> = tx
            .query_row("SELECT text FROM segments WHERE source_key = ?1", [&part_key], |row| row.get(0))
            .optional()
            .map_err(|error| format!("Actions lookup: {}", error))?;
        let Some(part) = part else { break };
        stored.push_str(&part);
        stored.push('\n');
    }
    let mut lines: Vec<String> = stored.lines().map(str::to_string).filter(|line| !line.is_empty()).collect();
    for action in actions {
        let action = secrets::redact(action).into_owned();
        if !lines.contains(&action) {
            lines.push(action);
        }
    }
    upsert_parts(tx, source, "action", &key, at, &lines.join("\n"), text::MAX_ANSWER_CHARS)
}

/// Writes the FTS row and the references of a segment whose old ones
/// were deleted (or never existed).
fn index_segment(
    tx: &rusqlite::Transaction,
    conversation_id: &str,
    workspace: Option<&str>,
    id: i64,
    role: &str,
    body: &str,
) -> Result<(), String> {
    let stemmed = crate::stemming::stem_text(body);
    tx.execute(
        "INSERT INTO segments_fts (rowid, body) VALUES (?1, ?2)",
        params![id, format!("{}\n{}", body, stemmed)],
    )
    .map_err(|error| format!("FTS insert: {}", error))?;

    let mut refs: Vec<Reference> = text::extract_references(body);
    refs.extend(super::actions::urls(body).into_iter().map(|value| Reference { kind: "url", value }));
    refs.extend(project_hints(body).into_iter().map(|value| Reference { kind: "project", value }));
    if role == "title" {
        if let Some(workspace) = workspace {
            refs.push(Reference { kind: "project", value: project_name(workspace) });
        }
    }
    for reference in refs {
        tx.execute(
            "INSERT OR IGNORE INTO refs (conversation_id, segment_id, kind, value) VALUES (?1, ?2, ?3, ?4)",
            params![conversation_id, id, reference.kind, reference.value],
        )
        .map_err(|error| format!("Refs insert: {}", error))?;
    }
    Ok(())
}

/// Project folders named in a message (`~/Cursor/App/<Project>/…`), the
/// only reliable project signal for conversations Cursor stored without
/// a workspace.
fn project_hints(body: &str) -> Vec<String> {
    let mut hints = Vec::new();
    let mut rest = body;
    while let Some(index) = rest.find("Cursor/App/") {
        let tail = &rest[index + "Cursor/App/".len()..];
        let tail = tail.strip_prefix("Clients/").unwrap_or(tail);
        let name: String = tail
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '-' || *c == '_' || *c == '.')
            .collect();
        if name.chars().count() >= 3 && !hints.contains(&name) {
            hints.push(name);
        }
        rest = &rest[index + 1..];
    }
    hints
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_project_from_paths() {
        assert_eq!(
            project_hints("cd ~/Cursor/App/Konta/apps/web puis /Users/a/Cursor/App/Clients/Atelier-Demo/x"),
            vec!["Konta".to_string(), "Atelier-Demo".to_string()]
        );
    }

    fn message(key: &str, role: Role, minute: u32, text: &str) -> Message {
        Message {
            key: key.to_string(),
            role,
            at: Some(format!("2026-10-28T10:{:02}:00.000Z", minute)),
            order: None,
            text: text.to_string(),
            actions: Vec::new(),
        }
    }

    fn segments(store: &ConversationStore) -> Vec<(String, String)> {
        let mut statement =
            store.conn.prepare("SELECT role, text FROM segments WHERE role != 'title' ORDER BY at, id").unwrap();
        statement.query_map([], |row| Ok((row.get(0)?, row.get(1)?))).unwrap().map(Result::unwrap).collect()
    }

    #[test]
    fn keeps_each_request_and_only_the_final_answer_of_its_turn() {
        let dir = std::env::temp_dir().join(format!("mp-conv-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("turns.db");
        let _ = std::fs::remove_file(&path);
        let mut store = ConversationStore::open(&path).unwrap();
        let source = SourceConversation {
            origin_id: "c1".to_string(),
            root_id: "c1".to_string(),
            tool: "cursor",
            title: Some("Rapprochement bancaire".to_string()),
            workspace: None,
            created_at: None,
            version: "1".to_string(),
            source_path: None,
        };
        let progress = "Je regarde le fichier de rapprochement et les colonnes de la table.";
        let first_draft = "Le rapprochement est corrigé, il reste le test à écrire pour la banque.";
        store
            .apply(&source, true, &[
                message("u1", Role::User, 1, "Corrige le rapprochement des virements Qonto"),
                message("a1", Role::Assistant, 2, progress),
                message("a2", Role::Assistant, 3, first_draft),
            ], &[])
            .unwrap();
        let final_answer = "Le rapprochement est corrigé et testé : les virements Qonto passent par verifyMatch.";
        store
            .apply(&source, true, &[message("a3", Role::Assistant, 4, final_answer)], &[])
            .unwrap();

        assert_eq!(
            segments(&store),
            vec![
                ("user".to_string(), "Corrige le rapprochement des virements Qonto".to_string()),
                ("assistant".to_string(), final_answer.to_string()),
            ]
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn stored(store: &ConversationStore) -> Vec<(String, String, String)> {
        let mut statement = store
            .conn
            .prepare("SELECT role, source_key, text FROM segments WHERE role != 'title' ORDER BY at, id")
            .unwrap();
        statement.query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?))).unwrap().map(Result::unwrap).collect()
    }

    #[test]
    fn a_message_stored_under_many_ids_is_kept_once() {
        let dir = std::env::temp_dir().join(format!("mp-conv-copies-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let mut store = ConversationStore::open(&dir.join("c.db")).unwrap();
        let source = SourceConversation {
            origin_id: "c1".to_string(),
            root_id: "c1".to_string(),
            tool: "cursor",
            title: None,
            workspace: None,
            created_at: None,
            version: "1".to_string(),
            source_path: None,
        };
        let plan = format!("The plan for the export: one file per month, named by its period. {}", "Each file holds the entries of its month. ".repeat(5));
        let answer = format!("The export is written and tested on three months of entries. {}", "Totals match the ledger. ".repeat(6));
        // As Cursor stores a long conversation: each message under several ids.
        let mut messages = vec![message("u1", Role::User, 1, "Écris l'export mensuel des écritures comptables")];
        for copy in 0..40 {
            messages.push(message(&format!("plan-{}", copy), Role::Assistant, 2, &plan));
        }
        for copy in 0..40 {
            messages.push(message(&format!("answer-{}", copy), Role::Assistant, 3, &answer));
        }
        store.apply(&source, true, &messages, &[]).unwrap();
        let count = |store: &ConversationStore, role: &str| stored(store).iter().filter(|row| row.0 == role).count();
        assert_eq!(count(&store, "step"), 1, "the plan once: {:?}", stored(&store));
        assert_eq!(count(&store, "assistant"), 1, "the answer once");
        // More copies come with a later sync, and in a later turn.
        let later = [
            message("plan-90", Role::Assistant, 4, &plan),
            message("answer-90", Role::Assistant, 5, &answer),
            message("u2", Role::User, 6, "Ajoute le total par journal dans chaque fichier"),
            message("plan-91", Role::Assistant, 7, &plan),
            message("a2", Role::Assistant, 8, "The total per journal is added at the end of each file, after the entries of the month."),
        ];
        store.apply(&source, true, &later, &[]).unwrap();
        assert_eq!(count(&store, "step"), 1, "no copy becomes a step: {:?}", stored(&store));
        assert_eq!(count(&store, "assistant"), 2, "one answer per turn");
        // Reading the same messages again changes nothing.
        assert_eq!(store.apply(&source, true, &messages, &[]).unwrap(), 0);
        assert_eq!(count(&store, "step"), 1);
        // Two long texts that begin alike and end differently are two texts.
        let opening = "Le plan détaillé de la migration, étape par étape. ".repeat(80);
        let first = format!("{}Conclusion : on migre en une fois.", opening);
        let second = format!("{}Conclusion : on migre en deux temps, par lots.", opening);
        assert!(first.chars().count() > text::MAX_ANSWER_CHARS);
        let revised = [
            message("u3", Role::User, 9, "Détaille le plan de migration du schéma de facturation"),
            message("long-1", Role::Assistant, 10, &first),
            message("long-2", Role::Assistant, 11, &second),
            message("a3", Role::Assistant, 12, "The two versions of the plan are above; the second one is the one I recommend for this schema."),
        ];
        store.apply(&source, true, &revised, &[]).unwrap();
        let all: String = stored(&store).iter().filter(|row| row.0 == "step").map(|row| row.2.as_str()).collect::<Vec<_>>().join(" ");
        assert!(all.contains("on migre en une fois"), "the first version is kept");
        assert!(all.contains("on migre en deux temps"), "and the revised one too");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_turn_keeps_its_draft_its_actions_and_long_texts_whole() {
        let dir = std::env::temp_dir().join(format!("mp-conv-turn-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let mut store = ConversationStore::open(&dir.join("c.db")).unwrap();
        let source = SourceConversation {
            origin_id: "c1".to_string(),
            root_id: "c1".to_string(),
            tool: "cursor",
            title: None,
            workspace: None,
            created_at: None,
            version: "1".to_string(),
            source_path: None,
        };
        let draft = format!(
            "Here is the post for the developer community: music cannot be attached to a scheduled post through the API. {}",
            "We tried the audio endpoint and the container flow. ".repeat(4)
        );
        let mut browse = message("a2", Role::Assistant, 3, "");
        browse.actions = vec![
            "navigate: https://developers.example.com/community/threads/new/?ref=nav".to_string(),
            "form_input: Music on a scheduled post".to_string(),
        ];
        let mut again = message("a3", Role::Assistant, 4, "");
        again.actions = vec!["navigate: https://developers.example.com/community/threads/new/?ref=nav".to_string()];
        let long_request = format!("Écris le message pour le forum. {}", "Le contexte complet de la demande est le suivant. ".repeat(100));
        store
            .apply(&source, true, &[
                message("u1", Role::User, 1, &long_request),
                message("a1", Role::Assistant, 2, &draft),
                browse,
                message("a4", Role::Assistant, 5, "Checking the page."),
                message("a5", Role::Assistant, 6, "The post is published on the developer community; I will watch for an answer."),
            ], &[])
            .unwrap();
        // The same turn read again, as while it is still running.
        store.apply(&source, true, &[again], &[]).unwrap();

        let rows = stored(&store);
        let of = |role: &str| -> Vec<&(String, String, String)> { rows.iter().filter(|row| row.0 == role).collect() };
        let request: String = of("user").iter().map(|row| row.2.as_str()).collect::<Vec<_>>().join(" ");
        assert!(of("user").len() >= 2, "a long request is stored in parts");
        assert!(of("user").iter().all(|row| row.2.chars().count() <= text::MAX_USER_CHARS));
        assert_eq!(request.matches("Le contexte complet").count(), 100, "nothing is cut");
        assert_eq!(of("step").len(), 1, "the draft is kept, the progress note is not: {:?}", of("step"));
        assert!(of("step")[0].2.starts_with("Here is the post"));
        assert_eq!(of("assistant").len(), 1);
        assert!(of("assistant")[0].2.starts_with("The post is published"));
        // A turn that ends on a word keeps its long text once, as a step.
        let reply = format!("Here is the reply for the thread: the limit is confirmed by the vendor. {}", "It applies to every scheduled post. ".repeat(6));
        let ending = [
            message("u9", Role::User, 20, "Prépare la réponse pour le fil de discussion du forum"),
            message("a9", Role::Assistant, 21, &reply),
        ];
        store.apply(&source, true, &ending, &[]).unwrap();
        // The open turn is read again whole, as a sync does.
        store.apply(&source, true, &[message("a9", Role::Assistant, 21, &reply), message("a10", Role::Assistant, 22, "Done.")], &[]).unwrap();
        let after = stored(&store);
        assert_eq!(after.iter().filter(|row| row.0 == "assistant").count(), 1, "the draft is no longer an answer");
        assert_eq!(after.iter().filter(|row| row.0 == "step").count(), 2);
        let rows = stored(&store);
        let of = |role: &str| -> Vec<&(String, String, String)> { rows.iter().filter(|row| row.0 == role && !row.1.contains("u9") && !row.1.contains("a9")).collect() };
        assert_eq!(of("action").len(), 1, "one list of actions per turn: {:?}", of("action"));
        assert_eq!(
            of("action")[0].2,
            "navigate: https://developers.example.com/community/threads/new/?ref=nav\nform_input: Music on a scheduled post",
            "each action once, in order"
        );
        let url: i64 = store
            .conn
            .query_row(
                "SELECT COUNT(*) FROM refs WHERE kind = 'url' AND value = 'https://developers.example.com/community/threads/new'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(url, 1);
        let count: i64 = store.conn.query_row("SELECT message_count FROM conversations", [], |row| row.get(0)).unwrap();
        assert_eq!(count, 3, "two requests and one answer, whatever their length");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_conversation_is_filed_under_the_project_of_its_folder() {
        let dir = std::env::temp_dir().join(format!("mp-conv-folder-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let mut store = ConversationStore::open(&dir.join("c.db")).unwrap();
        let source = SourceConversation {
            origin_id: "c1".to_string(),
            root_id: "c1".to_string(),
            tool: "cursor",
            title: None,
            workspace: Some("/Users/x/Cursor/App/Application/MonApp".to_string()),
            created_at: None,
            version: "1".to_string(),
            source_path: None,
        };
        let first = message("u1", Role::User, 1, "Regarde ~/Cursor/App/Planner/lib pour la publication programmée");
        store.apply(&source, true, &[first], &[]).unwrap();
        let project = |store: &ConversationStore| -> Option<String> {
            store.conn.query_row("SELECT project FROM conversations WHERE id = 'c1'", [], |row| row.get(0)).unwrap()
        };
        assert_eq!(project(&store).as_deref(), Some("Planner"), "the folder most mentioned, before the project is known");

        let known = |folder: &str| folder.ends_with("/MonApp").then(|| "monapp".to_string());
        assert_eq!(store.assign_known_projects(&known).unwrap(), 1);
        assert_eq!(project(&store).as_deref(), Some("monapp"));
        // A later message naming another folder does not move it back.
        let later = message("u2", Role::User, 2, "Et ~/Cursor/App/Planner/website pour la page de tarifs");
        store.apply(&SourceConversation { version: "2".to_string(), ..source }, true, &[later], &[]).unwrap();
        assert_eq!(project(&store).as_deref(), Some("monapp"));
        assert_eq!(store.assign_known_projects(&known).unwrap(), 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn redacts_secrets_stored_before_the_filter() {
        let dir = std::env::temp_dir().join(format!("mp-conv-redact-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("redact.db");
        let _ = std::fs::remove_file(&path);
        let mut store = ConversationStore::open(&path).unwrap();
        let source = SourceConversation {
            origin_id: "c1".to_string(),
            root_id: "c1".to_string(),
            tool: "cursor",
            title: Some("Déploiement du CI".to_string()),
            workspace: Some("/Users/x/Cursor/App/Konta".to_string()),
            created_at: None,
            version: "1".to_string(),
            source_path: None,
        };
        store
            .apply(&source, true, &[message("u1", Role::User, 1, "Corrige le déploiement du CI de Konta")], &[])
            .unwrap();
        // What an older version stored: the value in clear, indexed.
        let token = format!("ghp_{}", crate::secrets::tests::fake(36, 80));
        let leaked = format!("Le push passe avec token={} sur main", token);
        let id: i64 = store.conn.query_row("SELECT id FROM segments WHERE role = 'user'", [], |row| row.get(0)).unwrap();
        store.conn.execute("DELETE FROM segments_fts WHERE rowid = ?1", [id]).unwrap();
        store
            .conn
            .execute(
                "UPDATE segments SET text = ?2, text_hash = ?3, embed_state = 2, embedding = x'00' WHERE id = ?1",
                params![id, leaked, crate::db::content_hash(&leaked)],
            )
            .unwrap();
        store.conn.execute("INSERT INTO segments_fts (rowid, body) VALUES (?1, ?2)", params![id, leaked]).unwrap();
        store.conn.execute("UPDATE conversations SET title = ?1", [format!("CI {}", token)]).unwrap();
        let hits = |store: &ConversationStore, phrase: &str| -> i64 {
            store
                .conn
                .query_row(
                    "SELECT COUNT(*) FROM segments_fts WHERE segments_fts MATCH ?1",
                    [format!("\"{}\"", phrase)],
                    |row| row.get(0),
                )
                .unwrap()
        };

        let dry = store.redact_stored_secrets(false).unwrap();
        assert_eq!((dry.applied, dry.segments, dry.titles, dry.conversations), (false, 1, 1, 1));
        assert_eq!(dry.secrets_by_kind.get("github_token"), Some(&2));
        let unchanged: String = store.conn.query_row("SELECT text FROM segments WHERE id = ?1", [id], |row| row.get(0)).unwrap();
        assert_eq!(unchanged, leaked);
        assert_eq!(hits(&store, &token), 1);

        let applied = store.redact_stored_secrets(true).unwrap();
        assert_eq!((applied.applied, applied.segments, applied.titles), (true, 1, 1));
        let (text, hash, embed_state, embedding): (String, String, i64, Option<Vec<u8>>) = store
            .conn
            .query_row("SELECT text, text_hash, embed_state, embedding FROM segments WHERE id = ?1", [id], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
            })
            .unwrap();
        let masked = crate::secrets::mask("github_token", &token);
        assert_eq!(text, format!("Le push passe avec token={} sur main", masked));
        assert_eq!(hash, crate::db::content_hash(&text));
        assert_eq!((embed_state, embedding), (1, None));
        assert_eq!(hits(&store, &token), 0);
        assert_eq!(hits(&store, "push passe avec token"), 1);
        let title: String = store.conn.query_row("SELECT title FROM conversations", [], |row| row.get(0)).unwrap();
        assert_eq!(title, format!("CI {}", masked));
        for file in [path.clone(), dir.join("redact.db-wal")] {
            let bytes = std::fs::read(&file).unwrap_or_default();
            assert!(!bytes.windows(token.len()).any(|window| window == token.as_bytes()));
        }

        let again = store.redact_stored_secrets(true).unwrap();
        assert_eq!((again.segments, again.titles, again.conversations), (0, 0, 0));
        assert!(again.secrets_by_kind.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
