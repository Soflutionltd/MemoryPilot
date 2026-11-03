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
pub fn default_path() -> Result<PathBuf, String> {
    if let Ok(path) = std::env::var("MEMORYPILOT_CONVERSATIONS_DB") {
        return Ok(PathBuf::from(path));
    }
    let dir = dirs::home_dir().ok_or("Cannot find home directory")?.join(".MemoryPilot");
    std::fs::create_dir_all(&dir).map_err(|error| format!("Cannot create dir: {}", error))?;
    Ok(dir.join(FILE_NAME))
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
    pub(super) fn apply(
        &mut self,
        source: &SourceConversation,
        is_main: bool,
        messages: &[Message],
        seen: &[u64],
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
        // One segment per user message, one per turn for the agent: its
        // final answer. The progress notes in between ("checking X") are
        // nine messages out of ten and carry no recall value.
        let mut current_turn: Option<String> = tx
            .query_row(
                "SELECT source_key FROM segments WHERE origin_id = ?1 AND role = 'user' ORDER BY at DESC, id DESC LIMIT 1",
                [&source.origin_id],
                |row| row.get(0),
            )
            .optional()
            .map_err(|error| format!("Current turn: {}", error))?;
        let mut answer: Option<&Message> = None;
        let flush = |tx: &rusqlite::Transaction, turn: &Option<String>, answer: Option<&Message>| -> Result<usize, String> {
            let Some(message) = answer else { return Ok(0) };
            let body = text::truncate_chars(secrets::redact(message.text.trim()).trim(), text::MAX_ANSWER_CHARS);
            if body.chars().count() < MIN_ASSISTANT_CHARS {
                return Ok(0);
            }
            let turn_key = turn.clone().unwrap_or_else(|| format!("{}:{}:start", source.tool, source.origin_id));
            let key = format!("{}#answer", turn_key);
            upsert_segment(tx, source, "assistant", &key, message.at.as_deref(), &body)
        };
        for message in messages {
            match message.role {
                Role::Assistant => answer = Some(message),
                Role::User => {
                    written += flush(&tx, &current_turn, answer.take())?;
                    let key = format!("{}:{}:{}", source.tool, source.origin_id, message.key);
                    let body = text::clean_user_text(&secrets::redact(&message.text));
                    if body.chars().count() >= MIN_USER_CHARS {
                        written += upsert_segment(&tx, source, "user", &key, message.at.as_deref(), &body)?;
                    }
                    current_turn = Some(key);
                }
            }
        }
        written += flush(&tx, &current_turn, answer.take())?;

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

        tx.execute(
            "UPDATE conversations SET
                message_count = (SELECT COUNT(*) FROM segments WHERE conversation_id = ?1 AND role != 'title'),
                updated_at = COALESCE((SELECT MAX(at) FROM segments WHERE conversation_id = ?1), updated_at),
                project = COALESCE(
                    (SELECT value FROM refs WHERE conversation_id = ?1 AND kind = 'project'
                     GROUP BY value ORDER BY COUNT(*) DESC LIMIT 1),
                    project)
             WHERE id = ?1",
            [&source.root_id],
        )
        .map_err(|error| format!("Conversation totals: {}", error))?;
        tx.commit().map_err(|error| format!("Sync commit: {}", error))?;
        Ok(written)
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

/// 1: queued for the encoder, 0: never embedded (short agent notes).
fn embed_state(role: &str, body: &str) -> i64 {
    match role {
        "assistant" if body.chars().count() < EMBED_ASSISTANT_CHARS => 0,
        _ => 1,
    }
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
            project_hints("cd ~/Cursor/App/Konta/apps/web puis /Users/a/Cursor/App/Clients/Laetitia-Vagner/x"),
            vec!["Konta".to_string(), "Laetitia-Vagner".to_string()]
        );
    }

    fn message(key: &str, role: Role, minute: u32, text: &str) -> Message {
        Message {
            key: key.to_string(),
            role,
            at: Some(format!("2026-10-28T10:{:02}:00.000Z", minute)),
            order: None,
            text: text.to_string(),
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
