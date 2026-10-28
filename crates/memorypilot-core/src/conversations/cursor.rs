//! Reads Cursor's own chat store, read-only.
//!
//! Cursor keeps every agent conversation of every window in one SQLite
//! file (`globalStorage/state.vscdb`, table `cursorDiskKV`):
//! `composerData:<id>` holds a conversation's metadata and
//! `bubbleId:<id>:<bubble>` one message each (type 1 = user, 2 = agent).
//! The per-project `agent-transcripts` folders only hold a fraction of
//! them, so this store is the source of truth.

use super::{fold_timestamp, Message, Role, SourceConversation};
use rusqlite::{Connection, OpenFlags};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub fn default_store_path() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("MEMORYPILOT_CURSOR_STORE") {
        return Some(PathBuf::from(path));
    }
    let home = dirs::home_dir()?;
    let path = if cfg!(target_os = "macos") {
        home.join("Library/Application Support/Cursor/User/globalStorage/state.vscdb")
    } else if cfg!(target_os = "windows") {
        dirs::config_dir()?.join("Cursor/User/globalStorage/state.vscdb")
    } else {
        home.join(".config/Cursor/User/globalStorage/state.vscdb")
    };
    path.exists().then_some(path)
}

pub fn open(path: &Path) -> Result<Connection, String> {
    let conn = Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX | OpenFlags::SQLITE_OPEN_URI,
    )
    .map_err(|error| format!("Cursor store open: {}", error))?;
    conn.busy_timeout(std::time::Duration::from_secs(5))
        .map_err(|error| format!("Cursor store busy timeout: {}", error))?;
    Ok(conn)
}

#[derive(Deserialize)]
struct ComposerData {
    #[serde(rename = "composerId")]
    composer_id: Option<String>,
    name: Option<String>,
    #[serde(rename = "createdAt")]
    created_at: Option<i64>,
    #[serde(rename = "lastUpdatedAt")]
    last_updated_at: Option<i64>,
    #[serde(rename = "workspaceIdentifier")]
    workspace: Option<WorkspaceIdentifier>,
    #[serde(rename = "subagentComposerIds", default)]
    subagent_ids: Vec<String>,
    #[serde(rename = "subComposerIds", default)]
    sub_ids: Vec<String>,
    #[serde(rename = "fullConversationHeadersOnly", default)]
    headers: Vec<BubbleHeader>,
}

#[derive(Deserialize)]
struct WorkspaceIdentifier {
    uri: Option<WorkspaceUri>,
}

#[derive(Deserialize)]
struct WorkspaceUri {
    #[serde(rename = "fsPath")]
    fs_path: Option<String>,
}

#[derive(Deserialize)]
struct BubbleHeader {
    #[serde(rename = "bubbleId")]
    bubble_id: String,
}

#[derive(Deserialize)]
struct Bubble {
    #[serde(rename = "type")]
    kind: Option<u8>,
    text: Option<String>,
    #[serde(rename = "createdAt")]
    created_at: Option<serde_json::Value>,
}

pub struct Composer {
    pub conversation: SourceConversation,
    /// Bubble ids in display order, when Cursor still writes them
    /// (older conversations). Newer ones only carry `createdAt`.
    header_order: Vec<String>,
}

/// Every composer, with sub-agent and best-of-N composers pointed at the
/// conversation the user actually had.
pub fn list_composers(conn: &Connection) -> Result<Vec<Composer>, String> {
    let mut statement = conn
        .prepare("SELECT value FROM cursorDiskKV WHERE key >= 'composerData:' AND key < 'composerData;'")
        .map_err(|error| format!("Cursor composers: {}", error))?;
    let rows = statement
        .query_map([], |row| raw_value(row, 0))
        .map_err(|error| format!("Cursor composers: {}", error))?;

    let mut composers = Vec::new();
    let mut parent: HashMap<String, String> = HashMap::new();
    for row in rows {
        let Ok(Some(value)) = row else { continue };
        let Ok(data) = serde_json::from_slice::<ComposerData>(&value) else { continue };
        let Some(id) = data.composer_id.clone() else { continue };
        for child in data.subagent_ids.iter().chain(data.sub_ids.iter()) {
            parent.insert(child.clone(), id.clone());
        }
        let workspace = data
            .workspace
            .and_then(|w| w.uri)
            .and_then(|u| u.fs_path)
            .filter(|p| !p.is_empty());
        composers.push(Composer {
            conversation: SourceConversation {
                origin_id: id.clone(),
                root_id: id,
                tool: "cursor",
                title: data.name.filter(|n| !n.trim().is_empty()),
                workspace,
                created_at: data.created_at.and_then(millis_to_rfc3339),
                version: data.last_updated_at.or(data.created_at).unwrap_or(0).to_string(),
                source_path: None,
            },
            header_order: data.headers.into_iter().map(|h| h.bubble_id).collect(),
        });
    }

    for composer in &mut composers {
        let mut root = composer.conversation.origin_id.clone();
        let mut hops = 0;
        while let Some(up) = parent.get(&root) {
            root = up.clone();
            hops += 1;
            if hops > 8 {
                break;
            }
        }
        composer.conversation.root_id = root;
    }
    Ok(composers)
}

/// Bubble ids of one composer. Reads keys only (covering index), so it
/// stays cheap even for a conversation with hundreds of thousands of
/// tool-call bubbles.
pub fn bubble_ids(conn: &Connection, composer_id: &str) -> Result<Vec<String>, String> {
    let prefix = format!("bubbleId:{}:", composer_id);
    let upper = format!("bubbleId:{};", composer_id);
    let mut statement = conn
        .prepare("SELECT key FROM cursorDiskKV WHERE key >= ?1 AND key < ?2")
        .map_err(|error| format!("Cursor bubbles: {}", error))?;
    let rows = statement
        .query_map([&prefix, &upper], |row| row.get::<_, String>(0))
        .map_err(|error| format!("Cursor bubbles: {}", error))?;
    Ok(rows
        .filter_map(Result::ok)
        .filter_map(|key| key.strip_prefix(&prefix).map(str::to_string))
        .collect())
}

/// The readable messages among `ids`, in conversation order.
pub fn read_messages(conn: &Connection, composer: &Composer, ids: &[String]) -> Result<Vec<Message>, String> {
    let position: HashMap<&str, usize> = composer
        .header_order
        .iter()
        .enumerate()
        .map(|(index, id)| (id.as_str(), index))
        .collect();
    let base = composer.conversation.created_at.clone();
    let mut statement = conn
        .prepare_cached("SELECT value FROM cursorDiskKV WHERE key = ?1")
        .map_err(|error| format!("Cursor bubble: {}", error))?;

    let mut messages = Vec::new();
    for id in ids {
        let key = format!("bubbleId:{}:{}", composer.conversation.origin_id, id);
        let value = statement.query_row([&key], |row| raw_value(row, 0)).ok().flatten();
        let Some(value) = value else { continue };
        let Ok(bubble) = serde_json::from_slice::<Bubble>(&value) else { continue };
        let role = match bubble.kind {
            Some(1) => Role::User,
            Some(2) => Role::Assistant,
            _ => continue,
        };
        let Some(text) = bubble.text.filter(|t| !t.trim().is_empty()) else { continue };
        let at = bubble
            .created_at
            .as_ref()
            .and_then(timestamp_of)
            .or_else(|| base.clone());
        messages.push(Message {
            key: id.clone(),
            role,
            at,
            order: position.get(id.as_str()).copied(),
            text,
        });
    }
    messages.sort_by(|a, b| {
        let rank = |m: &Message| (m.order.unwrap_or(usize::MAX), m.at.clone(), m.key.clone());
        rank(a).cmp(&rank(b))
    });
    Ok(messages)
}

/// `cursorDiskKV.value` is declared BLOB but Cursor writes JSON as TEXT.
fn raw_value(row: &rusqlite::Row, index: usize) -> rusqlite::Result<Option<Vec<u8>>> {
    use rusqlite::types::ValueRef;
    Ok(match row.get_ref(index)? {
        ValueRef::Text(bytes) | ValueRef::Blob(bytes) => Some(bytes.to_vec()),
        _ => None,
    })
}

fn timestamp_of(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::String(text) if !text.is_empty() => fold_timestamp(text),
        serde_json::Value::Number(number) => number.as_i64().and_then(millis_to_rfc3339),
        _ => None,
    }
}

fn millis_to_rfc3339(millis: i64) -> Option<String> {
    chrono::DateTime::from_timestamp_millis(millis).map(|at| at.to_rfc3339_opts(chrono::SecondsFormat::Millis, true))
}
