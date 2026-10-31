//! Conversations pushed by an importer (claude.ai, Gemini) instead of
//! read from a local source. Stored like the Cursor and Claude Code ones:
//! each request, the final answer of each turn, nothing summarised.
//!
//! Idempotent per conversation id + message id: the ids already received
//! are kept in the origin's `seen` set, so sending the same conversation
//! again only writes the messages added since.

use super::store::ConversationStore;
use super::{fold_timestamp, hash_key, Message, Role, SourceConversation};
use serde_json::Value;

/// Tools an importer may push, as stored in `conversations.tool`.
pub const TOOLS: &[&str] = &["claude-ai", "gemini"];

#[derive(Debug, Clone)]
pub struct ImportedMessage {
    pub id: String,
    pub role: Role,
    pub text: String,
    pub at: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ImportedConversation {
    pub tool: &'static str,
    pub id: String,
    pub title: Option<String>,
    pub created_at: Option<String>,
    pub messages: Vec<ImportedMessage>,
}

#[derive(Debug, serde::Serialize)]
pub struct IngestReport {
    pub conversation_id: String,
    pub received: usize,
    pub new_messages: usize,
    pub segments_written: usize,
}

/// Reads and checks a conversation from tool arguments: `tool`,
/// `conversation_id`, optional `title` and `date`, and `messages`
/// (`id`, `role`, `text`, optional `date`). Nothing is written when any
/// part is invalid.
pub fn parse(args: &Value) -> Result<ImportedConversation, String> {
    let field = |value: &Value, key: &str| -> Option<String> {
        value
            .get(key)
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(String::from)
    };
    let date = |value: &Value, what: &str| -> Result<Option<String>, String> {
        match field(value, "date") {
            None => Ok(None),
            Some(raw) => fold_timestamp(&raw)
                .map(Some)
                .ok_or_else(|| format!("{}: date '{}' is not RFC 3339", what, raw)),
        }
    };

    let tool_name = field(args, "tool").ok_or("tool required")?;
    let tool = *TOOLS
        .iter()
        .find(|t| **t == tool_name)
        .ok_or_else(|| format!("tool must be one of: {}", TOOLS.join(", ")))?;
    let id = field(args, "conversation_id").ok_or("conversation_id required")?;
    let raw_messages = args
        .get("messages")
        .and_then(Value::as_array)
        .ok_or("messages must be an array")?;

    let mut messages = Vec::with_capacity(raw_messages.len());
    for (index, raw) in raw_messages.iter().enumerate() {
        let what = format!("messages[{}]", index);
        let message_id = field(raw, "id").ok_or_else(|| format!("{}: id required", what))?;
        let role = match field(raw, "role").as_deref() {
            Some("user" | "human") => Role::User,
            Some("assistant" | "model") => Role::Assistant,
            other => {
                return Err(format!(
                    "{}: role must be user (or human) or assistant (or model), got {:?}",
                    what, other
                ))
            }
        };
        let text = raw.get("text").and_then(Value::as_str).unwrap_or_default().to_string();
        messages.push(ImportedMessage { id: message_id, role, text, at: date(raw, &what)? });
    }

    Ok(ImportedConversation {
        tool,
        id,
        title: field(args, "title"),
        created_at: date(args, "conversation")?,
        messages,
    })
}

/// Writes the messages of `conversation` not received before.
pub fn ingest(store: &mut ConversationStore, conversation: &ImportedConversation) -> Result<IngestReport, String> {
    // Prefixed so an imported id can never collide with a local one.
    let root_id = format!("{}:{}", conversation.tool, conversation.id);
    let mut seen = store.seen_hashes(&root_id)?;
    seen.sort_unstable();

    let mut fresh = Vec::new();
    let mut added = std::collections::HashSet::new();
    for (order, message) in conversation.messages.iter().enumerate() {
        let hash = hash_key(&message.id);
        // Known from an earlier call, or repeated in this one.
        if seen.binary_search(&hash).is_ok() || !added.insert(hash) {
            continue;
        }
        fresh.push(Message {
            key: message.id.clone(),
            role: message.role,
            at: message.at.clone(),
            order: Some(order),
            text: message.text.clone(),
        });
    }
    let mut all_seen: Vec<u64> = seen.into_iter().chain(added).collect();
    all_seen.sort_unstable();

    let source = SourceConversation {
        origin_id: root_id.clone(),
        root_id: root_id.clone(),
        tool: conversation.tool,
        title: conversation.title.clone(),
        workspace: None,
        created_at: conversation.created_at.clone(),
        version: all_seen.len().to_string(),
        source_path: None,
    };
    let segments_written = store.apply(&source, true, &fresh, &all_seen)?;
    Ok(IngestReport {
        conversation_id: root_id,
        received: conversation.messages.len(),
        new_messages: fresh.len(),
        segments_written,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn temp_store(name: &str) -> (ConversationStore, std::path::PathBuf) {
        let dir = std::env::temp_dir().join(format!("mp-import-{}-{}", name, uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        (ConversationStore::open(&dir.join("conversations.db")).unwrap(), dir)
    }

    fn message(id: &str, role: &str, minute: u32, text: &str) -> Value {
        json!({ "id": id, "role": role, "text": text, "date": format!("2026-10-31T10:{:02}:00Z", minute) })
    }

    fn payload(messages: Vec<Value>) -> Value {
        json!({
            "tool": "claude-ai",
            "conversation_id": "abc-123",
            "title": "Plancher des épisodes",
            "date": "2026-10-31T10:00:00Z",
            "messages": messages,
        })
    }

    fn rows(store: &ConversationStore) -> Vec<(String, String)> {
        let mut statement = store
            .conn
            .prepare("SELECT role, text FROM segments WHERE role != 'title' ORDER BY at, id")
            .unwrap();
        statement.query_map([], |row| Ok((row.get(0)?, row.get(1)?))).unwrap().map(Result::unwrap).collect()
    }

    const ASK: &str = "Explique-moi le plancher des épisodes";
    const ANSWER: &str = "Le plancher est une note minimale en dessous de laquelle un épisode n'est pas montré.";
    const ASK_2: &str = "Fixe-le à 0,45 dans la configuration";
    const ANSWER_2: &str = "C'est fait : episode_absolute_floor vaut 0,45 par défaut, réglable par set_config.";

    #[test]
    fn stores_messages_and_only_adds_new_ones_on_resend() {
        let (mut store, dir) = temp_store("resend");
        let first = vec![message("m1", "user", 1, ASK), message("m2", "assistant", 2, ANSWER)];

        let report = ingest(&mut store, &parse(&payload(first.clone())).unwrap()).unwrap();
        assert_eq!(report.conversation_id, "claude-ai:abc-123");
        assert_eq!((report.received, report.new_messages, report.segments_written), (2, 2, 3));

        // Same conversation again: nothing new.
        let again = ingest(&mut store, &parse(&payload(first.clone())).unwrap()).unwrap();
        assert_eq!((again.received, again.new_messages, again.segments_written), (2, 0, 0));

        // Two more messages: only they are written.
        let mut longer = first;
        longer.push(message("m3", "human", 3, ASK_2));
        longer.push(message("m4", "model", 4, ANSWER_2));
        let more = ingest(&mut store, &parse(&payload(longer)).unwrap()).unwrap();
        assert_eq!((more.received, more.new_messages, more.segments_written), (4, 2, 2));

        assert_eq!(
            rows(&store),
            [("user", ASK), ("assistant", ANSWER), ("user", ASK_2), ("assistant", ANSWER_2)]
                .map(|(role, text)| (role.to_string(), text.to_string()))
        );
        let (tool, title, count): (String, String, i64) = store
            .conn
            .query_row(
                "SELECT tool, title, message_count FROM conversations WHERE id = 'claude-ai:abc-123'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert_eq!((tool.as_str(), title.as_str(), count), ("claude-ai", "Plancher des épisodes", 4));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn same_conversation_id_from_two_tools_stays_apart() {
        let (mut store, dir) = temp_store("tools");
        let messages = vec![message("m1", "user", 1, ASK)];
        ingest(&mut store, &parse(&payload(messages.clone())).unwrap()).unwrap();
        let mut gemini = payload(messages);
        gemini["tool"] = json!("gemini");
        let report = ingest(&mut store, &parse(&gemini).unwrap()).unwrap();
        assert_eq!((report.conversation_id.as_str(), report.new_messages), ("gemini:abc-123", 1));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn rejects_invalid_input_before_writing() {
        let mut bad_tool = payload(vec![]);
        bad_tool["tool"] = json!("chatgpt");
        assert!(parse(&bad_tool).unwrap_err().contains("claude-ai, gemini"));
        assert!(parse(&payload(vec![message("m1", "system", 1, ASK)])).unwrap_err().contains("messages[0]: role"));
        assert!(parse(&payload(vec![json!({ "role": "user", "text": ASK })])).unwrap_err().contains("id required"));
        assert!(parse(&payload(vec![json!({ "id": "m1", "role": "user", "text": ASK, "date": "hier" })]))
            .unwrap_err()
            .contains("RFC 3339"));
    }
}
