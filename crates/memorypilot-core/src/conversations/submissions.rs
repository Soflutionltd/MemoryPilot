//! Requests an agent sent to a third party, found in the actions of a turn.
//!
//! The agent that posts on a vendor's forum is asked to record it
//! (`log_external_request`) and sometimes does not. The index keeps the
//! actions of each turn: a turn where the agent typed text on a page of
//! a site known for requests (a developer forum, a feedback form, a
//! support desk) is taken for one, and proposed as a request to confirm.
//! Deliberately narrow: a page merely read, or a site outside the list,
//! is not a request.

use super::actions;
use super::store::ConversationStore;
use rusqlite::params;

/// Sites where what is typed is a request to someone else. More can be
/// added with the `external_hosts` setting.
pub const DEFAULT_HOSTS: &[&str] = &[
    "developers.facebook.com",
    "developer.apple.com",
    "feedbackassistant.apple.com",
    "appstoreconnect.apple.com",
    "issuetracker.google.com",
    "support.google.com",
    "community.openai.com",
    "community.cloudflare.com",
    "community.stripe.com",
    "support.stripe.com",
    "stackoverflow.com",
];

/// Tool names (their last part) that enter text in a page.
const TYPING: &[&str] = &["type", "fill", "form_input", "fl_act", "submit"];

pub(super) const SCHEMA: &str = "
    CREATE TABLE IF NOT EXISTS reported_submissions (
        source_key TEXT PRIMARY KEY,
        reported_at TEXT NOT NULL
    ) WITHOUT ROWID;
";

#[derive(Debug, Clone, PartialEq)]
pub struct Submission {
    /// The actions segment it was read from: reported once.
    pub source_key: String,
    pub conversation_id: String,
    pub project: Option<String>,
    pub at: Option<String>,
    pub host: String,
    pub url: String,
    /// What the user asked in that turn.
    pub request: String,
}

fn host_of(url: &str) -> &str {
    let rest = url.split("://").nth(1).unwrap_or(url);
    rest.split('/').next().unwrap_or(rest).trim_start_matches("www.")
}

fn types_text(action: &str) -> bool {
    let name = action.split(':').next().unwrap_or_default();
    let short = name.rsplit("__").next().unwrap_or(name).to_lowercase();
    TYPING.iter().any(|typing| short == *typing || short.ends_with(&format!("_{}", typing)) || short.starts_with(&format!("{}_", typing)))
}

/// The request of a turn's actions, when they type text on a known site.
fn submission_in(actions_text: &str, hosts: &[String]) -> Option<(String, String)> {
    if !actions_text.lines().any(types_text) {
        return None;
    }
    actions::urls(actions_text).into_iter().find_map(|url| {
        let host = host_of(&url).to_string();
        hosts.iter().any(|known| host == *known || host.ends_with(&format!(".{}", known))).then_some((host, url))
    })
}

impl ConversationStore {
    /// Turns that look like a request sent to a third party and were not
    /// reported yet, oldest first.
    pub fn unreported_submissions(&self, hosts: &[String]) -> Result<Vec<Submission>, String> {
        let mut statement = self
            .conn
            .prepare(
                "SELECT s.source_key, s.conversation_id, c.project, s.at, s.text,
                        (SELECT u.text FROM segments u WHERE u.source_key = substr(s.source_key, 1, length(s.source_key) - length('#actions')))
                 FROM segments s JOIN conversations c ON c.id = s.conversation_id
                 WHERE s.role = 'action' AND s.source_key LIKE '%#actions'
                   AND NOT EXISTS (SELECT 1 FROM reported_submissions r WHERE r.source_key = s.source_key)
                 ORDER BY COALESCE(s.at, ''), s.id",
            )
            .map_err(|error| format!("Submissions: {}", error))?;
        let rows = statement
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, Option<String>>(2)?,
                    row.get::<_, Option<String>>(3)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, Option<String>>(5)?,
                ))
            })
            .map_err(|error| format!("Submissions: {}", error))?;
        Ok(rows
            .filter_map(Result::ok)
            .filter_map(|(source_key, conversation_id, project, at, actions_text, request)| {
                let (host, url) = submission_in(&actions_text, hosts)?;
                Some(Submission { source_key, conversation_id, project, at, host, url, request: request.unwrap_or_default() })
            })
            .collect())
    }

    /// Notes that a submission was turned into a request, so that it is
    /// not proposed again once the request is edited or closed.
    pub fn mark_submission_reported(&self, source_key: &str) -> Result<(), String> {
        self.conn
            .execute(
                "INSERT OR IGNORE INTO reported_submissions (source_key, reported_at) VALUES (?1, ?2)",
                params![source_key, chrono::Utc::now().to_rfc3339()],
            )
            .map(|_| ())
            .map_err(|error| format!("Submission mark: {}", error))
    }

    /// Makes the next sync read every conversation again from its source,
    /// with the rules of this version (drafts, actions, whole texts).
    /// Nothing is deleted: a message read again replaces itself.
    pub fn forget_what_was_read(&self) -> Result<usize, String> {
        self.conn
            .execute("UPDATE origins SET version = NULL, seen = NULL", [])
            .map_err(|error| format!("Rebuild: {}", error))
    }
}

#[cfg(test)]
mod tests {
    use super::super::{Message, Role, SourceConversation};
    use super::*;

    fn turn(store: &mut ConversationStore, id: &str, request: &str, actions: &[&str]) {
        let source = SourceConversation {
            origin_id: id.to_string(),
            root_id: id.to_string(),
            tool: "cursor",
            title: None,
            workspace: None,
            created_at: None,
            version: "1".to_string(),
            source_path: None,
        };
        let messages = [
            Message {
                key: "u".to_string(),
                role: Role::User,
                at: Some("2026-10-12T10:00:00.000Z".to_string()),
                order: None,
                text: request.to_string(),
                actions: Vec::new(),
            },
            Message {
                key: "a".to_string(),
                role: Role::Assistant,
                at: Some("2026-10-12T10:05:00.000Z".to_string()),
                order: None,
                text: String::new(),
                actions: actions.iter().map(|action| action.to_string()).collect(),
            },
        ];
        store.apply(&source, true, &messages, &[]).unwrap();
    }

    #[test]
    fn text_typed_on_a_known_site_is_a_request_and_a_page_read_is_not() {
        let dir = std::env::temp_dir().join(format!("mp-submissions-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let mut store = ConversationStore::open(&dir.join("c.db")).unwrap();
        turn(
            &mut store,
            "posted",
            "Poste la question sur la musique d'une publication programmée dans leur communauté",
            &[
                "mcp__browser__navigate: https://developers.facebook.com/community/threads/new/?ref=nav",
                "mcp__browser__form_input: Music on a scheduled Instagram post",
            ],
        );
        turn(&mut store, "read", "Regarde si leur forum parle de la musique", &["mcp__browser__navigate: https://developers.facebook.com/community/"]);
        turn(
            &mut store,
            "elsewhere",
            "Remplis le formulaire de test de l'application",
            &["navigate: https://staging.example.com/signup", "browser_type: Jeanne Martin"],
        );
        let hosts: Vec<String> = DEFAULT_HOSTS.iter().map(|host| host.to_string()).collect();
        let found = store.unreported_submissions(&hosts).unwrap();
        assert_eq!(found.len(), 1, "{found:?}");
        assert_eq!(found[0].conversation_id, "posted");
        assert_eq!(found[0].host, "developers.facebook.com");
        assert_eq!(found[0].url, "https://developers.facebook.com/community/threads/new");
        assert!(found[0].request.starts_with("Poste la question sur la musique"));
        assert_eq!(found[0].at.as_deref(), Some("2026-10-12T10:05:00.000Z"));

        // A site added by the user is watched too; a reported one is not proposed again.
        let wider: Vec<String> = hosts.iter().cloned().chain(["example.com".to_string()]).collect();
        assert_eq!(store.unreported_submissions(&wider).unwrap().len(), 2);
        store.mark_submission_reported(&found[0].source_key).unwrap();
        assert!(store.unreported_submissions(&hosts).unwrap().is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn forgetting_what_was_read_makes_every_conversation_read_again() {
        let dir = std::env::temp_dir().join(format!("mp-rebuild-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let mut store = ConversationStore::open(&dir.join("c.db")).unwrap();
        turn(&mut store, "one", "Une demande suffisamment longue pour être gardée", &[]);
        assert_eq!(store.origin_version("one").unwrap().as_deref(), Some("1"));
        assert_eq!(store.forget_what_was_read().unwrap(), 1);
        assert_eq!(store.origin_version("one").unwrap(), None);
        let kept: i64 = store.conn.query_row("SELECT COUNT(*) FROM segments", [], |row| row.get(0)).unwrap();
        assert_eq!(kept, 1, "nothing is deleted");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
