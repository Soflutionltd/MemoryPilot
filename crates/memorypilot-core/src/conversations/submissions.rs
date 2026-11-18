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
const TYPING: &[&str] = &["type", "fill", "fill_form", "form_input", "fl_act", "submit"];
/// Tool names (their last part) that open a page in a browser. Fetching
/// a page or calling an API reads a site, it does not write on it.
const OPENING: &[&str] = &["navigate", "open", "goto"];
/// A request is a post or a message: this much text typed in one field
/// of the page. A name, a search or a one-time code is not one.
const MIN_TYPED_CHARS: usize = 80;
/// Arguments of a typing call that hold what was typed.
const TYPED_KEYS: &[&str] = &["text", "value", "content", "body", "message"];

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
    /// What was typed on the page, longest text first.
    pub typed: String,
}

fn host_of(url: &str) -> &str {
    let rest = url.split("://").nth(1).unwrap_or(url);
    rest.split('/').next().unwrap_or(rest).trim_start_matches("www.")
}

/// The last part of an action's tool name, lowercase.
fn tool_of(action: &str) -> String {
    let name = action.split(':').next().unwrap_or_default();
    name.rsplit("__").next().unwrap_or(name).to_lowercase()
}

fn is_one_of(tool: &str, names: &[&str]) -> bool {
    names.iter().any(|name| tool == *name || tool.ends_with(&format!("_{}", name)) || tool.starts_with(&format!("{}_", name)))
}

/// The value that follows `marker` in `raw`, up to the quote that closes
/// it. `quote` is the closing quote as written at that nesting level: a
/// quote inside the value carries one more backslash and is skipped. A
/// line cut short has no closing quote: the value runs to its end.
fn value_after<'a>(raw: &'a str, marker: &str, quote: &str) -> Option<&'a str> {
    let value = &raw[raw.find(marker)? + marker.len()..];
    let mut from = 0;
    while let Some(found) = value[from..].find(quote) {
        let at = from + found;
        if !value[..at].ends_with('\\') {
            return Some(&value[..at]);
        }
        from = at + quote.len();
    }
    Some(value.trim_end_matches('…'))
}

/// What a typing action typed: the text after the tool name, or the
/// value of the argument that holds it when the line is a JSON call
/// (Cursor nests it, escaped, inside its own envelope).
fn typed_text(action: &str) -> String {
    let rest = action.split_once(": ").map(|(_, rest)| rest).unwrap_or_default().trim();
    if !rest.starts_with('{') {
        return rest.to_string();
    }
    TYPED_KEYS
        .iter()
        .filter_map(|key| {
            // Nested in a string first (Cursor), then as a plain argument.
            value_after(rest, &format!("\\\"{}\\\":\\\"", key), "\\\"")
                .map(|value| value.replace("\\\\\\\"", "\"").replace("\\\\n", " "))
                .or_else(|| value_after(rest, &format!("\"{}\":\"", key), "\"").map(|value| value.replace("\\\"", "\"").replace("\\n", " ")))
        })
        .max_by_key(|value| value.chars().count())
        .unwrap_or_default()
}

/// The request of a turn's actions: text typed, at the length of a
/// message, on a page of a known site the agent had opened. Returns the
/// host, the page and what was typed.
fn submission_in(actions_text: &str, hosts: &[String]) -> Option<(String, String, String)> {
    let known = |host: &str| hosts.iter().any(|site| host == site || host.ends_with(&format!(".{}", site)));
    let mut page: Option<(String, String)> = None;
    let mut found: Option<(String, String)> = None;
    let mut typed: Vec<String> = Vec::new();
    for action in actions_text.lines() {
        let tool = tool_of(action);
        if is_one_of(&tool, OPENING) {
            page = actions::urls(action).into_iter().next().map(|url| (host_of(&url).to_string(), url)).filter(|(host, _)| known(host));
        } else if is_one_of(&tool, TYPING) {
            let Some(page) = &page else { continue };
            let text = typed_text(action);
            if !text.is_empty() {
                found.get_or_insert_with(|| page.clone());
                typed.push(text);
            }
        }
    }
    let (host, url) = found?;
    typed.sort_by_key(|text| std::cmp::Reverse(text.chars().count()));
    // One field at the length of a message: several short fields (a name,
    // an address) do not add up to one.
    if typed.first().is_none_or(|longest| longest.chars().count() < MIN_TYPED_CHARS) {
        return None;
    }
    Some((host, url, typed.join(" / ")))
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
                let (host, url, typed) = submission_in(&actions_text, hosts)?;
                Some(Submission { source_key, conversation_id, project, at, host, url, request: request.unwrap_or_default(), typed })
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
                "mcp__browser__form_input: A post scheduled through the content publishing API cannot carry a music track. Is there a supported way to attach one?",
            ],
        );
        // A name typed while creating an app, a page fetched and an API called are not requests.
        turn(
            &mut store,
            "app",
            "Crée l'application sur leur portail développeur",
            &[
                "mcp-cursor-ide-browser-cursor-ide-browser-browser_navigate: {\"tools\":[{\"name\":\"browser_navigate\",\"parameters\":\"{\\\"url\\\":\\\"https://developers.facebook.com/apps/creation/\\\"}\"}]}",
                "mcp-cursor-ide-browser-cursor-ide-browser-browser_type: {\"tools\":[{\"name\":\"browser_type\",\"parameters\":\"{\\\"ref\\\":\\\"e16\\\",\\\"text\\\":\\\"MonApp\\\"}\"}]}",
            ],
        );
        turn(
            &mut store,
            "fetched",
            "Lis leur documentation et remplis le formulaire de notre site de test",
            &[
                "web_fetch: https://developers.facebook.com/docs/graph-api/reference/user/accounts/",
                "run_terminal_command_v2: curl -sS https://api.appstoreconnect.apple.com/v1/apps",
                "navigate: https://staging.example.org/contact",
                "browser_type: Bonjour, voici un long message de test saisi dans le formulaire de contact de notre propre site de recette.",
            ],
        );
        turn(&mut store, "read", "Regarde si leur forum parle de la musique", &["mcp__browser__navigate: https://developers.facebook.com/community/"]);
        turn(
            &mut store,
            "elsewhere",
            "Remplis le formulaire de test de l'application",
            &["navigate: https://staging.example.com/signup", "browser_type: Jeanne Martin, responsable des essais, souhaite recevoir les notes de version de chaque mois."],
        );
        // Several short fields on a known site are a form, not a message.
        turn(
            &mut store,
            "form",
            "Mets à jour la fiche de contact du compte développeur",
            &[
                "navigate: https://developers.facebook.com/settings/contact/",
                "form_input: Jeanne Martin",
                "form_input: jeanne.martin@example.org",
                "form_input: 12 rue des Essais, 75000 Paris, France",
                "form_input: Responsable technique",
            ],
        );
        let hosts: Vec<String> = DEFAULT_HOSTS.iter().map(|host| host.to_string()).collect();
        let found = store.unreported_submissions(&hosts).unwrap();
        assert_eq!(found.len(), 1, "{found:?}");
        assert_eq!(found[0].conversation_id, "posted");
        assert_eq!(found[0].host, "developers.facebook.com");
        assert_eq!(found[0].url, "https://developers.facebook.com/community/threads/new");
        assert!(found[0].request.starts_with("Poste la question sur la musique"));
        assert!(found[0].typed.starts_with("A post scheduled through the content publishing API"), "{}", found[0].typed);
        assert!(found[0].typed.ends_with("Music on a scheduled Instagram post"));
        assert_eq!(found[0].at.as_deref(), Some("2026-10-12T10:05:00.000Z"));

        // A site added by the user is watched too; a reported one is not proposed again.
        let wider: Vec<String> = hosts.iter().cloned().chain(["example.com".to_string()]).collect();
        assert_eq!(store.unreported_submissions(&wider).unwrap().len(), 2);
        store.mark_submission_reported(&found[0].source_key).unwrap();
        assert!(store.unreported_submissions(&hosts).unwrap().is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_text_typed_is_read_whole_from_a_nested_call_even_with_quotes_in_it() {
        let nested = r#"mcp-cursor-ide-browser-cursor-ide-browser-browser_type: {"tools":[{"name":"browser_type","parameters":"{\"ref\":\"e16\",\"text\":\"Je dis \\\"bonjour\\\", puis je demande si une musique peut accompagner {une publication} programmée.\"}","serverName":"cursor-ide-browser"}]}"#;
        assert_eq!(typed_text(nested), "Je dis \"bonjour\", puis je demande si une musique peut accompagner {une publication} programmée.");
        let plain = r#"browser_type: {"ref":"e4","text":"Is \"music\", on a scheduled post, supported?"}"#;
        assert_eq!(typed_text(plain), "Is \"music\", on a scheduled post, supported?");
        let cut = r#"browser_type: {"tools":[{"name":"browser_type","parameters":"{\"text\":\"A long question about the content publishing API that the action line cut bef…"#;
        assert_eq!(typed_text(cut), "A long question about the content publishing API that the action line cut bef");
        assert_eq!(typed_text("mcp__browser__form_input: Music on a scheduled post"), "Music on a scheduled post");
        assert_eq!(typed_text(r#"browser_click: {"ref":"e9"}"#), "");
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
