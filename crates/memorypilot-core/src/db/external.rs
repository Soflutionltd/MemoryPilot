//! What was sent to a third party, and whether an answer came back.
//!
//! A post on a vendor's forum, a bug report, a support ticket or an
//! e-mail leaves no trace in the code, and the session that wrote it may
//! never be indexed. Asked later "did they answer?", an agent that finds
//! nothing concludes nothing was sent. Each request is therefore a
//! memory of its own kind, with where it went, when, and its status, and
//! the open ones are listed in one call and recalled at session start.

use super::*;
use serde::Serialize;
use serde_json::{json, Map, Value};

pub const EXTERNAL_REQUEST_KIND: &str = "external_request";
pub const EXTERNAL_STATUSES: &[&str] = &["pending", "answered", "closed"];
const PENDING: &str = "pending";
const METADATA_KEY: &str = "external";
const TAG: &str = "external-request";
const IMPORTANCE: i32 = 4;
const LISTED: usize = 500;

/// A request as tools show it.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ExternalRequest {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project: Option<String>,
    pub destination: String,
    pub subject: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    pub sent_at: String,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_reply_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
    /// How the request came to be recorded: by the agent that sent it, or
    /// found afterwards in a session and waiting to be confirmed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detected: Option<String>,
    /// Days since it was sent, while it waits for an answer.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub waiting_days: Option<i64>,
}

#[derive(Debug, Clone, Default)]
pub struct NewExternalRequest<'a> {
    pub destination: &'a str,
    pub subject: &'a str,
    pub url: Option<&'a str>,
    /// `YYYY-MM-DD` or RFC 3339; now when absent.
    pub sent_at: Option<&'a str>,
    pub project: Option<&'a str>,
    /// What was asked, in a few sentences: it is what a later search finds.
    pub details: Option<&'a str>,
    pub detected: Option<&'a str>,
}

/// A date as `YYYY-MM-DD` or RFC 3339, read as a UTC instant.
fn instant(raw: &str) -> Option<chrono::DateTime<chrono::Utc>> {
    let raw = raw.trim();
    chrono::DateTime::parse_from_rfc3339(raw).map(|at| at.with_timezone(&chrono::Utc)).ok().or_else(|| {
        chrono::NaiveDate::parse_from_str(raw, "%Y-%m-%d").ok()?.and_hms_opt(12, 0, 0).map(|at| at.and_utc())
    })
}

fn checked_date(field: &str, raw: &str) -> Result<String, String> {
    instant(raw).map(|_| raw.trim().to_string()).ok_or_else(|| format!("{}: expected YYYY-MM-DD or RFC 3339, got '{}'", field, raw))
}

fn day(raw: &str) -> &str {
    raw.get(..10).unwrap_or(raw)
}

/// The text a search reads: who was asked what, when, and where it stands.
fn content(request: &ExternalRequest, details: Option<&str>) -> String {
    let mut lines = vec![format!(
        "External request to {} ({}): {}",
        request.destination,
        day(&request.sent_at),
        request.subject
    )];
    if let Some(details) = details.map(str::trim).filter(|details| !details.is_empty()) {
        lines.push(details.to_string());
    }
    if let Some(url) = &request.url {
        lines.push(format!("URL: {}", url));
    }
    let status = match (&request.last_reply_at, request.status.as_str()) {
        (Some(at), _) => format!("Status: {} (last reply {})", request.status, day(at)),
        (None, PENDING) => "Status: pending, no reply yet".to_string(),
        (None, status) => format!("Status: {}", status),
    };
    lines.push(status);
    if let Some(note) = &request.note {
        lines.push(format!("Note: {}", note));
    }
    lines.join("\n")
}

fn metadata(request: &ExternalRequest, details: Option<&str>, existing: Option<&Value>) -> Value {
    let mut object = match existing {
        Some(Value::Object(object)) => object.clone(),
        _ => Map::new(),
    };
    object.insert(
        METADATA_KEY.into(),
        json!({
            "destination": request.destination,
            "subject": request.subject,
            "url": request.url,
            "sent_at": request.sent_at,
            "status": request.status,
            "last_reply_at": request.last_reply_at,
            "note": request.note,
            "detected": request.detected,
            "details": details,
        }),
    );
    Value::Object(object)
}

/// The request a memory holds; `None` for any other memory.
fn request_of(memory: &Memory, now: chrono::DateTime<chrono::Utc>) -> Option<(ExternalRequest, Option<String>)> {
    if memory.kind != EXTERNAL_REQUEST_KIND {
        return None;
    }
    let fields = memory.metadata.as_ref()?.get(METADATA_KEY)?;
    let text = |key: &str| fields.get(key).and_then(Value::as_str).map(str::to_string);
    let status = text("status").unwrap_or_else(|| PENDING.to_string());
    let sent_at = text("sent_at")?;
    let waiting_days = (status == PENDING).then(|| instant(&sent_at).map(|at| (now - at).num_days().max(0))).flatten();
    Some((
        ExternalRequest {
            id: memory.id.clone(),
            project: memory.project.clone(),
            destination: text("destination")?,
            subject: text("subject")?,
            url: text("url"),
            sent_at,
            status,
            last_reply_at: text("last_reply_at"),
            note: text("note"),
            detected: text("detected"),
            waiting_days,
        },
        text("details"),
    ))
}

impl Database {
    /// Records something sent to a third party. Logging the same request
    /// again updates it instead of adding a second one.
    pub fn log_external_request(&self, new: &NewExternalRequest, scope: &MemoryScope) -> Result<ExternalRequest, String> {
        let destination = new.destination.trim();
        let subject = new.subject.trim();
        if destination.is_empty() || subject.is_empty() {
            return Err("destination and subject are required".into());
        }
        let now = chrono::Utc::now();
        let sent_at = match new.sent_at.map(str::trim).filter(|raw| !raw.is_empty()) {
            Some(raw) => checked_date("sent_at", raw)?,
            None => now.to_rfc3339(),
        };
        let request = ExternalRequest {
            id: String::new(),
            project: None,
            destination: destination.to_string(),
            subject: subject.to_string(),
            url: new.url.map(str::trim).filter(|url| !url.is_empty()).map(str::to_string),
            sent_at,
            status: PENDING.to_string(),
            last_reply_at: None,
            note: None,
            detected: new.detected.map(str::to_string),
            waiting_days: None,
        };
        let tags = vec![TAG.to_string()];
        let (memory, _) = self.add_memory(
            &content(&request, new.details),
            EXTERNAL_REQUEST_KIND,
            new.project,
            &tags,
            "external-request",
            IMPORTANCE,
            None,
            Some(&metadata(&request, new.details, None)),
            scope,
        )?;
        request_of(&memory, now).map(|(request, _)| request).ok_or_else(|| "External request not stored".to_string())
    }

    /// Records an answer or the end of a request. `status` alone moves it;
    /// `last_reply_at` without a status means it was answered.
    pub fn update_external_request(
        &self,
        id: &str,
        status: Option<&str>,
        last_reply_at: Option<&str>,
        note: Option<&str>,
    ) -> Result<Option<ExternalRequest>, String> {
        let now = chrono::Utc::now();
        let Some(memory) = self.get_memory_by_id_or_prefix(id)? else { return Ok(None) };
        let Some((mut request, details)) = request_of(&memory, now) else {
            return Err(format!("{} is not an external request", id));
        };
        if let Some(status) = status {
            if !EXTERNAL_STATUSES.contains(&status) {
                return Err(format!("status must be one of: {}", EXTERNAL_STATUSES.join(", ")));
            }
            request.status = status.to_string();
        }
        if let Some(raw) = last_reply_at.map(str::trim).filter(|raw| !raw.is_empty()) {
            request.last_reply_at = Some(checked_date("last_reply_at", raw)?);
            if status.is_none() && request.status == PENDING {
                request.status = "answered".to_string();
            }
        }
        if let Some(note) = note.map(str::trim).filter(|note| !note.is_empty()) {
            request.note = Some(note.to_string());
        }
        // Confirmed by whoever updates it.
        if status.is_some() || last_reply_at.is_some() {
            request.detected = request.detected.filter(|how| how != "session");
        }
        let updated = self.update_memory_full(
            &memory.id,
            Some(&content(&request, details.as_deref())),
            None,
            None,
            None,
            None,
            Some(&metadata(&request, details.as_deref(), memory.metadata.as_ref())),
        )?;
        Ok(updated.and_then(|memory| request_of(&memory, now)).map(|(request, _)| request))
    }

    /// Requests sent to third parties, the longest-waiting first; only
    /// those still without an answer with `pending_only`.
    pub fn external_requests(&self, project: Option<&str>, pending_only: bool) -> Result<Vec<ExternalRequest>, String> {
        let now = chrono::Utc::now();
        let (memories, _) = self.list_memories(project, Some(EXTERNAL_REQUEST_KIND), None, LISTED, 0)?;
        let mut requests: Vec<ExternalRequest> = memories
            .iter()
            .filter_map(|memory| request_of(memory, now))
            .map(|(request, _)| request)
            .filter(|request| !pending_only || request.status == PENDING)
            .collect();
        requests.sort_by(|a, b| a.sent_at.cmp(&b.sent_at).then(a.id.cmp(&b.id)));
        Ok(requests)
    }
}

impl Database {
    /// What the conversation index found typed on a site known for
    /// requests (`conversations::submissions`, plus the sites of the
    /// `external_hosts` setting) becomes an external request to confirm,
    /// once. Returns how many were added.
    pub fn record_submissions(&self, store: &crate::conversations::ConversationStore) -> Result<usize, String> {
        let mut hosts: Vec<String> =
            crate::conversations::submissions::DEFAULT_HOSTS.iter().map(|host| host.to_string()).collect();
        if let Some(more) = self.get_config("external_hosts") {
            hosts.extend(
                more.split(',').map(|host| host.trim().trim_start_matches("www.").to_lowercase()).filter(|host| !host.is_empty()),
            );
        }
        let mut added = 0;
        for submission in store.unreported_submissions(&hosts)? {
            let request = submission.request.split_whitespace().collect::<Vec<_>>().join(" ");
            let subject = crate::text::preview(&request, 140);
            let project = submission.project.as_deref().and_then(|project| self.resolve_project(project));
            let details = format!(
                "Found in the actions of conversation {}: text was typed on {}. Not confirmed: check it was sent, then update or close it.",
                submission.conversation_id, submission.host
            );
            let new = NewExternalRequest {
                destination: &submission.host,
                subject: if subject.is_empty() { "Request typed on this site" } else { &subject },
                url: Some(&submission.url),
                sent_at: submission.at.as_deref(),
                project: project.as_deref(),
                details: Some(&details),
                detected: Some("session"),
            };
            self.log_external_request(&new, &MemoryScope::default())?;
            store.mark_submission_reported(&submission.source_key)?;
            added += 1;
        }
        Ok(added)
    }
}

/// One line per request still waiting, for the recall of a session.
pub(super) fn waiting_lines(requests: &[ExternalRequest], max: usize) -> Vec<String> {
    requests
        .iter()
        .take(max)
        .map(|request| {
            let waited = match request.waiting_days {
                Some(0) | None => "sent today".to_string(),
                Some(1) => "1 day without a reply".to_string(),
                Some(days) => format!("{} days without a reply", days),
            };
            let unconfirmed = if request.detected.as_deref() == Some("session") { ", to confirm" } else { "" };
            format!(
                "[EXT] {}: {} ({}{}) [id:{}]",
                request.destination,
                request.subject,
                waited,
                unconfirmed,
                crate::text::prefix(&request.id, 8)
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn open(name: &str) -> (Database, std::path::PathBuf) {
        let dir = std::env::temp_dir().join(format!("mp-external-{}-{}", name, uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).expect("dir");
        (Database::open_at(&dir.join("memory.db")).expect("db"), dir)
    }

    fn forum_post<'a>(sent_at: &'a str) -> NewExternalRequest<'a> {
        NewExternalRequest {
            destination: "forum développeurs Meta",
            subject: "Musique sur une publication Instagram programmée",
            url: Some("https://developers.facebook.com/community/threads/1234567890/"),
            sent_at: Some(sent_at),
            project: Some("monapp"),
            details: Some("Demande : comment joindre une musique à un post programmé par l'API, pas seulement à un reel."),
            detected: None,
        }
    }

    #[test]
    fn a_forum_post_is_listed_as_waiting_then_found_by_a_french_question() {
        let (db, dir) = open("replay");
        let sent = (chrono::Utc::now() - chrono::Duration::days(37)).format("%Y-%m-%d").to_string();
        let request = db.log_external_request(&forum_post(&sent), &MemoryScope::default()).expect("log");
        assert_eq!((request.status.as_str(), request.waiting_days), ("pending", Some(37)));
        assert_eq!(request.project.as_deref(), Some("monapp"));
        db.add_memory(
            "Le carrousel plante quand le conteneur n'est pas prêt au moment de publier.",
            "bug",
            Some("monapp"),
            &[],
            "test",
            3,
            None,
            None,
            &MemoryScope::default(),
        )
        .expect("other memory");

        // "What is waiting for an answer?" in one call.
        let waiting = db.external_requests(Some("monapp"), true).expect("pending");
        assert_eq!(waiting.len(), 1);
        assert_eq!(waiting[0].destination, "forum développeurs Meta");
        assert!(db.external_requests(Some("other"), true).expect("other project").is_empty());

        // The question that was asked, in French, without the word "forum".
        let found = db
            .search("a-t-on eu un retour de Meta sur la musique d'une publication programmée", 5, Some("monapp"), None, None, None)
            .expect("search");
        assert_eq!(found.first().map(|result| result.memory.id.as_str()), Some(request.id.as_str()), "{:?}", found.iter().map(|r| &r.memory.content).collect::<Vec<_>>());
        assert!(found[0].memory.content.contains("pending, no reply yet"));

        // An answer came: it leaves the waiting list and says when.
        let answered = db.update_external_request(&request.id, None, Some("2026-11-20"), Some("Meta: not supported for scheduled posts")).expect("update").expect("found");
        assert_eq!((answered.status.as_str(), answered.waiting_days), ("answered", None));
        assert!(db.external_requests(Some("monapp"), true).expect("pending").is_empty());
        let all = db.external_requests(None, false).expect("all");
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].last_reply_at.as_deref(), Some("2026-11-20"));
        assert!(db.get_memory(&request.id).expect("get").expect("memory").content.contains("last reply 2026-11-20"));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_post_typed_in_a_past_session_becomes_a_request_to_confirm_once() {
        use crate::conversations::{ConversationStore, Message, Role, SourceConversation};
        let (db, dir) = open("session");
        let mut store = ConversationStore::open(&dir.join("conversations.db")).expect("store");
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
        let turn = [
            Message {
                key: "u".to_string(),
                role: Role::User,
                at: Some("2026-10-12T10:00:00.000Z".to_string()),
                order: None,
                text: "Poste la question sur la musique d'une publication programmée dans leur communauté".to_string(),
                actions: Vec::new(),
            },
            Message {
                key: "a".to_string(),
                role: Role::Assistant,
                at: Some("2026-10-12T10:05:00.000Z".to_string()),
                order: None,
                text: String::new(),
                actions: vec![
                    "navigate: https://developers.facebook.com/community/threads/new/".to_string(),
                    "form_input: Music on a scheduled Instagram post".to_string(),
                ],
            },
        ];
        store.apply(&source, true, &turn, &[]).expect("apply");

        assert_eq!(db.record_submissions(&store).expect("record"), 1);
        let waiting = db.external_requests(None, true).expect("pending");
        assert_eq!(waiting.len(), 1);
        assert_eq!(waiting[0].destination, "developers.facebook.com");
        assert_eq!(waiting[0].detected.as_deref(), Some("session"));
        assert_eq!(waiting[0].sent_at, "2026-10-12T10:05:00.000Z");
        assert!(waiting[0].subject.starts_with("Poste la question sur la musique"));
        assert!(waiting_lines(&waiting, 5)[0].contains(", to confirm)"));

        // Closed by the user, it is not proposed again at the next indexing.
        db.update_external_request(&waiting[0].id, Some("closed"), None, None).expect("close");
        assert_eq!(db.record_submissions(&store).expect("again"), 0);
        assert_eq!(db.external_requests(None, false).expect("all").len(), 1);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_request_logged_twice_is_one_request_and_bad_input_is_refused() {
        let (db, dir) = open("twice");
        let first = db.log_external_request(&forum_post("2026-10-12"), &MemoryScope::default()).expect("first");
        let second = db.log_external_request(&forum_post("2026-10-12"), &MemoryScope::default()).expect("second");
        assert_eq!(first.id, second.id);
        assert_eq!(db.external_requests(None, false).expect("all").len(), 1);

        let missing = NewExternalRequest { destination: " ", subject: "x", ..Default::default() };
        assert!(db.log_external_request(&missing, &MemoryScope::default()).is_err());
        assert!(db.log_external_request(&forum_post("12/10/2026"), &MemoryScope::default()).is_err());
        assert!(db.update_external_request(&first.id, Some("lost"), None, None).is_err());
        assert_eq!(db.update_external_request("no-such-id", Some("closed"), None, None).expect("unknown"), None);
        let closed = db.update_external_request(&first.id, Some("closed"), None, None).expect("close").expect("found");
        assert_eq!(closed.status, "closed");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn the_waiting_requests_of_a_project_open_its_recall() {
        let (db, dir) = open("recall");
        let sent = (chrono::Utc::now() - chrono::Duration::days(3)).format("%Y-%m-%d").to_string();
        let request = db.log_external_request(&forum_post(&sent), &MemoryScope::default()).expect("log");
        let recall = db
            .recall(Some("monapp"), None, None, RecallMode::Safe, false, true, &MemoryScope::default())
            .expect("recall");
        let compact = recall["compact"].as_str().expect("compact");
        assert!(compact.contains("--- waiting on third parties ---"), "{compact}");
        assert!(compact.contains("[EXT] forum développeurs Meta: Musique sur une publication Instagram programmée (3 days without a reply)"), "{compact}");

        db.update_external_request(&request.id, Some("closed"), None, None).expect("close");
        let recall = db
            .recall(Some("monapp"), None, None, RecallMode::Safe, false, true, &MemoryScope::default())
            .expect("recall");
        assert!(!recall["compact"].as_str().expect("compact").contains("waiting on third parties"));
        let _ = std::fs::remove_dir_all(dir);
    }
}
