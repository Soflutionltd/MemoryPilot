//! Real failures, replayed.
//!
//! The public retrieval benchmarks ask for a passage in a conversation
//! that was recorded whole. What goes wrong in daily use lies around
//! that: a request sent to a third party that nobody noted, a session
//! that was never indexed, a project known under three names, a French
//! question about an English post, "nothing was ever sent" stored as a
//! fact. Each case here is one such failure, set up in a fresh database
//! and a fresh conversation index, with the real models. A case is added
//! whenever a new failure is found in use.

use super::*;
use crate::conversations::{self, ConversationStore, Message, Role, SourceConversation};
use serde_json::{json, Value};

type Case = fn(&Database, &mut ConversationStore) -> Result<String, String>;

const CASES: &[(&str, Case)] = &[
    ("a request sent to a third party is listed as waiting and found by a French question", request_is_found),
    ("an unconfirmed 'nothing was sent' does not outrank the fact", absence_ranks_below_fact),
    ("a project is one project under its other names", project_under_its_names),
    ("a word is found inside a long conversation without paging", word_in_long_conversation),
    ("weeks missing from a conversation are reported", hole_is_reported),
    ("a draft written mid-turn and a site visited are found", draft_and_site_are_found),
    ("a post typed in a past session becomes a request to confirm", past_post_is_recorded),
    ("a French question finds an English post", french_finds_english),
];

fn expect(condition: bool, failure: impl FnOnce() -> String) -> Result<(), String> {
    if condition {
        Ok(())
    } else {
        Err(failure())
    }
}

fn remember(db: &Database, content: &str, kind: &str, project: &str, metadata: Option<&Value>) -> Result<Memory, String> {
    db.add_memory(content, kind, Some(project), &[], "usage-bench", 3, None, metadata, &MemoryScope::default())
        .map(|(memory, _)| memory)
}

fn said(store: &mut ConversationStore, id: &str, messages: Vec<Message>) -> Result<(), String> {
    let source = SourceConversation {
        origin_id: id.to_string(),
        root_id: id.to_string(),
        tool: "cursor",
        title: None,
        workspace: None,
        created_at: None,
        version: messages.len().to_string(),
        source_path: None,
    };
    store.apply(&source, true, &messages, &[]).map(|_| ())
}

fn message(key: &str, role: Role, at: &str, text: &str, actions: &[&str]) -> Message {
    Message {
        key: key.to_string(),
        role,
        at: Some(at.to_string()),
        order: None,
        text: text.to_string(),
        actions: actions.iter().map(|action| action.to_string()).collect(),
    }
}

fn request_is_found(db: &Database, _: &mut ConversationStore) -> Result<String, String> {
    for other in [
        "Le carrousel plante quand le conteneur n'est pas prêt au moment de publier.",
        "Les reels ont besoin d'une image de couverture avant d'être programmés.",
        "Décision : les publications programmées passent par la file d'attente du serveur.",
        "La bibliothèque audio a été testée dans l'explorateur de l'API le 12/10.",
    ] {
        remember(db, other, "fact", "monapp", None)?;
    }
    let sent = (chrono::Utc::now() - chrono::Duration::days(37)).format("%Y-%m-%d").to_string();
    let request = db.log_external_request(
        &NewExternalRequest {
            destination: "forum développeurs Meta",
            subject: "Musique sur une publication Instagram programmée",
            url: Some("https://developers.facebook.com/community/threads/1234567890/"),
            sent_at: Some(&sent),
            project: Some("monapp"),
            details: Some("Demande : comment joindre une musique à un post programmé par l'API, pas seulement à un reel."),
            detected: None,
        },
        &MemoryScope::default(),
    )?;
    wait_for_vectors(db)?;
    let waiting = db.external_requests(Some("monapp"), true)?;
    expect(waiting.iter().any(|listed| listed.id == request.id && listed.waiting_days == Some(37)), || {
        format!("not in the waiting list: {:?}", waiting)
    })?;
    let found = db.search("a-t-on eu un retour de Meta sur la musique d'une publication programmée", 5, Some("monapp"), None, None, None)?;
    expect(found.first().is_some_and(|top| top.memory.id == request.id), || {
        format!("first result is not the request: {:?}", found.iter().map(|r| crate::text::preview(&r.memory.content, 60)).collect::<Vec<_>>())
    })?;
    Ok("listed after 37 days, first result of the French question".to_string())
}

fn absence_ranks_below_fact(db: &Database, _: &mut ConversationStore) -> Result<String, String> {
    let claim = "Musique des stories : aucune demande n'a jamais été envoyée au support de la plateforme vidéo.";
    let metadata = metadata_for_claim("fact", claim, None);
    let absence = remember(db, claim, "fact", "monsite", metadata.as_ref())?;
    let fact = remember(
        db,
        "Musique des stories : la demande au support de la plateforme vidéo a été envoyée le 3 octobre, ticket ouvert.",
        "fact",
        "monsite",
        None,
    )?;
    wait_for_vectors(db)?;
    expect(is_unverified(&absence), || "the absence was not marked unverified".to_string())?;
    let found = db.search("a-t-on envoyé une demande au support pour la musique des stories", 5, Some("monsite"), None, None, None)?;
    let rank = |id: &str| found.iter().position(|result| result.memory.id == id);
    expect(matches!((rank(&fact.id), rank(&absence.id)), (Some(fact), Some(absence)) if fact < absence), || {
        format!("fact at {:?}, absence at {:?}", rank(&fact.id), rank(&absence.id))
    })?;
    Ok("marked unverified, ranked after the fact, still shown".to_string())
}

fn project_under_its_names(db: &Database, _: &mut ConversationStore) -> Result<String, String> {
    remember(db, "Le planificateur publie les carrousels par le point d'accès conteneur.", "fact", "appli-planning", None)?;
    db.add_name_alias("Planner", "appli-planning")?;
    let resolved = db.resolve_project("planner");
    expect(resolved.as_deref() == Some("appli-planning"), || format!("planner resolves to {:?}", resolved))?;
    let (listed, _) = db.list_memories(resolved.as_deref(), None, None, 10, 0)?;
    expect(listed.len() == 1, || format!("{} memories under the alias", listed.len()))?;
    let refused = db.add_name_alias("appli-planning", "monapp");
    expect(refused.is_err(), || "a name holding memories was accepted as an alias".to_string())?;
    Ok("alias resolved on read; a name holding memories has to be merged".to_string())
}

fn word_in_long_conversation(_: &Database, store: &mut ConversationStore) -> Result<String, String> {
    let mut messages = Vec::new();
    for index in 0..400 {
        let text = if index == 257 {
            "Écris le message sur la musique des publications pour le forum des développeurs".to_string()
        } else {
            format!("Corrige la marge du bloc numéro {} dans la page des réglages du compte", index)
        };
        let at = format!("2026-10-{:02}T{:02}:{:02}:00.000Z", 1 + index / 60, 8 + (index % 60) / 6, (index % 6) * 10);
        messages.push(message(&format!("m{}", index), Role::User, &at, &text, &[]));
    }
    said(store, "long", messages)?;
    let found = store.search_within("long", "musique forum", 5, false)?.ok_or("conversation not indexed")?;
    expect(found.len() == 1 && found[0].snippet.contains("musique"), || format!("{} messages returned", found.len()))?;
    Ok("1 message out of 400 returned, with its id and date".to_string())
}

fn hole_is_reported(_: &Database, store: &mut ConversationStore) -> Result<String, String> {
    let days = ["2026-10-19", "2026-10-20", "2026-10-21", "2026-11-07", "2026-11-08"];
    let messages = days
        .iter()
        .enumerate()
        .map(|(index, day)| {
            message(&format!("d{}", index), Role::User, &format!("{}T10:00:00.000Z", day), "Continue le travail sur la publication programmée du compte", &[])
        })
        .collect();
    said(store, "holed", messages)?;
    let warnings = store.coverage_warnings(Some("holed"), chrono::Utc::now());
    expect(warnings.iter().any(|warning| warning.contains("between 2026-10-21 and 2026-11-07 (17 days)")), || {
        format!("no warning for the 17 days: {:?}", warnings)
    })?;
    Ok("17 days without a message reported when the conversation is searched".to_string())
}

fn posting_turn(id: &str) -> Vec<Message> {
    let draft = format!(
        "Here is the post for the developer community: music cannot be attached to a scheduled post through the API. {}",
        "We tried the audio endpoint and the container flow without success. ".repeat(4)
    );
    vec![
        message(&format!("{}-u", id), Role::User, "2026-10-12T10:00:00.000Z", "Poste la question sur la musique dans leur communauté de développeurs", &[]),
        message(&format!("{}-a1", id), Role::Assistant, "2026-10-12T10:02:00.000Z", &draft, &[]),
        message(
            &format!("{}-a2", id),
            Role::Assistant,
            "2026-10-12T10:05:00.000Z",
            "",
            &["navigate: https://developers.facebook.com/community/threads/new/", "form_input: Music on a scheduled Instagram post", "form_input: A post scheduled through the content publishing API cannot carry a music track. Is there a supported way to attach one?"],
        ),
        message(&format!("{}-a3", id), Role::Assistant, "2026-10-12T10:06:00.000Z", "C'est posté dans leur communauté, je surveille la réponse.", &[]),
    ]
}

fn draft_and_site_are_found(_: &Database, store: &mut ConversationStore) -> Result<String, String> {
    said(store, "posting", posting_turn("p"))?;
    let draft = store.search_within("posting", "audio endpoint container flow", 5, false)?.unwrap_or_default();
    expect(draft.first().is_some_and(|found| found.role == "step"), || format!("the draft was not found: {:?}", draft))?;
    let by_site = store.by_reference("url", "developers.facebook.com", 5)?;
    expect(by_site.iter().any(|(conversation, _)| conversation == "posting"), || format!("not found by its site: {:?}", by_site))?;
    Ok("the draft is searchable, the conversation is found by the site it went to".to_string())
}

fn past_post_is_recorded(db: &Database, store: &mut ConversationStore) -> Result<String, String> {
    // Reads the turn of the previous case, as an indexing run would.
    let added = db.record_submissions(store)?;
    expect(added == 1, || format!("{} requests recorded", added))?;
    let waiting = db.external_requests(None, true)?;
    let found = waiting.iter().find(|request| request.detected.as_deref() == Some("session"));
    expect(found.is_some_and(|request| request.destination == "developers.facebook.com"), || format!("{:?}", waiting))?;
    expect(db.record_submissions(store)? == 0, || "recorded twice".to_string())?;
    Ok("recorded once, to confirm, with its date and address".to_string())
}

fn french_finds_english(_: &Database, store: &mut ConversationStore) -> Result<String, String> {
    said(
        store,
        "english",
        vec![message(
            "e-u",
            Role::User,
            "2026-10-14T09:00:00.000Z",
            "Reply to the thread: we still cannot add music to a scheduled Instagram post, only to reels",
            &[],
        )],
    )?;
    conversations::embed_pending(store, None)?;
    let options = conversations::SearchOptions {
        query: "musique sur une publication programmée, réponse au fil de discussion".to_string(),
        limit: 3,
        use_vectors: true,
        ..Default::default()
    };
    let found = store.search(&options)?;
    // Conversations holding the very words of the question, in French,
    // are in the index too: the English one has to be among the three shown.
    let rank = found.iter().position(|hit| hit.conversation_id == "english");
    expect(rank.is_some(), || {
        format!("rank {:?} among {:?}", rank, found.iter().map(|hit| hit.conversation_id.as_str()).collect::<Vec<_>>())
    })?;
    Ok(format!("found at rank {}", rank.unwrap_or(0) + 1))
}

/// Blocks until every memory of the bench database has its vector.
fn wait_for_vectors(db: &Database) -> Result<(), String> {
    let started = std::time::Instant::now();
    loop {
        let pending: i64 = db
            .conn
            .query_row("SELECT COUNT(*) FROM memories WHERE embedding IS NULL", [], |row| row.get(0))
            .unwrap_or(0);
        if pending == 0 {
            return Ok(());
        }
        if started.elapsed().as_secs() > 120 {
            return Err(format!("timeout waiting for {} vectors", pending));
        }
        std::thread::sleep(std::time::Duration::from_millis(200));
    }
}

impl Database {
    /// Replays the real failures in a fresh database and a fresh
    /// conversation index. Every case reports what it checked or why it
    /// failed.
    pub fn benchmark_usage() -> Result<Value, String> {
        let dir = std::env::temp_dir().join(format!(
            "memorypilot-usage-bench-{}-{}",
            std::process::id(),
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0)
        ));
        std::fs::create_dir_all(&dir).map_err(|error| format!("Usage bench: {}", error))?;
        let outcome = (|| {
            let db = Database::open_at(&dir.join("memory.db"))?;
            let mut store = ConversationStore::open(&dir.join("conversations.db"))?;
            let cases: Vec<Value> = CASES
                .iter()
                .map(|(name, case)| match case(&db, &mut store) {
                    Ok(detail) => json!({ "case": name, "passed": true, "detail": detail }),
                    Err(failure) => json!({ "case": name, "passed": false, "failure": failure }),
                })
                .collect();
            let passed = cases.iter().filter(|case| case["passed"] == true).count();
            Ok(json!({ "benchmark": "usage-v1", "passed": passed, "total": cases.len(), "cases": cases }))
        })();
        let _ = std::fs::remove_dir_all(&dir);
        outcome
    }
}
