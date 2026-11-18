//! Claims that something never happened, kept apart from checked facts.
//!
//! "No request was ever sent to them" is written after a search that
//! found nothing, which proves little: the request may have been made in
//! a session that was never recorded. Stored as a plain fact, it then
//! comes back first and is believed. A memory that states such an
//! absence is marked `unverified` when it is written, ranks lower, and
//! says so wherever it is shown, until someone confirms it
//! (`resolve_memory` with status `verified`).

use super::{Memory, SearchResult};
use serde_json::{json, Map, Value};

/// The status of a claim nobody confirmed, in `metadata.status`.
pub const UNVERIFIED: &str = "unverified";
/// Kinds that state what is: a rule ("never commit a key") or a
/// decision is not a claim about the past.
const CLAIM_KINDS: &[&str] = &["fact", "note", "milestone"];
/// What an unconfirmed absence keeps of its rank.
const UNVERIFIED_WEIGHT: f64 = 0.6;

/// Words that deny, per language, and the auxiliaries that make the
/// sentence a statement about what happened rather than an instruction:
/// "n'a jamais été envoyé" has both, "ne jamais utiliser" and "aucune
/// erreur" have one.
const FRENCH_DENIALS: &[&str] = &["aucun", "aucune", "jamais", "rien", "personne"];
/// "n'a", not "n'est": "aucune erreur n'est apparue" describes a state,
/// "aucune demande n'a été envoyée" denies an event.
const FRENCH_AUXILIARIES: &[&str] = &["n'a", "n'ont", "n'ai", "n'avons", "n'avait", "n'avaient", "n'y a", "n'y avait"];
/// "Never" next to its auxiliary: "the API never returned an error" tells
/// how something behaves, "it was never sent" denies an event.
const ENGLISH_NEVER: &[&str] =
    &["was never", "were never", "has never", "have never", "had never", "never been", "never was", "never did"];
const ENGLISH_NONE: &[&str] = &["no", "nothing", "nobody", "no one", "none"];
const ENGLISH_NONE_AUXILIARIES: &[&str] = &["was", "were", "has been", "have been", "had been", "ever"];

/// Where `phrase` stands in `sentence` as whole words: "no" is in "no
/// request" but not in "another".
fn word_at(sentence: &str, phrase: &str) -> Option<usize> {
    let is_word = |c: char| c.is_alphanumeric() || c == '\'';
    sentence.match_indices(phrase).map(|(start, _)| start).find(|start| {
        let before = sentence[..*start].chars().next_back();
        let after = sentence[start + phrase.len()..].chars().next();
        !before.is_some_and(is_word) && !after.is_some_and(|c| c.is_alphanumeric())
    })
}

fn has_any(sentence: &str, phrases: &[&str]) -> bool {
    phrases.iter().any(|phrase| word_at(sentence, phrase).is_some())
}

/// "No request was sent" denies an event; "there was no error" describes
/// a result: the denial has to come before the auxiliary.
fn denies_before(sentence: &str, denials: &[&str], auxiliaries: &[&str]) -> bool {
    let first_denial = denials.iter().filter_map(|denial| word_at(sentence, denial)).min();
    let last_auxiliary = auxiliaries.iter().filter_map(|auxiliary| word_at(sentence, auxiliary)).max();
    matches!((first_denial, last_auxiliary), (Some(denial), Some(auxiliary)) if denial < auxiliary)
}

/// Whether `content` states, in French or English, that something never
/// happened or does not exist.
pub fn claims_an_absence(content: &str) -> bool {
    let text = content.to_lowercase().replace(['\u{2019}', '\u{02BC}'], "'");
    text.split(['.', '!', '?', '\n', ';', ':']).any(|sentence| {
        (has_any(sentence, FRENCH_DENIALS) && has_any(sentence, FRENCH_AUXILIARIES))
            || has_any(sentence, ENGLISH_NEVER)
            || denies_before(sentence, ENGLISH_NONE, ENGLISH_NONE_AUXILIARIES)
    })
}

/// The metadata a new memory is stored with: `metadata` plus the
/// `unverified` status when the memory claims an absence and its writer
/// gave no status of its own. `None` when nothing is to be added.
pub fn metadata_for_claim(kind: &str, content: &str, metadata: Option<&Value>) -> Option<Value> {
    if !CLAIM_KINDS.contains(&kind) || metadata.and_then(|m| m.get("status")).is_some() || !claims_an_absence(content) {
        return None;
    }
    let mut object = match metadata {
        Some(Value::Object(object)) => object.clone(),
        _ => Map::new(),
    };
    object.insert("status".into(), json!(UNVERIFIED));
    object.insert(
        "status_note".into(),
        json!("States that something never happened or does not exist; nobody confirmed it."),
    );
    Some(Value::Object(object))
}

pub fn is_unverified(memory: &Memory) -> bool {
    memory.metadata.as_ref().and_then(|m| m.get("status")).and_then(Value::as_str) == Some(UNVERIFIED)
}

/// What a reader is told about an unverified memory.
pub fn unverified_report() -> Value {
    json!({
        "status": UNVERIFIED,
        "reason": "States that something never happened or does not exist, and nobody confirmed it. A search that finds nothing does not prove it: check the source before relying on it, then resolve_memory with status verified, or correct the memory."
    })
}

/// Lowers the unconfirmed absences in a ranked list. It runs after the
/// reranker, whose scores replace everything computed before it.
pub(super) fn demote_unverified(results: &mut [SearchResult]) {
    if !results.iter().any(|result| is_unverified(&result.memory)) {
        return;
    }
    for result in results.iter_mut().filter(|result| is_unverified(&result.memory)) {
        result.score = (result.score * UNVERIFIED_WEIGHT * 10000.0).round() / 10000.0;
    }
    results.sort_by(|a, b| b.score.total_cmp(&a.score));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_absence_is_a_statement_about_what_happened() {
        for claim in [
            "Musique Instagram : aucune demande n'a jamais été envoyée à Meta, donc aucun retour à attendre.",
            "Rien n\u{2019}a été posté sur le forum des développeurs.",
            "Personne n'a répondu au rapport de bug.",
            "No request was ever sent to the platform team.",
            "The form has never been submitted.",
            "Nothing was posted on the developer forum.",
        ] {
            assert!(claims_an_absence(claim), "{claim}");
        }
        for other in [
            "Ne jamais utiliser wrangler pour déployer.",
            "Never commit a key to the repository.",
            "Build vérifié : 0 erreur, aucune erreur de type.",
            "La demande a été envoyée à Meta le 12/10, réponse attendue.",
            "There is no rate limit on this endpoint.",
            "The build was green and there was no error.",
            "Another change was made, no issue.",
            "Aucune erreur n'est apparue au build.",
            "Aucune migration n'est en attente.",
            "Tous les tests passent, aucun n'est ignoré.",
            "The API never returned an error, as the logs were clean.",
            "Le client n'a pas encore choisi la formule annuelle.",
        ] {
            assert!(!claims_an_absence(other), "{other}");
        }
    }

    fn ranked(id: &str, score: f64, status: Option<&str>) -> SearchResult {
        SearchResult {
            memory: Memory {
                id: id.to_string(),
                content: String::new(),
                kind: "fact".to_string(),
                project: None,
                tags: Vec::new(),
                source: "test".to_string(),
                importance: 3,
                expires_at: None,
                created_at: String::new(),
                updated_at: String::new(),
                metadata: status.map(|status| json!({ "status": status })),
                last_accessed_at: None,
                access_count: 0,
            },
            score,
            sources: Vec::new(),
        }
    }

    #[test]
    fn an_unconfirmed_absence_ranks_below_a_close_fact() {
        let mut results =
            vec![ranked("absence", 1.0, Some(UNVERIFIED)), ranked("fact", 0.8, None), ranked("far", 0.3, None)];
        demote_unverified(&mut results);
        let order: Vec<&str> = results.iter().map(|result| result.memory.id.as_str()).collect();
        assert_eq!(order, vec!["fact", "absence", "far"], "lowered, not hidden");
        assert!(results.windows(2).all(|pair| pair[0].score >= pair[1].score));

        // Once confirmed it keeps its rank, and a list without any is untouched.
        let mut confirmed = vec![ranked("absence", 1.0, Some("verified")), ranked("fact", 0.8, None)];
        demote_unverified(&mut confirmed);
        assert_eq!(confirmed[0].memory.id, "absence");
        assert_eq!(confirmed[0].score, 1.0);
    }

    #[test]
    fn only_a_claim_without_a_status_is_marked() {
        let claim = "Aucune demande n'a jamais été envoyée à Meta.";
        let marked = metadata_for_claim("fact", claim, Some(&json!({ "location": "forum" }))).expect("marked");
        assert_eq!(marked["status"], UNVERIFIED);
        assert_eq!(marked["location"], "forum");
        assert!(metadata_for_claim("preference", claim, None).is_none(), "a rule is not a claim");
        assert!(metadata_for_claim("fact", claim, Some(&json!({ "status": "verified" }))).is_none());
        assert!(metadata_for_claim("fact", "La demande est partie le 12/10.", None).is_none());
    }
}
