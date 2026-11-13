//! 1Password through its command-line tool. Secret values only travel on its
//! stdin/stdout, never on a command line.
//!
//! Which vault MemoryPilot may use, and which command to run, are the user's:
//! `~/.MemoryPilot/config/onepassword.json` (`{"vault": "...", "cli": "..."}`)
//! or the `MEMORYPILOT_OP_VAULT` / `MEMORYPILOT_OP_CLI` variables. `cli` names
//! another binary on the PATH, such as a wrapper that signs in a service
//! account; without it the standard one is used. Nothing is assumed about
//! the vault.

use serde_json::{json, Value};
use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;

use crate::connectors::{find_binary, run_with_timeout};

const SETTINGS_FILE: &str = "onepassword.json";
const DEFAULT_CLI: &str = "op";
pub const IMPORT_TAG: &str = "memorypilot-import";
pub const KEY_ITEM_TITLE: &str = "MemoryPilot vault key";
const LIST_TIMEOUT: Duration = Duration::from_secs(240);
const ITEM_TIMEOUT: Duration = Duration::from_secs(30);

/// One setting: its environment variable, else the key of `onepassword.json`.
fn setting(variable: &str, key: &str) -> Option<String> {
    let from_file = || {
        let path = crate::layout::path(crate::layout::Area::Config, SETTINGS_FILE)?;
        let settings: Value = serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()?;
        settings[key].as_str().map(String::from)
    };
    std::env::var(variable)
        .ok()
        .or_else(from_file)
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

/// The vault MemoryPilot reads and creates items in.
pub fn vault() -> Result<String, String> {
    setting("MEMORYPILOT_OP_VAULT", "vault").ok_or_else(|| {
        "No 1Password vault chosen: set MEMORYPILOT_OP_VAULT, or \"vault\" in ~/.MemoryPilot/config/onepassword.json".to_string()
    })
}

/// The configured command when it is on the PATH, else the standard one.
fn binary() -> Option<PathBuf> {
    setting("MEMORYPILOT_OP_CLI", "cli")
        .and_then(|cli| find_binary(&cli))
        .or_else(|| find_binary(DEFAULT_CLI))
}

fn op(args: &[&str], stdin: Option<&str>, timeout: Duration) -> Result<String, String> {
    let binary = binary().ok_or("1Password CLI not found (the standard one, or the \"cli\" of onepassword.json)")?;
    let mut command = Command::new(binary);
    command.args(args);
    run_with_timeout(command, stdin, timeout)
        .ok_or_else(|| format!("op {} failed or timed out", args.first().unwrap_or(&"")))
}

/// Every item of the vault, fields included (kept in memory, never printed).
pub fn vault_items(vault: &str) -> Result<Vec<Value>, String> {
    let list = op(
        &["item", "list", "--vault", vault, "--format", "json"],
        None,
        LIST_TIMEOUT,
    )?;
    let full = op(
        &["item", "get", "--vault", vault, "--format", "json", "-"],
        Some(&list),
        LIST_TIMEOUT,
    )?;
    serde_json::Deserializer::from_str(&full)
        .into_iter::<Value>()
        .map(|item| item.map_err(|e| format!("op item get output: {}", e)))
        .collect()
}

/// Whether an item is about the account a text describes: its title or one of
/// its URL hosts appears in the text. A reused password alone proves nothing.
fn describes(item: &Value, text: &str) -> bool {
    let text = text.to_lowercase();
    let title = item["title"].as_str().unwrap_or("").trim().to_lowercase();
    let hosts = item["urls"].as_array().into_iter().flatten().filter_map(|url| {
        let href = url["href"].as_str()?;
        let host = href.split("://").nth(1).unwrap_or(href).split(['/', '?', '#']).next()?;
        Some(host.trim_start_matches("www.").to_lowercase())
    });
    (title.chars().count() >= 4 && text.contains(&title))
        || hosts.into_iter().any(|host| host.contains('.') && text.contains(&host))
}

/// Reference to the first concealed field holding `value` in an item about
/// the account `text` describes.
pub fn find_value(vault: &str, items: &[Value], value: &str, text: &str) -> Option<String> {
    items.iter().filter(|item| describes(item, text)).find_map(|item| {
        let id = item["id"].as_str()?;
        item["fields"].as_array()?.iter().find_map(|field| {
            (field["type"] == "CONCEALED" && field["value"].as_str() == Some(value))
                .then(|| field["id"].as_str().map(|field_id| reference(vault, id, field_id)))
                .flatten()
        })
    })
}

pub fn find_title(items: &[Value], title: &str) -> Option<String> {
    items
        .iter()
        .find(|item| item["title"].as_str() == Some(title))
        .and_then(|item| item["id"].as_str())
        .map(String::from)
}

pub fn reference(vault: &str, item_id: &str, field: &str) -> String {
    format!("op://{}/{}/{}", vault, item_id, field)
}

fn first_email(text: &str) -> Option<&str> {
    text.split_whitespace()
        .map(|token| {
            token.trim_matches(|c: char| {
                !c.is_alphanumeric() && c != '@' && c != '.' && c != '_' && c != '-' && c != '+'
            })
        })
        .map(|token| token.trim_end_matches('.'))
        .find(|token| {
            token
                .split_once('@')
                .is_some_and(|(user, domain)| !user.is_empty() && domain.contains('.'))
        })
}

fn first_url(text: &str) -> Option<&str> {
    text.split_whitespace()
        .find(|token| token.starts_with("https://") || token.starts_with("http://"))
        .map(|token| token.trim_end_matches(['.', ',', ';', ')']))
}

/// Item template for a credential: a Login when a secret value is found
/// (password field), a Secure Note otherwise. The full text always goes in
/// the notes, so a wrong guess never loses anything. Returns the template and
/// the field the pointer should name.
pub fn item_template(
    title: &str,
    project: Option<&str>,
    text: &str,
    secret: Option<&str>,
) -> (Value, &'static str) {
    let mut tags = vec![IMPORT_TAG.to_string()];
    if let Some(project) = project {
        tags.push(project.to_string());
    }
    let notes = json!({ "id": "notesPlain", "type": "STRING", "purpose": "NOTES", "label": "notesPlain", "value": text });
    let Some(secret) = secret else {
        return (
            json!({ "title": title, "category": "SECURE_NOTE", "tags": tags, "fields": [notes] }),
            "notesPlain",
        );
    };
    let mut template = json!({
        "title": title,
        "category": "LOGIN",
        "tags": tags,
        "fields": [
            { "id": "username", "type": "STRING", "purpose": "USERNAME", "label": "username", "value": first_email(text).unwrap_or("") },
            { "id": "password", "type": "CONCEALED", "purpose": "PASSWORD", "label": "password", "value": secret },
            notes,
        ],
    });
    if let Some(url) = first_url(text) {
        template["urls"] = json!([{ "href": url, "primary": true }]);
    }
    (template, "password")
}

/// Creates the item and returns its id.
pub fn create_item(vault: &str, template: &Value) -> Result<String, String> {
    let created = op(
        &["item", "create", "--vault", vault, "--format", "json", "-"],
        Some(&template.to_string()),
        ITEM_TIMEOUT,
    )?;
    let created: Value =
        serde_json::from_str(&created).map_err(|e| format!("op item create output: {}", e))?;
    created["id"]
        .as_str()
        .map(String::from)
        .ok_or_else(|| "op item create returned no id".into())
}

/// Value behind an `op://` reference.
pub fn read(reference: &str) -> Result<String, String> {
    if !reference.starts_with("op://") {
        return Err(format!("not a 1Password reference: {}", reference));
    }
    op(&["read", "--no-newline", reference], None, ITEM_TIMEOUT)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn templates_put_the_secret_in_the_password_and_the_text_in_the_notes() {
        let text = "Admin demo https://demo.example.com : demo@example.com / password: Hx9!kLm2pQ";
        let (template, field) = item_template("Admin demo", Some("demo"), text, Some("Hx9!kLm2pQ"));
        assert_eq!(field, "password");
        assert_eq!(template["category"], "LOGIN");
        assert_eq!(template["fields"][0]["value"], "demo@example.com");
        assert_eq!(template["fields"][1]["value"], "Hx9!kLm2pQ");
        assert_eq!(template["fields"][2]["value"], text);
        assert_eq!(template["urls"][0]["href"], "https://demo.example.com");
        assert_eq!(template["tags"], json!([IMPORT_TAG, "demo"]));

        let (note, field) = item_template(
            "Note",
            None,
            "Code d'accès du portail : voir la fiche",
            None,
        );
        assert_eq!(field, "notesPlain");
        assert_eq!(note["category"], "SECURE_NOTE");
    }

    #[test]
    fn values_are_found_in_concealed_fields_only() {
        let items = vec![
            json!({ "id": "a1", "title": "Site", "fields": [
                { "id": "username", "type": "STRING", "value": "Hx9!kLm2pQ" },
                { "id": "password", "type": "CONCEALED", "value": "other" }
            ]}),
            json!({ "id": "b2", "title": "API démo", "fields": [
                { "id": "credential", "type": "CONCEALED", "value": "Hx9!kLm2pQ" }
            ]}),
        ];
        assert_eq!(
            find_value("Team", &items, "Hx9!kLm2pQ", "Clé API démo : Hx9!kLm2pQ").as_deref(),
            Some("op://Team/b2/credential")
        );
        assert_eq!(find_value("Team", &items, "missing", "Clé API : missing"), None);
        assert_eq!(find_value("Team", &items, "Hx9!kLm2pQ", "Admin MonApp : Hx9!kLm2pQ"), None, "same password, other account");

        let site = vec![json!({ "id": "c3", "title": "wpdebugtoolkit", "urls": [{ "href": "https://www.example.com/login" }], "fields": [
            { "id": "password", "type": "CONCEALED", "value": "Hx9!kLm2pQ" }
        ]})];
        assert_eq!(
            find_value("Team", &site, "Hx9!kLm2pQ", "Admin example.com : Hx9!kLm2pQ").as_deref(),
            Some("op://Team/c3/password"),
            "matched by URL host"
        );
        assert_eq!(reference("My vault", "c3", "password"), "op://My vault/c3/password");
        assert_eq!(find_title(&items, "API démo").as_deref(), Some("b2"));
    }
}
