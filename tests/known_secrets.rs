//! `--add-known-secret`: one value on stdin, only its print on disk, and the
//! value masked as `known` afterwards. Every path is a temporary folder.

use memorypilot_core::secrets::{self, KnownSecrets, KNOWN_SECRETS_FILE};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

fn temp_dir() -> PathBuf {
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let dir = std::env::temp_dir().join(format!("mp-known-cli-{}-{}", std::process::id(), nanos));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn add_known_secret(dir: &Path, stdin: &str) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_MemoryPilot"))
        .arg("--add-known-secret")
        .env("HOME", dir)
        .env("MEMORYPILOT_DB", dir.join("memory.db"))
        .env("MEMORYPILOT_CONVERSATIONS_DB", dir.join("conversations.db"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(stdin.as_bytes()).unwrap();
    child.wait_with_output().unwrap()
}

#[test]
fn add_known_secret_writes_only_the_print() {
    let dir = temp_dir();
    let file = dir.join(KNOWN_SECRETS_FILE);
    // 32 lowercase hex digits, made up for the test.
    let token = "3c9e1f0a7b2d4e6f8a1c3e5b7d9f0a2c";
    let text = format!("TOKEN_ID {} dans le .env", token);
    assert_eq!(secrets::redact_with(&text, &KnownSecrets::default()), text);

    let refused = add_known_secret(&dir, "\n");
    assert!(!refused.status.success());
    assert!(!file.exists());

    let added = add_known_secret(&dir, &format!("{}\n", token));
    assert!(added.status.success(), "{}", String::from_utf8_lossy(&added.stderr));
    let stdout = String::from_utf8_lossy(&added.stdout);
    let content = std::fs::read_to_string(&file).unwrap();
    assert_eq!(content.lines().count(), 1);
    let print = content.trim();
    assert!(print.len() == 64 && print.bytes().all(|b| b.is_ascii_hexdigit()));
    assert!(stdout.contains(&print[..4]) && !stdout.contains(print) && !stdout.contains(token));
    assert!(!content.contains(token));

    let known = secrets::load_known_secrets_from(Some(&file), None);
    assert_eq!(
        secrets::redact_with(&text, &known),
        format!("TOKEN_ID {} dans le .env", secrets::mask("known", token))
    );
    let _ = std::fs::remove_dir_all(dir);
}
