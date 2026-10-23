//! The `memorypilot-server` binary end to end: admin commands, then real
//! HTTP requests to `serve` (`scripts/test-postgres.sh`).

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::process::{Child, Command, Stdio};

const BINARY: &str = env!("CARGO_BIN_EXE_memorypilot-server");

fn command() -> Command {
    let url = std::env::var("MEMORYPILOT_TEST_PG_URL")
        .expect("MEMORYPILOT_TEST_PG_URL is not set (scripts/test-postgres.sh)");
    let mut command = Command::new(BINARY);
    command
        .env("MEMORYPILOT_PG_URL", url)
        .env("MEMORYPILOT_PG_TLS", "off")
        .env_remove("MEMORYPILOT_PG_CA");
    command
}

fn admin(args: &[&str]) -> String {
    let output = command().args(args).output().expect("run the binary");
    assert!(
        output.status.success(),
        "{:?}: {}",
        args,
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).expect("utf-8")
}

/// Kills the server when the test ends, even on failure.
struct Server {
    child: Child,
    address: String,
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn serve() -> Server {
    let mut child = command()
        .arg("serve")
        .env("MEMORYPILOT_BIND", "127.0.0.1:0")
        .env("MEMORYPILOT_WORKERS", "2")
        .env("MEMORYPILOT_EMBEDDER", "off")
        .stderr(Stdio::piped())
        .spawn()
        .expect("start the server");
    let mut lines = BufReader::new(child.stderr.take().expect("stderr")).lines();
    let address = lines
        .by_ref()
        .map_while(Result::ok)
        .find_map(|line| line.split("listening on http://").nth(1).map(String::from))
        .expect("the server announces its address");
    std::thread::spawn(move || lines.for_each(drop));
    Server { child, address }
}

/// One HTTP/1.1 request; returns the status, the headers and the body.
fn http(
    server: &Server,
    method: &str,
    path: &str,
    key: Option<&str>,
    body: &str,
) -> (u16, String, String) {
    let mut stream = TcpStream::connect(&server.address).expect("connect");
    let authorization = key.map_or(String::new(), |key| {
        format!("Authorization: Bearer {}\r\n", key)
    });
    write!(
        stream,
        "{} {} HTTP/1.1\r\nHost: test\r\nConnection: close\r\nContent-Type: application/json\r\n{}Content-Length: {}\r\n\r\n{}",
        method,
        path,
        authorization,
        body.len(),
        body
    )
    .expect("send");
    let mut answer = String::new();
    stream.read_to_string(&mut answer).expect("read");
    let (head, body) = answer.split_once("\r\n\r\n").expect("HTTP answer");
    let status = head
        .split(' ')
        .nth(1)
        .and_then(|code| code.parse().ok())
        .expect("status code");
    (status, head.to_string(), body.to_string())
}

fn line_value<'a>(output: &'a str, label: &str) -> &'a str {
    output
        .lines()
        .find_map(|line| line.strip_prefix(label))
        .map(str::trim)
        .unwrap_or_else(|| panic!("no '{}' in {}", label, output))
}

#[test]
#[ignore = "needs Postgres with pgvector: scripts/test-postgres.sh"]
fn the_binary_sets_up_a_tenant_and_serves_its_memories() {
    admin(&["migrate"]);
    let tenant = admin(&["create-tenant", "binary-test"]).trim().to_string();
    let space = admin(&["create-space", &tenant, "app"]).trim().to_string();
    let created = admin(&["create-key", &tenant, &space, "backend", "read,write"]);
    let (key_id, key) = (
        line_value(&created, "id:"),
        line_value(&created, "key:").to_string(),
    );
    let listed = admin(&["list-keys", &tenant]);
    assert!(
        listed.contains(key_id) && listed.contains("[read,write]"),
        "{}",
        listed
    );
    assert!(!listed.contains(&key), "the listing never shows the key");

    assert_eq!(
        admin(&["set-limits", &tenant, "120", "default"]).trim(),
        "limits set"
    );
    let server = serve();
    let (status, _, body) = http(&server, "GET", "/health", None, "");
    assert_eq!((status, body.contains("\"ok\"")), (200, true));

    let (status, head, _) = http(&server, "GET", "/v1/memories", None, "");
    assert_eq!(status, 401);
    assert!(head.contains("WWW-Authenticate: Bearer"), "{}", head);

    let (status, _, body) = http(
        &server,
        "POST",
        "/v1/memories",
        Some(&key),
        r#"{"content":"The binary test stores this memory","kind":"fact"}"#,
    );
    assert_eq!(status, 201, "{}", body);
    let memory: serde_json::Value = serde_json::from_str(&body).expect("json");
    let id = memory["memory"]["id"].as_str().expect("id");
    let (status, head, body) = http(
        &server,
        "GET",
        &format!("/v1/memories/{}", id),
        Some(&key),
        "",
    );
    assert_eq!(status, 200, "{}", body);
    assert!(head.contains("Content-Type: application/json"));
    assert!(head.contains("X-RateLimit-Limit: 120"), "{}", head);

    let (status, _, _) = http(
        &server,
        "POST",
        "/v1/memories",
        Some(&key),
        &format!(r#"{{"content":"{}"}}"#, "a".repeat(1024 * 1024)),
    );
    assert_eq!(status, 413, "bodies over 1 MiB are refused");

    let (status, head, body) = http(
        &server,
        "POST",
        "/mcp",
        Some(&key),
        r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#,
    );
    assert_eq!((status, body.as_str()), (202, ""), "{}", head);
    let (status, _, body) = http(
        &server,
        "POST",
        "/mcp",
        Some(&key),
        r#"{"jsonrpc":"2.0","id":1,"method":"tools/list"}"#,
    );
    assert_eq!(status, 200, "{}", body);
    assert!(body.contains("search_memory"));

    admin(&["revoke-key", &tenant, key_id]);
    let (status, _, _) = http(&server, "GET", "/v1/memories", Some(&key), "");
    assert_eq!(status, 401, "a revoked key stops working at once");

    let audit = admin(&["audit", &tenant, "10"]);
    for action in ["key.revoke", "memory.create", "limits.set", "key.create"] {
        assert!(audit.contains(action), "{} in {}", action, audit);
    }
    admin(&["usage", &tenant, "7"]);

    let exported = admin(&["export-tenant", &tenant]);
    assert_eq!(exported.lines().count(), 1, "{}", exported);
    assert!(exported.contains("The binary test stores this memory"));
    let wrong_name = command()
        .args(["delete-tenant", &tenant, "not-its-name"])
        .output()
        .expect("run");
    assert!(!wrong_name.status.success(), "the exact name is required");
    admin(&["delete-tenant", &tenant, "binary-test"]);
    assert_eq!(admin(&["export-tenant", &tenant]), "");

    let unknown = command().arg("create-space").output().expect("run");
    assert!(!unknown.status.success());
    let plain_refused = Command::new(BINARY)
        .arg("migrate")
        .env(
            "MEMORYPILOT_PG_URL",
            std::env::var("MEMORYPILOT_TEST_PG_URL").expect("url"),
        )
        .env_remove("MEMORYPILOT_PG_TLS")
        .env_remove("MEMORYPILOT_PG_CA")
        .output()
        .expect("run");
    assert!(
        !plain_refused.status.success(),
        "TLS is required unless explicitly turned off"
    );
}
