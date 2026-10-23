//! `memorypilot-server`: the multi-tenant REST API over Postgres, and the
//! admin commands that set up tenants, spaces and API keys.

use std::io::Read;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use memorypilot_server::api::{Api, Request, Response, MAX_BODY_BYTES};
use memorypilot_server::{Permission, PgAdmin, PgConnection, PgPool};
use serde_json::json;
use uuid::Uuid;

const USAGE: &str = "memorypilot-server [command]

Commands:
  serve                                     REST API (default); applies migrations first
  migrate                                   apply the missing migrations
  create-tenant <name>                      print the new tenant id
  create-space <tenant> <name>              print the new space id
  create-key <tenant> <space> <name> <permissions>
                                            permissions: comma separated, among read,write,subjects;
                                            the key is printed once and never stored
  revoke-key <tenant> <key id>
  list-keys <tenant>

Environment:
  MEMORYPILOT_PG_URL      postgres://user:password@host:port/database (required)
  MEMORYPILOT_PG_CA       extra CA certificate (PEM), e.g. Supabase's prod-ca-2021.crt
  MEMORYPILOT_PG_TLS      on (default: certificate and host name verified) or off (local only)
  MEMORYPILOT_BIND        address to listen on (default 127.0.0.1:8080)
  MEMORYPILOT_WORKERS     concurrent requests (default 8)";

const DEFAULT_BIND: &str = "127.0.0.1:8080";
const DEFAULT_WORKERS: usize = 8;
const MAX_WORKERS: usize = 256;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(error) = run(&args) {
        eprintln!("memorypilot-server: {}", error);
        std::process::exit(1);
    }
}

fn run(args: &[String]) -> Result<(), String> {
    let words: Vec<&str> = args.iter().map(String::as_str).collect();
    match words.as_slice() {
        [] | ["serve"] => serve(),
        ["help" | "--help" | "-h"] => {
            println!("{}", USAGE);
            Ok(())
        }
        ["migrate"] => {
            let applied = admin()?.migrate()?;
            println!("applied migrations: {:?}", applied);
            Ok(())
        }
        ["create-tenant", name] => {
            println!("{}", admin()?.create_tenant(name)?);
            Ok(())
        }
        ["create-space", tenant, name] => {
            println!("{}", admin()?.create_space(uuid(tenant)?, name)?);
            Ok(())
        }
        ["create-key", tenant, space, name, permissions] => {
            let permissions = permissions
                .split(',')
                .map(|name| Permission::parse(name.trim()))
                .collect::<Result<Vec<_>, _>>()?;
            let key = admin()?.create_api_key(uuid(tenant)?, uuid(space)?, name, &permissions)?;
            println!("id:  {}", key.id);
            println!("key: {}", key.secret);
            eprintln!("Store the key now: only its hash is kept.");
            Ok(())
        }
        ["revoke-key", tenant, key] => match admin()?.revoke_api_key(uuid(tenant)?, uuid(key)?)? {
            true => {
                println!("revoked");
                Ok(())
            }
            false => Err("no live key with this id for this tenant".into()),
        },
        ["list-keys", tenant] => {
            for key in admin()?.list_api_keys(uuid(tenant)?)? {
                let permissions: Vec<&str> = key.permissions.iter().map(|p| p.as_str()).collect();
                println!(
                    "{}  {}…  {}  space {}  [{}]  created {}  last used {}{}",
                    key.id,
                    key.prefix,
                    key.name,
                    key.space,
                    permissions.join(","),
                    key.created_at.to_rfc3339(),
                    key.last_used_at
                        .map_or("never".into(), |at| at.to_rfc3339()),
                    key.revoked_at
                        .map_or(String::new(), |at| format!("  REVOKED {}", at.to_rfc3339())),
                );
            }
            Ok(())
        }
        _ => Err(format!("unknown command\n\n{}", USAGE)),
    }
}

fn uuid(value: &str) -> Result<Uuid, String> {
    Uuid::parse_str(value).map_err(|_| format!("'{}' is not a UUID", value))
}

fn env(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .filter(|value| !value.trim().is_empty())
}

fn connection() -> Result<PgConnection, String> {
    let url = env("MEMORYPILOT_PG_URL").ok_or("MEMORYPILOT_PG_URL is not set")?;
    match env("MEMORYPILOT_PG_TLS").as_deref() {
        None | Some("on") => {
            PgConnection::verified(&url, env("MEMORYPILOT_PG_CA").as_deref().map(Path::new))
        }
        Some("off") => PgConnection::plain(&url),
        Some(other) => Err(format!("MEMORYPILOT_PG_TLS is on or off, not '{}'", other)),
    }
}

fn admin() -> Result<PgAdmin, String> {
    PgAdmin::connect(&connection()?)
}

fn serve() -> Result<(), String> {
    let connection = connection()?;
    let applied = PgAdmin::connect(&connection)?.migrate()?;
    if !applied.is_empty() {
        eprintln!("[memorypilot-server] applied migrations {:?}", applied);
    }
    let bind = env("MEMORYPILOT_BIND").unwrap_or_else(|| DEFAULT_BIND.into());
    let workers = match env("MEMORYPILOT_WORKERS") {
        None => DEFAULT_WORKERS,
        Some(value) => value
            .parse::<usize>()
            .ok()
            .filter(|n| (1..=MAX_WORKERS).contains(n))
            .ok_or_else(|| format!("MEMORYPILOT_WORKERS is a number from 1 to {}", MAX_WORKERS))?,
    };
    let api = Api::new(PgPool::new(connection, workers));
    let server = Arc::new(
        tiny_http::Server::http(&bind).map_err(|e| format!("cannot listen on {}: {}", bind, e))?,
    );
    let address = server
        .server_addr()
        .to_ip()
        .map_or_else(|| bind.clone(), |address| address.to_string());
    eprintln!("[memorypilot-server] listening on http://{}", address);
    let handles: Vec<_> = (0..workers)
        .map(|_| {
            let server = Arc::clone(&server);
            let api = api.clone();
            std::thread::spawn(move || {
                for request in server.incoming_requests() {
                    answer(&api, request);
                }
            })
        })
        .collect();
    for handle in handles {
        handle
            .join()
            .map_err(|_| "a worker stopped unexpectedly".to_string())?;
    }
    Ok(())
}

fn header<'a>(request: &'a tiny_http::Request, name: &'static str) -> Option<&'a str> {
    request
        .headers()
        .iter()
        .find(|header| header.field.equiv(name))
        .map(|header| header.value.as_str())
}

fn read_body(request: &mut tiny_http::Request) -> Result<Vec<u8>, Response> {
    let too_large = || Response {
        status: 413,
        body: json!({ "error": format!("The body is larger than {} bytes", MAX_BODY_BYTES) }),
    };
    if request
        .body_length()
        .is_some_and(|length| length > MAX_BODY_BYTES)
    {
        return Err(too_large());
    }
    let mut body = Vec::new();
    request
        .as_reader()
        .take(MAX_BODY_BYTES as u64 + 1)
        .read_to_end(&mut body)
        .map_err(|error| Response {
            status: 400,
            body: json!({ "error": format!("Cannot read the body: {}", error) }),
        })?;
    if body.len() > MAX_BODY_BYTES {
        return Err(too_large());
    }
    Ok(body)
}

/// One request: a panic in the handler answers 500 instead of killing the worker.
fn answer(api: &Api, mut request: tiny_http::Request) {
    let started = Instant::now();
    let response = match read_body(&mut request) {
        Err(response) => response,
        Ok(body) => std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            api.handle(&Request {
                method: request.method().as_str(),
                url: request.url(),
                authorization: header(&request, "Authorization"),
                subject: header(&request, "X-MemoryPilot-Subject"),
                body: &body,
            })
        }))
        .unwrap_or_else(|_| Response {
            status: 500,
            body: json!({ "error": "Internal error" }),
        }),
    };
    let path = request.url().split('?').next().unwrap_or("").to_string();
    let method = request.method().as_str().to_string();
    let status = response.status;
    let mut reply = tiny_http::Response::from_string(response.body.to_string())
        .with_status_code(status)
        .with_header(fixed_header("Content-Type: application/json"))
        .with_header(fixed_header("Cache-Control: no-store"));
    if status == 401 {
        reply = reply.with_header(fixed_header("WWW-Authenticate: Bearer"));
    }
    if let Err(error) = request.respond(reply) {
        eprintln!(
            "[memorypilot-server] {} {}: response not sent: {}",
            method, path, error
        );
    }
    eprintln!(
        "[memorypilot-server] {} {} {} {}ms",
        method,
        path,
        status,
        started.elapsed().as_millis()
    );
}

fn fixed_header(line: &str) -> tiny_http::Header {
    line.parse()
        .unwrap_or_else(|_| unreachable!("static header line: {}", line))
}
