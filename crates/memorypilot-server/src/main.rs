//! `memorypilot-server`: the multi-tenant REST API over Postgres, and the
//! admin commands that set up tenants, spaces and API keys.

use std::io::Read;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use memorypilot_server::api::{Api, Request, Response, MAX_BODY_BYTES};
use memorypilot_server::{
    embed_scopes, MasterKey, Permission, PgAdmin, PgConnection, PgPool, TenantLimits,
    DEFAULT_MAX_MEMORIES, DEFAULT_REQUESTS_PER_MINUTE,
};
use serde_json::{json, Value};
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
  set-limits <tenant> <requests per minute> <max memories>
                                            each a whole number or 'default'
  revoke-key <tenant> <key id>
  usage <tenant> <days>                     daily requests, writes and searches per space
  audit <tenant> <count>                    latest changes (ids only), newest first
  list-keys <tenant>
  export-tenant <tenant>                    every memory as JSON lines on stdout (portability)
  delete-tenant <tenant> <exact name>       erases the tenant and all its data, irreversibly
  generate-master-key                       32-byte key, printed once; put it in MEMORYPILOT_MASTER_KEY
  provision-key <tenant>                    wrap a data key for a tenant created before the master key
  prefetch-models                           download the embedding model into FASTEMBED_CACHE_PATH

Environment:
  MEMORYPILOT_PG_URL      postgres://user:password@host:port/database (required)
  MEMORYPILOT_PG_CA       extra CA certificate (PEM), e.g. Supabase's prod-ca-2021.crt
  MEMORYPILOT_PG_TLS      on (default: certificate and host name verified) or off (local only)
  MEMORYPILOT_BIND        address to listen on (default 127.0.0.1:8080)
  MEMORYPILOT_WORKERS     concurrent requests (default 8)
  MEMORYPILOT_DEFAULT_RPM requests per minute per key for tenants without their own (default 600)
  MEMORYPILOT_DEFAULT_MAX_MEMORIES
                          memories per tenant for tenants without their own (default 100000)
  MEMORYPILOT_EMBEDDER    on (default) or off: compute vectors in the background
                          (keep it on in one replica at least)
  MEMORYPILOT_ALLOWED_ORIGINS
                          browser origins allowed on /mcp, comma separated (default none)
  MEMORYPILOT_MASTER_KEY  32-byte key (base64) that wraps each tenant's data key;
                          required to store credentials. Generate with generate-master-key
  MEMORYPILOT_PUBLIC_URL  public origin of this server (OAuth issuer, e.g. https://memory.example)
  MEMORYPILOT_CONSOLE_DIR built SvelteKit console (default console/build), served at /console

Endpoints: GET /health; REST /v1/memories, /v1/search; MCP (Streamable HTTP) POST /mcp.
Authorization: Bearer <key>. X-MemoryPilot-Subject: <end user id> acts for one end user.";

const DEFAULT_BIND: &str = "127.0.0.1:8080";
const DEFAULT_WORKERS: usize = 8;
const MAX_WORKERS: usize = 256;
const EMBED_BATCH: usize = 32;
const EMBED_SCOPES_PER_PASS: usize = 100;
const EMBED_IDLE: Duration = Duration::from_secs(2);
const EMBED_RETRY: Duration = Duration::from_secs(30);
const USAGE_FLUSH: Duration = Duration::from_secs(10);

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
        ["prefetch-models"] => {
            memorypilot_core::embedding::prefetch()?;
            eprintln!(
                "models ready in {}",
                memorypilot_core::embedding::fastembed_cache_dir().display()
            );
            Ok(())
        }
        ["generate-master-key"] => {
            println!("{}", MasterKey::generate()?.to_base64());
            eprintln!(
                "Put this in MEMORYPILOT_MASTER_KEY (1Password). Losing it loses every credential."
            );
            Ok(())
        }
        ["provision-key", tenant] => match admin()?.provision_key(uuid(tenant)?)? {
            true => {
                println!("key provisioned");
                Ok(())
            }
            false => Err("no master key, unknown tenant, or a key already exists".into()),
        },
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
        ["set-limits", tenant, requests_per_minute, max_memories] => {
            let limits = TenantLimits {
                requests_per_minute: limit(requests_per_minute, "requests per minute")?,
                max_memories: limit(max_memories, "memory quota")?,
            };
            if limits.requests_per_minute == Some(0) {
                return Err("requests per minute must be at least 1".into());
            }
            match admin()?.set_limits(uuid(tenant)?, limits)? {
                true => {
                    println!("limits set");
                    Ok(())
                }
                false => Err("unknown tenant".into()),
            }
        }
        ["usage", tenant, days] => {
            let days: u32 = days
                .parse()
                .map_err(|_| format!("'{}' is not a number of days", days))?;
            for row in admin()?.usage(uuid(tenant)?, days)? {
                println!(
                    "{}  space {}  requests {}  writes {}  searches {}",
                    row.day, row.space, row.requests, row.writes, row.searches
                );
            }
            Ok(())
        }
        ["audit", tenant, limit] => {
            let limit: usize = limit
                .parse()
                .map_err(|_| format!("'{}' is not a number", limit))?;
            for entry in admin()?.audit_log(uuid(tenant)?, limit)? {
                let by = match entry.key {
                    Some(key) => format!("key {}", key),
                    None => entry.actor.clone(),
                };
                println!(
                    "{}  {}  {}  {}{}",
                    entry.at.to_rfc3339(),
                    by,
                    entry.action,
                    entry.target.unwrap_or_default(),
                    entry
                        .subject
                        .map_or(String::new(), |subject| format!("  end user {}", subject)),
                );
            }
            Ok(())
        }
        ["export-tenant", tenant] => {
            let mut out = std::io::BufWriter::new(std::io::stdout().lock());
            let written = admin()?.export_tenant(uuid(tenant)?, &mut out)?;
            std::io::Write::flush(&mut out).map_err(|e| format!("Export: {}", e))?;
            eprintln!("{} memories exported", written);
            Ok(())
        }
        ["delete-tenant", tenant, name] => {
            admin()?.delete_tenant(uuid(tenant)?, name)?;
            println!(
                "tenant deleted with all its spaces, end users, memories, keys, usage and audit"
            );
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

fn number_env<T: std::str::FromStr>(name: &str, default: T) -> Result<T, String> {
    match env(name) {
        None => Ok(default),
        Some(value) => value
            .trim()
            .parse()
            .map_err(|_| format!("{} is a whole number, not '{}'", name, value)),
    }
}

/// `default` (the server's default) or a whole number.
fn limit<T: std::str::FromStr>(value: &str, what: &str) -> Result<Option<T>, String> {
    match value {
        "default" => Ok(None),
        number => number
            .parse()
            .map(Some)
            .map_err(|_| format!("{} is a whole number or 'default', not '{}'", what, number)),
    }
}

fn admin() -> Result<PgAdmin, String> {
    let mut admin = PgAdmin::connect(&connection()?)?;
    if let Some(key) = MasterKey::from_env()? {
        admin.set_master_key(key);
    }
    Ok(admin)
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
    let origins = env("MEMORYPILOT_ALLOWED_ORIGINS")
        .map(|list| {
            list.split(',')
                .map(|origin| origin.trim().to_string())
                .filter(|origin| !origin.is_empty())
                .collect()
        })
        .unwrap_or_default();
    let pool = {
        let pool = PgPool::new(connection.clone(), workers);
        match MasterKey::from_env()? {
            Some(key) => {
                eprintln!("[memorypilot-server] tenant encryption on");
                pool.with_master_key(key)
            }
            None => pool,
        }
    };
    match env("MEMORYPILOT_EMBEDDER").as_deref() {
        None | Some("on") => {
            let (connection, pool) = (connection.clone(), pool.clone());
            std::thread::spawn(move || embed_in_background(connection, pool));
        }
        Some("off") => eprintln!("[memorypilot-server] background embedding off"),
        Some(other) => {
            return Err(format!(
                "MEMORYPILOT_EMBEDDER is on or off, not '{}'",
                other
            ))
        }
    }
    let api = Api::new(pool)
        .with_allowed_origins(origins)
        .with_public_url(env("MEMORYPILOT_PUBLIC_URL").unwrap_or_default())
        .with_default_limits(
            number_env("MEMORYPILOT_DEFAULT_RPM", DEFAULT_REQUESTS_PER_MINUTE)?,
            number_env("MEMORYPILOT_DEFAULT_MAX_MEMORIES", DEFAULT_MAX_MEMORIES)?,
        );
    {
        let api = api.clone();
        std::thread::spawn(move || record_usage_in_background(connection, api));
    }
    let server = Arc::new(
        tiny_http::Server::http(&bind).map_err(|e| format!("cannot listen on {}: {}", bind, e))?,
    );
    let address = server
        .server_addr()
        .to_ip()
        .map_or_else(|| bind.clone(), |address| address.to_string());
    eprintln!("[memorypilot-server] listening on http://{}", address);
    let console = std::sync::Arc::new(
        env("MEMORYPILOT_CONSOLE_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::path::PathBuf::from("console/build")),
    );
    let handles: Vec<_> = (0..workers)
        .map(|_| {
            let server = Arc::clone(&server);
            let api = api.clone();
            let console = Arc::clone(&console);
            std::thread::spawn(move || {
                for request in server.incoming_requests() {
                    answer(&api, &console, request);
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

/// Embeds pending memories of every tenant while there are some, then
/// polls; reconnects after a failure.
fn embed_in_background(connection: PgConnection, pool: PgPool) {
    let mut admin: Option<PgAdmin> = None;
    loop {
        let pass = match admin.take() {
            Some(session) => Ok(session),
            None => PgAdmin::connect(&connection),
        }
        .and_then(|mut session| {
            let scopes = session.pending_embedding_scopes(EMBED_SCOPES_PER_PASS)?;
            admin = Some(session);
            Ok(embed_scopes(&pool, &scopes, EMBED_BATCH))
        });
        match pass {
            Ok(0) => std::thread::sleep(EMBED_IDLE),
            Ok(_) => {}
            Err(error) => {
                eprintln!("[memorypilot-server] background embedding: {}", error);
                std::thread::sleep(EMBED_RETRY);
            }
        }
    }
}

/// Adds the usage counted in memory to the daily totals every few seconds;
/// what cannot be stored is kept for the next attempt.
fn record_usage_in_background(connection: PgConnection, api: Api) {
    let mut admin: Option<PgAdmin> = None;
    loop {
        std::thread::sleep(USAGE_FLUSH);
        let rows = api.drain_usage();
        if rows.is_empty() {
            continue;
        }
        let stored = match admin.take() {
            Some(session) => Ok(session),
            None => PgAdmin::connect(&connection),
        }
        .and_then(|mut session| {
            session.record_usage(&rows)?;
            admin = Some(session);
            Ok(())
        });
        if let Err(error) = stored {
            eprintln!("[memorypilot-server] usage not recorded yet: {}", error);
            api.restore_usage(rows);
        }
    }
}

fn read_body(request: &mut tiny_http::Request) -> Result<Vec<u8>, Response> {
    let too_large = || {
        Response::new(
            413,
            json!({ "error": format!("The body is larger than {} bytes", MAX_BODY_BYTES) }),
        )
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
        .map_err(|error| {
            Response::new(
                400,
                json!({ "error": format!("Cannot read the body: {}", error) }),
            )
        })?;
    if body.len() > MAX_BODY_BYTES {
        return Err(too_large());
    }
    Ok(body)
}

fn console_asset(root: &Path, url: &str) -> Option<(String, Vec<u8>)> {
    let rest = url.split('?').next()?.strip_prefix("/console")?;
    let relative = if rest.is_empty() || rest == "/" {
        "index.html"
    } else {
        rest.trim_start_matches('/')
    };
    if relative.contains("..") || Path::new(relative).is_absolute() {
        return None;
    }
    let path = root.join(relative);
    let bytes = std::fs::read(&path).ok().or_else(|| {
        if relative.contains('.') {
            None
        } else {
            std::fs::read(root.join("index.html")).ok()
        }
    })?;
    let content_type = match path.extension().and_then(|ext| ext.to_str()) {
        Some("js") => "text/javascript",
        Some("css") => "text/css",
        Some("svg") => "image/svg+xml",
        Some("json") => "application/json",
        _ => "text/html; charset=utf-8",
    };
    Some((content_type.into(), bytes))
}

/// One request: a panic in the handler answers 500 instead of killing the worker.
fn answer(api: &Api, console: &Path, mut request: tiny_http::Request) {
    let started = Instant::now();
    if request.method().as_str() == "GET" {
        if let Some((content_type, bytes)) = console_asset(console, request.url()) {
            let header = format!("Content-Type: {}", content_type);
            let mut reply = tiny_http::Response::from_data(bytes)
                .with_header(fixed_header(&header))
                .with_header(fixed_header("Cache-Control: no-store"));
            for (name, value) in memorypilot_server::api::FRAME_DENY {
                reply = reply.with_header(fixed_header(&format!("{}: {}", name, value)));
            }
            let _ = request.respond(reply);
            return;
        }
    }
    let response = match read_body(&mut request) {
        Err(response) => response,
        Ok(body) => std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let headers: Vec<(&str, &str)> = request
                .headers()
                .iter()
                .map(|header| (header.field.as_str().as_str(), header.value.as_str()))
                .collect();
            api.handle(&Request {
                method: request.method().as_str(),
                url: request.url(),
                headers: &headers,
                body: &body,
            })
        }))
        .unwrap_or_else(|_| Response::new(500, json!({ "error": "Internal error" }))),
    };
    let path = request.url().split('?').next().unwrap_or("").to_string();
    let method = request.method().as_str().to_string();
    let status = response.status;
    let mut reply = if let Some(html) = &response.html {
        tiny_http::Response::from_string(html.clone())
            .with_header(fixed_header("Content-Type: text/html; charset=utf-8"))
    } else {
        match &response.body {
            Value::Null => tiny_http::Response::from_string(String::new()),
            body => tiny_http::Response::from_string(body.to_string())
                .with_header(fixed_header("Content-Type: application/json")),
        }
    }
    .with_status_code(status)
    .with_header(fixed_header("Cache-Control: no-store"));
    if status == 401 {
        reply = reply.with_header(fixed_header(
            "WWW-Authenticate: Bearer realm=\"MemoryPilot\", resource_metadata=\"/.well-known/oauth-protected-resource\"",
        ));
    }
    for (name, value) in &response.headers {
        match tiny_http::Header::from_bytes(name.as_bytes(), value.as_bytes()) {
            Ok(header) => reply = reply.with_header(header),
            Err(()) => eprintln!("[memorypilot-server] header {} not sent", name),
        }
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
