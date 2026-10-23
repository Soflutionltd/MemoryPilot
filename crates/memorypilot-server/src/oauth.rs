//! OAuth 2.1 for MCP clients (claude.ai, ChatGPT): public clients, PKCE
//! S256, one access token per end user. The tenant is the API key pasted
//! on the consent page.

use std::collections::HashMap;

use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use serde_json::json;
use uuid::Uuid;

use crate::api::{Api, Request, Response};
use crate::auth::{key_hash, parse_all, ApiKey};
use crate::crypto::random_bytes;
use crate::{pg_error, PgPool, TenantLimits};

const TOKEN_PREFIX: &str = "mpt_";
const CODE_TTL_SECS: i64 = 300;
const TOKEN_TTL_SECS: i64 = 30 * 24 * 3600;

pub fn well_formed_token(presented: &str) -> bool {
    presented.strip_prefix(TOKEN_PREFIX).is_some_and(|body| {
        body.len() == URL_SAFE_NO_PAD.encode([0u8; 32]).len()
            && body
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_')
    })
}

pub fn resolve_token(pool: &PgPool, presented: &str) -> Result<Option<ApiKey>, String> {
    let mut client = pool.checkout()?;
    let rows = client
        .query(
            "SELECT tenant_id, space_id, key_id, subject_id, permissions, requests_per_minute, max_memories
             FROM memorypilot_resolve_token($1)",
            &[&key_hash(presented)],
        )
        .map_err(pg_error)?;
    rows.first()
        .map(|row| {
            Ok(ApiKey {
                id: row.get("key_id"),
                tenant: row.get("tenant_id"),
                space: row.get("space_id"),
                permissions: parse_all(row.get("permissions"))?,
                limits: TenantLimits {
                    requests_per_minute: row
                        .get::<_, Option<i32>>("requests_per_minute")
                        .map(|n| n.max(1) as u32),
                    max_memories: row
                        .get::<_, Option<i64>>("max_memories")
                        .map(|n| n.max(0) as u64),
                },
                subject: row.get("subject_id"),
            })
        })
        .transpose()
}

pub fn handle(api: &Api, request: &Request) -> Option<Response> {
    let (path, query) = request.url.split_once('?').unwrap_or((request.url, ""));
    let response = match (request.method, path) {
        ("GET", "/.well-known/oauth-authorization-server") => metadata(api),
        ("GET", "/.well-known/oauth-protected-resource") => resource(api),
        ("POST", "/register") | ("POST", "/oauth/register") => register(api, request),
        ("GET", "/authorize") | ("GET", "/oauth/authorize") => authorize_form(query),
        ("POST", "/authorize") | ("POST", "/oauth/authorize") => authorize(api, request),
        ("POST", "/token") | ("POST", "/oauth/token") => token(api, request),
        ("POST", "/revoke") | ("POST", "/oauth/revoke") => revoke(api, request),
        _ => return None,
    };
    Some(response)
}

fn issuer(api: &Api) -> String {
    if api.public_url().is_empty() {
        "http://127.0.0.1:8080".into()
    } else {
        api.public_url().to_string()
    }
}

fn metadata(api: &Api) -> Response {
    let issuer = issuer(api);
    Response::ok(json!({
        "issuer": issuer,
        "authorization_endpoint": format!("{}/oauth/authorize", issuer),
        "token_endpoint": format!("{}/oauth/token", issuer),
        "registration_endpoint": format!("{}/register", issuer),
        "revocation_endpoint": format!("{}/oauth/revoke", issuer),
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code"],
        "code_challenge_methods_supported": ["S256"],
        "token_endpoint_auth_methods_supported": ["none"],
        "scopes_supported": ["mcp"],
    }))
}

fn resource(api: &Api) -> Response {
    let issuer = issuer(api);
    Response::ok(json!({
        "resource": issuer,
        "authorization_servers": [issuer],
        "bearer_methods_supported": ["header"],
        "resource_documentation": "https://github.com/Soflutionltd/MemoryPilot",
    }))
}

fn register(api: &Api, request: &Request) -> Response {
    let body: serde_json::Value = match serde_json::from_slice(request.body) {
        Ok(body) => body,
        Err(error) => return bad(400, &format!("Invalid JSON: {}", error)),
    };
    let name = body["client_name"].as_str().unwrap_or("MCP client");
    let uris: Vec<String> = body["redirect_uris"]
        .as_array()
        .map(|uris| {
            uris.iter()
                .filter_map(|uri| uri.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    if name.is_empty()
        || name.len() > 200
        || uris.is_empty()
        || uris.iter().any(|uri| !safe_uri(uri))
    {
        return bad(
            400,
            "client_name and at least one https or localhost redirect_uri are required",
        );
    }
    let id = format!("mcp_{}", Uuid::new_v4().simple());
    if let Err(error) = insert_client(api, &id, name, &uris) {
        return Response::from(crate::api::Failure::from_store(error));
    }
    Response::new(
        201,
        json!({
            "client_id": id,
            "client_name": name,
            "redirect_uris": uris,
            "token_endpoint_auth_method": "none",
            "grant_types": ["authorization_code"],
            "response_types": ["code"],
        }),
    )
}

fn insert_client(api: &Api, id: &str, name: &str, uris: &[String]) -> Result<(), String> {
    let mut client = api.pool.checkout()?;
    client
        .execute(
            "INSERT INTO oauth_clients (id, name, redirect_uris) VALUES ($1, $2, $3)",
            &[&id, &name, &uris],
        )
        .map_err(pg_error)?;
    Ok(())
}

fn authorize_form(query: &str) -> Response {
    let params = form(query.as_bytes());
    Response::html(200, consent_page(&params, None))
}

fn authorize(api: &Api, request: &Request) -> Response {
    let params = form(request.body);
    let client_id = required(&params, "client_id");
    let redirect_uri = required(&params, "redirect_uri");
    let challenge = required(&params, "code_challenge");
    let api_key = required(&params, "api_key");
    let (client_id, redirect_uri, challenge, api_key) =
        match (client_id, redirect_uri, challenge, api_key) {
            (Ok(a), Ok(b), Ok(c), Ok(d)) => (a, b, c, d),
            _ => return Response::html(400, consent_page(&params, Some("Missing fields"))),
        };
    if params.get("code_challenge_method").map(String::as_str) != Some("S256") {
        return Response::html(400, consent_page(&params, Some("PKCE must be S256")));
    }
    if let Err(error) = registered_redirect(api, client_id, redirect_uri) {
        return Response::html(400, consent_page(&params, Some(&error)));
    }
    let key = match api.pool.authenticate(api_key) {
        Ok(Some(key)) => key,
        _ => {
            return Response::html(
                400,
                consent_page(&params, Some("Unknown or revoked API key")),
            )
        }
    };
    let subject = match params
        .get("subject")
        .map(String::as_str)
        .filter(|s| !s.is_empty())
    {
        Some(external_id) => match api.pool.subject(&key, external_id) {
            Ok(id) => Some(id),
            Err(error) => return Response::html(400, consent_page(&params, Some(&error))),
        },
        None => None,
    };
    let code = match issue_code(api, client_id, redirect_uri, challenge, &key, subject) {
        Ok(code) => code,
        Err(error) => return Response::from(crate::api::Failure::from_store(error)),
    };
    let mut location = format!("{}?code={}", redirect_uri, code);
    if let Some(state) = params.get("state") {
        location.push_str("&state=");
        location.push_str(&urlencoding(state));
    }
    let mut response = Response::new(302, serde_json::Value::Null);
    response.headers.push(("Location", location));
    response
}

fn token(api: &Api, request: &Request) -> Response {
    let params = form(request.body);
    if params.get("grant_type").map(String::as_str) != Some("authorization_code") {
        return bad(400, "grant_type must be authorization_code");
    }
    let code = match required(&params, "code") {
        Ok(code) => code,
        Err(error) => return bad(400, &error),
    };
    let verifier = match required(&params, "code_verifier") {
        Ok(verifier) => verifier,
        Err(error) => return bad(400, &error),
    };
    let redirect_uri = match required(&params, "redirect_uri") {
        Ok(uri) => uri,
        Err(error) => return bad(400, &error),
    };
    match exchange(
        api,
        code,
        verifier,
        redirect_uri,
        params.get("client_id").map(String::as_str),
    ) {
        Ok(secret) => Response::ok(json!({
            "access_token": secret,
            "token_type": "Bearer",
            "expires_in": TOKEN_TTL_SECS,
            "scope": "mcp",
        })),
        Err(error) => bad(400, &error),
    }
}

fn revoke(api: &Api, request: &Request) -> Response {
    let params = form(request.body);
    if let Some(token) = params.get("token") {
        let _ = revoke_token(api, token);
    }
    Response::new(200, serde_json::Value::Null)
}

fn registered_redirect(api: &Api, client_id: &str, redirect_uri: &str) -> Result<(), String> {
    let mut client = api.pool.checkout()?;
    let uris: Option<Vec<String>> = client
        .query_opt(
            "SELECT redirect_uris FROM oauth_clients WHERE id = $1",
            &[&client_id],
        )
        .map_err(pg_error)?
        .map(|row| row.get(0));
    match uris {
        Some(uris) if uris.iter().any(|uri| uri == redirect_uri) => Ok(()),
        Some(_) => Err("redirect_uri is not registered for this client".into()),
        None => Err("Unknown client_id".into()),
    }
}

fn issue_code(
    api: &Api,
    client_id: &str,
    redirect_uri: &str,
    challenge: &str,
    key: &ApiKey,
    subject: Option<Uuid>,
) -> Result<String, String> {
    let bytes = random_bytes::<32>()?;
    let code = URL_SAFE_NO_PAD.encode(bytes);
    let mut client = api.pool.checkout()?;
    client
        .execute(
            "INSERT INTO oauth_codes (code_hash, client_id, redirect_uri, code_challenge, tenant_id, space_id, key_id, subject_id, permissions, expires_at)
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, now() + make_interval(secs => $10))",
            &[
                &key_hash(&code),
                &client_id,
                &redirect_uri,
                &challenge,
                &key.tenant,
                &key.space,
                &key.id,
                &subject,
                &key.permissions.iter().map(|p| p.as_str().to_string()).collect::<Vec<_>>(),
                &(CODE_TTL_SECS as f64),
            ],
        )
        .map_err(pg_error)?;
    Ok(code)
}

fn exchange(
    api: &Api,
    code: &str,
    verifier: &str,
    redirect_uri: &str,
    client_id: Option<&str>,
) -> Result<String, String> {
    if s256(verifier) != peek_challenge(api, code, redirect_uri, client_id)? {
        return Err("code_verifier does not match the challenge".into());
    }
    let mut client = api.pool.checkout()?;
    let mut tx = client.transaction().map_err(pg_error)?;
    let row = tx
        .query_opt(
            "UPDATE oauth_codes SET used_at = now()
             WHERE code_hash = $1 AND used_at IS NULL AND expires_at > now() AND redirect_uri = $2
             RETURNING client_id, tenant_id, space_id, key_id, subject_id, permissions",
            &[&key_hash(code), &redirect_uri],
        )
        .map_err(pg_error)?
        .ok_or_else(|| "Invalid or expired code".to_string())?;
    let issued_client: String = row.get(0);
    if let Some(presented) = client_id {
        if presented != issued_client {
            return Err("client_id does not match the code".into());
        }
    }
    let tenant: Uuid = row.get(1);
    let space: Uuid = row.get(2);
    let key_id: Uuid = row.get(3);
    let subject: Option<Uuid> = row.get(4);
    let permissions: Vec<String> = row.get(5);
    tx.execute(
        "SELECT set_config('app.tenant_id', $1, true)",
        &[&tenant.to_string()],
    )
    .map_err(pg_error)?;
    let secret = format!(
        "{}{}",
        TOKEN_PREFIX,
        URL_SAFE_NO_PAD.encode(random_bytes::<32>()?)
    );
    let prefix = secret[..TOKEN_PREFIX.len() + 8].to_string();
    tx.execute(
        "INSERT INTO oauth_tokens (tenant_id, token_hash, prefix, client_id, space_id, key_id, subject_id, permissions, expires_at)
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8, now() + make_interval(secs => $9))",
        &[
            &tenant,
            &key_hash(&secret),
            &prefix,
            &issued_client,
            &space,
            &key_id,
            &subject,
            &permissions,
            &(TOKEN_TTL_SECS as f64),
        ],
    )
    .map_err(pg_error)?;
    tx.commit().map_err(pg_error)?;
    Ok(secret)
}

fn peek_challenge(
    api: &Api,
    code: &str,
    redirect_uri: &str,
    client_id: Option<&str>,
) -> Result<String, String> {
    let mut client = api.pool.checkout()?;
    let row = client
        .query_opt(
            "SELECT code_challenge, client_id FROM oauth_codes
             WHERE code_hash = $1 AND used_at IS NULL AND expires_at > now() AND redirect_uri = $2",
            &[&key_hash(code), &redirect_uri],
        )
        .map_err(pg_error)?
        .ok_or_else(|| "Invalid or expired code".to_string())?;
    let stored_client: String = row.get(1);
    if let Some(presented) = client_id {
        if presented != stored_client {
            return Err("client_id does not match the code".into());
        }
    }
    Ok(row.get(0))
}

fn revoke_token(api: &Api, token: &str) -> Result<(), String> {
    let mut client = api.pool.checkout()?;
    client
        .execute(
            "UPDATE oauth_tokens SET revoked_at = now() WHERE token_hash = $1 AND revoked_at IS NULL",
            &[&key_hash(token)],
        )
        .map_err(pg_error)?;
    Ok(())
}

fn s256(verifier: &str) -> String {
    URL_SAFE_NO_PAD
        .encode(ring::digest::digest(&ring::digest::SHA256, verifier.as_bytes()).as_ref())
}

fn form(body: &[u8]) -> HashMap<String, String> {
    form_urlencoded::parse(body).into_owned().collect()
}

fn required<'a>(params: &'a HashMap<String, String>, name: &str) -> Result<&'a str, String> {
    params
        .get(name)
        .map(String::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| format!("{} is required", name))
}

fn safe_uri(uri: &str) -> bool {
    (uri.starts_with("https://")
        || uri.starts_with("http://127.0.0.1")
        || uri.starts_with("http://localhost"))
        && uri.len() < 500
        && !uri.contains(['\n', '\r', ' '])
}

fn urlencoding(value: &str) -> String {
    form_urlencoded::byte_serialize(value.as_bytes()).collect()
}

fn bad(status: u16, message: &str) -> Response {
    Response::new(
        status,
        json!({ "error": "invalid_request", "error_description": message }),
    )
}

fn consent_page(params: &HashMap<String, String>, error: Option<&str>) -> String {
    let field = |name: &str| {
        params
            .get(name)
            .map(|value| html_escape(value))
            .unwrap_or_default()
    };
    let error = error
        .map(|message| format!("<p role=\"alert\">{}</p>", html_escape(message)))
        .unwrap_or_default();
    format!(
        r#"<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>MemoryPilot — connect</title>
<style>
body {{ font: 16px/1.45 ui-sans-serif, system-ui; background: #14120f; color: #f3efe6; margin: 0; }}
main {{ max-width: 28rem; margin: 4rem auto; padding: 2rem; border: 1px solid #ffffff1a; border-radius: 1.25rem; }}
label {{ display: block; margin: 1rem 0 0.35rem; font-weight: 600; }}
input {{ width: 100%; box-sizing: border-box; padding: 0.75rem 1rem; border-radius: 0.75rem; border: 1px solid #ffffff22; background: #1c1916; color: inherit; }}
button {{ margin-top: 1.5rem; width: 100%; padding: 0.85rem; border: 0; border-radius: 0.75rem; background: #e8a317; color: #14120f; font-weight: 700; }}
p[role=alert] {{ color: #ffb4a2; }}
</style></head>
<body><main>
<h1>Connect MemoryPilot</h1>
<p>Paste an API key. Optional end-user id: the token will only see that person's memories.</p>
{error}
<form method="post" action="/oauth/authorize">
<input type="hidden" name="client_id" value="{client_id}">
<input type="hidden" name="redirect_uri" value="{redirect_uri}">
<input type="hidden" name="state" value="{state}">
<input type="hidden" name="code_challenge" value="{code_challenge}">
<input type="hidden" name="code_challenge_method" value="S256">
<label for="api_key">API key</label>
<input id="api_key" name="api_key" type="password" autocomplete="off" required>
<label for="subject">End user id (optional)</label>
<input id="subject" name="subject" value="{subject}">
<button type="submit">Allow access</button>
</form>
</main></body></html>
"#,
        client_id = field("client_id"),
        redirect_uri = field("redirect_uri"),
        state = field("state"),
        code_challenge = field("code_challenge"),
        subject = field("subject"),
    )
}

fn html_escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}
