-- OAuth 2.1 for MCP (claude.ai, ChatGPT): public clients, PKCE S256,
-- one access token per end user. Clients are registered dynamically;
-- the tenant comes from the API key pasted on the consent page.

CREATE TABLE oauth_clients (
    id text PRIMARY KEY,
    name text NOT NULL,
    redirect_uris text[] NOT NULL CHECK (cardinality(redirect_uris) > 0),
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE oauth_codes (
    code_hash bytea PRIMARY KEY,
    client_id text NOT NULL REFERENCES oauth_clients (id) ON DELETE CASCADE,
    redirect_uri text NOT NULL,
    code_challenge text NOT NULL,
    tenant_id uuid NOT NULL REFERENCES tenants (id) ON DELETE CASCADE,
    space_id uuid NOT NULL,
    key_id uuid NOT NULL,
    subject_id uuid,
    permissions text[] NOT NULL,
    expires_at timestamptz NOT NULL,
    used_at timestamptz
);

CREATE TABLE oauth_tokens (
    tenant_id uuid NOT NULL REFERENCES tenants (id) ON DELETE CASCADE,
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    token_hash bytea NOT NULL UNIQUE,
    prefix text NOT NULL,
    client_id text NOT NULL,
    space_id uuid NOT NULL,
    key_id uuid NOT NULL,
    subject_id uuid,
    permissions text[] NOT NULL,
    expires_at timestamptz NOT NULL,
    revoked_at timestamptz,
    created_at timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE oauth_tokens ENABLE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON oauth_tokens
    USING (tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid);

CREATE FUNCTION memorypilot_resolve_token(candidate bytea)
RETURNS TABLE (
    tenant_id uuid,
    space_id uuid,
    key_id uuid,
    subject_id uuid,
    permissions text[],
    requests_per_minute integer,
    max_memories bigint
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
#variable_conflict use_column
BEGIN
    RETURN QUERY
        SELECT t.tenant_id, t.space_id, t.key_id, t.subject_id, t.permissions,
               n.requests_per_minute, n.max_memories
        FROM oauth_tokens t
        JOIN tenants n ON n.id = t.tenant_id
        WHERE t.token_hash = candidate
          AND t.revoked_at IS NULL
          AND t.expires_at > now();
END
$$;
REVOKE ALL ON FUNCTION memorypilot_resolve_token(bytea) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION memorypilot_resolve_token(bytea) TO memorypilot_app;
GRANT SELECT, INSERT ON oauth_clients TO memorypilot_app;
GRANT SELECT, INSERT, UPDATE ON oauth_codes TO memorypilot_app;
GRANT SELECT, INSERT, UPDATE ON oauth_tokens TO memorypilot_app;
