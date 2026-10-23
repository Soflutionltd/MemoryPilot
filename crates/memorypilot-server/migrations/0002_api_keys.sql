-- API keys: one per app (space) of a tenant. Only the SHA-256 of the key is
-- stored; the key itself is shown once, at creation. Permissions:
-- read (search, list, get), write (add, update, delete), subjects (act for
-- an end user named by the request, created on first use).
--
-- The application role cannot read this table: a key is resolved before
-- the tenant is known, so row level security would hide it. The lookup goes
-- through memorypilot_resolve_key, which returns only a live key's scope.

CREATE TABLE api_keys (
    tenant_id uuid NOT NULL,
    space_id uuid NOT NULL,
    id uuid NOT NULL DEFAULT gen_random_uuid(),
    name text NOT NULL,
    prefix text NOT NULL,
    key_hash bytea NOT NULL UNIQUE,
    permissions text[] NOT NULL CHECK (
        cardinality(permissions) > 0
        AND permissions <@ ARRAY['read', 'write', 'subjects']
    ),
    created_at timestamptz NOT NULL DEFAULT now(),
    last_used_at timestamptz,
    revoked_at timestamptz,
    PRIMARY KEY (tenant_id, id),
    UNIQUE (id),
    FOREIGN KEY (tenant_id, space_id) REFERENCES spaces (tenant_id, id) ON DELETE CASCADE
);

ALTER TABLE api_keys ENABLE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON api_keys
    USING (tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid)
    WITH CHECK (tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid);

CREATE FUNCTION memorypilot_resolve_key(candidate bytea)
RETURNS TABLE (tenant_id uuid, space_id uuid, key_id uuid, permissions text[])
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
#variable_conflict use_column
BEGIN
    RETURN QUERY
        SELECT k.tenant_id, k.space_id, k.id, k.permissions
        FROM api_keys k
        WHERE k.key_hash = candidate AND k.revoked_at IS NULL;
    UPDATE api_keys
    SET last_used_at = now()
    WHERE key_hash = candidate
      AND revoked_at IS NULL
      AND (last_used_at IS NULL OR last_used_at < now() - interval '1 minute');
END
$$;
REVOKE ALL ON FUNCTION memorypilot_resolve_key(bytea) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION memorypilot_resolve_key(bytea) TO memorypilot_app;

-- End users are created on first use by a key with the subjects permission.
GRANT INSERT ON subjects TO memorypilot_app;
