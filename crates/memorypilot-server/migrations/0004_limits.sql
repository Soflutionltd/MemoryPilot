-- Per-tenant limits; NULL means the server's default. The key lookup
-- returns them so a request needs no second query.

ALTER TABLE tenants
    ADD COLUMN requests_per_minute integer CHECK (requests_per_minute > 0),
    ADD COLUMN max_memories bigint CHECK (max_memories >= 0);

DROP FUNCTION memorypilot_resolve_key(bytea);

CREATE FUNCTION memorypilot_resolve_key(candidate bytea)
RETURNS TABLE (
    tenant_id uuid,
    space_id uuid,
    key_id uuid,
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
        SELECT k.tenant_id, k.space_id, k.id, k.permissions, t.requests_per_minute, t.max_memories
        FROM api_keys k
        JOIN tenants t ON t.id = k.tenant_id
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
