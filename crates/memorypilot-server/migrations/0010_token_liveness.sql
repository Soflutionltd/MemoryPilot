-- An access token dies with the API key it was issued from (revoke-key)
-- and with the end user it acts for (GDPR erasure), not only at expiry.

CREATE OR REPLACE FUNCTION memorypilot_resolve_token(candidate bytea)
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
        JOIN api_keys k ON k.tenant_id = t.tenant_id AND k.id = t.key_id
        WHERE t.token_hash = candidate
          AND t.revoked_at IS NULL
          AND t.expires_at > now()
          AND k.revoked_at IS NULL
          AND (t.subject_id IS NULL OR EXISTS (
              SELECT 1 FROM subjects s
              WHERE s.tenant_id = t.tenant_id AND s.id = t.subject_id
          ));
END
$$;
