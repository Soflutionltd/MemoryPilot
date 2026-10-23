-- Metering: daily counters per space, added to by the server every few
-- seconds (never one write per request). Audit: who changed what, by id
-- only (no memory content). The application role may add audit entries,
-- never read them; usage is written by the owner connection.

CREATE TABLE usage_daily (
    tenant_id uuid NOT NULL,
    space_id uuid NOT NULL,
    day date NOT NULL,
    requests bigint NOT NULL DEFAULT 0,
    writes bigint NOT NULL DEFAULT 0,
    searches bigint NOT NULL DEFAULT 0,
    PRIMARY KEY (tenant_id, space_id, day),
    FOREIGN KEY (tenant_id, space_id) REFERENCES spaces (tenant_id, id) ON DELETE CASCADE
);

CREATE TABLE audit_log (
    tenant_id uuid NOT NULL REFERENCES tenants (id) ON DELETE CASCADE,
    id bigint GENERATED ALWAYS AS IDENTITY,
    at timestamptz NOT NULL DEFAULT now(),
    space_id uuid,
    subject_id uuid,
    key_id uuid,
    actor text NOT NULL CHECK (actor IN ('key', 'admin')),
    action text NOT NULL,
    target text,
    PRIMARY KEY (tenant_id, id)
);
CREATE INDEX audit_log_recent ON audit_log (tenant_id, at DESC);

ALTER TABLE usage_daily ENABLE ROW LEVEL SECURITY;
ALTER TABLE audit_log ENABLE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON usage_daily
    USING (tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid)
    WITH CHECK (tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid);
CREATE POLICY tenant_isolation ON audit_log
    USING (tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid)
    WITH CHECK (tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid);

GRANT INSERT ON audit_log TO memorypilot_app;
