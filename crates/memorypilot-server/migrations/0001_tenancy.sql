-- Multi-tenant storage: tenant (a MemoryPilot customer) > space (one of
-- its apps or environments) > subject (an end user of that app).
-- Every data row carries tenant_id; row level security keeps the
-- application role inside the tenant set by `app.tenant_id` for the
-- transaction. The store also filters by tenant and space in every query.

CREATE EXTENSION IF NOT EXISTS vector;

DO $$
BEGIN
    CREATE ROLE memorypilot_app NOLOGIN;
EXCEPTION WHEN duplicate_object THEN
    NULL;
END
$$;
GRANT memorypilot_app TO CURRENT_USER;

CREATE TABLE tenants (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    name text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE spaces (
    tenant_id uuid NOT NULL REFERENCES tenants (id) ON DELETE CASCADE,
    id uuid NOT NULL DEFAULT gen_random_uuid(),
    name text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, id),
    UNIQUE (id),
    UNIQUE (tenant_id, name)
);

CREATE TABLE subjects (
    tenant_id uuid NOT NULL,
    space_id uuid NOT NULL,
    id uuid NOT NULL DEFAULT gen_random_uuid(),
    external_id text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, space_id, id),
    UNIQUE (id),
    UNIQUE (tenant_id, space_id, external_id),
    FOREIGN KEY (tenant_id, space_id) REFERENCES spaces (tenant_id, id) ON DELETE CASCADE
);

-- subject_id NULL: shared by the whole space. fts_content is the content
-- plus its stemmed projection, accents folded (computed by the store, as
-- SQLite's FTS5 index does).
CREATE TABLE memories (
    tenant_id uuid NOT NULL,
    space_id uuid NOT NULL,
    subject_id uuid,
    id text NOT NULL,
    content text NOT NULL,
    fts_content text NOT NULL,
    kind text NOT NULL,
    project text,
    tags jsonb NOT NULL DEFAULT '[]',
    source text NOT NULL,
    importance integer NOT NULL,
    expires_at timestamptz,
    metadata jsonb,
    embedding vector,
    content_hash text NOT NULL,
    created_at timestamptz NOT NULL,
    updated_at timestamptz NOT NULL,
    last_accessed_at timestamptz,
    access_count integer NOT NULL DEFAULT 0,
    fts tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('simple', fts_content), 'A')
        || setweight(to_tsvector('simple', tags::text), 'B')
        || setweight(to_tsvector('simple', coalesce(project, '')), 'C')
        || setweight(to_tsvector('simple', kind), 'D')
    ) STORED,
    PRIMARY KEY (tenant_id, id),
    FOREIGN KEY (tenant_id, space_id) REFERENCES spaces (tenant_id, id) ON DELETE CASCADE,
    FOREIGN KEY (tenant_id, space_id, subject_id) REFERENCES subjects (tenant_id, space_id, id) ON DELETE CASCADE
);
CREATE INDEX memories_scope_recent ON memories (tenant_id, space_id, project, updated_at DESC);
CREATE INDEX memories_content_hash ON memories (tenant_id, space_id, content_hash);
CREATE INDEX memories_fts ON memories USING gin (fts);

CREATE TABLE memory_links (
    tenant_id uuid NOT NULL,
    source_id text NOT NULL,
    target_id text NOT NULL,
    relation_type text NOT NULL DEFAULT 'relates_to',
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, source_id, target_id),
    FOREIGN KEY (tenant_id, source_id) REFERENCES memories (tenant_id, id) ON DELETE CASCADE,
    FOREIGN KEY (tenant_id, target_id) REFERENCES memories (tenant_id, id) ON DELETE CASCADE
);
CREATE INDEX memory_links_target ON memory_links (tenant_id, target_id);

CREATE TABLE memory_entities (
    tenant_id uuid NOT NULL,
    memory_id text NOT NULL,
    entity_kind text NOT NULL,
    entity_value text NOT NULL,
    FOREIGN KEY (tenant_id, memory_id) REFERENCES memories (tenant_id, id) ON DELETE CASCADE
);
CREATE INDEX memory_entities_memory ON memory_entities (tenant_id, memory_id);
CREATE INDEX memory_entities_value ON memory_entities (tenant_id, lower(entity_value));

CREATE TABLE knowledge_triples (
    tenant_id uuid NOT NULL,
    space_id uuid NOT NULL,
    subject_id uuid,
    id uuid NOT NULL DEFAULT gen_random_uuid(),
    subject text NOT NULL,
    predicate text NOT NULL,
    object text NOT NULL,
    valid_to timestamptz,
    source_memory_id text,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, id),
    FOREIGN KEY (tenant_id, space_id) REFERENCES spaces (tenant_id, id) ON DELETE CASCADE,
    FOREIGN KEY (tenant_id, space_id, subject_id) REFERENCES subjects (tenant_id, space_id, id) ON DELETE CASCADE,
    FOREIGN KEY (tenant_id, source_memory_id) REFERENCES memories (tenant_id, id) ON DELETE SET NULL (source_memory_id)
);
CREATE INDEX knowledge_triples_subject ON knowledge_triples (tenant_id, lower(subject));
CREATE INDEX knowledge_triples_object ON knowledge_triples (tenant_id, lower(object));
CREATE INDEX knowledge_triples_source ON knowledge_triples (tenant_id, source_memory_id);

CREATE TABLE projects (
    tenant_id uuid NOT NULL,
    space_id uuid NOT NULL,
    name text NOT NULL,
    path text NOT NULL DEFAULT '',
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, space_id, name),
    FOREIGN KEY (tenant_id, space_id) REFERENCES spaces (tenant_id, id) ON DELETE CASCADE
);

GRANT USAGE ON SCHEMA public TO memorypilot_app;
GRANT SELECT ON tenants, spaces, subjects TO memorypilot_app;
GRANT SELECT, INSERT, UPDATE, DELETE
    ON memories, memory_links, memory_entities, knowledge_triples, projects
    TO memorypilot_app;

ALTER TABLE tenants ENABLE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON tenants
    USING (id = nullif(current_setting('app.tenant_id', true), '')::uuid);

DO $$
DECLARE
    tenant_table text;
BEGIN
    FOREACH tenant_table IN ARRAY ARRAY[
        'spaces', 'subjects', 'memories', 'memory_links',
        'memory_entities', 'knowledge_triples', 'projects'
    ] LOOP
        EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', tenant_table);
        EXECUTE format(
            'CREATE POLICY tenant_isolation ON %I '
            'USING (tenant_id = nullif(current_setting(''app.tenant_id'', true), '''')::uuid) '
            'WITH CHECK (tenant_id = nullif(current_setting(''app.tenant_id'', true), '''')::uuid)',
            tenant_table
        );
    END LOOP;
END
$$;
