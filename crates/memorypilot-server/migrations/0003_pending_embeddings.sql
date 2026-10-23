-- The background embedder looks for memories without a vector across all
-- tenants; this keeps that lookup to the (few) pending rows.
CREATE INDEX memories_pending_embedding
    ON memories (tenant_id, space_id, subject_id)
    WHERE embedding IS NULL;
