-- Approximate nearest neighbours for large scopes. The embedding column has
-- no fixed dimension (the model is configurable), so there is one partial
-- index per supported dimension. pgvector 0.8 brings iterative index scans,
-- which keep a tenant filter from emptying the results.
DO $$
BEGIN
    IF (SELECT string_to_array(split_part(extversion, '-', 1), '.')::int[]
        FROM pg_extension WHERE extname = 'vector') < ARRAY[0, 8] THEN
        RAISE EXCEPTION 'MemoryPilot needs pgvector 0.8 or later: ALTER EXTENSION vector UPDATE';
    END IF;
END
$$;
CREATE INDEX memories_embedding_384 ON memories
    USING hnsw ((embedding::vector(384)) vector_cosine_ops) WHERE vector_dims(embedding) = 384;
CREATE INDEX memories_embedding_768 ON memories
    USING hnsw ((embedding::vector(768)) vector_cosine_ops) WHERE vector_dims(embedding) = 768;
CREATE INDEX memories_embedding_1024 ON memories
    USING hnsw ((embedding::vector(1024)) vector_cosine_ops) WHERE vector_dims(embedding) = 1024;
