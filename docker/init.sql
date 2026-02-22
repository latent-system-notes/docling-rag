CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id          TEXT PRIMARY KEY,
    doc_id      TEXT NOT NULL,
    text        TEXT NOT NULL,
    embedding   vector(768),
    text_search tsvector GENERATED ALWAYS AS (to_tsvector('simple', text)) STORED,
    page_num    INTEGER,
    doc_type    TEXT NOT NULL DEFAULT 'unknown',
    language    TEXT NOT NULL DEFAULT 'unknown',
    file_path   TEXT NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    chunk_index INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_chunks_text_search ON chunks USING gin (text_search);
CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks (doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_ingested_at ON chunks (ingested_at DESC);
