-- Migration 002: Normalize all paths to forward slashes and recompute doc_id
-- Makes doc_id deterministic from file_path so it's consistent across Windows/Linux.
-- Safe to run multiple times (idempotent).

BEGIN;

-- 1. Normalize file_path in chunks: backslashes → forward slashes
UPDATE chunks SET file_path = REPLACE(file_path, '\', '/')
WHERE file_path LIKE '%\%';

-- 2. Normalize path in path_permissions
UPDATE path_permissions SET path = RTRIM(REPLACE(path, '\', '/'), '/')
WHERE path LIKE '%\%' OR path LIKE '%/';

-- 3. Deduplicate path_permissions that became identical after normalization
DELETE FROM path_permissions pp
WHERE pp.id NOT IN (
    SELECT MIN(id) FROM path_permissions GROUP BY path, group_id
);

-- 4. Recompute doc_id as md5(file_path) so it's based on the normalized path
--    instead of the OS-specific absolute path.
--    Update chunks, then update document_permissions to match.

-- First, build a mapping of old_doc_id → new_doc_id
CREATE TEMP TABLE doc_id_map AS
SELECT DISTINCT
    doc_id AS old_doc_id,
    md5(file_path) AS new_doc_id
FROM chunks
WHERE doc_id != md5(file_path);

-- Update chunks
UPDATE chunks c
SET doc_id = m.new_doc_id
FROM doc_id_map m
WHERE c.doc_id = m.old_doc_id;

-- Update document_permissions
UPDATE document_permissions dp
SET doc_id = m.new_doc_id
FROM doc_id_map m
WHERE dp.doc_id = m.old_doc_id;

DROP TABLE doc_id_map;

-- 5. Recompute document_permissions from normalized paths
TRUNCATE document_permissions;

INSERT INTO document_permissions (doc_id, group_id)
SELECT DISTINCT c.doc_id, pp.group_id
FROM (SELECT DISTINCT doc_id, file_path FROM chunks) c
JOIN path_permissions pp
  ON c.file_path = pp.path
  OR c.file_path LIKE pp.path || '/%'
ON CONFLICT (doc_id, group_id) DO NOTHING;

COMMIT;
