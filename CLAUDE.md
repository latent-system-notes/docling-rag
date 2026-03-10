# Docling RAG — Development Guidelines

## Permission Logic (CRITICAL)

The RBAC system uses a "public by default, restrict on demand" model:

1. **Documents WITHOUT entries in `document_permissions`** = PUBLIC — all users can see them
2. **Documents WITH entries in `document_permissions`** = RESTRICTED — only users with matching groups can see them
3. **Admin users** (`is_admin=true`) bypass all permission checks (groups=None)

### SQL Pattern

All permission-filtered queries use this WHERE clause:

```sql
WHERE (
    c.doc_id NOT IN (SELECT DISTINCT dp.doc_id FROM document_permissions dp)
    OR c.doc_id IN (
        SELECT dp.doc_id FROM document_permissions dp
        JOIN groups g ON g.id = dp.group_id
        WHERE g.name = ANY(%s)
    )
)
```

- First condition: document has NO permission entries → public, everyone sees it
- Second condition: document has permission entries AND user's group matches → allowed

**NEVER remove the `NOT IN` clause** — it is intentional and ensures public documents remain accessible to all users.

### Where This Pattern Is Used

This exact SQL pattern is applied in **5 functions** in `src/storage/postgres.py`:

1. `search_vectors()` — vector similarity search
2. `search_fulltext()` — full-text search
3. `list_documents()` — My Documents page listing
4. `list_chunks()` — Chunks page listing
5. `count_chunks()` — Chunks pagination count

The MCP server (`src/mcp/server.py`) calls the same functions with the same `groups` parameter, so the same rules apply to MCP clients.

### Permission Lifecycle

- `path_permissions` — admin assigns groups to folder/file paths (tree-based, inherits downward)
- `document_permissions` — cache table, auto-refreshed when `path_permissions` change (add/remove)
- `compute_effective_groups(file_path)` — walks ancestor paths to find all matching groups
- At ingestion: permissions auto-computed from `path_permissions` and cached in `document_permissions`
- On path_permission change: `refresh_all_document_permissions()` recomputes cache for ALL documents
