import threading
from pathlib import Path

import numpy as np
from psycopg_pool import ConnectionPool
from pgvector.psycopg import register_vector

from ..config import config, COLLECTION_NAME, EMBEDDING_MODEL, get_logger, DEFAULT_TOP_K
from ..models import StorageError
from ..utils import make_doc_id

logger = get_logger(__name__)
_pool: ConnectionPool | None = None
_pool_lock = threading.Lock()


def _normalize_path(p: str) -> str:
    """Normalize path to absolute form with forward slashes, no trailing slash.

    Relative paths are resolved against DOCUMENTS_DIR so that all stored paths
    are always absolute and consistent regardless of the working directory.
    """
    n = p.replace("\\", "/").rstrip("/")
    try:
        docs_dir = config("DOCUMENTS_DIR")
        docs_dir_abs = str(docs_dir.resolve()).replace("\\", "/").rstrip("/")
        if not n.startswith(docs_dir_abs):
            # Relative path like "docs/data" — resolve via DOCUMENTS_DIR parent
            candidate = Path(n)
            resolved = str(candidate.resolve()).replace("\\", "/").rstrip("/")
            if resolved.startswith(docs_dir_abs) or docs_dir_abs.startswith(resolved):
                n = resolved
            else:
                # Path component matches inside DOCUMENTS_DIR
                docs_cfg = str(docs_dir).replace("\\", "/").rstrip("/")
                if n.startswith(docs_cfg + "/") or n == docs_cfg:
                    suffix = n[len(docs_cfg):]
                    n = docs_dir_abs + suffix
                elif n.startswith(docs_cfg.split("/")[-1] + "/"):
                    # e.g. "docs/file.pdf" when DOCUMENTS_DIR is "/workspace/docs"
                    dir_name = docs_cfg.split("/")[-1]
                    suffix = n[len(dir_name):]
                    n = docs_dir_abs + suffix
    except Exception:
        pass
    return n


def get_pool() -> ConnectionPool:
    global _pool
    if _pool is not None:
        return _pool
    with _pool_lock:
        if _pool is not None:
            return _pool
        try:
            conninfo = (
                f"host={config('POSTGRES_HOST')} port={config('POSTGRES_PORT')} "
                f"dbname={config('POSTGRES_DB')} user={config('POSTGRES_USER')} "
                f"password={config('POSTGRES_PASSWORD')} "
                f"connect_timeout=5"
            )
            pool = ConnectionPool(
                conninfo=conninfo,
                min_size=config("POSTGRES_POOL_MIN"),
                max_size=config("POSTGRES_POOL_MAX"),
                timeout=10,
                configure=lambda conn: register_vector(conn),
                check=ConnectionPool.check_connection,
            )
            _pool = pool
            return _pool
        except Exception as e:
            raise StorageError(f"Failed to connect to PostgreSQL: {e}") from e


def cleanup_pool() -> None:
    global _pool
    with _pool_lock:
        if _pool is not None:
            try:
                _pool.close(timeout=5)
            except Exception:
                pass
            _pool = None


def create_collection() -> None:
    try:
        with get_pool().connection() as conn:
            conn.execute("SELECT 1")
    except StorageError:
        raise
    except Exception as e:
        raise StorageError(f"Failed to verify database: {e}") from e


def add_vectors(ids: list[str], vectors: np.ndarray, metadata: list[dict]) -> None:
    try:
        with get_pool().connection() as conn:
            with conn.cursor() as cur:
                for i, (chunk_id, meta) in enumerate(zip(ids, metadata)):
                    cur.execute(
                        """INSERT INTO chunks (id, doc_id, text, embedding, page_num, doc_type, language, file_path, ingested_at, chunk_index)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (id) DO NOTHING""",
                        (
                            chunk_id,
                            meta.get("doc_id", ""),
                            meta.get("text", ""),
                            vectors[i].tolist(),
                            meta.get("page_num"),
                            meta.get("doc_type", "unknown"),
                            meta.get("language", "unknown"),
                            _normalize_path(meta.get("file_path", "")),
                            meta.get("ingested_at"),
                            meta.get("chunk_index", 0),
                        ),
                    )
            conn.commit()
    except StorageError:
        raise
    except Exception as e:
        raise StorageError(f"Failed to add vectors: {e}") from e


def search_vectors(query_vector: np.ndarray, top_k: int, groups: list[str] | None = None) -> list[dict]:
    try:
        if query_vector.ndim == 2:
            query_vector = query_vector[0]
        vec_list = query_vector.tolist()
        with get_pool().connection() as conn:
            if groups is not None:
                rows = conn.execute(
                    """SELECT c.id, c.text, c.doc_id, c.page_num, c.doc_type, c.language, c.file_path, c.ingested_at, c.chunk_index,
                        1 - (c.embedding <=> %s::vector) AS score
                    FROM chunks c
                    WHERE (
                        c.doc_id NOT IN (SELECT DISTINCT dp.doc_id FROM document_permissions dp)
                        OR c.doc_id IN (SELECT dp.doc_id FROM document_permissions dp JOIN groups g ON g.id = dp.group_id WHERE g.name = ANY(%s))
                    )
                    ORDER BY c.embedding <=> %s::vector LIMIT %s""",
                    (vec_list, groups, vec_list, top_k),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT id, text, doc_id, page_num, doc_type, language, file_path, ingested_at, chunk_index,
                        1 - (embedding <=> %s::vector) AS score
                    FROM chunks ORDER BY embedding <=> %s::vector LIMIT %s""",
                    (vec_list, vec_list, top_k),
                ).fetchall()
        return [
            {
                "id": r[0], "text": r[1], "doc_id": r[2], "page_num": r[3],
                "doc_type": r[4], "language": r[5], "file_path": r[6],
                "ingested_at": r[7].isoformat() if r[7] else "", "chunk_index": r[8],
                "score": float(r[9]),
            }
            for r in rows
        ]
    except StorageError:
        raise
    except Exception as e:
        raise StorageError(f"Failed to search vectors: {e}") from e


def search_fulltext(query: str, top_k: int = DEFAULT_TOP_K, groups: list[str] | None = None) -> list[tuple[str, float]]:
    tokens = query.lower().split()
    if not tokens:
        return []
    tsquery = " | ".join(tokens)
    try:
        with get_pool().connection() as conn:
            if groups is not None:
                rows = conn.execute(
                    """SELECT c.id, ts_rank_cd(c.text_search, to_tsquery('simple', %s)) AS rank
                    FROM chunks c
                    WHERE c.text_search @@ to_tsquery('simple', %s)
                      AND (
                        c.doc_id NOT IN (SELECT DISTINCT dp.doc_id FROM document_permissions dp)
                        OR c.doc_id IN (SELECT dp.doc_id FROM document_permissions dp JOIN groups g ON g.id = dp.group_id WHERE g.name = ANY(%s))
                      )
                    ORDER BY rank DESC LIMIT %s""",
                    (tsquery, tsquery, groups, top_k),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT id, ts_rank_cd(text_search, to_tsquery('simple', %s)) AS rank
                    FROM chunks WHERE text_search @@ to_tsquery('simple', %s)
                    ORDER BY rank DESC LIMIT %s""",
                    (tsquery, tsquery, top_k),
                ).fetchall()
        return [(r[0], float(r[1])) for r in rows]
    except StorageError:
        raise
    except Exception as e:
        logger.warning(f"Full-text search failed: {e}")
        return []


def get_chunks_by_ids(ids: list[str]) -> dict:
    if not ids:
        return {"ids": [], "documents": [], "metadatas": []}
    try:
        with get_pool().connection() as conn:
            rows = conn.execute(
                "SELECT id, text, doc_id, page_num, doc_type, language, file_path, ingested_at, chunk_index "
                "FROM chunks WHERE id = ANY(%s)",
                (ids,),
            ).fetchall()
        id_list, docs, metas = [], [], []
        for r in rows:
            id_list.append(r[0])
            docs.append(r[1])
            metas.append({
                "doc_id": r[2], "page_num": r[3], "doc_type": r[4],
                "language": r[5], "file_path": r[6],
                "ingested_at": r[7].isoformat() if r[7] else "", "chunk_index": r[8],
            })
        return {"ids": id_list, "documents": docs, "metadatas": metas}
    except StorageError:
        raise
    except Exception as e:
        raise StorageError(f"Failed to get chunks: {e}") from e


def get_document_count() -> int:
    try:
        with get_pool().connection() as conn:
            return conn.execute("SELECT COUNT(DISTINCT doc_id) FROM chunks").fetchone()[0]
    except StorageError:
        raise
    except Exception as e:
        raise StorageError(f"Failed to get document count: {e}") from e


def get_stats() -> dict:
    try:
        with get_pool().connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return {
            "collection": COLLECTION_NAME,
            "total_chunks": count,
            "embedding_model": EMBEDDING_MODEL,
            "vector_store": "postgresql+pgvector",
        }
    except StorageError:
        raise
    except Exception as e:
        raise StorageError(f"Failed to get stats: {e}") from e


def document_exists(file_path: str | Path) -> bool:
    doc_id = make_doc_id(file_path)
    try:
        with get_pool().connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM chunks WHERE doc_id = %s LIMIT 1", (doc_id,)
            ).fetchone()
        return row is not None
    except Exception:
        return False


def list_documents(limit: int | None = None, offset: int = 0, groups: list[str] | None = None) -> list[dict]:
    try:
        groups_filter = ""
        params: list = []
        if groups is not None:
            groups_filter = """WHERE (
                c.doc_id NOT IN (SELECT DISTINCT dp.doc_id FROM document_permissions dp)
                OR c.doc_id IN (SELECT dp.doc_id FROM document_permissions dp JOIN groups g ON g.id = dp.group_id WHERE g.name = ANY(%s))
            )"""
            params.append(groups)

        base = f"""SELECT c.doc_id, c.file_path, c.doc_type, c.language,
                    MAX(c.ingested_at) AS ingested_at, COUNT(*) AS num_chunks
                FROM chunks c {groups_filter}
                GROUP BY c.doc_id, c.file_path, c.doc_type, c.language
                ORDER BY MAX(c.ingested_at) DESC"""

        with get_pool().connection() as conn:
            if limit is not None:
                rows = conn.execute(base + " LIMIT %s OFFSET %s", (*params, limit, offset)).fetchall()
            else:
                rows = conn.execute(base + " OFFSET %s", (*params, offset)).fetchall()
        return [
            {
                "doc_id": r[0], "file_path": r[1], "doc_type": r[2], "language": r[3],
                "ingested_at": r[4].isoformat() if r[4] else "", "num_chunks": r[5],
            }
            for r in rows
        ]
    except StorageError:
        raise
    except Exception as e:
        raise StorageError(f"Failed to list documents: {e}") from e


def count_documents(groups: list[str] | None = None) -> int:
    try:
        where_clauses, params = [], []
        if groups is not None:
            where_clauses.append(
                "(c.doc_id NOT IN (SELECT DISTINCT dp.doc_id FROM document_permissions dp) "
                "OR c.doc_id IN (SELECT dp.doc_id FROM document_permissions dp "
                "JOIN groups g ON g.id = dp.group_id WHERE g.name = ANY(%s)))"
            )
            params.append(groups)
        where = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
        with get_pool().connection() as conn:
            return conn.execute(
                f"SELECT COUNT(*) FROM (SELECT DISTINCT c.doc_id FROM chunks c {where}) sub",
                params,
            ).fetchone()[0]
    except StorageError:
        raise
    except Exception as e:
        raise StorageError(f"Failed to count documents: {e}") from e


def remove_document_by_id(doc_id_or_partial: str) -> int:
    try:
        with get_pool().connection() as conn:
            if len(doc_id_or_partial) == 6:
                rows = conn.execute(
                    """SELECT DISTINCT doc_id FROM chunks WHERE doc_id LIKE %s""",
                    (f"%{doc_id_or_partial}",),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT DISTINCT doc_id FROM chunks WHERE doc_id = %s""",
                    (doc_id_or_partial,),
                ).fetchall()
        if not rows:
            return 0
        if len(rows) > 1:
            raise StorageError(f"Ambiguous ID: {doc_id_or_partial} matches multiple documents")
        return _remove_chunks(rows[0][0])
    except StorageError:
        raise
    except Exception as e:
        raise StorageError(f"Failed to remove document: {e}") from e


def remove_document(file_path: str | Path) -> int:
    doc_id = make_doc_id(file_path)
    return _remove_chunks(doc_id)


def _remove_chunks(doc_id: str) -> int:
    try:
        with get_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))
                conn.commit()
                return cur.rowcount
    except StorageError:
        raise
    except Exception as e:
        raise StorageError(f"Failed to remove chunks: {e}") from e


def reset_collection() -> None:
    try:
        with get_pool().connection() as conn:
            conn.execute("TRUNCATE TABLE chunks")
            conn.commit()
    except StorageError:
        raise
    except Exception as e:
        raise StorageError(f"Failed to reset collection: {e}") from e


# ============================================================
# RBAC: users, groups, permissions
# ============================================================

# --- Groups ---

def create_group(name: str, description: str = "") -> dict:
    try:
        with get_pool().connection() as conn:
            row = conn.execute(
                "INSERT INTO groups (name, description) VALUES (%s, %s) RETURNING id, name, description, created_at",
                (name, description),
            ).fetchone()
            conn.commit()
        return {"id": row[0], "name": row[1], "description": row[2], "created_at": row[3].isoformat()}
    except Exception as e:
        raise StorageError(f"Failed to create group: {e}") from e


def list_groups(search: str | None = None, limit: int | None = None, offset: int = 0) -> list[dict]:
    try:
        where, params = "", []
        if search:
            where = "WHERE name ILIKE %s OR description ILIKE %s"
            params = [f"%{search}%", f"%{search}%"]
        query = f"SELECT id, name, description, created_at FROM groups {where} ORDER BY name"
        if limit is not None:
            query += " LIMIT %s OFFSET %s"
            params.extend([limit, offset])
        else:
            query += " OFFSET %s"
            params.append(offset)
        with get_pool().connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return [{"id": r[0], "name": r[1], "description": r[2], "created_at": r[3].isoformat()} for r in rows]
    except Exception as e:
        raise StorageError(f"Failed to list groups: {e}") from e


def count_groups(search: str | None = None) -> int:
    try:
        where, params = "", []
        if search:
            where = "WHERE name ILIKE %s OR description ILIKE %s"
            params = [f"%{search}%", f"%{search}%"]
        with get_pool().connection() as conn:
            return conn.execute(f"SELECT COUNT(*) FROM groups {where}", params).fetchone()[0]
    except Exception as e:
        raise StorageError(f"Failed to count groups: {e}") from e


def get_group(group_id: int) -> dict | None:
    try:
        with get_pool().connection() as conn:
            row = conn.execute("SELECT id, name, description, created_at FROM groups WHERE id = %s", (group_id,)).fetchone()
        if not row:
            return None
        return {"id": row[0], "name": row[1], "description": row[2], "created_at": row[3].isoformat()}
    except Exception as e:
        raise StorageError(f"Failed to get group: {e}") from e


def update_group(group_id: int, name: str | None = None, description: str | None = None) -> dict | None:
    try:
        fields, params = [], []
        if name is not None:
            fields.append("name = %s"); params.append(name)
        if description is not None:
            fields.append("description = %s"); params.append(description)
        if not fields:
            return get_group(group_id)
        params.append(group_id)
        with get_pool().connection() as conn:
            conn.execute(f"UPDATE groups SET {', '.join(fields)} WHERE id = %s", params)
            conn.commit()
        return get_group(group_id)
    except Exception as e:
        raise StorageError(f"Failed to update group: {e}") from e


def delete_group(group_id: int) -> bool:
    try:
        with get_pool().connection() as conn:
            cur = conn.execute("DELETE FROM groups WHERE id = %s", (group_id,))
            conn.commit()
            return cur.rowcount > 0
    except Exception as e:
        raise StorageError(f"Failed to delete group: {e}") from e


# --- Users ---

def create_user(username: str, password_hash: str | None, display_name: str = "",
                email: str = "", is_admin: bool = False, auth_type: str = "local",
                must_change_password: bool = False) -> dict:
    try:
        with get_pool().connection() as conn:
            row = conn.execute(
                """INSERT INTO users (username, password_hash, display_name, email, is_admin, auth_type, must_change_password)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id, username, display_name, email, is_admin, is_active, auth_type, created_at, must_change_password""",
                (username, password_hash, display_name, email, is_admin, auth_type, must_change_password),
            ).fetchone()
            conn.commit()
        return _user_row_to_dict(row)
    except Exception as e:
        raise StorageError(f"Failed to create user: {e}") from e


def _user_row_to_dict(r) -> dict:
    return {
        "id": r[0], "username": r[1], "display_name": r[2], "email": r[3],
        "is_admin": r[4], "is_active": r[5], "auth_type": r[6], "created_at": r[7].isoformat(),
        "must_change_password": r[8],
    }

_USER_COLS = "id, username, display_name, email, is_admin, is_active, auth_type, created_at, must_change_password"


def get_user(user_id: int) -> dict | None:
    try:
        with get_pool().connection() as conn:
            row = conn.execute(f"SELECT {_USER_COLS} FROM users WHERE id = %s", (user_id,)).fetchone()
        return _user_row_to_dict(row) if row else None
    except Exception as e:
        raise StorageError(f"Failed to get user: {e}") from e


def get_user_by_username(username: str) -> dict | None:
    try:
        with get_pool().connection() as conn:
            row = conn.execute(
                f"SELECT {_USER_COLS}, password_hash FROM users WHERE username = %s", (username,)
            ).fetchone()
        if not row:
            return None
        d = _user_row_to_dict(row)
        d["password_hash"] = row[9]
        return d
    except Exception as e:
        raise StorageError(f"Failed to get user: {e}") from e


def list_users(search: str | None = None, limit: int | None = None, offset: int = 0) -> list[dict]:
    try:
        where, params = "", []
        if search:
            where = "WHERE username ILIKE %s OR display_name ILIKE %s OR email ILIKE %s"
            params = [f"%{search}%", f"%{search}%", f"%{search}%"]
        query = f"SELECT {_USER_COLS} FROM users {where} ORDER BY username"
        if limit is not None:
            query += " LIMIT %s OFFSET %s"
            params.extend([limit, offset])
        else:
            query += " OFFSET %s"
            params.append(offset)
        with get_pool().connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return [_user_row_to_dict(r) for r in rows]
    except Exception as e:
        raise StorageError(f"Failed to list users: {e}") from e


def count_users(search: str | None = None) -> int:
    try:
        where, params = "", []
        if search:
            where = "WHERE username ILIKE %s OR display_name ILIKE %s OR email ILIKE %s"
            params = [f"%{search}%", f"%{search}%", f"%{search}%"]
        with get_pool().connection() as conn:
            return conn.execute(f"SELECT COUNT(*) FROM users {where}", params).fetchone()[0]
    except Exception as e:
        raise StorageError(f"Failed to count users: {e}") from e


def update_user(user_id: int, **kwargs) -> dict | None:
    allowed = {"username", "password_hash", "display_name", "email", "is_admin", "is_active", "auth_type", "must_change_password"}
    fields, params = [], []
    for k, v in kwargs.items():
        if k in allowed and v is not None:
            fields.append(f"{k} = %s"); params.append(v)
    if not fields:
        return get_user(user_id)
    try:
        params.append(user_id)
        with get_pool().connection() as conn:
            conn.execute(f"UPDATE users SET {', '.join(fields)} WHERE id = %s", params)
            conn.commit()
        return get_user(user_id)
    except Exception as e:
        raise StorageError(f"Failed to update user: {e}") from e


def delete_user(user_id: int) -> bool:
    try:
        with get_pool().connection() as conn:
            cur = conn.execute("DELETE FROM users WHERE id = %s", (user_id,))
            conn.commit()
            return cur.rowcount > 0
    except Exception as e:
        raise StorageError(f"Failed to delete user: {e}") from e


# --- User-Group assignment ---

def assign_user_to_group(user_id: int, group_id: int) -> bool:
    try:
        with get_pool().connection() as conn:
            conn.execute(
                "INSERT INTO user_groups (user_id, group_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                (user_id, group_id),
            )
            conn.commit()
        return True
    except Exception as e:
        raise StorageError(f"Failed to assign user to group: {e}") from e


def remove_user_from_group(user_id: int, group_id: int) -> bool:
    try:
        with get_pool().connection() as conn:
            cur = conn.execute("DELETE FROM user_groups WHERE user_id = %s AND group_id = %s", (user_id, group_id))
            conn.commit()
            return cur.rowcount > 0
    except Exception as e:
        raise StorageError(f"Failed to remove user from group: {e}") from e


def get_user_groups(user_id: int) -> list[dict]:
    try:
        with get_pool().connection() as conn:
            rows = conn.execute(
                """SELECT g.id, g.name, g.description FROM groups g
                JOIN user_groups ug ON ug.group_id = g.id WHERE ug.user_id = %s ORDER BY g.name""",
                (user_id,),
            ).fetchall()
        return [{"id": r[0], "name": r[1], "description": r[2]} for r in rows]
    except Exception as e:
        raise StorageError(f"Failed to get user groups: {e}") from e


def get_user_group_names(user_id: int) -> list[str]:
    return [g["name"] for g in get_user_groups(user_id)]


# --- Path permissions ---

def add_path_permission(path: str, group_id: int) -> dict:
    path = _normalize_path(path)
    try:
        with get_pool().connection() as conn:
            row = conn.execute(
                """INSERT INTO path_permissions (path, group_id) VALUES (%s, %s)
                ON CONFLICT (path, group_id) DO NOTHING
                RETURNING id, path, group_id, created_at""",
                (path, group_id),
            ).fetchone()
            conn.commit()
        if row:
            return {"id": row[0], "path": row[1], "group_id": row[2], "created_at": row[3].isoformat()}
        return {"path": path, "group_id": group_id, "exists": True}
    except Exception as e:
        raise StorageError(f"Failed to add path permission: {e}") from e


def remove_path_permission(path: str, group_id: int) -> bool:
    path = _normalize_path(path)
    try:
        with get_pool().connection() as conn:
            cur = conn.execute("DELETE FROM path_permissions WHERE path = %s AND group_id = %s", (path, group_id))
            conn.commit()
            return cur.rowcount > 0
    except Exception as e:
        raise StorageError(f"Failed to remove path permission: {e}") from e


def list_path_permissions() -> list[dict]:
    try:
        with get_pool().connection() as conn:
            rows = conn.execute(
                """SELECT pp.id, pp.path, pp.group_id, g.name AS group_name, pp.created_at
                FROM path_permissions pp JOIN groups g ON g.id = pp.group_id
                ORDER BY pp.path, g.name"""
            ).fetchall()
        return [{"id": r[0], "path": r[1], "group_id": r[2], "group_name": r[3], "created_at": r[4].isoformat()} for r in rows]
    except Exception as e:
        raise StorageError(f"Failed to list path permissions: {e}") from e


def get_path_permissions_for_path(path: str) -> list[dict]:
    path = _normalize_path(path)
    try:
        with get_pool().connection() as conn:
            rows = conn.execute(
                """SELECT pp.id, pp.path, pp.group_id, g.name AS group_name
                FROM path_permissions pp JOIN groups g ON g.id = pp.group_id
                WHERE pp.path = %s ORDER BY g.name""",
                (path,),
            ).fetchall()
        return [{"id": r[0], "path": r[1], "group_id": r[2], "group_name": r[3]} for r in rows]
    except Exception as e:
        raise StorageError(f"Failed to get path permissions: {e}") from e


# --- Document permissions (effective, cached) ---

def compute_effective_groups(file_path: str) -> list[int]:
    """Compute effective group IDs for a file by matching all ancestor paths in path_permissions.

    Both file_path and stored path_permissions are normalized via _normalize_path
    (forward slashes, consistent DOCUMENTS_DIR prefix) so they always match.
    """
    normalized = _normalize_path(file_path)
    parts = normalized.split("/")
    # Build all ancestor paths: "docs", "docs/test", "docs/test/File.pdf"
    ancestors = []
    for i in range(1, len(parts) + 1):
        ancestors.append("/".join(parts[:i]))
    try:
        with get_pool().connection() as conn:
            rows = conn.execute(
                """SELECT DISTINCT pp.group_id
                FROM path_permissions pp
                WHERE REPLACE(REPLACE(pp.path, '\\', '/'), '//', '/') = ANY(%s)
                   OR RTRIM(REPLACE(REPLACE(pp.path, '\\', '/'), '//', '/'), '/') = ANY(%s)""",
                (ancestors, ancestors),
            ).fetchall()
        return [r[0] for r in rows]
    except Exception as e:
        raise StorageError(f"Failed to compute effective groups: {e}") from e


def set_document_permissions(doc_id: str, group_ids: list[int]) -> None:
    """Replace document_permissions for a doc_id with the given group_ids."""
    try:
        with get_pool().connection() as conn:
            conn.execute("DELETE FROM document_permissions WHERE doc_id = %s", (doc_id,))
            for gid in group_ids:
                conn.execute(
                    "INSERT INTO document_permissions (doc_id, group_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                    (doc_id, gid),
                )
            conn.commit()
    except Exception as e:
        raise StorageError(f"Failed to set document permissions: {e}") from e


def get_document_permissions(doc_id: str) -> list[dict]:
    try:
        with get_pool().connection() as conn:
            rows = conn.execute(
                """SELECT dp.doc_id, dp.group_id, g.name AS group_name
                FROM document_permissions dp JOIN groups g ON g.id = dp.group_id
                WHERE dp.doc_id = %s ORDER BY g.name""",
                (doc_id,),
            ).fetchall()
        return [{"doc_id": r[0], "group_id": r[1], "group_name": r[2]} for r in rows]
    except Exception as e:
        raise StorageError(f"Failed to get document permissions: {e}") from e


# ============================================================
# Settings
# ============================================================

def ensure_settings_table() -> None:
    """Create the settings table if it doesn't exist."""
    try:
        with get_pool().connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TIMESTAMPTZ DEFAULT now(),
                    updated_by TEXT
                )
            """)
            conn.commit()
    except Exception as e:
        raise StorageError(f"Failed to create settings table: {e}") from e


def list_settings() -> list[dict]:
    try:
        with get_pool().connection() as conn:
            rows = conn.execute(
                "SELECT key, value, updated_at, updated_by FROM settings ORDER BY key"
            ).fetchall()
        return [
            {"key": r[0], "value": r[1], "updated_at": r[2].isoformat() if r[2] else None, "updated_by": r[3]}
            for r in rows
        ]
    except Exception as e:
        raise StorageError(f"Failed to list settings: {e}") from e


def get_setting(key: str) -> dict | None:
    try:
        with get_pool().connection() as conn:
            row = conn.execute(
                "SELECT key, value, updated_at, updated_by FROM settings WHERE key = %s", (key,)
            ).fetchone()
        if not row:
            return None
        return {"key": row[0], "value": row[1], "updated_at": row[2].isoformat() if row[2] else None, "updated_by": row[3]}
    except Exception as e:
        raise StorageError(f"Failed to get setting: {e}") from e


def upsert_setting(key: str, value: str, updated_by: str | None = None) -> dict:
    try:
        with get_pool().connection() as conn:
            row = conn.execute(
                """INSERT INTO settings (key, value, updated_by, updated_at)
                VALUES (%s, %s, %s, now())
                ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_by = EXCLUDED.updated_by, updated_at = now()
                RETURNING key, value, updated_at, updated_by""",
                (key, value, updated_by),
            ).fetchone()
            conn.commit()
        return {"key": row[0], "value": row[1], "updated_at": row[2].isoformat() if row[2] else None, "updated_by": row[3]}
    except Exception as e:
        raise StorageError(f"Failed to upsert setting: {e}") from e


def delete_setting(key: str) -> bool:
    try:
        with get_pool().connection() as conn:
            cur = conn.execute("DELETE FROM settings WHERE key = %s", (key,))
            conn.commit()
            return cur.rowcount > 0
    except Exception as e:
        raise StorageError(f"Failed to delete setting: {e}") from e


def list_chunks(search: str | None = None, doc_id: str | None = None,
                file_name: str | None = None,
                limit: int = 20, offset: int = 0,
                groups: list[str] | None = None) -> list[dict]:
    try:
        where_clauses, params = [], []
        if groups is not None:
            where_clauses.append(
                "(c.doc_id NOT IN (SELECT DISTINCT dp.doc_id FROM document_permissions dp) "
                "OR c.doc_id IN (SELECT dp.doc_id FROM document_permissions dp "
                "JOIN groups g ON g.id = dp.group_id WHERE g.name = ANY(%s)))"
            )
            params.append(groups)
        if doc_id:
            where_clauses.append("c.doc_id::text ILIKE %s")
            params.append(f"%{doc_id}%")
        if file_name:
            where_clauses.append("c.file_path ILIKE %s")
            params.append(f"%{file_name}%")
        if search:
            where_clauses.append("c.text ILIKE %s")
            params.append(f"%{search}%")
        where = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
        query = f"""SELECT c.id, c.doc_id, c.text, c.page_num, c.doc_type,
                        c.language, c.file_path, c.ingested_at, c.chunk_index
                    FROM chunks c {where}
                    ORDER BY c.ingested_at DESC, c.chunk_index
                    LIMIT %s OFFSET %s"""
        params.extend([limit, offset])
        with get_pool().connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return [
            {
                "id": r[0], "doc_id": r[1], "text": r[2][:200] if r[2] else "",
                "page_num": r[3], "doc_type": r[4], "language": r[5],
                "file_path": r[6], "ingested_at": r[7].isoformat() if r[7] else "",
                "chunk_index": r[8],
            }
            for r in rows
        ]
    except StorageError:
        raise
    except Exception as e:
        raise StorageError(f"Failed to list chunks: {e}") from e


def count_chunks(search: str | None = None, doc_id: str | None = None,
                 file_name: str | None = None,
                 groups: list[str] | None = None) -> int:
    try:
        where_clauses, params = [], []
        if groups is not None:
            where_clauses.append(
                "(c.doc_id NOT IN (SELECT DISTINCT dp.doc_id FROM document_permissions dp) "
                "OR c.doc_id IN (SELECT dp.doc_id FROM document_permissions dp "
                "JOIN groups g ON g.id = dp.group_id WHERE g.name = ANY(%s)))"
            )
            params.append(groups)
        if doc_id:
            where_clauses.append("c.doc_id::text ILIKE %s")
            params.append(f"%{doc_id}%")
        if file_name:
            where_clauses.append("c.file_path ILIKE %s")
            params.append(f"%{file_name}%")
        if search:
            where_clauses.append("c.text ILIKE %s")
            params.append(f"%{search}%")
        where = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
        with get_pool().connection() as conn:
            return conn.execute(f"SELECT COUNT(*) FROM chunks c {where}", params).fetchone()[0]
    except StorageError:
        raise
    except Exception as e:
        raise StorageError(f"Failed to count chunks: {e}") from e


def refresh_all_document_permissions() -> int:
    """Recompute document_permissions for all documents based on current path_permissions.

    First normalizes any inconsistent paths in path_permissions to absolute form,
    then recomputes the document_permissions cache in batches to avoid loading
    all documents into memory at once.
    """
    _normalize_stored_paths()
    count = 0
    batch_size = 500
    offset = 0
    while True:
        docs = list_documents(limit=batch_size, offset=offset)
        if not docs:
            break
        for doc in docs:
            group_ids = compute_effective_groups(doc["file_path"])
            set_document_permissions(doc["doc_id"], group_ids)
            count += 1
        offset += batch_size
    return count


def _normalize_stored_paths():
    """Normalize all path_permissions entries to consistent form.

    Fixes legacy entries that may have absolute paths, backslashes, etc.
    """
    try:
        with get_pool().connection() as conn:
            rows = conn.execute("SELECT id, path FROM path_permissions").fetchall()
            for row_id, path in rows:
                normalized = _normalize_path(path)
                if normalized != path:
                    # Check if normalized path + same group already exists (avoid unique conflict)
                    existing = conn.execute(
                        """SELECT id FROM path_permissions
                        WHERE path = %s AND group_id = (SELECT group_id FROM path_permissions WHERE id = %s)
                        AND id != %s""",
                        (normalized, row_id, row_id),
                    ).fetchone()
                    if existing:
                        # Duplicate after normalization — delete this one
                        conn.execute("DELETE FROM path_permissions WHERE id = %s", (row_id,))
                    else:
                        conn.execute(
                            "UPDATE path_permissions SET path = %s WHERE id = %s",
                            (normalized, row_id),
                        )
                    logger.info(f"Normalized path_permission id={row_id}: {path!r} → {normalized!r}")
            conn.commit()
    except Exception as e:
        logger.warning(f"Failed to normalize stored paths: {e}")
