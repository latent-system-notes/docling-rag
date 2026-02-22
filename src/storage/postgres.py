import hashlib
import threading
from pathlib import Path

import numpy as np
from psycopg_pool import ConnectionPool
from pgvector.psycopg import register_vector

from ..config import config, COLLECTION_NAME, EMBEDDING_MODEL, get_logger
from ..models import StorageError

logger = get_logger(__name__)
_pool: ConnectionPool | None = None
_pool_lock = threading.Lock()


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
                f"password={config('POSTGRES_PASSWORD')}"
            )
            pool = ConnectionPool(
                conninfo=conninfo,
                min_size=config("POSTGRES_POOL_MIN"),
                max_size=config("POSTGRES_POOL_MAX"),
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
                            meta.get("file_path", ""),
                            meta.get("ingested_at"),
                            meta.get("chunk_index", 0),
                        ),
                    )
            conn.commit()
    except StorageError:
        raise
    except Exception as e:
        raise StorageError(f"Failed to add vectors: {e}") from e


def search_vectors(query_vector: np.ndarray, top_k: int) -> list[dict]:
    try:
        if query_vector.ndim == 2:
            query_vector = query_vector[0]
        vec_list = query_vector.tolist()
        with get_pool().connection() as conn:
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


def search_fulltext(query: str, top_k: int = 10) -> list[tuple[str, float]]:
    tokens = query.lower().split()
    if not tokens:
        return []
    tsquery = " | ".join(tokens)
    try:
        with get_pool().connection() as conn:
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
    doc_id = hashlib.md5(str(Path(file_path).absolute()).encode()).hexdigest()
    try:
        with get_pool().connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM chunks WHERE doc_id = %s LIMIT 1", (doc_id,)
            ).fetchone()
        return row is not None
    except Exception:
        return False


def list_documents(limit: int | None = None, offset: int = 0) -> list[dict]:
    try:
        with get_pool().connection() as conn:
            if limit is not None:
                rows = conn.execute(
                    """SELECT doc_id, file_path, doc_type, language,
                        MAX(ingested_at) AS ingested_at, COUNT(*) AS num_chunks
                    FROM chunks GROUP BY doc_id, file_path, doc_type, language
                    ORDER BY MAX(ingested_at) DESC LIMIT %s OFFSET %s""",
                    (limit, offset),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT doc_id, file_path, doc_type, language,
                        MAX(ingested_at) AS ingested_at, COUNT(*) AS num_chunks
                    FROM chunks GROUP BY doc_id, file_path, doc_type, language
                    ORDER BY MAX(ingested_at) DESC OFFSET %s""",
                    (offset,),
                ).fetchall()
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
    doc_id = hashlib.md5(str(Path(file_path).absolute()).encode()).hexdigest()
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
