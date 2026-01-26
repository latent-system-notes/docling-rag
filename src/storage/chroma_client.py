from pathlib import Path
from typing import Optional

import chromadb
import numpy as np
from chromadb.config import Settings as ChromaSettings

from ..config import settings, get_logger
from ..models import StorageError

logger = get_logger(__name__)

# Manual cache for ChromaDB client to allow cleanup
_chroma_client_cache: Optional[chromadb.ClientAPI] = None


def get_chroma_client() -> chromadb.ClientAPI:
    """Get ChromaDB persistent client.

    Uses manual caching to allow proper resource cleanup.
    Always uses persistent mode (SQLite).
    """
    global _chroma_client_cache

    # Return cached client if available
    if _chroma_client_cache is not None:
        return _chroma_client_cache

    logger.info(f"Connecting to ChromaDB at {settings.chroma_persist_dir}")

    try:
        settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)

        client = chromadb.PersistentClient(
            path=str(settings.chroma_persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        _chroma_client_cache = client
        return client
    except Exception as e:
        raise StorageError(f"Failed to connect to ChromaDB: {e}") from e


def cleanup_chroma_client() -> None:
    """Cleanup and release the cached ChromaDB client.

    Call this when you want to release database connections and resources.
    Useful for long-running processes or after batch operations.
    """
    global _chroma_client_cache

    if _chroma_client_cache is not None:
        # Try to clear the client properly
        try:
            # For persistent clients, this helps release SQLite connections
            if hasattr(_chroma_client_cache, 'clear_system_cache'):
                _chroma_client_cache.clear_system_cache()
        except Exception as e:
            logger.warning(f"Error during client cleanup: {e}")
        finally:
            _chroma_client_cache = None
            logger.info("ChromaDB client cache cleared")


def create_collection(collection_name: str | None = None) -> None:
    """Create or get a ChromaDB collection.

    Args:
        collection_name: Name of the collection (defaults to config setting)
    """
    collection_name = collection_name or settings.chroma_collection_name

    client = get_chroma_client()

    try:
        client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"Collection ready: {collection_name}")

    except Exception as e:
        raise StorageError(f"Failed to create collection: {e}") from e


def _add_vectors_chromadb_only(
    ids: list[str],
    vectors: np.ndarray,
    metadata: list[dict],
    collection_name: str | None = None,
) -> None:
    """Add vectors to ChromaDB only (without BM25).

    This is a low-level function used for atomic batch commits.
    Use add_vectors() for normal operations.

    Args:
        ids: List of chunk IDs
        vectors: Embedding vectors
        metadata: Chunk metadata (must include 'text' field)
        collection_name: Collection name (defaults to config setting)

    Raises:
        StorageError: If adding vectors fails
    """
    collection_name = collection_name or settings.chroma_collection_name
    client = get_chroma_client()

    try:
        collection = client.get_collection(collection_name)

        documents = [meta.get("text", "") for meta in metadata]
        metadatas = [
            {
                k: v if v is not None else ""
                for k, v in meta.items()
                if k != "text" and isinstance(v, (str, int, float, bool))
            }
            for meta in metadata
        ]

        collection.add(
            ids=ids,
            embeddings=vectors.tolist(),
            documents=documents,
            metadatas=metadatas,
        )

        logger.debug(f"Added {len(ids)} vectors to ChromaDB (BM25 not updated)")

    except Exception as e:
        raise StorageError(f"Failed to add vectors to ChromaDB: {e}") from e


def rollback_batch(ids: list[str], collection_name: str | None = None) -> None:
    """Remove a batch of chunks from ChromaDB (rollback operation).

    Used to rollback a batch when BM25 save fails during atomic commit.

    Args:
        ids: List of chunk IDs to remove
        collection_name: Collection name (defaults to config setting)
    """
    collection_name = collection_name or settings.chroma_collection_name
    client = get_chroma_client()

    try:
        collection = client.get_collection(collection_name)
        # ChromaDB allows deleting by ID list directly
        collection.delete(ids=ids)
        logger.warning(f"Rolled back {len(ids)} vectors from ChromaDB")

    except Exception as e:
        logger.error(f"Failed to rollback batch: {e}")
        raise StorageError(f"Failed to rollback batch: {e}") from e


def add_vectors(
    ids: list[str],
    vectors: np.ndarray,
    metadata: list[dict],
    collection_name: str | None = None,
) -> None:
    """Add vectors to ChromaDB and BM25 atomically.

    This function ensures both ChromaDB and BM25 are updated together.
    If BM25 save fails, ChromaDB changes are rolled back.

    Args:
        ids: List of chunk IDs
        vectors: Embedding vectors
        metadata: Chunk metadata (must include 'text' field)
        collection_name: Collection name (defaults to config setting)

    Raises:
        StorageError: If adding vectors fails
    """
    collection_name = collection_name or settings.chroma_collection_name

    documents = [meta.get("text", "") for meta in metadata]

    try:
        # Step 1: Add to ChromaDB
        _add_vectors_chromadb_only(ids, vectors, metadata, collection_name)

        # Step 2: Add to BM25 and save atomically
        from .bm25_index import get_bm25_index
        bm25_index = get_bm25_index()

        try:
            bm25_index.add_documents_atomic(documents, ids)
        except Exception as bm25_error:
            # Rollback ChromaDB changes
            logger.error(f"BM25 operation failed, rolling back ChromaDB changes")
            rollback_batch(ids, collection_name)
            raise

        logger.info(f"Successfully added {len(ids)} vectors to both ChromaDB and BM25")

    except Exception as e:
        raise StorageError(f"Failed to add vectors: {e}") from e


def search_vectors(
    query_vector: np.ndarray,
    top_k: int,
    collection_name: str | None = None,
    filter_expr: dict | None = None,
) -> list[dict]:
    collection_name = collection_name or settings.chroma_collection_name
    client = get_chroma_client()

    try:
        collection = client.get_collection(collection_name)

        # Flatten query_vector if it's 2D (single query returns [1, embedding_dim])
        if query_vector.ndim == 2:
            query_vector = query_vector[0]

        results = collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=top_k,
            where=filter_expr,
        )

        if not results["ids"] or not results["ids"][0]:
            return []

        formatted = []
        for i in range(len(results["ids"][0])):
            formatted.append(
                {
                    "id": results["ids"][0][i],
                    "distance": results["distances"][0][i],
                    "score": 1.0 - results["distances"][0][i],
                    "text": results["documents"][0][i],
                    **results["metadatas"][0][i],
                }
            )

        return formatted

    except Exception as e:
        raise StorageError(f"Failed to search vectors: {e}") from e


def delete_by_filter(filter_expr: dict, collection_name: str | None = None) -> None:
    collection_name = collection_name or settings.chroma_collection_name
    client = get_chroma_client()

    try:
        collection = client.get_collection(collection_name)
        collection.delete(where=filter_expr)
        logger.info(f"Deleted vectors matching filter")

    except Exception as e:
        raise StorageError(f"Failed to delete vectors: {e}") from e


def get_collection_stats(collection_name: str | None = None) -> dict:
    collection_name = collection_name or settings.chroma_collection_name
    client = get_chroma_client()

    try:
        collection = client.get_collection(collection_name)
        count = collection.count()
        return {"row_count": count}

    except Exception as e:
        raise StorageError(f"Failed to get collection stats: {e}") from e


# ============================================================================
# Collection Management
# ============================================================================


def initialize_collections() -> None:
    """Initialize the default collection."""
    create_collection()
    logger.info("Collections initialized")


def reset_collection(collection_name: str | None = None) -> None:
    """Delete and recreate a collection.

    Args:
        collection_name: Name of collection to reset (defaults to settings)
    """
    collection_name = collection_name or settings.chroma_collection_name
    client = get_chroma_client()

    try:
        client.delete_collection(collection_name)
        logger.info(f"Dropped collection: {collection_name}")
    except Exception:
        pass

    # Clear BM25 index
    from .bm25_index import get_bm25_index
    get_bm25_index().clear()

    create_collection(collection_name)


def get_stats() -> dict:
    """Get statistics about the RAG system.

    Returns:
        Dictionary with collection stats and configuration
    """
    stats = get_collection_stats()
    return {
        "collection": settings.chroma_collection_name,
        "total_chunks": stats["row_count"],
        "embedding_model": settings.embedding_model,
        "vector_store": "chromadb",
    }


def document_exists(file_path: str | Path) -> bool:
    """Check if document is already ingested by file path.

    Uses the same MD5 hash as ingestion to generate doc_id.
    This allows us to skip re-ingesting files that are already in the database.

    Args:
        file_path: Path to the document (can be string or Path object)

    Returns:
        True if document with this doc_id exists in ChromaDB

    Examples:
        >>> if not document_exists("paper.pdf"):
        ...     ingest_document("paper.pdf")
    """
    import hashlib

    file_path = Path(file_path) if isinstance(file_path, str) else file_path
    doc_id = hashlib.md5(str(file_path.absolute()).encode()).hexdigest()

    client = get_chroma_client()

    try:
        collection = client.get_collection(settings.chroma_collection_name)
        # Query by doc_id metadata field since chunk IDs have suffixes like _chunk_0
        results = collection.get(where={"doc_id": doc_id}, limit=1)
        return len(results['ids']) > 0
    except Exception:
        # If collection doesn't exist or any error occurs, assume document doesn't exist
        return False


def list_documents(limit: int | None = None, offset: int = 0) -> list[dict]:
    """List all unique documents in the collection.

    Returns a list of document information including:
    - doc_id: Unique document identifier
    - file_path: Original file path
    - doc_type: File extension
    - language: Detected language
    - num_chunks: Number of chunks
    - ingested_at: Timestamp of ingestion (ISO format)

    Args:
        limit: Maximum number of documents to return (None for all)
        offset: Number of documents to skip (for pagination)

    Returns:
        List of document dictionaries sorted by ingested_at (newest first)
    """
    client = get_chroma_client()
    collection = client.get_collection(settings.chroma_collection_name)

    # Get all items from collection
    # Note: ChromaDB doesn't support pagination in get(), so we fetch all and paginate in memory
    # For very large collections (10k+ docs), consider using HTTP mode with a remote server
    results = collection.get()

    if not results['ids']:
        return []

    # Group chunks by doc_id to get unique documents
    docs_map = {}
    for i, chunk_id in enumerate(results['ids']):
        metadata = results['metadatas'][i]
        doc_id = metadata.get('doc_id', 'unknown')

        # Only add document once (first chunk we encounter)
        if doc_id not in docs_map:
            docs_map[doc_id] = {
                'doc_id': doc_id,
                'file_path': metadata.get('file_path', 'unknown'),
                'doc_type': metadata.get('doc_type', 'unknown'),
                'language': metadata.get('language', 'unknown'),
                'ingested_at': metadata.get('ingested_at', 'unknown'),
                'num_chunks': 0  # Will count below
            }

        # Count chunks for this document
        docs_map[doc_id]['num_chunks'] += 1

    # Convert to list and sort by ingested_at (newest first)
    documents = list(docs_map.values())
    documents.sort(
        key=lambda x: x['ingested_at'] if x['ingested_at'] != 'unknown' else '',
        reverse=True
    )

    # Apply pagination
    if offset > 0:
        documents = documents[offset:]
    if limit is not None and limit > 0:
        documents = documents[:limit]

    return documents


def remove_document_by_id(doc_id_or_partial: str) -> int:
    """Remove a document by its ID (full or partial - last 6 chars).

    Args:
        doc_id_or_partial: Full doc_id or last 6 characters

    Returns:
        Number of chunks removed (0 if document not found)
    """
    client = get_chroma_client()
    collection = client.get_collection(settings.chroma_collection_name)

    # Get all documents
    all_docs = list_documents()

    # Find matching document
    matching_doc = None
    if len(doc_id_or_partial) == 6:
        # Partial ID - match last 6 chars
        matches = [doc for doc in all_docs if doc['doc_id'].endswith(doc_id_or_partial)]
        if len(matches) == 0:
            logger.warning(f"No document found with ID ending in: {doc_id_or_partial}")
            return 0
        elif len(matches) > 1:
            logger.error(f"Multiple documents match ID {doc_id_or_partial}: {[d['doc_id'] for d in matches]}")
            raise StorageError(f"Ambiguous ID: {doc_id_or_partial} matches multiple documents")
        matching_doc = matches[0]
    else:
        # Full ID - exact match
        matches = [doc for doc in all_docs if doc['doc_id'] == doc_id_or_partial]
        if len(matches) == 0:
            logger.warning(f"Document not found: {doc_id_or_partial}")
            return 0
        matching_doc = matches[0]

    doc_id = matching_doc['doc_id']

    # Get all chunks for this document
    results = collection.get(where={"doc_id": doc_id})

    if not results['ids']:
        return 0

    # Delete all chunks for this document
    num_chunks = len(results['ids'])
    chunk_ids = results['ids']
    collection.delete(where={"doc_id": doc_id})

    logger.info(f"Removed document {doc_id} ({num_chunks} chunks)")

    # Incrementally remove from BM25 index
    from .bm25_index import get_bm25_index
    bm25_index = get_bm25_index()
    bm25_index.remove_documents(chunk_ids)
    bm25_index.save()

    return num_chunks


def remove_document(file_path: str | Path) -> int:
    """Remove a document and all its chunks from ChromaDB.

    Args:
        file_path: Path to the document to remove

    Returns:
        Number of chunks removed (0 if document not found)
    """
    import hashlib

    file_path = Path(file_path) if isinstance(file_path, str) else file_path
    doc_id = hashlib.md5(str(file_path.absolute()).encode()).hexdigest()

    client = get_chroma_client()
    collection = client.get_collection(settings.chroma_collection_name)

    # Get all chunks for this document
    results = collection.get(where={"doc_id": doc_id})

    if not results['ids']:
        logger.warning(f"Document not found: {file_path}")
        return 0

    # Delete all chunks for this document
    chunk_ids = results['ids']
    collection.delete(where={"doc_id": doc_id})

    num_removed = len(chunk_ids)
    logger.info(f"Removed document {file_path}: {num_removed} chunks")

    # Incrementally remove from BM25 index
    from .bm25_index import get_bm25_index
    bm25_index = get_bm25_index()
    bm25_index.remove_documents(chunk_ids)
    bm25_index.save()

    return num_removed
