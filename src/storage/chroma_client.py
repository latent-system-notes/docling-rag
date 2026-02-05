import chromadb
import hashlib
import numpy as np
from chromadb.config import Settings as ChromaSettings
from pathlib import Path
from typing import Optional

from ..config import get_chroma_persist_dir, COLLECTION_NAME, EMBEDDING_MODEL, get_logger
from ..models import StorageError

logger = get_logger(__name__)
_chroma_client_cache: Optional[chromadb.ClientAPI] = None

def get_chroma_client() -> chromadb.ClientAPI:
    global _chroma_client_cache
    if _chroma_client_cache is not None:
        return _chroma_client_cache
    try:
        persist_dir = get_chroma_persist_dir()
        persist_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(persist_dir), settings=ChromaSettings(anonymized_telemetry=False))
        _chroma_client_cache = client
        return client
    except Exception as e:
        raise StorageError(f"Failed to connect to ChromaDB: {e}") from e

def cleanup_chroma_client() -> None:
    global _chroma_client_cache
    if _chroma_client_cache is not None:
        try:
            if hasattr(_chroma_client_cache, 'clear_system_cache'):
                _chroma_client_cache.clear_system_cache()
        except Exception:
            pass
        finally:
            _chroma_client_cache = None

def create_collection() -> None:
    try:
        get_chroma_client().get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    except Exception as e:
        raise StorageError(f"Failed to create collection: {e}") from e


CHROMA_MAX_BATCH_SIZE = 5000

def _add_vectors_chromadb_only(ids: list[str], vectors: np.ndarray, metadata: list[dict]) -> None:
    try:
        collection = get_chroma_client().get_collection(COLLECTION_NAME)
        documents = [meta.get("text", "") for meta in metadata]
        metadatas = [{k: v if v is not None else "" for k, v in meta.items() if k != "text" and isinstance(v, (str, int, float, bool))} for meta in metadata]
        for i in range(0, len(ids), CHROMA_MAX_BATCH_SIZE):
            end = i + CHROMA_MAX_BATCH_SIZE
            batch_embeddings = vectors[i:end].tolist()
            collection.add(ids=ids[i:end], embeddings=batch_embeddings, documents=documents[i:end],
                           metadatas=metadatas[i:end])
    except Exception as e:
        raise StorageError(f"Failed to add vectors to ChromaDB: {e}") from e

def rollback_batch(ids: list[str]) -> None:
    try:
        get_chroma_client().get_collection(COLLECTION_NAME).delete(ids=ids)
    except Exception as e:
        raise StorageError(f"Failed to rollback batch: {e}") from e


def add_vectors(ids: list[str], vectors: np.ndarray, metadata: list[dict]) -> None:
    documents = [meta.get("text", "") for meta in metadata]
    try:
        _add_vectors_chromadb_only(ids, vectors, metadata)
        from .bm25_index import get_bm25_index
        bm25_index = get_bm25_index()
        try:
            bm25_index.add_documents_atomic(documents, ids)
        except Exception:
            rollback_batch(ids)
            raise
    except Exception as e:
        raise StorageError(f"Failed to add vectors: {e}") from e

def search_vectors(query_vector: np.ndarray, top_k: int) -> list[dict]:
    try:
        collection = get_chroma_client().get_collection(COLLECTION_NAME)
        if query_vector.ndim == 2:
            query_vector = query_vector[0]
        results = collection.query(query_embeddings=[query_vector.tolist()], n_results=top_k)
        if not results["ids"] or not results["ids"][0]:
            return []
        return [{"id": results["ids"][0][i], "distance": results["distances"][0][i], "score": 1.0 - results["distances"][0][i],
                 "text": results["documents"][0][i], **results["metadatas"][0][i]} for i in range(len(results["ids"][0]))]
    except Exception as e:
        raise StorageError(f"Failed to search vectors: {e}") from e

def get_collection_stats() -> dict:
    try:
        return {"row_count": get_chroma_client().get_collection(COLLECTION_NAME).count()}
    except Exception as e:
        raise StorageError(f"Failed to get collection stats: {e}") from e

def initialize_collections() -> None:
    create_collection()

def reset_collection() -> None:
    try:
        get_chroma_client().delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    from .bm25_index import get_bm25_index
    get_bm25_index().clear()
    create_collection()

def get_stats() -> dict:
    stats = get_collection_stats()
    return {"collection": COLLECTION_NAME, "total_chunks": stats["row_count"],
            "embedding_model": EMBEDDING_MODEL, "vector_store": "chromadb"}

def document_exists(file_path: str | Path) -> bool:
    file_path = Path(file_path) if isinstance(file_path, str) else file_path
    doc_id = hashlib.md5(str(file_path.absolute()).encode()).hexdigest()
    try:
        collection = get_chroma_client().get_collection(COLLECTION_NAME)
        results = collection.get(where={"doc_id": doc_id}, limit=1)
        return len(results['ids']) > 0
    except Exception:
        return False

def _build_docs_map() -> dict[str, dict]:
    collection = get_chroma_client().get_collection(COLLECTION_NAME)
    results = collection.get(include=["metadatas"])
    if not results['ids']:
        return {}
    docs_map = {}
    for i, chunk_id in enumerate(results['ids']):
        metadata = results['metadatas'][i]
        doc_id = metadata.get('doc_id', 'unknown')
        if doc_id not in docs_map:
            docs_map[doc_id] = {'doc_id': doc_id, 'file_path': metadata.get('file_path', 'unknown'),
                'doc_type': metadata.get('doc_type', 'unknown'), 'language': metadata.get('language', 'unknown'),
                'ingested_at': metadata.get('ingested_at', 'unknown'), 'num_chunks': 0}
        docs_map[doc_id]['num_chunks'] += 1
    return docs_map


def get_document_count() -> int:
    return len(_build_docs_map())


def list_documents(limit: int | None = None, offset: int = 0) -> list[dict]:
    docs_map = _build_docs_map()
    if not docs_map:
        return []
    documents = sorted(docs_map.values(), key=lambda x: x['ingested_at'] if x['ingested_at'] != 'unknown' else '', reverse=True)
    if offset > 0:
        documents = documents[offset:]
    if limit and limit > 0:
        documents = documents[:limit]
    return documents

def remove_document_by_id(doc_id_or_partial: str) -> int:
    collection = get_chroma_client().get_collection(COLLECTION_NAME)
    all_docs = list_documents()
    matching_doc = None
    if len(doc_id_or_partial) == 6:
        matches = [doc for doc in all_docs if doc['doc_id'].endswith(doc_id_or_partial)]
        if len(matches) == 0:
            return 0
        elif len(matches) > 1:
            raise StorageError(f"Ambiguous ID: {doc_id_or_partial} matches multiple documents")
        matching_doc = matches[0]
    else:
        matches = [doc for doc in all_docs if doc['doc_id'] == doc_id_or_partial]
        if not matches:
            return 0
        matching_doc = matches[0]
    doc_id = matching_doc['doc_id']
    results = collection.get(where={"doc_id": doc_id})
    if not results['ids']:
        return 0
    num_chunks = len(results['ids'])
    chunk_ids = results['ids']
    collection.delete(where={"doc_id": doc_id})
    from .bm25_index import get_bm25_index
    bm25_index = get_bm25_index()
    bm25_index.remove_documents(chunk_ids)
    bm25_index.save()
    return num_chunks

def remove_document(file_path: str | Path) -> int:
    file_path = Path(file_path) if isinstance(file_path, str) else file_path
    doc_id = hashlib.md5(str(file_path.absolute()).encode()).hexdigest()
    collection = get_chroma_client().get_collection(COLLECTION_NAME)
    results = collection.get(where={"doc_id": doc_id})
    if not results['ids']:
        return 0
    chunk_ids = results['ids']
    collection.delete(where={"doc_id": doc_id})
    from .bm25_index import get_bm25_index
    bm25_index = get_bm25_index()
    bm25_index.remove_documents(chunk_ids)
    bm25_index.save()
    return len(chunk_ids)
