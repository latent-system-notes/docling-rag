import hashlib
from pathlib import Path
from typing import Optional

import chromadb
import numpy as np
from chromadb.config import Settings as ChromaSettings

from ..config import settings, get_logger
from ..models import StorageError

logger = get_logger(__name__)
_chroma_client_cache: Optional[chromadb.ClientAPI] = None

def get_chroma_client() -> chromadb.ClientAPI:
    global _chroma_client_cache
    if _chroma_client_cache is not None:
        return _chroma_client_cache
    try:
        settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(settings.chroma_persist_dir), settings=ChromaSettings(anonymized_telemetry=False))
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

def create_collection(collection_name: str | None = None) -> None:
    collection_name = collection_name or settings.chroma_collection_name
    try:
        get_chroma_client().get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
    except Exception as e:
        raise StorageError(f"Failed to create collection: {e}") from e

def _add_vectors_chromadb_only(ids: list[str], vectors: np.ndarray, metadata: list[dict], collection_name: str | None = None) -> None:
    collection_name = collection_name or settings.chroma_collection_name
    try:
        collection = get_chroma_client().get_collection(collection_name)
        documents = [meta.get("text", "") for meta in metadata]
        metadatas = [{k: v if v is not None else "" for k, v in meta.items() if k != "text" and isinstance(v, (str, int, float, bool))} for meta in metadata]
        collection.add(ids=ids, embeddings=vectors.tolist(), documents=documents, metadatas=metadatas)
    except Exception as e:
        raise StorageError(f"Failed to add vectors to ChromaDB: {e}") from e

def rollback_batch(ids: list[str], collection_name: str | None = None) -> None:
    collection_name = collection_name or settings.chroma_collection_name
    try:
        get_chroma_client().get_collection(collection_name).delete(ids=ids)
    except Exception as e:
        raise StorageError(f"Failed to rollback batch: {e}") from e

def add_vectors(ids: list[str], vectors: np.ndarray, metadata: list[dict], collection_name: str | None = None) -> None:
    collection_name = collection_name or settings.chroma_collection_name
    documents = [meta.get("text", "") for meta in metadata]
    try:
        _add_vectors_chromadb_only(ids, vectors, metadata, collection_name)
        from .bm25_index import get_bm25_index
        bm25_index = get_bm25_index()
        try:
            bm25_index.add_documents_atomic(documents, ids)
        except Exception:
            rollback_batch(ids, collection_name)
            raise
    except Exception as e:
        raise StorageError(f"Failed to add vectors: {e}") from e

def search_vectors(query_vector: np.ndarray, top_k: int, collection_name: str | None = None, filter_expr: dict | None = None) -> list[dict]:
    collection_name = collection_name or settings.chroma_collection_name
    try:
        collection = get_chroma_client().get_collection(collection_name)
        if query_vector.ndim == 2:
            query_vector = query_vector[0]
        results = collection.query(query_embeddings=[query_vector.tolist()], n_results=top_k, where=filter_expr)
        if not results["ids"] or not results["ids"][0]:
            return []
        return [{"id": results["ids"][0][i], "distance": results["distances"][0][i], "score": 1.0 - results["distances"][0][i],
                 "text": results["documents"][0][i], **results["metadatas"][0][i]} for i in range(len(results["ids"][0]))]
    except Exception as e:
        raise StorageError(f"Failed to search vectors: {e}") from e

def delete_by_filter(filter_expr: dict, collection_name: str | None = None) -> None:
    collection_name = collection_name or settings.chroma_collection_name
    try:
        get_chroma_client().get_collection(collection_name).delete(where=filter_expr)
    except Exception as e:
        raise StorageError(f"Failed to delete vectors: {e}") from e

def get_collection_stats(collection_name: str | None = None) -> dict:
    collection_name = collection_name or settings.chroma_collection_name
    try:
        return {"row_count": get_chroma_client().get_collection(collection_name).count()}
    except Exception as e:
        raise StorageError(f"Failed to get collection stats: {e}") from e

def initialize_collections() -> None:
    create_collection()

def reset_collection(collection_name: str | None = None) -> None:
    collection_name = collection_name or settings.chroma_collection_name
    try:
        get_chroma_client().delete_collection(collection_name)
    except Exception:
        pass
    from .bm25_index import get_bm25_index
    get_bm25_index().clear()
    create_collection(collection_name)

def get_stats() -> dict:
    stats = get_collection_stats()
    return {"collection": settings.chroma_collection_name, "total_chunks": stats["row_count"],
            "embedding_model": settings.embedding_model, "vector_store": "chromadb"}

def document_exists(file_path: str | Path) -> bool:
    file_path = Path(file_path) if isinstance(file_path, str) else file_path
    doc_id = hashlib.md5(str(file_path.absolute()).encode()).hexdigest()
    try:
        collection = get_chroma_client().get_collection(settings.chroma_collection_name)
        results = collection.get(where={"doc_id": doc_id}, limit=1)
        return len(results['ids']) > 0
    except Exception:
        return False

def list_documents(limit: int | None = None, offset: int = 0) -> list[dict]:
    collection = get_chroma_client().get_collection(settings.chroma_collection_name)
    results = collection.get()
    if not results['ids']:
        return []
    docs_map = {}
    for i, chunk_id in enumerate(results['ids']):
        metadata = results['metadatas'][i]
        doc_id = metadata.get('doc_id', 'unknown')
        if doc_id not in docs_map:
            docs_map[doc_id] = {'doc_id': doc_id, 'file_path': metadata.get('file_path', 'unknown'),
                'doc_type': metadata.get('doc_type', 'unknown'), 'language': metadata.get('language', 'unknown'),
                'ingested_at': metadata.get('ingested_at', 'unknown'), 'num_chunks': 0}
        docs_map[doc_id]['num_chunks'] += 1
    documents = sorted(docs_map.values(), key=lambda x: x['ingested_at'] if x['ingested_at'] != 'unknown' else '', reverse=True)
    if offset > 0:
        documents = documents[offset:]
    if limit and limit > 0:
        documents = documents[:limit]
    return documents

def remove_document_by_id(doc_id_or_partial: str) -> int:
    collection = get_chroma_client().get_collection(settings.chroma_collection_name)
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
    collection = get_chroma_client().get_collection(settings.chroma_collection_name)
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
