import hashlib
import numpy as np
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from ..config import get_chroma_persist_dir, COLLECTION_NAME, EMBEDDING_MODEL, get_logger
from ..models import StorageError

logger = get_logger(__name__)
_chroma_client_cache = None

CHROMA_MAX_BATCH_SIZE = 5000

def get_chroma_client() -> chromadb.ClientAPI:
    global _chroma_client_cache
    if _chroma_client_cache is not None:
        return _chroma_client_cache
    try:
        persist_dir = get_chroma_persist_dir()
        persist_dir.mkdir(parents=True, exist_ok=True)
        _chroma_client_cache = chromadb.PersistentClient(path=str(persist_dir), settings=ChromaSettings(anonymized_telemetry=False))
        return _chroma_client_cache
    except Exception as e:
        raise StorageError(f"Failed to connect to ChromaDB: {e}") from e

def cleanup_chroma_client() -> None:
    global _chroma_client_cache
    if _chroma_client_cache is not None:
        try:
            _chroma_client_cache.clear_system_cache()
        except Exception:
            pass
        _chroma_client_cache = None

def create_collection() -> None:
    get_chroma_client().get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

def _get_collection():
    return get_chroma_client().get_collection(COLLECTION_NAME)

def add_vectors(ids: list[str], vectors: np.ndarray, metadata: list[dict]) -> None:
    try:
        collection = _get_collection()
        documents = [meta.get("text", "") for meta in metadata]
        metadatas = [{k: v if v is not None else "" for k, v in meta.items() if k != "text" and isinstance(v, (str, int, float, bool))} for meta in metadata]
        for i in range(0, len(ids), CHROMA_MAX_BATCH_SIZE):
            end = i + CHROMA_MAX_BATCH_SIZE
            collection.add(ids=ids[i:end], embeddings=vectors[i:end].tolist(), documents=documents[i:end], metadatas=metadatas[i:end])
        from .bm25_index import get_bm25_index
        get_bm25_index().add_documents_atomic(documents, ids)
    except Exception as e:
        raise StorageError(f"Failed to add vectors: {e}") from e

def search_vectors(query_vector: np.ndarray, top_k: int) -> list[dict]:
    try:
        if query_vector.ndim == 2:
            query_vector = query_vector[0]
        results = _get_collection().query(query_embeddings=[query_vector.tolist()], n_results=top_k)
        if not results["ids"] or not results["ids"][0]:
            return []
        return [{"id": results["ids"][0][i], "distance": results["distances"][0][i], "score": 1.0 - results["distances"][0][i],
                 "text": results["documents"][0][i], **results["metadatas"][0][i]} for i in range(len(results["ids"][0]))]
    except Exception as e:
        raise StorageError(f"Failed to search vectors: {e}") from e

def get_stats() -> dict:
    count = _get_collection().count()
    return {"collection": COLLECTION_NAME, "total_chunks": count, "embedding_model": EMBEDDING_MODEL, "vector_store": "chromadb"}

def reset_collection() -> None:
    try:
        get_chroma_client().delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    from .bm25_index import get_bm25_index
    get_bm25_index().clear()
    create_collection()

def document_exists(file_path: str | Path) -> bool:
    doc_id = hashlib.md5(str(Path(file_path).absolute()).encode()).hexdigest()
    try:
        return len(_get_collection().get(where={"doc_id": doc_id}, limit=1)['ids']) > 0
    except Exception:
        return False

def _build_docs_map() -> dict[str, dict]:
    results = _get_collection().get(include=["metadatas"])
    if not results['ids']:
        return {}
    docs_map = {}
    for i, chunk_id in enumerate(results['ids']):
        meta = results['metadatas'][i]
        doc_id = meta.get('doc_id', 'unknown')
        if doc_id not in docs_map:
            docs_map[doc_id] = {'doc_id': doc_id, 'file_path': meta.get('file_path', 'unknown'),
                'doc_type': meta.get('doc_type', 'unknown'), 'language': meta.get('language', 'unknown'),
                'ingested_at': meta.get('ingested_at', 'unknown'), 'num_chunks': 0}
        docs_map[doc_id]['num_chunks'] += 1
    return docs_map

def list_documents(limit: int | None = None, offset: int = 0) -> list[dict]:
    docs = sorted(_build_docs_map().values(), key=lambda x: x.get('ingested_at', ''), reverse=True)
    docs = docs[offset:]
    return docs[:limit] if limit else docs

def _remove_chunks(doc_id: str) -> int:
    collection = _get_collection()
    results = collection.get(where={"doc_id": doc_id})
    if not results['ids']:
        return 0
    collection.delete(where={"doc_id": doc_id})
    from .bm25_index import get_bm25_index
    get_bm25_index().remove_documents(results['ids'])
    return len(results['ids'])

def remove_document_by_id(doc_id_or_partial: str) -> int:
    all_docs = list_documents()
    if len(doc_id_or_partial) == 6:
        matches = [d for d in all_docs if d['doc_id'].endswith(doc_id_or_partial)]
    else:
        matches = [d for d in all_docs if d['doc_id'] == doc_id_or_partial]
    if not matches:
        return 0
    if len(matches) > 1:
        raise StorageError(f"Ambiguous ID: {doc_id_or_partial} matches multiple documents")
    return _remove_chunks(matches[0]['doc_id'])

def remove_document(file_path: str | Path) -> int:
    doc_id = hashlib.md5(str(Path(file_path).absolute()).encode()).hexdigest()
    return _remove_chunks(doc_id)
