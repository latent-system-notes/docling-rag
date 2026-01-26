from datetime import datetime
from pathlib import Path

from ..config import get_logger
from ..utils import embed
from ..models import IngestionError, DocumentMetadata
from .chunker import chunk_document
from .document import extract_metadata, load_document
from ..storage.chroma_client import _add_vectors_chromadb_only, rollback_batch
from ..storage.bm25_index import get_bm25_index
from .checkpoint import load_checkpoint, create_checkpoint, update_checkpoint, delete_checkpoint, validate_checkpoint, validate_databases, cleanup_partial_data

logger = get_logger(__name__)

def ingest_document(file_path: str | Path, batch_size: int = 100, cleanup_after: bool = False, resume: bool = True) -> DocumentMetadata:
    start_time = datetime.now()
    page_count = None

    try:
        file_path = Path(file_path)
        if not file_path.exists():
            raise IngestionError(f"File not found: {file_path}")

        checkpoint = load_checkpoint(file_path) if resume else None
        chunks = None
        metadata = None
        start_batch = 0
        doc_id = None

        if checkpoint:
            if not validate_checkpoint(checkpoint, file_path):
                cleanup_partial_data(checkpoint.doc_id)
                delete_checkpoint(file_path)
                checkpoint = None
            elif not validate_databases(checkpoint, batch_size):
                cleanup_partial_data(checkpoint.doc_id)
                delete_checkpoint(file_path)
                checkpoint = None

        if checkpoint:
            doc, page_count = load_document(file_path)
            chunks = chunk_document(doc, doc_id=str(file_path))
            if not chunks:
                raise IngestionError(f"No chunks generated for {file_path}")
            if len(chunks) != checkpoint.total_chunks:
                cleanup_partial_data(checkpoint.doc_id)
                delete_checkpoint(file_path)
                checkpoint = None
            else:
                metadata = DocumentMetadata(**checkpoint.metadata)
                doc_id = checkpoint.doc_id
                start_batch = max(checkpoint.processed_batches) + 1 if checkpoint.processed_batches else 0
            del doc

        if not checkpoint:
            doc, page_count = load_document(file_path)
            chunks = chunk_document(doc, doc_id=str(file_path))
            if not chunks:
                raise IngestionError(f"No chunks generated for {file_path}")
            metadata = extract_metadata(doc, file_path, len(chunks), page_count)
            doc_id = metadata.doc_id
            create_checkpoint(file_path=file_path, doc_id=doc_id, total_chunks=len(chunks), metadata=metadata.model_dump())
            del doc
            start_batch = 0

        total_chunks = len(chunks)
        total_batches = (total_chunks + batch_size - 1) // batch_size

        for batch_idx in range(start_batch, total_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, total_chunks)
            batch_chunks = chunks[batch_start:batch_end]
            batch_texts = [chunk.text for chunk in batch_chunks]
            batch_ids = [chunk.id for chunk in batch_chunks]

            try:
                batch_embeddings = embed(batch_texts, show_progress=(batch_idx == start_batch))
                batch_metadata = [{"text": chunk.text, "doc_id": doc_id, "page_num": chunk.page_num,
                    "doc_type": metadata.doc_type, "language": metadata.language, "file_path": metadata.file_path,
                    "ingested_at": metadata.ingested_at.isoformat(), "chunk_index": chunk.metadata.get("index", batch_start)}
                    for chunk in batch_chunks]

                try:
                    _add_vectors_chromadb_only(batch_ids, batch_embeddings, batch_metadata)
                    bm25 = get_bm25_index()
                    try:
                        bm25.add_documents_atomic(batch_texts, batch_ids)
                    except Exception:
                        rollback_batch(batch_ids)
                        raise
                    update_checkpoint(file_path, batch_idx)
                except Exception as e:
                    raise IngestionError(f"Batch processing failed: {e}") from e
                del batch_chunks, batch_texts, batch_embeddings, batch_ids, batch_metadata
            except Exception:
                raise

        del chunks
        delete_checkpoint(file_path)
        logger.info(f"Ingested {file_path}: {total_chunks} chunks")

        if cleanup_after:
            from ..utils import cleanup_all_resources
            cleanup_all_resources()
        return metadata

    except Exception as e:
        if isinstance(e, IngestionError):
            raise
        raise IngestionError(f"Failed to ingest {file_path}: {e}") from e
