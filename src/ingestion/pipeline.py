from datetime import datetime
from pathlib import Path

from ..config import settings, get_logger
from ..utils import embed
from ..models import IngestionError, DocumentMetadata
from .chunker import chunk_document
from .document import extract_metadata, load_document
from ..storage.chroma_client import _add_vectors_chromadb_only, rollback_batch
from ..storage.bm25_index import get_bm25_index
from .checkpoint import (
    load_checkpoint,
    create_checkpoint,
    update_checkpoint,
    delete_checkpoint,
    validate_checkpoint,
    validate_databases,
    cleanup_partial_data,
)
from .audit_log import log_ingestion

logger = get_logger(__name__)


def ingest_document(
    file_path: str | Path,
    batch_size: int = 100,
    cleanup_after: bool = False,
    resume: bool = True
) -> DocumentMetadata:
    """Ingest a document into the RAG system with checkpoint-based resumption.

    This function supports resumable ingestion with atomic batch commits.
    If ingestion fails mid-way, it can be resumed from the last successful checkpoint.

    Memory-optimized version that processes embeddings in batches and avoids
    storing duplicate text in metadata.

    Args:
        file_path: Path to the document file
        batch_size: Number of chunks to process at once (default: 100)
        cleanup_after: If True, cleanup cached resources after ingestion
        resume: If True, resume from checkpoint if exists (default: True)

    Returns:
        DocumentMetadata with ingestion details

    Raises:
        IngestionError: If ingestion fails
    """
    start_time = datetime.now()
    page_count = None
    is_resumed = False

    try:
        file_path = Path(file_path)

        if not file_path.exists():
            raise IngestionError(f"File not found: {file_path}")

        # ===================================================================
        # Step 1: Check for existing checkpoint
        # ===================================================================
        checkpoint = load_checkpoint(file_path) if resume else None
        chunks = None
        metadata = None
        start_batch = 0
        doc_id = None

        if checkpoint:
            logger.info(f"Found existing checkpoint with {len(checkpoint.processed_batches)} completed batches")

            # Validate checkpoint is still valid
            if not validate_checkpoint(checkpoint, file_path):
                logger.warning("Checkpoint invalid (file modified), restarting fresh")
                cleanup_partial_data(checkpoint.doc_id)
                delete_checkpoint(file_path)
                checkpoint = None
            else:
                # Validate databases are consistent
                if not validate_databases(checkpoint, batch_size):
                    logger.error("Database inconsistency detected, restarting fresh")
                    cleanup_partial_data(checkpoint.doc_id)
                    delete_checkpoint(file_path)
                    checkpoint = None
                else:
                    logger.info("Checkpoint and databases validated, resuming ingestion")

        # ===================================================================
        # Step 2: Load and chunk document
        # ===================================================================
        if checkpoint:
            # Resume mode: Load document and regenerate chunks
            logger.info(f"Resuming ingestion from checkpoint...")
            doc, page_count = load_document(file_path)
            chunks = chunk_document(doc, doc_id=str(file_path))
            is_resumed = True

            if not chunks:
                raise IngestionError(f"No chunks generated for {file_path}")

            # Validate chunk count matches checkpoint
            if len(chunks) != checkpoint.total_chunks:
                logger.error(
                    f"Chunk count mismatch: checkpoint has {checkpoint.total_chunks}, "
                    f"but document now has {len(chunks)} chunks. Restarting fresh."
                )
                cleanup_partial_data(checkpoint.doc_id)
                delete_checkpoint(file_path)
                checkpoint = None
            else:
                # Use metadata from checkpoint
                metadata = DocumentMetadata(**checkpoint.metadata)
                doc_id = checkpoint.doc_id

                # Calculate which batches are already done
                if checkpoint.processed_batches:
                    start_batch = max(checkpoint.processed_batches) + 1
                else:
                    start_batch = 0

                logger.info(f"Resuming from batch {start_batch}")

            # Free the document from memory
            del doc

        if not checkpoint:
            # Fresh start
            logger.info(f"Loading document: {file_path}")
            doc, page_count = load_document(file_path)

            logger.info(f"Chunking document...")
            chunks = chunk_document(doc, doc_id=str(file_path))

            if not chunks:
                raise IngestionError(f"No chunks generated for {file_path}")

            metadata = extract_metadata(doc, file_path, len(chunks), page_count)
            doc_id = metadata.doc_id

            # Create checkpoint
            create_checkpoint(
                file_path=file_path,
                doc_id=doc_id,
                total_chunks=len(chunks),
                metadata=metadata.model_dump()
            )

            # Free the document from memory
            del doc
            start_batch = 0

        # ===================================================================
        # Step 3: Process chunks in batches with atomic commits
        # ===================================================================
        total_chunks = len(chunks)
        total_batches = (total_chunks + batch_size - 1) // batch_size

        # Determine if we have page information for progress display
        has_page_info = chunks[0].page_num is not None if chunks else False

        logger.info(f"Processing {total_chunks} chunks in {total_batches} batches of {batch_size}...")

        for batch_idx in range(start_batch, total_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, total_chunks)
            batch_chunks = chunks[batch_start:batch_end]

            batch_texts = [chunk.text for chunk in batch_chunks]
            batch_ids = [chunk.id for chunk in batch_chunks]

            try:
                # Generate embeddings for this batch
                batch_embeddings = embed(batch_texts, show_progress=(batch_idx == start_batch))

                # Prepare metadata WITHOUT duplicating text
                batch_metadata = [
                    {
                        "text": chunk.text,  # Text goes to documents parameter
                        "doc_id": doc_id,
                        "page_num": chunk.page_num,
                        "doc_type": metadata.doc_type,
                        "language": metadata.language,
                        "file_path": metadata.file_path,
                        "ingested_at": metadata.ingested_at.isoformat(),
                        "chunk_index": chunk.metadata.get("index", batch_start),
                    }
                    for chunk in batch_chunks
                ]

                # ============================================================
                # ATOMIC BATCH COMMIT: Both ChromaDB and BM25 must succeed
                # ============================================================
                try:
                    # Step 1: Add to ChromaDB only
                    _add_vectors_chromadb_only(batch_ids, batch_embeddings, batch_metadata)

                    # Step 2: Add to BM25 and save atomically
                    # If this fails, we rollback ChromaDB
                    bm25 = get_bm25_index()
                    try:
                        bm25.add_documents_atomic(batch_texts, batch_ids)
                    except Exception as bm25_error:
                        # Rollback: Remove from ChromaDB
                        logger.error(f"BM25 save failed, rolling back ChromaDB batch {batch_idx}")
                        rollback_batch(batch_ids)
                        raise

                    # Step 3: ONLY NOW update checkpoint (both databases succeeded)
                    update_checkpoint(file_path, batch_idx)

                    # Show progress with page information if available
                    if has_page_info:
                        # Get page range for this batch
                        page_nums = [c.page_num for c in batch_chunks if c.page_num is not None]
                        if page_nums:
                            min_page = min(page_nums)
                            max_page = max(page_nums)
                            page_info = f", pages {min_page}-{max_page}" if min_page != max_page else f", page {min_page}"
                        else:
                            page_info = ""
                    else:
                        page_info = ""

                    logger.info(
                        f"  Batch {batch_idx + 1}/{total_batches} completed "
                        f"({len(batch_ids)} chunks{page_info})"
                    )

                except Exception as e:
                    # Checkpoint NOT updated, will retry this batch on resume
                    logger.error(f"Batch {batch_idx} failed: {e}")
                    raise IngestionError(f"Batch processing failed: {e}") from e

                # Free batch from memory
                del batch_chunks, batch_texts, batch_embeddings, batch_ids, batch_metadata

            except Exception as e:
                # Let the exception propagate, checkpoint was not updated
                logger.error(f"Failed to process batch {batch_idx}: {e}")
                raise

        # ===================================================================
        # Step 4: Success - Delete checkpoint
        # ===================================================================
        # Free chunks list from memory
        del chunks

        delete_checkpoint(file_path)
        logger.info(f"Successfully ingested {file_path}: {total_chunks} chunks")

        # Log to audit CSV
        end_time = datetime.now()
        log_ingestion(
            file_path=file_path,
            doc_id=metadata.doc_id,
            doc_type=metadata.doc_type,
            language=metadata.language,
            num_pages=page_count,
            num_chunks=metadata.num_chunks,
            status="resumed" if is_resumed else "completed",
            start_time=start_time,
            end_time=end_time
        )

        # Optionally cleanup resources after ingestion
        if cleanup_after:
            from ..utils import cleanup_all_resources
            cleanup_all_resources()

        return metadata

    except Exception as e:
        # Log failure to audit CSV
        end_time = datetime.now()
        try:
            log_ingestion(
                file_path=file_path,
                doc_id=doc_id if doc_id else "unknown",
                doc_type=file_path.suffix.lstrip('.') if hasattr(file_path, 'suffix') else "unknown",
                language=metadata.language if metadata else "unknown",
                num_pages=page_count,
                num_chunks=len(chunks) if chunks else 0,
                status="failed",
                start_time=start_time,
                end_time=end_time
            )
        except Exception:
            # Don't fail the failure - silently ignore audit log errors
            pass

        if isinstance(e, IngestionError):
            raise
        raise IngestionError(f"Failed to ingest {file_path}: {e}") from e
