"""
Parallel document ingestion pipeline.

Architecture:
    [Worker 1] -> Parse docs ->
    [Worker 2] -> Parse docs -> [Queue] -> [Single Writer Thread] -> SQLite (ChromaDB + BM25)
    [Worker 3] -> Parse docs ->

Workers do expensive parsing in parallel, single writer handles all DB operations
to avoid SQLite write contention.
"""
import multiprocessing as mp
import threading
import hashlib
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty as QueueEmpty
from typing import List, Optional, Callable, Dict, Any

from datetime import datetime

from src.config import get_logger, set_session_id
from src.ingestion.lock import IngestionLock, IngestionLockError
from src.ingestion.document import load_document, extract_metadata
from src.ingestion.chunker import chunk_document
from src.ingestion.audit_log import log_ingestion
from src.storage.chroma_client import _add_vectors_chromadb_only, rollback_batch
from src.storage.bm25_index import get_bm25_index
from src.utils import embed
from src.ingestion.checkpoint import (
    load_checkpoint,
    create_checkpoint,
    update_checkpoint,
    delete_checkpoint,
    validate_checkpoint,
    validate_databases,
    cleanup_partial_data,
    _compute_file_hash
)
from src.ingestion.status import get_status_manager, StatusManager

logger = get_logger(__name__)

# Default supported file extensions
DEFAULT_EXTENSIONS = ['.pdf', '.docx', '.pptx', '.xlsx', '.txt', '.md', '.html']


@dataclass
class ParallelIngestionConfig:
    """Configuration for parallel document ingestion."""
    num_workers: int = 4
    batch_size: int = 100
    extensions: Optional[List[str]] = None
    recursive: bool = True
    force: bool = False
    resume: bool = True

    def __post_init__(self):
        if self.extensions is None:
            self.extensions = DEFAULT_EXTENSIONS.copy()


@dataclass
class ParallelIngestionResult:
    """Result of parallel document ingestion."""
    total_files: int
    processed: int
    failed: int
    skipped: int
    resumed: int
    total_chunks: int
    duration_seconds: float
    session_id: str = None
    errors: List[dict] = field(default_factory=list)


def collect_files(
    directory: Path,
    extensions: List[str] = None,
    recursive: bool = True
) -> List[Path]:
    """Collect all files to ingest from directory.

    Files are sorted by size (largest first) to optimize parallel processing.
    This ensures large files (which take longer) start processing immediately
    while workers are available, preventing idle workers at the end.

    Args:
        directory: Root directory to scan
        extensions: List of file extensions to include (with dot, e.g., '.pdf')
        recursive: Whether to scan subdirectories

    Returns:
        List of file paths sorted by size (largest first)
    """
    if extensions is None:
        extensions = DEFAULT_EXTENSIONS

    files = []
    for ext in extensions:
        if recursive:
            files.extend(directory.rglob(f'*{ext}'))
        else:
            files.extend(directory.glob(f'*{ext}'))

    # Sort by file size (largest first) to optimize parallel processing
    # Large files take longer, so they should start first
    return sorted(files, key=lambda f: f.stat().st_size, reverse=True)


def _parse_document_worker(
    worker_id: int,
    file_queue: mp.Queue,
    chunk_queue: mp.Queue,
    session_id: str = None,
    status_db_path: Path = None
):
    """Worker process that parses documents and extracts chunks.

    CPU-intensive work happens here in parallel.

    Args:
        worker_id: Unique worker identifier
        file_queue: Queue of file paths to process
        chunk_queue: Queue to send parsed chunks to writer
        session_id: Session ID for status tracking
        status_db_path: Path to status database (required for Windows spawn)
    """
    # Set process name for logging - will appear as %(processName)s in log format
    mp.current_process().name = f"Worker-{worker_id}"
    # Set session_id for logging context in this worker process
    if session_id:
        set_session_id(session_id)
    logger.info("Worker process started")

    # Initialize status manager for this worker
    # On Windows (spawn), we need the explicit path since project settings aren't inherited
    status_mgr = None
    if session_id:
        try:
            import os
            status_mgr = StatusManager(db_path=status_db_path) if status_db_path else StatusManager()
            status_mgr.register_worker(session_id, worker_id, "parser", pid=os.getpid())
            logger.info(f"Status manager initialized (db: {status_mgr.db_path}, session: {session_id})")
        except Exception as e:
            logger.warning(f"Failed to initialize status manager: {e}")

    processed = 0
    failed = 0
    stopped_by_signal = False

    while True:
        try:
            # Check for stop signal
            if status_mgr and session_id:
                try:
                    if status_mgr.check_stop_signal(session_id, worker_id):
                        logger.info("Received stop signal, shutting down...")
                        status_mgr.mark_signal_processed(session_id, worker_id)
                        stopped_by_signal = True
                        break
                except Exception as e:
                    logger.warning(f"Error checking stop signal: {e}")

            # Get next file from queue
            file_path = file_queue.get(timeout=1)

            if file_path is None:  # Poison pill
                break

            try:
                logger.info(f"Parsing {file_path.name}")

                # Update status: parsing started
                if status_mgr and session_id:
                    try:
                        status_mgr.update_worker_status(
                            session_id, worker_id, "parser",
                            status="parsing",
                            current_file=file_path.name,
                            file_started=True
                        )
                    except Exception:
                        pass

                # Helper function to check for stop signal during processing
                def check_stop():
                    if status_mgr and session_id:
                        try:
                            if status_mgr.check_stop_signal(session_id, worker_id):
                                return True
                        except Exception:
                            pass
                    return False

                # PARSE DOCUMENT (CPU-intensive, happens in parallel)
                # Step 1: Load document (OCR, table extraction, etc.)
                doc, page_count = load_document(file_path)

                # Check for stop signal after document loading
                if check_stop():
                    logger.info("Received stop signal after document load, shutting down...")
                    status_mgr.mark_signal_processed(session_id, worker_id)
                    stopped_by_signal = True
                    break

                # Step 2: Compute doc_id and file_hash for checkpoint
                doc_id = hashlib.md5(str(file_path.absolute()).encode()).hexdigest()
                file_hash = _compute_file_hash(file_path)

                # Step 3: Chunk document
                chunks = chunk_document(doc, doc_id=str(file_path))

                # Check for stop signal after chunking
                if check_stop():
                    logger.info("Received stop signal after chunking, shutting down...")
                    status_mgr.mark_signal_processed(session_id, worker_id)
                    stopped_by_signal = True
                    break

                if not chunks:
                    logger.warning(f"No chunks extracted from {file_path.name}")
                    failed += 1

                    # Update status: file failed
                    if status_mgr and session_id:
                        try:
                            status_mgr.update_worker_status(
                                session_id, worker_id, "parser",
                                status="idle",
                                file_failed=True,
                                error_message="No chunks extracted"
                            )
                        except Exception:
                            pass

                    chunk_queue.put({
                        'worker_id': worker_id,
                        'file': str(file_path),
                        'status': 'failed',
                        'error': 'No chunks extracted',
                        'doc_id': doc_id,
                        'doc_type': file_path.suffix.lstrip('.'),
                        'num_pages': page_count
                    })
                    continue

                # Step 4: Extract metadata
                metadata = extract_metadata(doc, file_path, len(chunks), page_count)

                # Free document from memory
                del doc

                # Step 5: Generate embeddings (parallelized across workers)
                # Update status: embedding
                if status_mgr and session_id:
                    try:
                        status_mgr.update_worker_status(
                            session_id, worker_id, "parser",
                            status="embedding",
                            current_file=file_path.name
                        )
                    except Exception:
                        pass

                logger.info(f"Generating embeddings for {file_path.name} ({len(chunks)} chunks)")
                chunk_texts = [chunk.text for chunk in chunks]
                embeddings = embed(chunk_texts, show_progress=False)
                logger.info(f"Embeddings generated for {file_path.name}")

                # Check for stop signal after embedding
                if check_stop():
                    logger.info("Received stop signal after embedding, shutting down...")
                    status_mgr.mark_signal_processed(session_id, worker_id)
                    stopped_by_signal = True
                    break

                # Send parsed chunks WITH embeddings to writer
                # Convert metadata to dict for serialization
                chunk_queue.put({
                    'worker_id': worker_id,
                    'file': str(file_path),
                    'status': 'parsed',
                    'chunks': chunks,
                    'embeddings': embeddings,  # Pre-computed embeddings
                    'metadata': metadata.model_dump(),  # Convert to dict for checkpoint
                    'num_chunks': len(chunks),
                    'doc_id': doc_id,
                    'file_hash': file_hash,
                    'page_count': page_count
                })

                processed += 1
                logger.info(f"Parsed {file_path.name} ({len(chunks)} chunks, embeddings ready)")

                # Update status: file completed
                if status_mgr and session_id:
                    try:
                        status_mgr.update_worker_status(
                            session_id, worker_id, "parser",
                            status="idle",
                            file_completed=True
                        )
                    except Exception:
                        pass

            except Exception as e:
                failed += 1
                logger.error(f"{file_path.name} - {e}")

                # Update status: file failed
                if status_mgr and session_id:
                    try:
                        status_mgr.update_worker_status(
                            session_id, worker_id, "parser",
                            status="idle",
                            file_failed=True,
                            error_message=str(e)
                        )
                    except Exception:
                        pass

                chunk_queue.put({
                    'worker_id': worker_id,
                    'file': str(file_path),
                    'status': 'failed',
                    'error': str(e)
                })

        except mp.queues.Empty:
            # Update heartbeat during idle
            if status_mgr and session_id:
                try:
                    status_mgr.update_worker_status(
                        session_id, worker_id, "parser",
                        status="idle"
                    )
                except Exception:
                    pass
            continue
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            break

    # Update final status
    if status_mgr and session_id:
        try:
            status_mgr.update_worker_status(
                session_id, worker_id, "parser",
                status="stopped" if stopped_by_signal else "finished"
            )
        except Exception:
            pass

    logger.info(f"Finished: {processed} parsed, {failed} failed")
    chunk_queue.put({
        'worker_id': worker_id,
        'status': 'worker_done',
        'processed': processed,
        'failed': failed
    })


def _writer_thread_func(
    chunk_queue: mp.Queue,
    stats: Dict,
    batch_size: int = 100,
    session_id: str = None,
    status_db_path: Path = None
):
    """Single writer thread that consumes parsed chunks and writes to databases.

    All SQLite writes happen here serially (no contention).

    Args:
        chunk_queue: Queue of parsed chunks from workers
        stats: Shared statistics dict
        batch_size: Size of batches for database writes
        session_id: Session ID for status tracking
        status_db_path: Path to status database
    """
    # Set thread name for internal tracking (note: %(processName)s shows process name, not thread name)
    threading.current_thread().name = "Writer"
    logger.info("Writer thread started")

    # Initialize status manager for writer
    status_mgr = None
    if session_id:
        try:
            status_mgr = StatusManager(db_path=status_db_path) if status_db_path else StatusManager()
            status_mgr.register_worker(session_id, 0, "writer")
        except Exception as e:
            logger.warning(f"Failed to initialize status manager: {e}")

    # Initialize BM25 index (ChromaDB accessed via module functions)
    bm25_index = get_bm25_index()

    workers_done = 0
    total_workers = stats.get('num_workers', 0)

    while workers_done < total_workers:
        try:
            # Get next chunk batch from queue
            item = chunk_queue.get(timeout=1)

            if item['status'] == 'worker_done':
                workers_done += 1
                logger.info(f" Worker {item['worker_id']} done "
                           f"({workers_done}/{total_workers})")
                continue

            elif item['status'] == 'failed':
                stats['failed'] += 1
                stats['errors'].append({
                    'file': item.get('file', 'unknown'),
                    'error': item.get('error', 'Unknown error')
                })
                # Log failure to audit CSV
                try:
                    file_path = Path(item.get('file', 'unknown'))
                    log_ingestion(
                        file_path=file_path,
                        doc_id=item.get('doc_id', 'unknown'),
                        doc_type=item.get('doc_type', file_path.suffix.lstrip('.')),
                        language='unknown',
                        num_pages=item.get('num_pages'),
                        num_chunks=0,
                        status="failed",
                        start_time=datetime.now(),  # Approximate
                        end_time=datetime.now(),
                        session_id=session_id
                    )
                except Exception:
                    pass  # Don't fail ingestion due to logging errors
                continue

            elif item['status'] == 'parsed':
                try:
                    chunks = item['chunks']
                    embeddings = item.get('embeddings')  # Pre-computed by workers
                    metadata = item['metadata']
                    file_path = Path(item['file'])
                    doc_id = item['doc_id']
                    file_hash = item['file_hash']
                    file_start_time = datetime.now()  # Track start time for audit log

                    # Update status: writing started
                    if status_mgr and session_id:
                        try:
                            status_mgr.update_worker_status(
                                session_id, 0, "writer",
                                status="writing",
                                current_file=file_path.name,
                                file_started=True
                            )
                        except Exception:
                            pass

                    # CHECKPOINT MANAGEMENT
                    # Load or create checkpoint
                    checkpoint = load_checkpoint(file_path)
                    start_batch = 0
                    resumed_chunk_count = 0

                    if not checkpoint:
                        # Create new checkpoint
                        checkpoint = create_checkpoint(
                            file_path=file_path,
                            doc_id=doc_id,
                            total_chunks=len(chunks),
                            metadata=metadata
                        )
                    else:
                        # Resume from last completed batch
                        if checkpoint.processed_batches:
                            start_batch = max(checkpoint.processed_batches) + 1
                        else:
                            start_batch = 0

                        resumed_chunk_count = start_batch * batch_size
                        stats['resumed'] = stats.get('resumed', 0) + 1
                        stats['resumed_chunks'] = stats.get('resumed_chunks', 0) + resumed_chunk_count

                        logger.info(f" Resuming {file_path.name} from batch {start_batch}")

                    # WRITE TO DATABASES (serialized, no contention)
                    # Process in batches to avoid memory issues
                    total_batches = (len(chunks) + batch_size - 1) // batch_size

                    for batch_idx in range(start_batch, total_batches):
                        batch_start = batch_idx * batch_size
                        batch_end = min(batch_start + batch_size, len(chunks))
                        batch_chunks = chunks[batch_start:batch_end]

                        # Extract data from Chunk objects (not dicts)
                        batch_chunk_ids = [chunk.id for chunk in batch_chunks]
                        batch_texts = [chunk.text for chunk in batch_chunks]

                        # Build metadata for each chunk (matching pipeline.py format)
                        # Handle ingested_at datetime conversion
                        ingested_at = metadata.get("ingested_at")
                        if hasattr(ingested_at, 'isoformat'):
                            ingested_at_str = ingested_at.isoformat()
                        elif isinstance(ingested_at, str):
                            ingested_at_str = ingested_at
                        else:
                            ingested_at_str = datetime.now().isoformat()

                        batch_metadatas = [
                            {
                                "text": chunk.text,
                                "doc_id": doc_id,
                                "page_num": chunk.page_num,
                                "doc_type": metadata.get("doc_type", "unknown"),
                                "language": metadata.get("language", "unknown"),
                                "file_path": metadata.get("file_path", str(file_path)),
                                "ingested_at": ingested_at_str,
                                "chunk_index": chunk.metadata.get("index", batch_start + i),
                            }
                            for i, chunk in enumerate(batch_chunks)
                        ]

                        # Use pre-computed embeddings from worker, or generate if not available
                        if embeddings is not None:
                            batch_embeddings = embeddings[batch_start:batch_end]
                        else:
                            # Fallback: generate embeddings (for backwards compatibility)
                            batch_embeddings = embed(batch_texts, show_progress=(batch_idx == start_batch))

                        # ATOMIC BATCH COMMIT
                        try:
                            # Step 1: Add to ChromaDB
                            _add_vectors_chromadb_only(batch_chunk_ids, batch_embeddings, batch_metadatas)

                            # Step 2: Add to BM25 index atomically
                            # If this fails, rollback ChromaDB
                            try:
                                bm25_index.add_documents_atomic(batch_texts, batch_chunk_ids)
                            except Exception as bm25_error:
                                # Rollback: Remove from ChromaDB
                                logger.error(f" BM25 save failed, rolling back ChromaDB batch {batch_idx}")
                                rollback_batch(batch_chunk_ids)
                                raise

                            # Step 3: Update checkpoint (only if both succeeded)
                            update_checkpoint(file_path, batch_idx)

                        except Exception as e:
                            # Checkpoint NOT updated - batch will retry on resume
                            logger.error(f" Batch {batch_idx} failed: {e}")
                            raise

                    # Delete checkpoint on successful completion
                    delete_checkpoint(file_path)

                    # Update stats
                    stats['processed'] += 1
                    stats['total_chunks'] += item['num_chunks']

                    # Update status: file completed
                    if status_mgr and session_id:
                        try:
                            status_mgr.update_worker_status(
                                session_id, 0, "writer",
                                status="idle",
                                file_completed=True
                            )
                            status_mgr.update_session_stats(
                                session_id,
                                processed_delta=1,
                                chunks_delta=item['num_chunks']
                            )
                        except Exception:
                            pass

                    logger.info(f"Stored {item['num_chunks']} chunks from "
                               f"{file_path.name}")

                    # Log to audit CSV
                    try:
                        log_ingestion(
                            file_path=file_path,
                            doc_id=doc_id,
                            doc_type=metadata.get('doc_type', file_path.suffix.lstrip('.')),
                            language=metadata.get('language', 'unknown'),
                            num_pages=metadata.get('num_pages'),
                            num_chunks=item['num_chunks'],
                            status="completed",
                            start_time=file_start_time,
                            end_time=datetime.now(),
                            session_id=session_id
                        )
                    except Exception:
                        pass  # Don't fail ingestion due to audit log errors

                except Exception as e:
                    stats['failed'] += 1
                    stats['errors'].append({
                        'file': item.get('file', 'unknown'),
                        'error': str(e)
                    })

                    # Update status: file failed
                    if status_mgr and session_id:
                        try:
                            status_mgr.update_worker_status(
                                session_id, 0, "writer",
                                status="idle",
                                file_failed=True,
                                error_message=str(e)
                            )
                            status_mgr.update_session_stats(
                                session_id,
                                failed_delta=1
                            )
                        except Exception:
                            pass

                    logger.error(f"Failed to store chunks: {e}")

                    # Log failure to audit CSV
                    try:
                        log_ingestion(
                            file_path=file_path,
                            doc_id=doc_id if doc_id else "unknown",
                            doc_type=metadata.get('doc_type', file_path.suffix.lstrip('.')) if metadata else "unknown",
                            language=metadata.get('language', 'unknown') if metadata else "unknown",
                            num_pages=metadata.get('num_pages') if metadata else None,
                            num_chunks=item.get('num_chunks', 0),
                            status="failed",
                            start_time=file_start_time,
                            end_time=datetime.now(),
                            session_id=session_id
                        )
                    except Exception:
                        pass  # Don't fail on audit log errors

        except QueueEmpty:
            # Update heartbeat during idle
            if status_mgr and session_id:
                try:
                    status_mgr.update_worker_status(
                        session_id, 0, "writer",
                        status="idle"
                    )
                except Exception:
                    pass
            continue
        except Exception as e:
            logger.error(f"Error: {e}")
            continue

    # Update final status
    if status_mgr and session_id:
        try:
            status_mgr.update_worker_status(
                session_id, 0, "writer",
                status="finished"
            )
        except Exception:
            pass

    logger.info("Finished")
    stats['writer_done'] = True


def _filter_files_by_checkpoint(
    files: List[Path],
    stats: Dict,
    force: bool = False
) -> List[Path]:
    """Filter files based on checkpoint status and database existence.

    Args:
        files: List of file paths to filter
        stats: Statistics dict to update
        force: If True, reprocess all files regardless of status

    Returns:
        List of files that need processing
    """
    from src.storage.chroma_client import document_exists

    logger.info("Checking for existing checkpoints and indexed documents...")
    files_to_process = []
    skipped_completed = 0
    skipped_indexed = 0
    files_to_resume = 0

    for file_path in files:
        # If force mode, process all files
        if force:
            files_to_process.append(file_path)
            continue

        checkpoint = load_checkpoint(file_path)

        if checkpoint:
            # Validate checkpoint (file hash + database consistency)
            if not validate_checkpoint(checkpoint, file_path):
                logger.warning(f"Invalid checkpoint for {file_path.name}, will reprocess")
                cleanup_partial_data(checkpoint.doc_id)
                delete_checkpoint(file_path)
                files_to_process.append(file_path)
            elif not validate_databases(checkpoint, batch_size=100):
                logger.warning(f"Database inconsistency for {file_path.name}, will reprocess")
                cleanup_partial_data(checkpoint.doc_id)
                delete_checkpoint(file_path)
                files_to_process.append(file_path)
            else:
                # Valid checkpoint - check if complete
                total_batches = (checkpoint.total_chunks + 99) // 100  # batch_size = 100
                completed_batches = len(checkpoint.processed_batches)

                if completed_batches >= total_batches:
                    # Fully completed - skip this file
                    logger.info(f"Skipping completed file: {file_path.name}")
                    delete_checkpoint(file_path)
                    skipped_completed += 1
                else:
                    # Partial - will resume
                    logger.info(f"Will resume {file_path.name} from batch {completed_batches}")
                    files_to_process.append(file_path)
                    files_to_resume += 1
        else:
            # No checkpoint - check if already indexed in database
            if document_exists(file_path):
                logger.info(f"Skipping already indexed: {file_path.name}")
                skipped_indexed += 1
            else:
                # Not indexed - fresh start
                files_to_process.append(file_path)

    # Update stats
    stats['skipped'] = skipped_completed + skipped_indexed
    stats['skipped_indexed'] = skipped_indexed
    stats['skipped_completed'] = skipped_completed
    stats['files_to_resume'] = files_to_resume

    total_skipped = skipped_completed + skipped_indexed
    logger.info(f"Processing {len(files_to_process)} files "
               f"(skipped: {total_skipped} [{skipped_indexed} indexed, {skipped_completed} completed], "
               f"resuming: {files_to_resume}, new: {len(files_to_process) - files_to_resume})")

    return files_to_process


def parallel_ingest_documents(
    source: Path,
    config: ParallelIngestionConfig = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> ParallelIngestionResult:
    """Main entry point for parallel document ingestion.

    Args:
        source: Directory containing documents to ingest
        config: Configuration options (uses defaults if None)
        progress_callback: Optional callback for progress updates,
                          called with dict containing 'processed', 'failed',
                          'total', 'rate', 'eta_minutes'

    Returns:
        ParallelIngestionResult with statistics

    Raises:
        ValueError: If source is not a directory or doesn't exist
        IngestionLockError: If another ingestion is already running
    """
    from src.config import settings

    if config is None:
        config = ParallelIngestionConfig()

    if not source.exists():
        raise ValueError(f"Directory not found: {source}")

    if not source.is_dir():
        raise ValueError(f"Not a directory: {source}")

    # Acquire project-level lock to prevent concurrent ingestion
    project_dir = settings.chroma_persist_dir.parent
    lock = IngestionLock(project_dir)

    if not lock.acquire():
        # Get info about existing session for helpful error message
        lock_info = lock.get_lock_info()
        status_mgr = get_status_manager()
        active = status_mgr.get_active_session()

        if active:
            raise IngestionLockError(
                f"Ingestion already running (session {active.session_id}, "
                f"started {active.started_at.strftime('%H:%M:%S')}). "
                f"Use 'rag status' to monitor or 'rag stop' to cancel."
            )
        elif lock_info:
            raise IngestionLockError(
                f"Another ingestion process is running (PID {lock_info.get('pid')}, "
                f"started {lock_info.get('started_at', 'unknown')}). "
                f"If this is stale, delete {lock.lock_file}"
            )
        else:
            raise IngestionLockError("Another ingestion process is running.")

    # Variables for cleanup handler
    session_id = None
    cleanup_done = False
    original_sigint = signal.getsignal(signal.SIGINT)
    original_sigterm = signal.getsignal(signal.SIGTERM)

    def cleanup_handler(signum, frame):
        """Handle interrupt signals by releasing lock and marking session."""
        nonlocal cleanup_done
        if cleanup_done:
            return
        cleanup_done = True

        logger.info("Received interrupt signal, cleaning up...")
        lock.release()
        if session_id:
            try:
                status_mgr = get_status_manager()
                status_mgr.complete_session(session_id, "interrupted")
            except Exception:
                pass

        # Re-raise the signal to allow normal interrupt handling
        signal.signal(signal.SIGINT, original_sigint)
        signal.signal(signal.SIGTERM, original_sigterm)
        raise KeyboardInterrupt

    # Install signal handlers
    signal.signal(signal.SIGINT, cleanup_handler)
    signal.signal(signal.SIGTERM, cleanup_handler)

    try:
        return _parallel_ingest_documents_impl(source, config, progress_callback, lock)
    finally:
        # Restore original signal handlers
        signal.signal(signal.SIGINT, original_sigint)
        signal.signal(signal.SIGTERM, original_sigterm)
        # Release lock
        lock.release()


def _parallel_ingest_documents_impl(
    source: Path,
    config: ParallelIngestionConfig,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]],
    lock: IngestionLock,
) -> ParallelIngestionResult:
    """Implementation of parallel document ingestion (called after lock acquired).

    Args:
        source: Directory containing documents to ingest
        config: Configuration options
        progress_callback: Optional callback for progress updates
        lock: The acquired ingestion lock

    Returns:
        ParallelIngestionResult with statistics
    """

    logger.info("=" * 60)
    logger.info("PARALLEL DOCUMENT INGESTION")
    logger.info("=" * 60)
    logger.info(f"Workers: {config.num_workers} (parallel parsing)")
    logger.info(f"Writer:  1 (serial database writes)")
    logger.info("=" * 60)

    # Collect files
    logger.info(f"Scanning {source}...")
    files = collect_files(source, config.extensions, config.recursive)

    if not files:
        logger.warning(f"No documents found in {source}")
        return ParallelIngestionResult(
            total_files=0,
            processed=0,
            failed=0,
            skipped=0,
            resumed=0,
            total_chunks=0,
            duration_seconds=0.0,
            errors=[]
        )

    logger.info(f"Found {len(files)} documents to ingest")

    # Create queues
    file_queue = mp.Queue()
    chunk_queue = mp.Queue()

    # Shared statistics (managed by writer)
    manager = mp.Manager()
    stats = manager.dict({
        'num_workers': config.num_workers,
        'processed': 0,
        'failed': 0,
        'total_chunks': 0,
        'resumed': 0,
        'resumed_chunks': 0,
        'skipped': 0,
        'files_to_resume': 0,
        'writer_done': False,
        'errors': manager.list()
    })

    # Filter files based on checkpoint status and database existence
    if config.resume:
        files_to_process = _filter_files_by_checkpoint(files, stats, force=config.force)
    else:
        files_to_process = files

    if not files_to_process:
        logger.info("All files already processed!")
        return ParallelIngestionResult(
            total_files=len(files),
            processed=0,
            failed=0,
            skipped=stats.get('skipped', 0),
            resumed=0,
            total_chunks=0,
            duration_seconds=0.0,
            errors=[]
        )

    # Create ingestion session for status tracking
    session_id = None
    status_db_path = None
    try:
        status_mgr = get_status_manager()
        # Get the db path to pass to workers (needed for Windows spawn)
        status_db_path = status_mgr.db_path
        session_id = status_mgr.create_session(
            source_path=str(source),
            num_workers=config.num_workers,
            total_files=len(files),
            skipped_files=stats.get('skipped', 0)
        )
        # Set session_id for logging context
        set_session_id(session_id)
    except Exception as e:
        logger.warning(f"Failed to create status session: {e}")

    # Populate file queue
    for file_path in files_to_process:
        file_queue.put(file_path)

    # Add poison pills for workers
    for _ in range(config.num_workers):
        file_queue.put(None)

    # Start writer thread (single thread, all DB writes)
    writer = threading.Thread(
        target=_writer_thread_func,
        args=(chunk_queue, stats, config.batch_size, session_id, status_db_path),
        daemon=True
    )
    writer.start()

    # Start worker processes (parallel parsing)
    logger.info(f"Starting {config.num_workers} worker processes...")
    workers = []
    for i in range(config.num_workers):
        p = mp.Process(
            target=_parse_document_worker,
            args=(i, file_queue, chunk_queue, session_id, status_db_path)
        )
        p.start()
        workers.append(p)

    # Monitor progress
    start_time = time.time()
    last_processed = 0

    while not stats.get('writer_done', False):
        time.sleep(2)

        current_processed = stats['processed']
        current_failed = stats['failed']
        total_done = current_processed + current_failed

        if total_done > last_processed:
            elapsed = time.time() - start_time
            rate = total_done / elapsed if elapsed > 0 else 0
            remaining = len(files_to_process) - total_done
            eta = remaining / rate if rate > 0 else 0

            # Calculate processing breakdown
            new_count = current_processed - stats.get('resumed', 0)
            resumed_count = stats.get('resumed', 0)
            skipped_count = stats.get('skipped', 0)

            logger.info(f"Progress: {total_done}/{len(files_to_process)} "
                       f"({new_count} new, {resumed_count} resumed, {skipped_count} skipped) | "
                       f"{rate:.1f} docs/sec | ETA: {eta/60:.1f} min")

            # Call progress callback if provided
            if progress_callback:
                progress_callback({
                    'processed': current_processed,
                    'failed': current_failed,
                    'total': len(files_to_process),
                    'rate': rate,
                    'eta_minutes': eta / 60,
                    'resumed': resumed_count,
                    'skipped': skipped_count
                })

            last_processed = total_done

    # Wait for all workers to finish
    for p in workers:
        p.join()

    # Wait for writer to finish
    writer.join(timeout=30)

    # Final summary
    elapsed = time.time() - start_time

    logger.info("")
    logger.info("=" * 60)
    logger.info("INGESTION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total files:     {len(files)}")
    logger.info(f"Processed:       {stats['processed']}")
    logger.info(f"  - New:         {stats['processed'] - stats.get('resumed', 0)}")
    logger.info(f"  - Resumed:     {stats.get('resumed', 0)}")
    logger.info(f"Skipped:         {stats.get('skipped', 0)} (already completed)")
    logger.info(f"Failed:          {stats['failed']}")
    logger.info(f"Total chunks:    {stats['total_chunks']}")
    if stats.get('resumed_chunks', 0) > 0:
        logger.info(f"  - Resumed:     {stats['resumed_chunks']} chunks skipped")
    logger.info(f"Time:            {elapsed/60:.1f} minutes")
    if stats['processed'] > 0:
        logger.info(f"Rate:            {stats['processed']/elapsed:.2f} docs/sec")
    logger.info("=" * 60)

    # Complete the session
    if session_id:
        try:
            status_mgr = get_status_manager()
            status_mgr.complete_session(session_id, "completed")
        except Exception as e:
            logger.warning(f"Failed to complete status session: {e}")

    # Convert manager list to regular list
    errors = list(stats.get('errors', []))

    return ParallelIngestionResult(
        total_files=len(files),
        processed=stats['processed'],
        failed=stats['failed'],
        skipped=stats.get('skipped', 0),
        resumed=stats.get('resumed', 0),
        total_chunks=stats['total_chunks'],
        duration_seconds=elapsed,
        session_id=session_id,
        errors=errors
    )
