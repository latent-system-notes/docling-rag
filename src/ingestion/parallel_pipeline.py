import hashlib
import multiprocessing as mp
import signal
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from queue import Empty as QueueEmpty
from typing import List, Optional, Callable, Dict, Any

from src.config import get_logger
from src.ingestion.lock import IngestionLock, IngestionLockError
from src.ingestion.document import load_document, extract_metadata
from src.ingestion.chunker import chunk_document
from src.storage.chroma_client import _add_vectors_chromadb_only, rollback_batch
from src.storage.bm25_index import get_bm25_index
from src.utils import embed
from src.ingestion.checkpoint import load_checkpoint, create_checkpoint, update_checkpoint, delete_checkpoint, validate_checkpoint, validate_databases, cleanup_partial_data, _compute_file_hash

logger = get_logger(__name__)
DEFAULT_EXTENSIONS = ['.pdf', '.docx', '.pptx', '.xlsx', '.txt', '.md', '.html']

@dataclass
class ParallelIngestionConfig:
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
    total_files: int
    processed: int
    failed: int
    skipped: int
    resumed: int
    total_chunks: int
    duration_seconds: float
    errors: List[dict] = field(default_factory=list)

def collect_files(directory: Path, extensions: List[str] = None, recursive: bool = True) -> List[Path]:
    if extensions is None:
        extensions = DEFAULT_EXTENSIONS
    files = []
    for ext in extensions:
        files.extend(directory.rglob(f'*{ext}') if recursive else directory.glob(f'*{ext}'))
    return sorted(files, key=lambda f: f.stat().st_size, reverse=True)

def _parse_document_worker(worker_id: int, file_queue: mp.Queue, chunk_queue: mp.Queue):
    mp.current_process().name = f"Worker-{worker_id}"
    processed, failed = 0, 0

    while True:
        try:
            file_path = file_queue.get(timeout=1)
            if file_path is None:
                break
            try:
                doc, page_count = load_document(file_path)
                doc_id = hashlib.md5(str(file_path.absolute()).encode()).hexdigest()
                file_hash = _compute_file_hash(file_path)
                chunks = chunk_document(doc, doc_id=str(file_path))

                if not chunks:
                    failed += 1
                    chunk_queue.put({'worker_id': worker_id, 'file': str(file_path), 'status': 'failed', 'error': 'No chunks extracted',
                        'doc_id': doc_id, 'doc_type': file_path.suffix.lstrip('.'), 'num_pages': page_count})
                    continue

                metadata = extract_metadata(doc, file_path, len(chunks), page_count)
                del doc

                chunk_texts = [chunk.text for chunk in chunks]
                embeddings = embed(chunk_texts, show_progress=False)

                chunk_queue.put({'worker_id': worker_id, 'file': str(file_path), 'status': 'parsed', 'chunks': chunks,
                    'embeddings': embeddings, 'metadata': metadata.model_dump(), 'num_chunks': len(chunks),
                    'doc_id': doc_id, 'file_hash': file_hash, 'page_count': page_count})
                processed += 1

            except Exception as e:
                failed += 1
                logger.error(f"{file_path.name} - {e}")
                chunk_queue.put({'worker_id': worker_id, 'file': str(file_path), 'status': 'failed', 'error': str(e)})

        except mp.queues.Empty:
            continue
        except Exception:
            break

    chunk_queue.put({'worker_id': worker_id, 'status': 'worker_done', 'processed': processed, 'failed': failed})

def _writer_thread_func(chunk_queue: mp.Queue, stats: Dict, batch_size: int = 100):
    threading.current_thread().name = "Writer"
    bm25_index = get_bm25_index()
    workers_done = 0
    total_workers = stats.get('num_workers', 0)

    while workers_done < total_workers:
        try:
            item = chunk_queue.get(timeout=1)
            if item['status'] == 'worker_done':
                workers_done += 1
                continue
            elif item['status'] == 'failed':
                stats['failed'] += 1
                stats['errors'].append({'file': item.get('file', 'unknown'), 'error': item.get('error', 'Unknown error')})
                continue
            elif item['status'] == 'parsed':
                try:
                    chunks = item['chunks']
                    embeddings = item.get('embeddings')
                    metadata = item['metadata']
                    file_path = Path(item['file'])
                    doc_id = item['doc_id']

                    checkpoint = load_checkpoint(file_path)
                    start_batch = 0
                    if not checkpoint:
                        checkpoint = create_checkpoint(file_path=file_path, doc_id=doc_id, total_chunks=len(chunks), metadata=metadata)
                    else:
                        start_batch = max(checkpoint.processed_batches) + 1 if checkpoint.processed_batches else 0
                        stats['resumed'] = stats.get('resumed', 0) + 1
                        stats['resumed_chunks'] = stats.get('resumed_chunks', 0) + start_batch * batch_size

                    total_batches = (len(chunks) + batch_size - 1) // batch_size
                    for batch_idx in range(start_batch, total_batches):
                        batch_start = batch_idx * batch_size
                        batch_end = min(batch_start + batch_size, len(chunks))
                        batch_chunks = chunks[batch_start:batch_end]
                        batch_chunk_ids = [chunk.id for chunk in batch_chunks]
                        batch_texts = [chunk.text for chunk in batch_chunks]

                        ingested_at = metadata.get("ingested_at")
                        if hasattr(ingested_at, 'isoformat'):
                            ingested_at_str = ingested_at.isoformat()
                        elif isinstance(ingested_at, str):
                            ingested_at_str = ingested_at
                        else:
                            ingested_at_str = datetime.now().isoformat()

                        batch_metadatas = [{"text": chunk.text, "doc_id": doc_id, "page_num": chunk.page_num,
                            "doc_type": metadata.get("doc_type", "unknown"), "language": metadata.get("language", "unknown"),
                            "file_path": metadata.get("file_path", str(file_path)), "ingested_at": ingested_at_str,
                            "chunk_index": chunk.metadata.get("index", batch_start + i)} for i, chunk in enumerate(batch_chunks)]

                        batch_embeddings = embeddings[batch_start:batch_end] if embeddings is not None else embed(batch_texts, show_progress=(batch_idx == start_batch))

                        try:
                            _add_vectors_chromadb_only(batch_chunk_ids, batch_embeddings, batch_metadatas)
                            try:
                                bm25_index.add_documents_atomic(batch_texts, batch_chunk_ids)
                            except Exception:
                                rollback_batch(batch_chunk_ids)
                                raise
                            update_checkpoint(file_path, batch_idx)
                        except Exception as e:
                            raise

                    delete_checkpoint(file_path)
                    stats['processed'] += 1
                    stats['total_chunks'] += item['num_chunks']

                except Exception as e:
                    stats['failed'] += 1
                    stats['errors'].append({'file': item.get('file', 'unknown'), 'error': str(e)})

        except QueueEmpty:
            continue
        except Exception:
            continue

    stats['writer_done'] = True

def _filter_files_by_checkpoint(files: List[Path], stats: Dict, force: bool = False) -> List[Path]:
    from src.storage.chroma_client import document_exists
    files_to_process = []
    skipped_completed, skipped_indexed, files_to_resume = 0, 0, 0

    for file_path in files:
        if force:
            files_to_process.append(file_path)
            continue
        checkpoint = load_checkpoint(file_path)
        if checkpoint:
            if not validate_checkpoint(checkpoint, file_path):
                cleanup_partial_data(checkpoint.doc_id)
                delete_checkpoint(file_path)
                files_to_process.append(file_path)
            elif not validate_databases(checkpoint, batch_size=100):
                cleanup_partial_data(checkpoint.doc_id)
                delete_checkpoint(file_path)
                files_to_process.append(file_path)
            else:
                total_batches = (checkpoint.total_chunks + 99) // 100
                completed_batches = len(checkpoint.processed_batches)
                if completed_batches >= total_batches:
                    delete_checkpoint(file_path)
                    skipped_completed += 1
                else:
                    files_to_process.append(file_path)
                    files_to_resume += 1
        else:
            if document_exists(file_path):
                skipped_indexed += 1
            else:
                files_to_process.append(file_path)

    stats['skipped'] = skipped_completed + skipped_indexed
    stats['skipped_indexed'] = skipped_indexed
    stats['skipped_completed'] = skipped_completed
    stats['files_to_resume'] = files_to_resume
    return files_to_process

def parallel_ingest_documents(source: Path, config: ParallelIngestionConfig = None, progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> ParallelIngestionResult:
    from src.config import settings
    if config is None:
        config = ParallelIngestionConfig()
    if not source.exists():
        raise ValueError(f"Directory not found: {source}")
    if not source.is_dir():
        raise ValueError(f"Not a directory: {source}")

    project_dir = settings.chroma_persist_dir.parent
    lock = IngestionLock(project_dir)
    if not lock.acquire():
        lock_info = lock.get_lock_info()
        if lock_info:
            raise IngestionLockError(f"Another ingestion process is running (PID {lock_info.get('pid')}). Delete {lock.lock_file} if stale.")
        else:
            raise IngestionLockError("Another ingestion process is running.")

    cleanup_done = False
    original_sigint = signal.getsignal(signal.SIGINT)
    original_sigterm = signal.getsignal(signal.SIGTERM)

    def cleanup_handler(signum, frame):
        nonlocal cleanup_done
        if cleanup_done:
            return
        cleanup_done = True
        lock.release()
        signal.signal(signal.SIGINT, original_sigint)
        signal.signal(signal.SIGTERM, original_sigterm)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, cleanup_handler)
    signal.signal(signal.SIGTERM, cleanup_handler)
    try:
        return _parallel_ingest_impl(source, config, progress_callback, lock)
    finally:
        signal.signal(signal.SIGINT, original_sigint)
        signal.signal(signal.SIGTERM, original_sigterm)
        lock.release()

def _parallel_ingest_impl(source: Path, config: ParallelIngestionConfig, progress_callback: Optional[Callable[[Dict[str, Any]], None]], lock: IngestionLock) -> ParallelIngestionResult:
    logger.info(f"Parallel ingestion: {config.num_workers} workers, {source}")
    files = collect_files(source, config.extensions, config.recursive)
    if not files:
        return ParallelIngestionResult(total_files=0, processed=0, failed=0, skipped=0, resumed=0, total_chunks=0, duration_seconds=0.0, errors=[])

    file_queue = mp.Queue()
    chunk_queue = mp.Queue()
    manager = mp.Manager()
    stats = manager.dict({'num_workers': config.num_workers, 'processed': 0, 'failed': 0, 'total_chunks': 0, 'resumed': 0,
        'resumed_chunks': 0, 'skipped': 0, 'files_to_resume': 0, 'writer_done': False, 'errors': manager.list()})

    files_to_process = _filter_files_by_checkpoint(files, stats, force=config.force) if config.resume else files
    if not files_to_process:
        return ParallelIngestionResult(total_files=len(files), processed=0, failed=0, skipped=stats.get('skipped', 0),
            resumed=0, total_chunks=0, duration_seconds=0.0, errors=[])

    for file_path in files_to_process:
        file_queue.put(file_path)
    for _ in range(config.num_workers):
        file_queue.put(None)

    writer = threading.Thread(target=_writer_thread_func, args=(chunk_queue, stats, config.batch_size), daemon=True)
    writer.start()

    workers = []
    for i in range(config.num_workers):
        p = mp.Process(target=_parse_document_worker, args=(i, file_queue, chunk_queue))
        p.start()
        workers.append(p)

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
            logger.info(f"Progress: {total_done}/{len(files_to_process)} | {rate:.1f} docs/sec | ETA: {eta/60:.1f} min")
            if progress_callback:
                progress_callback({'processed': current_processed, 'failed': current_failed, 'total': len(files_to_process),
                    'rate': rate, 'eta_minutes': eta / 60, 'resumed': stats.get('resumed', 0), 'skipped': stats.get('skipped', 0)})
            last_processed = total_done

    for p in workers:
        p.join()
    writer.join(timeout=30)

    elapsed = time.time() - start_time
    logger.info(f"Completed: {stats['processed']} processed, {stats['failed']} failed, {stats['total_chunks']} chunks in {elapsed/60:.1f} min")

    return ParallelIngestionResult(total_files=len(files), processed=stats['processed'], failed=stats['failed'],
        skipped=stats.get('skipped', 0), resumed=stats.get('resumed', 0), total_chunks=stats['total_chunks'],
        duration_seconds=elapsed, errors=list(stats.get('errors', [])))
