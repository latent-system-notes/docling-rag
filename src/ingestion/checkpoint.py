"""Checkpoint system for resumable document ingestion.

This module provides checkpoint-based progress tracking to enable resuming
failed ingestion operations without starting from scratch.

Key features:
- Checkpoint creation and management
- File hash validation to detect modifications
- Database consistency validation (ChromaDB + BM25)
- Automatic cleanup of stale checkpoints
"""
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from ..config import settings, get_logger
from ..models import IngestionCheckpoint

logger = get_logger(__name__)


def _compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of file for change detection.

    Args:
        file_path: Path to file

    Returns:
        SHA256 hash as hex string
    """
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        # Read in 8KB chunks to handle large files
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_checkpoint_path(file_path: Path) -> Path:
    """Get checkpoint file path for a given document.

    Args:
        file_path: Path to document file

    Returns:
        Path to checkpoint JSON file
    """
    # Use MD5 of file path as checkpoint filename (same as doc_id)
    doc_id = hashlib.md5(str(file_path.absolute()).encode()).hexdigest()
    checkpoint_dir = settings.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir / f"{doc_id}.json"


def create_checkpoint(
    file_path: Path,
    doc_id: str,
    total_chunks: int,
    metadata: dict
) -> IngestionCheckpoint:
    """Create a new checkpoint for document ingestion.

    Args:
        file_path: Path to document being ingested
        doc_id: Document ID
        total_chunks: Total number of chunks in document
        metadata: Document metadata (doc_type, language, etc.)

    Returns:
        Created checkpoint object
    """
    file_hash = _compute_file_hash(file_path)

    checkpoint = IngestionCheckpoint(
        doc_id=doc_id,
        file_path=str(file_path.absolute()),
        file_hash=file_hash,
        total_chunks=total_chunks,
        processed_batches=[],
        last_batch=-1,
        timestamp=datetime.now(),
        metadata=metadata
    )

    checkpoint_path = get_checkpoint_path(file_path)
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint.model_dump(), f, indent=2, default=str)

    logger.info(f"Created checkpoint for {file_path.name} ({total_chunks} chunks)")
    return checkpoint


def load_checkpoint(file_path: Path) -> Optional[IngestionCheckpoint]:
    """Load existing checkpoint for a document.

    Args:
        file_path: Path to document

    Returns:
        Checkpoint object if exists, None otherwise
    """
    checkpoint_path = get_checkpoint_path(file_path)

    if not checkpoint_path.exists():
        return None

    try:
        with open(checkpoint_path, 'r') as f:
            data = json.load(f)

        checkpoint = IngestionCheckpoint(**data)
        logger.info(
            f"Loaded checkpoint for {file_path.name}: "
            f"{len(checkpoint.processed_batches)} batches completed"
        )
        return checkpoint

    except Exception as e:
        logger.error(f"Failed to load checkpoint from {checkpoint_path}: {e}")
        # Delete corrupted checkpoint
        try:
            checkpoint_path.unlink()
            logger.warning("Deleted corrupted checkpoint")
        except Exception:
            pass
        return None


def update_checkpoint(file_path: Path, batch_index: int) -> None:
    """Update checkpoint with completed batch.

    Args:
        file_path: Path to document
        batch_index: Index of completed batch
    """
    checkpoint = load_checkpoint(file_path)
    if not checkpoint:
        logger.warning(f"No checkpoint found to update for {file_path}")
        return

    # Add batch to processed list if not already there
    if batch_index not in checkpoint.processed_batches:
        checkpoint.processed_batches.append(batch_index)
        checkpoint.last_batch = max(checkpoint.processed_batches)
        checkpoint.timestamp = datetime.now()

        # Save updated checkpoint
        checkpoint_path = get_checkpoint_path(file_path)
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint.model_dump(), f, indent=2, default=str)

        logger.debug(f"Updated checkpoint: batch {batch_index} completed")


def delete_checkpoint(file_path: Path) -> None:
    """Delete checkpoint file.

    Args:
        file_path: Path to document
    """
    checkpoint_path = get_checkpoint_path(file_path)

    if checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info(f"Deleted checkpoint for {file_path.name}")


def validate_checkpoint(checkpoint: IngestionCheckpoint, file_path: Path) -> bool:
    """Validate checkpoint is still valid for current file.

    Checks:
    1. File still exists
    2. File hasn't been modified (hash match)

    Args:
        checkpoint: Checkpoint to validate
        file_path: Current file path

    Returns:
        True if checkpoint is valid, False otherwise
    """
    # Check file exists
    if not file_path.exists():
        logger.warning(f"File no longer exists: {file_path}")
        return False

    # Check file hash matches (detect modifications)
    current_hash = _compute_file_hash(file_path)
    if current_hash != checkpoint.file_hash:
        logger.warning(
            f"File has been modified since checkpoint "
            f"(hash mismatch: {current_hash[:8]} != {checkpoint.file_hash[:8]})"
        )
        return False

    logger.info("Checkpoint validation passed")
    return True


def validate_databases(checkpoint: IngestionCheckpoint, batch_size: int) -> bool:
    """Validate ChromaDB and BM25 are consistent with checkpoint.

    This critical check ensures both databases have exactly the chunks
    that were marked as completed in the checkpoint.

    Args:
        checkpoint: Checkpoint to validate against
        batch_size: Batch size used during ingestion

    Returns:
        True if databases are consistent, False otherwise
    """
    from ..storage.chroma_client import get_chroma_client
    from ..storage.bm25_index import get_bm25_index

    try:
        # Calculate expected number of chunks processed
        if not checkpoint.processed_batches:
            # No batches completed yet
            expected_chunks = 0
        else:
            # Each batch has batch_size chunks, except possibly the last one
            expected_chunks = len(checkpoint.processed_batches) * batch_size
            # Cap at total chunks (in case last batch was partial)
            expected_chunks = min(expected_chunks, checkpoint.total_chunks)

        # Get actual chunks from ChromaDB
        client = get_chroma_client()
        collection = client.get_collection(settings.chroma_collection_name)

        # Query for all chunks with this doc_id
        results = collection.get(where={"doc_id": checkpoint.doc_id})
        chromadb_chunk_count = len(results['ids']) if results['ids'] else 0

        if chromadb_chunk_count != expected_chunks:
            logger.error(
                f"ChromaDB chunk count mismatch: "
                f"Expected {expected_chunks}, found {chromadb_chunk_count}"
            )
            return False

        # Get actual chunks from BM25
        bm25 = get_bm25_index()

        # Load BM25 if not already loaded
        if not bm25._is_loaded and bm25.index_path.exists():
            bm25.load()

        # Count chunks in BM25 for this document
        # Need to check which chunk IDs from this doc exist in BM25
        bm25_chunk_ids = set(bm25.doc_ids) if bm25.doc_ids else set()
        doc_chunk_ids = set(results['ids']) if results['ids'] else set()

        # Check if all ChromaDB chunks are in BM25
        missing_in_bm25 = doc_chunk_ids - bm25_chunk_ids
        if missing_in_bm25:
            logger.error(
                f"BM25 missing {len(missing_in_bm25)} chunks that exist in ChromaDB"
            )
            return False

        logger.info(
            f"Database validation passed: "
            f"{expected_chunks} chunks in both ChromaDB and BM25"
        )
        return True

    except Exception as e:
        logger.error(f"Database validation failed: {e}")
        return False


def cleanup_partial_data(doc_id: str) -> None:
    """Remove partial/orphaned data from both databases.

    Called when validation fails to ensure clean state before restart.

    Args:
        doc_id: Document ID to cleanup
    """
    from ..storage.chroma_client import get_chroma_client
    from ..storage.bm25_index import get_bm25_index

    logger.warning(f"Cleaning up partial data for doc_id: {doc_id}")

    try:
        # Remove from ChromaDB
        client = get_chroma_client()
        collection = client.get_collection(settings.chroma_collection_name)

        # Get all chunk IDs for this document
        results = collection.get(where={"doc_id": doc_id})
        if results['ids']:
            chunk_ids = results['ids']
            collection.delete(where={"doc_id": doc_id})
            logger.info(f"Removed {len(chunk_ids)} orphaned chunks from ChromaDB")

            # Remove from BM25
            bm25 = get_bm25_index()
            bm25.remove_documents(chunk_ids)
            bm25.save()
            logger.info(f"Removed {len(chunk_ids)} orphaned chunks from BM25")

    except Exception as e:
        logger.error(f"Failed to cleanup partial data: {e}")


def list_checkpoints() -> list[dict]:
    """List all active checkpoints.

    Returns:
        List of checkpoint info dicts with:
        - file_path: Document path
        - doc_id: Document ID
        - total_chunks: Total chunks
        - processed_batches: Number of completed batches
        - timestamp: Last update time
    """
    checkpoint_dir = settings.checkpoint_dir

    if not checkpoint_dir.exists():
        return []

    checkpoints = []
    for checkpoint_file in checkpoint_dir.glob("*.json"):
        try:
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)

            checkpoint = IngestionCheckpoint(**data)
            checkpoints.append({
                'file_path': checkpoint.file_path,
                'doc_id': checkpoint.doc_id,
                'total_chunks': checkpoint.total_chunks,
                'processed_batches': len(checkpoint.processed_batches),
                'timestamp': checkpoint.timestamp.isoformat() if isinstance(checkpoint.timestamp, datetime) else checkpoint.timestamp,
            })
        except Exception as e:
            logger.warning(f"Failed to load checkpoint {checkpoint_file}: {e}")

    return checkpoints


def clean_all_checkpoints() -> int:
    """Delete all checkpoint files.

    Returns:
        Number of checkpoints deleted
    """
    checkpoint_dir = settings.checkpoint_dir

    if not checkpoint_dir.exists():
        return 0

    count = 0
    for checkpoint_file in checkpoint_dir.glob("*.json"):
        try:
            checkpoint_file.unlink()
            count += 1
        except Exception as e:
            logger.warning(f"Failed to delete checkpoint {checkpoint_file}: {e}")

    logger.info(f"Deleted {count} checkpoint files")
    return count


def clean_stale_checkpoints(days: int = None) -> int:
    """Delete checkpoints older than specified days.

    Args:
        days: Age threshold in days (defaults to config setting)

    Returns:
        Number of checkpoints deleted
    """
    days = days or settings.checkpoint_retention_days
    checkpoint_dir = settings.checkpoint_dir

    if not checkpoint_dir.exists():
        return 0

    cutoff_date = datetime.now() - timedelta(days=days)
    count = 0

    for checkpoint_file in checkpoint_dir.glob("*.json"):
        try:
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)

            checkpoint = IngestionCheckpoint(**data)

            # Convert timestamp to datetime if it's a string
            timestamp = checkpoint.timestamp
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)

            if timestamp < cutoff_date:
                checkpoint_file.unlink()
                count += 1
                logger.debug(f"Deleted stale checkpoint: {checkpoint_file.name}")

        except Exception as e:
            logger.warning(f"Failed to process checkpoint {checkpoint_file}: {e}")

    if count > 0:
        logger.info(f"Deleted {count} stale checkpoints (older than {days} days)")

    return count
