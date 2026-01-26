import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from ..config import settings, get_logger
from ..models import IngestionCheckpoint

logger = get_logger(__name__)

def _compute_file_hash(file_path: Path) -> str:
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()

def get_checkpoint_path(file_path: Path) -> Path:
    doc_id = hashlib.md5(str(file_path.absolute()).encode()).hexdigest()
    checkpoint_dir = settings.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir / f"{doc_id}.json"

def create_checkpoint(file_path: Path, doc_id: str, total_chunks: int, metadata: dict) -> IngestionCheckpoint:
    file_hash = _compute_file_hash(file_path)
    checkpoint = IngestionCheckpoint(doc_id=doc_id, file_path=str(file_path.absolute()), file_hash=file_hash,
        total_chunks=total_chunks, processed_batches=[], last_batch=-1, timestamp=datetime.now(), metadata=metadata)
    checkpoint_path = get_checkpoint_path(file_path)
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint.model_dump(), f, indent=2, default=str)
    return checkpoint

def load_checkpoint(file_path: Path) -> Optional[IngestionCheckpoint]:
    checkpoint_path = get_checkpoint_path(file_path)
    if not checkpoint_path.exists():
        return None
    try:
        with open(checkpoint_path, 'r') as f:
            data = json.load(f)
        return IngestionCheckpoint(**data)
    except Exception:
        try:
            checkpoint_path.unlink()
        except Exception:
            pass
        return None

def update_checkpoint(file_path: Path, batch_index: int) -> None:
    checkpoint = load_checkpoint(file_path)
    if not checkpoint:
        return
    if batch_index not in checkpoint.processed_batches:
        checkpoint.processed_batches.append(batch_index)
        checkpoint.last_batch = max(checkpoint.processed_batches)
        checkpoint.timestamp = datetime.now()
        checkpoint_path = get_checkpoint_path(file_path)
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint.model_dump(), f, indent=2, default=str)

def delete_checkpoint(file_path: Path) -> None:
    checkpoint_path = get_checkpoint_path(file_path)
    if checkpoint_path.exists():
        checkpoint_path.unlink()

def validate_checkpoint(checkpoint: IngestionCheckpoint, file_path: Path) -> bool:
    if not file_path.exists():
        return False
    current_hash = _compute_file_hash(file_path)
    return current_hash == checkpoint.file_hash

def validate_databases(checkpoint: IngestionCheckpoint, batch_size: int) -> bool:
    from ..storage.chroma_client import get_chroma_client
    from ..storage.bm25_index import get_bm25_index
    try:
        expected_chunks = min(len(checkpoint.processed_batches) * batch_size, checkpoint.total_chunks) if checkpoint.processed_batches else 0
        client = get_chroma_client()
        collection = client.get_collection(settings.chroma_collection_name)
        results = collection.get(where={"doc_id": checkpoint.doc_id})
        chromadb_chunk_count = len(results['ids']) if results['ids'] else 0
        if chromadb_chunk_count != expected_chunks:
            return False
        bm25 = get_bm25_index()
        if not bm25._is_loaded:
            bm25.load()
        return bm25.num_docs > 0 or chromadb_chunk_count == 0
    except Exception:
        return False

def cleanup_partial_data(doc_id: str) -> None:
    from ..storage.chroma_client import get_chroma_client
    from ..storage.bm25_index import get_bm25_index
    try:
        client = get_chroma_client()
        collection = client.get_collection(settings.chroma_collection_name)
        results = collection.get(where={"doc_id": doc_id})
        if results['ids']:
            chunk_ids = results['ids']
            collection.delete(where={"doc_id": doc_id})
            bm25 = get_bm25_index()
            bm25.remove_documents(chunk_ids)
            bm25.save()
    except Exception:
        pass

def list_checkpoints() -> list[dict]:
    checkpoint_dir = settings.checkpoint_dir
    if not checkpoint_dir.exists():
        return []
    checkpoints = []
    for checkpoint_file in checkpoint_dir.glob("*.json"):
        try:
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
            checkpoint = IngestionCheckpoint(**data)
            checkpoints.append({'file_path': checkpoint.file_path, 'doc_id': checkpoint.doc_id,
                'total_chunks': checkpoint.total_chunks, 'processed_batches': len(checkpoint.processed_batches),
                'timestamp': checkpoint.timestamp.isoformat() if isinstance(checkpoint.timestamp, datetime) else checkpoint.timestamp})
        except Exception:
            pass
    return checkpoints

def clean_all_checkpoints() -> int:
    checkpoint_dir = settings.checkpoint_dir
    if not checkpoint_dir.exists():
        return 0
    count = 0
    for checkpoint_file in checkpoint_dir.glob("*.json"):
        try:
            checkpoint_file.unlink()
            count += 1
        except Exception:
            pass
    return count

def clean_stale_checkpoints(days: int = None) -> int:
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
            timestamp = checkpoint.timestamp
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
            if timestamp < cutoff_date:
                checkpoint_file.unlink()
                count += 1
        except Exception:
            pass
    return count
