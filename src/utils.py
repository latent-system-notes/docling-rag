import fnmatch
import gc
import logging
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

import numpy as np
from langdetect import LangDetectException, detect
from rich.console import Console

from .config import device, config, EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL, get_logger
from .models import EmbeddingError

logger = get_logger(__name__)
_console = Console()
_embedder_cache = None

@contextmanager
def _quiet_logging():
    import warnings
    old_filter = warnings.filters[:]
    old_disable = logging.root.manager.disable
    logging.disable(logging.WARNING)
    warnings.filterwarnings("ignore")
    try:
        yield
    finally:
        logging.disable(old_disable)
        warnings.filters[:] = old_filter

def detect_language(text: str) -> str:
    if not text:
        return "unknown"
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"

def get_embedder():
    global _embedder_cache
    if _embedder_cache is not None:
        return _embedder_cache
    from sentence_transformers import SentenceTransformer
    local_path = get_embedding_model_path()
    if not local_path.exists():
        raise EmbeddingError(f"Embedding model not found at {local_path}. Run 'rag models --download' first.")
    logger.info(f"Loading embedding model from {local_path}")
    with _quiet_logging(), _console.status("Loading embedding model..."):
        _embedder_cache = SentenceTransformer(str(local_path), device=device, local_files_only=True, trust_remote_code=False)
    return _embedder_cache

def cleanup_embedder() -> None:
    global _embedder_cache
    _embedder_cache = None
    _free_cuda_cache()

def _free_cuda_cache() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

def embed(texts: str | list[str], show_progress: bool = False) -> np.ndarray:
    texts = [texts] if isinstance(texts, str) else texts
    if not texts:
        raise EmbeddingError("No texts provided for embedding")
    return get_embedder().encode(texts, batch_size=EMBEDDING_BATCH_SIZE, show_progress_bar=show_progress, convert_to_numpy=True)

def get_embedding_model_path() -> Path:
    return config("MODELS_DIR") / "embedding" / EMBEDDING_MODEL.split("/")[-1]

def download_embedding_model() -> None:
    from sentence_transformers import SentenceTransformer
    path = get_embedding_model_path()
    path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading embedding model: {EMBEDDING_MODEL}")
    SentenceTransformer(EMBEDDING_MODEL).save(str(path))

SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.pptx', '.xlsx', '.html', '.htm', '.md', '.png', '.jpg', '.jpeg', '.tiff', '.tif', '.wav', '.mp3'}
EXCLUDE_PATTERNS = {'.*', '__*', '*.tmp', '*.temp', '~*', '*.bak', '*.backup', 'Thumbs.db', '.DS_Store'}

def is_supported_file(file_path: Path) -> bool:
    return file_path.suffix.lower() in SUPPORTED_EXTENSIONS

def discover_files(root_path: Path, recursive: bool = True) -> Iterator[Path]:
    for f in root_path.glob("**/*" if recursive else "*"):
        try:
            if f.is_file() and is_supported_file(f) and not any(fnmatch.fnmatch(f.name, p) for p in EXCLUDE_PATTERNS):
                yield f
        except (PermissionError, OSError):
            continue

def is_file_modified(file_path: Path, ingested_at_iso: str) -> bool:
    if ingested_at_iso == 'unknown':
        return True
    try:
        return file_path.exists() and datetime.fromtimestamp(file_path.stat().st_mtime) > datetime.fromisoformat(ingested_at_iso)
    except Exception:
        return False

@contextmanager
def managed_resources():
    try:
        yield
    finally:
        cleanup_all_resources()

def cleanup_all_resources() -> None:
    cleanup_embedder()
    from .ingestion.chunker import cleanup_chunker
    cleanup_chunker()
    from .storage.chroma_client import cleanup_chroma_client
    cleanup_chroma_client()
    from .storage.bm25_index import get_bm25_index
    get_bm25_index().close()
    gc.collect()
    _free_cuda_cache()
