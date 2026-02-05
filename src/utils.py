import fnmatch
import gc
import logging
import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
from langdetect import LangDetectException, detect
from rich.console import Console

from .config import device, config, EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL, get_logger

_console = Console()

@contextmanager
def _quiet_logging():
    """Temporarily suppress all verbose logs for clean spinner display."""
    import warnings
    old_filter = warnings.filters[:]
    old_disable = logging.root.manager.disable
    logging.disable(logging.WARNING)  # Globally disable INFO and below
    warnings.filterwarnings("ignore")
    try:
        yield
    finally:
        logging.disable(old_disable)
        warnings.filters[:] = old_filter
from .models import EmbeddingError

logger = get_logger(__name__)
_embedder_cache: Optional[any] = None

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

    local_path = get_model_paths()["embedding"]
    if not local_path.exists():
        raise EmbeddingError(f"Embedding model not found at {local_path}. Run 'rag models --download' first.")

    logger.info(f"Loading embedding model from {local_path}")
    with _quiet_logging(), _console.status("Loading embedding model..."):
        model = SentenceTransformer(str(local_path), device=device, local_files_only=True, trust_remote_code=False)
    _embedder_cache = model
    return model

def cleanup_embedder() -> None:
    global _embedder_cache
    if _embedder_cache is not None:
        try:
            if hasattr(_embedder_cache, 'to'):
                _embedder_cache.to('cpu')
            del _embedder_cache
        except Exception:
            pass
        finally:
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
    try:
        texts = [texts] if isinstance(texts, str) else texts
        if not texts:
            raise EmbeddingError("No texts provided for embedding")
        embedder = get_embedder()
        return embedder.encode(texts, batch_size=EMBEDDING_BATCH_SIZE, show_progress_bar=show_progress, convert_to_numpy=True)
    except Exception as e:
        raise EmbeddingError(f"Failed to generate embeddings: {e}") from e

def get_model_paths() -> dict[str, Path]:
    models_dir = config("MODELS_DIR")
    embedding_name = EMBEDDING_MODEL.split("/")[-1]
    return {
        "embedding": models_dir / "embedding" / embedding_name,
        "docling_layout": Path(os.environ.get("HF_HOME", str(models_dir / ".cache"))) / "hub"
    }

def download_embedding_model() -> None:
    from sentence_transformers import SentenceTransformer
    path = get_model_paths()["embedding"]
    path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    model.save(str(path))

def download_docling_models() -> None:
    from huggingface_hub import snapshot_download
    logger.info("Downloading Docling layout model")
    snapshot_download(repo_id="docling-project/docling-layout-heron", revision="main")
    logger.info("Downloading Docling table structure model")
    snapshot_download(repo_id="docling-project/docling-models", revision="v2.3.0")

def verify_models_exist() -> dict[str, bool]:
    paths = get_model_paths()
    models_dir = config("MODELS_DIR")
    hf_cache = Path(os.environ.get("HF_HOME", str(models_dir / ".cache"))) / "hub"
    docling_layout_exists = hf_cache.exists() and len(list(hf_cache.glob("models--docling-project--docling-layout-heron"))) > 0
    docling_table_exists = hf_cache.exists() and len(list(hf_cache.glob("models--docling-project--docling-models"))) > 0
    return {"embedding": paths["embedding"].exists(), "docling_layout": docling_layout_exists, "docling_table": docling_table_exists}

SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.pptx', '.xlsx', '.html', '.htm', '.md', '.png', '.jpg', '.jpeg', '.tiff', '.tif', '.wav', '.mp3'}
EXCLUDE_PATTERNS = {'.*', '__*', '*.tmp', '*.temp', '~*', '*.bak', '*.backup', 'Thumbs.db', '.DS_Store'}

def is_supported_file(file_path: Path) -> bool:
    return file_path.suffix.lower() in SUPPORTED_EXTENSIONS

def should_exclude_file(file_path: Path) -> bool:
    return any(fnmatch.fnmatch(file_path.name, pattern) for pattern in EXCLUDE_PATTERNS)

def discover_files(root_path: Path, recursive: bool = True,
                   include_extensions: set[str] | None = None) -> Iterator[Path]:
    """Yield supported files one at a time (memory-efficient for large directories)."""
    extensions = include_extensions or SUPPORTED_EXTENSIONS
    pattern = "**/*" if recursive else "*"
    for f in root_path.glob(pattern):
        try:
            if f.is_file() and is_supported_file(f) and not should_exclude_file(f):
                yield f
        except (PermissionError, OSError):
            # Skip inaccessible files (permission denied, locked, broken symlinks)
            continue

def is_file_modified(file_path: Path, ingested_at_iso: str) -> bool:
    if not file_path.exists() or ingested_at_iso == 'unknown':
        return ingested_at_iso == 'unknown'
    try:
        ingested_at = datetime.fromisoformat(ingested_at_iso)
        file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
        return file_mtime > ingested_at
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
    from .ingestion.document import cleanup_converter
    cleanup_converter()
    from .ingestion.chunker import cleanup_chunker
    cleanup_chunker()
    from .storage.chroma_client import cleanup_chroma_client
    cleanup_chroma_client()
    from .storage.bm25_index import get_bm25_index
    get_bm25_index().close()
    gc.collect()
    _free_cuda_cache()
