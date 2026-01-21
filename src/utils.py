"""Utility functions for language detection, embeddings, and model downloading."""
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import numpy as np
from langdetect import LangDetectException, detect

from .config import settings, get_logger
from .models import EmbeddingError

logger = get_logger(__name__)

# Manual cache for embedding model to allow cleanup
_embedder_cache: Optional[any] = None


# ============================================================================
# Language Detection
# ============================================================================


def detect_language(text: str) -> str:
    """Detect the language of the given text.

    Args:
        text: Input text to detect language from

    Returns:
        ISO 639-1 language code (e.g., 'en', 'fr') or 'unknown'
    """
    if not settings.language_detection_enabled or not text:
        return "unknown"

    try:
        lang = detect(text)
        return lang
    except LangDetectException:
        return "unknown"


# ============================================================================
# Text Embeddings
# ============================================================================


def get_embedder():
    """Load the embedding model (cached for reuse).

    Embeddings convert text into numbers (vectors) that capture semantic meaning.
    Similar texts have similar vectors, enabling semantic search.

    Uses manual caching to allow proper resource cleanup.

    Returns:
        Loaded SentenceTransformer model
    """
    global _embedder_cache

    # Return cached model if available
    if _embedder_cache is not None:
        return _embedder_cache

    from sentence_transformers import SentenceTransformer

    local_path = get_model_paths()["embedding"]

    if not local_path.exists():
        raise EmbeddingError(
            f"Embedding model not found at {local_path}. "
            "Run 'rag models --download' first."
        )

    logger.info(f"Loading embedding model from {local_path}")
    model = SentenceTransformer(str(local_path), device=settings.device)
    _embedder_cache = model
    return model


def cleanup_embedder() -> None:
    """Cleanup and release the cached embedding model.

    Call this to free ~420MB+ of memory when the model is no longer needed.
    Useful for batch processing or after ingesting large document sets.
    """
    global _embedder_cache

    if _embedder_cache is not None:
        # Clear model from memory
        try:
            # Move model tensors to CPU if on GPU (though we use CPU-only mode)
            if hasattr(_embedder_cache, 'to'):
                _embedder_cache.to('cpu')
            # Delete the model
            del _embedder_cache
        except Exception as e:
            logger.warning(f"Error during embedder cleanup: {e}")
        finally:
            _embedder_cache = None
            logger.info("Embedding model cache cleared (freed ~420MB)")


def embed(texts: str | list[str], show_progress: bool = False) -> np.ndarray:
    """Convert text(s) into numerical vectors (embeddings).

    This is the core of semantic search - we convert both documents and queries
    to vectors, then find documents with similar vectors to the query.

    Args:
        texts: Single text or list of texts to embed
        show_progress: Show progress bar for batch processing

    Returns:
        NumPy array of embeddings (shape: [num_texts, embedding_dim])
    """
    try:
        # Normalize input to list
        texts = [texts] if isinstance(texts, str) else texts

        if not texts:
            raise EmbeddingError("No texts provided for embedding")

        embedder = get_embedder()

        # Generate embeddings with optional progress bar
        embeddings = embedder.encode(
            texts,
            batch_size=settings.embedding_batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

        return embeddings

    except Exception as e:
        raise EmbeddingError(f"Failed to generate embeddings: {e}") from e


# ============================================================================
# Model Downloading
# ============================================================================


def get_model_paths() -> dict[str, Path]:
    """Return local path for embedding model."""
    embedding_name = settings.embedding_model.split("/")[-1]
    return {"embedding": settings.models_dir / "embedding" / embedding_name}


def download_embedding_model() -> None:
    """Download embedding model to local directory"""
    from sentence_transformers import SentenceTransformer

    path = get_model_paths()["embedding"]
    path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading embedding model: {settings.embedding_model}")
    model = SentenceTransformer(settings.embedding_model)
    model.save(str(path))
    logger.info(f"Downloaded embedding model to {path}")


def verify_models_exist() -> dict[str, bool]:
    """Check if embedding model exists locally"""
    return {"embedding": get_model_paths()["embedding"].exists()}


# ============================================================================
# File Discovery and Filtering
# ============================================================================


SUPPORTED_EXTENSIONS = {
    # Documents
    '.pdf', '.docx', '.pptx', '.xlsx',
    # Web
    '.html', '.htm',
    # Markup
    '.md',
    # Images (with OCR)
    '.png', '.jpg', '.jpeg', '.tiff', '.tif',
    # Audio (with ASR)
    '.wav', '.mp3'
}

EXCLUDE_PATTERNS = {
    '.*',           # Hidden files
    '__*',          # Python special
    '*.tmp',        # Temp files
    '*.temp',
    '~*',           # Office temp
    '*.bak',        # Backups
    '*.backup',
    'Thumbs.db',    # System
    '.DS_Store'
}


def is_supported_file(file_path: Path) -> bool:
    """Check if file extension is supported for ingestion.

    Args:
        file_path: Path to the file to check

    Returns:
        True if file extension is in supported list
    """
    return file_path.suffix.lower() in SUPPORTED_EXTENSIONS


def should_exclude_file(file_path: Path) -> bool:
    """Check if file matches exclusion patterns.

    Args:
        file_path: Path to the file to check

    Returns:
        True if file matches any exclusion pattern
    """
    import fnmatch
    name = file_path.name
    return any(fnmatch.fnmatch(name, pattern) for pattern in EXCLUDE_PATTERNS)


def discover_files(
    root_path: Path,
    recursive: bool = True,
    include_extensions: set[str] | None = None,
    limit: int | None = None
) -> list[Path]:
    """Discover files in directory with filtering.

    Scans a directory and returns a list of valid files to ingest,
    automatically filtering by supported extensions and exclusion patterns.

    Args:
        root_path: Directory to scan
        recursive: Scan subdirectories (default: True)
        include_extensions: Override default supported extensions
        limit: Maximum number of files to return (None for all)

    Returns:
        List of valid file paths to ingest, sorted by path

    Examples:
        >>> files = discover_files(Path("./documents"))
        >>> files = discover_files(Path("./docs"), recursive=False)
        >>> files = discover_files(Path("./data"), include_extensions={'.pdf', '.md'})
        >>> files = discover_files(Path("./huge_dir"), limit=100)
    """
    extensions = include_extensions or SUPPORTED_EXTENSIONS

    # Recursive or flat scan
    pattern = "**/*" if recursive else "*"
    all_files = root_path.glob(pattern)

    # Filter: files only, supported types, not excluded
    valid_files = [
        f for f in all_files
        if f.is_file()
        and is_supported_file(f)
        and not should_exclude_file(f)
    ]

    # Sort for consistent ordering
    valid_files = sorted(valid_files)

    # Apply limit if specified
    if limit is not None and limit > 0:
        valid_files = valid_files[:limit]

    return valid_files


# ============================================================================
# File Modification Detection
# ============================================================================


def is_file_modified(file_path: Path, ingested_at_iso: str) -> bool:
    """Check if file was modified after ingestion.

    Compares file's last modification time with ingestion timestamp.

    Args:
        file_path: Path to the file
        ingested_at_iso: ISO format timestamp from ChromaDB

    Returns:
        True if file was modified after ingestion
    """
    from datetime import datetime

    if not file_path.exists():
        return False

    if ingested_at_iso == 'unknown':
        return True  # Assume modified if timestamp missing

    try:
        # Parse ISO timestamp
        ingested_at = datetime.fromisoformat(ingested_at_iso)

        # Get file's last modification time
        file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)

        # File is modified if mtime is newer than ingested_at
        return file_mtime > ingested_at

    except Exception as e:
        logger.warning(f"Error checking modification time for {file_path}: {e}")
        return False


# ============================================================================
# Global Cleanup
# ============================================================================


@contextmanager
def managed_resources():
    """Context manager for automatic resource cleanup.

    Automatically cleans up cached resources when exiting the context.
    Useful for batch operations or CLI commands.

    Example:
        >>> with managed_resources():
        ...     ingest_document("paper.pdf")
        ...     query("What is this about?")
        ... # Resources automatically cleaned up here
    """
    try:
        yield
    finally:
        cleanup_all_resources()


def cleanup_all_resources() -> None:
    """Cleanup all cached resources (embedding model, ChromaDB client, BM25 index).

    Use this to free memory after batch operations or in long-running processes.
    Frees approximately 420MB+ from embedding model plus database connections.
    """
    import gc

    # Clean up embedding model
    cleanup_embedder()

    # Clean up ChromaDB client
    from .storage.chroma_client import cleanup_chroma_client
    cleanup_chroma_client()

    # Clear BM25 index from memory (but don't delete the saved file)
    from .storage.bm25_index import get_bm25_index
    bm25_index = get_bm25_index()
    if bm25_index.index is not None:
        bm25_index.index = None
        bm25_index.documents = []
        bm25_index.doc_ids = []
        logger.info("BM25 index cleared from memory")

    # Force garbage collection to free memory
    gc.collect()
    logger.info("All resources cleaned up and garbage collected")
