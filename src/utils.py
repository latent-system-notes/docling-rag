"""Utility functions for language detection, embeddings, and model downloading."""
from functools import cache
from pathlib import Path

import numpy as np
from langdetect import LangDetectException, detect
from rich.progress import Progress

from .config import settings, get_logger
from .models import EmbeddingError

logger = get_logger(__name__)


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


@cache
def get_embedder():
    """Load the embedding model (cached for reuse).

    Embeddings convert text into numbers (vectors) that capture semantic meaning.
    Similar texts have similar vectors, enabling semantic search.

    Returns:
        Loaded SentenceTransformer model
    """
    from sentence_transformers import SentenceTransformer

    local_path = get_model_paths()["embedding"]

    if not local_path.exists():
        raise EmbeddingError(
            f"Embedding model not found at {local_path}. "
            "Run 'rag models --download' first."
        )

    logger.info(f"Loading embedding model from {local_path}")
    return SentenceTransformer(str(local_path), device=settings.device)


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
        if show_progress and len(texts) > 1:
            embeddings = embedder.encode(
                texts,
                batch_size=settings.embedding_batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
            )
        else:
            embeddings = embedder.encode(
                texts,
                batch_size=settings.embedding_batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

        return embeddings

    except Exception as e:
        raise EmbeddingError(f"Failed to generate embeddings: {e}") from e


def embed_batch(texts: list[str], desc: str = "Embedding") -> np.ndarray:
    """Embed multiple texts with progress bar (convenience wrapper)."""
    return embed(texts, show_progress=True)


# ============================================================================
# Model Downloading
# ============================================================================


def get_model_paths() -> dict[str, Path]:
    """Return local paths for all models based on settings.

    Models are stored as: ./models/{type}/{model-name}/
    We extract the model name from HuggingFace IDs like "org/model-name"
    """
    base = settings.models_dir

    # Extract model names from "organization/model-name" format
    embedding_name = settings.embedding_model.split("/")[-1]
    granite_name = settings.granite_model_id.split("/")[-1]

    return {
        "embedding": base / "embedding" / embedding_name,
        "granite_tokenizer": base / "granite" / granite_name / "tokenizer",
        "granite_model": base / "granite" / granite_name / "model",
    }


def download_embedding_model() -> None:
    """Download embedding model to local directory"""
    from sentence_transformers import SentenceTransformer

    path = get_model_paths()["embedding"]
    path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading embedding model: {settings.embedding_model}")
    model = SentenceTransformer(settings.embedding_model)
    model.save(str(path))
    logger.info(f"Downloaded embedding model to {path}")


def download_granite_model() -> None:
    """Download Granite model and tokenizer to local directory"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok_path = get_model_paths()["granite_tokenizer"]
    model_path = get_model_paths()["granite_model"]

    tok_path.mkdir(parents=True, exist_ok=True)
    model_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading Granite tokenizer: {settings.granite_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(settings.granite_model_id)
    tokenizer.save_pretrained(str(tok_path))
    logger.info(f"Downloaded Granite tokenizer to {tok_path}")

    logger.info(f"Downloading Granite model: {settings.granite_model_id}")
    model = AutoModelForCausalLM.from_pretrained(settings.granite_model_id)
    model.save_pretrained(str(model_path))
    logger.info(f"Downloaded Granite model to {model_path}")


def download_all_models() -> None:
    """Download all models with progress tracking"""
    with Progress() as progress:
        task = progress.add_task("Downloading models", total=2)

        progress.update(task, description="Downloading embedding model...")
        download_embedding_model()
        progress.advance(task)

        progress.update(task, description="Downloading Granite model...")
        download_granite_model()
        progress.advance(task)

    logger.info("All models downloaded successfully")


def verify_models_exist() -> dict[str, bool]:
    """Check if all models exist locally"""
    paths = get_model_paths()
    return {
        "embedding": paths["embedding"].exists(),
        "granite": paths["granite_model"].exists(),
    }


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
    include_extensions: set[str] | None = None
) -> list[Path]:
    """Discover files in directory with filtering.

    Scans a directory and returns a list of valid files to ingest,
    automatically filtering by supported extensions and exclusion patterns.

    Args:
        root_path: Directory to scan
        recursive: Scan subdirectories (default: True)
        include_extensions: Override default supported extensions

    Returns:
        List of valid file paths to ingest, sorted by path

    Examples:
        >>> files = discover_files(Path("./documents"))
        >>> files = discover_files(Path("./docs"), recursive=False)
        >>> files = discover_files(Path("./data"), include_extensions={'.pdf', '.md'})
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

    return sorted(valid_files)  # Sort for consistent ordering


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
