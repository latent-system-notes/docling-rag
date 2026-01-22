import logging
import os
import warnings
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from .models import ChunkingMethod, Device

# Suppress tokenizer and model warnings
warnings.filterwarnings("ignore", message=".*Token indices sequence length.*")
warnings.filterwarnings("ignore", message=".*incorrect regex pattern.*")
warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================================
# Global Logging Configuration
# ============================================================================

import time


class UppercaseFormatter(logging.Formatter):
    """Custom formatter with uppercase month abbreviation."""
    def formatTime(self, record, datefmt=None):
        ct = self.converter(record.created)
        if datefmt:
            s = time.strftime(datefmt, ct)
            return s.upper()
        else:
            return super().formatTime(record, datefmt)


# Configure root logger to apply format globally to all loggers
_root_logger = logging.getLogger()
_root_logger.setLevel(logging.INFO)

# Remove any existing handlers
for handler in _root_logger.handlers[:]:
    _root_logger.removeHandler(handler)

# Add our custom handler
_handler = logging.StreamHandler()
_formatter = UppercaseFormatter(
    fmt="[%(levelname)s] [%(asctime)s] %(message)s",
    datefmt="%d-%b %H:%M:%S"
)
_handler.setFormatter(_formatter)
_root_logger.addHandler(_handler)

# Configure third-party loggers - remove their handlers to use our global format
for lib_name in ["RapidOCR", "transformers", "sentence_transformers", "chromadb", "httpx"]:
    lib_logger = logging.getLogger(lib_name)
    # Remove any existing handlers
    for handler in lib_logger.handlers[:]:
        lib_logger.removeHandler(handler)
    # Set to use root logger handler (propagate=True)
    lib_logger.propagate = True

# Set log levels for third-party loggers
logging.getLogger("RapidOCR").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


def enforce_logging_format():
    """Enforce our logging format on third-party libraries that add their own handlers."""
    rapidocr_logger = logging.getLogger("RapidOCR")
    # Remove any handlers they added
    for handler in rapidocr_logger.handlers[:]:
        rapidocr_logger.removeHandler(handler)
    # Ensure they use root logger
    rapidocr_logger.propagate = True
    # Set log level to WARNING (suppress INFO logs)
    rapidocr_logger.setLevel(logging.WARNING)

# ============================================================================
# Force CPU-only Mode for PyTorch
# ============================================================================

# Set environment variable to disable CUDA before any PyTorch import
# This ensures all PyTorch operations use CPU only, even if GPU is available
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"  # Disable MPS (Apple Silicon GPU)

# ============================================================================
# Force Offline Mode for HuggingFace
# ============================================================================

# NOTE: Offline mode is NOT set here globally because it would prevent downloads.
# Instead, we enforce offline mode via `local_files_only=True` parameter
# when loading models in utils.py and other modules.

# Set cache directory to local models folder
# This ensures any caching operations use our local directory
if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = str(Path("./models/.cache").absolute())

# OCR engine is configured via settings.ocr_engine (defaults to "auto")

# Import torch after setting environment variables
try:
    import torch
    # Explicitly set default device to CPU
    torch.set_default_device("cpu")
    # Disable cuDNN if available
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.enabled = False
except ImportError:
    # torch might not be imported yet, will be set when it loads
    pass


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="RAG_", case_sensitive=False)

    models_dir: Path = Path("./models")

    chroma_persist_dir: str = "./data/chroma"
    chroma_collection_name: str = "documents"

    # Checkpoint directory for resumable ingestion
    checkpoint_dir: Path = Path("./data/checkpoints")
    checkpoint_retention_days: int = 7

    # ChromaDB connection mode
    chroma_mode: str = "persistent"  # "persistent" or "http"

    # ChromaDB HTTP server settings (only used when mode="http")
    chroma_server_host: str = "localhost"
    chroma_server_port: int = 8000
    chroma_server_ssl: bool = False
    chroma_server_api_key: str = ""  # Optional: for authentication

    embedding_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    embedding_batch_size: int = 32

    chunking_method: ChunkingMethod = "hybrid"
    max_tokens: int = 512

    enable_ocr: bool = True
    enable_asr: bool = True
    ocr_languages: str = "eng+ara"
    ocr_engine: str = "auto"  # auto, rapidocr, easyocr, tesseract, ocrmac

    default_top_k: int = 5

    # MCP Server Configuration
    mcp_server_name: str = "docling-rag"
    mcp_transport: str = "streamable-http"
    mcp_host: str = "0.0.0.0"
    mcp_port: int = 8080
    mcp_instructions: str = "You are a RAG (Retrieval-Augmented Generation) assistant with access to indexed documents. Use query_rag to retrieve relevant context and list_all_documents to browse available documents."
    mcp_enable_cleanup: bool = True

    # MCP Tool Descriptions
    mcp_tool_query_description: str = """Query the RAG system to retrieve relevant document chunks based on semantic similarity.

Args:
    query_text (str): The search query or question to find relevant context for
    top_k (int, optional): Number of top results to return. Defaults to 5.

Returns:
    QueryResult containing:
    - query: The original query text
    - context: List of SearchResult objects with:
        - chunk: Document chunk (text, page_num, doc_id, metadata)
        - score: Relevance score (0-1, higher is better)
        - distance: Vector distance (lower is better)

Use this to retrieve context chunks that can be used to answer questions or provide relevant information from indexed documents.

Examples:
    query_rag("What is machine learning?", top_k=3)
    query_rag("Safety procedures for evacuation", top_k=10)"""

    mcp_tool_list_docs_description: str = """List all indexed documents with metadata and pagination support.

Returns document information including file paths, types, languages, chunk counts, and ingestion timestamps.

Args:
    limit (int, optional): Maximum number of documents to return. None returns all.
    offset (int, optional): Number of documents to skip for pagination. Defaults to 0.

Returns:
    Dictionary containing:
    - total: Total number of documents in the system
    - showing: Number of documents in this response
    - offset: Current pagination offset
    - documents: List of document objects with:
        - doc_id: Unique document identifier
        - file_path: Path to original file
        - doc_type: File extension (pdf, docx, etc.)
        - language: Detected language
        - num_chunks: Number of chunks created
        - ingested_at: Timestamp when ingested

Examples:
    list_all_documents()              # Get all documents
    list_all_documents(limit=20)      # Get first 20
    list_all_documents(limit=20, offset=20)  # Get next 20 (page 2)"""

    language_detection_enabled: bool = True

    device: Device = "cpu"
    log_level: str = "INFO"


settings = Settings()


# ============================================================================
# Logging
# ============================================================================


def get_logger(name: str) -> logging.Logger:
    """Get or create a logger with the configured log level.

    Logging is configured globally, so this just returns a logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(settings.log_level)
    return logger
