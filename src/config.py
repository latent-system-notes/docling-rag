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
    """Custom formatter with uppercase month abbreviation and optional session_id."""
    def formatTime(self, record, datefmt=None):
        ct = self.converter(record.created)
        if datefmt:
            s = time.strftime(datefmt, ct)
            return s.upper()
        else:
            return super().formatTime(record, datefmt)

    def format(self, record):
        # Add session_id to record if available
        if not hasattr(record, 'session_id'):
            record.session_id = _current_session_id
        return super().format(record)


# Global session ID for logging context
_current_session_id = ""


def set_session_id(session_id: str):
    """Set the current session ID for logging context."""
    global _current_session_id
    _current_session_id = session_id if session_id else ""


def get_session_id() -> str:
    """Get the current session ID."""
    return _current_session_id


def clear_session_id():
    """Clear the current session ID."""
    global _current_session_id
    _current_session_id = ""


# Configure root logger to apply format globally to all loggers
_root_logger = logging.getLogger()
_root_logger.setLevel(logging.INFO)

# Remove any existing handlers
for handler in _root_logger.handlers[:]:
    _root_logger.removeHandler(handler)

# Add our custom handler
_handler = logging.StreamHandler()
_formatter = UppercaseFormatter(
    fmt="[%(levelname)s] [%(asctime)s] [%(processName)s] [%(session_id)s] %(message)s",
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
# Device Configuration (CPU/GPU)
# ============================================================================

# Check device setting from environment BEFORE importing PyTorch
# RAG_DEVICE can be: cpu, cuda, mps (Apple Silicon), or auto
_device_setting = os.environ.get("RAG_DEVICE", "cpu").lower()

if _device_setting == "cpu":
    # Force CPU-only mode - disable all GPU backends
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
elif _device_setting == "auto":
    # Auto-detect: let PyTorch decide the best available device
    # Don't set any environment variables, PyTorch will auto-detect
    pass
elif _device_setting == "cuda":
    # NVIDIA GPU mode - ensure CUDA is not blocked
    if "CUDA_VISIBLE_DEVICES" in os.environ and os.environ["CUDA_VISIBLE_DEVICES"] == "":
        del os.environ["CUDA_VISIBLE_DEVICES"]
elif _device_setting == "mps":
    # Apple Silicon GPU mode
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# ============================================================================
# Force Offline Mode for HuggingFace
# ============================================================================

# Set cache directory to local models folder FIRST
# This ensures any caching operations use our local directory
if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = str(Path("./models/.cache").absolute())

# Check if offline mode should be enforced
# Offline mode is enabled by default for security and airgapped deployments
# Set RAG_OFFLINE_MODE=false or OFFLINE_MODE=false to disable
_offline_mode = os.environ.get("RAG_OFFLINE_MODE", os.environ.get("OFFLINE_MODE", "true")).lower()
if _offline_mode in ("true", "1", "yes"):
    # Enforce offline mode - prevent any network access for model downloads
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

# OCR engine is configured via settings.ocr_engine (defaults to "auto")

# Import torch after setting environment variables
try:
    import torch

    if _device_setting == "cpu":
        # Explicitly set default device to CPU
        torch.set_default_device("cpu")
        # Disable cuDNN if available
        if hasattr(torch.backends, 'cudnn'):
            torch.backends.cudnn.enabled = False
    elif _device_setting == "cuda":
        if torch.cuda.is_available():
            torch.set_default_device("cuda")
        else:
            warnings.warn("CUDA requested but not available, falling back to CPU")
            torch.set_default_device("cpu")
    elif _device_setting == "mps":
        if torch.backends.mps.is_available():
            torch.set_default_device("mps")
        else:
            warnings.warn("MPS requested but not available, falling back to CPU")
            torch.set_default_device("cpu")
    elif _device_setting == "auto":
        # Auto-detect best device
        if torch.cuda.is_available():
            torch.set_default_device("cuda")
        elif torch.backends.mps.is_available():
            torch.set_default_device("mps")
        else:
            torch.set_default_device("cpu")
except ImportError:
    # torch might not be imported yet, will be set when it loads
    pass


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="RAG_", case_sensitive=False)

    models_dir: Path = Path("./models")

    chroma_persist_dir: Path = Path("./data/chroma")
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

    # Offline mode - prevents any network access for model downloads
    # Set to False only when running 'rag models --download'
    offline_mode: bool = True

    default_top_k: int = 5

    # MCP Server Configuration
    mcp_server_name: str = "docling-rag"
    mcp_transport: str = "streamable-http"
    mcp_host: str = "0.0.0.0"
    mcp_port: int = 8080
    mcp_instructions: str = "You are a RAG (Retrieval-Augmented Generation) assistant with access to indexed documents. Use query_rag to retrieve relevant context and list_all_documents to browse available documents."
    mcp_enable_cleanup: bool = True

    # MCP Monitoring Configuration
    mcp_metrics_enabled: bool = True
    mcp_metrics_retention_days: int = 7

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
# Project Integration
# ============================================================================


def get_active_project_settings() -> dict | None:
    """Get settings from the active project if one exists.

    Returns:
        Dict with project settings or None if no active project
    """
    try:
        from .project import get_project_manager
        pm = get_project_manager()
        project = pm.get_active_project()
        if project:
            paths = pm.get_project_paths(project.name)
            return {
                # Basic
                "name": project.name,
                "port": project.port,
                "device": project.device,
                "chroma_persist_dir": paths["chroma_path"],
                "db_path": paths["db_path"],
                "docs_path": paths["docs_path"],
                # Document Processing
                "enable_ocr": project.enable_ocr,
                "ocr_engine": project.ocr_engine,
                "ocr_languages": project.ocr_languages,
                "enable_asr": project.enable_asr,
                # Embedding & Chunking
                "embedding_model": project.embedding_model,
                "chunking_method": project.chunking_method,
                "max_tokens": project.max_tokens,
                # Retrieval
                "default_top_k": project.default_top_k,
                # MCP Server
                "mcp_server_name": project.mcp_server_name,
                "mcp_transport": project.mcp_transport,
                "mcp_host": project.mcp_host,
                "mcp_enable_cleanup": project.mcp_enable_cleanup,
                # Logging
                "log_level": project.log_level,
            }
    except Exception:
        pass
    return None


def apply_project_settings() -> bool:
    """Apply active project settings to the global settings.

    Returns:
        True if project settings were applied, False otherwise
    """
    global settings

    project = get_active_project_settings()
    if not project:
        return False

    # Override ALL settings with project values
    # Paths
    settings.chroma_persist_dir = project["chroma_persist_dir"]

    # Document Processing
    settings.enable_ocr = project["enable_ocr"]
    settings.ocr_engine = project["ocr_engine"]
    settings.ocr_languages = project["ocr_languages"]
    settings.enable_asr = project["enable_asr"]

    # Embedding & Chunking
    settings.embedding_model = project["embedding_model"]
    settings.chunking_method = project["chunking_method"]
    settings.max_tokens = project["max_tokens"]

    # Retrieval
    settings.default_top_k = project["default_top_k"]

    # MCP Server
    settings.mcp_port = project["port"]
    settings.mcp_server_name = project["mcp_server_name"]
    settings.mcp_transport = project["mcp_transport"]
    settings.mcp_host = project["mcp_host"]
    settings.mcp_enable_cleanup = project["mcp_enable_cleanup"]

    # Device
    settings.device = project["device"]

    # Logging
    settings.log_level = project["log_level"]

    return True


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
