"""Configuration module for the docling-rag system.

Settings are simplified to:
- Environment variables (RAG_ prefix) for infrastructure settings
- Hardcoded constants for document processing defaults
- Project config for project-specific paths and port
"""
import logging
import os
import time
import warnings
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Suppress tokenizer and model warnings
warnings.filterwarnings("ignore", message=".*Token indices sequence length.*")
warnings.filterwarnings("ignore", message=".*incorrect regex pattern.*")
warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================================
# Global Logging Configuration
# ============================================================================


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


# ============================================================================
# Settings Class
# ============================================================================


# ============================================================================
# Hardcoded Defaults (not configurable - these are module-level constants)
# ============================================================================

# Document Processing
ENABLE_OCR: bool = False  # OCR disabled - not needed
OCR_ENGINE: str = "auto"
OCR_LANGUAGES: str = "eng+ara"
ENABLE_ASR: bool = True

# Embedding & Chunking
EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
EMBEDDING_BATCH_SIZE: int = 32
CHUNKING_METHOD: str = "hybrid"
MAX_TOKENS: int = 512

# Retrieval
DEFAULT_TOP_K: int = 5

# MCP
MCP_TRANSPORT: str = "streamable-http"
MCP_ENABLE_CLEANUP: bool = True
MCP_HOST: str = "0.0.0.0"
MCP_METRICS_ENABLED: bool = True
MCP_METRICS_RETENTION_DAYS: int = 7

# ChromaDB - always persistent mode
CHROMA_MODE: str = "persistent"
COLLECTION_NAME: str = "documents"

# Infrastructure
OFFLINE_MODE: bool = True  # Always offline except during model download
LANGUAGE_DETECTION_ENABLED: bool = True
MODELS_DIR: Path = Path("./models")
CHECKPOINT_RETENTION_DAYS: int = 7

# Log level
LOG_LEVEL: str = "INFO"


class Settings(BaseSettings):
    """Global settings.

    Only ONE setting is configurable via environment variable:
    - RAG_DEVICE: cpu, cuda, mps, auto

    Everything else is hardcoded as module-level constants above.
    Project-specific settings (name, port, data_dir, docs_dir) come from ProjectConfig.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="RAG_",
        case_sensitive=False,
        extra="ignore"
    )

    # === Only 1 configurable setting from environment ===
    device: str = "cpu"  # cpu, cuda, mps, auto

    # MCP Tool Descriptions
    mcp_instructions: str = "You are a RAG assistant with access to indexed documents. Use query_rag to retrieve relevant context and list_all_documents to browse available documents."

    mcp_tool_query_description: str = """Query the RAG system to retrieve relevant document chunks.

Args:
    query_text: Search query or question
    top_k: Number of results (default: 5)

Returns relevant chunks with scores."""

    mcp_tool_list_docs_description: str = """List all indexed documents with metadata.

Args:
    limit: Max documents to return
    offset: Pagination offset

Returns document list with file paths, types, and chunk counts."""

    # === Hardcoded path properties ===
    @property
    def models_dir(self) -> Path:
        return MODELS_DIR

    @property
    def checkpoint_dir(self) -> Path:
        """Checkpoint dir is inside the project's data_dir."""
        paths = get_project_paths()
        if paths:
            return paths["data_dir"] / "checkpoints"
        return Path("./data/checkpoints")  # Fallback

    # === Compatibility properties ===
    # These provide backward compatibility with code using old property names
    # They reference the module-level constants defined above

    @property
    def enable_ocr(self) -> bool:
        return ENABLE_OCR

    @property
    def ocr_engine(self) -> str:
        return OCR_ENGINE

    @property
    def ocr_languages(self) -> str:
        return OCR_LANGUAGES

    @property
    def enable_asr(self) -> bool:
        return ENABLE_ASR

    @property
    def embedding_model(self) -> str:
        return EMBEDDING_MODEL

    @property
    def chunking_method(self) -> str:
        return CHUNKING_METHOD

    @property
    def max_tokens(self) -> int:
        return MAX_TOKENS

    @property
    def default_top_k(self) -> int:
        return DEFAULT_TOP_K

    @property
    def mcp_transport(self) -> str:
        return MCP_TRANSPORT

    @property
    def mcp_enable_cleanup(self) -> bool:
        return MCP_ENABLE_CLEANUP

    @property
    def chroma_collection_name(self) -> str:
        return COLLECTION_NAME

    @property
    def log_level(self) -> str:
        return LOG_LEVEL

    @property
    def embedding_batch_size(self) -> int:
        return EMBEDDING_BATCH_SIZE

    @property
    def offline_mode(self) -> bool:
        return OFFLINE_MODE

    @property
    def chroma_mode(self) -> str:
        return CHROMA_MODE

    @property
    def mcp_host(self) -> str:
        return MCP_HOST

    @property
    def mcp_metrics_enabled(self) -> bool:
        return MCP_METRICS_ENABLED

    @property
    def mcp_metrics_retention_days(self) -> int:
        return MCP_METRICS_RETENTION_DAYS

    @property
    def language_detection_enabled(self) -> bool:
        return LANGUAGE_DETECTION_ENABLED

    @property
    def checkpoint_retention_days(self) -> int:
        return CHECKPOINT_RETENTION_DAYS

    # Uppercase aliases for direct access (e.g., settings.ENABLE_OCR)
    @property
    def ENABLE_OCR(self) -> bool:
        return ENABLE_OCR

    @property
    def OCR_ENGINE(self) -> str:
        return OCR_ENGINE

    @property
    def OCR_LANGUAGES(self) -> str:
        return OCR_LANGUAGES

    @property
    def ENABLE_ASR(self) -> bool:
        return ENABLE_ASR

    @property
    def EMBEDDING_MODEL(self) -> str:
        return EMBEDDING_MODEL

    @property
    def CHUNKING_METHOD(self) -> str:
        return CHUNKING_METHOD

    @property
    def MAX_TOKENS(self) -> int:
        return MAX_TOKENS

    @property
    def DEFAULT_TOP_K(self) -> int:
        return DEFAULT_TOP_K

    @property
    def MCP_TRANSPORT(self) -> str:
        return MCP_TRANSPORT

    @property
    def MCP_ENABLE_CLEANUP(self) -> bool:
        return MCP_ENABLE_CLEANUP

    @property
    def COLLECTION_NAME(self) -> str:
        return COLLECTION_NAME

    @property
    def LOG_LEVEL(self) -> str:
        return LOG_LEVEL

    @property
    def chroma_persist_dir(self) -> Path:
        """Get ChromaDB persist directory from active project."""
        paths = get_project_paths()
        if paths:
            return paths["chroma_path"]
        return Path("./data/chroma")  # Fallback

    @property
    def mcp_port(self) -> int:
        """Get MCP port from active project."""
        from .project import get_project_manager
        pm = get_project_manager()
        active = pm.get_active_project()
        if active:
            return active.port
        return 9090  # Default

    @property
    def mcp_server_name(self) -> str:
        """Get MCP server name from active project."""
        from .project import get_project_manager
        pm = get_project_manager()
        active = pm.get_active_project()
        if active:
            return active.mcp_server_name
        return "docling-rag"  # Default


# Create the settings instance
settings = Settings()


# ============================================================================
# Helper Functions
# ============================================================================


def get_project_paths() -> dict[str, Path] | None:
    """Get active project paths (data_dir, docs_dir, chroma_path, bm25_path).

    Returns:
        Dict with path information or None if no active project
    """
    try:
        from .project import get_project_manager
        pm = get_project_manager()
        active = pm.get_active_project()
        if not active:
            return None
        return pm.get_project_paths(active.name)
    except Exception:
        return None


def invalidate_cache():
    """Placeholder for backward compatibility."""
    pass


def apply_project_settings() -> bool:
    """Check if an active project exists.

    Kept for backward compatibility.

    Returns:
        True if an active project exists, False otherwise
    """
    return get_project_paths() is not None


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
