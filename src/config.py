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


class InfrastructureSettings(BaseSettings):
    """Infrastructure settings that are truly global and should NOT vary per project.

    These settings are read from environment variables with RAG_ prefix.
    Project-specific settings are handled by EffectiveSettings class.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="RAG_",
        case_sensitive=False,
        extra="ignore"  # Ignore removed env vars (RAG_ENABLE_OCR, etc.)
    )

    # === Global Infrastructure Settings ===
    models_dir: Path = Path("./models")

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

    # Performance tuning
    embedding_batch_size: int = 32

    # Offline mode - prevents any network access for model downloads
    # Set to False only when running 'rag models --download'
    offline_mode: bool = True

    # Feature flags
    language_detection_enabled: bool = True

    # Device - system-wide default (project can override)
    device: Device = "cpu"

    # MCP Monitoring Configuration (global)
    mcp_metrics_enabled: bool = True
    mcp_metrics_retention_days: int = 7

    # MCP Tool Descriptions (global - same for all projects)
    mcp_instructions: str = "You are a RAG (Retrieval-Augmented Generation) assistant with access to indexed documents. Use query_rag to retrieve relevant context and list_all_documents to browse available documents."

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


# Global infrastructure settings instance
_infra_settings = InfrastructureSettings()


class EffectiveSettings:
    """Unified settings: project config > env vars > defaults.

    This class provides a single source of truth for all settings.
    Project-specific settings are read from the active project.
    Infrastructure settings are read from environment variables.

    Usage:
        from src.config import settings
        print(settings.enable_ocr)  # Reads from project or default
        print(settings.models_dir)  # Reads from infrastructure settings
    """

    def __init__(self):
        self._project_cache = None
        self._project_cache_valid = False

    def _get_project(self) -> dict | None:
        """Get active project settings (cached)."""
        if not self._project_cache_valid:
            self._project_cache = get_active_project_settings()
            self._project_cache_valid = True
        return self._project_cache

    def invalidate_cache(self):
        """Call after switching projects to refresh settings."""
        self._project_cache_valid = False
        self._project_cache = None

    # === Infrastructure Settings (from env vars) ===
    @property
    def models_dir(self) -> Path:
        return _infra_settings.models_dir

    @property
    def offline_mode(self) -> bool:
        return _infra_settings.offline_mode

    @property
    def checkpoint_dir(self) -> Path:
        return _infra_settings.checkpoint_dir

    @property
    def checkpoint_retention_days(self) -> int:
        return _infra_settings.checkpoint_retention_days

    @property
    def chroma_mode(self) -> str:
        return _infra_settings.chroma_mode

    @property
    def chroma_server_host(self) -> str:
        return _infra_settings.chroma_server_host

    @property
    def chroma_server_port(self) -> int:
        return _infra_settings.chroma_server_port

    @property
    def chroma_server_ssl(self) -> bool:
        return _infra_settings.chroma_server_ssl

    @property
    def chroma_server_api_key(self) -> str:
        return _infra_settings.chroma_server_api_key

    @property
    def embedding_batch_size(self) -> int:
        return _infra_settings.embedding_batch_size

    @property
    def language_detection_enabled(self) -> bool:
        return _infra_settings.language_detection_enabled

    @property
    def mcp_metrics_enabled(self) -> bool:
        return _infra_settings.mcp_metrics_enabled

    @property
    def mcp_metrics_retention_days(self) -> int:
        return _infra_settings.mcp_metrics_retention_days

    @property
    def mcp_instructions(self) -> str:
        return _infra_settings.mcp_instructions

    @property
    def mcp_tool_query_description(self) -> str:
        return _infra_settings.mcp_tool_query_description

    @property
    def mcp_tool_list_docs_description(self) -> str:
        return _infra_settings.mcp_tool_list_docs_description

    # === Project Settings (project > default) ===
    @property
    def enable_ocr(self) -> bool:
        project = self._get_project()
        if project:
            return project["enable_ocr"]
        return True  # Default: True

    @property
    def ocr_engine(self) -> str:
        project = self._get_project()
        if project:
            return project["ocr_engine"]
        return "auto"  # Default

    @property
    def ocr_languages(self) -> str:
        project = self._get_project()
        if project:
            return project["ocr_languages"]
        return "eng+ara"  # Default

    @property
    def enable_asr(self) -> bool:
        project = self._get_project()
        if project:
            return project["enable_asr"]
        return True  # Default: True

    @property
    def embedding_model(self) -> str:
        project = self._get_project()
        if project:
            return project["embedding_model"]
        return "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"  # Default

    @property
    def chunking_method(self) -> str:
        project = self._get_project()
        if project:
            return project["chunking_method"]
        return "hybrid"  # Default

    @property
    def max_tokens(self) -> int:
        project = self._get_project()
        if project:
            return project["max_tokens"]
        return 512  # Default

    @property
    def default_top_k(self) -> int:
        project = self._get_project()
        if project:
            return project["default_top_k"]
        return 5  # Default

    @property
    def device(self) -> str:
        """Device: project config > env var > 'cpu'."""
        project = self._get_project()
        if project and project.get("device"):
            return project["device"]
        return _infra_settings.device  # Falls back to RAG_DEVICE env var

    @property
    def log_level(self) -> str:
        project = self._get_project()
        if project:
            return project["log_level"]
        return "INFO"  # Default

    # === MCP Project Settings ===
    @property
    def mcp_server_name(self) -> str:
        project = self._get_project()
        if project:
            return project["mcp_server_name"]
        return "docling-rag"  # Default

    @property
    def mcp_transport(self) -> str:
        project = self._get_project()
        if project:
            return project["mcp_transport"]
        return "streamable-http"  # Default

    @property
    def mcp_host(self) -> str:
        project = self._get_project()
        if project:
            return project["mcp_host"]
        return "0.0.0.0"  # Default

    @property
    def mcp_port(self) -> int:
        project = self._get_project()
        if project:
            return project["port"]
        return 8080  # Default

    @property
    def mcp_enable_cleanup(self) -> bool:
        project = self._get_project()
        if project:
            return project["mcp_enable_cleanup"]
        return True  # Default

    # === Path Settings (project > default) ===
    @property
    def chroma_persist_dir(self) -> Path:
        project = self._get_project()
        if project:
            return Path(project["chroma_persist_dir"])
        return Path("./data/chroma")  # Default

    @property
    def chroma_collection_name(self) -> str:
        return "documents"  # Always the same

    # === Utility Methods ===
    def model_dump(self) -> dict:
        """Return all settings as a dictionary (for compatibility)."""
        return {
            # Infrastructure
            "models_dir": self.models_dir,
            "offline_mode": self.offline_mode,
            "checkpoint_dir": self.checkpoint_dir,
            "checkpoint_retention_days": self.checkpoint_retention_days,
            "chroma_mode": self.chroma_mode,
            "chroma_server_host": self.chroma_server_host,
            "chroma_server_port": self.chroma_server_port,
            "chroma_server_ssl": self.chroma_server_ssl,
            "chroma_server_api_key": self.chroma_server_api_key,
            "embedding_batch_size": self.embedding_batch_size,
            "language_detection_enabled": self.language_detection_enabled,
            "mcp_metrics_enabled": self.mcp_metrics_enabled,
            "mcp_metrics_retention_days": self.mcp_metrics_retention_days,
            "mcp_instructions": self.mcp_instructions,
            "mcp_tool_query_description": self.mcp_tool_query_description,
            "mcp_tool_list_docs_description": self.mcp_tool_list_docs_description,
            # Project-specific
            "enable_ocr": self.enable_ocr,
            "ocr_engine": self.ocr_engine,
            "ocr_languages": self.ocr_languages,
            "enable_asr": self.enable_asr,
            "embedding_model": self.embedding_model,
            "chunking_method": self.chunking_method,
            "max_tokens": self.max_tokens,
            "default_top_k": self.default_top_k,
            "device": self.device,
            "log_level": self.log_level,
            "mcp_server_name": self.mcp_server_name,
            "mcp_transport": self.mcp_transport,
            "mcp_host": self.mcp_host,
            "mcp_port": self.mcp_port,
            "mcp_enable_cleanup": self.mcp_enable_cleanup,
            "chroma_persist_dir": self.chroma_persist_dir,
            "chroma_collection_name": self.chroma_collection_name,
        }

    def get_setting_source(self, key: str) -> str:
        """Return the source of a setting value ('project', 'env', or 'default')."""
        infrastructure_keys = {
            "models_dir", "offline_mode", "checkpoint_dir", "checkpoint_retention_days",
            "chroma_mode", "chroma_server_host", "chroma_server_port", "chroma_server_ssl",
            "chroma_server_api_key", "embedding_batch_size", "language_detection_enabled",
            "mcp_metrics_enabled", "mcp_metrics_retention_days", "mcp_instructions",
            "mcp_tool_query_description", "mcp_tool_list_docs_description",
        }

        if key in infrastructure_keys:
            return "env"

        project = self._get_project()
        if project:
            return "project"

        return "default"


# Create the effective settings instance
effective_settings = EffectiveSettings()

# Backward compatibility alias
settings = effective_settings


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
    """Check if an active project exists and invalidate settings cache.

    This function is kept for backward compatibility. With EffectiveSettings,
    project settings are read dynamically, so this just invalidates the cache
    and returns whether a project is active.

    Returns:
        True if an active project exists, False otherwise
    """
    # Invalidate cache to ensure fresh project settings are read
    effective_settings.invalidate_cache()

    # Check if there's an active project
    project = get_active_project_settings()
    return project is not None


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
