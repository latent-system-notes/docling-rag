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

    mcp_server_name: str = "docling-rag"

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
