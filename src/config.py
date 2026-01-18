import logging
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from .models import AnswerMode, ChunkingMethod, Device, Quantization


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="RAG_", case_sensitive=False)

    base_dir: Path = Path.cwd()
    models_dir: Path = Path("./models")
    data_dir: Path = Path("./data")

    chroma_persist_dir: str = "./data/chroma"
    chroma_collection_name: str = "documents"

    # ChromaDB connection mode
    chroma_mode: str = "persistent"  # "persistent" or "http"

    # ChromaDB HTTP server settings (only used when mode="http")
    chroma_server_host: str = "localhost"
    chroma_server_port: int = 8000
    chroma_server_ssl: bool = False
    chroma_server_api_key: str = ""  # Optional: for authentication

    embedding_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    embedding_dim: int = 768
    embedding_batch_size: int = 32

    chunking_method: ChunkingMethod = "hybrid"
    max_tokens: int = 512
    chunk_overlap_tokens: int = 50

    enable_ocr: bool = True
    enable_asr: bool = True
    ocr_languages: str = "eng+ara"

    default_top_k: int = 5

    granite_model_id: str = "ibm-granite/granite-3.1-2b-instruct"
    granite_quantization: Quantization = "none"
    answer_mode: AnswerMode = "both"
    max_context_chunks: int = 5
    max_new_tokens: int = 512
    temperature: float = 0.7

    mcp_server_name: str = "dockling-rag"

    language_detection_enabled: bool = True

    device: Device = "cpu"
    log_level: str = "INFO"


settings = Settings()


# ============================================================================
# Logging
# ============================================================================


def get_logger(name: str) -> logging.Logger:
    """Get or create a logger with the configured log level.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(settings.log_level)

    return logger
