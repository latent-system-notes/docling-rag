import logging
import os
import time
import warnings
from pathlib import Path

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.filterwarnings("ignore", message=".*Token indices sequence length.*")
warnings.filterwarnings("ignore", message=".*incorrect regex pattern.*")
warnings.filterwarnings("ignore", category=FutureWarning)

class _Fmt(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        return time.strftime(datefmt, self.converter(record.created)).upper() if datefmt else super().formatTime(record, datefmt)

_h = logging.StreamHandler()
_h.setFormatter(_Fmt("[%(levelname)s] [%(asctime)s] %(message)s", "%d-%b %H:%M:%S"))
logging.getLogger().handlers = [_h]
logging.getLogger().setLevel(logging.INFO)
for lib in ["RapidOCR", "transformers", "sentence_transformers", "chromadb", "httpx"]:
    logging.getLogger(lib).setLevel(logging.WARNING)

device = os.environ.get("RAG_DEVICE", "cpu").lower()
if device == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

try:
    import torch
    torch.set_default_device({"cpu": "cpu", "cuda": "cuda" if torch.cuda.is_available() else "cpu",
                              "mps": "mps" if torch.backends.mps.is_available() else "cpu"}.get(device, "cpu"))
except ImportError:
    pass

# HARDCODED CONSTANTS
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
EMBEDDING_BATCH_SIZE = 32
MAX_TOKENS = 512
DEFAULT_TOP_K = 5
COLLECTION_NAME = "documents"
MCP_HOST = "0.0.0.0"
MCP_TRANSPORT = "streamable-http"

# DEFAULTS for env-configurable params
_ENV_DEFAULTS = {
    "MCP_SERVER_NAME": "docling-rag",
    "MCP_PORT": "9090",
    "DATA_DIR": "./data",
    "MODELS_DIR": "./models",
    "DOCUMENTS_DIR": "./documents",
    "MCP_INSTRUCTIONS": "RAG assistant with document access. Use query_rag and list_all_documents.",
    "MCP_TOOL_QUERY_DESC": "Query RAG for relevant chunks. Args: query_text, top_k",
    "MCP_TOOL_LIST_DOCS_DESC": "List indexed documents. Args: limit, offset",
}

def config(key: str):
    """Get env-configurable value. Call at runtime, not import time."""
    value = os.environ.get(key) or _ENV_DEFAULTS.get(key)
    if key == "MCP_PORT":
        return int(value)
    if key in ("DATA_DIR", "MODELS_DIR", "DOCUMENTS_DIR"):
        return Path(value)
    return value

def get_chroma_persist_dir() -> Path:
    return config("DATA_DIR") / "chroma"

def _setup_hf_env():
    """Setup HuggingFace environment. Called after env is loaded."""
    models_dir = config("MODELS_DIR")
    if "HF_HOME" not in os.environ:
        os.environ["HF_HOME"] = str((models_dir / ".cache").absolute())
    for k in ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"]:
        os.environ[k] = "1"

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger
