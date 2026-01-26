import logging
import os
import time
import warnings
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

warnings.filterwarnings("ignore", message=".*Token indices sequence length.*")
warnings.filterwarnings("ignore", category=FutureWarning)

# Logging
class _Fmt(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        return time.strftime(datefmt, self.converter(record.created)).upper() if datefmt else super().formatTime(record, datefmt)

_h = logging.StreamHandler()
_h.setFormatter(_Fmt("[%(levelname)s] [%(asctime)s] %(message)s", "%d-%b %H:%M:%S"))
logging.getLogger().handlers = [_h]
logging.getLogger().setLevel(logging.INFO)
for lib in ["RapidOCR", "transformers", "sentence_transformers", "chromadb", "httpx"]:
    logging.getLogger(lib).setLevel(logging.WARNING)

# Device setup
_dev = os.environ.get("RAG_DEVICE", "cpu").lower()
if _dev == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = str(Path("./models/.cache").absolute())
for k in ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"]:
    os.environ[k] = "1"

try:
    import torch
    torch.set_default_device({"cpu": "cpu", "cuda": "cuda" if torch.cuda.is_available() else "cpu",
                              "mps": "mps" if torch.backends.mps.is_available() else "cpu"}.get(_dev, "cpu"))
except ImportError:
    pass

# Hardcoded constants
ENABLE_OCR = False
ENABLE_ASR = True
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
EMBEDDING_BATCH_SIZE = 32
CHUNKING_METHOD = "hybrid"
MAX_TOKENS = 512
DEFAULT_TOP_K = 5
MCP_TRANSPORT = "streamable-http"
MCP_HOST = "0.0.0.0"
COLLECTION_NAME = "documents"
MODELS_DIR = Path("./models")

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="RAG_", extra="ignore")
    device: str = "cpu"

    mcp_instructions: str = "RAG assistant with document access. Use query_rag and list_all_documents."
    mcp_tool_query_description: str = "Query RAG for relevant chunks. Args: query_text, top_k"
    mcp_tool_list_docs_description: str = "List indexed documents. Args: limit, offset"

    @property
    def models_dir(self): return MODELS_DIR
    @property
    def checkpoint_dir(self):
        p = get_project_paths()
        return p["data_dir"] / "checkpoints" if p else Path("./data/checkpoints")
    @property
    def chroma_persist_dir(self):
        p = get_project_paths()
        return p["chroma_path"] if p else Path("./data/chroma")
    @property
    def mcp_port(self):
        from .project import get_project_manager
        a = get_project_manager().get_active_project()
        return a.port if a else 9090
    @property
    def mcp_server_name(self):
        from .project import get_project_manager
        a = get_project_manager().get_active_project()
        return a.mcp_server_name if a else "docling-rag"

    # Compatibility aliases
    enable_ocr = property(lambda s: ENABLE_OCR)
    enable_asr = property(lambda s: ENABLE_ASR)
    embedding_model = property(lambda s: EMBEDDING_MODEL)
    embedding_batch_size = property(lambda s: EMBEDDING_BATCH_SIZE)
    chunking_method = property(lambda s: CHUNKING_METHOD)
    max_tokens = property(lambda s: MAX_TOKENS)
    default_top_k = property(lambda s: DEFAULT_TOP_K)
    mcp_transport = property(lambda s: MCP_TRANSPORT)
    mcp_host = property(lambda s: MCP_HOST)
    chroma_collection_name = property(lambda s: COLLECTION_NAME)
    offline_mode = property(lambda s: True)
    chroma_mode = property(lambda s: "persistent")
    language_detection_enabled = property(lambda s: True)
    log_level = property(lambda s: "INFO")
    checkpoint_retention_days = property(lambda s: 7)

settings = Settings()

def get_project_paths():
    try:
        from .project import get_project_manager
        pm = get_project_manager()
        a = pm.get_active_project()
        return pm.get_project_paths(a.name) if a else None
    except:
        return None

def apply_project_settings():
    return get_project_paths() is not None

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger
