import logging
import os
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*Token indices sequence length.*")
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
if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = str(Path("./models/.cache").absolute())
for k in ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"]:
    os.environ[k] = "1"

try:
    import torch
    torch.set_default_device({"cpu": "cpu", "cuda": "cuda" if torch.cuda.is_available() else "cpu",
                              "mps": "mps" if torch.backends.mps.is_available() else "cpu"}.get(device, "cpu"))
except ImportError:
    pass

MODELS_DIR = Path("./models")
CHROMA_PERSIST_DIR = Path("./data/chroma")
CHECKPOINT_DIR = Path("./data/checkpoints")
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
EMBEDDING_BATCH_SIZE = 32
MAX_TOKENS = 512
DEFAULT_TOP_K = 5
COLLECTION_NAME = "documents"
CHECKPOINT_RETENTION_DAYS = 7

MCP_HOST = "0.0.0.0"
MCP_PORT = 9090
MCP_TRANSPORT = "streamable-http"
MCP_SERVER_NAME = "docling-rag"
MCP_INSTRUCTIONS = "RAG assistant with document access. Use query_rag and list_all_documents."
MCP_TOOL_QUERY_DESC = "Query RAG for relevant chunks. Args: query_text, top_k"
MCP_TOOL_LIST_DOCS_DESC = "List indexed documents. Args: limit, offset"

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger
