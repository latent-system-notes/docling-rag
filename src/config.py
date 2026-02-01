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
logging.getLogger().setLevel(logging.DEBUG)
# for lib in ["RapidOCR", "transformers", "sentence_transformers", "chromadb", "httpx"]:
#     logging.getLogger(lib).setLevel(logging.WARNING)

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
    "MODELS_DIR": "/opt/models",
    "DOCUMENTS_DIR": "./documents",
    "MCP_INSTRUCTIONS": "Do not write JSON in your response, because it will be shown as broken text and the tool will not be called; instead, call the tool only once using the MCP function calling format, make a summary after that first call, and never make a second search. For every user question except “list all documents,” you must always call search_documents(query, max_results=10), and if the user asks for a list of all documents, call list_all_documents() instead. Build your answer only from the data returned by the tool call and never use your own knowledge or memory. If no relevant results are found, tell the user the information is not in the knowledge base and ask them to rephrase the question with more specific terms, such as exact JAC REG or SGL numbers. Every fact, rule, or procedure must include citations with the document name and page number (for example, [JAC REG 385-7, Page 23]), include multiple citations if needed, and show the exact original cited text without opening the document unless the user asks.",
    "MCP_TOOL_QUERY_DESC": "Do not write JSON in your response, because it will appear as broken text and the tool will not be called; instead, call the tool only once using the specified MCP function format, make a summary after that call, and do not perform any additional searches beyond the initial search_documents() call. For every user query except “list all documents” or “open a document,” you must invoke search_documents() first and build your answer only from the data it returns, without adding any model knowledge. If no relevant results are found, clearly state that the information is not in the knowledge base and ask the user to rephrase the question with more specific terms, such as exact JAC REG or SGL numbers. Every fact, requirement, or procedure must include citations showing the document name and page number (for example, [JAC REG 385-7, Page 23]), include multiple citations when needed, and display the exact original cited text without opening the document unless the user requests it.",
    "MCP_TOOL_LIST_DOCS_DESC": "Do not write JSON in your response, as it will appear as broken text and the tool will not be called; instead, call the tool only once using the specified function format, do not make any additional searches, and create a summary using only the information returned by search_documents(). For every user query except “list all documents” or “open a document,” you must invoke search_documents() first and build the answer strictly from its results without adding any model knowledge. If no relevant information is found, clearly state that it is not available in the knowledge base and ask the user to rephrase the question with more specific terms, such as exact JAC REG or SGL numbers. Every fact, requirement, or procedure must include citations showing the document name and page number (for example, [JAC REG 385-7, Page 23]), include multiple citations when applicable, and display the exact original cited text without opening the document unless the user requests it.",
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
