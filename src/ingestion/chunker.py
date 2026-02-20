import hashlib

from docling.chunking import HybridChunker
from docling_core.types.doc import DoclingDocument
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer

from ..config import MAX_TOKENS, get_logger
from ..models import ChunkingError, Chunk
from ..utils import get_embedding_model_path

logger = get_logger(__name__)
_chunker_cache = None


def get_chunker() -> HybridChunker:
    global _chunker_cache
    if _chunker_cache is not None:
        return _chunker_cache
    local_model_path = get_embedding_model_path()
    if not local_model_path.exists():
        raise ChunkingError(f"Tokenizer model not found at {local_model_path}. Run 'rag models --download' first.")
    tokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(str(local_model_path), local_files_only=True, trust_remote_code=False),
        max_tokens=MAX_TOKENS)
    _chunker_cache = HybridChunker(tokenizer=tokenizer, max_tokens=MAX_TOKENS)
    return _chunker_cache


def cleanup_chunker() -> None:
    global _chunker_cache
    _chunker_cache = None


def _extract_page_num(chunk) -> int | None:
    try:
        meta = chunk.meta.model_dump() if hasattr(chunk.meta, "model_dump") else chunk.meta
        if isinstance(meta, dict):
            return meta["doc_items"][0]["prov"][0]["page_no"]
    except (AttributeError, IndexError, KeyError, TypeError):
        pass
    return None


def chunk_document(doc: DoclingDocument, doc_id: str) -> list[Chunk]:
    try:
        result = []
        for idx, chunk in enumerate(get_chunker().chunk(doc)):
            chunk_id = hashlib.md5(f"{doc_id}_{idx}_{chunk.text}".encode()).hexdigest()
            result.append(Chunk(id=chunk_id, text=chunk.text, doc_id=doc_id,
                                page_num=_extract_page_num(chunk) if hasattr(chunk, "meta") and chunk.meta else None,
                                metadata={"index": idx}))
        return result
    except Exception as e:
        raise ChunkingError(f"Failed to chunk document: {e}") from e
