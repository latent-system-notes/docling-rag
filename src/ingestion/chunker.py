import hashlib
from typing import Iterable

from docling.chunking import HybridChunker
from docling_core.types.doc import DoclingDocument
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer

from ..config import MAX_TOKENS, get_logger
from ..models import ChunkingError, Chunk
from ..utils import get_model_paths

logger = get_logger(__name__)

def chunk_document(doc: DoclingDocument, doc_id: str) -> list[Chunk]:
    try:
        local_model_path = get_model_paths()["embedding"]
        if not local_model_path.exists():
            raise ChunkingError(f"Tokenizer model not found at {local_model_path}. Run 'rag models --download' first.")
        hf_tokenizer = AutoTokenizer.from_pretrained(str(local_model_path), local_files_only=True, trust_remote_code=False)
        tokenizer = HuggingFaceTokenizer(tokenizer=hf_tokenizer, max_tokens=MAX_TOKENS)
        chunker = HybridChunker(tokenizer=tokenizer, max_tokens=MAX_TOKENS)

        chunks: Iterable = chunker.chunk(doc)
        result = []
        for idx, chunk in enumerate(chunks):
            chunk_text = chunk.text
            chunk_id = hashlib.md5(f"{doc_id}_{idx}_{chunk_text}".encode()).hexdigest()
            page_num = None
            if hasattr(chunk, "meta") and chunk.meta:
                try:
                    meta_dict = chunk.meta.model_dump() if hasattr(chunk.meta, "model_dump") else chunk.meta
                    if isinstance(meta_dict, dict):
                        page_num = meta_dict.get("doc_items", [{}])[0].get("prov", [{}])[0].get("page_no")
                except (AttributeError, IndexError, KeyError, TypeError):
                    pass
            result.append(Chunk(id=chunk_id, text=chunk_text, doc_id=doc_id, page_num=page_num, metadata={"index": idx}))
        return result
    except Exception as e:
        raise ChunkingError(f"Failed to chunk document: {e}") from e
