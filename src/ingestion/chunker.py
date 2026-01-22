import hashlib
from typing import Iterable

from docling.chunking import HybridChunker
from docling_core.types.doc import DoclingDocument
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer

from ..config import settings, get_logger
from ..models import ChunkingError, Chunk
from ..utils import get_model_paths

logger = get_logger(__name__)


def chunk_document(
    doc: DoclingDocument,
    doc_id: str,
    method: str | None = None,
    max_tokens: int | None = None,
) -> list[Chunk]:
    try:
        method = method or settings.chunking_method
        max_tokens = max_tokens or settings.max_tokens

        if method == "hybrid":
            # Load tokenizer from local path with offline mode enforced
            from transformers import AutoTokenizer

            local_model_path = get_model_paths()["embedding"]

            if not local_model_path.exists():
                raise ChunkingError(
                    f"Tokenizer model not found at {local_model_path}. "
                    "OFFLINE MODE: Cannot download models because HF_HUB_OFFLINE=1. "
                    "Run 'rag models --download' first (requires internet connection)."
                )

            # Load tokenizer with strict offline mode
            hf_tokenizer = AutoTokenizer.from_pretrained(
                str(local_model_path),
                local_files_only=True,  # Enforce offline mode
                trust_remote_code=False  # Security
            )

            # Wrap in Docling's HuggingFaceTokenizer
            tokenizer = HuggingFaceTokenizer(
                tokenizer=hf_tokenizer,
                max_tokens=max_tokens
            )

            # Create chunker with offline tokenizer
            chunker = HybridChunker(
                tokenizer=tokenizer,
                max_tokens=max_tokens,
            )
        else:
            from docling.chunking import HierarchicalChunker

            chunker = HierarchicalChunker()

        chunks: Iterable = chunker.chunk(doc)
        result = []

        for idx, chunk in enumerate(chunks):
            chunk_text = chunk.text
            chunk_id = hashlib.md5(f"{doc_id}_{idx}_{chunk_text}".encode()).hexdigest()

            page_num = None
            if hasattr(chunk, "meta") and chunk.meta:
                try:
                    # Convert DocMeta to dict if it's a pydantic model
                    meta_dict = chunk.meta.model_dump() if hasattr(chunk.meta, "model_dump") else chunk.meta
                    if isinstance(meta_dict, dict):
                        page_num = meta_dict.get("doc_items", [{}])[0].get("prov", [{}])[0].get("page_no")
                except (AttributeError, IndexError, KeyError, TypeError):
                    # If metadata structure is unexpected, just skip page number extraction
                    pass

            result.append(
                Chunk(
                    id=chunk_id,
                    text=chunk_text,
                    doc_id=doc_id,
                    page_num=page_num,
                    metadata={"index": idx},
                )
            )

        logger.info(f"Created {len(result)} chunks for document {doc_id}")
        return result

    except Exception as e:
        raise ChunkingError(f"Failed to chunk document: {e}") from e
