import hashlib
from pathlib import Path

from ..config import get_logger
from ..utils import embed
from ..models import IngestionError, DocumentMetadata
from .chunker import chunk_document
from .document import extract_metadata, load_document
from ..storage.chroma_client import add_vectors

logger = get_logger(__name__)

def ingest_document(file_path: str | Path) -> DocumentMetadata:
    file_path = Path(file_path)
    if not file_path.exists():
        raise IngestionError(f"File not found: {file_path}")

    doc, page_count = load_document(file_path)
    chunks = chunk_document(doc, doc_id=str(file_path))
    if not chunks:
        del doc
        raise IngestionError(f"No chunks extracted from {file_path}")

    metadata = extract_metadata(doc, file_path, len(chunks), page_count)
    del doc

    doc_id = hashlib.md5(str(file_path.absolute()).encode()).hexdigest()
    chunk_ids = [c.id for c in chunks]
    texts = [c.text for c in chunks]
    metadatas = [
        {"text": c.text, "doc_id": doc_id, "page_num": c.page_num,
         "doc_type": metadata.doc_type, "language": metadata.language, "file_path": metadata.file_path,
         "ingested_at": metadata.ingested_at.isoformat(), "chunk_index": c.metadata.get("index", i)}
        for i, c in enumerate(chunks)
    ]
    num_chunks = len(chunks)
    del chunks

    embeddings = embed(texts, show_progress=True)
    del texts
    add_vectors(chunk_ids, embeddings, metadatas)
    del chunk_ids, embeddings, metadatas

    logger.info(f"Ingested {file_path}: {num_chunks} chunks")
    return metadata
