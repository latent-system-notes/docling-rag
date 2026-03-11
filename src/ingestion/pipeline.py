from pathlib import Path

from ..config import get_logger
from ..utils import embed, make_doc_id, display_path
from ..models import IngestionError, DocumentMetadata
from .chunker import chunk_document
from .document import extract_metadata, load_document
from ..storage.postgres import add_vectors, compute_effective_groups, set_document_permissions

logger = get_logger(__name__)


def ingest_document(file_path: str | Path, ocr_mode: str = "smart") -> DocumentMetadata:
    file_path = Path(file_path)
    if not file_path.exists():
        raise IngestionError(f"File not found: {file_path}")

    normalized_path = display_path(file_path)
    doc_id = make_doc_id(file_path)

    doc, page_count = load_document(file_path, ocr_mode=ocr_mode)
    chunks = chunk_document(doc, doc_id=normalized_path)
    if not chunks:
        del doc
        raise IngestionError(f"No chunks extracted from {file_path}")

    metadata = extract_metadata(doc, file_path, len(chunks), page_count)
    del doc

    metadata.file_path = normalized_path

    chunk_ids = [c.id for c in chunks]
    texts = [c.text for c in chunks]
    metadatas = [
        {"text": c.text, "doc_id": doc_id, "page_num": c.page_num,
         "doc_type": metadata.doc_type, "language": metadata.language, "file_path": normalized_path,
         "ingested_at": metadata.ingested_at.isoformat(), "chunk_index": c.metadata.get("index", i)}
        for i, c in enumerate(chunks)
    ]
    num_chunks = len(chunks)
    del chunks

    embeddings = embed(texts, show_progress=True)
    del texts
    add_vectors(chunk_ids, embeddings, metadatas)
    del chunk_ids, embeddings, metadatas

    group_ids = compute_effective_groups(normalized_path)
    if group_ids:
        set_document_permissions(doc_id, group_ids)
        logger.info(f"Assigned {len(group_ids)} group(s) to {normalized_path}")

    logger.info(f"Ingested {normalized_path}: {num_chunks} chunks")
    return metadata
