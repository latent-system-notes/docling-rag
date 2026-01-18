from pathlib import Path

from ..config import settings, get_logger
from ..utils import embed_batch
from ..models import IngestionError, DocumentMetadata
from .chunker import chunk_document
from .document import extract_metadata, load_document
from ..storage.chroma_client import add_vectors

logger = get_logger(__name__)


def ingest_document(
    file_path: str | Path,
    doc_type: str = "",
) -> DocumentMetadata:
    try:
        file_path = Path(file_path)

        if not file_path.exists():
            raise IngestionError(f"File not found: {file_path}")

        logger.info(f"Ingesting document: {file_path}")

        doc = load_document(file_path)
        chunks = chunk_document(doc, doc_id=str(file_path))

        if not chunks:
            raise IngestionError(f"No chunks generated for {file_path}")

        metadata = extract_metadata(doc, file_path, len(chunks))

        chunk_texts = [chunk.text for chunk in chunks]
        embeddings = embed_batch(chunk_texts, desc=f"Embedding {file_path.name}")

        chunk_ids = [chunk.id for chunk in chunks]
        chunk_metadata = [
            {
                "text": chunk.text,
                "doc_id": chunk.doc_id,
                "page_num": chunk.page_num,
                "doc_type": metadata.doc_type,
                "language": metadata.language,
                "file_path": metadata.file_path,
                "ingested_at": metadata.ingested_at.isoformat(),
                "metadata": chunk.metadata,
            }
            for chunk in chunks
        ]

        add_vectors(chunk_ids, embeddings, chunk_metadata)

        logger.info(f"Successfully ingested {file_path}: {len(chunks)} chunks")
        return metadata

    except Exception as e:
        raise IngestionError(f"Failed to ingest {file_path}: {e}") from e


def ingest_batch(file_paths: list[Path]) -> list[DocumentMetadata]:
    results = []

    for file_path in file_paths:
        try:
            metadata = ingest_document(file_path)
            results.append(metadata)
        except Exception as e:
            logger.error(f"Failed to ingest {file_path}: {e}")

    return results
