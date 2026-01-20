from pathlib import Path

from ..config import settings, get_logger
from ..utils import embed
from ..models import IngestionError, DocumentMetadata
from .chunker import chunk_document
from .document import extract_metadata, load_document
from ..storage.chroma_client import add_vectors

logger = get_logger(__name__)


def ingest_document(file_path: str | Path) -> DocumentMetadata:
    """Ingest a document into the RAG system.

    Args:
        file_path: Path to the document file

    Returns:
        DocumentMetadata with ingestion details

    Raises:
        IngestionError: If ingestion fails
    """
    try:
        file_path = Path(file_path)

        if not file_path.exists():
            raise IngestionError(f"File not found: {file_path}")

        logger.info(f"Loading document: {file_path}")

        doc = load_document(file_path)

        logger.info(f"Chunking document...")
        chunks = chunk_document(doc, doc_id=str(file_path))

        if not chunks:
            raise IngestionError(f"No chunks generated for {file_path}")

        metadata = extract_metadata(doc, file_path, len(chunks))

        chunk_texts = [chunk.text for chunk in chunks]
        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        embeddings = embed(chunk_texts, show_progress=True)

        chunk_ids = [chunk.id for chunk in chunks]
        chunk_metadata = [
            {
                "text": chunk.text,
                "doc_id": metadata.doc_id,  # Use the MD5 hash from metadata, not chunk.doc_id
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
