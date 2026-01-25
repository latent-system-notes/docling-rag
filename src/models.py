"""Data models, types, and exceptions for the docling-rag system."""
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

# ============================================================================
# Type Aliases
# ============================================================================

ChunkingMethod = Literal["hybrid", "hierarchical"]
Device = Literal["cpu", "cuda", "mps", "auto"]


# ============================================================================
# Data Models
# ============================================================================


class Chunk(BaseModel):
    id: str
    text: str
    doc_id: str
    page_num: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentMetadata(BaseModel):
    doc_id: str
    doc_type: str
    language: str
    file_path: str
    num_chunks: int
    num_pages: int | None = None
    ingested_at: datetime = Field(default_factory=datetime.now)


class IngestionCheckpoint(BaseModel):
    """Checkpoint for resumable ingestion."""
    doc_id: str
    file_path: str
    file_hash: str
    total_chunks: int
    processed_batches: list[int] = Field(default_factory=list)
    last_batch: int = -1
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    chunk: Chunk
    score: float
    distance: float


class QueryResult(BaseModel):
    query: str
    context: list[SearchResult]


# ============================================================================
# Exceptions
# ============================================================================


class DoclingRagError(Exception):
    """Base exception for all RAG errors"""
    pass


class DocumentLoadError(DoclingRagError):
    """Failed to load document"""
    pass


class ChunkingError(DoclingRagError):
    """Failed to chunk document"""
    pass


class EmbeddingError(DoclingRagError):
    """Failed to generate embeddings"""
    pass


class StorageError(DoclingRagError):
    """Failed storage operation"""
    pass


class IngestionError(DoclingRagError):
    """Failed to ingest document"""
    pass
