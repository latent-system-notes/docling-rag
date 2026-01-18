"""Data models, types, and exceptions for the docling-rag system."""
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

# ============================================================================
# Type Aliases
# ============================================================================

AnswerMode = Literal["granite", "context_only", "both"]
ChunkingMethod = Literal["hybrid", "hierarchical"]
Quantization = Literal["none", "4bit", "8bit"]
Device = Literal["cpu", "cuda", "mps"]


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
    ingested_at: datetime = Field(default_factory=datetime.now)


class SearchResult(BaseModel):
    chunk: Chunk
    score: float
    distance: float


class QueryResult(BaseModel):
    query: str
    context: list[SearchResult]
    answer: str | None = None
    mode: AnswerMode


# ============================================================================
# Exceptions
# ============================================================================


class DocklingRagError(Exception):
    """Base exception for all RAG errors"""
    pass


class DocumentLoadError(DocklingRagError):
    """Failed to load document"""
    pass


class ChunkingError(DocklingRagError):
    """Failed to chunk document"""
    pass


class EmbeddingError(DocklingRagError):
    """Failed to generate embeddings"""
    pass


class StorageError(DocklingRagError):
    """Failed storage operation"""
    pass


class SearchError(DocklingRagError):
    """Failed search operation"""
    pass


class GenerationError(DocklingRagError):
    """Failed to generate answer"""
    pass


class IngestionError(DocklingRagError):
    """Failed to ingest document"""
    pass
