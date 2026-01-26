from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, Field

ChunkingMethod = Literal["hybrid", "hierarchical"]
Device = Literal["cpu", "cuda", "mps", "auto"]

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

class DoclingRagError(Exception): pass
class DocumentLoadError(DoclingRagError): pass
class ChunkingError(DoclingRagError): pass
class EmbeddingError(DoclingRagError): pass
class StorageError(DoclingRagError): pass
class IngestionError(DoclingRagError): pass
