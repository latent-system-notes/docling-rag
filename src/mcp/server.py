from pathlib import Path

from fastmcp import FastMCP

from ..config import settings
from ..ingestion.pipeline import ingest_document
from ..query import query
from ..storage.chroma_client import get_stats, initialize_collections, delete_by_filter
from ..models import AnswerMode, DocumentMetadata, QueryResult

mcp = FastMCP(settings.mcp_server_name)


@mcp.tool()
async def query_rag(
    query_text: str,
    top_k: int = 5,
    mode: AnswerMode | None = None,
) -> QueryResult:
    """Query RAG system. Modes: granite=Granite answers, context_only=raw context for Claude, both=both"""
    return query(query_text, top_k, mode)


@mcp.tool()
async def ingest_doc(file_path: str, doc_type: str = "") -> DocumentMetadata:
    """Ingest a document into the RAG system"""
    return ingest_document(Path(file_path), doc_type)


@mcp.tool()
async def get_statistics() -> dict:
    """Get RAG system statistics"""
    return get_stats()


@mcp.tool()
async def delete_document(doc_id: str) -> dict:
    """Delete a document and all its chunks"""
    delete_by_filter({"doc_id": doc_id})
    return {"status": "deleted", "doc_id": doc_id}


@mcp.tool()
async def initialize_system() -> dict:
    """Initialize the RAG system (create collections)"""
    initialize_collections()
    return {"status": "initialized"}


def run_server():
    mcp.run()


if __name__ == "__main__":
    run_server()
