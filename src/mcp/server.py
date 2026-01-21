import atexit
import signal
import sys

from fastmcp import FastMCP

from ..config import settings, get_logger
from ..query import query
from ..storage.chroma_client import get_stats, list_documents
from ..models import QueryResult
from ..utils import cleanup_all_resources

logger = get_logger(__name__)
mcp = FastMCP(settings.mcp_server_name)


@mcp.tool()
async def query_rag(
    query_text: str,
    top_k: int = 5,
) -> QueryResult:
    """Query RAG system and return relevant context chunks for LLM reasoning"""
    return query(query_text, top_k)


@mcp.tool()
async def get_statistics() -> dict:
    """Get RAG system statistics"""
    return get_stats()


@mcp.tool()
async def list_all_documents(
    limit: int | None = None,
    offset: int = 0,
) -> dict:
    """List all indexed documents with metadata and pagination support.

    Returns document information including file paths, types, languages,
    chunk counts, and ingestion timestamps.

    Args:
        limit: Maximum number of documents to return (None for all)
        offset: Number of documents to skip for pagination (default: 0)

    Returns:
        Dictionary with:
        - documents: List of document metadata
        - total: Total number of documents in database
        - showing: Number of documents returned
        - offset: Current offset

    Examples:
        list_all_documents()                    # Get all documents
        list_all_documents(limit=20)            # Get first 20 documents
        list_all_documents(limit=20, offset=20) # Get next 20 documents (page 2)
    """
    documents = list_documents(limit=limit, offset=offset)
    total = len(list_documents())  # Get total count

    return {
        "documents": documents,
        "total": total,
        "showing": len(documents),
        "offset": offset,
    }


def _cleanup_on_shutdown():
    """Cleanup resources on server shutdown."""
    logger.info("Shutting down MCP server, cleaning up resources...")
    cleanup_all_resources()
    logger.info("Cleanup complete")


def _handle_signal(signum, frame):
    """Handle shutdown signals (SIGINT, SIGTERM)."""
    logger.info(f"Received signal {signum}, initiating shutdown...")
    _cleanup_on_shutdown()
    sys.exit(0)


def run_server():
    """Run the MCP server with proper resource cleanup on shutdown."""
    # Register cleanup handlers
    atexit.register(_cleanup_on_shutdown)
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    logger.info("Starting MCP server with resource cleanup handlers registered")

    try:
        mcp.run()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        _cleanup_on_shutdown()
    except Exception as e:
        logger.error(f"Server error: {e}")
        _cleanup_on_shutdown()
        raise


if __name__ == "__main__":
    run_server()
