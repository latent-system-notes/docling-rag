import atexit
import signal
import sys

from fastmcp import FastMCP

from ..config import settings, get_logger
from ..query import query
from ..storage.chroma_client import list_documents
from ..models import QueryResult
from ..utils import cleanup_all_resources

logger = get_logger(__name__)
mcp = FastMCP(
    name=settings.mcp_server_name,
    instructions=settings.mcp_instructions,
)


@mcp.tool(description=settings.mcp_tool_query_description)
async def query_rag(
    query_text: str,
    top_k: int = 5,
) -> QueryResult:
    """Query RAG system and return relevant context chunks for LLM reasoning"""
    return query(query_text, top_k)


@mcp.tool(description=settings.mcp_tool_list_docs_description)
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
    """Run the MCP server with streamable-http transport and proper resource cleanup."""
    # Register cleanup handlers
    if settings.mcp_enable_cleanup:
        atexit.register(_cleanup_on_shutdown)
        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)

    logger.info(f"Starting MCP server (transport={settings.mcp_transport}, host={settings.mcp_host}, port={settings.mcp_port})")

    try:
        mcp.run(
            transport=settings.mcp_transport,
            host=settings.mcp_host,
            port=settings.mcp_port,
        )
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        if settings.mcp_enable_cleanup:
            _cleanup_on_shutdown()
    except Exception as e:
        logger.error(f"Server error: {e}")
        if settings.mcp_enable_cleanup:
            _cleanup_on_shutdown()
        raise


if __name__ == "__main__":
    run_server()
