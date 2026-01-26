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
mcp = FastMCP(name=settings.mcp_server_name, instructions=settings.mcp_instructions)

@mcp.tool(description=settings.mcp_tool_query_description)
async def query_rag(query_text: str, top_k: int = 5) -> QueryResult:
    return query(query_text, top_k)

@mcp.tool(description=settings.mcp_tool_list_docs_description)
async def list_all_documents(limit: int | None = None, offset: int = 0) -> dict:
    documents = list_documents(limit=limit, offset=offset)
    total = len(list_documents())
    return {"documents": documents, "total": total, "showing": len(documents), "offset": offset}

def _cleanup_on_shutdown():
    logger.info("Shutting down MCP server...")
    cleanup_all_resources()

def _handle_signal(signum, frame):
    _cleanup_on_shutdown()
    sys.exit(0)

def run_server():
    atexit.register(_cleanup_on_shutdown)
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    logger.info(f"Starting MCP server ({settings.mcp_host}:{settings.mcp_port})")
    try:
        mcp.run(transport=settings.mcp_transport, host=settings.mcp_host, port=settings.mcp_port)
    except KeyboardInterrupt:
        _cleanup_on_shutdown()
    except Exception as e:
        logger.error(f"Server error: {e}")
        _cleanup_on_shutdown()
        raise

if __name__ == "__main__":
    run_server()
