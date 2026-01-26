import atexit
import signal
import sys

from fastmcp import FastMCP

from ..config import MCP_SERVER_NAME, MCP_INSTRUCTIONS, MCP_TOOL_QUERY_DESC, MCP_TOOL_LIST_DOCS_DESC, MCP_HOST, MCP_PORT, MCP_TRANSPORT, get_logger
from ..query import query
from ..storage.chroma_client import list_documents
from ..models import QueryResult
from ..utils import cleanup_all_resources

logger = get_logger(__name__)
mcp = FastMCP(name=MCP_SERVER_NAME, instructions=MCP_INSTRUCTIONS)

@mcp.tool(description=MCP_TOOL_QUERY_DESC)
async def query_rag(query_text: str, top_k: int = 5) -> QueryResult:
    return query(query_text, top_k)

@mcp.tool(description=MCP_TOOL_LIST_DOCS_DESC)
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

    logger.info(f"Starting MCP server ({MCP_HOST}:{MCP_PORT})")
    try:
        mcp.run(transport=MCP_TRANSPORT, host=MCP_HOST, port=MCP_PORT)
    except KeyboardInterrupt:
        _cleanup_on_shutdown()
    except Exception as e:
        logger.error(f"Server error: {e}")
        _cleanup_on_shutdown()
        raise

if __name__ == "__main__":
    run_server()
