import atexit
import signal
import sys

from ..config import config, MCP_HOST, MCP_TRANSPORT, get_logger
from ..query import query as Query
from ..storage.chroma_client import list_documents
from ..models import QueryResult
from ..utils import cleanup_all_resources

logger = get_logger(__name__)

def _cleanup_on_shutdown():
    logger.info("Shutting down MCP server...")
    cleanup_all_resources()

def _handle_signal(signum, frame):
    _cleanup_on_shutdown()
    sys.exit(0)

def run_server():
    from fastmcp import FastMCP

    # Lazy initialization - read config at runtime
    server_name = config("MCP_SERVER_NAME")
    instructions = config("MCP_INSTRUCTIONS")
    query_desc = config("MCP_TOOL_QUERY_DESC")
    list_desc = config("MCP_TOOL_LIST_DOCS_DESC")
    port = config("MCP_PORT")

    mcp = FastMCP(name=server_name, instructions=instructions)

    @mcp.tool(name="search_documents",description=query_desc)
    async def search_documents(query: str, max_results: int = 5) -> QueryResult:
        return Query(query, max_results)

    @mcp.tool(name="list_all_documents",description=list_desc)
    async def list_all_documents(limit: int | None = 50, offset: int = 0) -> dict:
        documents = list_documents(limit=limit, offset=offset)
        total = len(list_documents())
        return {"documents": documents, "total": total, "showing": len(documents), "offset": offset}

    atexit.register(_cleanup_on_shutdown)
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    logger.info(f"Starting MCP server ({MCP_HOST}:{port})")
    try:
        mcp.run(transport=MCP_TRANSPORT, host=MCP_HOST, port=port)
    except KeyboardInterrupt:
        _cleanup_on_shutdown()
    except Exception as e:
        logger.error(f"Server error: {e}")
        _cleanup_on_shutdown()
        raise

if __name__ == "__main__":
    run_server()
