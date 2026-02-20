import atexit
import signal
import sys

from ..config import config, MCP_HOST, MCP_TRANSPORT, get_logger
from ..query import query as Query
from ..storage.chroma_client import list_documents
from ..models import QueryResult
from ..utils import cleanup_all_resources

logger = get_logger(__name__)

def run_server():
    from fastmcp import FastMCP

    server_name = config("MCP_SERVER_NAME")
    port = config("MCP_PORT")
    mcp = FastMCP(name=server_name, instructions=config("MCP_INSTRUCTIONS"))

    @mcp.tool(name="search_documents", description=config("MCP_TOOL_QUERY_DESC"))
    async def search_documents(query: str, max_results: int = 5) -> QueryResult:
        return Query(query, max_results)

    @mcp.tool(name="list_all_documents", description=config("MCP_TOOL_LIST_DOCS_DESC"))
    async def list_all_documents(limit: int | None = 50, offset: int = 0) -> dict:
        from ..storage.chroma_client import get_document_count
        docs = list_documents(limit=limit, offset=offset)
        return {"documents": docs, "total": get_document_count(), "showing": len(docs), "offset": offset}

    atexit.register(cleanup_all_resources)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: (cleanup_all_resources(), sys.exit(0)))

    logger.info(f"Starting MCP server ({MCP_HOST}:{port})")
    try:
        mcp.run(transport=MCP_TRANSPORT, host=MCP_HOST, port=port)
    except KeyboardInterrupt:
        cleanup_all_resources()
