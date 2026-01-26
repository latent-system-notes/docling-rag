import atexit
import os
import signal
import sys

from fastmcp import FastMCP

from ..config import settings, get_logger
from ..query import query
from ..storage.chroma_client import list_documents
from ..models import QueryResult
from ..utils import cleanup_all_resources
from .monitoring import track_query
from .status import get_mcp_status_manager

logger = get_logger(__name__)
mcp = FastMCP(name=settings.mcp_server_name, instructions=settings.mcp_instructions)

@mcp.tool(description=settings.mcp_tool_query_description)
@track_query
async def query_rag(query_text: str, top_k: int = 5) -> QueryResult:
    return query(query_text, top_k)

@mcp.tool(description=settings.mcp_tool_list_docs_description)
async def list_all_documents(limit: int | None = None, offset: int = 0) -> dict:
    documents = list_documents(limit=limit, offset=offset)
    total = len(list_documents())
    return {"documents": documents, "total": total, "showing": len(documents), "offset": offset}

def _cleanup_on_shutdown():
    logger.info("Shutting down MCP server...")
    try:
        get_mcp_status_manager().mark_server_stopped()
    except Exception:
        pass
    cleanup_all_resources()

def _handle_signal(signum, frame):
    _cleanup_on_shutdown()
    sys.exit(0)

def run_server():
    if settings.mcp_enable_cleanup:
        atexit.register(_cleanup_on_shutdown)
        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)

    if settings.mcp_metrics_enabled:
        try:
            manager = get_mcp_status_manager()
            manager.register_server(pid=os.getpid(), host=settings.mcp_host, port=settings.mcp_port)
            manager.cleanup_old_metrics(retention_days=settings.mcp_metrics_retention_days)
        except Exception:
            pass

    logger.info(f"Starting MCP server ({settings.mcp_host}:{settings.mcp_port})")
    try:
        mcp.run(transport=settings.mcp_transport, host=settings.mcp_host, port=settings.mcp_port)
    except KeyboardInterrupt:
        if settings.mcp_enable_cleanup:
            _cleanup_on_shutdown()
    except Exception as e:
        logger.error(f"Server error: {e}")
        try:
            get_mcp_status_manager().mark_server_crashed()
        except Exception:
            pass
        if settings.mcp_enable_cleanup:
            _cleanup_on_shutdown()
        raise

if __name__ == "__main__":
    run_server()
