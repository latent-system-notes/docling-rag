import atexit
import signal
import sys

from ..config import config, MCP_HOST, get_logger
from ..query import query as Query
from ..storage.postgres import list_documents
from ..models import QueryResult
from ..utils import cleanup_all_resources

logger = get_logger(__name__)


def _create_mcp():
    """Create and configure the FastMCP instance with tools."""
    from fastmcp import FastMCP

    server_name = config("MCP_SERVER_NAME")
    mcp = FastMCP(name=server_name, instructions=config("MCP_INSTRUCTIONS"))

    @mcp.tool(name="search_documents", description=config("MCP_TOOL_QUERY_DESC"))
    async def search_documents(query: str, max_results: int = 5, groups: list[str] | None = None) -> QueryResult:
        return Query(query, max_results, groups=groups)

    @mcp.tool(name="list_all_documents", description=config("MCP_TOOL_LIST_DOCS_DESC"))
    async def list_all_documents(limit: int | None = 50, offset: int = 0, groups: list[str] | None = None) -> dict:
        from ..storage.postgres import get_document_count
        docs = list_documents(limit=limit, offset=offset, groups=groups)
        return {"documents": docs, "total": get_document_count(), "showing": len(docs), "offset": offset}

    return mcp


def create_combined_app():
    """Create a single FastAPI app that serves:
    - /mcp       → MCP streamable-http endpoint
    - /api/*     → Dashboard REST API routes
    - /*         → Static frontend (SPA fallback)
    """
    from ..dashboard.app import create_app
    from ..storage.postgres import ensure_settings_table
    from ..config import load_settings_from_db

    # Ensure settings table exists and load overrides before MCP is created
    try:
        ensure_settings_table()
        load_settings_from_db()
    except Exception as e:
        logger.warning(f"Could not initialize settings: {e}")

    app = create_app()

    # Mount MCP as a sub-application at /mcp
    mcp = _create_mcp()
    mcp_asgi = mcp.http_app(path="/mcp", transport="streamable-http")
    app.mount("/mcp", mcp_asgi)

    return app


def reload_mcp(app):
    """Hot-reload MCP: refresh settings from DB, recreate MCP instance, remount."""
    from ..config import load_settings_from_db

    load_settings_from_db()
    app.routes[:] = [r for r in app.routes if not (hasattr(r, 'path') and r.path == '/mcp')]
    mcp = _create_mcp()
    mcp_asgi = mcp.http_app(path="/mcp", transport="streamable-http")
    app.mount("/mcp", mcp_asgi)
    logger.info("MCP instance reloaded with updated settings")


def run_server():
    """Run the combined server (MCP + Dashboard + Frontend) on a single port."""
    import uvicorn

    port = config("MCP_PORT")
    app = create_combined_app()

    atexit.register(cleanup_all_resources)
    _signals = [signal.SIGINT]
    if sys.platform != "win32":
        _signals.append(signal.SIGTERM)
    for sig in _signals:
        signal.signal(sig, lambda *_: sys.exit(0))

    logger.info(f"Starting server on http://{MCP_HOST}:{port}")
    logger.info(f"  MCP endpoint: http://{MCP_HOST}:{port}/mcp")
    logger.info(f"  API endpoint: http://{MCP_HOST}:{port}/api")
    logger.info(f"  Dashboard:    http://{MCP_HOST}:{port}/")

    try:
        uvicorn.run(app, host=MCP_HOST, port=port)
    except KeyboardInterrupt:
        pass
