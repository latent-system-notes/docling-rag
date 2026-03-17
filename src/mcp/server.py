import atexit
import signal
import sys
import threading
import time

from ..config import config, MCP_HOST, get_logger
from ..query import query as Query
from ..storage.postgres import list_documents
from ..models import QueryResult
from ..utils import cleanup_all_resources

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# User group cache — avoids DB lookups on every MCP request
# ---------------------------------------------------------------------------

_user_cache: dict[str, tuple[list[str] | None, float]] = {}  # username → (groups, timestamp)
_user_cache_lock = threading.Lock()
_USER_CACHE_TTL = 300  # 5 minutes


def _resolve_user_from_headers() -> tuple[list[str] | None, str | None]:
    """Extract user email from MCP request headers and resolve their groups.

    Looks for X-OpenWebUI-User-Email header, extracts the username,
    looks up or auto-creates the user, and returns their groups.

    Uses a TTL cache to avoid hitting the DB on every MCP request.

    Returns:
        (groups, username) — groups is None if user is admin,
        list[str] for regular users (may be empty).
        Returns ([], None) on error to deny access to restricted docs.
    """
    from fastmcp.server.dependencies import get_http_request

    try:
        request = get_http_request()
    except RuntimeError:
        return None, None

    # Header may be sent as X-OpenWebUI-User-Email (case-insensitive)
    email = request.headers.get("x-openwebui-user-email")
    if not email or not email.strip():
        return None, None

    email = email.strip()
    username = email.split("@")[0].lower()
    if not username:
        return None, None

    # Check cache first
    now = time.monotonic()
    with _user_cache_lock:
        cached = _user_cache.get(username)
        if cached is not None:
            groups, ts = cached
            if now - ts < _USER_CACHE_TTL:
                return groups, username
            # Expired — remove stale entry
            del _user_cache[username]

    # Cache miss — resolve from DB
    try:
        from ..storage.postgres import (
            get_user_by_username, create_user, get_user_group_names,
        )

        user = get_user_by_username(username)

        if user is None:
            # Auto-create user so admin can assign groups later
            logger.info(f"Auto-creating user '{username}' from MCP header (email: {email})")
            user = create_user(
                username=username,
                password_hash=None,
                display_name=username,
                email=email,
                is_admin=False,
                auth_type="mcp",
                must_change_password=False,
            )

        if not user.get("is_active", True):
            logger.warning(f"MCP request from inactive user '{username}', treating as restricted")
            with _user_cache_lock:
                _user_cache[username] = ([], now)
            return [], username

        # Admin bypasses all permission filtering
        if user.get("is_admin"):
            with _user_cache_lock:
                _user_cache[username] = (None, now)
            return None, username

        groups = get_user_group_names(user["id"]) or []
        with _user_cache_lock:
            _user_cache[username] = (groups, now)
        return groups, username

    except Exception as e:
        logger.warning(f"Failed to resolve MCP user '{username}': {e}")
        # On error, return empty groups (public docs only) — never fall back to client param
        return [], None


def _create_mcp():
    """Create and configure the FastMCP instance with tools."""
    from fastmcp import FastMCP

    server_name = config("MCP_SERVER_NAME")
    mcp = FastMCP(name=server_name, instructions=config("MCP_INSTRUCTIONS"))

    @mcp.tool(name="search_documents", description=config("MCP_TOOL_QUERY_DESC"))
    async def search_documents(query: str, max_results: int = 5, groups: list[str] | None = None) -> QueryResult:
        resolved_groups, username = _resolve_user_from_headers()
        # If header present, always use resolved groups (never trust client param)
        # If no header, fall back to explicit groups param
        effective_groups = resolved_groups if username is not None else groups
        if username:
            logger.info(f"MCP search by '{username}', groups={effective_groups}")
        return Query(query, max_results, groups=effective_groups)

    @mcp.tool(name="list_all_documents", description=config("MCP_TOOL_LIST_DOCS_DESC"))
    async def list_all_documents(limit: int | None = 50, offset: int = 0, groups: list[str] | None = None) -> dict:
        from ..storage.postgres import get_document_count
        resolved_groups, username = _resolve_user_from_headers()
        effective_groups = resolved_groups if username is not None else groups
        if username:
            logger.info(f"MCP list_docs by '{username}', groups={effective_groups}")
        docs = list_documents(limit=limit, offset=offset, groups=effective_groups)
        return {"documents": docs, "total": get_document_count(), "showing": len(docs), "offset": offset}

    return mcp


def _create_mcp_asgi():
    """Create the MCP ASGI sub-application."""
    mcp = _create_mcp()
    return mcp.http_app(path="/", transport="streamable-http")


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

    # Create MCP ASGI app and pass it to create_app so it's mounted
    # BEFORE the SPA catch-all route (route order matters in Starlette)
    mcp_asgi = _create_mcp_asgi()
    app = create_app(mcp_app=mcp_asgi)

    return app


def reload_mcp(app):
    """Hot-reload MCP: refresh settings from DB, recreate MCP instance, remount."""
    from ..config import load_settings_from_db

    load_settings_from_db()
    app.routes[:] = [r for r in app.routes if not (hasattr(r, 'path') and r.path == '/mcp')]

    # Re-create and re-mount MCP; insert before the SPA catch-all (last route)
    mcp_asgi = _create_mcp_asgi()
    app.mount("/mcp", mcp_asgi)
    mcp_route = app.routes.pop()
    # Insert before the last route (SPA catch-all)
    app.routes.insert(-1, mcp_route)

    # Clear user cache so new MCP instance picks up fresh groups
    with _user_cache_lock:
        _user_cache.clear()

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
