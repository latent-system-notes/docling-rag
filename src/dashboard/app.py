from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from .routes import auth, users, groups, permissions, search, settings, ingestion

# React build output directory (frontend/dist after `npm run build`)
FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"


def create_app() -> FastAPI:
    app = FastAPI(title="Docling RAG Dashboard", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API routes — all under /api
    app.include_router(auth.router)
    app.include_router(users.router)
    app.include_router(groups.router)
    app.include_router(permissions.router)
    app.include_router(search.router)
    app.include_router(settings.router)
    app.include_router(ingestion.router)

    # Serve React static build if available
    if FRONTEND_DIR.is_dir():
        # Mount static assets (JS, CSS, images) — but NOT at root to avoid catching SPA routes
        assets_dir = FRONTEND_DIR / "assets"
        if assets_dir.is_dir():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

        # Serve other static files (favicon, etc.) and SPA fallback
        # Paths starting with /api or /mcp are handled by their own routers/mounts
        @app.get("/{full_path:path}")
        async def serve_spa(request: Request, full_path: str):
            if full_path.startswith(("api/", "api", "mcp/", "mcp")):
                return HTMLResponse("Not found", status_code=404)
            # Try to serve the exact file first (e.g. favicon.ico, robots.txt)
            file_path = FRONTEND_DIR / full_path
            if full_path and file_path.is_file() and ".." not in full_path:
                return FileResponse(file_path)
            # SPA fallback: always return index.html for client-side routing
            index = FRONTEND_DIR / "index.html"
            if index.is_file():
                return FileResponse(index)
            return HTMLResponse("<h1>Frontend not built</h1><p>Run <code>cd frontend && npm run build</code></p>", status_code=503)
    else:
        @app.get("/")
        async def no_frontend():
            return HTMLResponse("<h1>Frontend not built</h1><p>Run <code>cd frontend && npm run build</code></p>", status_code=503)

    return app
