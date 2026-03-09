import shutil
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File as FastAPIFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ...config import config, get_logger
from ...utils import SUPPORTED_EXTENSIONS
from ..deps import require_admin

router = APIRouter(prefix="/api/files", tags=["files"], dependencies=[Depends(require_admin)])
logger = get_logger(__name__)

MAX_UPLOAD_SIZE = 500 * 1024 * 1024  # 500 MB


def _docs_dir() -> Path:
    return Path(config("DOCUMENTS_DIR")).resolve()


def _safe_resolve(rel_path: str) -> Path:
    """Resolve a relative path inside DOCUMENTS_DIR, rejecting traversal attempts."""
    docs = _docs_dir()
    if not rel_path or rel_path in (".", "/"):
        return docs
    # Reject obvious traversal
    if ".." in rel_path.replace("\\", "/").split("/"):
        raise HTTPException(status_code=400, detail="Path traversal not allowed")
    resolved = (docs / rel_path).resolve()
    if not str(resolved).startswith(str(docs)):
        raise HTTPException(status_code=400, detail="Path traversal not allowed")
    return resolved


def _rel_path(abs_path: Path) -> str:
    """Return forward-slash relative path from DOCUMENTS_DIR."""
    return str(abs_path.relative_to(_docs_dir())).replace("\\", "/")


def _file_info(p: Path) -> dict:
    stat = p.stat()
    is_dir = p.is_dir()
    return {
        "name": p.name,
        "path": _rel_path(p),
        "type": "directory" if is_dir else "file",
        "size": 0 if is_dir else stat.st_size,
        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "extension": "" if is_dir else p.suffix.lower(),
    }


# ---------------------------------------------------------------------------
# List directory contents
# ---------------------------------------------------------------------------

@router.get("")
async def list_directory(path: str = Query("", description="Relative path inside documents dir")):
    target = _safe_resolve(path)
    if not target.exists():
        raise HTTPException(status_code=404, detail="Directory not found")
    if not target.is_dir():
        raise HTTPException(status_code=400, detail="Path is not a directory")

    items = []
    for child in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
        if child.name.startswith("."):
            continue
        items.append(_file_info(child))

    return {
        "path": path or "",
        "items": items,
    }


# ---------------------------------------------------------------------------
# Upload files
# ---------------------------------------------------------------------------

@router.post("/upload")
async def upload_files(
    path: str = Query("", description="Target directory (relative)"),
    files: list[UploadFile] = FastAPIFile(...),
):
    target_dir = _safe_resolve(path)
    if not target_dir.exists():
        target_dir.mkdir(parents=True, exist_ok=True)
    if not target_dir.is_dir():
        raise HTTPException(status_code=400, detail="Target path is not a directory")

    uploaded = []
    errors = []

    for f in files:
        ext = Path(f.filename).suffix.lower()
        if ext and ext not in SUPPORTED_EXTENSIONS:
            errors.append(f"{f.filename}: unsupported file type '{ext}'")
            continue

        dest = target_dir / f.filename
        try:
            content = await f.read()
            if len(content) > MAX_UPLOAD_SIZE:
                errors.append(f"{f.filename}: exceeds max upload size ({MAX_UPLOAD_SIZE // (1024*1024)} MB)")
                continue
            dest.write_bytes(content)
            uploaded.append(_rel_path(dest))
            logger.info(f"Uploaded: {_rel_path(dest)}")
        except Exception as e:
            errors.append(f"{f.filename}: {e}")
        finally:
            await f.close()

    return {
        "uploaded": uploaded,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Create directory
# ---------------------------------------------------------------------------

class MkdirRequest(BaseModel):
    path: str
    name: str


@router.post("/mkdir")
async def create_directory(body: MkdirRequest):
    parent = _safe_resolve(body.path)
    if not parent.is_dir():
        raise HTTPException(status_code=400, detail="Parent path is not a directory")

    name = body.name.strip()
    if not name or "/" in name or "\\" in name or ".." in name:
        raise HTTPException(status_code=400, detail="Invalid folder name")

    new_dir = parent / name
    if new_dir.exists():
        raise HTTPException(status_code=409, detail="Folder already exists")

    new_dir.mkdir(parents=False, exist_ok=False)
    logger.info(f"Created directory: {_rel_path(new_dir)}")
    return {"ok": True, "path": _rel_path(new_dir)}


# ---------------------------------------------------------------------------
# Delete file or directory
# ---------------------------------------------------------------------------

class DeleteRequest(BaseModel):
    path: str


@router.delete("")
async def delete_item(body: DeleteRequest):
    target = _safe_resolve(body.path)
    docs = _docs_dir()

    if target == docs:
        raise HTTPException(status_code=400, detail="Cannot delete the root documents directory")
    if not target.exists():
        raise HTTPException(status_code=404, detail="Path not found")

    name = target.name
    if target.is_dir():
        shutil.rmtree(target)
        logger.info(f"Deleted directory: {body.path}")
    else:
        target.unlink()
        logger.info(f"Deleted file: {body.path}")

    return {"ok": True, "deleted": body.path, "name": name}


# ---------------------------------------------------------------------------
# Rename / move
# ---------------------------------------------------------------------------

class RenameRequest(BaseModel):
    path: str
    new_name: str


@router.put("/rename")
async def rename_item(body: RenameRequest):
    target = _safe_resolve(body.path)
    if not target.exists():
        raise HTTPException(status_code=404, detail="Path not found")

    new_name = body.new_name.strip()
    if not new_name or "/" in new_name or "\\" in new_name or ".." in new_name:
        raise HTTPException(status_code=400, detail="Invalid name")

    new_path = target.parent / new_name
    if new_path.exists():
        raise HTTPException(status_code=409, detail="A file or folder with that name already exists")

    target.rename(new_path)
    logger.info(f"Renamed: {body.path} -> {_rel_path(new_path)}")
    return {"ok": True, "old_path": body.path, "new_path": _rel_path(new_path)}


# ---------------------------------------------------------------------------
# Download file
# ---------------------------------------------------------------------------

@router.get("/download")
async def download_file(path: str = Query(..., description="Relative path to file")):
    target = _safe_resolve(path)
    if not target.exists():
        raise HTTPException(status_code=404, detail="File not found")
    if not target.is_file():
        raise HTTPException(status_code=400, detail="Path is not a file")

    return FileResponse(
        path=str(target),
        filename=target.name,
        media_type="application/octet-stream",
    )
