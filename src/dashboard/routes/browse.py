import platform
import string
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from ...config import config
from ..deps import require_admin

router = APIRouter(prefix="/api/browse", tags=["browse"], dependencies=[Depends(require_admin)])

_is_windows = platform.system() == "Windows"


@router.get("/directories")
async def browse_directories(path: str = ""):
    """Browse server directories. Empty path returns drives (Windows) or root dirs (Linux)."""
    if not path:
        if _is_windows:
            items = []
            for letter in string.ascii_uppercase:
                drive = Path(f"{letter}:/")
                if drive.exists():
                    items.append({"name": f"{letter}:\\", "path": f"{letter}:\\"})
            return {"path": "", "parent": None, "is_windows": True, "items": items}
        else:
            return _list_subdirs(Path("/"))

    resolved = Path(path).resolve()
    if not resolved.exists():
        raise HTTPException(status_code=404, detail="Path does not exist")
    if not resolved.is_dir():
        raise HTTPException(status_code=400, detail="Path is not a directory")

    return _list_subdirs(resolved)


def _list_subdirs(directory: Path) -> dict:
    items = []
    try:
        for entry in sorted(directory.iterdir()):
            if not entry.is_dir():
                continue
            if entry.name.startswith("."):
                continue
            try:
                # Check accessibility
                list(entry.iterdir())
                items.append({"name": entry.name, "path": str(entry)})
            except PermissionError:
                pass
    except PermissionError:
        pass

    # Compute parent
    parent_path = directory.parent
    if _is_windows:
        # Drive root (e.g. C:\) -> parent is null to show drive list
        if directory == parent_path:
            parent = None
        else:
            parent = str(parent_path)
    else:
        if str(directory) == "/":
            parent = None
        else:
            parent = str(parent_path)

    return {
        "path": str(directory),
        "parent": parent,
        "is_windows": _is_windows,
        "items": items,
    }


@router.get("/document-folders")
async def browse_document_folders(path: str = ""):
    """Browse folders and files inside DOCUMENTS_DIR for the INCLUDE_FOLDERS picker."""
    docs_dir = Path(config("DOCUMENTS_DIR")).resolve()
    if not docs_dir.is_dir():
        raise HTTPException(status_code=400, detail=f"DOCUMENTS_DIR does not exist: {docs_dir}")

    if path:
        target = (docs_dir / path).resolve()
        # Prevent path traversal
        if not str(target).startswith(str(docs_dir)):
            raise HTTPException(status_code=400, detail="Invalid path")
        if not target.is_dir():
            raise HTTPException(status_code=404, detail="Path does not exist")
    else:
        target = docs_dir

    items = []
    try:
        for entry in sorted(target.iterdir()):
            if entry.name.startswith("."):
                continue
            rel = str(entry.relative_to(docs_dir)).replace("\\", "/")
            try:
                if entry.is_dir():
                    items.append({"name": entry.name, "type": "directory", "relative_path": rel})
                elif entry.is_file():
                    items.append({"name": entry.name, "type": "file"})
            except PermissionError:
                pass
    except PermissionError:
        pass

    # Sort: directories first, then files
    items.sort(key=lambda x: (0 if x["type"] == "directory" else 1, x["name"].lower()))

    relative_path = str(target.relative_to(docs_dir)).replace("\\", "/") if target != docs_dir else ""
    if relative_path == ".":
        relative_path = ""

    parent = None
    if relative_path:
        p = Path(relative_path).parent
        parent = "" if str(p) == "." else str(p).replace("\\", "/")

    return {
        "base": str(docs_dir),
        "relative_path": relative_path,
        "parent": parent,
        "items": items,
    }
