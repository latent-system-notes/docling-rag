from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel

from ...config import config
from ...storage.postgres import (
    add_path_permission, remove_path_permission, list_path_permissions,
    get_document_permissions, list_documents, refresh_all_document_permissions,
    _normalize_path,
)
from ..deps import require_admin, get_current_user

router = APIRouter(prefix="/api/permissions", tags=["permissions"])


class PathPermissionRequest(BaseModel):
    path: str
    group_id: int


# --- Path permissions (admin only) ---

@router.get("/paths", dependencies=[Depends(require_admin)])
async def list_all_path_permissions():
    return list_path_permissions()


@router.post("/paths", status_code=status.HTTP_201_CREATED, dependencies=[Depends(require_admin)])
async def add_new_path_permission(req: PathPermissionRequest):
    return add_path_permission(req.path, req.group_id)


@router.delete("/paths", dependencies=[Depends(require_admin)])
async def remove_existing_path_permission(req: PathPermissionRequest):
    if not remove_path_permission(req.path, req.group_id):
        raise HTTPException(status_code=404, detail="Path permission not found")
    return {"ok": True}


@router.post("/refresh", dependencies=[Depends(require_admin)])
async def refresh_permissions():
    """Recompute document_permissions for all documents based on current path_permissions."""
    count = refresh_all_document_permissions()
    return {"refreshed": count}


# --- Lazy tree: returns only immediate children of a given path ---

@router.get("/tree/children", dependencies=[Depends(require_admin)])
async def get_tree_children(path: str = Query("", alias="path")):
    """Return immediate children of a directory path.

    If path is empty, returns the root node (DOCUMENTS_DIR) with its immediate children.
    Otherwise, returns the children of the specified path.
    """
    docs_dir = config("DOCUMENTS_DIR")
    docs_dir_str = str(docs_dir).replace("\\", "/").rstrip("/")

    # Build permission lookup (only needs paths that start with the requested prefix)
    path_perms = list_path_permissions()
    perm_map: dict[str, list[dict]] = {}
    for pp in path_perms:
        normalized = pp["path"].replace("\\", "/").rstrip("/")
        perm_map.setdefault(normalized, []).append(
            {"group_id": pp["group_id"], "group_name": pp["group_name"]}
        )

    requested = path.replace("\\", "/").rstrip("/") if path else ""

    if not requested:
        # Return the root node with its immediate children
        fs_root = Path(docs_dir)
        if not fs_root.exists():
            return {
                "name": docs_dir_str,
                "path": docs_dir_str,
                "type": "directory",
                "groups": perm_map.get(docs_dir_str, []),
                "children": [],
            }

        children = _list_children(fs_root, docs_dir_str, perm_map)
        return {
            "name": fs_root.name or docs_dir_str,
            "path": docs_dir_str,
            "type": "directory",
            "groups": perm_map.get(docs_dir_str, []),
            "children": children,
        }
    else:
        # Resolve the requested path to filesystem
        # The requested path is like "docs/test/subfolder"
        # We need to find the actual filesystem path
        fs_path = _resolve_fs_path(docs_dir_str, requested)
        if not fs_path or not fs_path.is_dir():
            return []

        return _list_children(fs_path, requested, perm_map)


def _resolve_fs_path(docs_dir_str: str, requested: str) -> Path | None:
    """Resolve a normalized tree path back to a filesystem Path."""
    docs_dir = Path(docs_dir_str.replace("/", "\\") if "\\" in str(Path(".")) else docs_dir_str)

    if requested == docs_dir_str:
        return docs_dir

    # requested might be "docs/test/subfolder", docs_dir_str might be "docs/test"
    # so the relative part is "subfolder"
    if requested.startswith(docs_dir_str + "/"):
        relative = requested[len(docs_dir_str) + 1:]
        candidate = docs_dir / relative.replace("/", "\\") if "\\" in str(Path(".")) else docs_dir / relative
        if candidate.exists():
            return candidate

    # Fallback: try as-is
    candidate = Path(requested)
    if candidate.exists():
        return candidate

    return None


def _list_children(fs_dir: Path, parent_path: str, perm_map: dict) -> list[dict]:
    """List immediate children of a filesystem directory as tree nodes."""
    children = []
    try:
        for child in sorted(fs_dir.iterdir()):
            if child.name.startswith("."):
                continue
            child_path = parent_path + "/" + child.name
            is_dir = child.is_dir()
            node = {
                "name": child.name,
                "path": child_path,
                "type": "directory" if is_dir else "file",
                "groups": perm_map.get(child_path, []),
            }
            if is_dir:
                # Don't include children — they'll be loaded lazily
                # But indicate it has children so the UI can show an expand arrow
                try:
                    has_children = any(True for _ in child.iterdir())
                except PermissionError:
                    has_children = False
                node["has_children"] = has_children
            children.append(node)
    except PermissionError:
        pass
    return children


# --- Document permissions (read-only for non-admin) ---

@router.get("/documents/{doc_id}")
async def get_doc_permissions(doc_id: str, user: dict = Depends(get_current_user)):
    return get_document_permissions(doc_id)
