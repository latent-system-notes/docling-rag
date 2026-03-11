from pathlib import Path

from fastapi import APIRouter, Depends, Query

from ...query import query as query_fn
from ...storage.postgres import list_documents, count_documents
from ..deps import get_current_user

router = APIRouter(prefix="/api/search", tags=["search"])


@router.get("")
async def search_documents(
    q: str = Query(..., min_length=1),
    top_k: int = Query(10, ge=1, le=100),
    user: dict = Depends(get_current_user),
):
    """Search documents filtered by user's groups. Admin sees everything."""
    groups = None if user.get("is_admin") else (user.get("groups") or [])
    result = query_fn(q, top_k, groups=groups)
    return {
        "query": result.query,
        "total_results": len(result.context),
        "results": [
            {
                "rank": i,
                "score": round(r.score, 4),
                "text": r.chunk.text,
                "file": Path(r.chunk.metadata.get("file_path", "")).name,
                "file_path": r.chunk.metadata.get("file_path", ""),
                "page": r.chunk.page_num,
                "doc_type": r.chunk.metadata.get("doc_type", ""),
            }
            for i, r in enumerate(result.context, 1)
        ],
    }


@router.get("/documents")
async def list_accessible_documents(
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=500),
    user: dict = Depends(get_current_user),
):
    """List documents the current user can access. Admin sees everything."""
    groups = None if user.get("is_admin") else (user.get("groups") or [])
    offset = (page - 1) * page_size
    docs = list_documents(limit=page_size, offset=offset, groups=groups)
    total = count_documents(groups=groups)
    return {"items": docs, "total": total, "page": page, "page_size": page_size}
