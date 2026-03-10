from pathlib import Path

from fastapi import APIRouter, Depends, Query

from ...query import query as query_fn
from ...storage.postgres import list_documents
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
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    user: dict = Depends(get_current_user),
):
    """List documents the current user can access. Admin sees everything."""
    groups = None if user.get("is_admin") else (user.get("groups") or [])
    docs = list_documents(limit=limit, offset=offset, groups=groups)
    return {"documents": docs, "showing": len(docs), "offset": offset}
