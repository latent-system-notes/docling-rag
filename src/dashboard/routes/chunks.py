from fastapi import APIRouter, Depends, Query

from ...storage.postgres import list_chunks, count_chunks
from ..deps import get_current_user

router = APIRouter(prefix="/api/chunks", tags=["chunks"])


@router.get("")
async def get_chunks(
    q: str = Query("", alias="q"),
    doc_id: str = Query(""),
    file_name: str = Query(""),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    user: dict = Depends(get_current_user),
):
    groups = None if user.get("is_admin") else (user.get("groups") or [])
    search = q.strip() or None
    did = doc_id.strip() or None
    fname = file_name.strip() or None
    offset = (page - 1) * page_size
    items = list_chunks(search=search, doc_id=did, file_name=fname, limit=page_size, offset=offset, groups=groups)
    total = count_chunks(search=search, doc_id=did, file_name=fname, groups=groups)
    return {"items": items, "total": total, "page": page, "page_size": page_size}
