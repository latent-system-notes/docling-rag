import re

from fastapi import APIRouter, HTTPException, Query, status, Depends
from pydantic import BaseModel

from ...storage.postgres import create_group, list_groups, count_groups, get_group, update_group, delete_group
from ..deps import require_admin

router = APIRouter(prefix="/api/groups", tags=["groups"], dependencies=[Depends(require_admin)])

KEBAB_RE = re.compile(r'^[a-z0-9]+(-[a-z0-9]+)*$')


def _to_kebab(name: str) -> str:
    name = name.lower().strip()
    name = re.sub(r'\s+', '-', name)
    name = re.sub(r'[^a-z0-9-]', '', name)
    name = re.sub(r'-{2,}', '-', name)
    name = name.strip('-')
    return name


def _validate_group_name(name: str) -> str:
    name = _to_kebab(name)
    if not name:
        raise HTTPException(status_code=400, detail="Group name is required")
    if not KEBAB_RE.match(name):
        raise HTTPException(status_code=400, detail="Group name must be kebab-case (lowercase letters, numbers, hyphens)")
    return name


class CreateGroupRequest(BaseModel):
    name: str
    description: str = ""


class UpdateGroupRequest(BaseModel):
    name: str | None = None
    description: str | None = None


@router.get("")
async def list_all_groups(search: str = Query("", alias="q"), page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100)):
    s = search.strip() or None
    offset = (page - 1) * page_size
    items = list_groups(search=s, limit=page_size, offset=offset)
    total = count_groups(search=s)
    return {"items": items, "total": total, "page": page, "page_size": page_size}


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_new_group(req: CreateGroupRequest):
    name = _validate_group_name(req.name)
    return create_group(name, req.description)


@router.get("/{group_id}")
async def get_single_group(group_id: int):
    group = get_group(group_id)
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")
    return group


@router.patch("/{group_id}")
async def update_existing_group(group_id: int, req: UpdateGroupRequest):
    name = _validate_group_name(req.name) if req.name is not None else None
    return update_group(group_id, name, req.description)


@router.delete("/{group_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_existing_group(group_id: int):
    if not delete_group(group_id):
        raise HTTPException(status_code=404, detail="Group not found")
