import re

from fastapi import APIRouter, HTTPException, Query, status, Depends
from pydantic import BaseModel

from ...auth.auth import hash_password
from ...storage.postgres import (
    create_user, list_users, count_users, get_user, update_user, delete_user,
    assign_user_to_group, remove_user_from_group, get_user_groups,
)
from ..deps import require_admin

router = APIRouter(prefix="/api/users", tags=["users"], dependencies=[Depends(require_admin)])

USERNAME_RE = re.compile(r'^[a-z0-9_-]+$')


def _validate_username(username: str) -> str:
    username = username.lower().strip()
    if not username or not USERNAME_RE.match(username):
        raise HTTPException(status_code=400, detail="Username may only contain lowercase letters, numbers, dash, and underscore")
    return username


class ResetPasswordRequest(BaseModel):
    new_password: str


class CreateUserRequest(BaseModel):
    username: str
    password: str | None = None
    display_name: str = ""
    email: str = ""
    is_admin: bool = False
    auth_type: str = "local"


class UpdateUserRequest(BaseModel):
    display_name: str | None = None
    email: str | None = None
    is_admin: bool | None = None
    is_active: bool | None = None
    auth_type: str | None = None


class AssignGroupRequest(BaseModel):
    group_id: int


@router.get("")
async def list_all_users(search: str = Query("", alias="q"), page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100)):
    s = search.strip() or None
    offset = (page - 1) * page_size
    items = list_users(search=s, limit=page_size, offset=offset)
    total = count_users(search=s)
    return {"items": items, "total": total, "page": page, "page_size": page_size}


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_new_user(req: CreateUserRequest):
    username = _validate_username(req.username)
    pw_hash = hash_password(req.password) if req.password else None
    if req.auth_type == "local" and not pw_hash:
        raise HTTPException(status_code=400, detail="Password required for local users")
    return create_user(username, pw_hash, req.display_name, req.email, req.is_admin, req.auth_type)


@router.get("/{user_id}")
async def get_single_user(user_id: int):
    user = get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@router.patch("/{user_id}")
async def update_existing_user(user_id: int, req: UpdateUserRequest):
    kwargs = req.model_dump(exclude_none=True)
    if "password" in kwargs:
        kwargs["password_hash"] = hash_password(kwargs.pop("password"))
    return update_user(user_id, **kwargs)


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_existing_user(user_id: int):
    if not delete_user(user_id):
        raise HTTPException(status_code=404, detail="User not found")


@router.post("/{user_id}/reset-password")
async def reset_user_password(user_id: int, req: ResetPasswordRequest):
    user = get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if len(req.new_password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
    update_user(user_id, password_hash=hash_password(req.new_password), must_change_password=True)
    return {"ok": True, "message": f"Password reset for {user['username']}. User will be prompted to change on next login."}


@router.get("/{user_id}/groups")
async def get_user_group_list(user_id: int):
    return get_user_groups(user_id)


@router.post("/{user_id}/groups", status_code=status.HTTP_201_CREATED)
async def assign_group(user_id: int, req: AssignGroupRequest):
    assign_user_to_group(user_id, req.group_id)
    return {"ok": True}


@router.delete("/{user_id}/groups/{group_id}", status_code=status.HTTP_204_NO_CONTENT)
async def unassign_group(user_id: int, group_id: int):
    remove_user_from_group(user_id, group_id)
