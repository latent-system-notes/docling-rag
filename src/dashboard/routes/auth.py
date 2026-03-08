from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from ...auth.auth import authenticate, create_token, hash_password, verify_password
from ...storage.postgres import get_user_by_username, update_user
from ..deps import get_current_user, Depends

router = APIRouter(prefix="/api/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str
    password: str


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict


@router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest):
    user = authenticate(req.username, req.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    token = create_token(user)
    safe_user = {k: v for k, v in user.items() if k != "password_hash"}
    return TokenResponse(access_token=token, user=safe_user)


@router.post("/change-password")
async def change_password(req: ChangePasswordRequest, current_user: dict = Depends(get_current_user)):
    user = get_user_by_username(current_user["sub"])
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if not verify_password(req.current_password, user["password_hash"]):
        raise HTTPException(status_code=400, detail="Current password is incorrect")
    if len(req.new_password) < 6:
        raise HTTPException(status_code=400, detail="New password must be at least 6 characters")
    update_user(user["id"], password_hash=hash_password(req.new_password), must_change_password=False)
    return {"ok": True, "message": "Password changed successfully"}


@router.get("/me")
async def me(user: dict = Depends(get_current_user)):
    return user
