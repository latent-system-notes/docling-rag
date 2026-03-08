import os
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt

from ..config import config, get_logger
from ..storage.postgres import get_user_by_username, get_user_group_names

logger = get_logger(__name__)

ALGORITHM = "HS256"
TOKEN_EXPIRE_HOURS = 24


def _get_secret() -> str:
    return config("DASHBOARD_SECRET") or os.environ.get("DASHBOARD_SECRET") or "change-me-in-production"


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode(), password_hash.encode())
    except Exception:
        return False


def authenticate(username: str, password: str) -> dict | None:
    """Authenticate via local DB or LDAP. Returns user dict or None."""
    user = get_user_by_username(username)

    if not user or not user.get("is_active", False):
        return None

    if user["auth_type"] == "local":
        if not user.get("password_hash") or not verify_password(password, user["password_hash"]):
            return None
        return user

    if user["auth_type"] == "ldap":
        from .ldap_client import ldap_authenticate
        if not ldap_authenticate(username, password):
            return None
        return user

    return None


def create_token(user: dict) -> str:
    groups = get_user_group_names(user["id"])
    payload = {
        "sub": user["username"],
        "user_id": user["id"],
        "is_admin": user["is_admin"],
        "groups": groups,
        "exp": datetime.now(timezone.utc) + timedelta(hours=TOKEN_EXPIRE_HOURS),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, _get_secret(), algorithm=ALGORITHM)


def decode_token(token: str) -> dict | None:
    try:
        return jwt.decode(token, _get_secret(), algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        logger.debug("Token expired")
        return None
    except jwt.InvalidTokenError:
        logger.debug("Invalid token")
        return None
