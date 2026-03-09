import os

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from ...config import config, SETTINGS_REGISTRY
from ...storage.postgres import get_setting, list_settings, upsert_setting, delete_setting
from ..deps import require_admin

router = APIRouter(prefix="/api/settings", tags=["settings"], dependencies=[Depends(require_admin)])


class UpdateSettingRequest(BaseModel):
    value: str


@router.get("")
async def get_all_settings():
    """Return all registered settings with their effective values and metadata."""
    db_overrides = {s["key"]: s for s in list_settings()}
    result = []
    for key, meta in SETTINGS_REGISTRY.items():
        db_row = db_overrides.get(key)
        effective_value = str(config(key)) if config(key) is not None else ""
        result.append({
            "key": key,
            "value": effective_value,
            "has_override": db_row is not None,
            "updated_at": db_row["updated_at"] if db_row else None,
            "updated_by": db_row["updated_by"] if db_row else None,
            **meta,
        })
    return result


@router.put("/{key}")
async def update_setting(key: str, body: UpdateSettingRequest, user: dict = Depends(require_admin)):
    """Upsert a setting value. Only keys in SETTINGS_REGISTRY are allowed."""
    if key not in SETTINGS_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Setting '{key}' is not in the allowed registry")

    meta = SETTINGS_REGISTRY[key]
    username = user.get("sub", "admin")
    row = upsert_setting(key, body.value, updated_by=username)

    # Always update os.environ so config() / reload picks up the value
    os.environ[key] = body.value

    return {"ok": True, "restart_required": meta.get("restart_required", False), "reload_mcp": meta.get("reload_mcp", False), "setting": row}


@router.delete("/{key}")
async def remove_setting(key: str, user: dict = Depends(require_admin)):
    """Remove a DB override, reverting to env/default."""
    if key not in SETTINGS_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Setting '{key}' is not in the allowed registry")

    deleted = delete_setting(key)
    if not deleted:
        raise HTTPException(status_code=404, detail="No override found for this setting")

    # Remove from os.environ so config() falls back to default
    meta = SETTINGS_REGISTRY[key]
    os.environ.pop(key, None)

    return {"ok": True, "restart_required": meta.get("restart_required", False), "reload_mcp": meta.get("reload_mcp", False)}


@router.post("/reload-mcp")
async def reload_mcp_endpoint(request: Request, user: dict = Depends(require_admin)):
    """Reload the MCP instance with current settings."""
    from ...mcp.server import reload_mcp
    try:
        reload_mcp(request.app)
        return {"ok": True, "message": "MCP reloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload MCP: {e}")
