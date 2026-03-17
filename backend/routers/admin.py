import os
import logging
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from core.auth import generate_api_key, get_supabase, get_redis
from admin_ui import ADMIN_UI

logger = logging.getLogger("deeptrack.admin")
router = APIRouter()

ADMIN_SECRET = os.environ.get("ADMIN_SECRET")

def _check_admin(x_admin_secret: str = Header(...)):
    if x_admin_secret != ADMIN_SECRET:
        raise HTTPException(403, "Forbidden")

class CreateKeyRequest(BaseModel):
    owner:         str
    user_id:       str
    monthly_limit: int = 100
    notes:         str = ""

class UpdateLimitRequest(BaseModel):
    monthly_limit: int

@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def admin_dashboard():
    return ADMIN_UI

@router.post("/keys", summary="Create API key for a client")
def create_key(body: CreateKeyRequest, _=Depends(_check_admin)):
    key = generate_api_key(
        owner         = body.owner,
        user_id       = body.user_id,
        monthly_limit = body.monthly_limit,
        notes         = body.notes,
    )
    return {"key": key, "owner": body.owner, "monthly_limit": body.monthly_limit}


@router.get("/keys", summary="List all API keys")
def list_keys(_=Depends(_check_admin)):
    db  = get_supabase()
    res = (
        db.table("api_details")
        .select("id, owner, user_id, api_key, monthly_limit, is_active, notes, created_at")
        .order("created_at", desc=True)
        .execute()
    )
    return {"keys": res.data, "total": len(res.data)}


@router.patch("/keys/{key_id}/limit", summary="Update monthly limit")
def update_limit(key_id: str, body: UpdateLimitRequest, _=Depends(_check_admin)):
    db = get_supabase()
    db.table("api_details").update({"monthly_limit": body.monthly_limit}).eq("id", key_id).execute()
    return {"key_id": key_id, "monthly_limit": body.monthly_limit}


@router.patch("/keys/{key_id}/deactivate", summary="Deactivate a key")
def deactivate_key(key_id: str, _=Depends(_check_admin)):
    db = get_supabase()
    db.table("api_details").update({"is_active": False}).eq("id", key_id).execute()
    return {"key_id": key_id, "is_active": False}


@router.patch("/keys/{key_id}/activate", summary="Activate a key")
def activate_key(key_id: str, _=Depends(_check_admin)):
    db = get_supabase()
    db.table("api_details").update({"is_active": True}).eq("id", key_id).execute()
    return {"key_id": key_id, "is_active": True}


@router.get("/keys/{key_id}", summary="Get a single key")
def get_key(key_id: str, _=Depends(_check_admin)):
    db  = get_supabase()
    res = (
        db.table("api_details")
        .select("id, owner, user_id, monthly_limit, is_active, notes, created_at")
        .eq("id", key_id)
        .single()
        .execute()
    )
    if not res.data:
        raise HTTPException(404, "Key not found")
    return res.data


@router.get("/users", summary="List all users")
def list_users(_=Depends(_check_admin)):
    db  = get_supabase()
    res = (
        db.table("users")
        .select("id, name, email")
        .order("created_at", desc=True)
        .execute()
    )
    return {"users": res.data}


@router.get("/usage/month", summary="This month's usage count for every key")
def usage_month(_=Depends(_check_admin)):
    """Returns dict of key_id -> scans used this month."""
    r = get_redis()
    if not r:
        return {"usage": {}, "note": "Redis not connected"}

    month   = datetime.now(timezone.utc).strftime("%Y-%m")
    keys    = r.keys(f"ratelimit:*:{month}")
    usage   = {}
    for redis_key in keys:
        parts  = redis_key.split(":")
        key_id = parts[1] if len(parts) == 3 else None
        if key_id:
            val           = r.get(redis_key)
            usage[key_id] = int(val) if val else 0

    return {"usage": usage, "month": month}