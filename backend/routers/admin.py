import os
import logging
from datetime import datetime, timezone
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Annotated, Optional
from core.auth import (
    generate_api_key,
    get_supabase,
    get_redis,
    PLAN_LIMITS,
    get_default_limit,
    verify_auth0_admin_bearer,
)
from admin_ui import ADMIN_UI

load_dotenv()
logger = logging.getLogger("deeptrack.admin")
router = APIRouter()

ADMIN_SECRET = os.getenv("ADMIN_SECRET")


def _check_admin(
    request: Request,
    authorization: Annotated[Optional[str], Header()] = None,
    x_admin_secret: Annotated[Optional[str], Header()] = None,
):
    """
    Auth0 bearer token takes priority in all environments.
    Falls back to X-Admin-Secret for local scripting/tests only.
    """
    if authorization and authorization.lower().startswith("bearer "):
        claims = verify_auth0_admin_bearer(authorization)
        request.state.admin_claims = claims
        return

    # fallback: X-Admin-Secret for local curl/scripts
    if not ADMIN_SECRET:
        raise HTTPException(status_code=500, detail="Missing DEEPTRACK_ADMIN_SECRET environment variable")
    if x_admin_secret != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden")


class CreateKeyRequest(BaseModel):
    owner:         str
    user_id:       str
    track:         str = "api"
    plan:          str = "payg"
    monthly_limit: Optional[int] = None
    notes:         str = ""


class UpdateLimitRequest(BaseModel):
    monthly_limit: int


class UpdatePlanRequest(BaseModel):
    track: str
    plan:  str
    monthly_limit: Optional[int] = None


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def admin_dashboard():
    return ADMIN_UI


@router.get("/plans", summary="List all plans with default limits")
def list_plans(_=Depends(_check_admin)):
    return {"plans": PLAN_LIMITS}


@router.post("/keys", summary="Create API key for a client")
def create_key(body: CreateKeyRequest, _=Depends(_check_admin)):
    key = generate_api_key(
        owner         = body.owner,
        user_id       = body.user_id,
        track         = body.track,
        plan          = body.plan,
        monthly_limit = body.monthly_limit,
        notes         = body.notes,
    )
    limit = body.monthly_limit or get_default_limit(body.track, body.plan)
    return {
        "key":           key,
        "owner":         body.owner,
        "track":         body.track,
        "plan":          body.plan,
        "monthly_limit": limit,
    }


@router.get("/keys", summary="List all API keys")
def list_keys(_=Depends(_check_admin)):
    db  = get_supabase()
    res = (
        db.table("api_details")
        .select("id, owner, user_id, api_key, track, plan, monthly_limit, is_active, notes, created_at")
        .order("created_at", desc=True)
        .execute()
    )
    return {"keys": res.data or [], "total": len(res.data or [])}


@router.patch("/keys/{key_id}/limit", summary="Override monthly limit for a key")
def update_limit(key_id: str, body: UpdateLimitRequest, _=Depends(_check_admin)):
    db = get_supabase()
    db.table("api_details").update({"monthly_limit": body.monthly_limit}).eq("id", key_id).execute()
    return {"key_id": key_id, "monthly_limit": body.monthly_limit}


@router.patch("/keys/{key_id}/plan", summary="Change plan (resets limit to plan default unless overridden)")
def update_plan(key_id: str, body: UpdatePlanRequest, _=Depends(_check_admin)):
    if body.track not in PLAN_LIMITS:
        raise HTTPException(400, f"Invalid track. Must be: {list(PLAN_LIMITS.keys())}")
    if body.plan not in PLAN_LIMITS[body.track]:
        raise HTTPException(400, f"Invalid plan. Must be: {list(PLAN_LIMITS[body.track].keys())}")

    limit = body.monthly_limit or get_default_limit(body.track, body.plan)
    db    = get_supabase()
    db.table("api_details").update({
        "track":         body.track,
        "plan":          body.plan,
        "monthly_limit": limit,
    }).eq("id", key_id).execute()
    return {"key_id": key_id, "track": body.track, "plan": body.plan, "monthly_limit": limit}


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


@router.get("/usage/month", summary="This month's scan count for every key")
def usage_month(_=Depends(_check_admin)):
    r = get_redis()
    if not r:
        return {"usage": {}, "note": "Redis not connected"}

    month = datetime.now(timezone.utc).strftime("%Y-%m")
    keys  = r.keys(f"ratelimit:*:{month}")
    usage = {}
    for redis_key in keys:
        parts  = redis_key.split(":")
        key_id = parts[1] if len(parts) == 3 else None
        if key_id:
            val           = r.get(redis_key)
            usage[key_id] = int(val) if val else 0

    return {"usage": usage, "month": month}