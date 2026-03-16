import os
import logging
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from core.auth import generate_api_key, get_supabase
from admin_ui import ADMIN_UI

load_dotenv()
logger = logging.getLogger("deeptrack.admin")
router = APIRouter()

ADMIN_SECRET = os.environ.get("ADMIN_SECRET")

def _check_admin(x_admin_secret: str = Header(...)):
    if x_admin_secret != ADMIN_SECRET:
        raise HTTPException(403, "Forbidden")

class CreateKeyRequest(BaseModel):
    owner:       str
    user_id:     str = "00000000-0000-0000-0000-000000000000"
    daily_limit: int = 10
    notes:       str = ""

class UpdateLimitRequest(BaseModel):
    daily_limit: int

@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def admin_dashboard():
    """
    Admin dashboard — open in browser, enter ADMIN_SECRET to authenticate.
    No server-side session — the secret is used as a header on every API call.
    """
    return ADMIN_UI

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

@router.post("/keys", summary="Create API key for a client")
def create_key(body: CreateKeyRequest, _=Depends(_check_admin)):
    """Generate a new API key. Raw key is only returned once — store it."""
    key = generate_api_key(
        owner       = body.owner,
        user_id     = body.user_id,
        daily_limit = body.daily_limit,
        notes       = body.notes,
    )
    return {
        "key":         key,
        "owner":       body.owner,
        "daily_limit": body.daily_limit,
    }


@router.get("/keys", summary="List all API keys")
def list_keys(_=Depends(_check_admin)):
    db  = get_supabase()
    res = (
        db.table("api_details")
        .select("id, owner, user_id, api_key, daily_limit, is_active, notes, created_at")
        .order("created_at", desc=True)
        .execute()
    )
    return {"keys": res.data, "total": len(res.data)}


@router.patch("/keys/{key_id}/limit", summary="Update daily limit")
def update_limit(key_id: str, body: UpdateLimitRequest, _=Depends(_check_admin)):
    db = get_supabase()
    db.table("api_details").update({"daily_limit": body.daily_limit}).eq("id", key_id).execute()
    return {"key_id": key_id, "daily_limit": body.daily_limit}


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
        .select("id, owner, user_id, daily_limit, is_active, notes, created_at")
        .eq("id", key_id)
        .single()
        .execute()
    )
    if not res.data:
        raise HTTPException(404, "Key not found")
    return res.data