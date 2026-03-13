import os
import uuid
import logging
from dotenv import load_dotenv
from fastapi import Request, HTTPException
from supabase import create_client, Client

load_dotenv()
logger = logging.getLogger("deeptrack.auth")

supabase_client: Client = None

def get_supabase():
    """
    creates a supabase client instance
    """
    global supabase_client
    if supabase_client is None:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_KEY")
        if not url or not key:
            raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set.")
        supabase_client = create_client(url, key)
    return supabase_client


def generate_api_key(owner: str, user_id: str, daily_limit: int = 10, notes: str = "") -> str:
    """
    Generate and store a new API key for a user.
    Returns the raw key, only shown once.
    """
    raw_key = "dt_" + uuid.uuid4().hex + uuid.uuid4().hex
    db = get_supabase()
    db.table("api_details").insert({
        "user_id":     user_id,
        "api_key":     raw_key,
        "daily_limit": daily_limit,
        "is_active":   True,
        "owner":       owner,
        "notes":       notes,
    }).execute()
    logger.info(f"Generated API key for owner='{owner}' limit={daily_limit}/day")
    return raw_key


async def verify_api_key(request: Request) -> dict:
    """
    FastAPI dependency — validates key and attaches api_limit to request.state.
    SlowAPI reads request.state.api_limit to apply the per-key rate limit.
    """
    raw_key = request.headers.get("X-API-Key")
    if not raw_key:
        raise HTTPException(
            status_code=401,
            detail="Missing X-API-Key header."
        )

    db = get_supabase()

    try:
        res = (
            db.table("api_details")
            .select("id, user_id, owner, daily_limit, is_active")
            .eq("api_key", raw_key)
            .single()
            .execute()
        )
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid API key.")

    if not res.data or not res.data["is_active"]:
        raise HTTPException(status_code=401, detail="Invalid or inactive API key.")

    # Attach limit to request state — SlowAPI callable reads this
    request.state.api_key_row = res.data
    request.state.api_limit   = res.data["daily_limit"]

    return res.data