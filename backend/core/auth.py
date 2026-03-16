import os
import uuid
import logging
from datetime import datetime, date, timedelta, timezone
from dotenv import load_dotenv
from fastapi import Request, HTTPException
from supabase import create_client
import redis as redis_lib

load_dotenv()
logger = logging.getLogger("deeptrack.auth")
supabase_client= None
redis_client = None

def get_supabase():
    global supabase_client
    if supabase_client is None:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_KEY")
        if not url or not key:
            raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set.")
        supabase_client = create_client(url, key)
    return supabase_client


def get_redis():
    global redis_client
    if redis_client is None:
        url = os.getenv("REDIS_URL")
        if url:
            redis_client = redis_lib.from_url(url, decode_responses=True)
            logger.info("Redis client connected.")
        else:
            logger.warning("REDIS_URL not set — rate limiting disabled.")
    return redis_client

def _next_midnight() -> int:
    """Unix timestamp of next UTC midnight, used to expire Redis counters."""
    tomorrow = date.today() + timedelta(days=1)
    midnight = datetime(tomorrow.year, tomorrow.month, tomorrow.day, tzinfo=timezone.utc)
    return int(midnight.timestamp())

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
    """Validates key and checks limit. Does not increment the counter."""
    raw_key = request.headers.get("X-API-Key")
    if not raw_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header.")

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

    key_row     = res.data
    daily_limit = key_row["daily_limit"]

    # check current usage without incrementing
    r = get_redis()
    if r:
        today     = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        redis_key = f"ratelimit:{key_row['id']}:{today}"
        val       = r.get(redis_key)
        used      = int(val) if val else 0
        if used >= daily_limit:
            raise HTTPException(
                status_code=429,
                detail=f"Daily limit of {daily_limit} requests exceeded. "
                       f"Upgrade at deeptrack.io/pricing"
            )

    request.state.api_key_row = key_row
    request.state.api_limit   = daily_limit
    return key_row


def increment_usage(request: Request):
    """
    Increments the Redis counter for this key.
    Call this only after a successful prediction.
    """
    r = get_redis()
    if not r:
        return
    key_row   = request.state.api_key_row
    today     = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    redis_key = f"ratelimit:{key_row['id']}:{today}"
    used      = r.incr(redis_key)
    if used == 1:
        r.expireat(redis_key, _next_midnight())