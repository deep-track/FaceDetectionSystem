import os
import uuid
import logging
from datetime import datetime, timezone
from calendar import monthrange
from dotenv import load_dotenv
from fastapi import Request, HTTPException
from supabase import create_client, Client
import redis as redis_lib

load_dotenv()
logger = logging.getLogger("deeptrack.auth")

supabase_client = None
redis_client    = None

def get_supabase() -> Client:
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


def _current_month() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


def _end_of_month_timestamp() -> int:
    """Unix timestamp of the last second of the current UTC month."""
    now      = datetime.now(timezone.utc)
    last_day = monthrange(now.year, now.month)[1]
    eom      = datetime(now.year, now.month, last_day, 23, 59, 59, tzinfo=timezone.utc)
    return int(eom.timestamp())


def generate_api_key(owner: str, user_id: str, monthly_limit: int = 100, notes: str = "") -> str:
    """
    Generate and store a new API key for a user.
    Returns the raw key
    """
    raw_key = "dt_" + uuid.uuid4().hex + uuid.uuid4().hex
    db = get_supabase()
    db.table("api_details").insert({
        "user_id":       user_id,
        "api_key":       raw_key,
        "monthly_limit": monthly_limit,
        "is_active":     True,
        "owner":         owner,
        "notes":         notes,
    }).execute()
    logger.info(f"Generated API key for owner='{owner}' limit={monthly_limit}/month")
    return raw_key


async def verify_api_key(request: Request) -> dict:
    """
    FastAPI dependency, validates key and checks monthly limit via Redis.
    """
    raw_key = request.headers.get("X-API-Key")
    if not raw_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header.")

    db = get_supabase()

    try:
        res = (
            db.table("api_details")
            .select("id, user_id, owner, monthly_limit, is_active")
            .eq("api_key", raw_key)
            .single()
            .execute()
        )
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid API key.")

    if not res.data or not res.data["is_active"]:
        raise HTTPException(status_code=401, detail="Invalid or inactive API key.")

    key_row       = res.data
    monthly_limit = key_row["monthly_limit"]

    # check usage without incrementing
    r = get_redis()
    if r:
        redis_key = f"ratelimit:{key_row['id']}:{_current_month()}"
        val       = r.get(redis_key)
        used      = int(val) if val else 0
        if used >= monthly_limit:
            raise HTTPException(
                status_code=429,
                detail=f"Monthly limit of {monthly_limit} scans exceeded. "
                       f"Upgrade your plan at deeptrack.io/pricing"
            )

    request.state.api_key_row   = key_row
    request.state.monthly_limit = monthly_limit
    return key_row


def increment_usage(request: Request):
    """
    Increments the Redis counter for this key.
    Called only after a successful prediction.
    """
    r = get_redis()
    if not r:
        return
    key_row   = request.state.api_key_row
    redis_key = f"ratelimit:{key_row['id']}:{_current_month()}"
    used      = r.incr(redis_key)
    if used == 1:
        r.expireat(redis_key, _end_of_month_timestamp())