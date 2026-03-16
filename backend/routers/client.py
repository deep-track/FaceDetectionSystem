from core.auth import get_redis, verify_api_key
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, Request

router = APIRouter()

@router.get("/usage/me")
async def my_usage(request: Request, _key: dict = Depends(verify_api_key)):
    """ counter for checking daily limits left"""
    key_row     = request.state.api_key_row
    daily_limit = key_row["daily_limit"]

    r = get_redis()
    used_today = 0
    if r:
        today     = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        redis_key = f"ratelimit:{key_row['id']}:{today}"
        val       = r.get(redis_key)
        used_today = int(val) if val else 0

    return {
        "daily_limit":  daily_limit,
        "used_today":   used_today,
        "remaining":    max(daily_limit - used_today, 0),
        "resets_at":    "midnight UTC",
    }