from core.auth import get_redis, verify_api_key, get_plan_rate
from datetime import datetime, timezone
from calendar import monthrange
from fastapi import APIRouter, Depends, Request

router = APIRouter()

@router.get("/usage/me")
async def my_usage(request: Request, _key: dict = Depends(verify_api_key)):
    """Returns this month's usage stats for the authenticated key."""
    key_row       = request.state.api_key_row
    monthly_limit = request.state.monthly_limit
    track         = request.state.track
    plan          = request.state.plan

    r          = get_redis()
    used_month = 0
    if r:
        redis_key  = f"ratelimit:{key_row['id']}:{datetime.now(timezone.utc).strftime('%Y-%m')}"
        val        = r.get(redis_key)
        used_month = int(val) if val else 0

    now       = datetime.now(timezone.utc)
    last_day  = monthrange(now.year, now.month)[1]
    resets_at = f"{now.year}-{now.month:02d}-{last_day:02d}"
    is_unlimited = monthly_limit >= 999999

    return {
        "owner":           key_row.get("owner"),
        "track":           track,
        "plan":            plan,
        "rate_per_scan":   get_plan_rate(track, plan),
        "monthly_limit":   None if is_unlimited else monthly_limit,
        "used_this_month": used_month,
        "remaining":       None if is_unlimited else max(monthly_limit - used_month, 0),
        "resets_at":       resets_at,
    }