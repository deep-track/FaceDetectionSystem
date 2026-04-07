import os
import uuid
import logging
from datetime import datetime, timezone
from calendar import monthrange
from dotenv import load_dotenv
from fastapi import Request, HTTPException
from supabase import create_client, Client
import redis as redis_lib
import jwt
from jwt import PyJWKClient

load_dotenv()
logger = logging.getLogger("deeptrack.auth")

supabase_client = None
redis_client    = None
_auth0_jwks_client: PyJWKClient | None = None


def is_production_environment() -> bool:
    v = (os.getenv("ENVIRONMENT") or os.getenv("APP_ENV") or "").lower().strip()
    return v in ("production", "prod")


def _auth0_issuer() -> str:
    explicit = (os.getenv("AUTH0_ISSUER") or "").strip().rstrip("/")
    if explicit:
        return explicit + "/"
    domain = (os.getenv("AUTH0_DOMAIN") or "").strip()
    if not domain:
        return ""
    return f"https://{domain}/"


def _auth0_jwks_url() -> str:
    iss = _auth0_issuer().rstrip("/")
    if not iss:
        return ""
    return f"{iss}/.well-known/jwks.json"


def verify_auth0_admin_bearer(authorization: str | None) -> dict:
    """
    Validates Authorization: Bearer <JWT> (Auth0 access token: Google, SSO, etc.).
    Requires AUTH0_AUDIENCE and AUTH0_DOMAIN or AUTH0_ISSUER.
    """
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization Bearer token.")

    token = authorization.split(None, 1)[1].strip()
    audience = (os.getenv("AUTH0_AUDIENCE") or "").strip()
    issuer   = _auth0_issuer()

    if not audience or not issuer:
        logger.error("Auth0 admin: AUTH0_AUDIENCE and AUTH0_DOMAIN (or AUTH0_ISSUER) must be set in production.")
        raise HTTPException(status_code=500, detail="Auth0 is not configured.")

    global _auth0_jwks_client
    if _auth0_jwks_client is None:
        jwks_url = _auth0_jwks_url()
        if not jwks_url:
            raise HTTPException(status_code=500, detail="Auth0 is not configured.")
        _auth0_jwks_client = PyJWKClient(jwks_url)

    try:
        signing_key = _auth0_jwks_client.get_signing_key_from_jwt(token)
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            audience=audience,
            issuer=issuer,
            leeway=60,
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired.")
    except jwt.InvalidTokenError as e:
        logger.warning("Auth0 JWT rejected: %s", e)
        raise HTTPException(status_code=401, detail="Invalid or unauthorized token.")
    except Exception as e:
        logger.warning("Auth0 JWT verification error: %s", e)
        raise HTTPException(status_code=401, detail="Invalid or unauthorized token.")

    return payload

# Default monthly limits per plan — used when creating a key.
# monthly_limit is stored in DB and can be overridden per client.
PLAN_LIMITS = {
    "api": {
        "payg":       999999,  # pay as you go, no limits
        "starter":    5000,
        "growth":     25000,
        "scale":      100000,
        "enterprise": 999999,
    },
    "platform": {
        "trial":      20,
        "starter":    150,
        "pro":        600,
        "business":   2500,
        "enterprise": 999999,
    },
}

# Per-scan rates, for display and billing reference
PLAN_RATES = {
    "api": {
        "payg":       0.40,
        "starter":    0.39,
        "growth":     0.37,
        "scale":      0.36,
        "enterprise": 0.35,
    },
    "platform": {
        "trial":      0.00,
        "starter":    0.50,
        "pro":        0.43,
        "business":   0.40,
        "enterprise": 0.00,
    },
}


def get_default_limit(track: str, plan: str) -> int:
    """Returns the default monthly limit for a plan. Used at key creation time."""
    return PLAN_LIMITS.get(track, {}).get(plan, 100)


def get_plan_rate(track: str, plan: str) -> float:
    return PLAN_RATES.get(track, {}).get(plan, 0.40)


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
    now      = datetime.now(timezone.utc)
    last_day = monthrange(now.year, now.month)[1]
    eom      = datetime(now.year, now.month, last_day, 23, 59, 59, tzinfo=timezone.utc)
    return int(eom.timestamp())


def generate_api_key(
    owner:         str,
    user_id:       str,
    track:         str = "api",
    plan:          str = "payg",
    monthly_limit: int = None,  # if None, uses plan default
    notes:         str = "",
) -> str:
    """
    Generate and store a new API key.
    monthly_limit defaults to the plan's standard limit but can be overridden.
    Returns the raw key — only shown once.
    """
    if monthly_limit is None:
        monthly_limit = get_default_limit(track, plan)

    raw_key = "dt_" + uuid.uuid4().hex + uuid.uuid4().hex
    db = get_supabase()
    db.table("api_details").insert({
        "user_id":       user_id,
        "api_key":       raw_key,
        "track":         track,
        "plan":          plan,
        "monthly_limit": monthly_limit,
        "is_active":     True,
        "owner":         owner,
        "notes":         notes,
    }).execute()
    logger.info(f"Generated API key for owner='{owner}' track='{track}' plan='{plan}' limit={monthly_limit}/month")
    return raw_key


async def verify_api_key(request: Request) -> dict:
    """
    FastAPI dependency — validates key and checks monthly limit via Redis.
    Limit is read from api_details.monthly_limit (not derived from plan).
    Does not increment the counter — see increment_usage().
    """
    raw_key = request.headers.get("X-API-Key")
    if not raw_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header.")

    db = get_supabase()

    try:
        res = (
            db.table("api_details")
            .select("id, user_id, owner, track, plan, monthly_limit, is_active")
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
    request.state.track         = key_row["track"]
    request.state.plan          = key_row["plan"]
    return key_row


def increment_usage(request: Request):
    """Increments the Redis counter. Call only after a successful prediction."""
    r = get_redis()
    if not r:
        return
    key_row   = request.state.api_key_row
    redis_key = f"ratelimit:{key_row['id']}:{_current_month()}"
    used      = r.incr(redis_key)
    if used == 1:
        r.expireat(redis_key, _end_of_month_timestamp())