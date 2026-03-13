import os
import logging
from dotenv import load_dotenv
from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

load_dotenv()
logger = logging.getLogger("deeptrack.limiter")

REDIS_URL = os.getenv("REDIS_URL")

def _make_limiter() -> Limiter:
    if REDIS_URL:
        logger.info(f"Rate limiter using Redis: {REDIS_URL[:30]}...")
        return Limiter(
            key_func=get_api_key_identifier,
            storage_uri=REDIS_URL,
        )
    else:
        logger.warning("REDIS_URL not set — rate limiter using in-memory storage, use for dev only")
        return Limiter(key_func=get_api_key_identifier)


def get_api_key_identifier(request: Request) -> str:
    """
    SlowAPI key function — identifies each client by their API key.
    Falls back to IP address for unauthenticated requests (demo UI).
    """
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return f"apikey:{api_key}"
    return f"ip:{get_remote_address(request)}"


def get_api_limit(request: Request) -> str:
    """
    SlowAPI dynamic limit callable.
    Returns the per-key limit string, e.g. '1000/day'.
    verify_api_key must run first (as a Depends) to set request.state.api_limit.
    """
    limit = getattr(request.state, "api_limit", 10)
    return f"{limit}/day"

# Global limiter instance — imported by main.py and routers
limiter = _make_limiter()