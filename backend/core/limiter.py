import os
import logging
from dotenv import load_dotenv
from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

load_dotenv()
logger = logging.getLogger("deeptrack.limiter")
REDIS_URL = os.getenv("REDIS_URL")

def get_api_key_identifier(request: Request) -> str:
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return f"apikey:{api_key}"
    return f"ip:{get_remote_address(request)}"


def _make_limiter() -> Limiter:
    if REDIS_URL:
        logger.info(f"Rate limiter using Redis")
        return Limiter(key_func=get_api_key_identifier, storage_uri=REDIS_URL)
    logger.warning("REDIS_URL not set — using in-memory (dev only)")
    return Limiter(key_func=get_api_key_identifier)

limiter = _make_limiter()