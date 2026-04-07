import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pytest
from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("SUPABASE_URL",         os.getenv("SUPABASE_URL",         "https://test.supabase.co"))
os.environ.setdefault("SUPABASE_SERVICE_KEY", os.getenv("SUPABASE_SERVICE_KEY", "test-key"))
os.environ.setdefault("ADMIN_SECRET",         os.getenv("ADMIN_SECRET",         "test-admin-secret"))
os.environ.setdefault("ENVIRONMENT",          os.getenv("ENVIRONMENT",          "development"))
os.environ.setdefault("REDIS_URL",            os.getenv("REDIS_URL",            "redis://localhost:6379"))

from helpers import VALID_KEY, ADMIN_SECRET


@pytest.fixture(scope="session")
def app():
    """Lazy import so tests that only need auth logic avoid loading ML stacks (cv2, tensorflow, …)."""
    from main import app as fastapi_app
    return fastapi_app


@pytest.fixture
def client(app):
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        yield c

@pytest.fixture
def valid_api_key_headers():
    return {"X-API-Key": VALID_KEY}

@pytest.fixture
def admin_headers():
    return {"X-Admin-Secret": ADMIN_SECRET}