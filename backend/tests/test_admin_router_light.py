"""
Admin router tests without loading main (no tensorflow / cv2 / mediapipe).
Uses a minimal FastAPI app with only the admin router mounted.
"""
import pytest
from unittest.mock import patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from routers.admin import router as admin_router
from helpers import ADMIN_SECRET


@pytest.fixture
def admin_app_dev(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "development")
    app = FastAPI()
    app.include_router(admin_router, prefix="/admin")
    return app


@pytest.fixture
def admin_app_prod(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    app = FastAPI()
    app.include_router(admin_router, prefix="/admin")
    return app


class TestAdminSecretDev:
    def test_no_secret_403(self, admin_app_dev):
        with TestClient(admin_app_dev) as c:
            r = c.get("/admin/plans")
        assert r.status_code == 403

    def test_wrong_secret_403(self, admin_app_dev):
        with TestClient(admin_app_dev) as c:
            r = c.get("/admin/plans", headers={"X-Admin-Secret": "wrong"})
        assert r.status_code == 403

    def test_correct_secret_ok(self, admin_app_dev):
        with TestClient(admin_app_dev) as c:
            r = c.get("/admin/plans", headers={"X-Admin-Secret": ADMIN_SECRET})
        assert r.status_code == 200
        assert "plans" in r.json()


class TestAdminAuth0Prod:
    def test_missing_bearer_401(self, admin_app_prod):
        with TestClient(admin_app_prod) as c:
            r = c.get("/admin/plans")
        assert r.status_code == 401

    def test_bearer_calls_verify(self, admin_app_prod):
        with patch("routers.admin.verify_auth0_admin_bearer", return_value={"sub": "auth0|1"}):
            with TestClient(admin_app_prod) as c:
                r = c.get("/admin/plans", headers={"Authorization": "Bearer eyJ.test.sig"})
        assert r.status_code == 200
