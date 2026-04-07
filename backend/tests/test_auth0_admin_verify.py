"""Unit tests for Auth0 admin JWT verification (no full app / ML imports)."""
import pytest
from unittest.mock import MagicMock, patch
from fastapi import HTTPException

import core.auth as auth


@pytest.fixture(autouse=True)
def reset_jwks_client():
    auth._auth0_jwks_client = None
    yield
    auth._auth0_jwks_client = None


@pytest.fixture
def auth0_env(monkeypatch):
    monkeypatch.setenv("AUTH0_DOMAIN", "example.auth0.com")
    monkeypatch.setenv("AUTH0_AUDIENCE", "https://deeptrack-api/")
    monkeypatch.delenv("AUTH0_ISSUER", raising=False)


def test_is_production_true(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    assert auth.is_production_environment() is True
    monkeypatch.setenv("ENVIRONMENT", "prod")
    assert auth.is_production_environment() is True


def test_is_production_false(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "development")
    assert auth.is_production_environment() is False
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    assert auth.is_production_environment() is False


def test_verify_missing_header_401(auth0_env):
    with pytest.raises(HTTPException) as ei:
        auth.verify_auth0_admin_bearer(None)
    assert ei.value.status_code == 401


def test_verify_not_bearer_401(auth0_env):
    with pytest.raises(HTTPException) as ei:
        auth.verify_auth0_admin_bearer("Basic xyz")
    assert ei.value.status_code == 401


def test_verify_missing_config_500(monkeypatch):
    monkeypatch.delenv("AUTH0_AUDIENCE", raising=False)
    monkeypatch.delenv("AUTH0_DOMAIN", raising=False)
    monkeypatch.delenv("AUTH0_ISSUER", raising=False)
    with pytest.raises(HTTPException) as ei:
        auth.verify_auth0_admin_bearer("Bearer eyJhbGciOiJSUzI1NiJ9.e30.sig")
    assert ei.value.status_code == 500


def test_verify_success_decodes_claims(auth0_env):
    signing_key = MagicMock()
    signing_key.key = "mock-public-key"
    mock_jwks = MagicMock()
    mock_jwks.get_signing_key_from_jwt.return_value = signing_key

    with patch.object(auth, "PyJWKClient", return_value=mock_jwks):
        with patch.object(auth.jwt, "decode", return_value={"sub": "auth0|admin1", "email": "a@x.com"}):
            out = auth.verify_auth0_admin_bearer("Bearer header.payload.sig")
    assert out["sub"] == "auth0|admin1"
    assert out["email"] == "a@x.com"
    mock_jwks.get_signing_key_from_jwt.assert_called_once()
