import pytest
from unittest.mock import patch, MagicMock
from helpers import (
    make_supabase_mock, make_redis_mock,
    MOCK_KEY_ROW_API, VALID_KEY, INVALID_KEY, INACTIVE_KEY,
)

class TestAuthMissingKey:
    def test_image_predict_no_key_returns_401(self, client):
        resp = client.post("/v1/image/predict")
        assert resp.status_code == 401
        assert "Missing X-API-Key" in resp.json()["detail"]

    def test_video_predict_no_key_returns_401(self, client):
        resp = client.post("/v1/video/predict/video")
        assert resp.status_code == 401

    def test_usage_no_key_returns_401(self, client):
        resp = client.get("/v1/client/usage/me")
        assert resp.status_code == 401

class TestAuthInvalidKey:
    def test_invalid_key_returns_401(self, client):
        with patch("core.auth.get_supabase") as mock_sb:
            mock_db = MagicMock()
            mock_db.table.return_value.select.return_value\
                .eq.return_value.single.return_value\
                .execute.side_effect = Exception("not found")
            mock_sb.return_value = mock_db

            resp = client.get(
                "/v1/client/usage/me",
                headers={"X-API-Key": INVALID_KEY}
            )
        assert resp.status_code == 401
        assert "Invalid API key" in resp.json()["detail"]

    def test_key_not_starting_with_dt_still_validated_by_server(self, client):
        """Keys without dt_ prefix are still checked server-side."""
        with patch("core.auth.get_supabase") as mock_sb:
            mock_db = MagicMock()
            mock_db.table.return_value.select.return_value\
                .eq.return_value.single.return_value\
                .execute.side_effect = Exception("not found")
            mock_sb.return_value = mock_db

            resp = client.get(
                "/v1/client/usage/me",
                headers={"X-API-Key": "not_a_real_key"}
            )
        assert resp.status_code == 401


class TestAuthInactiveKey:
    def test_inactive_key_returns_401(self, client):
        with patch("core.auth.get_supabase") as mock_sb:
            mock_sb.return_value = make_supabase_mock(MOCK_KEY_ROW_API, active=False)

            resp = client.get(
                "/v1/client/usage/me",
                headers={"X-API-Key": INACTIVE_KEY}
            )
        assert resp.status_code == 401
        assert "Invalid or inactive" in resp.json()["detail"]


class TestAuthValidKey:
    def test_valid_key_reaches_endpoint(self, client):
        with patch("core.auth.get_supabase") as mock_sb, \
             patch("core.auth.get_redis") as mock_redis:
            mock_sb.return_value    = make_supabase_mock(MOCK_KEY_ROW_API)
            mock_redis.return_value = make_redis_mock(used=0)

            resp = client.get(
                "/v1/client/usage/me",
                headers={"X-API-Key": VALID_KEY}
            )
        assert resp.status_code == 200

    def test_valid_key_response_has_expected_fields(self, client):
        with patch("core.auth.get_supabase") as mock_sb, \
             patch("core.auth.get_redis") as mock_redis:
            mock_sb.return_value    = make_supabase_mock(MOCK_KEY_ROW_API)
            mock_redis.return_value = make_redis_mock(used=10)

            resp = client.get(
                "/v1/client/usage/me",
                headers={"X-API-Key": VALID_KEY}
            )
        data = resp.json()
        assert "monthly_limit"   in data
        assert "used_this_month" in data
        assert "remaining"       in data
        assert "resets_at"       in data
        assert "track"           in data
        assert "plan"            in data