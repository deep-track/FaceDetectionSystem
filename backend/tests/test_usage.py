"""Tests for /v1/client/usage/me endpoint."""
import pytest
from unittest.mock import patch
from helpers import make_supabase_mock, make_redis_mock, VALID_KEY, MOCK_KEY_ROW_API, MOCK_KEY_ROW_PLATFORM

SB_PATCH    = "core.auth.get_supabase"
REDIS_PATCH = "core.auth.get_redis"

class TestUsageEndpoint:
    def test_returns_200_with_valid_key(self, client):
        with patch(SB_PATCH) as mock_sb, patch(REDIS_PATCH) as mock_redis:
            mock_sb.return_value    = make_supabase_mock(MOCK_KEY_ROW_API)
            mock_redis.return_value = make_redis_mock(used=0)
            resp = client.get("/v1/client/usage/me", headers={"X-API-Key": VALID_KEY})
        assert resp.status_code == 200

    def test_response_schema(self, client):
        with patch(SB_PATCH) as mock_sb, patch(REDIS_PATCH) as mock_redis:
            mock_sb.return_value    = make_supabase_mock(MOCK_KEY_ROW_API)
            mock_redis.return_value = make_redis_mock(used=50)
            resp = client.get("/v1/client/usage/me", headers={"X-API-Key": VALID_KEY})
        data = resp.json()
        for field in ["monthly_limit", "used_this_month", "remaining", "resets_at", "track", "plan", "rate_per_scan"]:
            assert field in data

    def test_used_this_month_reflects_redis_counter(self, client):
        with patch(SB_PATCH) as mock_sb, \
             patch(REDIS_PATCH) as mock_redis, \
             patch("routers.client.get_redis") as mock_client_redis:
            mock_sb.return_value    = make_supabase_mock(MOCK_KEY_ROW_API)
            mock_redis.return_value = make_redis_mock(used=123)
            # also patch get_redis in the client router directly
            mock_client_redis.return_value = make_redis_mock(used=123)

            resp = client.get("/v1/client/usage/me", headers={"X-API-Key": VALID_KEY})
        assert resp.json()["used_this_month"] == 123

    def test_remaining_is_limit_minus_used(self, client):
        key_row = {**MOCK_KEY_ROW_API, "monthly_limit": 5000}
        with patch(SB_PATCH) as mock_sb, \
             patch(REDIS_PATCH) as mock_redis, \
             patch("routers.client.get_redis") as mock_client_redis:
            mock_sb.return_value           = make_supabase_mock(key_row)
            mock_redis.return_value        = make_redis_mock(used=200)
            mock_client_redis.return_value = make_redis_mock(used=200)

            resp = client.get("/v1/client/usage/me", headers={"X-API-Key": VALID_KEY})
        assert resp.json()["remaining"] == 4800

    def test_remaining_never_goes_below_zero(self, client):
        key_row = {**MOCK_KEY_ROW_API, "monthly_limit": 5}
        with patch(SB_PATCH) as mock_sb, \
             patch(REDIS_PATCH) as mock_redis, \
             patch("routers.client.get_redis") as mock_client_redis:
            mock_sb.return_value           = make_supabase_mock(key_row)
            mock_redis.return_value        = make_redis_mock(used=5)
            mock_client_redis.return_value = make_redis_mock(used=999)

            # at limit — verify_api_key blocks, so test at limit-1
            key_row2 = {**key_row, "monthly_limit": 1000}
            mock_sb.return_value           = make_supabase_mock(key_row2)
            mock_redis.return_value        = make_redis_mock(used=0)
            mock_client_redis.return_value = make_redis_mock(used=9999)

            resp = client.get("/v1/client/usage/me", headers={"X-API-Key": VALID_KEY})
        assert resp.json()["remaining"] == 0

    def test_resets_at_is_end_of_month(self, client):
        with patch(SB_PATCH) as mock_sb, patch(REDIS_PATCH) as mock_redis:
            mock_sb.return_value    = make_supabase_mock(MOCK_KEY_ROW_API)
            mock_redis.return_value = make_redis_mock(used=0)
            resp = client.get("/v1/client/usage/me", headers={"X-API-Key": VALID_KEY})
        resets_at = resp.json()["resets_at"]
        assert len(resets_at) == 10
        assert resets_at[4]   == "-"
        assert resets_at[7]   == "-"

    def test_api_growth_rate_per_scan(self, client):
        key_row = {**MOCK_KEY_ROW_API, "plan": "growth"}
        with patch(SB_PATCH) as mock_sb, patch(REDIS_PATCH) as mock_redis:
            mock_sb.return_value    = make_supabase_mock(key_row)
            mock_redis.return_value = make_redis_mock(used=0)
            resp = client.get("/v1/client/usage/me", headers={"X-API-Key": VALID_KEY})
        assert resp.json()["rate_per_scan"] == 0.37

    def test_platform_pro_rate_per_scan(self, client):
        key_row = {**MOCK_KEY_ROW_PLATFORM, "plan": "pro"}
        with patch(SB_PATCH) as mock_sb, patch(REDIS_PATCH) as mock_redis:
            mock_sb.return_value    = make_supabase_mock(key_row)
            mock_redis.return_value = make_redis_mock(used=0)
            resp = client.get("/v1/client/usage/me", headers={"X-API-Key": VALID_KEY})
        assert resp.json()["rate_per_scan"] == 0.43

    def test_unlimited_plan_returns_none_for_remaining(self, client):
        key_row = {**MOCK_KEY_ROW_API, "plan": "payg", "monthly_limit": 999999}
        with patch(SB_PATCH) as mock_sb, \
             patch(REDIS_PATCH) as mock_redis, \
             patch("routers.client.get_redis") as mock_client_redis:
            mock_sb.return_value           = make_supabase_mock(key_row)
            mock_redis.return_value        = make_redis_mock(used=0)
            mock_client_redis.return_value = make_redis_mock(used=5000)

            resp = client.get("/v1/client/usage/me", headers={"X-API-Key": VALID_KEY})
        data = resp.json()
        assert data["remaining"]     is None
        assert data["monthly_limit"] is None