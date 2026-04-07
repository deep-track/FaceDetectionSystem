import pytest
from unittest.mock import patch, MagicMock
from helpers import (
    make_supabase_mock, make_redis_mock,
    MOCK_KEY_ROW_API, MOCK_KEY_ROW_PLATFORM, VALID_KEY,
)

class TestRateLimitEnforcement:
    def test_request_within_limit_passes(self, client):
        with patch("core.auth.get_supabase") as mock_sb, \
             patch("core.auth.get_redis") as mock_redis:
            mock_sb.return_value    = make_supabase_mock(MOCK_KEY_ROW_API)
            mock_redis.return_value = make_redis_mock(used=100)  # well under 5000

            resp = client.get(
                "/v1/client/usage/me",
                headers={"X-API-Key": VALID_KEY}
            )
        assert resp.status_code == 200

    def test_request_at_limit_is_blocked(self, client):
        key_row = {**MOCK_KEY_ROW_API, "monthly_limit": 5}
        with patch("core.auth.get_supabase") as mock_sb, \
             patch("core.auth.get_redis") as mock_redis:
            mock_sb.return_value    = make_supabase_mock(key_row)
            mock_redis.return_value = make_redis_mock(used=5)  # at limit

            resp = client.get(
                "/v1/client/usage/me",
                headers={"X-API-Key": VALID_KEY}
            )
        assert resp.status_code == 429

    def test_request_over_limit_returns_429(self, client):
        key_row = {**MOCK_KEY_ROW_API, "monthly_limit": 3}
        with patch("core.auth.get_supabase") as mock_sb, \
             patch("core.auth.get_redis") as mock_redis:
            mock_sb.return_value    = make_supabase_mock(key_row)
            mock_redis.return_value = make_redis_mock(used=10)  # way over

            resp = client.get(
                "/v1/client/usage/me",
                headers={"X-API-Key": VALID_KEY}
            )
        assert resp.status_code == 429
        assert "Monthly limit" in resp.json()["detail"]
        assert "deeptrack.io/pricing" in resp.json()["detail"]

    def test_429_message_includes_limit_value(self, client):
        key_row = {**MOCK_KEY_ROW_API, "monthly_limit": 150}
        with patch("core.auth.get_supabase") as mock_sb, \
             patch("core.auth.get_redis") as mock_redis:
            mock_sb.return_value    = make_supabase_mock(key_row)
            mock_redis.return_value = make_redis_mock(used=150)

            resp = client.get(
                "/v1/client/usage/me",
                headers={"X-API-Key": VALID_KEY}
            )
        assert "150" in resp.json()["detail"]


class TestRateLimitNoRedis:
    def test_no_redis_allows_requests_through(self, client):
        """If Redis is unavailable, requests should not be blocked."""
        with patch("core.auth.get_supabase") as mock_sb, \
             patch("core.auth.get_redis") as mock_redis:
            mock_sb.return_value    = make_supabase_mock(MOCK_KEY_ROW_API)
            mock_redis.return_value = None  # Redis down

            resp = client.get(
                "/v1/client/usage/me",
                headers={"X-API-Key": VALID_KEY}
            )
        assert resp.status_code == 200


class TestAPITrackLimits:
    """Verify each API plan's limit is enforced correctly."""

    @pytest.mark.parametrize("plan,limit", [
        ("payg",       999999),
        ("starter",    5000),
        ("growth",     25000),
        ("scale",      100000),
        ("enterprise", 999999),
    ])
    def test_api_plan_limit_is_correct(self, client, plan, limit):
        from core.auth import get_default_limit
        assert get_default_limit("api", plan) == limit


class TestPlatformTrackLimits:
    """Verify each Platform plan's limit is enforced correctly."""

    @pytest.mark.parametrize("plan,limit", [
        ("trial",      20),
        ("starter",    150),
        ("pro",        600),
        ("business",   2500),
        ("enterprise", 999999),
    ])
    def test_platform_plan_limit_is_correct(self, client, plan, limit):
        from core.auth import get_default_limit
        assert get_default_limit("platform", plan) == limit

    def test_platform_trial_blocks_at_20(self, client):
        key_row = {**MOCK_KEY_ROW_PLATFORM, "plan": "trial", "monthly_limit": 20}
        with patch("core.auth.get_supabase") as mock_sb, \
             patch("core.auth.get_redis") as mock_redis:
            mock_sb.return_value    = make_supabase_mock(key_row)
            mock_redis.return_value = make_redis_mock(used=20)

            resp = client.get(
                "/v1/client/usage/me",
                headers={"X-API-Key": VALID_KEY}
            )
        assert resp.status_code == 429

    def test_custom_limit_override_is_respected(self, client):
        """CTO overrides platform/pro from 600 to 1000 — 1000 should be enforced."""
        key_row = {**MOCK_KEY_ROW_PLATFORM, "plan": "pro", "monthly_limit": 1000}
        with patch("core.auth.get_supabase") as mock_sb, \
             patch("core.auth.get_redis") as mock_redis:
            mock_sb.return_value    = make_supabase_mock(key_row)
            mock_redis.return_value = make_redis_mock(used=999)  # under custom limit

            resp = client.get(
                "/v1/client/usage/me",
                headers={"X-API-Key": VALID_KEY}
            )
        assert resp.status_code == 200

    def test_custom_limit_override_blocks_at_override_value(self, client):
        key_row = {**MOCK_KEY_ROW_PLATFORM, "plan": "pro", "monthly_limit": 1000}
        with patch("core.auth.get_supabase") as mock_sb, \
             patch("core.auth.get_redis") as mock_redis:
            mock_sb.return_value    = make_supabase_mock(key_row)
            mock_redis.return_value = make_redis_mock(used=1000)  # at custom limit

            resp = client.get(
                "/v1/client/usage/me",
                headers={"X-API-Key": VALID_KEY}
            )
        assert resp.status_code == 429