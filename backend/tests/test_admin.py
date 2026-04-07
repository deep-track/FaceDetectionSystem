"""Tests for admin endpoints."""
import pytest
from unittest.mock import patch, MagicMock
from helpers import make_supabase_mock, make_redis_mock, ADMIN_SECRET

# patch target must be where the function is used, not where it's defined
SB_PATCH    = "routers.admin.get_supabase"
REDIS_PATCH = "routers.admin.get_redis"


class TestAdminAuth:
    def test_no_secret_returns_403(self, client):
        """Optional X-Admin-Secret → forbidden when missing or wrong (dev/test mode)."""
        resp = client.get("/admin/keys")
        assert resp.status_code == 403

    def test_wrong_secret_returns_403(self, client):
        resp = client.get("/admin/keys", headers={"X-Admin-Secret": "wrong"})
        assert resp.status_code == 403

    def test_correct_secret_passes(self, client):
        with patch(SB_PATCH) as mock_sb:
            mock_result      = MagicMock()
            mock_result.data = []
            mock_sb.return_value.table.return_value \
                .select.return_value \
                .order.return_value \
                .execute.return_value = mock_result

            resp = client.get("/admin/keys", headers={"X-Admin-Secret": ADMIN_SECRET})
        assert resp.status_code == 200


class TestAdminListKeys:
    def test_returns_keys_list(self, client):
        from helpers import MOCK_KEY_ROW_API
        with patch(SB_PATCH) as mock_sb:
            mock_result      = MagicMock()
            mock_result.data = [
                {**MOCK_KEY_ROW_API, "api_key": "dt_abc123", "notes": "", "created_at": "2026-03-01"}
            ]
            mock_sb.return_value.table.return_value \
                .select.return_value \
                .order.return_value \
                .execute.return_value = mock_result

            resp = client.get("/admin/keys", headers={"X-Admin-Secret": ADMIN_SECRET})
        data = resp.json()
        assert "keys"  in data
        assert "total" in data
        assert data["total"] == 1

    def test_empty_keys_returns_empty_list(self, client):
        with patch(SB_PATCH) as mock_sb:
            mock_result      = MagicMock()
            mock_result.data = []
            mock_sb.return_value.table.return_value \
                .select.return_value \
                .order.return_value \
                .execute.return_value = mock_result

            resp = client.get("/admin/keys", headers={"X-Admin-Secret": ADMIN_SECRET})
        assert resp.json()["total"] == 0


class TestAdminCreateKey:
    def test_create_key_returns_dt_prefixed_key(self, client):
        with patch(SB_PATCH) as mock_sb:
            mock_sb.return_value.table.return_value \
                .insert.return_value \
                .execute.return_value = MagicMock()

            resp = client.post(
                "/admin/keys",
                headers={"X-Admin-Secret": ADMIN_SECRET, "Content-Type": "application/json"},
                json={"owner": "Test Corp", "user_id": "user-001", "track": "api", "plan": "starter"},
            )
        assert resp.status_code == 200
        assert resp.json()["key"].startswith("dt_")

    def test_create_key_sets_correct_monthly_limit(self, client):
        with patch(SB_PATCH) as mock_sb:
            mock_sb.return_value.table.return_value \
                .insert.return_value \
                .execute.return_value = MagicMock()

            resp = client.post(
                "/admin/keys",
                headers={"X-Admin-Secret": ADMIN_SECRET, "Content-Type": "application/json"},
                json={"owner": "Test", "user_id": "user-001", "track": "api", "plan": "growth"},
            )
        assert resp.json()["monthly_limit"] == 25000

    def test_create_key_with_limit_override(self, client):
        with patch(SB_PATCH) as mock_sb:
            mock_sb.return_value.table.return_value \
                .insert.return_value \
                .execute.return_value = MagicMock()

            resp = client.post(
                "/admin/keys",
                headers={"X-Admin-Secret": ADMIN_SECRET, "Content-Type": "application/json"},
                json={"owner": "Gotham", "user_id": "user-001", "track": "api", "plan": "growth", "monthly_limit": 30000},
            )
        assert resp.json()["monthly_limit"] == 30000

    def test_create_platform_trial_key(self, client):
        with patch(SB_PATCH) as mock_sb:
            mock_sb.return_value.table.return_value \
                .insert.return_value \
                .execute.return_value = MagicMock()

            resp = client.post(
                "/admin/keys",
                headers={"X-Admin-Secret": ADMIN_SECRET, "Content-Type": "application/json"},
                json={"owner": "New User", "user_id": "user-002", "track": "platform", "plan": "trial"},
            )
        assert resp.status_code == 200
        assert resp.json()["monthly_limit"] == 20
        assert resp.json()["track"] == "platform"


class TestAdminToggleKey:
    def test_deactivate_key(self, client):
        with patch(SB_PATCH) as mock_sb:
            mock_sb.return_value.table.return_value \
                .update.return_value \
                .eq.return_value \
                .execute.return_value = MagicMock()

            resp = client.patch(
                "/admin/keys/11111111-1111-1111-1111-111111111111/deactivate",
                headers={"X-Admin-Secret": ADMIN_SECRET},
            )
        assert resp.status_code == 200
        assert resp.json()["is_active"] == False

    def test_activate_key(self, client):
        with patch(SB_PATCH) as mock_sb:
            mock_sb.return_value.table.return_value \
                .update.return_value \
                .eq.return_value \
                .execute.return_value = MagicMock()

            resp = client.patch(
                "/admin/keys/11111111-1111-1111-1111-111111111111/activate",
                headers={"X-Admin-Secret": ADMIN_SECRET},
            )
        assert resp.status_code == 200
        assert resp.json()["is_active"] == True


class TestAdminUpdatePlan:
    def test_update_plan_resets_limit_to_default(self, client):
        with patch(SB_PATCH) as mock_sb:
            mock_sb.return_value.table.return_value \
                .update.return_value \
                .eq.return_value \
                .execute.return_value = MagicMock()

            resp = client.patch(
                "/admin/keys/11111111-1111-1111-1111-111111111111/plan",
                headers={"X-Admin-Secret": ADMIN_SECRET, "Content-Type": "application/json"},
                json={"track": "api", "plan": "scale"},
            )
        assert resp.status_code == 200
        assert resp.json()["monthly_limit"] == 100000

    def test_update_plan_invalid_track_returns_400(self, client):
        resp = client.patch(
            "/admin/keys/11111111-1111-1111-1111-111111111111/plan",
            headers={"X-Admin-Secret": ADMIN_SECRET, "Content-Type": "application/json"},
            json={"track": "invalid", "plan": "starter"},
        )
        assert resp.status_code == 400

    def test_update_plan_invalid_plan_returns_400(self, client):
        resp = client.patch(
            "/admin/keys/11111111-1111-1111-1111-111111111111/plan",
            headers={"X-Admin-Secret": ADMIN_SECRET, "Content-Type": "application/json"},
            json={"track": "api", "plan": "nonexistent"},
        )
        assert resp.status_code == 400


class TestAdminUsageMonth:
    def test_returns_usage_dict(self, client):
        with patch(REDIS_PATCH) as mock_redis:
            mock_r            = MagicMock()
            mock_r.keys.return_value = ["ratelimit:uuid-001:2026-03", "ratelimit:uuid-002:2026-03"]
            mock_r.get.side_effect   = lambda k: "42" if "001" in k else "7"
            mock_redis.return_value  = mock_r

            resp = client.get("/admin/usage/month", headers={"X-Admin-Secret": ADMIN_SECRET})
        data = resp.json()
        assert "usage" in data
        assert "month" in data

    def test_no_redis_returns_empty_usage(self, client):
        with patch(REDIS_PATCH) as mock_redis:
            mock_redis.return_value = None
            resp = client.get("/admin/usage/month", headers={"X-Admin-Secret": ADMIN_SECRET})
        assert resp.json()["usage"] == {}