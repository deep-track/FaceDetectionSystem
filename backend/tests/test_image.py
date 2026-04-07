"""Tests for image prediction endpoint."""
import pytest
from unittest.mock import patch, MagicMock
from helpers import make_supabase_mock, make_redis_mock, VALID_KEY, MOCK_KEY_ROW_API, small_jpg

# patch where the functions are called from
SB_PATCH    = "core.auth.get_supabase"
REDIS_PATCH = "core.auth.get_redis"

MOCK_PREDICTION = {
    "prediction":            "FAKE",
    "confidence_percentage": 87.3,
    "raw_scores": {"Real": 12.7, "Fake": 87.3},
}


class TestImagePredictAuth:
    def test_no_key_returns_401(self, client):
        resp = client.post("/v1/image/predict")
        assert resp.status_code == 401

    def test_invalid_key_returns_401(self, client):
        with patch(SB_PATCH) as mock_sb, patch(REDIS_PATCH) as mock_redis:
            mock_db = MagicMock()
            mock_db.table.return_value.select.return_value \
                .eq.return_value.single.return_value \
                .execute.side_effect = Exception("not found")
            mock_sb.return_value    = mock_db
            mock_redis.return_value = make_redis_mock()

            resp = client.post(
                "/v1/image/predict",
                headers={"X-API-Key": "dt_badkey"},
                files={"file": ("test.jpg", small_jpg(), "image/jpeg")},
            )
        assert resp.status_code == 401


class TestImagePredictValidation:
    def test_missing_file_returns_422(self, client):
        with patch(SB_PATCH) as mock_sb, patch(REDIS_PATCH) as mock_redis:
            mock_sb.return_value    = make_supabase_mock(MOCK_KEY_ROW_API)
            mock_redis.return_value = make_redis_mock()
            # set a mock predictor so model-not-loaded doesn't fire first
            client.app.state.image_predictor = MagicMock()

            resp = client.post("/v1/image/predict", headers={"X-API-Key": VALID_KEY})
        assert resp.status_code == 422

    def test_unsupported_file_type_returns_400(self, client):
        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = MOCK_PREDICTION
        with patch(SB_PATCH) as mock_sb, patch(REDIS_PATCH) as mock_redis:
            mock_sb.return_value    = make_supabase_mock(MOCK_KEY_ROW_API)
            mock_redis.return_value = make_redis_mock()
            client.app.state.image_predictor = mock_predictor

            resp = client.post(
                "/v1/image/predict",
                headers={"X-API-Key": VALID_KEY},
                files={"file": ("test.pdf", b"fake pdf", "application/pdf")},
            )
        # PIL will fail to open it → 400
        assert resp.status_code == 400


class TestImagePredictSuccess:
    def test_valid_image_returns_200(self, client):
        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = MOCK_PREDICTION
        with patch(SB_PATCH) as mock_sb, \
             patch(REDIS_PATCH) as mock_redis, \
             patch("core.auth.increment_usage"):
            mock_sb.return_value    = make_supabase_mock(MOCK_KEY_ROW_API)
            mock_redis.return_value = make_redis_mock()
            client.app.state.image_predictor = mock_predictor

            resp = client.post(
                "/v1/image/predict",
                headers={"X-API-Key": VALID_KEY},
                files={"file": ("photo.jpg", small_jpg(), "image/jpeg")},
            )
        assert resp.status_code == 200

    def test_response_has_prediction_fields(self, client):
        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = MOCK_PREDICTION
        with patch(SB_PATCH) as mock_sb, \
             patch(REDIS_PATCH) as mock_redis, \
             patch("core.auth.increment_usage"):
            mock_sb.return_value    = make_supabase_mock(MOCK_KEY_ROW_API)
            mock_redis.return_value = make_redis_mock()
            client.app.state.image_predictor = mock_predictor

            resp = client.post(
                "/v1/image/predict",
                headers={"X-API-Key": VALID_KEY},
                files={"file": ("photo.jpg", small_jpg(), "image/jpeg")},
            )
        data = resp.json()
        assert "prediction"            in data
        assert "confidence_percentage" in data
        assert "raw_scores"            in data
        assert "filename"              in data

    def test_response_verdict_is_valid_value(self, client):
        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = MOCK_PREDICTION
        with patch(SB_PATCH) as mock_sb, \
             patch(REDIS_PATCH) as mock_redis, \
             patch("core.auth.increment_usage"):
            mock_sb.return_value    = make_supabase_mock(MOCK_KEY_ROW_API)
            mock_redis.return_value = make_redis_mock()
            client.app.state.image_predictor = mock_predictor

            resp = client.post(
                "/v1/image/predict",
                headers={"X-API-Key": VALID_KEY},
                files={"file": ("photo.jpg", small_jpg(), "image/jpeg")},
            )
        assert resp.json()["prediction"] in ("Real", "Fake", "REAL", "FAKE", "UNCERTAIN")

    def test_model_not_loaded_returns_503(self, client):
        with patch(SB_PATCH) as mock_sb, patch(REDIS_PATCH) as mock_redis:
            mock_sb.return_value    = make_supabase_mock(MOCK_KEY_ROW_API)
            mock_redis.return_value = make_redis_mock()
            client.app.state.image_predictor = None

            resp = client.post(
                "/v1/image/predict",
                headers={"X-API-Key": VALID_KEY},
                files={"file": ("photo.jpg", small_jpg(), "image/jpeg")},
            )
        assert resp.status_code == 503