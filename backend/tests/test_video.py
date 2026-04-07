"""Tests for video prediction endpoints."""
import io
import time
import pytest
from unittest.mock import patch, MagicMock
from helpers import make_supabase_mock, make_redis_mock, VALID_KEY, MOCK_KEY_ROW_API

SB_PATCH    = "core.auth.get_supabase"
REDIS_PATCH = "core.auth.get_redis"

SMALL_MP4 = b'\x00\x00\x00\x18ftypmp42' + b'\x00' * 100

MOCK_VIDEO_RESULT = {
    "label": "FAKE", "confidence": 91.2, "fake_prob": 0.912,
    "face_pct": 98.0, "total_frames": 240, "n_segments": 4, "segments": [],
}


class TestVideoAuth:
    def test_no_key_returns_401(self, client):
        resp = client.post("/v1/video/predict/video")
        assert resp.status_code == 401

    def test_jobs_endpoint_requires_key(self, client):
        resp = client.get("/v1/video/jobs/some-job-id")
        assert resp.status_code == 401


class TestVideoUploadValidation:
    def test_unsupported_format_returns_400(self, client):
        mock_predictor = MagicMock()
        with patch(SB_PATCH) as mock_sb, patch(REDIS_PATCH) as mock_redis:
            mock_sb.return_value    = make_supabase_mock(MOCK_KEY_ROW_API)
            mock_redis.return_value = make_redis_mock()
            client.app.state.video_predictor = mock_predictor
            client.app.state.jobs            = {}

            resp = client.post(
                "/v1/video/predict/video",
                headers={"X-API-Key": VALID_KEY},
                files={"file": ("clip.wmv", SMALL_MP4, "video/x-ms-wmv")},
            )
        assert resp.status_code == 400
        assert "Unsupported format" in resp.json()["detail"]

    def test_file_too_large_returns_413(self, client):
        mock_predictor = MagicMock()
        large_file     = b'\x00' * (51 * 1024 * 1024)
        with patch(SB_PATCH) as mock_sb, patch(REDIS_PATCH) as mock_redis:
            mock_sb.return_value    = make_supabase_mock(MOCK_KEY_ROW_API)
            mock_redis.return_value = make_redis_mock()
            client.app.state.video_predictor = mock_predictor
            client.app.state.jobs            = {}

            resp = client.post(
                "/v1/video/predict/video",
                headers={"X-API-Key": VALID_KEY},
                files={"file": ("big.mp4", large_file, "video/mp4")},
            )
        assert resp.status_code == 413

    def test_missing_file_returns_422(self, client):
        mock_predictor = MagicMock()
        with patch(SB_PATCH) as mock_sb, patch(REDIS_PATCH) as mock_redis:
            mock_sb.return_value    = make_supabase_mock(MOCK_KEY_ROW_API)
            mock_redis.return_value = make_redis_mock()
            client.app.state.video_predictor = mock_predictor

            resp = client.post("/v1/video/predict/video", headers={"X-API-Key": VALID_KEY})
        assert resp.status_code == 422


class TestVideoUploadSuccess:
    def test_valid_upload_returns_job_id(self, client):
        mock_predictor = MagicMock()
        with patch(SB_PATCH) as mock_sb, \
             patch(REDIS_PATCH) as mock_redis, \
             patch("routers.video.video_executor") as mock_exec:
            mock_sb.return_value             = make_supabase_mock(MOCK_KEY_ROW_API)
            mock_redis.return_value          = make_redis_mock()
            mock_exec.submit                 = MagicMock()
            client.app.state.video_predictor = mock_predictor
            client.app.state.jobs            = {}

            resp = client.post(
                "/v1/video/predict/video",
                headers={"X-API-Key": VALID_KEY},
                files={"file": ("clip.mp4", SMALL_MP4, "video/mp4")},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert "job_id"   in data
        assert "status"   in data
        assert "poll_url" in data
        assert data["status"] == "queued"

    def test_poll_url_contains_job_id(self, client):
        mock_predictor = MagicMock()
        with patch(SB_PATCH) as mock_sb, \
             patch(REDIS_PATCH) as mock_redis, \
             patch("routers.video.video_executor") as mock_exec:
            mock_sb.return_value             = make_supabase_mock(MOCK_KEY_ROW_API)
            mock_redis.return_value          = make_redis_mock()
            mock_exec.submit                 = MagicMock()
            client.app.state.video_predictor = mock_predictor
            client.app.state.jobs            = {}

            resp = client.post(
                "/v1/video/predict/video",
                headers={"X-API-Key": VALID_KEY},
                files={"file": ("clip.mp4", SMALL_MP4, "video/mp4")},
            )
        data = resp.json()
        assert data["job_id"] in data["poll_url"]

    def test_model_not_loaded_returns_503(self, client):
        with patch(SB_PATCH) as mock_sb, patch(REDIS_PATCH) as mock_redis:
            mock_sb.return_value             = make_supabase_mock(MOCK_KEY_ROW_API)
            mock_redis.return_value          = make_redis_mock()
            client.app.state.video_predictor = None

            resp = client.post(
                "/v1/video/predict/video",
                headers={"X-API-Key": VALID_KEY},
                files={"file": ("clip.mp4", SMALL_MP4, "video/mp4")},
            )
        assert resp.status_code == 503


class TestJobPolling:
    def test_get_job_not_found_returns_404(self, client):
        with patch(SB_PATCH) as mock_sb, patch(REDIS_PATCH) as mock_redis:
            mock_sb.return_value    = make_supabase_mock(MOCK_KEY_ROW_API)
            mock_redis.return_value = make_redis_mock()
            client.app.state.jobs   = {}

            resp = client.get(
                "/v1/video/jobs/nonexistent-job-id",
                headers={"X-API-Key": VALID_KEY},
            )
        assert resp.status_code == 404

    def test_get_job_queued_status(self, client):
        job_id = "test-job-123"
        with patch(SB_PATCH) as mock_sb, patch(REDIS_PATCH) as mock_redis:
            mock_sb.return_value    = make_supabase_mock(MOCK_KEY_ROW_API)
            mock_redis.return_value = make_redis_mock()
            client.app.state.jobs   = {
                job_id: {
                    "status": "queued", "filename": "clip.mp4",
                    "size_mb": 2.1, "created_at": time.time(),
                    "result": None, "error": None,
                }
            }
            resp = client.get(f"/v1/video/jobs/{job_id}", headers={"X-API-Key": VALID_KEY})
        assert resp.status_code == 200
        assert resp.json()["status"] == "queued"

    def test_get_job_done_returns_result(self, client):
        job_id = "test-job-done"
        with patch(SB_PATCH) as mock_sb, patch(REDIS_PATCH) as mock_redis:
            mock_sb.return_value    = make_supabase_mock(MOCK_KEY_ROW_API)
            mock_redis.return_value = make_redis_mock()
            client.app.state.jobs   = {
                job_id: {
                    "status": "done", "filename": "clip.mp4",
                    "size_mb": 2.1, "created_at": time.time(),
                    "result": MOCK_VIDEO_RESULT, "error": None,
                }
            }
            resp = client.get(f"/v1/video/jobs/{job_id}", headers={"X-API-Key": VALID_KEY})
        data = resp.json()
        assert data["status"]              == "done"
        assert data["result"]["label"]     == "FAKE"
        assert data["result"]["fake_prob"] == 0.912

    def test_get_job_error_status(self, client):
        job_id = "test-job-error"
        with patch(SB_PATCH) as mock_sb, patch(REDIS_PATCH) as mock_redis:
            mock_sb.return_value    = make_supabase_mock(MOCK_KEY_ROW_API)
            mock_redis.return_value = make_redis_mock()
            client.app.state.jobs   = {
                job_id: {
                    "status": "error", "filename": "clip.mp4",
                    "size_mb": 2.1, "created_at": time.time(),
                    "result": None, "error": "No face detected",
                }
            }
            resp = client.get(f"/v1/video/jobs/{job_id}", headers={"X-API-Key": VALID_KEY})
        data = resp.json()
        assert data["status"] == "error"
        assert data["error"]  == "No face detected"