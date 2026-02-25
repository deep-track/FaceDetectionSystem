# FakeCatcher API Reference

**Version:** 1.0 | **Base URL:** `/` | **Protocol:** HTTP / WebSocket | **Auth:** None required

FakeCatcher detects deepfakes by analyzing subtle physiological signals (rPPG — remote photoplethysmography) extracted from facial regions, fed through a CNN classifier. It supports real-time frame analysis, async video processing, and WebSocket streaming.

---

## Table of Contents

- [Endpoints](#endpoints)
- [Frame Prediction](#frame-prediction)
- [Video Prediction](#video-prediction)
- [Job Management](#job-management)
- [System Endpoints](#system-endpoints)
- [WebSocket Streaming](#websocket-streaming)
- [Reference: Labels, Errors, Rate Limits](#reference)

---

## Endpoints

| Method | Endpoint | Auth | Description | Key Constraints |
|--------|----------|------|-------------|-----------------|
| `POST` | `/predict/frame` | — | Submit a single Base64-encoded JPEG frame | Buffers 128 frames before predicting (~4s) |
| `POST` | `/predict/video` | — | Upload a video file for async analysis | `.mp4 .avi .mov .mkv` only; max 50 MB |
| `GET` | `/jobs/{job_id}` | — | Poll status and result of a video job | Job TTL: 2 hours |
| `GET` | `/jobs` | — | List all active jobs | — |
| `GET` | `/status` | — | Inspect the current frame buffer state | — |
| `POST` | `/reset` | — | Clear the frame buffer | — |
| `GET` | `/health` | — | Liveness check for model and server | — |
| `GET` | `/metrics` | — | System-wide stats (workers, queue depth, limits) | — |
| `WS` | `/ws/predict` | — | Stream frames for real-time prediction | Same 128-frame buffer requirement |

---

## Frame Prediction

### `POST /predict/frame`

Submit one JPEG frame at a time. Predictions are emitted after every 128-frame buffer is filled (~4 seconds at 30 fps). Subsequent frames continue the rolling buffer.

**Request body**
```json
{
  "image": "<base64-encoded JPEG>"
}
```

**Response — buffering** (`< 128 frames`)
```json
{
  "status": "buffering",
  "fill_pct": 42,
  "frames_seen": 54,
  "message": "Buffering... 42% — need 128 frames (~4s)"
}
```

**Response — prediction** (`>= 128 frames`)
```json
{
  "status": "prediction",
  "label": "REAL",
  "confidence": 92.3,
  "fake_prob": 0.0765,
  "frames_seen": 128
}
```

> **Note:** `fake_prob` is a float in `[0, 1]`. `confidence` is derived from the distance from the 0.5 decision boundary and expressed as a percentage.

---

## Video Prediction

### `POST /predict/video`

Upload a video file for asynchronous deepfake analysis. Returns immediately with a `job_id`; poll `/jobs/{job_id}` for results.

**Request** — `multipart/form-data`

| Field | Type | Notes |
|-------|------|-------|
| `file` | binary | Accepted formats: `.mp4`, `.avi`, `.mov`, `.mkv`; max 50 MB |

**Response — job queued** `202`
```json
{
  "job_id": "3f2a1b4c-...",
  "status": "queued",
  "filename": "video.mp4",
  "size_mb": 12.5,
  "poll_url": "/jobs/3f2a1b4c-...",
  "message": "Job queued. Poll /jobs/{job_id} for result."
}
```

**Rate limit:** 30 upload requests per minute. Up to 4 jobs can process concurrently.

---

## Job Management

### `GET /jobs/{job_id}`

Retrieve the current status or final result of a video analysis job.

**Response — completed job**
```json
{
  "job_id": "3f2a1b4c-...",
  "status": "done",
  "filename": "video.mp4",
  "size_mb": 12.5,
  "result": {
    "label": "FAKE",
    "confidence": 88.4,
    "fake_prob": 0.8843,
    "total_frames": 842,
    "face_pct": 97.3,
    "n_segments": 12
  },
  "error": null,
  "age_sec": 45
}
```

**Job status values**

| Status | Meaning |
|--------|---------|
| `queued` | Waiting for an available worker |
| `processing` | Actively being analysed |
| `done` | Completed successfully — `result` is populated |
| `error` | Processing failed — `error` contains the reason |

**Result fields**

| Field | Type | Description |
|-------|------|-------------|
| `label` | string | `REAL`, `FAKE`, or `UNCERTAIN` |
| `confidence` | float | Confidence percentage (0–100) |
| `fake_prob` | float | Raw model probability (0–1) |
| `total_frames` | int | Frames analysed |
| `face_pct` | float | % of frames with a detectable face |
| `n_segments` | int | Number of 128-frame segments used |

> **TTL:** Jobs expire 2 hours after creation. After expiry, `GET /jobs/{job_id}` returns `404`.

---

### `GET /jobs`

List all active (non-expired) jobs.

```json
{
  "jobs": {
    "3f2a1b4c-...": {
      "status": "processing",
      "filename": "video.mp4",
      "age_sec": 32
    }
  },
  "counts": {
    "processing": 1
  },
  "total": 1
}
```

---

## System Endpoints

### `GET /status`

Returns the current state of the frame buffer used by `/predict/frame` and `/ws/predict`.

```json
{
  "buffer_fill_pct": 50,
  "frames_seen": 64,
  "omega": 128,
  "last_result": null,
  "model": "data/cnn_model.keras"
}
```

---

### `POST /reset`

Clears the frame buffer. Useful when switching video sources mid-session.

```json
{
  "status": "reset",
  "message": "Frame buffer cleared."
}
```

> **Warning:** This affects all concurrent callers sharing the same buffer instance. Not safe for multi-tenant deployments without per-client buffer isolation.

---

### `GET /health`

Lightweight liveness check. Returns `200` when both the model and HTTP server are operational.

```json
{
  "status": "ok",
  "model": "operational",
  "server": "operational"
}
```

Use this endpoint for load-balancer health probes.

---

### `GET /metrics`

Returns system-wide operational stats.

```json
{
  "jobs": {
    "queued": 1,
    "processing": 2
  },
  "total_jobs": 3,
  "workers": 4,
  "rate_limit_rpm": 30,
  "max_upload_mb": 50,
  "auth_enabled": false,
  "job_ttl_hours": 2
}
```

---

## WebSocket Streaming

### `WS /ws/predict`

Stream JPEG frames for continuous, real-time predictions. Shares the same 128-frame buffer logic as the REST frame endpoint.

**Send** — raw Base64-encoded JPEG string (one frame per message)

**Receive — buffering**
```json
{
  "status": "buffering",
  "fill_pct": 30,
  "frames_seen": 38
}
```

**Receive — prediction** (emitted every 128 frames)
```json
{
  "status": "prediction",
  "label": "REAL",
  "confidence": 94.1,
  "fake_prob": 0.058,
  "frames_seen": 256,
  "segments_seen": 3
}
```

`segments_seen` reflects how many complete 128-frame windows have been evaluated in this session.

---

## Reference

### Prediction Labels

| Label | Meaning |
|-------|---------|
| `REAL` | Signal is consistent with authentic physiological patterns |
| `FAKE` | Signal is inconsistent — deepfake likely |
| `UNCERTAIN` | Insufficient signal quality to classify (poor lighting, face not visible, etc.) |

---

### Rate Limits & Constraints

| Limit | Value |
|-------|-------|
| Video uploads | 30 requests / minute |
| Max video file size | 50 MB |
| Concurrent video jobs | 4 |
| Job retention (TTL) | 2 hours |
| Min frames for prediction | 128 (~4s at 30 fps) |
| Supported video formats | `.mp4`, `.avi`, `.mov`, `.mkv` |

---

### Error Responses

All errors follow standard HTTP status codes with a `detail` field.

| Code | Meaning | Example trigger |
|------|---------|-----------------|
| `400` | Bad request | Malformed Base64, missing field |
| `404` | Not found | Unknown or expired `job_id` |
| `413` | Payload too large | Video exceeds 50 MB |
| `429` | Rate limit exceeded | > 30 uploads/minute |
| `500` | Internal server error | Model crash |
| `503` | Service unavailable | Workers exhausted |

**Example**
```json
{
  "detail": "Unsupported format '.wmv'. Use: .mp4, .avi, .mov, .mkv"
}
```

---

## Architecture Overview

```
Input (frame / video / stream)
        │
        ▼
  1. Face Detection
        │
        ▼
  2. Signal Extraction (rPPG)
        │
        ▼
  3. rPPG Map Generation
        │
        ▼
  4. CNN Classification
        │
        ▼
  5. Result Aggregation  ──►  REAL / FAKE / UNCERTAIN
```

---

## Tips for Best Accuracy

- Ensure the subject's face is **clearly visible and well-lit**
- Videos should be **at least 4 seconds** (128 frames minimum)
- Avoid heavy compression artifacts — they degrade rPPG signal quality
- `face_pct` in the video result indicates how reliably the face was tracked; values below ~70% may produce less reliable predictions