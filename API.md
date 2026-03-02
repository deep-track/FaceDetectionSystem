# DeepTrack API Reference

**Version:** 1.0 | **Base URL:** `/` | **Protocol:** HTTP / WebSocket | **Auth:** None required

DeepTrack is a unified deepfake detection API with two independent analysis engines:

- **`/v1/video/*`** — FakeCatcher: rPPG biological signal analysis via CNN (webcam & video upload)
- **`/v1/image/*`** — DeepTrack: Swin Transformer visual classification (single image upload)

## Table of Contents

- [Endpoint Overview](#endpoint-overview)
- [System Endpoints](#system-endpoints)
- [Video API](#video-api)
  - [Frame Prediction](#frame-prediction)
  - [Video Upload](#video-upload)
  - [Job Management](#job-management)
  - [WebSocket Streaming](#websocket-streaming)
- [Image API](#image-api)
  - [Image Prediction](#image-prediction)
- [Reference](#reference)

## Endpoint Overview

### System

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Landing page with links to both demo UIs |
| `GET` | `/v1/health` | Liveness check — reports status of both models |
| `GET` | `/v1/status` | Detailed server status, buffer state, job queue |
| `GET` | `/docs` | Interactive Swagger UI |

### Video (`/v1/video/*`)

| Method | Endpoint | Description | Key Constraints |
|--------|----------|-------------|-----------------|
| `GET` | `/v1/video/` | Video demo UI (webcam + upload tabs) | — |
| `POST` | `/v1/video/predict/frame` | Submit a single Base64-encoded JPEG frame | Buffers 128 frames before predicting (~4s) |
| `POST` | `/v1/video/predict/video` | Upload a video file for async analysis | `.mp4 .avi .mov .mkv` only; max 50 MB |
| `GET` | `/v1/video/jobs/{job_id}` | Poll status and result of a video job | Job TTL: 2 hours |
| `GET` | `/v1/video/jobs` | List all active jobs | — |
| `POST` | `/v1/video/reset` | Clear the frame buffer | — |
| `WS` | `/v1/video/ws` | Stream frames for real-time prediction | Same 128-frame buffer requirement |

### Image (`/v1/image/*`)

| Method | Endpoint | Description | Key Constraints |
|--------|----------|-------------|-----------------|
| `GET` | `/v1/image/` | Image demo UI | — |
| `POST` | `/v1/image/predict` | Classify a single image as Real or Fake | JPEG, PNG, WEBP, BMP |

## System Endpoints

### `GET /v1/health`

Lightweight liveness check. Reports operational status of both models independently — one model being unavailable does not affect the other.

```json
{
  "status": "ok",
  "uptime_seconds": 3842,
  "video_model": "operational",
  "image_model": "operational"
}
```

`video_model` or `image_model` will be `"unavailable"` if the respective model failed to load at startup. Use this endpoint for load-balancer health probes.

### `GET /v1/status`

Detailed server status including frame buffer state and job queue.

```json
{
  "uptime_seconds": 3842,
  "video": {
    "model_loaded": true,
    "buffer_fill_pct": 50,
    "frames_seen": 64,
    "jobs": {
      "queued": 1,
      "processing": 2
    },
    "total_jobs": 3,
    "max_workers": 4,
    "max_upload_mb": 50,
    "job_ttl_hours": 2
  },
  "image": {
    "model_loaded": true
  }
}
```


## Video API

### Frame Prediction

#### `POST /v1/video/predict/frame`

Submit one JPEG frame at a time. Predictions are emitted after every 128-frame buffer is filled (~4 seconds at 30 fps). Subsequent frames continue the rolling buffer with 50% overlap.

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

> `fake_prob` is a float in `[0, 1]`. `confidence` is derived from the distance to the 0.5 decision boundary, expressed as a percentage.

### Video Upload

#### `POST /v1/video/predict/video`

Upload a video file for asynchronous deepfake analysis. Returns immediately with a `job_id`; poll `/v1/video/jobs/{job_id}` for results.

**Request** — `multipart/form-data`

| Field | Type | Notes |
|-------|------|-------|
| `file` | binary | Accepted: `.mp4`, `.avi`, `.mov`, `.mkv`; max 50 MB |

**Response — job queued** `200`
```json
{
  "job_id": "3f2a1b4c-...",
  "status": "queued",
  "filename": "video.mp4",
  "size_mb": 12.5,
  "poll_url": "/v1/video/jobs/3f2a1b4c-..."
}
```

Up to 4 jobs process concurrently. Additional jobs wait in a queue.

### Job Management

#### `GET /v1/video/jobs/{job_id}`

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
    "n_segments": 12,
    "segments": [
      {
        "segment": 1,
        "start_sec": 0.0,
        "end_sec": 4.3,
        "label": "FAKE",
        "confidence": 91.2,
        "fake_prob": 0.912
      }
    ]
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
| `done` | Completed — `result` is populated |
| `error` | Failed — `error` contains the reason |

**Result fields**

| Field | Type | Description |
|-------|------|-------------|
| `label` | string | `REAL`, `FAKE`, or `UNCERTAIN` |
| `confidence` | float | Confidence percentage (0–100) |
| `fake_prob` | float | Raw model probability (0–1) |
| `total_frames` | int | Total frames analysed |
| `face_pct` | float | % of frames with a detectable face |
| `n_segments` | int | Number of 128-frame segments evaluated |
| `segments` | array | Per-segment breakdown with timestamps |

> **TTL:** Jobs expire 2 hours after creation. After expiry, `GET /v1/video/jobs/{job_id}` returns `404`.

#### `GET /v1/video/jobs`

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

#### `POST /v1/video/reset`

Clears the shared frame buffer. Use when switching video sources mid-session.

```json
{ "status": "reset" }
```

> **Warning:** This affects all concurrent callers sharing the same buffer. Not safe for multi-tenant deployments without per-client buffer isolation. The WebSocket endpoint (`/v1/video/ws`) uses per-connection buffers and is unaffected.

### WebSocket Streaming

#### `WS /v1/video/ws`

Stream JPEG frames for continuous real-time predictions. Each WebSocket connection maintains its own isolated frame buffer — independent of the REST frame endpoint and other connections.

**Send** — raw Base64-encoded JPEG string (one frame per message)

**Receive — buffering**
```json
{
  "status": "buffering",
  "fill_pct": 30,
  "frames_seen": 38
}
```

**Receive — prediction**
```json
{
  "status": "prediction",
  "label": "REAL",
  "confidence": 94.1,
  "fake_prob": 0.058,
  "warning": null,
  "frames_seen": 256,
  "segments_seen": 3
}
```

**Receive — uncertain signal**
```json
{
  "status": "prediction",
  "label": "UNCERTAIN",
  "confidence": 0.0,
  "fake_prob": null,
  "warning": "Face not detected in 45% of frames — check lighting and camera angle",
  "frames_seen": 128,
  "segments_seen": 1
}
```

`segments_seen` reflects how many complete 128-frame windows have been evaluated in this connection session. Predictions are smoothed across the last 3 segments to reduce noise.


## Image API

### Image Prediction

#### `POST /v1/image/predict`

Classify a single image as Real or Fake using the Swin Transformer model.

**Request** — `multipart/form-data`

| Field | Type | Notes |
|-------|------|-------|
| `file` | binary | Accepted: JPEG, PNG, WEBP, BMP |

**Response**
```json
{
  "filename": "photo.jpg",
  "prediction": "Fake",
  "confidence_percentage": 94.27,
  "raw_scores": {
    "Real": 5.73,
    "Fake": 94.27
  }
}
```

**Response fields**

| Field | Type | Description |
|-------|------|-------------|
| `filename` | string | Original uploaded filename |
| `prediction` | string | `Real` or `Fake` |
| `confidence_percentage` | float | Confidence of the winning class (0–100) |
| `raw_scores.Real` | float | Softmax probability for Real class (0–100) |
| `raw_scores.Fake` | float | Softmax probability for Fake class (0–100) |

> `raw_scores` always sum to 100. `confidence_percentage` equals the higher of the two scores.

## Reference

### Prediction Labels

#### Video (`/v1/video/*`)

| Label | Meaning |
|-------|---------|
| `REAL` | rPPG signal consistent with authentic physiological patterns |
| `FAKE` | rPPG signal inconsistent — deepfake likely |
| `UNCERTAIN` | Insufficient signal quality to classify (poor lighting, face not detected, low SNR) |

#### Image (`/v1/image/*`)

| Label | Meaning |
|-------|---------|
| `Real` | Visual features consistent with an authentic image |
| `Fake` | Visual features consistent with a deepfake |

---

### Constraints

| Constraint | Value |
|------------|-------|
| Max video file size | 50 MB |
| Concurrent video jobs | 4 |
| Job retention (TTL) | 2 hours |
| Min frames for video prediction | 128 (~4s at 30 fps) |
| Supported video formats | `.mp4`, `.avi`, `.mov`, `.mkv` |
| Supported image formats | JPEG, PNG, WEBP, BMP |
| Frame buffer overlap | 50% (new prediction every 64 frames) |
| WebSocket smoothing window | 3 segments |

---

### Error Responses

All errors follow standard HTTP status codes with a `detail` field.

| Code | Meaning | Example trigger |
|------|---------|-----------------|
| `400` | Bad request | Malformed Base64, unsupported file format |
| `404` | Not found | Unknown or expired `job_id` |
| `413` | Payload too large | Video exceeds 50 MB |
| `500` | Internal server error | Model inference failure |
| `503` | Service unavailable | Model not loaded at startup |

**Example**
```json
{
  "detail": "Unsupported format '.wmv'. Use: .mp4, .avi, .mov, .mkv"
}
```

---

## Architecture Overview

### Video Pipeline (`/v1/video/*`)

```
Input (frame / video / WebSocket stream)
        │
        ▼
  1. Face Detection (MediaPipe FaceLandmarker)
        │
        ▼
  2. Per-subregion RGB Signal Extraction (32 facial landmarks)
        │
        ▼
  3. CHROM rPPG + PSD Map Generation (128 × 64)
        │
        ▼
  4. Signal Quality Check (SNR, face detection rate)
        │
        ▼
  5. CNN Classification
        │
        ▼
  6. Segment Aggregation + Smoothing
        │
        ▼
  REAL / FAKE / UNCERTAIN
```

### Image Pipeline (`/v1/image/*`)

```
Input (JPEG / PNG / WEBP / BMP)
        │
        ▼
  1. Resize to 224×224, ImageNet normalisation
        │
        ▼
  2. Swin Transformer (swin_small_patch4_window7_224)
        │
        ▼
  3. Custom classifier head (LayerNorm → 512 → 2)
        │
        ▼
  4. Softmax
        │
        ▼
  Real / Fake
```

## Tips for Best Accuracy

### Video
- Ensure the subject's face is **clearly visible and well-lit**
- Videos should be **at least 4 seconds** (128 frames minimum)
- Avoid heavy compression artifacts — they degrade the rPPG signal
- `face_pct` below ~70% in the video result indicates unreliable face tracking and may reduce accuracy
- For webcam use, select the correct camera source before starting — switching mid-session requires a buffer reset

### Image
- Use **high-resolution, uncompressed** images where possible
- The model was trained on face-centric images — full-face crops produce the most reliable results
- Heavy JPEG compression or social-media resizing may reduce confidence