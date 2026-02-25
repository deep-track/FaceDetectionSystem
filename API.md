# FakeCatcher API Documentation

**Version:** 1.0\
**Base URL:** `/`\
**Protocol:** HTTP / WebSocket\
**Content-Type:** `application/json` (unless otherwise specified)

FakeCatcher API provides deepfake detection using remote
photoplethysmography (rPPG) analysis and a CNN model. It supports:

-   Real-time frame prediction (REST & WebSocket)
-   Video upload and asynchronous processing
-   Job tracking and management
-   Health and system monitoring

------------------------------------------------------------------------

# Table of Contents

-   [Overview](#overview)
-   [Authentication](#authentication)
-   [Rate Limits](#rate-limits)
-   [Endpoints Summary](#endpoints-summary)
-   [Frame Prediction](#frame-prediction)
-   [Video Prediction](#video-prediction)
-   [Job Management](#job-management)
-   [System Endpoints](#system-endpoints)
-   [WebSocket Endpoint](#websocket-endpoint)
-   [Prediction Labels](#prediction-labels)
-   [Error Handling](#error-handling)

------------------------------------------------------------------------

# Overview

FakeCatcher analyzes subtle physiological signals extracted from facial
regions to detect deepfake videos and images.

Prediction types:

  Type     Input         Output
  -------- ------------- -----------------------------
  Frame    base64 JPEG   REAL / FAKE / UNCERTAIN
  Video    video file    Full analysis with segments
  Stream   live frames   Continuous predictions

------------------------------------------------------------------------

# Authentication

Authentication has been removed and is not required.

------------------------------------------------------------------------

# Rate Limits

Default limits:

  Limit                   Value
  ----------------------- ---------------
  Video upload requests   30 per minute
  Max video size          50 MB
  Concurrent video jobs   4
  Job retention           2 hours

------------------------------------------------------------------------

# Endpoints Summary

  Method   Endpoint           Description
  -------- ------------------ -----------------------------
  POST     `/predict/frame`   Submit frame for prediction
  POST     `/predict/video`   Upload video for analysis
  GET      `/jobs/{job_id}`   Get video job result
  GET      `/jobs`            List all jobs
  GET      `/status`          Frame buffer status
  POST     `/reset`           Reset frame buffer
  GET      `/health`          Health check
  GET      `/metrics`         System metrics
  WS       `/ws/predict`      Real-time prediction stream

------------------------------------------------------------------------

# Frame Prediction

## POST /predict/frame

Submit a single frame for prediction.

Prediction occurs after 128 frames are buffered.

## Request

``` json
{
  "image": "base64_encoded_jpeg"
}
```

## Response (Buffering)

``` json
{
  "status": "buffering",
  "fill_pct": 42,
  "frames_seen": 54,
  "message": "Buffering... 42% — need 128 frames (~4s)"
}
```

## Response (Prediction)

``` json
{
  "status": "prediction",
  "label": "REAL",
  "confidence": 92.3,
  "fake_prob": 0.0765,
  "frames_seen": 128
}
```

------------------------------------------------------------------------

# Video Prediction

## POST /predict/video

Uploads a video for analysis.

Returns immediately with job ID.

## Request

Content-Type: multipart/form-data

Field:

    file: video file (.mp4, .avi, .mov, .mkv)

## Response

``` json
{
  "job_id": "uuid",
  "status": "queued",
  "filename": "video.mp4",
  "size_mb": 12.5,
  "poll_url": "/jobs/uuid",
  "message": "Job queued. Poll /jobs/{job_id} for result."
}
```

------------------------------------------------------------------------

# Job Management

## GET /jobs/{job_id}

Retrieve job status.

## Response

``` json
{
  "job_id": "uuid",
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

## Job Status Values

  Status       Description
  ------------ -------------------------
  queued       Waiting to be processed
  processing   Currently analyzing
  done         Completed successfully
  error        Failed

------------------------------------------------------------------------

## GET /jobs

List all active jobs.

``` json
{
  "jobs": {
    "uuid": {
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

------------------------------------------------------------------------

# System Endpoints

## GET /status

Frame buffer status.

``` json
{
  "buffer_fill_pct": 50,
  "frames_seen": 64,
  "omega": 128,
  "last_result": null,
  "model": "data/cnn_model.keras"
}
```

------------------------------------------------------------------------

## POST /reset

Reset frame buffer.

``` json
{
  "status": "reset",
  "message": "Frame buffer cleared."
}
```

------------------------------------------------------------------------

## GET /health

Check system health.

``` json
{
  "status": "ok",
  "model": "operational",
  "server": "operational"
}
```

------------------------------------------------------------------------

## GET /metrics

System metrics.

``` json
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

------------------------------------------------------------------------

# WebSocket Endpoint

## WS /ws/predict

Real-time prediction.

## Send

Base64 JPEG string

## Receive (Buffering)

``` json
{
  "status": "buffering",
  "fill_pct": 30,
  "frames_seen": 38
}
```

## Receive (Prediction)

``` json
{
  "status": "prediction",
  "label": "REAL",
  "confidence": 94.1,
  "fake_prob": 0.058,
  "frames_seen": 256,
  "segments_seen": 3
}
```

------------------------------------------------------------------------

# Prediction Labels

  Label       Meaning
  ----------- -----------------------------
  REAL        Likely authentic
  FAKE        Likely deepfake
  UNCERTAIN   Insufficient signal quality

------------------------------------------------------------------------

# Error Handling

Errors follow HTTP status codes.

  Code   Meaning
  ------ ---------------------
  400    Bad request
  404    Resource not found
  413    File too large
  429    Rate limit exceeded
  500    Internal error
  503    Service unavailable

Example:

``` json
{
  "detail": "Unsupported format '.wmv'. Use: .mp4, .avi, .mov, .mkv"
}
```

------------------------------------------------------------------------

# Architecture Overview

Pipeline:

1.  Face detection
2.  Signal extraction
3.  rPPG map generation
4.  CNN classification
5.  Result aggregation

------------------------------------------------------------------------

# Notes

-   Minimum frames required: 128
-   Minimum video length: \~4 seconds
-   Face must be visible clearly
-   Better lighting improves accuracy

------------------------------------------------------------------------
