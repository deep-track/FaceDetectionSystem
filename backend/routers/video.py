import cv2
import base64
import logging
import os
import tempfile
import time
import uuid
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from core.auth import verify_api_key
from core.limiter import limiter, get_api_limit
from core.video_predictor import (
    FrameBuffer, N_SUBREGIONS, OMEGA, FS,
    build_ppg_map, classify_prob, signal_quality,
)

logger = logging.getLogger("deeptrack.video")
router = APIRouter()

MAX_UPLOAD_MB     = 50
MAX_VIDEO_WORKERS = 4
JOB_TTL_SECONDS   = 7200

video_executor = ThreadPoolExecutor(
    max_workers=MAX_VIDEO_WORKERS,
    thread_name_prefix="deeptrack-video",
)


class FrameRequest(BaseModel):
    image: str  # base64-encoded JPEG


def _purge_old_jobs(jobs: dict):
    cutoff = time.time() - JOB_TTL_SECONDS
    stale  = [jid for jid, j in jobs.items() if j["created_at"] < cutoff]
    for jid in stale:
        del jobs[jid]
    if stale:
        logger.info(f"Purged {len(stale)} stale jobs")


def _run_video_job(job_id: str, tmp_path: str, predictor, jobs: dict):
    try:
        jobs[job_id]["status"] = "processing"
        result = predictor.predict_video(tmp_path)
        jobs[job_id].update({"status": "done", "result": result})
        logger.info(f"Job {job_id} done: {result['label']} ({result['confidence']}%)")
    except ValueError as e:
        jobs[job_id].update({"status": "error", "error": str(e)})
        logger.warning(f"Job {job_id} validation error: {e}")
    except Exception as e:
        jobs[job_id].update({"status": "error", "error": f"Processing error: {e}"})
        logger.error(f"Job {job_id} failed: {e}")
    finally:
        for _ in range(5):
            try:
                os.unlink(tmp_path)
                break
            except PermissionError:
                time.sleep(0.2)


'''
protected endpoints, they require X-API-KEY to access
'''

@router.post("/predict/video")
@limiter.limit(get_api_limit)
async def predict_video(
    request: Request,
    file: UploadFile = File(...),
    _key: dict = Depends(verify_api_key),
):
    predictor = request.app.state.video_predictor
    jobs      = request.app.state.jobs
    if predictor is None:
        raise HTTPException(503, "Video model not loaded.")

    allowed = {".mp4", ".avi", ".mov", ".mkv"}
    ext     = os.path.splitext(file.filename or "")[1].lower()
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported format '{ext}'. Use: {', '.join(allowed)}")

    data = await file.read()
    mb   = len(data) / 1024 / 1024
    if mb > MAX_UPLOAD_MB:
        raise HTTPException(413, f"File too large ({mb:.1f}MB). Max is {MAX_UPLOAD_MB}MB.")

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    _purge_old_jobs(jobs)
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "queued", "filename": file.filename,
        "size_mb": round(mb, 2), "created_at": time.time(),
        "result": None, "error": None,
    }
    video_executor.submit(_run_video_job, job_id, tmp_path, predictor, jobs)
    logger.info(f"Job {job_id} queued: {file.filename} ({mb:.1f}MB)")

    return {"job_id": job_id, "status": "queued", "filename": file.filename,
            "size_mb": round(mb, 2), "poll_url": f"/v1/video/jobs/{job_id}"}


@router.get("/jobs/{job_id}")
async def get_job(
    job_id: str,
    request: Request,
    _key: dict = Depends(verify_api_key),
):
    jobs = request.app.state.jobs
    job  = jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found.")
    return {"job_id": job_id, "status": job["status"], "filename": job["filename"],
            "size_mb": job["size_mb"], "result": job.get("result"),
            "error": job.get("error"), "age_sec": round(time.time() - job["created_at"])}


@router.get("/jobs")
async def list_jobs(
    request: Request,
    _key: dict = Depends(verify_api_key),
):
    jobs = request.app.state.jobs
    summary = {
        jid: {"status": j["status"], "filename": j["filename"],
              "age_sec": round(time.time() - j["created_at"])}
        for jid, j in jobs.items()
    }
    counts = {}
    for j in jobs.values():
        counts[j["status"]] = counts.get(j["status"], 0) + 1
    return {"jobs": summary, "counts": counts, "total": len(jobs)}

'''
open endpoints, don't require keys
'''

@router.post("/predict/frame")
async def predict_frame(req: FrameRequest, request: Request):
    predictor = request.app.state.video_predictor
    buffer    = request.app.state.video_buffer
    if predictor is None:
        raise HTTPException(503, "Video model not loaded.")
    try:
        img_bytes = base64.b64decode(req.image)
        arr       = np.frombuffer(img_bytes, dtype=np.uint8)
        frame     = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(400, "Could not decode image")
    except Exception as e:
        raise HTTPException(400, f"Bad image data: {e}")

    rgb_vals = predictor.extract_frame_rgb(frame)
    buffer.push(rgb_vals)

    if buffer.ready():
        ppg_map = buffer.get_map()
        result  = predictor.predict_map(ppg_map)
        buffer.last_result = result
        return {"status": "prediction", "label": result["label"],
                "confidence": result["confidence"], "fake_prob": result["fake_prob"],
                "frames_seen": buffer.frame_count}

    return {"status": "buffering", "fill_pct": buffer.fill_pct,
            "frames_seen": buffer.frame_count,
            "message": f"Buffering... {buffer.fill_pct}% — need {OMEGA} frames (~{OMEGA//FS}s)"}


@router.post("/reset")
async def reset(request: Request):
    request.app.state.video_buffer.reset()
    return {"status": "reset"}


@router.websocket("/ws")
async def websocket_predict(ws: WebSocket):
    await ws.accept()
    predictor = ws.app.state.video_predictor
    ws_buffer = FrameBuffer()
    logger.info("WebSocket client connected")
    try:
        while True:
            data = await ws.receive_text()
            try:
                img_bytes = base64.b64decode(data)
                arr       = np.frombuffer(img_bytes, dtype=np.uint8)
                frame     = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is None:
                    await ws.send_json({"error": "Could not decode frame"})
                    continue
            except Exception as e:
                await ws.send_json({"error": str(e)})
                continue

            rgb_vals = predictor.extract_frame_rgb(frame)
            ws_buffer.push(rgb_vals)

            if ws_buffer.ready():
                ppg_map = ws_buffer.get_map()
                q_ok, q_reason, snr = signal_quality(
                    [list(ws_buffer.R[i]) for i in range(N_SUBREGIONS)],
                    [list(ws_buffer.G[i]) for i in range(N_SUBREGIONS)],
                    [list(ws_buffer.B[i]) for i in range(N_SUBREGIONS)],
                )
                result = predictor.predict_map(ppg_map, q_ok, q_reason, snr)
                if result["fake_prob"] is not None:
                    ws_buffer.prob_history.append(result["fake_prob"])
                smoothed = float(np.mean(ws_buffer.prob_history)) if ws_buffer.prob_history else None
                if smoothed is not None and result["label"] != "UNCERTAIN":
                    label, conf          = classify_prob(smoothed)
                    result["label"]      = label
                    result["confidence"] = conf
                    result["fake_prob"]  = round(smoothed, 4)
                ws_buffer.last_result = result
                await ws.send_json({
                    "status": "prediction", "label": result["label"],
                    "confidence": result["confidence"], "fake_prob": result["fake_prob"],
                    "warning": result.get("warning"), "frames_seen": ws_buffer.frame_count,
                    "segments_seen": len(ws_buffer.prob_history),
                })
            else:
                await ws.send_json({
                    "status": "buffering", "fill_pct": ws_buffer.fill_pct,
                    "frames_seen": ws_buffer.frame_count,
                })
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")


@router.get("/", response_class=HTMLResponse)
async def video_ui():
    return r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>DeepTrack — Video Analysis</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg:        #080b0f;
      --surface:   #0e1318;
      --border:    #1c2530;
      --border2:   #243040;
      --text:      #c8d8e8;
      --muted:     #4a6070;
      --accent:    #00aaff;
      --accent2:   #0066cc;
      --real:      #00e87a;
      --fake:      #ff3355;
      --uncertain: #ffaa00;
      --mono:      'IBM Plex Mono', monospace;
      --sans:      'IBM Plex Sans', sans-serif;
    }
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg); color: var(--text); font-family: var(--sans);
      font-size: 14px; min-height: 100vh; display: flex;
      flex-direction: column; align-items: center; padding: 0 0 60px;
    }
    header {
      width: 100%; border-bottom: 1px solid var(--border);
      padding: 18px 40px; display: flex; align-items: center;
      gap: 16px; background: var(--surface);
    }
    .logo-mark {
      width: 32px; height: 32px; border: 2px solid var(--accent); border-radius: 4px;
      display: flex; align-items: center; justify-content: center;
      font-family: var(--mono); font-size: 13px; font-weight: 600;
      color: var(--accent); letter-spacing: -1px;
    }
    .logo-text { font-family: var(--mono); font-size: 15px; font-weight: 600; color: var(--text); letter-spacing: 0.08em; }
    .logo-text span { color: var(--accent); }
    .header-badge {
      margin-left: auto; font-family: var(--mono); font-size: 10px; color: var(--muted);
      border: 1px solid var(--border2); border-radius: 3px; padding: 3px 8px;
      letter-spacing: 0.1em; text-transform: uppercase;
    }
    main { width: 100%; max-width: 860px; padding: 32px 20px 0; }
    .page-title { font-family: var(--mono); font-size: 11px; color: var(--muted); letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 24px; }
    .tabs { display: flex; border-bottom: 1px solid var(--border); margin-bottom: 28px; }
    .tab-btn {
      font-family: var(--mono); font-size: 12px; font-weight: 500; letter-spacing: 0.08em;
      text-transform: uppercase; color: var(--muted); background: none; border: none;
      border-bottom: 2px solid transparent; padding: 10px 20px; cursor: pointer;
      transition: color .15s, border-color .15s; margin-bottom: -1px;
    }
    .tab-btn:hover { color: var(--text); }
    .tab-btn.active { color: var(--accent); border-bottom-color: var(--accent); }
    .tab-panel { display: none; }
    .tab-panel.active { display: block; }
    .card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 24px; margin-bottom: 16px; }
    .card-label { font-family: var(--mono); font-size: 10px; color: var(--muted); letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 14px; }
    .verdict-block {
      display: flex; align-items: center; gap: 20px; padding: 20px;
      background: var(--bg); border: 1px solid var(--border); border-radius: 6px;
      margin-bottom: 20px; min-height: 72px;
    }
    .verdict-icon { width: 44px; height: 44px; border-radius: 6px; display: flex; align-items: center; justify-content: center; font-size: 20px; flex-shrink: 0; }
    .verdict-icon.REAL      { background: rgba(0,232,122,.12); border: 1px solid rgba(0,232,122,.3); }
    .verdict-icon.FAKE      { background: rgba(255,51,85,.12);  border: 1px solid rgba(255,51,85,.3); }
    .verdict-icon.UNCERTAIN { background: rgba(255,170,0,.12);  border: 1px solid rgba(255,170,0,.3); }
    .verdict-icon.idle      { background: var(--surface); border: 1px solid var(--border); }
    .verdict-label { font-family: var(--mono); font-size: 22px; font-weight: 600; letter-spacing: 0.05em; }
    .verdict-label.REAL      { color: var(--real); }
    .verdict-label.FAKE      { color: var(--fake); }
    .verdict-label.UNCERTAIN { color: var(--uncertain); }
    .verdict-label.idle      { color: var(--muted); font-size: 14px; font-weight: 400; }
    .verdict-sub { font-family: var(--mono); font-size: 11px; color: var(--muted); margin-top: 4px; }
    .prob-row { display: flex; align-items: center; gap: 12px; margin-bottom: 10px; }
    .prob-key { font-family: var(--mono); font-size: 10px; font-weight: 600; letter-spacing: 0.1em; width: 52px; text-transform: uppercase; }
    .prob-key.real { color: var(--real); }
    .prob-key.fake { color: var(--fake); }
    .prob-track { flex: 1; height: 6px; background: var(--border); border-radius: 3px; overflow: hidden; }
    .prob-fill { height: 100%; border-radius: 3px; width: 0%; transition: width .4s cubic-bezier(.4,0,.2,1); }
    .prob-fill.real { background: var(--real); }
    .prob-fill.fake { background: var(--fake); }
    .prob-pct { font-family: var(--mono); font-size: 11px; color: var(--muted); width: 40px; text-align: right; }
    .buffer-row { display: flex; align-items: center; gap: 12px; margin-bottom: 14px; }
    .buffer-label { font-family: var(--mono); font-size: 10px; color: var(--muted); width: 52px; text-transform: uppercase; letter-spacing: 0.08em; }
    .buffer-track { flex: 1; height: 3px; background: var(--border); border-radius: 2px; overflow: hidden; }
    .buffer-fill { height: 100%; background: var(--accent); border-radius: 2px; width: 0%; transition: width .3s ease; }
    .buffer-pct { font-family: var(--mono); font-size: 11px; color: var(--muted); width: 36px; text-align: right; }
    .meta-row { display: flex; gap: 24px; flex-wrap: wrap; }
    .meta-item { display: flex; flex-direction: column; gap: 2px; }
    .meta-key { font-family: var(--mono); font-size: 9px; color: var(--muted); letter-spacing: 0.12em; text-transform: uppercase; }
    .meta-val { font-family: var(--mono); font-size: 13px; color: var(--text); }
    #video { width: 100%; border-radius: 6px; border: 1px solid var(--border); display: block; background: #000; aspect-ratio: 4/3; object-fit: cover; }
    .cam-row { display: flex; align-items: center; gap: 10px; margin-bottom: 14px; }
    .cam-row label { font-family: var(--mono); font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.1em; white-space: nowrap; }
    select {
      flex: 1; background: var(--bg); color: var(--text); border: 1px solid var(--border2);
      border-radius: 5px; padding: 7px 28px 7px 10px; font-family: var(--sans); font-size: 13px;
      cursor: pointer; outline: none; appearance: none; -webkit-appearance: none;
      background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6' fill='none'%3E%3Cpath d='M1 1l4 4 4-4' stroke='%234a6070' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E");
      background-repeat: no-repeat; background-position: right 10px center;
    }
    select:focus { border-color: var(--accent2); }
    .btn-row { display: flex; gap: 10px; margin-top: 16px; }
    .btn { font-family: var(--mono); font-size: 12px; font-weight: 500; letter-spacing: 0.08em; text-transform: uppercase; border: none; border-radius: 5px; padding: 9px 20px; cursor: pointer; transition: background .15s, opacity .15s; }
    .btn-primary { background: var(--accent); color: #000; }
    .btn-primary:hover { background: #22bbff; }
    .btn-primary:disabled { opacity: .4; cursor: not-allowed; }
    .btn-secondary { background: var(--border2); color: var(--text); }
    .btn-secondary:hover { background: #2e3d50; }
    .btn-full { width: 100%; text-align: center; }
    .drop-zone { border: 1.5px dashed var(--border2); border-radius: 8px; padding: 36px 20px; text-align: center; cursor: pointer; transition: border-color .2s, background .2s; margin-bottom: 14px; }
    .drop-zone:hover, .drop-zone.dragover { border-color: var(--accent); background: rgba(0,170,255,.04); }
    .drop-zone .dz-icon { font-size: 28px; margin-bottom: 10px; opacity: .5; }
    .drop-zone p { font-size: 13px; color: var(--muted); line-height: 1.6; }
    .drop-zone strong { color: var(--text); }
    #file-name { font-family: var(--mono); font-size: 11px; color: var(--accent); margin-bottom: 12px; min-height: 16px; }
    #file-input { display: none; }
    .spinner { display: none; width: 20px; height: 20px; border: 2px solid var(--border2); border-top-color: var(--accent); border-radius: 50%; animation: spin .7s linear infinite; margin: 0 auto 16px; }
    @keyframes spin { to { transform: rotate(360deg); } }
    .seg-section { margin-top: 20px; }
    .seg-table { width: 100%; border-collapse: collapse; font-size: 12px; }
    .seg-table th { font-family: var(--mono); font-size: 9px; font-weight: 500; color: var(--muted); letter-spacing: 0.12em; text-transform: uppercase; padding: 8px 10px; text-align: left; border-bottom: 1px solid var(--border); }
    .seg-table td { padding: 8px 10px; border-bottom: 1px solid var(--border); font-family: var(--mono); font-size: 12px; color: var(--text); }
    .seg-table tr:last-child td { border-bottom: none; }
    .seg-table tr.row-fake      td { background: rgba(255,51,85,.04); }
    .seg-table tr.row-real      td { background: rgba(0,232,122,.03); }
    .seg-table tr.row-uncertain td { background: rgba(255,170,0,.04); }
    .pill { display: inline-block; font-family: var(--mono); font-size: 10px; font-weight: 600; letter-spacing: 0.08em; padding: 2px 8px; border-radius: 3px; }
    .pill.REAL      { background: rgba(0,232,122,.15); color: var(--real); }
    .pill.FAKE      { background: rgba(255,51,85,.15);  color: var(--fake); }
    .pill.UNCERTAIN { background: rgba(255,170,0,.15);  color: var(--uncertain); }
    .tip { font-size: 11px; color: var(--muted); margin-top: 14px; line-height: 1.5; }
    .tip strong { color: var(--text); }
  </style>
</head>
<body>
<header>
  <div class="logo-mark">DT</div>
  <div class="logo-text">Deep<span>Track</span></div>
  <div class="header-badge">rPPG Engine v1</div>
</header>
<main>
  <div class="page-title">// Video Analysis</div>
  <div class="tabs">
    <button class="tab-btn active" onclick="switchTab('webcam', this)">Live Webcam</button>
    <button class="tab-btn"        onclick="switchTab('upload', this)">Upload Video</button>
  </div>
  <div class="tab-panel active" id="tab-webcam">
    <div class="card">
      <div class="card-label">Camera Input</div>
      <div class="cam-row">
        <label>Source</label>
        <select id="cam-sel"><option>Click Start to detect cameras</option></select>
        <button class="btn btn-secondary" style="padding:7px 12px;font-size:11px" onclick="refreshCameras()">Refresh</button>
      </div>
      <video id="video" autoplay muted playsinline></video>
    </div>
    <div class="card">
      <div class="card-label">Detection</div>
      <div class="verdict-block">
        <div class="verdict-icon idle" id="wc-verdict-icon">⬡</div>
        <div>
          <div class="verdict-label idle" id="wc-verdict-label">Awaiting signal</div>
          <div class="verdict-sub" id="wc-verdict-sub">Select a camera and press Start</div>
        </div>
      </div>
      <div class="buffer-row">
        <span class="buffer-label">Buffer</span>
        <div class="buffer-track"><div class="buffer-fill" id="wc-buf-fill"></div></div>
        <span class="buffer-pct" id="wc-buf-pct">0%</span>
      </div>
      <div id="wc-probs" style="display:none">
        <div class="prob-row">
          <span class="prob-key real">Real</span>
          <div class="prob-track"><div class="prob-fill real" id="wc-real-fill"></div></div>
          <span class="prob-pct" id="wc-real-pct">—</span>
        </div>
        <div class="prob-row">
          <span class="prob-key fake">Fake</span>
          <div class="prob-track"><div class="prob-fill fake" id="wc-fake-fill"></div></div>
          <span class="prob-pct" id="wc-fake-pct">—</span>
        </div>
      </div>
      <div class="meta-row" id="wc-meta" style="margin-top:16px; display:none">
        <div class="meta-item"><span class="meta-key">Frames</span><span class="meta-val" id="wc-frames">—</span></div>
        <div class="meta-item"><span class="meta-key">P(fake)</span><span class="meta-val" id="wc-pfake">—</span></div>
        <div class="meta-item"><span class="meta-key">Segments</span><span class="meta-val" id="wc-segs">—</span></div>
      </div>
      <div class="btn-row">
        <button class="btn btn-primary" id="startBtn" onclick="startCapture()">▶ Start</button>
        <button class="btn btn-secondary" onclick="resetBuffer()">↺ Reset</button>
      </div>
      <p class="tip"><strong>Tip:</strong> For deepfake filter testing, select OBS Virtual Camera with DeepFaceLive running.</p>
    </div>
  </div>
  <div class="tab-panel" id="tab-upload">
    <div class="card">
      <div class="card-label">Video File</div>
      <div class="drop-zone" id="drop-zone"
           onclick="document.getElementById('file-input').click()"
           ondragover="onDragOver(event)" ondragleave="onDragLeave(event)" ondrop="onDrop(event)">
        <div class="dz-icon">🎬</div>
        <p><strong>Click to upload</strong> or drag and drop</p>
        <p style="font-size:11px;margin-top:4px">MP4 · AVI · MOV · MKV &nbsp;·&nbsp; Max 50MB &nbsp;·&nbsp; Face must be visible &nbsp;·&nbsp; Min ~4s</p>
      </div>
      <input type="file" id="file-input" accept=".mp4,.avi,.mov,.mkv" onchange="onFileSelected(this.files[0])">
      <div id="file-name"></div>
      <button class="btn btn-primary btn-full" id="analyze-btn" onclick="analyzeVideo()" disabled>Analyze Video</button>
    </div>
    <div class="card" id="up-result-card" style="display:none">
      <div class="card-label">Result</div>
      <div class="spinner" id="spinner"></div>
      <div class="verdict-block" id="up-verdict-block" style="display:none">
        <div class="verdict-icon idle" id="up-verdict-icon">⬡</div>
        <div>
          <div class="verdict-label idle" id="up-verdict-label">—</div>
          <div class="verdict-sub" id="up-verdict-sub">—</div>
        </div>
      </div>
      <div id="up-probs" style="display:none">
        <div class="prob-row">
          <span class="prob-key real">Real</span>
          <div class="prob-track"><div class="prob-fill real" id="up-real-fill"></div></div>
          <span class="prob-pct" id="up-real-pct">—</span>
        </div>
        <div class="prob-row">
          <span class="prob-key fake">Fake</span>
          <div class="prob-track"><div class="prob-fill fake" id="up-fake-fill"></div></div>
          <span class="prob-pct" id="up-fake-pct">—</span>
        </div>
      </div>
      <div class="meta-row" id="up-meta" style="display:none; margin-top:16px">
        <div class="meta-item"><span class="meta-key">Segments</span><span class="meta-val" id="up-segs">—</span></div>
        <div class="meta-item"><span class="meta-key">Frames</span><span class="meta-val" id="up-frames">—</span></div>
        <div class="meta-item"><span class="meta-key">Face visible</span><span class="meta-val" id="up-face">—</span></div>
        <div class="meta-item"><span class="meta-key">P(fake)</span><span class="meta-val" id="up-pfake">—</span></div>
      </div>
      <div class="seg-section" id="seg-section" style="display:none">
        <div class="card-label" style="margin-top:20px">Segment Breakdown</div>
        <table class="seg-table">
          <thead><tr><th>#</th><th>Time</th><th>Verdict</th><th>Confidence</th><th>P(fake)</th></tr></thead>
          <tbody id="seg-tbody"></tbody>
        </table>
      </div>
    </div>
  </div>
</main>
<canvas id="canvas" width="480" height="360" style="display:none"></canvas>
<script>
  function switchTab(name, btn) {
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.getElementById('tab-' + name).classList.add('active');
    btn.classList.add('active');
  }
  const ICONS = { REAL: '✓', FAKE: '✕', UNCERTAIN: '?', idle: '⬡' };
  function setVerdict(prefix, label, sub, state) {
    document.getElementById(prefix+'-verdict-icon').className   = 'verdict-icon '+state;
    document.getElementById(prefix+'-verdict-icon').textContent = ICONS[state]||'⬡';
    document.getElementById(prefix+'-verdict-label').className  = 'verdict-label '+state;
    document.getElementById(prefix+'-verdict-label').textContent = label;
    document.getElementById(prefix+'-verdict-sub').textContent  = sub;
  }
  function updateProbs(prefix, fakeProb) {
    const realPct=((1-fakeProb)*100).toFixed(1), fakePct=(fakeProb*100).toFixed(1);
    document.getElementById(prefix+'-probs').style.display    = 'block';
    document.getElementById(prefix+'-real-fill').style.width  = realPct+'%';
    document.getElementById(prefix+'-real-pct').textContent   = realPct+'%';
    document.getElementById(prefix+'-fake-fill').style.width  = fakePct+'%';
    document.getElementById(prefix+'-fake-pct').textContent   = fakePct+'%';
  }
  let ws, captureInterval, currentStream = null;
  async function refreshCameras() {
    try { const tmp=await navigator.mediaDevices.getUserMedia({video:true}); tmp.getTracks().forEach(t=>t.stop()); } catch(e){}
    const devices=await navigator.mediaDevices.enumerateDevices();
    const cameras=devices.filter(d=>d.kind==='videoinput');
    const sel=document.getElementById('cam-sel');
    sel.innerHTML='';
    cameras.forEach((cam,idx)=>{
      const opt=document.createElement('option');
      opt.value=cam.deviceId; opt.textContent=cam.label||('Camera '+(idx+1));
      if(cam.label&&cam.label.toLowerCase().includes('obs')) opt.selected=true;
      sel.appendChild(opt);
    });
  }
  async function startCapture() {
    const sel=document.getElementById('cam-sel');
    if(!sel.value||sel.options[0]?.value===''){
      try {
        const tmp=await navigator.mediaDevices.getUserMedia({video:true}); tmp.getTracks().forEach(t=>t.stop());
        const devices=await navigator.mediaDevices.enumerateDevices();
        const cameras=devices.filter(d=>d.kind==='videoinput');
        sel.innerHTML='';
        cameras.forEach((cam,idx)=>{
          const opt=document.createElement('option');
          opt.value=cam.deviceId; opt.textContent=cam.label||('Camera '+(idx+1));
          if(cam.label&&cam.label.toLowerCase().includes('obs')) opt.selected=true;
          sel.appendChild(opt);
        });
      } catch(e){ setVerdict('wc','Camera Error',e.message,'UNCERTAIN'); return; }
    }
    const deviceId=sel.value;
    if(currentStream) currentStream.getTracks().forEach(t=>t.stop());
    const constraints=deviceId?{video:{deviceId:{exact:deviceId},width:{ideal:1280},height:{ideal:720}}}:{video:{width:{ideal:1280},height:{ideal:720}}};
    try { currentStream=await navigator.mediaDevices.getUserMedia(constraints); }
    catch(e){ setVerdict('wc','Camera Error',e.message,'UNCERTAIN'); return; }
    document.getElementById('video').srcObject=currentStream;
    const track=currentStream.getVideoTracks()[0];
    const settings=track.getSettings();
    const canvas=document.getElementById('canvas');
    canvas.width=settings.width||1280; canvas.height=settings.height||720;
    const wsProto=location.protocol==='https:'?'wss:':'ws:';
    ws=new WebSocket(wsProto+'//'+location.host+'/v1/video/ws');
    ws.onmessage=(e)=>{
      const d=JSON.parse(e.data);
      if(d.status==='prediction'){
        if(d.label==='UNCERTAIN'){
          setVerdict('wc','UNCERTAIN',d.warning||'Signal quality too low','UNCERTAIN');
          document.getElementById('wc-probs').style.display='none';
        } else {
          setVerdict('wc',d.label,d.confidence+'% confidence',d.label);
          updateProbs('wc',d.fake_prob);
        }
        document.getElementById('wc-buf-fill').style.width='100%';
        document.getElementById('wc-buf-pct').textContent='100%';
        document.getElementById('wc-meta').style.display='flex';
        document.getElementById('wc-frames').textContent=d.frames_seen;
        document.getElementById('wc-pfake').textContent=d.fake_prob;
        document.getElementById('wc-segs').textContent=d.segments_seen;
      } else {
        setVerdict('wc','Buffering...','Collecting signal — hold still','idle');
        document.getElementById('wc-buf-fill').style.width=d.fill_pct+'%';
        document.getElementById('wc-buf-pct').textContent=d.fill_pct+'%';
      }
    };
    ws.onclose=()=>setVerdict('wc','Disconnected','WebSocket closed','idle');
    const ctx=canvas.getContext('2d'); const vid=document.getElementById('video');
    captureInterval=setInterval(()=>{
      if(ws.readyState!==WebSocket.OPEN) return;
      ctx.drawImage(vid,0,0,canvas.width,canvas.height);
      ws.send(canvas.toDataURL('image/jpeg',0.85).split(',')[1]);
    },1000/15);
    document.getElementById('startBtn').textContent='◼ Running';
  }
  async function resetBuffer() {
    if(captureInterval) clearInterval(captureInterval);
    if(ws) ws.close();
    await fetch('/v1/video/reset',{method:'POST'});
    setVerdict('wc','Awaiting signal','Select a camera and press Start','idle');
    document.getElementById('wc-buf-fill').style.width='0%';
    document.getElementById('wc-buf-pct').textContent='0%';
    document.getElementById('wc-probs').style.display='none';
    document.getElementById('wc-meta').style.display='none';
    document.getElementById('startBtn').textContent='▶ Start';
  }
  let selectedFile=null;
  function onFileSelected(file){
    if(!file) return;
    selectedFile=file;
    document.getElementById('file-name').textContent=file.name+'  ('+(file.size/1024/1024).toFixed(1)+' MB)';
    document.getElementById('analyze-btn').disabled=false;
    document.getElementById('up-result-card').style.display='none';
  }
  function onDragOver(e){e.preventDefault();document.getElementById('drop-zone').classList.add('dragover');}
  function onDragLeave(){document.getElementById('drop-zone').classList.remove('dragover');}
  function onDrop(e){e.preventDefault();document.getElementById('drop-zone').classList.remove('dragover');if(e.dataTransfer.files[0])onFileSelected(e.dataTransfer.files[0]);}
  function showResult(data){
    document.getElementById('up-verdict-block').style.display='flex';
    setVerdict('up',data.label,data.confidence+'% confidence',data.label);
    updateProbs('up',data.fake_prob);
    document.getElementById('up-meta').style.display='flex';
    document.getElementById('up-segs').textContent=data.n_segments;
    document.getElementById('up-frames').textContent=data.total_frames;
    document.getElementById('up-face').textContent=data.face_pct+'%';
    document.getElementById('up-pfake').textContent=data.fake_prob;
    const tbody=document.getElementById('seg-tbody'); tbody.innerHTML='';
    (data.segments||[]).forEach(s=>{
      const tr=document.createElement('tr'); tr.className='row-'+s.label.toLowerCase();
      tr.innerHTML='<td>'+s.segment+'</td><td>'+s.start_sec+'s – '+s.end_sec+'s</td>'+
        '<td><span class="pill '+s.label+'">'+s.label+'</span></td>'+
        '<td>'+s.confidence+'%</td><td>'+s.fake_prob+'</td>';
      tbody.appendChild(tr);
    });
    document.getElementById('seg-section').style.display='block';
  }
  async function pollJob(jobId){
    const MAX_POLLS=120; let interval=1500;
    for(let i=0;i<MAX_POLLS;i++){
      await new Promise(r=>setTimeout(r,interval));
      interval=Math.min(Math.floor(interval*1.4),6000);
      const resp=await fetch('/v1/video/jobs/'+jobId);
      if(!resp.ok) throw new Error('Poll failed: '+resp.statusText);
      const job=await resp.json();
      if(job.status==='queued') setVerdict('up','Queued','Waiting for worker...','idle');
      else if(job.status==='processing') setVerdict('up','Processing','Analysing rPPG signal...','idle');
      if(job.status==='done') return job.result;
      if(job.status==='error') throw new Error(job.error||'Job failed');
    }
    throw new Error('Timed out waiting for result');
  }
  async function analyzeVideo(){
    if(!selectedFile) return;
    document.getElementById('up-result-card').style.display='block';
    document.getElementById('spinner').style.display='block';
    document.getElementById('up-verdict-block').style.display='flex';
    document.getElementById('up-probs').style.display='none';
    document.getElementById('up-meta').style.display='none';
    document.getElementById('seg-section').style.display='none';
    document.getElementById('analyze-btn').disabled=true;
    setVerdict('up','Uploading...','Sending file to server','idle');
    const formData=new FormData(); formData.append('file',selectedFile);
    try{
      const resp=await fetch('/v1/video/predict/video',{method:'POST',body:formData});
      const data=await resp.json();
      if(!resp.ok){ setVerdict('up','Error',data.detail||resp.statusText,'UNCERTAIN'); return; }
      setVerdict('up','Queued','Waiting for worker...','idle');
      const result=await pollJob(data.job_id);
      document.getElementById('spinner').style.display='none';
      showResult(result);
    } catch(e){
      document.getElementById('spinner').style.display='none';
      setVerdict('up','Failed',e.message,'UNCERTAIN');
    } finally { document.getElementById('analyze-btn').disabled=false; }
  }
</script>
</body>
</html>"""