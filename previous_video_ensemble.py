import cv2
import base64
import logging
import os
import tempfile
import time
import uuid
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from fastapi import APIRouter, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

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
        return {
            "status":      "prediction",
            "label":       result["label"],
            "confidence":  result["confidence"],
            "fake_prob":   result["fake_prob"],
            "frames_seen": buffer.frame_count,
            "per_model":   result.get("per_model"),
            "dominant":    result.get("dominant"),
        }

    return {
        "status":      "buffering",
        "fill_pct":    buffer.fill_pct,
        "frames_seen": buffer.frame_count,
        "message":     f"Buffering... {buffer.fill_pct}% — need {OMEGA} frames (~{OMEGA//FS}s)",
    }


@router.post("/predict/video")
async def predict_video(request: Request, file: UploadFile = File(...)):
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
        "status":     "queued",
        "filename":   file.filename,
        "size_mb":    round(mb, 2),
        "created_at": time.time(),
        "result":     None,
        "error":      None,
    }
    video_executor.submit(_run_video_job, job_id, tmp_path, predictor, jobs)
    logger.info(f"Job {job_id} queued: {file.filename} ({mb:.1f}MB)")

    return {
        "job_id":   job_id,
        "status":   "queued",
        "filename": file.filename,
        "size_mb":  round(mb, 2),
        "poll_url": f"/v1/video/jobs/{job_id}",
    }


@router.get("/jobs/{job_id}")
async def get_job(job_id: str, request: Request):
    jobs = request.app.state.jobs
    job  = jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found.")
    return {
        "job_id":   job_id,
        "status":   job["status"],
        "filename": job["filename"],
        "size_mb":  job["size_mb"],
        "result":   job.get("result"),
        "error":    job.get("error"),
        "age_sec":  round(time.time() - job["created_at"]),
    }


@router.get("/jobs")
async def list_jobs(request: Request):
    jobs    = request.app.state.jobs
    summary = {
        jid: {
            "status":   j["status"],
            "filename": j["filename"],
            "age_sec":  round(time.time() - j["created_at"]),
        }
        for jid, j in jobs.items()
    }
    counts = {}
    for j in jobs.values():
        counts[j["status"]] = counts.get(j["status"], 0) + 1
    return {"jobs": summary, "counts": counts, "total": len(jobs)}


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
                    "status":        "prediction",
                    "label":         result["label"],
                    "confidence":    result["confidence"],
                    "fake_prob":     result["fake_prob"],
                    "warning":       result.get("warning"),
                    "frames_seen":   ws_buffer.frame_count,
                    "segments_seen": len(ws_buffer.prob_history),
                })
            else:
                await ws.send_json({
                    "status":      "buffering",
                    "fill_pct":    ws_buffer.fill_pct,
                    "frames_seen": ws_buffer.frame_count,
                })
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")


@router.get("/", response_class=HTMLResponse)
async def video_ui():
    return """<!DOCTYPE html>
<html>
<head>
  <title>FakeCatcher - Live Demo</title>
  <link href="https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    *{box-sizing:border-box}
    body{font-family:'DM Mono',monospace;background:#0d0d0d;color:#eee;
         display:flex;flex-direction:column;align-items:center;padding:30px;margin:0;
         background-image:linear-gradient(rgba(232,255,71,.02)1px,transparent 1px),
           linear-gradient(90deg,rgba(232,255,71,.02)1px,transparent 1px);
         background-size:32px 32px}
    h1{font-family:'Syne',sans-serif;font-weight:800;font-size:2rem;
       letter-spacing:-.02em;margin-bottom:4px}
    h1 span{color:#e8ff47}
    p.sub{color:#555;font-size:.75rem;margin-top:0;margin-bottom:20px;
          letter-spacing:.1em;text-transform:uppercase}
    .tabs{display:flex;gap:4px;margin-bottom:0}
    .tab-btn{padding:10px 28px;border:1px solid #222;border-bottom:none;
             border-radius:6px 6px 0 0;font-size:.85em;font-family:'DM Mono',monospace;
             cursor:pointer;background:#111;color:#555;letter-spacing:.05em}
    .tab-btn.active{background:#161616;color:#e8ff47;border-color:#333}
    .tab-panel{display:none;flex-direction:column;align-items:center;width:540px;
               background:#161616;border:1px solid #333;border-radius:0 8px 8px 8px;padding:24px}
    .tab-panel.active{display:flex}
    .status{font-size:1.8em;margin:16px 0 8px;font-weight:700;min-height:1.4em;
            font-family:'Syne',sans-serif;letter-spacing:-.01em}
    .FAKE{color:#f55}.REAL{color:#5f5}.UNCERTAIN{color:#fa0}
    .bar-wrap{width:100%;background:#1a1a1a;border-radius:3px;height:6px;margin-bottom:6px}
    .bar{height:6px;background:#e8ff47;border-radius:3px;width:0%;transition:width .4s}
    .log{font-size:.72em;color:#555;margin-bottom:12px;min-height:1.2em;letter-spacing:.04em}
    .prob-section{width:100%;margin-top:8px;display:none}
    .prob-row{display:flex;align-items:center;margin:5px 0;gap:10px}
    .prob-label{width:46px;font-weight:500;font-size:.82em}
    .prob-track{flex:1;background:#1a1a1a;border-radius:3px;height:20px;overflow:hidden}
    .prob-fill{height:100%;border-radius:3px;transition:width .4s ease;display:flex;
               align-items:center;padding-left:8px;font-size:.72em;font-weight:700;
               color:#000;white-space:nowrap}
    .real-fill{background:#5f5}.fake-fill{background:#f55}
    .prob-pct{width:46px;text-align:right;font-size:.8em}
    #video{border:1px solid #2a2a2a;border-radius:6px;width:100%}
    .cam-row{display:flex;align-items:center;gap:8px;width:100%;margin-bottom:10px}
    #cam-sel{flex:1;background:#111;color:#eee;border:1px solid #333;border-radius:6px;
             padding:6px 10px;font-size:.82em;font-family:'DM Mono',monospace;cursor:pointer}
    .btn-row{display:flex;gap:8px;margin-top:12px}
    button.primary{padding:10px 26px;font-size:.88em;border:none;border-radius:6px;
                   cursor:pointer;background:#e8ff47;color:#000;font-family:'DM Mono',monospace;
                   font-weight:500;letter-spacing:.06em}
    button.primary:hover{background:#d4eb30}
    button.secondary{padding:10px 20px;font-size:.88em;border:1px solid #333;border-radius:6px;
                     cursor:pointer;background:#111;color:#aaa;font-family:'DM Mono',monospace}
    button.secondary:hover{border-color:#555;color:#eee}
    .drop-zone{width:100%;border:1px dashed #333;border-radius:8px;padding:36px 20px;
               text-align:center;cursor:pointer;transition:border-color .2s,background .2s;
               margin-bottom:16px}
    .drop-zone:hover,.drop-zone.dragover{border-color:#e8ff47;background:#161f00}
    .drop-zone p{margin:6px 0;color:#555;font-size:.82em}
    .drop-zone .icon{font-size:2rem;margin-bottom:8px}
    #file-input{display:none}
    #file-name{font-size:.78em;color:#e8ff47;margin-bottom:10px;min-height:1.2em}
    #analyze-btn{width:100%}
    #seg-section{width:100%;margin-top:16px;display:none}
    #seg-section h3{font-size:.75em;color:#555;margin:0 0 8px;letter-spacing:.1em;
                    text-transform:uppercase}
    table{width:100%;border-collapse:collapse;font-size:.78em}
    th{background:#111;color:#555;padding:6px 8px;text-align:left;
       font-size:.7em;letter-spacing:.08em;text-transform:uppercase}
    td{padding:5px 8px;border-bottom:1px solid #1a1a1a}
    tr.fake-row{background:#1f1010}tr.real-row{background:#101f10}
    tr.uncertain-row{background:#1f1a0a}
    .pill{display:inline-block;padding:2px 8px;border-radius:3px;font-size:.8em;font-weight:700}
    .pill.FAKE{background:#f55;color:#000}.pill.REAL{background:#5f5;color:#000}
    .pill.UNCERTAIN{background:#fa0;color:#000}
    .spinner{display:none;width:28px;height:28px;border:3px solid #222;
             border-top-color:#e8ff47;border-radius:50%;
             animation:spin .7s linear infinite;margin:12px auto}
    @keyframes spin{to{transform:rotate(360deg)}}

    /* ── Scorecard ── */
    #scorecard{display:none;width:100%;margin-top:20px}
    .sc-card{border:1px solid #252525;border-radius:8px;overflow:hidden}
    .sc-header{background:#111;padding:12px 18px;border-bottom:1px solid #222;
               display:flex;align-items:center;justify-content:space-between}
    .sc-title{font-family:'Syne',sans-serif;font-weight:700;font-size:.88rem;
              letter-spacing:.04em;color:#f0f0f0}
    .sc-dominant{font-size:.65rem;letter-spacing:.14em;text-transform:uppercase;color:#e8ff47}
    .sc-body{padding:16px 18px;background:#0f0f0f}
    .sc-row{display:flex;align-items:center;margin:8px 0;gap:10px}
    .sc-name{width:148px;font-size:.68rem;letter-spacing:.06em;
             text-transform:uppercase;color:#666;flex-shrink:0;transition:color .2s}
    .sc-name.dominant{color:#e8ff47}
    .sc-track{flex:1;background:#1a1a1a;border-radius:2px;height:16px;
              overflow:hidden;position:relative}
    .sc-fill{height:100%;border-radius:2px;
             transition:width .7s cubic-bezier(.4,0,.2,1);
             display:flex;align-items:center;padding-left:7px;
             font-size:.65rem;font-weight:700;color:#000;white-space:nowrap}
    .sc-pct{width:40px;text-align:right;font-size:.72rem;font-weight:500;
            color:#aaa;flex-shrink:0}
    .sc-fake{background:linear-gradient(90deg,#d44,#f55)}
    .sc-uncertain{background:linear-gradient(90deg,#d90,#fa0)}
    .sc-real{background:linear-gradient(90deg,#3a3,#5f5)}
    .sc-ensemble{background:linear-gradient(90deg,#b8cc20,#e8ff47)}
    .sc-divider{height:1px;background:#1e1e1e;margin:12px 0}
    .sc-verdict-chip{display:inline-block;padding:1px 7px;border-radius:2px;
                     font-size:.62rem;font-weight:700;margin-left:6px;vertical-align:middle}
    .sc-verdict-chip.FAKE{background:#f55;color:#000}
    .sc-verdict-chip.REAL{background:#5f5;color:#000}
    .sc-verdict-chip.UNCERTAIN{background:#fa0;color:#000}
  </style>
</head>
<body>
  <h1>Fake<span>Catcher</span></h1>
  <p class="sub">Deepfake detection via rPPG biological signals</p>
  <div class="tabs">
    <button class="tab-btn active" onclick="switchTab('webcam',this)">Live Webcam</button>
    <button class="tab-btn"        onclick="switchTab('upload',this)">Upload Video</button>
  </div>

  <!-- ── WEBCAM TAB ── -->
  <div class="tab-panel active" id="tab-webcam">
    <div class="cam-row">
      <label style="font-size:.75em;color:#555;white-space:nowrap;letter-spacing:.06em">CAMERA</label>
      <select id="cam-sel"><option>Loading...</option></select>
      <button class="secondary" style="padding:6px 12px;font-size:.78em" onclick="refreshCameras()">↺</button>
    </div>
    <video id="video" width="480" height="360" autoplay muted></video>
    <div class="status" id="wc-status" style="color:#333">Select a camera and press Start</div>
    <div class="bar-wrap"><div class="bar" id="wc-bar"></div></div>
    <div class="log" id="wc-log"></div>
    <div class="prob-section" id="wc-probs">
      <div class="prob-row">
        <span class="prob-label" style="color:#5f5">REAL</span>
        <div class="prob-track"><div class="prob-fill real-fill" id="wc-real-fill"></div></div>
        <span class="prob-pct" id="wc-real-pct">—</span>
      </div>
      <div class="prob-row">
        <span class="prob-label" style="color:#f55">FAKE</span>
        <div class="prob-track"><div class="prob-fill fake-fill" id="wc-fake-fill"></div></div>
        <span class="prob-pct" id="wc-fake-pct">—</span>
      </div>
    </div>
    <div class="btn-row">
      <button class="primary"   id="startBtn" onclick="startCapture()">▶ Start</button>
      <button class="secondary" id="resetBtn" onclick="resetBuffer()">↺ Reset</button>
    </div>
    <p style="font-size:.68em;color:#333;margin-top:14px;letter-spacing:.04em">
      Tip: select OBS Virtual Camera with DeepFaceLive for deepfake filter testing
    </p>
  </div>

  <!-- ── UPLOAD TAB ── -->
  <div class="tab-panel" id="tab-upload">
    <div class="drop-zone" id="drop-zone"
         onclick="document.getElementById('file-input').click()"
         ondragover="onDragOver(event)" ondragleave="onDragLeave(event)" ondrop="onDrop(event)">
      <div class="icon">⬆</div>
      <p><strong>Click to upload</strong> or drag and drop a video</p>
      <p style="font-size:.75em;color:#444">MP4 · AVI · MOV · MKV &nbsp;|&nbsp; Face must be visible &nbsp;|&nbsp; Min ~4s</p>
    </div>
    <input type="file" id="file-input" accept=".mp4,.avi,.mov,.mkv"
           onchange="onFileSelected(this.files[0])">
    <div id="file-name"></div>
    <button class="primary" id="analyze-btn" onclick="analyzeVideo()" disabled>Analyze Video</button>
    <div class="spinner" id="spinner"></div>
    <div class="status" id="up-status"></div>
    <div class="bar-wrap" style="display:none" id="up-bar-wrap">
      <div class="bar" id="up-bar"></div>
    </div>
    <div class="log" id="up-log"></div>

    <div class="prob-section" id="up-probs">
      <div class="prob-row">
        <span class="prob-label" style="color:#5f5">REAL</span>
        <div class="prob-track"><div class="prob-fill real-fill" id="up-real-fill"></div></div>
        <span class="prob-pct" id="up-real-pct">—</span>
      </div>
      <div class="prob-row">
        <span class="prob-label" style="color:#f55">FAKE</span>
        <div class="prob-track"><div class="prob-fill fake-fill" id="up-fake-fill"></div></div>
        <span class="prob-pct" id="up-fake-pct">—</span>
      </div>
    </div>

    <div id="seg-section">
      <h3>Segment Breakdown</h3>
      <table>
        <thead>
          <tr><th>#</th><th>Time</th><th>Result</th><th>Confidence</th><th>P(fake)</th></tr>
        </thead>
        <tbody id="seg-tbody"></tbody>
      </table>
    </div>

    <div id="scorecard">
      <div class="sc-card">
        <div class="sc-header">
          <span class="sc-title">Model Scorecard</span>
          <span class="sc-dominant" id="sc-dominant"></span>
        </div>
        <div class="sc-body">
          <div id="sc-rows"></div>
          <div class="sc-divider"></div>
          <div class="sc-row" id="sc-ensemble-row"></div>
        </div>
      </div>
    </div>
  </div>

<script>
  function switchTab(name, btn) {
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.getElementById('tab-' + name).classList.add('active');
    btn.classList.add('active');
  }

  function updateProbs(prefix, fakeProb) {
    const realPct = ((1 - fakeProb) * 100).toFixed(1);
    const fakePct = (fakeProb * 100).toFixed(1);
    document.getElementById(prefix + '-probs').style.display = 'block';
    const rFill = document.getElementById(prefix + '-real-fill');
    const fFill = document.getElementById(prefix + '-fake-fill');
    rFill.style.width = realPct + '%';
    rFill.textContent = realPct > 10 ? realPct + '%' : '';
    document.getElementById(prefix + '-real-pct').textContent = realPct + '%';
    fFill.style.width = fakePct + '%';
    fFill.textContent = fakePct > 10 ? fakePct + '%' : '';
    document.getElementById(prefix + '-fake-pct').textContent = fakePct + '%';
  }

  function showScorecard(perModel, dominant, ensembleProb) {
    const container = document.getElementById('sc-rows');
    container.innerHTML = '';

    const displayNames = {
      'DeepFakeDetection': 'DeepFakeDetection',
      'Deepfakes':         'Deepfakes',
    };

    const fills = [];

    Object.entries(perModel).forEach(([key, fakeProb]) => {
      const conf       = fakeProb >= 0.5 ? fakeProb * 100 : (1 - fakeProb) * 100;
      const verdict    = fakeProb >= 0.55 ? 'FAKE'
                       : fakeProb >= 0.35 ? 'UNCERTAIN'
                       : 'REAL';
      const colorClass = verdict === 'FAKE'     ? 'sc-fake'
                       : verdict === 'UNCERTAIN' ? 'sc-uncertain'
                       : 'sc-real';
      const isDominant = key === dominant;

      const row = document.createElement('div');
      row.className = 'sc-row';
      row.innerHTML =
        '<span class="sc-name' + (isDominant ? ' dominant' : '') + '">' +
          (isDominant ? '▶ ' : '') + (displayNames[key] || key) +
        '</span>' +
        '<div class="sc-track">' +
          '<div class="sc-fill ' + colorClass + '" style="width:0%" id="scf-' + key + '">' +
            (conf > 18 ? conf.toFixed(0) + '%' : '') +
          '</div>' +
        '</div>' +
        '<span class="sc-pct">' + conf.toFixed(1) + '%</span>' +
        '<span class="sc-verdict-chip ' + verdict + '">' + verdict + '</span>';

      container.appendChild(row);
      fills.push({ id: 'scf-' + key, target: conf.toFixed(1) });
    });

    const ensConf    = ensembleProb >= 0.5 ? ensembleProb * 100 : (1 - ensembleProb) * 100;
    const ensVerdict = ensembleProb >= 0.55 ? 'FAKE'
                     : ensembleProb >= 0.35 ? 'UNCERTAIN'
                     : 'REAL';
    document.getElementById('sc-ensemble-row').innerHTML =
      '<span class="sc-name dominant" style="color:#e8ff47">Weighted Ensemble</span>' +
      '<div class="sc-track">' +
        '<div class="sc-fill sc-ensemble" style="width:0%" id="scf-ensemble">' +
          (ensConf > 18 ? ensConf.toFixed(0) + '%' : '') +
        '</div>' +
      '</div>' +
      '<span class="sc-pct" style="color:#e8ff47">' + ensConf.toFixed(1) + '%</span>' +
      '<span class="sc-verdict-chip ' + ensVerdict + '">' + ensVerdict + '</span>';

    fills.push({ id: 'scf-ensemble', target: ensConf.toFixed(1) });

    document.getElementById('sc-dominant').textContent =
      dominant ? 'Dominant · ' + dominant : '';
    document.getElementById('scorecard').style.display = 'block';

    requestAnimationFrame(() => {
      setTimeout(() => {
        fills.forEach(f => {
          const el = document.getElementById(f.id);
          if (el) el.style.width = f.target + '%';
        });
      }, 80);
    });
  }

  let ws, interval, currentStream = null;

  async function refreshCameras() {
    try {
      const tmp = await navigator.mediaDevices.getUserMedia({ video: true });
      tmp.getTracks().forEach(t => t.stop());
    } catch (e) {}
    const devices = await navigator.mediaDevices.enumerateDevices();
    const cameras = devices.filter(d => d.kind === 'videoinput');
    const sel = document.getElementById('cam-sel');
    sel.innerHTML = '';
    cameras.forEach((cam, idx) => {
      const opt = document.createElement('option');
      opt.value = cam.deviceId;
      opt.textContent = cam.label || ('Camera ' + (idx + 1));
      if (cam.label && cam.label.toLowerCase().includes('obs')) opt.selected = true;
      sel.appendChild(opt);
    });
  }

  async function startCapture() {
    const deviceId = document.getElementById('cam-sel').value;
    if (currentStream) currentStream.getTracks().forEach(t => t.stop());
    const constraints = deviceId ? { video: { deviceId: { exact: deviceId } } } : { video: true };
    try {
      currentStream = await navigator.mediaDevices.getUserMedia(constraints);
    } catch (e) {
      document.getElementById('wc-status').textContent = 'Camera error: ' + e.message;
      return;
    }
    document.getElementById('video').srcObject = currentStream;
    const wsProtocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(wsProtocol + '//' + location.host + '/v1/video/ws');
    ws.onmessage = (e) => {
      const d = JSON.parse(e.data);
      const statusEl = document.getElementById('wc-status');
      if (d.status === 'prediction') {
        if (d.label === 'UNCERTAIN') {
          statusEl.textContent = 'UNCERTAIN';
          statusEl.className   = 'status UNCERTAIN';
          document.getElementById('wc-log').textContent = d.warning || 'Signal quality too low';
          document.getElementById('wc-bar').style.width = '0%';
          document.getElementById('wc-probs').style.display = 'none';
        } else {
          statusEl.textContent = d.label + ' (' + d.confidence + '%)';
          statusEl.className   = 'status ' + d.label;
          document.getElementById('wc-bar').style.width = '100%';
          document.getElementById('wc-log').textContent =
            'Frames: ' + d.frames_seen + '  |  P(fake): ' + d.fake_prob;
          updateProbs('wc', d.fake_prob);
        }
      } else {
        statusEl.textContent = 'Buffering... ' + d.fill_pct + '%';
        statusEl.className   = 'status';
        document.getElementById('wc-bar').style.width = d.fill_pct + '%';
      }
    };
    ws.onclose = () => {
      document.getElementById('wc-status').textContent = 'Disconnected';
    };
    const canvas = document.getElementById('canvas') || (() => {
      const c = document.createElement('canvas');
      c.width = 480; c.height = 360; c.style.display = 'none';
      c.id = 'canvas';
      document.body.appendChild(c);
      return c;
    })();
    const ctx = canvas.getContext('2d');
    interval = setInterval(() => {
      if (ws.readyState !== WebSocket.OPEN) return;
      ctx.drawImage(document.getElementById('video'), 0, 0, 480, 360);
      ws.send(canvas.toDataURL('image/jpeg', .6).split(',')[1]);
    }, 1000 / 15);
    document.getElementById('startBtn').textContent = '● Running';
  }

  async function resetBuffer() {
    await fetch('/v1/video/reset', { method: 'POST' });
    document.getElementById('wc-status').textContent = 'Buffer reset';
    document.getElementById('wc-status').className   = 'status';
    document.getElementById('wc-bar').style.width    = '0%';
    document.getElementById('wc-probs').style.display = 'none';
  }

  let selectedFile = null;

  function onFileSelected(file) {
    if (!file) return;
    selectedFile = file;
    document.getElementById('file-name').textContent =
      file.name + '  (' + (file.size / 1024 / 1024).toFixed(1) + ' MB)';
    document.getElementById('analyze-btn').disabled   = false;
    document.getElementById('up-status').textContent  = '';
    document.getElementById('up-status').className    = 'status';
    document.getElementById('up-probs').style.display    = 'none';
    document.getElementById('seg-section').style.display = 'none';
    document.getElementById('scorecard').style.display   = 'none';
    document.getElementById('up-bar-wrap').style.display = 'none';
  }

  function onDragOver(e) {
    e.preventDefault();
    document.getElementById('drop-zone').classList.add('dragover');
  }
  function onDragLeave(e) {
    document.getElementById('drop-zone').classList.remove('dragover');
  }
  function onDrop(e) {
    e.preventDefault();
    document.getElementById('drop-zone').classList.remove('dragover');
    if (e.dataTransfer.files[0]) onFileSelected(e.dataTransfer.files[0]);
  }

  function showResult(data) {
    const statusEl = document.getElementById('up-status');
    statusEl.textContent = data.label + ' (' + data.confidence + '%)';
    statusEl.className   = 'status ' + data.label;
    document.getElementById('up-bar-wrap').style.display = 'block';
    document.getElementById('up-bar').style.width        = '100%';
    document.getElementById('up-log').textContent =
      data.n_segments + ' segments  |  ' + data.total_frames +
      ' frames  |  face ' + data.face_pct + '%';
    updateProbs('up', data.fake_prob);

    const tbody = document.getElementById('seg-tbody');
    tbody.innerHTML = '';
    (data.segments || []).forEach(s => {
      const tr = document.createElement('tr');
      tr.className = s.label === 'FAKE' ? 'fake-row'
                   : s.label === 'UNCERTAIN' ? 'uncertain-row' : 'real-row';
      tr.innerHTML =
        '<td>' + s.segment + '</td>' +
        '<td>' + s.start_sec + 's – ' + s.end_sec + 's</td>' +
        '<td><span class="pill ' + s.label + '">' + s.label + '</span></td>' +
        '<td>' + s.confidence + '%</td>' +
        '<td>' + s.fake_prob + '</td>';
      tbody.appendChild(tr);
    });
    document.getElementById('seg-section').style.display = 'block';

    if (data.per_model) {
      showScorecard(data.per_model, data.dominant, data.fake_prob);
    }
  }

  async function pollJob(jobId) {
    for (let i = 0; i < 180; i++) {
      await new Promise(r => setTimeout(r, 2000));
      const resp = await fetch('/v1/video/jobs/' + jobId);
      if (!resp.ok) throw new Error('Poll failed: ' + resp.statusText);
      const job = await resp.json();
      if (job.status === 'queued' || job.status === 'processing')
        document.getElementById('up-status').textContent =
          job.status === 'queued' ? 'Queued — waiting for worker...' : 'Processing...';
      if (job.status === 'done')  return job.result;
      if (job.status === 'error') throw new Error(job.error || 'Job failed');
    }
    throw new Error('Timed out waiting for result');
  }

  async function analyzeVideo() {
    if (!selectedFile) return;
    document.getElementById('spinner').style.display     = 'block';
    document.getElementById('analyze-btn').disabled      = true;
    document.getElementById('up-status').textContent     = 'Uploading...';
    document.getElementById('up-status').className       = 'status';
    document.getElementById('up-probs').style.display    = 'none';
    document.getElementById('seg-section').style.display = 'none';
    document.getElementById('scorecard').style.display   = 'none';
    document.getElementById('up-bar-wrap').style.display = 'none';

    const formData = new FormData();
    formData.append('file', selectedFile);
    try {
      const resp = await fetch('/v1/video/predict/video', { method: 'POST', body: formData });
      const data = await resp.json();
      if (!resp.ok) {
        document.getElementById('up-status').textContent =
          'Error: ' + (data.detail || resp.statusText);
        return;
      }
      document.getElementById('up-status').textContent = 'Queued — waiting for worker...';
      const result = await pollJob(data.job_id);
      showResult(result);
    } catch (e) {
      document.getElementById('up-status').textContent = 'Failed: ' + e.message;
    } finally {
      document.getElementById('spinner').style.display = 'none';
      document.getElementById('analyze-btn').disabled  = false;
    }
  }

  window.addEventListener('load', refreshCameras);
</script>
</body>
</html>"""