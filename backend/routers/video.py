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

MAX_UPLOAD_MB    = 50
MAX_VIDEO_WORKERS = 4
JOB_TTL_SECONDS  = 7200

video_executor = ThreadPoolExecutor(
    max_workers=MAX_VIDEO_WORKERS,
    thread_name_prefix="deeptrack-video",
)


class FrameRequest(BaseModel):
    image: str  # base64-encoded JPEG

# helper funcs
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

# endpoints
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
        "status": "queued", "filename": file.filename,
        "size_mb": round(mb, 2), "created_at": time.time(),
        "result": None, "error": None,
    }
    video_executor.submit(_run_video_job, job_id, tmp_path, predictor, jobs)
    logger.info(f"Job {job_id} queued: {file.filename} ({mb:.1f}MB)")

    return {"job_id": job_id, "status": "queued", "filename": file.filename,
            "size_mb": round(mb, 2), "poll_url": f"/v1/video/jobs/{job_id}"}


@router.get("/jobs/{job_id}")
async def get_job(job_id: str, request: Request):
    jobs = request.app.state.jobs
    job  = jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found.")
    return {"job_id": job_id, "status": job["status"], "filename": job["filename"],
            "size_mb": job["size_mb"], "result": job.get("result"),
            "error": job.get("error"), "age_sec": round(time.time() - job["created_at"])}


@router.get("/jobs")
async def list_jobs(request: Request):
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
    return """<!DOCTYPE html>
<html>
<head>
  <title>FakeCatcher - Live Demo</title>
  <style>
    *{box-sizing:border-box}
    body{font-family:Arial,sans-serif;background:#111;color:#eee;
         display:flex;flex-direction:column;align-items:center;padding:30px;margin:0}
    h1{color:#4af;margin-bottom:6px}
    p.sub{color:#888;font-size:.85em;margin-top:0;margin-bottom:20px}
    .tabs{display:flex;gap:4px;margin-bottom:20px}
    .tab-btn{padding:10px 28px;border:none;border-radius:6px 6px 0 0;font-size:1em;
             cursor:pointer;background:#2a2a2a;color:#aaa}
    .tab-btn.active{background:#1e3a5f;color:#4af;font-weight:bold}
    .tab-panel{display:none;flex-direction:column;align-items:center;width:520px;
               background:#1a1a1a;border-radius:0 8px 8px 8px;padding:24px}
    .tab-panel.active{display:flex}
    .status{font-size:2em;margin:16px 0 8px;font-weight:bold;min-height:1.4em}
    .FAKE{color:#f55}.REAL{color:#5f5}.UNCERTAIN{color:#fa0}
    .bar-wrap{width:100%;background:#333;border-radius:4px;height:14px;margin-bottom:6px}
    .bar{height:14px;background:#4af;border-radius:4px;width:0%;transition:width .3s}
    .log{font-size:.8em;color:#888;margin-bottom:12px;min-height:1.2em}
    .prob-section{width:100%;margin-top:8px;display:none}
    .prob-row{display:flex;align-items:center;margin:5px 0;gap:10px}
    .prob-label{width:46px;font-weight:bold;font-size:.9em}
    .prob-track{flex:1;background:#333;border-radius:4px;height:20px;overflow:hidden}
    .prob-fill{height:100%;border-radius:4px;transition:width .4s ease;display:flex;
               align-items:center;padding-left:8px;font-size:.78em;font-weight:bold;
               color:#000;white-space:nowrap}
    .real-fill{background:#5f5}.fake-fill{background:#f55}
    .prob-pct{width:46px;text-align:right;font-size:.85em}
    #video{border:3px solid #333;border-radius:8px;width:100%}
    .cam-row{display:flex;align-items:center;gap:8px;width:100%;margin-bottom:10px}
    #cam-sel{flex:1;background:#222;color:#eee;border:1px solid #555;border-radius:6px;
             padding:6px 10px;font-size:.88em;cursor:pointer}
    .btn-row{display:flex;gap:8px;margin-top:12px}
    button.primary{padding:10px 26px;font-size:1em;border:none;border-radius:6px;
                   cursor:pointer;background:#4af;color:#000}
    button.secondary{padding:10px 20px;font-size:1em;border:none;border-radius:6px;
                     cursor:pointer;background:#444;color:#eee}
    .drop-zone{width:100%;border:2px dashed #555;border-radius:10px;padding:40px 20px;
               text-align:center;cursor:pointer;transition:border-color .2s;margin-bottom:16px}
    .drop-zone:hover,.drop-zone.dragover{border-color:#4af}
    .drop-zone p{margin:6px 0;color:#aaa;font-size:.9em}
    .drop-zone .icon{font-size:2.5em}
    #file-input{display:none}
    #file-name{font-size:.85em;color:#5af;margin-bottom:10px;min-height:1.2em}
    #analyze-btn{width:100%}
    #seg-section{width:100%;margin-top:16px;display:none}
    #seg-section h3{font-size:.95em;color:#aaa;margin:0 0 8px}
    table{width:100%;border-collapse:collapse;font-size:.82em}
    th{background:#2a2a2a;color:#888;padding:6px 8px;text-align:left}
    td{padding:5px 8px;border-bottom:1px solid #2a2a2a}
    tr.fake-row{background:#2a1515}tr.real-row{background:#152a15}tr.uncertain-row{background:#2a2010}
    .pill{display:inline-block;padding:2px 8px;border-radius:10px;font-size:.85em;font-weight:bold}
    .pill.FAKE{background:#f55;color:#000}.pill.REAL{background:#5f5;color:#000}.pill.UNCERTAIN{background:#fa0;color:#000}
    .spinner{display:none;width:36px;height:36px;border:4px solid #333;border-top-color:#4af;
             border-radius:50%;animation:spin .8s linear infinite;margin:12px auto}
    @keyframes spin{to{transform:rotate(360deg)}}
  </style>
</head>
<body>
  <h1>FakeCatcher</h1>
  <p class="sub">Deepfake detection via rPPG biological signals</p>
  <div class="tabs">
    <button class="tab-btn active" onclick="switchTab('webcam',this)">Live Webcam</button>
    <button class="tab-btn"        onclick="switchTab('upload',this)">Upload Video</button>
  </div>

  <div class="tab-panel active" id="tab-webcam">
    <div class="cam-row">
      <label style="font-size:.85em;color:#888;white-space:nowrap">Camera:</label>
      <select id="cam-sel"><option>Loading...</option></select>
      <button class="secondary" style="padding:6px 12px;font-size:.85em" onclick="refreshCameras()">Refresh</button>
    </div>
    <video id="video" width="480" height="360" autoplay muted></video>
    <div class="status" id="wc-status">Select a camera and press Start</div>
    <div class="bar-wrap"><div class="bar" id="wc-bar"></div></div>
    <div class="log" id="wc-log"></div>
    <div class="prob-section" id="wc-probs">
      <div class="prob-row">
        <span class="prob-label" style="color:#5f5">REAL</span>
        <div class="prob-track"><div class="prob-fill real-fill" id="wc-real-fill"></div></div>
        <span class="prob-pct" id="wc-real-pct">-</span>
      </div>
      <div class="prob-row">
        <span class="prob-label" style="color:#f55">FAKE</span>
        <div class="prob-track"><div class="prob-fill fake-fill" id="wc-fake-fill"></div></div>
        <span class="prob-pct" id="wc-fake-pct">-</span>
      </div>
    </div>
    <div class="btn-row">
      <button class="primary" id="startBtn" onclick="startCapture()">&#9654; Start</button>
      <button class="secondary" id="resetBtn" onclick="resetBuffer()">&#8635; Reset</button>
    </div>
    <p style="font-size:.75em;color:#555;margin-top:14px">
      Tip: for deepfake filter testing, select OBS Virtual Camera with DeepFaceLive running
    </p>
  </div>

  <div class="tab-panel" id="tab-upload">
    <div class="drop-zone" id="drop-zone" onclick="document.getElementById('file-input').click()"
         ondragover="onDragOver(event)" ondragleave="onDragLeave(event)" ondrop="onDrop(event)">
      <div class="icon">&#127909;</div>
      <p><strong>Click to upload</strong> or drag and drop a video</p>
      <p style="font-size:.8em">MP4, AVI, MOV, MKV &nbsp;|&nbsp; Face must be visible &nbsp;|&nbsp; Min ~4s</p>
    </div>
    <input type="file" id="file-input" accept=".mp4,.avi,.mov,.mkv" onchange="onFileSelected(this.files[0])">
    <div id="file-name"></div>
    <button class="primary" id="analyze-btn" onclick="analyzeVideo()" disabled>Analyze Video</button>
    <div class="spinner" id="spinner"></div>
    <div class="status" id="up-status"></div>
    <div class="bar-wrap" style="display:none" id="up-bar-wrap"><div class="bar" id="up-bar"></div></div>
    <div class="log" id="up-log"></div>
    <div class="prob-section" id="up-probs">
      <div class="prob-row">
        <span class="prob-label" style="color:#5f5">REAL</span>
        <div class="prob-track"><div class="prob-fill real-fill" id="up-real-fill"></div></div>
        <span class="prob-pct" id="up-real-pct">-</span>
      </div>
      <div class="prob-row">
        <span class="prob-label" style="color:#f55">FAKE</span>
        <div class="prob-track"><div class="prob-fill fake-fill" id="up-fake-fill"></div></div>
        <span class="prob-pct" id="up-fake-pct">-</span>
      </div>
    </div>
    <div id="seg-section">
      <h3>Segment breakdown</h3>
      <table><thead><tr><th>#</th><th>Time</th><th>Result</th><th>Confidence</th><th>P(fake)</th></tr></thead>
      <tbody id="seg-tbody"></tbody></table>
    </div>
  </div>

<script>
  function switchTab(name,btn){
    document.querySelectorAll('.tab-panel').forEach(p=>p.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));
    document.getElementById('tab-'+name).classList.add('active');btn.classList.add('active');
  }
  function updateProbs(prefix,fakeProb){
    const realPct=((1-fakeProb)*100).toFixed(1),fakePct=(fakeProb*100).toFixed(1);
    document.getElementById(prefix+'-probs').style.display='block';
    const rFill=document.getElementById(prefix+'-real-fill');
    const fFill=document.getElementById(prefix+'-fake-fill');
    rFill.style.width=realPct+'%';rFill.textContent=realPct>10?realPct+'%':'';
    document.getElementById(prefix+'-real-pct').textContent=realPct+'%';
    fFill.style.width=fakePct+'%';fFill.textContent=fakePct>10?fakePct+'%':'';
    document.getElementById(prefix+'-fake-pct').textContent=fakePct+'%';
  }
  let ws,interval,currentStream=null;
  async function refreshCameras(){
    try{const tmp=await navigator.mediaDevices.getUserMedia({video:true});tmp.getTracks().forEach(t=>t.stop())}catch(e){}
    const devices=await navigator.mediaDevices.enumerateDevices();
    const cameras=devices.filter(d=>d.kind==='videoinput');
    const sel=document.getElementById('cam-sel');sel.innerHTML='';
    cameras.forEach((cam,idx)=>{
      const opt=document.createElement('option');opt.value=cam.deviceId;
      opt.textContent=cam.label||('Camera '+(idx+1));
      if(cam.label&&cam.label.toLowerCase().includes('obs'))opt.selected=true;
      sel.appendChild(opt);
    });
  }
  async function startCapture(){
    const deviceId=document.getElementById('cam-sel').value;
    if(currentStream)currentStream.getTracks().forEach(t=>t.stop());
    const constraints=deviceId?{video:{deviceId:{exact:deviceId}}}:{video:true};
    try{currentStream=await navigator.mediaDevices.getUserMedia(constraints)}
    catch(e){document.getElementById('wc-status').textContent='Camera error: '+e.message;return}
    document.getElementById('video').srcObject=currentStream;
    const wsProtocol=location.protocol==='https:'?'wss:':'ws:';
    ws=new WebSocket(wsProtocol+'//'+location.host+'/v1/video/ws');
    ws.onmessage=(e)=>{
      const d=JSON.parse(e.data);
      const statusEl=document.getElementById('wc-status');
      if(d.status==='prediction'){
        if(d.label==='UNCERTAIN'){
          statusEl.textContent='UNCERTAIN';statusEl.className='status UNCERTAIN';
          document.getElementById('wc-log').textContent=d.warning||'Signal quality too low';
          document.getElementById('wc-bar').style.width='0%';
          document.getElementById('wc-probs').style.display='none';
        }else{
          statusEl.textContent=d.label+' ('+d.confidence+'%)';statusEl.className='status '+d.label;
          document.getElementById('wc-bar').style.width='100%';
          document.getElementById('wc-log').textContent='Frames seen: '+d.frames_seen+' | P(fake): '+d.fake_prob;
          updateProbs('wc',d.fake_prob);
        }
      }else{
        statusEl.textContent='Buffering... '+d.fill_pct+'%';statusEl.className='status';
        document.getElementById('wc-bar').style.width=d.fill_pct+'%';
      }
    };
    ws.onclose=()=>{document.getElementById('wc-status').textContent='Disconnected'};
    const canvas=document.getElementById('canvas')||(() =>{
      const c=document.createElement('canvas');c.width=480;c.height=360;c.style.display='none';
      document.body.appendChild(c);return c;
    })();
    const ctx=canvas.getContext('2d');
    interval=setInterval(()=>{
      if(ws.readyState!==WebSocket.OPEN)return;
      ctx.drawImage(document.getElementById('video'),0,0,480,360);
      ws.send(canvas.toDataURL('image/jpeg',.6).split(',')[1]);
    },1000/15);
    document.getElementById('startBtn').textContent='Running';
  }
  async function resetBuffer(){
    await fetch('/v1/video/reset',{method:'POST'});
    document.getElementById('wc-status').textContent='Buffer reset';
    document.getElementById('wc-bar').style.width='0%';
    document.getElementById('wc-probs').style.display='none';
  }
  let selectedFile=null;
  function onFileSelected(file){
    if(!file)return;selectedFile=file;
    document.getElementById('file-name').textContent=file.name+' ('+(file.size/1024/1024).toFixed(1)+' MB)';
    document.getElementById('analyze-btn').disabled=false;
    document.getElementById('up-status').textContent='';
    document.getElementById('up-status').className='status';
    document.getElementById('up-probs').style.display='none';
    document.getElementById('seg-section').style.display='none';
    document.getElementById('up-bar-wrap').style.display='none';
  }
  function onDragOver(e){e.preventDefault();document.getElementById('drop-zone').classList.add('dragover')}
  function onDragLeave(e){document.getElementById('drop-zone').classList.remove('dragover')}
  function onDrop(e){e.preventDefault();document.getElementById('drop-zone').classList.remove('dragover');
    if(e.dataTransfer.files[0])onFileSelected(e.dataTransfer.files[0])}
  function showResult(data){
    const statusEl=document.getElementById('up-status');
    statusEl.textContent=data.label+' ('+data.confidence+'%)';statusEl.className='status '+data.label;
    document.getElementById('up-bar-wrap').style.display='block';
    document.getElementById('up-bar').style.width='100%';
    document.getElementById('up-log').textContent=
      data.n_segments+' segments analysed | '+data.total_frames+' frames | face detected '+data.face_pct+'% of frames';
    updateProbs('up',data.fake_prob);
    const tbody=document.getElementById('seg-tbody');tbody.innerHTML='';
    (data.segments||[]).forEach(s=>{
      const tr=document.createElement('tr');
      tr.className=s.label==='FAKE'?'fake-row':s.label==='UNCERTAIN'?'uncertain-row':'real-row';
      tr.innerHTML='<td>'+s.segment+'</td><td>'+s.start_sec+'s - '+s.end_sec+'s</td>'+
        '<td><span class="pill '+s.label+'">'+s.label+'</span></td>'+
        '<td>'+s.confidence+'%</td><td>'+s.fake_prob+'</td>';
      tbody.appendChild(tr);
    });
    document.getElementById('seg-section').style.display='block';
  }
  async function pollJob(jobId){
    for(let i=0;i<180;i++){
      await new Promise(r=>setTimeout(r,2000));
      const resp=await fetch('/v1/video/jobs/'+jobId);
      if(!resp.ok)throw new Error('Poll failed: '+resp.statusText);
      const job=await resp.json();
      if(job.status==='queued'||job.status==='processing')
        document.getElementById('up-status').textContent=job.status==='queued'?'Queued — waiting for worker...':'Processing...';
      if(job.status==='done')return job.result;
      if(job.status==='error')throw new Error(job.error||'Job failed');
    }
    throw new Error('Timed out waiting for result');
  }
  async function analyzeVideo(){
    if(!selectedFile)return;
    document.getElementById('spinner').style.display='block';
    document.getElementById('analyze-btn').disabled=true;
    document.getElementById('up-status').textContent='Uploading...';
    document.getElementById('up-status').className='status';
    document.getElementById('up-probs').style.display='none';
    document.getElementById('seg-section').style.display='none';
    document.getElementById('up-bar-wrap').style.display='none';
    const formData=new FormData();formData.append('file',selectedFile);
    try{
      const resp=await fetch('/v1/video/predict/video',{method:'POST',body:formData});
      const data=await resp.json();
      if(!resp.ok){document.getElementById('up-status').textContent='Error: '+(data.detail||resp.statusText);return}
      document.getElementById('up-status').textContent='Queued — waiting for worker...';
      const result=await pollJob(data.job_id);showResult(result);
    }catch(e){document.getElementById('up-status').textContent='Failed: '+e.message}
    finally{document.getElementById('spinner').style.display='none';document.getElementById('analyze-btn').disabled=false}
  }
  window.addEventListener('load',refreshCameras);
</script>
</body>
</html>"""