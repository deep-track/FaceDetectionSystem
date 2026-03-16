import io
import logging
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image
from core.auth import verify_api_key

logger = logging.getLogger("deeptrack.image")
router = APIRouter()

@router.post("/predict")
async def predict_image(
    request: Request,
    file: UploadFile = File(...),
    _key: dict = Depends(verify_api_key),
):
    """Classify a single image as Real or Fake."""
    predictor = request.app.state.image_predictor
    if predictor is None:
        raise HTTPException(503, "Image model not loaded.")
    try:
        image_bytes = await file.read()
        image       = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        result      = predictor.predict(image)
        return {"filename": file.filename, **result}
    except Exception as e:
        raise HTTPException(400, f"Error processing image: {e}")


@router.get("/", response_class=HTMLResponse)
async def image_ui():
    """DeepTrack image upload demo UI."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DeepTrack — Image Analysis</title>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
  :root{--bg:#0a0a0a;--surface:#111;--border:#222;--accent:#e8ff47;
        --text:#f0f0f0;--muted:#555;--real:#47ffa3;--fake:#ff4757}
  body{background:var(--bg);color:var(--text);font-family:'DM Mono',monospace;
       min-height:100vh;display:flex;flex-direction:column;align-items:center;
       padding:0 0 80px;
       background-image:linear-gradient(rgba(232,255,71,.03)1px,transparent 1px),
         linear-gradient(90deg,rgba(232,255,71,.03)1px,transparent 1px);
       background-size:40px 40px}
  header{width:100%;max-width:100%;margin-bottom:0;padding:20px 40px;
         border-bottom:1px solid var(--border)}
  .header-inner{max-width:680px;margin:0 auto}
  .wordmark{font-family:'Syne',sans-serif;font-weight:800;
            font-size:clamp(2rem,5vw,2.8rem);letter-spacing:-.03em;line-height:1}
  .wordmark span{color:var(--accent)}
  .tagline{margin-top:8px;font-size:.72rem;letter-spacing:.18em;
           text-transform:uppercase;color:var(--muted)}
  .status-pill{display:inline-flex;align-items:center;gap:6px;margin-top:14px;
               padding:4px 12px;border:1px solid var(--border);
               font-size:.7rem;letter-spacing:.1em;color:var(--muted)}
  .status-pill::before{content:'';width:6px;height:6px;border-radius:50%;
    background:var(--real);box-shadow:0 0 6px var(--real);animation:pulse 2s infinite}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}

  .key-bar{width:100%;background:#0d1117;border-bottom:1px solid #1c2530;
           padding:10px 40px;display:flex;align-items:center;gap:12px}
  .key-bar-inner{max-width:680px;width:100%;margin:0 auto;display:flex;align-items:center;gap:12px}
  .key-label{font-size:.65rem;letter-spacing:.14em;text-transform:uppercase;
             color:#4a6070;white-space:nowrap}
  .key-input{flex:1;max-width:480px;background:#080b0f;border:1px solid #243040;
             color:#c8d8e8;font-family:'DM Mono',monospace;font-size:12px;
             padding:7px 12px;outline:none;transition:border-color .15s}
  .key-input:focus{border-color:#e8ff47}
  .key-status{font-size:.65rem;letter-spacing:.1em;color:#4a6070;text-transform:uppercase;
              white-space:nowrap}

  .content{width:100%;max-width:680px;padding:40px 24px 0}
  .card{width:100%;background:var(--surface);border:1px solid var(--border);padding:40px}
  .drop-zone{border:1.5px dashed var(--border);padding:52px 24px;text-align:center;
             cursor:pointer;transition:border-color .2s,background .2s}
  .drop-zone:hover,.drop-zone.over{border-color:var(--accent);background:rgba(232,255,71,.03)}
  .drop-zone input{display:none}
  .drop-icon{font-size:2rem;margin-bottom:16px;display:block;opacity:.4}
  .drop-label{font-family:'Syne',sans-serif;font-size:1.05rem;font-weight:700}
  .drop-sub{margin-top:6px;font-size:.7rem;color:var(--muted);letter-spacing:.08em}
  .preview-strip{display:none;margin-top:24px;gap:10px;flex-wrap:wrap}
  .preview-strip.visible{display:flex}
  .thumb-wrap{position:relative;width:76px;height:76px;border:1px solid var(--border);overflow:hidden;flex-shrink:0}
  .thumb-wrap img{width:100%;height:100%;object-fit:cover;display:block}
  .thumb-remove{position:absolute;top:2px;right:3px;background:none;border:none;
                color:var(--muted);font-size:.75rem;cursor:pointer;transition:color .15s;line-height:1}
  .thumb-remove:hover{color:var(--fake)}
  .progress-wrap{margin-top:20px;height:2px;background:var(--border);display:none;overflow:hidden}
  .progress-wrap.visible{display:block}
  .progress-bar{height:100%;background:var(--accent);animation:scan 1.6s ease-in-out infinite}
  @keyframes scan{0%{margin-left:0%;width:30%}50%{margin-left:70%;width:30%}100%{margin-left:0%;width:30%}}
  .btn-analyze{margin-top:28px;width:100%;padding:14px;background:var(--accent);color:#000;
               font-family:'Syne',sans-serif;font-weight:700;font-size:.9rem;
               letter-spacing:.12em;text-transform:uppercase;border:none;cursor:pointer;
               transition:opacity .15s,transform .1s}
  .btn-analyze:hover:not(:disabled){opacity:.88;transform:translateY(-1px)}
  .btn-analyze:disabled{opacity:.25;cursor:not-allowed;transform:none}
  #results{width:100%;max-width:680px;padding:0 24px;margin-top:24px}
  .result-card{background:var(--surface);border:1px solid var(--border);
               border-left:3px solid var(--muted);margin-bottom:12px;overflow:hidden;
               animation:slideIn .25s ease}
  @keyframes slideIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
  .result-card.REAL{border-left-color:var(--real)}
  .result-card.FAKE{border-left-color:var(--fake)}
  .result-inner{display:grid;grid-template-columns:72px 1fr auto;align-items:stretch}
  .result-thumb{width:72px;height:72px;object-fit:cover;display:block;border-right:1px solid var(--border)}
  .result-body{padding:12px 16px;display:flex;flex-direction:column;justify-content:center;gap:4px;min-width:0}
  .result-filename{font-size:.68rem;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .result-verdict{font-family:'Syne',sans-serif;font-weight:800;font-size:1.2rem;letter-spacing:-.02em;line-height:1}
  .result-card.REAL .result-verdict{color:var(--real)}
  .result-card.FAKE .result-verdict{color:var(--fake)}
  .prob-row{display:flex;align-items:center;gap:8px;margin-top:6px}
  .prob-label{font-size:.6rem;letter-spacing:.1em;text-transform:uppercase;width:28px;color:var(--muted);flex-shrink:0}
  .prob-track{flex:1;height:3px;background:var(--border);overflow:hidden}
  .prob-fill{height:100%;width:0%;transition:width .7s cubic-bezier(.16,1,.3,1)}
  .prob-fill.real{background:var(--real)}
  .prob-fill.fake{background:var(--fake)}
  .prob-pct{font-size:.62rem;width:36px;text-align:right;color:var(--muted)}
  .confidence-badge{padding:0 20px;display:flex;flex-direction:column;align-items:center;
                    justify-content:center;border-left:1px solid var(--border);gap:2px;flex-shrink:0}
  .conf-value{font-family:'Syne',sans-serif;font-weight:700;font-size:1.4rem;line-height:1}
  .result-card.REAL .conf-value{color:var(--real)}
  .result-card.FAKE .conf-value{color:var(--fake)}
  .conf-label{font-size:.56rem;letter-spacing:.12em;text-transform:uppercase;color:var(--muted)}
  .error-card{background:var(--surface);border:1px solid var(--fake);border-left:3px solid var(--fake);
              padding:14px 18px;margin-bottom:12px;font-size:.74rem;color:var(--fake);animation:slideIn .25s ease}
  .error-card strong{display:block;margin-bottom:4px;font-family:'Syne',sans-serif}
  footer{margin-top:56px;font-size:.62rem;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);text-align:center}
</style>
</head>
<body>

<header>
  <div class="header-inner">
    <div class="wordmark">Deep<span>Track</span></div>
    <div class="tagline">Swin Transformer &nbsp;·&nbsp; Image Authenticity Analysis</div>
    <div class="status-pill">MODEL ONLINE &nbsp;·&nbsp; /v1/image/predict</div>
  </div>
</header>

<div class="key-bar">
  <div class="key-bar-inner">
    <span class="key-label">API Key</span>
    <input type="password" id="api-key-input" class="key-input"
           placeholder="dt_your_key_here" oninput="saveKey()">
    <span class="key-status" id="key-status"></span>
  </div>
</div>

<div class="content">
  <div class="card">
    <div class="drop-zone" id="dropZone"
         ondragover="onDragOver(event)" ondragleave="onDragLeave(event)" ondrop="onDrop(event)"
         onclick="document.getElementById('fileInput').click()">
      <input type="file" id="fileInput" accept="image/*" multiple onchange="onFiles(this.files)">
      <span class="drop-icon">⬡</span>
      <div class="drop-label">Drop images or click to browse</div>
      <div class="drop-sub">JPG &nbsp;·&nbsp; PNG &nbsp;·&nbsp; WEBP &nbsp;·&nbsp; BMP &nbsp;|&nbsp; Multiple files supported</div>
    </div>
    <div class="preview-strip" id="previewStrip"></div>
    <div class="progress-wrap" id="progressWrap"><div class="progress-bar"></div></div>
    <button class="btn-analyze" id="analyzeBtn" disabled onclick="runAnalysis()">Analyze Images</button>
  </div>
</div>

<div id="results"></div>
<footer>DeepTrack &nbsp;·&nbsp; Swin-T deepfake classifier &nbsp;·&nbsp; internal build</footer>

<script>
  let pendingFiles = [];

  function saveKey() {
    const k = document.getElementById('api-key-input').value.trim();
    const status = document.getElementById('key-status');
    if (k.startsWith('dt_')) {
      status.textContent = '✓ Key set';
      status.style.color = '#47ffa3';
    } else if (k) {
      status.textContent = '⚠ Must start with dt_';
      status.style.color = '#ffaa00';
    } else {
      status.textContent = '';
    }
  }

  function getKey() {
    return document.getElementById('api-key-input').value.trim();
  }

  function onDragOver(e){e.preventDefault();document.getElementById('dropZone').classList.add('over')}
  function onDragLeave(){document.getElementById('dropZone').classList.remove('over')}
  function onDrop(e){e.preventDefault();document.getElementById('dropZone').classList.remove('over');onFiles(e.dataTransfer.files)}

  function onFiles(fileList) {
    const valid = ['image/jpeg','image/png','image/webp','image/bmp','image/gif'];
    for (const f of fileList) if (valid.includes(f.type)) pendingFiles.push(f);
    renderPreviews();
  }

  function renderPreviews() {
    const strip = document.getElementById('previewStrip');
    strip.innerHTML = '';
    if (!pendingFiles.length) {
      strip.classList.remove('visible');
      document.getElementById('analyzeBtn').disabled = true;
      document.querySelector('.drop-label').textContent = 'Drop images or click to browse';
      return;
    }
    strip.classList.add('visible');
    document.getElementById('analyzeBtn').disabled = false;
    document.querySelector('.drop-label').textContent =
      pendingFiles.length === 1 ? '1 image selected' : `${pendingFiles.length} images selected`;
    pendingFiles.forEach((f, idx) => {
      const url  = URL.createObjectURL(f);
      const wrap = document.createElement('div');
      wrap.className = 'thumb-wrap';
      wrap.innerHTML = `<img src="${url}"><button class="thumb-remove" onclick="removeFile(${idx},event)">&#x2715;</button>`;
      strip.appendChild(wrap);
    });
  }

  function removeFile(idx, e) { e.stopPropagation(); pendingFiles.splice(idx, 1); renderPreviews(); }

  let _uid = 0; const _idMap = new WeakMap();
  function uid(f) { if (!_idMap.has(f)) _idMap.set(f, ++_uid); return _idMap.get(f); }
  function esc(s) { return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }

  async function runAnalysis() {
    if (!pendingFiles.length) return;

    const key = getKey();
    if (!key) { alert('Please enter your API key above before analyzing.'); return; }

    const btn      = document.getElementById('analyzeBtn');
    const progress = document.getElementById('progressWrap');
    const results  = document.getElementById('results');
    btn.disabled = true; progress.classList.add('visible'); results.innerHTML = '';

    const batch = [...pendingFiles]; pendingFiles = []; renderPreviews();

    for (const file of batch) {
      const thumbUrl = URL.createObjectURL(file);
      const id       = uid(file);
      const fd       = new FormData(); fd.append('file', file);
      try {
        const resp = await fetch('/v1/image/predict', {
          method:  'POST',
          headers: { 'X-API-Key': key },
          body:    fd,
        });
        const data = await resp.json();
        if (!resp.ok) {
          results.insertAdjacentHTML('beforeend',
            `<div class="error-card"><strong>${esc(file.name)}</strong>${esc(data.detail || resp.statusText)}</div>`);
          continue;
        }
        const verdict  = data.prediction.toUpperCase();
        const realPct  = data.raw_scores.Real;
        const fakePct  = data.raw_scores.Fake;
        const conf     = data.confidence_percentage;
        const card     = document.createElement('div');
        card.className = `result-card ${verdict}`;
        card.innerHTML = `<div class="result-inner">
          <img class="result-thumb" src="${thumbUrl}" alt="">
          <div class="result-body">
            <div class="result-filename">${esc(file.name)}</div>
            <div class="result-verdict">${verdict}</div>
            <div class="prob-row"><span class="prob-label">Real</span>
              <div class="prob-track"><div class="prob-fill real" id="r${id}"></div></div>
              <span class="prob-pct">${realPct}%</span></div>
            <div class="prob-row"><span class="prob-label">Fake</span>
              <div class="prob-track"><div class="prob-fill fake" id="f${id}"></div></div>
              <span class="prob-pct">${fakePct}%</span></div>
          </div>
          <div class="confidence-badge">
            <div class="conf-value">${conf}%</div>
            <div class="conf-label">confidence</div>
          </div></div>`;
        results.appendChild(card);
        requestAnimationFrame(() => requestAnimationFrame(() => {
          const r = document.getElementById(`r${id}`);
          const f = document.getElementById(`f${id}`);
          if (r) r.style.width = realPct + '%';
          if (f) f.style.width = fakePct + '%';
        }));
      } catch (err) {
        results.insertAdjacentHTML('beforeend',
          `<div class="error-card"><strong>${esc(file.name)}</strong>${esc(err.message)}</div>`);
      }
    }
    progress.classList.remove('visible'); btn.disabled = false;
  }
</script>
</body>
</html>"""