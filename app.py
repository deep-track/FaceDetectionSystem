import os
import cv2
import sys
import base64
import logging
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from scipy.signal import butter, filtfilt, welch
from pydantic import BaseModel
from tensorflow import keras
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# config
CNN_MODEL_PATH  = "data/cnn_model.keras"
LANDMARKER_PATH = "data/face_landmarker.task"
OMEGA           = 128       # frames per segment (~4.3s @ 30fps)
N_SUBREGIONS    = 32
FS              = 30
LOW_HZ, HIGH_HZ = 0.7, 14.0

MID_FACE_32 = [
    1, 4, 5, 6, 8, 9, 10, 151,
    195, 197, 168, 107, 66, 105, 63, 70,
    336, 296, 334, 293, 300, 417,
    351, 399, 175, 152, 377, 400,
    378, 379, 365, 397
]

def butterworth_filter(signal):
    nyq  = FS / 2.0
    b, a = butter(4, [LOW_HZ/nyq, min(HIGH_HZ/nyq, 0.999)], btype='band')
    return filtfilt(b, a, signal) if len(signal) >= 15 else signal.copy()


def chrom_ppg_segment(R, G, B):
    total = R + G + B + 1e-9
    Rn, Gn, Bn = R/total, G/total, B/total
    Xs = 3*Rn - 2*Gn
    Ys = 1.5*Rn + Gn - 1.5*Bn
    return Xs - (np.std(Xs) / (np.std(Ys) + 1e-9)) * Ys


def norm_0_255(arr):
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-9:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mn) / (mx - mn) * 255.0).astype(np.float32)


def psd_binned(signal, n_bins):
    _, p  = welch(signal, fs=FS, nperseg=min(len(signal), 128))
    x_old = np.linspace(0, 1, len(p))
    x_new = np.linspace(0, 1, n_bins)
    return np.interp(x_new, x_old, p).astype(np.float32)


def build_ppg_map(R_bufs, G_bufs, B_bufs):
    """Build one omega×64 PPG map from RGB buffers of length omega."""
    ppg_cols = np.zeros((OMEGA, N_SUBREGIONS), dtype=np.float32)
    psd_cols = np.zeros((OMEGA, N_SUBREGIONS), dtype=np.float32)
    for i in range(N_SUBREGIONS):
        R_arr = np.array(R_bufs[i], dtype=np.float64)
        G_arr = np.array(G_bufs[i], dtype=np.float64)
        B_arr = np.array(B_bufs[i], dtype=np.float64)
        sig = butterworth_filter(chrom_ppg_segment(R_arr, G_arr, B_arr))
        ppg_cols[:, i] = norm_0_255(sig)
        psd_cols[:, i] = norm_0_255(psd_binned(sig, n_bins=OMEGA))
    spectral_map = np.concatenate([ppg_cols, psd_cols], axis=1)  # (128, 64)
    return (spectral_map[np.newaxis, ..., np.newaxis] / 255.0).astype(np.float32)


def sample_patch(frame_bgr, landmarks, lm_idx, h, w, patch_px=8):
    lm = landmarks[lm_idx]
    cx, cy = int(lm.x * w), int(lm.y * h)
    x0, x1 = max(0, cx - patch_px), min(w, cx + patch_px)
    y0, y1 = max(0, cy - patch_px), min(h, cy + patch_px)
    if x1 <= x0 or y1 <= y0:
        return None
    patch = frame_bgr[y0:y1, x0:x1].astype(np.float64)
    return patch[:, :, 2].mean(), patch[:, :, 1].mean(), patch[:, :, 0].mean()


class FrameBuffer:
    """
    Maintains a sliding window of OMEGA frames worth of per-subregion
    R/G/B means. When full, builds a PPG map and runs CNN inference.
    Slides by OMEGA//2 (50% overlap) so predictions come every ~2s.
    """
    def __init__(self):
        self.R = [deque(maxlen=OMEGA) for _ in range(N_SUBREGIONS)]
        self.G = [deque(maxlen=OMEGA) for _ in range(N_SUBREGIONS)]
        self.B = [deque(maxlen=OMEGA) for _ in range(N_SUBREGIONS)]
        self.frame_count  = 0
        self.last_result  = None
        self.step         = OMEGA // 2   # predict every 64 frames

    def push(self, rgb_values: list):
        """
        Push one frame's RGB means.
        rgb_values: list of (R, G, B) tuples, one per subregion.
                    Use (0,0,0) if subregion not detected.
        """
        for i, (r, g, b) in enumerate(rgb_values):
            self.R[i].append(r)
            self.G[i].append(g)
            self.B[i].append(b)
        self.frame_count += 1

    def ready(self) -> bool:
        """True when buffer is full AND it's time for a new prediction."""
        return (len(self.R[0]) == OMEGA and
                self.frame_count % self.step == 0)

    def get_map(self) -> np.ndarray:
        return build_ppg_map(
            [list(self.R[i]) for i in range(N_SUBREGIONS)],
            [list(self.G[i]) for i in range(N_SUBREGIONS)],
            [list(self.B[i]) for i in range(N_SUBREGIONS)],
        )

    def reset(self):
        for i in range(N_SUBREGIONS):
            self.R[i].clear(); self.G[i].clear(); self.B[i].clear()
        self.frame_count = 0
        self.last_result = None

    @property
    def fill_pct(self):
        return int(len(self.R[0]) / OMEGA * 100)

# model + landmarker
class Predictor:
    def __init__(self):
        logger.info("Loading CNN model...")
        self.model = keras.models.load_model(CNN_MODEL_PATH)
        logger.info("CNN model loaded.")

        logger.info("Loading MediaPipe FaceLandmarker...")
        base_opts = mp_python.BaseOptions(model_asset_path=LANDMARKER_PATH)
        options   = mp_vision.FaceLandmarkerOptions(
            base_options=base_opts,
            running_mode=mp_vision.RunningMode.IMAGE,   # IMAGE mode for per-frame API use
            num_faces=1,
            min_face_detection_confidence=0.4,
            min_face_presence_confidence=0.4,
        )
        self.landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self.lm_indices = MID_FACE_32[:N_SUBREGIONS]
        logger.info("FaceLandmarker loaded.")

    def extract_frame_rgb(self, frame_bgr):
        """
        Run face landmarker on one frame (IMAGE mode — no timestamp needed).
        Returns list of (R, G, B) tuples for each subregion, or zeros if no face.
        """
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result    = self.landmarker.detect(mp_image)

        if not result.face_landmarks:
            return [(0.0, 0.0, 0.0)] * N_SUBREGIONS

        landmarks = result.face_landmarks[0]
        rgb_vals  = []
        for lm_idx in self.lm_indices:
            rgb = sample_patch(frame_bgr, landmarks, lm_idx, h, w)
            rgb_vals.append(rgb if rgb is not None else (0.0, 0.0, 0.0))
        return rgb_vals

    def predict_map(self, ppg_map: np.ndarray) -> dict:
        """Run CNN on a (1, 128, 64, 1) PPG map. Returns prediction dict."""
        prob  = float(self.model.predict(ppg_map, verbose=0)[0][0])
        label = 'FAKE' if prob >= 0.5 else 'REAL'
        conf  = prob if prob >= 0.5 else 1.0 - prob
        return {
            'label':      label,
            'confidence': round(conf * 100, 1),
            'fake_prob':  round(prob, 4),
        }

    def close(self):
        self.landmarker.close()

# fastapi setup
predictor: Optional[Predictor]   = None
buffer:    Optional[FrameBuffer] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup
    global predictor, buffer
    if not os.path.exists(CNN_MODEL_PATH):
        logger.error(f"CNN model not found: {CNN_MODEL_PATH}")
        logger.error("Set CNN_MODEL env var or place model at data/cnn_model.keras")
        sys.exit(1)
    if not os.path.exists(LANDMARKER_PATH):
        logger.error(f"Landmarker not found: {LANDMARKER_PATH}")
        sys.exit(1)
    predictor = Predictor()
    buffer    = FrameBuffer()
    logger.info("API ready.")

    yield  # server runs here

    # ── Shutdown
    if predictor:
        predictor.close()


app = FastAPI(title="FakeCatcher API", version="1.0", lifespan=lifespan)

# http endpoints
class FrameRequest(BaseModel):
    image: str   # base64-encoded JPEG


@app.post("/predict/frame")
async def predict_frame(req: FrameRequest):
    """
    Send one webcam frame as a base64 JPEG.
    Returns prediction once enough frames are buffered (omega=128).
    Before that, returns buffering status with fill percentage.

    Example (Python client):
        import base64, requests, cv2
        cap = cv2.VideoCapture(0)
        while True:
            _, frame = cap.read()
            _, buf = cv2.imencode('.jpg', frame)
            b64 = base64.b64encode(buf).decode()
            resp = requests.post('http://localhost:8000/predict/frame',
                                 json={'image': b64})
            print(resp.json())
    """
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
            'status':       'prediction',
            'label':        result['label'],
            'confidence':   result['confidence'],
            'fake_prob':    result['fake_prob'],
            'frames_seen':  buffer.frame_count,
        }

    return {
        'status':      'buffering',
        'fill_pct':    buffer.fill_pct,
        'frames_seen': buffer.frame_count,
        'message':     f"Buffering... {buffer.fill_pct}% — need {OMEGA} frames (~{OMEGA//FS}s)"
    }


@app.get("/status")
async def status():
    return {
        'buffer_fill_pct': buffer.fill_pct,
        'frames_seen':     buffer.frame_count,
        'omega':           OMEGA,
        'last_result':     buffer.last_result,
        'model':           CNN_MODEL_PATH,
    }


@app.post("/reset")
async def reset():
    buffer.reset()
    return {'status': 'reset', 'message': 'Frame buffer cleared.'}


@app.websocket("/ws/predict")
async def websocket_predict(ws: WebSocket):
    """
    WebSocket endpoint for real-time webcam streaming.

    Client sends: base64-encoded JPEG string (one per frame)
    Server sends: JSON with prediction or buffering status

    JavaScript example (browser):
        const ws = new WebSocket('ws://localhost:8000/ws/predict');
        // Send frames from getUserMedia canvas
        ws.send(canvas.toDataURL('image/jpeg', 0.7).split(',')[1]);
        ws.onmessage = (e) => console.log(JSON.parse(e.data));
    """
    await ws.accept()
    ws_buffer = FrameBuffer()   # each WebSocket connection gets its own buffer
    logger.info("WebSocket client connected")

    try:
        while True:
            data = await ws.receive_text()

            try:
                img_bytes = base64.b64decode(data)
                arr       = np.frombuffer(img_bytes, dtype=np.uint8)
                frame     = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is None:
                    await ws.send_json({'error': 'Could not decode frame'})
                    continue
            except Exception as e:
                await ws.send_json({'error': str(e)})
                continue

            rgb_vals = predictor.extract_frame_rgb(frame)
            ws_buffer.push(rgb_vals)

            if ws_buffer.ready():
                ppg_map = ws_buffer.get_map()
                result  = predictor.predict_map(ppg_map)
                ws_buffer.last_result = result
                await ws.send_json({
                    'status':      'prediction',
                    'label':       result['label'],
                    'confidence':  result['confidence'],
                    'fake_prob':   result['fake_prob'],
                    'frames_seen': ws_buffer.frame_count,
                })
            else:
                await ws.send_json({
                    'status':      'buffering',
                    'fill_pct':    ws_buffer.fill_pct,
                    'frames_seen': ws_buffer.frame_count,
                })

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")


# browser UI
@app.get("/", response_class=HTMLResponse)
async def demo_ui():
    """Simple browser demo — open http://localhost:8000 to use your webcam live."""
    return """
<!DOCTYPE html>
<html>
<head>
  <title>FakeCatcher — Live Demo</title>
  <style>
    body { font-family: Arial, sans-serif; background: #111; color: #eee;
           display: flex; flex-direction: column; align-items: center; padding: 30px; }
    h1   { color: #4af; }
    #video   { border: 3px solid #333; border-radius: 8px; }
    #status  { font-size: 2em; margin: 20px 0; font-weight: bold; }
    #bar-wrap { width: 480px; background: #333; border-radius: 4px; height: 16px; }
    #bar      { height: 16px; background: #4af; border-radius: 4px;
                width: 0%; transition: width 0.2s; }
    #log     { font-size: 0.85em; color: #aaa; margin-top: 10px; }
    .FAKE    { color: #f55; }
    .REAL    { color: #5f5; }
    button   { margin: 8px; padding: 10px 24px; font-size: 1em;
               border: none; border-radius: 6px; cursor: pointer; }
    #startBtn { background: #4af; color: #000; }
    #resetBtn { background: #555; color: #eee; }

    /* real/fake probability bars */
    #prob-section { width: 480px; margin-top: 16px; display: none; }
    .prob-row     { display: flex; align-items: center; margin: 6px 0; gap: 10px; }
    .prob-label   { width: 46px; font-weight: bold; font-size: 0.95em; }
    .prob-track   { flex: 1; background: #333; border-radius: 4px; height: 22px;
                    overflow: hidden; }
    .prob-fill    { height: 100%; border-radius: 4px; transition: width 0.4s ease;
                    display: flex; align-items: center; padding-left: 8px;
                    font-size: 0.8em; font-weight: bold; color: #000; white-space: nowrap; }
    #real-fill    { background: #5f5; }
    #fake-fill    { background: #f55; }
    .prob-pct     { width: 46px; text-align: right; font-size: 0.9em; }
    #real-pct     { color: #5f5; }
    #fake-pct     { color: #f55; }
  </style>
</head>
<body>
  <h1>FakeCatcher — Live Webcam Demo</h1>
  <video id="video" width="480" height="360" autoplay muted></video>
  <canvas id="canvas" width="480" height="360" style="display:none"></canvas>
  <div id="status">Press Start to begin</div>
  <div id="status">Please give the system 2-10s to boot up</div>
  <div id="bar-wrap"><div id="bar"></div></div>
  <div id="log"></div>

  <!-- real/fake probability bars (hidden until first prediction) -->
  <div id="prob-section">
    <div class="prob-row">
      <span class="prob-label" style="color:#5f5">REAL</span>
      <div class="prob-track">
        <div class="prob-fill" id="real-fill"></div>
      </div>
      <span class="prob-pct" id="real-pct">—</span>
    </div>
    <div class="prob-row">
      <span class="prob-label" style="color:#f55">FAKE</span>
      <div class="prob-track">
        <div class="prob-fill" id="fake-fill"></div>
      </div>
      <span class="prob-pct" id="fake-pct">—</span>
    </div>
  </div>

  <div>
    <button id="startBtn" onclick="startCapture()">&#9654; Start</button>
    <button id="resetBtn" onclick="resetBuffer()">&#8635; Reset</button>
  </div>

<script>
  let ws, interval, streaming = false;

  function updateProbBars(fakeProb) {
    const realProb = 1.0 - fakeProb;
    const realPct  = (realProb * 100).toFixed(1);
    const fakePct  = (fakeProb * 100).toFixed(1);

    document.getElementById('prob-section').style.display = 'block';

    document.getElementById('real-fill').style.width = realPct + '%';
    document.getElementById('real-fill').textContent  = realPct > 10 ? realPct + '%' : '';
    document.getElementById('real-pct').textContent   = realPct + '%';

    document.getElementById('fake-fill').style.width = fakePct + '%';
    document.getElementById('fake-fill').textContent  = fakePct > 10 ? fakePct + '%' : '';
    document.getElementById('fake-pct').textContent   = fakePct + '%';
  }

  async function startCapture() {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    document.getElementById('video').srcObject = stream;

    const wsProtocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(wsProtocol + '//' + location.host + '/ws/predict');

    ws.onmessage = (e) => {
      const d = JSON.parse(e.data);
      const statusEl = document.getElementById('status');
      const barEl    = document.getElementById('bar');
      const logEl    = document.getElementById('log');

      if (d.status === 'prediction') {
        statusEl.textContent = d.label + ' (' + d.confidence + '%)';
        statusEl.className   = d.label;
        barEl.style.width    = '100%';
        logEl.textContent    = 'Frames seen: ' + d.frames_seen +
                               ' | P(fake): ' + d.fake_prob;
        updateProbBars(d.fake_prob);
      } else {
        statusEl.textContent = 'Buffering... ' + d.fill_pct + '%';
        statusEl.className   = '';
        barEl.style.width    = d.fill_pct + '%';
      }
    };

    ws.onclose = () => {
      document.getElementById('status').textContent = 'Disconnected';
    };

    const canvas = document.getElementById('canvas');
    const ctx    = canvas.getContext('2d');

    interval = setInterval(() => {
      if (ws.readyState !== WebSocket.OPEN) return;
      ctx.drawImage(document.getElementById('video'), 0, 0, 480, 360);
      const b64 = canvas.toDataURL('image/jpeg', 0.6).split(',')[1];
      ws.send(b64);
    }, 1000 / 15);   // 15 fps to server (model trained at 30fps but this saves bandwidth)

    streaming = true;
    document.getElementById('startBtn').textContent = 'Running';
  }

  async function resetBuffer() {
    await fetch('/reset', { method: 'POST' });
    document.getElementById('status').textContent = 'Buffer reset';
    document.getElementById('bar').style.width = '0%';
    document.getElementById('prob-section').style.display = 'none';
  }
</script>
</body>
</html>
"""