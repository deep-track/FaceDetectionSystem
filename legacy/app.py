import os
import cv2
import sys
import time
import uuid
import base64
import asyncio
import logging
import tempfile
import numpy as np
import mediapipe as mp
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from pydantic import BaseModel
from scipy.signal import butter, filtfilt, welch
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# config
CNN_MODEL_PATH  = "data/cnn_model.keras"
LANDMARKER_PATH = "data/face_landmarker.task"
OMEGA           = 128
N_SUBREGIONS    = 32
FS              = 30
LOW_HZ, HIGH_HZ = 0.7, 14.0
THRESHOLD_FAKE      = 0.65
THRESHOLD_UNCERTAIN = 0.45
SMOOTHING_WINDOW    = 3    # segments to average on webcam
MAX_UPLOAD_MB       = 50   # reject files larger than 50mb
MAX_VIDEO_WORKERS   = 4    # concurrent video jobs
JOB_TTL_SECONDS     = 7200 # keep jobs for 2 hours

MID_FACE_32 = [
    1, 4, 5, 6, 8, 9, 10, 151,
    195, 197, 168, 107, 66, 105, 63, 70,
    336, 296, 334, 293, 300, 417,
    351, 399, 175, 152, 377, 400,
    378, 379, 365, 397
]

# in-memory job store for async video processing
jobs: dict = {}

# bounded thread pool, limits concurrent video jobs to max workers
video_executor = ThreadPoolExecutor(max_workers=MAX_VIDEO_WORKERS,
                                    thread_name_prefix='fakecatcher-video')

class FrameRequest(BaseModel):
    image: str   # base64-encoded JPEG

# signal helper funcs
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

def signal_quality(R_bufs, G_bufs, B_bufs):
    # returns (quality_ok, reason, snr) — UNCERTAIN label shown if quality fails
    # fails when face missing >40% of frames or signal SNR too low (poor lighting)
    snrs, zero_counts = [], []
    for i in range(min(8, N_SUBREGIONS)):   # sample 8 subregions for speed
        R_arr = np.array(R_bufs[i], dtype=np.float64)
        zero_pct = (R_arr == 0).mean()
        zero_counts.append(zero_pct)
        if zero_pct < 0.6:
            sig = butterworth_filter(chrom_ppg_segment(R_arr,
                      np.array(G_bufs[i], dtype=np.float64),
                      np.array(B_bufs[i], dtype=np.float64)))
            snr = np.mean(sig ** 2) / (np.std(np.diff(sig)) ** 2 + 1e-9)
            snrs.append(snr)

    avg_zero = np.mean(zero_counts)
    avg_snr  = np.mean(snrs) if snrs else 0.0

    if avg_zero > 0.40:
        return False, f"Face not detected in {avg_zero*100:.0f}% of frames — check lighting and camera angle", avg_snr
    if avg_snr < 0.008:
        return False, f"Signal quality too low (SNR={avg_snr:.4f}) — improve lighting for accurate results", avg_snr
    return True, "ok", avg_snr

def classify_prob(prob: float) -> tuple:
    # three-zone classification: REAL | UNCERTAIN | FAKE
    # avoids false accusations on marginal predictions near the boundary
    if prob >= THRESHOLD_FAKE:
        return 'FAKE', round(prob * 100, 1)
    elif prob >= THRESHOLD_UNCERTAIN:
        return 'UNCERTAIN', round((1.0 - abs(prob - 0.5) * 2) * 100, 1)
    else:
        return 'REAL', round((1.0 - prob) * 100, 1)

def sample_patch(frame_bgr, landmarks, lm_idx, h, w, patch_px=8):
    lm = landmarks[lm_idx]
    cx, cy = int(lm.x * w), int(lm.y * h)
    x0, x1 = max(0, cx - patch_px), min(w, cx + patch_px)
    y0, y1 = max(0, cy - patch_px), min(h, cy + patch_px)
    if x1 <= x0 or y1 <= y0:
        return None
    patch = frame_bgr[y0:y1, x0:x1].astype(np.float64)
    return patch[:, :, 2].mean(), patch[:, :, 1].mean(), patch[:, :, 0].mean()

# frame buffer with a sliding window, 50% overlap
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
        self.step         = OMEGA // 2                      # predict every 64 frames
        self.prob_history = deque(maxlen=SMOOTHING_WINDOW)  # rolling window of recent probs

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
        self.frame_count  = 0
        self.last_result  = None
        self.prob_history.clear()

    @property
    def fill_pct(self):
        return int(len(self.R[0]) / OMEGA * 100)

# init model and landmarker
class Predictor:
    def __init__(self):
        logger.info("Loading CNN model...")
        import tensorflow as tf
        from tensorflow import keras
        self.model = keras.models.load_model(CNN_MODEL_PATH)
        logger.info("CNN model loaded.")

        logger.info("Loading MediaPipe FaceLandmarker...")
        base_opts = mp_python.BaseOptions(model_asset_path=LANDMARKER_PATH)

        # image mode for webcam
        image_opts = mp_vision.FaceLandmarkerOptions(
            base_options=base_opts,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.4,
            min_face_presence_confidence=0.4,
        )
        self.landmarker      = mp_vision.FaceLandmarker.create_from_options(image_opts)
        self.lm_indices      = MID_FACE_32[:N_SUBREGIONS]
        self.landmarker_path = LANDMARKER_PATH  # stored for per-request VIDEO mode instances
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

    def predict_map(self, ppg_map: np.ndarray,
                    quality_ok: bool = True, quality_reason: str = "ok",
                    snr: float = 1.0) -> dict:
        """Run CNN on a (1, 128, 64, 1) PPG map. Returns prediction dict."""
        if not quality_ok:
            # signal too noisy to make a reliable call, return UNCERTAIN instead of guessing
            return {
                'label':      'UNCERTAIN',
                'confidence': 0.0,
                'fake_prob':  None,
                'warning':    quality_reason,
            }
        prob        = float(self.model.predict(ppg_map, verbose=0)[0][0])
        label, conf = classify_prob(prob)
        return {
            'label':      label,
            'confidence': conf,
            'fake_prob':  round(prob, 4),
            'snr':        round(snr, 4),
        }

    def predict_video(self, video_path: str) -> dict:
        """
        Run full PPG extraction + CNN inference on a video file.
        Processes all segments and aggregates by averaging fake_prob
        across segments — same majority-vote approach used in training.
        Returns per-segment results and a final video-level verdict.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0 or video_fps > 240:
            video_fps = FS

        # collect per-subregion RGB signals using VIDEO mode — same as training
        # fresh landmarker per call so timestamps always start from 0 (shared
        # instance would reject ts=0 if a previous upload already advanced it)
        map_R = [[] for _ in range(N_SUBREGIONS)]
        map_G = [[] for _ in range(N_SUBREGIONS)]
        map_B = [[] for _ in range(N_SUBREGIONS)]
        frame_idx   = 0
        face_frames = 0

        base_opts  = mp_python.BaseOptions(model_asset_path=self.landmarker_path)
        video_opts = mp_vision.FaceLandmarkerOptions(
            base_options=base_opts,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        with mp_vision.FaceLandmarker.create_from_options(video_opts) as video_lm:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break

                h, w      = frame_bgr.shape[:2]
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                ts_ms     = int(frame_idx * 1000.0 / video_fps)
                result    = video_lm.detect_for_video(mp_image, ts_ms)

                if not result.face_landmarks:
                    for i in range(N_SUBREGIONS):
                        map_R[i].append(0.0); map_G[i].append(0.0); map_B[i].append(0.0)
                else:
                    face_frames += 1
                    landmarks = result.face_landmarks[0]
                    for i, lm_idx in enumerate(self.lm_indices):
                        rgb = sample_patch(frame_bgr, landmarks, lm_idx, h, w)
                        r, g, b = rgb if rgb is not None else (0.0, 0.0, 0.0)
                        map_R[i].append(r); map_G[i].append(g); map_B[i].append(b)

                frame_idx += 1

        cap.release()

        total_frames = frame_idx
        if total_frames < OMEGA:
            raise ValueError(
                f"Video too short: {total_frames} frames, need at least {OMEGA} "
                f"(~{OMEGA//FS}s at 30fps)"
            )

        face_pct = round(face_frames / total_frames * 100, 1) if total_frames else 0
        if face_pct < 20:
            raise ValueError(
                f"Face detected in only {face_pct}% of frames — "
                "ensure the face is clearly visible throughout the video"
            )

        # slice into overlapping segments and predict each one
        step     = OMEGA // 2
        segments = []
        probs    = []

        for start in range(0, total_frames - OMEGA + 1, step):
            end     = start + OMEGA
            ppg_map = build_ppg_map(
                [map_R[i][start:end] for i in range(N_SUBREGIONS)],
                [map_G[i][start:end] for i in range(N_SUBREGIONS)],
                [map_B[i][start:end] for i in range(N_SUBREGIONS)],
            )
            result = self.predict_map(ppg_map)
            probs.append(result['fake_prob'])
            segments.append({
                'segment':    len(segments) + 1,
                'start_sec':  round(start / FS, 1),
                'end_sec':    round(end   / FS, 1),
                'label':      result['label'],
                'confidence': result['confidence'],
                'fake_prob':  result['fake_prob'],
            })

        # video-level verdict: average prob across all segments
        avg_prob          = float(np.mean(probs))
        label, confidence = classify_prob(avg_prob)

        return {
            'label':        label,
            'confidence':   confidence,
            'fake_prob':    round(avg_prob, 4),
            'total_frames': total_frames,
            'face_pct':     face_pct,
            'n_segments':   len(segments),
            'segments':     segments,
        }

    def close(self):
        self.landmarker.close()

# init fastapi app
predictor: Optional[Predictor]   = None
buffer:    Optional[FrameBuffer] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
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

    # shutdown
    video_executor.shutdown(wait=False)
    if predictor:
        predictor.close()

app = FastAPI(title="FakeCatcher API", version="1.0", lifespan=lifespan)

@app.post("/predict/frame")
async def predict_frame(req: FrameRequest):
    """
    Send one webcam frame as a base64 JPEG.
    Returns prediction once enough frames are buffered (omega=128).
    Before that, returns buffering status with fill percentage.
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
            'status':      'prediction',
            'label':       result['label'],
            'confidence':  result['confidence'],
            'fake_prob':   result['fake_prob'],
            'frames_seen': buffer.frame_count,
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


@app.get("/health")
async def health():
    """Liveness check, returns 200 if server and model are ready."""
    if predictor is None:
        raise HTTPException(503, "Model not loaded")
    return {'status': 'ok', 'model': 'operational', 'server': 'operational'}


@app.get("/metrics")
async def metrics():
    """Server metrics for monitoring during stress tests."""
    job_counts = {}
    for j in jobs.values():
        job_counts[j['status']] = job_counts.get(j['status'], 0) + 1
    return {
        'jobs':          job_counts,
        'total_jobs':    len(jobs),
        'workers':       MAX_VIDEO_WORKERS,
        'max_upload_mb': MAX_UPLOAD_MB,
        'job_ttl_hours': JOB_TTL_SECONDS // 3600,
    }


@app.post("/reset")
async def reset():
    buffer.reset()
    return {'status': 'reset', 'message': 'Frame buffer cleared.'}


def _run_video_job(job_id: str, tmp_path: str, filename: str):
    """Worker function, runs in thread pool, updates jobs dict when done."""
    try:
        jobs[job_id]['status'] = 'processing'
        result = predictor.predict_video(tmp_path)
        jobs[job_id].update({'status': 'done', 'result': result})
        logger.info(f"Job {job_id} done: {result['label']} ({result['confidence']}%)")
    except ValueError as e:
        jobs[job_id].update({'status': 'error', 'error': str(e)})
        logger.warning(f"Job {job_id} validation error: {e}")
    except Exception as e:
        jobs[job_id].update({'status': 'error', 'error': f"Processing error: {e}"})
        logger.error(f"Job {job_id} failed: {e}")
    finally:
        # release temp file, retry on Windows file lock
        for _ in range(5):
            try:
                os.unlink(tmp_path)
                break
            except PermissionError:
                time.sleep(0.2)


def _purge_old_jobs():
    """Remove jobs older than JOB_TTL_SECONDS to prevent memory growth."""
    cutoff = time.time() - JOB_TTL_SECONDS
    stale  = [jid for jid, j in jobs.items() if j['created_at'] < cutoff]
    for jid in stale:
        del jobs[jid]
    if stale:
        logger.info(f"Purged {len(stale)} stale jobs")


@app.post("/predict/video")
async def predict_video(file: UploadFile = File(...)):
    """
    Upload a video file for deepfake analysis.
    Returns a job_id immediately — poll GET /jobs/{job_id} for the result.
    This is non-blocking so multiple uploads can be processed concurrently.
    """
    # file format check
    allowed = {'.mp4', '.avi', '.mov', '.mkv'}
    ext     = os.path.splitext(file.filename or '')[1].lower()
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported format '{ext}'. Use: {', '.join(allowed)}")

    # read and size-check before writing to disk
    data = await file.read()
    mb   = len(data) / 1024 / 1024
    if mb > MAX_UPLOAD_MB:
        raise HTTPException(413, f"File too large ({mb:.1f}MB). Max is {MAX_UPLOAD_MB}MB.")

    # write to temp file — cv2.VideoCapture needs a real path
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    # create job record and submit to bounded thread pool
    _purge_old_jobs()
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        'status':     'queued',
        'filename':   file.filename,
        'size_mb':    round(mb, 2),
        'created_at': time.time(),
        'result':     None,
        'error':      None,
    }
    video_executor.submit(_run_video_job, job_id, tmp_path, file.filename)
    logger.info(f"Job {job_id} queued: {file.filename} ({mb:.1f}MB)")

    return {
        'job_id':   job_id,
        'status':   'queued',
        'filename': file.filename,
        'size_mb':  round(mb, 2),
        'poll_url': f"/jobs/{job_id}",
        'message':  f"Job queued. Poll /jobs/{job_id} for result.",
    }


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """Poll for the result of a video prediction job."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found. Jobs expire after {JOB_TTL_SECONDS//3600}h.")
    return {
        'job_id':   job_id,
        'status':   job['status'],
        'filename': job['filename'],
        'size_mb':  job['size_mb'],
        'result':   job.get('result'),
        'error':    job.get('error'),
        'age_sec':  round(time.time() - job['created_at']),
    }


@app.get("/jobs")
async def list_jobs():
    """List all active jobs — useful for monitoring stress test progress."""
    summary = {
        jid: {
            'status':   j['status'],
            'filename': j['filename'],
            'age_sec':  round(time.time() - j['created_at']),
        }
        for jid, j in jobs.items()
    }
    counts = {}
    for j in jobs.values():
        counts[j['status']] = counts.get(j['status'], 0) + 1
    return {'jobs': summary, 'counts': counts, 'total': len(jobs)}


@app.websocket("/ws/predict")
async def websocket_predict(ws: WebSocket):
    """
    WebSocket endpoint for real-time webcam streaming.

    Client sends: base64-encoded JPEG string (one per frame)
    Server sends: JSON with prediction or buffering status
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
                # run signal quality check before prediction
                q_ok, q_reason, snr = signal_quality(
                    [list(ws_buffer.R[i]) for i in range(N_SUBREGIONS)],
                    [list(ws_buffer.G[i]) for i in range(N_SUBREGIONS)],
                    [list(ws_buffer.B[i]) for i in range(N_SUBREGIONS)],
                )
                result = predictor.predict_map(ppg_map, q_ok, q_reason, snr)

                # smooth verdict across last SMOOTHING_WINDOW segments
                # avoids single noisy windows causing false FAKE on real faces
                if result['fake_prob'] is not None:
                    ws_buffer.prob_history.append(result['fake_prob'])
                smoothed_prob = float(np.mean(ws_buffer.prob_history)) if ws_buffer.prob_history else None
                if smoothed_prob is not None and result['label'] != 'UNCERTAIN':
                    label, conf          = classify_prob(smoothed_prob)
                    result['label']      = label
                    result['confidence'] = conf
                    result['fake_prob']  = round(smoothed_prob, 4)

                ws_buffer.last_result = result
                await ws.send_json({
                    'status':        'prediction',
                    'label':         result['label'],
                    'confidence':    result['confidence'],
                    'fake_prob':     result['fake_prob'],
                    'warning':       result.get('warning'),
                    'frames_seen':   ws_buffer.frame_count,
                    'segments_seen': len(ws_buffer.prob_history),
                })
            else:
                await ws.send_json({
                    'status':      'buffering',
                    'fill_pct':    ws_buffer.fill_pct,
                    'frames_seen': ws_buffer.frame_count,
                })

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")


@app.get("/", response_class=HTMLResponse)
async def demo_ui():
    """Browser demo with two tabs: live webcam and video file upload."""
    return """
<!DOCTYPE html>
<html>
<head>
  <title>FakeCatcher - Live Demo</title>
  <style>
    * { box-sizing: border-box; }
    body { font-family: Arial, sans-serif; background: #111; color: #eee;
           display: flex; flex-direction: column; align-items: center; padding: 30px; margin: 0; }
    h1   { color: #4af; margin-bottom: 6px; }
    p.sub { color: #888; font-size: 0.85em; margin-top: 0; margin-bottom: 20px; }

    /* tabs */
    .tabs      { display: flex; gap: 4px; margin-bottom: 20px; }
    .tab-btn   { padding: 10px 28px; border: none; border-radius: 6px 6px 0 0;
                 font-size: 1em; cursor: pointer; background: #2a2a2a; color: #aaa; }
    .tab-btn.active { background: #1e3a5f; color: #4af; font-weight: bold; }
    .tab-panel { display: none; flex-direction: column; align-items: center; width: 520px;
                 background: #1a1a1a; border-radius: 0 8px 8px 8px; padding: 24px; }
    .tab-panel.active { display: flex; }

    /* shared result area */
    .status    { font-size: 2em; margin: 16px 0 8px; font-weight: bold; min-height: 1.4em; }
    .FAKE      { color: #f55; }
    .REAL      { color: #5f5; }
    .UNCERTAIN { color: #fa0; }   /* amber — signal too noisy to classify */
    .bar-wrap { width: 100%; background: #333; border-radius: 4px; height: 14px; margin-bottom: 6px; }
    .bar      { height: 14px; background: #4af; border-radius: 4px; width: 0%;
                transition: width 0.3s; }
    .log      { font-size: 0.8em; color: #888; margin-bottom: 12px; min-height: 1.2em; }
    .prob-section { width: 100%; margin-top: 8px; display: none; }
    .prob-row     { display: flex; align-items: center; margin: 5px 0; gap: 10px; }
    .prob-label   { width: 46px; font-weight: bold; font-size: 0.9em; }
    .prob-track   { flex: 1; background: #333; border-radius: 4px; height: 20px; overflow: hidden; }
    .prob-fill    { height: 100%; border-radius: 4px; transition: width 0.4s ease;
                    display: flex; align-items: center; padding-left: 8px;
                    font-size: 0.78em; font-weight: bold; color: #000; white-space: nowrap; }
    .real-fill    { background: #5f5; }
    .fake-fill    { background: #f55; }
    .prob-pct     { width: 46px; text-align: right; font-size: 0.85em; }

    /* webcam tab */
    #video    { border: 3px solid #333; border-radius: 8px; width: 100%; }
    .cam-row  { display: flex; align-items: center; gap: 8px; width: 100%; margin-bottom: 10px; }
    #cam-sel  { flex: 1; background: #222; color: #eee; border: 1px solid #555;
                border-radius: 6px; padding: 6px 10px; font-size: 0.88em; cursor: pointer; }
    .btn-row  { display: flex; gap: 8px; margin-top: 12px; }
    button.primary { padding: 10px 26px; font-size: 1em; border: none; border-radius: 6px;
                     cursor: pointer; background: #4af; color: #000; }
    button.secondary { padding: 10px 20px; font-size: 1em; border: none; border-radius: 6px;
                       cursor: pointer; background: #444; color: #eee; }

    /* upload tab */
    .drop-zone { width: 100%; border: 2px dashed #555; border-radius: 10px;
                 padding: 40px 20px; text-align: center; cursor: pointer;
                 transition: border-color 0.2s; margin-bottom: 16px; }
    .drop-zone:hover, .drop-zone.dragover { border-color: #4af; }
    .drop-zone p { margin: 6px 0; color: #aaa; font-size: 0.9em; }
    .drop-zone .icon { font-size: 2.5em; }
    #file-input { display: none; }
    #file-name  { font-size: 0.85em; color: #5af; margin-bottom: 10px; min-height: 1.2em; }
    #analyze-btn { width: 100%; }

    /* segment breakdown table */
    #seg-section { width: 100%; margin-top: 16px; display: none; }
    #seg-section h3 { font-size: 0.95em; color: #aaa; margin: 0 0 8px; }
    table { width: 100%; border-collapse: collapse; font-size: 0.82em; }
    th    { background: #2a2a2a; color: #888; padding: 6px 8px; text-align: left; }
    td    { padding: 5px 8px; border-bottom: 1px solid #2a2a2a; }
    tr.fake-row      { background: #2a1515; }
    tr.real-row      { background: #152a15; }
    tr.uncertain-row { background: #2a2010; }
    .pill { display: inline-block; padding: 2px 8px; border-radius: 10px;
            font-size: 0.85em; font-weight: bold; }
    .pill.FAKE      { background: #f55; color: #000; }
    .pill.REAL      { background: #5f5; color: #000; }
    .pill.UNCERTAIN { background: #fa0; color: #000; }

    /* spinner */
    .spinner { display: none; width: 36px; height: 36px; border: 4px solid #333;
               border-top-color: #4af; border-radius: 50%;
               animation: spin 0.8s linear infinite; margin: 12px auto; }
    @keyframes spin { to { transform: rotate(360deg); } }
  </style>
</head>
<body>
  <h1>FakeCatcher</h1>
  <p class="sub">Deepfake detection via rPPG biological signals</p>

  <!-- tab buttons -->
  <div class="tabs">
    <button class="tab-btn active" onclick="switchTab('webcam', this)">Live Webcam</button>
    <button class="tab-btn"        onclick="switchTab('upload', this)">Upload Video</button>
  </div>

  <!-- ── WEBCAM TAB ── -->
  <div class="tab-panel active" id="tab-webcam">
    <div class="cam-row">
      <label style="font-size:0.85em;color:#888;white-space:nowrap">Camera:</label>
      <select id="cam-sel"><option>Loading...</option></select>
      <button class="secondary" style="padding:6px 12px;font-size:0.85em"
              onclick="refreshCameras()">Refresh</button>
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
      <button class="primary"   id="startBtn"  onclick="startCapture()">&#9654; Start</button>
      <button class="secondary" id="resetBtn"  onclick="resetBuffer()">&#8635; Reset</button>
    </div>
    <p style="font-size:0.75em;color:#555;margin-top:14px">
      Tip: for deepfake filter testing, select OBS Virtual Camera with DeepFaceLive running
    </p>
  </div>

  <!-- ── UPLOAD TAB ── -->
  <div class="tab-panel" id="tab-upload">
    <div class="drop-zone" id="drop-zone" onclick="document.getElementById('file-input').click()"
         ondragover="onDragOver(event)" ondragleave="onDragLeave(event)" ondrop="onDrop(event)">
      <div class="icon">&#127909;</div>
      <p><strong>Click to upload</strong> or drag and drop a video</p>
      <p style="font-size:0.8em">MP4, AVI, MOV, MKV &nbsp;|&nbsp; Face must be visible &nbsp;|&nbsp; Min ~4s</p>
    </div>
    <input type="file" id="file-input" accept=".mp4,.avi,.mov,.mkv"
           onchange="onFileSelected(this.files[0])">
    <div id="file-name"></div>

    <button class="primary" id="analyze-btn" onclick="analyzeVideo()" disabled>
      Analyze Video
    </button>
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
        <span class="prob-pct" id="up-real-pct">-</span>
      </div>
      <div class="prob-row">
        <span class="prob-label" style="color:#f55">FAKE</span>
        <div class="prob-track"><div class="prob-fill fake-fill" id="up-fake-fill"></div></div>
        <span class="prob-pct" id="up-fake-pct">-</span>
      </div>
    </div>

    <!-- per-segment breakdown table, shown after prediction -->
    <div id="seg-section">
      <h3>Segment breakdown</h3>
      <table>
        <thead>
          <tr><th>#</th><th>Time</th><th>Result</th><th>Confidence</th><th>P(fake)</th></tr>
        </thead>
        <tbody id="seg-tbody"></tbody>
      </table>
    </div>
  </div>

<script>
  // ── Tab switching
  function switchTab(name, btn) {
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.getElementById('tab-' + name).classList.add('active');
    btn.classList.add('active');
  }

  // ── Shared prob bar updater
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

  // ────────────────────────────────────────
  //  WEBCAM TAB
  // ────────────────────────────────────────
  let ws, interval, currentStream = null;

  async function refreshCameras() {
    try {
      const tmp = await navigator.mediaDevices.getUserMedia({ video: true });
      tmp.getTracks().forEach(t => t.stop());
    } catch(e) {}

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
    } catch(e) {
      document.getElementById('wc-status').textContent = 'Camera error: ' + e.message;
      return;
    }
    document.getElementById('video').srcObject = currentStream;

    const wsProtocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(wsProtocol + '//' + location.host + '/ws/predict');

    ws.onmessage = (e) => {
      const d = JSON.parse(e.data);
      const statusEl = document.getElementById('wc-status');
      if (d.status === 'prediction') {
        if (d.label === 'UNCERTAIN') {
          // signal quality too low — show warning in amber instead of REAL/FAKE
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
            'Frames seen: ' + d.frames_seen + ' | P(fake): ' + d.fake_prob;
          updateProbs('wc', d.fake_prob);
        }
      } else {
        statusEl.textContent = 'Buffering... ' + d.fill_pct + '%';
        statusEl.className   = 'status';
        document.getElementById('wc-bar').style.width = d.fill_pct + '%';
      }
    };
    ws.onclose = () => { document.getElementById('wc-status').textContent = 'Disconnected'; };

    const canvas = document.getElementById('canvas') || (() => {
      const c = document.createElement('canvas');
      c.width = 480; c.height = 360; c.style.display = 'none';
      document.body.appendChild(c); return c;
    })();
    const ctx = canvas.getContext('2d');

    interval = setInterval(() => {
      if (ws.readyState !== WebSocket.OPEN) return;
      ctx.drawImage(document.getElementById('video'), 0, 0, 480, 360);
      ws.send(canvas.toDataURL('image/jpeg', 0.6).split(',')[1]);
    }, 1000 / 15);   // 15 fps to server

    document.getElementById('startBtn').textContent = 'Running';
  }

  async function resetBuffer() {
    await fetch('/reset', { method: 'POST' });
    document.getElementById('wc-status').textContent = 'Buffer reset';
    document.getElementById('wc-bar').style.width = '0%';
    document.getElementById('wc-probs').style.display = 'none';
  }

  // ────────────────────────────────────────
  //  UPLOAD TAB
  // ────────────────────────────────────────
  let selectedFile = null;

  function onFileSelected(file) {
    if (!file) return;
    selectedFile = file;
    document.getElementById('file-name').textContent = file.name + ' (' + (file.size / 1024 / 1024).toFixed(1) + ' MB)';
    document.getElementById('analyze-btn').disabled = false;
    // Reset previous results
    document.getElementById('up-status').textContent = '';
    document.getElementById('up-status').className = 'status';
    document.getElementById('up-probs').style.display = 'none';
    document.getElementById('seg-section').style.display = 'none';
    document.getElementById('up-bar-wrap').style.display = 'none';
  }

  function onDragOver(e)  { e.preventDefault(); document.getElementById('drop-zone').classList.add('dragover'); }
  function onDragLeave(e) { document.getElementById('drop-zone').classList.remove('dragover'); }
  function onDrop(e) {
    e.preventDefault();
    document.getElementById('drop-zone').classList.remove('dragover');
    if (e.dataTransfer.files[0]) onFileSelected(e.dataTransfer.files[0]);
  }

  function showResult(data) {
    // render a completed result dict into the upload UI
    const statusEl = document.getElementById('up-status');
    statusEl.textContent = data.label + ' (' + data.confidence + '%)';
    statusEl.className   = 'status ' + data.label;

    document.getElementById('up-bar-wrap').style.display = 'block';
    document.getElementById('up-bar').style.width = '100%';
    document.getElementById('up-log').textContent =
      data.n_segments + ' segments analysed | ' +
      data.total_frames + ' frames | face detected ' + data.face_pct + '% of frames';

    updateProbs('up', data.fake_prob);

    const tbody = document.getElementById('seg-tbody');
    tbody.innerHTML = '';
    (data.segments || []).forEach(s => {
      const tr = document.createElement('tr');
      tr.className = s.label === 'FAKE' ? 'fake-row' : s.label === 'UNCERTAIN' ? 'uncertain-row' : 'real-row';
      tr.innerHTML =
        '<td>' + s.segment + '</td>' +
        '<td>' + s.start_sec + 's - ' + s.end_sec + 's</td>' +
        '<td><span class="pill ' + s.label + '">' + s.label + '</span></td>' +
        '<td>' + s.confidence + '%</td>' +
        '<td>' + s.fake_prob + '</td>';
      tbody.appendChild(tr);
    });
    document.getElementById('seg-section').style.display = 'block';
  }

  async function pollJob(jobId) {
    // poll /jobs/{jobId} every 2s until done or error
    const MAX_POLLS = 180;   // 6 minutes max
    for (let i = 0; i < MAX_POLLS; i++) {
      await new Promise(r => setTimeout(r, 2000));
      const resp = await fetch('/jobs/' + jobId);
      if (!resp.ok) throw new Error('Poll failed: ' + resp.statusText);
      const job  = await resp.json();
      if (job.status === 'queued' || job.status === 'processing') {
        document.getElementById('up-status').textContent =
          job.status === 'queued' ? 'Queued — waiting for worker...' : 'Processing...';
      }
      if (job.status === 'done')  return job.result;
      if (job.status === 'error') throw new Error(job.error || 'Job failed');
    }
    throw new Error('Timed out waiting for result');
  }

  async function analyzeVideo() {
    if (!selectedFile) return;

    // Show spinner, disable button
    document.getElementById('spinner').style.display = 'block';
    document.getElementById('analyze-btn').disabled = true;
    document.getElementById('up-status').textContent = 'Uploading...';
    document.getElementById('up-status').className = 'status';
    document.getElementById('up-probs').style.display = 'none';
    document.getElementById('seg-section').style.display = 'none';
    document.getElementById('up-bar-wrap').style.display = 'none';

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      // submit — returns job_id immediately
      const resp = await fetch('/predict/video', { method: 'POST', body: formData });
      const data = await resp.json();
      if (!resp.ok) {
        document.getElementById('up-status').textContent = 'Error: ' + (data.detail || resp.statusText);
        return;
      }

      // poll until done then render result
      document.getElementById('up-status').textContent = 'Queued — waiting for worker...';
      const result = await pollJob(data.job_id);
      showResult(result);

    } catch(e) {
      document.getElementById('up-status').textContent = 'Failed: ' + e.message;
    } finally {
      document.getElementById('spinner').style.display = 'none';
      document.getElementById('analyze-btn').disabled = false;
    }
  }

  // Populate camera list on page load
  window.addEventListener('load', refreshCameras);
</script>
</body>
</html>
"""