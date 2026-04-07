import os
import cv2
import logging
import numpy as np
import mediapipe as mp
from collections import deque
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from scipy.signal import butter, filtfilt, welch

logger = logging.getLogger("deeptrack.video")

CNN_MODEL_PATH  = "data/cnn_model.keras"
LANDMARKER_PATH = "data/face_landmarker.task"
HF_REPO_ID      = "dkkinyua/fakecatcher"

OMEGA               = 128
N_SUBREGIONS        = 32
FS                  = 30
LOW_HZ              = 0.7
HIGH_HZ             = 14.0
THRESHOLD_FAKE      = 0.65
THRESHOLD_UNCERTAIN = 0.45
SMOOTHING_WINDOW    = 3

MID_FACE_32 = [
    1, 4, 5, 6, 8, 9, 10, 151,
    195, 197, 168, 107, 66, 105, 63, 70,
    336, 296, 334, 293, 300, 417,
    351, 399, 175, 152, 377, 400,
    378, 379, 365, 397,
]


def _resolve_path(local_path: str, hf_filename: str) -> str:
    if os.path.exists(local_path):
        logger.info(f"Using local file: {local_path}")
        return local_path
    logger.info(f"{local_path} not found — downloading {hf_filename} from HuggingFace...")
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=hf_filename,
        token=os.getenv("HUGGINGFACE_TOKEN"),
    )
    logger.info(f"Downloaded: {path}")
    return path


# ── Signal helpers ────────────────────────────────────────────────────────────

def butterworth_filter(signal):
    nyq  = FS / 2.0
    b, a = butter(4, [LOW_HZ / nyq, min(HIGH_HZ / nyq, 0.999)], btype="band")
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
    ppg_cols = np.zeros((OMEGA, N_SUBREGIONS), dtype=np.float32)
    psd_cols = np.zeros((OMEGA, N_SUBREGIONS), dtype=np.float32)
    for i in range(N_SUBREGIONS):
        R_arr = np.array(R_bufs[i], dtype=np.float64)
        G_arr = np.array(G_bufs[i], dtype=np.float64)
        B_arr = np.array(B_bufs[i], dtype=np.float64)
        sig = butterworth_filter(chrom_ppg_segment(R_arr, G_arr, B_arr))
        ppg_cols[:, i] = norm_0_255(sig)
        psd_cols[:, i] = norm_0_255(psd_binned(sig, n_bins=OMEGA))
    spectral_map = np.concatenate([ppg_cols, psd_cols], axis=1)
    return (spectral_map[np.newaxis, ..., np.newaxis] / 255.0).astype(np.float32)

def signal_quality(R_bufs, G_bufs, B_bufs):
    snrs, zero_counts = [], []
    for i in range(min(8, N_SUBREGIONS)):
        R_arr    = np.array(R_bufs[i], dtype=np.float64)
        zero_pct = (R_arr == 0).mean()
        zero_counts.append(zero_pct)
        if zero_pct < 0.6:
            sig = butterworth_filter(chrom_ppg_segment(
                R_arr,
                np.array(G_bufs[i], dtype=np.float64),
                np.array(B_bufs[i], dtype=np.float64),
            ))
            snr = np.mean(sig ** 2) / (np.std(np.diff(sig)) ** 2 + 1e-9)
            snrs.append(snr)
    avg_zero = np.mean(zero_counts)
    avg_snr  = np.mean(snrs) if snrs else 0.0
    if avg_zero > 0.40:
        return False, f"Face not detected in {avg_zero*100:.0f}% of frames", avg_snr
    if avg_snr < 0.008:
        return False, f"Signal quality too low (SNR={avg_snr:.4f})", avg_snr
    return True, "ok", avg_snr

def classify_prob(prob: float) -> tuple:
    if prob >= THRESHOLD_FAKE:
        return "FAKE", round(prob * 100, 1)
    elif prob >= THRESHOLD_UNCERTAIN:
        return "UNCERTAIN", round((1.0 - abs(prob - 0.5) * 2) * 100, 1)
    else:
        return "REAL", round((1.0 - prob) * 100, 1)

def sample_patch(frame_bgr, landmarks, lm_idx, h, w, patch_px=8):
    lm = landmarks[lm_idx]
    cx, cy = int(lm.x * w), int(lm.y * h)
    x0, x1 = max(0, cx - patch_px), min(w, cx + patch_px)
    y0, y1 = max(0, cy - patch_px), min(h, cy + patch_px)
    if x1 <= x0 or y1 <= y0:
        return None
    patch = frame_bgr[y0:y1, x0:x1].astype(np.float64)
    return patch[:, :, 2].mean(), patch[:, :, 1].mean(), patch[:, :, 0].mean()


# ── FrameBuffer ───────────────────────────────────────────────────────────────

class FrameBuffer:
    def __init__(self):
        self.R = [deque(maxlen=OMEGA) for _ in range(N_SUBREGIONS)]
        self.G = [deque(maxlen=OMEGA) for _ in range(N_SUBREGIONS)]
        self.B = [deque(maxlen=OMEGA) for _ in range(N_SUBREGIONS)]
        self.frame_count  = 0
        self.last_result  = None
        self.step         = OMEGA // 2
        self.prob_history = deque(maxlen=SMOOTHING_WINDOW)

    def push(self, rgb_values: list):
        for i, (r, g, b) in enumerate(rgb_values):
            self.R[i].append(r); self.G[i].append(g); self.B[i].append(b)
        self.frame_count += 1

    def ready(self) -> bool:
        return len(self.R[0]) == OMEGA and self.frame_count % self.step == 0

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


# ── VideoPredictor ────────────────────────────────────────────────────────────

class VideoPredictor:
    def __init__(self):
        cnn_path        = _resolve_path(CNN_MODEL_PATH,  "cnn_model.keras")
        landmarker_path = _resolve_path(LANDMARKER_PATH, "face_landmarker.task")

        logger.info("Loading CNN model...")
        import tensorflow as tf
        from tensorflow import keras
        self.model = keras.models.load_model(cnn_path)
        logger.info("CNN model loaded.")

        logger.info("Loading MediaPipe FaceLandmarker (IMAGE mode)...")
        base_opts  = mp_python.BaseOptions(model_asset_path=landmarker_path)
        image_opts = mp_vision.FaceLandmarkerOptions(
            base_options=base_opts,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.4,
            min_face_presence_confidence=0.4,
        )
        self.landmarker      = mp_vision.FaceLandmarker.create_from_options(image_opts)
        self.lm_indices      = MID_FACE_32[:N_SUBREGIONS]
        self.landmarker_path = landmarker_path
        logger.info("FaceLandmarker loaded.")

    def extract_frame_rgb(self, frame_bgr):
        h, w      = frame_bgr.shape[:2]
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
        """Single-map predict — used by webcam/frame endpoints."""
        if not quality_ok:
            return {"label": "UNCERTAIN", "confidence": 0.0,
                    "fake_prob": None, "warning": quality_reason}
        prob        = float(self.model.predict(ppg_map, verbose=0)[0][0])
        label, conf = classify_prob(prob)
        return {"label": label, "confidence": conf,
                "fake_prob": round(prob, 4), "snr": round(snr, 4)}

    def _batch_predict(self, ppg_maps: list) -> list:
        """
        One model.predict() call for all segments instead of one per segment.
        Reduces inference calls from n_segments down to 1.
        Signal integrity is completely unchanged — same frames, same overlap.
        Returns list of (prob, label, confidence) tuples.
        """
        batch = np.concatenate(ppg_maps, axis=0)          # (n_segs, 128, 64, 1)
        preds = self.model.predict(batch, verbose=0)[:, 0] # (n_segs,)
        results = []
        for prob in preds:
            prob = round(float(prob), 4)
            label, conf = classify_prob(prob)
            results.append((prob, label, conf))
        return results

    def predict_video(self, video_path: str) -> dict:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0 or video_fps > 240:
            video_fps = FS

        # ── Phase 1: full frame extraction — every frame, no skipping ─────────
        map_R = [[] for _ in range(N_SUBREGIONS)]
        map_G = [[] for _ in range(N_SUBREGIONS)]
        map_B = [[] for _ in range(N_SUBREGIONS)]
        frame_idx = face_frames = 0

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
                f"Video too short: {total_frames} frames, need ≥{OMEGA} (~{OMEGA//FS}s at 30fps)")
        face_pct = round(face_frames / total_frames * 100, 1) if total_frames else 0
        if face_pct < 20:
            raise ValueError(
                f"Face detected in only {face_pct}% of frames — ensure face is visible")

        # ── Phase 2: build all PPG maps — 50% overlap preserved ───────────────
        step     = OMEGA // 2
        ppg_maps = []
        seg_meta = []
        for start in range(0, total_frames - OMEGA + 1, step):
            end = start + OMEGA
            ppg_maps.append(build_ppg_map(
                [map_R[i][start:end] for i in range(N_SUBREGIONS)],
                [map_G[i][start:end] for i in range(N_SUBREGIONS)],
                [map_B[i][start:end] for i in range(N_SUBREGIONS)],
            ))
            seg_meta.append((start, end))

        # ── Phase 3: single batch predict for all segments ────────────────────
        batch_results = self._batch_predict(ppg_maps)

        segments = []
        probs    = []
        for idx, ((start, end), (prob, label, conf)) in enumerate(zip(seg_meta, batch_results)):
            probs.append(prob)
            segments.append({
                "segment":    idx + 1,
                "start_sec":  round(start / FS, 1),
                "end_sec":    round(end   / FS, 1),
                "label":      label,
                "confidence": conf,
                "fake_prob":  prob,
            })

        avg_prob          = float(np.mean(probs))
        label, confidence = classify_prob(avg_prob)

        return {
            "label":        label,
            "confidence":   confidence,
            "fake_prob":    round(avg_prob, 4),
            "total_frames": total_frames,
            "face_pct":     face_pct,
            "n_segments":   len(segments),
            "segments":     segments,
        }

    def close(self):
        self.landmarker.close()