import os
import argparse
import numpy as np
import cv2
import mediapipe as mp
from scipy.signal import butter, filtfilt, welch
from tqdm import tqdm

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# config
FS      = 30
LOW_HZ  = 0.7
HIGH_HZ = 14.0

# 32 mid-face landmark indices for sub-region PPG extraction (Section 4.4.1)
# These cover forehead, nose bridge, cheeks and chin — all skin-rich areas
MID_FACE_32 = [
    1, 4, 5, 6, 8, 9, 10, 151,           # nose & forehead
    195, 197, 168, 107, 66, 105, 63, 70,  # upper mid-face
    336, 296, 334, 293, 300, 417,          # lower mid-face
    351, 399, 175, 152, 377, 400,          # chin area
    378, 379, 365, 397                     # lower cheeks
]

# set up landmarker
def build_landmarker(model_path):
    """FaceLandmarker in VIDEO mode for sequential frame processing."""
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.FaceLandmarker.create_from_options(options)


def sample_landmark_patch(frame_bgr, landmarks, lm_idx, h, w, patch_px=8):
    """
    Extract mean R, G, B from a small square patch centred on one landmark.
    Returns (R, G, B) or None if out of bounds.
    """
    lm  = landmarks[lm_idx]
    cx  = int(lm.x * w)
    cy  = int(lm.y * h)
    x0, x1 = max(0, cx - patch_px), min(w, cx + patch_px)
    y0, y1 = max(0, cy - patch_px), min(h, cy + patch_px)
    if x1 <= x0 or y1 <= y0:
        return None
    patch = frame_bgr[y0:y1, x0:x1].astype(np.float64)
    R = patch[:, :, 2].mean()
    G = patch[:, :, 1].mean()
    B = patch[:, :, 0].mean()
    return R, G, B

# helper funcs for signals
def butterworth_filter(signal, fs=FS, low=LOW_HZ, high=HIGH_HZ):
    nyq   = fs / 2.0
    low_n = low / nyq
    high_n= min(high / nyq, 0.999)
    b, a  = butter(4, [low_n, high_n], btype='band')
    if len(signal) < 15:
        return signal.copy()
    return filtfilt(b, a, signal)


def norm_0_255(arr):
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-9:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mn) / (mx - mn) * 255.0).astype(np.float32)


def psd_binned(signal, n_bins, fs=FS):
    """Compute PSD and resample to exactly n_bins values."""
    nperseg = min(len(signal), 128)
    _, p    = welch(signal, fs=fs, nperseg=nperseg)
    x_old   = np.linspace(0, 1, len(p))
    x_new   = np.linspace(0, 1, n_bins)
    return np.interp(x_new, x_old, p).astype(np.float32)


def chrom_ppg_segment(R_arr, G_arr, B_arr):
    """
    Chrominance-PPG with segment-level alpha (de Haan & Jeanne 2013).
    All inputs are 1-D arrays of per-frame means.
    """
    total = R_arr + G_arr + B_arr + 1e-9
    Rn = R_arr / total;  Gn = G_arr / total;  Bn = B_arr / total
    Xs = 3*Rn - 2*Gn
    Ys = 1.5*Rn + Gn - 1.5*Bn
    alpha = np.std(Xs) / (np.std(Ys) + 1e-9)
    return Xs - alpha * Ys

def extract_ppg_maps(video_path, omega, n_subregions, model_path):
    """
    Extract spectral PPG maps of shape (omega, 64) from a video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0 or video_fps > 240:
        video_fps = FS

    lm_indices = MID_FACE_32[:n_subregions]

    # Per-landmark frame buffers: R, G, B means
    R_bufs = [[] for _ in range(n_subregions)]
    G_bufs = [[] for _ in range(n_subregions)]
    B_bufs = [[] for _ in range(n_subregions)]

    frame_idx = 0

    with build_landmarker(model_path) as landmarker:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            h, w = frame_bgr.shape[:2]
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int(frame_idx * 1000.0 / video_fps)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            frame_idx += 1

            if not result.face_landmarks:
                # No face: append zeros to keep buffers aligned
                for i in range(n_subregions):
                    R_bufs[i].append(0.0)
                    G_bufs[i].append(0.0)
                    B_bufs[i].append(0.0)
                continue

            landmarks = result.face_landmarks[0]

            for i, lm_idx in enumerate(lm_indices):
                rgb_vals = sample_landmark_patch(frame_bgr, landmarks, lm_idx, h, w)
                if rgb_vals is None:
                    R_bufs[i].append(0.0);  G_bufs[i].append(0.0);  B_bufs[i].append(0.0)
                else:
                    R, G, B = rgb_vals
                    R_bufs[i].append(R);  G_bufs[i].append(G);  B_bufs[i].append(B)

    cap.release()

    n_frames = min(len(R_bufs[i]) for i in range(n_subregions))
    if n_frames < omega:
        return []

    maps = []
    step = omega // 2

    for start in range(0, n_frames - omega + 1, step):
        end = start + omega
        ppg_cols = np.zeros((omega, n_subregions), dtype=np.float32)
        psd_cols = np.zeros((omega, n_subregions), dtype=np.float32)

        for i in range(n_subregions):
            R_arr = np.array(R_bufs[i][start:end], dtype=np.float64)
            G_arr = np.array(G_bufs[i][start:end], dtype=np.float64)
            B_arr = np.array(B_bufs[i][start:end], dtype=np.float64)

            # Chrominance PPG with segment-level alpha
            sig = chrom_ppg_segment(R_arr, G_arr, B_arr)
            sig = butterworth_filter(sig)

            # Temporal column
            ppg_cols[:, i] = norm_0_255(sig)

            # PSD column (resampled to omega bins)
            bins = psd_binned(sig, n_bins=omega)
            psd_cols[:, i] = norm_0_255(bins)

        # Concatenate → (omega, 2*n_subregions) = (omega, 64)
        spectral_map = np.concatenate([ppg_cols, psd_cols], axis=1)
        maps.append(spectral_map)

    return maps

# build dataset
def process_directory(video_dir, label, omega, n_subregions, model_path, limit=None):
    X, y, names = [], [], []
    files = sorted(
        f for f in os.listdir(video_dir)
        if f.lower().endswith(('.mp4', '.avi', '.mov'))
    )
    if limit is not None:
        files = files[:limit]
    tag = "Real" if label == 0 else "Fake"
    for fname in tqdm(files, desc=f"{tag} ({os.path.basename(video_dir)}) [{len(files)} videos]"):
        path = os.path.join(video_dir, fname)
        try:
            maps = extract_ppg_maps(path, omega, n_subregions, model_path)
            for m in maps:
                X.append(m)
                y.append(label)
                names.append(fname)
        except Exception as e:
            print(f"  [ERROR] {fname}: {e}")
    return X, y, names

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--real_dir',     required=True)
    ap.add_argument('--fake_dir',     required=True)
    ap.add_argument('--output',       default='data/ppg_maps.npz')
    ap.add_argument('--omega',        type=int, default=128)
    ap.add_argument('--n_subregions', type=int, default=32)
    ap.add_argument('--model',        default='data/face_landmarker.task',
                    help='Path to face_landmarker.task model file')
    ap.add_argument('--limit',        type=int, default=None,
                    help='Max videos per class, e.g. --limit 200 (default: all)')
    args = ap.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(
            f"Model not found: {args.model}\n"
            "Download from: https://storage.googleapis.com/mediapipe-models/"
            "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        )

    print(f"\n{'='*60}")
    print("  FakeCatcher — Spectral PPG Map Extraction")
    print(f"  model        : {args.model}")
    print(f"  omega        : {args.omega} frames (~{args.omega/FS:.1f}s @ {FS}fps)")
    print(f"  n_subregions : {args.n_subregions}  →  map: ({args.omega}, {args.n_subregions*2})")
    print(f"  real_dir     : {args.real_dir}")
    print(f"  fake_dir     : {args.fake_dir}")
    print(f"  output       : {args.output}")
    print(f"{'='*60}\n")

    Xr, yr, nr = process_directory(args.real_dir, 0, args.omega, args.n_subregions, args.model, limit=args.limit)
    Xf, yf, nf = process_directory(args.fake_dir, 1, args.omega, args.n_subregions, args.model, limit=args.limit)

    # Stack → (N, omega, 64, 1), normalise to [0,1]
    X     = np.array(Xr + Xf, dtype=np.float32)[..., np.newaxis] / 255.0
    y     = np.array(yr + yf, dtype=np.int32)
    names = np.array(nr + nf)

    np.savez(args.output, X=X, y=y, video_names=names)

    print(f"\n{'='*60}")
    print(f"  Saved {len(X)} maps → {args.output}")
    print(f"  Map shape     : {X.shape}")
    print(f"  Real segments : {int((y==0).sum())}")
    print(f"  Fake segments : {int((y==1).sum())}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()