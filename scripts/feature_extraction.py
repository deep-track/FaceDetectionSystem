"""
FakeCatcher - Signal Feature Extraction for SVM
================================================
Extracts 126-dimensional biological signal features from portrait videos.
Based on: "FakeCatcher: Detection of Synthetic Portrait Videos using Biological Signals"

Uses the new MediaPipe Tasks API (mediapipe >= 0.10.x) with FaceLandmarker.
Requires: data/face_landmarker.task  (download from Google once, reuse forever)

Download the model if you haven't already:
    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

Usage:
    python 1_signal_feature_extraction.py \
        --real_dir "C:/path/to/original" \
        --fake_dir "C:/path/to/Deepfakes" \
        --output   features_svm.npz \
        --omega    128 \
        --model    data/face_landmarker.task
"""

import os
import argparse
import numpy as np
import cv2
import mediapipe as mp
from scipy.signal import butter, filtfilt, welch
from scipy.stats import entropy as scipy_entropy
from tqdm import tqdm

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# config
FS      = 30     # assumed frame rate (Hz)
LOW_HZ  = 0.7    # Butterworth bandpass low (Hz)
HIGH_HZ = 14.0   # Butterworth bandpass high (Hz)

# MediaPipe landmark indices for three facial ROIs
# Left cheek, mid-face (nose bridge/forehead), right cheek
LEFT_CHEEK_IDX  = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148]
MID_FACE_IDX    = [1, 4, 5, 195, 197, 6, 168, 8, 9, 151]
RIGHT_CHEEK_IDX = [454, 323, 361, 288, 397, 365, 379, 378, 400, 377]

# face landmarker
def build_landmarker(model_path):
    """
    Create a FaceLandmarker in VIDEO mode.
    VIDEO mode processes frames sequentially with monotonically increasing
    timestamps — ideal for reading video files frame-by-frame.
    """
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


def get_landmark_pixels(frame_bgr, landmarks, indices):
    """
    Given a list of NormalizedLandmark objects and target indices,
    build a convex-hull mask and extract all pixel values inside.

    Returns pixel array of shape (N, 3) in BGR, or None.
    """
    h, w = frame_bgr.shape[:2]

    pts = np.array(
        [[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in indices],
        dtype=np.int32
    )

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)

    pixels = frame_bgr[mask > 0]
    if len(pixels) == 0:
        return None
    return pixels.astype(np.float64)

# helper funcs
def butterworth_filter(signal, fs=FS, low=LOW_HZ, high=HIGH_HZ):
    """Zero-phase Butterworth bandpass filter (Section 7 of paper)."""
    nyq   = fs / 2.0
    low_n = low / nyq
    high_n= min(high / nyq, 0.999)
    b, a  = butter(4, [low_n, high_n], btype='band')
    if len(signal) < 15:
        return signal.copy()
    return filtfilt(b, a, signal)


def chrom_ppg_segment(R_arr, G_arr, B_arr):
    """
    Chrominance-based PPG from arrays of per-frame R, G, B means.
    """
    total = R_arr + G_arr + B_arr + 1e-9
    Rn = R_arr / total
    Gn = G_arr / total
    Bn = B_arr / total
    Xs = 3*Rn - 2*Gn
    Ys = 1.5*Rn + Gn - 1.5*Bn
    alpha = np.std(Xs) / (np.std(Ys) + 1e-9)
    return Xs - alpha * Ys

# extract signals from rois
def extract_roi_signals(video_path, omega, model_path):
    """
    Open a video and extract 6 biological signals per frame:
      GL, GM, GR  — mean green channel from left/mid/right ROI
      CL, CM, CR  — chrominance-PPG from left/mid/right ROI

    Uses MediaPipe FaceLandmarker (VIDEO mode) for precise ROI landmark masking.
    Timestamps are derived from the actual video frame index and FPS.

    Returns list of segment dicts (keys: GL, GM, GR, CL, CM, CR).
    Each value is a float32 array of length omega.
    Returns [] if the video has fewer than omega usable frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    # Get actual FPS from video metadata (fall back to 30)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0 or video_fps > 240:
        video_fps = FS

    # Raw per-frame buffers for R, G, B means per ROI
    # We collect RGB separately so chrom_ppg can use segment-level alpha
    raw = {
        'GL': [], 'GM': [], 'GR': [],          # green channel (scalar/frame)
        'RL': [], 'RM': [], 'RR': [],           # R channel
        'BL': [], 'BM': [], 'BR': [],           # B channel
        'GL_g': [], 'GM_g': [], 'GR_g': [],     # G channel (for chrom)
    }

    frame_idx = 0

    with build_landmarker(model_path) as landmarker:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            # Convert BGR → RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            # Timestamp must be monotonically increasing integers (ms)
            timestamp_ms = int(frame_idx * 1000.0 / video_fps)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if not result.face_landmarks:
                frame_idx += 1
                continue   # no face detected this frame, skip

            landmarks = result.face_landmarks[0]   # first (only) face

            for tag, indices in [
                ('L', LEFT_CHEEK_IDX),
                ('M', MID_FACE_IDX),
                ('R', RIGHT_CHEEK_IDX),
            ]:
                pixels = get_landmark_pixels(frame_bgr, landmarks, indices)
                if pixels is None:
                    continue

                R = np.mean(pixels[:, 2])
                G = np.mean(pixels[:, 1])
                B = np.mean(pixels[:, 0])

                raw['G'  + tag].append(G)          # green-PPG value
                raw['R'  + tag].append(R)
                raw['GL_g' if tag == 'L' else ('GM_g' if tag == 'M' else 'GR_g')].append(G)
                raw['B'  + tag].append(B)

            frame_idx += 1

    cap.release()

    # ── Determine usable frame count (all 9 per-frame buffers must be same length)
    green_keys = ['GL', 'GM', 'GR']
    rgb_keys   = ['RL', 'RM', 'RR', 'BL', 'BM', 'BR', 'GL_g', 'GM_g', 'GR_g']
    all_keys   = green_keys + rgb_keys
    lengths    = [len(raw[k]) for k in all_keys]
    if not lengths or min(lengths) < omega:
        return []

    n_frames = min(lengths)
    step     = omega // 2   # 50% overlap
    segments = []

    for start in range(0, n_frames - omega + 1, step):
        end = start + omega
        seg = {}

        # Green-channel PPG (GL, GM, GR)
        for k in green_keys:
            arr    = np.array(raw[k][start:end], dtype=np.float64)
            seg[k] = butterworth_filter(arr).astype(np.float32)

        # Chrominance PPG (CL, CM, CR) — compute alpha at segment level
        for tag, r_key, g_key, b_key, c_key in [
            ('L', 'RL', 'GL_g', 'BL', 'CL'),
            ('M', 'RM', 'GM_g', 'BM', 'CM'),
            ('R', 'RR', 'GR_g', 'BR', 'CR'),
        ]:
            R_arr = np.array(raw[r_key][start:end], dtype=np.float64)
            G_arr = np.array(raw[g_key][start:end], dtype=np.float64)
            B_arr = np.array(raw[b_key][start:end], dtype=np.float64)
            c_sig = chrom_ppg_segment(R_arr, G_arr, B_arr)
            seg[c_key] = butterworth_filter(c_sig).astype(np.float32)

        segments.append(seg)

    return segments

# feature extraction from paper
def _psd(sig):
    _, p = welch(sig.astype(np.float64), fs=FS, nperseg=min(len(sig), 256))
    return p


def _spec_ac(sig):
    p  = _psd(sig)
    ac = np.correlate(p, p, mode='full')
    return ac[len(p) - 1:]


def F1_features(a, b):
    """Mean and max of cross power spectral density."""
    cpsd = _psd(a) * _psd(b)
    return [float(np.mean(cpsd)), float(np.max(cpsd))]


def F3_features(sig):
    """4 spectral-autocorrelation features."""
    ac  = _spec_ac(sig)
    thr = np.mean(ac) + np.std(ac)
    nb  = float(np.sum(ac > thr))
    return [
        nb,
        float(np.sum(np.diff(np.sign(ac - np.mean(ac))) != 0)),
        float(np.mean(ac[ac > thr])) if nb > 0 else 0.0,
        float(np.max(ac)),
    ]


def F4_features(sig):
    """7 statistical time-domain features."""
    win  = max(1, FS)
    wins = [sig[i:i+win] for i in range(0, len(sig) - win + 1, win)]
    diffs = np.diff(sig)
    ac    = np.correlate(sig - sig.mean(), sig - sig.mean(), mode='full')
    hist, _ = np.histogram(sig, bins=50, density=True)
    return [
        float(np.std(sig)),
        float(np.std([np.mean(w) for w in wins])) if wins else 0.0,
        float(np.sqrt(np.mean(diffs**2))) if len(diffs) else 0.0,
        float(np.mean([np.std(np.diff(w)) for w in wins])) if wins else 0.0,
        float(np.std(diffs)) if len(diffs) else 0.0,
        float(np.mean(ac[len(sig) - 1:])),
        float(scipy_entropy(hist + 1e-9)),
    ]


def build_feature_vector(seg):
    """
    Assemble 126-feature vector from one segment dict.
    """
    GL = seg['GL'];  GM = seg['GM'];  GR = seg['GR']
    CL = seg['CL'];  CM = seg['CM'];  CR = seg['CR']
    S  = [GL, GM, GR, CL, CM, CR]
    DC = [np.abs(CL - CM), np.abs(CL - CR), np.abs(CR - CM)]

    feats = []

    # F1(log DC)
    log_DC = [np.log(np.abs(d) + 1e-9) for d in DC]
    for i in range(len(log_DC)):
        for j in range(i + 1, len(log_DC)):
            feats.extend(F1_features(log_DC[i], log_DC[j]))

    # F3 on log(S)
    for s in [np.log(np.abs(x) + 1e-9) for x in S]:
        feats.extend(F3_features(s))

    # F3 on cross-PSD of DC pairs
    for i in range(len(DC)):
        for j in range(i + 1, len(DC)):
            feats.extend(F3_features(_psd(DC[i]) * _psd(DC[j])))

    # F4 on log(S)
    for s in [np.log(np.abs(x) + 1e-9) for x in S]:
        feats.extend(F4_features(s))

    # F4 on cross-PSD of DC pairs
    for i in range(len(DC)):
        for j in range(i + 1, len(DC)):
            feats.extend(F4_features(_psd(DC[i]) * _psd(DC[j])))

    for s in S:
        ac = _spec_ac(s)
        feats.extend([float(np.mean(ac)), float(np.max(ac))])

    return np.array(feats, dtype=np.float32)

# build dataset
def process_directory(video_dir, label, omega, model_path, limit=None):
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
            segs = extract_roi_signals(path, omega, model_path)
            for seg in segs:
                fv = build_feature_vector(seg)
                if np.isfinite(fv).all():
                    X.append(fv)
                    y.append(label)
                    names.append(fname)
        except Exception as e:
            print(f"  [ERROR] {fname}: {e}")
    return X, y, names

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--real_dir', required=True,  help='Folder of real videos')
    ap.add_argument('--fake_dir', required=True,  help='Folder of fake videos')
    ap.add_argument('--output',   default='features_svm.npz')
    ap.add_argument('--omega',    type=int, default=128,
                    help='Segment length in frames (default 128 ≈ 4.3s @ 30fps)')
    ap.add_argument('--model',    default='data/face_landmarker.task',
                    help='Path to face_landmarker.task model file')
    ap.add_argument('--limit',    type=int, default=None,
                    help='Max videos per class, e.g. --limit 200 (default: all)')
    args = ap.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(
            f"Model not found: {args.model}\n"
            "Download from: https://storage.googleapis.com/mediapipe-models/"
            "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        )

    print(f"\n{'='*60}")
    print("  FakeCatcher — SVM Feature Extraction")
    print(f"  model     : {args.model}")
    print(f"  omega     : {args.omega} frames (~{args.omega/FS:.1f}s @ {FS}fps)")
    print(f"  real_dir  : {args.real_dir}")
    print(f"  fake_dir  : {args.fake_dir}")
    print(f"  output    : {args.output}")
    print(f"{'='*60}\n")

    Xr, yr, nr = process_directory(args.real_dir, label=0, omega=args.omega, model_path=args.model, limit=args.limit)
    Xf, yf, nf = process_directory(args.fake_dir, label=1, omega=args.omega, model_path=args.model, limit=args.limit)

    X     = np.array(Xr + Xf, dtype=np.float32)
    y     = np.array(yr + yf, dtype=np.int32)
    names = np.array(nr + nf)

    np.savez(args.output, X=X, y=y, video_names=names)

    print(f"\n{'='*60}")
    print(f"  Saved {len(X)} segments → {args.output}")
    print(f"  Real segments : {int((y==0).sum())}")
    print(f"  Fake segments : {int((y==1).sum())}")
    if len(X):
        print(f"  Feature dims  : {X.shape[1]}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()