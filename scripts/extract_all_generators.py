"""
FakeCatcher - Multi-Generator Extraction
=========================================
Extracts SVM features + PPG maps from fake generator directories and
optionally merges with existing .npz files so you don't re-extract
real videos or already-processed generators.

-- TWO MODES --

1. FRESH — extract real + fakes from scratch:
    python extract_all_generators.py ^
        --real_dir     "C:/path/to/original" ^
        --fake_dirs    "C:/Face2Face" "C:/FaceSwap" ^
        --out_features data/features_svm_all.npz ^
        --out_maps     data/ppg_maps_all.npz ^
        --model        data/face_landmarker.task ^
        --limit        1000

2. MERGE — extract new fakes only, merge into existing .npz files:
    python extract_all_generators.py ^
        --fake_dirs         "C:/DeepfakeDetection" "C:/Face2Face" ^
                            "C:/FaceShifter" "C:/FaceSwap" "C:/NeuralTextures" ^
        --existing_features data/features_svm.npz ^
        --existing_maps     data/ppg_maps.npz ^
        --out_features      data/features_svm_all.npz ^
        --out_maps          data/ppg_maps_all.npz ^
        --model             data/face_landmarker.task ^
        --limit             1000

Note on --limit:
    Limits videos PER FAKE DIRECTORY.
    In FRESH mode, real videos are sampled at limit*num_generators to stay balanced.
    In MERGE mode, the existing real segments are reused as-is.
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


# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
FS           = 30
LOW_HZ       = 0.7
HIGH_HZ      = 14.0
N_SUBREGIONS = 32

LEFT_CHEEK_IDX  = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148]
MID_FACE_IDX    = [1, 4, 5, 195, 197, 6, 168, 8, 9, 151]
RIGHT_CHEEK_IDX = [454, 323, 361, 288, 397, 365, 379, 378, 400, 377]

MID_FACE_32 = [
    1, 4, 5, 6, 8, 9, 10, 151,
    195, 197, 168, 107, 66, 105, 63, 70,
    336, 296, 334, 293, 300, 417,
    351, 399, 175, 152, 377, 400,
    378, 379, 365, 397
]


# ─────────────────────────────────────────────
#  MEDIAPIPE
# ─────────────────────────────────────────────

def build_landmarker(model_path):
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


# ─────────────────────────────────────────────
#  PIXEL HELPERS
# ─────────────────────────────────────────────

def get_landmark_pixels(frame_bgr, landmarks, indices):
    h, w = frame_bgr.shape[:2]
    pts  = np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)]
                     for i in indices], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    pixels = frame_bgr[mask > 0]
    return pixels.astype(np.float64) if len(pixels) > 0 else None


def sample_patch(frame_bgr, landmarks, lm_idx, h, w, patch_px=8):
    lm = landmarks[lm_idx]
    cx, cy = int(lm.x * w), int(lm.y * h)
    x0, x1 = max(0, cx - patch_px), min(w, cx + patch_px)
    y0, y1 = max(0, cy - patch_px), min(h, cy + patch_px)
    if x1 <= x0 or y1 <= y0:
        return None
    patch = frame_bgr[y0:y1, x0:x1].astype(np.float64)
    return patch[:, :, 2].mean(), patch[:, :, 1].mean(), patch[:, :, 0].mean()


# ─────────────────────────────────────────────
#  SIGNAL HELPERS
# ─────────────────────────────────────────────

def butterworth_filter(signal):
    nyq  = FS / 2.0
    b, a = butter(4, [LOW_HZ / nyq, min(HIGH_HZ / nyq, 0.999)], btype='band')
    return filtfilt(b, a, signal) if len(signal) >= 15 else signal.copy()


def chrom_ppg_segment(R_arr, G_arr, B_arr):
    total = R_arr + G_arr + B_arr + 1e-9
    Rn, Gn, Bn = R_arr/total, G_arr/total, B_arr/total
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


# ─────────────────────────────────────────────
#  SINGLE-PASS VIDEO EXTRACTION
# ─────────────────────────────────────────────

def extract_video(video_path, omega, model_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], []

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0 or video_fps > 240:
        video_fps = FS

    lm_map_indices = MID_FACE_32[:N_SUBREGIONS]
    svm_raw = {k: [] for k in ['RL','RM','RR','GL','GM','GR','BL','BM','BR']}
    map_R = [[] for _ in range(N_SUBREGIONS)]
    map_G = [[] for _ in range(N_SUBREGIONS)]
    map_B = [[] for _ in range(N_SUBREGIONS)]
    frame_idx = 0

    with build_landmarker(model_path) as landmarker:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            h, w = frame_bgr.shape[:2]
            mp_image     = mp.Image(image_format=mp.ImageFormat.SRGB,
                                    data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            timestamp_ms = int(frame_idx * 1000.0 / video_fps)
            result       = landmarker.detect_for_video(mp_image, timestamp_ms)
            frame_idx   += 1

            if not result.face_landmarks:
                for i in range(N_SUBREGIONS):
                    map_R[i].append(0.0); map_G[i].append(0.0); map_B[i].append(0.0)
                continue

            landmarks = result.face_landmarks[0]

            for tag, indices in [('L', LEFT_CHEEK_IDX),
                                  ('M', MID_FACE_IDX),
                                  ('R', RIGHT_CHEEK_IDX)]:
                pixels = get_landmark_pixels(frame_bgr, landmarks, indices)
                if pixels is not None:
                    svm_raw['R'+tag].append(np.mean(pixels[:, 2]))
                    svm_raw['G'+tag].append(np.mean(pixels[:, 1]))
                    svm_raw['B'+tag].append(np.mean(pixels[:, 0]))

            for i, lm_idx in enumerate(lm_map_indices):
                rgb = sample_patch(frame_bgr, landmarks, lm_idx, h, w)
                if rgb is None:
                    map_R[i].append(0.0); map_G[i].append(0.0); map_B[i].append(0.0)
                else:
                    map_R[i].append(rgb[0]); map_G[i].append(rgb[1]); map_B[i].append(rgb[2])

    cap.release()

    step = omega // 2
    svm_segments, map_segments = [], []

    # SVM segments
    svm_lengths = [len(svm_raw[k]) for k in svm_raw]
    if svm_lengths and min(svm_lengths) >= omega:
        n_svm = min(svm_lengths)
        for start in range(0, n_svm - omega + 1, step):
            end = start + omega
            seg = {}
            for tag in ('L', 'M', 'R'):
                arr = np.array(svm_raw['G'+tag][start:end], dtype=np.float64)
                seg['G'+tag] = butterworth_filter(arr).astype(np.float32)
            for tag, c_key in [('L','CL'), ('M','CM'), ('R','CR')]:
                R_arr = np.array(svm_raw['R'+tag][start:end], dtype=np.float64)
                G_arr = np.array(svm_raw['G'+tag][start:end], dtype=np.float64)
                B_arr = np.array(svm_raw['B'+tag][start:end], dtype=np.float64)
                seg[c_key] = butterworth_filter(
                    chrom_ppg_segment(R_arr, G_arr, B_arr)).astype(np.float32)
            svm_segments.append(seg)

    # PPG map segments
    map_n = min(len(map_R[i]) for i in range(N_SUBREGIONS))
    if map_n >= omega:
        for start in range(0, map_n - omega + 1, step):
            end = start + omega
            ppg_cols = np.zeros((omega, N_SUBREGIONS), dtype=np.float32)
            psd_cols = np.zeros((omega, N_SUBREGIONS), dtype=np.float32)
            for i in range(N_SUBREGIONS):
                R_arr = np.array(map_R[i][start:end], dtype=np.float64)
                G_arr = np.array(map_G[i][start:end], dtype=np.float64)
                B_arr = np.array(map_B[i][start:end], dtype=np.float64)
                sig = butterworth_filter(chrom_ppg_segment(R_arr, G_arr, B_arr))
                ppg_cols[:, i] = norm_0_255(sig)
                psd_cols[:, i] = norm_0_255(psd_binned(sig, n_bins=omega))
            map_segments.append(np.concatenate([ppg_cols, psd_cols], axis=1))

    return svm_segments, map_segments


# ─────────────────────────────────────────────
#  SVM FEATURE VECTOR
# ─────────────────────────────────────────────

def _psd(sig):
    _, p = welch(sig.astype(np.float64), fs=FS, nperseg=min(len(sig), 256))
    return p

def _spec_ac(sig):
    p = _psd(sig)
    ac = np.correlate(p, p, mode='full')
    return ac[len(p)-1:]

def F1(a, b):
    cpsd = _psd(a) * _psd(b)
    return [float(np.mean(cpsd)), float(np.max(cpsd))]

def F3(sig):
    ac  = _spec_ac(sig)
    thr = np.mean(ac) + np.std(ac)
    nb  = float(np.sum(ac > thr))
    return [nb,
            float(np.sum(np.diff(np.sign(ac - np.mean(ac))) != 0)),
            float(np.mean(ac[ac > thr])) if nb > 0 else 0.0,
            float(np.max(ac))]

def F4(sig):
    win  = max(1, FS)
    wins = [sig[i:i+win] for i in range(0, len(sig)-win+1, win)]
    diffs = np.diff(sig)
    ac    = np.correlate(sig-sig.mean(), sig-sig.mean(), mode='full')
    hist, _ = np.histogram(sig, bins=50, density=True)
    return [float(np.std(sig)),
            float(np.std([np.mean(w) for w in wins])) if wins else 0.0,
            float(np.sqrt(np.mean(diffs**2))) if len(diffs) else 0.0,
            float(np.mean([np.std(np.diff(w)) for w in wins])) if wins else 0.0,
            float(np.std(diffs)) if len(diffs) else 0.0,
            float(np.mean(ac[len(sig)-1:])),
            float(scipy_entropy(hist + 1e-9))]

def build_feature_vector(seg):
    GL=seg['GL']; GM=seg['GM']; GR=seg['GR']
    CL=seg['CL']; CM=seg['CM']; CR=seg['CR']
    S  = [GL, GM, GR, CL, CM, CR]
    DC = [np.abs(CL-CM), np.abs(CL-CR), np.abs(CR-CM)]
    feats = []
    log_DC = [np.log(np.abs(d)+1e-9) for d in DC]
    for i in range(len(log_DC)):
        for j in range(i+1, len(log_DC)):
            feats.extend(F1(log_DC[i], log_DC[j]))
    for s in [np.log(np.abs(x)+1e-9) for x in S]:
        feats.extend(F3(s))
    for i in range(len(DC)):
        for j in range(i+1, len(DC)):
            feats.extend(F3(_psd(DC[i])*_psd(DC[j])))
    for s in [np.log(np.abs(x)+1e-9) for x in S]:
        feats.extend(F4(s))
    for i in range(len(DC)):
        for j in range(i+1, len(DC)):
            feats.extend(F4(_psd(DC[i])*_psd(DC[j])))
    for s in S:
        ac = _spec_ac(s)
        feats.extend([float(np.mean(ac)), float(np.max(ac))])
    return np.array(feats, dtype=np.float32)


# ─────────────────────────────────────────────
#  DIRECTORY PROCESSOR
# ─────────────────────────────────────────────

def process_directory(video_dir, label, omega, model_path, limit, tag):
    files = sorted(f for f in os.listdir(video_dir)
                   if f.lower().endswith(('.mp4', '.avi', '.mov')))
    if limit is not None:
        files = files[:limit]

    svm_X, svm_y, svm_names = [], [], []
    map_X, map_y, map_names = [], [], []

    for fname in tqdm(files, desc=f"{tag} [{len(files)} videos]"):
        path        = os.path.join(video_dir, fname)
        unique_name = f"{tag}/{fname}"   # prefix keeps names unique across generators
        try:
            svm_segs, map_segs = extract_video(path, omega, model_path)
            for seg in svm_segs:
                fv = build_feature_vector(seg)
                if np.isfinite(fv).all():
                    svm_X.append(fv); svm_y.append(label); svm_names.append(unique_name)
            for m in map_segs:
                map_X.append(m); map_y.append(label); map_names.append(unique_name)
        except Exception as e:
            print(f"  [ERROR] {fname}: {e}")

    return (svm_X, svm_y, svm_names), (map_X, map_y, map_names)


# ─────────────────────────────────────────────
#  NPZ MERGE HELPER
# ─────────────────────────────────────────────

def load_existing_npz(path, is_maps=False):
    """
    Load an existing .npz produced by extract_all.py or this script.
    Returns (X_list, y_list, names_list) ready to concatenate.
    Maps are returned already normalised to [0,1] with channel dim.
    """
    print(f"  Loading: {path}")
    data  = np.load(path, allow_pickle=True)
    X     = data['X'].astype(np.float32)
    y     = data['y'].astype(np.int32)
    names = list(data['video_names'])

    if is_maps and X.max() > 1.5:
        X = X / 255.0   # normalise old files saved as 0-255

    print(f"    {len(X)} segments  "
          f"(real: {int((y==0).sum())}  fake: {int((y==1).sum())})")
    return list(X), list(y), names


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="FakeCatcher — extract from multiple fake generators, "
                    "with optional merge into existing .npz files"
    )

    # ── Input sources
    ap.add_argument('--real_dir',          default=None,
                    help='[FRESH mode] Folder of real (original) videos. '
                         'Not needed in MERGE mode.')
    ap.add_argument('--fake_dirs',         required=True, nargs='+',
                    help='One or more fake generator folders e.g. '
                         '"C:/DeepfakeDetection" "C:/Face2Face" "C:/FaceShifter" '
                         '"C:/FaceSwap" "C:/NeuralTextures"')

    # ── Existing files to merge with (MERGE mode)
    ap.add_argument('--existing_features', default=None,
                    help='[MERGE mode] Existing features_svm.npz to merge new fakes into')
    ap.add_argument('--existing_maps',     default=None,
                    help='[MERGE mode] Existing ppg_maps.npz to merge new fakes into')

    # ── Outputs
    ap.add_argument('--out_features',      default='data/features_svm_all.npz')
    ap.add_argument('--out_maps',          default='data/ppg_maps_all.npz')

    # ── Extraction settings
    ap.add_argument('--omega',             type=int, default=128)
    ap.add_argument('--model',             default='data/face_landmarker.task')
    ap.add_argument('--limit',             type=int, default=None,
                    help='Max videos PER FAKE DIRECTORY (default: all). '
                         'In FRESH mode real videos scale to limit*num_generators.')
    args = ap.parse_args()

    # ── Validate mode
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")

    merge_mode = args.existing_features is not None or args.existing_maps is not None
    fresh_mode = args.real_dir is not None

    if not merge_mode and not fresh_mode:
        ap.error("Provide either --real_dir (fresh) "
                 "or --existing_features + --existing_maps (merge)")
    if merge_mode and (args.existing_features is None or args.existing_maps is None):
        ap.error("Merge mode requires BOTH --existing_features and --existing_maps")

    os.makedirs(os.path.dirname(args.out_features) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(args.out_maps)     or '.', exist_ok=True)

    generator_tags = [os.path.basename(d.rstrip('/\\')) for d in args.fake_dirs]

    print(f"\n{'='*60}")
    print(f"  FakeCatcher — Multi-Generator Extraction")
    print(f"  Mode         : {'MERGE into existing' if merge_mode else 'FRESH'}")
    print(f"  Generators   : {', '.join(generator_tags)}")
    print(f"  Limit/gen    : {args.limit or 'all'}")
    print(f"  omega        : {args.omega}")
    print(f"{'='*60}\n")

    # ── Seed accumulator from existing files (MERGE) or empty (FRESH)
    if merge_mode:
        print("Loading existing .npz files...")
        all_svm_X, all_svm_y, all_svm_n = load_existing_npz(
            args.existing_features, is_maps=False)
        all_map_X, all_map_y, all_map_n = load_existing_npz(
            args.existing_maps, is_maps=True)
        print()
    else:
        all_svm_X, all_svm_y, all_svm_n = [], [], []
        all_map_X, all_map_y, all_map_n = [], [], []

    # ── Real videos — FRESH mode only
    if fresh_mode:
        real_limit = (args.limit * len(args.fake_dirs)) if args.limit else None
        print(f"Extracting real videos (limit: {real_limit or 'all'})...")
        (sX, sy, sn), (mX, my, mn) = process_directory(
            args.real_dir, label=0,
            omega=args.omega, model_path=args.model,
            limit=real_limit, tag='real'
        )
        all_svm_X += sX; all_svm_y += sy; all_svm_n += sn
        all_map_X += mX; all_map_y += my; all_map_n += mn

    # ── Each fake generator
    print(f"Extracting {len(args.fake_dirs)} fake generator(s)...")
    for fake_dir, tag in zip(args.fake_dirs, generator_tags):
        (sX, sy, sn), (mX, my, mn) = process_directory(
            fake_dir, label=1,
            omega=args.omega, model_path=args.model,
            limit=args.limit, tag=tag
        )
        all_svm_X += sX; all_svm_y += sy; all_svm_n += sn
        all_map_X += mX; all_map_y += my; all_map_n += mn

    # ── Save SVM features
    svm_X = np.array(all_svm_X, dtype=np.float32)
    svm_y = np.array(all_svm_y, dtype=np.int32)
    svm_n = np.array(all_svm_n)
    np.savez(args.out_features, X=svm_X, y=svm_y, video_names=svm_n)

    # ── Save PPG maps — ensure (N, omega, 64, 1) normalised [0,1]
    map_arr = np.array(all_map_X, dtype=np.float32)
    if map_arr.ndim == 3:
        map_arr = map_arr[..., np.newaxis]   # add channel dim if missing
    if map_arr.max() > 1.5:
        map_arr = map_arr / 255.0            # normalise if not already done
    map_y = np.array(all_map_y, dtype=np.int32)
    map_n = np.array(all_map_n)
    np.savez(args.out_maps, X=map_arr, y=map_y, video_names=map_n)

    # ── Summary
    print(f"\n{'='*60}")
    print("  Done.")
    print(f"\n  SVM  → {args.out_features}")
    print(f"    Real: {int((svm_y==0).sum())}  "
          f"Fake: {int((svm_y==1).sum())}  "
          f"Total: {len(svm_X)}")
    print(f"\n  Maps → {args.out_maps}")
    print(f"    Real: {int((map_y==0).sum())}  "
          f"Fake: {int((map_y==1).sum())}  "
          f"Total: {len(map_arr)}")
    if len(map_arr):
        print(f"    Shape: {map_arr.shape}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()