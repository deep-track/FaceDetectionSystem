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
FS      = 30
LOW_HZ  = 0.7
HIGH_HZ = 14.0

# Three ROI landmark groups for SVM signals
LEFT_CHEEK_IDX  = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148]
MID_FACE_IDX    = [1, 4, 5, 195, 197, 6, 168, 8, 9, 151]
RIGHT_CHEEK_IDX = [454, 323, 361, 288, 397, 365, 379, 378, 400, 377]

# 32 mid-face landmark indices for PPG map sub-regions
MID_FACE_32 = [
    1, 4, 5, 6, 8, 9, 10, 151,
    195, 197, 168, 107, 66, 105, 63, 70,
    336, 296, 334, 293, 300, 417,
    351, 399, 175, 152, 377, 400,
    378, 379, 365, 397
]

# config landmarker
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


# helper funcs
def get_landmark_pixels(frame_bgr, landmarks, indices):
    """Convex-hull mask over landmark indices → pixel array (N,3) BGR."""
    h, w = frame_bgr.shape[:2]
    pts  = np.array(
        [[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in indices],
        dtype=np.int32
    )
    mask   = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    pixels = frame_bgr[mask > 0]
    return pixels.astype(np.float64) if len(pixels) > 0 else None


def sample_patch(frame_bgr, landmarks, lm_idx, h, w, patch_px=8):
    """Small square patch centred on one landmark → (R, G, B) means or None."""
    lm  = landmarks[lm_idx]
    cx, cy = int(lm.x * w), int(lm.y * h)
    x0, x1 = max(0, cx - patch_px), min(w, cx + patch_px)
    y0, y1 = max(0, cy - patch_px), min(h, cy + patch_px)
    if x1 <= x0 or y1 <= y0:
        return None
    patch = frame_bgr[y0:y1, x0:x1].astype(np.float64)
    return patch[:, :, 2].mean(), patch[:, :, 1].mean(), patch[:, :, 0].mean()

# signal helper funcs
def butterworth_filter(signal):
    nyq   = FS / 2.0
    b, a  = butter(4, [LOW_HZ / nyq, min(HIGH_HZ / nyq, 0.999)], btype='band')
    return filtfilt(b, a, signal) if len(signal) >= 15 else signal.copy()


def chrom_ppg_segment(R_arr, G_arr, B_arr):
    """Chrominance PPG with segment-level alpha (de Haan & Jeanne 2013)."""
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

# video extraction
def extract_video(video_path, omega, n_subregions, model_path):
    """
    Read the video exactly ONCE.  For every frame, collect:
      - R/G/B means from LEFT, MID, RIGHT landmark regions  → for SVM signals
      - R/G/B means from each of n_subregions mid-face patches → for PPG maps

    Then slice into segments and compute both outputs.

    Returns:
        svm_segments  : list of dicts with keys GL/GM/GR/CL/CM/CR
        map_segments  : list of (omega, 2*n_subregions) float32 arrays
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], []

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0 or video_fps > 240:
        video_fps = FS

    lm_map_indices = MID_FACE_32[:n_subregions]

    # SVM buffers: R, G, B per ROI per frame
    svm_raw = {
        'RL': [], 'RM': [], 'RR': [],
        'GL': [], 'GM': [], 'GR': [],
        'BL': [], 'BM': [], 'BR': [],
    }

    # pppg map buffers, R, G, B per sub-region per frame
    map_R = [[] for _ in range(n_subregions)]
    map_G = [[] for _ in range(n_subregions)]
    map_B = [[] for _ in range(n_subregions)]

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
                # No face, pad all map buffers with zeros to stay aligned
                for i in range(n_subregions):
                    map_R[i].append(0.0); map_G[i].append(0.0); map_B[i].append(0.0)
                continue

            landmarks = result.face_landmarks[0]

            # SVM ROI signals (left / mid / right cheek)
            for tag, indices in [('L', LEFT_CHEEK_IDX),
                                  ('M', MID_FACE_IDX),
                                  ('R', RIGHT_CHEEK_IDX)]:
                pixels = get_landmark_pixels(frame_bgr, landmarks, indices)
                if pixels is not None:
                    svm_raw['R' + tag].append(np.mean(pixels[:, 2]))
                    svm_raw['G' + tag].append(np.mean(pixels[:, 1]))
                    svm_raw['B' + tag].append(np.mean(pixels[:, 0]))

            # ppg map sub-region patches
            for i, lm_idx in enumerate(lm_map_indices):
                rgb = sample_patch(frame_bgr, landmarks, lm_idx, h, w)
                if rgb is None:
                    map_R[i].append(0.0); map_G[i].append(0.0); map_B[i].append(0.0)
                else:
                    map_R[i].append(rgb[0]); map_G[i].append(rgb[1]); map_B[i].append(rgb[2])

    cap.release()

    step = omega // 2  # 50% overlap between segments

    # building svm segments
    svm_lengths = [len(svm_raw[k]) for k in svm_raw]
    svm_segments = []

    if svm_lengths and min(svm_lengths) >= omega:
        n_svm = min(svm_lengths)
        for start in range(0, n_svm - omega + 1, step):
            end = start + omega
            seg = {}

            # green channel ppg
            for tag in ('L', 'M', 'R'):
                arr        = np.array(svm_raw['G' + tag][start:end], dtype=np.float64)
                seg['G' + tag] = butterworth_filter(arr).astype(np.float32)

            # chrominance ppg (alpha computed per segment)
            for tag, c_key in [('L', 'CL'), ('M', 'CM'), ('R', 'CR')]:
                R_arr = np.array(svm_raw['R' + tag][start:end], dtype=np.float64)
                G_arr = np.array(svm_raw['G' + tag][start:end], dtype=np.float64)
                B_arr = np.array(svm_raw['B' + tag][start:end], dtype=np.float64)
                c_sig = chrom_ppg_segment(R_arr, G_arr, B_arr)
                seg[c_key] = butterworth_filter(c_sig).astype(np.float32)

            svm_segments.append(seg)

    # build ppg map segments
    map_n_frames = min(len(map_R[i]) for i in range(n_subregions))
    map_segments = []

    if map_n_frames >= omega:
        for start in range(0, map_n_frames - omega + 1, step):
            end      = start + omega
            ppg_cols = np.zeros((omega, n_subregions), dtype=np.float32)
            psd_cols = np.zeros((omega, n_subregions), dtype=np.float32)

            for i in range(n_subregions):
                R_arr = np.array(map_R[i][start:end], dtype=np.float64)
                G_arr = np.array(map_G[i][start:end], dtype=np.float64)
                B_arr = np.array(map_B[i][start:end], dtype=np.float64)

                sig = chrom_ppg_segment(R_arr, G_arr, B_arr)
                sig = butterworth_filter(sig)

                ppg_cols[:, i] = norm_0_255(sig)
                psd_cols[:, i] = norm_0_255(psd_binned(sig, n_bins=omega))

            map_segments.append(
                np.concatenate([ppg_cols, psd_cols], axis=1)  # (omega, 64)
            )

    return svm_segments, map_segments

# svm feature vectors as in paper
def _psd(sig):
    _, p = welch(sig.astype(np.float64), fs=FS, nperseg=min(len(sig), 256))
    return p

def _spec_ac(sig):
    p  = _psd(sig)
    ac = np.correlate(p, p, mode='full')
    return ac[len(p) - 1:]

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
    win   = max(1, FS)
    wins  = [sig[i:i+win] for i in range(0, len(sig) - win + 1, win)]
    diffs = np.diff(sig)
    ac    = np.correlate(sig - sig.mean(), sig - sig.mean(), mode='full')
    hist, _ = np.histogram(sig, bins=50, density=True)
    return [float(np.std(sig)),
            float(np.std([np.mean(w) for w in wins])) if wins else 0.0,
            float(np.sqrt(np.mean(diffs**2))) if len(diffs) else 0.0,
            float(np.mean([np.std(np.diff(w)) for w in wins])) if wins else 0.0,
            float(np.std(diffs)) if len(diffs) else 0.0,
            float(np.mean(ac[len(sig) - 1:])),
            float(scipy_entropy(hist + 1e-9))]

def build_feature_vector(seg):
    GL = seg['GL'];  GM = seg['GM'];  GR = seg['GR']
    CL = seg['CL'];  CM = seg['CM'];  CR = seg['CR']
    S  = [GL, GM, GR, CL, CM, CR]
    DC = [np.abs(CL - CM), np.abs(CL - CR), np.abs(CR - CM)]
    feats = []

    log_DC = [np.log(np.abs(d) + 1e-9) for d in DC]
    for i in range(len(log_DC)):
        for j in range(i + 1, len(log_DC)):
            feats.extend(F1(log_DC[i], log_DC[j]))

    for s in [np.log(np.abs(x) + 1e-9) for x in S]:
        feats.extend(F3(s))

    for i in range(len(DC)):
        for j in range(i + 1, len(DC)):
            feats.extend(F3(_psd(DC[i]) * _psd(DC[j])))

    for s in [np.log(np.abs(x) + 1e-9) for x in S]:
        feats.extend(F4(s))

    for i in range(len(DC)):
        for j in range(i + 1, len(DC)):
            feats.extend(F4(_psd(DC[i]) * _psd(DC[j])))

    for s in S:
        ac = _spec_ac(s)
        feats.extend([float(np.mean(ac)), float(np.max(ac))])

    return np.array(feats, dtype=np.float32)

def process_directory(video_dir, label, omega, n_subregions, model_path, limit):
    """
    Process all videos in a directory in a SINGLE PASS,
    collecting both SVM features and PPG maps simultaneously.
    """
    files = sorted(
        f for f in os.listdir(video_dir)
        if f.lower().endswith(('.mp4', '.avi', '.mov'))
    )
    if limit is not None:
        files = files[:limit]

    tag = "Real" if label == 0 else "Fake"

    svm_X, svm_y, svm_names = [], [], []
    map_X, map_y, map_names = [], [], []

    for fname in tqdm(files, desc=f"{tag} ({os.path.basename(video_dir)}) [{len(files)} videos]"):
        path = os.path.join(video_dir, fname)
        try:
            svm_segs, map_segs = extract_video(path, omega, n_subregions, model_path)

            # SVM features
            for seg in svm_segs:
                fv = build_feature_vector(seg)
                if np.isfinite(fv).all():
                    svm_X.append(fv)
                    svm_y.append(label)
                    svm_names.append(fname)

            # PPG maps
            for m in map_segs:
                map_X.append(m)
                map_y.append(label)
                map_names.append(fname)

        except Exception as e:
            print(f"  [ERROR] {fname}: {e}")

    return (svm_X, svm_y, svm_names), (map_X, map_y, map_names)

def main():
    ap = argparse.ArgumentParser(
        description="FakeCatcher — combined SVM + CNN extraction in one video pass"
    )
    ap.add_argument('--real_dir',      required=True,  help='Folder of real videos')
    ap.add_argument('--fake_dir',      required=True,  help='Folder of fake videos')
    ap.add_argument('--out_features',  default='data/features_svm.npz',
                    help='Output path for SVM feature file')
    ap.add_argument('--out_maps',      default='data/ppg_maps.npz',
                    help='Output path for CNN PPG map file')
    ap.add_argument('--omega',         type=int, default=128,
                    help='Segment length in frames (default 128 ≈ 4.3s @ 30fps)')
    ap.add_argument('--n_subregions',  type=int, default=32,
                    help='Mid-face sub-regions for PPG maps (default 32)')
    ap.add_argument('--model',         default='data/face_landmarker.task',
                    help='Path to face_landmarker.task model file')
    ap.add_argument('--limit',         type=int, default=None,
                    help='Max videos per class, e.g. --limit 300 (default: all)')
    args = ap.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(
            f"Model not found: {args.model}\n"
            "Download from: https://storage.googleapis.com/mediapipe-models/"
            "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        )

    os.makedirs(os.path.dirname(args.out_features) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(args.out_maps)     or '.', exist_ok=True)

    print(f"\n{'='*60}")
    print("  FakeCatcher — Combined Extraction (single video pass)")
    print(f"  model        : {args.model}")
    print(f"  omega        : {args.omega} frames (~{args.omega/FS:.1f}s @ {FS}fps)")
    print(f"  n_subregions : {args.n_subregions}")
    print(f"  real_dir     : {args.real_dir}")
    print(f"  fake_dir     : {args.fake_dir}")
    print(f"  out_features : {args.out_features}")
    print(f"  out_maps     : {args.out_maps}")
    print(f"  limit        : {args.limit or 'all'} videos per class")
    print(f"{'='*60}\n")

    # process real videos
    (svm_Xr, svm_yr, svm_nr), (map_Xr, map_yr, map_nr) = process_directory(
        args.real_dir, label=0,
        omega=args.omega, n_subregions=args.n_subregions,
        model_path=args.model, limit=args.limit
    )

    # process fake videos
    (svm_Xf, svm_yf, svm_nf), (map_Xf, map_yf, map_nf) = process_directory(
        args.fake_dir, label=1,
        omega=args.omega, n_subregions=args.n_subregions,
        model_path=args.model, limit=args.limit
    )

    # save SVM features
    svm_X = np.array(svm_Xr + svm_Xf, dtype=np.float32)
    svm_y = np.array(svm_yr + svm_yf, dtype=np.int32)
    svm_n = np.array(svm_nr + svm_nf)
    np.savez(args.out_features, X=svm_X, y=svm_y, video_names=svm_n)

    # save PPG maps 
    map_X = np.array(map_Xr + map_Xf, dtype=np.float32)[..., np.newaxis] / 255.0
    map_y = np.array(map_yr + map_yf, dtype=np.int32)
    map_n = np.array(map_nr + map_nf)
    np.savez(args.out_maps, X=map_X, y=map_y, video_names=map_n)

    # summary
    print(f"\n{'='*60}")
    print("  Extraction complete — both outputs saved")
    print(f"\n  SVM features → {args.out_features}")
    print(f"    Segments  : {len(svm_X)}  "
          f"(real: {int((svm_y==0).sum())}  fake: {int((svm_y==1).sum())})")
    if len(svm_X):
        print(f"    Feature dims : {svm_X.shape[1]}")

    print(f"\n  PPG maps → {args.out_maps}")
    print(f"    Segments  : {len(map_X)}  "
          f"(real: {int((map_y==0).sum())}  fake: {int((map_y==1).sum())})")
    if len(map_X):
        print(f"    Map shape    : {map_X.shape[1:]}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()