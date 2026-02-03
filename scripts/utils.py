import io
import json
import cv2
import numpy as np
from app.client import supabase

EYE_L = [33, 133, 159, 145]
EYE_R = [362, 263, 386, 374]
MOUTH = [61, 291, 13, 14]
NOSE = [1]

LEFT_CHEEK = [50, 187, 207, 216, 215]
RIGHT_CHEEK = [280, 411, 427, 436, 435]

all_feature_windows = []
all_signal_windows = []
RECORDING_CONTEXT = {}

def mean_landmarks(landmarks, indices):
    """ calculate the mean of landmarks in the x, y and z axes """
    xs, ys, zs = [], [], []

    for i in indices:
        lm = landmarks[i]
        xs.append(lm.x)
        ys.append(lm.y)
        zs.append(lm.z)

    return (
        sum(xs) / len(xs),
        sum(ys) / len(ys),
        sum(zs) / len(zs)
    )

def eye_openness(landmarks, upper, lower):
    """ check openness of the eye using the upper and lower eyelid """
    return abs(landmarks[upper].y - landmarks[lower].y)

def mouth_openness(landmarks, upper, lower):
    """ check mouth openness using the upper and lower lip """
    return abs(landmarks[upper].y - landmarks[lower].y)

def extract_motion_features(prev_frame, curr_frame):
    """ extract motion features from previous and current frame """
    features = []

    prev_lm = prev_frame["landmarks"]
    curr_lm = curr_frame["landmarks"]

    # region motion
    for region in [EYE_L, EYE_R, MOUTH, NOSE]:
        px, py, pz = mean_landmarks(prev_lm, region)
        cx, cy, cz = mean_landmarks(curr_lm, region)
        features.extend([cx - px, cy - py, cz - pz])

    # eye motion
    features.append(eye_openness(curr_lm, 159, 145))
    features.append(eye_openness(curr_lm, 386, 374))

    # mouth motion
    features.append(mouth_openness(curr_lm, 13, 14))

    # head pose motion
    if prev_frame["pose"] is not None and curr_frame["pose"] is not None:
        prev_pose = np.array(prev_frame["pose"].data, dtype=np.float32)
        curr_pose = np.array(curr_frame["pose"].data, dtype=np.float32)
        features.extend((curr_pose - prev_pose).flatten().tolist())
    else:
        features.extend([0.0] * 16)

    return features

def extract_roi_signal(frame, landmarks, indices):
    """ extract mean RGB signal from a ROI """
    h, w, _ = frame.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    pts = np.array([
        [int(landmarks[i].x * w), int(landmarks[i].y * h)]
        for i in indices
    ], dtype=np.int32)

    cv2.fillConvexPoly(mask, pts, 255)
    roi_pixels = frame[mask == 255]

    if len(roi_pixels) == 0:
        return np.zeros(3)

    return roi_pixels.mean(axis=0)

def extract_phys_features(window_frames):
    """ extract physiological features from cheek rPPG """
    left_green = np.array([
        extract_roi_signal(f["frame"], f["landmarks"], LEFT_CHEEK)[1]
        for f in window_frames
    ])

    right_green = np.array([
        extract_roi_signal(f["frame"], f["landmarks"], RIGHT_CHEEK)[1]
        for f in window_frames
    ])

    left_stats = [left_green.mean(), left_green.std(), left_green.max(), left_green.min()]
    right_stats = [right_green.mean(), right_green.std(), right_green.max(), right_green.min()]

    corr = np.corrcoef(left_green, right_green)[0, 1] if len(left_green) > 1 else 0.0
    if np.isnan(corr):
        corr = 0.0

    return left_stats + right_stats + [corr]

def process_window(window_frames):
    """ process motion + physiological features for a window """
    motion_features = []

    for i in range(1, len(window_frames)):
        motion_features.extend(
            extract_motion_features(window_frames[i - 1], window_frames[i])
        )

    phys_features = extract_phys_features(window_frames)

    return np.concatenate([motion_features, phys_features])


def extract_cheek_signal(frame, landmarks, cheek_indices):
    """ extract green rppg signal from the cheeks """
    h, w, _ = frame.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    points = []
    for idx in cheek_indices:
        lm = landmarks[idx]
        points.append([int(lm.x * w), int(lm.y * h)])

    points = np.array(points, dtype=np.int32)
    cv2.fillConvexPoly(mask, points, 255)

    green_channel = frame[:, :, 1]
    cheek_pixels = green_channel[mask == 255]

    if len(cheek_pixels) == 0:
        return 0.0

    return np.mean(cheek_pixels)

def process_and_store_window(window_frames, left_signal, right_signal):
    """ process window into features and aligned signals """

    feature_vec = process_window(window_frames)

    signal_window = {
        "left_cheek": np.array(left_signal),
        "right_cheek": np.array(right_signal),
    }

    all_feature_windows.append(feature_vec)
    all_signal_windows.append(signal_window)

def set_recording_context(email, session_id, label):
    RECORDING_CONTEXT["email"] = email
    RECORDING_CONTEXT["session_id"] = session_id
    RECORDING_CONTEXT["label"] = label

def save_recording():
    """ save features and signals from this recording """
    
    email = RECORDING_CONTEXT["email"]
    session_id = RECORDING_CONTEXT["session_id"]
    label = RECORDING_CONTEXT["label"]

    base = f"data/{email}/{session_id}"

    # Convert features to bytes buffer
    features_buffer = io.BytesIO()
    np.savez(features_buffer, X=np.array(all_feature_windows))
    features_buffer.seek(0)  # Reset buffer position to start

    # Convert signals to bytes buffer
    signals_buffer = io.BytesIO()
    np.savez(signals_buffer, signals=np.array(all_signal_windows, dtype=object))
    signals_buffer.seek(0)  # Reset buffer position to start

    # Upload features directly from memory
    supabase.storage.from_("data").upload(
        f"{base}/features.npz",
        features_buffer.getvalue()
    )

    # Upload signals directly from memory
    supabase.storage.from_("data").upload(
        f"{base}/signals.npz",
        signals_buffer.getvalue()
    )

    # Create metadata
    metadata = {
        "email": email,
        "label": label,
        "num_windows": len(all_feature_windows),
        "pipeline_version": "v1.0.0"
    }

    # Upload metadata directly from memory
    supabase.storage.from_("data").upload(
        f"{base}/metadata.json",
        json.dumps(metadata).encode("utf-8")
    )
    
    # Clear the recording data after successful upload
    all_feature_windows.clear()
    all_signal_windows.clear()
    RECORDING_CONTEXT.clear()