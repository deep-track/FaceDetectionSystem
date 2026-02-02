import cv2
import time
import numpy as np
import mediapipe as mp
from collections import deque
from mediapipe.tasks.python import vision

model_path = r"data\face_landmarker.task"

# config
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
FaceLandmarkerResult = vision.FaceLandmarkerResult
VisionRunningMode = vision.RunningMode

latest_result = None
# landmarks
EYE_L = [33, 133, 159, 145]
EYE_R = [362, 263, 386, 374]
MOUTH = [61, 291, 13, 14]
NOSE = [1]
LEFT_CHEEK = [205, 50, 187, 84]   # Fakecatcher left cheek
RIGHT_CHEEK = [425, 280, 437, 348] # Fakecatcher right cheek

batch_counter = 0
BATCH_SIZE = 50
WINDOW_SIZE = 30
all_windows = []

# buffers
frame_buffer = deque(maxlen=WINDOW_SIZE)
feature_buffer = deque(maxlen=WINDOW_SIZE)

# create face landmarker instance
def result_callback(result, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result
    print("Callback fired at, ", timestamp_ms)

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_faces=1,             
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    result_callback=result_callback
)

landmarker = FaceLandmarker.create_from_options(options)

# process windows
def process_windows(window):
    """ process window into np array and save as buffer """
    global all_windows
    all_windows.append(np.array(window))

def save_recording(filename=r"data/feature_windows.npz"):
    """Save all windows from this recording into one file"""
    if not all_windows:
        print("No windows to save!")
        return

    data = np.array(all_windows)
    np.savez(filename, windows=data)
    print(f"Saved {len(all_windows)} windows to {filename}")

def mean_landmarks(landmarks, indices):
    """ calculate the mean of landmarks in the x, y and z axes"""
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

    # get region motion
    for region in [EYE_L, EYE_R, MOUTH, NOSE]:
        px, py, pz = mean_landmarks(prev_lm, region)
        cx, cy, cz = mean_landmarks(curr_lm, region)
        features.extend([cx - px, cy - py, cz - pz])

    # eye motion
    features.append(eye_openness(curr_lm, 159, 145))  # left
    features.append(eye_openness(curr_lm, 386, 374))  # right

    # mouth motion
    features.append(mouth_openness(curr_lm, 13, 14))

    # check for head pose
    if prev_frame["pose"] is not None and curr_frame["pose"] is not None:
        prev_pose = np.array(prev_frame["pose"].data, dtype=np.float32)
        curr_pose = np.array(curr_frame["pose"].data, dtype=np.float32)
        features.extend((curr_pose - prev_pose).flatten().tolist())
    else:
        features.extend([0.0]*16)

    return features

def extract_roi_signal(frame, landmarks, indices):
    """ extract mean RGB signal from a ROI """
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    pts = np.array([[int(landmarks[i].x*frame.shape[1]),
                     int(landmarks[i].y*frame.shape[0])] for i in indices])
    cv2.fillPoly(mask, [pts], 1)
    roi_pixels = frame[mask==1]
    if len(roi_pixels) == 0:
        return np.zeros(3)
    return roi_pixels.mean(axis=0)  # mean RGB

def extract_phys_features(window_frames):
    """ extract Fakecatcher physiological features for left/right cheek """
    left_green = np.array([extract_roi_signal(f["frame"], f["landmarks"], LEFT_CHEEK)[1]
                           for f in window_frames])
    right_green = np.array([extract_roi_signal(f["frame"], f["landmarks"], RIGHT_CHEEK)[1]
                            for f in window_frames])
    # stats
    left_stats = [left_green.mean(), left_green.std(), left_green.max(), left_green.min()]
    right_stats = [right_green.mean(), right_green.std(), right_green.max(), right_green.min()]
    # correlation
    corr = np.corrcoef(left_green, right_green)[0,1] if len(left_green)>1 else 0.0
    return left_stats + right_stats + [corr]

def process_window(window_frames):
    """ process motion + physiological features for a window """
    motion_features = []
    for i in range(1, len(window_frames)):
        motion_features.extend(extract_motion_features(window_frames[i-1], window_frames[i]))
    phys_features = extract_phys_features(window_frames)
    return np.concatenate([motion_features, phys_features])

def process_webcam(landmarker):
    cap = cv2.VideoCapture(0)
    #reduce cpu load
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    start_time = time.time()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        if latest_result is None:
            print("Waiting for MediaPipe results...")

        # Convert to MediaPipe Image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )

        timestamp_ms = int((time.time() - start_time) * 1000)

        landmarker.detect_async(mp_image, timestamp_ms)

        # Draw face mesh if result exists
        if latest_result and latest_result.face_landmarks:
            frame_data = {
                "timestamp": timestamp_ms,
                "landmarks": latest_result.face_landmarks[0],
                "blendshapes": (
                    latest_result.face_blendshapes[0]
                    if latest_result.face_blendshapes else None
                ),
                "pose": (
                    latest_result.facial_transformation_matrixes[0]
                    if latest_result.facial_transformation_matrixes else None
                ),
                "frame": frame.copy(), # keep frame for ROI signals
            }

            frame_buffer.append(frame_data) # append frame data to deque

            # once we have a full window, process it
            if len(frame_buffer) == WINDOW_SIZE:
                window_array = process_window(list(frame_buffer))
                all_windows.append(window_array)
                frame_buffer.popleft() # slide window

        cv2.imshow("Face Mesh (Production)", frame)

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

    save_recording()

if __name__ == "__main__":
    process_webcam(landmarker)
