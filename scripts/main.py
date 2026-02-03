import cv2
import time
import numpy as np
import mediapipe as mp
from collections import deque
from mediapipe.tasks.python import vision
from scripts.utils import extract_cheek_signal, extract_motion_features, extract_phys_features, process_and_store_window, save_recording

model_path = r"data\face_landmarker.task"

# config
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode

latest_result = None

# landmarks (MediaPipe canonical indices)
EYE_L = [33, 133, 159, 145]
EYE_R = [362, 263, 386, 374]
MOUTH = [61, 291, 13, 14]
NOSE = [1]

LEFT_CHEEK = [50, 187, 207, 216, 215]
RIGHT_CHEEK = [280, 411, 427, 436, 435]

WINDOW_SIZE = 30

# buffers
frame_buffer = deque(maxlen=WINDOW_SIZE)
left_cheek_signal = deque(maxlen=WINDOW_SIZE)
right_cheek_signal = deque(maxlen=WINDOW_SIZE)

# window-level storage
all_feature_windows = []
all_signal_windows = []

# result callback
def result_callback(result, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

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

def main():
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    start_time = time.time()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        elapsed_time = time.time() - start_time
        if elapsed_time >= 45:
            print("45 second limit reached, closing window.")
            break

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )

        timestamp_ms = int((time.time() - start_time) * 1000)
        landmarker.detect_async(mp_image, timestamp_ms)

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
                "frame": frame.copy(),
            }

            frame_buffer.append(frame_data)

            left_val = extract_cheek_signal(
                frame, frame_data["landmarks"], LEFT_CHEEK
            )
            right_val = extract_cheek_signal(
                frame, frame_data["landmarks"], RIGHT_CHEEK
            )

            left_cheek_signal.append(left_val)
            right_cheek_signal.append(right_val)

            if len(frame_buffer) == WINDOW_SIZE:
                process_and_store_window(
                    list(frame_buffer),
                    list(left_cheek_signal),
                    list(right_cheek_signal)
                )
                frame_buffer.popleft()

        cv2.imshow("Face Mesh (Production)", frame)

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

    save_recording()

if __name__ == "__main__":
    main()