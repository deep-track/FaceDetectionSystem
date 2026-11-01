"""
Face cropping used by BOTH dataset preparation and the image predictor, so the
Swin model sees exactly the same kind of input at training and at inference time.
"""
import logging
import threading

import numpy as np
from PIL import Image

logger = logging.getLogger("deeptrack.facecrop")


def square_box_from_landmarks(xs, ys, width, height, margin=0.3):
    """
    Square pixel box (x0, y0, x1, y1) around a face.

    xs, ys   normalised [0, 1] landmark coordinates
    margin   extra context around the landmark extent, as a fraction of its size
    The box stays inside the image; it is shifted (not squashed) at the borders and
    shrinks only if the face is larger than the image's short side.
    """
    px = np.asarray(xs, dtype=np.float64) * width
    py = np.asarray(ys, dtype=np.float64) * height
    x_min, x_max, y_min, y_max = px.min(), px.max(), py.min(), py.max()
    side = max(x_max - x_min, y_max - y_min) * (1.0 + margin)
    side = max(2.0, min(side, float(min(width, height))))
    cx, cy = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
    x0 = min(max(cx - side / 2.0, 0.0), width - side)
    y0 = min(max(cy - side / 2.0, 0.0), height - side)
    return int(round(x0)), int(round(y0)), int(round(x0 + side)), int(round(y0 + side))


class FaceCropper:
    """Crops the largest-confidence face found by the MediaPipe FaceLandmarker."""

    def __init__(self, landmarker_path: str, margin: float = 0.3):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        self._mp = mp
        self.margin = margin
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=landmarker_path),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.4,
            min_face_presence_confidence=0.4,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(opts)
        self._lock = threading.Lock()   # the landmarker is shared between request threads

    def crop(self, image: Image.Image):
        """Return (cropped_image, face_found). Without a face the original image is returned."""
        rgb = np.ascontiguousarray(np.asarray(image.convert("RGB"), dtype=np.uint8))
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        with self._lock:
            result = self._landmarker.detect(mp_image)
        if not result.face_landmarks:
            return image, False
        lms = result.face_landmarks[0]
        h, w = rgb.shape[:2]
        box = square_box_from_landmarks([p.x for p in lms], [p.y for p in lms], w, h, self.margin)
        return image.crop(box), True

    def close(self):
        self._landmarker.close()
