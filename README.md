# FaceDetectionSystem
A face detection system built with Python, opencv, mediapipe, scipy, numpy, to detect and extract rPPG signals from various face regions e.g. left cheek, right cheek, forehead

## To start webcam 

To start your webcam and check for real/fake, run
```bash
uvicorn app:app --reload --port 8000
```

And the application will be live at `http://localhost:8000`