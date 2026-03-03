# FaceDetectionSystem
A face detection system built with Python, opencv, mediapipe, scipy, numpy, to extract rPPG signals from various face regions e.g. left cheek, right cheek, forehead and detect real / fake content from live video footage

Make sure to download the FaceForensics dataset from Google or on [Kaggle](https://www.kaggle.com/datasets/xdxd003/ff-c23)

## Setup Instructions
### 1. Clone this repo
```bash
git clone https://github.com/dkkinyua/FaceDetectionSystem.git
```
### 2. Install and activate virtual enviornment
```bash
python -m venv env_name
source env_name/bin/activate # Linux/MacOS
.\env_name\Scripts\activate # Windows
```

### 3. Install required packages
> NOTE: Make sure your virtual environment is active to prevent package mismatch
```bash
pip install -r requirements.txt
```
## To run extraction and training scripts
### 1. Scripts
#### A. Extract both features for SVM and PPG Maps for CNN training
This script extracts features for all deepfake videos and the original videos dataset
Run this
```bash
python scripts/main.py ^ 
    --real_dir "path/to/real/videos/orig.mp4" -- ^
    --fake_dir "path/to/fake/videos1/fake_1.mp4" "path/to/fake/videos2/fake_2.mp4" ^
    --out_features data/features_svm_all.npz ^
    --out_maps     data/ppg_maps_all.npz ^
    --model        data/face_landmarker.task ^
    --limit        1000
```

This will run for approximately 6-10 hours and `data/features_svm_all.npz` and `data/ppg_maps_all.npz` will be produced

### 2. Training scripts
To run the training scripts,

```bash
# For CNN
python training\train_cnn.py --maps data/ppg_maps.npz --output cnn_model.keras --epochs 20 --batch 15 --test_split 0.4

# For SVM
python training\train_svm.py --features data/features_svm_1000.npz --output data/svm_model.pkl
```
## To start backend application and configuration

To start your webcam and check for real/fake probabilities, run
```bash
cd backend
uvicorn backend.main:app --reload --port 8000
```

And the application will be live at `http://localhost:8000`

### For frontend integration and CORS config
For the backend to communicate with the frontend, Cross Origin Resource Sharing (CORS) needs to be set up.
For the backend, set up your `.env` to protect the origins and head over to line 83 of `backend/main.py` at `fastapi.middleware.cors.CORSMiddleware` and change from `allow_origins=["*"]` to  `allow_origins=[f"{origin_paths}"]` to prevent the backend being accessible from all origins.

This change will be made to the backend code before pushing to the prod server upon receiving the frontend server URLs.
Check the example below on how to do this:

```python

# set up CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://deeptrack-staging.com"], # mask these in .env using os..getenv(ENV_VAR)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

```

## Uploading videos
Upload fake/real videos for analysis when booting up the live server, since recording deepfake videos and showing them directly to the webcam might be a lot of work to do.

`VIDEO_MODE` is used when uploading videos for processing, while `IMAGE_MODE` is used for webcam streaming for per-frame analysis.