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
## Models

Model files live in `data/` (override the folder with `DATA_DIR`). Nothing is downloaded from anyone else's account.

| File | Used by | Status |
|------|---------|--------|
| `data/cnn_model.keras` | video detector (rPPG CNN) | included |
| `data/face_landmarker.task` | video detector, face cropping | included; downloaded from Google's public MediaPipe bucket if missing |
| `data/best_swin.pth` | image detector (Swin Transformer) | **not included, train it (below)**. Until it exists the image endpoints report `unavailable`; the video detector works normally |
| `data/svm_model.pkl` | nothing (trained, not used by the API) | included |

If you host your own model files on HuggingFace, set `HF_REPO_ID=<user>/<repo>` (and `HUGGINGFACE_TOKEN` if the repo is private) and any file missing from `data/` is fetched from there. `python build.py` checks that everything needed is in place.

### Training the image detector
```bash
pip install -r requirements-image-model.txt

# 1. crop faces from your real/fake videos or images and split them train/val/test (by video)
python scripts/prepare_image_dataset.py --real_dir "path/to/real" --fake_dirs "path/to/fakeA" "path/to/fakeB" --out_dir data/image_dataset

# 2. train (use a GPU, e.g. Google Colab or Kaggle) and write data/best_swin.pth
python training/train_swin.py --data_dir data/image_dataset --epochs 12 --batch 32
```
The training script reports false-positive rate (a real face wrongly called fake) and, if you pass `--groups_csv` with a skin-tone label per test image, the same numbers per group. Check those before trusting the model for the general population. The API crops the face before classifying, exactly like the training data (`FACE_CROP=0` turns this off).

## To run extraction and training scripts
### 1. Scripts
#### A. Extract SVM features and PPG maps for CNN training
Use `scripts/extract_all_generators.py` to extract from the real videos and one or more folders of fake videos (one folder per generator, e.g. Deepfakes, Face2Face, FaceSwap):

```bash
python scripts/extract_all_generators.py ^
    --real_dir     "path/to/original_videos" ^
    --fake_dirs    "path/to/Deepfakes" "path/to/Face2Face" ^
    --out_features data/features_svm_all.npz ^
    --out_maps     data/ppg_maps_all.npz ^
    --model        data/face_landmarker.task ^
    --limit        1000
```

(Use `\` instead of `^` on Linux/macOS.) It can also merge newly extracted generators into existing `.npz` files, see the header of the script. For a single real folder and a single fake folder, `scripts/main.py` takes `--real_dir` and `--fake_dir` (one path each).

This will run for approximately 6-10 hours and produces `data/features_svm_all.npz` and `data/ppg_maps_all.npz`.

### 2. Training scripts
Train/test splits are made **by video**, not by segment: every video is cut into overlapping segments, and splitting by segment would leak near-identical data into the test set.

```bash
# For CNN
python training/train_cnn.py --maps data/ppg_maps_all.npz --output data/cnn_model.keras --epochs 20 --batch 15 --test_split 0.4

# For SVM (trained and saved, but not used by the API at the moment)
python training/train_svm.py --features data/features_svm_all.npz --output data/svm_model.pkl
```

## To start the backend

The API has two detectors: the video detector (rPPG signals + CNN, `/v1/video/*`) and the image detector (Swin Transformer, `/v1/image/*`). Models are loaded in the background after the server starts; check `GET /v1/health` to see when each one is `operational`.

Model files are looked up in `data/` first (`cnn_model.keras`, `face_landmarker.task`, `best_swin.pth`) and otherwise downloaded from HuggingFace. Set `HUGGINGFACE_TOKEN` in your `.env` or environment if the repository is private. `build.py` pre-downloads the models during a Render build.

Run from the **project root** (not from inside `backend/`):

```bash
uvicorn backend.main:app --reload --port 8000
```

The application will be live at `http://localhost:8000`, with interactive docs at `/docs`. See `API.md` for every endpoint.

### For frontend integration and CORS config
By default the backend allows every origin (`*`). Before deploying, set the `ALLOWED_ORIGINS` environment variable to a comma-separated list of the frontend URLs:

```bash
ALLOWED_ORIGINS="http://localhost:5173,https://deeptrack-staging.com"
```

No code change is needed; `backend/main.py` reads this variable when it starts.

## Uploading videos
Upload fake/real videos for analysis when booting up the live server, since recording deepfake videos and showing them directly to the webcam might be a lot of work to do.

Uploaded videos are processed with MediaPipe's VIDEO running mode (tracking across frames), while webcam/WebSocket frames use IMAGE mode (each frame analysed on its own).