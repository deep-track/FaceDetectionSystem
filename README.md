# FaceDetectionSystem
A face detection system built with Python, opencv, mediapipe, scipy, numpy, to detect and extract rPPG signals from various face regions e.g. left cheek, right cheek, forehead

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

## To start webcam 

To start your webcam and check for real/fake, run
```bash
uvicorn app:app --reload --port 8000
```

And the application will be live at `http://localhost:8000`