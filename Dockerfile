# DeepTrack API on AWS (App Runner / ECS). CPU-only.
#   docker build -t deeptrack-api .
#   docker run -p 8080:8080 -e DEEPTRACK_API_KEYS=<key> deeptrack-api
FROM python:3.12-slim

# OpenCV and MediaPipe need these system libraries.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libportaudio2 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Runtime dependencies only (requirements.txt is the full development
# environment). CPU builds of torch/torchvision, so pip doesn't pull the
# multi-gigabyte CUDA wheels.
COPY requirements-server.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
      torch==2.14.1+cpu torchvision==0.29.1+cpu -r requirements-server.txt

COPY backend ./backend
# The video models ship in the repo (data/cnn_model.keras, data/face_landmarker.task).
# The image model (best_swin.pth) comes from HuggingFace when HUGGINGFACE_TOKEN
# is set; without it the image endpoints report unavailable.
COPY data ./data

ENV PORT=8080 PYTHONUNBUFFERED=1
EXPOSE 8080
# Run from the repo root: models load from data/ and imports resolve from backend/.
CMD ["sh", "-c", "uvicorn main:app --app-dir backend --host 0.0.0.0 --port ${PORT}"]
