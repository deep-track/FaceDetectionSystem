# DeepTrack API on AWS (ECS Fargate, infra/ecs/deploy.sh). CPU-only.
#   docker build -t deeptrack-api .
#   docker run -p 8080:8080 -e DEEPTRACK_API_KEYS=<key> deeptrack-api
FROM python:3.12-slim

# OpenCV and MediaPipe need these system libraries (MediaPipe's
# FaceLandmarker loads libEGL/libGLESv2 even on CPU).
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libegl1 libgles2 libportaudio2 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Runtime dependencies only (requirements.txt is the full development
# environment). The image model (torch, a few GB) is left out unless
# WITH_IMAGE_MODEL=1; without it the image endpoints report unavailable and
# video works normally.
ARG WITH_IMAGE_MODEL=0
COPY requirements-server.txt requirements-image.txt ./
RUN pip install --no-cache-dir -r requirements-server.txt \
 && if [ "$WITH_IMAGE_MODEL" = "1" ]; then pip install --no-cache-dir -r requirements-image.txt; fi

COPY backend ./backend
# The video models ship in the repo (data/cnn_model.keras, data/face_landmarker.task).
# The image model's weights (best_swin.pth) come from HuggingFace when
# HUGGINGFACE_TOKEN is set (and the image was built with WITH_IMAGE_MODEL=1).
COPY data ./data

ENV PORT=8080 PYTHONUNBUFFERED=1
EXPOSE 8080
# Run from the repo root: models load from data/ and imports resolve from backend/.
CMD ["sh", "-c", "uvicorn main:app --app-dir backend --host 0.0.0.0 --port ${PORT}"]
