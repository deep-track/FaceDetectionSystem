"""
build.py - make sure the model files exist before the server starts (Render / Docker build step).

  face_landmarker.task   downloaded from Google's public MediaPipe bucket if missing
  cnn_model.keras        video detector - must be in data/ (or in the repo named by HF_REPO_ID)
  best_swin.pth          image detector - train it with training/train_swin.py, put it in data/
                         (or in the repo named by HF_REPO_ID). If it is missing the server still
                         starts; only the image endpoints report "unavailable".

Optional environment variables:
  HF_REPO_ID           your own HuggingFace model repo, used for files not found in data/
  HUGGINGFACE_TOKEN    token for that repo if it is private
  HF_HOME              HuggingFace cache location (must match between build and runtime)
"""
import os
import sys

from backend.core.model_files import resolve_model_file, LANDMARKER_URL, DATA_DIR

# (filename, public download URL, required for the server to be useful)
FILES = [
    ("face_landmarker.task", LANDMARKER_URL, True),
    ("cnn_model.keras",      None,           True),
    ("best_swin.pth",        None,           False),
]

print("=" * 50)
print("DeepTrack - checking model files")
print(f"data directory: {DATA_DIR}")
print("=" * 50)

missing_required = []
for filename, url, required in FILES:
    try:
        path = resolve_model_file(filename, url)
        print(f"OK       {filename} -> {path}")
    except Exception as e:
        tag = "MISSING " if required else "optional"
        print(f"{tag} {filename}: {e}")
        if required:
            missing_required.append(filename)

if missing_required:
    print(f"\nRequired model file(s) missing: {', '.join(missing_required)}")
    sys.exit(1)
print("\nModel check finished.")
