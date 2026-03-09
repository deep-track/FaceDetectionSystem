"""
Downloads the HF models on Render at app startup and deploy
"""
import os
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

load_dotenv()

REPO_ID = "dkkinyua/fakecatcher"
TOKEN   = os.getenv("HUGGINGFACE_TOKEN")

FILES = [
    "face_landmarker.task",
    "best_swin.pth",
    "cnn_model.keras",
]

print("=" * 50)
print("DeepTrack — pre-downloading models")
print("=" * 50)

for filename in FILES:
    print(f"\nDownloading {filename}...")
    try:
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            token=TOKEN,
        )
        print(f"{filename} cached at {path}")
    except Exception as e:
        print(f"{filename} failed: {e}")
        raise  # fail the build if any model is missing

print("\n✅ All models downloaded successfully.")