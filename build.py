"""
build.py — pre-download models from HuggingFace during Render build step
"""
import os
from huggingface_hub import hf_hub_download
# forces consistent cache path between build step and running server
os.environ["HF_HOME"] = "/opt/render/.cache/huggingface"

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
            cache_dir=os.path.join(os.environ["HF_HOME"], "hub"),
        )
        print(f"{filename} cached at {path}")
    except Exception as e:
        print(f"{filename} failed: {e}")
        raise

print("\n✅ All models downloaded successfully.")