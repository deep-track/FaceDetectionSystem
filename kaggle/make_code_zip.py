"""
Pack the code needed to train the Swin image detector on Kaggle into kaggle/deeptrack-code.zip.

    python kaggle/make_code_zip.py

Then on Kaggle: Create > New Dataset > upload deeptrack-code.zip (keep it Private).
Re-run this and upload a new dataset version whenever the training code changes.
"""
import os
import zipfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deeptrack-code.zip")

FILES = [
    "backend/__init__.py",
    "backend/core/__init__.py",
    "backend/core/model_files.py",
    "backend/core/face_crop.py",
    "backend/core/image_predictor.py",
    "scripts/prepare_image_dataset.py",
    "training/train_swin.py",
    "training/evaluate_swin.py",
    "data/face_landmarker.task",
]

missing = [f for f in FILES if not os.path.exists(os.path.join(ROOT, f))]
if missing:
    raise SystemExit("Missing files, run this from a complete project folder:\n  " + "\n  ".join(missing))

with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for f in FILES:
        z.write(os.path.join(ROOT, f), f)

print(f"Wrote {OUT} ({os.path.getsize(OUT) / 1e6:.1f} MB, {len(FILES)} files)")
