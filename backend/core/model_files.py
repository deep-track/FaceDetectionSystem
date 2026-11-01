"""
Where the model files live and how they are found.

Lookup order for a file such as ``cnn_model.keras``:
  1. ``<project>/data/<file>``  (or ``$DATA_DIR/<file>`` if DATA_DIR is set)
  2. a public download URL, for files that have one (the MediaPipe face landmarker)
  3. a HuggingFace repo named in ``$HF_REPO_ID`` (optional; ``$HUGGINGFACE_TOKEN`` if private)

Nothing is fetched from any account that is not explicitly configured.
"""
import logging
import os
import tempfile
import urllib.request
import zipfile

logger = logging.getLogger("deeptrack.models")

# backend/core/model_files.py -> project root is two levels up
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR     = os.path.abspath(os.getenv("DATA_DIR") or os.path.join(PROJECT_ROOT, "data"))

# Google's public MediaPipe model (also the one in data/face_landmarker.task)
LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


def data_path(filename: str) -> str:
    return os.path.join(DATA_DIR, filename)


def _download(url: str, dest: str) -> str:
    """Download to a temp file first so a failed download never leaves a broken model behind."""
    logger.info(f"Downloading {url}")
    try:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(dest), suffix=".part")
    except OSError:
        # read-only project dir (some hosts): fall back to the system temp dir
        dest = os.path.join(tempfile.gettempdir(), os.path.basename(dest))
        tmp_fd, tmp_path = tempfile.mkstemp(dir=tempfile.gettempdir(), suffix=".part")
    try:
        with os.fdopen(tmp_fd, "wb") as out, urllib.request.urlopen(url, timeout=60) as resp:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        if dest.endswith(".task") and not zipfile.is_zipfile(tmp_path):
            raise ValueError("Downloaded file is not a valid MediaPipe .task bundle")
        os.replace(tmp_path, dest)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    logger.info(f"Saved to {dest}")
    return dest


def resolve_model_file(filename: str, public_url: str = None) -> str:
    """Return a local path for ``filename`` or raise FileNotFoundError with instructions."""
    path = data_path(filename)
    if os.path.exists(path):
        logger.info(f"Using local file: {path}")
        return path

    if public_url:
        return _download(public_url, path)

    repo = os.getenv("HF_REPO_ID", "").strip()
    if repo:
        from huggingface_hub import hf_hub_download
        logger.info(f"{path} not found - fetching {filename} from HuggingFace repo {repo}")
        return hf_hub_download(repo_id=repo, filename=filename,
                               token=os.getenv("HUGGINGFACE_TOKEN") or None)

    raise FileNotFoundError(
        f"Model file '{filename}' not found at {path}. Put it in the data/ folder "
        f"(see README, 'Models'), or set HF_REPO_ID to a HuggingFace repo that contains it."
    )
