import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("deeptrack.main")

_startup_time: float = 0.0

async def _load_models(app: FastAPI):
    """
    move all heavy imports such as tensorflow, mediapipe
    to this function so easier port binding in prod
    """
    loop = asyncio.get_running_loop()
    logger.info("=== DeepTrack model loading (background) ===")

    try:
        from backend.core.video_predictor import VideoPredictor, FrameBuffer
        vp = await loop.run_in_executor(None, VideoPredictor)
        app.state.video_predictor = vp
        app.state.video_buffer    = FrameBuffer()
        logger.info("Video predictor ready.")
    except FileNotFoundError as e:       # a model file is missing: expected, no traceback needed
        app.state.model_errors["video"] = str(e)
        logger.error(f"Video predictor unavailable: {e}")
    except Exception as e:
        app.state.model_errors["video"] = str(e)
        logger.exception("Video predictor failed to load")

    try:
        from backend.core.image_predictor import ImagePredictor
        ip = await loop.run_in_executor(None, ImagePredictor)
        app.state.image_predictor = ip
        logger.info("Image predictor ready.")
    except FileNotFoundError as e:       # e.g. best_swin.pth has not been trained yet
        app.state.model_errors["image"] = str(e)
        logger.error(f"Image predictor unavailable: {e}")
    except Exception as e:
        app.state.model_errors["image"] = str(e)
        logger.exception("Image predictor failed to load")

    logger.info("=== Model loading finished ===")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _startup_time
    _startup_time = time.time()

    app.state.video_predictor = None
    app.state.video_buffer    = None
    app.state.image_predictor = None
    app.state.jobs            = {}
    app.state.frame_buffers   = {}   # per-session buffers for POST /predict/frame
    app.state.model_errors    = {}   # "video"/"image" -> load error message

    asyncio.create_task(_load_models(app))
    logger.info("=== DeepTrack server started — models loading in background ===")

    yield

    logger.info("Shutting down...")
    if app.state.video_predictor:
        app.state.video_predictor.close()
    if app.state.image_predictor:
        app.state.image_predictor.close()
    video.shutdown_video_executor()


app = FastAPI(
    title="DeepTrack API",
    version="1.0",
    description=(
        "Unified deepfake detection API.\n\n"
        "- `/v1/video/*` — rPPG biological signal analysis (video)\n"
        "- `/v1/image/*` — Swin Transformer visual analysis (image)"
    ),
    lifespan=lifespan,
)

# Comma-separated list, e.g. ALLOWED_ORIGINS="http://localhost:5173,https://app.example.com".
# Defaults to "*" (open) so existing deployments keep working; set it in production.
ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()
] or ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# moved these imports here for same binding issue
from backend.routers import video, image
app.include_router(video.router, prefix="/v1/video", tags=["Video"])
app.include_router(image.router, prefix="/v1/image", tags=["Image"])


def _model_state(name: str, predictor) -> str:
    if predictor is not None:
        return "operational"
    return "unavailable" if name in app.state.model_errors else "loading"


@app.get("/v1/health", tags=["System"])
async def health():
    return {
        "status":         "ok",
        "uptime_seconds": round(time.time() - _startup_time),
        "video_model":    _model_state("video", app.state.video_predictor),
        "image_model":    _model_state("image", app.state.image_predictor),
    }


@app.get("/v1/status", tags=["System"])
async def status():
    MAX_VIDEO_WORKERS = video.MAX_VIDEO_WORKERS
    MAX_UPLOAD_MB     = video.MAX_UPLOAD_MB
    JOB_TTL_SECONDS   = video.JOB_TTL_SECONDS

    jobs   = app.state.jobs or {}
    counts = {}
    for j in jobs.values():
        counts[j["status"]] = counts.get(j["status"], 0) + 1

    vb = app.state.video_buffer
    return {
        "uptime_seconds": round(time.time() - _startup_time),
        "video": {
            "model_loaded":    app.state.video_predictor is not None,
            "buffer_fill_pct": vb.fill_pct if vb else None,
            "frames_seen":     vb.frame_count if vb else None,
            "jobs":            counts,
            "total_jobs":      len(jobs),
            "max_workers":     MAX_VIDEO_WORKERS,
            "max_upload_mb":   MAX_UPLOAD_MB,
            "job_ttl_hours":   JOB_TTL_SECONDS // 3600 if JOB_TTL_SECONDS else None,
        },
        "image": {
            "model_loaded": app.state.image_predictor is not None,
        },
    }


@app.get("/", response_class=HTMLResponse, tags=["System"])
async def home():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><title>DeepTrack API</title>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Mono&display=swap" rel="stylesheet">
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:#0a0a0a;color:#f0f0f0;font-family:'DM Mono',monospace;min-height:100vh;
       display:flex;flex-direction:column;align-items:center;justify-content:center;padding:40px 24px;
       background-image:linear-gradient(rgba(232,255,71,.03)1px,transparent 1px),
         linear-gradient(90deg,rgba(232,255,71,.03)1px,transparent 1px);background-size:40px 40px}
  .wordmark{font-family:'Syne',sans-serif;font-weight:800;font-size:clamp(2.8rem,7vw,4.2rem);
            letter-spacing:-.03em;line-height:1}
  .wordmark span{color:#e8ff47}
  .sub{margin-top:10px;font-size:.72rem;letter-spacing:.18em;text-transform:uppercase;color:#555}
  .cards{display:flex;gap:16px;margin-top:48px;flex-wrap:wrap;justify-content:center}
  .card{border:1px solid #222;background:#111;padding:32px 36px;text-decoration:none;
        color:inherit;transition:border-color .2s,transform .15s;min-width:200px;text-align:center}
  .card:hover{border-color:#e8ff47;transform:translateY(-3px)}
  .card-icon{font-size:2rem;margin-bottom:14px;display:block}
  .card-title{font-family:'Syne',sans-serif;font-weight:700;font-size:1.1rem;margin-bottom:6px}
  .card-desc{font-size:.7rem;color:#555;letter-spacing:.06em}
  .card-route{margin-top:14px;font-size:.65rem;color:#e8ff47;letter-spacing:.08em}
  footer{margin-top:56px;font-size:.62rem;letter-spacing:.1em;text-transform:uppercase;color:#333}
</style>
</head>
<body>
  <div class="wordmark">Deep<span>Track</span></div>
  <div class="sub">Unified Deepfake Detection API · v1.0</div>
  <div class="cards">
    <a class="card" href="/v1/video/"><span class="card-icon">🎬</span>
      <div class="card-title">Video Analysis</div>
      <div class="card-desc">rPPG biological signal detection</div>
      <div class="card-route">/v1/video/</div></a>
    <a class="card" href="/v1/image/"><span class="card-icon">🖼</span>
      <div class="card-title">Image Analysis</div>
      <div class="card-desc">Swin Transformer visual classification</div>
      <div class="card-route">/v1/image/</div></a>
    <a class="card" href="/docs"><span class="card-icon">📄</span>
      <div class="card-title">API Docs</div>
      <div class="card-desc">Interactive Swagger UI</div>
      <div class="card-route">/docs</div></a>
  </div>
  <footer>DeepTrack · internal build</footer>
</body>
</html>"""