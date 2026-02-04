from app.config import settings
from app.client import get_current_user
from app.recording import start_recording
from app.models import StartRecordingRequest, RecordingResponse
from fastapi import FastAPI, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

app = FastAPI(
    title="Training app for webcam product",
    version='1.0.0'
)
templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html", {"request": request}
    )

@app.get("/health")
async def health():
    return {
        "detail": "Health OK."
    }

@app.get("/signup", response_class=HTMLResponse)
def signup(request: Request):
    return templates.TemplateResponse(
        "signup.html",
        {
            "request": request,
            "SUPABASE_URL": settings.SUPABASE_URL,
            "SUPABASE_ANON_KEY": settings.SUPABASE_ANON_KEY,
        }
    )

@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse(
        "login.html",
        {
            "request": request,
            "SUPABASE_URL": settings.SUPABASE_URL,
            "SUPABASE_ANON_KEY": settings.SUPABASE_ANON_KEY,
        }
    )

@app.get("/docs/recording", response_class=HTMLResponse)
def recording_docs(request: Request):
    return templates.TemplateResponse(
        "recording_docs.html", {"request": request}
    )

@app.post("/recordings/start", response_model=RecordingResponse)
def start(req: StartRecordingRequest, user=Depends(get_current_user)):
    session_id = start_recording(
        email=user.email,
        label=req.label
    )

    return {
        "session_id": session_id,
        "message": "Recording complete and uploaded"
    }