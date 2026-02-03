import uuid
from scripts.main import main as run_webcam
from scripts.utils import set_recording_context

def start_recording(email: str, label: str):
    session_id = str(uuid.uuid4())

    set_recording_context(
        email=email,
        session_id=session_id,
        label=label
    )

    run_webcam()

    return session_id
