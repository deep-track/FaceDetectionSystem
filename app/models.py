from pydantic import BaseModel
from typing import Literal

class StartRecordingRequest(BaseModel):
    label: Literal["real", "fake"]

class RecordingResponse(BaseModel):
    session_id: str
    message: str