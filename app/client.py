import os
from app.config import settings
from fastapi import HTTPException, Header
from supabase import create_client

# init client
supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)

# verify current user
def get_current_user(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid auth header")

    token = authorization.replace("Bearer ", "")
    user = supabase.auth.get_user(token)

    if user is None:
        raise HTTPException(status_code=401, detail="Invalid token")

    return user.user
