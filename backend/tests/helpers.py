"""Shared test helpers — imported directly by test files."""
import io
import os
from unittest.mock import MagicMock

VALID_KEY     = os.getenv("DEEPTRACK_KEY", "dt_testkey123")
INVALID_KEY   = "dt_invalidkey000"
INACTIVE_KEY  = "dt_inactivekey000"
OVERLIMIT_KEY = "dt_overlimitkey00"
ADMIN_SECRET  = os.getenv("ADMIN_SECRET", "test-admin-secret")

MOCK_KEY_ROW_API = {
    "id":            "uuid-api-001",
    "user_id":       "user-uuid-001",
    "owner":         "Gotham Media",
    "track":         "api",
    "plan":          "starter",
    "monthly_limit": 5000,
    "is_active":     True,
}

MOCK_KEY_ROW_PLATFORM = {
    "id":            "uuid-plat-001",
    "user_id":       "user-uuid-002",
    "owner":         "Acme Corp",
    "track":         "platform",
    "plan":          "pro",
    "monthly_limit": 600,
    "is_active":     True,
}


def make_supabase_mock(key_row: dict = None, active: bool = True):
    """Build a mock Supabase client that returns the given key row on .single()."""
    mock_db     = MagicMock()
    mock_result = MagicMock()
    if key_row:
        mock_result.data = {**key_row, "is_active": active}
    else:
        mock_result.data = None
    # chain: .table().select().eq().single().execute()
    mock_db.table.return_value \
        .select.return_value \
        .eq.return_value \
        .single.return_value \
        .execute.return_value = mock_result
    # chain: .table().select().order().execute() — for list endpoints
    mock_list_result      = MagicMock()
    mock_list_result.data = [key_row] if key_row else []
    mock_db.table.return_value \
        .select.return_value \
        .order.return_value \
        .execute.return_value = mock_list_result
    return mock_db


def make_redis_mock(used: int = 0):
    """Build a mock Redis client with a given usage counter."""
    mock_redis = MagicMock()
    mock_redis.get.return_value      = str(used) if used > 0 else None
    mock_redis.incr.return_value     = used + 1
    mock_redis.expireat.return_value = True
    mock_redis.keys.return_value     = []
    return mock_redis


def small_jpg() -> bytes:
    """Generate a real minimal JPEG that PIL can open."""
    try:
        from PIL import Image
        img    = Image.new("RGB", (10, 10), color=(255, 0, 0))
        buf    = io.BytesIO()
        img.save(buf, format="JPEG")
        return buf.getvalue()
    except ImportError:
        # fallback minimal JPEG if PIL not available
        return (
            b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00'
            b'\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t'
            b'\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a'
            b'\xff\xd9'
        )