from dataclasses import dataclass

import httpx
from fastapi import Header, HTTPException

from src.config import settings


@dataclass
class AdminUser:
    id: str
    email: str


def require_admin(authorization: str | None = Header(default=None)) -> AdminUser:
    if not settings.SUPABASE_URL or not settings.SUPABASE_ANON_KEY:
        raise HTTPException(status_code=503, detail="Supabase authentication is not configured")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authentication required")
    try:
        response = httpx.get(
            f"{settings.SUPABASE_URL}/auth/v1/user",
            headers={"apikey": settings.SUPABASE_ANON_KEY, "Authorization": authorization},
            timeout=10,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to communicate with Supabase Auth: {exc}")
    if response.status_code != 200:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    payload = response.json()
    email = (payload.get("email") or "").lower()
    if email not in settings.ADMIN_EMAILS:
        raise HTTPException(status_code=403, detail="Admin access required")
    return AdminUser(id=payload["id"], email=email)
