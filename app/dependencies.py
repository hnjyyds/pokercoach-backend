from __future__ import annotations

from fastapi import Header, HTTPException, status

from app.data import TOKENS, USERS
from app.schemas import User


def require_current_user(authorization: str | None = Header(default=None)) -> User:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")

    token = authorization.removeprefix("Bearer ").strip()
    email = TOKENS.get(token)
    if not email or email not in USERS:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    return USERS[email]
