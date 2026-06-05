from __future__ import annotations

from fastapi import APIRouter, status

from app.data import DEMO_USER, TOKENS, USERS
from app.schemas import AuthResponse, LoginRequest, RegisterRequest, User


router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=AuthResponse)
def login(payload: LoginRequest) -> AuthResponse:
    user = USERS.get(payload.email.lower()) or DEMO_USER
    token = f"mock-token-{user.id}"
    TOKENS[token] = user.email
    return AuthResponse(token=token, user=user)


@router.post("/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
def register(payload: RegisterRequest) -> AuthResponse:
    email = payload.email.lower()
    user = User(
        id=f"usr_{len(USERS) + 1}",
        name=payload.name,
        email=email,
        level="新手入门",
        streak_days=0,
        skill_score=520,
    )
    USERS[email] = user
    token = f"mock-token-{user.id}"
    TOKENS[token] = email
    return AuthResponse(token=token, user=user)
