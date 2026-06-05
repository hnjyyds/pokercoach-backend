from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import FileResponse

from app.config import STATIC_DIR


router = APIRouter(tags=["pages"])


@router.get("/", include_in_schema=False)
def prototype() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
