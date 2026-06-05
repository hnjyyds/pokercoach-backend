from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import STATIC_DIR
from app.routers import auth, battle, pages, system, tools, training, users


def create_app() -> FastAPI:
    app = FastAPI(
        title="Poker Coach Mock API",
        version="0.1.0",
        summary="Mock backend for a Texas Hold'em training app prototype.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    app.include_router(system.router)
    app.include_router(pages.router)
    app.include_router(auth.router)
    app.include_router(users.router)
    app.include_router(training.router)
    app.include_router(tools.router)
    app.include_router(battle.router)
    return app


app = create_app()
