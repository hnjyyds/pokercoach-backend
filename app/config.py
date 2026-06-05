from __future__ import annotations

import os
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
BASE_DIR = APP_DIR.parent
STATIC_DIR = APP_DIR / "static"

_ENV_LOADED = False


def load_environment() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return

    for path in (BASE_DIR / ".env", BASE_DIR / ".env.local"):
        load_env_file(path)
    _ENV_LOADED = True


def load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = unquote_env_value(value.strip())


def unquote_env_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


load_environment()
