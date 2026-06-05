from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def disable_live_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("POKERCOACH_LLM_ENABLED", "false")
