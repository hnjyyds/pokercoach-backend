from __future__ import annotations

import json

import httpx
import pytest

from app.llm import (
    LLMConfig,
    LLMProviderDisabledError,
    LLMProviderError,
    LLMProviderResponseError,
    OpenAIResponsesProvider,
)


def test_provider_is_disabled_without_openai_key() -> None:
    provider = OpenAIResponsesProvider(config=LLMConfig.from_env({}))

    assert provider.enabled is False
    with pytest.raises(LLMProviderDisabledError) as exc:
        provider.generate_text("hello")

    assert exc.value.code == "llm_provider_disabled"


def test_provider_posts_openai_responses_api_shape() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth_scheme"] = request.headers["authorization"].split(" ", 1)[0]
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "resp_test",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Use pot odds first."}],
                    }
                ],
            },
        )

    provider = OpenAIResponsesProvider(
        config=LLMConfig.from_env(
            {
                "OPENAI_API_KEY": "fake-key-for-tests",
                "OPENAI_BASE_URL": "https://llm.example/v1/",
                "OPENAI_MODEL": "test-model",
            }
        ),
        transport=httpx.MockTransport(handler),
    )

    result = provider.generate_text(
        "Explain pot odds.",
        instructions="Answer as a poker coach.",
        max_output_tokens=128,
    )

    assert captured["url"] == "https://llm.example/v1/responses"
    assert captured["auth_scheme"] == "Bearer"
    assert captured["body"] == {
        "model": "test-model",
        "input": "Explain pot odds.",
        "instructions": "Answer as a poker coach.",
        "max_output_tokens": 128,
    }
    assert result.text == "Use pot odds first."
    assert result.raw["id"] == "resp_test"


def test_provider_posts_structured_output_shape() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"output_text": '{"answer":"fold"}'})

    provider = OpenAIResponsesProvider(
        config=LLMConfig.from_env(
            {
                "OPENAI_API_KEY": "fake-key-for-tests",
                "OPENAI_BASE_URL": "https://llm.example/v1",
                "OPENAI_MODEL": "test-model",
            }
        ),
        transport=httpx.MockTransport(handler),
    )

    result = provider.generate_json(
        "Pick the best action.",
        schema_name="poker_action",
        schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["answer"],
            "properties": {"answer": {"type": "string"}},
        },
    )

    assert result.data == {"answer": "fold"}
    assert captured["body"] == {
        "model": "test-model",
        "input": "Pick the best action.",
        "text": {
            "format": {
                "type": "json_schema",
                "name": "poker_action",
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["answer"],
                    "properties": {"answer": {"type": "string"}},
                },
                "strict": True,
            }
        },
    }


def test_provider_accepts_top_level_output_text() -> None:
    provider = OpenAIResponsesProvider(
        config=LLMConfig(api_key="fake-key-for-tests"),
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"output_text": "Top-level text"})),
    )

    assert provider.generate_text("hello").text == "Top-level text"


def test_provider_raises_controlled_http_errors() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={"error": {"code": "rate_limit_exceeded", "message": "raw upstream detail"}},
        )

    provider = OpenAIResponsesProvider(
        config=LLMConfig(api_key="fake-key-for-tests"),
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(LLMProviderError) as exc:
        provider.generate_text("hello")

    assert exc.value.status_code == 429
    assert exc.value.code == "rate_limit_exceeded"
    assert str(exc.value) == "LLM provider returned HTTP 429"
    assert "raw upstream detail" not in str(exc.value)


def test_provider_raises_controlled_request_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection failed with private details", request=request)

    provider = OpenAIResponsesProvider(
        config=LLMConfig(api_key="fake-key-for-tests"),
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(LLMProviderError) as exc:
        provider.generate_text("hello")

    assert exc.value.code == "llm_provider_request_failed"
    assert str(exc.value) == "LLM provider request failed"
    assert "private details" not in str(exc.value)


def test_provider_rejects_invalid_success_payloads() -> None:
    provider = OpenAIResponsesProvider(
        config=LLMConfig(api_key="fake-key-for-tests"),
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"output": []})),
    )

    with pytest.raises(LLMProviderResponseError) as exc:
        provider.generate_text("hello")

    assert exc.value.code == "llm_provider_invalid_response"
