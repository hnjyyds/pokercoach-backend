from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Mapping

import httpx


DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENAI_MODEL = "gpt-5.4-mini"


@dataclass(frozen=True)
class LLMConfig:
    api_key: str | None
    base_url: str = DEFAULT_OPENAI_BASE_URL
    model: str = DEFAULT_OPENAI_MODEL
    timeout_seconds: float = 30.0

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> LLMConfig:
        if environ is None:
            from app.config import load_environment

            load_environment()
        source = os.environ if environ is None else environ
        api_key = source.get("OPENAI_API_KEY", "").strip() or None
        base_url = source.get("OPENAI_BASE_URL", "").strip() or DEFAULT_OPENAI_BASE_URL
        model = source.get("OPENAI_MODEL", "").strip() or DEFAULT_OPENAI_MODEL
        enabled_value = source.get("POKERCOACH_LLM_ENABLED", "true").strip().lower()
        if enabled_value in {"0", "false", "no", "off"}:
            api_key = None
        return cls(api_key=api_key, base_url=base_url, model=model)


@dataclass(frozen=True)
class LLMResult:
    text: str
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class LLMJSONResult:
    data: Mapping[str, Any]
    raw: Mapping[str, Any]


class LLMProviderError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "llm_provider_error",
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


class LLMProviderDisabledError(LLMProviderError):
    def __init__(self) -> None:
        super().__init__("LLM provider is disabled", code="llm_provider_disabled")


class LLMProviderResponseError(LLMProviderError):
    pass


class OpenAIResponsesProvider:
    def __init__(
        self,
        config: LLMConfig | None = None,
        *,
        client: httpx.Client | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.config = config or LLMConfig.from_env()
        self._owns_client = client is None
        self._client = client or httpx.Client(
            timeout=self.config.timeout_seconds,
            transport=transport,
        )

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    @property
    def responses_url(self) -> str:
        return f"{self.config.base_url.rstrip('/')}/responses"

    def generate_text(
        self,
        prompt: str,
        *,
        instructions: str | None = None,
        max_output_tokens: int | None = None,
    ) -> LLMResult:
        cleaned_prompt = prompt.strip()
        if not cleaned_prompt:
            raise ValueError("prompt cannot be empty")

        body: dict[str, Any] = {
            "model": self.config.model,
            "input": cleaned_prompt,
        }
        if instructions:
            body["instructions"] = instructions
        if max_output_tokens is not None:
            body["max_output_tokens"] = max_output_tokens

        return self._create_text_response(body)

    def generate_json(
        self,
        prompt: str,
        *,
        schema_name: str,
        schema: Mapping[str, Any],
        instructions: str | None = None,
        max_output_tokens: int | None = None,
    ) -> LLMJSONResult:
        cleaned_prompt = prompt.strip()
        if not cleaned_prompt:
            raise ValueError("prompt cannot be empty")

        body: dict[str, Any] = {
            "model": self.config.model,
            "input": cleaned_prompt,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "schema": schema,
                    "strict": True,
                }
            },
        }
        if instructions:
            body["instructions"] = instructions
        if max_output_tokens is not None:
            body["max_output_tokens"] = max_output_tokens

        result = self._create_text_response(body)
        return LLMJSONResult(data=parse_json_text(result.text), raw=result.raw)

    def _create_text_response(self, body: Mapping[str, Any]) -> LLMResult:
        response = self._post_response(body)
        payload = self._parse_json(response)
        text = extract_response_text(payload)
        if text is None:
            raise LLMProviderResponseError(
                "LLM provider returned no text output",
                code="llm_provider_invalid_response",
                status_code=response.status_code,
            )
        return LLMResult(text=text, raw=payload)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> OpenAIResponsesProvider:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _post_response(self, body: Mapping[str, Any]) -> httpx.Response:
        if not self.enabled:
            raise LLMProviderDisabledError()

        try:
            response = self._client.post(
                self.responses_url,
                json=body,
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
            )
        except httpx.TimeoutException as exc:
            raise LLMProviderError("LLM provider request timed out", code="llm_provider_timeout") from exc
        except httpx.HTTPError as exc:
            raise LLMProviderError("LLM provider request failed", code="llm_provider_request_failed") from exc

        if response.is_success:
            return response

        raise LLMProviderError(
            f"LLM provider returned HTTP {response.status_code}",
            code=extract_error_code(response),
            status_code=response.status_code,
        )

    def _parse_json(self, response: httpx.Response) -> Mapping[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise LLMProviderResponseError(
                "LLM provider returned invalid JSON",
                code="llm_provider_invalid_json",
                status_code=response.status_code,
            ) from exc

        if not isinstance(payload, Mapping):
            raise LLMProviderResponseError(
                "LLM provider returned an invalid payload",
                code="llm_provider_invalid_response",
                status_code=response.status_code,
            )
        return payload


def get_llm_provider() -> OpenAIResponsesProvider:
    return OpenAIResponsesProvider()


def extract_error_code(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return "llm_provider_http_error"

    if not isinstance(payload, Mapping):
        return "llm_provider_http_error"

    error = payload.get("error")
    if not isinstance(error, Mapping):
        return "llm_provider_http_error"

    code = error.get("code")
    return code if isinstance(code, str) and code else "llm_provider_http_error"


def extract_response_text(payload: Mapping[str, Any]) -> str | None:
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text:
        return output_text

    chunks: list[str] = []
    output = payload.get("output")
    if not isinstance(output, list):
        return None

    for item in output:
        if not isinstance(item, Mapping):
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for content_item in content:
            if not isinstance(content_item, Mapping):
                continue
            text = content_item.get("text")
            if isinstance(text, str) and content_item.get("type") in {"output_text", "text"}:
                chunks.append(text)

    text = "\n".join(chunks).strip()
    return text or None


def parse_json_text(text: str) -> Mapping[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```").strip()
        cleaned = cleaned.removesuffix("```").strip()
    try:
        parsed = json.loads(cleaned)
    except ValueError as exc:
        raise LLMProviderResponseError(
            "LLM provider returned invalid structured JSON",
            code="llm_provider_invalid_structured_json",
        ) from exc
    if not isinstance(parsed, Mapping):
        raise LLMProviderResponseError(
            "LLM provider returned a non-object structured payload",
            code="llm_provider_invalid_structured_json",
        )
    return parsed
