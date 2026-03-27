from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

try:
    from google import genai  # type: ignore
    from google.genai import types  # type: ignore

    GENAI_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency surface
    genai = None  # type: ignore
    types = None  # type: ignore
    GENAI_AVAILABLE = False

from ...logging import LoggingConfig
from ..errors import (
    AuthError,
    InvalidRequestError,
    LLMError,
    RateLimitedError,
    ServerError,
)
from ..errors import TimeoutError as LLMTimeoutError
from ..settings import LLMSettings
from ..types import ChatClient, EmbeddingsClient, LLMProvider, TokenCounter

logger = LoggingConfig.get_logger(__name__)


def _map_genai_exception(exc: Exception) -> LLMError:
    exc_str = str(exc).lower()
    if "api key" in exc_str or "not found" in exc_str or "unauthorized" in exc_str:
        return AuthError(str(exc))
    if "rate limit" in exc_str or "quota" in exc_str:
        return RateLimitedError(str(exc))
    if "timeout" in exc_str or "deadline" in exc_str:
        return LLMTimeoutError(str(exc))
    if "invalid" in exc_str or "bad request" in exc_str:
        return InvalidRequestError(str(exc))
    return ServerError(str(exc))


class _GoogleGenAIEmbeddings(EmbeddingsClient):
    def __init__(
        self,
        client: Any,
        model: str,
        *,
        task_type: str | None = None,
        output_dimensionality: int | None = None,
    ):
        self._client = client
        self._model = model
        self._task_type = task_type
        self._output_dimensionality = output_dimensionality

    async def embed(self, inputs: list[str]) -> list[list[float]]:
        if not self._client:
            raise NotImplementedError("Google GenAI client not available")

        import asyncio

        results: list[list[float]] = []

        started = datetime.now(UTC)
        try:
            for text in inputs:
                config = None
                if self._task_type or self._output_dimensionality:
                    config = types.EmbedContentConfig(
                        task_type=self._task_type,
                        output_dimensionality=self._output_dimensionality,
                    )

                response = await asyncio.to_thread(
                    self._client.models.embed_content,
                    model=self._model,
                    contents=text,
                    config=config,
                )
                if response.embeddings and response.embeddings[0].values:
                    results.append(response.embeddings[0].values)
                else:
                    raise ServerError("Empty embedding response from Google GenAI")

            duration_ms = int((datetime.now(UTC) - started).total_seconds() * 1000)
            try:
                logger.info(
                    "LLM request",
                    provider="google_genai",
                    operation="embeddings",
                    model=self._model,
                    inputs=len(inputs),
                    latency_ms=duration_ms,
                )
            except Exception:
                pass

            return results
        except Exception as exc:
            mapped = _map_genai_exception(exc)
            try:
                logger.warning(
                    "LLM error",
                    provider="google_genai",
                    operation="embeddings",
                    model=self._model,
                    error=type(exc).__name__,
                )
            except Exception:
                pass
            raise mapped


class _GoogleGenAIChat(ChatClient):
    def __init__(self, client: Any, model: str):
        self._client = client
        self._model = model

    async def chat(
        self, messages: list[dict[str, Any]], **kwargs: Any
    ) -> dict[str, Any]:
        if not self._client:
            raise NotImplementedError("Google GenAI client not available")

        import asyncio

        contents = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part(text=content)],
                )
            )

        generate_content_config = types.GenerateContentConfig(
            temperature=kwargs.get("temperature"),
            max_output_tokens=kwargs.get("max_tokens"),
            top_p=kwargs.get("top_p"),
            top_k=kwargs.get("top_k"),
            stop_sequence=kwargs.get("stop"),
        )

        started = datetime.now(UTC)
        try:
            response = await asyncio.to_thread(
                self._client.models.generate_content,
                model=self._model,
                contents=contents,
                config=generate_content_config,
            )

            duration_ms = int((datetime.now(UTC) - started).total_seconds() * 1000)
            try:
                logger.info(
                    "LLM request",
                    provider="google_genai",
                    operation="chat",
                    model=self._model,
                    messages=len(messages),
                    latency_ms=duration_ms,
                )
            except Exception:
                pass

            text = ""
            if response.candidates:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    text = candidate.content.parts[0].text or ""

            return {
                "text": text,
                "raw": response,
                "usage": None,
                "model": self._model,
            }
        except Exception as exc:
            mapped = _map_genai_exception(exc)
            try:
                logger.warning(
                    "LLM error",
                    provider="google_genai",
                    operation="chat",
                    model=self._model,
                    error=type(exc).__name__,
                )
            except Exception:
                pass
            raise mapped


class _NoopTokenizer(TokenCounter):
    def count(self, text: str) -> int:
        return len(text)


class GoogleGenAIProvider(LLMProvider):
    def __init__(self, settings: LLMSettings):
        self._settings = settings
        if not GENAI_AVAILABLE:
            self._client = None
        else:
            api_key = settings.api_key or ""
            self._client = genai.Client(api_key=api_key)

    def embeddings(self) -> EmbeddingsClient:
        model = self._settings.models.get("embeddings", "text-embedding-004")
        provider_options = self._settings.provider_options or {}
        return _GoogleGenAIEmbeddings(
            self._client,
            model,
            task_type=provider_options.get("task_type"),
            output_dimensionality=provider_options.get("output_dimensionality"),
        )

    def chat(self) -> ChatClient:
        model = self._settings.models.get("chat", "gemini-2.0-flash")
        return _GoogleGenAIChat(self._client, model)

    def tokenizer(self) -> TokenCounter:
        return _NoopTokenizer()
