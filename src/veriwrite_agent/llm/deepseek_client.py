"""DeepSeek adapter implemented through its OpenAI-compatible API."""

from __future__ import annotations

from typing import Any, Sequence

from openai import OpenAI

from veriwrite_agent.config.settings import LLMSettings
from veriwrite_agent.llm.base import ChatMessage, LLMResponseError


class DeepSeekClient:
    """Hide provider SDK details behind the project-level LLMClient interface."""

    def __init__(self, settings: LLMSettings, *, sdk_client: Any | None = None) -> None:
        self._settings = settings
        self._client = sdk_client or OpenAI(
            api_key=settings.api_key.get_secret_value(),
            base_url=str(settings.base_url),
            timeout=settings.timeout_seconds,
            max_retries=settings.max_retries,
        )

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        response_format: dict[str, str] | None = None,
    ) -> str:
        request: dict[str, object] = {
            "model": self._settings.model,
            "messages": list(messages),
            "stream": False,
            "temperature": self._settings.temperature,
            "max_tokens": self._settings.max_tokens,
        }
        if response_format is not None:
            request["response_format"] = response_format

        response = self._client.chat.completions.create(**request)
        content = response.choices[0].message.content
        if not content:
            raise LLMResponseError("The LLM provider returned empty content")
        return content
