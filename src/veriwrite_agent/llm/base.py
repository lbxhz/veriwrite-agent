"""Types shared by every LLM provider implementation."""

from __future__ import annotations

from typing import Literal, Protocol, Sequence, TypedDict


class ChatMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


class LLMResponseError(RuntimeError):
    """Raised when a provider returns no usable response content."""


class LLMOutputTruncatedError(LLMResponseError):
    """Raised when a provider stops because the configured output limit was reached."""


class LLMClient(Protocol):
    """The stable interface consumed by services that need an LLM."""

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        response_format: dict[str, str] | None = None,
    ) -> str:
        """Return response text without exposing provider-specific objects."""
