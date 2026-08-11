"""Deterministic fake LLM clients used by tests without API cost."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

from veriwrite_agent.llm.base import ChatMessage


@dataclass
class FakeLLMClient:
    response_text: str
    calls: list[dict[str, object]] = field(default_factory=list)

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        response_format: dict[str, str] | None = None,
    ) -> str:
        self.calls.append(
            {
                "messages": list(messages),
                "response_format": response_format,
            }
        )
        return self.response_text


@dataclass
class ScriptedLLMClient:
    """Return an exact response sequence and fail on an unexpected extra call.

    Unlike :class:`FakeLLMClient`, this client makes call count part of the test
    contract.  It is useful for bounded Agent-loop smoke tests because an accidental
    retry or a non-converging controller loop fails immediately instead of silently
    reusing a plausible response.
    """

    responses: Iterable[str]
    calls: list[dict[str, object]] = field(default_factory=list, init=False)
    _remaining: list[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._remaining = list(self.responses)

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        response_format: dict[str, str] | None = None,
    ) -> str:
        call_number = len(self.calls) + 1
        self.calls.append(
            {
                "messages": list(messages),
                "response_format": response_format,
            }
        )
        if not self._remaining:
            raise AssertionError(
                f"scripted LLM received unexpected call {call_number}; "
                "the response budget is exhausted"
            )
        return self._remaining.pop(0)

    @property
    def remaining_response_count(self) -> int:
        return len(self._remaining)

    def assert_exhausted(self) -> None:
        """Assert that the workflow made every expected model call."""

        if self._remaining:
            raise AssertionError(
                f"scripted LLM has {len(self._remaining)} unconsumed response(s)"
            )
