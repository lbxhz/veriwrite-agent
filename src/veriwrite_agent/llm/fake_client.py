"""Deterministic fake LLM used by tests without API cost."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

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

