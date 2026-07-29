"""Deterministic literature search provider used by tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from veriwrite_agent.models.literature_discovery import (
    LiteratureCandidate,
    LiteratureSearchPlan,
)


@dataclass
class FakeLiteratureSearchProvider:
    candidates: list[LiteratureCandidate]
    calls: list[LiteratureSearchPlan] = field(default_factory=list)

    def search(self, plan: LiteratureSearchPlan) -> Iterable[LiteratureCandidate]:
        self.calls.append(plan)
        yield from self.candidates
