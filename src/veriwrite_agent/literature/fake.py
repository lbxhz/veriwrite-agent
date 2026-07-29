"""Deterministic literature search provider used by tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from veriwrite_agent.models.literature_discovery import (
    LiteratureCandidate,
    LiteratureSearchPlan,
)
from veriwrite_agent.models.literature_verification import (
    AuthoritativeMetadataEvidence,
    DoiResolutionEvidence,
)


@dataclass
class FakeLiteratureSearchProvider:
    candidates: list[LiteratureCandidate]
    calls: list[LiteratureSearchPlan] = field(default_factory=list)

    def search(self, plan: LiteratureSearchPlan) -> Iterable[LiteratureCandidate]:
        self.calls.append(plan)
        yield from self.candidates


@dataclass
class FakeDoiResolver:
    evidence_by_doi: dict[str, DoiResolutionEvidence]
    calls: list[str] = field(default_factory=list)

    def resolve(self, doi: str) -> DoiResolutionEvidence:
        self.calls.append(doi)
        return self.evidence_by_doi[doi]


@dataclass
class FakeAuthoritativeMetadataProvider:
    evidence_by_doi: dict[str, AuthoritativeMetadataEvidence]
    calls: list[str] = field(default_factory=list)

    def fetch(self, doi: str) -> AuthoritativeMetadataEvidence:
        self.calls.append(doi)
        return self.evidence_by_doi[doi]
