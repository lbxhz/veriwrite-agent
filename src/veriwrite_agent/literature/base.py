"""Stable application-facing interfaces for literature infrastructure."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from veriwrite_agent.models.literature_discovery import (
    JournalRankingLookup,
    LiteratureCandidate,
    LiteratureSearchPlan,
)
from veriwrite_agent.models.literature_verification import (
    AuthoritativeMetadataEvidence,
    DoiResolutionEvidence,
)


class LiteratureSearchError(RuntimeError):
    """Raised when a scholarly search provider cannot return usable results."""


class LiteratureSearchProvider(Protocol):
    """Return DOI-backed candidates without exposing provider response objects."""

    def search(self, plan: LiteratureSearchPlan) -> Iterable[LiteratureCandidate]:
        """Yield candidates lazily until the plan or consumer stops."""


class JournalRankingProvider(Protocol):
    """Look up a versioned journal classification for one discipline."""

    @property
    def available_disciplines(self) -> tuple[str, ...]:
        """Return exact discipline names accepted by lookup."""

    def lookup(self, journal_title: str, discipline: str) -> JournalRankingLookup:
        """Return source-backed match, absence, or an internal catalog conflict."""


class DoiResolver(Protocol):
    """Resolve one canonical DOI without exposing an HTTP client."""

    def resolve(self, doi: str) -> DoiResolutionEvidence:
        """Return the DOI landing-page resolution evidence."""


class AuthoritativeMetadataProvider(Protocol):
    """Fetch registration-authority metadata for one canonical DOI."""

    def fetch(self, doi: str) -> AuthoritativeMetadataEvidence:
        """Return raw and parsed authority metadata, or an explicit failure."""
