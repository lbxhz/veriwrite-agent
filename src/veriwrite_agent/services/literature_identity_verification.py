"""Use Crossref RIS as the canonical V0.2.1 record and validate its DOI."""

from __future__ import annotations

from collections.abc import Iterable

from veriwrite_agent.literature.base import (
    AuthoritativeMetadataProvider,
    DoiResolver,
)
from veriwrite_agent.models.literature_discovery import LiteratureCandidate
from veriwrite_agent.models.literature_verification import (
    LiteratureVerificationBatch,
    LiteratureVerificationResult,
)


class LiteratureIdentityVerificationService:
    """Verify candidate identity without asking an LLM to judge facts."""

    def __init__(
        self,
        resolver: DoiResolver,
        metadata_provider: AuthoritativeMetadataProvider,
    ) -> None:
        self._resolver = resolver
        self._metadata_provider = metadata_provider

    def verify(self, candidate: LiteratureCandidate) -> LiteratureVerificationResult:
        authority = self._metadata_provider.fetch(candidate.doi)
        if authority.status != "available" or authority.metadata is None:
            return LiteratureVerificationResult(
                status="excluded",
                candidate=candidate,
                authority=authority,
                reason_codes=[f"authority_metadata_{authority.status}"],
            )

        metadata = authority.metadata
        missing_fields = [
            field
            for field, value in (
                ("doi", metadata.doi),
                ("title", metadata.title),
                ("authors", metadata.authors),
                ("year", metadata.year),
                ("journal_title", metadata.journal_title),
            )
            if not value
        ]
        if missing_fields:
            return LiteratureVerificationResult(
                status="excluded",
                candidate=candidate,
                authority=authority,
                reason_codes=[
                    f"authority_missing_{field}" for field in missing_fields
                ],
            )
        if metadata.doi != candidate.doi:
            return LiteratureVerificationResult(
                status="excluded",
                candidate=candidate,
                authority=authority,
                reason_codes=["ris_doi_mismatch"],
            )

        resolution = self._resolver.resolve(metadata.doi)
        if resolution.status in {"unresolvable", "unavailable"}:
            return LiteratureVerificationResult(
                status="excluded",
                candidate=candidate,
                resolution=resolution,
                authority=authority,
                reason_codes=[f"doi_{resolution.status}"],
            )
        warnings = (
            ["landing_page_unavailable"]
            if resolution.status == "landing_unavailable"
            else []
        )
        return LiteratureVerificationResult(
            status="verified",
            candidate=candidate,
            resolution=resolution,
            authority=authority,
            warning_codes=warnings,
        )

    def verify_many(
        self,
        candidates: Iterable[LiteratureCandidate],
    ) -> LiteratureVerificationBatch:
        results: list[LiteratureVerificationResult] = []
        seen_dois: set[str] = set()
        for candidate in candidates:
            if candidate.doi in seen_dois:
                continue
            seen_dois.add(candidate.doi)
            results.append(self.verify(candidate))
        return LiteratureVerificationBatch(results=results)
