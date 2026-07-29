"""Data contracts for DOI resolution and bibliographic identity verification."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import Field, field_validator, model_validator

from veriwrite_agent.models.literature_discovery import (
    LiteratureCandidate,
    canonicalize_doi,
)
from veriwrite_agent.models.requirements import StrictModel

DoiResolutionStatus = Literal[
    "resolved",
    "unresolvable",
    "landing_unavailable",
    "unavailable",
]
AuthorityMetadataStatus = Literal[
    "available",
    "not_found",
    "unsupported",
    "invalid",
    "unavailable",
]
VerificationStatus = Literal["verified", "excluded"]


class DoiResolutionEvidence(StrictModel):
    """Evidence that doi.org did or did not resolve a DOI to a landing page."""

    doi: str
    status: DoiResolutionStatus
    resolver_url: str = Field(min_length=1)
    final_url: str | None = None
    http_status: int | None = Field(default=None, ge=100, le=599)
    attempts: int = Field(ge=1, le=3)
    checked_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    reason: str = Field(min_length=1)

    @field_validator("doi")
    @classmethod
    def normalize_doi(cls, value: str) -> str:
        return canonicalize_doi(value)

    @model_validator(mode="after")
    def resolved_evidence_needs_a_target(self) -> DoiResolutionEvidence:
        if self.status == "resolved" and (
            not self.final_url
            or self.http_status is None
            or not 200 <= self.http_status < 400
        ):
            raise ValueError(
                "resolved DOI evidence needs a final URL and successful HTTP status"
            )
        if self.status == "landing_unavailable" and not self.final_url:
            raise ValueError("landing_unavailable evidence needs the redirected URL")
        return self


class RisBibliographicMetadata(StrictModel):
    """Normalized bibliographic fields parsed from one authority-provided RIS record."""

    doi: str | None = None
    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: int | None = Field(default=None, ge=1000, le=2100)
    journal_title: str | None = None
    publisher: str | None = None
    url: str | None = None
    ris_type: str | None = None

    @field_validator("doi")
    @classmethod
    def normalize_optional_doi(cls, value: str | None) -> str | None:
        return canonicalize_doi(value) if value else None

    @field_validator(
        "title",
        "journal_title",
        "publisher",
        "url",
        "ris_type",
        mode="after",
    )
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        clean = " ".join(value.split())
        return clean or None

    @field_validator("authors", mode="after")
    @classmethod
    def normalize_authors(cls, values: list[str]) -> list[str]:
        return [" ".join(value.split()) for value in values if value.strip()]


class AuthoritativeMetadataEvidence(StrictModel):
    """Raw RIS and parsed fields returned through DOI content negotiation."""

    doi: str
    status: AuthorityMetadataStatus
    source_url: str = Field(min_length=1)
    media_type: Literal["application/x-research-info-systems"] = (
        "application/x-research-info-systems"
    )
    metadata: RisBibliographicMetadata | None = None
    raw_ris: str | None = None
    attempts: int = Field(ge=1, le=3)
    checked_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    reason: str = Field(min_length=1)

    @field_validator("doi")
    @classmethod
    def normalize_doi(cls, value: str) -> str:
        return canonicalize_doi(value)

    @model_validator(mode="after")
    def status_must_match_payload(self) -> AuthoritativeMetadataEvidence:
        if self.status == "available" and (
            self.metadata is None or not self.raw_ris
        ):
            raise ValueError("available authority evidence needs parsed and raw RIS")
        if self.status != "available" and (
            self.metadata is not None or self.raw_ris is not None
        ):
            raise ValueError("unavailable authority evidence cannot contain RIS data")
        return self


class LiteratureVerificationResult(StrictModel):
    """Final V0.2.1 identity decision for one Crossref candidate."""

    schema_version: Literal["0.2.1"] = "0.2.1"
    status: VerificationStatus
    candidate: LiteratureCandidate
    resolution: DoiResolutionEvidence | None = None
    authority: AuthoritativeMetadataEvidence | None = None
    reason_codes: list[str] = Field(default_factory=list)
    warning_codes: list[str] = Field(default_factory=list)
    decided_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def verified_results_need_complete_evidence(self) -> LiteratureVerificationResult:
        if self.resolution is not None and self.resolution.doi != self.candidate.doi:
            raise ValueError("resolution DOI must match candidate DOI")
        if self.authority is not None and self.authority.doi != self.candidate.doi:
            raise ValueError("authority DOI must match candidate DOI")
        if self.status == "verified":
            if self.resolution is None:
                raise ValueError("verified results need DOI resolution evidence")
            if self.resolution.status not in {"resolved", "landing_unavailable"}:
                raise ValueError("verified results need a DOI that reached a landing target")
            if self.authority is None or self.authority.status != "available":
                raise ValueError("verified results need available authority metadata")
            metadata = self.authority.metadata
            if metadata is None or not all(
                (
                    metadata.doi,
                    metadata.title,
                    metadata.authors,
                    metadata.year,
                    metadata.journal_title,
                )
            ):
                raise ValueError("verified results need complete authority RIS identity")
            if metadata.doi != self.candidate.doi:
                raise ValueError("verified RIS DOI must match the requested candidate DOI")
            if self.reason_codes:
                raise ValueError("verified results cannot contain exclusion reasons")
            if (
                self.resolution.status == "landing_unavailable"
                and "landing_page_unavailable" not in self.warning_codes
            ):
                raise ValueError(
                    "restricted landing pages must remain visible as a warning"
                )
        elif not self.reason_codes:
            raise ValueError("excluded results need at least one reason")
        return self


class LiteratureVerificationBatch(StrictModel):
    """Batch wrapper used by the future V0.2 orchestrator and UI."""

    schema_version: Literal["0.2.1"] = "0.2.1"
    results: list[LiteratureVerificationResult] = Field(default_factory=list)

    @property
    def verified_records(self) -> list[LiteratureVerificationResult]:
        return [result for result in self.results if result.status == "verified"]

    @property
    def excluded_records(self) -> list[LiteratureVerificationResult]:
        return [result for result in self.results if result.status == "excluded"]
