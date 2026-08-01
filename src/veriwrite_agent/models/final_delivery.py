"""Contracts for the final, user-confirmed MVP paper deliverable."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import Field, field_validator, model_validator

from veriwrite_agent.models.executable_policy import ExecutableRequirementPolicy
from veriwrite_agent.models.literature_discovery import canonicalize_doi
from veriwrite_agent.models.requirements import StrictModel


class FinalMatterProposal(StrictModel):
    """LLM proposal generated only after the confirmed body exists."""

    title: str = Field(min_length=1)
    abstract: str = Field(min_length=80)
    keywords: list[str] = Field(min_length=3, max_length=8)
    conclusion: str = Field(min_length=80)

    @field_validator("keywords", mode="after")
    @classmethod
    def keywords_must_be_unique(cls, values: list[str]) -> list[str]:
        cleaned = [" ".join(value.split()) for value in values if value.strip()]
        if len(cleaned) != len(set(value.casefold() for value in cleaned)):
            raise ValueError("keywords must be unique")
        return cleaned


class FinalReferenceEntry(StrictModel):
    citation_key: str = Field(min_length=1)
    index: int = Field(ge=1)
    doi: str
    authors: list[str] = Field(default_factory=list)
    year: int = Field(ge=1000, le=2100)
    title: str = Field(min_length=1)
    journal: str | None = None
    publisher: str | None = None
    source_type: str = Field(min_length=1)
    is_foreign: bool
    formatted_text: str = Field(min_length=1)

    @field_validator("doi")
    @classmethod
    def normalize_doi(cls, value: str) -> str:
        return canonicalize_doi(value)


class FinalPaperAuditIssue(StrictModel):
    code: str = Field(min_length=1)
    severity: Literal["warning", "blocking"]
    requirement_path: str = Field(min_length=1)
    detail: str = Field(min_length=1)


class FinalPaperAudit(StrictModel):
    policy_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    counted_units: int = Field(ge=0)
    reference_count: int = Field(ge=0)
    foreign_reference_count: int = Field(ge=0)
    issues: list[FinalPaperAuditIssue] = Field(default_factory=list)
    deferred_checks: list[str] = Field(default_factory=lambda: ["claim_entailment"])

    @property
    def blocking_count(self) -> int:
        return sum(issue.severity == "blocking" for issue in self.issues)


class FinalPaperPackage(StrictModel):
    """Complete Markdown paper, references, audit, and confirmation state."""

    schema_version: Literal["mvp-1.0"] = "mvp-1.0"
    status: Literal["needs_revision", "ready_for_confirmation", "confirmed"]
    requirement_policy: ExecutableRequirementPolicy
    title: str = Field(min_length=1)
    abstract: str = Field(min_length=1)
    keywords: list[str] = Field(min_length=1)
    body_markdown: str = Field(min_length=1)
    conclusion: str = Field(min_length=1)
    ai_declaration: str | None = None
    references: list[FinalReferenceEntry] = Field(default_factory=list)
    markdown: str = Field(min_length=1)
    audit: FinalPaperAudit
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    confirmed_by: str | None = None
    confirmed_at: datetime | None = None

    @model_validator(mode="after")
    def status_must_match_audit_and_confirmation(self) -> FinalPaperPackage:
        blocking = self.audit.blocking_count > 0
        if self.status == "needs_revision" and not blocking:
            raise ValueError("needs_revision requires a blocking audit issue")
        if self.status in {"ready_for_confirmation", "confirmed"} and blocking:
            raise ValueError("a releasable final paper cannot contain blockers")
        if self.status == "confirmed":
            if not self.confirmed_by or self.confirmed_at is None:
                raise ValueError("confirmed final papers need confirmation audit fields")
        elif self.confirmed_by is not None or self.confirmed_at is not None:
            raise ValueError("unconfirmed final papers cannot claim confirmation")
        return self
