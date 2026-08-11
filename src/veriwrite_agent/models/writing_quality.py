"""Structured editorial-review contracts for V0.4 chapter prose."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from veriwrite_agent.models.requirements import StrictModel

QualityIssueCode = Literal[
    "paragraph_repetition",
    "topic_drift",
    "coherence_gap",
    "terminology_inconsistent",
    "academic_style_problem",
    "unsupported_claim",
    "overstated_evidence",
    "false_self_attribution",
    "oversized_paragraph",
]

ManuscriptIssueCode = Literal[
    "cross_section_repetition",
    "paragraph_repetition",
    "section_role_overlap",
    "academic_style_problem",
    "false_self_attribution",
    "oversized_paragraph",
    "coherence_gap",
    "terminology_inconsistent",
    "global_coherence_gap",
]


class ParagraphQualityFinding(StrictModel):
    paragraph_number: int = Field(ge=1)
    code: QualityIssueCode
    severity: Literal["warning", "blocking"] = "warning"
    detail: str = Field(min_length=1, max_length=500)
    revision_instruction: str = Field(min_length=1, max_length=500)
    claim_kind: Literal[
        "evidence_fact", "reasoned_synthesis", "author_analysis"
    ] | None = None
    evidence_card_ids: list[str] = Field(default_factory=list, max_length=5)


class SectionQualityReview(StrictModel):
    section_id: str = Field(pattern=r"^[a-z][a-z0-9_]{1,39}$")
    findings: list[ParagraphQualityFinding] = Field(default_factory=list, max_length=12)


class ManuscriptQualityFinding(StrictModel):
    """One body-paragraph action produced by the full-manuscript editor."""

    section_id: str = Field(pattern=r"^[a-z][a-z0-9_]{1,39}$")
    paragraph_number: int = Field(ge=1)
    code: ManuscriptIssueCode
    severity: Literal["warning", "blocking"] = "warning"
    disposition: Literal["report_only", "targeted_repair"] = "report_only"
    detail: str = Field(min_length=1, max_length=500)
    revision_instruction: str = Field(min_length=1, max_length=500)


class ManuscriptQualityReview(StrictModel):
    """Auditable result of a cross-section editorial pass."""

    review_status: Literal["completed", "deterministic_fallback"] = "completed"
    findings: list[ManuscriptQualityFinding] = Field(default_factory=list, max_length=16)
    review_error: str | None = Field(default=None, max_length=1000)


class ManuscriptEditorialCheckpoint(StrictModel):
    """Persistent result of the independent full-manuscript editing stage."""

    schema_version: Literal["0.4-editor.0"] = "0.4-editor.0"
    body_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["passed", "needs_revision"]
    review: ManuscriptQualityReview
    blocking_count: int = Field(ge=0)
    warning_count: int = Field(ge=0)
    completed_at: datetime
