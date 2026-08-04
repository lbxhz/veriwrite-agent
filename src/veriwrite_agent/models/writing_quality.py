"""Structured editorial-review contracts for V0.4 chapter prose."""

from __future__ import annotations

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
]


class ParagraphQualityFinding(StrictModel):
    paragraph_number: int = Field(ge=1)
    code: QualityIssueCode
    detail: str = Field(min_length=1, max_length=500)
    revision_instruction: str = Field(min_length=1, max_length=500)
    claim_kind: Literal[
        "evidence_fact", "reasoned_synthesis", "author_analysis"
    ] | None = None
    evidence_card_ids: list[str] = Field(default_factory=list, max_length=5)


class SectionQualityReview(StrictModel):
    section_id: str = Field(pattern=r"^[a-z][a-z0-9_]{1,39}$")
    findings: list[ParagraphQualityFinding] = Field(default_factory=list, max_length=12)
