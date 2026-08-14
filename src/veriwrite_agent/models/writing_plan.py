"""Evidence-first planning contracts for reliable V0.4 paragraph writing."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Literal

from pydantic import Field, field_validator, model_validator

from veriwrite_agent.models.literature_discovery import canonicalize_doi
from veriwrite_agent.models.requirements import StrictModel
from veriwrite_agent.models.writing import (
    ParagraphRole,
    SectionEvidenceItem,
    SectionSourceRecord,
)

ArgumentMove = Literal[
    "frame_problem",
    "compare_studies",
    "synthesize_consensus",
    "analyze_difference",
    "evaluate_limitation",
    "author_judgment",
    "legacy_unspecified",
]


class ParagraphPlanProposal(StrictModel):
    """Semantic paragraph plan using short aliases instead of authority IDs."""

    role: ParagraphRole
    purpose: str = Field(min_length=1)
    claim_focus: str = Field(min_length=1)
    central_question: str = "legacy_unspecified"
    argument_move: ArgumentMove = "legacy_unspecified"
    comparison_axis: str | None = None
    relative_weight: int = Field(ge=1, le=10)
    # Tolerate a model over-selecting evidence on the first attempt; the deterministic
    # allocator trims to the executable contract (five evidence cards per paragraph)
    # before the compiled WritingParagraphPlan is built.
    evidence_refs: list[str] = Field(default_factory=list, max_length=8)
    # The proposal contract must accommodate the executable policy rather than
    # imposing a second, smaller hard-coded citation-cluster limit. Compilation
    # still enforces SectionEvidencePacket.max_sources_per_paragraph.
    source_refs: list[str] = Field(default_factory=list, max_length=8)

    @field_validator("evidence_refs", mode="after")
    @classmethod
    def evidence_refs_must_be_unique(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("paragraph evidence refs must be unique")
        if any(not value.startswith("E") or not value[1:].isdigit() for value in values):
            raise ValueError("paragraph evidence refs must use E-number aliases")
        return values

    @field_validator("source_refs", mode="after")
    @classmethod
    def source_refs_must_be_unique(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("paragraph source refs must be unique")
        if any(not value.startswith("S") or not value[1:].isdigit() for value in values):
            raise ValueError("paragraph source refs must use S-number aliases")
        return values


class SectionPlanProposal(StrictModel):
    """One LLM-proposed paragraph sequence before deterministic compilation."""

    section_id: str = Field(pattern=r"^[a-z][a-z0-9_]{1,39}$")
    # Initial planning still requests at least three paragraphs.  A global structural
    # editor may legitimately collapse a short section to two and then semantically
    # replan those two surviving roles.
    paragraphs: list[ParagraphPlanProposal] = Field(min_length=2, max_length=12)


class WritingParagraphPlan(StrictModel):
    """One code-compiled paragraph assignment with real evidence authority."""

    paragraph_id: str = Field(pattern=r"^[a-z][a-z0-9_]{1,39}_p[0-9]{2}$")
    section_id: str = Field(pattern=r"^[a-z][a-z0-9_]{1,39}$")
    paragraph_number: int = Field(ge=1, le=99)
    role: ParagraphRole
    purpose: str = Field(min_length=1)
    claim_focus: str = Field(min_length=1)
    central_question: str = "legacy_unspecified"
    argument_move: ArgumentMove = "legacy_unspecified"
    comparison_axis: str | None = None
    target_words: int = Field(ge=50)
    coverage_only: bool = False
    evidence_card_ids: list[str] = Field(default_factory=list, max_length=5)
    source_dois: list[str] = Field(default_factory=list, max_length=8)
    # Deferred full-text enhancement: a paragraph downgraded during evidence recovery
    # keeps its original argument intent here so a later PDF-backed pass can upgrade it
    # back from background to detailed evidence without losing the planned claim.
    deferred_argument: ArgumentMove | None = None
    deferred_comparison_axis: str | None = None
    deferred_purpose: str | None = None
    deferred_claim_focus: str | None = None
    deferred_central_question: str | None = None
    deferred_recovery_dois: list[str] = Field(default_factory=list)

    @field_validator("evidence_card_ids", "source_dois", mode="after")
    @classmethod
    def support_values_must_be_unique(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("planned paragraph support values must be unique")
        return values

    @field_validator("source_dois")
    @classmethod
    def normalize_dois(cls, values: list[str]) -> list[str]:
        return [canonicalize_doi(value) for value in values]

    @model_validator(mode="after")
    def planned_paragraph_needs_support(self) -> WritingParagraphPlan:
        if not self.evidence_card_ids and not self.source_dois:
            raise ValueError("every planned paragraph needs source support")
        if self.role == "detailed_evidence" and not self.evidence_card_ids:
            raise ValueError("detailed paragraph plans require evidence cards")
        return self


class WritingSectionPlan(StrictModel):
    """Audited paragraph sequence for one confirmed outline section."""

    section_id: str = Field(pattern=r"^[a-z][a-z0-9_]{1,39}$")
    title: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    target_words: int = Field(ge=100)
    counting_policy: Literal[
        "chinese_chars_and_english_words", "words"
    ] = "chinese_chars_and_english_words"
    # Initial planning still proposes at least three paragraphs, but a confirmed
    # full-manuscript edit may merge a redundant paragraph and leave two substantive
    # paragraphs in a short section.
    paragraphs: list[WritingParagraphPlan] = Field(min_length=2, max_length=12)

    @model_validator(mode="after")
    def paragraphs_must_match_section_and_budget(self) -> WritingSectionPlan:
        numbers = [paragraph.paragraph_number for paragraph in self.paragraphs]
        if numbers != list(range(1, len(self.paragraphs) + 1)):
            raise ValueError("planned paragraph numbers must be consecutive")
        if any(paragraph.section_id != self.section_id for paragraph in self.paragraphs):
            raise ValueError("planned paragraphs must belong to their section")
        if sum(paragraph.target_words for paragraph in self.paragraphs) != self.target_words:
            raise ValueError("planned paragraph targets must sum to the section target")
        return self


class GroundedWritingPlan(StrictModel):
    """One recoverable, user-confirmable plan compiled from actual V0.3 evidence."""

    schema_version: Literal["0.4-plan.0"] = "0.4-plan.0"
    status: Literal["draft", "confirmed"] = "draft"
    topic: str = Field(min_length=1)
    output_language: Literal[
        "Chinese", "English", "bilingual", "pending_confirmation"
    ] = "pending_confirmation"
    plan_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    required_source_dois: list[str] = Field(default_factory=list)
    sections: list[WritingSectionPlan] = Field(min_length=1)
    confirmed_by: str | None = None
    confirmed_at: datetime | None = None

    @model_validator(mode="after")
    def status_must_match_confirmation(self) -> GroundedWritingPlan:
        section_ids = [section.section_id for section in self.sections]
        if len(section_ids) != len(set(section_ids)):
            raise ValueError("writing plan section IDs must be unique")
        if self.status == "confirmed":
            if not self.confirmed_by or self.confirmed_at is None:
                raise ValueError("confirmed writing plans need confirmation audit fields")
        elif self.confirmed_by is not None or self.confirmed_at is not None:
            raise ValueError("draft writing plans cannot claim confirmation")
        required = [canonicalize_doi(doi) for doi in self.required_source_dois]
        if len(required) != len(set(required)):
            raise ValueError("required writing-plan source DOI values must be unique")
        planned = {
            doi
            for section in self.sections
            for paragraph in section.paragraphs
            for doi in paragraph.source_dois
        }
        missing = [doi for doi in required if doi not in planned]
        if missing:
            raise ValueError("writing plan does not cover every required source DOI")
        self.required_source_dois = required
        return self

    def confirm(self, *, confirmed_by: str) -> GroundedWritingPlan:
        name = confirmed_by.strip()
        if not name:
            raise ValueError("confirmed_by cannot be blank")
        return self.model_copy(
            update={
                "status": "confirmed",
                "confirmed_by": name,
                "confirmed_at": datetime.now(timezone.utc),
            }
        )


class ParagraphTextProposal(StrictModel):
    """LLM paragraph prose without any citation-selection authority."""

    text: str = Field(min_length=1)

    @field_validator("text", mode="before")
    @classmethod
    def normalize_control_characters(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        normalized = re.sub(r"[\x00-\x1f\x7f]+", " ", value)
        normalized = " ".join(normalized.split())
        if not normalized:
            raise ValueError("paragraph text cannot be blank after normalization")
        return normalized


class ParagraphEvidencePacket(StrictModel):
    """Minimal locked context for writing exactly one planned paragraph."""

    schema_version: Literal["0.4-paragraph.0"] = "0.4-paragraph.0"
    section_id: str = Field(pattern=r"^[a-z][a-z0-9_]{1,39}$")
    section_title: str = Field(min_length=1)
    paragraph: WritingParagraphPlan
    counting_policy: Literal[
        "chinese_chars_and_english_words", "words"
    ] = "chinese_chars_and_english_words"
    output_language: Literal[
        "Chinese", "English", "bilingual", "pending_confirmation"
    ] = "pending_confirmation"
    evidence_items: list[SectionEvidenceItem] = Field(default_factory=list, max_length=5)
    sources: list[SectionSourceRecord] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def locked_inputs_must_match_plan(self) -> ParagraphEvidencePacket:
        if self.paragraph.section_id != self.section_id:
            raise ValueError("paragraph packet section does not match its plan")
        evidence_ids = [item.evidence_id for item in self.evidence_items]
        if evidence_ids != self.paragraph.evidence_card_ids:
            raise ValueError("paragraph packet evidence order must match its plan")
        source_dois = [source.doi for source in self.sources]
        if source_dois != self.paragraph.source_dois:
            raise ValueError("paragraph packet source order must match its plan")
        if any(item.doi not in source_dois for item in self.evidence_items):
            raise ValueError("paragraph evidence must resolve to a locked source")
        return self
