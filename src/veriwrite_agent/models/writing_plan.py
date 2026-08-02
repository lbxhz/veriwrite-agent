"""Evidence-first planning contracts for reliable V0.4 paragraph writing."""

from __future__ import annotations

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


class ParagraphPlanProposal(StrictModel):
    """Semantic paragraph plan using short aliases instead of authority IDs."""

    role: ParagraphRole
    purpose: str = Field(min_length=1)
    claim_focus: str = Field(min_length=1)
    relative_weight: int = Field(ge=1, le=10)
    evidence_refs: list[str] = Field(default_factory=list, max_length=5)
    source_refs: list[str] = Field(default_factory=list, max_length=3)

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
    paragraphs: list[ParagraphPlanProposal] = Field(min_length=3, max_length=12)


class WritingParagraphPlan(StrictModel):
    """One code-compiled paragraph assignment with real evidence authority."""

    paragraph_id: str = Field(pattern=r"^[a-z][a-z0-9_]{1,39}_p[0-9]{2}$")
    section_id: str = Field(pattern=r"^[a-z][a-z0-9_]{1,39}$")
    paragraph_number: int = Field(ge=1, le=99)
    role: ParagraphRole
    purpose: str = Field(min_length=1)
    claim_focus: str = Field(min_length=1)
    target_words: int = Field(ge=80)
    evidence_card_ids: list[str] = Field(default_factory=list, max_length=5)
    source_dois: list[str] = Field(default_factory=list, max_length=8)

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
    paragraphs: list[WritingParagraphPlan] = Field(min_length=3, max_length=12)

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
    plan_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
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


class ParagraphEvidencePacket(StrictModel):
    """Minimal locked context for writing exactly one planned paragraph."""

    schema_version: Literal["0.4-paragraph.0"] = "0.4-paragraph.0"
    section_id: str = Field(pattern=r"^[a-z][a-z0-9_]{1,39}$")
    section_title: str = Field(min_length=1)
    paragraph: WritingParagraphPlan
    counting_policy: Literal[
        "chinese_chars_and_english_words", "words"
    ] = "chinese_chars_and_english_words"
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
