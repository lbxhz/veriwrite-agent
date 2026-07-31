"""Contracts handed from the confirmed V0.3 evidence library to V0.4 writing."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import Field, model_validator

from veriwrite_agent.models.evidence import EvidenceLibrary
from veriwrite_agent.models.requirement_workflow import ConfirmedRequirementSpec
from veriwrite_agent.models.requirements import StrictModel


class WritingOutlineSection(StrictModel):
    section_id: str = Field(pattern=r"^[a-z][a-z0-9_]{1,39}$")
    title: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    target_words: int = Field(ge=100)
    research_questions: list[str] = Field(default_factory=list)
    core_dois: list[str] = Field(default_factory=list)
    supporting_dois: list[str] = Field(default_factory=list)
    evidence_card_ids: list[str] = Field(default_factory=list)
    evidence_gap: bool = False


class WritingOutlineDraft(StrictModel):
    schema_version: Literal["0.3.0"] = "0.3.0"
    topic: str = Field(min_length=1)
    writing_through_line: str = Field(min_length=1)
    target_words: int = Field(ge=200)
    sections: list[WritingOutlineSection] = Field(min_length=1)
    unresolved_gaps: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def section_word_budgets_must_match_total(self) -> WritingOutlineDraft:
        if sum(section.target_words for section in self.sections) != self.target_words:
            raise ValueError("section target_words must sum to the outline total")
        return self


class ConfirmedWritingOutline(StrictModel):
    schema_version: Literal["0.3.0"] = "0.3.0"
    status: Literal["confirmed"] = "confirmed"
    outline: WritingOutlineDraft
    confirmed_by: str = Field(min_length=1)
    confirmed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    confirmation_note: str | None = None

    @model_validator(mode="after")
    def confirmed_outline_cannot_have_gaps(self) -> ConfirmedWritingOutline:
        if self.outline.unresolved_gaps or any(
            section.evidence_gap for section in self.outline.sections
        ):
            raise ValueError("writing outline evidence gaps must be resolved first")
        return self


class V04WritingHandoff(StrictModel):
    """Immutable gate: V0.4 only starts from three confirmed contracts."""

    schema_version: Literal["0.4-handoff.0"] = "0.4-handoff.0"
    status: Literal["ready_for_writing"] = "ready_for_writing"
    requirement: ConfirmedRequirementSpec
    outline: ConfirmedWritingOutline
    evidence_library: EvidenceLibrary
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @model_validator(mode="after")
    def all_inputs_must_be_confirmed_and_connected(self) -> V04WritingHandoff:
        if self.evidence_library.status != "confirmed":
            raise ValueError("V0.4 requires a confirmed evidence library")
        library_dois = {record.doi for record in self.evidence_library.records}
        library_card_ids = {
            card.evidence_id for card in self.evidence_library.evidence_cards
        }
        for section in self.outline.outline.sections:
            if any(doi not in library_dois for doi in section.core_dois):
                raise ValueError("outline core_dois must exist in the library")
            if any(doi not in library_dois for doi in section.supporting_dois):
                raise ValueError("outline supporting_dois must exist in the library")
            if any(
                evidence_id not in library_card_ids
                for evidence_id in section.evidence_card_ids
            ):
                raise ValueError(
                    "outline evidence_card_ids must exist in the library"
                )
        return self
