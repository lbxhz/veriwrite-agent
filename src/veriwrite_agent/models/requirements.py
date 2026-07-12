"""Structured requirements shared by every VeriWrite module."""

from __future__ import annotations

from math import ceil
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator


class StrictModel(BaseModel):
    """Reject unknown fields so mistakes fail early."""

    model_config = ConfigDict(extra="forbid")


class SourceEvidence(StrictModel):
    """A source snippet explaining why a parsed field has its value."""

    field: str = Field(min_length=1)
    source_text: str = Field(min_length=1)
    note: str | None = None


class LengthRequirement(StrictModel):
    minimum_chars: int | None = Field(default=None, ge=1)
    target_chars: int | None = Field(default=None, ge=1)
    figures_excluded: bool = False
    counting_policy: Literal["pending_confirmation", "chinese_chars_and_english_words"] = (
        "pending_confirmation"
    )

    @model_validator(mode="after")
    def target_must_not_be_below_minimum(self) -> LengthRequirement:
        if (
            self.minimum_chars is not None
            and self.target_chars is not None
            and self.target_chars < self.minimum_chars
        ):
            raise ValueError("target_chars cannot be below minimum_chars")
        return self


class ReferenceRequirement(StrictModel):
    minimum_total: int | None = Field(default=None, ge=1)
    minimum_foreign_ratio: float | None = Field(default=None, gt=0, le=1)
    recent_year_window: int | None = Field(default=None, ge=1)
    recent_year_rule_strength: Literal["hard", "soft_preference", "unspecified"] = (
        "unspecified"
    )
    preferred_source_types: list[str] = Field(default_factory=list)
    discouraged_source_types: list[str] = Field(default_factory=list)
    citation_order: Literal["first_appearance", "unspecified"] = "unspecified"
    in_text_style: Literal["numeric_superscript", "author_year", "unspecified"] = "unspecified"
    max_references_per_citation_cluster: int | None = Field(default=None, ge=1)
    bibliography_style: str = "pending_confirmation"
    all_bibliography_items_must_be_cited_and_discussed: bool = False

    @computed_field
    @property
    def minimum_foreign_count(self) -> int | None:
        if self.minimum_total is None or self.minimum_foreign_ratio is None:
            return None
        return ceil(self.minimum_total * self.minimum_foreign_ratio)


class StructureRequirement(StrictModel):
    required_or_recommended_sections: list[str] = Field(default_factory=list)
    must_include_original_analysis: bool = False
    must_not_list_titles_or_abstracts_only: bool = False


class FormattingRequirement(StrictModel):
    paper_size: str | None = None
    body_font: str | None = None
    body_font_size: str | None = None
    line_spacing: float | None = Field(default=None, gt=0)


class RequirementSpec(StrictModel):
    schema_version: str = "0.1.0"
    document_type: str
    institution: str | None = None
    course_name: str | None = None
    topic: str | None = None
    topic_source: Literal["explicit", "user_confirmation_required"] = (
        "user_confirmation_required"
    )
    required_theme_elements: list[str] = Field(default_factory=list)
    deliverables: list[str] = Field(default_factory=list)
    length: LengthRequirement = Field(default_factory=LengthRequirement)
    structure: StructureRequirement = Field(default_factory=StructureRequirement)
    references: ReferenceRequirement = Field(default_factory=ReferenceRequirement)
    formatting: FormattingRequirement = Field(default_factory=FormattingRequirement)
    workflow_conditions: list[str] = Field(default_factory=list)
    ambiguities: list[str] = Field(default_factory=list)
    source_evidence: list[SourceEvidence] = Field(default_factory=list)

