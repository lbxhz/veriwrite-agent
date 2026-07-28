"""Structured requirements shared by every VeriWrite module."""

from __future__ import annotations

from math import ceil
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


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
    preferred_source_types: list[str] = Field(
        default_factory=list,
        description=(
            "Canonical source types such as journal_article, book, thesis, "
            "conference_paper, and technical_report."
        ),
    )
    discouraged_source_types: list[str] = Field(
        default_factory=list,
        description=(
            "Canonical source types such as journal_article, book, thesis, "
            "conference_paper, and technical_report."
        ),
    )
    citation_order: Literal["first_appearance", "unspecified"] = "unspecified"
    in_text_style: Literal["numeric_superscript", "author_year", "unspecified"] = "unspecified"
    max_references_per_citation_cluster: int | None = Field(default=None, ge=1)
    bibliography_style: str = "pending_confirmation"
    all_bibliography_items_must_be_cited_and_discussed: bool = False

    @field_validator(
        "preferred_source_types",
        "discouraged_source_types",
        mode="before",
    )
    @classmethod
    def normalize_source_type_aliases(cls, value: object) -> object:
        if not isinstance(value, list):
            return value
        aliases = {
            "重要学术期刊论文": "journal_article",
            "期刊论文": "journal_article",
            "journal_article": "journal_article",
            "专著": "book",
            "book": "book",
            "monograph": "book",
            "硕博学位论文": "thesis",
            "硕士学位论文": "thesis",
            "博士学位论文": "thesis",
            "dissertation": "thesis",
            "thesis": "thesis",
            "会议论文": "conference_paper",
            "conference_paper": "conference_paper",
            "报告": "technical_report",
            "report": "technical_report",
            "technical_report": "technical_report",
        }
        normalized: list[object] = []
        for item in value:
            replacement = aliases.get(item, item) if isinstance(item, str) else item
            if replacement not in normalized:
                normalized.append(replacement)
        return normalized

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
    schema_version: str = "0.1.1"
    document_type: str = Field(
        description=(
            "Canonical document type. Use research_direction_literature_review "
            "for a 研究方向文献综述."
        )
    )
    institution: str | None = None
    school_or_department: str | None = Field(
        default=None,
        description="学院、院系或部门；不要与大学名称合并。",
    )
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

    @field_validator("deliverables", mode="before")
    @classmethod
    def normalize_deliverables(cls, value: object) -> object:
        if not isinstance(value, list):
            return value
        normalized: list[object] = []
        for item in value:
            replacements = (
                ["文献综述正文", "参考文献"]
                if item == "文献综述正文和参考文献"
                else [item]
            )
            for replacement in replacements:
                if replacement not in normalized:
                    normalized.append(replacement)
        return normalized

    @field_validator("required_theme_elements", mode="before")
    @classmethod
    def normalize_theme_elements(cls, value: object) -> object:
        if not isinstance(value, list):
            return value
        normalized: list[object] = []
        canonical_phrases = (
            ("人工智能", "人工智能"),
            ("新一代信息技术", "新一代信息技术"),
            ("专业领域", "专业领域"),
            ("多学科交叉", "多学科交叉"),
        )
        for item in value:
            if not isinstance(item, str):
                if item not in normalized:
                    normalized.append(item)
                continue
            matches = [
                canonical
                for phrase, canonical in canonical_phrases
                if phrase in item
            ]
            replacements = matches or [item.strip()]
            for replacement in replacements:
                if replacement not in normalized:
                    normalized.append(replacement)
        return normalized

    @field_validator("document_type", mode="before")
    @classmethod
    def normalize_document_type(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        aliases = {
            "文献综述": "research_direction_literature_review",
            "研究方向文献综述": "research_direction_literature_review",
            "research_direction_literature_review": (
                "research_direction_literature_review"
            ),
        }
        return aliases.get(value.strip(), value.strip())
