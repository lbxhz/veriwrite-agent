"""Executable policy compiled from a user-confirmed V0.1 requirement."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from veriwrite_agent.models.requirements import (
    AIUsagePolicy,
    FormattingRequirement,
    PolicyRule,
    SelectionPolicy,
    StrictModel,
    SubmissionRequirement,
)


class ExecutableLengthPolicy(StrictModel):
    """One unambiguous counting contract shared by writing and final audit."""

    counting_policy: Literal["chinese_chars_and_english_words", "words"]
    minimum_units: int | None = Field(default=None, ge=1)
    target_units: int = Field(ge=1)
    maximum_units: int | None = Field(default=None, ge=1)
    figures_excluded: bool = False
    excluded_components: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def bounds_must_be_consistent(self) -> ExecutableLengthPolicy:
        if self.minimum_units is not None and self.target_units < self.minimum_units:
            raise ValueError("target_units cannot be below minimum_units")
        if self.maximum_units is not None and self.target_units > self.maximum_units:
            raise ValueError("target_units cannot exceed maximum_units")
        return self


class ExecutableReferencePolicy(StrictModel):
    """Reference constraints that retrieval, selection, and delivery can enforce."""

    minimum_total: int = Field(ge=1)
    target_total: int = Field(ge=1)
    target_origin: Literal["explicit_target", "minimum_only", "system_default"]
    target_is_approximate: bool = False
    minimum_foreign_ratio: float | None = Field(default=None, gt=0, le=1)
    minimum_foreign_count: int | None = Field(default=None, ge=1)
    recent_year_window: int | None = Field(default=None, ge=1)
    recent_year_rule_strength: Literal["hard", "soft_preference", "unspecified"] = "unspecified"
    year_from: int | None = Field(default=None, ge=1900, le=2100)
    year_to: int | None = Field(default=None, ge=1900, le=2100)
    preferred_source_types: list[str] = Field(default_factory=list)
    discouraged_source_types: list[str] = Field(default_factory=list)
    citation_order: Literal["first_appearance", "unspecified"] = "unspecified"
    in_text_style: Literal["numeric_superscript", "author_year", "unspecified"] = "unspecified"
    max_references_per_citation_cluster: int | None = Field(default=None, ge=1)
    bibliography_style: str = Field(min_length=1)
    style_examples: list[str] = Field(default_factory=list)
    required_management_tools: list[str] = Field(default_factory=list)
    source_restriction_rules: list[PolicyRule] = Field(default_factory=list)
    all_bibliography_items_must_be_cited_and_discussed: bool = False

    @model_validator(mode="after")
    def reference_bounds_must_be_consistent(self) -> ExecutableReferencePolicy:
        if self.target_total < self.minimum_total:
            raise ValueError("target_total cannot be below minimum_total")
        if (
            self.minimum_foreign_count is not None
            and self.minimum_foreign_count > self.target_total
        ):
            raise ValueError("minimum_foreign_count cannot exceed target_total")
        if (
            self.year_from is not None
            and self.year_to is not None
            and self.year_to < self.year_from
        ):
            raise ValueError("year_to cannot be before year_from")
        return self


class ExecutableStructurePolicy(StrictModel):
    required_sections: list[str] = Field(default_factory=list)
    must_include_original_analysis: bool = False
    must_not_list_titles_or_abstracts_only: bool = False


class PolicyCoverageItem(StrictModel):
    """Explain where one V0.1 requirement family is consumed downstream."""

    requirement_path: str = Field(min_length=1)
    enforcement: Literal["enforced", "audited", "user_gate"]
    consumers: list[str] = Field(min_length=1)
    note: str = Field(min_length=1)


class ExecutableRequirementPolicy(StrictModel):
    """Immutable runtime contract consumed from V0.2 through final delivery."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    requirement_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    confirmed_by: str = Field(min_length=1)
    document_type: str = Field(min_length=1)
    institution: str | None = None
    school_or_department: str | None = None
    course_name: str | None = None
    output_language: Literal["Chinese", "English", "bilingual", "pending_confirmation"]
    topic: str = Field(min_length=1)
    required_theme_elements: list[str] = Field(default_factory=list)
    deliverables: list[str] = Field(default_factory=list)
    length: ExecutableLengthPolicy
    references: ExecutableReferencePolicy
    structure: ExecutableStructurePolicy
    formatting: FormattingRequirement
    workflow_conditions: list[str] = Field(default_factory=list)
    policy_rules: list[PolicyRule] = Field(default_factory=list)
    selection_policy: SelectionPolicy
    submission: SubmissionRequirement
    ai_usage: AIUsagePolicy
    acknowledged_issue_ids: list[str] = Field(default_factory=list)
    remaining_warning_ids: list[str] = Field(default_factory=list)
    resolution_notes: list[str] = Field(default_factory=list)
    unresolved_requirements: list[str] = Field(default_factory=list)
    coverage: list[PolicyCoverageItem] = Field(min_length=1)

    @model_validator(mode="after")
    def coverage_paths_must_be_unique(self) -> ExecutableRequirementPolicy:
        paths = [item.requirement_path for item in self.coverage]
        if len(paths) != len(set(paths)):
            raise ValueError("policy coverage paths must be unique")
        return self
