"""Data contracts for the V0.1 requirement-review workflow."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import Field, model_validator

from veriwrite_agent.models.requirements import RequirementSpec, StrictModel


class ParserRun(StrictModel):
    """Result of one parser without hiding parser failures."""

    parser_name: Literal["rule_based", "llm"]
    status: Literal["succeeded", "failed"]
    spec: RequirementSpec | None = None
    error: str | None = None

    @model_validator(mode="after")
    def result_must_match_status(self) -> ParserRun:
        if self.status == "succeeded" and self.spec is None:
            raise ValueError("a successful parser run must contain a spec")
        if self.status == "succeeded" and self.error is not None:
            raise ValueError("a successful parser run cannot contain an error")
        if self.status == "failed" and not self.error:
            raise ValueError("a failed parser run must contain an error")
        if self.status == "failed" and self.spec is not None:
            raise ValueError("a failed parser run cannot contain a spec")
        return self


class RequirementConflict(StrictModel):
    """A field on which two valid parser results disagree."""

    field: str = Field(min_length=1)
    rule_value: Any = None
    llm_value: Any = None
    provisional_value: Any = None
    reason: str = Field(min_length=1)


class ReconciliationResult(StrictModel):
    """Provisional requirement plus every disagreement kept for review."""

    merged_spec: RequirementSpec
    conflicts: list[RequirementConflict] = Field(default_factory=list)


class CompletenessIssue(StrictModel):
    """A missing, ambiguous, or operational condition requiring attention."""

    issue_id: str = Field(min_length=1)
    field: str | None = None
    severity: Literal["blocking", "warning"]
    message: str = Field(min_length=1)
    requires_user_confirmation: bool = False


class CompletenessReport(StrictModel):
    """Machine-readable review checklist for a provisional requirement."""

    issues: list[CompletenessIssue] = Field(default_factory=list)

    @property
    def blocking_count(self) -> int:
        return sum(issue.severity == "blocking" for issue in self.issues)

    @property
    def warning_count(self) -> int:
        return sum(issue.severity == "warning" for issue in self.issues)


class RequirementReviewPackage(StrictModel):
    """Everything a user needs to review before confirming requirements."""

    workflow_schema_version: str = "0.1.2"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    parser_mode: Literal["rule_only", "dual"]
    rule_run: ParserRun
    llm_run: ParserRun | None = None
    reconciliation: ReconciliationResult
    completeness: CompletenessReport
    status: Literal["needs_resolution", "ready_for_confirmation"]

    @model_validator(mode="after")
    def status_must_match_blocking_issues(self) -> RequirementReviewPackage:
        expected = (
            "needs_resolution" if self.completeness.blocking_count else "ready_for_confirmation"
        )
        if self.status != expected:
            raise ValueError("review status does not match completeness report")
        return self


class RequirementConfirmation(StrictModel):
    """User decisions applied to a review package."""

    confirmed_by: str = Field(default="user", min_length=1)
    field_updates: dict[str, Any] = Field(default_factory=dict)
    acknowledged_issue_ids: list[str] = Field(default_factory=list)
    note: str | None = None


class ConfirmedRequirementSpec(StrictModel):
    """Stable hand-off contract consumed by V0.2 and later versions."""

    workflow_schema_version: str = "0.1.2"
    status: Literal["confirmed"] = "confirmed"
    confirmed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    confirmed_by: str = Field(min_length=1)
    requirement: RequirementSpec
    acknowledged_issue_ids: list[str] = Field(default_factory=list)
    remaining_warnings: list[CompletenessIssue] = Field(default_factory=list)

    @model_validator(mode="after")
    def handoff_cannot_contain_blocking_issues(self) -> ConfirmedRequirementSpec:
        if any(issue.severity == "blocking" for issue in self.remaining_warnings):
            raise ValueError("a confirmed hand-off cannot contain blocking issues")
        return self
