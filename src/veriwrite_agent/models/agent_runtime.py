"""Runtime contracts for the bounded VeriWrite Agent loop.

The contracts in this module carry references and decisions, not the complete paper
artifacts. Existing V0.1-V0.5 models remain the source of truth for artifact contents.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from veriwrite_agent.models.requirements import StrictModel

AgentStage = Literal[
    "requirements",
    "literature",
    "evidence",
    "planning",
    "writing",
    "editing",
    "delivery",
]

ArtifactKind = Literal[
    "requirement_spec",
    "requirement_policy",
    "literature_blueprint",
    "literature_result",
    "literature_verification",
    "evidence_library",
    "writing_handoff",
    "writing_plan",
    "writing_project",
    "body_draft",
    "manuscript_review",
    "final_package",
    "quality_scorecard",
]


def _unique_nonempty(values: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        clean = " ".join(value.split())
        if clean and clean not in seen:
            normalized.append(clean)
            seen.add(clean)
    return normalized


class ArtifactReference(StrictModel):
    """Stable pointer to one validated stage artifact stored outside AgentState."""

    artifact_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{2,79}$")
    kind: ArtifactKind
    schema_version: str = Field(min_length=1, max_length=40)
    fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    storage_key: str = Field(min_length=1, max_length=240)
    status: Literal[
        "draft",
        "confirmed",
        "ready",
        "needs_revision",
        "superseded",
        "invalid",
    ]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RefineLiteratureSearchAction(StrictModel):
    kind: Literal["refine_literature_search"] = "refine_literature_search"
    gap_ids: list[str] = Field(min_length=1, max_length=20)
    queries: list[str] = Field(min_length=1, max_length=20)
    excluded_dois: list[str] = Field(default_factory=list, max_length=500)
    target_additional_count: int = Field(ge=1, le=100)
    allow_topic_boundary_change: bool = False

    @field_validator("gap_ids", "queries", "excluded_dois", mode="after")
    @classmethod
    def list_values_must_be_unique(cls, values: list[str]) -> list[str]:
        return _unique_nonempty(values)


class AcquireFullTextAction(StrictModel):
    kind: Literal["acquire_full_text"] = "acquire_full_text"
    dois: list[str] = Field(min_length=1, max_length=100)
    allow_open_access_download: bool = True
    allow_manual_fallback: bool = True

    @field_validator("dois", mode="after")
    @classmethod
    def dois_must_be_unique(cls, values: list[str]) -> list[str]:
        return _unique_nonempty(values)


class RebuildEvidenceAction(StrictModel):
    kind: Literal["rebuild_evidence"] = "rebuild_evidence"
    affected_section_ids: list[str] = Field(min_length=1, max_length=20)
    source_dois: list[str] = Field(min_length=1, max_length=100)
    preserve_verified_documents: bool = True

    @field_validator("affected_section_ids", "source_dois", mode="after")
    @classmethod
    def identifiers_must_be_unique(cls, values: list[str]) -> list[str]:
        return _unique_nonempty(values)


class ReviseWritingPlanAction(StrictModel):
    kind: Literal["revise_writing_plan"] = "revise_writing_plan"
    affected_section_ids: list[str] = Field(min_length=1, max_length=20)
    finding_ids: list[str] = Field(min_length=1, max_length=30)
    preserve_unaffected_sections: bool = True

    @field_validator("affected_section_ids", "finding_ids", mode="after")
    @classmethod
    def identifiers_must_be_unique(cls, values: list[str]) -> list[str]:
        return _unique_nonempty(values)


class WriteOrReviseSectionsAction(StrictModel):
    kind: Literal["write_or_revise_sections"] = "write_or_revise_sections"
    section_ids: list[str] = Field(min_length=1, max_length=20)
    paragraph_numbers: dict[str, list[int]] = Field(default_factory=dict)
    mode: Literal["draft", "targeted_revision"]
    max_revision_rounds: int = Field(default=3, ge=1, le=5)

    @field_validator("section_ids", mode="after")
    @classmethod
    def sections_must_be_unique(cls, values: list[str]) -> list[str]:
        return _unique_nonempty(values)

    @model_validator(mode="after")
    def targeted_revision_requires_paragraphs(self) -> WriteOrReviseSectionsAction:
        unknown = set(self.paragraph_numbers).difference(self.section_ids)
        if unknown:
            raise ValueError("paragraph repair targets must belong to selected sections")
        if self.mode == "targeted_revision" and not self.paragraph_numbers:
            raise ValueError("targeted_revision requires paragraph_numbers")
        if any(
            not numbers or len(numbers) != len(set(numbers)) or min(numbers) < 1
            for numbers in self.paragraph_numbers.values()
        ):
            raise ValueError("paragraph_numbers must contain unique positive integers")
        return self


class RunCriticAction(StrictModel):
    kind: Literal["run_critic"] = "run_critic"
    scope: Literal["section", "manuscript", "final_delivery"]
    section_ids: list[str] = Field(default_factory=list, max_length=20)
    include_external_score: bool = False

    @field_validator("section_ids", mode="after")
    @classmethod
    def sections_must_be_unique(cls, values: list[str]) -> list[str]:
        return _unique_nonempty(values)

    @model_validator(mode="after")
    def section_scope_requires_targets(self) -> RunCriticAction:
        if self.scope == "section" and not self.section_ids:
            raise ValueError("section critic requires section_ids")
        if self.scope != "section" and self.section_ids:
            raise ValueError("whole-manuscript critics cannot declare section-only scope")
        return self


class AssembleFinalDeliveryAction(StrictModel):
    kind: Literal["assemble_final_delivery"] = "assemble_final_delivery"
    require_external_score: bool = False
    export_docx: bool = True


class RequestUserInputAction(StrictModel):
    kind: Literal["request_user_input"] = "request_user_input"
    request_type: Literal[
        "requirement_conflict",
        "manual_pdf_download",
        "policy_change",
        "final_confirmation",
    ]
    prompt: str = Field(min_length=1, max_length=1000)
    affected_artifact_ids: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("affected_artifact_ids", mode="after")
    @classmethod
    def artifact_ids_must_be_unique(cls, values: list[str]) -> list[str]:
        return _unique_nonempty(values)


AgentActionPayload = Annotated[
    RefineLiteratureSearchAction
    | AcquireFullTextAction
    | RebuildEvidenceAction
    | ReviseWritingPlanAction
    | WriteOrReviseSectionsAction
    | RunCriticAction
    | AssembleFinalDeliveryAction
    | RequestUserInputAction,
    Field(discriminator="kind"),
]


def action_idempotency_key(
    input_artifact_ids: list[str],
    payload: AgentActionPayload,
) -> str:
    """Derive an executor cache key from validated inputs and action semantics."""

    serialized = json.dumps(
        {
            "input_artifact_ids": sorted(input_artifact_ids),
            "payload": payload.model_dump(mode="json"),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class AgentActionRequest(StrictModel):
    """One planner/controller request; executors may run only validated actions."""

    schema_version: Literal["agent-action.0"] = "agent-action.0"
    action_id: str = Field(pattern=r"^act_[0-9a-f]{16}$")
    requested_by: Literal["planner", "controller"]
    reason_code: str = Field(pattern=r"^[a-z][a-z0-9_]{2,79}$")
    rationale: str = Field(min_length=1, max_length=1200)
    input_artifact_ids: list[str] = Field(default_factory=list, max_length=30)
    payload: AgentActionPayload
    idempotency_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    requires_user_approval: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("input_artifact_ids", mode="after")
    @classmethod
    def artifact_ids_must_be_unique(cls, values: list[str]) -> list[str]:
        return _unique_nonempty(values)

    @model_validator(mode="after")
    def user_actions_require_approval(self) -> AgentActionRequest:
        is_user_request = self.payload.kind == "request_user_input"
        if is_user_request != self.requires_user_approval:
            raise ValueError(
                "request_user_input is exactly the action type that requires user approval"
            )
        expected_key = action_idempotency_key(
            self.input_artifact_ids,
            self.payload,
        )
        if self.idempotency_key != expected_key:
            raise ValueError("idempotency_key does not match action inputs and payload")
        return self


class AgentExecutionError(StrictModel):
    category: Literal[
        "validation",
        "model_output",
        "network",
        "timeout",
        "authorization",
        "evidence_gap",
        "policy_violation",
        "internal",
    ]
    code: str = Field(pattern=r"^[a-z][a-z0-9_]{2,79}$")
    detail: str = Field(min_length=1, max_length=2000)
    retriable: bool
    retry_after_seconds: int | None = Field(default=None, ge=1, le=3600)

    @model_validator(mode="after")
    def retry_delay_requires_retriable_error(self) -> AgentExecutionError:
        if self.retry_after_seconds is not None and not self.retriable:
            raise ValueError("retry_after_seconds requires retriable=true")
        return self


class ToolObservation(StrictModel):
    """What actually happened when an executor attempted one action."""

    schema_version: Literal["agent-observation.0"] = "agent-observation.0"
    observation_id: str = Field(pattern=r"^obs_[0-9a-f]{16}$")
    action_id: str = Field(pattern=r"^act_[0-9a-f]{16}$")
    idempotency_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["succeeded", "partial", "failed", "blocked"]
    attempt: int = Field(default=1, ge=1, le=20)
    output_artifacts: list[ArtifactReference] = Field(default_factory=list, max_length=30)
    metrics: dict[str, int | float | str | bool] = Field(default_factory=dict)
    issue_codes: list[str] = Field(default_factory=list, max_length=30)
    error: AgentExecutionError | None = None
    reused_existing_result: bool = False
    started_at: datetime
    completed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("issue_codes", mode="after")
    @classmethod
    def issues_must_be_unique(cls, values: list[str]) -> list[str]:
        return _unique_nonempty(values)

    @model_validator(mode="after")
    def status_must_match_error(self) -> ToolObservation:
        if self.completed_at < self.started_at:
            raise ValueError("completed_at cannot be before started_at")
        if self.status == "succeeded" and self.error is not None:
            raise ValueError("successful observations cannot contain an error")
        if self.status in {"failed", "blocked"} and self.error is None:
            raise ValueError("failed or blocked observations require an error")
        if self.status == "partial" and not self.output_artifacts and self.error is None:
            raise ValueError("partial observations require an output or an error")
        return self


class CriticFinding(StrictModel):
    finding_id: str = Field(pattern=r"^find_[0-9a-f]{16}$")
    code: str = Field(pattern=r"^[a-z][a-z0-9_]{2,79}$")
    category: Literal[
        "requirement",
        "literature",
        "evidence",
        "argument",
        "citation",
        "structure",
        "language",
        "runtime",
    ]
    severity: Literal["warning", "blocking"]
    responsibility_stage: AgentStage
    artifact_id: str | None = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9_-]{2,79}$",
    )
    location: str | None = Field(default=None, max_length=240)
    detail: str = Field(min_length=1, max_length=1600)
    suggested_action: Literal[
        "report_only",
        "retry",
        "revise",
        "rollback",
        "request_user",
        "stop",
    ]


class QualityDimensionScore(StrictModel):
    dimension: str = Field(pattern=r"^[a-z][a-z0-9_]{2,79}$")
    score: float = Field(ge=0, le=100)
    confidence: float = Field(ge=0, le=1)
    rationale: str = Field(min_length=1, max_length=1000)


class CriticReport(StrictModel):
    """Independent evaluation output; it has no authority to mutate artifacts."""

    schema_version: Literal["agent-critic.0"] = "agent-critic.0"
    report_id: str = Field(pattern=r"^crit_[0-9a-f]{16}$")
    scope: Literal["section", "manuscript", "final_delivery", "runtime"]
    evaluated_artifact_ids: list[str] = Field(min_length=1, max_length=30)
    outcome: Literal["pass", "revise", "rollback", "blocked"]
    findings: list[CriticFinding] = Field(default_factory=list, max_length=50)
    scores: list[QualityDimensionScore] = Field(default_factory=list, max_length=30)
    evaluator: str = Field(min_length=1, max_length=120)
    rubric_version: str | None = Field(default=None, max_length=80)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("evaluated_artifact_ids", mode="after")
    @classmethod
    def artifact_ids_must_be_unique(cls, values: list[str]) -> list[str]:
        return _unique_nonempty(values)

    @model_validator(mode="after")
    def outcome_must_match_findings(self) -> CriticReport:
        blocking = [finding for finding in self.findings if finding.severity == "blocking"]
        if self.outcome == "pass" and blocking:
            raise ValueError("passing critic reports cannot contain blocking findings")
        if self.outcome != "pass" and not self.findings:
            raise ValueError("non-passing critic reports require findings")
        if self.outcome == "revise" and not any(
            finding.suggested_action in {"retry", "revise"}
            for finding in blocking
        ):
            raise ValueError("revise outcome requires an executable blocking revision")
        if self.outcome == "rollback" and not any(
            finding.suggested_action == "rollback" for finding in blocking
        ):
            raise ValueError("rollback outcome requires a blocking rollback finding")
        return self


class ControllerDecision(StrictModel):
    """The only contract allowed to advance, retry, roll back, or finish a run."""

    schema_version: Literal["agent-decision.0"] = "agent-decision.0"
    decision_id: str = Field(pattern=r"^dec_[0-9a-f]{16}$")
    decision_type: Literal[
        "continue",
        "retry",
        "revise",
        "rollback",
        "request_user",
        "finish",
        "stop",
    ]
    current_stage: AgentStage
    target_stage: AgentStage | None = None
    based_on_observation_ids: list[str] = Field(default_factory=list, max_length=30)
    based_on_critic_report_ids: list[str] = Field(default_factory=list, max_length=30)
    reason_code: str = Field(pattern=r"^[a-z][a-z0-9_]{2,79}$")
    explanation: str = Field(min_length=1, max_length=1600)
    next_action: AgentActionRequest | None = None
    stop_reason: str | None = Field(default=None, max_length=1200)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator(
        "based_on_observation_ids",
        "based_on_critic_report_ids",
        mode="after",
    )
    @classmethod
    def evidence_ids_must_be_unique(cls, values: list[str]) -> list[str]:
        return _unique_nonempty(values)

    @model_validator(mode="after")
    def decision_must_be_executable(self) -> ControllerDecision:
        active_decisions = {"continue", "retry", "revise", "rollback", "request_user"}
        if (self.decision_type in active_decisions) != (self.next_action is not None):
            raise ValueError("active controller decisions require exactly one next_action")
        if self.next_action is not None and self.next_action.requested_by != "controller":
            raise ValueError("controller decisions may issue only controller-owned actions")
        if self.decision_type == "request_user":
            if (
                self.next_action is None
                or self.next_action.payload.kind != "request_user_input"
            ):
                raise ValueError("request_user decisions require a request_user_input action")
        if self.decision_type == "rollback":
            if self.target_stage is None:
                raise ValueError("rollback decisions require target_stage")
            stage_order = [
                "requirements",
                "literature",
                "evidence",
                "planning",
                "writing",
                "editing",
                "delivery",
            ]
            if stage_order.index(self.target_stage) >= stage_order.index(self.current_stage):
                raise ValueError("rollback target_stage must precede current_stage")
        elif self.target_stage is not None and self.decision_type not in {"continue", "revise"}:
            raise ValueError("target_stage is not valid for this decision type")
        if self.decision_type == "finish":
            if self.current_stage != "delivery":
                raise ValueError("only delivery can finish an Agent run")
            if not self.based_on_critic_report_ids:
                raise ValueError("finish decisions require at least one critic report")
        if self.decision_type == "stop" and not self.stop_reason:
            raise ValueError("stop decisions require stop_reason")
        if self.decision_type != "stop" and self.stop_reason is not None:
            raise ValueError("stop_reason is valid only for stop decisions")
        return self


class AgentBudget(StrictModel):
    max_model_calls: int = Field(default=100, ge=1, le=10000)
    used_model_calls: int = Field(default=0, ge=0)
    max_total_tokens: int | None = Field(default=None, ge=1)
    used_total_tokens: int = Field(default=0, ge=0)
    max_recovery_rounds: int = Field(default=8, ge=0, le=100)
    used_recovery_rounds: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def usage_cannot_exceed_budget(self) -> AgentBudget:
        if self.used_model_calls > self.max_model_calls:
            raise ValueError("model-call budget has been exceeded")
        if (
            self.max_total_tokens is not None
            and self.used_total_tokens > self.max_total_tokens
        ):
            raise ValueError("token budget has been exceeded")
        if self.used_recovery_rounds > self.max_recovery_rounds:
            raise ValueError("recovery-round budget has been exceeded")
        return self


class AgentState(StrictModel):
    """Small current-state snapshot; full event history is stored separately."""

    schema_version: Literal["agent-state.0"] = "agent-state.0"
    run_id: str = Field(pattern=r"^run_[0-9a-f]{16}$")
    project_id: str = Field(min_length=1, max_length=120)
    lifecycle: Literal[
        "running",
        "waiting_user",
        "completed",
        "failed",
        "stopped",
    ] = "running"
    current_stage: AgentStage
    requirement_policy_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifacts: list[ArtifactReference] = Field(default_factory=list, max_length=200)
    active_action_id: str | None = Field(
        default=None,
        pattern=r"^act_[0-9a-f]{16}$",
    )
    pending_user_action_id: str | None = Field(
        default=None,
        pattern=r"^act_[0-9a-f]{16}$",
    )
    latest_observation_ids: list[str] = Field(default_factory=list, max_length=50)
    latest_critic_report_ids: list[str] = Field(default_factory=list, max_length=50)
    latest_decision_id: str | None = Field(
        default=None,
        pattern=r"^dec_[0-9a-f]{16}$",
    )
    blocker_codes: list[str] = Field(default_factory=list, max_length=50)
    revision_rounds_by_stage: dict[AgentStage, int] = Field(default_factory=dict)
    budget: AgentBudget = Field(default_factory=AgentBudget)
    event_sequence: int = Field(default=0, ge=0)
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator(
        "latest_observation_ids",
        "latest_critic_report_ids",
        "blocker_codes",
        mode="after",
    )
    @classmethod
    def identifiers_must_be_unique(cls, values: list[str]) -> list[str]:
        return _unique_nonempty(values)

    @model_validator(mode="after")
    def lifecycle_must_match_runtime_fields(self) -> AgentState:
        artifact_ids = [artifact.artifact_id for artifact in self.artifacts]
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("AgentState artifact IDs must be unique")
        if self.updated_at < self.started_at:
            raise ValueError("updated_at cannot be before started_at")
        if self.lifecycle == "running":
            if self.pending_user_action_id is not None:
                raise ValueError("running state cannot contain a pending user action")
        elif self.lifecycle == "waiting_user":
            if self.active_action_id is not None or self.pending_user_action_id is None:
                raise ValueError(
                    "waiting_user requires one pending user action and no active action"
                )
        else:
            if self.active_action_id is not None or self.pending_user_action_id is not None:
                raise ValueError("terminal states cannot contain pending actions")
        if self.lifecycle == "completed":
            if self.current_stage != "delivery" or self.blocker_codes:
                raise ValueError(
                    "completed state requires delivery stage and no blocker codes"
                )
        if any(rounds < 0 for rounds in self.revision_rounds_by_stage.values()):
            raise ValueError("revision round counters cannot be negative")
        return self


def agent_state_fingerprint(state: AgentState) -> str:
    serialized = json.dumps(
        state.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class AgentCheckpoint(StrictModel):
    """Recoverable state snapshot written before and after consequential actions."""

    schema_version: Literal["agent-checkpoint.0"] = "agent-checkpoint.0"
    checkpoint_id: str = Field(pattern=r"^ckpt_[0-9a-f]{16}$")
    sequence: int = Field(ge=0)
    reason: Literal[
        "run_started",
        "before_action",
        "after_action",
        "after_decision",
        "user_pause",
        "error",
        "manual",
    ]
    state: AgentState
    state_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_checkpoint_id: str | None = Field(
        default=None,
        pattern=r"^ckpt_[0-9a-f]{16}$",
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def fingerprint_must_match_state(self) -> AgentCheckpoint:
        if self.state_fingerprint != agent_state_fingerprint(self.state):
            raise ValueError("checkpoint fingerprint does not match AgentState")
        if self.sequence == 0 and self.parent_checkpoint_id is not None:
            raise ValueError("initial checkpoint cannot have a parent")
        if self.sequence > 0 and self.parent_checkpoint_id is None:
            raise ValueError("non-initial checkpoint requires parent_checkpoint_id")
        return self
