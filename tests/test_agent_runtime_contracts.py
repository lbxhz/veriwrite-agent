from datetime import datetime, timezone

import pytest
from pydantic import TypeAdapter, ValidationError

from veriwrite_agent.models.agent_runtime import (
    AgentActionPayload,
    AgentActionRequest,
    AgentBudget,
    AgentCheckpoint,
    AgentExecutionError,
    AgentState,
    ArtifactReference,
    ControllerDecision,
    CriticFinding,
    CriticReport,
    ToolObservation,
    action_idempotency_key,
    agent_state_fingerprint,
)

NOW = datetime(2026, 8, 9, tzinfo=timezone.utc)
HASH = "a" * 64


def artifact(
    artifact_id: str = "writing_project_v1",
    *,
    status: str = "confirmed",
) -> ArtifactReference:
    return ArtifactReference(
        artifact_id=artifact_id,
        kind="writing_project",
        schema_version="0.4.0",
        fingerprint=HASH,
        storage_key="v04_writing_project_json",
        status=status,
        created_at=NOW,
    )


def controller_action(
    *,
    payload: dict | None = None,
    requires_user_approval: bool = False,
) -> AgentActionRequest:
    payload_model = TypeAdapter(AgentActionPayload).validate_python(
        payload
        or {
            "kind": "write_or_revise_sections",
            "section_ids": ["theme1"],
            "paragraph_numbers": {"theme1": [2]},
            "mode": "targeted_revision",
        }
    )
    input_artifact_ids = ["writing_project_v1"]
    return AgentActionRequest.model_validate(
        {
            "action_id": "act_0123456789abcdef",
            "requested_by": "controller",
            "reason_code": "critic_requires_revision",
            "rationale": "Only the affected paragraph needs another evidence-bound pass.",
            "input_artifact_ids": input_artifact_ids,
            "payload": payload_model,
            "idempotency_key": action_idempotency_key(
                input_artifact_ids,
                payload_model,
            ),
            "requires_user_approval": requires_user_approval,
            "created_at": NOW,
        }
    )


def blocking_finding(
    *,
    suggested_action: str = "revise",
    responsibility_stage: str = "writing",
) -> CriticFinding:
    return CriticFinding(
        finding_id="find_0123456789abcdef",
        code="false_self_attribution",
        category="argument",
        severity="blocking",
        responsibility_stage=responsibility_stage,
        artifact_id="writing_project_v1",
        location="theme1:2",
        detail="The review claims a cited method as the current paper's work.",
        suggested_action=suggested_action,
    )


def state(**updates) -> AgentState:
    values = {
        "run_id": "run_0123456789abcdef",
        "project_id": "course-paper-1",
        "lifecycle": "running",
        "current_stage": "writing",
        "requirement_policy_fingerprint": HASH,
        "artifacts": [artifact()],
        "budget": AgentBudget(max_model_calls=20, used_model_calls=3),
        "started_at": NOW,
        "updated_at": NOW,
    }
    values.update(updates)
    return AgentState.model_validate(values)


def test_action_payload_is_discriminated_and_strict() -> None:
    request = controller_action()

    assert request.payload.kind == "write_or_revise_sections"
    assert request.payload.paragraph_numbers == {"theme1": [2]}

    with pytest.raises(ValidationError, match="paragraph_numbers"):
        controller_action(
            payload={
                "kind": "write_or_revise_sections",
                "section_ids": ["theme1"],
                "mode": "targeted_revision",
            }
        )

    with pytest.raises(ValidationError, match="Extra inputs"):
        controller_action(
            payload={
                "kind": "assemble_final_delivery",
                "unexpected_permission": True,
            }
        )


def test_only_user_request_actions_require_user_approval() -> None:
    user_payload = {
        "kind": "request_user_input",
        "request_type": "manual_pdf_download",
        "prompt": "Please download the one unavailable core PDF.",
        "affected_artifact_ids": ["writing_project_v1"],
    }

    request = controller_action(
        payload=user_payload,
        requires_user_approval=True,
    )

    assert request.payload.kind == "request_user_input"
    with pytest.raises(ValidationError, match="requires user approval"):
        controller_action(payload=user_payload)
    with pytest.raises(ValidationError, match="requires user approval"):
        controller_action(requires_user_approval=True)


def test_action_idempotency_key_is_owned_by_deterministic_code() -> None:
    request = controller_action()
    payload = request.model_dump()
    payload["idempotency_key"] = "b" * 64

    with pytest.raises(ValidationError, match="idempotency_key"):
        AgentActionRequest.model_validate(payload)


def test_tool_observation_status_cannot_hide_execution_failure() -> None:
    error = AgentExecutionError(
        category="network",
        code="crossref_timeout",
        detail="Crossref did not answer before the configured timeout.",
        retriable=True,
        retry_after_seconds=10,
    )
    failed = ToolObservation(
        observation_id="obs_0123456789abcdef",
        action_id="act_0123456789abcdef",
        idempotency_key=HASH,
        status="failed",
        error=error,
        started_at=NOW,
        completed_at=NOW,
    )

    assert failed.error is not None and failed.error.retriable
    with pytest.raises(ValidationError, match="require an error"):
        failed.model_copy(update={"error": None}).__class__.model_validate(
            failed.model_copy(update={"error": None}).model_dump()
        )
    with pytest.raises(ValidationError, match="cannot contain an error"):
        ToolObservation.model_validate(
            failed.model_copy(update={"status": "succeeded"}).model_dump()
        )


def test_critic_outcome_requires_matching_findings() -> None:
    with pytest.raises(ValidationError, match="passing critic"):
        CriticReport(
            report_id="crit_0123456789abcdef",
            scope="manuscript",
            evaluated_artifact_ids=["writing_project_v1"],
            outcome="pass",
            findings=[blocking_finding()],
            evaluator="independent-reviewer",
            created_at=NOW,
        )

    report = CriticReport(
        report_id="crit_0123456789abcdef",
        scope="manuscript",
        evaluated_artifact_ids=["writing_project_v1"],
        outcome="rollback",
        findings=[
            blocking_finding(
                suggested_action="rollback",
                responsibility_stage="evidence",
            )
        ],
        evaluator="independent-reviewer",
        created_at=NOW,
    )

    assert report.outcome == "rollback"


def test_controller_rollback_must_target_an_earlier_stage() -> None:
    decision = ControllerDecision(
        decision_id="dec_0123456789abcdef",
        decision_type="rollback",
        current_stage="writing",
        target_stage="evidence",
        based_on_critic_report_ids=["crit_0123456789abcdef"],
        reason_code="detailed_claim_lacks_full_text",
        explanation="Evidence must be rebuilt before the affected section is revised.",
        next_action=controller_action(
            payload={
                "kind": "rebuild_evidence",
                "affected_section_ids": ["theme1"],
                "source_dois": ["10.1000/example"],
            }
        ),
        created_at=NOW,
    )

    assert decision.target_stage == "evidence"
    with pytest.raises(ValidationError, match="must precede"):
        ControllerDecision.model_validate(
            decision.model_copy(update={"target_stage": "delivery"}).model_dump()
        )


def test_finish_decision_requires_delivery_critic_evidence() -> None:
    finish = ControllerDecision(
        decision_id="dec_0123456789abcdef",
        decision_type="finish",
        current_stage="delivery",
        based_on_critic_report_ids=["crit_0123456789abcdef"],
        reason_code="all_release_gates_passed",
        explanation="The final package has no blocking audit or critic findings.",
        created_at=NOW,
    )

    assert finish.next_action is None
    with pytest.raises(ValidationError, match="critic report"):
        ControllerDecision.model_validate(
            finish.model_copy(update={"based_on_critic_report_ids": []}).model_dump()
        )


def test_agent_state_separates_current_state_from_pending_user_work() -> None:
    waiting = state(
        lifecycle="waiting_user",
        active_action_id=None,
        pending_user_action_id="act_0123456789abcdef",
    )

    assert waiting.lifecycle == "waiting_user"
    with pytest.raises(ValidationError, match="pending user action"):
        AgentState.model_validate(
            waiting.model_copy(update={"pending_user_action_id": None}).model_dump()
        )
    with pytest.raises(ValidationError, match="delivery stage"):
        state(lifecycle="completed", current_stage="writing")
    with pytest.raises(ValidationError, match="no blocker"):
        state(
            lifecycle="completed",
            current_stage="delivery",
            blocker_codes=["citation_mismatch"],
        )


def test_budget_contract_prevents_unbounded_recovery() -> None:
    with pytest.raises(ValidationError, match="model-call budget"):
        AgentBudget(max_model_calls=3, used_model_calls=4)
    with pytest.raises(ValidationError, match="recovery-round budget"):
        AgentBudget(max_recovery_rounds=2, used_recovery_rounds=3)


def test_checkpoint_detects_state_tampering_and_requires_parent_chain() -> None:
    active = state()
    initial = AgentCheckpoint(
        checkpoint_id="ckpt_0123456789abcdef",
        sequence=0,
        reason="run_started",
        state=active,
        state_fingerprint=agent_state_fingerprint(active),
        created_at=NOW,
    )

    assert initial.state_fingerprint == agent_state_fingerprint(active)
    with pytest.raises(ValidationError, match="fingerprint"):
        AgentCheckpoint.model_validate(
            initial.model_copy(update={"state_fingerprint": "b" * 64}).model_dump()
        )
    with pytest.raises(ValidationError, match="requires parent"):
        AgentCheckpoint(
            checkpoint_id="ckpt_fedcba9876543210",
            sequence=1,
            reason="after_action",
            state=active,
            state_fingerprint=agent_state_fingerprint(active),
            created_at=NOW,
        )


def test_agent_state_rejects_unknown_runtime_fields() -> None:
    payload = state().model_dump()
    payload["planner_private_thoughts"] = "must not enter persisted state"

    with pytest.raises(ValidationError, match="Extra inputs"):
        AgentState.model_validate(payload)
