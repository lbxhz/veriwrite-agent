import json
from datetime import datetime, timezone

import pytest
from pydantic import TypeAdapter

from veriwrite_agent.models.agent_runtime import (
    AgentActionPayload,
    AgentActionRequest,
    AgentCheckpoint,
    AgentState,
    ArtifactReference,
    ControllerDecision,
    CriticReport,
    ToolObservation,
    action_idempotency_key,
    agent_state_fingerprint,
)
from veriwrite_agent.models.requirements import RequirementSpec
from veriwrite_agent.services.agent_artifacts import (
    active_artifact,
    artifact_fingerprint,
    artifact_reference_from_model,
    register_artifact,
)
from veriwrite_agent.services.agent_runtime_store import AgentRuntimeStore

NOW = datetime(2026, 8, 9, tzinfo=timezone.utc)
HASH = "a" * 64


def runtime_state(**updates: object) -> AgentState:
    values: dict[str, object] = {
        "run_id": "run_0123456789abcdef",
        "project_id": "course-paper-1",
        "current_stage": "requirements",
        "requirement_policy_fingerprint": HASH,
        "started_at": NOW,
        "updated_at": NOW,
    }
    values.update(updates)
    return AgentState.model_validate(values)


def action(
    *,
    action_id: str = "act_0123456789abcdef",
    rationale: str = "Evaluate the current manuscript before deciding the next step.",
) -> AgentActionRequest:
    payload = TypeAdapter(AgentActionPayload).validate_python(
        {"kind": "run_critic", "scope": "manuscript"}
    )
    return AgentActionRequest(
        action_id=action_id,
        requested_by="controller",
        reason_code="manuscript_requires_audit",
        rationale=rationale,
        input_artifact_ids=["body_draft_0123456789abcdef"],
        payload=payload,
        idempotency_key=action_idempotency_key(
            ["body_draft_0123456789abcdef"], payload
        ),
        created_at=NOW,
    )


def observation(request: AgentActionRequest) -> ToolObservation:
    return ToolObservation(
        observation_id="obs_0123456789abcdef",
        action_id=request.action_id,
        idempotency_key=request.idempotency_key,
        status="succeeded",
        started_at=NOW,
        completed_at=NOW,
    )


def checkpoint(
    state: AgentState,
    *,
    checkpoint_id: str,
    sequence: int,
    parent_checkpoint_id: str | None = None,
) -> AgentCheckpoint:
    return AgentCheckpoint(
        checkpoint_id=checkpoint_id,
        sequence=sequence,
        reason="run_started" if sequence == 0 else "after_decision",
        state=state,
        state_fingerprint=agent_state_fingerprint(state),
        parent_checkpoint_id=parent_checkpoint_id,
        created_at=NOW,
    )


def test_artifact_adapter_is_stable_and_does_not_copy_the_payload() -> None:
    spec = RequirementSpec(
        document_type="research_direction_literature_review",
        topic="大气遥感",
        topic_source="explicit",
        output_language="Chinese",
    )

    first = artifact_reference_from_model(
        spec,
        storage_key="mvp/requirement_spec.json",
    )
    second = artifact_reference_from_model(
        spec,
        storage_key="mvp/requirement_spec.json",
    )

    assert first.artifact_id == second.artifact_id
    assert first.fingerprint == artifact_fingerprint(spec)
    assert first.kind == "requirement_spec"
    assert first.status == "draft"
    assert "大气遥感" not in first.model_dump_json()


def test_register_artifact_is_idempotent_and_supersedes_only_same_kind() -> None:
    first = artifact_reference_from_model(
        RequirementSpec(document_type="review", topic="大气遥感"),
        storage_key="requirements/v1.json",
    )
    second = artifact_reference_from_model(
        RequirementSpec(document_type="review", topic="大气遥感反演"),
        storage_key="requirements/v2.json",
    )
    state = register_artifact(runtime_state(), first)
    unchanged = register_artifact(state, first)
    updated = register_artifact(unchanged, second)

    assert unchanged is state
    assert updated.event_sequence == 2
    assert updated.artifacts[0].status == "superseded"
    assert active_artifact(updated, "requirement_spec") == second


def test_register_artifact_compacts_old_superseded_history_at_state_limit() -> None:
    history = [
        ArtifactReference(
            artifact_id=f"writing_project_history_{index:03d}",
            kind="writing_project",
            schema_version="0.4.0",
            fingerprint=f"{index:064x}",
            storage_key=f"history/{index}.json",
            status="superseded",
            created_at=NOW,
        )
        for index in range(199)
    ]
    active = ArtifactReference(
        artifact_id="writing_project_active_000",
        kind="writing_project",
        schema_version="0.4.0",
        fingerprint="f" * 64,
        storage_key="active.json",
        status="draft",
        created_at=NOW,
    )
    state = runtime_state(artifacts=[*history, active])
    replacement = ArtifactReference(
        artifact_id="writing_project_active_001",
        kind="writing_project",
        schema_version="0.4.0",
        fingerprint="e" * 64,
        storage_key="active-v2.json",
        status="draft",
        created_at=NOW,
    )

    updated = register_artifact(state, replacement)

    assert len(updated.artifacts) == 200
    assert active_artifact(updated, "writing_project") == replacement
    assert updated.artifacts[0].artifact_id == "writing_project_history_001"


def test_successful_observation_is_reusable_by_idempotency_key(tmp_path) -> None:
    store = AgentRuntimeStore(tmp_path)
    request = action()
    result = observation(request)

    store.save_action(request)
    store.save_observation(result)

    assert store.load_action(request.action_id) == request
    assert store.load_success_by_idempotency(request.idempotency_key) == result
    store.save_action(request)
    store.save_observation(result)


def test_runtime_store_rejects_conflicting_events_and_mismatched_observations(
    tmp_path,
) -> None:
    store = AgentRuntimeStore(tmp_path)
    request = action()
    store.save_action(request)

    with pytest.raises(ValueError, match="different content"):
        store.save_action(action(rationale="A different reason for the same action ID."))

    mismatched = observation(request).model_copy(update={"idempotency_key": "b" * 64})
    with pytest.raises(ValueError, match="idempotency_key"):
        store.save_observation(mismatched)


def test_checkpoint_chain_recovers_even_if_pointer_or_extra_file_is_corrupt(
    tmp_path,
) -> None:
    store = AgentRuntimeStore(tmp_path)
    initial_state = runtime_state()
    initial = checkpoint(
        initial_state,
        checkpoint_id="ckpt_0123456789abcdef",
        sequence=0,
    )
    next_state = initial_state.model_copy(
        update={"event_sequence": 1, "updated_at": NOW}
    )
    next_checkpoint = checkpoint(
        next_state,
        checkpoint_id="ckpt_fedcba9876543210",
        sequence=1,
        parent_checkpoint_id=initial.checkpoint_id,
    )

    store.save_checkpoint(initial)
    store.save_checkpoint(next_checkpoint)
    store.save_checkpoint(next_checkpoint)
    (tmp_path / "latest_checkpoint.json").write_text("{broken", encoding="utf-8")
    (tmp_path / "checkpoints" / "00000002_ckpt_aaaaaaaaaaaaaaaa.json").write_text(
        "{broken", encoding="utf-8"
    )

    assert store.load_latest_checkpoint() == next_checkpoint
    assert store.load_state() == next_state


def test_checkpoint_pointer_selects_a_valid_branch_after_concurrent_fork(
    tmp_path,
) -> None:
    store = AgentRuntimeStore(tmp_path)
    state = runtime_state()
    initial = checkpoint(
        state,
        checkpoint_id="ckpt_0123456789abcdef",
        sequence=0,
    )
    selected = checkpoint(
        state.model_copy(update={"event_sequence": 1}),
        checkpoint_id="ckpt_1111111111111111",
        sequence=1,
        parent_checkpoint_id=initial.checkpoint_id,
    )
    sibling = checkpoint(
        state.model_copy(update={"event_sequence": 2}),
        checkpoint_id="ckpt_2222222222222222",
        sequence=1,
        parent_checkpoint_id=initial.checkpoint_id,
    )
    store.save_checkpoint(initial)
    store.save_checkpoint(selected)
    sibling_path = (
        tmp_path / "checkpoints" / "00000001_ckpt_2222222222222222.json"
    )
    sibling_path.write_text(sibling.model_dump_json(indent=2), encoding="utf-8")
    (tmp_path / "latest_checkpoint.json").write_text(
        json.dumps(
            {
                "checkpoint_id": selected.checkpoint_id,
                "sequence": selected.sequence,
                "file": "00000001_ckpt_1111111111111111.json",
            }
        ),
        encoding="utf-8",
    )

    assert store.load_latest_checkpoint() == selected
    assert store.load_state() == selected.state


def test_checkpoint_store_uses_unique_longest_valid_branch_when_pointer_is_stale(
    tmp_path,
) -> None:
    store = AgentRuntimeStore(tmp_path)
    state = runtime_state()
    root = checkpoint(
        state,
        checkpoint_id="ckpt_0123456789abcdef",
        sequence=0,
    )
    short_branch = checkpoint(
        state.model_copy(update={"event_sequence": 1}),
        checkpoint_id="ckpt_1111111111111111",
        sequence=1,
        parent_checkpoint_id=root.checkpoint_id,
    )
    long_branch = checkpoint(
        state.model_copy(update={"event_sequence": 2}),
        checkpoint_id="ckpt_2222222222222222",
        sequence=1,
        parent_checkpoint_id=root.checkpoint_id,
    )
    tip = checkpoint(
        state.model_copy(update={"event_sequence": 3}),
        checkpoint_id="ckpt_3333333333333333",
        sequence=2,
        parent_checkpoint_id=long_branch.checkpoint_id,
    )
    store.save_checkpoint(root)
    for item in (short_branch, long_branch, tip):
        path = (
            tmp_path
            / "checkpoints"
            / f"{item.sequence:08d}_{item.checkpoint_id}.json"
        )
        path.write_text(item.model_dump_json(indent=2), encoding="utf-8")
    (tmp_path / "latest_checkpoint.json").write_text(
        json.dumps(
            {
                "checkpoint_id": "ckpt_ffffffffffffffff",
                "sequence": 3,
                "file": "00000003_ckpt_ffffffffffffffff.json",
            }
        ),
        encoding="utf-8",
    )

    assert store.load_latest_checkpoint() == tip


def test_checkpoint_store_rejects_skipped_or_wrong_parent(tmp_path) -> None:
    store = AgentRuntimeStore(tmp_path)
    state = runtime_state()
    initial = checkpoint(
        state,
        checkpoint_id="ckpt_0123456789abcdef",
        sequence=0,
    )
    store.save_checkpoint(initial)
    invalid = checkpoint(
        state,
        checkpoint_id="ckpt_fedcba9876543210",
        sequence=2,
        parent_checkpoint_id=initial.checkpoint_id,
    )

    with pytest.raises(ValueError, match="sequence"):
        store.save_checkpoint(invalid)


def test_controller_decision_requires_stored_evidence_and_persists_next_action(
    tmp_path,
) -> None:
    store = AgentRuntimeStore(tmp_path)
    request = action()
    result = observation(request)
    next_request = action(action_id="act_fedcba9876543210")
    decision = ControllerDecision(
        decision_id="dec_0123456789abcdef",
        decision_type="continue",
        current_stage="editing",
        target_stage="delivery",
        based_on_observation_ids=[result.observation_id],
        reason_code="critic_execution_completed",
        explanation="The controller can now perform the next bounded audit action.",
        next_action=next_request,
        created_at=NOW,
    )

    with pytest.raises(ValueError, match="unknown observations"):
        store.save_decision(decision)

    store.save_action(request)
    store.save_observation(result)
    store.save_decision(decision)

    assert store.load_decision(decision.decision_id) == decision
    assert store.load_action(next_request.action_id) == next_request


def test_finish_decision_requires_a_stored_critic_report(tmp_path) -> None:
    store = AgentRuntimeStore(tmp_path)
    report = CriticReport(
        report_id="crit_0123456789abcdef",
        scope="final_delivery",
        evaluated_artifact_ids=["final_package_0123456789abcdef"],
        outcome="pass",
        evaluator="independent-reviewer",
        created_at=NOW,
    )
    decision = ControllerDecision(
        decision_id="dec_0123456789abcdef",
        decision_type="finish",
        current_stage="delivery",
        based_on_critic_report_ids=[report.report_id],
        reason_code="all_release_gates_passed",
        explanation="The final delivery critic found no blocking issue.",
        created_at=NOW,
    )

    with pytest.raises(ValueError, match="unknown critic reports"):
        store.save_decision(decision)
    store.save_critic_report(report)
    store.save_decision(decision)

    assert store.load_critic_report(report.report_id) == report
    assert store.load_decision(decision.decision_id) == decision
