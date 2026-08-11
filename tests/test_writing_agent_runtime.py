from datetime import datetime, timezone
from types import SimpleNamespace

from veriwrite_agent.models.agent_runtime import (
    AgentState,
    ArtifactReference,
)
from veriwrite_agent.models.writing_plan import (
    GroundedWritingPlan,
    WritingParagraphPlan,
    WritingSectionPlan,
)
from veriwrite_agent.services.agent_runtime_store import AgentRuntimeStore
from veriwrite_agent.services.writing_agent_runtime import (
    WritingAgentContext,
    WritingAgentRuntimeService,
)

NOW = datetime(2026, 8, 9, tzinfo=timezone.utc)


def reference(kind: str, artifact_id: str, fingerprint: str) -> ArtifactReference:
    return ArtifactReference.model_validate(
        {
            "artifact_id": artifact_id,
            "kind": kind,
            "schema_version": "test.1",
            "fingerprint": fingerprint,
            "storage_key": f"snapshot.state.{artifact_id}",
            "status": "ready",
            "created_at": NOW,
        }
    )


def plan() -> GroundedWritingPlan:
    paragraphs = [
        WritingParagraphPlan(
            paragraph_id=f"methods_p0{number}",
            section_id="methods",
            paragraph_number=number,
            role="synthesis",
            purpose="综合证据。",
            claim_focus="不同方法具有不同适用条件。",
            central_question="方法边界是什么？",
            argument_move="synthesize_consensus",
            target_words=300,
            evidence_card_ids=["ev_1"],
            source_dois=["10.1000/core"],
        )
        for number in (1, 2)
    ]
    return GroundedWritingPlan(
        status="confirmed",
        topic="大气遥感",
        output_language="Chinese",
        plan_fingerprint="b" * 64,
        sections=[
            WritingSectionPlan(
                section_id="methods",
                title="反演方法",
                purpose="比较反演方法。",
                target_words=600,
                counting_policy="chinese_chars_and_english_words",
                paragraphs=paragraphs,
            )
        ],
        confirmed_by="student",
        confirmed_at=NOW,
    )


def context() -> WritingAgentContext:
    policy = reference(
        "requirement_policy", "requirement_policy_0123456789abcdef", "1" * 64
    )
    handoff = reference(
        "writing_handoff", "writing_handoff_0123456789abcdef", "2" * 64
    )
    plan_reference = reference(
        "writing_plan", "writing_plan_0123456789abcdef", "3" * 64
    )
    project = reference(
        "writing_project", "writing_project_0123456789abcdef", "4" * 64
    )
    state = AgentState(
        run_id="run_0123456789abcdef",
        project_id="course-paper-1",
        current_stage="writing",
        requirement_policy_fingerprint="1" * 64,
        artifacts=[policy, handoff, plan_reference, project],
        started_at=NOW,
        updated_at=NOW,
    )
    return WritingAgentContext(
        state=state,
        policy_reference=policy,
        handoff_reference=handoff,
        plan_reference=plan_reference,
        project_reference=project,
    )


def test_runtime_records_action_observation_critic_decision_and_checkpoint(
    tmp_path,
    monkeypatch,
) -> None:
    store = AgentRuntimeStore(tmp_path)
    runtime = WritingAgentRuntimeService(store)
    active = context()
    prepared = runtime.prepare_section_action(active, section_ids=["methods"])
    output_reference = reference(
        "writing_project", "writing_project_fedcba9876543210", "5" * 64
    )
    monkeypatch.setattr(
        "veriwrite_agent.services.writing_agent_runtime.artifact_reference_from_model",
        lambda *_args, **_kwargs: output_reference,
    )
    result = SimpleNamespace(
        project=SimpleNamespace(),
        events=(SimpleNamespace(stage="confirmed"),),
        stopped_section_id=None,
        stop_code=None,
        completed=True,
    )

    recorded = runtime.record_section_result(prepared, result, plan())

    assert store.load_action(prepared.action.action_id) == prepared.action
    assert store.load_observation(recorded.observation.observation_id) == (
        recorded.observation
    )
    assert store.load_critic_report(recorded.assessment.critic.report_id) == (
        recorded.assessment.critic
    )
    assert store.load_decision(recorded.assessment.decision.decision_id) == (
        recorded.assessment.decision
    )
    assert recorded.state.active_action_id is None
    assert recorded.state.current_stage == "editing"
    assert recorded.state.latest_decision_id == recorded.assessment.decision.decision_id
    assert store.load_latest_checkpoint().state == recorded.state


def test_prepare_action_reuses_a_successful_idempotent_result(
    tmp_path,
) -> None:
    store = AgentRuntimeStore(tmp_path)
    runtime = WritingAgentRuntimeService(store)
    active = context()
    prepared = runtime.prepare_section_action(active, section_ids=["methods"])
    from veriwrite_agent.models.agent_runtime import ToolObservation

    observation = ToolObservation(
        observation_id="obs_0123456789abcdef",
        action_id=prepared.action.action_id,
        idempotency_key=prepared.action.idempotency_key,
        status="succeeded",
        started_at=prepared.action.created_at,
        completed_at=prepared.action.created_at,
    )
    store.save_observation(observation)
    idle_state = prepared.state.model_copy(update={"active_action_id": None})
    resumed = WritingAgentContext(
        state=idle_state,
        policy_reference=active.policy_reference,
        handoff_reference=active.handoff_reference,
        plan_reference=active.plan_reference,
        project_reference=active.project_reference,
    )

    duplicate = runtime.prepare_section_action(resumed, section_ids=["methods"])

    assert duplicate.cached_observation == observation
    assert duplicate.action.idempotency_key == prepared.action.idempotency_key
