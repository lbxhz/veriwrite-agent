"""Persisted runtime bridge between the V0.4 executor and Agent controller."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

from veriwrite_agent.models.agent_runtime import (
    AgentActionRequest,
    AgentCheckpoint,
    AgentExecutionError,
    AgentState,
    ArtifactReference,
    ToolObservation,
    WriteOrReviseSectionsAction,
    action_idempotency_key,
    agent_state_fingerprint,
)
from veriwrite_agent.models.executable_policy import ExecutableRequirementPolicy
from veriwrite_agent.models.writing import V04WritingProject
from veriwrite_agent.models.writing_handoff import V04WritingHandoff
from veriwrite_agent.models.writing_plan import GroundedWritingPlan
from veriwrite_agent.services.agent_artifacts import (
    active_artifact,
    artifact_reference_from_model,
    register_artifact,
)
from veriwrite_agent.services.agent_runtime_store import AgentRuntimeStore
from veriwrite_agent.services.writing_agent_controller import (
    WritingAgentAssessment,
    WritingAgentController,
)
from veriwrite_agent.services.writing_autopilot import ContinuousWritingResult


@dataclass(frozen=True)
class WritingAgentContext:
    state: AgentState
    policy_reference: ArtifactReference
    handoff_reference: ArtifactReference
    plan_reference: ArtifactReference
    project_reference: ArtifactReference


@dataclass(frozen=True)
class PreparedWritingAction:
    state: AgentState
    action: AgentActionRequest
    cached_observation: ToolObservation | None = None


@dataclass(frozen=True)
class RecordedWritingTransition:
    state: AgentState
    observation: ToolObservation
    assessment: WritingAgentAssessment
    project_reference: ArtifactReference


class WritingAgentRuntimeService:
    """Keep execution records out of Streamlit and make refreshes recoverable."""

    def __init__(
        self,
        store: AgentRuntimeStore,
        *,
        controller: WritingAgentController | None = None,
    ) -> None:
        self._store = store
        self._controller = controller or WritingAgentController()

    def initialize(
        self,
        *,
        run_id: str,
        project_id: str,
        policy: ExecutableRequirementPolicy,
        handoff: V04WritingHandoff,
        plan: GroundedWritingPlan,
        project: V04WritingProject,
    ) -> WritingAgentContext:
        references = (
            artifact_reference_from_model(
                policy,
                storage_key="mvp_snapshot.state.recovered_executable_policy_json",
            ),
            artifact_reference_from_model(
                handoff,
                storage_key="mvp_snapshot.state.v03_writing_handoff_json",
            ),
            artifact_reference_from_model(
                plan,
                storage_key="mvp_snapshot.state.v04_writing_plan_json",
            ),
            artifact_reference_from_model(
                project,
                storage_key="mvp_snapshot.state.v04_writing_project_json",
            ),
        )
        state = self._store.load_state()
        if state is None:
            state = AgentState(
                run_id=run_id,
                project_id=project_id,
                current_stage="writing",
                requirement_policy_fingerprint=policy.requirement_fingerprint,
            )
            for reference in references:
                state = register_artifact(state, reference)
            self._checkpoint(state, reason="run_started")
        else:
            if state.run_id != run_id:
                raise ValueError("Agent runtime directory belongs to another run")
            if state.project_id != project_id:
                raise ValueError("Agent runtime project_id does not match the active project")
            if state.requirement_policy_fingerprint != policy.requirement_fingerprint:
                raise ValueError("confirmed requirement policy changed; start a new Agent run")
            previous_fingerprint = agent_state_fingerprint(state)
            for reference in references:
                state = register_artifact(state, reference)
            if agent_state_fingerprint(state) != previous_fingerprint:
                self._checkpoint(state, reason="manual")

        return WritingAgentContext(
            state=state,
            policy_reference=references[0],
            handoff_reference=references[1],
            plan_reference=references[2],
            project_reference=references[3],
        )

    def prepare_section_action(
        self,
        context: WritingAgentContext,
        *,
        section_ids: list[str],
    ) -> PreparedWritingAction:
        if not section_ids:
            raise ValueError("section writing action requires at least one section")
        state = context.state
        if state.active_action_id is not None:
            existing = self._store.load_action(state.active_action_id)
            if existing is None:
                raise ValueError("active Agent action is missing from the event store")
            cached = self._store.load_success_by_idempotency(existing.idempotency_key)
            return PreparedWritingAction(
                state=state,
                action=existing,
                cached_observation=cached,
            )

        payload = WriteOrReviseSectionsAction(
            section_ids=section_ids,
            mode="draft",
        )
        inputs = [
            context.plan_reference.artifact_id,
            context.project_reference.artifact_id,
        ]
        action = AgentActionRequest(
            action_id=self._id("act"),
            requested_by="planner",
            reason_code="execute_pending_sections",
            rationale=(
                "Generate only pending chapters; the executor may preserve passed "
                "paragraphs and target previously identified failures."
            ),
            input_artifact_ids=inputs,
            payload=payload,
            idempotency_key=action_idempotency_key(inputs, payload),
        )
        cached = self._store.load_success_by_idempotency(action.idempotency_key)
        if cached is not None:
            original_action = self._store.load_action(cached.action_id)
            if original_action is None:
                raise ValueError("cached observation lost its original action")
            action = original_action
        if cached is None:
            self._store.save_action(action)
            state = self._validated_state(
                state,
                active_action_id=action.action_id,
                current_stage="writing",
                lifecycle="running",
                pending_user_action_id=None,
                event_sequence=state.event_sequence + 1,
                updated_at=datetime.now(timezone.utc),
            )
            self._checkpoint(state, reason="before_action")
        return PreparedWritingAction(
            state=state,
            action=action,
            cached_observation=cached,
        )

    def record_section_result(
        self,
        prepared: PreparedWritingAction,
        result: ContinuousWritingResult,
        plan: GroundedWritingPlan,
    ) -> RecordedWritingTransition:
        project_reference = artifact_reference_from_model(
            result.project,
            storage_key="mvp_snapshot.state.v04_writing_project_json",
        )
        issue_codes = self._result_issue_codes(result)
        observation = prepared.cached_observation or self._observation(
            prepared.action,
            result,
            project_reference=project_reference,
            issue_codes=issue_codes,
        )
        if prepared.cached_observation is None:
            self._store.save_observation(observation)

        state = register_artifact(prepared.state, project_reference)
        plan_reference = active_artifact(state, "writing_plan")
        if plan_reference is None:
            raise ValueError("AgentState lost the active writing plan")
        assessment = self._controller.assess_section_run(
            result,
            plan,
            plan_reference=plan_reference,
            project_reference=project_reference,
        )
        self._store.save_critic_report(assessment.critic)
        self._store.save_decision(assessment.decision)

        blocking_codes = [
            finding.code
            for finding in assessment.critic.findings
            if finding.severity == "blocking"
        ]
        lifecycle = "running"
        pending_user_action_id = None
        if assessment.decision.decision_type == "request_user":
            lifecycle = "waiting_user"
            pending_user_action_id = assessment.decision.next_action.action_id
        elif assessment.decision.decision_type == "finish":
            lifecycle = "completed"
        elif assessment.decision.decision_type == "stop":
            lifecycle = "stopped"
        target_stage = assessment.decision.target_stage or "writing"
        revision_rounds = dict(state.revision_rounds_by_stage)
        if assessment.decision.decision_type in {"retry", "revise", "rollback"}:
            revision_rounds[target_stage] = revision_rounds.get(target_stage, 0) + 1
        state = self._validated_state(
            state,
            lifecycle=lifecycle,
            current_stage=target_stage,
            active_action_id=None,
            pending_user_action_id=pending_user_action_id,
            latest_observation_ids=[
                *state.latest_observation_ids[-49:],
                observation.observation_id,
            ],
            latest_critic_report_ids=[
                *state.latest_critic_report_ids[-49:],
                assessment.critic.report_id,
            ],
            latest_decision_id=assessment.decision.decision_id,
            blocker_codes=blocking_codes,
            revision_rounds_by_stage=revision_rounds,
            event_sequence=state.event_sequence + 1,
            updated_at=datetime.now(timezone.utc),
        )
        self._checkpoint(state, reason="after_decision")
        return RecordedWritingTransition(
            state=state,
            observation=observation,
            assessment=assessment,
            project_reference=project_reference,
        )

    def record_assessment(
        self,
        state: AgentState,
        assessment: WritingAgentAssessment,
        *,
        output_reference: ArtifactReference | None = None,
    ) -> AgentState:
        """Persist a V0.5 critic/controller pair and advance only through its decision."""

        if output_reference is not None:
            state = register_artifact(state, output_reference)
        self._store.save_critic_report(assessment.critic)
        self._store.save_decision(assessment.decision)
        blocking_codes = [
            finding.code
            for finding in assessment.critic.findings
            if finding.severity == "blocking"
        ]
        lifecycle = "running"
        pending_user_action_id = None
        if assessment.decision.decision_type == "request_user":
            lifecycle = "waiting_user"
            pending_user_action_id = assessment.decision.next_action.action_id
        elif assessment.decision.decision_type == "finish":
            lifecycle = "completed"
        elif assessment.decision.decision_type == "stop":
            lifecycle = "stopped"
        target_stage = assessment.decision.target_stage or state.current_stage
        revision_rounds = dict(state.revision_rounds_by_stage)
        if assessment.decision.decision_type in {"retry", "revise", "rollback"}:
            revision_rounds[target_stage] = revision_rounds.get(target_stage, 0) + 1
        updated = self._validated_state(
            state,
            lifecycle=lifecycle,
            current_stage=target_stage,
            active_action_id=None,
            pending_user_action_id=pending_user_action_id,
            latest_critic_report_ids=[
                *state.latest_critic_report_ids[-49:],
                assessment.critic.report_id,
            ],
            latest_decision_id=assessment.decision.decision_id,
            blocker_codes=blocking_codes,
            revision_rounds_by_stage=revision_rounds,
            event_sequence=state.event_sequence + 1,
            updated_at=datetime.now(timezone.utc),
        )
        self._checkpoint(updated, reason="after_decision")
        return updated

    def _observation(
        self,
        action: AgentActionRequest,
        result: ContinuousWritingResult,
        *,
        project_reference: ArtifactReference,
        issue_codes: list[str],
    ) -> ToolObservation:
        now = datetime.now(timezone.utc)
        status = "succeeded"
        error = None
        if result.stopped_section_id is not None:
            if result.stop_code == "policy_blocked":
                status = "blocked"
                error = AgentExecutionError(
                    category="policy_violation",
                    code="ai_writing_policy_blocked",
                    detail=result.stop_reason or "AI writing is blocked by policy.",
                    retriable=False,
                )
            elif result.stop_code in {"generation_failed", "review_failed"}:
                status = "failed"
                error = AgentExecutionError(
                    category="model_output",
                    code=result.stop_code,
                    detail=result.stop_reason or "A model stage failed.",
                    retriable=True,
                )
            else:
                status = "partial"
        return ToolObservation(
            observation_id=self._id("obs"),
            action_id=action.action_id,
            idempotency_key=action.idempotency_key,
            status=status,
            output_artifacts=[project_reference],
            metrics={
                "event_count": len(result.events),
                "body_complete": result.completed,
            },
            issue_codes=issue_codes,
            error=error,
            started_at=action.created_at,
            completed_at=now,
        )

    @staticmethod
    def _result_issue_codes(result: ContinuousWritingResult) -> list[str]:
        codes: list[str] = []
        if result.stop_code:
            codes.append(result.stop_code)
        if result.stopped_section_id is None:
            return codes
        state = next(
            section
            for section in result.project.sections
            if section.section_id == result.stopped_section_id
        )
        if state.draft is not None:
            codes.extend(issue.code for issue in state.draft.issues)
        return list(dict.fromkeys(codes))

    def _checkpoint(self, state: AgentState, *, reason: str) -> None:
        latest = self._store.load_latest_checkpoint()
        checkpoint = AgentCheckpoint.model_validate(
            {
                "checkpoint_id": self._id("ckpt"),
                "sequence": 0 if latest is None else latest.sequence + 1,
                "reason": reason,
                "state": state,
                "state_fingerprint": agent_state_fingerprint(state),
                "parent_checkpoint_id": (
                    None if latest is None else latest.checkpoint_id
                ),
            }
        )
        self._store.save_checkpoint(checkpoint)

    @staticmethod
    def _validated_state(state: AgentState, **updates: object) -> AgentState:
        return AgentState.model_validate(
            state.model_copy(update=updates).model_dump(mode="json")
        )

    @staticmethod
    def _id(prefix: str) -> str:
        return f"{prefix}_{uuid4().hex[:16]}"
