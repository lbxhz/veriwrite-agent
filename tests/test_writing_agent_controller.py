from datetime import datetime, timezone
from types import SimpleNamespace

from veriwrite_agent.models.agent_runtime import ArtifactReference
from veriwrite_agent.models.writing_plan import (
    GroundedWritingPlan,
    WritingParagraphPlan,
    WritingSectionPlan,
)
from veriwrite_agent.models.writing_quality import (
    ManuscriptEditorialCheckpoint,
    ManuscriptQualityFinding,
    ManuscriptQualityReview,
)
from veriwrite_agent.services.writing_agent_controller import WritingAgentController
from veriwrite_agent.services.writing_evidence_recovery import (
    ParagraphEvidenceGap,
    WritingEvidenceRecoveryRequest,
)

NOW = datetime(2026, 8, 9, tzinfo=timezone.utc)


def reference(kind: str, artifact_id: str) -> ArtifactReference:
    return ArtifactReference.model_validate(
        {
            "artifact_id": artifact_id,
            "kind": kind,
            "schema_version": "test.1",
            "fingerprint": "a" * 64,
            "storage_key": f"snapshot.state.{artifact_id}",
            "status": "ready",
            "created_at": NOW,
        }
    )


def writing_plan() -> GroundedWritingPlan:
    section = WritingSectionPlan(
        section_id="methods",
        title="反演方法",
        purpose="比较反演方法的能力边界。",
        target_words=600,
        counting_policy="chinese_chars_and_english_words",
        paragraphs=[
            WritingParagraphPlan(
                paragraph_id="methods_p01",
                section_id="methods",
                paragraph_number=1,
                role="synthesis",
                purpose="综合比较方法。",
                claim_focus="不同反演方法具有不同适用边界。",
                central_question="这些方法如何比较？",
                argument_move="compare_studies",
                target_words=300,
                evidence_card_ids=["ev_1"],
                source_dois=["10.1000/core"],
            ),
            WritingParagraphPlan(
                paragraph_id="methods_p02",
                section_id="methods",
                paragraph_number=2,
                role="synthesis",
                purpose="归纳适用边界。",
                claim_focus="方法选择取决于观测条件与数据质量。",
                central_question="方法选择受到哪些条件限制？",
                argument_move="synthesize_consensus",
                target_words=300,
                evidence_card_ids=["ev_1"],
                source_dois=["10.1000/core"],
            ),
        ],
    )
    return GroundedWritingPlan(
        status="confirmed",
        topic="大气遥感",
        output_language="Chinese",
        plan_fingerprint="b" * 64,
        sections=[section],
        confirmed_by="student",
        confirmed_at=NOW,
    )


def evidence_recovery(*, require_download: bool) -> WritingEvidenceRecoveryRequest:
    gap = ParagraphEvidenceGap(
        section_id="methods",
        section_title="反演方法",
        paragraph_number=1,
        reason="comparison_requires_full_text",
        claim_focus="比较两类反演方法。",
        central_question="两类方法的差异是什么？",
        missing_full_text_dois=(
            ["10.1000/background"] if require_download else []
        ),
        available_direct_evidence_dois=["10.1000/core"],
        search_queries=["atmospheric retrieval comparison"],
        detail="比较需要至少两篇可追溯全文，目前只有一篇。",
    )
    return WritingEvidenceRecoveryRequest(
        status="pending_full_text" if require_download else "pending_search",
        source_plan_fingerprint="b" * 64,
        affected_section_ids=["methods"],
        gaps=[gap],
        requested_core_dois=(
            ["10.1000/background"] if require_download else []
        ),
        search_queries_by_section={
            "methods": ["atmospheric retrieval comparison"]
        },
    )


def test_passed_section_run_advances_to_global_manuscript_editor() -> None:
    result = SimpleNamespace(stopped_section_id=None)

    assessment = WritingAgentController().assess_section_run(
        result,
        writing_plan(),
        plan_reference=reference("writing_plan", "writing_plan_0123456789abcdef"),
        project_reference=reference(
            "writing_project", "writing_project_0123456789abcdef"
        ),
    )

    assert assessment.critic.outcome == "pass"
    assert assessment.decision.decision_type == "continue"
    assert assessment.decision.target_stage == "editing"
    assert assessment.decision.next_action.payload.kind == "run_critic"
    assert assessment.decision.next_action.payload.scope == "manuscript"


def test_evidence_gap_rolls_back_to_full_text_without_rewriting() -> None:
    result = SimpleNamespace(
        stopped_section_id="methods",
        stop_code="evidence_gap",
        stop_reason="missing full text",
        recovery_request=evidence_recovery(require_download=True),
    )

    assessment = WritingAgentController().assess_section_run(
        result,
        writing_plan(),
        plan_reference=reference("writing_plan", "writing_plan_0123456789abcdef"),
        project_reference=reference(
            "writing_project", "writing_project_0123456789abcdef"
        ),
    )

    assert assessment.critic.outcome == "rollback"
    assert assessment.decision.target_stage == "evidence"
    assert assessment.decision.next_action.payload.kind == "acquire_full_text"
    assert assessment.decision.next_action.payload.dois == ["10.1000/background"]


def test_evidence_gap_without_candidate_rolls_back_to_targeted_search() -> None:
    result = SimpleNamespace(
        stopped_section_id="methods",
        stop_code="evidence_gap",
        stop_reason="missing comparison source",
        recovery_request=evidence_recovery(require_download=False),
    )

    assessment = WritingAgentController().assess_section_run(
        result,
        writing_plan(),
        plan_reference=reference("writing_plan", "writing_plan_0123456789abcdef"),
        project_reference=reference(
            "writing_project", "writing_project_0123456789abcdef"
        ),
    )

    assert assessment.decision.target_stage == "literature"
    assert assessment.decision.next_action.payload.kind == "refine_literature_search"
    assert assessment.decision.next_action.payload.queries == [
        "atmospheric retrieval comparison"
    ]


def test_exhausted_rewrites_change_the_plan_instead_of_regenerating_the_chapter() -> None:
    issue = SimpleNamespace(
        severity="blocking",
        code="topic_drift",
        paragraph_number=1,
        detail="The contextual source became the paragraph's main subject.",
    )
    section = SimpleNamespace(
        section_id="methods",
        draft=SimpleNamespace(issues=[issue]),
    )
    result = SimpleNamespace(
        stopped_section_id="methods",
        stop_code="review_exhausted",
        stop_reason="same critique after three rounds",
        recovery_request=None,
        project=SimpleNamespace(sections=[section]),
    )

    assessment = WritingAgentController().assess_section_run(
        result,
        writing_plan(),
        plan_reference=reference("writing_plan", "writing_plan_0123456789abcdef"),
        project_reference=reference(
            "writing_project", "writing_project_0123456789abcdef"
        ),
    )

    assert assessment.decision.decision_type == "revise"
    assert assessment.decision.target_stage == "planning"
    assert assessment.decision.next_action.payload.kind == "revise_writing_plan"
    assert assessment.decision.next_action.payload.affected_section_ids == ["methods"]


def test_deterministic_plan_binding_blocker_revises_plan_without_rewriting() -> None:
    issue = SimpleNamespace(
        severity="blocking",
        code="source_permission_exceeded",
        paragraph_number=1,
        detail="The locked source cannot serve this paragraph role.",
    )
    section = SimpleNamespace(
        section_id="methods",
        draft=SimpleNamespace(issues=[issue], citations=[]),
    )
    result = SimpleNamespace(
        stopped_section_id="methods",
        stop_code="deterministic_blocked",
        stop_reason="non-prose dependency problem",
        recovery_request=None,
        project=SimpleNamespace(sections=[section]),
    )

    assessment = WritingAgentController().assess_section_run(
        result,
        writing_plan(),
        plan_reference=reference("writing_plan", "writing_plan_0123456789abcdef"),
        project_reference=reference(
            "writing_project", "writing_project_0123456789abcdef"
        ),
    )

    assert assessment.decision.decision_type == "revise"
    assert assessment.decision.target_stage == "planning"
    assert assessment.decision.next_action.payload.kind == "revise_writing_plan"


def test_deterministic_text_blocker_keeps_targeted_rewrite_route() -> None:
    issue = SimpleNamespace(
        severity="blocking",
        code="workflow_instruction_leak",
        paragraph_number=2,
        detail="The paragraph exposes an internal edit instruction.",
    )
    section = SimpleNamespace(
        section_id="methods",
        draft=SimpleNamespace(issues=[issue], citations=[]),
    )
    result = SimpleNamespace(
        stopped_section_id="methods",
        stop_code="deterministic_blocked",
        stop_reason="prose validation failed",
        recovery_request=None,
        project=SimpleNamespace(sections=[section]),
    )

    assessment = WritingAgentController().assess_section_run(
        result,
        writing_plan(),
        plan_reference=reference("writing_plan", "writing_plan_0123456789abcdef"),
        project_reference=reference(
            "writing_project", "writing_project_0123456789abcdef"
        ),
    )

    assert assessment.decision.decision_type == "retry"
    assert assessment.decision.next_action.payload.kind == "write_or_revise_sections"
    assert assessment.decision.next_action.payload.paragraph_numbers == {"methods": [2]}


def test_deterministic_evidence_integrity_blocker_rebuilds_evidence() -> None:
    issue = SimpleNamespace(
        severity="blocking",
        code="unconfirmed_evidence",
        paragraph_number=1,
        detail="The evidence card is not confirmed.",
    )
    citation = SimpleNamespace(doi="10.1000/core")
    section = SimpleNamespace(
        section_id="methods",
        draft=SimpleNamespace(issues=[issue], citations=[citation]),
    )
    result = SimpleNamespace(
        stopped_section_id="methods",
        stop_code="deterministic_blocked",
        stop_reason="evidence integrity failed",
        recovery_request=None,
        project=SimpleNamespace(sections=[section]),
    )

    assessment = WritingAgentController().assess_section_run(
        result,
        writing_plan(),
        plan_reference=reference("writing_plan", "writing_plan_0123456789abcdef"),
        project_reference=reference(
            "writing_project", "writing_project_0123456789abcdef"
        ),
    )

    assert assessment.decision.decision_type == "rollback"
    assert assessment.decision.target_stage == "evidence"
    assert assessment.decision.next_action.payload.kind == "rebuild_evidence"
    assert assessment.decision.next_action.payload.source_dois == ["10.1000/core"]


def test_manuscript_editor_reopens_only_located_paragraphs() -> None:
    checkpoint = ManuscriptEditorialCheckpoint(
        body_fingerprint="c" * 64,
        status="needs_revision",
        review=ManuscriptQualityReview(
            findings=[
                ManuscriptQualityFinding(
                    section_id="methods",
                    paragraph_number=1,
                    code="cross_section_repetition",
                    severity="blocking",
                    disposition="targeted_repair",
                    detail="This paragraph repeats the preceding chapter.",
                    revision_instruction="Retain only the comparison unique to this section.",
                )
            ]
        ),
        blocking_count=1,
        warning_count=0,
        completed_at=NOW,
    )

    assessment = WritingAgentController().assess_manuscript_editor(
        checkpoint,
        project_reference=reference(
            "writing_project", "writing_project_0123456789abcdef"
        ),
    )

    assert assessment.decision.decision_type == "revise"
    assert assessment.decision.target_stage == "writing"
    assert assessment.decision.next_action.payload.kind == "write_or_revise_sections"
    assert assessment.decision.next_action.payload.paragraph_numbers == {
        "methods": [1]
    }


def test_passed_manuscript_editor_advances_to_final_assembly() -> None:
    checkpoint = ManuscriptEditorialCheckpoint(
        body_fingerprint="c" * 64,
        status="passed",
        review=ManuscriptQualityReview(),
        blocking_count=0,
        warning_count=0,
        completed_at=NOW,
    )

    assessment = WritingAgentController().assess_manuscript_editor(
        checkpoint,
        project_reference=reference(
            "writing_project", "writing_project_0123456789abcdef"
        ),
    )

    assert assessment.critic.outcome == "pass"
    assert assessment.decision.target_stage == "delivery"
    assert assessment.decision.next_action.payload.kind == "assemble_final_delivery"


def test_releasable_final_package_waits_for_user_confirmation() -> None:
    package = SimpleNamespace(
        status="ready_for_confirmation",
        audit=SimpleNamespace(issues=[]),
    )

    assessment = WritingAgentController().assess_final_package(
        package,
        writing_plan(),
        package_reference=reference(
            "final_package", "final_package_0123456789abcdef"
        ),
    )

    assert assessment.critic.outcome == "pass"
    assert assessment.decision.decision_type == "request_user"
    assert assessment.decision.next_action.payload.request_type == "final_confirmation"


def test_reference_shortfall_at_delivery_rolls_back_to_literature() -> None:
    issue = SimpleNamespace(
        code="reference_count_below_minimum",
        severity="blocking",
        requirement_path="references.minimum_total",
        detail="required=60; actual=48",
    )
    package = SimpleNamespace(
        status="needs_revision",
        audit=SimpleNamespace(issues=[issue]),
        requirement_policy=SimpleNamespace(topic="大气遥感"),
    )

    assessment = WritingAgentController().assess_final_package(
        package,
        writing_plan(),
        package_reference=reference(
            "final_package", "final_package_0123456789abcdef"
        ),
    )

    assert assessment.critic.outcome == "rollback"
    assert assessment.decision.target_stage == "literature"
    assert assessment.decision.next_action.payload.kind == "refine_literature_search"
