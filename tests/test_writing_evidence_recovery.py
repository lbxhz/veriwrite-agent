import hashlib
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from veriwrite_agent.models.evidence import (
    DocumentAcquisition,
    DocumentPage,
    EvidenceCard,
    EvidenceLibrary,
    EvidenceQuote,
    LiteratureLibraryRecord,
)
from veriwrite_agent.models.literature_selection import (
    ConfirmedLiteratureSearchBlueprint,
    LiteratureSearchBlueprint,
    LiteratureThemePlan,
)
from veriwrite_agent.models.writing import (
    SectionEvidenceItem,
    SectionEvidencePacket,
    SectionSourceRecord,
)
from veriwrite_agent.models.writing_plan import (
    GroundedWritingPlan,
    WritingParagraphPlan,
    WritingSectionPlan,
)
from veriwrite_agent.services.writing_evidence_recovery import (
    WritingEvidenceRecoveryRequest,
    WritingEvidenceRecoveryService,
    deferred_enhancement_targets,
    deferred_recovery_dois,
    downgrade_unresolved_evidence_claims,
    preserve_unresolved_deferred_sections,
    upgrade_deferred_evidence_claims,
)
from veriwrite_agent.services.writing_quality import mark_manuscript_editor_targets
from veriwrite_agent.ui import writing_console

CORE_DOI = "10.1000/core"
SECOND_CORE_DOI = "10.1000/core-2"
BACKGROUND_DOI = "10.1000/background"


def _section_plan(*, use_background_comparison: bool) -> WritingSectionPlan:
    paragraphs = [
        WritingParagraphPlan(
            paragraph_id="method_p01",
            section_id="method",
            paragraph_number=1,
            role="detailed_evidence",
            purpose="Present the verified result.",
            claim_focus="The retrieval result reduces uncertainty.",
            central_question="What does the verified result establish?",
            argument_move="frame_problem",
            target_words=100,
            evidence_card_ids=["ev_core_result"],
            source_dois=[CORE_DOI],
        ),
        WritingParagraphPlan(
            paragraph_id="method_p02",
            section_id="method",
            paragraph_number=2,
            role="background" if use_background_comparison else "detailed_evidence",
            purpose="Compare model architectures and accuracy.",
            claim_focus="Compare the performance of two retrieval models.",
            central_question="Which model is more accurate and why?",
            argument_move="compare_studies",
            comparison_axis="accuracy and architecture",
            target_words=100,
            evidence_card_ids=(
                []
                if use_background_comparison
                else ["ev_core_result", "ev_second_result"]
            ),
            source_dois=(
                [BACKGROUND_DOI]
                if use_background_comparison
                else [CORE_DOI, SECOND_CORE_DOI]
            ),
        ),
        WritingParagraphPlan(
            paragraph_id="method_p03",
            section_id="method",
            paragraph_number=3,
            role="synthesis",
            purpose="State a bounded conclusion.",
            claim_focus="The evidence supports a scoped conclusion.",
            central_question="What conclusion follows?",
            argument_move="synthesize_consensus",
            target_words=100,
            evidence_card_ids=["ev_core_result"],
            source_dois=[CORE_DOI],
        ),
    ]
    return WritingSectionPlan(
        section_id="method",
        title="Retrieval methods",
        purpose="Compare atmospheric retrieval methods.",
        target_words=300,
        counting_policy="words",
        paragraphs=paragraphs,
    )


def _section_with_single_source(
    section: WritingSectionPlan,
    *,
    section_id: str,
    source_doi: str,
) -> WritingSectionPlan:
    return section.model_copy(
        update={
            "section_id": section_id,
            "title": section_id.title(),
            "paragraphs": [
                paragraph.model_copy(
                    update={
                        "paragraph_id": f"{section_id}_p{paragraph.paragraph_number:02d}",
                        "section_id": section_id,
                        "source_dois": [source_doi],
                    }
                )
                for paragraph in section.paragraphs
            ],
        }
    )


def test_recovery_merge_reopens_only_sections_needed_by_new_source_contract() -> None:
    template = _section_plan(use_background_comparison=False)
    generated_method = _section_with_single_source(
        template,
        section_id="method",
        source_doi=CORE_DOI,
    )
    generated_context = _section_with_single_source(
        template,
        section_id="context",
        source_doi=BACKGROUND_DOI,
    )
    accepted_old_method = _section_with_single_source(
        template,
        section_id="method",
        source_doi=SECOND_CORE_DOI,
    )

    sections, preserved = (
        writing_console._reopen_minimum_sections_for_source_coverage(
            reference_sections=[generated_method, generated_context],
            merged_sections=[accepted_old_method, generated_context],
            preserved_section_ids={"method", "context"},
            required_source_dois=[CORE_DOI, BACKGROUND_DOI],
        )
    )

    assert preserved == {"context"}
    assert sections[0] == generated_method
    assert sections[1] == generated_context


def _packet() -> SectionEvidencePacket:
    return SectionEvidencePacket(
        section_id="method",
        title="Retrieval methods",
        purpose="Compare atmospheric retrieval methods.",
        target_words=300,
        counting_policy="words",
        evidence_items=[
            SectionEvidenceItem(
                evidence_id="ev_core_result",
                doi=CORE_DOI,
                normalized_claim="The core study reports a retrieval result.",
                evidence_type="result",
                support_strength="direct",
                supporting_quotes=[
                    EvidenceQuote(page_number=3, exact_text="Verified result text.")
                ],
            ),
            SectionEvidenceItem(
                evidence_id="ev_second_result",
                doi=SECOND_CORE_DOI,
                normalized_claim="The second study reports a comparison result.",
                evidence_type="result",
                support_strength="direct",
                supporting_quotes=[
                    EvidenceQuote(page_number=5, exact_text="Second verified result.")
                ],
            ),
        ],
        sources=[
            SectionSourceRecord(
                doi=CORE_DOI,
                citation_key="core2025",
                title="Core retrieval study",
                year=2025,
                evidence_tier="A_core",
                permitted_use="detailed_claims",
                admission_status="admitted",
                centrality="central",
                supported_claim="Supports the retrieval result.",
                suitable_section_id="method",
            ),
            SectionSourceRecord(
                doi=SECOND_CORE_DOI,
                citation_key="second2025",
                title="Second core retrieval study",
                year=2025,
                evidence_tier="A_core",
                permitted_use="detailed_claims",
                admission_status="admitted",
                centrality="central",
                supported_claim="Supports comparison of retrieval results.",
                suitable_section_id="method",
            ),
            SectionSourceRecord(
                doi=BACKGROUND_DOI,
                citation_key="background2025",
                title="Metadata-only comparison candidate",
                year=2025,
                evidence_tier="C_background",
                permitted_use="background_only",
                admission_status="admitted",
                centrality="supporting",
                supported_claim="Provides general retrieval background.",
                suitable_section_id="method",
            ),
        ],
    )


def test_audit_routes_metadata_only_comparison_to_full_text_recovery() -> None:
    service = WritingEvidenceRecoveryService()

    gaps = service.audit_section(
        _section_plan(use_background_comparison=True),
        _packet(),
    )
    request = service.request(plan_fingerprint="a" * 64, gaps=gaps)

    assert len(gaps) == 1
    assert gaps[0].paragraph_number == 2
    assert gaps[0].reason == "comparison_requires_full_text"
    assert request.status == "pending_full_text"
    assert request.requested_core_dois == [BACKGROUND_DOI]
    assert request.unavailable_full_text_dois == []
    assert request.affected_section_ids == ["method"]


def test_audit_accepts_a_comparison_that_uses_only_direct_full_text_evidence() -> None:
    gaps = WritingEvidenceRecoveryService().audit_section(
        _section_plan(use_background_comparison=False),
        _packet(),
    )

    assert gaps == ()


def test_audit_accepts_general_background_sources_beside_two_direct_sources() -> None:
    plan = _section_plan(use_background_comparison=False)
    comparison = plan.paragraphs[1].model_copy(
        update={
            "role": "synthesis",
            "source_dois": [CORE_DOI, SECOND_CORE_DOI, BACKGROUND_DOI],
        }
    )
    plan = plan.model_copy(
        update={"paragraphs": [plan.paragraphs[0], comparison, plan.paragraphs[2]]}
    )

    gaps = WritingEvidenceRecoveryService().audit_section(plan, _packet())

    assert gaps == ()


def test_audit_does_not_escalate_a_pure_editorial_rewrite_to_pdf_recovery() -> None:
    plan = _section_plan(use_background_comparison=True)
    editorial = plan.paragraphs[1].model_copy(
        update={
            "purpose": "Global manuscript repair: remove repeated technical details.",
            "claim_focus": "Retain only one bounded editorial judgment.",
            "central_question": "What unique judgment should remain?",
            "argument_move": "author_judgment",
            "comparison_axis": None,
        }
    )
    plan = plan.model_copy(
        update={"paragraphs": [plan.paragraphs[0], editorial, plan.paragraphs[2]]}
    )

    gaps = WritingEvidenceRecoveryService().audit_section(plan, _packet())

    assert gaps == ()


def test_recovery_request_can_carry_reviewer_feedback_without_search_gap() -> None:
    request = WritingEvidenceRecoveryRequest(
        status="ready_to_resume",
        source_plan_fingerprint="a" * 64,
        affected_section_ids=["method"],
        repair_feedback_by_section={
            "method": ["第 2 段 topic_drift：更换中心判断与证据分配。"]
        },
        planning_repair_round=1,
    )

    assert request.gaps == []
    assert request.max_recovery_rounds == 4
    assert request.max_planning_repair_rounds == 2


def test_visible_recovery_status_tracks_internal_phase_without_stage_navigation() -> None:
    gap = WritingEvidenceRecoveryService().audit_section(
        _section_plan(use_background_comparison=True),
        _packet(),
    )[0]
    request = WritingEvidenceRecoveryService().request(
        plan_fingerprint="a" * 64,
        gaps=(gap,),
    ).model_copy(update={"status": "pending_search"})
    state = {
        writing_console.EVIDENCE_RECOVERY_REQUEST_KEY: request.model_dump_json()
    }

    assert writing_console.active_writing_recovery_status(state) == "pending_search"

    state[writing_console.EVIDENCE_RECOVERY_REQUEST_KEY] = request.model_copy(
        update={"status": "resolved"}
    ).model_dump_json()
    assert writing_console.active_writing_recovery_status(state) is None


def test_cross_stage_cleanup_preserves_active_agent_identity(monkeypatch) -> None:
    gap = WritingEvidenceRecoveryService().audit_section(
        _section_plan(use_background_comparison=True),
        _packet(),
    )[0]
    request = WritingEvidenceRecoveryService().request(
        plan_fingerprint="a" * 64,
        gaps=(gap,),
    )
    state = {
        writing_console.EVIDENCE_RECOVERY_REQUEST_KEY: request.model_dump_json(),
        writing_console.V04_AGENT_RUN_ID_KEY: "run_0123456789abcdef",
        writing_console.WRITING_PLAN_KEY: "old-plan",
    }
    monkeypatch.setattr(writing_console, "st", SimpleNamespace(session_state=state))

    writing_console.clear_writing_state()

    assert writing_console.WRITING_PLAN_KEY not in state
    assert state[writing_console.V04_AGENT_RUN_ID_KEY] == "run_0123456789abcdef"


def test_main_pause_clears_every_automatic_transition_but_keeps_run_identity() -> None:
    state = {
        writing_console.V04_AGENT_RUN_ID_KEY: "run_0123456789abcdef",
        writing_console.V04_AUTOPILOT_REQUESTED_KEY: True,
        writing_console.EVIDENCE_RECOVERY_AUTO_PLAN_KEY: True,
        writing_console.EVIDENCE_RECOVERY_AUTO_RESUME_KEY: True,
        writing_console.FINAL_DELIVERY_AUTO_RESUME_KEY: True,
        "literature_auto_run_requested": True,
        "literature_auto_advance_requested": True,
        writing_console.V04_PROJECT_KEY: "saved-project",
    }

    assert writing_console.pause_writing_agent(state)

    assert state[writing_console.V04_AGENT_RUN_ID_KEY] == "run_0123456789abcdef"
    assert state[writing_console.V04_PROJECT_KEY] == "saved-project"
    assert not writing_console._agent_auto_run_requested(state)
    assert "已暂停" in state["mvp_flash"]


def test_failure_ledger_survives_reruns_and_is_category_scoped() -> None:
    state: dict[str, object] = {}

    assert writing_console._record_failure_attempt(
        state,
        category="transient_generation",
        section_id="methods",
    ) == 1
    assert writing_console._record_failure_attempt(
        state,
        category="transient_generation",
        section_id="methods",
    ) == 2
    assert writing_console._record_failure_attempt(
        state,
        category="section_plan_repair",
        section_id="methods",
    ) == 1

    writing_console._clear_failure_attempts(
        state,
        category="transient_generation",
    )

    assert writing_console._failure_ledger(state) == {
        "section_plan_repair:methods": 1
    }


def test_connection_failure_is_the_only_kind_eligible_for_transient_resume() -> None:
    connection = SimpleNamespace(
        stop_code="generation_failed",
        stop_reason="chapter generation failed: Connection error.; 4/12 cached",
    )
    contract = SimpleNamespace(
        stop_code="generation_failed",
        stop_reason="paragraph plan references authority outside its section packet",
    )

    assert writing_console._is_transient_generation_failure(connection)
    assert not writing_console._is_transient_generation_failure(contract)


def test_recovery_shell_pause_control_accepts_checkpoint_only_state(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        writing_console,
        "st",
        SimpleNamespace(session_state={}),
    )

    # The recovery shell has no top-level V0.4 project and intentionally relies on
    # the control to resolve one from its checkpoint only when autorun is active.
    writing_console._render_agent_pause_control()


def test_persistent_control_can_restore_project_from_recovery_checkpoint(
    monkeypatch,
) -> None:
    state = {
        writing_console.EVIDENCE_RECOVERY_CHECKPOINT_KEY: json.dumps(
            {"writing_project_json": "saved-recovery-project"}
        )
    }
    recovered = SimpleNamespace(project_id="paper-from-checkpoint")
    monkeypatch.setattr(
        writing_console.V04WritingProject,
        "model_validate_json",
        classmethod(lambda _cls, payload: recovered if payload else None),
    )

    assert writing_console._control_project_from_state(state) is recovered


def test_ready_recovery_checkpoint_does_not_bypass_pause() -> None:
    state = {
        writing_console.EVIDENCE_RECOVERY_REQUEST_KEY: json.dumps(
            {"status": "ready_to_resume"}
        ),
        writing_console.EVIDENCE_RECOVERY_CHECKPOINT_KEY: "saved-checkpoint",
    }

    assert not writing_console._consume_writing_plan_auto_request(
        state,
        auto_failures=0,
    )

    state[writing_console.EVIDENCE_RECOVERY_AUTO_PLAN_KEY] = True
    assert writing_console._consume_writing_plan_auto_request(
        state,
        auto_failures=0,
    )
    assert writing_console.EVIDENCE_RECOVERY_AUTO_PLAN_KEY not in state


def test_user_can_decline_one_blocked_pdf_batch_and_resume_with_bounded_claims(
    monkeypatch,
) -> None:
    gap = WritingEvidenceRecoveryService().audit_section(
        _section_plan(use_background_comparison=True),
        _packet(),
    )[0]
    request = WritingEvidenceRecoveryService().request(
        plan_fingerprint="a" * 64,
        gaps=(gap,),
    ).model_copy(update={"status": "blocked"})
    state = {
        writing_console.EVIDENCE_RECOVERY_REQUEST_KEY: request.model_dump_json(),
        writing_console.EVIDENCE_RECOVERY_CHECKPOINT_KEY: json.dumps(
            {
                "writing_plan_json": "saved-plan",
                "writing_project_json": "saved-project",
            }
        ),
    }
    monkeypatch.setattr(
        writing_console,
        "st",
        SimpleNamespace(session_state=state),
    )
    fake_plan = SimpleNamespace()
    fake_handoff = SimpleNamespace(model_dump_json=lambda **_: "saved-handoff")
    fake_project = SimpleNamespace(handoff=fake_handoff)
    monkeypatch.setattr(
        writing_console.GroundedWritingPlan,
        "model_validate_json",
        classmethod(lambda _cls, _value: fake_plan),
    )
    monkeypatch.setattr(
        writing_console.V04WritingProject,
        "model_validate_json",
        classmethod(lambda _cls, _value: fake_project),
    )
    captured = {}

    def bounded_downgrade(project, plan, declined_request):
        captured["project"] = project
        captured["plan"] = plan
        captured["request"] = declined_request
        return True

    monkeypatch.setattr(
        writing_console,
        "_begin_bounded_claim_downgrade",
        bounded_downgrade,
    )

    assert writing_console.continue_without_restricted_full_text()

    assert captured["project"] is fake_project
    assert captured["plan"] is fake_plan
    assert captured["request"].unavailable_full_text_dois == [BACKGROUND_DOI]
    assert state["v03_writing_handoff_json"] == "saved-handoff"
    assert state[writing_console.V04_AUTOPILOT_REQUESTED_KEY] is True
    assert state[writing_console.EVIDENCE_RECOVERY_AUTO_RESUME_KEY] is True


def test_unresolved_comparison_is_downgraded_to_metadata_bounded_background() -> None:
    plan = _section_plan(use_background_comparison=True)
    gap = WritingEvidenceRecoveryService().audit_section(plan, _packet())[0]
    writing_plan = GroundedWritingPlan(
        topic="Atmospheric retrieval",
        output_language="English",
        plan_fingerprint="a" * 64,
        sections=[plan],
    )

    downgraded = downgrade_unresolved_evidence_claims(writing_plan, [gap])

    paragraph = downgraded.sections[0].paragraphs[1]
    assert paragraph.role == "background"
    assert paragraph.argument_move == "frame_problem"
    assert paragraph.evidence_card_ids == []
    assert paragraph.source_dois == [BACKGROUND_DOI]
    # 降级时必须保留原始意图，供后续 PDF 增强 pass 恢复为详细对比。
    assert paragraph.deferred_argument == "compare_studies"
    assert paragraph.deferred_comparison_axis == "accuracy and architecture"
    assert paragraph.deferred_purpose == writing_plan.sections[0].paragraphs[1].purpose
    assert (
        paragraph.deferred_claim_focus
        == writing_plan.sections[0].paragraphs[1].claim_focus
    )
    assert (
        paragraph.deferred_central_question
        == writing_plan.sections[0].paragraphs[1].central_question
    )
    assert paragraph.deferred_recovery_dois == [BACKGROUND_DOI]
    assert downgraded.plan_fingerprint != writing_plan.plan_fingerprint
    assert WritingEvidenceRecoveryService().audit_section(
        downgraded.sections[0], _packet()
    ) == ()
    assert WritingEvidenceRecoveryService().validate_resolution(
        downgraded,
        [_packet()],
        affected_section_ids=["method"],
    ) == ()


def test_targeted_repair_evidence_audit_ignores_accepted_untouched_paragraphs() -> None:
    gaps = WritingEvidenceRecoveryService().audit_section(
        _section_plan(use_background_comparison=True),
        _packet(),
        paragraph_numbers={1},
    )

    assert gaps == ()


def test_manuscript_editor_target_is_not_reclassified_as_new_evidence_claim() -> None:
    section = _section_plan(use_background_comparison=True)
    plan = GroundedWritingPlan(
        topic="Atmospheric retrieval",
        output_language="English",
        plan_fingerprint="a" * 64,
        sections=[section],
    )

    marked = mark_manuscript_editor_targets(
        plan,
        {("method", 2): ["Remove repeated background and retain one cautious judgment."]},
    )

    paragraph = marked.sections[0].paragraphs[1]
    assert paragraph.argument_move == "author_judgment"
    assert paragraph.role == "background"
    assert paragraph.source_dois == [BACKGROUND_DOI]
    assert marked.plan_fingerprint != plan.plan_fingerprint
    assert WritingEvidenceRecoveryService().audit_section(
        marked.sections[0], _packet(), paragraph_numbers={2}
    ) == ()


def test_resolution_gate_rejects_a_downgraded_plan_with_unknown_support() -> None:
    plan = _section_plan(use_background_comparison=True)
    gap = WritingEvidenceRecoveryService().audit_section(plan, _packet())[0]
    writing_plan = GroundedWritingPlan(
        topic="Atmospheric retrieval",
        output_language="English",
        plan_fingerprint="a" * 64,
        sections=[plan],
    )
    downgraded = downgrade_unresolved_evidence_claims(writing_plan, [gap])
    bad_paragraph = downgraded.sections[0].paragraphs[1].model_copy(
        update={"source_dois": ["10.1000/not-in-packet"]}
    )
    bad_section = downgraded.sections[0].model_copy(
        update={
            "paragraphs": [
                downgraded.sections[0].paragraphs[0],
                bad_paragraph,
                downgraded.sections[0].paragraphs[2],
            ]
        }
    )
    invalid = downgraded.model_copy(update={"sections": [bad_section]})

    errors = WritingEvidenceRecoveryService().validate_resolution(
        invalid,
        [_packet()],
        affected_section_ids=["method"],
    )

    assert any("unknown source 10.1000/not-in-packet" in error for error in errors)


def test_recovery_queries_are_added_without_weakening_the_search_boundary() -> None:
    service = WritingEvidenceRecoveryService()
    gaps = service.audit_section(
        _section_plan(use_background_comparison=True),
        _packet(),
    )
    request = service.request(plan_fingerprint="a" * 64, gaps=gaps)
    original = ConfirmedLiteratureSearchBlueprint(
        confirmed_by="student",
        blueprint=LiteratureSearchBlueprint(
            topic="Atmospheric retrieval",
            discipline="Atmospheric science",
            writing_through_line="Compare retrieval methods",
            target_total=3,
            max_candidates=100,
            themes=[
                LiteratureThemePlan(
                    theme_id="method",
                    section_title="Retrieval methods",
                    section_purpose="Compare retrieval methods",
                    research_questions=["How do retrieval methods differ?"],
                    primary_keywords=["atmospheric retrieval"],
                    search_queries=["atmospheric retrieval"],
                    target_count=2,
                ),
                LiteratureThemePlan(
                    theme_id="future",
                    section_title="Future work",
                    section_purpose="Review future retrieval research",
                    research_questions=["What remains unresolved?"],
                    primary_keywords=["retrieval future"],
                    search_queries=["atmospheric retrieval future"],
                    target_count=1,
                ),
            ],
        ),
    )

    enriched = service.enrich_search_blueprint(original, request)

    assert enriched.blueprint.target_total == original.blueprint.target_total
    assert enriched.blueprint.requirement_policy == original.blueprint.requirement_policy
    assert enriched.blueprint.max_candidates == 100
    assert enriched.blueprint.themes[0].search_queries[0] != "atmospheric retrieval"
    assert "atmospheric retrieval" in enriched.blueprint.themes[0].search_queries


def test_recovery_retry_history_is_scoped_to_the_affected_chapter() -> None:
    old_gap = WritingEvidenceRecoveryService().audit_section(
        _section_plan(use_background_comparison=True),
        _packet(),
    )[0]
    previous = WritingEvidenceRecoveryService().request(
        plan_fingerprint="a" * 64,
        gaps=(old_gap,),
    ).model_copy(update={"status": "ready_to_resume", "recovery_round": 4})
    different_gap = old_gap.model_copy(
        update={
            "section_id": "future",
            "section_title": "Future work",
        }
    )
    different_chapter = WritingEvidenceRecoveryService().request(
        plan_fingerprint="b" * 64,
        gaps=(different_gap,),
    )
    same_chapter = WritingEvidenceRecoveryService().request(
        plan_fingerprint="b" * 64,
        gaps=(old_gap,),
    )

    assert not writing_console._same_evidence_recovery_incident(
        previous,
        different_chapter,
    )
    assert writing_console._same_evidence_recovery_incident(previous, same_chapter)


def test_reviewer_replan_budget_does_not_leak_from_resolved_or_other_chapter() -> None:
    previous = WritingEvidenceRecoveryRequest(
        status="resolved",
        source_plan_fingerprint="a" * 64,
        affected_section_ids=["challenges"],
        repair_feedback_by_section={"challenges": ["reassign support"]},
        planning_repair_round=3,
        max_planning_repair_rounds=3,
    )

    assert not writing_console._same_reviewer_plan_repair_incident(
        previous,
        "assimilation",
    )
    assert not writing_console._same_reviewer_plan_repair_incident(
        previous,
        "challenges",
    )


def test_reviewer_replan_budget_is_retained_for_same_unresolved_chapter() -> None:
    previous = WritingEvidenceRecoveryRequest(
        status="ready_to_resume",
        source_plan_fingerprint="a" * 64,
        affected_section_ids=["assimilation"],
        repair_feedback_by_section={"assimilation": ["reassign support"]},
        planning_repair_round=2,
        max_planning_repair_rounds=3,
    )

    assert writing_console._same_reviewer_plan_repair_incident(
        previous,
        "assimilation",
    )
    assert not writing_console._same_reviewer_plan_repair_incident(
        previous,
        "forecast",
    )


def test_reviewer_plan_repair_starts_fresh_after_resolved_other_chapter(
    monkeypatch,
) -> None:
    previous = WritingEvidenceRecoveryRequest(
        status="resolved",
        source_plan_fingerprint="a" * 64,
        affected_section_ids=["challenges"],
        repair_feedback_by_section={"challenges": ["reassign support"]},
        planning_repair_round=3,
        max_planning_repair_rounds=3,
    )
    session_state = {
        writing_console.EVIDENCE_RECOVERY_REQUEST_KEY: previous.model_dump_json()
    }
    monkeypatch.setattr(
        writing_console,
        "st",
        SimpleNamespace(session_state=session_state),
    )
    captured: list[WritingEvidenceRecoveryRequest] = []
    monkeypatch.setattr(
        writing_console,
        "_begin_structural_plan_repair",
        lambda project, plan, request: captured.append(request),
    )
    section_state = SimpleNamespace(
        section_id="assimilation",
        draft=SimpleNamespace(
            issues=[
                SimpleNamespace(
                    severity="blocking",
                    code="source_permission_exceeded",
                    paragraph_number=2,
                    detail="background-only source cannot provide section support",
                )
            ]
        ),
    )
    plan = SimpleNamespace(
        plan_fingerprint="b" * 64,
        sections=[
            SimpleNamespace(
                section_id="assimilation",
                paragraphs=[
                    SimpleNamespace(
                        paragraph_number=2,
                        claim_focus="Compare assimilation methods",
                    )
                ],
            )
        ]
    )

    assert writing_console._begin_reviewer_plan_repair(
        SimpleNamespace(),
        plan,
        section_state,
    )
    assert captured[0].affected_section_ids == ["assimilation"]
    assert captured[0].planning_repair_round == 1
    assert captured[0].unavailable_full_text_dois == []


def test_editorial_evidence_gap_is_not_routed_to_pdf_acquisition() -> None:
    gap = WritingEvidenceRecoveryService().audit_section(
        _section_plan(use_background_comparison=True),
        _packet(),
    )[0]
    request = WritingEvidenceRecoveryService().request(
        plan_fingerprint="a" * 64,
        gaps=(gap,),
    )
    project = SimpleNamespace(
        sections=[
            SimpleNamespace(
                section_id="method",
                draft=SimpleNamespace(
                    issues=[
                        SimpleNamespace(
                            code="final_audit_repair",
                            severity="blocking",
                            paragraph_number=2,
                        )
                    ]
                ),
            )
        ]
    )

    assert writing_console._evidence_gaps_are_editorial_targets(project, request)


def test_section_plan_repair_budget_survives_resolved_request_and_stops(
    monkeypatch,
) -> None:
    state = {
        writing_console.V04_FAILURE_LEDGER_KEY: json.dumps(
            {"section_plan_repair:method": 3}
        )
    }
    monkeypatch.setattr(
        writing_console,
        "st",
        SimpleNamespace(session_state=state),
    )
    paused: list[object] = []
    monkeypatch.setattr(
        writing_console,
        "pause_writing_agent",
        lambda _state, project=None: paused.append(project) or True,
    )
    monkeypatch.setattr(writing_console, "_autosave_current_project", lambda: True)
    monkeypatch.setattr(
        writing_console,
        "_begin_structural_plan_repair",
        lambda *_: pytest.fail("exhausted repair budget must not start replanning"),
    )
    project = SimpleNamespace()
    section_state = SimpleNamespace(
        section_id="method",
        draft=SimpleNamespace(
            issues=[
                SimpleNamespace(
                    severity="blocking",
                    code="unsupported_claim",
                    paragraph_number=1,
                    detail="claim exceeds evidence",
                )
            ]
        ),
    )

    assert not writing_console._begin_reviewer_plan_repair(
        project,
        SimpleNamespace(),
        section_state,
    )
    assert paused == [project]
    assert "跨页面熔断" in state["mvp_flash"]


def test_exhausted_replanned_gap_downgrades_without_more_search() -> None:
    gap = WritingEvidenceRecoveryService().audit_section(
        _section_plan(use_background_comparison=True),
        _packet(),
    )[0]
    request = WritingEvidenceRecoveryService().request(
        plan_fingerprint="a" * 64,
        gaps=(gap,),
    ).model_copy(
        update={
            "status": "ready_to_resume",
            "recovery_round": 4,
            "max_recovery_rounds": 4,
            "planning_repair_round": 1,
        }
    )
    project = SimpleNamespace(sections=[])

    assert writing_console._evidence_recovery_can_downgrade_without_search(
        project,
        request,
    )


def test_structural_plan_repair_actually_queues_automatic_replanning(
    monkeypatch,
) -> None:
    gap = WritingEvidenceRecoveryService().audit_section(
        _section_plan(use_background_comparison=True),
        _packet(),
    )[0]
    request = WritingEvidenceRecoveryService().request(
        plan_fingerprint="a" * 64,
        gaps=(gap,),
    ).model_copy(update={"recovery_round": 4, "planning_repair_round": 1})
    state = {
        writing_console.WRITING_PLAN_KEY: "old-plan",
        writing_console.V04_PROJECT_KEY: "old-project",
    }
    monkeypatch.setattr(
        writing_console,
        "st",
        SimpleNamespace(session_state=state),
    )
    fake_handoff = SimpleNamespace(model_dump_json=lambda **_: "handoff")
    fake_project = SimpleNamespace(
        handoff=fake_handoff,
        model_dump_json=lambda **_: "project",
    )
    fake_plan = SimpleNamespace(model_dump_json=lambda **_: "plan")

    writing_console._begin_structural_plan_repair(
        fake_project,
        fake_plan,
        request,
    )

    queued = WritingEvidenceRecoveryRequest.model_validate_json(
        state[writing_console.EVIDENCE_RECOVERY_REQUEST_KEY]
    )
    feedback = json.loads(state[writing_console.WRITING_PLAN_REPAIR_FEEDBACK_KEY])
    assert queued.status == "ready_to_resume"
    assert feedback["method"]
    assert writing_console.WRITING_PLAN_KEY not in state
    assert writing_console.V04_PROJECT_KEY not in state
    assert state[writing_console.EVIDENCE_RECOVERY_AUTO_PLAN_KEY] is True
    assert state["v04_force_writing_plan_regeneration"] is True
    assert state["mvp_navigation_request"] == "writing"


def test_evidence_recovery_at_limit_downgrades_instead_of_replanning(
    monkeypatch,
) -> None:
    """recovery 达到上限时必须走降级，而不是继续重新规划。

    回归保护：_begin_evidence_recovery 的边界判断曾用 ``>``，导致
    recovery_round == max_recovery_rounds（4 == 4）时不触发降级，
    而是继续设置 v04_force_writing_plan_regeneration 重规划，形成死循环。
    """
    gap = WritingEvidenceRecoveryService().audit_section(
        _section_plan(use_background_comparison=True),
        _packet(),
    )[0]
    request = WritingEvidenceRecoveryService().request(
        plan_fingerprint="a" * 64,
        gaps=(gap,),
    ).model_copy(
        update={
            "recovery_round": 4,
            "max_recovery_rounds": 4,
            "planning_repair_round": 2,
            "max_planning_repair_rounds": 2,
        }
    )
    monkeypatch.setattr(writing_console, "st", SimpleNamespace(session_state={}))
    monkeypatch.setattr(
        writing_console,
        "_evidence_gaps_are_editorial_targets",
        lambda project, req: False,
    )
    downgrade_calls: list[WritingEvidenceRecoveryRequest] = []
    monkeypatch.setattr(
        writing_console,
        "_begin_bounded_claim_downgrade",
        lambda project, plan, req: downgrade_calls.append(req) or True,
    )
    # 让修复前的"重新规划"路径也能走到断言点，而不因缺少 mock 崩溃。
    monkeypatch.setattr(
        writing_console,
        "_best_evidence_recovery_checkpoint",
        lambda project, plan, req: {},
    )
    monkeypatch.setattr(writing_console, "_selected_literature_dois", lambda: set())
    monkeypatch.setattr(
        writing_console,
        "route_evidence_recovery_to_search",
        lambda req: None,
    )

    result = writing_console._begin_evidence_recovery(
        SimpleNamespace(),
        SimpleNamespace(),
        request,
    )

    assert downgrade_calls, (
        "recovery_round 达到上限时应降级为背景主张，而不是继续重新规划"
    )
    assert result is True


def test_first_missing_pdf_batch_is_deferred_without_cross_stage_loop(
    monkeypatch,
) -> None:
    gap = WritingEvidenceRecoveryService().audit_section(
        _section_plan(use_background_comparison=True),
        _packet(),
    )[0]
    request = WritingEvidenceRecoveryService().request(
        plan_fingerprint="a" * 64,
        gaps=(gap,),
    )
    monkeypatch.setattr(writing_console, "st", SimpleNamespace(session_state={}))
    downgrade_calls: list[WritingEvidenceRecoveryRequest] = []
    monkeypatch.setattr(
        writing_console,
        "_begin_bounded_claim_downgrade",
        lambda project, plan, req: downgrade_calls.append(req) or True,
    )
    monkeypatch.setattr(
        writing_console,
        "route_evidence_recovery_to_search",
        lambda req: pytest.fail("missing PDF must not re-enter V0.2/V0.3 search"),
    )

    result = writing_console._begin_evidence_recovery(
        SimpleNamespace(),
        SimpleNamespace(),
        request,
    )

    assert result is True
    assert downgrade_calls == [request]
    assert request.recovery_round == 1


def test_deferred_upgrade_of_pending_chapter_updates_plan_without_reopen(
    monkeypatch,
) -> None:
    old_paragraph = SimpleNamespace(
        deferred_argument="frame_problem",
        paragraph_number=5,
    )
    upgraded_paragraph = SimpleNamespace(
        deferred_argument=None,
        paragraph_number=5,
    )
    plan = SimpleNamespace(
        sections=[SimpleNamespace(section_id="theme_application_forecast", paragraphs=[old_paragraph])]
    )
    upgraded_plan = SimpleNamespace(
        sections=[
            SimpleNamespace(
                section_id="theme_application_forecast",
                paragraphs=[upgraded_paragraph],
            )
        ],
        model_dump_json=lambda **_: "upgraded-plan",
    )
    project = SimpleNamespace(
        handoff=SimpleNamespace(evidence_library=SimpleNamespace()),
        sections=[
            SimpleNamespace(
                section_id="theme_application_forecast",
                status="pending",
                draft=None,
            )
        ],
        model_dump_json=lambda **_: "pending-project",
    )
    project.model_copy = lambda **_: project
    state: dict[str, object] = {}
    monkeypatch.setattr(writing_console, "st", SimpleNamespace(session_state=state))
    monkeypatch.setattr(
        writing_console,
        "upgrade_deferred_evidence_claims",
        lambda _plan, _library: upgraded_plan,
    )
    monkeypatch.setattr(
        writing_console,
        "_reopen_targeted_paragraphs",
        lambda *_: pytest.fail("a pending chapter has no draft to reopen"),
    )

    assert writing_console._begin_deferred_evidence_enhancement(project, plan)
    assert state[writing_console.WRITING_PLAN_KEY] == "upgraded-plan"
    assert state[writing_console.V04_PROJECT_KEY] == "pending-project"


FIRST_RECOVERED_DOI = "10.1000/recovered-1"
SECOND_RECOVERED_DOI = "10.1000/recovered-2"


def _recovered_library(
    full_text_dois: list[str],
    *,
    metadata_only_dois: list[str] | None = None,
) -> EvidenceLibrary:
    records: list[LiteratureLibraryRecord] = []
    documents: list[DocumentAcquisition] = []
    pages: list[DocumentPage] = []
    cards: list[EvidenceCard] = []
    for index, doi in enumerate(full_text_dois):
        sha = hashlib.sha256(doi.encode("utf-8")).hexdigest()
        documents.append(
            DocumentAcquisition(
                doi=doi,
                status="available",
                method="user_upload",
                source_url=f"https://doi.org/{doi}",
                local_path=f"runtime/recovered_{index}.pdf",
                sha256=sha,
                media_type="application/pdf",
                file_size_bytes=4096,
                attempts=1,
            )
        )
        pages.append(
            DocumentPage(
                doi=doi,
                document_sha256=sha,
                page_number=1,
                text=f"Verified full-text evidence for {doi}.",
                extraction_method="native_text",
            )
        )
        cards.append(
            EvidenceCard(
                evidence_id=f"ev_rec_{index}",
                doi=doi,
                theme_id="method",
                evidence_type="result",
                normalized_claim=f"Verified result from {doi}.",
                supporting_quotes=[
                    EvidenceQuote(
                        page_number=1,
                        exact_text=f"Verified full-text evidence for {doi}.",
                    )
                ],
                source_document_sha256=sha,
                support_strength="direct",
                review_status="confirmed",
            )
        )
        records.append(
            LiteratureLibraryRecord(
                doi=doi,
                title=f"Recovered study {index}",
                authors=["Doe, Jane"],
                year=2025,
                journal="Journal",
                source_url=f"https://doi.org/{doi}",
                theme_ids=["method"],
                evidence_tier="A_core",
                evidence_status="full_text_verified",
                permitted_use="detailed_claims",
                admission_status="admitted",
                centrality="central",
                supported_claim=f"Supports detailed comparison evidence for {doi}.",
                suitable_section_id="method",
                use_boundary="Use for detailed retrieval comparison evidence.",
            )
        )
    for doi in metadata_only_dois or []:
        records.append(
            LiteratureLibraryRecord(
                doi=doi,
                title="Metadata-only candidate",
                authors=["Roe, Ana"],
                year=2024,
                source_url=f"https://doi.org/{doi}",
                theme_ids=["method"],
                evidence_tier="C_background",
                evidence_status="metadata_verified",
                permitted_use="background_only",
            )
        )
    return EvidenceLibrary(
        status="confirmed",
        records=records,
        documents=documents,
        pages=pages,
        evidence_cards=cards,
        confirmed_by="student",
        confirmed_at=datetime.now(timezone.utc),
    )


def _deferred_plan(
    *,
    deferred_argument: str,
    deferred_dois: list[str],
    comparison_axis: str | None = None,
) -> GroundedWritingPlan:
    paragraph = WritingParagraphPlan(
        paragraph_id="method_p01",
        section_id="method",
        paragraph_number=1,
        role="background",
        purpose="State a bounded background overview and the limits of the available records.",
        claim_focus="The available records permit only a general description of its scope.",
        central_question="What is the general scope of this direction?",
        argument_move="frame_problem",
        comparison_axis=None,
        target_words=100,
        evidence_card_ids=[],
        source_dois=list(deferred_dois),
        deferred_argument=deferred_argument,
        deferred_comparison_axis=comparison_axis,
        deferred_purpose="Compare the original planned retrieval evidence.",
        deferred_claim_focus="The original planned comparison claim.",
        deferred_central_question="What did the original comparison establish?",
        deferred_recovery_dois=list(deferred_dois),
    )
    tail = WritingParagraphPlan(
        paragraph_id="method_p02",
        section_id="method",
        paragraph_number=2,
        role="synthesis",
        purpose="State a bounded conclusion.",
        claim_focus="The evidence supports a scoped conclusion.",
        central_question="What conclusion follows?",
        argument_move="synthesize_consensus",
        target_words=100,
        evidence_card_ids=[],
        source_dois=list(deferred_dois),
    )
    section = WritingSectionPlan(
        section_id="method",
        title="Retrieval methods",
        purpose="Compare atmospheric retrieval methods.",
        target_words=200,
        counting_policy="words",
        paragraphs=[paragraph, tail],
    )
    return GroundedWritingPlan(
        topic="Atmospheric retrieval",
        output_language="English",
        plan_fingerprint="a" * 64,
        sections=[section],
    )


def test_deferred_comparison_is_upgraded_when_both_pdfs_arrive() -> None:
    dois = [FIRST_RECOVERED_DOI, SECOND_RECOVERED_DOI]
    plan = _deferred_plan(
        deferred_argument="compare_studies",
        deferred_dois=dois,
        comparison_axis="accuracy and architecture",
    )
    library = _recovered_library(dois)

    upgraded = upgrade_deferred_evidence_claims(plan, library)

    paragraph = upgraded.sections[0].paragraphs[0]
    assert paragraph.role == "detailed_evidence"
    assert paragraph.argument_move == "compare_studies"
    assert paragraph.comparison_axis == "accuracy and architecture"
    assert paragraph.evidence_card_ids == ["ev_rec_0", "ev_rec_1"]
    assert paragraph.source_dois == dois
    assert paragraph.purpose == "Compare the original planned retrieval evidence."
    assert paragraph.claim_focus == "The original planned comparison claim."
    assert paragraph.central_question == "What did the original comparison establish?"
    assert paragraph.deferred_argument is None
    assert paragraph.deferred_comparison_axis is None
    assert paragraph.deferred_recovery_dois == []
    assert upgraded.plan_fingerprint != plan.plan_fingerprint


def test_replan_cannot_erase_deferred_marker_without_recovered_pdf() -> None:
    dois = [FIRST_RECOVERED_DOI, SECOND_RECOVERED_DOI]
    previous = _deferred_plan(
        deferred_argument="compare_studies",
        deferred_dois=dois,
        comparison_axis="accuracy and architecture",
    )
    detailed = previous.sections[0].paragraphs[0].model_copy(
        update={
            "role": "detailed_evidence",
            "purpose": "Make the detailed comparison again.",
            "claim_focus": "One model is more accurate than the other.",
            "deferred_argument": None,
            "deferred_comparison_axis": None,
            "deferred_purpose": None,
            "deferred_claim_focus": None,
            "deferred_central_question": None,
            "deferred_recovery_dois": [],
        }
    )
    candidate = previous.model_copy(
        update={
            "plan_fingerprint": "b" * 64,
            "sections": [
                previous.sections[0].model_copy(
                    update={
                        "paragraphs": [
                            detailed,
                            previous.sections[0].paragraphs[1],
                        ]
                    }
                )
            ],
        }
    )
    metadata_only = _recovered_library([], metadata_only_dois=dois)

    guarded = preserve_unresolved_deferred_sections(
        previous,
        candidate,
        metadata_only,
    )

    assert guarded.sections[0] == previous.sections[0]
    assert guarded.sections[0].paragraphs[0].deferred_argument == "compare_studies"
    assert guarded.plan_fingerprint not in {
        previous.plan_fingerprint,
        candidate.plan_fingerprint,
    }


def test_replan_may_replace_section_after_all_deferred_pdfs_are_recovered() -> None:
    dois = [FIRST_RECOVERED_DOI, SECOND_RECOVERED_DOI]
    previous = _deferred_plan(
        deferred_argument="compare_studies",
        deferred_dois=dois,
        comparison_axis="accuracy and architecture",
    )
    candidate = previous.model_copy(update={"plan_fingerprint": "b" * 64})

    guarded = preserve_unresolved_deferred_sections(
        previous,
        candidate,
        _recovered_library(dois),
    )

    assert guarded is candidate


def test_deferred_comparison_stays_background_when_a_pdf_is_missing() -> None:
    dois = [FIRST_RECOVERED_DOI, SECOND_RECOVERED_DOI]
    plan = _deferred_plan(
        deferred_argument="compare_studies",
        deferred_dois=dois,
        comparison_axis="accuracy and architecture",
    )
    library = _recovered_library(
        [FIRST_RECOVERED_DOI],
        metadata_only_dois=[SECOND_RECOVERED_DOI],
    )

    upgraded = upgrade_deferred_evidence_claims(plan, library)

    assert upgraded is plan
    paragraph = upgraded.sections[0].paragraphs[0]
    assert paragraph.role == "background"
    assert paragraph.argument_move == "frame_problem"
    assert paragraph.deferred_argument == "compare_studies"


def test_deferred_detail_claim_is_upgraded_with_a_single_pdf() -> None:
    plan = _deferred_plan(
        deferred_argument="evaluate_limitation",
        deferred_dois=[FIRST_RECOVERED_DOI],
    )
    library = _recovered_library([FIRST_RECOVERED_DOI])

    upgraded = upgrade_deferred_evidence_claims(plan, library)

    paragraph = upgraded.sections[0].paragraphs[0]
    assert paragraph.role == "detailed_evidence"
    assert paragraph.argument_move == "evaluate_limitation"
    assert paragraph.evidence_card_ids == ["ev_rec_0"]
    assert paragraph.deferred_argument is None
    assert paragraph.deferred_recovery_dois == []


def test_deferred_recovery_dois_aggregates_the_batch_download_list() -> None:
    plan = _deferred_plan(
        deferred_argument="compare_studies",
        deferred_dois=[FIRST_RECOVERED_DOI, SECOND_RECOVERED_DOI],
        comparison_axis="accuracy and architecture",
    )

    assert deferred_recovery_dois(plan) == [FIRST_RECOVERED_DOI, SECOND_RECOVERED_DOI]

    no_deferred = _deferred_plan(
        deferred_argument="compare_studies",
        deferred_dois=[FIRST_RECOVERED_DOI, SECOND_RECOVERED_DOI],
        comparison_axis="accuracy and architecture",
    ).model_copy(
        update={
            "sections": [
                plan.sections[0].model_copy(
                    update={
                        "paragraphs": [
                            plan.sections[0].paragraphs[0].model_copy(
                                update={"deferred_recovery_dois": []}
                            ),
                            plan.sections[0].paragraphs[1],
                        ]
                    }
                )
            ]
        }
    )
    assert deferred_recovery_dois(no_deferred) == []


def test_deferred_enhancement_targets_reports_only_upgradable_paragraphs() -> None:
    dois = [FIRST_RECOVERED_DOI, SECOND_RECOVERED_DOI]
    plan = _deferred_plan(
        deferred_argument="compare_studies",
        deferred_dois=dois,
        comparison_axis="accuracy and architecture",
    )
    library = _recovered_library(dois)

    assert deferred_enhancement_targets(plan, library) == {("method", 1)}


def test_deferred_enhancement_targets_is_empty_when_pdfs_are_missing() -> None:
    dois = [FIRST_RECOVERED_DOI, SECOND_RECOVERED_DOI]
    plan = _deferred_plan(
        deferred_argument="compare_studies",
        deferred_dois=dois,
        comparison_axis="accuracy and architecture",
    )
    library = _recovered_library(
        [FIRST_RECOVERED_DOI],
        metadata_only_dois=[SECOND_RECOVERED_DOI],
    )

    assert deferred_enhancement_targets(plan, library) == set()


def test_upgrade_ignores_paragraphs_without_deferred_intent() -> None:
    plan = _deferred_plan(
        deferred_argument="compare_studies",
        deferred_dois=[FIRST_RECOVERED_DOI, SECOND_RECOVERED_DOI],
        comparison_axis="accuracy and architecture",
    )
    # 清除 defer 标记，模拟一个从未降级的普通段落。
    plain = plan.model_copy(
        update={
            "sections": [
                plan.sections[0].model_copy(
                    update={
                        "paragraphs": [
                            plan.sections[0].paragraphs[0].model_copy(
                                update={
                                    "deferred_argument": None,
                                    "deferred_comparison_axis": None,
                                    "deferred_recovery_dois": [],
                                }
                            ),
                            plan.sections[0].paragraphs[1],
                        ]
                    }
                )
            ]
        }
    )
    library = _recovered_library([FIRST_RECOVERED_DOI, SECOND_RECOVERED_DOI])

    upgraded = upgrade_deferred_evidence_claims(plain, library)

    assert upgraded is plain
    assert upgraded.sections[0].paragraphs[0].role == "background"
