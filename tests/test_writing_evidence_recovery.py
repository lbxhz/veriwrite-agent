import json
from types import SimpleNamespace

from veriwrite_agent.models.evidence import EvidenceQuote
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
    downgrade_unresolved_evidence_claims,
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
