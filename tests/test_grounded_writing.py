import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from veriwrite_agent.llm.fake_client import FakeLLMClient, ScriptedLLMClient
from veriwrite_agent.models.evidence import (
    DocumentAcquisition,
    DocumentPage,
    EvidenceCard,
    EvidenceLibrary,
    EvidenceQuote,
    LiteratureLibraryRecord,
)
from veriwrite_agent.models.requirement_workflow import ConfirmedRequirementSpec
from veriwrite_agent.models.requirements import (
    AIUsagePolicy,
    ReferenceRequirement,
    RequirementSpec,
    TopicBoundary,
)
from veriwrite_agent.models.writing import (
    DraftParagraphProposal,
    SectionDraftIssue,
    SectionDraftProposal,
    V04WritingProject,
)
from veriwrite_agent.models.writing_plan import (
    GroundedWritingPlan,
    ParagraphTextProposal,
    SectionPlanProposal,
)
from veriwrite_agent.models.writing_quality import (
    ManuscriptQualityFinding,
    ManuscriptQualityReview,
)
from veriwrite_agent.models.writing_handoff import (
    ConfirmedWritingOutline,
    V04WritingHandoff,
    WritingOutlineDraft,
    WritingOutlineSection,
)
from veriwrite_agent.services.grounded_writing import (
    GroundedSectionDraftService,
    GroundedWritingError,
    LLMGroundedSectionWriter,
    SectionEvidencePacketBuilder,
    WritingProjectService,
    count_writing_units,
)
from veriwrite_agent.services.agent_runtime_store import AgentRuntimeStore
from veriwrite_agent.services.manuscript_structural_editing import (
    merge_redundant_manuscript_paragraphs,
    semantically_replan_manuscript_sections,
)
from veriwrite_agent.services.requirement_policy import RequirementPolicyCompiler
from veriwrite_agent.services.writing_planning import (
    GroundedWritingPlanner,
    LLMGroundedParagraphWriter,
    ParagraphEvidencePacketBuilder,
    ParagraphWritingRuntimeCache,
    PlannedSectionDraftService,
    WritingPlanError,
    WritingPlanBudgetExceeded,
    WritingPlanRuntimeCache,
    _assign_required_sources_to_problem_paragraphs,
    _repair_compiled_required_source_coverage,
    align_writing_plan_language,
    rebase_writing_plan_authority,
    repair_writing_plan_source_coverage,
)
from veriwrite_agent.services.writing_quality import (
    FullManuscriptEditorialService,
    LLMManuscriptQualityReviewer,
    LLMSectionQualityReviewer,
    _validate_manuscript_review,
    apply_section_quality_review,
    mark_section_quality_review_failed,
    refine_writing_plan_for_manuscript_review,
)
from veriwrite_agent.services.writing_autopilot import (
    ContinuousSectionWritingService,
    ContinuousWritingPolicy,
)
from veriwrite_agent.services.writing_agent_runtime import WritingAgentRuntimeService
from veriwrite_agent.services.writing_evidence_recovery import merge_recovery_handoffs
from veriwrite_agent.ui.writing_console import (
    _merge_recovery_checkpoint_progress,
    _reopen_confirmed_state_for_plan_changes,
    _synchronize_project_handoff,
    reopen_entire_body_for_regeneration,
)

DOI = "10.1000/core.1"
SUPPORTING_DOI = "10.1000/support.1"
BACKGROUND_DOI = "10.1000/background.1"
SHA = "a" * 64
EVIDENCE_ID = "ev_method_direct_001"


class SequenceLLMClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def complete(
        self,
        messages,
        *,
        response_format: dict[str, str] | None = None,
    ) -> str:
        self.calls.append(
            {
                "messages": list(messages),
                "response_format": response_format,
            }
        )
        return self.responses.pop(0)


def handoff() -> V04WritingHandoff:
    document = DocumentAcquisition(
        doi=DOI,
        status="available",
        method="user_upload",
        source_url=f"https://doi.org/{DOI}",
        local_path="runtime/core.pdf",
        sha256=SHA,
        media_type="application/pdf",
        file_size_bytes=4096,
        attempts=1,
    )
    page = DocumentPage(
        doi=DOI,
        document_sha256=SHA,
        page_number=4,
        text="The retrieval model reduced uncertainty over the study region.",
        extraction_method="native_text",
    )
    card = EvidenceCard(
        evidence_id=EVIDENCE_ID,
        doi=DOI,
        theme_id="method",
        evidence_type="result",
        normalized_claim="The retrieval model reduced regional uncertainty.",
        supporting_quotes=[
            EvidenceQuote(
                page_number=4,
                exact_text=(
                    "The retrieval model reduced uncertainty over the study region."
                ),
            )
        ],
        source_document_sha256=SHA,
        support_strength="direct",
        review_status="confirmed",
    )
    core = LiteratureLibraryRecord(
        doi=DOI,
        title="Verified retrieval model",
        authors=["Sun, Lin"],
        year=2025,
        journal="Atmospheric Research",
        source_url=f"https://doi.org/{DOI}",
        theme_ids=["method"],
        evidence_tier="A_core",
        evidence_status="full_text_verified",
        permitted_use="detailed_claims",
        admission_status="admitted",
        centrality="central",
        supported_claim="Supports comparison of atmospheric retrieval methods.",
        suitable_section_id="method",
        use_boundary="Use only for atmospheric retrieval method evidence.",
    )
    supporting = LiteratureLibraryRecord(
        doi=SUPPORTING_DOI,
        title="Verified background record",
        authors=["Chen, An"],
        year=2024,
        journal="Journal",
        abstract="Remote sensing supports regional atmospheric observation.",
        source_url=f"https://doi.org/{SUPPORTING_DOI}",
        theme_ids=["method"],
        evidence_tier="B_supporting",
        evidence_status="metadata_verified",
        permitted_use="section_support",
        admission_status="admitted",
        centrality="supporting",
        supported_claim="Supports the atmospheric retrieval background.",
        suitable_section_id="method",
        use_boundary="Use only as section-level atmospheric retrieval support.",
    )
    library = EvidenceLibrary(
        status="confirmed",
        records=[core, supporting],
        documents=[document],
        pages=[page],
        evidence_cards=[card],
        confirmed_by="student",
        confirmed_at=datetime.now(timezone.utc),
    )
    outline = ConfirmedWritingOutline(
        outline=WritingOutlineDraft(
            topic="Atmospheric remote sensing",
            writing_through_line="Compare evidence-backed retrieval methods.",
            target_words=300,
            sections=[
                WritingOutlineSection(
                    section_id="method",
                    title="Retrieval methods",
                    purpose="Compare recent retrieval methods.",
                    target_words=300,
                    research_questions=["How are retrieval errors reduced?"],
                    core_dois=[DOI],
                    supporting_dois=[SUPPORTING_DOI],
                    evidence_card_ids=[EVIDENCE_ID],
                )
            ],
        ),
        confirmed_by="student",
    )
    requirement = ConfirmedRequirementSpec(
        confirmed_by="student",
        confirmed_at=datetime.now(timezone.utc),
        requirement=RequirementSpec(
            document_type="literature_review",
            output_language="English",
            topic="Atmospheric remote sensing",
            topic_boundary=TopicBoundary(
                central_question="How can atmospheric retrieval uncertainty be reduced?",
                included_objects=["atmospheric retrieval methods"],
                excluded_objects=["soil moisture"],
                contextual_only_topics=["edge computing"],
                origin="explicit",
            ),
        ),
    )
    return V04WritingHandoff(
        requirement=requirement,
        outline=outline,
        evidence_library=library,
    )


def chinese_handoff() -> V04WritingHandoff:
    active = handoff()
    requirement = active.requirement.requirement.model_copy(
        update={"output_language": "Chinese"}
    )
    return active.model_copy(
        update={
            "requirement": active.requirement.model_copy(
                update={"requirement": requirement}
            ),
            "requirement_policy": None,
        }
    )


def two_section_handoff() -> V04WritingHandoff:
    active = handoff()
    first = active.outline.outline.sections[0]
    second = first.model_copy(
        update={
            "section_id": "discussion",
            "title": "Evidence synthesis",
            "purpose": "Synthesize the verified retrieval evidence.",
            "research_questions": ["What conclusion follows from the evidence?"],
        }
    )
    outline_draft = active.outline.outline.model_copy(
        update={"target_words": 600, "sections": [first, second]}
    )
    return active.model_copy(
        update={
            "outline": active.outline.model_copy(update={"outline": outline_draft})
        }
    )


def test_recovery_handoff_keeps_unaffected_outline_and_verified_evidence() -> None:
    previous = two_section_handoff()
    supporting = previous.evidence_library.records[1]
    current_library = EvidenceLibrary.model_validate(
        previous.evidence_library.model_copy(
            update={
                "records": [supporting],
                "documents": [],
                "extractions": [],
                "page_selections": [],
                "pages": [],
                "evidence_cards": [],
                "literature_matrix": [],
            }
        ).model_dump(mode="json")
    )
    current_sections = [
        section.model_copy(
            update={
                "core_dois": [],
                "supporting_dois": [SUPPORTING_DOI],
                "evidence_card_ids": [],
            }
        )
        for section in previous.outline.outline.sections
    ]
    current = V04WritingHandoff.model_validate(
        previous.model_copy(
            update={
                "evidence_library": current_library,
                "outline": previous.outline.model_copy(
                    update={
                        "outline": previous.outline.outline.model_copy(
                            update={"sections": current_sections}
                        )
                    }
                ),
            }
        ).model_dump(mode="json")
    )

    merged = merge_recovery_handoffs(
        previous,
        current,
        affected_section_ids={"discussion"},
    )

    records = {record.doi: record for record in merged.evidence_library.records}
    sections = {
        section.section_id: section for section in merged.outline.outline.sections
    }
    assert records[DOI].evidence_status == "full_text_verified"
    assert EVIDENCE_ID in {
        card.evidence_id for card in merged.evidence_library.evidence_cards
    }
    assert sections["method"] == previous.outline.outline.sections[0]
    assert sections["discussion"].core_dois == []


def test_recovery_checkpoint_unions_compatible_confirmed_progress() -> None:
    active = handoff()
    plan = GroundedWritingPlanner(FakeLLMClient(plan_response())).plan(active).confirm(
        confirmed_by="student"
    )
    packet = SectionEvidencePacketBuilder().build(active, "method")
    draft = GroundedSectionDraftService().create(
        packet,
        SectionDraftProposal(
            section_id="method",
            paragraphs=[
                DraftParagraphProposal(
                    role=paragraph.role,
                    text=f"Evidence-bound paragraph {paragraph.paragraph_number}.",
                    evidence_card_ids=paragraph.evidence_card_ids,
                    source_dois=paragraph.source_dois,
                )
                for paragraph in plan.sections[0].paragraphs
            ],
        ),
    )
    projects = WritingProjectService()
    accepted = projects.save_draft(projects.start(active), quality_passed(draft))
    accepted = projects.confirm_section(accepted, "method", confirmed_by="student")
    pending = projects.start(active)
    existing = {
        "writing_plan_json": plan.model_dump_json(),
        "writing_project_json": accepted.model_dump_json(),
        "handoff_json": active.model_dump_json(),
    }
    candidate = {
        "writing_plan_json": plan.model_dump_json(),
        "writing_project_json": pending.model_dump_json(),
        "handoff_json": active.model_dump_json(),
    }

    merged = _merge_recovery_checkpoint_progress(existing, candidate)
    restored = V04WritingProject.model_validate_json(merged["writing_project_json"])

    assert restored.sections[0].status == "confirmed"
    assert restored.sections[0].draft == accepted.sections[0].draft

    excluded = _merge_recovery_checkpoint_progress(
        existing,
        candidate,
        excluded_section_ids={"method"},
    )
    excluded_project = V04WritingProject.model_validate_json(
        excluded["writing_project_json"]
    )
    assert excluded_project.sections[0].status == "pending"

    changed_paragraphs = list(plan.sections[0].paragraphs)
    changed_paragraphs[1] = changed_paragraphs[1].model_copy(
        update={"purpose": "Use newly admitted bounded background authority."}
    )
    changed_section = plan.sections[0].model_copy(
        update={"paragraphs": changed_paragraphs}
    )
    targeted = _reopen_confirmed_state_for_plan_changes(
        accepted.sections[0],
        previous_plan=plan.sections[0],
        current_plan=changed_section,
        handoff=active,
    )

    assert targeted is not None
    assert targeted.status == "needs_review"
    assert targeted.draft is not None
    assert targeted.draft.paragraphs == accepted.sections[0].draft.paragraphs
    repair_issues = [
        issue for issue in targeted.draft.issues if issue.code == "final_audit_repair"
    ]
    assert [issue.paragraph_number for issue in repair_issues] == [2]


def blocked_handoff() -> V04WritingHandoff:
    active = handoff()
    blocked_requirement = active.requirement.requirement.model_copy(
        update={
            "ai_policy": AIUsagePolicy(
                prohibited_uses=["不允许 AI 生成任何句子或段落内容。"],
            )
        }
    )
    return active.model_copy(
        update={
            "requirement": active.requirement.model_copy(
                update={"requirement": blocked_requirement}
            )
        }
    )


def handoff_with_background_source() -> V04WritingHandoff:
    active = handoff()
    background = LiteratureLibraryRecord(
        doi=BACKGROUND_DOI,
        title="Metadata-only background record",
        authors=["Park, Min"],
        year=2025,
        journal="Background Journal",
        abstract="Remote sensing and AI are used in environmental monitoring.",
        source_url=f"https://doi.org/{BACKGROUND_DOI}",
        theme_ids=["method"],
        evidence_tier="C_background",
        evidence_status="metadata_verified",
        permitted_use="background_only",
        admission_status="admitted",
        centrality="supporting",
        supported_claim="Provides background on environmental remote sensing.",
        suitable_section_id="method",
        use_boundary="Use only as brief background, never as the section subject.",
    )
    library = active.evidence_library.model_copy(
        update={"records": [*active.evidence_library.records, background]}
    )
    section = active.outline.outline.sections[0].model_copy(
        update={
            "supporting_dois": [
                SUPPORTING_DOI,
                BACKGROUND_DOI,
            ]
        }
    )
    outline_draft = active.outline.outline.model_copy(update={"sections": [section]})
    outline = active.outline.model_copy(update={"outline": outline_draft})
    return active.model_copy(
        update={"evidence_library": library, "outline": outline}
    )


def proposal(
    *,
    role: str = "detailed_evidence",
    evidence_ids: list[str] | None = None,
    source_dois: list[str] | None = None,
    text: str = "The retrieval model reduces regional uncertainty.",
) -> SectionDraftProposal:
    return SectionDraftProposal(
        section_id="method",
        paragraphs=[
            DraftParagraphProposal(
                role=role,
                text=text,
                evidence_card_ids=(
                    [EVIDENCE_ID] if evidence_ids is None else evidence_ids
                ),
                source_dois=[] if source_dois is None else source_dois,
            )
        ],
    )


def quality_passed(draft):
    return draft.model_copy(
        update={
            "quality_review_status": "passed",
            "quality_review_rounds": max(1, draft.quality_review_rounds),
            "quality_reviewed_at": datetime.now(timezone.utc),
        }
    )


def plan_response(*, evidence_ref: str = "E001") -> str:
    return json.dumps(
        {
            "section_id": "method",
            "paragraphs": [
                {
                    "role": "detailed_evidence",
                    "purpose": "Present the verified retrieval result.",
                    "claim_focus": "The verified model reduces regional uncertainty.",
                    "central_question": "How does the verified model reduce uncertainty?",
                    "argument_move": "frame_problem",
                    "comparison_axis": "retrieval uncertainty",
                    "relative_weight": 3,
                    "evidence_refs": [evidence_ref],
                    "source_refs": [],
                },
                {
                    "role": "section_support",
                    "purpose": "Connect the result to the wider research context.",
                    "claim_focus": "Remote sensing supports regional observation.",
                    "central_question": "What boundary limits the supporting study?",
                    "argument_move": "evaluate_limitation",
                    "comparison_axis": "scope of application",
                    "relative_weight": 2,
                    "evidence_refs": [],
                    "source_refs": ["S002"],
                },
                {
                    "role": "synthesis",
                    "purpose": "Synthesize the core and supporting literature.",
                    "claim_focus": "The evidence motivates regional retrieval workflows.",
                    "central_question": "What shared conclusion follows from the studies?",
                    "argument_move": "synthesize_consensus",
                    "comparison_axis": "regional retrieval workflow",
                    "relative_weight": 1,
                    "evidence_refs": ["E001"],
                    "source_refs": ["S002"],
                },
            ],
        }
    )


def test_builds_section_packet_from_confirmed_handoff() -> None:
    packet = SectionEvidencePacketBuilder().build(handoff(), "method")

    assert packet.evidence_items[0].evidence_id == EVIDENCE_ID
    assert [source.doi for source in packet.sources] == [DOI, SUPPORTING_DOI]
    assert packet.sources[0].citation_key == "sun2025_core1"


def test_planner_compiles_short_aliases_to_locked_real_authority() -> None:
    client = FakeLLMClient(plan_response())

    plan = GroundedWritingPlanner(client).plan(handoff())

    section = plan.sections[0]
    assert plan.status == "draft"
    assert plan.required_source_dois == []
    assert len(section.paragraphs) == 3
    assert sum(item.target_words for item in section.paragraphs) == 300
    assert section.paragraphs[0].evidence_card_ids == [EVIDENCE_ID]
    assert section.paragraphs[0].source_dois == [DOI]
    assert section.paragraphs[1].source_dois == [SUPPORTING_DOI]
    assert len(client.calls) == 1
    assert plan.output_language == "English"
    prompt_payload = json.loads(client.calls[0]["messages"][1]["content"])
    assert prompt_payload["evidence_catalog"][0]["ref"] == "E001"
    assert prompt_payload["source_catalog"][1]["ref"] == "S002"
    assert prompt_payload["source_catalog"][1]["allowed_roles"] == [
        "section_support",
        "background",
        "synthesis",
    ]


def test_planner_stops_before_exceeding_model_call_budget() -> None:
    client = ScriptedLLMClient([plan_response()])

    with pytest.raises(WritingPlanBudgetExceeded):
        GroundedWritingPlanner(client, max_model_calls=1).plan(
            two_section_handoff()
        )

    assert len(client.calls) == 1


def test_required_source_replaces_optional_ref_when_repair_capacity_is_full() -> None:
    active_handoff = handoff_with_background_source()
    packet = SectionEvidencePacketBuilder().build(active_handoff, "method")
    packet = packet.model_copy(
        update={
            "required_source_dois": [BACKGROUND_DOI],
            "max_sources_per_paragraph": 1,
        }
    )
    proposal_payload = json.loads(plan_response())
    proposal_payload["paragraphs"][2]["evidence_refs"] = []
    proposal_payload["paragraphs"][2]["source_refs"] = ["S002"]
    proposal_plan = SectionPlanProposal.model_validate(proposal_payload)
    evidence_aliases = {
        "E001": packet.evidence_items[0],
    }
    source_aliases = {
        f"S{index:03d}": source
        for index, source in enumerate(packet.sources, 1)
    }

    repaired = _assign_required_sources_to_problem_paragraphs(
        packet,
        proposal_plan,
        evidence_aliases=evidence_aliases,
        source_aliases=source_aliases,
        repair_invalid_permissions=True,
    )

    assert repaired.paragraphs[2].source_refs == ["S003"]


def test_required_source_uses_fourth_policy_authorized_source_slot() -> None:
    active_handoff = handoff_with_background_source()
    packet = SectionEvidencePacketBuilder().build(active_handoff, "method")
    fourth = packet.sources[2].model_copy(
        update={
            "doi": "10.1000/background.2",
            "title": "Additional bounded background source",
        }
    )
    packet = packet.model_copy(
        update={
            "sources": [*packet.sources, fourth],
            "required_source_dois": [fourth.doi],
            "max_sources_per_paragraph": 4,
        }
    )
    proposal_payload = json.loads(plan_response())
    proposal_payload["paragraphs"][0].update(
        {
            "role": "synthesis",
            "evidence_refs": [],
            "source_refs": ["S001", "S002", "S003"],
        }
    )
    proposal_payload["paragraphs"][2].update(
        {
            "role": "detailed_evidence",
            "evidence_refs": ["E001"],
            "source_refs": [],
        }
    )
    proposal_plan = SectionPlanProposal.model_validate(proposal_payload)
    evidence_aliases = {"E001": packet.evidence_items[0]}
    source_aliases = {
        f"S{index:03d}": source
        for index, source in enumerate(packet.sources, 1)
    }

    repaired = _assign_required_sources_to_problem_paragraphs(
        packet,
        proposal_plan,
        evidence_aliases=evidence_aliases,
        source_aliases=source_aliases,
        repair_invalid_permissions=True,
    )

    assert repaired.paragraphs[0].source_refs == [
        "S001",
        "S002",
        "S003",
        "S004",
    ]


def test_missing_required_source_replaces_duplicate_required_occurrence() -> None:
    active_handoff = handoff_with_background_source()
    packet = SectionEvidencePacketBuilder().build(active_handoff, "method")
    supporting_card = packet.evidence_items[0].model_copy(
        update={
            "evidence_id": "ev_method_support_002",
            "doi": SUPPORTING_DOI,
        }
    )
    supporting_source = packet.sources[1].model_copy(
        update={"permitted_use": "detailed_claims"}
    )
    packet = packet.model_copy(
        update={
            "evidence_items": [*packet.evidence_items, supporting_card],
            "sources": [
                packet.sources[0],
                supporting_source,
                packet.sources[2],
            ],
            "required_source_dois": [SUPPORTING_DOI, BACKGROUND_DOI],
            "max_sources_per_paragraph": 1,
        }
    )
    proposal_payload = json.loads(plan_response())
    proposal_payload["paragraphs"][0].update(
        {
            "role": "synthesis",
            "evidence_refs": [],
            "source_refs": ["S002"],
        }
    )
    proposal_payload["paragraphs"][1].update(
        {
            "role": "detailed_evidence",
            "evidence_refs": ["E002"],
            "source_refs": [],
        }
    )
    proposal_payload["paragraphs"][2].update(
        {
            "role": "detailed_evidence",
            "evidence_refs": ["E001"],
            "source_refs": [],
        }
    )
    proposal_plan = SectionPlanProposal.model_validate(proposal_payload)
    evidence_aliases = {
        "E001": packet.evidence_items[0],
        "E002": supporting_card,
    }
    source_aliases = {
        f"S{index:03d}": source
        for index, source in enumerate(packet.sources, 1)
    }

    repaired = _assign_required_sources_to_problem_paragraphs(
        packet,
        proposal_plan,
        evidence_aliases=evidence_aliases,
        source_aliases=source_aliases,
        repair_invalid_permissions=True,
    )

    assert repaired.paragraphs[0].source_refs == ["S003"]
    assert repaired.paragraphs[1].evidence_refs == ["E002"]


def test_required_sources_allocate_restricted_permission_before_flexible_source() -> None:
    active_handoff = handoff_with_background_source()
    packet = SectionEvidencePacketBuilder().build(active_handoff, "method")
    packet = packet.model_copy(
        update={
            "required_source_dois": [SUPPORTING_DOI, BACKGROUND_DOI],
            "max_sources_per_paragraph": 1,
        }
    )
    proposal_plan = SectionPlanProposal.model_validate(
        {
            "section_id": "method",
            "paragraphs": [
                {
                    "role": "synthesis",
                    "purpose": "establish the contextual boundary",
                    "claim_focus": "background constraints and context",
                    "central_question": "Which contextual boundary matters?",
                    "argument_move": "synthesize_consensus",
                    "comparison_axis": "context",
                    "relative_weight": 5,
                    "evidence_refs": [],
                    "source_refs": [],
                },
                {
                    "role": "section_support",
                    "purpose": "compare the supporting result",
                    "claim_focus": "supporting methodological result",
                    "central_question": "Which result supports the section?",
                    "argument_move": "compare_studies",
                    "comparison_axis": "method",
                    "relative_weight": 5,
                    "evidence_refs": [],
                    "source_refs": [],
                },
            ],
        }
    )
    source_aliases = {
        f"S{index:03d}": source
        for index, source in enumerate(packet.sources, 1)
    }

    repaired = _assign_required_sources_to_problem_paragraphs(
        packet,
        proposal_plan,
        evidence_aliases={},
        source_aliases=source_aliases,
        repair_invalid_permissions=True,
    )

    assert repaired.paragraphs[0].source_refs == ["S003"]
    assert repaired.paragraphs[1].source_refs == ["S002"]


def test_semantic_replan_requires_only_sources_previously_used_by_section() -> None:
    active_handoff = handoff_with_background_source()
    plan = GroundedWritingPlanner(FakeLLMClient(plan_response())).plan(active_handoff)
    section = plan.sections[0]
    paragraphs = []
    for paragraph in section.paragraphs:
        source_dois = [
            doi for doi in paragraph.source_dois if doi != BACKGROUND_DOI
        ]
        if not source_dois and not paragraph.evidence_card_ids:
            source_dois = [DOI]
        paragraphs.append(paragraph.model_copy(update={"source_dois": source_dois}))
    section = section.model_copy(update={"paragraphs": paragraphs})
    plan = GroundedWritingPlan.model_validate(
        plan.model_copy(update={"sections": [section]}).model_dump(mode="json")
    )

    class CapturePlanner:
        packet = None

        def replan_section(self, packet, *, paragraph_count):
            self.packet = packet
            assert paragraph_count == len(section.paragraphs)
            return section

    planner = CapturePlanner()
    project = WritingProjectService().start(active_handoff)

    semantically_replan_manuscript_sections(
        plan,
        project,
        section_ids={section.section_id},
        planner=planner,
    )

    assert planner.packet is not None
    assert BACKGROUND_DOI not in planner.packet.required_source_dois
    assert set(planner.packet.required_source_dois) <= {
        doi
        for paragraph in section.paragraphs
        for doi in paragraph.source_dois
    } | {DOI}


def test_semantic_replan_failure_keeps_refined_section_for_targeted_repair() -> None:
    active_handoff = handoff_with_background_source()
    plan = GroundedWritingPlanner(FakeLLMClient(plan_response())).plan(active_handoff)
    project = WritingProjectService().start(active_handoff)

    class FailingPlanner:
        def replan_section(self, packet, *, paragraph_count):
            raise WritingPlanError("synthetic capacity conflict")

    result = semantically_replan_manuscript_sections(
        plan,
        project,
        section_ids={plan.sections[0].section_id},
        planner=FailingPlanner(),
    )

    assert result.sections == plan.sections


def test_global_editor_refines_plan_before_targeted_rewrite() -> None:
    plan = GroundedWritingPlanner(FakeLLMClient(plan_response())).plan(handoff())
    original = plan.sections[0]
    finding = ManuscriptQualityFinding(
        section_id="method",
        paragraph_number=3,
        code="cross_section_repetition",
        disposition="targeted_repair",
        detail="The synthesis repeats both earlier paragraphs.",
        revision_instruction=(
            "Keep only the distinct boundary judgment and omit repeated study details."
        ),
    )

    refined = refine_writing_plan_for_manuscript_review(
        plan,
        ManuscriptQualityReview(findings=[finding]),
        evidence_doi_by_id={EVIDENCE_ID: DOI},
    )

    target = refined.sections[0].paragraphs[2]
    assert refined.plan_fingerprint != plan.plan_fingerprint
    assert target.target_words <= 240
    assert target.claim_focus == finding.revision_instruction
    assert target.argument_move == "author_judgment"
    assert target.source_dois == [DOI]
    assert target.evidence_card_ids == [EVIDENCE_ID]
    assert sum(
        paragraph.target_words for paragraph in refined.sections[0].paragraphs
    ) == original.target_words


def test_global_editor_warning_cannot_drive_an_endless_rewrite_loop() -> None:
    plan = GroundedWritingPlanner(FakeLLMClient(plan_response())).plan(handoff())
    warning = ManuscriptQualityFinding(
        section_id="method",
        paragraph_number=3,
        code="global_coherence_gap",
        severity="warning",
        disposition="targeted_repair",
        detail="A transition could be clearer.",
        revision_instruction="Add a short transition.",
    )

    normalized = _validate_manuscript_review(
        ManuscriptQualityReview(findings=[warning]),
        plan,
    )

    assert normalized.findings[0].disposition == "report_only"


def test_material_manuscript_warning_is_promoted_to_targeted_repair() -> None:
    plan = GroundedWritingPlanner(FakeLLMClient(plan_response())).plan(handoff())
    warning = ManuscriptQualityFinding(
        section_id="method",
        paragraph_number=3,
        code="cross_section_repetition",
        severity="warning",
        disposition="report_only",
        detail="The paragraph repeats a prior argument.",
        revision_instruction="Merge it into another paragraph.",
    )

    normalized = _validate_manuscript_review(
        ManuscriptQualityReview(findings=[warning]),
        plan,
    )

    finding = normalized.findings[0]
    assert finding.severity == "blocking"
    assert finding.disposition == "targeted_repair"
    assert "不得移动段落" in finding.revision_instruction


def test_manuscript_reviewer_deduplicates_the_same_paragraph_issue() -> None:
    plan = GroundedWritingPlanner(FakeLLMClient(plan_response())).plan(handoff())
    duplicate = ManuscriptQualityFinding(
        section_id="method",
        paragraph_number=3,
        code="cross_section_repetition",
        severity="warning",
        disposition="report_only",
        detail="The paragraph repeats a prior argument.",
        revision_instruction="Retain only the unique synthesis.",
    )

    normalized = _validate_manuscript_review(
        ManuscriptQualityReview(findings=[duplicate, duplicate]),
        plan,
    )

    assert len(normalized.findings) == 1
    assert normalized.findings[0].severity == "blocking"
    assert normalized.findings[0].disposition == "targeted_repair"


def test_v05_explicitly_carries_deferred_chapter_finding() -> None:
    active = handoff()
    plan = GroundedWritingPlanner(FakeLLMClient(plan_response())).plan(active).confirm(
        confirmed_by="student"
    )
    packet = SectionEvidencePacketBuilder().build(active, "method")
    paragraphs = [
        DraftParagraphProposal(
            role=item.role,
            text=f"Evidence-bound paragraph {item.paragraph_number}.",
            evidence_card_ids=item.evidence_card_ids,
            source_dois=item.source_dois,
        )
        for item in plan.sections[0].paragraphs
    ]
    draft = GroundedSectionDraftService().create(
        packet,
        SectionDraftProposal(section_id="method", paragraphs=paragraphs),
    )
    draft = quality_passed(
        draft.model_copy(
            update={
                "issues": [
                    *draft.issues,
                    SectionDraftIssue(
                        code="terminology_inconsistent",
                        severity="warning",
                        paragraph_number=2,
                        detail="V0.4 deferred inconsistent terminology.",
                    ),
                    SectionDraftIssue(
                        code="quality_review_deferred",
                        severity="warning",
                        detail="Local repair converged without removing the finding.",
                    ),
                ]
            }
        )
    )
    projects = WritingProjectService()
    project = projects.save_draft(projects.start(active), draft)
    project = projects.confirm_section(project, "method", confirmed_by="student")
    client = FakeLLMClient(json.dumps({"findings": []}))

    checkpoint = FullManuscriptEditorialService(
        LLMManuscriptQualityReviewer(client)
    ).run(plan, project)

    assert checkpoint.status == "passed"
    assert checkpoint.warning_count == 1
    assert checkpoint.review.findings[0].code == "terminology_inconsistent"
    payload = json.loads(client.calls[0]["messages"][1]["content"])
    assert payload["deferred_chapter_findings"][0]["code"] == (
        "terminology_inconsistent"
    )


def test_v05_reviewer_can_promote_deferred_repetition_to_targeted_repair() -> None:
    active = handoff()
    plan = GroundedWritingPlanner(FakeLLMClient(plan_response())).plan(active).confirm(
        confirmed_by="student"
    )
    packet = SectionEvidencePacketBuilder().build(active, "method")
    paragraphs = [
        DraftParagraphProposal(
            role=item.role,
            text=f"Distinct evidence-bound paragraph {item.paragraph_number}.",
            evidence_card_ids=item.evidence_card_ids,
            source_dois=item.source_dois,
        )
        for item in plan.sections[0].paragraphs
    ]
    draft = GroundedSectionDraftService().create(
        packet,
        SectionDraftProposal(section_id="method", paragraphs=paragraphs),
    ).model_copy(
        update={
            "issues": [
                SectionDraftIssue(
                    code="paragraph_repetition",
                    severity="warning",
                    paragraph_number=2,
                    detail="V0.4 deferred repeated argument.",
                ),
                SectionDraftIssue(
                    code="quality_review_deferred",
                    severity="warning",
                    detail="Local repair converged without removing the finding.",
                ),
            ]
        }
    )
    draft = quality_passed(draft)
    projects = WritingProjectService()
    project = projects.save_draft(projects.start(active), draft)
    project = projects.confirm_section(project, "method", confirmed_by="student")
    response = json.dumps(
        {
            "findings": [
                {
                    "section_id": "method",
                    "paragraph_number": 2,
                    "code": "paragraph_repetition",
                    "severity": "warning",
                    "disposition": "report_only",
                    "detail": "The repeated argument remains material in full context.",
                    "revision_instruction": "Retain only the distinct comparison.",
                }
            ]
        }
    )

    checkpoint = FullManuscriptEditorialService(
        LLMManuscriptQualityReviewer(FakeLLMClient(response))
    ).run(plan, project)

    assert checkpoint.status == "needs_revision"
    finding = checkpoint.review.findings[0]
    assert finding.code == "paragraph_repetition"
    assert finding.severity == "blocking"
    assert finding.disposition == "targeted_repair"


def test_structural_editor_merges_redundant_paragraph_and_support_scope() -> None:
    active = handoff()
    plan = GroundedWritingPlanner(FakeLLMClient(plan_response())).plan(active)
    packet = SectionEvidencePacketBuilder().build(active, "method")
    paragraphs = [
        DraftParagraphProposal(
            role=item.role,
            text=f"Evidence-bound paragraph {item.paragraph_number}.",
            evidence_card_ids=item.evidence_card_ids,
            source_dois=item.source_dois,
        )
        for item in plan.sections[0].paragraphs
    ]
    draft = GroundedSectionDraftService().create(
        packet,
        SectionDraftProposal(section_id="method", paragraphs=paragraphs),
    )
    service = WritingProjectService()
    project = service.save_draft(service.start(active), quality_passed(draft))
    project = service.confirm_section(
        project,
        "method",
        confirmed_by="student",
    )
    finding = ManuscriptQualityFinding(
        section_id="method",
        paragraph_number=3,
        code="cross_section_repetition",
        severity="blocking",
        disposition="targeted_repair",
        detail="The final paragraph repeats the preceding synthesis.",
        revision_instruction="Delete or merge the redundant paragraph.",
    )

    edited = merge_redundant_manuscript_paragraphs(
        plan,
        project,
        ManuscriptQualityReview(findings=[finding]),
    )

    assert len(edited.plan.sections[0].paragraphs) == 2
    assert len(edited.project.sections[0].draft.paragraphs) == 2
    assert edited.target_remap[("method", 3)] == ("method", 2)
    assert {
        doi
        for paragraph in edited.plan.sections[0].paragraphs
        for doi in paragraph.source_dois
    } == {
        DOI,
        SUPPORTING_DOI,
    }
    assert all(
        len(paragraph.source_dois) == 1
        for paragraph in edited.plan.sections[0].paragraphs
    )
    assert sum(
        paragraph.target_words for paragraph in edited.plan.sections[0].paragraphs
    ) == plan.sections[0].target_words
    assert edited.project.status == "drafting"


def test_v05_rejection_reopens_every_v04_paragraph_without_losing_plan() -> None:
    active = handoff()
    plan = GroundedWritingPlanner(FakeLLMClient(plan_response())).plan(active)
    plan = plan.confirm(confirmed_by="student")
    packet = SectionEvidencePacketBuilder().build(active, "method")
    draft = GroundedSectionDraftService().create(
        packet,
        SectionDraftProposal(
            section_id="method",
            paragraphs=[
                DraftParagraphProposal(
                    role=item.role,
                    text=f"Evidence-bound paragraph {item.paragraph_number}.",
                    evidence_card_ids=item.evidence_card_ids,
                    source_dois=item.source_dois,
                )
                for item in plan.sections[0].paragraphs
            ],
        ),
    )
    service = WritingProjectService()
    project = service.save_draft(service.start(active), quality_passed(draft))
    project = service.confirm_section(project, "method", confirmed_by="student")
    state = {
        "v04_writing_plan_json": plan.model_dump_json(indent=2),
        "v04_writing_project_json": project.model_dump_json(indent=2),
        "mvp_navigation": "delivery",
    }

    assert reopen_entire_body_for_regeneration(state) is True

    reopened = type(project).model_validate_json(state["v04_writing_project_json"])
    reopened_draft = reopened.sections[0].draft
    assert reopened.status == "drafting"
    assert reopened.sections[0].status == "needs_review"
    assert reopened_draft is not None
    assert len(reopened_draft.issues) == len(reopened_draft.paragraphs)
    assert {issue.paragraph_number for issue in reopened_draft.issues} == {1, 2, 3}
    assert state["v04_writing_plan_json"] == plan.model_dump_json(indent=2)
    assert state["mvp_navigation_request"] == "writing"
    assert state["v04_autopilot_requested"] is True
    assert "mvp_final_repair_checkpoint_json" in state


def test_planner_injects_independent_review_feedback_into_replanning_prompt() -> None:
    client = FakeLLMClient(plan_response())
    feedback = {"method": ["第 2 段 [topic_drift]：偏离章节研究对象。"]}

    GroundedWritingPlanner(
        client,
        repair_feedback_by_section=feedback,
    ).plan(handoff())

    prompt_payload = json.loads(client.calls[0]["messages"][1]["content"])
    assert prompt_payload["repair_feedback"] == feedback["method"]
    assert "mandatory diagnosis" in client.calls[0]["messages"][0]["content"]


def test_section_packet_rejects_legacy_unreviewed_literature() -> None:
    active = handoff()
    legacy_records = [
        record.model_copy(
            update={
                "admission_status": "legacy_unreviewed",
                "centrality": "peripheral",
                "supported_claim": None,
                "suitable_section_id": None,
                "use_boundary": None,
            }
        )
        for record in active.evidence_library.records
    ]
    legacy = active.model_copy(
        update={
            "evidence_library": active.evidence_library.model_copy(
                update={"records": legacy_records}
            )
        }
    )

    with pytest.raises(GroundedWritingError, match="topic admission"):
        SectionEvidencePacketBuilder().build(legacy, "method")


def test_planner_downgrades_comparison_move_when_only_one_source_is_locked() -> None:
    payload = json.loads(plan_response())
    payload["paragraphs"][2]["source_refs"] = []

    plan = GroundedWritingPlanner(FakeLLMClient(json.dumps(payload))).plan(handoff())

    paragraph = plan.sections[0].paragraphs[2]
    assert paragraph.source_dois == [DOI]
    assert paragraph.argument_move == "frame_problem"
    assert paragraph.comparison_axis is None


def test_planner_assigns_omitted_required_source_to_existing_problem_paragraph() -> None:
    payload = json.loads(plan_response())
    for paragraph in payload["paragraphs"]:
        paragraph["source_refs"] = []

    plan = GroundedWritingPlanner(FakeLLMClient(json.dumps(payload))).plan(handoff())

    assert SUPPORTING_DOI in {
        doi
        for paragraph in plan.sections[0].paragraphs
        for doi in paragraph.source_dois
    }
    assert len(plan.sections[0].paragraphs) == 3


def test_planner_repairs_once_then_rejects_unknown_aliases() -> None:
    client = SequenceLLMClient(
        [
            plan_response(evidence_ref="E999"),
            plan_response(evidence_ref="E999"),
        ]
    )

    with pytest.raises(WritingPlanError, match="unknown short"):
        GroundedWritingPlanner(client).plan(handoff())

    assert len(client.calls) == 2


def test_planner_drops_unneeded_low_permission_source_from_supported_paragraph() -> None:
    active_handoff = handoff_with_background_source()
    payload = json.loads(plan_response())
    payload["paragraphs"][0]["source_refs"] = ["S003"]
    client = FakeLLMClient(json.dumps(payload))

    plan = GroundedWritingPlanner(client).plan(active_handoff)

    assert plan.sections[0].paragraphs[0].source_dois == [DOI]


def test_planner_repairs_background_source_used_for_section_support() -> None:
    active_handoff = handoff_with_background_source()
    invalid_payload = json.loads(plan_response())
    invalid_payload["paragraphs"][1]["source_refs"] = ["S003"]
    valid_payload = json.loads(plan_response())
    valid_payload["paragraphs"][1].update(
        {
            "role": "background",
            "source_refs": ["S003"],
        }
    )
    client = SequenceLLMClient(
        [json.dumps(invalid_payload), json.dumps(valid_payload)]
    )

    plan = GroundedWritingPlanner(client).plan(active_handoff)

    repaired = plan.sections[0].paragraphs[1]
    assert repaired.role == "background"
    assert repaired.source_dois == [BACKGROUND_DOI]
    assert len(client.calls) == 2
    prompt_payload = json.loads(client.calls[0]["messages"][1]["content"])
    assert prompt_payload["source_catalog"][2]["allowed_roles"] == [
        "background",
        "synthesis",
    ]
    assert "S003" not in prompt_payload["allowed_support_refs_by_role"][
        "section_support"
    ]["source_refs"]
    assert "S003" in prompt_payload["allowed_support_refs_by_role"]["background"][
        "source_refs"
    ]


def test_planner_deterministically_replaces_repeated_permission_mismatch() -> None:
    active_handoff = handoff_with_background_source()
    invalid_payload = json.loads(plan_response())
    invalid_payload["paragraphs"][1]["source_refs"] = ["S003"]
    repeated = json.dumps(invalid_payload)
    client = SequenceLLMClient([repeated, repeated])

    plan = GroundedWritingPlanner(client).plan(active_handoff)

    repaired = plan.sections[0].paragraphs[1]
    assert repaired.role == "section_support"
    assert repaired.source_dois
    assert BACKGROUND_DOI not in repaired.source_dois
    assert len(client.calls) == 2


def test_replanner_trims_repeated_excess_paragraph_plan() -> None:
    payload = json.loads(plan_response())
    payload["paragraphs"].insert(
        2,
        {
            "role": "section_support",
            "purpose": "Repeat a lower-priority contextual bridge.",
            "claim_focus": "The supporting record provides section context.",
            "central_question": "What context frames the retrieval problem?",
            "argument_move": "evaluate_limitation",
            "comparison_axis": "scope of application",
            "relative_weight": 1,
            "evidence_refs": [],
            "source_refs": ["S002"],
        },
    )
    repeated = json.dumps(payload)
    client = SequenceLLMClient([repeated, repeated])
    packet = SectionEvidencePacketBuilder().build(handoff(), "method")

    section = GroundedWritingPlanner(client).replan_section(
        packet,
        paragraph_count=3,
    )

    assert len(section.paragraphs) == 3
    assert len(client.calls) == 2


def test_compiled_plan_preserves_required_source_over_optional_metadata() -> None:
    active_handoff = handoff_with_background_source()
    plan = GroundedWritingPlanner(FakeLLMClient(plan_response())).plan(
        active_handoff
    )
    packet = SectionEvidencePacketBuilder().build(active_handoff, "method").model_copy(
        update={
            "required_source_dois": [BACKGROUND_DOI],
            "max_sources_per_paragraph": 2,
        }
    )
    paragraphs = list(plan.sections[0].paragraphs)
    paragraphs[2] = paragraphs[2].model_copy(
        update={
            "evidence_card_ids": [EVIDENCE_ID],
            "source_dois": [DOI, SUPPORTING_DOI],
        }
    )

    repaired = _repair_compiled_required_source_coverage(packet, paragraphs)

    assert repaired[2].source_dois == [DOI, BACKGROUND_DOI]
    assert repaired[2].evidence_card_ids == [EVIDENCE_ID]


def test_compiled_plan_replaces_duplicate_required_occurrence_for_missing_source() -> None:
    active_handoff = handoff_with_background_source()
    plan = GroundedWritingPlanner(FakeLLMClient(plan_response())).plan(
        active_handoff
    )
    packet = SectionEvidencePacketBuilder().build(active_handoff, "method").model_copy(
        update={
            "required_source_dois": [DOI, SUPPORTING_DOI, BACKGROUND_DOI],
            "max_sources_per_paragraph": 2,
        }
    )
    paragraphs = list(plan.sections[0].paragraphs)
    paragraphs[0] = paragraphs[0].model_copy(
        update={
            "evidence_card_ids": [EVIDENCE_ID],
            "source_dois": [DOI, SUPPORTING_DOI],
        }
    )
    paragraphs[1] = paragraphs[1].model_copy(
        update={"source_dois": [DOI, SUPPORTING_DOI]}
    )
    paragraphs[2] = paragraphs[2].model_copy(
        update={
            "evidence_card_ids": [],
            "source_dois": [DOI, SUPPORTING_DOI],
        }
    )

    repaired = _repair_compiled_required_source_coverage(packet, paragraphs)

    covered = {doi for paragraph in repaired for doi in paragraph.source_dois}
    assert covered == {DOI, SUPPORTING_DOI, BACKGROUND_DOI}
    assert BACKGROUND_DOI in repaired[2].source_dois
    assert len(repaired[2].source_dois) == 2


def test_compiled_plan_repurposes_redundant_detail_for_background_source() -> None:
    active_handoff = handoff_with_background_source()
    plan = GroundedWritingPlanner(FakeLLMClient(plan_response())).plan(
        active_handoff
    )
    packet = SectionEvidencePacketBuilder().build(active_handoff, "method").model_copy(
        update={
            "required_source_dois": [DOI, SUPPORTING_DOI, BACKGROUND_DOI],
            "max_sources_per_paragraph": 1,
        }
    )
    paragraphs = list(plan.sections[0].paragraphs)
    paragraphs[0] = paragraphs[0].model_copy(
        update={
            "role": "detailed_evidence",
            "evidence_card_ids": [EVIDENCE_ID],
            "source_dois": [DOI],
        }
    )
    paragraphs[1] = paragraphs[1].model_copy(
        update={
            "role": "detailed_evidence",
            "evidence_card_ids": [EVIDENCE_ID],
            "source_dois": [DOI],
        }
    )
    paragraphs[2] = paragraphs[2].model_copy(
        update={
            "role": "synthesis",
            "evidence_card_ids": [],
            "source_dois": [SUPPORTING_DOI],
        }
    )

    repaired = _repair_compiled_required_source_coverage(packet, paragraphs)

    covered = {doi for paragraph in repaired for doi in paragraph.source_dois}
    assert covered == {DOI, SUPPORTING_DOI, BACKGROUND_DOI}
    assert len(repaired) == len(paragraphs)
    assert repaired[1].role == "background"
    assert repaired[1].evidence_card_ids == []
    assert repaired[1].source_dois == [BACKGROUND_DOI]
    assert repaired[1].coverage_only is False


def test_planner_covers_every_source_required_by_reference_policy() -> None:
    active_handoff = handoff_with_background_source()
    requirement_spec = active_handoff.requirement.requirement.model_copy(
        update={
            "references": ReferenceRequirement(
                minimum_total=3,
                target_total=3,
                bibliography_style="APA 7th",
                max_references_per_citation_cluster=4,
                all_bibliography_items_must_be_cited_and_discussed=True,
            )
        }
    )
    confirmed_requirement = active_handoff.requirement.model_copy(
        update={"requirement": requirement_spec}
    )
    policy = RequirementPolicyCompiler(current_year=2026).compile(
        confirmed_requirement
    )
    active_handoff = active_handoff.model_copy(
        update={
            "requirement": confirmed_requirement,
            "requirement_policy": policy,
        }
    )

    payload = json.loads(plan_response())
    payload["paragraphs"][2]["source_refs"].append("S003")
    plan = GroundedWritingPlanner(FakeLLMClient(json.dumps(payload))).plan(
        active_handoff
    )

    planned_dois = {
        doi
        for section in plan.sections
        for paragraph in section.paragraphs
        for doi in paragraph.source_dois
    }
    assert set(plan.required_source_dois) == {
        DOI,
        SUPPORTING_DOI,
        BACKGROUND_DOI,
    }
    assert planned_dois == set(plan.required_source_dois)
    assert max(
        len(paragraph.source_dois)
        for section in plan.sections
        for paragraph in section.paragraphs
    ) <= 4
    assert all(
        paragraph.coverage_only is False
        for section in plan.sections
        for paragraph in section.paragraphs
    )
    assert all(
        "coverage policy" not in paragraph.purpose.casefold()
        for section in plan.sections
        for paragraph in section.paragraphs
    )


def test_source_coverage_repair_uses_existing_problem_paragraphs() -> None:
    active_handoff = handoff_with_background_source()
    legacy = GroundedWritingPlanner(FakeLLMClient(plan_response())).plan(
        active_handoff
    )
    requirement_spec = active_handoff.requirement.requirement.model_copy(
        update={
            "references": ReferenceRequirement(
                minimum_total=3,
                target_total=3,
                bibliography_style="APA 7th",
                max_references_per_citation_cluster=4,
                all_bibliography_items_must_be_cited_and_discussed=True,
            )
        }
    )
    confirmed_requirement = active_handoff.requirement.model_copy(
        update={"requirement": requirement_spec}
    )
    active_handoff = active_handoff.model_copy(
        update={
            "requirement": confirmed_requirement,
            "requirement_policy": RequirementPolicyCompiler(current_year=2026).compile(
                confirmed_requirement
            ),
        }
    )
    repair = repair_writing_plan_source_coverage(active_handoff, legacy)

    assert len(repair.plan.sections[0].paragraphs) == len(
        legacy.sections[0].paragraphs
    )
    assert BACKGROUND_DOI in {
        doi
        for paragraph in repair.plan.sections[0].paragraphs
        for doi in paragraph.source_dois
    }
    assert all(
        paragraph.coverage_only is False
        for paragraph in repair.plan.sections[0].paragraphs
    )


def test_planner_routes_unlisted_policy_required_source_without_replanning() -> None:
    active_handoff = handoff_with_background_source()
    section = active_handoff.outline.outline.sections[0].model_copy(
        update={"supporting_dois": [SUPPORTING_DOI]}
    )
    active_handoff = active_handoff.model_copy(
        update={
            "outline": active_handoff.outline.model_copy(
                update={
                    "outline": active_handoff.outline.outline.model_copy(
                        update={"sections": [section]}
                    )
                }
            )
        }
    )
    requirement_spec = active_handoff.requirement.requirement.model_copy(
        update={
            "references": ReferenceRequirement(
                minimum_total=3,
                target_total=3,
                bibliography_style="APA 7th",
                max_references_per_citation_cluster=4,
                all_bibliography_items_must_be_cited_and_discussed=True,
            )
        }
    )
    confirmed_requirement = active_handoff.requirement.model_copy(
        update={"requirement": requirement_spec}
    )
    active_handoff = active_handoff.model_copy(
        update={
            "requirement": confirmed_requirement,
            "requirement_policy": RequirementPolicyCompiler(current_year=2026).compile(
                confirmed_requirement
            ),
        }
    )
    planning_packet = SectionEvidencePacketBuilder().build(
        active_handoff,
        "method",
        include_policy_required_routes=False,
    )
    writing_packet = SectionEvidencePacketBuilder().build(active_handoff, "method")
    client = FakeLLMClient(plan_response())

    plan = GroundedWritingPlanner(client).plan(active_handoff)

    assert BACKGROUND_DOI not in {source.doi for source in planning_packet.sources}
    assert BACKGROUND_DOI in {source.doi for source in writing_packet.sources}
    assert BACKGROUND_DOI in {
        doi
        for paragraph in plan.sections[0].paragraphs
        for doi in paragraph.source_dois
    }
    assert len(client.calls) == 1


def test_source_coverage_repair_accepts_an_explicit_recovery_bibliography() -> None:
    active_handoff = handoff_with_background_source()
    plan = GroundedWritingPlanner(FakeLLMClient(plan_response())).plan(
        active_handoff
    )

    repair = repair_writing_plan_source_coverage(
        active_handoff,
        plan,
        required_source_dois=[DOI, SUPPORTING_DOI],
    )

    assert repair.plan.required_source_dois == [DOI, SUPPORTING_DOI]
    assert BACKGROUND_DOI not in repair.plan.required_source_dois


def test_authority_rebase_removes_sources_that_left_the_current_handoff() -> None:
    previous_handoff = handoff_with_background_source()
    requirement_spec = previous_handoff.requirement.requirement.model_copy(
        update={
            "references": ReferenceRequirement(
                minimum_total=3,
                target_total=3,
                bibliography_style="APA 7th",
                max_references_per_citation_cluster=4,
                all_bibliography_items_must_be_cited_and_discussed=True,
            )
        }
    )
    confirmed_requirement = previous_handoff.requirement.model_copy(
        update={"requirement": requirement_spec}
    )
    previous_handoff = previous_handoff.model_copy(
        update={
            "requirement": confirmed_requirement,
            "requirement_policy": RequirementPolicyCompiler(current_year=2026).compile(
                confirmed_requirement
            ),
        }
    )
    payload = json.loads(plan_response())
    payload["paragraphs"][2]["source_refs"].append("S003")
    previous_plan = GroundedWritingPlanner(
        FakeLLMClient(json.dumps(payload))
    ).plan(previous_handoff)
    current_section = previous_handoff.outline.outline.sections[0].model_copy(
        update={"supporting_dois": [SUPPORTING_DOI]}
    )
    current_handoff = previous_handoff.model_copy(
        update={
            "outline": previous_handoff.outline.model_copy(
                update={
                    "outline": previous_handoff.outline.outline.model_copy(
                        update={"sections": [current_section]}
                    )
                }
            ),
            "evidence_library": previous_handoff.evidence_library.model_copy(
                update={
                    "records": [
                        record
                        for record in previous_handoff.evidence_library.records
                        if record.doi != BACKGROUND_DOI
                    ]
                }
            ),
        }
    )

    repair = rebase_writing_plan_authority(current_handoff, previous_plan)

    assert repair.plan.required_source_dois == [DOI, SUPPORTING_DOI]
    assert BACKGROUND_DOI not in {
        doi
        for section in repair.plan.sections
        for paragraph in section.paragraphs
        for doi in paragraph.source_dois
    }
    assert repair.changed_paragraph_numbers == {"method": (3,)}


def test_handoff_sync_preserves_draft_and_reopens_only_rebased_paragraph() -> None:
    previous_handoff = handoff_with_background_source()
    previous_plan = GroundedWritingPlanner(
        FakeLLMClient(plan_response())
    ).plan(previous_handoff).confirm(confirmed_by="student")
    # Put the removable metadata source on one synthesis paragraph without making it
    # the paragraph's only authority.
    section = previous_plan.sections[0]
    paragraphs = list(section.paragraphs)
    paragraphs[2] = paragraphs[2].model_copy(
        update={
            "source_dois": [*paragraphs[2].source_dois, BACKGROUND_DOI]
        }
    )
    previous_plan = GroundedWritingPlan.model_validate(
        previous_plan.model_copy(
            update={"sections": [section.model_copy(update={"paragraphs": paragraphs})]}
        ).model_dump(mode="json")
    )
    packet = SectionEvidencePacketBuilder().build(previous_handoff, "method")
    draft = GroundedSectionDraftService().create(
        packet,
        SectionDraftProposal(
            section_id="method",
            paragraphs=[
                DraftParagraphProposal(
                    role=item.role,
                    text=f"Evidence-bound paragraph {item.paragraph_number}.",
                    evidence_card_ids=item.evidence_card_ids,
                    source_dois=item.source_dois,
                )
                for item in previous_plan.sections[0].paragraphs
            ],
        ),
    )
    projects = WritingProjectService()
    project = projects.save_draft(projects.start(previous_handoff), quality_passed(draft))
    project = projects.confirm_section(project, "method", confirmed_by="student")
    current_section = previous_handoff.outline.outline.sections[0].model_copy(
        update={"supporting_dois": [SUPPORTING_DOI]}
    )
    current_handoff = previous_handoff.model_copy(
        update={
            "outline": previous_handoff.outline.model_copy(
                update={
                    "outline": previous_handoff.outline.outline.model_copy(
                        update={"sections": [current_section]}
                    )
                }
            ),
            "evidence_library": previous_handoff.evidence_library.model_copy(
                update={
                    "records": [
                        record
                        for record in previous_handoff.evidence_library.records
                        if record.doi != BACKGROUND_DOI
                    ]
                }
            ),
        }
    )
    current_plan = rebase_writing_plan_authority(
        current_handoff,
        previous_plan,
    ).plan

    synchronized = _synchronize_project_handoff(
        project,
        previous_plan=previous_plan,
        current_plan=current_plan,
        handoff=current_handoff,
    )

    state = synchronized.sections[0]
    assert state.status == "needs_review"
    assert state.draft is not None
    assert state.draft.paragraphs == project.sections[0].draft.paragraphs
    assert [
        issue.paragraph_number
        for issue in state.draft.issues
        if issue.code == "final_audit_repair"
    ] == [3]


def test_paragraph_writer_cannot_return_self_selected_evidence_ids() -> None:
    active_handoff = handoff()
    plan = GroundedWritingPlanner(FakeLLMClient(plan_response())).plan(active_handoff)
    section_packet = SectionEvidencePacketBuilder().build(active_handoff, "method")
    paragraph_packet = ParagraphEvidencePacketBuilder().build(
        section_packet,
        plan.sections[0].paragraphs[0],
    )
    client = FakeLLMClient(
        json.dumps(
            {
                "text": "The locked evidence supports the claim.",
                "evidence_card_ids": ["ev_invented_001"],
            }
        )
    )

    with pytest.raises(GroundedWritingError, match="paragraph output"):
        LLMGroundedParagraphWriter(client).write(paragraph_packet)
    assert len(client.calls) == 2


def test_chinese_language_contract_flows_to_plan_packet_and_repairs_prose() -> None:
    active_handoff = chinese_handoff()
    plan = GroundedWritingPlanner(FakeLLMClient(plan_response())).plan(active_handoff)
    section_packet = SectionEvidencePacketBuilder().build(active_handoff, "method")
    paragraph_packet = ParagraphEvidencePacketBuilder().build(
        section_packet,
        plan.sections[0].paragraphs[0],
    )
    client = SequenceLLMClient(
        [
            json.dumps(
                {
                    "text": (
                        "This paragraph is written entirely in English and therefore "
                        "violates the confirmed Chinese output language."
                    )
                }
            ),
            json.dumps(
                {
                    "text": (
                        "现有证据表明，该反演模型能够降低研究区域的不确定性，"
                        "因此可为区域大气遥感反演流程提供直接的方法依据。"
                    )
                },
                ensure_ascii=False,
            ),
        ]
    )

    result = LLMGroundedParagraphWriter(client).write(paragraph_packet)

    assert plan.output_language == "Chinese"
    assert section_packet.output_language == "Chinese"
    assert paragraph_packet.output_language == "Chinese"
    assert result.text.startswith("现有证据表明")
    assert len(client.calls) == 2
    assert "natural academic Chinese" in client.calls[0]["messages"][0]["content"]


def test_review_writer_repairs_false_self_attribution_with_source_authors() -> None:
    active_handoff = chinese_handoff()
    plan = GroundedWritingPlanner(FakeLLMClient(plan_response())).plan(active_handoff)
    section_packet = SectionEvidencePacketBuilder().build(active_handoff, "method")
    paragraph_packet = ParagraphEvidencePacketBuilder().build(
        section_packet,
        plan.sections[0].paragraphs[0],
    )
    client = SequenceLLMClient(
        [
            json.dumps(
                {"text": "本文提出了一种新的大气反演方法，并完成了实验验证。"},
                ensure_ascii=False,
            ),
            json.dumps(
                {
                    "text": (
                        "Smith 等提出了该大气反演方法，并在锁定证据所述研究区域"
                        "完成验证；该结果为本综述比较现有方法提供了依据。"
                    )
                },
                ensure_ascii=False,
            ),
        ]
    )

    result = LLMGroundedParagraphWriter(client).write(paragraph_packet)

    assert result.text.startswith("Smith 等提出")
    assert len(client.calls) == 2
    assert "literature review" in client.calls[0]["messages"][0]["content"]
    assert "source author names" in client.calls[1]["messages"][-1]["content"]


def test_cached_plan_migration_adds_language_and_marks_legacy_coverage() -> None:
    active_handoff = chinese_handoff()
    plan = GroundedWritingPlanner(FakeLLMClient(plan_response())).plan(active_handoff)
    payload = plan.model_dump(mode="json")
    payload["output_language"] = "pending_confirmation"
    payload["plan_fingerprint"] = "c" * 64
    payload["sections"][0]["paragraphs"][1]["purpose"] = (
        "Map the scope of additional verified literature required by the "
        "bibliography coverage policy."
    )
    legacy = type(plan).model_validate(payload)

    migrated = align_writing_plan_language(active_handoff, legacy)

    assert migrated.output_language == "Chinese"
    assert migrated.sections[0].paragraphs[1].coverage_only is True
    assert migrated.plan_fingerprint != legacy.plan_fingerprint


def test_draft_language_audit_blocks_english_prose_in_chinese_project() -> None:
    packet = SectionEvidencePacketBuilder().build(chinese_handoff(), "method")

    draft = GroundedSectionDraftService().create(
        packet,
        proposal(
            text=(
                "This paragraph remains entirely in English even though the confirmed "
                "course-paper output language is Chinese."
            )
        ),
    )

    assert draft.status == "needs_review"
    assert "language_mismatch" in [issue.code for issue in draft.issues]


def test_chapter_quality_reviewer_creates_targeted_editorial_issue() -> None:
    active_handoff = handoff()
    plan = GroundedWritingPlanner(FakeLLMClient(plan_response())).plan(active_handoff)
    packet = SectionEvidencePacketBuilder().build(active_handoff, "method")
    draft = GroundedSectionDraftService().create(packet, proposal())
    response = json.dumps(
        {
            "section_id": "method",
            "findings": [
                {
                    "paragraph_number": 1,
                    "code": "overstated_evidence",
                    "detail": "The causal wording is stronger than the locked result.",
                    "revision_instruction": "Use associative and qualified wording.",
                    "claim_kind": "evidence_fact",
                    "evidence_card_ids": [EVIDENCE_ID],
                }
            ],
        }
    )

    review = LLMSectionQualityReviewer(FakeLLMClient(response)).review(
        plan.sections[0],
        draft,
        packet,
        output_language="English",
    )
    reviewed = apply_section_quality_review(draft, review)

    assert reviewed.issues[-1].code == "overstated_evidence"
    assert reviewed.issues[-1].severity == "blocking"
    assert reviewed.status == "needs_review"
    assert reviewed.issues[-1].paragraph_number == 1
    assert "qualified wording" in reviewed.issues[-1].detail


def test_chapter_quality_reviewer_deduplicates_repeated_findings() -> None:
    active_handoff = handoff()
    plan = GroundedWritingPlanner(FakeLLMClient(plan_response())).plan(active_handoff)
    packet = SectionEvidencePacketBuilder().build(active_handoff, "method")
    draft = GroundedSectionDraftService().create(packet, proposal())
    finding = {
        "paragraph_number": 1,
        "code": "topic_drift",
        "severity": "warning",
        "detail": "The paragraph gives too much space to context.",
        "revision_instruction": "Keep the chapter's central question in focus.",
    }
    client = FakeLLMClient(
        json.dumps({"section_id": "method", "findings": [finding, finding]})
    )

    review = LLMSectionQualityReviewer(client).review(
        plan.sections[0],
        draft,
        packet,
        output_language="English",
    )

    assert len(review.findings) == 1
    assert review.findings[0].code == "topic_drift"
    assert len(client.calls) == 1


def test_chapter_quality_reviewer_drops_out_of_scope_evidence_ids() -> None:
    active_handoff = handoff()
    plan = GroundedWritingPlanner(FakeLLMClient(plan_response())).plan(active_handoff)
    packet = SectionEvidencePacketBuilder().build(active_handoff, "method")
    draft = GroundedSectionDraftService().create(packet, proposal())
    response = json.dumps(
        {
            "section_id": "method",
            "findings": [
                {
                    "paragraph_number": 1,
                    "code": "overstated_evidence",
                    "detail": "The wording is too strong.",
                    "revision_instruction": "Qualify the sentence.",
                    "claim_kind": "evidence_fact",
                    "evidence_card_ids": ["ev_outside_locked_scope"],
                }
            ],
        }
    )

    review = LLMSectionQualityReviewer(FakeLLMClient(response)).review(
        plan.sections[0],
        draft,
        packet,
        output_language="English",
    )

    assert review.findings[0].evidence_card_ids == []


def test_chapter_quality_reviewer_drops_false_author_attribution_alarm() -> None:
    active_handoff = handoff()
    plan = GroundedWritingPlanner(FakeLLMClient(plan_response())).plan(active_handoff)
    packet = SectionEvidencePacketBuilder().build(active_handoff, "method")
    draft = GroundedSectionDraftService().create(
        packet,
        proposal(text="Smith et al. proposed the evidence-bound retrieval method."),
    )
    response = json.dumps(
        {
            "section_id": "method",
            "findings": [
                {
                    "paragraph_number": 1,
                    "code": "false_self_attribution",
                    "severity": "blocking",
                    "detail": "The verb proposed may look like self attribution.",
                    "revision_instruction": "Use the cited author as subject.",
                }
            ],
        }
    )

    review = LLMSectionQualityReviewer(FakeLLMClient(response)).review(
        plan.sections[0],
        draft,
        packet,
        output_language="English",
    )

    assert review.findings == []


def test_quality_review_reopens_needs_review_draft_when_only_warnings_remain() -> None:
    active_handoff = handoff()
    plan = GroundedWritingPlanner(FakeLLMClient(plan_response())).plan(active_handoff)
    packet = SectionEvidencePacketBuilder().build(active_handoff, "method")
    draft = GroundedSectionDraftService().create(packet, proposal()).model_copy(
        update={
            "status": "needs_review",
            "issues": [
                SectionDraftIssue(
                    code="unsupported_claim",
                    severity="blocking",
                    detail="An earlier editorial review blocked this draft.",
                    paragraph_number=1,
                )
            ],
        }
    )
    response = json.dumps(
        {
            "section_id": "method",
            "findings": [
                {
                    "paragraph_number": 1,
                    "code": "academic_style_problem",
                    "detail": "The sentence is verbose.",
                    "revision_instruction": "Make it concise.",
                }
            ],
        }
    )

    review = LLMSectionQualityReviewer(FakeLLMClient(response)).review(
        plan.sections[0],
        draft,
        packet,
        output_language="English",
    )
    reviewed = apply_section_quality_review(draft, review)

    assert reviewed.status == "draft"
    assert reviewed.quality_review_status == "passed"
    assert all(issue.severity == "warning" for issue in reviewed.issues)


def test_continuous_writing_confirms_a_clean_independently_reviewed_section() -> None:
    active_handoff = handoff()
    plan = GroundedWritingPlanner(FakeLLMClient(plan_response())).plan(
        active_handoff
    ).confirm(confirmed_by="student")
    paragraph_text = " ".join(["grounded"] * 100)
    writer_client = FakeLLMClient(
        json.dumps({"text": paragraph_text})
    )
    reviewer_client = FakeLLMClient(
        json.dumps({"section_id": "method", "findings": []})
    )
    project = WritingProjectService().start(active_handoff)
    checkpoints = []

    result = ContinuousSectionWritingService(
        writer=LLMGroundedParagraphWriter(writer_client),
        reviewer=LLMSectionQualityReviewer(reviewer_client),
    ).run(
        project,
        plan,
        confirmed_by="student (continuous authorization)",
        on_checkpoint=lambda active, event: checkpoints.append(
            (active.status, event.stage)
        ),
    )

    draft = result.project.sections[0].draft
    assert result.completed
    assert result.stopped_section_id is None
    assert draft is not None
    assert draft.status == "confirmed"
    assert draft.quality_review_status == "passed"
    assert draft.quality_review_rounds == 1
    assert len(writer_client.calls) == 3
    assert len(reviewer_client.calls) == 1
    assert ("body_complete", "confirmed") in checkpoints


def test_continuous_writing_degrades_reviewer_contract_failure_without_stopping() -> None:
    active_handoff = handoff()
    plan = GroundedWritingPlanner(FakeLLMClient(plan_response())).plan(
        active_handoff
    ).confirm(confirmed_by="student")
    writer_client = FakeLLMClient(
        json.dumps({"text": " ".join(["grounded"] * 100)})
    )
    reviewer_client = SequenceLLMClient(["not-json", "still-not-json"])

    result = ContinuousSectionWritingService(
        writer=LLMGroundedParagraphWriter(writer_client),
        reviewer=LLMSectionQualityReviewer(reviewer_client),
    ).run(
        WritingProjectService().start(active_handoff),
        plan,
        confirmed_by="student (continuous authorization)",
    )

    draft = result.project.sections[0].draft
    assert result.completed
    assert result.stopped_section_id is None
    assert draft is not None
    assert draft.status == "confirmed"
    assert draft.quality_review_status == "passed"
    assert any(issue.code == "quality_review_degraded" for issue in draft.issues)
    assert len(reviewer_client.calls) == 2


def test_continuous_writing_routes_evidence_gap_before_calling_llms() -> None:
    active_handoff = handoff_with_background_source()
    draft_plan = GroundedWritingPlanner(FakeLLMClient(plan_response())).plan(
        active_handoff
    )
    section = draft_plan.sections[0]
    background_comparison = section.paragraphs[1].model_copy(
        update={
            "role": "background",
            "purpose": "Compare model architectures and accuracy.",
            "claim_focus": "Compare the performance of two retrieval models.",
            "central_question": "Which model is more accurate and why?",
            "argument_move": "compare_studies",
            "comparison_axis": "accuracy and architecture",
            "evidence_card_ids": [],
            "source_dois": [BACKGROUND_DOI],
        }
    )
    section = section.model_copy(
        update={
            "paragraphs": [
                section.paragraphs[0],
                background_comparison,
                section.paragraphs[2],
            ]
        }
    )
    plan = GroundedWritingPlan.model_validate(
        draft_plan.model_copy(update={"sections": [section]}).model_dump(mode="json")
    ).confirm(confirmed_by="student")
    writer_client = FakeLLMClient(json.dumps({"text": "should not be called"}))
    reviewer_client = FakeLLMClient(
        json.dumps({"section_id": "method", "findings": []})
    )

    result = ContinuousSectionWritingService(
        writer=LLMGroundedParagraphWriter(writer_client),
        reviewer=LLMSectionQualityReviewer(reviewer_client),
    ).run(
        WritingProjectService().start(active_handoff),
        plan,
        confirmed_by="student",
    )

    assert result.stop_code == "evidence_gap"
    assert result.recovery_request is not None
    assert result.recovery_request.requested_core_dois == [BACKGROUND_DOI]
    assert writer_client.calls == []
    assert reviewer_client.calls == []


def test_continuous_writing_batches_evidence_gaps_from_all_pending_sections() -> None:
    active_handoff = handoff_with_background_source()
    first_outline = active_handoff.outline.outline.sections[0]
    first_background = next(
        record
        for record in active_handoff.evidence_library.records
        if record.doi == BACKGROUND_DOI
    )
    second_background_doi = "10.1000/background.2"
    second_background = first_background.model_copy(
        update={
            "doi": second_background_doi,
            "title": "Second metadata-only background record",
            "source_url": f"https://doi.org/{second_background_doi}",
        }
    )
    second_outline = first_outline.model_copy(
        update={
            "section_id": "discussion",
            "title": "Evidence synthesis",
            "purpose": "Synthesize the verified retrieval evidence.",
            "research_questions": ["What conclusion follows from the evidence?"],
            "supporting_dois": [SUPPORTING_DOI, second_background_doi],
        }
    )
    outline = active_handoff.outline.outline.model_copy(
        update={
            "target_words": 600,
            "sections": [first_outline, second_outline],
        }
    )
    active_handoff = active_handoff.model_copy(
        update={
            "outline": active_handoff.outline.model_copy(update={"outline": outline}),
            "evidence_library": active_handoff.evidence_library.model_copy(
                update={
                    "records": [
                        *active_handoff.evidence_library.records,
                        second_background,
                    ]
                }
            ),
        }
    )

    discussion_response = json.loads(plan_response())
    discussion_response["section_id"] = "discussion"
    draft_plan = GroundedWritingPlanner(
        ScriptedLLMClient([plan_response(), json.dumps(discussion_response)])
    ).plan(active_handoff)
    gap_sections = []
    for section, background_doi in zip(
        draft_plan.sections,
        (BACKGROUND_DOI, second_background_doi),
        strict=True,
    ):
        comparison = section.paragraphs[1].model_copy(
            update={
                "role": "background",
                "purpose": "Compare model architectures and accuracy.",
                "claim_focus": "Compare the performance of two retrieval models.",
                "central_question": "Which model is more accurate and why?",
                "argument_move": "compare_studies",
                "comparison_axis": "accuracy and architecture",
                "evidence_card_ids": [],
                "source_dois": [background_doi],
            }
        )
        gap_sections.append(
            section.model_copy(
                update={
                    "paragraphs": [
                        section.paragraphs[0],
                        comparison,
                        section.paragraphs[2],
                    ]
                }
            )
        )
    plan = GroundedWritingPlan.model_validate(
        draft_plan.model_copy(update={"sections": gap_sections}).model_dump(mode="json")
    ).confirm(confirmed_by="student")
    writer = FakeLLMClient(json.dumps({"text": "must not be called"}))
    reviewer = FakeLLMClient(
        json.dumps({"section_id": "method", "findings": []})
    )

    result = ContinuousSectionWritingService(
        writer=LLMGroundedParagraphWriter(writer),
        reviewer=LLMSectionQualityReviewer(reviewer),
    ).run(
        WritingProjectService().start(active_handoff),
        plan,
        confirmed_by="student",
    )

    assert result.stop_code == "evidence_gap"
    assert result.recovery_request is not None
    assert result.recovery_request.affected_section_ids == ["method", "discussion"]
    assert result.recovery_request.requested_core_dois == [
        BACKGROUND_DOI,
        second_background_doi,
    ]
    assert writer.calls == []
    assert reviewer.calls == []


def test_non_prose_deterministic_blocker_does_not_call_writer_or_reviewer() -> None:
    active_handoff = handoff()
    plan = GroundedWritingPlanner(FakeLLMClient(plan_response())).plan(
        active_handoff
    ).confirm(confirmed_by="student")
    packet = SectionEvidencePacketBuilder().build(active_handoff, "method")
    blocked = GroundedSectionDraftService().create(
        packet,
        proposal(evidence_ids=["ev_missing_from_packet"]),
    )
    projects = WritingProjectService()
    project = projects.save_draft(projects.start(active_handoff), blocked)
    writer = FakeLLMClient(json.dumps({"text": "must not be called"}))
    reviewer = FakeLLMClient(json.dumps({"section_id": "method", "findings": []}))

    result = ContinuousSectionWritingService(
        writer=LLMGroundedParagraphWriter(writer),
        reviewer=LLMSectionQualityReviewer(reviewer),
    ).run(
        project,
        plan,
        confirmed_by="student (continuous authorization)",
    )

    assert result.stop_code == "deterministic_blocked"
    assert "unknown_evidence_card" in (result.stop_reason or "")
    assert writer.calls == []
    assert reviewer.calls == []


def test_continuous_writing_advances_through_the_next_section() -> None:
    active_handoff = two_section_handoff()
    discussion_plan = json.loads(plan_response())
    discussion_plan["section_id"] = "discussion"
    plan = GroundedWritingPlanner(
        SequenceLLMClient([plan_response(), json.dumps(discussion_plan)])
    ).plan(active_handoff).confirm(confirmed_by="student")
    writer_client = FakeLLMClient(
        json.dumps({"text": " ".join(["grounded"] * 100)})
    )
    reviewer_client = SequenceLLMClient(
        [
            json.dumps({"section_id": "method", "findings": []}),
            json.dumps({"section_id": "discussion", "findings": []}),
        ]
    )

    result = ContinuousSectionWritingService(
        writer=LLMGroundedParagraphWriter(writer_client),
        reviewer=LLMSectionQualityReviewer(reviewer_client),
    ).run(
        WritingProjectService().start(active_handoff),
        plan,
        confirmed_by="student (continuous authorization)",
    )

    assert result.completed
    assert [state.status for state in result.project.sections] == [
        "confirmed",
        "confirmed",
    ]
    assert len(writer_client.calls) == 6
    assert len(reviewer_client.calls) == 2


def test_scripted_fake_llm_smoke_covers_the_complete_v04_agent_path(
    tmp_path: Path,
) -> None:
    """Exercise V0.4 orchestration and persistence without any network calls."""

    active_handoff = two_section_handoff()
    discussion_plan = json.loads(plan_response())
    discussion_plan["section_id"] = "discussion"
    planner_client = ScriptedLLMClient(
        [plan_response(), json.dumps(discussion_plan)]
    )
    plan = GroundedWritingPlanner(planner_client).plan(active_handoff).confirm(
        confirmed_by="fake-smoke"
    )
    policy = RequirementPolicyCompiler(current_year=2026).compile(
        active_handoff.requirement
    )
    project = WritingProjectService().start(active_handoff)

    store = AgentRuntimeStore(tmp_path / "agent-runtime")
    runtime = WritingAgentRuntimeService(store)
    context = runtime.initialize(
        run_id="run_0123456789abcdef",
        project_id="fake-v04-smoke",
        policy=policy,
        handoff=active_handoff,
        plan=plan,
        project=project,
    )
    prepared = runtime.prepare_section_action(
        context,
        section_ids=[section.section_id for section in plan.sections],
    )

    writer_client = ScriptedLLMClient(
        [
            json.dumps(
                {
                    "text": (
                        f"Paragraph {number} presents verified retrieval evidence "
                        + " ".join(["grounded"] * 92)
                    )
                }
            )
            for number in range(1, 7)
        ]
    )
    reviewer_client = ScriptedLLMClient(
        [
            json.dumps({"section_id": "method", "findings": []}),
            json.dumps({"section_id": "discussion", "findings": []}),
        ]
    )
    checkpoint_stages: list[str] = []
    result = ContinuousSectionWritingService(
        writer=LLMGroundedParagraphWriter(writer_client),
        reviewer=LLMSectionQualityReviewer(reviewer_client),
    ).run(
        project,
        plan,
        confirmed_by="fake-smoke (continuous authorization)",
        on_checkpoint=lambda _project, event: checkpoint_stages.append(event.stage),
    )
    transition = runtime.record_section_result(prepared, result, plan)

    planner_client.assert_exhausted()
    writer_client.assert_exhausted()
    reviewer_client.assert_exhausted()
    assert result.completed
    assert [section.status for section in result.project.sections] == [
        "confirmed",
        "confirmed",
    ]
    assert set(checkpoint_stages) >= {"generating", "reviewing", "confirmed"}
    assert transition.observation.status == "succeeded"
    assert transition.assessment.critic.outcome == "pass"
    assert transition.assessment.decision.decision_type == "continue"
    assert transition.state.current_stage == "editing"
    assert transition.state.lifecycle == "running"
    assert store.load_latest_checkpoint() is not None


def test_continuous_writing_repairs_only_reviewed_paragraph_then_rechecks() -> None:
    active_handoff = handoff()
    plan = GroundedWritingPlanner(FakeLLMClient(plan_response())).plan(
        active_handoff
    ).confirm(confirmed_by="student")
    initial_texts = [
        f"Paragraph {number} " + " ".join(["grounded"] * 98)
        for number in range(1, 4)
    ]
    initial = [json.dumps({"text": text}) for text in initial_texts]
    revised_text = "Revised paragraph " + " ".join(["focused"] * 98)
    revised = json.dumps({"text": revised_text})
    writer_client = SequenceLLMClient([*initial, revised])
    finding = json.dumps(
        {
            "section_id": "method",
            "findings": [
                {
                    "paragraph_number": 2,
                    "code": "academic_style_problem",
                    "severity": "blocking",
                    "detail": "The paragraph is generic.",
                    "revision_instruction": "State the scoped comparison directly.",
                }
            ],
        }
    )
    clean = json.dumps({"section_id": "method", "findings": []})
    reviewer_client = SequenceLLMClient([finding, clean])

    result = ContinuousSectionWritingService(
        writer=LLMGroundedParagraphWriter(writer_client),
        reviewer=LLMSectionQualityReviewer(reviewer_client),
        policy=ContinuousWritingPolicy(max_revision_passes=1),
    ).run(
        WritingProjectService().start(active_handoff),
        plan,
        confirmed_by="student (continuous authorization)",
    )

    draft = result.project.sections[0].draft
    assert result.completed
    assert draft is not None
    assert draft.paragraphs[0].text == initial_texts[0]
    assert draft.paragraphs[1].text == revised_text
    assert draft.paragraphs[2].text == initial_texts[2]
    assert draft.quality_review_status == "passed"
    assert draft.quality_review_rounds == 2
    assert len(writer_client.calls) == 4
    assert len(reviewer_client.calls) == 2


def test_continuous_writing_stops_when_review_findings_remain() -> None:
    active_handoff = handoff()
    plan = GroundedWritingPlanner(FakeLLMClient(plan_response())).plan(
        active_handoff
    ).confirm(confirmed_by="student")
    writer_client = FakeLLMClient(
        json.dumps({"text": " ".join(["grounded"] * 100)})
    )
    finding = json.dumps(
        {
            "section_id": "method",
            "findings": [
                {
                    "paragraph_number": 1,
                    "code": "topic_drift",
                    "severity": "blocking",
                    "detail": "The paragraph leaves the chapter's defined scope.",
                    "revision_instruction": "Return to the planned atmospheric claim.",
                }
            ],
        }
    )

    result = ContinuousSectionWritingService(
        writer=LLMGroundedParagraphWriter(writer_client),
        reviewer=LLMSectionQualityReviewer(FakeLLMClient(finding)),
        policy=ContinuousWritingPolicy(max_revision_passes=1),
    ).run(
        WritingProjectService().start(active_handoff),
        plan,
        confirmed_by="student (continuous authorization)",
    )

    draft = result.project.sections[0].draft
    assert not result.completed
    assert result.stopped_section_id == "method"
    assert draft is not None
    assert draft.status == "needs_review"
    assert draft.quality_review_status == "findings"
    assert any(issue.code == "topic_drift" for issue in draft.issues)
    assert result.stop_code == "review_exhausted"
    assert "停止盲目重写" in (result.stop_reason or "")


def test_continuous_writing_defers_persistent_style_findings() -> None:
    active_handoff = handoff()
    plan = GroundedWritingPlanner(FakeLLMClient(plan_response())).plan(
        active_handoff
    ).confirm(confirmed_by="student")
    writer_client = FakeLLMClient(
        json.dumps({"text": " ".join(["grounded"] * 100)})
    )
    style_finding = json.dumps(
        {
            "section_id": "method",
            "findings": [
                {
                    "paragraph_number": 1,
                    "code": "academic_style_problem",
                    "severity": "blocking",
                    "detail": "The paragraph remains formulaic.",
                    "revision_instruction": "Use a direct scholarly judgment.",
                }
            ],
        }
    )

    result = ContinuousSectionWritingService(
        writer=LLMGroundedParagraphWriter(writer_client),
        reviewer=LLMSectionQualityReviewer(FakeLLMClient(style_finding)),
        policy=ContinuousWritingPolicy(
            max_revision_passes=3,
            max_total_review_rounds=6,
        ),
    ).run(
        WritingProjectService().start(active_handoff),
        plan,
        confirmed_by="student (continuous authorization)",
    )

    draft = result.project.sections[0].draft
    assert result.completed
    assert draft is not None
    assert draft.status == "confirmed"
    assert draft.quality_review_status == "passed"
    assert any(
        issue.code == "academic_style_problem" and issue.severity == "warning"
        for issue in draft.issues
    )
    assert any(issue.code == "quality_review_deferred" for issue in draft.issues)
    assert len(draft.quality_review_history) == 2
    assert (
        draft.quality_review_history[0].blocking_signatures
        == draft.quality_review_history[1].blocking_signatures
    )
    assert len(writer_client.calls) == 4


def test_continuous_writing_resumes_legacy_reviewer_failure_without_calls() -> None:
    active_handoff = handoff()
    plan = GroundedWritingPlanner(FakeLLMClient(plan_response())).plan(
        active_handoff
    ).confirm(confirmed_by="student")
    packet = SectionEvidencePacketBuilder().build(active_handoff, "method")
    failed_draft = mark_section_quality_review_failed(
        GroundedSectionDraftService().create(packet, proposal()),
        "legacy malformed reviewer JSON",
    ).model_copy(update={"quality_review_rounds": 3})
    project = WritingProjectService().save_draft(
        WritingProjectService().start(active_handoff),
        failed_draft,
    )
    writer_client = FakeLLMClient(json.dumps({"text": "unused"}))
    reviewer_client = FakeLLMClient(
        json.dumps({"section_id": "method", "findings": []})
    )

    result = ContinuousSectionWritingService(
        writer=LLMGroundedParagraphWriter(writer_client),
        reviewer=LLMSectionQualityReviewer(reviewer_client),
    ).run(
        project,
        plan,
        confirmed_by="student (continuous authorization)",
    )

    draft = result.project.sections[0].draft
    assert result.completed
    assert draft is not None
    assert draft.status == "confirmed"
    assert draft.quality_review_status == "passed"
    assert any(issue.code == "quality_review_degraded" for issue in draft.issues)
    assert writer_client.calls == []
    assert reviewer_client.calls == []


def test_continuous_writing_accepts_nonblocking_editorial_advice() -> None:
    active_handoff = handoff()
    plan = GroundedWritingPlanner(FakeLLMClient(plan_response())).plan(
        active_handoff
    ).confirm(confirmed_by="student")
    writer_client = FakeLLMClient(
        json.dumps({"text": " ".join(["grounded"] * 100)})
    )
    reviewer = FakeLLMClient(
        json.dumps(
            {
                "section_id": "method",
                "findings": [
                    {
                        "paragraph_number": 1,
                        "code": "academic_style_problem",
                        "severity": "warning",
                        "detail": "The opening could be shorter.",
                        "revision_instruction": "Optionally tighten the opening.",
                    }
                ],
            }
        )
    )

    result = ContinuousSectionWritingService(
        writer=LLMGroundedParagraphWriter(writer_client),
        reviewer=LLMSectionQualityReviewer(reviewer),
    ).run(
        WritingProjectService().start(active_handoff),
        plan,
        confirmed_by="student (continuous authorization)",
    )

    draft = result.project.sections[0].draft
    assert result.completed
    assert draft is not None
    assert draft.quality_review_status == "passed"
    assert len(writer_client.calls) == 3
    assert any(issue.severity == "warning" for issue in draft.issues)


def test_continuous_writing_confirms_already_passed_draft_without_rewriting() -> None:
    active_handoff = handoff()
    plan = GroundedWritingPlanner(FakeLLMClient(plan_response())).plan(
        active_handoff
    ).confirm(confirmed_by="student")
    packet = SectionEvidencePacketBuilder().build(active_handoff, "method")
    draft = GroundedSectionDraftService().create(packet, proposal()).model_copy(
        update={
            "quality_review_status": "passed",
            "quality_review_rounds": 1,
            "quality_reviewed_at": datetime.now(timezone.utc),
        }
    )
    project = WritingProjectService().save_draft(
        WritingProjectService().start(active_handoff),
        draft,
    )
    writer = FakeLLMClient(json.dumps({"text": "must not be called"}))
    reviewer = FakeLLMClient(json.dumps({"section_id": "method", "findings": []}))

    result = ContinuousSectionWritingService(
        writer=LLMGroundedParagraphWriter(writer),
        reviewer=LLMSectionQualityReviewer(reviewer),
    ).run(
        project,
        plan,
        confirmed_by="student (continuous authorization)",
    )

    assert result.completed
    assert result.project.sections[0].status == "confirmed"
    assert writer.calls == []
    assert reviewer.calls == []


def test_manual_section_writing_stops_after_review_without_confirming() -> None:
    active_handoff = handoff()
    plan = GroundedWritingPlanner(FakeLLMClient(plan_response())).plan(
        active_handoff
    ).confirm(confirmed_by="student")

    result = ContinuousSectionWritingService(
        writer=LLMGroundedParagraphWriter(
            FakeLLMClient(json.dumps({"text": " ".join(["grounded"] * 100)}))
        ),
        reviewer=LLMSectionQualityReviewer(
            FakeLLMClient(json.dumps({"section_id": "method", "findings": []}))
        ),
    ).run(
        WritingProjectService().start(active_handoff),
        plan,
        confirmed_by="student",
        section_id="method",
        auto_confirm=False,
    )

    draft = result.project.sections[0].draft
    assert not result.completed
    assert draft is not None
    assert draft.status == "draft"
    assert draft.quality_review_status == "passed"
    assert result.events[-1].stage == "ready"


def test_paragraph_writer_recovers_literal_json_control_characters() -> None:
    active_handoff = handoff()
    plan = GroundedWritingPlanner(FakeLLMClient(plan_response())).plan(active_handoff)
    section_packet = SectionEvidencePacketBuilder().build(active_handoff, "method")
    paragraph_packet = ParagraphEvidencePacketBuilder().build(
        section_packet,
        plan.sections[0].paragraphs[0],
    )
    client = FakeLLMClient(
        '{"text":"First line\nsecond line\twith a control\u0001 character."}'
    )

    result = LLMGroundedParagraphWriter(client).write(paragraph_packet)

    assert result.text == "First line second line with a control character."
    assert len(client.calls) == 1


def test_paragraph_writer_repairs_malformed_json_once() -> None:
    active_handoff = handoff()
    plan = GroundedWritingPlanner(FakeLLMClient(plan_response())).plan(active_handoff)
    section_packet = SectionEvidencePacketBuilder().build(active_handoff, "method")
    paragraph_packet = ParagraphEvidencePacketBuilder().build(
        section_packet,
        plan.sections[0].paragraphs[0],
    )
    client = SequenceLLMClient(
        [
            "not-json",
            json.dumps({"text": "The repaired paragraph remains evidence-bound."}),
        ]
    )

    result = LLMGroundedParagraphWriter(client).write(paragraph_packet)

    assert result.text == "The repaired paragraph remains evidence-bound."
    assert len(client.calls) == 2
    repair_message = client.calls[1]["messages"][-1]["content"]
    assert "Repair only the JSON encoding" in repair_message


def test_paragraph_writer_shortens_runaway_output_once() -> None:
    active_handoff = handoff()
    plan = GroundedWritingPlanner(FakeLLMClient(plan_response())).plan(active_handoff)
    section_packet = SectionEvidencePacketBuilder().build(active_handoff, "method")
    paragraph_packet = ParagraphEvidencePacketBuilder().build(
        section_packet,
        plan.sections[0].paragraphs[0],
    )
    runaway = " ".join(["runaway"] * 200)
    shortened = " ".join(["grounded"] * 70)
    client = SequenceLLMClient(
        [
            json.dumps({"text": runaway}),
            json.dumps({"text": shortened}),
        ]
    )

    result = LLMGroundedParagraphWriter(client).write(paragraph_packet)

    assert result.text == shortened
    assert len(client.calls) == 2
    repair_message = client.calls[1]["messages"][-1]["content"]
    assert "Rewrite only the paragraph text" in repair_message
    assert "no more than 176 counted units" in repair_message


def test_paragraph_writer_compacts_repeated_overrun_at_sentence_boundary() -> None:
    active_handoff = handoff()
    plan = GroundedWritingPlanner(FakeLLMClient(plan_response())).plan(active_handoff)
    section_packet = SectionEvidencePacketBuilder().build(active_handoff, "method")
    paragraph_packet = ParagraphEvidencePacketBuilder().build(
        section_packet,
        plan.sections[0].paragraphs[0],
    )
    sentence = " ".join(["grounded"] * 20) + "."
    still_too_long = " ".join([sentence] * 10)
    client = SequenceLLMClient(
        [json.dumps({"text": still_too_long})] * 3
    )

    result = LLMGroundedParagraphWriter(client).write(paragraph_packet)

    counted = count_writing_units(result.text, counting_policy="words")
    assert counted == 160
    assert result.text.endswith(".")
    assert len(client.calls) == 3
    final_repair_message = client.calls[2]["messages"][-1]["content"]
    assert "safety margin below the hard limit of 176" in final_repair_message


def test_paragraph_writer_accepts_negligible_planning_floor_shortfall() -> None:
    active_handoff = handoff()
    plan = GroundedWritingPlanner(FakeLLMClient(plan_response())).plan(active_handoff)
    section_packet = SectionEvidencePacketBuilder().build(active_handoff, "method")
    paragraph_packet = ParagraphEvidencePacketBuilder().build(
        section_packet,
        plan.sections[0].paragraphs[0],
    )
    paragraph_packet = paragraph_packet.model_copy(
        update={
            "paragraph": paragraph_packet.paragraph.model_copy(
                update={"target_words": 313}
            )
        }
    )
    client = FakeLLMClient(
        json.dumps({"text": " ".join(["grounded"] * 264)})
    )

    result = LLMGroundedParagraphWriter(client).write(paragraph_packet)

    assert count_writing_units(result.text, counting_policy="words") == 264
    assert len(client.calls) == 1


def test_paragraph_writer_still_repairs_material_planning_floor_shortfall() -> None:
    active_handoff = handoff()
    plan = GroundedWritingPlanner(FakeLLMClient(plan_response())).plan(active_handoff)
    section_packet = SectionEvidencePacketBuilder().build(active_handoff, "method")
    paragraph_packet = ParagraphEvidencePacketBuilder().build(
        section_packet,
        plan.sections[0].paragraphs[0],
    )
    paragraph_packet = paragraph_packet.model_copy(
        update={
            "paragraph": paragraph_packet.paragraph.model_copy(
                update={"target_words": 313}
            )
        }
    )
    client = SequenceLLMClient(
        [json.dumps({"text": " ".join(["grounded"] * 120)})] * 3
    )

    with pytest.raises(GroundedWritingError, match="tolerated minimum"):
        LLMGroundedParagraphWriter(client).write(paragraph_packet)

    assert len(client.calls) == 3


def test_paragraph_writer_removes_self_authored_citation_once() -> None:
    active_handoff = handoff()
    plan = GroundedWritingPlanner(FakeLLMClient(plan_response())).plan(active_handoff)
    section_packet = SectionEvidencePacketBuilder().build(active_handoff, "method")
    paragraph_packet = ParagraphEvidencePacketBuilder().build(
        section_packet,
        plan.sections[0].paragraphs[0],
    )
    invalid = " ".join(["grounded"] * 65) + " [@invented2025]"
    repaired = " ".join(["grounded"] * 70)
    client = SequenceLLMClient(
        [
            json.dumps({"text": invalid}),
            json.dumps({"text": repaired}),
        ]
    )

    result = LLMGroundedParagraphWriter(client).write(paragraph_packet)

    assert result.text == repaired
    assert len(client.calls) == 2
    repair_message = client.calls[1]["messages"][-1]["content"]
    assert "Remove every citation marker" in repair_message


def test_paragraph_cache_rejects_runaway_cached_text(tmp_path: Path) -> None:
    active_handoff = handoff()
    plan = GroundedWritingPlanner(FakeLLMClient(plan_response())).plan(active_handoff)
    section_packet = SectionEvidencePacketBuilder().build(active_handoff, "method")
    paragraph_packet = ParagraphEvidencePacketBuilder().build(
        section_packet,
        plan.sections[0].paragraphs[0],
    )
    cache = ParagraphWritingRuntimeCache(
        tmp_path,
        plan_fingerprint=plan.plan_fingerprint,
    )
    valid = ParagraphTextProposal(text=" ".join(["grounded"] * 70))
    cache.save(paragraph_packet, valid)
    path = (
        tmp_path
        / plan.plan_fingerprint[:16]
        / "method"
        / "method_p01.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["proposal"]["text"] = " ".join(["runaway"] * 200)
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert cache.load(paragraph_packet) is None

    payload["proposal"]["text"] = " ".join(["grounded"] * 70) + " [@invented]"
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert cache.load(paragraph_packet) is None


def test_planner_cache_preserves_completed_section_plan(tmp_path: Path) -> None:
    active_handoff = handoff()
    cache = WritingPlanRuntimeCache(tmp_path, handoff=active_handoff)
    first_client = FakeLLMClient(plan_response())
    first = GroundedWritingPlanner(first_client, cache=cache).plan(active_handoff)
    second_client = FakeLLMClient("not-json")

    second = GroundedWritingPlanner(second_client, cache=cache).plan(active_handoff)

    assert second == first
    assert len(first_client.calls) == 1
    assert second_client.calls == []


def test_planned_writer_uses_locked_bindings_and_reuses_paragraph_cache(
    tmp_path: Path,
) -> None:
    active_handoff = handoff()
    plan = GroundedWritingPlanner(FakeLLMClient(plan_response())).plan(active_handoff)
    section_plan = plan.sections[0]
    section_packet = SectionEvidencePacketBuilder().build(active_handoff, "method")
    cache = ParagraphWritingRuntimeCache(
        tmp_path,
        plan_fingerprint=plan.plan_fingerprint,
    )
    first_client = FakeLLMClient(
        json.dumps({"text": "The verified result supports this planned paragraph."})
    )
    service = PlannedSectionDraftService()

    first = service.draft(
        section_packet,
        section_plan,
        LLMGroundedParagraphWriter(first_client),
        cache=cache,
    )
    second_client = FakeLLMClient("not-json")
    second = service.draft(
        section_packet,
        section_plan,
        LLMGroundedParagraphWriter(second_client),
        cache=cache,
    )

    assert first.model_dump(exclude={"generated_at"}) == second.model_dump(
        exclude={"generated_at"}
    )
    assert first.status == "draft"
    assert len(first.paragraphs) == 3
    assert first.paragraphs[0].evidence_card_ids == [EVIDENCE_ID]
    assert "[@sun2025_core1]" in first.markdown
    assert "p. 4" not in first.markdown
    assert len(first_client.calls) == 3
    assert second_client.calls == []

    project_service = WritingProjectService()
    first = quality_passed(first)
    project = project_service.save_draft(
        project_service.start(active_handoff),
        first,
    )
    project = project_service.confirm_section(
        project,
        "method",
        confirmed_by="student",
    )
    body = project_service.assemble_body(project)
    assert project.status == "body_complete"
    assert body.source_dois == [DOI, SUPPORTING_DOI]

    targeted_client = FakeLLMClient(
        json.dumps({"text": "Only the blocked paragraph is regenerated."})
    )
    targeted = service.draft(
        section_packet,
        section_plan,
        LLMGroundedParagraphWriter(targeted_client),
        cache=cache,
        force_paragraph_numbers={2},
    )
    assert len(targeted_client.calls) == 1
    assert targeted.paragraphs[0].text == first.paragraphs[0].text
    assert targeted.paragraphs[1].text == "Only the blocked paragraph is regenerated."

    no_cache_client = FakeLLMClient(
        json.dumps({"text": "Only the explicitly targeted paragraph changes."})
    )
    reused = service.draft(
        section_packet,
        section_plan,
        LLMGroundedParagraphWriter(no_cache_client),
        existing_draft=first,
        force_paragraph_numbers={2},
    )
    assert len(no_cache_client.calls) == 1
    assert reused.paragraphs[0].text == first.paragraphs[0].text
    assert reused.paragraphs[1].text == "Only the explicitly targeted paragraph changes."
    assert reused.paragraphs[2].text == first.paragraphs[2].text
    repair_payload = json.loads(no_cache_client.calls[0]["messages"][1]["content"])
    assert repair_payload["locked_evidence_packet"]["paragraph"]["paragraph_number"] == 2
    assert [
        item["paragraph_number"] for item in repair_payload["editorial_context"]
    ] == [1, 3]


def test_paragraph_cache_resumes_after_mid_section_failure(tmp_path: Path) -> None:
    active_handoff = handoff()
    plan = GroundedWritingPlanner(FakeLLMClient(plan_response())).plan(active_handoff)
    section_packet = SectionEvidencePacketBuilder().build(active_handoff, "method")
    cache = ParagraphWritingRuntimeCache(
        tmp_path,
        plan_fingerprint=plan.plan_fingerprint,
    )
    valid = json.dumps({"text": "A valid locked-evidence paragraph."})
    interrupted_client = SequenceLLMClient([valid, "not-json", "not-json"])
    service = PlannedSectionDraftService()

    with pytest.raises(GroundedWritingError, match="paragraph output"):
        service.draft(
            section_packet,
            plan.sections[0],
            LLMGroundedParagraphWriter(interrupted_client),
            cache=cache,
        )

    resumed_client = FakeLLMClient(valid)
    draft = service.draft(
        section_packet,
        plan.sections[0],
        LLMGroundedParagraphWriter(resumed_client),
        cache=cache,
    )

    assert draft.status == "draft"
    assert len(interrupted_client.calls) == 3
    assert len(resumed_client.calls) == 2


def test_llm_writes_prose_but_code_adds_citation_and_page() -> None:
    packet = SectionEvidencePacketBuilder().build(handoff(), "method")
    client = FakeLLMClient(
        response_text=json.dumps(proposal().model_dump())
    )

    draft = LLMGroundedSectionWriter(client).draft(packet)

    assert draft.status == "draft"
    assert "[@sun2025_core1]" in draft.markdown
    assert "p. 4" not in draft.markdown
    assert draft.citations[0].doi == DOI
    assert draft.citations[0].page_numbers == [4]
    assert client.calls[0]["response_format"] == {"type": "json_object"}
    system_prompt = client.calls[0]["messages"][0]["content"]
    assert "Do not state specific numbers" in system_prompt


def test_writer_repairs_missing_support_without_rewriting_prose() -> None:
    packet = SectionEvidencePacketBuilder().build(handoff(), "method")
    detailed_text = "The retrieval model reduces regional uncertainty."
    synthesis_text = "Together, the studies motivate regional retrieval workflows."
    valid_text = "The verified result provides direct regional evidence."
    unbound_response = json.dumps(
        {
            "section_id": "method",
            "paragraphs": [
                {
                    "role": "detailed_evidence",
                    "text": detailed_text,
                    "evidence_card_ids": [],
                    "source_dois": [DOI],
                },
                {
                    "role": "synthesis",
                    "text": synthesis_text,
                    "evidence_card_ids": [],
                    "source_dois": [],
                },
                {
                    "role": "detailed_evidence",
                    "text": valid_text,
                    "evidence_card_ids": [EVIDENCE_ID],
                    "source_dois": [],
                },
            ],
        }
    )
    repair_response = json.dumps(
        {
            "section_id": "method",
            "bindings": [
                {
                    "paragraph_number": 1,
                    "evidence_card_ids": [EVIDENCE_ID],
                    "source_dois": [],
                },
                {
                    "paragraph_number": 2,
                    "evidence_card_ids": [],
                    "source_dois": [SUPPORTING_DOI],
                },
            ],
        }
    )
    client = SequenceLLMClient([unbound_response, repair_response])

    draft = LLMGroundedSectionWriter(client).draft(packet)

    assert len(client.calls) == 2
    assert [paragraph.text for paragraph in draft.paragraphs] == [
        detailed_text,
        synthesis_text,
        valid_text,
    ]
    assert draft.paragraphs[0].evidence_card_ids == [EVIDENCE_ID]
    assert draft.paragraphs[1].source_dois == [SUPPORTING_DOI]
    assert draft.paragraphs[2].evidence_card_ids == [EVIDENCE_ID]
    assert draft.status == "draft"
    assert "[@sun2025_core1]" in draft.markdown
    assert "p. 4" not in draft.markdown
    repair_prompt = client.calls[1]["messages"][0]["content"]
    assert "do not return or rewrite paragraph text" in repair_prompt
    repair_payload = json.loads(client.calls[1]["messages"][1]["content"])
    assert [item["paragraph_number"] for item in repair_payload["paragraphs"]] == [1, 2]


def test_repaired_unknown_evidence_id_remains_blocking() -> None:
    packet = SectionEvidencePacketBuilder().build(handoff(), "method")
    unbound_response = json.dumps(
        {
            "section_id": "method",
            "paragraphs": [
                {
                    "role": "detailed_evidence",
                    "text": "The retrieval model reduces regional uncertainty.",
                    "evidence_card_ids": [],
                    "source_dois": [DOI],
                }
            ],
        }
    )
    repair_response = json.dumps(
        {
            "section_id": "method",
            "bindings": [
                {
                    "paragraph_number": 1,
                    "evidence_card_ids": ["ev_method_unknown_001"],
                    "source_dois": [],
                }
            ],
        }
    )
    client = SequenceLLMClient([unbound_response, repair_response])

    draft = LLMGroundedSectionWriter(client).draft(packet)

    assert draft.status == "needs_review"
    assert any(issue.code == "unknown_evidence_card" for issue in draft.issues)


def test_support_repair_must_bind_every_paragraph() -> None:
    packet = SectionEvidencePacketBuilder().build(handoff(), "method")
    unbound_response = json.dumps(
        {
            "section_id": "method",
            "paragraphs": [
                {
                    "role": "detailed_evidence",
                    "text": "The retrieval model reduces regional uncertainty.",
                    "evidence_card_ids": [],
                    "source_dois": [DOI],
                },
                {
                    "role": "synthesis",
                    "text": "The evidence supports further regional evaluation.",
                    "evidence_card_ids": [],
                    "source_dois": [],
                },
            ],
        }
    )
    incomplete_repair = json.dumps(
        {
            "section_id": "method",
            "bindings": [
                {
                    "paragraph_number": 1,
                    "evidence_card_ids": [EVIDENCE_ID],
                    "source_dois": [],
                }
            ],
        }
    )
    client = SequenceLLMClient([unbound_response, incomplete_repair])

    with pytest.raises(GroundedWritingError, match="every paragraph exactly once"):
        LLMGroundedSectionWriter(client).draft(packet)


def test_confirmed_ai_policy_blocks_provider_call_but_keeps_evidence_packet() -> None:
    packet = SectionEvidencePacketBuilder().build(blocked_handoff(), "method")
    client = FakeLLMClient(response_text="{}")

    assert packet.ai_writing_mode == "generation_blocked"
    with pytest.raises(GroundedWritingError, match="prohibited"):
        LLMGroundedSectionWriter(client).draft(packet)
    assert client.calls == []


def test_unknown_evidence_card_blocks_section_confirmation() -> None:
    packet = SectionEvidencePacketBuilder().build(handoff(), "method")
    draft = GroundedSectionDraftService().create(
        packet,
        proposal(evidence_ids=["ev_method_unknown_001"]),
    )

    assert draft.status == "needs_review"
    assert draft.issues[0].code == "unknown_evidence_card"
    with pytest.raises(GroundedWritingError, match="blocking"):
        GroundedSectionDraftService().confirm(
            draft,
            confirmed_by="student",
        )


def test_metadata_source_cannot_support_a_detailed_claim() -> None:
    packet = SectionEvidencePacketBuilder().build(handoff(), "method")
    draft = GroundedSectionDraftService().create(
        packet,
        proposal(
            source_dois=[SUPPORTING_DOI],
        ),
    )

    assert draft.status == "needs_review"
    assert any(
        issue.code == "source_permission_exceeded"
        for issue in draft.issues
    )


def test_llm_cannot_insert_its_own_doi_citation() -> None:
    packet = SectionEvidencePacketBuilder().build(handoff(), "method")
    draft = GroundedSectionDraftService().create(
        packet,
        proposal(
            text=(
                "The model reduces uncertainty "
                "(doi:10.9999/invented.1)."
            ),
        ),
    )

    assert draft.status == "needs_review"
    assert any(issue.code == "llm_authored_citation" for issue in draft.issues)


def test_internal_coverage_instruction_cannot_enter_submit_ready_prose() -> None:
    packet = SectionEvidencePacketBuilder().build(handoff(), "method")
    draft = GroundedSectionDraftService().create(
        packet,
        proposal(text="为满足参考文献覆盖政策，本段补充讨论上述研究。"),
    )

    assert draft.status == "needs_review"
    assert any(issue.code == "workflow_instruction_leak" for issue in draft.issues)


def test_internal_global_edit_instruction_cannot_enter_submit_ready_prose() -> None:
    packet = SectionEvidencePacketBuilder().build(handoff(), "method")
    draft = GroundedSectionDraftService().create(
        packet,
        proposal(text="本段不再重复前述核心判断，而是作为过渡引出后续讨论。"),
    )

    assert draft.status == "needs_review"
    assert any(issue.code == "workflow_instruction_leak" for issue in draft.issues)


def test_project_requires_human_confirmation_before_body_assembly() -> None:
    active_handoff = handoff()
    packet = SectionEvidencePacketBuilder().build(active_handoff, "method")
    draft = GroundedSectionDraftService().create(packet, proposal())
    service = WritingProjectService()
    project = service.save_draft(service.start(active_handoff), draft)

    with pytest.raises(GroundedWritingError, match="confirmed"):
        service.assemble_body(project)

    with pytest.raises(GroundedWritingError, match="quality review passes"):
        service.confirm_section(
            project,
            "method",
            confirmed_by="student",
        )

    project = service.save_draft(project, quality_passed(draft))
    project = service.confirm_section(
        project,
        "method",
        confirmed_by="student",
    )
    body = service.assemble_body(project)

    assert project.status == "body_complete"
    assert body.source_dois == [DOI]
    assert body.markdown.startswith("# Atmospheric remote sensing")
