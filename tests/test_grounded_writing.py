import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from veriwrite_agent.llm.fake_client import FakeLLMClient
from veriwrite_agent.models.evidence import (
    DocumentAcquisition,
    DocumentPage,
    EvidenceCard,
    EvidenceLibrary,
    EvidenceQuote,
    LiteratureLibraryRecord,
)
from veriwrite_agent.models.requirement_workflow import ConfirmedRequirementSpec
from veriwrite_agent.models.requirements import AIUsagePolicy, RequirementSpec
from veriwrite_agent.models.writing import (
    DraftParagraphProposal,
    SectionDraftProposal,
)
from veriwrite_agent.models.writing_plan import ParagraphTextProposal
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
)
from veriwrite_agent.services.writing_planning import (
    GroundedWritingPlanner,
    LLMGroundedParagraphWriter,
    ParagraphEvidencePacketBuilder,
    ParagraphWritingRuntimeCache,
    PlannedSectionDraftService,
    WritingPlanError,
    WritingPlanRuntimeCache,
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
        ),
    )
    return V04WritingHandoff(
        requirement=requirement,
        outline=outline,
        evidence_library=library,
    )


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


def plan_response(*, evidence_ref: str = "E001") -> str:
    return json.dumps(
        {
            "section_id": "method",
            "paragraphs": [
                {
                    "role": "detailed_evidence",
                    "purpose": "Present the verified retrieval result.",
                    "claim_focus": "The verified model reduces regional uncertainty.",
                    "relative_weight": 3,
                    "evidence_refs": [evidence_ref],
                    "source_refs": [],
                },
                {
                    "role": "section_support",
                    "purpose": "Connect the result to the wider research context.",
                    "claim_focus": "Remote sensing supports regional observation.",
                    "relative_weight": 2,
                    "evidence_refs": [],
                    "source_refs": ["S002"],
                },
                {
                    "role": "synthesis",
                    "purpose": "Synthesize the core and supporting literature.",
                    "claim_focus": "The evidence motivates regional retrieval workflows.",
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
    assert len(section.paragraphs) == 3
    assert sum(item.target_words for item in section.paragraphs) == 300
    assert section.paragraphs[0].evidence_card_ids == [EVIDENCE_ID]
    assert section.paragraphs[0].source_dois == [DOI]
    assert section.paragraphs[1].source_dois == [SUPPORTING_DOI]
    assert len(client.calls) == 1
    prompt_payload = json.loads(client.calls[0]["messages"][1]["content"])
    assert prompt_payload["evidence_catalog"][0]["ref"] == "E001"
    assert prompt_payload["source_catalog"][1]["ref"] == "S002"
    assert prompt_payload["source_catalog"][1]["allowed_roles"] == [
        "section_support",
        "background",
        "synthesis",
    ]


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
        json.dumps({"text": "The locked evidence supports this planned paragraph."})
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
    assert "[@sun2025_core1, p. 4]" in first.markdown
    assert len(first_client.calls) == 3
    assert second_client.calls == []

    project_service = WritingProjectService()
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
    assert "[@sun2025_core1, p. 4]" in draft.markdown
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
    assert "[@sun2025_core1, p. 4]" in draft.markdown
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


def test_project_requires_human_confirmation_before_body_assembly() -> None:
    active_handoff = handoff()
    packet = SectionEvidencePacketBuilder().build(active_handoff, "method")
    draft = GroundedSectionDraftService().create(packet, proposal())
    service = WritingProjectService()
    project = service.save_draft(service.start(active_handoff), draft)

    with pytest.raises(GroundedWritingError, match="confirmed"):
        service.assemble_body(project)

    project = service.confirm_section(
        project,
        "method",
        confirmed_by="student",
    )
    body = service.assemble_body(project)

    assert project.status == "body_complete"
    assert body.source_dois == [DOI]
    assert body.markdown.startswith("# Atmospheric remote sensing")
