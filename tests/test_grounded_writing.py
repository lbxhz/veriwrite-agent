import json
from datetime import datetime, timezone

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

DOI = "10.1000/core.1"
SUPPORTING_DOI = "10.1000/support.1"
SHA = "a" * 64
EVIDENCE_ID = "ev_method_direct_001"


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


def test_builds_section_packet_from_confirmed_handoff() -> None:
    packet = SectionEvidencePacketBuilder().build(handoff(), "method")

    assert packet.evidence_items[0].evidence_id == EVIDENCE_ID
    assert [source.doi for source in packet.sources] == [DOI, SUPPORTING_DOI]
    assert packet.sources[0].citation_key == "sun2025_core1"


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
