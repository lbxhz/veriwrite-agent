"""Tests for the post-draft evidence assembly and enhancement handoff chain."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import pytest

from veriwrite_agent.models.evidence import (
    DocumentAcquisition,
    DocumentPage,
    EvidenceCard,
    EvidenceLibrary,
    EvidenceQuote,
    LiteratureLibraryRecord,
    PdfInspectionBatch,
)
from veriwrite_agent.models.literature_selection import (
    BalancedLiteratureSelection,
    LiteratureSearchBlueprint,
    LiteratureThemePlan,
)
from veriwrite_agent.models.literature_verification import LiteratureVerificationBatch
from veriwrite_agent.models.requirement_workflow import ConfirmedRequirementSpec
from veriwrite_agent.models.requirements import RequirementSpec, TopicBoundary
from veriwrite_agent.models.writing_plan import (
    GroundedWritingPlan,
    WritingParagraphPlan,
    WritingSectionPlan,
)
from veriwrite_agent.services import evidence_assembly
from veriwrite_agent.services.evidence_assembly import build_deferred_enhancement_handoff
from veriwrite_agent.services.evidence_library import (
    EvidenceLibraryBuilder,
    EvidenceLibraryConfirmationService,
)
from veriwrite_agent.services.requirement_policy import RequirementPolicyCompiler
from veriwrite_agent.services.writing_handoff import (
    WritingHandoffService,
    WritingOutlineBuilder,
)
from veriwrite_agent.services.writing_evidence_recovery import deferred_section_ids

BACKGROUND_DOI = "10.1000/bg"
METHOD_DOI = "10.1000/method"
RECOVERED_DOI = "10.1000/recovered"


def _sha(doi: str) -> str:
    return hashlib.sha256(doi.encode("utf-8")).hexdigest()


def _document(doi: str) -> DocumentAcquisition:
    return DocumentAcquisition(
        doi=doi,
        status="available",
        method="user_upload",
        source_url=f"https://doi.org/{doi}",
        local_path=f"runtime/{doi.replace('/', '_')}.pdf",
        sha256=_sha(doi),
        media_type="application/pdf",
        file_size_bytes=4096,
        attempts=1,
    )


def _page(doi: str) -> DocumentPage:
    return DocumentPage(
        doi=doi,
        document_sha256=_sha(doi),
        page_number=1,
        text=f"Grounded source text for {doi}.",
        extraction_method="native_text",
    )


def _card(doi: str, theme_id: str, index: int) -> EvidenceCard:
    return EvidenceCard(
        evidence_id=f"ev_{theme_id}_{index}",
        doi=doi,
        theme_id=theme_id,
        evidence_type="result",
        normalized_claim=f"Grounded result from {doi}.",
        supporting_quotes=[
            EvidenceQuote(page_number=1, exact_text=f"Grounded source text for {doi}.")
        ],
        source_document_sha256=_sha(doi),
        support_strength="direct",
        review_status="confirmed",
    )


def _full_text_record(doi: str, theme_id: str, title: str) -> LiteratureLibraryRecord:
    return LiteratureLibraryRecord(
        doi=doi,
        title=title,
        authors=["Doe, Jane"],
        year=2025,
        journal="Journal",
        source_url=f"https://doi.org/{doi}",
        theme_ids=[theme_id],
        evidence_tier="A_core",
        evidence_status="full_text_verified",
        permitted_use="detailed_claims",
        admission_status="admitted",
        centrality="central",
        supported_claim=f"Supports {theme_id} evidence.",
        suitable_section_id=theme_id,
        use_boundary=f"Use for {theme_id} evidence.",
    )


def _metadata_record(doi: str, theme_id: str, title: str) -> LiteratureLibraryRecord:
    return LiteratureLibraryRecord(
        doi=doi,
        title=title,
        authors=["Roe, Ana"],
        year=2024,
        source_url=f"https://doi.org/{doi}",
        theme_ids=[theme_id],
        evidence_tier="B_supporting",
        evidence_status="metadata_verified",
        permitted_use="section_support",
        admission_status="admitted",
        centrality="supporting",
        supported_claim=f"Supports {theme_id} comparison context.",
        suitable_section_id=theme_id,
        use_boundary=f"Use for {theme_id} supporting evidence.",
    )


def _requirement() -> ConfirmedRequirementSpec:
    return ConfirmedRequirementSpec(
        confirmed_by="student",
        confirmed_at=datetime.now(timezone.utc),
        requirement=RequirementSpec(
            document_type="literature_review",
            topic="Atmospheric remote sensing",
            topic_boundary=TopicBoundary(
                central_question="How do atmospheric retrieval methods differ?",
                included_objects=["atmospheric retrieval methods"],
                excluded_objects=["soil moisture"],
                contextual_only_topics=["edge computing"],
                origin="explicit",
            ),
        ),
    )


def _blueprint() -> LiteratureSearchBlueprint:
    return LiteratureSearchBlueprint(
        topic="Atmospheric remote sensing",
        discipline="Remote sensing",
        writing_through_line="From background to retrieval methods.",
        target_total=2,
        themes=[
            LiteratureThemePlan(
                theme_id="background",
                section_title="Background",
                section_purpose="Explain the research background.",
                research_questions=["Why is retrieval needed?"],
                primary_keywords=["atmosphere"],
                search_queries=["atmospheric remote sensing"],
                target_count=1,
            ),
            LiteratureThemePlan(
                theme_id="method",
                section_title="Methods",
                section_purpose="Compare retrieval methods.",
                research_questions=["Which methods are used?"],
                primary_keywords=["retrieval"],
                search_queries=["satellite retrieval method"],
                target_count=1,
            ),
        ],
    )


def _selection() -> BalancedLiteratureSelection:
    return BalancedLiteratureSelection(
        blueprint=_blueprint(),
        selected=[],
        shortages={"background": 1, "method": 1},
        target_reached=False,
    )


def _previous_handoff():
    requirement = _requirement()
    policy = RequirementPolicyCompiler().compile(requirement)
    records = [
        _full_text_record(BACKGROUND_DOI, "background", "Background paper"),
        _full_text_record(METHOD_DOI, "method", "Method paper"),
        _metadata_record(RECOVERED_DOI, "method", "Recovered candidate"),
    ]
    documents = [_document(BACKGROUND_DOI), _document(METHOD_DOI)]
    pages = [_page(BACKGROUND_DOI), _page(METHOD_DOI)]
    cards = [_card(BACKGROUND_DOI, "background", 1), _card(METHOD_DOI, "method", 1)]
    library = EvidenceLibraryBuilder().build(
        records=records,
        documents=documents,
        pages=pages,
        evidence_cards=cards,
    )
    confirmed = EvidenceLibraryConfirmationService().confirm(
        library,
        confirmed_by="student",
    )
    outline = WritingOutlineBuilder().build(_blueprint(), confirmed, policy=policy)
    confirmed_outline = WritingHandoffService().confirm_outline(
        outline,
        confirmed_by="student",
    )
    return WritingHandoffService().create(
        requirement=requirement,
        outline=confirmed_outline,
        evidence_library=confirmed,
        policy=policy,
    )


def _recovered_library() -> EvidenceLibrary:
    records = [
        _full_text_record(BACKGROUND_DOI, "background", "Background paper"),
        _full_text_record(METHOD_DOI, "method", "Method paper"),
        _full_text_record(RECOVERED_DOI, "method", "Recovered candidate"),
    ]
    documents = [_document(BACKGROUND_DOI), _document(METHOD_DOI), _document(RECOVERED_DOI)]
    pages = [_page(BACKGROUND_DOI), _page(METHOD_DOI), _page(RECOVERED_DOI)]
    cards = [
        _card(BACKGROUND_DOI, "background", 1),
        _card(METHOD_DOI, "method", 1),
        _card(RECOVERED_DOI, "method", 2),
    ]
    return EvidenceLibraryBuilder().build(
        records=records,
        documents=documents,
        pages=pages,
        evidence_cards=cards,
    )


def _deferred_section() -> WritingSectionPlan:
    deferred = WritingParagraphPlan(
        paragraph_id="method_p01",
        section_id="method",
        paragraph_number=1,
        role="background",
        purpose="State a bounded background overview.",
        claim_focus="The available records permit only a general description.",
        central_question="What is the general scope of this direction?",
        argument_move="frame_problem",
        target_words=100,
        evidence_card_ids=[],
        source_dois=[RECOVERED_DOI],
        deferred_argument="compare_studies",
        deferred_comparison_axis="accuracy and architecture",
        deferred_recovery_dois=[RECOVERED_DOI],
    )
    plain = WritingParagraphPlan(
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
        source_dois=[METHOD_DOI],
    )
    return WritingSectionPlan(
        section_id="method",
        title="Retrieval methods",
        purpose="Compare atmospheric retrieval methods.",
        target_words=200,
        counting_policy="words",
        paragraphs=[deferred, plain],
    )


def _plan(*, with_deferred: bool) -> GroundedWritingPlan:
    section = _deferred_section()
    if not with_deferred:
        section = section.model_copy(
            update={
                "paragraphs": [
                    section.paragraphs[0].model_copy(
                        update={
                            "deferred_argument": None,
                            "deferred_comparison_axis": None,
                            "deferred_recovery_dois": [],
                        }
                    ),
                    section.paragraphs[1],
                ]
            }
        )
    return GroundedWritingPlan(
        topic="Atmospheric retrieval",
        output_language="English",
        plan_fingerprint="a" * 64,
        sections=[section],
    )


def test_deferred_section_ids_returns_only_sections_with_deferred_paragraphs() -> None:
    assert deferred_section_ids(_plan(with_deferred=True)) == {"method"}
    assert deferred_section_ids(_plan(with_deferred=False)) == set()


def test_build_deferred_enhancement_handoff_merges_recovered_full_text(
    monkeypatch,
    tmp_path,
) -> None:
    previous = _previous_handoff()
    recovered = _recovered_library()
    monkeypatch.setattr(
        evidence_assembly,
        "build_evidence_library",
        lambda *args, **kwargs: recovered,
    )

    merged = build_deferred_enhancement_handoff(
        previous,
        selection=_selection(),
        batch=PdfInspectionBatch(download_directory=str(tmp_path), inspected_file_count=0),
        affected_section_ids={"method"},
        confirmed_requirement=previous.requirement,
        verifications=LiteratureVerificationBatch(),
        policy=previous.requirement_policy,
        cache_root=tmp_path,
    )

    recovered_record = next(
        record for record in merged.evidence_library.records if record.doi == RECOVERED_DOI
    )
    assert recovered_record.evidence_status == "full_text_verified"
    assert recovered_record.evidence_tier == "A_core"
    assert recovered_record.permitted_use == "detailed_claims"
    recovered_cards = [
        card for card in merged.evidence_library.evidence_cards if card.doi == RECOVERED_DOI
    ]
    assert recovered_cards
    assert all(
        card.review_status == "confirmed" and card.support_strength == "direct"
        for card in recovered_cards
    )
    # 未受影响章节的既有全文保持不变。
    background_record = next(
        record for record in merged.evidence_library.records if record.doi == BACKGROUND_DOI
    )
    assert background_record.evidence_status == "full_text_verified"


def test_build_deferred_enhancement_handoff_rejects_unresolved_library(
    monkeypatch,
    tmp_path,
) -> None:
    previous = _previous_handoff()
    unresolved = _recovered_library().model_copy(
        update={"unresolved_issues": [f"core_pdf_missing:{RECOVERED_DOI}"]}
    )
    monkeypatch.setattr(
        evidence_assembly,
        "build_evidence_library",
        lambda *args, **kwargs: unresolved,
    )

    with pytest.raises(ValueError, match="仍有全文未就绪"):
        build_deferred_enhancement_handoff(
            previous,
            selection=_selection(),
            batch=PdfInspectionBatch(download_directory=str(tmp_path), inspected_file_count=0),
            affected_section_ids={"method"},
            confirmed_requirement=previous.requirement,
            verifications=LiteratureVerificationBatch(),
            policy=previous.requirement_policy,
            cache_root=tmp_path,
        )
