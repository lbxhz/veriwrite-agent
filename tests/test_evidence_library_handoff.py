from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from veriwrite_agent.models.evidence import (
    DocumentAcquisition,
    DocumentPage,
    EvidenceCard,
    EvidenceQuote,
    LiteratureLibraryRecord,
)
from veriwrite_agent.models.literature_selection import (
    LiteratureSearchBlueprint,
    LiteratureThemePlan,
)
from veriwrite_agent.models.requirement_workflow import ConfirmedRequirementSpec
from veriwrite_agent.models.requirements import RequirementSpec
from veriwrite_agent.services.evidence_library import (
    EvidenceLibraryBuilder,
    EvidenceLibraryConfirmationService,
)
from veriwrite_agent.services.writing_handoff import (
    WritingHandoffService,
    WritingOutlineBuilder,
)

DOI = "10.1000/library.1"
SHA = "d" * 64


def document() -> DocumentAcquisition:
    return DocumentAcquisition(
        doi=DOI,
        status="available",
        method="user_upload",
        source_url=f"https://doi.org/{DOI}",
        local_path="runtime/papers/library.pdf",
        sha256=SHA,
        media_type="application/pdf",
        file_size_bytes=4096,
        attempts=1,
    )


def record() -> LiteratureLibraryRecord:
    return LiteratureLibraryRecord(
        doi=DOI,
        title="Verified full-text paper",
        authors=["A. Author"],
        year=2025,
        journal="Journal",
        source_url=f"https://doi.org/{DOI}",
        theme_ids=["background", "method"],
        evidence_tier="A_core",
        evidence_status="full_text_verified",
        permitted_use="detailed_claims",
    )


def card(theme_id: str, evidence_type: str) -> EvidenceCard:
    return EvidenceCard(
        evidence_id=f"ev_{theme_id}_{evidence_type}_001",
        doi=DOI,
        theme_id=theme_id,
        evidence_type=evidence_type,
        normalized_claim=f"Grounded {evidence_type} claim.",
        supporting_quotes=[
            EvidenceQuote(page_number=1, exact_text="Grounded source text.")
        ],
        source_document_sha256=SHA,
        support_strength="direct",
    )


def page() -> DocumentPage:
    return DocumentPage(
        doi=DOI,
        document_sha256=SHA,
        page_number=1,
        text="Grounded source text.",
        extraction_method="native_text",
    )


def blueprint() -> LiteratureSearchBlueprint:
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


def test_builds_confirms_and_hands_off_a_grounded_library() -> None:
    cards = [card("background", "background"), card("method", "method")]
    draft_library = EvidenceLibraryBuilder().build(
        records=[record()],
        documents=[document()],
        pages=[page()],
        evidence_cards=cards,
    )
    confirmed_library = EvidenceLibraryConfirmationService().confirm(
        draft_library,
        confirmed_by="student",
    )
    outline = WritingOutlineBuilder().build(
        blueprint(),
        confirmed_library,
        target_words=1000,
    )
    confirmed_outline = WritingHandoffService().confirm_outline(
        outline,
        confirmed_by="student",
    )
    requirement = ConfirmedRequirementSpec(
        confirmed_by="student",
        confirmed_at=datetime.now(timezone.utc),
        requirement=RequirementSpec(
            document_type="literature_review",
            topic="Atmospheric remote sensing",
        ),
    )

    handoff = WritingHandoffService().create(
        requirement=requirement,
        outline=confirmed_outline,
        evidence_library=confirmed_library,
    )

    assert handoff.status == "ready_for_writing"
    assert confirmed_library.literature_matrix[0].methods[0].value
    assert sum(
        section.target_words for section in outline.sections
    ) == outline.target_words


def test_metadata_only_record_cannot_support_detailed_claims() -> None:
    with pytest.raises(ValidationError, match="metadata-only"):
        LiteratureLibraryRecord(
            doi="10.1000/background.1",
            title="Metadata record",
            year=2024,
            source_url="https://doi.org/10.1000/background.1",
            theme_ids=["background"],
            evidence_tier="B_supporting",
            evidence_status="metadata_verified",
            permitted_use="detailed_claims",
        )


def test_outline_with_missing_full_text_cannot_be_confirmed() -> None:
    metadata = LiteratureLibraryRecord(
        doi="10.1000/background.1",
        title="Metadata record",
        year=2024,
        source_url="https://doi.org/10.1000/background.1",
        theme_ids=["background", "method"],
        evidence_tier="B_supporting",
        evidence_status="metadata_verified",
        permitted_use="section_support",
    )
    library = EvidenceLibraryBuilder().build(
        records=[metadata],
        documents=[],
        evidence_cards=[],
    )
    outline = WritingOutlineBuilder().build(
        blueprint(),
        library,
        target_words=1000,
    )

    with pytest.raises(ValidationError, match="gaps"):
        WritingHandoffService().confirm_outline(
            outline,
            confirmed_by="student",
        )


def test_smoke_outline_consolidates_themes_around_one_verified_pdf() -> None:
    metadata = LiteratureLibraryRecord(
        doi="10.1000/support.1",
        title="Metadata-only supporting paper",
        year=2025,
        source_url="https://doi.org/10.1000/support.1",
        theme_ids=["method"],
        evidence_tier="B_supporting",
        evidence_status="metadata_verified",
        permitted_use="section_support",
    )
    library = EvidenceLibraryBuilder().build(
        records=[record(), metadata],
        documents=[document()],
        pages=[page()],
        evidence_cards=[card("background", "background")],
    )

    outline = WritingOutlineBuilder().build(
        blueprint(),
        library,
        target_words=800,
        smoke_test=True,
    )

    assert len(outline.sections) == 1
    assert outline.sections[0].core_dois == [DOI]
    assert outline.sections[0].supporting_dois == ["10.1000/support.1"]
    assert len(outline.sections[0].evidence_card_ids) == 1
    assert outline.sections[0].evidence_gap is False
    assert outline.unresolved_gaps == []
    WritingHandoffService().confirm_outline(outline, confirmed_by="student")
