from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from veriwrite_agent.models.evidence import (
    DocumentAcquisition,
    DocumentPage,
    EvidenceBackedValue,
    EvidenceCard,
    EvidenceLibrary,
    EvidenceQuote,
    LiteratureMatrixRow,
)
from veriwrite_agent.services.evidence_grounding import EvidenceGroundingValidator

DOI = "10.1000/evidence.1"
SHA = "a" * 64


def page(*, sha256: str = SHA) -> DocumentPage:
    return DocumentPage(
        doi=DOI,
        document_sha256=sha256,
        page_number=7,
        text=(
            "The retrieval method combines multispectral observations "
            "with a physical atmospheric model."
        ),
        extraction_method="native_text",
    )


def card(
    *,
    quote: str = "combines multispectral observations with a physical atmospheric model",
    sha256: str = SHA,
    review_status: str = "needs_review",
) -> EvidenceCard:
    return EvidenceCard(
        evidence_id="ev_method_001",
        doi=DOI,
        theme_id="retrieval_method",
        evidence_type="method",
        normalized_claim="The study combines multispectral data with a physical model.",
        supporting_quotes=[
            EvidenceQuote(
                page_number=7,
                exact_text=quote,
                section_title="Methods",
            )
        ],
        source_document_sha256=sha256,
        support_strength="direct",
        review_status=review_status,
    )


def document() -> DocumentAcquisition:
    return DocumentAcquisition(
        doi=DOI,
        status="available",
        method="automatic_download",
        source_url="https://publisher.example/paper.pdf",
        local_path="runtime/papers/evidence.pdf",
        sha256=SHA,
        media_type="application/pdf",
        file_size_bytes=1024,
        attempts=1,
    )


def test_accepts_a_quote_found_on_the_claimed_pdf_page() -> None:
    result = EvidenceGroundingValidator().validate([page()], [card()])

    assert result.valid is True
    assert result.issues == []


def test_rejects_an_llm_quote_not_present_on_the_page() -> None:
    result = EvidenceGroundingValidator().validate(
        [page()],
        [card(quote="The model achieved perfect accuracy in every region.")],
    )

    assert result.valid is False
    assert result.issues[0].code == "quote_not_found_on_page"


def test_rejects_a_quote_attached_to_a_different_pdf_hash() -> None:
    result = EvidenceGroundingValidator().validate(
        [page()],
        [card(sha256="b" * 64)],
    )

    assert result.valid is False
    assert result.issues[0].code == "document_identity_mismatch"


def test_matrix_cells_must_reference_evidence_from_the_same_paper() -> None:
    other_doi = "10.1000/evidence.2"
    other_document = document().model_copy(update={"doi": other_doi})
    matrix = LiteratureMatrixRow(
        doi=other_doi,
        title="Other paper",
        theme_ids=["retrieval_method"],
        methods=[
            EvidenceBackedValue(
                value="Physical retrieval",
                evidence_card_ids=["ev_method_001"],
            )
        ],
    )

    with pytest.raises(ValidationError, match="same paper"):
        EvidenceLibrary(
            documents=[document(), other_document],
            evidence_cards=[card()],
            literature_matrix=[matrix],
        )


def test_confirmed_library_requires_reviewed_cards_and_audit_fields() -> None:
    matrix = LiteratureMatrixRow(
        doi=DOI,
        title="Evidence paper",
        theme_ids=["retrieval_method"],
        methods=[
            EvidenceBackedValue(
                value="Physical retrieval",
                evidence_card_ids=["ev_method_001"],
            )
        ],
    )

    library = EvidenceLibrary(
        status="confirmed",
        documents=[document()],
        evidence_cards=[card(review_status="confirmed")],
        literature_matrix=[matrix],
        confirmed_by="student",
        confirmed_at=datetime.now(timezone.utc),
    )

    assert library.status == "confirmed"
