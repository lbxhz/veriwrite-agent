import json

import pytest

from veriwrite_agent.llm.fake_client import FakeLLMClient
from veriwrite_agent.models.evidence import DocumentPage
from veriwrite_agent.services.evidence_card_extraction import (
    EvidenceCardExtractionError,
    LLMEvidenceCardExtractor,
)

DOI = "10.1000/cards.1"
SHA = "c" * 64


def page() -> DocumentPage:
    return DocumentPage(
        doi=DOI,
        document_sha256=SHA,
        page_number=4,
        text=(
            "The study combines satellite observations with a physical "
            "retrieval model to estimate aerosol optical depth."
        ),
        extraction_method="native_text",
    )


def response(quote: str) -> str:
    return json.dumps(
        {
            "proposals": [
                {
                    "evidence_type": "method",
                    "normalized_claim": (
                        "The method combines satellite observations with "
                        "a physical retrieval model."
                    ),
                    "supporting_quotes": [
                        {
                            "page_number": 4,
                            "exact_text": quote,
                            "section_title": "Methods",
                        }
                    ],
                    "support_strength": "direct",
                }
            ]
        }
    )


def test_builds_cards_but_code_assigns_identity_fields() -> None:
    client = FakeLLMClient(
        response(
            "combines satellite observations with a physical retrieval model"
        )
    )

    cards = LLMEvidenceCardExtractor(client).extract(
        doi=DOI,
        title="Aerosol retrieval",
        theme_id="retrieval_method",
        section_purpose="Compare retrieval methods.",
        pages=[page()],
    )

    assert len(cards) == 1
    assert cards[0].doi == DOI
    assert cards[0].source_document_sha256 == SHA
    assert cards[0].evidence_id.startswith("ev_retrieval_method_")
    assert client.calls[0]["response_format"] == {"type": "json_object"}


def test_rejects_a_quote_that_is_not_on_the_pdf_page() -> None:
    client = FakeLLMClient(response("The method achieved perfect accuracy."))

    with pytest.raises(EvidenceCardExtractionError, match="grounding"):
        LLMEvidenceCardExtractor(client).extract(
            doi=DOI,
            title="Aerosol retrieval",
            theme_id="retrieval_method",
            section_purpose="Compare retrieval methods.",
            pages=[page()],
        )
