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


def response(passage_id: str = "page_4_passage_1") -> str:
    return json.dumps(
        {
            "selections": [
                {
                    "evidence_type": "method",
                    "normalized_claim": (
                        "The method combines satellite observations with "
                        "a physical retrieval model."
                    ),
                    "passage_ids": [passage_id],
                    "support_strength": "direct",
                }
            ]
        }
    )


def test_builds_cards_but_code_assigns_identity_fields() -> None:
    client = FakeLLMClient(response())

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
    assert cards[0].supporting_quotes[0].exact_text == page().text
    assert client.calls[0]["response_format"] == {"type": "json_object"}
    assert "Return at most 4 selections" in client.calls[0]["messages"][0]["content"]


def test_rejects_an_unknown_passage_id() -> None:
    client = FakeLLMClient(response("page_4_passage_999"))

    with pytest.raises(EvidenceCardExtractionError, match="unknown passage"):
        LLMEvidenceCardExtractor(client).extract(
            doi=DOI,
            title="Aerosol retrieval",
            theme_id="retrieval_method",
            section_purpose="Compare retrieval methods.",
            pages=[page()],
        )


def test_rejects_oversized_structured_output() -> None:
    oversized = {"selections": [json.loads(response())["selections"][0] for _ in range(2)]}
    client = FakeLLMClient(json.dumps(oversized))

    with pytest.raises(EvidenceCardExtractionError, match="card limit"):
        LLMEvidenceCardExtractor(
            client,
            max_cards_per_batch=1,
        ).extract(
            doi=DOI,
            title="Aerosol retrieval",
            theme_id="retrieval_method",
            section_purpose="Compare retrieval methods.",
            pages=[page()],
        )
