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


def multi_passage_page() -> DocumentPage:
    return DocumentPage(
        doi=DOI,
        document_sha256=SHA,
        page_number=4,
        text=" ".join(
            [
                "Satellite observations provide a detailed view of atmospheric states "
                "for retrieval experiments across several representative regions.",
                "A physical retrieval model constrains the machine learning estimates "
                "and preserves consistency with the measured radiative quantities.",
                "Independent validation data are used to calculate prediction errors "
                "and compare the proposed approach with established baseline methods.",
                "The reported results show improved retrieval accuracy under the tested "
                "conditions while identifying limitations for future regional studies.",
            ]
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


def test_deduplicates_passage_ids_without_weakening_grounding() -> None:
    client = FakeLLMClient(
        json.dumps(
            {
                "selections": [
                    {
                        "evidence_type": "method",
                        "normalized_claim": "The study combines observations and modelling.",
                        "passage_ids": [
                            "page_4_passage_1",
                            "page_4_passage_2",
                            "page_4_passage_2",
                        ],
                        "support_strength": "direct",
                    }
                ]
            }
        )
    )

    cards = LLMEvidenceCardExtractor(client, max_quote_chars=150).extract(
        doi=DOI,
        title="Aerosol retrieval",
        theme_id="retrieval_method",
        section_purpose="Compare retrieval methods.",
        pages=[multi_passage_page()],
    )

    assert len(cards[0].supporting_quotes) == 2
    assert len({quote.exact_text for quote in cards[0].supporting_quotes}) == 2


def test_keeps_first_three_unique_passage_ids_when_provider_returns_four() -> None:
    client = FakeLLMClient(
        json.dumps(
            {
                "selections": [
                    {
                        "evidence_type": "method",
                        "normalized_claim": "The study combines observations and modelling.",
                        "passage_ids": [
                            "page_4_passage_1",
                            "page_4_passage_2",
                            "page_4_passage_3",
                            "page_4_passage_4",
                        ],
                        "support_strength": "direct",
                    }
                ]
            }
        )
    )

    cards = LLMEvidenceCardExtractor(client, max_quote_chars=150).extract(
        doi=DOI,
        title="Aerosol retrieval",
        theme_id="retrieval_method",
        section_purpose="Compare retrieval methods.",
        pages=[multi_passage_page()],
    )

    assert len(cards[0].supporting_quotes) == 3


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
