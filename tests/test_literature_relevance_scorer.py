import json
from dataclasses import dataclass, field
from typing import Sequence

import pytest

from veriwrite_agent.llm.base import ChatMessage
from veriwrite_agent.models.literature_discovery import LiteratureCandidate
from veriwrite_agent.models.literature_selection import (
    LiteratureSearchBlueprint,
    LiteratureThemePlan,
)
from veriwrite_agent.models.literature_verification import (
    AuthoritativeMetadataEvidence,
    DoiResolutionEvidence,
    LiteratureVerificationResult,
    RisBibliographicMetadata,
)
from veriwrite_agent.services.literature_relevance_scorer import (
    LLMLiteratureRelevanceScorer,
    RelevanceScoringError,
)


@dataclass
class SequenceLLMClient:
    responses: list[str]
    calls: list[list[ChatMessage]] = field(default_factory=list)

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        response_format: dict[str, str] | None = None,
    ) -> str:
        del response_format
        self.calls.append(list(messages))
        return self.responses.pop(0)


def blueprint() -> LiteratureSearchBlueprint:
    return LiteratureSearchBlueprint(
        topic="Atmospheric remote sensing",
        discipline="大气科学",
        writing_through_line="Objects and methods",
        target_total=2,
        themes=[
            LiteratureThemePlan(
                theme_id="aerosol",
                section_title="Aerosol",
                section_purpose="Review aerosol retrieval",
                research_questions=["How are aerosols retrieved?"],
                primary_keywords=["aerosol"],
                search_queries=["satellite aerosol retrieval"],
                target_count=1,
            ),
            LiteratureThemePlan(
                theme_id="methane",
                section_title="Methane",
                section_purpose="Review methane retrieval",
                research_questions=["How is methane retrieved?"],
                primary_keywords=["methane"],
                search_queries=["satellite methane retrieval"],
                target_count=1,
            ),
        ],
    )


def verified(doi: str, title: str) -> LiteratureVerificationResult:
    candidate = LiteratureCandidate(
        doi=doi,
        title=title,
        authors=["A. Author"],
        year=2025,
        journal_title="Remote Sensing of Environment",
        source_provider="crossref",
    )
    return LiteratureVerificationResult(
        status="verified",
        candidate=candidate,
        resolution=DoiResolutionEvidence(
            doi=doi,
            status="resolved",
            resolver_url=f"https://doi.org/{doi}",
            final_url=f"https://publisher.example/{doi}",
            http_status=200,
            attempts=1,
            reason="resolved",
        ),
        authority=AuthoritativeMetadataEvidence(
            doi=doi,
            status="available",
            source_url=f"https://doi.org/{doi}",
            metadata=RisBibliographicMetadata(
                doi=doi,
                title=title,
                authors=["Author, A."],
                year=2025,
                journal_title="Remote Sensing of Environment",
            ),
            raw_ris="TY  - JOUR\nER  -",
            attempts=1,
            reason="available",
        ),
    )


def assessment_response(dois: list[str]) -> str:
    return json.dumps(
        {
            "assessments": [
                {
                    "doi": doi,
                    "theme_scores": [
                        {
                            "theme_id": "aerosol",
                            "score": 0.9 if "aerosol" in doi else 0.2,
                            "rationale": "Title and abstract fit",
                            "matched_concepts": ["retrieval"],
                        },
                        {
                            "theme_id": "methane",
                            "score": 0.9 if "methane" in doi else 0.2,
                            "rationale": "Title and abstract fit",
                            "matched_concepts": ["retrieval"],
                        },
                    ],
                    "best_theme_id": (
                        "aerosol" if "aerosol" in doi else "methane"
                    ),
                }
                for doi in dois
            ]
        }
    )


def test_scores_only_the_verified_supplied_dois_and_all_themes() -> None:
    papers = [
        verified("10.1000/aerosol", "Satellite aerosol retrieval"),
        verified("10.1000/methane", "Satellite methane retrieval"),
    ]
    client = SequenceLLMClient(
        [assessment_response([paper.candidate.doi for paper in papers])]
    )

    assessments = LLMLiteratureRelevanceScorer(client).score(
        blueprint(),
        papers,
    )

    assert [item.doi for item in assessments] == [
        "10.1000/aerosol",
        "10.1000/methane",
    ]
    assert {
        score.theme_id for score in assessments[0].theme_scores
    } == {"aerosol", "methane"}


def test_rejects_invented_doi_even_after_one_scope_repair() -> None:
    wrong = assessment_response(["10.1000/invented"])
    client = SequenceLLMClient([wrong, wrong])

    with pytest.raises(RelevanceScoringError, match="every supplied DOI"):
        LLMLiteratureRelevanceScorer(client).score(
            blueprint(),
            [verified("10.1000/aerosol", "Satellite aerosol retrieval")],
        )

    assert len(client.calls) == 2
