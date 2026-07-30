from veriwrite_agent.models.literature_discovery import (
    JournalRankingLookup,
    JournalRankingRecord,
    LiteratureCandidate,
)
from veriwrite_agent.models.literature_selection import (
    LiteratureRelevanceAssessment,
    LiteratureSearchBlueprint,
    LiteratureSelectionCandidate,
    LiteratureThemePlan,
    ThemeRelevanceScore,
)
from veriwrite_agent.models.literature_verification import (
    AuthoritativeMetadataEvidence,
    DoiResolutionEvidence,
    LiteratureVerificationResult,
    RisBibliographicMetadata,
)
from veriwrite_agent.services.literature_selector import (
    BalancedLiteratureSelector,
)


def blueprint(*, aerosol_target: int = 1, methane_target: int = 1) -> LiteratureSearchBlueprint:
    return LiteratureSearchBlueprint(
        topic="Atmospheric remote sensing",
        discipline="大气科学",
        writing_through_line="Objects and methods",
        target_total=aerosol_target + methane_target,
        themes=[
            LiteratureThemePlan(
                theme_id="aerosol",
                section_title="Aerosol",
                section_purpose="Review aerosol retrieval",
                research_questions=["How are aerosols retrieved?"],
                primary_keywords=["aerosol"],
                search_queries=["satellite aerosol retrieval"],
                target_count=aerosol_target,
            ),
            LiteratureThemePlan(
                theme_id="methane",
                section_title="Methane",
                section_purpose="Review methane retrieval",
                research_questions=["How is methane retrieved?"],
                primary_keywords=["methane"],
                search_queries=["satellite methane retrieval"],
                target_count=methane_target,
            ),
        ],
    )


def selection_candidate(
    *,
    doi: str,
    title: str,
    year: int,
    tier: str | None,
    aerosol_score: float,
    methane_score: float,
) -> LiteratureSelectionCandidate:
    candidate = LiteratureCandidate(
        doi=doi,
        title=title,
        authors=["A. Author"],
        year=year,
        journal_title="Example Journal",
        source_provider="crossref",
    )
    verification = LiteratureVerificationResult(
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
                year=year,
                journal_title="Example Journal",
            ),
            raw_ris="TY  - JOUR\nER  -",
            attempts=1,
            reason="available",
        ),
    )
    ranking_records = (
        [
            JournalRankingRecord(
                category="理工类",
                discipline="大气科学",
                journal_title="Example Journal",
                normalized_title="example journal",
                tier=tier,
                source_workbook="catalog.xlsx",
                source_row=2,
            )
        ]
        if tier is not None
        else []
    )
    ranking = JournalRankingLookup(
        status="matched" if tier is not None else "not_found",
        query_title="Example Journal",
        normalized_title="example journal",
        discipline="大气科学",
        records=ranking_records,
        reason="matched" if tier is not None else "not found",
    )
    relevance = LiteratureRelevanceAssessment(
        doi=doi,
        theme_scores=[
            ThemeRelevanceScore(
                theme_id="aerosol",
                score=aerosol_score,
                rationale="semantic fit",
            ),
            ThemeRelevanceScore(
                theme_id="methane",
                score=methane_score,
                rationale="semantic fit",
            ),
        ],
        best_theme_id=(
            "aerosol" if aerosol_score >= methane_score else "methane"
        ),
    )
    return LiteratureSelectionCandidate(
        verification=verification,
        ranking=ranking,
        relevance=relevance,
    )


def test_relevance_has_priority_over_journal_tier_and_year() -> None:
    more_relevant = selection_candidate(
        doi="10.1000/relevant",
        title="Direct aerosol evidence",
        year=2022,
        tier="T3",
        aerosol_score=0.95,
        methane_score=0.1,
    )
    more_prestigious_and_recent = selection_candidate(
        doi="10.1000/prestigious",
        title="Partly related aerosol evidence",
        year=2026,
        tier="T1",
        aerosol_score=0.8,
        methane_score=0.1,
    )

    result = BalancedLiteratureSelector().select(
        blueprint(),
        [
            more_prestigious_and_recent,
            more_relevant,
            selection_candidate(
                doi="10.1000/methane",
                title="Methane evidence",
                year=2025,
                tier="T2",
                aerosol_score=0.1,
                methane_score=0.9,
            ),
        ],
    )

    aerosol = next(item for item in result.selected if item.theme_id == "aerosol")
    assert aerosol.doi == "10.1000/relevant"


def test_equal_relevance_prefers_tier_then_recency() -> None:
    candidates = [
        selection_candidate(
            doi="10.1000/t2-new",
            title="T2 newer",
            year=2026,
            tier="T2",
            aerosol_score=0.9,
            methane_score=0.1,
        ),
        selection_candidate(
            doi="10.1000/t1-old",
            title="T1 older",
            year=2022,
            tier="T1",
            aerosol_score=0.9,
            methane_score=0.1,
        ),
        selection_candidate(
            doi="10.1000/t1-new",
            title="T1 newer",
            year=2025,
            tier="T1",
            aerosol_score=0.9,
            methane_score=0.1,
        ),
        selection_candidate(
            doi="10.1000/methane",
            title="Methane",
            year=2025,
            tier="T3",
            aerosol_score=0.1,
            methane_score=0.9,
        ),
    ]

    result = BalancedLiteratureSelector().select(blueprint(), candidates)

    aerosol = next(item for item in result.selected if item.theme_id == "aerosol")
    assert aerosol.doi == "10.1000/t1-new"


def test_unranked_paper_can_be_selected_but_loses_an_equal_relevance_tie() -> None:
    candidates = [
        selection_candidate(
            doi="10.1000/unranked",
            title="Unranked but real",
            year=2026,
            tier=None,
            aerosol_score=0.9,
            methane_score=0.1,
        ),
        selection_candidate(
            doi="10.1000/t6",
            title="Locally ranked",
            year=2022,
            tier="T6",
            aerosol_score=0.9,
            methane_score=0.1,
        ),
        selection_candidate(
            doi="10.1000/methane",
            title="Methane",
            year=2025,
            tier=None,
            aerosol_score=0.1,
            methane_score=0.9,
        ),
    ]

    result = BalancedLiteratureSelector().select(blueprint(), candidates)

    aerosol = next(item for item in result.selected if item.theme_id == "aerosol")
    methane = next(item for item in result.selected if item.theme_id == "methane")
    assert aerosol.doi == "10.1000/t6"
    assert methane.cug_tier is None
    assert methane.ranking_status == "not_found"


def test_theme_quotas_prevent_one_topic_from_filling_the_final_pool() -> None:
    candidates = [
        selection_candidate(
            doi="10.1000/aerosol-1",
            title="Aerosol one",
            year=2025,
            tier="T1",
            aerosol_score=0.95,
            methane_score=0.1,
        ),
        selection_candidate(
            doi="10.1000/aerosol-2",
            title="Aerosol two",
            year=2024,
            tier="T1",
            aerosol_score=0.9,
            methane_score=0.1,
        ),
        selection_candidate(
            doi="10.1000/methane-1",
            title="Methane one",
            year=2025,
            tier="T3",
            aerosol_score=0.1,
            methane_score=0.9,
        ),
        selection_candidate(
            doi="10.1000/methane-2",
            title="Methane two",
            year=2024,
            tier="T3",
            aerosol_score=0.1,
            methane_score=0.85,
        ),
    ]

    result = BalancedLiteratureSelector().select(
        blueprint(aerosol_target=2, methane_target=2),
        candidates,
    )

    assert result.target_reached is True
    assert result.shortages == {}
    assert [item.theme_id for item in result.selected].count("aerosol") == 2
    assert [item.theme_id for item in result.selected].count("methane") == 2
