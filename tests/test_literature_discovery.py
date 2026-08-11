from veriwrite_agent.literature.cug_catalog import CugJournalRankingProvider
from veriwrite_agent.literature.fake import FakeLiteratureSearchProvider
from veriwrite_agent.literature.norwegian_register import (
    NorwegianRegisterRankingProvider,
)
from veriwrite_agent.models.literature_discovery import (
    LiteratureCandidate,
    LiteratureSearchPlan,
)
from veriwrite_agent.services.literature_discovery import (
    LiteratureDiscoveryService,
)


def candidate(
    index: int,
    *,
    journal: str = "Remote Sensing of Environment",
    year: int = 2025,
    source_type: str = "journal-article",
) -> LiteratureCandidate:
    return LiteratureCandidate(
        doi=f"10.1000/example.{index}",
        title=f"Candidate {index}",
        authors=["Test Author"],
        year=year,
        journal_title=journal,
        source_type=source_type,
        source_provider="crossref",
        source_url=f"https://doi.org/10.1000/example.{index}",
    )


def plan(*, accepted_tiers: list[str] | None = None) -> LiteratureSearchPlan:
    return LiteratureSearchPlan(
        topic="GeoAI",
        discipline="测绘科学与技术",
        primary_keywords=["GeoAI"],
        search_queries=["GeoAI GIS"],
        accepted_tiers=accepted_tiers or ["T1"],
        year_from=2020,
        year_to=2026,
    )


def test_stops_after_fifty_eligible_candidates() -> None:
    search = FakeLiteratureSearchProvider(
        [candidate(index) for index in range(80)]
    )
    service = LiteratureDiscoveryService(
        search,
        CugJournalRankingProvider.from_default_catalog(),
    )

    result = service.discover(plan())

    assert result.target_reached is True
    assert result.needs_user_confirmation is False
    assert result.scanned_count == 50
    assert len(result.eligible_records) == 50
    assert len(search.calls) == 1


def test_deduplicates_doi_and_explains_hard_filter_failures() -> None:
    duplicate = candidate(1)
    search = FakeLiteratureSearchProvider(
        [
            duplicate,
            duplicate.model_copy(update={"title": "Duplicate presentation"}),
            candidate(2, journal="Remote Sensing"),
            candidate(3, journal="Imaginary GeoAI Journal"),
            candidate(4, year=2018),
        ]
    )
    service = LiteratureDiscoveryService(
        search,
        CugJournalRankingProvider.from_default_catalog(),
    )

    result = service.discover(plan())

    assert result.target_reached is False
    assert result.needs_user_confirmation is True
    assert result.scanned_count == 4
    assert result.duplicate_count == 1
    assert len(result.eligible_records) == 1
    reasons = {
        reason
        for decision in result.excluded_records
        for reason in decision.reason_codes
    }
    assert "journal_tier_not_accepted" in reasons
    assert "journal_not_in_cug_2023_catalog" in reasons
    assert "publication_year_below_requirement" in reasons


def test_preferred_ranking_policy_keeps_unranked_real_candidates() -> None:
    search = FakeLiteratureSearchProvider(
        [candidate(1, journal="Journal Outside Local Catalog")]
    )
    service = LiteratureDiscoveryService(
        search,
        CugJournalRankingProvider.from_default_catalog(),
    )
    preferred_plan = plan().model_copy(
        update={
            "journal_ranking_policy": "preferred",
            "target_eligible_count": 1,
        }
    )

    result = service.discover(preferred_plan)

    assert result.target_reached is True
    assert len(result.eligible_records) == 1
    assert result.eligible_records[0].ranking.status == "not_found"


def test_preferred_ranking_does_not_crash_on_punctuation_only_journal_title() -> None:
    search = FakeLiteratureSearchProvider([candidate(1, journal="—")])
    service = LiteratureDiscoveryService(
        search,
        CugJournalRankingProvider.from_default_catalog(),
    )
    preferred_plan = plan().model_copy(
        update={
            "journal_ranking_policy": "preferred",
            "target_eligible_count": 1,
        }
    )

    result = service.discover(preferred_plan)

    assert result.target_reached is True
    assert result.eligible_records[0].ranking.status == "not_found"
    assert result.eligible_records[0].ranking.normalized_title == ""


def test_discovery_preserves_independent_norwegian_ranking_evidence() -> None:
    paper = candidate(1).model_copy(update={"issns": ["0034-4257"]})
    service = LiteratureDiscoveryService(
        FakeLiteratureSearchProvider([paper]),
        CugJournalRankingProvider.from_default_catalog(),
        NorwegianRegisterRankingProvider.from_default_catalog(),
    )
    one_result_plan = plan().model_copy(update={"target_eligible_count": 1})

    result = service.discover(one_result_plan)

    norwegian = result.eligible_records[0].norwegian_ranking
    assert norwegian is not None
    assert norwegian.resolved_level == 2
    assert norwegian.match_basis == "issn"


def test_discovery_skips_dois_persisted_by_an_earlier_search_window() -> None:
    search = FakeLiteratureSearchProvider([candidate(1), candidate(2)])
    service = LiteratureDiscoveryService(
        search,
        CugJournalRankingProvider.from_default_catalog(),
    )

    result = service.discover(
        plan().model_copy(update={"target_eligible_count": 1}),
        known_dois={candidate(1).doi},
        stop_when_target_reached=False,
    )

    assert result.duplicate_count == 1
    assert result.scanned_count == 1
    assert [item.candidate.doi for item in result.decisions] == [candidate(2).doi]
