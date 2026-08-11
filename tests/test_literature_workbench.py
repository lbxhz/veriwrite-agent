import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from veriwrite_agent.literature.cug_catalog import CugJournalRankingProvider
from veriwrite_agent.literature.fake import (
    FakeAuthoritativeMetadataProvider,
    FakeDoiResolver,
)
from veriwrite_agent.llm.fake_client import FakeLLMClient
from veriwrite_agent.models.literature_discovery import (
    LiteratureCandidate,
    LiteratureSearchPlan,
)
from veriwrite_agent.models.literature_selection import (
    ConfirmedLiteratureSearchBlueprint,
    LiteratureRelevanceAssessment,
    LiteratureSearchBlueprint,
    LiteratureThemePlan,
    ThemeRelevanceScore,
)
from veriwrite_agent.models.requirements import TopicBoundary
from veriwrite_agent.models.literature_verification import (
    AuthoritativeMetadataEvidence,
    DoiResolutionEvidence,
    LiteratureVerificationBatch,
    LiteratureVerificationResult,
    RisBibliographicMetadata,
)
from veriwrite_agent.services.literature_blueprint_search import (
    LiteratureBlueprintSearchExpander,
)
from veriwrite_agent.services.literature_discovery import LiteratureDiscoveryService
from veriwrite_agent.services.literature_identity_verification import (
    LiteratureIdentityVerificationService,
)
from veriwrite_agent.services.literature_relevance_scorer import (
    LLMLiteratureRelevanceScorer,
)
from veriwrite_agent.services.literature_query_refinement import (
    LiteratureQueryRefinementBatch,
    ThemeQueryRefinement,
)
from veriwrite_agent.ui.literature_workbench import (
    LiteratureWorkbench,
    _seed_recovery_caches,
    adaptive_candidate_capacity,
    blueprint_run_id,
)


@dataclass
class ThemeSearchProvider:
    calls: list[LiteratureSearchPlan] = field(default_factory=list)

    def search(
        self,
        plan: LiteratureSearchPlan,
    ) -> Iterable[LiteratureCandidate]:
        self.calls.append(plan)
        if "Aerosol" in plan.topic:
            yield candidate(
                "10.1000/aerosol",
                "Satellite aerosol retrieval",
            )
        else:
            yield candidate(
                "10.1000/methane",
                "Satellite methane retrieval",
            )


@dataclass
class WindowedThemeSearchProvider:
    calls: list[LiteratureSearchPlan] = field(default_factory=list)

    def search(
        self,
        plan: LiteratureSearchPlan,
    ) -> Iterable[LiteratureCandidate]:
        self.calls.append(plan)
        query = plan.search_queries[0]
        offset = plan.query_offsets.get(query, 0)
        limit = plan.query_limits.get(query, plan.max_candidates)
        if "Aerosol" in plan.topic:
            window = {
                0: candidate("10.1000/aerosol", "Satellite aerosol retrieval"),
            }
        else:
            window = {
                0: candidate("10.1000/context", "General remote sensing context"),
                1: candidate("10.1000/methane", "Satellite methane retrieval"),
            }
        for position in range(offset, offset + limit):
            if position in window:
                yield window[position]


@dataclass
class SemanticRecoverySearchProvider:
    calls: list[LiteratureSearchPlan] = field(default_factory=list)

    def search(
        self,
        plan: LiteratureSearchPlan,
    ) -> Iterable[LiteratureCandidate]:
        self.calls.append(plan)
        if "Aerosol" in plan.topic:
            yield candidate("10.1000/aerosol", "Satellite aerosol retrieval")
        elif "direct methane spectroscopy" in plan.search_queries:
            yield candidate("10.1000/methane", "Direct methane spectroscopy")
        elif any(offset == 0 for offset in plan.query_offsets.values()):
            yield candidate("10.1000/context", "General remote sensing context")


@dataclass
class StubShortageQueryRefiner:
    calls: list[dict[str, int]] = field(default_factory=list)

    def refine(
        self,
        blueprint: LiteratureSearchBlueprint,
        shortages: dict[str, int],
        *,
        previous_recovery_queries: dict[str, list[str]] | None = None,
    ) -> LiteratureQueryRefinementBatch:
        del blueprint, previous_recovery_queries
        self.calls.append(shortages)
        return LiteratureQueryRefinementBatch(
            themes=[
                ThemeQueryRefinement(
                    theme_id="methane",
                    search_queries=[
                        "direct methane spectroscopy",
                        "atmospheric methane instrument comparison",
                    ],
                )
            ]
        )


@dataclass
class RuleBasedRelevanceScorer:
    calls: list[list[str]] = field(default_factory=list)

    def score(
        self,
        blueprint: LiteratureSearchBlueprint,
        records: list[LiteratureVerificationResult],
    ) -> list[LiteratureRelevanceAssessment]:
        self.calls.append([record.candidate.doi for record in records])
        results: list[LiteratureRelevanceAssessment] = []
        for record in records:
            doi = record.candidate.doi
            admitted_theme = (
                "aerosol"
                if doi.endswith("/aerosol")
                else "methane" if doi.endswith("/methane") else None
            )
            results.append(
                LiteratureRelevanceAssessment(
                    doi=doi,
                    theme_scores=[
                        ThemeRelevanceScore(
                            theme_id=theme.theme_id,
                            score=0.95 if theme.theme_id == admitted_theme else 0.1,
                            rationale="deterministic adaptive-search test",
                        )
                        for theme in blueprint.themes
                    ],
                    best_theme_id=admitted_theme or blueprint.themes[0].theme_id,
                    admission_status="admit" if admitted_theme else "reject",
                    centrality="central" if admitted_theme else "out_of_scope",
                    supported_claim=(
                        "Supports the target retrieval comparison."
                        if admitted_theme
                        else None
                    ),
                    suitable_section_id=admitted_theme,
                    use_boundary=(
                        "Use only in the matched atmospheric retrieval section."
                        if admitted_theme
                        else None
                    ),
                    exclusion_reason=(None if admitted_theme else "too broad"),
                )
            )
        return results


def candidate(doi: str, title: str) -> LiteratureCandidate:
    return LiteratureCandidate(
        doi=doi,
        title=title,
        authors=["A. Author"],
        year=2025,
        journal_title="Remote Sensing of Environment",
        source_provider="crossref",
    )


def confirmed_blueprint() -> ConfirmedLiteratureSearchBlueprint:
    return ConfirmedLiteratureSearchBlueprint(
        confirmed_by="student",
        blueprint=LiteratureSearchBlueprint(
            topic="Atmospheric remote sensing",
            discipline="测绘科学与技术",
            writing_through_line="Objects and methods",
            target_total=2,
            max_candidates=20,
            relevance_threshold=0.5,
            topic_boundary=TopicBoundary(
                central_question="How are atmospheric constituents retrieved?",
                included_objects=["aerosol", "methane"],
                excluded_objects=["soil moisture"],
                origin="explicit",
            ),
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
        ),
    )


def authority(doi: str, title: str) -> AuthoritativeMetadataEvidence:
    return AuthoritativeMetadataEvidence(
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
        raw_ris=(
            "TY  - JOUR\n"
            f"TI  - {title}\n"
            "AU  - Author, A.\n"
            "PY  - 2025\n"
            "JO  - Remote Sensing of Environment\n"
            f"DO  - {doi}\n"
            "ER  -\n"
        ),
        attempts=1,
        reason="available",
    )


def resolution(doi: str) -> DoiResolutionEvidence:
    return DoiResolutionEvidence(
        doi=doi,
        status="resolved",
        resolver_url=f"https://doi.org/{doi}",
        final_url=f"https://publisher.example/{doi}",
        http_status=200,
        attempts=1,
        reason="resolved",
    )


def relevance_response() -> str:
    return json.dumps(
        {
            "assessments": [
                {
                    "doi": "10.1000/aerosol",
                    "admission_status": "admit",
                    "centrality": "central",
                    "supported_claim": "Supports comparison of aerosol retrieval methods.",
                    "suitable_section_id": "aerosol",
                    "use_boundary": "Use only for atmospheric aerosol retrieval.",
                    "theme_scores": [
                        {
                            "theme_id": "aerosol",
                            "score": 0.95,
                            "rationale": "direct aerosol study",
                        },
                        {
                            "theme_id": "methane",
                            "score": 0.1,
                            "rationale": "not a methane study",
                        },
                    ],
                    "best_theme_id": "aerosol",
                },
                {
                    "doi": "10.1000/methane",
                    "admission_status": "admit",
                    "centrality": "central",
                    "supported_claim": "Supports comparison of methane retrieval methods.",
                    "suitable_section_id": "methane",
                    "use_boundary": "Use only for atmospheric methane retrieval.",
                    "theme_scores": [
                        {
                            "theme_id": "aerosol",
                            "score": 0.1,
                            "rationale": "not an aerosol study",
                        },
                        {
                            "theme_id": "methane",
                            "score": 0.95,
                            "rationale": "direct methane study",
                        },
                    ],
                    "best_theme_id": "methane",
                },
            ]
        }
    )


def test_runs_full_v02_flow_and_resumes_from_stage_caches(
    tmp_path: Path,
) -> None:
    search = ThemeSearchProvider()
    resolver = FakeDoiResolver(
        {
            doi: resolution(doi)
            for doi in ("10.1000/aerosol", "10.1000/methane")
        }
    )
    metadata = FakeAuthoritativeMetadataProvider(
        {
            "10.1000/aerosol": authority(
                "10.1000/aerosol",
                "Satellite aerosol retrieval",
            ),
            "10.1000/methane": authority(
                "10.1000/methane",
                "Satellite methane retrieval",
            ),
        }
    )
    llm = FakeLLMClient(relevance_response())
    workbench = LiteratureWorkbench(
        planner=None,
        search_expander=LiteratureBlueprintSearchExpander(pool_multiplier=1),
        discovery_service=LiteratureDiscoveryService(
            search,
            CugJournalRankingProvider.from_default_catalog(),
        ),
        verification_service=LiteratureIdentityVerificationService(
            resolver,
            metadata,
        ),
        relevance_scorer=LLMLiteratureRelevanceScorer(llm),
    )
    progress: list[tuple[str, int, int, str]] = []
    resume_progress: list[tuple[str, int, int, str]] = []

    first = workbench.run(
        confirmed_blueprint(),
        cache_root=tmp_path,
        progress=lambda *items: progress.append(items),
    )
    verification_cache = first.run_dir / "verification_cache.json"
    cached_batch = LiteratureVerificationBatch.model_validate_json(
        verification_cache.read_text(encoding="utf-8")
    )
    unrelated_doi = "10.1000/unrelated-cache"
    cached_batch.results.append(
        LiteratureVerificationResult(
            candidate=candidate(unrelated_doi, "Unrelated cached study"),
            status="verified",
            resolution=resolution(unrelated_doi),
            authority=authority(unrelated_doi, "Unrelated cached study"),
        )
    )
    verification_cache.write_text(
        cached_batch.model_dump_json(indent=2),
        encoding="utf-8",
    )
    second = workbench.run(
        confirmed_blueprint(),
        cache_root=tmp_path,
        progress=lambda *items: resume_progress.append(items),
    )

    assert first.selection.target_reached is True
    assert len(first.selection.selected) == 2
    assert {item.theme_id for item in first.selection.selected} == {
        "aerosol",
        "methane",
    }
    assert first.ris_text.count("TY  - JOUR") == 2
    assert (first.run_dir / "final_result.json").is_file()
    assert second.run_id == first.run_id
    assert len(search.calls) == 2
    assert resolver.calls == ["10.1000/aerosol", "10.1000/methane"]
    assert len(llm.calls) == 1
    assert any(stage == "complete" for stage, *_ in progress)
    assert any(
        stage == "relevance" and current == 0 and total == 2
        for stage, current, total, _ in progress
    )
    assert any(
        stage == "relevance" and current == 2 and total == 2
        for stage, current, total, _ in resume_progress
    )


def test_candidate_pool_multiplier_participates_in_run_cache_identity() -> None:
    confirmed = confirmed_blueprint()

    assert blueprint_run_id(confirmed, pool_multiplier=2) != blueprint_run_id(
        confirmed,
        pool_multiplier=4,
    )


def test_shortage_recovery_capacity_scales_beyond_legacy_fixed_limit() -> None:
    blueprint = confirmed_blueprint().blueprint.model_copy(
        update={"target_total": 60, "max_candidates": 300}
    )

    assert adaptive_candidate_capacity(blueprint) == 900


def test_recovery_run_seeds_only_reusable_verified_stage_caches(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    for name in (
        "discovery_cache.json",
        "verification_cache.json",
        "relevance_cache.json",
        "final_result.json",
    ):
        (source / name).write_text(f'{{"name": "{name}"}}', encoding="utf-8")

    _seed_recovery_caches(source, target)

    assert (target / "discovery_cache.json").is_file()
    assert (target / "verification_cache.json").is_file()
    assert (target / "relevance_cache.json").is_file()
    assert not (target / "final_result.json").exists()


def test_automatically_expands_only_the_shortage_theme_without_repeating_windows(
    tmp_path: Path,
) -> None:
    search = WindowedThemeSearchProvider()
    dois_and_titles = {
        "10.1000/aerosol": "Satellite aerosol retrieval",
        "10.1000/context": "General remote sensing context",
        "10.1000/methane": "Satellite methane retrieval",
    }
    resolver = FakeDoiResolver(
        {doi: resolution(doi) for doi in dois_and_titles}
    )
    metadata = FakeAuthoritativeMetadataProvider(
        {
            doi: authority(doi, title)
            for doi, title in dois_and_titles.items()
        }
    )
    relevance = RuleBasedRelevanceScorer()
    workbench = LiteratureWorkbench(
        planner=None,
        search_expander=LiteratureBlueprintSearchExpander(pool_multiplier=1),
        discovery_service=LiteratureDiscoveryService(
            search,
            CugJournalRankingProvider.from_default_catalog(),
        ),
        verification_service=LiteratureIdentityVerificationService(
            resolver,
            metadata,
        ),
        relevance_scorer=relevance,  # type: ignore[arg-type]
    )

    first = workbench.run(confirmed_blueprint(), cache_root=tmp_path)
    calls_after_first_run = len(search.calls)
    second = workbench.run(confirmed_blueprint(), cache_root=tmp_path)

    assert first.selection.target_reached is True
    assert {item.doi for item in first.selection.selected} == {
        "10.1000/aerosol",
        "10.1000/methane",
    }
    aerosol_calls = [call for call in search.calls if "Aerosol" in call.topic]
    methane_calls = [call for call in search.calls if "Methane" in call.topic]
    assert len(aerosol_calls) == 1
    assert [call.query_offsets[call.search_queries[0]] for call in methane_calls] == [
        0,
        1,
    ]
    assert len(search.calls) == calls_after_first_run
    assert second.selection == first.selection


def test_automatically_rewrites_shortage_queries_after_depth_expansion(
    tmp_path: Path,
) -> None:
    search = SemanticRecoverySearchProvider()
    dois_and_titles = {
        "10.1000/aerosol": "Satellite aerosol retrieval",
        "10.1000/context": "General remote sensing context",
        "10.1000/methane": "Direct methane spectroscopy",
    }
    resolver = FakeDoiResolver(
        {doi: resolution(doi) for doi in dois_and_titles}
    )
    metadata = FakeAuthoritativeMetadataProvider(
        {
            doi: authority(doi, title)
            for doi, title in dois_and_titles.items()
        }
    )
    relevance = RuleBasedRelevanceScorer()
    refiner = StubShortageQueryRefiner()
    workbench = LiteratureWorkbench(
        planner=None,
        search_expander=LiteratureBlueprintSearchExpander(pool_multiplier=1),
        discovery_service=LiteratureDiscoveryService(
            search,
            CugJournalRankingProvider.from_default_catalog(),
        ),
        verification_service=LiteratureIdentityVerificationService(
            resolver,
            metadata,
        ),
        relevance_scorer=relevance,  # type: ignore[arg-type]
        shortage_query_refiner=refiner,  # type: ignore[arg-type]
    )
    run_dir = tmp_path / blueprint_run_id(
        confirmed_blueprint(),
        pool_multiplier=1,
    )
    run_dir.mkdir(parents=True)
    invalid_cache = run_dir / "query_refinement_1.json"
    invalid_cache.write_text(
        json.dumps(
            {
                "themes": [
                    {
                        "theme_id": "methane",
                        "search_queries": [
                            "duplicate methane query",
                            "duplicate methane query",
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = workbench.run(confirmed_blueprint(), cache_root=tmp_path)
    calls_after_first_run = len(search.calls)
    resumed = workbench.run(confirmed_blueprint(), cache_root=tmp_path)

    assert result.selection.target_reached is True
    assert {item.doi for item in result.selection.selected} == {
        "10.1000/aerosol",
        "10.1000/methane",
    }
    assert refiner.calls == [{"methane": 1}]
    assert any(
        "direct methane spectroscopy" in call.search_queries
        for call in search.calls
    )
    assert (result.run_dir / "query_refinement_1.json").is_file()
    assert (result.run_dir / "query_refinement_1.rejected.json").is_file()
    assert len(search.calls) == calls_after_first_run
    assert resumed.selection == result.selection
