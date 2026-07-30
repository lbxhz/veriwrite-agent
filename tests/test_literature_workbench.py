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
    LiteratureSearchBlueprint,
    LiteratureThemePlan,
)
from veriwrite_agent.models.literature_verification import (
    AuthoritativeMetadataEvidence,
    DoiResolutionEvidence,
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
from veriwrite_agent.ui.literature_workbench import LiteratureWorkbench


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

    first = workbench.run(
        confirmed_blueprint(),
        cache_root=tmp_path,
        progress=lambda *items: progress.append(items),
    )
    second = workbench.run(
        confirmed_blueprint(),
        cache_root=tmp_path,
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
