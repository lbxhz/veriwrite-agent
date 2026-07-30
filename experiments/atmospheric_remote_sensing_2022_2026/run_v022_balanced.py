"""Run the V0.2.2 outline-guided, balanced literature selection regression."""

from __future__ import annotations

import json
from pathlib import Path

from veriwrite_agent.config.settings import LLMSettings
from veriwrite_agent.literature.crossref import CrossrefSearchProvider
from veriwrite_agent.literature.cug_catalog import CugJournalRankingProvider
from veriwrite_agent.literature.doi import DoiOrgResolver, DoiRisMetadataProvider
from veriwrite_agent.llm.deepseek_client import DeepSeekClient
from veriwrite_agent.models.literature_discovery import CandidateDecision
from veriwrite_agent.models.literature_selection import (
    LiteratureRelevanceAssessmentBatch,
    LiteratureSearchBlueprint,
    LiteratureSelectionCandidate,
)
from veriwrite_agent.models.literature_verification import (
    LiteratureVerificationBatch,
)
from veriwrite_agent.models.requirement_workflow import ConfirmedRequirementSpec
from veriwrite_agent.models.requirements import ReferenceRequirement, RequirementSpec
from veriwrite_agent.services.literature_blueprint_planner import (
    LiteratureBlueprintPlanner,
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
from veriwrite_agent.services.literature_selector import BalancedLiteratureSelector

OUTPUT_DIR = Path(__file__).resolve().parent
YEAR_TO = 2026
TOPIC = (
    "大气遥感：利用卫星、激光雷达和光谱遥感研究气溶胶、云、"
    "温室气体与空气质量"
)


def main() -> None:
    settings = LLMSettings()
    llm = DeepSeekClient(settings)
    ranking_provider = CugJournalRankingProvider.from_default_catalog()
    confirmed = ConfirmedRequirementSpec(
        confirmed_by="v0.2.2-regression",
        requirement=RequirementSpec(
            document_type="research_direction_literature_review",
            topic=TOPIC,
            topic_source="explicit",
            required_theme_elements=[
                "气溶胶遥感",
                "云遥感",
                "温室气体遥感",
                "空气质量遥感",
            ],
            references=ReferenceRequirement(
                target_total=20,
                recent_year_window=5,
                recent_year_rule_strength="hard",
            ),
        ),
    )

    blueprint_path = OUTPUT_DIR / "v022_search_blueprint.json"
    if blueprint_path.is_file():
        blueprint = LiteratureSearchBlueprint.model_validate_json(
            blueprint_path.read_text(encoding="utf-8")
        )
    else:
        blueprint = LiteratureBlueprintPlanner(
            llm,
            ranking_provider.available_disciplines,
            current_year=YEAR_TO,
        ).plan(confirmed)
        _write_json("v022_search_blueprint.json", blueprint.model_dump(mode="json"))
    themed_plans = LiteratureBlueprintSearchExpander(pool_multiplier=2).expand(
        blueprint
    )

    prefilter_path = OUTPUT_DIR / "v022_prefiltered_candidates.json"
    if prefilter_path.is_file():
        prefilter = json.loads(prefilter_path.read_text(encoding="utf-8"))
        discovery_diagnostics = prefilter["discovery"]
        decisions = [
            CandidateDecision.model_validate(item)
            for item in prefilter["eligible_decisions"]
        ]
        decisions_by_doi = {
            decision.candidate.doi: decision for decision in decisions
        }
    else:
        discovery_service = LiteratureDiscoveryService(
            CrossrefSearchProvider(),
            ranking_provider,
        )
        decisions_by_doi: dict[str, CandidateDecision] = {}
        discovery_diagnostics: list[dict[str, object]] = []
        for themed_plan in themed_plans:
            result = discovery_service.discover(themed_plan.plan)
            discovery_diagnostics.append(
                {
                    "theme_id": themed_plan.theme_id,
                    "search_plan": themed_plan.plan.model_dump(mode="json"),
                    "scanned_count": result.scanned_count,
                    "duplicate_count": result.duplicate_count,
                    "eligible_count": len(result.eligible_records),
                    "excluded_count": len(result.excluded_records),
                    "target_reached": result.target_reached,
                }
            )
            for decision in result.eligible_records:
                decisions_by_doi.setdefault(decision.candidate.doi, decision)
        _write_json(
            "v022_prefiltered_candidates.json",
            {
                "discovery": discovery_diagnostics,
                "eligible_decisions": [
                    decision.model_dump(mode="json")
                    for decision in decisions_by_doi.values()
                ],
            },
        )

    verifier = LiteratureIdentityVerificationService(
        DoiOrgResolver(
            timeout_seconds=10,
            max_attempts=1,
            minimum_request_interval_seconds=0.2,
        ),
        DoiRisMetadataProvider(
            timeout_seconds=10,
            max_attempts=1,
            minimum_request_interval_seconds=0.2,
        ),
    )
    verification_cache_path = OUTPUT_DIR / "v022_verification_cache.json"
    if verification_cache_path.is_file():
        verifications = LiteratureVerificationBatch.model_validate_json(
            verification_cache_path.read_text(encoding="utf-8")
        )
    else:
        verifications = LiteratureVerificationBatch()
    verified_dois = {result.candidate.doi for result in verifications.results}
    for decision in decisions_by_doi.values():
        if decision.candidate.doi in verified_dois:
            continue
        verifications.results.append(verifier.verify(decision.candidate))
        verification_cache_path.write_text(
            verifications.model_dump_json(indent=2),
            encoding="utf-8",
        )
    verified = verifications.verified_records

    relevance_cache_path = OUTPUT_DIR / "v022_relevance_cache.json"
    if relevance_cache_path.is_file():
        relevance_batch = LiteratureRelevanceAssessmentBatch.model_validate_json(
            relevance_cache_path.read_text(encoding="utf-8")
        )
    else:
        relevance_batch = LiteratureRelevanceAssessmentBatch()
    scored_dois = {item.doi for item in relevance_batch.assessments}
    pending = [
        result for result in verified if result.candidate.doi not in scored_dois
    ]
    scorer = LLMLiteratureRelevanceScorer(llm, batch_size=10)
    for start in range(0, len(pending), 10):
        relevance_batch.assessments.extend(
            scorer.score(blueprint, pending[start : start + 10])
        )
        relevance_cache_path.write_text(
            relevance_batch.model_dump_json(indent=2),
            encoding="utf-8",
        )
    relevance = relevance_batch.assessments
    relevance_by_doi = {item.doi: item for item in relevance}
    candidates = [
        LiteratureSelectionCandidate(
            verification=result,
            ranking=decisions_by_doi[result.candidate.doi].ranking,
            relevance=relevance_by_doi[result.candidate.doi],
        )
        for result in verified
    ]
    selection = BalancedLiteratureSelector().select(blueprint, candidates)

    verification_by_doi = {
        result.candidate.doi: result for result in verifications.results
    }
    selected_rows: list[dict[str, object]] = []
    selected_ris: list[str] = []
    for record in selection.selected:
        verification = verification_by_doi[record.doi]
        authority = verification.authority
        if authority is None or authority.metadata is None:
            raise RuntimeError("selected paper lost authority evidence")
        selected_rows.append(
            {
                **record.model_dump(mode="json"),
                "authors": authority.metadata.authors,
                "journal": authority.metadata.journal_title,
                "publisher": authority.metadata.publisher,
                "final_url": (
                    verification.resolution.final_url
                    if verification.resolution is not None
                    else None
                ),
                "warnings": verification.warning_codes,
            }
        )
        selected_ris.append(authority.raw_ris or "")

    _write_json(
        "v022_balanced_results.json",
        {
            "blueprint": blueprint.model_dump(mode="json"),
            "discovery": discovery_diagnostics,
            "unique_prefiltered_count": len(decisions_by_doi),
            "verified_count": len(verified),
            "excluded_count": len(verifications.excluded_records),
            "target_reached": selection.target_reached,
            "shortages": selection.shortages,
            "selected": selected_rows,
        },
    )
    (OUTPUT_DIR / "v022_selected.ris").write_text(
        "\n".join(selected_ris),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "themes": [
                    {
                        "theme_id": theme.theme_id,
                        "title": theme.section_title,
                        "quota": theme.target_count,
                    }
                    for theme in blueprint.themes
                ],
                "unique_prefiltered_count": len(decisions_by_doi),
                "verified_count": len(verified),
                "selected_count": len(selection.selected),
                "shortages": selection.shortages,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _write_json(name: str, value: object) -> None:
    (OUTPUT_DIR / name).write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
