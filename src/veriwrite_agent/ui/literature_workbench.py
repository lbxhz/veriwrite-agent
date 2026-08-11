"""Resumable application layer for the V0.2 literature console."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from veriwrite_agent.config.settings import LLMSettings
from veriwrite_agent.literature.crossref import CrossrefSearchProvider
from veriwrite_agent.literature.cug_catalog import CugJournalRankingProvider
from veriwrite_agent.literature.doi import DoiOrgResolver, DoiRisMetadataProvider
from veriwrite_agent.literature.norwegian_register import (
    NorwegianRegisterRankingProvider,
)
from veriwrite_agent.llm.deepseek_client import DeepSeekClient
from veriwrite_agent.models.literature_discovery import (
    CandidateDecision,
    LiteratureSearchPlan,
)
from veriwrite_agent.models.literature_selection import (
    BalancedLiteratureSelection,
    ConfirmedLiteratureSearchBlueprint,
    LiteratureRelevanceAssessmentBatch,
    LiteratureSearchBlueprint,
    LiteratureSelectionCandidate,
    ThemedLiteratureSearchPlan,
)
from veriwrite_agent.models.literature_verification import (
    LiteratureVerificationBatch,
)
from veriwrite_agent.models.requirement_workflow import ConfirmedRequirementSpec
from veriwrite_agent.services.literature_blueprint_planner import (
    LiteratureBlueprintPlanner,
)
from veriwrite_agent.services.literature_blueprint_search import (
    LiteratureBlueprintSearchExpander,
    allocate_query_budget,
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
    LiteratureShortageQueryRefiner,
)
from veriwrite_agent.services.literature_selector import BalancedLiteratureSelector

ProgressCallback = Callable[[str, int, int, str], None]
RELEVANCE_CHECKPOINT_BATCH_SIZE = 4
SEMANTIC_RECOVERY_MAX_ROUNDS = 12
SEMANTIC_RECOVERY_MULTIPLIER = 6
MAX_STAGNANT_RECOVERY_ROUNDS = 3
LITERATURE_RESULT_SCHEMA_VERSION = "0.2.6-console"


@dataclass(frozen=True)
class LiteratureWorkbenchResult:
    """Serializable hand-off rendered and downloaded by the Streamlit UI."""

    run_id: str
    run_dir: Path
    selection: BalancedLiteratureSelection
    verifications: LiteratureVerificationBatch
    diagnostics: tuple[dict[str, object], ...]
    prefiltered_count: int
    ris_text: str
    automatic_search_exhausted: bool
    stop_reason: str
    automatic_rounds: int
    candidate_capacity: int

    @property
    def verified_count(self) -> int:
        return len(self.verifications.verified_records)

    @property
    def verification_excluded_count(self) -> int:
        return len(self.verifications.excluded_records)

    @property
    def verification_exclusion_reason_counts(self) -> dict[str, int]:
        return dict(
            Counter(
                reason
                for result in self.verifications.excluded_records
                for reason in result.reason_codes
            )
        )

    def result_payload(self) -> dict[str, object]:
        return {
            "schema_version": LITERATURE_RESULT_SCHEMA_VERSION,
            "run_id": self.run_id,
            "run_directory": str(self.run_dir),
            "prefiltered_count": self.prefiltered_count,
            "verified_count": self.verified_count,
            "verification_excluded_count": self.verification_excluded_count,
            "verification_exclusion_reason_counts": (
                self.verification_exclusion_reason_counts
            ),
            "discovery": list(self.diagnostics),
            "selection": self.selection.model_dump(mode="json"),
            "automatic_search_exhausted": self.automatic_search_exhausted,
            "stop_reason": self.stop_reason,
            "automatic_rounds": self.automatic_rounds,
            "candidate_capacity": self.candidate_capacity,
        }

    def result_json(self) -> str:
        return json.dumps(self.result_payload(), ensure_ascii=False, indent=2)


class LiteratureWorkbench:
    """Plan and run the V0.2 workflow with stage-level persistent caches."""

    def __init__(
        self,
        *,
        planner: LiteratureBlueprintPlanner | None,
        search_expander: LiteratureBlueprintSearchExpander,
        discovery_service: LiteratureDiscoveryService,
        verification_service: LiteratureIdentityVerificationService,
        relevance_scorer: LLMLiteratureRelevanceScorer,
        shortage_query_refiner: LiteratureShortageQueryRefiner | None = None,
        selector: BalancedLiteratureSelector | None = None,
    ) -> None:
        self._planner = planner
        self._search_expander = search_expander
        self._discovery_service = discovery_service
        self._verification_service = verification_service
        self._relevance_scorer = relevance_scorer
        self._shortage_query_refiner = shortage_query_refiner
        self._selector = selector or BalancedLiteratureSelector()

    @classmethod
    def live(
        cls,
        *,
        pool_multiplier: int = 2,
        doi_max_attempts: int = 3,
    ) -> LiteratureWorkbench:
        settings = LLMSettings()
        llm = DeepSeekClient(settings.for_structured_output())
        ranking = CugJournalRankingProvider.from_default_catalog()
        norwegian_ranking = NorwegianRegisterRankingProvider.from_default_catalog()
        return cls(
            planner=LiteratureBlueprintPlanner(
                llm,
                ranking.available_disciplines,
            ),
            search_expander=LiteratureBlueprintSearchExpander(
                pool_multiplier=pool_multiplier
            ),
            discovery_service=LiteratureDiscoveryService(
                CrossrefSearchProvider(),
                ranking,
                norwegian_ranking,
            ),
            verification_service=LiteratureIdentityVerificationService(
                DoiOrgResolver(max_attempts=doi_max_attempts),
                DoiRisMetadataProvider(max_attempts=doi_max_attempts),
            ),
            relevance_scorer=LLMLiteratureRelevanceScorer(llm),
            shortage_query_refiner=LiteratureShortageQueryRefiner(llm),
        )

    def plan(
        self,
        confirmed_requirement: ConfirmedRequirementSpec,
    ) -> LiteratureSearchBlueprint:
        if self._planner is None:
            raise RuntimeError("this workbench has no blueprint planner")
        return self._planner.plan(confirmed_requirement)

    def run(
        self,
        confirmed: ConfirmedLiteratureSearchBlueprint,
        *,
        cache_root: Path,
        progress: ProgressCallback | None = None,
        seed_run_dir: Path | None = None,
    ) -> LiteratureWorkbenchResult:
        run_id = blueprint_run_id(
            confirmed,
            pool_multiplier=self._search_expander.pool_multiplier,
        )
        run_dir = cache_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        if seed_run_dir is not None:
            _seed_recovery_caches(seed_run_dir, run_dir)
        _write_json(
            run_dir / "confirmed_blueprint.json",
            confirmed.model_dump(mode="json"),
        )

        diagnostics: list[dict[str, object]] = []
        decisions_by_doi: dict[str, CandidateDecision] = {}
        verifications = LiteratureVerificationBatch()
        selection: BalancedLiteratureSelection | None = None
        shortages: dict[str, int] | None = None
        automatic_rounds = 0
        stagnant_rounds = 0
        for round_index in range(self._search_expander.max_rounds):
            themed_plans = self._search_expander.expand(
                confirmed,
                round_index=round_index,
                shortages=shortages,
            )
            if not themed_plans:
                break
            automatic_rounds += 1
            decisions_by_doi, diagnostics = self._discover(
                themed_plans,
                run_dir,
                progress,
                round_index=round_index,
            )
            verifications = self._verify(
                list(decisions_by_doi.values()),
                run_dir,
                progress,
            )
            assessments = self._score(
                confirmed,
                verifications,
                run_dir,
                progress,
            )
            selection = self._select(
                confirmed,
                decisions_by_doi,
                verifications,
                assessments,
            )
            if selection.target_reached:
                break
            shortages = selection.shortages

        if (
            selection is not None
            and not selection.target_reached
            and self._shortage_query_refiner is not None
        ):
            for recovery_index in range(SEMANTIC_RECOVERY_MAX_ROUNDS):
                reused_refinement = (
                    run_dir / f"query_refinement_{recovery_index + 1}.json"
                ).is_file()
                semantic_plans = self._semantic_recovery_plans(
                    confirmed,
                    shortages=selection.shortages,
                    decisions_by_doi=decisions_by_doi,
                    run_dir=run_dir,
                    recovery_index=recovery_index,
                )
                if not semantic_plans:
                    break
                automatic_rounds += 1
                previous_candidate_count = len(decisions_by_doi)
                decisions_by_doi, diagnostics = self._discover(
                    semantic_plans,
                    run_dir,
                    progress,
                    round_index=self._search_expander.max_rounds + recovery_index,
                )
                verifications = self._verify(
                    list(decisions_by_doi.values()),
                    run_dir,
                    progress,
                )
                assessments = self._score(
                    confirmed,
                    verifications,
                    run_dir,
                    progress,
                )
                selection = self._select(
                    confirmed,
                    decisions_by_doi,
                    verifications,
                    assessments,
                )
                if selection.target_reached:
                    break
                if (
                    len(decisions_by_doi) <= previous_candidate_count
                    and not reused_refinement
                ):
                    stagnant_rounds += 1
                elif len(decisions_by_doi) > previous_candidate_count:
                    stagnant_rounds = 0
                if stagnant_rounds >= MAX_STAGNANT_RECOVERY_ROUNDS:
                    break

        if selection is not None and not selection.target_reached:
            selection = self._selector.select_with_internal_quota_reallocation(
                confirmed.blueprint,
                [
                    LiteratureSelectionCandidate(
                        verification=verification,
                        ranking=decisions_by_doi[verification.candidate.doi].ranking,
                        norwegian_ranking=(
                            decisions_by_doi[
                                verification.candidate.doi
                            ].norwegian_ranking
                        ),
                        relevance={
                            assessment.doi: assessment
                            for assessment in assessments.assessments
                        }[verification.candidate.doi],
                    )
                    for verification in verifications.verified_records
                ],
            )

        if selection is None:
            raise RuntimeError("adaptive literature retrieval produced no search window")
        candidate_capacity = adaptive_candidate_capacity(confirmed.blueprint)
        if selection.target_reached:
            stop_reason = "target_reached"
        elif len(decisions_by_doi) >= candidate_capacity:
            stop_reason = "candidate_capacity_exhausted"
        elif stagnant_rounds >= MAX_STAGNANT_RECOVERY_ROUNDS:
            stop_reason = "search_stagnated"
        else:
            stop_reason = "automatic_round_limit_reached"
        ris_text = self._selected_ris(selection, verifications)
        result = LiteratureWorkbenchResult(
            run_id=run_id,
            run_dir=run_dir,
            selection=selection,
            verifications=verifications,
            diagnostics=tuple(diagnostics),
            prefiltered_count=len(decisions_by_doi),
            ris_text=ris_text,
            automatic_search_exhausted=not selection.target_reached,
            stop_reason=stop_reason,
            automatic_rounds=automatic_rounds,
            candidate_capacity=candidate_capacity,
        )
        _write_json(run_dir / "final_result.json", result.result_payload())
        (run_dir / "selected.ris").write_text(ris_text, encoding="utf-8")
        _notify(progress, "complete", 1, 1, "V0.2 文献选择完成")
        return result

    def _semantic_recovery_plans(
        self,
        confirmed: ConfirmedLiteratureSearchBlueprint,
        *,
        shortages: dict[str, int],
        decisions_by_doi: dict[str, CandidateDecision],
        run_dir: Path,
        recovery_index: int,
    ) -> list[ThemedLiteratureSearchPlan]:
        """Create cached, boundary-preserving query rewrites for remaining gaps."""

        if self._shortage_query_refiner is None:
            return []
        positive_shortages = {
            theme_id: count for theme_id, count in shortages.items() if count > 0
        }
        remaining_capacity = max(
            0,
            adaptive_candidate_capacity(confirmed.blueprint) - len(decisions_by_doi),
        )
        if not positive_shortages or remaining_capacity <= 0:
            return []

        previous_queries: dict[str, list[str]] = {}
        for previous_index in range(recovery_index):
            previous_path = run_dir / f"query_refinement_{previous_index + 1}.json"
            if not previous_path.is_file():
                continue
            previous_batch = _load_cached_query_refinement(previous_path)
            if previous_batch is None:
                continue
            for theme in previous_batch.themes:
                previous_queries.setdefault(theme.theme_id, []).extend(
                    theme.search_queries
                )

        cache_path = run_dir / f"query_refinement_{recovery_index + 1}.json"
        refinement = _load_cached_query_refinement(cache_path)
        if refinement is None:
            if cache_path.is_file():
                rejected_path = cache_path.with_name(
                    f"{cache_path.stem}.rejected.json"
                )
                if not rejected_path.exists():
                    shutil.copy2(cache_path, rejected_path)
            refinement = self._shortage_query_refiner.refine(
                confirmed.blueprint,
                positive_shortages,
                previous_recovery_queries=previous_queries,
            )
            cache_path.write_text(
                refinement.model_dump_json(indent=2),
                encoding="utf-8",
            )

        themes_by_id = {
            theme.theme_id: theme for theme in confirmed.blueprint.themes
        }
        plans: list[ThemedLiteratureSearchPlan] = []
        for refined in refinement.themes:
            shortage = positive_shortages.get(refined.theme_id, 0)
            theme = themes_by_id.get(refined.theme_id)
            if theme is None or shortage <= 0 or remaining_capacity <= 0:
                continue
            requested = max(12, shortage * SEMANTIC_RECOVERY_MULTIPLIER)
            budget = min(100, requested, remaining_capacity)
            query_limits = allocate_query_budget(refined.search_queries, budget)
            plan = LiteratureSearchPlan(
                topic=f"{confirmed.blueprint.topic}：{theme.section_title}",
                discipline=confirmed.blueprint.discipline,
                primary_keywords=theme.primary_keywords,
                related_keywords=theme.related_keywords,
                search_queries=refined.search_queries,
                accepted_tiers=confirmed.blueprint.accepted_tiers,
                year_from=confirmed.blueprint.year_from,
                year_to=confirmed.blueprint.year_to,
                journal_ranking_policy=confirmed.blueprint.journal_ranking_policy,
                target_eligible_count=budget,
                max_candidates=budget,
                query_offsets={query: 0 for query in refined.search_queries},
                query_limits=query_limits,
                requirement_policy=confirmed.blueprint.requirement_policy,
            )
            plans.append(
                ThemedLiteratureSearchPlan(
                    theme_id=refined.theme_id,
                    plan=plan,
                )
            )
            remaining_capacity -= budget
        return plans

    def _discover(
        self,
        themed_plans: list[ThemedLiteratureSearchPlan],
        run_dir: Path,
        progress: ProgressCallback | None,
        *,
        round_index: int,
    ) -> tuple[dict[str, CandidateDecision], list[dict[str, object]]]:
        cache_path = run_dir / "discovery_cache.json"
        cached = _read_json(cache_path, default={})
        diagnostics = list(cached.get("diagnostics", []))
        decisions = [
            CandidateDecision.model_validate(item)
            for item in cached.get(
                "candidate_decisions",
                cached.get("eligible_decisions", []),
            )
        ]
        all_decisions_by_doi = {
            decision.candidate.doi: decision for decision in decisions
        }
        query_depths = _restore_query_depths(cached, diagnostics)
        total = len(themed_plans)
        for index, themed in enumerate(themed_plans, 1):
            remaining_plan = _remaining_search_window(
                themed.plan,
                query_depths.get(themed.theme_id, {}),
            )
            if remaining_plan is None:
                _notify(
                    progress,
                    "discovery",
                    index,
                    total,
                    f"已恢复主题检索窗口：{themed.theme_id}",
                )
                continue
            _notify(
                progress,
                "discovery",
                index - 1,
                total,
                f"第 {round_index + 1} 轮正在检索缺口主题：{themed.theme_id}",
            )
            result = self._discovery_service.discover(
                remaining_plan,
                known_dois=set(all_decisions_by_doi),
                stop_when_target_reached=False,
            )
            reason_counts = Counter(
                reason
                for decision in result.excluded_records
                for reason in decision.reason_codes
            )
            diagnostics.append(
                {
                    "theme_id": themed.theme_id,
                    "round_index": round_index,
                    "scanned_count": result.scanned_count,
                    "duplicate_count": result.duplicate_count,
                    "eligible_count": len(result.eligible_records),
                    "excluded_count": len(result.excluded_records),
                    "target_reached": result.target_reached,
                    "exclusion_reason_counts": dict(reason_counts),
                    "search_plan": remaining_plan.model_dump(mode="json"),
                }
            )
            for decision in result.decisions:
                all_decisions_by_doi.setdefault(decision.candidate.doi, decision)
            theme_depths = query_depths.setdefault(themed.theme_id, {})
            for query in remaining_plan.search_queries:
                theme_depths[query] = max(
                    theme_depths.get(query, 0),
                    remaining_plan.query_offsets[query]
                    + remaining_plan.query_limits[query],
                )
            eligible_decisions = {
                doi: decision
                for doi, decision in all_decisions_by_doi.items()
                if decision.status == "eligible"
            }
            _write_json(
                cache_path,
                {
                    "schema_version": "0.2-discovery.2",
                    "query_depths": query_depths,
                    "diagnostics": diagnostics,
                    "candidate_decisions": [
                        decision.model_dump(mode="json")
                        for decision in all_decisions_by_doi.values()
                    ],
                    "eligible_decisions": [
                        decision.model_dump(mode="json")
                        for decision in eligible_decisions.values()
                    ],
                },
            )
            _notify(
                progress,
                "discovery",
                index,
                total,
                f"主题 {themed.theme_id} 本轮检索已缓存",
            )
        eligible_decisions = {
            doi: decision
            for doi, decision in all_decisions_by_doi.items()
            if decision.status == "eligible"
        }
        return eligible_decisions, diagnostics

    def _select(
        self,
        confirmed: ConfirmedLiteratureSearchBlueprint,
        decisions_by_doi: dict[str, CandidateDecision],
        verifications: LiteratureVerificationBatch,
        assessments: LiteratureRelevanceAssessmentBatch,
    ) -> BalancedLiteratureSelection:
        relevance_by_doi = {
            assessment.doi: assessment for assessment in assessments.assessments
        }
        candidates = [
            LiteratureSelectionCandidate(
                verification=verification,
                ranking=decisions_by_doi[verification.candidate.doi].ranking,
                norwegian_ranking=(
                    decisions_by_doi[verification.candidate.doi].norwegian_ranking
                ),
                relevance=relevance_by_doi[verification.candidate.doi],
            )
            for verification in verifications.verified_records
        ]
        return self._selector.select(confirmed.blueprint, candidates)

    def _verify(
        self,
        decisions: list[CandidateDecision],
        run_dir: Path,
        progress: ProgressCallback | None,
    ) -> LiteratureVerificationBatch:
        cache_path = run_dir / "verification_cache.json"
        if cache_path.is_file():
            batch = LiteratureVerificationBatch.model_validate_json(
                cache_path.read_text(encoding="utf-8")
            )
        else:
            batch = LiteratureVerificationBatch()
        active_dois = {decision.candidate.doi for decision in decisions}
        completed = {result.candidate.doi for result in batch.results}
        total = len(decisions)
        for index, decision in enumerate(decisions, 1):
            doi = decision.candidate.doi
            if doi in completed:
                _notify(progress, "verification", index, total, f"已恢复 DOI：{doi}")
                continue
            _notify(
                progress,
                "verification",
                index - 1,
                total,
                f"正在验证 DOI：{doi}",
            )
            batch.results.append(
                self._verification_service.verify(decision.candidate)
            )
            completed.add(doi)
            cache_path.write_text(
                batch.model_dump_json(indent=2),
                encoding="utf-8",
            )
            _notify(progress, "verification", index, total, f"DOI 验证完成：{doi}")
        return LiteratureVerificationBatch(
            results=[
                result
                for result in batch.results
                if result.candidate.doi in active_dois
            ]
        )

    def _score(
        self,
        confirmed: ConfirmedLiteratureSearchBlueprint,
        verifications: LiteratureVerificationBatch,
        run_dir: Path,
        progress: ProgressCallback | None,
    ) -> LiteratureRelevanceAssessmentBatch:
        cache_path = run_dir / "relevance_cache.json"
        if cache_path.is_file():
            cached_payload = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached_payload.get("schema_version") == "0.2-admission.1":
                batch = LiteratureRelevanceAssessmentBatch.model_validate(
                    cached_payload
                )
            else:
                # Pre-admission relevance scores cannot be reused: they never answered
                # which concrete claim a paper supports or whether it is out of scope.
                batch = LiteratureRelevanceAssessmentBatch()
        else:
            batch = LiteratureRelevanceAssessmentBatch()
        verified_records = verifications.verified_records
        verified_dois = {result.candidate.doi for result in verified_records}
        scored = {
            assessment.doi
            for assessment in batch.assessments
            if assessment.doi in verified_dois
        }
        pending = [
            result
            for result in verified_records
            if result.candidate.doi not in scored
        ]
        total = len(verified_records)
        completed_before_run = total - len(pending)
        for start in range(0, len(pending), RELEVANCE_CHECKPOINT_BATCH_SIZE):
            chunk = pending[start : start + RELEVANCE_CHECKPOINT_BATCH_SIZE]
            completed = completed_before_run + start
            _notify(
                progress,
                "relevance",
                completed,
                total,
                "正在评估已验证论文与各主题的相关性；结果将按微批次缓存",
            )
            batch.assessments.extend(
                self._relevance_scorer.score(confirmed.blueprint, chunk)
            )
            cache_path.write_text(
                batch.model_dump_json(indent=2),
                encoding="utf-8",
            )
            _notify(
                progress,
                "relevance",
                completed + len(chunk),
                total,
                "相关性评分微批次已缓存",
            )
        if not pending:
            _notify(
                progress,
                "relevance",
                total,
                total,
                "已恢复全部相关性评分",
            )
        return batch

    @staticmethod
    def _selected_ris(
        selection: BalancedLiteratureSelection,
        verifications: LiteratureVerificationBatch,
    ) -> str:
        verification_by_doi = {
            result.candidate.doi: result for result in verifications.results
        }
        records: list[str] = []
        for selected in selection.selected:
            authority = verification_by_doi[selected.doi].authority
            if authority is None or not authority.raw_ris:
                raise RuntimeError("selected paper is missing cached authority RIS")
            records.append(authority.raw_ris.strip())
        return "\n\n".join(records) + ("\n" if records else "")


def _load_cached_query_refinement(
    cache_path: Path,
) -> LiteratureQueryRefinementBatch | None:
    """Return a valid cached plan; old malformed caches are repaired by the caller."""

    if not cache_path.is_file():
        return None
    try:
        return LiteratureQueryRefinementBatch.model_validate_json(
            cache_path.read_text(encoding="utf-8")
        )
    except ValueError:
        return None


def _remaining_search_window(
    plan: LiteratureSearchPlan,
    cached_depths: dict[str, int],
) -> LiteratureSearchPlan | None:
    """Return only the portion of a query window not already persisted locally."""

    planned_offsets = plan.query_offsets or {
        query: 0 for query in plan.search_queries
    }
    planned_limits = plan.query_limits or allocate_query_budget(
        plan.search_queries,
        plan.max_candidates,
    )
    offsets: dict[str, int] = {}
    limits: dict[str, int] = {}
    for query in plan.search_queries:
        window_start = planned_offsets[query]
        window_end = window_start + planned_limits[query]
        remaining_start = max(window_start, cached_depths.get(query, 0))
        if remaining_start >= window_end:
            continue
        offsets[query] = remaining_start
        limits[query] = window_end - remaining_start
    if not limits:
        return None
    remaining_total = sum(limits.values())
    return LiteratureSearchPlan.model_validate(
        plan.model_copy(
            update={
                "search_queries": list(limits),
                "max_candidates": remaining_total,
                "target_eligible_count": min(
                    plan.target_eligible_count,
                    remaining_total,
                ),
                "query_offsets": offsets,
                "query_limits": limits,
            }
        ).model_dump(mode="python")
    )


def _restore_query_depths(
    cached: dict[str, object],
    diagnostics: list[dict[str, object]],
) -> dict[str, dict[str, int]]:
    """Restore adaptive cursors and migrate pre-window discovery diagnostics."""

    restored: dict[str, dict[str, int]] = {}
    raw_depths = cached.get("query_depths")
    if isinstance(raw_depths, dict):
        for theme_id, values in raw_depths.items():
            if not isinstance(theme_id, str) or not isinstance(values, dict):
                continue
            restored[theme_id] = {
                str(query): int(depth)
                for query, depth in values.items()
                if isinstance(query, str)
                and isinstance(depth, int)
                and not isinstance(depth, bool)
                and depth >= 0
            }

    for diagnostic in diagnostics:
        theme_id = diagnostic.get("theme_id")
        payload = diagnostic.get("search_plan")
        if not isinstance(theme_id, str) or not isinstance(payload, dict):
            continue
        try:
            plan = LiteratureSearchPlan.model_validate(payload)
        except ValueError:
            continue
        offsets = plan.query_offsets or {query: 0 for query in plan.search_queries}
        if plan.query_limits:
            limits = plan.query_limits
        else:
            # Older runs stopped as soon as their eligible target was reached, so
            # only the observed candidate count is safe to mark as consumed.
            observed = diagnostic.get("scanned_count", 0)
            duplicates = diagnostic.get("duplicate_count", 0)
            consumed = (
                int(observed) + int(duplicates)
                if isinstance(observed, int)
                and not isinstance(observed, bool)
                and isinstance(duplicates, int)
                and not isinstance(duplicates, bool)
                else 0
            )
            limits = allocate_query_budget(
                plan.search_queries,
                min(plan.max_candidates, max(0, consumed)),
            )
        theme_depths = restored.setdefault(theme_id, {})
        for query in plan.search_queries:
            theme_depths[query] = max(
                theme_depths.get(query, 0),
                offsets[query] + limits[query],
            )
    return restored


def adaptive_candidate_capacity(blueprint: LiteratureSearchBlueprint) -> int:
    """Keep semantic shortage recovery larger than the fixed first-pass pool."""

    return min(
        1000,
        max(
            blueprint.max_candidates,
            300,
            blueprint.target_total * 15,
        ),
    )


def blueprint_run_id(
    confirmed: ConfirmedLiteratureSearchBlueprint,
    *,
    pool_multiplier: int = 2,
) -> str:
    """Stable cache identity for the exact blueprint and retrieval scale."""

    canonical = json.dumps(
        {
            "pipeline_version": "0.2.4-topic-admission-gate",
            "blueprint": confirmed.blueprint.model_dump(mode="json"),
            "pool_multiplier": pool_multiplier,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _seed_recovery_caches(source_run_dir: Path, target_run_dir: Path) -> None:
    """Reuse verified candidates while a recovery run adds targeted queries."""

    source = source_run_dir.resolve()
    target = target_run_dir.resolve()
    if source == target or not source.is_dir():
        return
    for name in (
        "discovery_cache.json",
        "verification_cache.json",
        "relevance_cache.json",
    ):
        source_file = source / name
        target_file = target / name
        if not source_file.is_file():
            continue
        if not target_file.exists():
            shutil.copy2(source_file, target_file)
            continue
        if name == "discovery_cache.json":
            source_payload = _read_json(source_file, default={})
            target_payload = _read_json(target_file, default={})
            _write_json(
                target_file,
                _merge_discovery_caches(source_payload, target_payload),
            )
        elif name == "verification_cache.json":
            source_batch = LiteratureVerificationBatch.model_validate_json(
                source_file.read_text(encoding="utf-8")
            )
            target_batch = LiteratureVerificationBatch.model_validate_json(
                target_file.read_text(encoding="utf-8")
            )
            merged = {
                result.candidate.doi: result
                for result in [*source_batch.results, *target_batch.results]
            }
            target_file.write_text(
                LiteratureVerificationBatch(
                    results=list(merged.values())
                ).model_dump_json(indent=2),
                encoding="utf-8",
            )
        elif name == "relevance_cache.json":
            source_batch = LiteratureRelevanceAssessmentBatch.model_validate_json(
                source_file.read_text(encoding="utf-8")
            )
            target_batch = LiteratureRelevanceAssessmentBatch.model_validate_json(
                target_file.read_text(encoding="utf-8")
            )
            merged = {
                assessment.doi: assessment
                for assessment in [
                    *source_batch.assessments,
                    *target_batch.assessments,
                ]
            }
            target_file.write_text(
                LiteratureRelevanceAssessmentBatch(
                    assessments=list(merged.values())
                ).model_dump_json(indent=2),
                encoding="utf-8",
            )


def _merge_discovery_caches(
    source: dict[str, object],
    target: dict[str, object],
) -> dict[str, object]:
    """Merge candidates and query cursors without replaying prior windows."""

    source_decisions = source.get(
        "candidate_decisions",
        source.get("eligible_decisions", []),
    )
    target_decisions = target.get(
        "candidate_decisions",
        target.get("eligible_decisions", []),
    )
    decisions: dict[str, CandidateDecision] = {}
    for payload in [
        *(source_decisions if isinstance(source_decisions, list) else []),
        *(target_decisions if isinstance(target_decisions, list) else []),
    ]:
        decision = CandidateDecision.model_validate(payload)
        decisions[decision.candidate.doi] = decision

    query_depths: dict[str, dict[str, int]] = {}
    for payload in (source.get("query_depths"), target.get("query_depths")):
        if not isinstance(payload, dict):
            continue
        for theme_id, raw_queries in payload.items():
            if not isinstance(theme_id, str) or not isinstance(raw_queries, dict):
                continue
            theme_depths = query_depths.setdefault(theme_id, {})
            for query, depth in raw_queries.items():
                if (
                    isinstance(query, str)
                    and isinstance(depth, int)
                    and not isinstance(depth, bool)
                    and depth >= 0
                ):
                    theme_depths[query] = max(theme_depths.get(query, 0), depth)

    diagnostics: list[dict[str, object]] = []
    seen_diagnostics: set[str] = set()
    for payload in (source.get("diagnostics"), target.get("diagnostics")):
        if not isinstance(payload, list):
            continue
        for item in payload:
            if not isinstance(item, dict):
                continue
            identity = json.dumps(item, ensure_ascii=False, sort_keys=True)
            if identity in seen_diagnostics:
                continue
            seen_diagnostics.add(identity)
            diagnostics.append(item)

    eligible = [
        decision
        for decision in decisions.values()
        if decision.status == "eligible"
    ]
    return {
        "schema_version": "0.2-discovery.2",
        "query_depths": query_depths,
        "diagnostics": diagnostics,
        "candidate_decisions": [
            decision.model_dump(mode="json") for decision in decisions.values()
        ],
        "eligible_decisions": [
            decision.model_dump(mode="json") for decision in eligible
        ],
    }


def _notify(
    callback: ProgressCallback | None,
    stage: str,
    current: int,
    total: int,
    message: str,
) -> None:
    if callback is not None:
        callback(stage, current, total, message)


def _read_json(path: Path, *, default: object) -> dict[str, object]:
    if not path.is_file():
        if not isinstance(default, dict):
            raise TypeError("default cache value must be a dictionary")
        return default
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"cache file root must be an object: {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
