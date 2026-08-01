"""Resumable application layer for the V0.2 literature console."""

from __future__ import annotations

import hashlib
import json
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
from veriwrite_agent.models.literature_discovery import CandidateDecision
from veriwrite_agent.models.literature_selection import (
    BalancedLiteratureSelection,
    ConfirmedLiteratureSearchBlueprint,
    LiteratureRelevanceAssessmentBatch,
    LiteratureSearchBlueprint,
    LiteratureSelectionCandidate,
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
)
from veriwrite_agent.services.literature_discovery import LiteratureDiscoveryService
from veriwrite_agent.services.literature_identity_verification import (
    LiteratureIdentityVerificationService,
)
from veriwrite_agent.services.literature_relevance_scorer import (
    LLMLiteratureRelevanceScorer,
)
from veriwrite_agent.services.literature_selector import BalancedLiteratureSelector

ProgressCallback = Callable[[str, int, int, str], None]


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
            "schema_version": "0.2.3-console",
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
        selector: BalancedLiteratureSelector | None = None,
    ) -> None:
        self._planner = planner
        self._search_expander = search_expander
        self._discovery_service = discovery_service
        self._verification_service = verification_service
        self._relevance_scorer = relevance_scorer
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
    ) -> LiteratureWorkbenchResult:
        run_id = blueprint_run_id(
            confirmed,
            pool_multiplier=self._search_expander.pool_multiplier,
        )
        run_dir = cache_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            run_dir / "confirmed_blueprint.json",
            confirmed.model_dump(mode="json"),
        )

        decisions_by_doi, diagnostics = self._discover(
            confirmed,
            run_dir,
            progress,
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
        selection = self._selector.select(confirmed.blueprint, candidates)
        ris_text = self._selected_ris(selection, verifications)
        result = LiteratureWorkbenchResult(
            run_id=run_id,
            run_dir=run_dir,
            selection=selection,
            verifications=verifications,
            diagnostics=tuple(diagnostics),
            prefiltered_count=len(decisions_by_doi),
            ris_text=ris_text,
        )
        _write_json(run_dir / "final_result.json", result.result_payload())
        (run_dir / "selected.ris").write_text(ris_text, encoding="utf-8")
        _notify(progress, "complete", 1, 1, "V0.2 文献选择完成")
        return result

    def _discover(
        self,
        confirmed: ConfirmedLiteratureSearchBlueprint,
        run_dir: Path,
        progress: ProgressCallback | None,
    ) -> tuple[dict[str, CandidateDecision], list[dict[str, object]]]:
        cache_path = run_dir / "discovery_cache.json"
        cached = _read_json(cache_path, default={})
        diagnostics = list(cached.get("diagnostics", []))
        completed = set(cached.get("completed_theme_ids", []))
        decisions = [
            CandidateDecision.model_validate(item)
            for item in cached.get("eligible_decisions", [])
        ]
        decisions_by_doi = {
            decision.candidate.doi: decision for decision in decisions
        }
        themed_plans = self._search_expander.expand(confirmed)
        total = len(themed_plans)
        for index, themed in enumerate(themed_plans, 1):
            if themed.theme_id in completed:
                _notify(
                    progress,
                    "discovery",
                    index,
                    total,
                    f"已恢复主题：{themed.theme_id}",
                )
                continue
            _notify(
                progress,
                "discovery",
                index - 1,
                total,
                f"正在检索主题：{themed.theme_id}",
            )
            result = self._discovery_service.discover(themed.plan)
            reason_counts = Counter(
                reason
                for decision in result.excluded_records
                for reason in decision.reason_codes
            )
            diagnostics.append(
                {
                    "theme_id": themed.theme_id,
                    "scanned_count": result.scanned_count,
                    "duplicate_count": result.duplicate_count,
                    "eligible_count": len(result.eligible_records),
                    "excluded_count": len(result.excluded_records),
                    "target_reached": result.target_reached,
                    "exclusion_reason_counts": dict(reason_counts),
                    "search_plan": themed.plan.model_dump(mode="json"),
                }
            )
            for decision in result.eligible_records:
                decisions_by_doi.setdefault(decision.candidate.doi, decision)
            completed.add(themed.theme_id)
            _write_json(
                cache_path,
                {
                    "completed_theme_ids": sorted(completed),
                    "diagnostics": diagnostics,
                    "eligible_decisions": [
                        decision.model_dump(mode="json")
                        for decision in decisions_by_doi.values()
                    ],
                },
            )
            _notify(
                progress,
                "discovery",
                index,
                total,
                f"主题 {themed.theme_id} 检索完成",
            )
        return decisions_by_doi, diagnostics

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
        return batch

    def _score(
        self,
        confirmed: ConfirmedLiteratureSearchBlueprint,
        verifications: LiteratureVerificationBatch,
        run_dir: Path,
        progress: ProgressCallback | None,
    ) -> LiteratureRelevanceAssessmentBatch:
        cache_path = run_dir / "relevance_cache.json"
        if cache_path.is_file():
            batch = LiteratureRelevanceAssessmentBatch.model_validate_json(
                cache_path.read_text(encoding="utf-8")
            )
        else:
            batch = LiteratureRelevanceAssessmentBatch()
        scored = {assessment.doi for assessment in batch.assessments}
        pending = [
            result
            for result in verifications.verified_records
            if result.candidate.doi not in scored
        ]
        total = len(pending)
        batch_size = 20
        for start in range(0, total, batch_size):
            chunk = pending[start : start + batch_size]
            _notify(
                progress,
                "relevance",
                start,
                total,
                "正在评估已验证论文与各主题的相关性",
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
                min(start + len(chunk), total),
                total,
                "相关性评分批次已缓存",
            )
        if not pending:
            _notify(
                progress,
                "relevance",
                len(batch.assessments),
                len(batch.assessments),
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


def blueprint_run_id(
    confirmed: ConfirmedLiteratureSearchBlueprint,
    *,
    pool_multiplier: int = 2,
) -> str:
    """Stable cache identity for the exact blueprint and retrieval scale."""

    canonical = json.dumps(
        {
            "pipeline_version": "0.2.3-norwegian-register-2025",
            "blueprint": confirmed.blueprint.model_dump(mode="json"),
            "pool_multiplier": pool_multiplier,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


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
