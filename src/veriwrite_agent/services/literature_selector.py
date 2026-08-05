"""Deterministically select a balanced pool from verified, scored papers."""

from __future__ import annotations

from math import ceil

from veriwrite_agent.models.literature_discovery import CugTier
from veriwrite_agent.models.literature_selection import (
    BalancedLiteratureSelection,
    LiteratureSearchBlueprint,
    LiteratureSelectionCandidate,
    SelectedLiteratureRecord,
)
from veriwrite_agent.services.requirement_policy import is_foreign_literature

TIER_RANK: dict[CugTier, int] = {
    "T1": 1,
    "T2": 2,
    "T3": 3,
    "T4": 4,
    "T5": 5,
    "T6": 6,
}
NORWEGIAN_LEVEL_RANK = {2: 1, 1: 2, 0: 3}


class BalancedLiteratureSelector:
    """Apply relevance first, then journal tier and publication year."""

    def select(
        self,
        blueprint: LiteratureSearchBlueprint,
        candidates: list[LiteratureSelectionCandidate],
    ) -> BalancedLiteratureSelection:
        theme_ids = {theme.theme_id for theme in blueprint.themes}
        for candidate in candidates:
            scored_theme_ids = {score.theme_id for score in candidate.relevance.theme_scores}
            if scored_theme_ids != theme_ids:
                raise ValueError("every candidate must be scored against every blueprint theme")

        selected: list[SelectedLiteratureRecord] = []
        used_dois: set[str] = set()
        theme_counts = {theme.theme_id: 0 for theme in blueprint.themes}
        supporting_counts = {theme.theme_id: 0 for theme in blueprint.themes}
        themes = sorted(
            blueprint.themes,
            key=lambda theme: (theme.priority, theme.theme_id),
        )

        progress = True
        while progress:
            progress = False
            for theme in themes:
                if theme_counts[theme.theme_id] >= theme.target_count:
                    continue
                eligible = [
                    candidate
                    for candidate in candidates
                    if candidate.verification.candidate.doi not in used_dois
                    and candidate.relevance.admission_status == "admit"
                    and candidate.relevance.suitable_section_id == theme.theme_id
                    and candidate.relevance.centrality in {"central", "supporting"}
                    and self._score(candidate, theme.theme_id) >= blueprint.relevance_threshold
                    and (
                        candidate.relevance.centrality == "central"
                        or supporting_counts[theme.theme_id]
                        < ceil(theme.target_count * blueprint.max_contextual_share)
                    )
                ]
                if not eligible:
                    continue
                minimum_foreign_count = (
                    blueprint.requirement_policy.references.minimum_foreign_count
                    if blueprint.requirement_policy is not None
                    else None
                )
                foreign_selected = sum(item.is_foreign for item in selected)
                prefer_foreign = (
                    minimum_foreign_count is not None and foreign_selected < minimum_foreign_count
                )
                eligible.sort(
                    key=lambda candidate: self._sort_key(
                        candidate,
                        theme.theme_id,
                        prefer_foreign=prefer_foreign,
                        blueprint=blueprint,
                    )
                )
                chosen = eligible[0]
                record = self._selected_record(chosen, theme.theme_id)
                selected.append(record)
                used_dois.add(record.doi)
                theme_counts[theme.theme_id] += 1
                if chosen.relevance.centrality == "supporting":
                    supporting_counts[theme.theme_id] += 1
                progress = True

        shortages = {
            theme.theme_id: theme.target_count - theme_counts[theme.theme_id]
            for theme in themes
            if theme_counts[theme.theme_id] < theme.target_count
        }
        policy_issues: list[str] = []
        if blueprint.requirement_policy is not None:
            minimum_foreign_count = blueprint.requirement_policy.references.minimum_foreign_count
            foreign_count = sum(item.is_foreign for item in selected)
            if minimum_foreign_count is not None and foreign_count < minimum_foreign_count:
                policy_issues.append(
                    "minimum_foreign_count_not_reached:"
                    f"required={minimum_foreign_count}:actual={foreign_count}"
                )
        admission_exclusions: dict[str, int] = {}
        for candidate in candidates:
            if candidate.relevance.admission_status == "admit":
                continue
            key = (
                candidate.relevance.exclusion_reason
                or candidate.relevance.admission_status
            )
            admission_exclusions[key] = admission_exclusions.get(key, 0) + 1
        return BalancedLiteratureSelection(
            blueprint=blueprint,
            selected=selected,
            shortages=shortages,
            policy_issues=policy_issues,
            admission_exclusions=admission_exclusions,
            target_reached=(len(selected) == blueprint.target_total and not policy_issues),
        )

    @staticmethod
    def _score(
        candidate: LiteratureSelectionCandidate,
        theme_id: str,
    ) -> float:
        return next(
            score.score for score in candidate.relevance.theme_scores if score.theme_id == theme_id
        )

    def _sort_key(
        self,
        candidate: LiteratureSelectionCandidate,
        theme_id: str,
        *,
        prefer_foreign: bool,
        blueprint: LiteratureSearchBlueprint,
    ) -> tuple[int, int, float, int, int, int, int, str]:
        score = self._score(candidate, theme_id)
        tier = candidate.ranking.resolved_tier
        norwegian_level = (
            candidate.norwegian_ranking.resolved_level
            if candidate.norwegian_ranking is not None
            else None
        )
        metadata = candidate.verification.authority
        if metadata is None or metadata.metadata is None:
            raise ValueError("selection candidate evidence is incomplete")
        year = metadata.metadata.year
        if year is None:
            raise ValueError("verified selection candidates need a year")
        foreign_rank = 0
        if prefer_foreign and not is_foreign_literature(
            language=candidate.verification.candidate.language,
            title=metadata.metadata.title or candidate.verification.candidate.title,
        ):
            foreign_rank = 1
        source_rank = 0
        if blueprint.requirement_policy is not None:
            source_type = candidate.verification.candidate.source_type.casefold()
            preferred = {
                value.casefold()
                for value in blueprint.requirement_policy.references.preferred_source_types
            }
            discouraged = {
                value.casefold()
                for value in blueprint.requirement_policy.references.discouraged_source_types
            }
            if source_type in discouraged:
                source_rank = 2
            elif preferred and source_type not in preferred:
                source_rank = 1
        return (
            foreign_rank,
            0 if candidate.relevance.centrality == "central" else 1,
            -score,
            source_rank,
            TIER_RANK[tier] if tier is not None else 7,
            (NORWEGIAN_LEVEL_RANK[norwegian_level] if norwegian_level is not None else 4),
            -year,
            candidate.verification.candidate.doi,
        )

    def _selected_record(
        self,
        candidate: LiteratureSelectionCandidate,
        theme_id: str,
    ) -> SelectedLiteratureRecord:
        score = self._score(candidate, theme_id)
        tier = candidate.ranking.resolved_tier
        norwegian_ranking = candidate.norwegian_ranking
        norwegian_level = (
            norwegian_ranking.resolved_level if norwegian_ranking is not None else None
        )
        authority = candidate.verification.authority
        if authority is None or authority.metadata is None:
            raise ValueError("selection candidate evidence is incomplete")
        metadata = authority.metadata
        if metadata.doi is None or metadata.title is None or metadata.year is None:
            raise ValueError("verified RIS is missing selection fields")
        source_candidate = candidate.verification.candidate
        foreign = is_foreign_literature(
            language=source_candidate.language,
            title=metadata.title,
        )
        ranking_reason = (
            f"中国地质大学（武汉）2023版期刊等级{tier}"
            if tier is not None
            else ("地大2023版目录未给出该学科唯一等级；未据此判假，作为同等相关性下的末位偏好")
        )
        norwegian_reason = (
            f"挪威国家学术出版渠道目录2025等级：Level {norwegian_level}"
            if norwegian_level is not None
            else "挪威国家学术出版渠道目录2025未取得唯一等级"
        )
        return SelectedLiteratureRecord(
            doi=metadata.doi,
            title=metadata.title,
            authors=metadata.authors,
            journal=metadata.journal_title or source_candidate.journal_title,
            publisher=metadata.publisher or source_candidate.publisher,
            language=source_candidate.language,
            source_type=source_candidate.source_type,
            is_foreign=foreign,
            theme_id=theme_id,
            relevance_score=score,
            centrality=candidate.relevance.centrality,
            supported_claim=candidate.relevance.supported_claim,
            suitable_section_id=candidate.relevance.suitable_section_id,
            use_boundary=candidate.relevance.use_boundary,
            cug_tier=tier,
            ranking_status=candidate.ranking.status,
            norwegian_level=norwegian_level,
            norwegian_ranking_status=(
                norwegian_ranking.status if norwegian_ranking is not None else None
            ),
            norwegian_match_basis=(
                norwegian_ranking.match_basis if norwegian_ranking is not None else None
            ),
            year=metadata.year,
            selection_reasons=[
                "通过Crossref RIS与DOI真实性验证",
                f"与章节主题相关性得分{score:.2f}",
                ranking_reason,
                norwegian_reason,
                "foreign_literature" if foreign else "domestic_or_chinese_literature",
                f"发表年份{metadata.year}",
            ],
        )
