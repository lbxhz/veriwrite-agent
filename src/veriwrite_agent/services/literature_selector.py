"""Deterministically select a balanced pool from verified, scored papers."""

from __future__ import annotations

from veriwrite_agent.models.literature_discovery import CugTier
from veriwrite_agent.models.literature_selection import (
    BalancedLiteratureSelection,
    LiteratureSearchBlueprint,
    LiteratureSelectionCandidate,
    SelectedLiteratureRecord,
)

TIER_RANK: dict[CugTier, int] = {
    "T1": 1,
    "T2": 2,
    "T3": 3,
    "T4": 4,
    "T5": 5,
    "T6": 6,
}


class BalancedLiteratureSelector:
    """Apply relevance first, then journal tier and publication year."""

    def select(
        self,
        blueprint: LiteratureSearchBlueprint,
        candidates: list[LiteratureSelectionCandidate],
    ) -> BalancedLiteratureSelection:
        theme_ids = {theme.theme_id for theme in blueprint.themes}
        for candidate in candidates:
            scored_theme_ids = {
                score.theme_id for score in candidate.relevance.theme_scores
            }
            if scored_theme_ids != theme_ids:
                raise ValueError(
                    "every candidate must be scored against every blueprint theme"
                )

        selected: list[SelectedLiteratureRecord] = []
        used_dois: set[str] = set()
        theme_counts = {theme.theme_id: 0 for theme in blueprint.themes}
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
                    and self._score(candidate, theme.theme_id)
                    >= blueprint.relevance_threshold
                ]
                if not eligible:
                    continue
                eligible.sort(
                    key=lambda candidate: self._sort_key(
                        candidate,
                        theme.theme_id,
                    )
                )
                chosen = eligible[0]
                record = self._selected_record(chosen, theme.theme_id)
                selected.append(record)
                used_dois.add(record.doi)
                theme_counts[theme.theme_id] += 1
                progress = True

        shortages = {
            theme.theme_id: theme.target_count - theme_counts[theme.theme_id]
            for theme in themes
            if theme_counts[theme.theme_id] < theme.target_count
        }
        return BalancedLiteratureSelection(
            blueprint=blueprint,
            selected=selected,
            shortages=shortages,
            target_reached=len(selected) == blueprint.target_total,
        )

    @staticmethod
    def _score(
        candidate: LiteratureSelectionCandidate,
        theme_id: str,
    ) -> float:
        return next(
            score.score
            for score in candidate.relevance.theme_scores
            if score.theme_id == theme_id
        )

    def _sort_key(
        self,
        candidate: LiteratureSelectionCandidate,
        theme_id: str,
    ) -> tuple[float, int, int, str]:
        score = self._score(candidate, theme_id)
        tier = candidate.ranking.resolved_tier
        metadata = candidate.verification.authority
        if metadata is None or metadata.metadata is None:
            raise ValueError("selection candidate evidence is incomplete")
        year = metadata.metadata.year
        if year is None:
            raise ValueError("verified selection candidates need a year")
        return (
            -score,
            TIER_RANK[tier] if tier is not None else 7,
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
        authority = candidate.verification.authority
        if authority is None or authority.metadata is None:
            raise ValueError("selection candidate evidence is incomplete")
        metadata = authority.metadata
        if (
            metadata.doi is None
            or metadata.title is None
            or metadata.year is None
        ):
            raise ValueError("verified RIS is missing selection fields")
        ranking_reason = (
            f"中国地质大学（武汉）2023版期刊等级{tier}"
            if tier is not None
            else (
                "地大2023版目录未给出该学科唯一等级；"
                "未据此判假，作为同等相关性下的末位偏好"
            )
        )
        return SelectedLiteratureRecord(
            doi=metadata.doi,
            title=metadata.title,
            theme_id=theme_id,
            relevance_score=score,
            cug_tier=tier,
            ranking_status=candidate.ranking.status,
            year=metadata.year,
            selection_reasons=[
                "通过Crossref RIS与DOI真实性验证",
                f"与章节主题相关性得分{score:.2f}",
                ranking_reason,
                f"发表年份{metadata.year}",
            ],
        )
