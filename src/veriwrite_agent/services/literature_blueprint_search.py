"""Expand provisional outline themes into bounded executable search plans."""

from __future__ import annotations

import math

from veriwrite_agent.models.literature_discovery import LiteratureSearchPlan
from veriwrite_agent.models.literature_selection import (
    ConfirmedLiteratureSearchBlueprint,
    ThemedLiteratureSearchPlan,
)


class UnconfirmedLiteratureBlueprintError(ValueError):
    """Raised when retrieval is attempted with an unconfirmed draft."""


class LiteratureBlueprintSearchExpander:
    """Allocate candidate-pool budget across provisional outline themes."""

    def __init__(self, *, pool_multiplier: int = 3) -> None:
        if not 1 <= pool_multiplier <= 10:
            raise ValueError("pool_multiplier must be between 1 and 10")
        self._pool_multiplier = pool_multiplier

    @property
    def pool_multiplier(self) -> int:
        """Candidate-pool scale that participates in cache identity."""

        return self._pool_multiplier

    def expand(
        self,
        confirmed: ConfirmedLiteratureSearchBlueprint,
    ) -> list[ThemedLiteratureSearchPlan]:
        if not isinstance(confirmed, ConfirmedLiteratureSearchBlueprint):
            raise UnconfirmedLiteratureBlueprintError(
                "literature retrieval requires a user-confirmed search blueprint"
            )
        blueprint = confirmed.blueprint
        per_theme_max = max(
            1,
            math.ceil(blueprint.max_candidates / len(blueprint.themes)),
        )
        results: list[ThemedLiteratureSearchPlan] = []
        for theme in blueprint.themes:
            pool_target = min(
                100,
                per_theme_max,
                max(theme.target_count, theme.target_count * self._pool_multiplier),
            )
            plan = LiteratureSearchPlan(
                topic=f"{blueprint.topic}：{theme.section_title}",
                discipline=blueprint.discipline,
                primary_keywords=theme.primary_keywords,
                related_keywords=theme.related_keywords,
                search_queries=theme.search_queries,
                accepted_tiers=blueprint.accepted_tiers,
                year_from=blueprint.year_from,
                year_to=blueprint.year_to,
                journal_ranking_policy=blueprint.journal_ranking_policy,
                target_eligible_count=pool_target,
                max_candidates=per_theme_max,
                requirement_policy=blueprint.requirement_policy,
            )
            results.append(
                ThemedLiteratureSearchPlan(
                    theme_id=theme.theme_id,
                    plan=plan,
                )
            )
        return results
