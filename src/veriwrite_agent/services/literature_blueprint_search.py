"""Expand a confirmed blueprint into non-overlapping adaptive search windows."""

from __future__ import annotations

from veriwrite_agent.models.literature_discovery import LiteratureSearchPlan
from veriwrite_agent.models.literature_selection import (
    ConfirmedLiteratureSearchBlueprint,
    ThemedLiteratureSearchPlan,
)


class UnconfirmedLiteratureBlueprintError(ValueError):
    """Raised when retrieval is attempted with an unconfirmed draft."""


class LiteratureBlueprintSearchExpander:
    """Allocate target-driven windows that grow only when selection has shortages."""

    maximum_pool_multiplier = 10
    expansion_step = 2

    def __init__(self, *, pool_multiplier: int = 3) -> None:
        if not 1 <= pool_multiplier <= 10:
            raise ValueError("pool_multiplier must be between 1 and 10")
        self._pool_multiplier = pool_multiplier

    @property
    def pool_multiplier(self) -> int:
        """Initial candidate-pool scale that participates in cache identity."""

        return self._pool_multiplier

    @property
    def max_rounds(self) -> int:
        """Maximum automatic rounds needed to reach the ten-times safety cap."""

        remaining = self.maximum_pool_multiplier - self._pool_multiplier
        return 1 + max(0, (remaining + self.expansion_step - 1) // self.expansion_step)

    def expand(
        self,
        confirmed: ConfirmedLiteratureSearchBlueprint,
        *,
        round_index: int = 0,
        shortages: dict[str, int] | None = None,
    ) -> list[ThemedLiteratureSearchPlan]:
        if not isinstance(confirmed, ConfirmedLiteratureSearchBlueprint):
            raise UnconfirmedLiteratureBlueprintError(
                "literature retrieval requires a user-confirmed search blueprint"
            )
        if round_index < 0:
            raise ValueError("round_index cannot be negative")
        blueprint = confirmed.blueprint
        multiplier = min(
            self.maximum_pool_multiplier,
            self._pool_multiplier + round_index * self.expansion_step,
        )
        previous_multiplier = (
            0
            if round_index == 0
            else min(
                self.maximum_pool_multiplier,
                self._pool_multiplier + (round_index - 1) * self.expansion_step,
            )
        )
        results: list[ThemedLiteratureSearchPlan] = []
        for theme in blueprint.themes:
            if shortages is not None and shortages.get(theme.theme_id, 0) <= 0:
                continue
            cumulative_budget = min(
                1000,
                theme.target_count * multiplier,
            )
            previous_budget = min(
                1000,
                theme.target_count * previous_multiplier,
            )
            current_offsets = allocate_query_budget(
                theme.search_queries,
                previous_budget,
            )
            cumulative_limits = allocate_query_budget(
                theme.search_queries,
                cumulative_budget,
            )
            query_limits = {
                query: cumulative_limits[query] - current_offsets[query]
                for query in theme.search_queries
            }
            window_size = sum(query_limits.values())
            if window_size <= 0:
                continue
            shortage = (
                shortages.get(theme.theme_id, theme.target_count)
                if shortages is not None
                else theme.target_count
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
                target_eligible_count=min(
                    100,
                    window_size,
                    max(1, shortage * self._pool_multiplier),
                ),
                max_candidates=window_size,
                query_offsets=current_offsets,
                query_limits=query_limits,
                requirement_policy=blueprint.requirement_policy,
            )
            results.append(
                ThemedLiteratureSearchPlan(
                    theme_id=theme.theme_id,
                    plan=plan,
                )
            )
        return results


def allocate_query_budget(
    queries: list[str],
    total: int,
) -> dict[str, int]:
    """Allocate an exact cumulative depth across stable query order."""

    base, remainder = divmod(total, len(queries))
    return {
        query: base + (1 if index < remainder else 0)
        for index, query in enumerate(queries)
    }
