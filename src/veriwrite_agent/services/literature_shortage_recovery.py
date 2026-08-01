"""Audited V0.2 retrieval adjustment and shortage recovery planning."""

from __future__ import annotations

from dataclasses import dataclass

from veriwrite_agent.models.literature_selection import (
    ConfirmedLiteratureSearchBlueprint,
    LiteratureSearchBlueprint,
)
from veriwrite_agent.services.literature_blueprint_confirmation import (
    LiteratureBlueprintConfirmationService,
)


@dataclass(frozen=True)
class LiteratureShortageRecoveryPlan:
    """A user-triggered retrieval expansion that does not weaken requirements."""

    confirmed_blueprint: ConfirmedLiteratureSearchBlueprint
    pool_multiplier: int
    previous_max_candidates: int
    expanded_max_candidates: int


@dataclass(frozen=True)
class LiteratureRetrievalAdjustmentPlan:
    """A user-confirmed change to target size and retrieval capacity."""

    confirmed_blueprint: ConfirmedLiteratureSearchBlueprint
    previous_target_total: int
    target_total: int
    previous_max_candidates: int
    max_candidates: int
    previous_pool_multiplier: int
    pool_multiplier: int
    theme_target_counts: dict[str, int]


class LiteratureShortageRecoveryService:
    """Adjust retrieval capacity without silently weakening V0.1 requirements."""

    def minimum_allowed_target(
        self,
        confirmed: ConfirmedLiteratureSearchBlueprint,
    ) -> int:
        """Return the V0.1 floor that a V0.2 adjustment cannot cross."""

        blueprint = confirmed.blueprint
        policy = blueprint.requirement_policy
        minimum = max(2, len(blueprint.themes))
        if policy is None:
            return minimum
        minimum = max(minimum, policy.references.minimum_total)
        if (
            policy.references.target_origin == "explicit_target"
            and not policy.references.target_is_approximate
        ):
            minimum = max(minimum, policy.references.target_total)
        return minimum

    def adjust_retrieval(
        self,
        confirmed: ConfirmedLiteratureSearchBlueprint,
        *,
        target_total: int,
        max_candidates: int,
        current_pool_multiplier: int,
        pool_multiplier: int,
    ) -> LiteratureRetrievalAdjustmentPlan:
        """Apply explicit UI choices and proportionally redistribute theme quotas."""

        blueprint = confirmed.blueprint
        minimum_target = self.minimum_allowed_target(confirmed)
        if target_total < minimum_target:
            raise ValueError(
                f"target_total cannot be below the V0.1 floor ({minimum_target})"
            )
        if target_total > 100:
            raise ValueError("target_total cannot exceed 100 in the MVP")
        if target_total < len(blueprint.themes):
            raise ValueError("target_total must allocate at least one paper per theme")
        if not 20 <= max_candidates <= 1000:
            raise ValueError("max_candidates must be between 20 and 1000")
        if max_candidates < target_total:
            raise ValueError("max_candidates cannot be below target_total")
        if not 1 <= current_pool_multiplier <= 10:
            raise ValueError("current_pool_multiplier must be between 1 and 10")
        if not 1 <= pool_multiplier <= 10:
            raise ValueError("pool_multiplier must be between 1 and 10")
        if (
            target_total == blueprint.target_total
            and max_candidates == blueprint.max_candidates
            and pool_multiplier == current_pool_multiplier
        ):
            raise ValueError("at least one retrieval parameter must change")

        target_counts = self._redistribute_theme_targets(
            confirmed,
            target_total=target_total,
        )
        updated_themes = [
            theme.model_copy(update={"target_count": target_counts[theme.theme_id]})
            for theme in blueprint.themes
        ]
        adjusted_blueprint = LiteratureSearchBlueprint.model_validate(
            blueprint.model_copy(
                update={
                    "target_total": target_total,
                    "max_candidates": max_candidates,
                    "themes": updated_themes,
                }
            ).model_dump(mode="python")
        )
        adjusted = LiteratureBlueprintConfirmationService().confirm(
            adjusted_blueprint,
            confirmed_by=confirmed.confirmed_by,
            note=(
                "用户在V0.2结果界面明确调整检索参数："
                f"最终目标 {blueprint.target_total}→{target_total}，"
                f"候选扫描上限 {blueprint.max_candidates}→{max_candidates}，"
                f"候选池倍率 {current_pool_multiplier}→{pool_multiplier}。"
                "系统按原主题配额比例重新分配目标；主题、检索词、相关性阈值"
                "和V0.1执行策略均未被静默修改。"
            ),
            expected_policy=blueprint.requirement_policy,
        )
        return LiteratureRetrievalAdjustmentPlan(
            confirmed_blueprint=adjusted,
            previous_target_total=blueprint.target_total,
            target_total=target_total,
            previous_max_candidates=blueprint.max_candidates,
            max_candidates=max_candidates,
            previous_pool_multiplier=current_pool_multiplier,
            pool_multiplier=pool_multiplier,
            theme_target_counts=target_counts,
        )

    def expand_candidate_pool(
        self,
        confirmed: ConfirmedLiteratureSearchBlueprint,
        *,
        current_pool_multiplier: int,
        shortages: dict[str, int],
    ) -> LiteratureShortageRecoveryPlan:
        if not shortages or not any(count > 0 for count in shortages.values()):
            raise ValueError("candidate-pool recovery requires a positive theme shortage")
        if not 1 <= current_pool_multiplier <= 10:
            raise ValueError("current_pool_multiplier must be between 1 and 10")

        blueprint = confirmed.blueprint
        next_multiplier = min(10, max(4, current_pool_multiplier + 2))
        next_max_candidates = min(
            1000,
            max(500, blueprint.max_candidates + 200),
        )
        if (
            next_multiplier == current_pool_multiplier
            and next_max_candidates == blueprint.max_candidates
        ):
            raise ValueError("candidate pool already uses the maximum recovery capacity")

        expanded_blueprint = blueprint.model_copy(
            update={"max_candidates": next_max_candidates}
        )
        shortage_text = ", ".join(
            f"{theme_id}={count}"
            for theme_id, count in sorted(shortages.items())
            if count > 0
        )
        recovered = LiteratureBlueprintConfirmationService().confirm(
            expanded_blueprint,
            confirmed_by=confirmed.confirmed_by,
            note=(
                "用户在V0.2缺口界面明确触发扩大候选池；"
                f"缺口为 {shortage_text}；"
                f"候选池倍率 {current_pool_multiplier}→{next_multiplier}，"
                f"max_candidates {blueprint.max_candidates}→{next_max_candidates}。"
                "主题、配额、相关性阈值和V0.1执行策略均未降低。"
            ),
            expected_policy=blueprint.requirement_policy,
        )
        return LiteratureShortageRecoveryPlan(
            confirmed_blueprint=recovered,
            pool_multiplier=next_multiplier,
            previous_max_candidates=blueprint.max_candidates,
            expanded_max_candidates=next_max_candidates,
        )

    @staticmethod
    def _redistribute_theme_targets(
        confirmed: ConfirmedLiteratureSearchBlueprint,
        *,
        target_total: int,
    ) -> dict[str, int]:
        """Use capped largest-deficit allocation while preserving theme proportions."""

        themes = confirmed.blueprint.themes
        if target_total > len(themes) * 30:
            raise ValueError("target_total exceeds the per-theme quota capacity")
        original_total = sum(theme.target_count for theme in themes)
        ideals = [target_total * theme.target_count / original_total for theme in themes]
        allocations = [1 for _ in themes]
        for _ in range(target_total - len(themes)):
            candidates = [index for index, value in enumerate(allocations) if value < 30]
            if not candidates:
                raise ValueError("theme quota capacity is exhausted")
            selected_index = max(
                candidates,
                key=lambda index: (
                    ideals[index] - allocations[index],
                    themes[index].priority,
                    -index,
                ),
            )
            allocations[selected_index] += 1
        return {
            theme.theme_id: allocations[index]
            for index, theme in enumerate(themes)
        }
