import pytest

from veriwrite_agent.models.literature_selection import (
    ConfirmedLiteratureSearchBlueprint,
    LiteratureSearchBlueprint,
    LiteratureThemePlan,
)
from veriwrite_agent.services.literature_shortage_recovery import (
    LiteratureShortageRecoveryService,
)


def _confirmed_blueprint() -> ConfirmedLiteratureSearchBlueprint:
    return ConfirmedLiteratureSearchBlueprint(
        confirmed_by="student",
        blueprint=LiteratureSearchBlueprint(
            topic="Atmospheric remote sensing",
            discipline="Atmospheric science",
            writing_through_line="Development and challenges",
            target_total=4,
            max_candidates=300,
            themes=[
                LiteratureThemePlan(
                    theme_id="history",
                    section_title="History",
                    section_purpose="Review the history",
                    research_questions=["How did the field develop?"],
                    primary_keywords=["history"],
                    search_queries=["atmospheric remote sensing history"],
                    target_count=2,
                ),
                LiteratureThemePlan(
                    theme_id="future",
                    section_title="Future",
                    section_purpose="Review future work",
                    research_questions=["What comes next?"],
                    primary_keywords=["future"],
                    search_queries=["atmospheric remote sensing future"],
                    target_count=2,
                ),
            ],
        ),
    )


def test_recovery_expands_capacity_without_weakening_blueprint() -> None:
    original = _confirmed_blueprint()

    recovery = LiteratureShortageRecoveryService().expand_candidate_pool(
        original,
        current_pool_multiplier=2,
        shortages={"future": 2},
    )

    assert recovery.pool_multiplier == 4
    assert recovery.previous_max_candidates == 300
    assert recovery.expanded_max_candidates == 500
    assert recovery.confirmed_blueprint.blueprint.max_candidates == 500
    assert recovery.confirmed_blueprint.blueprint.target_total == 4
    assert recovery.confirmed_blueprint.blueprint.themes == original.blueprint.themes
    assert recovery.confirmed_blueprint.blueprint.relevance_threshold == 0.6
    assert "future=2" in (recovery.confirmed_blueprint.confirmation_note or "")


def test_recovery_can_expand_again_until_the_supported_limit() -> None:
    original = _confirmed_blueprint()
    first = LiteratureShortageRecoveryService().expand_candidate_pool(
        original,
        current_pool_multiplier=2,
        shortages={"future": 2},
    )

    second = LiteratureShortageRecoveryService().expand_candidate_pool(
        first.confirmed_blueprint,
        current_pool_multiplier=first.pool_multiplier,
        shortages={"future": 1},
    )

    assert second.pool_multiplier == 6
    assert second.expanded_max_candidates == 700


def test_user_can_adjust_target_and_retrieval_capacity_together() -> None:
    original = _confirmed_blueprint()

    adjustment = LiteratureShortageRecoveryService().adjust_retrieval(
        original,
        target_total=6,
        max_candidates=450,
        current_pool_multiplier=2,
        pool_multiplier=5,
    )

    blueprint = adjustment.confirmed_blueprint.blueprint
    assert blueprint.target_total == 6
    assert blueprint.max_candidates == 450
    assert adjustment.pool_multiplier == 5
    assert adjustment.theme_target_counts == {"history": 3, "future": 3}
    assert sum(theme.target_count for theme in blueprint.themes) == 6
    assert [theme.search_queries for theme in blueprint.themes] == [
        theme.search_queries for theme in original.blueprint.themes
    ]


def test_user_adjustment_rejects_noop_and_target_below_theme_floor() -> None:
    original = _confirmed_blueprint()
    service = LiteratureShortageRecoveryService()

    with pytest.raises(ValueError, match="at least one retrieval parameter"):
        service.adjust_retrieval(
            original,
            target_total=4,
            max_candidates=300,
            current_pool_multiplier=2,
            pool_multiplier=2,
        )

    with pytest.raises(ValueError, match="V0.1 floor"):
        service.adjust_retrieval(
            original,
            target_total=1,
            max_candidates=300,
            current_pool_multiplier=2,
            pool_multiplier=3,
        )
