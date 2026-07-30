import pytest

from veriwrite_agent.models.literature_selection import (
    LiteratureSearchBlueprint,
    LiteratureThemePlan,
)
from veriwrite_agent.services.literature_blueprint_confirmation import (
    LiteratureBlueprintConfirmationService,
)
from veriwrite_agent.services.literature_blueprint_search import (
    LiteratureBlueprintSearchExpander,
    UnconfirmedLiteratureBlueprintError,
)


def blueprint() -> LiteratureSearchBlueprint:
    return LiteratureSearchBlueprint(
        topic="Atmospheric remote sensing",
        discipline="大气科学",
        writing_through_line="Observations, methods, and applications",
        target_total=6,
        max_candidates=60,
        year_from=2022,
        year_to=2026,
        themes=[
            LiteratureThemePlan(
                theme_id="aerosol",
                section_title="Aerosols",
                section_purpose="Review aerosol retrieval",
                research_questions=["How are aerosols retrieved?"],
                primary_keywords=["aerosol remote sensing"],
                search_queries=["satellite aerosol retrieval"],
                target_count=3,
            ),
            LiteratureThemePlan(
                theme_id="methane",
                section_title="Methane",
                section_purpose="Review methane retrieval",
                research_questions=["How is methane retrieved?"],
                primary_keywords=["methane remote sensing"],
                search_queries=["satellite methane retrieval"],
                target_count=3,
            ),
        ],
    )


def test_expands_every_confirmed_theme_into_an_independent_bounded_search() -> None:
    confirmed = LiteratureBlueprintConfirmationService().confirm(
        blueprint(),
        confirmed_by="student",
    )

    plans = LiteratureBlueprintSearchExpander(pool_multiplier=3).expand(confirmed)

    assert [item.theme_id for item in plans] == ["aerosol", "methane"]
    assert [item.plan.target_eligible_count for item in plans] == [9, 9]
    assert [item.plan.max_candidates for item in plans] == [30, 30]
    assert plans[0].plan.topic == "Atmospheric remote sensing：Aerosols"
    assert plans[0].plan.year_from == 2022
    assert plans[0].plan.year_to == 2026
    assert plans[0].plan.journal_ranking_policy == "preferred"


def test_unconfirmed_draft_cannot_start_external_retrieval() -> None:
    with pytest.raises(
        UnconfirmedLiteratureBlueprintError,
        match="user-confirmed",
    ):
        LiteratureBlueprintSearchExpander().expand(blueprint())  # type: ignore[arg-type]
