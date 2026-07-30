from veriwrite_agent.models.literature_selection import (
    LiteratureSearchBlueprint,
    LiteratureThemePlan,
)
from veriwrite_agent.services.literature_blueprint_confirmation import (
    LiteratureBlueprintConfirmationService,
)


def blueprint() -> LiteratureSearchBlueprint:
    return LiteratureSearchBlueprint(
        topic="Atmospheric remote sensing",
        discipline="大气科学",
        writing_through_line="Objects and methods",
        target_total=2,
        themes=[
            LiteratureThemePlan(
                theme_id="aerosol",
                section_title="Aerosol",
                section_purpose="Review aerosol retrieval",
                research_questions=["How are aerosols retrieved?"],
                primary_keywords=["aerosol"],
                search_queries=["satellite aerosol retrieval"],
                target_count=1,
            ),
            LiteratureThemePlan(
                theme_id="methane",
                section_title="Methane",
                section_purpose="Review methane retrieval",
                research_questions=["How is methane retrieved?"],
                primary_keywords=["methane"],
                search_queries=["satellite methane retrieval"],
                target_count=1,
            ),
        ],
    )


def test_confirmation_preserves_the_exact_user_approved_blueprint() -> None:
    draft = blueprint()

    confirmed = LiteratureBlueprintConfirmationService().confirm(
        draft,
        confirmed_by="student",
        note="主题和每部分数量已确认",
    )

    assert confirmed.status == "confirmed"
    assert confirmed.confirmed_by == "student"
    assert confirmed.confirmation_note == "主题和每部分数量已确认"
    assert confirmed.blueprint == draft
