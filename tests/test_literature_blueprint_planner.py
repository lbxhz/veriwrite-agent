import json

import pytest

from veriwrite_agent.llm.fake_client import FakeLLMClient
from veriwrite_agent.models.requirement_workflow import ConfirmedRequirementSpec
from veriwrite_agent.models.requirements import (
    ReferenceRequirement,
    RequirementSpec,
    TopicBoundary,
)
from veriwrite_agent.services.literature_blueprint_planner import (
    BlueprintPlanningError,
    LiteratureBlueprintPlanner,
)


def confirmed_requirement(
    *,
    topic: str | None = "大气遥感近五年的研究进展",
) -> ConfirmedRequirementSpec:
    return ConfirmedRequirementSpec(
        confirmed_by="tester",
        requirement=RequirementSpec(
            document_type="research_direction_literature_review",
            topic=topic,
            topic_source=(
                "explicit" if topic is not None else "user_confirmation_required"
            ),
            references=ReferenceRequirement(
                target_total=6,
                recent_year_window=5,
                recent_year_rule_strength="hard",
            ),
        ),
    )


def valid_blueprint_response(*, discipline: str = "大气科学") -> str:
    return json.dumps(
        {
            "topic": "LLM不得覆盖用户主题",
            "discipline": discipline,
            "writing_through_line": "从观测对象、反演方法到应用验证",
            "topic_boundary": {
                "central_question": "近五年大气遥感的观测对象与反演方法如何演进？",
                "included_objects": ["气溶胶", "温室气体", "大气遥感观测系统"],
                "excluded_objects": ["土壤水分", "考古", "健身物联网", "海底油气"],
                "contextual_only_topics": ["云计算", "边缘计算"],
                "origin": "agent_proposed",
            },
            "target_total": 6,
            "themes": [
                {
                    "theme_id": "aerosol",
                    "section_title": "气溶胶遥感",
                    "section_purpose": "梳理气溶胶观测与反演进展",
                    "research_questions": ["近年主要传感器和反演方法是什么？"],
                    "primary_keywords": ["aerosol remote sensing"],
                    "related_keywords": ["AOD retrieval"],
                    "search_queries": ["satellite aerosol retrieval"],
                    "target_count": 3,
                    "priority": 1,
                },
                {
                    "theme_id": "greenhouse_gas",
                    "section_title": "温室气体遥感",
                    "section_purpose": "比较温室气体监测技术",
                    "research_questions": ["甲烷和二氧化碳如何被卫星反演？"],
                    "primary_keywords": ["greenhouse gas remote sensing"],
                    "related_keywords": ["methane satellite"],
                    "search_queries": ["satellite methane carbon dioxide retrieval"],
                    "target_count": 3,
                    "priority": 1,
                },
            ],
        },
        ensure_ascii=False,
    )


def test_llm_designs_themes_but_code_enforces_confirmed_bounds() -> None:
    client = FakeLLMClient(valid_blueprint_response())
    planner = LiteratureBlueprintPlanner(
        client,
        ["大气科学", "测绘科学与技术"],
        current_year=2026,
    )

    blueprint = planner.plan(confirmed_requirement())

    assert blueprint.outline_status == "provisional_search_blueprint"
    assert blueprint.topic == "大气遥感近五年的研究进展"
    assert blueprint.target_total == 6
    assert sum(theme.target_count for theme in blueprint.themes) == 6
    assert blueprint.year_from == 2022
    assert blueprint.year_to == 2026
    assert blueprint.topic_boundary.is_actionable is True
    assert blueprint.accepted_tiers == ["T1", "T2", "T3", "T4", "T5", "T6"]
    assert client.calls[0]["response_format"] == {"type": "json_object"}


def test_confirmed_topic_boundary_overrides_llm_proposal() -> None:
    confirmed = confirmed_requirement()
    requirement = confirmed.requirement.model_copy(
        update={
            "topic_boundary": TopicBoundary(
                central_question="大气成分遥感如何提高反演可靠性？",
                included_objects=["大气成分"],
                excluded_objects=["土壤水分"],
                contextual_only_topics=["边缘计算"],
                origin="explicit",
            )
        }
    )

    blueprint = LiteratureBlueprintPlanner(
        FakeLLMClient(valid_blueprint_response()),
        ["大气科学"],
        current_year=2026,
    ).plan(confirmed.model_copy(update={"requirement": requirement}))

    assert blueprint.topic_boundary.central_question == "大气成分遥感如何提高反演可靠性？"
    assert blueprint.topic_boundary.included_objects == ["大气成分"]


def test_unsupported_discipline_is_rejected() -> None:
    planner = LiteratureBlueprintPlanner(
        FakeLLMClient(valid_blueprint_response(discipline="不存在的学科")),
        ["大气科学"],
    )

    with pytest.raises(BlueprintPlanningError, match="unsupported discipline"):
        planner.plan(confirmed_requirement())


def test_missing_confirmed_topic_blocks_blueprint_planning() -> None:
    planner = LiteratureBlueprintPlanner(
        FakeLLMClient(valid_blueprint_response()),
        ["大气科学"],
    )

    with pytest.raises(BlueprintPlanningError, match="research topic"):
        planner.plan(confirmed_requirement(topic=None))
