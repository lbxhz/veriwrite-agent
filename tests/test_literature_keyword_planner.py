import json

import pytest

from veriwrite_agent.llm.fake_client import FakeLLMClient
from veriwrite_agent.models.requirement_workflow import ConfirmedRequirementSpec
from veriwrite_agent.models.requirements import (
    ReferenceRequirement,
    RequirementSpec,
)
from veriwrite_agent.services.literature_keyword_planner import (
    KeywordPlanningError,
    LiteratureKeywordPlanner,
)


def confirmed_requirement() -> ConfirmedRequirementSpec:
    return ConfirmedRequirementSpec(
        confirmed_by="tester",
        requirement=RequirementSpec(
            document_type="research_direction_literature_review",
            topic="地理智能 GeoAI 在 GIS 中的发展与应用",
            topic_source="explicit",
            references=ReferenceRequirement(
                recent_year_window=5,
                recent_year_rule_strength="hard",
            ),
        ),
    )


def test_llm_generates_plan_but_code_enforces_time_window() -> None:
    response = json.dumps(
        {
            "topic": "translated topic that must not replace the confirmed topic",
            "discipline": "测绘科学与技术",
            "primary_keywords": ["GeoAI", "GeoAI"],
            "related_keywords": ["GIS", "spatial artificial intelligence"],
            "search_queries": ["GeoAI GIS", "geospatial artificial intelligence"],
            "accepted_tiers": ["T1", "T2"],
            "target_eligible_count": 50,
            "max_candidates": 300,
        },
        ensure_ascii=False,
    )
    client = FakeLLMClient(response)
    planner = LiteratureKeywordPlanner(
        client,
        ["测绘科学与技术", "地理科学"],
        current_year=2026,
    )

    plan = planner.plan(confirmed_requirement())

    assert plan.topic == "地理智能 GeoAI 在 GIS 中的发展与应用"
    assert plan.primary_keywords == ["GeoAI"]
    assert plan.discipline == "测绘科学与技术"
    assert plan.year_from == 2022
    assert plan.year_to == 2026
    assert client.calls[0]["response_format"] == {"type": "json_object"}


def test_unsupported_discipline_is_rejected() -> None:
    response = json.dumps(
        {
            "topic": "GeoAI",
            "discipline": "不存在的学科",
            "primary_keywords": ["GeoAI"],
            "search_queries": ["GeoAI"],
            "target_eligible_count": 50,
            "max_candidates": 300,
        },
        ensure_ascii=False,
    )
    planner = LiteratureKeywordPlanner(
        FakeLLMClient(response),
        ["测绘科学与技术"],
    )

    with pytest.raises(KeywordPlanningError, match="unsupported discipline"):
        planner.plan(confirmed_requirement())


def test_missing_confirmed_topic_blocks_search_planning() -> None:
    confirmed = ConfirmedRequirementSpec(
        confirmed_by="tester",
        requirement=RequirementSpec(
            document_type="research_direction_literature_review"
        ),
    )
    planner = LiteratureKeywordPlanner(
        FakeLLMClient("{}"),
        ["测绘科学与技术"],
    )

    with pytest.raises(KeywordPlanningError, match="research topic"):
        planner.plan(confirmed)
