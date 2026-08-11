import json

from veriwrite_agent.llm.fake_client import FakeLLMClient
from veriwrite_agent.models.literature_selection import (
    LiteratureSearchBlueprint,
    LiteratureThemePlan,
)
from veriwrite_agent.models.requirements import TopicBoundary
from veriwrite_agent.services.literature_query_refinement import (
    LiteratureShortageQueryRefiner,
)


def blueprint() -> LiteratureSearchBlueprint:
    return LiteratureSearchBlueprint(
        topic="Atmospheric remote sensing",
        discipline="大气科学",
        writing_through_line="Observation, retrieval, and application",
        target_total=2,
        topic_boundary=TopicBoundary(
            central_question="How is the atmosphere observed remotely?",
            included_objects=["cloud", "aerosol"],
            excluded_objects=["soil moisture"],
            origin="explicit",
        ),
        themes=[
            LiteratureThemePlan(
                theme_id="observation",
                section_title="Observation systems",
                section_purpose="Compare direct atmospheric observing systems",
                research_questions=["Which observing systems provide atmospheric data?"],
                primary_keywords=["lidar", "radar"],
                search_queries=["atmospheric remote sensing lidar radar"],
                target_count=1,
            ),
            LiteratureThemePlan(
                theme_id="application",
                section_title="Applications",
                section_purpose="Review atmospheric applications",
                research_questions=["How are observations applied?"],
                primary_keywords=["air quality"],
                search_queries=["satellite air quality application"],
                target_count=1,
            ),
        ],
    )


def test_rewrites_only_shortage_theme_with_new_queries() -> None:
    client = FakeLLMClient(
        json.dumps(
            {
                "themes": [
                    {
                        "theme_id": "observation",
                        "search_queries": [
                            "atmospheric observing system lidar instrument review",
                            "satellite radar atmospheric measurement comparison",
                        ],
                    }
                ]
            }
        )
    )

    result = LiteratureShortageQueryRefiner(client).refine(
        blueprint(),
        {"observation": 1},
    )

    assert [theme.theme_id for theme in result.themes] == ["observation"]
    assert len(result.themes[0].search_queries) == 2
    assert len(client.calls) == 1


def test_falls_back_when_repair_still_reuses_existing_queries() -> None:
    client = FakeLLMClient(
        json.dumps(
            {
                "themes": [
                    {
                        "theme_id": "observation",
                        "search_queries": [
                            "atmospheric remote sensing lidar radar",
                            "satellite radar atmospheric measurement comparison",
                        ],
                    }
                ]
            }
        )
    )

    result = LiteratureShortageQueryRefiner(client).refine(
        blueprint(),
        {"observation": 1},
        previous_recovery_queries={
            "observation": ["satellite radar atmospheric measurement comparison"]
        },
    )

    assert len(client.calls) == 2
    assert result.schema_version == "0.2-query-refinement.fallback.1"
    assert len(result.themes[0].search_queries) >= 2
    assert "atmospheric remote sensing lidar radar" not in {
        query.casefold() for query in result.themes[0].search_queries
    }


def test_falls_back_when_llm_deduplication_leaves_only_one_query() -> None:
    client = FakeLLMClient(
        json.dumps(
            {
                "themes": [
                    {
                        "theme_id": "observation",
                        "search_queries": [
                            "atmospheric observation network review",
                            "atmospheric observation network review",
                        ],
                    }
                ]
            }
        )
    )

    result = LiteratureShortageQueryRefiner(client).refine(
        blueprint(),
        {"observation": 1},
    )

    assert len(client.calls) == 2
    assert result.schema_version == "0.2-query-refinement.fallback.1"
    assert len(result.themes[0].search_queries) >= 2


def test_deterministic_fallback_supports_all_twelve_recovery_rounds() -> None:
    source = blueprint()
    previous: dict[str, list[str]] = {"observation": []}

    for _ in range(12):
        result = LiteratureShortageQueryRefiner._fallback_batch(
            source,
            shortage_ids={"observation"},
            previous_recovery_queries=previous,
        )
        queries = result.themes[0].search_queries
        assert len(queries) == 4
        assert not set(queries) & set(previous["observation"])
        previous["observation"].extend(queries)

    assert len(previous["observation"]) == 48


def test_filters_an_existing_query_when_two_new_queries_remain() -> None:
    client = FakeLLMClient(
        json.dumps(
            {
                "themes": [
                    {
                        "theme_id": "observation",
                        "search_queries": [
                            "atmospheric remote sensing lidar radar",
                            "atmospheric observing system instrument review",
                            "satellite radar atmospheric measurement comparison",
                        ],
                    }
                ]
            }
        )
    )

    result = LiteratureShortageQueryRefiner(client).refine(
        blueprint(),
        {"observation": 1},
    )

    assert result.themes[0].search_queries == [
        "atmospheric observing system instrument review",
        "satellite radar atmospheric measurement comparison",
    ]
    assert len(client.calls) == 1
