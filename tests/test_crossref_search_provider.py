import json
from urllib.error import URLError
from urllib.parse import parse_qs, urlparse

import pytest
from pydantic import ValidationError

from veriwrite_agent.literature.crossref import CrossrefSearchProvider
from veriwrite_agent.models.literature_discovery import LiteratureSearchPlan


class FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._body = json.dumps(payload).encode()

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def search_plan() -> LiteratureSearchPlan:
    return LiteratureSearchPlan(
        topic="GeoAI",
        discipline="测绘科学与技术",
        primary_keywords=["GeoAI"],
        search_queries=["GeoAI GIS"],
        year_from=2020,
        year_to=2026,
    )


def test_rejects_boolean_syntax_unsupported_by_bibliographic_query() -> None:
    with pytest.raises(ValidationError, match="without uppercase Boolean operators"):
        LiteratureSearchPlan(
            topic="GeoAI",
            discipline="测绘科学与技术",
            primary_keywords=["GeoAI"],
            search_queries=["GeoAI AND GIS"],
        )


def payload() -> dict[str, object]:
    return {
        "message": {
            "items": [
                {
                    "DOI": "10.1000/GeoAI.1",
                    "title": ["A GeoAI Study"],
                    "author": [{"given": "San", "family": "Zhang"}],
                    "published": {"date-parts": [[2025, 3, 1]]},
                    "container-title": ["Remote Sensing of Environment"],
                    "publisher": "Example Publisher",
                    "type": "journal-article",
                    "URL": "https://doi.org/10.1000/GeoAI.1",
                    "abstract": "<jats:p>GeoAI abstract.</jats:p>",
                }
            ],
            "next-cursor": "unused",
        }
    }


def test_maps_crossref_response_and_sends_responsible_request() -> None:
    calls: list[object] = []

    def opener(request: object, *, timeout: float) -> FakeResponse:
        calls.append((request, timeout))
        return FakeResponse(payload())

    provider = CrossrefSearchProvider(
        mailto="student@example.com",
        opener=opener,
        rows_per_request=10,
    )

    candidates = list(provider.search(search_plan()))

    assert len(candidates) == 1
    assert candidates[0].doi == "10.1000/geoai.1"
    assert candidates[0].authors == ["San Zhang"]
    assert candidates[0].year == 2025
    assert candidates[0].abstract == "GeoAI abstract."
    request, timeout = calls[0]
    assert "query.bibliographic=GeoAI+GIS" in request.full_url
    assert "type%3Ajournal-article" in request.full_url
    assert "mailto=student%40example.com" in request.full_url
    assert "veriwrite-agent" in request.headers["User-agent"]
    assert timeout == 20


def test_retries_transient_network_errors_then_succeeds() -> None:
    attempts = 0
    sleeps: list[float] = []

    def opener(_request: object, *, timeout: float) -> FakeResponse:
        nonlocal attempts
        attempts += 1
        assert timeout == 20
        if attempts < 3:
            raise URLError("temporary")
        return FakeResponse(payload())

    provider = CrossrefSearchProvider(
        opener=opener,
        sleeper=sleeps.append,
        max_retries=2,
        rows_per_request=10,
        minimum_request_interval_seconds=0,
    )

    candidates = list(provider.search(search_plan()))

    assert len(candidates) == 1
    assert attempts == 3
    assert sleeps == [0.5, 1.0]


def test_throttles_public_list_requests_to_one_per_second() -> None:
    current_time = 100.0
    sleeps: list[float] = []
    calls = 0

    def clock() -> float:
        return current_time

    def sleeper(seconds: float) -> None:
        nonlocal current_time
        sleeps.append(seconds)
        current_time += seconds

    def opener(_request: object, *, timeout: float) -> FakeResponse:
        nonlocal calls
        calls += 1
        assert timeout == 20
        if calls == 1:
            first_page = payload()
            first_page["message"]["next-cursor"] = "page-2"  # type: ignore[index]
            return FakeResponse(first_page)
        return FakeResponse({"message": {"items": [], "next-cursor": "unused"}})

    provider = CrossrefSearchProvider(
        opener=opener,
        sleeper=sleeper,
        clock=clock,
        rows_per_request=1,
    )

    candidates = list(provider.search(search_plan()))

    assert len(candidates) == 1
    assert calls == 2
    assert sleeps == [1.0]


def test_interleaves_candidates_from_all_queries_before_consumer_stops() -> None:
    def item(doi: str, title: str) -> dict[str, object]:
        return {
            "DOI": doi,
            "title": [title],
            "published": {"date-parts": [[2025]]},
            "container-title": ["Remote Sensing of Environment"],
            "type": "journal-article",
        }

    pages = {
        "aerosol remote sensing": [
            item("10.1000/aerosol.1", "Aerosol One"),
            item("10.1000/aerosol.2", "Aerosol Two"),
        ],
        "methane remote sensing": [
            item("10.1000/methane.1", "Methane One"),
            item("10.1000/methane.2", "Methane Two"),
        ],
    }

    def opener(request: object, *, timeout: float) -> FakeResponse:
        assert timeout == 20
        query = parse_qs(urlparse(request.full_url).query)["query.bibliographic"][0]
        return FakeResponse(
            {
                "message": {
                    "items": pages[query],
                    "next-cursor": "unused",
                }
            }
        )

    plan = LiteratureSearchPlan(
        topic="Atmospheric remote sensing",
        discipline="大气科学",
        primary_keywords=["atmospheric remote sensing"],
        search_queries=[
            "aerosol remote sensing",
            "methane remote sensing",
        ],
        target_eligible_count=2,
        max_candidates=4,
    )
    provider = CrossrefSearchProvider(
        opener=opener,
        rows_per_request=2,
        minimum_request_interval_seconds=0,
    )

    dois = [candidate.doi for candidate in provider.search(plan)]

    assert dois == [
        "10.1000/aerosol.1",
        "10.1000/methane.1",
        "10.1000/aerosol.2",
        "10.1000/methane.2",
    ]
