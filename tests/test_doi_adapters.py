from email.message import Message
from http.client import RemoteDisconnected
from urllib.error import HTTPError, URLError

from veriwrite_agent.literature.doi import (
    DoiOrgResolver,
    DoiRisMetadataProvider,
)
from veriwrite_agent.literature.ris import RisParseError, parse_ris

VALID_RIS = """TY  - JOUR
TI  - A GeoAI Study
AU  - Zhang, San
PY  - 2025/03/01
JO  - Annals of GIS
DO  - 10.1000/GEOAI.1
PB  - Example Publisher
UR  - https://doi.org/10.1000/geoai.1
ER  -
"""


class FakeResponse:
    def __init__(
        self,
        *,
        url: str,
        body: bytes = b"",
        status: int = 200,
        content_type: str = "text/plain; charset=utf-8",
    ) -> None:
        self._url = url
        self._body = body
        self.status = status
        self.headers = Message()
        self.headers["Content-Type"] = content_type

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def geturl(self) -> str:
        return self._url

    def getcode(self) -> int:
        return self.status

    def read(self, _amount: int = -1) -> bytes:
        return self._body


def test_doi_resolver_returns_final_landing_url() -> None:
    calls: list[object] = []

    def opener(request: object, *, timeout: float) -> FakeResponse:
        calls.append((request, timeout))
        return FakeResponse(url="https://publisher.example/article/geoai")

    result = DoiOrgResolver(
        opener=opener,
        minimum_request_interval_seconds=0,
    ).resolve("https://doi.org/10.1000/GEOAI.1")

    assert result.status == "resolved"
    assert result.doi == "10.1000/geoai.1"
    assert result.final_url == "https://publisher.example/article/geoai"
    request, timeout = calls[0]
    assert request.full_url == "https://doi.org/10.1000/geoai.1"
    assert request.headers["Accept"] == "text/html"
    assert timeout == 20


def test_doi_resolver_distinguishes_missing_doi_from_network_failure() -> None:
    def missing(request: object, *, timeout: float) -> FakeResponse:
        raise HTTPError(request.full_url, 404, "not found", {}, None)

    missing_result = DoiOrgResolver(
        opener=missing,
        minimum_request_interval_seconds=0,
    ).resolve("10.1000/missing")

    attempts = 0
    sleeps: list[float] = []

    def unavailable(_request: object, *, timeout: float) -> FakeResponse:
        nonlocal attempts
        attempts += 1
        raise URLError("temporary")

    unavailable_result = DoiOrgResolver(
        opener=unavailable,
        sleeper=sleeps.append,
        minimum_request_interval_seconds=0,
    ).resolve("10.1000/unavailable")

    assert missing_result.status == "unresolvable"
    assert missing_result.attempts == 1
    assert unavailable_result.status == "unavailable"
    assert unavailable_result.attempts == 3
    assert attempts == 3
    assert sleeps == [0.5, 1.0]


def test_doi_adapters_retry_remote_disconnects_without_aborting_the_batch() -> None:
    resolver_attempts = 0
    metadata_attempts = 0

    def resolver_opener(_request: object, *, timeout: float) -> FakeResponse:
        nonlocal resolver_attempts
        resolver_attempts += 1
        if resolver_attempts == 1:
            raise RemoteDisconnected("remote closed the connection")
        return FakeResponse(url="https://publisher.example/article/geoai")

    def metadata_opener(_request: object, *, timeout: float) -> FakeResponse:
        nonlocal metadata_attempts
        metadata_attempts += 1
        if metadata_attempts == 1:
            raise RemoteDisconnected("remote closed the connection")
        return FakeResponse(
            url="https://api.crossref.org/works/10.1000/geoai.1/transform",
            body=VALID_RIS.encode(),
            content_type="application/x-research-info-systems; charset=utf-8",
        )

    resolution = DoiOrgResolver(
        opener=resolver_opener,
        sleeper=lambda _seconds: None,
        minimum_request_interval_seconds=0,
    ).resolve("10.1000/geoai.1")
    metadata = DoiRisMetadataProvider(
        opener=metadata_opener,
        sleeper=lambda _seconds: None,
        minimum_request_interval_seconds=0,
    ).fetch("10.1000/geoai.1")

    assert resolution.status == "resolved"
    assert resolution.attempts == 2
    assert metadata.status == "available"
    assert metadata.attempts == 2


def test_ris_provider_uses_content_negotiation_and_parses_identity() -> None:
    calls: list[object] = []

    def opener(request: object, *, timeout: float) -> FakeResponse:
        calls.append((request, timeout))
        return FakeResponse(
            url="https://api.crossref.org/works/10.1000/geoai.1/transform",
            body=VALID_RIS.encode(),
            content_type="application/x-research-info-systems; charset=utf-8",
        )

    result = DoiRisMetadataProvider(
        opener=opener,
        minimum_request_interval_seconds=0,
    ).fetch("10.1000/geoai.1")

    assert result.status == "available"
    assert result.metadata is not None
    assert result.metadata.doi == "10.1000/geoai.1"
    assert result.metadata.title == "A GeoAI Study"
    assert result.metadata.authors == ["Zhang, San"]
    assert result.metadata.year == 2025
    assert result.metadata.journal_title == "Annals of GIS"
    request, timeout = calls[0]
    assert request.headers["Accept"] == "application/x-research-info-systems"
    assert timeout == 20


def test_ris_provider_rejects_non_ris_authority_response() -> None:
    def opener(_request: object, *, timeout: float) -> FakeResponse:
        return FakeResponse(
            url="https://publisher.example/article",
            body=b"<html>landing page</html>",
            content_type="text/html",
        )

    result = DoiRisMetadataProvider(
        opener=opener,
        minimum_request_interval_seconds=0,
    ).fetch("10.1000/geoai.1")

    assert result.status == "invalid"
    assert result.metadata is None
    assert result.raw_ris is None


def test_ris_parser_requires_exactly_one_record() -> None:
    try:
        parse_ris("not RIS")
    except RisParseError as exc:
        assert "exactly one" in str(exc)
    else:
        raise AssertionError("invalid RIS should fail")
