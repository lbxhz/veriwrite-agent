"""doi.org adapters for landing-page resolution and RIS content negotiation."""

from __future__ import annotations

import time
from collections.abc import Callable
from http.client import HTTPException
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from veriwrite_agent.literature.ris import RisParseError, parse_ris
from veriwrite_agent.models.literature_discovery import canonicalize_doi
from veriwrite_agent.models.literature_verification import (
    AuthoritativeMetadataEvidence,
    DoiResolutionEvidence,
)

HttpOpener = Callable[..., Any]
RIS_MEDIA_TYPE = "application/x-research-info-systems"
MAX_RIS_BYTES = 1_000_000


class _RequestThrottle:
    def __init__(
        self,
        interval_seconds: float,
        sleeper: Callable[[float], None],
        clock: Callable[[], float],
    ) -> None:
        if interval_seconds < 0:
            raise ValueError("minimum_request_interval_seconds cannot be negative")
        self._interval_seconds = interval_seconds
        self._sleeper = sleeper
        self._clock = clock
        self._next_request_at = 0.0

    def wait(self) -> None:
        now = self._clock()
        started_at = now
        if now < self._next_request_at:
            self._sleeper(self._next_request_at - now)
            started_at = self._next_request_at
        self._next_request_at = started_at + self._interval_seconds


class DoiOrgResolver:
    """Resolve a DOI to the final landing URL using the DOI proxy."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 20,
        max_attempts: int = 3,
        opener: HttpOpener = urlopen,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        minimum_request_interval_seconds: float = 1.0,
    ) -> None:
        _validate_http_settings(timeout_seconds, max_attempts)
        self._timeout_seconds = timeout_seconds
        self._max_attempts = max_attempts
        self._opener = opener
        self._sleeper = sleeper
        self._throttle = _RequestThrottle(
            minimum_request_interval_seconds,
            sleeper,
            clock,
        )

    def resolve(self, doi: str) -> DoiResolutionEvidence:
        canonical = canonicalize_doi(doi)
        resolver_url = _doi_url(canonical)
        request = Request(
            resolver_url,
            headers={
                "Accept": "text/html",
                "User-Agent": "veriwrite-agent/0.2.1",
            },
        )
        last_error: Exception | None = None

        for attempt in range(1, self._max_attempts + 1):
            self._throttle.wait()
            try:
                with self._opener(request, timeout=self._timeout_seconds) as response:
                    final_url = response.geturl()
                    status = _response_status(response)
                return DoiResolutionEvidence(
                    doi=canonical,
                    status="resolved",
                    resolver_url=resolver_url,
                    final_url=final_url,
                    http_status=status,
                    attempts=attempt,
                    reason="doi.org resolved the DOI to a reachable landing page.",
                )
            except HTTPError as exc:
                last_error = exc
                final_url = exc.geturl()
                redirected = final_url.rstrip("/") != resolver_url.rstrip("/")
                if redirected and exc.code < 500:
                    return DoiResolutionEvidence(
                        doi=canonical,
                        status="landing_unavailable",
                        resolver_url=resolver_url,
                        final_url=final_url,
                        http_status=exc.code,
                        attempts=attempt,
                        reason=(
                            "doi.org redirected successfully, but the landing page "
                            f"returned HTTP {exc.code}."
                        ),
                    )
                if not redirected and exc.code in {400, 404}:
                    return DoiResolutionEvidence(
                        doi=canonical,
                        status="unresolvable",
                        resolver_url=resolver_url,
                        http_status=exc.code,
                        attempts=attempt,
                        reason=f"doi.org could not resolve the DOI (HTTP {exc.code}).",
                    )
                if exc.code != 429 and exc.code < 500:
                    break
            except (URLError, TimeoutError, OSError, HTTPException) as exc:
                last_error = exc
            if attempt < self._max_attempts:
                self._sleeper(0.5 * (2 ** (attempt - 1)))

        return DoiResolutionEvidence(
            doi=canonical,
            status="unavailable",
            resolver_url=resolver_url,
            attempts=self._max_attempts,
            reason=(
                "DOI resolution remained unavailable after "
                f"{self._max_attempts} attempts: {type(last_error).__name__}."
            ),
        )


class DoiRisMetadataProvider:
    """Retrieve authority-generated RIS through DOI content negotiation."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 20,
        max_attempts: int = 3,
        opener: HttpOpener = urlopen,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        minimum_request_interval_seconds: float = 1.0,
    ) -> None:
        _validate_http_settings(timeout_seconds, max_attempts)
        self._timeout_seconds = timeout_seconds
        self._max_attempts = max_attempts
        self._opener = opener
        self._sleeper = sleeper
        self._throttle = _RequestThrottle(
            minimum_request_interval_seconds,
            sleeper,
            clock,
        )

    def fetch(self, doi: str) -> AuthoritativeMetadataEvidence:
        canonical = canonicalize_doi(doi)
        source_url = _doi_url(canonical)
        request = Request(
            source_url,
            headers={
                "Accept": RIS_MEDIA_TYPE,
                "User-Agent": "veriwrite-agent/0.2.1",
            },
        )
        last_error: Exception | None = None

        for attempt in range(1, self._max_attempts + 1):
            self._throttle.wait()
            try:
                with self._opener(request, timeout=self._timeout_seconds) as response:
                    final_url = response.geturl()
                    status = _response_status(response)
                    body = response.read(MAX_RIS_BYTES + 1)
                if status == 204:
                    return _missing_metadata(
                        canonical,
                        final_url,
                        "not_found",
                        attempt,
                        "The DOI registration agency returned no metadata.",
                    )
                if len(body) > MAX_RIS_BYTES:
                    return _missing_metadata(
                        canonical,
                        final_url,
                        "invalid",
                        attempt,
                        "Authority RIS exceeded the one-megabyte safety limit.",
                    )
                text = _decode_response(body, response)
                try:
                    metadata = parse_ris(text)
                except RisParseError as exc:
                    return _missing_metadata(
                        canonical,
                        final_url,
                        "invalid",
                        attempt,
                        f"Authority response was not usable RIS: {exc}.",
                    )
                return AuthoritativeMetadataEvidence(
                    doi=canonical,
                    status="available",
                    source_url=final_url,
                    metadata=metadata,
                    raw_ris=text,
                    attempts=attempt,
                    reason="RIS was returned by the DOI registration authority.",
                )
            except HTTPError as exc:
                last_error = exc
                status = (
                    "not_found"
                    if exc.code == 404
                    else "unsupported"
                    if exc.code == 406
                    else None
                )
                if status is not None:
                    return _missing_metadata(
                        canonical,
                        exc.geturl(),
                        status,
                        attempt,
                        f"DOI metadata request returned HTTP {exc.code}.",
                    )
                if exc.code != 429 and exc.code < 500:
                    break
            except (URLError, TimeoutError, OSError, HTTPException) as exc:
                last_error = exc
            if attempt < self._max_attempts:
                self._sleeper(0.5 * (2 ** (attempt - 1)))

        return _missing_metadata(
            canonical,
            source_url,
            "unavailable",
            self._max_attempts,
            (
                "Authority RIS remained unavailable after "
                f"{self._max_attempts} attempts: {type(last_error).__name__}."
            ),
        )


def _doi_url(doi: str) -> str:
    return f"https://doi.org/{quote(doi, safe='/')}"


def _validate_http_settings(timeout_seconds: float, max_attempts: int) -> None:
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if not 1 <= max_attempts <= 3:
        raise ValueError("max_attempts must be between 1 and 3")


def _response_status(response: Any) -> int:
    status = getattr(response, "status", None)
    if isinstance(status, int):
        return status
    return int(response.getcode())


def _decode_response(body: bytes, response: Any) -> str:
    headers = getattr(response, "headers", None)
    charset = None
    if headers is not None and hasattr(headers, "get_content_charset"):
        charset = headers.get_content_charset()
    for encoding in (charset, "utf-8-sig", "utf-8", "latin-1"):
        if not encoding:
            continue
        try:
            return body.decode(encoding)
        except (LookupError, UnicodeDecodeError):
            continue
    raise UnicodeDecodeError("utf-8", body, 0, len(body), "unsupported encoding")


def _missing_metadata(
    doi: str,
    source_url: str,
    status: str,
    attempts: int,
    reason: str,
) -> AuthoritativeMetadataEvidence:
    return AuthoritativeMetadataEvidence(
        doi=doi,
        status=status,
        source_url=source_url,
        attempts=attempts,
        reason=reason,
    )
