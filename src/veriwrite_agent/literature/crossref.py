"""Crossref REST adapter for DOI-backed literature discovery."""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Callable, Iterable
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from pydantic import ValidationError

from veriwrite_agent.literature.base import LiteratureSearchError
from veriwrite_agent.models.literature_discovery import (
    LiteratureCandidate,
    LiteratureSearchPlan,
)

HttpOpener = Callable[..., Any]


class CrossrefSearchProvider:
    """Search Crossref responsibly and expose only project-level candidates."""

    def __init__(
        self,
        *,
        mailto: str | None = None,
        timeout_seconds: float = 20,
        max_retries: int = 2,
        rows_per_request: int = 100,
        opener: HttpOpener = urlopen,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        minimum_request_interval_seconds: float | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        if not 1 <= rows_per_request <= 1000:
            raise ValueError("rows_per_request must be between 1 and 1000")
        if (
            minimum_request_interval_seconds is not None
            and minimum_request_interval_seconds < 0
        ):
            raise ValueError("minimum_request_interval_seconds cannot be negative")
        self._mailto = mailto.strip() if mailto else None
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._rows_per_request = rows_per_request
        self._opener = opener
        self._sleeper = sleeper
        self._clock = clock
        self._minimum_request_interval_seconds = (
            minimum_request_interval_seconds
            if minimum_request_interval_seconds is not None
            else (1 / 3 if self._mailto else 1.0)
        )
        self._next_request_at = 0.0

    def search(self, plan: LiteratureSearchPlan) -> Iterable[LiteratureCandidate]:
        queries = plan.search_queries
        per_query_limit = math.ceil(plan.max_candidates / len(queries))
        total_yielded = 0

        for query in queries:
            cursor = "*"
            yielded_for_query = 0
            seen_cursors: set[str] = set()

            while (
                yielded_for_query < per_query_limit
                and total_yielded < plan.max_candidates
            ):
                rows = min(
                    self._rows_per_request,
                    per_query_limit - yielded_for_query,
                    plan.max_candidates - total_yielded,
                )
                payload = self._request_page(plan, query, cursor, rows)
                message = payload.get("message")
                if not isinstance(message, dict):
                    raise LiteratureSearchError(
                        "Crossref response does not contain a message object"
                    )
                items = message.get("items")
                if not isinstance(items, list):
                    raise LiteratureSearchError(
                        "Crossref response does not contain an items list"
                    )

                for item in items:
                    candidate = self._candidate_from_item(item)
                    if candidate is None:
                        continue
                    yield candidate
                    yielded_for_query += 1
                    total_yielded += 1
                    if (
                        yielded_for_query >= per_query_limit
                        or total_yielded >= plan.max_candidates
                    ):
                        break

                if len(items) < rows:
                    break
                next_cursor = message.get("next-cursor")
                if (
                    not isinstance(next_cursor, str)
                    or not next_cursor
                    or next_cursor in seen_cursors
                ):
                    break
                seen_cursors.add(next_cursor)
                cursor = next_cursor

    def _request_page(
        self,
        plan: LiteratureSearchPlan,
        query: str,
        cursor: str,
        rows: int,
    ) -> dict[str, Any]:
        filters = [f"type:{plan.work_type}"]
        if plan.year_from is not None:
            filters.append(f"from-pub-date:{plan.year_from}-01-01")
        if plan.year_to is not None:
            filters.append(f"until-pub-date:{plan.year_to}-12-31")
        params = {
            "query.bibliographic": query,
            "filter": ",".join(filters),
            "rows": rows,
            "cursor": cursor,
        }
        if self._mailto:
            params["mailto"] = self._mailto
        url = f"https://api.crossref.org/works?{urlencode(params)}"
        agent = "veriwrite-agent/0.2"
        if self._mailto:
            agent += f" (mailto:{self._mailto})"
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": agent,
            },
        )

        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            self._wait_for_request_slot()
            try:
                with self._opener(
                    request,
                    timeout=self._timeout_seconds,
                ) as response:
                    body = response.read()
                payload = json.loads(body.decode("utf-8"))
                if not isinstance(payload, dict):
                    raise LiteratureSearchError(
                        "Crossref response root is not an object"
                    )
                return payload
            except HTTPError as exc:
                last_error = exc
                if exc.code != 429 and exc.code < 500:
                    break
            except (URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
            if attempt < self._max_retries:
                self._sleeper(0.5 * (2**attempt))

        raise LiteratureSearchError(
            f"Crossref search failed after {self._max_retries + 1} attempts"
        ) from last_error

    def _wait_for_request_slot(self) -> None:
        now = self._clock()
        request_started_at = now
        if now < self._next_request_at:
            self._sleeper(self._next_request_at - now)
            request_started_at = self._next_request_at
        self._next_request_at = (
            request_started_at + self._minimum_request_interval_seconds
        )

    @staticmethod
    def _candidate_from_item(item: object) -> LiteratureCandidate | None:
        if not isinstance(item, dict):
            return None
        doi = item.get("DOI")
        title = CrossrefSearchProvider._first_text(item.get("title"))
        journal = CrossrefSearchProvider._first_text(item.get("container-title"))
        if not isinstance(doi, str) or title is None or journal is None:
            return None

        authors: list[str] = []
        raw_authors = item.get("author")
        if isinstance(raw_authors, list):
            for author in raw_authors:
                if not isinstance(author, dict):
                    continue
                name = " ".join(
                    str(author.get(part, "")).strip()
                    for part in ("given", "family")
                    if str(author.get(part, "")).strip()
                )
                if name:
                    authors.append(name)

        abstract = item.get("abstract")
        if isinstance(abstract, str):
            abstract = re.sub(r"<[^>]+>", " ", abstract)
            abstract = " ".join(abstract.split()) or None
        else:
            abstract = None

        try:
            return LiteratureCandidate(
                doi=doi,
                title=title,
                authors=authors,
                year=CrossrefSearchProvider._publication_year(item),
                journal_title=journal,
                publisher=(
                    item.get("publisher")
                    if isinstance(item.get("publisher"), str)
                    else None
                ),
                source_type=(
                    item.get("type")
                    if isinstance(item.get("type"), str)
                    else "journal-article"
                ),
                source_provider="crossref",
                source_url=(
                    item.get("URL") if isinstance(item.get("URL"), str) else None
                ),
                abstract=abstract,
            )
        except ValidationError:
            return None

    @staticmethod
    def _first_text(value: object) -> str | None:
        if not isinstance(value, list):
            return None
        for item in value:
            if isinstance(item, str) and item.strip():
                return " ".join(item.split())
        return None

    @staticmethod
    def _publication_year(item: dict[str, Any]) -> int | None:
        for field in (
            "published",
            "published-print",
            "published-online",
            "issued",
        ):
            value = item.get(field)
            if not isinstance(value, dict):
                continue
            date_parts = value.get("date-parts")
            if (
                isinstance(date_parts, list)
                and date_parts
                and isinstance(date_parts[0], list)
                and date_parts[0]
                and isinstance(date_parts[0][0], int)
            ):
                return date_parts[0][0]
        return None
