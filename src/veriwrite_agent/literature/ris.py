"""Small deterministic parser for the RIS fields needed by V0.2.1."""

from __future__ import annotations

import re
from collections import defaultdict

from pydantic import ValidationError

from veriwrite_agent.models.literature_verification import (
    RisBibliographicMetadata,
)

RIS_LINE = re.compile(r"^([A-Z0-9]{2})  - ?(.*)$")


class RisParseError(ValueError):
    """Raised when an authority response is not one usable RIS record."""


def parse_ris(text: str) -> RisBibliographicMetadata:
    """Parse one RIS record while preserving only identity-relevant fields."""

    fields: dict[str, list[str]] = defaultdict(list)
    current_tag: str | None = None
    record_count = 0

    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        match = RIS_LINE.match(raw_line)
        if match:
            tag, value = match.groups()
            current_tag = tag
            if tag == "TY":
                record_count += 1
            if tag == "ER":
                current_tag = None
                continue
            fields[tag].append(value.strip())
        elif current_tag and raw_line[:1].isspace() and raw_line.strip():
            fields[current_tag][-1] = f"{fields[current_tag][-1]} {raw_line.strip()}"

    if record_count != 1:
        raise RisParseError("authority response must contain exactly one RIS record")

    year = _parse_year(_first(fields, "PY", "Y1", "DA"))
    try:
        return RisBibliographicMetadata(
            doi=_first(fields, "DO"),
            title=_first(fields, "TI", "T1"),
            authors=[*fields.get("AU", []), *fields.get("A1", [])],
            year=year,
            journal_title=_first(fields, "JO", "JF", "T2"),
            publisher=_first(fields, "PB"),
            url=_first(fields, "UR"),
            ris_type=_first(fields, "TY"),
        )
    except (ValidationError, ValueError) as exc:
        raise RisParseError("RIS identity fields violate the data contract") from exc


def _first(fields: dict[str, list[str]], *tags: str) -> str | None:
    for tag in tags:
        for value in fields.get(tag, []):
            if value.strip():
                return value.strip()
    return None


def _parse_year(value: str | None) -> int | None:
    if value is None:
        return None
    match = re.search(r"\b(1[0-9]{3}|20[0-9]{2}|2100)\b", value)
    return int(match.group(1)) if match else None
