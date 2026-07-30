"""Local adapter for the CUG Wuhan 2023 journal classification directory."""

from __future__ import annotations

import csv
import html
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

from veriwrite_agent.models.literature_discovery import (
    JournalRankingLookup,
    JournalRankingRecord,
)

EXPECTED_COLUMNS = {
    "category",
    "discipline",
    "journal_title",
    "tier",
    "source_workbook",
    "source_row",
}


class JournalCatalogError(ValueError):
    """Raised when the local catalog cannot satisfy its data contract."""


def normalize_journal_title(value: str) -> str:
    """Create a conservative title key without guessing journal aliases."""

    value = html.unescape(value)
    normalized = unicodedata.normalize("NFKC", value)
    normalized = normalized.replace("\u00a0", " ")
    normalized = re.sub(r"[‐‑‒–—―]", "-", normalized)
    normalized = re.sub(r"\s*&\s*", " AND ", normalized)
    normalized = normalized.upper()
    normalized = re.sub(r"[^0-9A-Z\u3400-\u9FFF]+", " ", normalized)
    return " ".join(normalized.split())


class CugJournalRankingProvider:
    """Resolve exact normalized titles against a versioned local CSV snapshot."""

    def __init__(self, catalog_path: str | Path) -> None:
        self._catalog_path = Path(catalog_path)
        self._index: dict[tuple[str, str], list[JournalRankingRecord]] = defaultdict(list)
        self._disciplines: set[str] = set()
        self._record_count = 0
        self._load()

    @classmethod
    def from_default_catalog(cls) -> CugJournalRankingProvider:
        repository_root = Path(__file__).resolve().parents[3]
        return cls(
            repository_root
            / "data"
            / "cug_wuhan_journal_classification_2023.csv"
        )

    @property
    def available_disciplines(self) -> tuple[str, ...]:
        return tuple(sorted(self._disciplines))

    @property
    def catalog_path(self) -> Path:
        return self._catalog_path

    @property
    def record_count(self) -> int:
        return self._record_count

    def lookup(self, journal_title: str, discipline: str) -> JournalRankingLookup:
        query_title = " ".join(journal_title.split())
        normalized = normalize_journal_title(query_title)
        clean_discipline = " ".join(discipline.split())
        records = list(self._index.get((clean_discipline, normalized), []))
        if not records:
            return JournalRankingLookup(
                status="not_found",
                query_title=query_title,
                normalized_title=normalized,
                discipline=clean_discipline,
                reason=(
                    "该期刊未出现在中国地质大学（武汉）2023版所选学科目录中。"
                ),
            )

        tiers = {record.tier for record in records}
        if len(tiers) > 1:
            return JournalRankingLookup(
                status="ambiguous",
                query_title=query_title,
                normalized_title=normalized,
                discipline=clean_discipline,
                records=records,
                reason=(
                    "地大2023版源目录在同一学科中为规范化后的同一期刊给出了多个等级。"
                ),
            )

        return JournalRankingLookup(
            status="matched",
            query_title=query_title,
            normalized_title=normalized,
            discipline=clean_discipline,
            records=records,
            reason=(
                "期刊名称与中国地质大学（武汉）2023版所选学科目录规范化匹配。"
            ),
        )

    def _load(self) -> None:
        if not self._catalog_path.is_file():
            raise JournalCatalogError(f"journal catalog does not exist: {self._catalog_path}")

        with self._catalog_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or [])
            if columns != EXPECTED_COLUMNS:
                raise JournalCatalogError(
                    "journal catalog columns do not match the expected contract"
                )
            for row_number, row in enumerate(reader, start=2):
                try:
                    journal_title = " ".join(row["journal_title"].split())
                    discipline = " ".join(row["discipline"].split())
                    record = JournalRankingRecord(
                        category=row["category"],
                        discipline=discipline,
                        journal_title=journal_title,
                        normalized_title=normalize_journal_title(journal_title),
                        tier=row["tier"],
                        source_workbook=row["source_workbook"],
                        source_row=int(row["source_row"]),
                    )
                except (TypeError, ValueError) as exc:
                    raise JournalCatalogError(
                        f"invalid journal catalog row {row_number}"
                    ) from exc
                self._disciplines.add(record.discipline)
                self._index[(record.discipline, record.normalized_title)].append(record)
                self._record_count += 1

        if not self._index:
            raise JournalCatalogError("journal catalog contains no records")
