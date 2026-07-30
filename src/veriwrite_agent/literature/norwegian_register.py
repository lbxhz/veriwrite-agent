"""Fixed-year adapter for the open Norwegian Register journal classification."""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from veriwrite_agent.literature.cug_catalog import normalize_journal_title
from veriwrite_agent.models.literature_discovery import (
    NorwegianJournalRankingLookup,
    NorwegianJournalRankingRecord,
    canonicalize_issn,
)

EXPECTED_COLUMNS = {
    "journal_id",
    "original_title",
    "international_title",
    "print_issn",
    "online_issn",
    "scientific_field",
    "level_2025",
}


class NorwegianRegisterCatalogError(ValueError):
    """Raised when the vendored Norwegian Register snapshot is malformed."""


class NorwegianRegisterRankingProvider:
    """Resolve journals by ISSN first and normalized title only as a fallback."""

    def __init__(self, catalog_path: str | Path) -> None:
        self._catalog_path = Path(catalog_path)
        self._issn_index: dict[str, list[NorwegianJournalRankingRecord]] = (
            defaultdict(list)
        )
        self._title_index: dict[str, list[NorwegianJournalRankingRecord]] = (
            defaultdict(list)
        )
        self._record_count = 0
        self._load()

    @classmethod
    def from_default_catalog(cls) -> NorwegianRegisterRankingProvider:
        repository_root = Path(__file__).resolve().parents[3]
        return cls(repository_root / "data" / "norwegian_register_journals_2025.csv")

    @property
    def catalog_path(self) -> Path:
        return self._catalog_path

    @property
    def record_count(self) -> int:
        return self._record_count

    def lookup(
        self,
        journal_title: str,
        issns: list[str] | tuple[str, ...] = (),
    ) -> NorwegianJournalRankingLookup:
        query_title = " ".join(journal_title.split())
        query_issns: list[str] = []
        for issn in issns:
            try:
                normalized = canonicalize_issn(issn)
            except ValueError:
                continue
            if normalized not in query_issns:
                query_issns.append(normalized)

        records = self._unique_records(
            record
            for issn in query_issns
            for record in self._issn_index.get(issn, ())
        )
        match_basis = "issn"
        if not records:
            normalized_title = normalize_journal_title(query_title)
            records = self._unique_records(self._title_index.get(normalized_title, ()))
            match_basis = "title"

        if not records:
            return NorwegianJournalRankingLookup(
                status="not_found",
                query_title=query_title,
                query_issns=query_issns,
                match_basis="none",
                reason=(
                    "The journal was not assigned level 0, 1, or 2 in the "
                    "Norwegian Register 2025 snapshot."
                ),
            )

        levels = {record.level for record in records}
        if len(levels) > 1:
            return NorwegianJournalRankingLookup(
                status="ambiguous",
                query_title=query_title,
                query_issns=query_issns,
                match_basis=match_basis,
                records=records,
                reason=(
                    "The Norwegian Register 2025 snapshot produced conflicting "
                    f"levels through {match_basis} matching."
                ),
            )

        return NorwegianJournalRankingLookup(
            status="matched",
            query_title=query_title,
            query_issns=query_issns,
            match_basis=match_basis,
            records=records,
            reason=(
                "The journal matched the Norwegian Register 2025 snapshot "
                f"through {match_basis}."
            ),
        )

    @staticmethod
    def _unique_records(
        records: Iterable[NorwegianJournalRankingRecord],
    ) -> list[NorwegianJournalRankingRecord]:
        unique: dict[str, NorwegianJournalRankingRecord] = {}
        for record in records:
            unique.setdefault(record.journal_id, record)
        return list(unique.values())

    def _load(self) -> None:
        if not self._catalog_path.is_file():
            raise NorwegianRegisterCatalogError(
                f"Norwegian Register catalog does not exist: {self._catalog_path}"
            )

        with self._catalog_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or [])
            if columns != EXPECTED_COLUMNS:
                raise NorwegianRegisterCatalogError(
                    "Norwegian Register columns do not match the expected contract"
                )
            for source_row, row in enumerate(reader, start=2):
                try:
                    original_title = " ".join(row["original_title"].split())
                    international_title = (
                        " ".join(row["international_title"].split()) or None
                    )
                    normalized_titles = []
                    for title in (original_title, international_title):
                        if not title:
                            continue
                        normalized = normalize_journal_title(title)
                        if normalized and normalized not in normalized_titles:
                            normalized_titles.append(normalized)
                    print_issn = self._optional_issn(row["print_issn"])
                    online_issn = self._optional_issn(row["online_issn"])
                    record = NorwegianJournalRankingRecord(
                        journal_id=row["journal_id"],
                        original_title=original_title,
                        international_title=international_title,
                        normalized_titles=normalized_titles,
                        print_issn=print_issn,
                        online_issn=online_issn,
                        scientific_field=" ".join(row["scientific_field"].split())
                        or None,
                        level=int(row["level_2025"]),
                        source_row=source_row,
                    )
                except (TypeError, ValueError) as exc:
                    raise NorwegianRegisterCatalogError(
                        f"invalid Norwegian Register row {source_row}"
                    ) from exc

                for normalized_title in record.normalized_titles:
                    self._title_index[normalized_title].append(record)
                for issn in (record.print_issn, record.online_issn):
                    if issn:
                        self._issn_index[issn].append(record)
                self._record_count += 1

        if not self._title_index or not self._issn_index:
            raise NorwegianRegisterCatalogError(
                "Norwegian Register catalog contains no usable records"
            )

    @staticmethod
    def _optional_issn(value: str) -> str | None:
        clean = value.strip()
        if not clean:
            return None
        try:
            return canonicalize_issn(clean)
        except ValueError:
            return None
