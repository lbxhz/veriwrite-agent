"""Data contracts for V0.2 literature discovery and local journal ranking."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import Field, field_validator, model_validator

from veriwrite_agent.models.requirements import StrictModel

CugTier = Literal["T1", "T2", "T3", "T4", "T5", "T6"]
NorwegianLevel = Literal[0, 1, 2]
RankingLookupStatus = Literal["matched", "not_found", "ambiguous"]
CandidateDecisionStatus = Literal["eligible", "excluded"]


def canonicalize_doi(value: str) -> str:
    """Return the canonical DOI name used as the project-wide identity key."""

    normalized = value.strip()
    normalized = re.sub(
        r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)",
        "",
        normalized,
        flags=re.IGNORECASE,
    ).lower()
    if not re.fullmatch(r"10\.\d{4,9}/\S+", normalized):
        raise ValueError("doi must be a canonical DOI name")
    return normalized


def canonicalize_issn(value: str) -> str:
    """Return a hyphenated ISSN key, including a checksum guard."""

    compact = re.sub(r"[^0-9X]", "", value.upper())
    if not re.fullmatch(r"\d{7}[\dX]", compact):
        raise ValueError("issn must contain eight valid ISSN characters")
    digits = [int(character) for character in compact[:7]]
    check_value = 10 if compact[-1] == "X" else int(compact[-1])
    if (sum((8 - index) * digit for index, digit in enumerate(digits)) + check_value) % 11:
        raise ValueError("issn checksum is invalid")
    return f"{compact[:4]}-{compact[4:]}"


class LiteratureSearchPlan(StrictModel):
    """LLM-produced, contract-validated instructions for deterministic search."""

    schema_version: str = "0.2.0"
    topic: str = Field(min_length=1)
    discipline: str = Field(min_length=1)
    primary_keywords: list[str] = Field(min_length=1)
    related_keywords: list[str] = Field(default_factory=list)
    search_queries: list[str] = Field(min_length=1, max_length=8)
    accepted_tiers: list[CugTier] = Field(
        default_factory=lambda: ["T1", "T2", "T3", "T4", "T5", "T6"]
    )
    year_from: int | None = Field(default=None, ge=1900, le=2100)
    year_to: int | None = Field(default=None, ge=1900, le=2100)
    work_type: Literal["journal-article"] = "journal-article"
    journal_ranking_policy: Literal["required", "preferred"] = "required"
    target_eligible_count: int = Field(default=50, ge=1, le=100)
    max_candidates: int = Field(default=300, ge=1, le=1000)

    @field_validator(
        "primary_keywords",
        "related_keywords",
        "search_queries",
        mode="after",
    )
    @classmethod
    def normalize_text_lists(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for value in values:
            clean = " ".join(value.split())
            fingerprint = clean.casefold()
            if clean and fingerprint not in seen:
                normalized.append(clean)
                seen.add(fingerprint)
        return normalized

    @field_validator("search_queries", mode="after")
    @classmethod
    def reject_unsupported_boolean_syntax(cls, values: list[str]) -> list[str]:
        for value in values:
            if re.search(r"\b(?:AND|OR|NOT)\b", value):
                raise ValueError(
                    "Crossref bibliographic queries must be free-text phrases "
                    "without uppercase Boolean operators"
                )
        return values

    @field_validator("accepted_tiers", mode="after")
    @classmethod
    def normalize_tier_order(cls, values: list[CugTier]) -> list[CugTier]:
        order = ("T1", "T2", "T3", "T4", "T5", "T6")
        return [tier for tier in order if tier in values]

    @model_validator(mode="after")
    def validate_search_bounds(self) -> LiteratureSearchPlan:
        if not self.primary_keywords:
            raise ValueError("at least one primary keyword is required")
        if not self.search_queries:
            raise ValueError("at least one search query is required")
        if not self.accepted_tiers:
            raise ValueError("at least one accepted journal tier is required")
        if (
            self.year_from is not None
            and self.year_to is not None
            and self.year_to < self.year_from
        ):
            raise ValueError("year_to cannot be before year_from")
        return self


class LiteratureCandidate(StrictModel):
    """One DOI-backed candidate returned by a scholarly metadata provider."""

    doi: str
    title: str = Field(min_length=1)
    authors: list[str] = Field(default_factory=list)
    year: int | None = Field(default=None, ge=1000, le=2100)
    journal_title: str = Field(min_length=1)
    issns: list[str] = Field(default_factory=list)
    publisher: str | None = None
    source_type: str = "journal-article"
    source_provider: str = Field(min_length=1)
    source_url: str | None = None
    abstract: str | None = None

    @field_validator("doi")
    @classmethod
    def normalize_doi(cls, value: str) -> str:
        return canonicalize_doi(value)

    @field_validator(
        "title",
        "journal_title",
        "publisher",
        "source_type",
        "source_provider",
        "source_url",
        mode="after",
    )
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = " ".join(value.split())
        return normalized or None

    @field_validator("authors", mode="after")
    @classmethod
    def normalize_authors(cls, values: list[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            clean = " ".join(value.split())
            fingerprint = clean.casefold()
            if clean and fingerprint not in seen:
                result.append(clean)
                seen.add(fingerprint)
        return result

    @field_validator("issns", mode="after")
    @classmethod
    def normalize_issns(cls, values: list[str]) -> list[str]:
        result: list[str] = []
        for value in values:
            try:
                normalized = canonicalize_issn(value)
            except ValueError:
                continue
            if normalized not in result:
                result.append(normalized)
        return result


class JournalRankingRecord(StrictModel):
    """One row preserved from the CUG Wuhan 2023 journal directory."""

    ranking_system: Literal["CUG_WUHAN_TIER"] = "CUG_WUHAN_TIER"
    edition: Literal[2023] = 2023
    category: Literal["理工类", "人文社科类"]
    discipline: str = Field(min_length=1)
    journal_title: str = Field(min_length=1)
    normalized_title: str = Field(min_length=1)
    tier: CugTier
    source_workbook: str = Field(min_length=1)
    source_row: int = Field(ge=2)


class JournalRankingLookup(StrictModel):
    """Explain whether a journal has one usable local classification."""

    status: RankingLookupStatus
    query_title: str = Field(min_length=1)
    normalized_title: str = Field(min_length=1)
    discipline: str = Field(min_length=1)
    records: list[JournalRankingRecord] = Field(default_factory=list)
    reason: str = Field(min_length=1)

    @property
    def resolved_tier(self) -> CugTier | None:
        if self.status != "matched" or not self.records:
            return None
        return self.records[0].tier

    @model_validator(mode="after")
    def status_must_match_evidence(self) -> JournalRankingLookup:
        if self.status == "not_found" and self.records:
            raise ValueError("not_found lookups cannot contain records")
        if self.status in {"matched", "ambiguous"} and not self.records:
            raise ValueError("matched or ambiguous lookups need source records")
        if self.status == "matched" and len({record.tier for record in self.records}) != 1:
            raise ValueError("matched lookups must resolve to one tier")
        if self.status == "ambiguous" and len({record.tier for record in self.records}) < 2:
            raise ValueError("ambiguous lookups need conflicting tiers")
        return self


class NorwegianJournalRankingRecord(StrictModel):
    """One journal row from the open Norwegian Register 2025 classification."""

    ranking_system: Literal["NORWEGIAN_REGISTER"] = "NORWEGIAN_REGISTER"
    edition: Literal[2025] = 2025
    journal_id: str = Field(min_length=1)
    original_title: str = Field(min_length=1)
    international_title: str | None = None
    normalized_titles: list[str] = Field(default_factory=list)
    print_issn: str | None = None
    online_issn: str | None = None
    scientific_field: str | None = None
    level: NorwegianLevel
    source_row: int = Field(ge=2)


class NorwegianJournalRankingLookup(StrictModel):
    """Explain the fixed-year Norwegian Register match independently of CUG."""

    status: RankingLookupStatus
    query_title: str = Field(min_length=1)
    query_issns: list[str] = Field(default_factory=list)
    match_basis: Literal["issn", "title", "none"]
    records: list[NorwegianJournalRankingRecord] = Field(default_factory=list)
    reason: str = Field(min_length=1)

    @property
    def resolved_level(self) -> NorwegianLevel | None:
        if self.status != "matched" or not self.records:
            return None
        return self.records[0].level

    @model_validator(mode="after")
    def status_must_match_evidence(self) -> NorwegianJournalRankingLookup:
        if self.status == "not_found":
            if self.records or self.match_basis != "none":
                raise ValueError("not_found lookups cannot contain matched evidence")
            return self
        if not self.records or self.match_basis == "none":
            raise ValueError("matched or ambiguous lookups need source evidence")
        levels = {record.level for record in self.records}
        if self.status == "matched" and len(levels) != 1:
            raise ValueError("matched lookups must resolve to one Norwegian level")
        if self.status == "ambiguous" and len(levels) < 2:
            raise ValueError("ambiguous lookups need conflicting Norwegian levels")
        return self


class CandidateDecision(StrictModel):
    """Deterministic eligibility decision for one discovered work."""

    status: CandidateDecisionStatus
    candidate: LiteratureCandidate
    ranking: JournalRankingLookup
    norwegian_ranking: NorwegianJournalRankingLookup | None = None
    reason_codes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def eligible_records_need_a_resolved_ranking(self) -> CandidateDecision:
        if self.status == "eligible":
            if self.reason_codes:
                raise ValueError("eligible candidates cannot contain exclusion reasons")
        elif not self.reason_codes:
            raise ValueError("excluded candidates need at least one reason")
        return self


class LiteratureDiscoveryResult(StrictModel):
    """Batch outcome handed to the UI and later RIS exporter."""

    schema_version: str = "0.2.0"
    plan: LiteratureSearchPlan
    scanned_count: int = Field(ge=0)
    duplicate_count: int = Field(ge=0)
    decisions: list[CandidateDecision] = Field(default_factory=list)
    target_reached: bool
    needs_user_confirmation: bool

    @property
    def eligible_records(self) -> list[CandidateDecision]:
        return [decision for decision in self.decisions if decision.status == "eligible"]

    @property
    def excluded_records(self) -> list[CandidateDecision]:
        return [decision for decision in self.decisions if decision.status == "excluded"]

    @model_validator(mode="after")
    def flags_must_match_counts(self) -> LiteratureDiscoveryResult:
        expected_target = len(self.eligible_records) >= self.plan.target_eligible_count
        if self.target_reached != expected_target:
            raise ValueError("target_reached does not match eligible record count")
        if self.needs_user_confirmation == self.target_reached:
            raise ValueError(
                "needs_user_confirmation must be true exactly when the target was not reached"
            )
        return self
