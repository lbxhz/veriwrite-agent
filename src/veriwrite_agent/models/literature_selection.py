"""Contracts for outline-guided discovery, relevance, and balanced selection."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Literal

from pydantic import Field, field_validator, model_validator

from veriwrite_agent.models.executable_policy import ExecutableRequirementPolicy
from veriwrite_agent.models.literature_discovery import (
    CugTier,
    JournalRankingLookup,
    LiteratureSearchPlan,
    NorwegianJournalRankingLookup,
    NorwegianLevel,
    RankingLookupStatus,
    canonicalize_doi,
)
from veriwrite_agent.models.literature_verification import (
    LiteratureVerificationResult,
)
from veriwrite_agent.models.requirements import StrictModel


class LiteratureThemePlan(StrictModel):
    """One literature-bearing provisional outline section."""

    theme_id: str = Field(pattern=r"^[a-z][a-z0-9_]{1,39}$")
    section_title: str = Field(min_length=1)
    section_purpose: str = Field(min_length=1)
    research_questions: list[str] = Field(min_length=1, max_length=4)
    primary_keywords: list[str] = Field(min_length=1, max_length=10)
    related_keywords: list[str] = Field(default_factory=list, max_length=20)
    search_queries: list[str] = Field(min_length=1, max_length=4)
    target_count: int = Field(ge=1, le=30)
    priority: int = Field(default=1, ge=1, le=10)

    @field_validator(
        "research_questions",
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
    def reject_boolean_query_syntax(cls, values: list[str]) -> list[str]:
        if any(re.search(r"\b(?:AND|OR|NOT)\b", value) for value in values):
            raise ValueError("Crossref search phrases cannot use uppercase Boolean operators")
        return values


class LiteratureSearchBlueprint(StrictModel):
    """A provisional outline that controls literature discovery, not final writing."""

    schema_version: Literal["0.2.2"] = "0.2.2"
    outline_status: Literal["provisional_search_blueprint"] = "provisional_search_blueprint"
    topic: str = Field(min_length=1)
    discipline: str = Field(min_length=1)
    writing_through_line: str = Field(min_length=1)
    target_total: int = Field(ge=2, le=100)
    themes: list[LiteratureThemePlan] = Field(min_length=2, max_length=8)
    accepted_tiers: list[CugTier] = Field(
        default_factory=lambda: ["T1", "T2", "T3", "T4", "T5", "T6"]
    )
    journal_ranking_policy: Literal["preferred"] = "preferred"
    year_from: int | None = Field(default=None, ge=1900, le=2100)
    year_to: int | None = Field(default=None, ge=1900, le=2100)
    max_candidates: int = Field(default=300, ge=20, le=1000)
    relevance_threshold: float = Field(default=0.6, ge=0, le=1)
    requirement_policy: ExecutableRequirementPolicy | None = None

    @field_validator("accepted_tiers", mode="after")
    @classmethod
    def normalize_tier_order(cls, values: list[CugTier]) -> list[CugTier]:
        order = ("T1", "T2", "T3", "T4", "T5", "T6")
        return [tier for tier in order if tier in values]

    @model_validator(mode="after")
    def validate_blueprint(self) -> LiteratureSearchBlueprint:
        theme_ids = [theme.theme_id for theme in self.themes]
        if len(theme_ids) != len(set(theme_ids)):
            raise ValueError("theme_id values must be unique")
        if sum(theme.target_count for theme in self.themes) != self.target_total:
            raise ValueError("theme target_count values must sum to target_total")
        if not self.accepted_tiers:
            raise ValueError("accepted_tiers cannot be empty")
        if (
            self.year_from is not None
            and self.year_to is not None
            and self.year_to < self.year_from
        ):
            raise ValueError("year_to cannot be before year_from")
        return self


class ConfirmedLiteratureSearchBlueprint(StrictModel):
    """Exact search blueprint approved by a user before external retrieval."""

    schema_version: Literal["0.2.2"] = "0.2.2"
    status: Literal["confirmed"] = "confirmed"
    confirmed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    confirmed_by: str = Field(min_length=1)
    confirmation_note: str | None = None
    blueprint: LiteratureSearchBlueprint


class ThemedLiteratureSearchPlan(StrictModel):
    """One executable Crossref plan derived from a blueprint theme."""

    theme_id: str = Field(min_length=1)
    plan: LiteratureSearchPlan


class ThemeRelevanceScore(StrictModel):
    """Semantic fit of one verified paper to one provisional section."""

    theme_id: str = Field(min_length=1)
    score: float = Field(ge=0, le=1)
    rationale: str = Field(min_length=1)
    matched_concepts: list[str] = Field(default_factory=list)


class LiteratureRelevanceAssessment(StrictModel):
    """LLM assessment constrained to existing DOI and blueprint themes."""

    doi: str
    theme_scores: list[ThemeRelevanceScore] = Field(min_length=1)
    best_theme_id: str = Field(min_length=1)

    @field_validator("doi")
    @classmethod
    def normalize_doi(cls, value: str) -> str:
        return canonicalize_doi(value)

    @model_validator(mode="after")
    def best_theme_must_have_the_highest_score(
        self,
    ) -> LiteratureRelevanceAssessment:
        scores = {item.theme_id: item.score for item in self.theme_scores}
        if len(scores) != len(self.theme_scores):
            raise ValueError("theme relevance scores must be unique by theme_id")
        if self.best_theme_id not in scores:
            raise ValueError("best_theme_id must appear in theme_scores")
        if scores[self.best_theme_id] != max(scores.values()):
            raise ValueError("best_theme_id must have the highest relevance score")
        return self


class LiteratureRelevanceAssessmentBatch(StrictModel):
    assessments: list[LiteratureRelevanceAssessment] = Field(default_factory=list)


class LiteratureSelectionCandidate(StrictModel):
    """All evidence needed for deterministic final literature ranking."""

    verification: LiteratureVerificationResult
    ranking: JournalRankingLookup
    norwegian_ranking: NorwegianJournalRankingLookup | None = None
    relevance: LiteratureRelevanceAssessment

    @model_validator(mode="after")
    def evidence_must_describe_one_eligible_doi(
        self,
    ) -> LiteratureSelectionCandidate:
        doi = self.verification.candidate.doi
        if self.verification.status != "verified":
            raise ValueError("selection candidates must be verified")
        if self.relevance.doi != doi:
            raise ValueError("relevance DOI must match verification DOI")
        return self


class SelectedLiteratureRecord(StrictModel):
    """One selected paper and the explainable reasons for its position."""

    doi: str
    title: str = Field(min_length=1)
    authors: list[str] = Field(default_factory=list)
    journal: str | None = None
    publisher: str | None = None
    language: str | None = None
    source_type: str = "journal-article"
    is_foreign: bool
    theme_id: str = Field(min_length=1)
    relevance_score: float = Field(ge=0, le=1)
    cug_tier: CugTier | None = None
    ranking_status: RankingLookupStatus
    norwegian_level: NorwegianLevel | None = None
    norwegian_ranking_status: RankingLookupStatus | None = None
    norwegian_match_basis: Literal["issn", "title", "none"] | None = None
    year: int = Field(ge=1000, le=2100)
    selection_reasons: list[str] = Field(min_length=1)

    @field_validator("doi")
    @classmethod
    def normalize_doi(cls, value: str) -> str:
        return canonicalize_doi(value)


class BalancedLiteratureSelection(StrictModel):
    """Balanced final pool plus section-level shortages for the UI."""

    schema_version: Literal["0.2.3"] = "0.2.3"
    blueprint: LiteratureSearchBlueprint
    selected: list[SelectedLiteratureRecord] = Field(default_factory=list)
    shortages: dict[str, int] = Field(default_factory=dict)
    policy_issues: list[str] = Field(default_factory=list)
    target_reached: bool

    @model_validator(mode="after")
    def result_must_respect_theme_quotas(self) -> BalancedLiteratureSelection:
        dois = [record.doi for record in self.selected]
        if len(dois) != len(set(dois)):
            raise ValueError("a DOI can only be selected once")
        targets = {theme.theme_id: theme.target_count for theme in self.blueprint.themes}
        counts = {
            theme_id: sum(record.theme_id == theme_id for record in self.selected)
            for theme_id in targets
        }
        if any(counts[theme_id] > targets[theme_id] for theme_id in targets):
            raise ValueError("selected records cannot exceed a theme quota")
        expected_shortages = {
            theme_id: targets[theme_id] - counts[theme_id]
            for theme_id in targets
            if counts[theme_id] < targets[theme_id]
        }
        if self.shortages != expected_shortages:
            raise ValueError("shortages must match the selected theme counts")
        expected_reached = (
            len(self.selected) == self.blueprint.target_total and not self.policy_issues
        )
        if self.target_reached != expected_reached:
            raise ValueError("target_reached does not match the selected count")
        return self
