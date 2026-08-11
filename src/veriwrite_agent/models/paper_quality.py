"""Quantitative, version-comparable evaluation contracts for final papers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import Field, model_validator

from veriwrite_agent.models.requirements import StrictModel

PaperQualityMetricCode = Literal[
    "requirement_compliance",
    "reference_integrity",
    "evidence_traceability",
    "topic_relevance",
    "analysis_synthesis",
    "presentation_quality",
]


class PaperQualityMetric(StrictModel):
    code: PaperQualityMetricCode
    label: str = Field(min_length=1)
    score: float = Field(ge=0, le=100)
    weight: float = Field(gt=0, le=1)
    weighted_points: float = Field(ge=0, le=100)
    basis: list[str] = Field(min_length=1)
    limitation: str | None = None

    @model_validator(mode="after")
    def weighted_points_must_match(self) -> PaperQualityMetric:
        expected = round(self.score * self.weight, 2)
        if abs(self.weighted_points - expected) > 0.01:
            raise ValueError("weighted_points must equal score multiplied by weight")
        return self


class PaperQualityScorecard(StrictModel):
    schema_version: Literal["paper-quality.0"] = "paper-quality.0"
    paper_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation_method: Literal[
        "deterministic-proxy-v1", "process-aware-proxy-v2"
    ] = "process-aware-proxy-v2"
    release_gate: Literal["passed", "blocked"]
    overall_score: float = Field(ge=0, le=100)
    grade: Literal["excellent", "strong", "acceptable", "weak"]
    metrics: list[PaperQualityMetric] = Field(min_length=6, max_length=6)
    blocking_issues: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def aggregate_must_match_metrics(self) -> PaperQualityScorecard:
        if abs(sum(metric.weight for metric in self.metrics) - 1.0) > 0.001:
            raise ValueError("paper quality metric weights must sum to one")
        expected = round(sum(metric.weighted_points for metric in self.metrics), 2)
        if abs(self.overall_score - expected) > 0.01:
            raise ValueError("overall_score must equal weighted metric points")
        if (self.release_gate == "blocked") != bool(self.blocking_issues):
            raise ValueError("release gate must match blocking issues")
        return self


class PaperQualityComparison(StrictModel):
    schema_version: Literal["paper-quality-comparison.0"] = (
        "paper-quality-comparison.0"
    )
    baseline_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    overall_delta: float = Field(ge=-100, le=100)
    metric_deltas: dict[PaperQualityMetricCode, float]
    improved_metrics: list[PaperQualityMetricCode] = Field(default_factory=list)
    regressed_metrics: list[PaperQualityMetricCode] = Field(default_factory=list)
