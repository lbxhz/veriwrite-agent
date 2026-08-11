"""Contracts for evaluating uploaded papers outside the writing workflow."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import Field, model_validator

from veriwrite_agent.models.requirements import StrictModel

StandaloneMetricCode = Literal[
    "requirement_compliance",
    "citation_consistency",
    "topic_focus",
    "analysis_synthesis",
    "structure_organization",
    "language_style",
]


class StandaloneSemanticMetric(StrictModel):
    code: Literal[
        "requirement_compliance",
        "citation_consistency",
        "topic_focus",
        "analysis_synthesis",
        "structure_organization",
        "language_style",
    ]
    score: float = Field(ge=0, le=100)
    rationale: str = Field(min_length=1, max_length=600)


class StandaloneFinding(StrictModel):
    dimension: StandaloneMetricCode
    severity: Literal["minor", "major"]
    location: str = Field(min_length=1, max_length=120)
    detail: str = Field(min_length=1, max_length=500)
    recommendation: str = Field(min_length=1, max_length=500)


class StandaloneSemanticReview(StrictModel):
    inferred_title: str = Field(min_length=1, max_length=300)
    inferred_topic: str = Field(min_length=1, max_length=500)
    metrics: list[StandaloneSemanticMetric] = Field(min_length=6, max_length=6)
    findings: list[StandaloneFinding] = Field(default_factory=list, max_length=24)

    @model_validator(mode="after")
    def semantic_metric_codes_must_be_complete(self) -> StandaloneSemanticReview:
        expected = {
            "requirement_compliance",
            "citation_consistency",
            "topic_focus",
            "analysis_synthesis",
            "structure_organization",
            "language_style",
        }
        actual = {metric.code for metric in self.metrics}
        if actual != expected or len(actual) != len(self.metrics):
            raise ValueError("semantic review must contain every metric exactly once")
        return self


class StandaloneQualityMetric(StrictModel):
    code: StandaloneMetricCode
    label: str = Field(min_length=1)
    score: float = Field(ge=0, le=100)
    weight: float = Field(gt=0, le=1)
    weighted_points: float = Field(ge=0, le=100)
    basis: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def weighted_points_must_match(self) -> StandaloneQualityMetric:
        expected = round(self.score * self.weight, 2)
        if abs(self.weighted_points - expected) > 0.01:
            raise ValueError("weighted_points must equal score multiplied by weight")
        return self


class StandalonePaperEvaluation(StrictModel):
    schema_version: Literal["standalone-paper-quality.1"] = (
        "standalone-paper-quality.1"
    )
    evaluation_method: Literal["document-quality-judge-v2"] = (
        "document-quality-judge-v2"
    )
    paper_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    criteria_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_filename: str = Field(min_length=1, max_length=255)
    source_format: Literal["docx", "pdf"]
    extraction_method: str = Field(min_length=1)
    page_count: int | None = Field(default=None, ge=1)
    counted_units: int = Field(ge=1)
    reference_count: int = Field(ge=0)
    citation_marker_count: int = Field(ge=0)
    inferred_title: str = Field(min_length=1, max_length=300)
    inferred_topic: str = Field(min_length=1, max_length=500)
    overall_score: float = Field(ge=0, le=100)
    grade: Literal["excellent", "strong", "acceptable", "weak"]
    metrics: list[StandaloneQualityMetric] = Field(min_length=6, max_length=6)
    findings: list[StandaloneFinding] = Field(default_factory=list, max_length=24)
    reviewer_model: str = Field(min_length=1)
    limitations: list[str] = Field(default_factory=list)
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def aggregate_must_match_metrics(self) -> StandalonePaperEvaluation:
        if abs(sum(metric.weight for metric in self.metrics) - 1.0) > 0.001:
            raise ValueError("standalone metric weights must sum to one")
        if len({metric.code for metric in self.metrics}) != len(self.metrics):
            raise ValueError("standalone metric codes must be unique")
        expected = round(sum(metric.weighted_points for metric in self.metrics), 2)
        if abs(self.overall_score - expected) > 0.01:
            raise ValueError("overall_score must equal weighted metric points")
        return self
