"""Contracts for the independent hermes-rubric MCP writing evaluator."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import Field, model_validator

from veriwrite_agent.models.requirements import StrictModel

_METHOD_PATTERN = re.compile(
    r"^hermes-rubric-(?P<version>v1|v2)\|"
    r"(?:(?:scope=(?P<scope>[^|]+)\|))?rubric=(?P<rubric>[^|]+)\|"
    r"rubric_hash=(?P<hash>[0-9a-f]{64})\|backend=deepseek-openai$"
)


class EvaluatorRubricSummary(StrictModel):
    """One immutable evaluator rubric advertised over MCP."""

    id: str = Field(min_length=1)
    file: str = Field(min_length=1)
    dimensions: list[str] = Field(min_length=1)
    rubric_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class EvaluatorEvidenceCitation(StrictModel):
    quote: str = Field(min_length=1)
    evidence_id: str = Field(min_length=1)
    location: str = Field(min_length=1)
    source_class: str = Field(min_length=1)


class EvaluatorDimensionEvidence(StrictModel):
    dim_id: str = Field(min_length=1)
    dim_name: str = Field(min_length=1)
    evidence_found: bool
    confidence: str = Field(min_length=1)
    hedge: bool
    citations: list[EvaluatorEvidenceCitation] = Field(default_factory=list)
    evidence_summary: str = ""
    source_class_mix: dict[str, int] = Field(default_factory=dict)


class EvaluatorDimensionScore(StrictModel):
    dim_id: str = Field(min_length=1)
    dim_name: str = Field(min_length=1)
    score: float = Field(ge=0, le=10)
    score_rationale: str = Field(min_length=1)
    evidence_drove_score: str = ""
    hedge_applied: bool
    citation_source_weight: float = Field(ge=0)


class EvaluatorDimensionSummary(StrictModel):
    dim_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    score: float = Field(ge=0, le=10)
    score_100: float = Field(ge=0, le=100)
    weight: float = Field(gt=0)
    hedge: bool

    @model_validator(mode="after")
    def scales_must_match(self) -> EvaluatorDimensionSummary:
        if abs(self.score_100 - self.score * 10) > 0.11:
            raise ValueError("score_100 must be the 0-100 form of score")
        return self


class ExternalWritingEvaluation(StrictModel):
    """Validated full response from ``evaluate_writing``."""

    rubric_id: str = Field(min_length=1)
    rubric: dict[str, Any]
    evidence_citations: list[EvaluatorDimensionEvidence] = Field(min_length=1)
    per_dim_scores: list[EvaluatorDimensionScore] = Field(min_length=1)
    aggregate: float = Field(ge=0, le=10)
    aggregate_100: float = Field(ge=0, le=100)
    max_possible: float = Field(gt=0, le=10)
    hedge_dims: list[str] = Field(default_factory=list)
    hedge_note: str = ""
    dim_summaries: list[EvaluatorDimensionSummary] = Field(min_length=1)
    evaluation_method: str = Field(min_length=1)
    receipt: dict[str, Any]

    @property
    def rubric_hash(self) -> str:
        match = _METHOD_PATTERN.fullmatch(self.evaluation_method)
        if match is None:  # guarded by model validation
            raise ValueError("invalid evaluation_method")
        return match.group("hash")

    @property
    def is_full_document_measurement(self) -> bool:
        method = _METHOD_PATTERN.fullmatch(self.evaluation_method)
        if method is None or method.group("version") != "v2":
            return False
        pipeline = self.receipt.get("pipeline")
        return bool(
            method.group("scope") == "full-document"
            and isinstance(pipeline, dict)
            and pipeline.get("target_scope") == "full_document"
            and pipeline.get("target_truncated") is False
            and pipeline.get("target_coverage_ratio") == 1.0
            and pipeline.get("target_visible_bytes")
            == pipeline.get("target_total_bytes")
        )

    @model_validator(mode="after")
    def validate_measurement_identity(self) -> ExternalWritingEvaluation:
        method = _METHOD_PATTERN.fullmatch(self.evaluation_method)
        if method is None:
            raise ValueError("unsupported external evaluation_method fingerprint")
        if method.group("rubric") != self.rubric_id:
            raise ValueError("evaluation_method rubric does not match rubric_id")
        if abs(self.aggregate_100 - self.aggregate * 10) > 0.11:
            raise ValueError("aggregate_100 must be the 0-100 form of aggregate")
        score_ids = {item.dim_id for item in self.per_dim_scores}
        summary_ids = {item.dim_id for item in self.dim_summaries}
        evidence_ids = {item.dim_id for item in self.evidence_citations}
        if score_ids != summary_ids or score_ids != evidence_ids:
            raise ValueError("evidence, scores, and summaries must cover identical dimensions")
        pipeline = self.receipt.get("pipeline")
        if not isinstance(pipeline, dict):
            raise ValueError("receipt.pipeline is required")
        if pipeline.get("stage_1_rubric_hash_sha256") != method.group("hash"):
            raise ValueError("receipt rubric hash does not match evaluation_method")
        if self.receipt.get("backend") != "deepseek-openai":
            raise ValueError("external evaluator must use the DeepSeek backend")
        if method.group("version") == "v2" and not self.is_full_document_measurement:
            raise ValueError("v2 external evaluation must cover the full document")
        return self


class PairwiseDimensionDelta(StrictModel):
    name: str = Field(min_length=1)
    a: float = Field(ge=0, le=100)
    b: float = Field(ge=0, le=100)
    delta_b_minus_a: float = Field(ge=-100, le=100)


class ExternalWritingComparison(StrictModel):
    """Version-isolated response from ``evaluate_pairwise``."""

    rubric_id: str = Field(min_length=1)
    overall_delta_b_minus_a: float = Field(ge=-100, le=100)
    dim_deltas: dict[str, PairwiseDimensionDelta]
    preferred: Literal["A", "B", "tie"]
    result_a: ExternalWritingEvaluation
    result_b: ExternalWritingEvaluation

    @model_validator(mode="after")
    def evaluations_must_be_comparable(self) -> ExternalWritingComparison:
        if self.result_a.evaluation_method != self.result_b.evaluation_method:
            raise ValueError("different evaluator fingerprints are not comparable")
        if self.result_a.rubric_id != self.rubric_id:
            raise ValueError("pairwise rubric does not match embedded evaluations")
        expected = round(
            self.result_b.aggregate_100 - self.result_a.aggregate_100,
            1,
        )
        if abs(self.overall_delta_b_minus_a - expected) > 0.11:
            raise ValueError("pairwise aggregate delta does not match embedded results")
        return self
