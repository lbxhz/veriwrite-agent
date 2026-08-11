"""MCP stdio client for the independent ``veriwrite-evaluator`` package."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from veriwrite_agent.config.settings import LLMSettings
from veriwrite_agent.models.external_writing_evaluation import (
    EvaluatorRubricSummary,
    ExternalWritingComparison,
    ExternalWritingEvaluation,
)


class ExternalEvaluatorError(RuntimeError):
    """Raised when the isolated evaluator cannot provide a trusted result."""


EXTERNAL_QUALITY_WARNING_THRESHOLD = 70.0


def external_quality_warning(
    evaluation: ExternalWritingEvaluation,
    *,
    threshold: float = EXTERNAL_QUALITY_WARNING_THRESHOLD,
) -> str | None:
    """Return a visible comparison warning without changing internal release gates."""

    if not 0 <= threshold <= 100:
        raise ValueError("external quality warning threshold must be between 0 and 100")
    if evaluation.aggregate_100 >= threshold:
        return None
    return (
        "内部事实、证据、引用与课程要求门禁已独立运行，但外部写作质量评分为 "
        f"{evaluation.aggregate_100:.1f}/100，低于对照阈值 {threshold:.1f}。"
        "该信号不改变内部 release gate，建议在确认交付前重点复核结构、论证推进、"
        "重复和学术表达。"
    )


@dataclass(frozen=True)
class ExternalEvaluatorConfig:
    """Safe process and evaluation settings for the MCP child process."""

    command: str = sys.executable
    module: str = "veriwrite_evaluator.mcp_server"
    rubric_id: str = "academic_writing_zh_v1"
    target_window_bytes: int = 120000
    max_full_document_bytes: int = 160000
    batch: bool = True
    timeout_seconds: float = 300.0
    cwd: Path | None = None
    environment: dict[str, str] | None = None

    def __post_init__(self) -> None:
        if self.target_window_bytes < 1000:
            raise ValueError("target_window_bytes must be at least 1000")
        if self.max_full_document_bytes < self.target_window_bytes:
            raise ValueError(
                "max_full_document_bytes must cover target_window_bytes"
            )
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

    @classmethod
    def from_veriwrite_environment(
        cls,
        *,
        cwd: Path | None = None,
    ) -> ExternalEvaluatorConfig:
        """Pass the existing VeriWrite secret to a pinned ``deepseek-chat`` judge."""

        settings = LLMSettings()
        target_window_bytes = int(
            os.getenv("VW_EVAL_TARGET_WINDOW_BYTES", "120000")
        )
        max_full_document_bytes = int(
            os.getenv("VW_EVAL_MAX_FULL_DOCUMENT_BYTES", "160000")
        )
        batch = os.getenv("VW_EVAL_BATCH", "1") not in {
            "0",
            "false",
            "False",
        }
        timeout_seconds = float(os.getenv("VW_EVAL_TIMEOUT_SECONDS", "300"))
        environment = dict(os.environ)
        environment.update(
            {
                "VW_EVAL_API_KEY": (
                    os.getenv("VW_EVAL_API_KEY")
                    or settings.api_key.get_secret_value()
                ),
                "VW_EVAL_BASE_URL": (
                    os.getenv("VW_EVAL_BASE_URL") or str(settings.base_url)
                ),
                # Contract-heavy evaluator output must not inherit a reasoning/
                # flash model selected for ordinary prose generation.
                "VW_EVAL_MODEL": "deepseek-chat",
                "VW_EVAL_TEMPERATURE": "0.0",
                "VW_EVAL_BATCH": "1" if batch else "0",
                "VW_EVAL_TARGET_WINDOW_BYTES": str(target_window_bytes),
                "VW_EVAL_TIMEOUT_SECONDS": str(timeout_seconds),
            }
        )
        return cls(
            cwd=cwd,
            environment=environment,
            target_window_bytes=target_window_bytes,
            max_full_document_bytes=max_full_document_bytes,
            batch=batch,
            timeout_seconds=timeout_seconds,
        )


class ExternalWritingEvaluatorClient:
    """Typed synchronous facade over the evaluator's MCP stdio tools."""

    _TOOLS = {
        "list_rubrics",
        "get_rubric",
        "evaluate_writing",
        "evaluate_pairwise",
    }

    def __init__(self, config: ExternalEvaluatorConfig | None = None) -> None:
        self._config = config or ExternalEvaluatorConfig()

    def list_rubrics(self) -> list[EvaluatorRubricSummary]:
        payload = self._call_tool("list_rubrics", {})
        if not isinstance(payload, list):
            raise ExternalEvaluatorError("list_rubrics returned a non-list payload")
        return [EvaluatorRubricSummary.model_validate(item) for item in payload]

    def get_rubric(self, rubric_id: str | None = None) -> dict[str, Any]:
        payload = self._call_tool(
            "get_rubric",
            {"rubric_id": rubric_id or self._config.rubric_id},
        )
        if not isinstance(payload, dict):
            raise ExternalEvaluatorError("get_rubric returned a non-object payload")
        return payload

    def evaluate_writing(
        self,
        target_text: str,
        *,
        rubric_id: str | None = None,
    ) -> ExternalWritingEvaluation:
        if not target_text.strip():
            raise ValueError("target_text cannot be blank")
        target_bytes = len(target_text.encode("utf-8"))
        window = self._full_document_window(target_bytes)
        payload = self._call_tool(
            "evaluate_writing",
            {
                "target_text": target_text,
                "rubric_id": rubric_id or self._config.rubric_id,
                "target_window_bytes": window,
                "batch": self._config.batch,
            },
        )
        evaluation = ExternalWritingEvaluation.model_validate(payload)
        self._verify_target_receipt(evaluation, target_text)
        return evaluation

    def evaluate_pairwise(
        self,
        text_a: str,
        text_b: str,
        *,
        rubric_id: str | None = None,
    ) -> ExternalWritingComparison:
        if not text_a.strip() or not text_b.strip():
            raise ValueError("pairwise texts cannot be blank")
        largest_target = max(
            len(text_a.encode("utf-8")),
            len(text_b.encode("utf-8")),
        )
        window = self._full_document_window(largest_target)
        payload = self._call_tool(
            "evaluate_pairwise",
            {
                "text_a": text_a,
                "text_b": text_b,
                "rubric_id": rubric_id or self._config.rubric_id,
                "target_window_bytes": window,
                "batch": self._config.batch,
            },
        )
        comparison = ExternalWritingComparison.model_validate(payload)
        self._verify_target_receipt(comparison.result_a, text_a)
        self._verify_target_receipt(comparison.result_b, text_b)
        return comparison

    def _full_document_window(self, target_bytes: int) -> int:
        if target_bytes > self._config.max_full_document_bytes:
            raise ExternalEvaluatorError(
                "paper is too large for a trusted full-document evaluation: "
                f"{target_bytes} bytes exceeds "
                f"{self._config.max_full_document_bytes} bytes"
            )
        return max(self._config.target_window_bytes, target_bytes)

    def _call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        if name not in self._TOOLS:
            raise ValueError(f"unsupported evaluator MCP tool: {name}")
        try:
            return asyncio.run(self._call_tool_async(name, arguments))
        except ExternalEvaluatorError:
            raise
        except Exception as exc:
            raise ExternalEvaluatorError(
                f"evaluator MCP call failed for {name}: {exc}"
            ) from exc

    async def _call_tool_async(self, name: str, arguments: dict[str, Any]) -> Any:
        params = StdioServerParameters(
            command=self._config.command,
            args=["-m", self._config.module],
            env=self._config.environment,
            cwd=str(self._config.cwd) if self._config.cwd else None,
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await asyncio.wait_for(
                    session.initialize(),
                    timeout=self._config.timeout_seconds,
                )
                result = await asyncio.wait_for(
                    session.call_tool(name, arguments),
                    timeout=self._config.timeout_seconds,
                )
        if result.isError:
            detail = " ".join(
                str(getattr(item, "text", "")) for item in result.content
            ).strip()
            raise ExternalEvaluatorError(detail or f"{name} returned an MCP error")
        structured = result.structuredContent
        if isinstance(structured, dict):
            payload = structured.get("result", structured)
            if payload is not None:
                return payload
        for item in result.content:
            text = getattr(item, "text", None)
            if not isinstance(text, str):
                continue
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                continue
        raise ExternalEvaluatorError(f"{name} returned no JSON payload")

    @staticmethod
    def _verify_target_receipt(
        evaluation: ExternalWritingEvaluation,
        target_text: str,
    ) -> None:
        inputs = evaluation.receipt.get("inputs")
        if not isinstance(inputs, dict):
            raise ExternalEvaluatorError("evaluation receipt has no inputs")
        recorded_hash = inputs.get("target_hash_sha256")
        actual_hash = hashlib.sha256(target_text.encode("utf-8")).hexdigest()
        if (
            not isinstance(recorded_hash, str)
            or len(recorded_hash) < 16
            or not actual_hash.startswith(recorded_hash)
        ):
            raise ExternalEvaluatorError("evaluation receipt does not match target text")
        if not evaluation.is_full_document_measurement:
            raise ExternalEvaluatorError(
                "external evaluation did not cover the complete paper"
            )
        pipeline = evaluation.receipt["pipeline"]
        if pipeline.get("target_total_bytes") != len(target_text.encode("utf-8")):
            raise ExternalEvaluatorError(
                "evaluation receipt reports a different full-document size"
            )


def comparable_external_evaluations(
    baseline: ExternalWritingEvaluation,
    candidate: ExternalWritingEvaluation,
) -> bool:
    """Prevent cross-rubric or cross-backend score comparisons."""

    return baseline.evaluation_method == candidate.evaluation_method
