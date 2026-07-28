"""Testable application layer used by the local Streamlit workbench."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any, Literal

from veriwrite_agent.config.settings import LLMSettings
from veriwrite_agent.llm.deepseek_client import DeepSeekClient
from veriwrite_agent.models.requirement_workflow import RequirementReviewPackage
from veriwrite_agent.services.llm_requirement_parser import LLMRequirementParser
from veriwrite_agent.services.requirement_input import extract_requirement_text
from veriwrite_agent.services.requirement_parser import RuleBasedRequirementParser
from veriwrite_agent.services.requirement_pipeline import RequirementReviewPipeline


@dataclass(frozen=True)
class SampleCase:
    label: str
    path: Path
    focus: str


@dataclass(frozen=True)
class WorkbenchResult:
    review: RequirementReviewPackage
    source_name: str
    source_format: str
    extracted_text: str
    extraction_method: str
    extraction_warnings: tuple[str, ...]
    ocr_average_confidence: float | None
    elapsed_seconds: float


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def built_in_samples() -> list[SampleCase]:
    fixture_dir = project_root() / "tests" / "fixtures"
    return [
        SampleCase(
            "完整课程要求",
            fixture_dir / "course_requirement.txt",
            "综合硬约束、软约束、歧义与模板示例",
        ),
        SampleCase(
            "外文比例表达",
            fixture_dir / "foreign_ratio_review.txt",
            "外文三分之一与近年文献偏好",
        ),
        SampleCase(
            "指定章节结构",
            fixture_dir / "specified_sections_review.txt",
            "章节、原创分析与禁止摘要罗列",
        ),
        SampleCase(
            "图表与格式要求",
            fixture_dir / "figures_and_format_review.txt",
            "图件、公式、字体、纸张与行距",
        ),
        SampleCase(
            "冲突字数表达",
            fixture_dir / "ambiguous_length_review.txt",
            "“左右”与“以上”的冲突及模板题目隔离",
        ),
    ]


def prepare_review_from_path(
    path: Path,
    *,
    mode: Literal["rule", "dual"],
) -> WorkbenchResult:
    started = perf_counter()
    extraction = extract_requirement_text(path)
    review = _build_pipeline(mode).prepare(extraction.text)
    return WorkbenchResult(
        review=review,
        source_name=path.name,
        source_format=path.suffix.lower() or "text",
        extracted_text=extraction.text,
        extraction_method=extraction.method,
        extraction_warnings=extraction.warnings,
        ocr_average_confidence=extraction.ocr_average_confidence,
        elapsed_seconds=perf_counter() - started,
    )


def prepare_review_from_upload(
    filename: str,
    payload: bytes,
    *,
    mode: Literal["rule", "dual"],
) -> WorkbenchResult:
    suffix = Path(filename).suffix.lower()
    with TemporaryDirectory(prefix="veriwrite-upload-") as temp_dir:
        path = Path(temp_dir) / f"uploaded{suffix}"
        path.write_bytes(payload)
        result = prepare_review_from_path(path, mode=mode)
    return WorkbenchResult(
        review=result.review,
        source_name=filename,
        source_format=suffix or "text",
        extracted_text=result.extracted_text,
        extraction_method=result.extraction_method,
        extraction_warnings=result.extraction_warnings,
        ocr_average_confidence=result.ocr_average_confidence,
        elapsed_seconds=result.elapsed_seconds,
    )


def comparison_rows(review: RequirementReviewPackage) -> list[dict[str, str]]:
    rule_values = flatten_spec(
        review.rule_run.spec.model_dump(mode="json")
        if review.rule_run.spec is not None
        else {}
    )
    llm_values = flatten_spec(
        review.llm_run.spec.model_dump(mode="json")
        if review.llm_run is not None and review.llm_run.spec is not None
        else {}
    )
    conflict_fields = {
        conflict.field for conflict in review.reconciliation.conflicts
    }
    rows: list[dict[str, str]] = []
    for field in sorted(set(rule_values) | set(llm_values)):
        if field == "source_evidence":
            continue
        rule_value = rule_values.get(field)
        llm_value = llm_values.get(field)
        if field in conflict_fields:
            status = "需用户确认"
        elif review.llm_run is None:
            status = "仅规则模式"
        elif rule_value == llm_value:
            status = "一致"
        elif _is_empty(rule_value):
            status = "LLM 补充"
        elif _is_empty(llm_value):
            status = "规则补充"
        else:
            status = "规范化后一致"
        rows.append(
            {
                "字段": field,
                "规则结果": display_value(rule_value),
                "LLM结果": display_value(llm_value),
                "判断": status,
            }
        )
    return rows


def diagnostic_messages(
    review: RequirementReviewPackage,
) -> tuple[list[str], list[str]]:
    advantages: list[str] = []
    problems: list[str] = []
    rows = comparison_rows(review)

    if review.llm_run is not None and review.llm_run.status == "succeeded":
        advantages.append("规则与 LLM 两条路径均成功，能够进行独立交叉检查。")
    elif review.parser_mode == "rule_only":
        advantages.append("规则模式无需 API，结果稳定且可重复。")
    else:
        error = _explain_llm_error(
            review.llm_run.error if review.llm_run is not None else None
        )
        problems.append(
            "LLM 路径失败，本次结果仅由规则解析器兜底。"
            f"原因：{error}"
        )

    normalized = sum(row["判断"] == "规范化后一致" for row in rows)
    if normalized:
        advantages.append(f"规范化层自动消除了 {normalized} 项表达差异。")

    llm_additions = sum(row["判断"] == "LLM 补充" for row in rows)
    if llm_additions:
        advantages.append(f"LLM 补充了 {llm_additions} 个规则路径缺失字段。")

    conflict_count = len(review.reconciliation.conflicts)
    if conflict_count:
        problems.append(f"仍有 {conflict_count} 个实质冲突需要用户判断。")
    elif review.llm_run is not None and review.llm_run.status == "succeeded":
        advantages.append("双路比较没有发现遗留字段冲突。")
    elif review.parser_mode == "rule_only":
        advantages.append("规则结果内部没有待裁决字段冲突。")
    else:
        problems.append("LLM 未成功返回，本次无法判断双路之间是否存在冲突。")

    blocking = review.completeness.blocking_count
    warnings = review.completeness.warning_count
    if blocking:
        problems.append(f"存在 {blocking} 个阻塞项，当前不能交给 V0.2。")
    if warnings:
        problems.append(f"存在 {warnings} 个警告，需要确认或保留为已知风险。")

    advantages.append("所有候选和用户修改都受 RequirementSpec 数据合同约束。")
    return advantages, problems


def flatten_spec(
    value: dict[str, Any],
    prefix: tuple[str, ...] = (),
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, child in value.items():
        path = (*prefix, key)
        if isinstance(child, dict):
            result.update(flatten_spec(child, path))
        else:
            result[".".join(path)] = child
    return result


def display_value(value: Any) -> str:
    if value is None:
        return "—"
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _build_pipeline(
    mode: Literal["rule", "dual"],
) -> RequirementReviewPipeline:
    llm_parser = None
    if mode == "dual":
        settings = LLMSettings()
        llm_parser = LLMRequirementParser(DeepSeekClient(settings))
    return RequirementReviewPipeline(
        RuleBasedRequirementParser(),
        llm_parser=llm_parser,
    )


def _is_empty(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def _explain_llm_error(error: str | None) -> str:
    if not error:
        return "未返回错误详情"
    normalized = " ".join(error.split())[:500]
    error_kinds = (
        ("AuthenticationError", "API Key 或鉴权失败"),
        ("RateLimitError", "API 限流或账户额度不足"),
        ("APITimeoutError", "API 请求超时"),
        ("APIConnectionError", "无法连接 LLM 接口"),
        ("LLMOutputValidationError", "LLM 返回内容未通过 RequirementSpec 校验"),
    )
    for marker, explanation in error_kinds:
        if marker in normalized:
            return f"{explanation}（{normalized}）"
    return normalized
