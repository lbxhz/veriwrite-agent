"""Generate a provisional, outline-guided literature search blueprint."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime

from pydantic import ValidationError

from veriwrite_agent.llm.base import LLMClient
from veriwrite_agent.models.literature_selection import (
    LiteratureSearchBlueprint,
)
from veriwrite_agent.models.requirement_workflow import ConfirmedRequirementSpec


class BlueprintPlanningError(ValueError):
    """Raised when a supported provisional search blueprint cannot be produced."""


class LiteratureBlueprintPlanner:
    """Use an LLM for semantic section design while code enforces hard bounds."""

    def __init__(
        self,
        client: LLMClient,
        available_disciplines: Sequence[str],
        *,
        current_year: int | None = None,
    ) -> None:
        disciplines = tuple(dict.fromkeys(item.strip() for item in available_disciplines))
        if not disciplines or any(not item for item in disciplines):
            raise ValueError("available_disciplines must contain non-empty names")
        self._client = client
        self._available_disciplines = disciplines
        self._current_year = current_year or datetime.now().year

    def plan(
        self,
        confirmed: ConfirmedRequirementSpec,
    ) -> LiteratureSearchBlueprint:
        requirement = confirmed.requirement
        if not requirement.topic:
            raise BlueprintPlanningError(
                "confirmed requirements do not contain a research topic"
            )
        target_total = (
            requirement.references.target_total
            or requirement.references.minimum_total
            or 50
        )
        target_total = min(max(target_total, 2), 100)
        schema = json.dumps(
            LiteratureSearchBlueprint.model_json_schema(),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        source = json.dumps(
            {
                "topic": requirement.topic,
                "document_type": requirement.document_type,
                "required_theme_elements": requirement.required_theme_elements,
                "required_sections": (
                    requirement.structure.required_or_recommended_sections
                ),
                "length": requirement.length.model_dump(mode="json"),
                "reference_requirements": requirement.references.model_dump(
                    mode="json"
                ),
                "policy_rules": [
                    rule.model_dump(mode="json")
                    for rule in requirement.policy_rules
                ],
                "available_disciplines": self._available_disciplines,
                "target_total": target_total,
            },
            ensure_ascii=False,
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "你是研究型文献检索蓝图规划器，只返回JSON，不返回Markdown。"
                    "根据已确认需求生成2至8个需要文献支撑的临时章节主题。"
                    "这些章节用于指导检索，不是不可修改的最终论文大纲。"
                    "不要把摘要、关键词、结论或参考文献列表作为检索主题。"
                    "每个主题必须包含研究问题、英文关键词、1至4条Crossref自由文本检索短语"
                    "和文献配额；所有配额之和必须等于target_total。"
                    "检索短语不得使用大写AND、OR、NOT或字段语法。"
                    "discipline必须原样选自available_disciplines。"
                    "不要生成论文、作者或DOI。"
                    "accepted_tiers保留T1至T6；高水平期刊是后续软排序，不是此处硬排除。"
                    f"输出必须符合以下JSON Schema：{schema}"
                ),
            },
            {"role": "user", "content": source},
        ]
        blueprint = self._complete_and_validate(messages)
        if blueprint.discipline not in self._available_disciplines:
            raise BlueprintPlanningError(
                f"LLM selected unsupported discipline: {blueprint.discipline}"
            )

        references = requirement.references
        year_from: int | None = None
        year_to: int | None = None
        if (
            references.recent_year_window is not None
            and references.recent_year_rule_strength == "hard"
        ):
            year_from = (
                self._current_year - references.recent_year_window + 1
            )
            year_to = self._current_year
        try:
            return LiteratureSearchBlueprint.model_validate(
                {
                    **blueprint.model_dump(mode="python"),
                    "topic": requirement.topic,
                    "target_total": target_total,
                    "accepted_tiers": ["T1", "T2", "T3", "T4", "T5", "T6"],
                    "year_from": year_from,
                    "year_to": year_to,
                    "max_candidates": 300,
                }
            )
        except ValidationError as exc:
            raise BlueprintPlanningError(
                "LLM theme quotas do not match the confirmed reference target: "
                f"{self._format_errors(exc)}"
            ) from exc

    def _complete_and_validate(
        self,
        messages: list[dict[str, str]],
    ) -> LiteratureSearchBlueprint:
        raw = self._client.complete(
            messages,
            response_format={"type": "json_object"},
        )
        try:
            return LiteratureSearchBlueprint.model_validate_json(raw)
        except ValidationError as first_error:
            repaired = self._client.complete(
                [
                    *messages,
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": (
                            "上一个JSON未通过数据合同。只修复JSON并返回完整对象。"
                            f"错误：{self._format_errors(first_error)}"
                        ),
                    },
                ],
                response_format={"type": "json_object"},
            )
            try:
                return LiteratureSearchBlueprint.model_validate_json(repaired)
            except ValidationError as repair_error:
                raise BlueprintPlanningError(
                    "LLM blueprint still violates the data contract after repair: "
                    f"{self._format_errors(repair_error)}"
                ) from repair_error

    @staticmethod
    def _format_errors(error: ValidationError) -> str:
        details = [
            {
                "field": ".".join(str(part) for part in item["loc"]),
                "message": item["msg"],
                "type": item["type"],
            }
            for item in error.errors(include_url=False)
        ]
        return json.dumps(details[:20], ensure_ascii=False)
