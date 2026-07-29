"""Use an LLM once to turn confirmed requirements into a bounded search plan."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime

from pydantic import ValidationError

from veriwrite_agent.llm.base import LLMClient
from veriwrite_agent.models.literature_discovery import LiteratureSearchPlan
from veriwrite_agent.models.requirement_workflow import ConfirmedRequirementSpec


class KeywordPlanningError(ValueError):
    """Raised when the LLM cannot produce a valid, supported search plan."""


class LiteratureKeywordPlanner:
    """Keep semantic query generation separate from deterministic retrieval."""

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

    def plan(self, confirmed: ConfirmedRequirementSpec) -> LiteratureSearchPlan:
        requirement = confirmed.requirement
        if not requirement.topic:
            raise KeywordPlanningError("confirmed requirements do not contain a research topic")

        schema = json.dumps(
            LiteratureSearchPlan.model_json_schema(),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        source = json.dumps(
            {
                "topic": requirement.topic,
                "required_theme_elements": requirement.required_theme_elements,
                "reference_requirements": requirement.references.model_dump(mode="json"),
                "policy_rules": [
                    rule.model_dump(mode="json") for rule in requirement.policy_rules
                ],
                "available_disciplines": self._available_disciplines,
            },
            ensure_ascii=False,
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "你是文献检索计划生成器，只返回JSON对象，不返回Markdown。"
                    "根据已确认的研究主题生成英文核心关键词、同义词和1至8条Crossref检索短语。"
                    "每条检索短语使用简洁的自由文本，不使用大写AND、OR、NOT或字段查询语法。"
                    "discipline必须从available_disciplines中原样选择最合适的一项。"
                    "不要判断论文真假，不要编造文献、DOI或期刊等级。"
                    "accepted_tiers只有在需求明确限制T等级时才收窄，否则保留T1至T6。"
                    "target_eligible_count必须是50，max_candidates必须是300。"
                    f"输出必须符合以下JSON Schema：{schema}"
                ),
            },
            {"role": "user", "content": source},
        ]
        raw = self._client.complete(
            messages,
            response_format={"type": "json_object"},
        )
        try:
            plan = LiteratureSearchPlan.model_validate_json(raw)
        except ValidationError as first_error:
            repair_messages = [
                *messages,
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": (
                        "上一个JSON未通过数据合同。只修复JSON并返回完整对象。"
                        f"错误：{self._format_errors(first_error)}"
                    ),
                },
            ]
            repaired = self._client.complete(
                repair_messages,
                response_format={"type": "json_object"},
            )
            try:
                plan = LiteratureSearchPlan.model_validate_json(repaired)
            except ValidationError as repair_error:
                raise KeywordPlanningError(
                    "LLM search plan still violates LiteratureSearchPlan after repair: "
                    f"{self._format_errors(repair_error)}"
                ) from repair_error

        if plan.discipline not in self._available_disciplines:
            raise KeywordPlanningError(
                f"LLM selected unsupported discipline: {plan.discipline}"
            )

        updates: dict[str, object] = {"topic": requirement.topic}
        references = requirement.references
        if (
            references.recent_year_window is not None
            and references.recent_year_rule_strength == "hard"
        ):
            updates["year_from"] = self._current_year - references.recent_year_window + 1
            updates["year_to"] = self._current_year
        return LiteratureSearchPlan.model_validate(
            {
                **plan.model_dump(mode="python"),
                **updates,
            }
        )

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
