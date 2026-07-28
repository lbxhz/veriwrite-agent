"""Parse course requirements with any client that implements LLMClient."""

from __future__ import annotations

import json

from pydantic import ValidationError

from veriwrite_agent.llm.base import LLMClient
from veriwrite_agent.models.requirements import RequirementSpec


class LLMOutputValidationError(ValueError):
    """Raised when LLM JSON does not satisfy RequirementSpec."""


class LLMRequirementParser:
    """Semantic parser whose output is constrained by RequirementSpec."""

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def parse(self, text: str) -> RequirementSpec:
        schema = json.dumps(
            RequirementSpec.model_json_schema(),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "你是课程要求解析器。只返回JSON对象，不要返回Markdown。"
                    "缺失信息使用null或空列表，不得猜测。"
                    "模板中的示例题目不得当作用户真实题目。"
                    "大学名称写入institution，学院或院系写入school_or_department。"
                    "同义文献类型必须使用JSON Schema描述中的规范英文值。"
                    "只有原文确实冲突或缺失时才记录ambiguities；"
                    "原文明确允许A或B不属于歧义。"
                    f"输出必须符合以下JSON Schema：{schema}"
                ),
            },
            {"role": "user", "content": text},
        ]
        raw = self._client.complete(
            messages,
            response_format={"type": "json_object"},
        )
        try:
            return RequirementSpec.model_validate_json(raw)
        except ValidationError as exc:
            raise LLMOutputValidationError(
                "LLM output did not satisfy RequirementSpec"
            ) from exc
