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
                    "若文件包含多个教师或多个可选方向，必须分别写入profiles，"
                    "不得把不同教师的题目、禁用来源或处罚规则混成一个题目。"
                    "统一要求写入顶层；教师专属要求写入对应profile。"
                    "“30篇左右”写入references.target_total并将"
                    "target_is_approximate设为true，不得误写成minimum_total。"
                    "单词范围写入minimum_words和maximum_words，不得换算为中文字数。"
                    "禁止来源、学术规范、AI使用、选题、提交方式与处罚均须保留。"
                    "OCR可能把SAR/InSAR识别为SARInSAR、把AI识别为Al，"
                    "只有上下文明确时才可进行这种字符纠正。"
                    "只有原文确实冲突或缺失时才记录ambiguities；"
                    "原文明确允许A或B不属于歧义。"
                    "When a usable topic is present, also produce a conservative "
                    "topic_boundary. State one central research question, the research "
                    "objects that are in scope, objects that are clearly out of scope, "
                    "and adjacent technologies that may appear only as supporting context. "
                    "If these boundaries are inferred rather than quoted from the source, "
                    "set origin to agent_proposed; do not present the proposal as an "
                    "explicit teacher requirement. "
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
        except ValidationError as first_error:
            repair_messages = [
                *messages,
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": (
                        "上一个JSON未通过数据合同。请只修复JSON并返回完整对象，"
                        "不要解释，不得丢失原文已经提取的要求。字段错误如下："
                        f"{self._format_errors(first_error)}"
                    ),
                },
            ]
            repaired = self._client.complete(
                repair_messages,
                response_format={"type": "json_object"},
            )
            try:
                return RequirementSpec.model_validate_json(repaired)
            except ValidationError as repair_error:
                raise LLMOutputValidationError(
                    "LLM JSON 自动修复后仍未通过 RequirementSpec；"
                    f"字段错误：{self._format_errors(repair_error)}"
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
