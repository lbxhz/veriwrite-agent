"""Render a RequirementReviewPackage as a user-facing confirmation form."""

from __future__ import annotations

import json
from typing import Any

from veriwrite_agent.models.requirement_workflow import RequirementReviewPackage


class RequirementReviewRenderer:
    """Keep presentation separate from parsing and reconciliation logic."""

    def render_markdown(self, review: RequirementReviewPackage) -> str:
        spec = review.reconciliation.merged_spec
        lines = [
            "# 课程要求确认单",
            "",
            f"- 状态：`{review.status}`",
            f"- 解析模式：`{review.parser_mode}`",
            f"- 生成时间：`{review.created_at.isoformat()}`",
            "",
            "## 合并后的临时要求",
            "",
            "| 字段 | 临时值 |",
            "| --- | --- |",
            self._row("document_type", spec.document_type),
            self._row("institution", spec.institution),
            self._row("school_or_department", spec.school_or_department),
            self._row("topic", spec.topic),
            self._row("length.minimum_chars", spec.length.minimum_chars),
            self._row("length.target_chars", spec.length.target_chars),
            self._row(
                "length.counting_policy",
                spec.length.counting_policy,
            ),
            self._row(
                "references.minimum_total",
                spec.references.minimum_total,
            ),
            self._row(
                "references.minimum_foreign_ratio",
                spec.references.minimum_foreign_ratio,
            ),
            self._row(
                "references.bibliography_style",
                spec.references.bibliography_style,
            ),
            "",
            "## 两条解析路径",
            "",
            "| 解析器 | 状态 | 说明 |",
            "| --- | --- | --- |",
            self._parser_row(
                review.rule_run.parser_name,
                review.rule_run.status,
                review.rule_run.error,
            ),
        ]
        if review.llm_run is not None:
            lines.append(
                self._parser_row(
                    review.llm_run.parser_name,
                    review.llm_run.status,
                    review.llm_run.error,
                )
            )
        else:
            lines.append("| llm | 未运行 | 本次使用 rule-only 模式 |")

        lines.extend(["", "## 候选结果对照", ""])
        rule_values = self._flatten(
            review.rule_run.spec.model_dump(mode="json")
            if review.rule_run.spec is not None
            else {}
        )
        llm_values = self._flatten(
            review.llm_run.spec.model_dump(mode="json")
            if review.llm_run is not None and review.llm_run.spec is not None
            else {}
        )
        fields = sorted(
            field
            for field in set(rule_values) | set(llm_values)
            if field != "source_evidence"
        )
        if llm_values:
            lines.extend(
                [
                    "| 字段 | 规则候选 | LLM 候选 |",
                    "| --- | --- | --- |",
                ]
            )
            for field in fields:
                lines.append(
                    f"| {self._escape(field)} | "
                    f"{self._format(rule_values.get(field))} | "
                    f"{self._format(llm_values.get(field))} |"
                )
        else:
            lines.extend(["| 字段 | 规则候选 |", "| --- | --- |"])
            for field in fields:
                lines.append(
                    f"| {self._escape(field)} | "
                    f"{self._format(rule_values.get(field))} |"
                )

        lines.extend(["", "## 解析冲突", ""])
        if review.reconciliation.conflicts:
            lines.extend(
                [
                    "| 字段 | 规则结果 | LLM 结果 | 临时值 |",
                    "| --- | --- | --- | --- |",
                ]
            )
            for conflict in review.reconciliation.conflicts:
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            self._escape(conflict.field),
                            self._format(conflict.rule_value),
                            self._format(conflict.llm_value),
                            self._format(conflict.provisional_value),
                        ]
                    )
                    + " |"
                )
        else:
            lines.append("没有发现双路字段冲突。")

        lines.extend(["", "## 待处理事项", ""])
        if review.completeness.issues:
            for issue in review.completeness.issues:
                checkbox = "[ ]" if issue.requires_user_confirmation else "[-]"
                lines.append(
                    f"- {checkbox} `{issue.issue_id}` "
                    f"（{issue.severity}）：{issue.message}"
                )
        else:
            lines.append("没有待处理事项，可以进行最终确认。")

        lines.extend(
            [
                "",
                "## 如何确认",
                "",
                "在确认答案 JSON 的 `field_updates` 中补充或修正字段；",
                "对无需修改但已经知悉的问题，将其编号加入 "
                "`acknowledged_issue_ids`。确认命令会再次执行 Pydantic "
                "和完整性检查。",
                "",
            ]
        )
        return "\n".join(lines)

    @classmethod
    def _row(cls, field: str, value: Any) -> str:
        return f"| {cls._escape(field)} | {cls._format(value)} |"

    @classmethod
    def _parser_row(
        cls,
        parser_name: str,
        status: str,
        error: str | None,
    ) -> str:
        return (
            f"| {cls._escape(parser_name)} | {cls._escape(status)} | "
            f"{cls._format(error)} |"
        )

    @classmethod
    def _format(cls, value: Any) -> str:
        if value is None:
            return "`null`"
        rendered = json.dumps(value, ensure_ascii=False, sort_keys=True)
        return cls._escape(rendered)

    @staticmethod
    def _escape(value: str) -> str:
        return value.replace("|", "\\|").replace("\n", " ")

    @classmethod
    def _flatten(
        cls,
        value: dict[str, Any],
        prefix: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, child in value.items():
            path = (*prefix, key)
            if isinstance(child, dict):
                result.update(cls._flatten(child, path))
            else:
                result[".".join(path)] = child
        return result
