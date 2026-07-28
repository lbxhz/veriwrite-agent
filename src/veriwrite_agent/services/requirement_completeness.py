"""Check whether a provisional requirement is safe to confirm and hand off."""

from __future__ import annotations

from collections.abc import Sequence

from veriwrite_agent.models.requirement_workflow import (
    CompletenessIssue,
    CompletenessReport,
    ParserRun,
    RequirementConflict,
)
from veriwrite_agent.models.requirements import RequirementSpec


class RequirementCompletenessChecker:
    """Turn missing information and conflicts into an explicit checklist."""

    def check(
        self,
        spec: RequirementSpec,
        *,
        conflicts: Sequence[RequirementConflict] = (),
        parser_runs: Sequence[ParserRun] = (),
    ) -> CompletenessReport:
        issues: list[CompletenessIssue] = []

        if not spec.topic:
            issues.append(
                CompletenessIssue(
                    issue_id="missing_topic",
                    field="topic",
                    severity="blocking",
                    message="研究主题尚未确认，无法安全进入文献检索阶段。",
                    requires_user_confirmation=True,
                )
            )

        for conflict in conflicts:
            issues.append(
                CompletenessIssue(
                    issue_id=f"unresolved_conflict:{conflict.field}",
                    field=conflict.field,
                    severity="blocking",
                    message=f"规则解析和 LLM 对字段 {conflict.field} 的结果不一致。",
                    requires_user_confirmation=True,
                )
            )

        if spec.length.minimum_chars is None:
            issues.append(
                CompletenessIssue(
                    issue_id="missing_minimum_chars",
                    field="length.minimum_chars",
                    severity="warning",
                    message="课程文件没有给出明确的最低字数。",
                    requires_user_confirmation=True,
                )
            )

        if spec.references.minimum_total is None:
            issues.append(
                CompletenessIssue(
                    issue_id="missing_minimum_references",
                    field="references.minimum_total",
                    severity="warning",
                    message="课程文件没有给出明确的最低参考文献数量。",
                    requires_user_confirmation=True,
                )
            )

        if spec.references.bibliography_style == "pending_confirmation":
            issues.append(
                CompletenessIssue(
                    issue_id="pending_bibliography_style",
                    field="references.bibliography_style",
                    severity="warning",
                    message="参考文献著录标准尚待用户或教师确认。",
                    requires_user_confirmation=True,
                )
            )

        if spec.length.counting_policy == "pending_confirmation":
            issues.append(
                CompletenessIssue(
                    issue_id="pending_counting_policy",
                    field="length.counting_policy",
                    severity="warning",
                    message="字数统计口径尚待确认。",
                    requires_user_confirmation=True,
                )
            )

        for index, ambiguity in enumerate(spec.ambiguities):
            issues.append(
                CompletenessIssue(
                    issue_id=f"source_ambiguity:{index}",
                    field="ambiguities",
                    severity="warning",
                    message=ambiguity,
                    requires_user_confirmation=True,
                )
            )

        for run in parser_runs:
            if run.status == "failed":
                issues.append(
                    CompletenessIssue(
                        issue_id=f"parser_failed:{run.parser_name}",
                        severity="warning",
                        message=(
                            f"{run.parser_name} 解析失败；系统保留了另一条可用结果。"
                        ),
                        requires_user_confirmation=False,
                    )
                )

        if parser_runs and not any(
            run.parser_name == "llm" for run in parser_runs
        ):
            issues.append(
                CompletenessIssue(
                    issue_id="single_parser_mode",
                    severity="warning",
                    message="本次审查包只使用了规则解析，未执行 LLM 交叉检查。",
                    requires_user_confirmation=False,
                )
            )

        return CompletenessReport(issues=issues)
