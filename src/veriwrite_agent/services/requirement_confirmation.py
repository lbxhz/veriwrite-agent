"""Apply explicit user decisions and produce the confirmed V0.1 hand-off."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from pydantic import ValidationError

from veriwrite_agent.models.requirement_workflow import (
    ConfirmedRequirementSpec,
    RequirementConfirmation,
    RequirementReviewPackage,
)
from veriwrite_agent.models.requirements import RequirementSpec, SourceEvidence
from veriwrite_agent.services.requirement_completeness import (
    RequirementCompletenessChecker,
)


class RequirementConfirmationError(ValueError):
    """Raised when a confirmation leaves blocking or unacknowledged issues."""


class RequirementConfirmationService:
    """Validate user choices instead of treating confirmation as free-form edits."""

    _protected_roots = {"schema_version", "source_evidence", "ambiguities"}

    def __init__(
        self,
        completeness_checker: RequirementCompletenessChecker | None = None,
    ) -> None:
        self._completeness_checker = (
            completeness_checker or RequirementCompletenessChecker()
        )

    def confirm(
        self,
        review: RequirementReviewPackage,
        confirmation: RequirementConfirmation,
    ) -> ConfirmedRequirementSpec:
        spec_data = review.reconciliation.merged_spec.model_dump(mode="json")

        for field, value in confirmation.field_updates.items():
            self._apply_field_update(spec_data, field, value)
        if "topic" in confirmation.field_updates:
            spec_data["topic_source"] = "explicit"

        evidence = list(spec_data["source_evidence"])
        for field, value in confirmation.field_updates.items():
            evidence.append(
                SourceEvidence(
                    field=field,
                    source_text=f"用户确认：{confirmation.confirmed_by}",
                    note=(
                        confirmation.note
                        or f"确认值：{value!r}"
                    ),
                ).model_dump(mode="json")
            )
        spec_data["source_evidence"] = evidence

        try:
            confirmed_spec = RequirementSpec.model_validate(spec_data)
        except ValidationError as exc:
            raise RequirementConfirmationError(
                "用户确认后的数据不符合 RequirementSpec 数据合同。"
            ) from exc

        unresolved_conflicts = [
            conflict
            for conflict in review.reconciliation.conflicts
            if conflict.field not in confirmation.field_updates
        ]
        report = self._completeness_checker.check(
            confirmed_spec,
            conflicts=unresolved_conflicts,
            parser_runs=[
                run
                for run in (review.rule_run, review.llm_run)
                if run is not None
            ],
        )
        blocking = [issue.issue_id for issue in report.issues if issue.severity == "blocking"]
        if blocking:
            raise RequirementConfirmationError(
                "仍有阻塞项未解决：" + ", ".join(blocking)
            )

        acknowledged = set(confirmation.acknowledged_issue_ids)
        known_issue_ids = {
            issue.issue_id
            for issue in [*review.completeness.issues, *report.issues]
        }
        unknown_acknowledgements = sorted(acknowledged - known_issue_ids)
        if unknown_acknowledgements:
            raise RequirementConfirmationError(
                "确认答案包含未知问题编号："
                + ", ".join(unknown_acknowledgements)
            )
        pending_acknowledgements = [
            issue.issue_id
            for issue in report.issues
            if issue.requires_user_confirmation
            and issue.issue_id not in acknowledged
        ]
        if pending_acknowledgements:
            raise RequirementConfirmationError(
                "以下非阻塞问题需要修改字段或明确确认："
                + ", ".join(pending_acknowledgements)
            )

        return ConfirmedRequirementSpec(
            confirmed_by=confirmation.confirmed_by,
            requirement=confirmed_spec,
            acknowledged_issue_ids=sorted(acknowledged),
            remaining_warnings=[
                issue for issue in report.issues if issue.severity == "warning"
            ],
        )

    def _apply_field_update(
        self,
        spec_data: dict[str, Any],
        field: str,
        value: Any,
    ) -> None:
        path = field.split(".")
        if not field or any(not part for part in path):
            raise RequirementConfirmationError(f"无效字段路径：{field!r}")
        if path[0] in self._protected_roots:
            raise RequirementConfirmationError(f"字段不允许直接修改：{field}")

        candidate = deepcopy(spec_data)
        current: Any = candidate
        for part in path[:-1]:
            if not isinstance(current, dict) or part not in current:
                raise RequirementConfirmationError(f"未知字段：{field}")
            current = current[part]
        if not isinstance(current, dict) or path[-1] not in current:
            raise RequirementConfirmationError(f"未知字段：{field}")
        current[path[-1]] = value

        try:
            RequirementSpec.model_validate(candidate)
        except ValidationError as exc:
            raise RequirementConfirmationError(
                f"字段 {field} 的确认值不合法。"
            ) from exc
        spec_data.clear()
        spec_data.update(candidate)
