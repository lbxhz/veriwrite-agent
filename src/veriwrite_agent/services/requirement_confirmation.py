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
        self._completeness_checker = completeness_checker or RequirementCompletenessChecker()

    def confirm(
        self,
        review: RequirementReviewPackage,
        confirmation: RequirementConfirmation,
    ) -> ConfirmedRequirementSpec:
        spec_data = review.reconciliation.merged_spec.model_dump(mode="json")

        for field, value in confirmation.field_updates.items():
            self._apply_field_update(spec_data, field, value)
        confirmed_topic = spec_data["topic"] if "topic" in confirmation.field_updates else None
        confirmed_boundary = {
            field.split(".", 1)[1]: value
            for field, value in confirmation.field_updates.items()
            if field.startswith("topic_boundary.")
        }
        if spec_data.get("selected_profile_id"):
            self._materialize_selected_profile(spec_data)
        if confirmed_topic is not None:
            spec_data["topic"] = confirmed_topic
            spec_data["topic_source"] = "explicit"
        spec_data["topic_boundary"].update(confirmed_boundary)

        evidence = list(spec_data["source_evidence"])
        for field, value in confirmation.field_updates.items():
            evidence.append(
                SourceEvidence(
                    field=field,
                    source_text=f"用户确认：{confirmation.confirmed_by}",
                    note=(confirmation.note or f"确认值：{value!r}"),
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
            parser_runs=[run for run in (review.rule_run, review.llm_run) if run is not None],
        )
        blocking = [issue.issue_id for issue in report.issues if issue.severity == "blocking"]
        if blocking:
            raise RequirementConfirmationError("仍有阻塞项未解决：" + ", ".join(blocking))

        acknowledged = set(confirmation.acknowledged_issue_ids)
        known_issue_ids = {
            issue.issue_id for issue in [*review.completeness.issues, *report.issues]
        }
        unknown_acknowledgements = sorted(acknowledged - known_issue_ids)
        if unknown_acknowledgements:
            raise RequirementConfirmationError(
                "确认答案包含未知问题编号：" + ", ".join(unknown_acknowledgements)
            )
        pending_acknowledgements = [
            issue.issue_id
            for issue in report.issues
            if issue.requires_user_confirmation and issue.issue_id not in acknowledged
        ]
        if pending_acknowledgements:
            raise RequirementConfirmationError(
                "以下非阻塞问题需要修改字段或明确确认：" + ", ".join(pending_acknowledgements)
            )

        return ConfirmedRequirementSpec(
            confirmed_by=confirmation.confirmed_by,
            requirement=confirmed_spec,
            acknowledged_issue_ids=sorted(acknowledged),
            remaining_warnings=[issue for issue in report.issues if issue.severity == "warning"],
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
            if isinstance(current, list):
                try:
                    current = current[int(part)]
                except (ValueError, IndexError) as exc:
                    raise RequirementConfirmationError(f"未知字段：{field}") from exc
            elif isinstance(current, dict) and part in current:
                current = current[part]
            else:
                raise RequirementConfirmationError(f"未知字段：{field}")
        if isinstance(current, list):
            try:
                current[int(path[-1])] = value
            except (ValueError, IndexError) as exc:
                raise RequirementConfirmationError(f"未知字段：{field}") from exc
        elif isinstance(current, dict) and path[-1] in current:
            current[path[-1]] = value
        else:
            raise RequirementConfirmationError(f"未知字段：{field}")

        try:
            RequirementSpec.model_validate(candidate)
        except ValidationError as exc:
            raise RequirementConfirmationError(f"字段 {field} 的确认值不合法。") from exc
        spec_data.clear()
        spec_data.update(candidate)

    @staticmethod
    def _materialize_selected_profile(spec_data: dict[str, Any]) -> None:
        """Copy the selected option into the effective top-level hand-off."""

        selected_id = spec_data["selected_profile_id"]
        selected = next(
            (
                profile
                for profile in spec_data.get("profiles", [])
                if profile.get("profile_id") == selected_id
            ),
            None,
        )
        if selected is None:
            return

        if selected.get("topic"):
            spec_data["topic"] = selected["topic"]
            spec_data["topic_source"] = "explicit"
        selected_boundary = selected.get("topic_boundary", {})
        if selected_boundary.get("central_question"):
            spec_data["topic_boundary"] = deepcopy(selected_boundary)
        if selected.get("output_language") != "pending_confirmation":
            spec_data["output_language"] = selected["output_language"]

        for field in (
            "required_theme_elements",
            "deliverables",
            "workflow_conditions",
            "policy_rules",
        ):
            spec_data[field] = RequirementConfirmationService._stable_union(
                spec_data.get(field, []),
                selected.get(field, []),
            )

        global_length = spec_data["length"]
        profile_length = selected["length"]
        for field in (
            "minimum_chars",
            "target_chars",
            "minimum_words",
            "maximum_words",
            "target_words",
        ):
            if global_length.get(field) is None and profile_length.get(field) is not None:
                global_length[field] = profile_length[field]
        global_length["figures_excluded"] = (
            global_length["figures_excluded"] or profile_length["figures_excluded"]
        )
        global_length["excluded_components"] = RequirementConfirmationService._stable_union(
            global_length["excluded_components"],
            profile_length["excluded_components"],
        )
        if (
            global_length["counting_policy"] == "pending_confirmation"
            and profile_length["counting_policy"] != "pending_confirmation"
        ):
            global_length["counting_policy"] = profile_length["counting_policy"]

        global_structure = spec_data["structure"]
        profile_structure = selected["structure"]
        global_structure["required_or_recommended_sections"] = (
            RequirementConfirmationService._stable_union(
                global_structure["required_or_recommended_sections"],
                profile_structure["required_or_recommended_sections"],
            )
        )
        for field in (
            "must_include_original_analysis",
            "must_not_list_titles_or_abstracts_only",
        ):
            global_structure[field] = global_structure[field] or profile_structure[field]

        global_refs = spec_data["references"]
        profile_refs = selected["references"]
        for field in (
            "minimum_total",
            "target_total",
            "minimum_foreign_ratio",
            "recent_year_window",
            "max_references_per_citation_cluster",
        ):
            if global_refs.get(field) is None and profile_refs.get(field) is not None:
                global_refs[field] = profile_refs[field]
        global_refs["target_is_approximate"] = (
            global_refs["target_is_approximate"] or profile_refs["target_is_approximate"]
        )
        for field in (
            "preferred_source_types",
            "discouraged_source_types",
            "style_examples",
            "required_management_tools",
            "restriction_rules",
        ):
            global_refs[field] = RequirementConfirmationService._stable_union(
                global_refs[field],
                profile_refs[field],
            )
        if (
            global_refs["bibliography_style"] == "pending_confirmation"
            and profile_refs["bibliography_style"] != "pending_confirmation"
        ):
            global_refs["bibliography_style"] = profile_refs["bibliography_style"]

    @staticmethod
    def _stable_union(left: list[Any], right: list[Any]) -> list[Any]:
        result: list[Any] = []
        for item in [*left, *right]:
            if item not in result:
                result.append(item)
        return result
