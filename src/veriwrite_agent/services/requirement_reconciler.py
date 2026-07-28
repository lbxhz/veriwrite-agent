"""Deterministically reconcile rule-based and LLM requirement candidates."""

from __future__ import annotations

from copy import deepcopy
from math import isclose
from typing import Any, Iterator

from pydantic import ValidationError

from veriwrite_agent.models.requirement_workflow import (
    ReconciliationResult,
    RequirementConflict,
)
from veriwrite_agent.models.requirements import RequirementSpec


class RequirementReconciler:
    """Merge safe values and expose disagreements instead of guessing."""

    _union_fields = {"ambiguities", "source_evidence"}
    _set_like_fields = {
        "deliverables",
        "required_theme_elements",
        "references.preferred_source_types",
        "references.discouraged_source_types",
    }

    def reconcile(
        self,
        rule_spec: RequirementSpec,
        llm_spec: RequirementSpec,
    ) -> ReconciliationResult:
        merged = rule_spec.model_dump(mode="json")
        llm_values = llm_spec.model_dump(mode="json")
        conflicts: list[RequirementConflict] = []

        for path, llm_value in self._leaf_values(llm_values):
            rule_value = self._get_path(merged, path)
            field = ".".join(path)

            if field in self._union_fields:
                combined = self._stable_union(rule_value, llm_value)
                self._set_path(merged, path, combined)
                continue

            if self._equivalent(field, rule_value, llm_value) or self._is_empty(
                llm_value
            ):
                continue

            if self._is_empty(rule_value):
                candidate = deepcopy(merged)
                self._set_path(candidate, path, llm_value)
                try:
                    RequirementSpec.model_validate(candidate)
                except ValidationError:
                    conflicts.append(
                        RequirementConflict(
                            field=field,
                            rule_value=rule_value,
                            llm_value=llm_value,
                            provisional_value=rule_value,
                            reason=(
                                "The LLM value could not be merged without "
                                "violating the RequirementSpec data contract."
                            ),
                        )
                    )
                else:
                    merged = candidate
                continue

            conflicts.append(
                RequirementConflict(
                    field=field,
                    rule_value=rule_value,
                    llm_value=llm_value,
                    provisional_value=rule_value,
                    reason=(
                        "Both parsers produced non-empty values. The rule-based "
                        "value is retained provisionally until the user decides."
                    ),
                )
            )

        return ReconciliationResult(
            merged_spec=RequirementSpec.model_validate(merged),
            conflicts=conflicts,
        )

    @classmethod
    def _leaf_values(
        cls,
        value: dict[str, Any],
        prefix: tuple[str, ...] = (),
    ) -> Iterator[tuple[tuple[str, ...], Any]]:
        for key, child in value.items():
            path = (*prefix, key)
            if isinstance(child, dict):
                yield from cls._leaf_values(child, path)
            else:
                yield path, child

    @staticmethod
    def _get_path(value: dict[str, Any], path: tuple[str, ...]) -> Any:
        current: Any = value
        for part in path:
            current = current[part]
        return current

    @staticmethod
    def _set_path(
        value: dict[str, Any],
        path: tuple[str, ...],
        replacement: Any,
    ) -> None:
        current = value
        for part in path[:-1]:
            current = current[part]
        current[path[-1]] = deepcopy(replacement)

    @staticmethod
    def _is_empty(value: Any) -> bool:
        return value is None or value == "" or value == [] or value == {}

    @classmethod
    def _equivalent(cls, field: str, left: Any, right: Any) -> bool:
        if left == right:
            return True
        if (
            field in cls._set_like_fields
            and isinstance(left, list)
            and isinstance(right, list)
            and all(isinstance(item, str) for item in [*left, *right])
        ):
            return set(left) == set(right)
        if (
            isinstance(left, (int, float))
            and not isinstance(left, bool)
            and isinstance(right, (int, float))
            and not isinstance(right, bool)
        ):
            return isclose(left, right, rel_tol=1e-3, abs_tol=1e-6)
        return False

    @staticmethod
    def _stable_union(left: Any, right: Any) -> list[Any]:
        result: list[Any] = []
        for item in [*(left or []), *(right or [])]:
            if item not in result:
                result.append(item)
        return result
