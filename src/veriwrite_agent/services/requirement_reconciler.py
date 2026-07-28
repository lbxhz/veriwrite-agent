"""Deterministically reconcile rule-based and LLM requirement candidates."""

from __future__ import annotations

import re
from copy import deepcopy
from difflib import SequenceMatcher
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
        self._align_profiles(merged, llm_values)
        conflicts: list[RequirementConflict] = []

        for path, llm_value in self._leaf_values(llm_values):
            rule_value = self._get_path(merged, path)
            field = ".".join(path)

            if field in self._union_fields:
                combined = self._stable_union(rule_value, llm_value)
                self._set_path(merged, path, combined)
                continue

            if isinstance(rule_value, list) and isinstance(llm_value, list):
                self._set_path(
                    merged,
                    path,
                    self._stable_union(rule_value, llm_value),
                )
                continue

            if self._equivalent(field, rule_value, llm_value) or self._is_empty(
                field,
                llm_value,
            ):
                continue

            if self._is_empty(field, rule_value):
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
            elif key == "profiles" and isinstance(child, list):
                for index, profile in enumerate(child):
                    if isinstance(profile, dict):
                        yield from cls._leaf_values(
                            profile,
                            (*path, str(index)),
                        )
            else:
                yield path, child

    @staticmethod
    def _get_path(value: dict[str, Any], path: tuple[str, ...]) -> Any:
        current: Any = value
        for part in path:
            if isinstance(current, list):
                current = current[int(part)]
            else:
                current = current[part]
        return current

    @staticmethod
    def _set_path(
        value: dict[str, Any],
        path: tuple[str, ...],
        replacement: Any,
    ) -> None:
        current: Any = value
        for part in path[:-1]:
            if isinstance(current, list):
                current = current[int(part)]
            else:
                current = current[part]
        if isinstance(current, list):
            current[int(path[-1])] = deepcopy(replacement)
        else:
            current[path[-1]] = deepcopy(replacement)

    @staticmethod
    def _is_empty(field: str, value: Any) -> bool:
        if value is None or value == "" or value == [] or value == {}:
            return True
        if isinstance(value, str) and value in {
            "pending_confirmation",
            "user_confirmation_required",
            "unspecified",
        }:
            return True
        if isinstance(value, bool) and value is False:
            return True
        return False

    @classmethod
    def _equivalent(cls, field: str, left: Any, right: Any) -> bool:
        if left == right:
            return True
        if (
            re.fullmatch(r"profiles\.\d+\.topic", field)
            and isinstance(left, str)
            and isinstance(right, str)
        ):
            return cls._topic_similarity(left, right) >= 0.3
        if (
            (field.endswith("consequence") or field.endswith("violation_consequence"))
            and isinstance(left, str)
            and isinstance(right, str)
        ):
            left_outcome = cls._outcome_signature(left)
            right_outcome = cls._outcome_signature(right)
            if left_outcome and left_outcome == right_outcome:
                return True
        if (
            field.endswith("bibliography_style")
            and isinstance(left, str)
            and isinstance(right, str)
        ):

            def normalize_style(value: str) -> str:
                return re.sub(
                    r"[\s，,。]+|期刊格式",
                    "",
                    value.casefold(),
                )

            left_style = normalize_style(left)
            right_style = normalize_style(right)
            if left_style in right_style or right_style in left_style:
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
    def _topic_similarity(left: str, right: str) -> float:
        anchors = (
            "AI",
            "大地测量",
            "卫星导航",
            "定位",
            "卫星遥感",
            "温室气体",
            "SAR",
            "InSAR",
            "遥感",
            "GeoAI",
            "GIS",
            "理论",
            "方法",
            "应用",
        )
        left_anchors = {
            anchor.casefold() for anchor in anchors if anchor.casefold() in left.casefold()
        }
        right_anchors = {
            anchor.casefold() for anchor in anchors if anchor.casefold() in right.casefold()
        }
        if len(left_anchors & right_anchors) >= 2:
            return 1.0

        def normalize(value: str) -> str:
            return re.sub(
                r"[\W_]+",
                "",
                value.casefold()
                .replace("介绍", "")
                .replace("并进行总结", "")
                .replace("当前存在的不足和问题", ""),
            )

        return SequenceMatcher(
            None,
            normalize(left),
            normalize(right),
            autojunk=False,
        ).ratio()

    @staticmethod
    def _outcome_signature(value: str) -> str | None:
        normalized = re.sub(r"\s+", "", value)
        score = re.search(r"(\d{1,3})分", normalized)
        if score:
            return f"score:{score.group(1)}"
        if "不及格" in normalized:
            return "fail"
        if "重做" in normalized:
            return "redo"
        return None

    @staticmethod
    def _stable_union(left: Any, right: Any) -> list[Any]:
        result: list[Any] = []
        for item in [*(left or []), *(right or [])]:
            if item not in result:
                result.append(item)
        return result

    @staticmethod
    def _align_profiles(
        merged: dict[str, Any],
        llm_values: dict[str, Any],
    ) -> None:
        """Align selectable profiles by id or teacher before field merging."""

        rule_profiles = merged.get("profiles", [])
        llm_profiles = llm_values.get("profiles", [])
        if not rule_profiles and llm_profiles:
            merged["profiles"] = deepcopy(llm_profiles)
            return
        if not rule_profiles or not llm_profiles:
            return

        unused = list(llm_profiles)
        aligned: list[dict[str, Any]] = []
        for rule_profile in rule_profiles:
            match = next(
                (
                    profile
                    for profile in unused
                    if profile.get("profile_id") == rule_profile.get("profile_id")
                    or (
                        profile.get("teacher")
                        and profile.get("teacher") == rule_profile.get("teacher")
                    )
                ),
                None,
            )
            if match is None:
                aligned.append(deepcopy(rule_profile))
                continue
            aligned_match = deepcopy(match)
            aligned_match["profile_id"] = rule_profile.get("profile_id")
            aligned.append(aligned_match)
            unused.remove(match)
        for profile in unused:
            merged["profiles"].append(deepcopy(profile))
            aligned.append(profile)
        llm_values["profiles"] = aligned
