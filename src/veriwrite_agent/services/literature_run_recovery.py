"""Recover a lost UI session from durable V0.2 runtime artifacts."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from veriwrite_agent.models.executable_policy import ExecutableRequirementPolicy
from veriwrite_agent.models.literature_selection import (
    BalancedLiteratureSelection,
    ConfirmedLiteratureSearchBlueprint,
)
from veriwrite_agent.models.literature_verification import LiteratureVerificationBatch
from veriwrite_agent.models.requirement_workflow import ConfirmedRequirementSpec
from veriwrite_agent.models.requirements import (
    LengthRequirement,
    ReferenceRequirement,
    RequirementSpec,
    StructureRequirement,
)


@dataclass(frozen=True)
class RecoverableLiteratureRun:
    """Validated V0.2 artifacts plus a downstream-safe requirement context."""

    run_id: str
    run_dir: Path
    modified_at: float
    confirmed_requirement: ConfirmedRequirementSpec
    executable_policy: ExecutableRequirementPolicy
    confirmed_blueprint_json: str
    result_json: str
    verification_json: str
    ris_text: str
    selected_count: int
    target_total: int
    pool_multiplier: int

    def session_state(self) -> dict[str, str | int | bool]:
        blueprint = ConfirmedLiteratureSearchBlueprint.model_validate_json(
            self.confirmed_blueprint_json
        ).blueprint
        serialized_blueprint = blueprint.model_dump_json(indent=2)
        return {
            "confirmed_json": self.confirmed_requirement.model_dump_json(indent=2),
            "recovered_executable_policy_json": self.executable_policy.model_dump_json(indent=2),
            "requirement_recovered_from_executable_policy": True,
            "literature_blueprint_json": serialized_blueprint,
            "literature_blueprint_editor": serialized_blueprint,
            "literature_confirmed_blueprint_json": self.confirmed_blueprint_json,
            "literature_result_json": self.result_json,
            "literature_ris": self.ris_text,
            "literature_verification_json": self.verification_json,
            "literature_run_dir": str(self.run_dir),
            "literature_pool_multiplier": self.pool_multiplier,
        }


class LiteratureRunRecoveryService:
    """Find the newest complete, internally consistent V0.2 run."""

    _MAX_RECOVERY_CANDIDATES = 12

    def latest(self, cache_root: Path) -> RecoverableLiteratureRun | None:
        if not cache_root.is_dir():
            return None
        candidates = sorted(
            (
                directory
                for directory in cache_root.iterdir()
                if directory.is_dir() and (directory / "final_result.json").is_file()
            ),
            key=lambda directory: (directory / "final_result.json").stat().st_mtime,
            reverse=True,
        )
        # Runtime may contain many historical search attempts. Validating every
        # multi-megabyte result on each blank Streamlit render makes the workbench
        # appear hung and can take longer than the UI timeout. Disaster recovery is
        # intentionally recent-only; older projects should come from their autosave.
        for directory in candidates[: self._MAX_RECOVERY_CANDIDATES]:
            try:
                return self._load(directory)
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                continue
        return None

    def _load(self, run_dir: Path) -> RecoverableLiteratureRun:
        required = {
            "confirmed": run_dir / "confirmed_blueprint.json",
            "result": run_dir / "final_result.json",
            "verification": run_dir / "verification_cache.json",
            "ris": run_dir / "selected.ris",
        }
        if any(not path.is_file() for path in required.values()):
            raise ValueError("V0.2 run is incomplete")

        confirmed_json = required["confirmed"].read_text(encoding="utf-8")
        result_json = required["result"].read_text(encoding="utf-8")
        verification_json = required["verification"].read_text(encoding="utf-8")
        confirmed = ConfirmedLiteratureSearchBlueprint.model_validate_json(confirmed_json)
        payload = json.loads(result_json)
        selection = BalancedLiteratureSelection.model_validate(payload["selection"])
        LiteratureVerificationBatch.model_validate_json(verification_json)
        if selection.blueprint != confirmed.blueprint:
            raise ValueError("V0.2 result does not match its confirmed blueprint")
        policy = confirmed.blueprint.requirement_policy
        if policy is None:
            raise ValueError("V0.2 run has no executable V0.1 policy")

        pool_multiplier = 2
        diagnostics = payload.get("discovery", [])
        target_by_theme = {
            theme.theme_id: theme.target_count for theme in confirmed.blueprint.themes
        }
        observed_ratios = []
        for diagnostic in diagnostics:
            theme_id = diagnostic.get("theme_id")
            target = target_by_theme.get(theme_id)
            search_plan = diagnostic.get("search_plan")
            if target and isinstance(search_plan, dict):
                eligible_target = search_plan.get("target_eligible_count")
                if isinstance(eligible_target, int):
                    observed_ratios.append(math.ceil(eligible_target / target))
        if observed_ratios:
            pool_multiplier = max(1, min(10, max(observed_ratios)))

        return RecoverableLiteratureRun(
            run_id=str(payload.get("run_id") or run_dir.name),
            run_dir=run_dir,
            modified_at=required["result"].stat().st_mtime,
            confirmed_requirement=_requirement_context_from_policy(policy),
            executable_policy=policy,
            confirmed_blueprint_json=confirmed_json,
            result_json=result_json,
            verification_json=verification_json,
            ris_text=required["ris"].read_text(encoding="utf-8"),
            selected_count=len(selection.selected),
            target_total=selection.blueprint.target_total,
            pool_multiplier=pool_multiplier,
        )


def _requirement_context_from_policy(
    policy: ExecutableRequirementPolicy,
) -> ConfirmedRequirementSpec:
    """Rebuild display metadata while keeping the original policy as authority."""

    if policy.length.counting_policy == "words":
        length = LengthRequirement(
            minimum_words=policy.length.minimum_units,
            target_words=policy.length.target_units,
            maximum_words=policy.length.maximum_units,
            figures_excluded=policy.length.figures_excluded,
            excluded_components=policy.length.excluded_components,
            counting_policy="words",
        )
    else:
        length = LengthRequirement(
            minimum_chars=policy.length.minimum_units,
            target_chars=policy.length.target_units,
            figures_excluded=policy.length.figures_excluded,
            excluded_components=policy.length.excluded_components,
            counting_policy="chinese_chars_and_english_words",
        )
    references = ReferenceRequirement(
        minimum_total=policy.references.minimum_total,
        target_total=(
            policy.references.target_total
            if policy.references.target_origin == "explicit_target"
            else None
        ),
        target_is_approximate=policy.references.target_is_approximate,
        minimum_foreign_ratio=policy.references.minimum_foreign_ratio,
        recent_year_window=policy.references.recent_year_window,
        recent_year_rule_strength=policy.references.recent_year_rule_strength,
        preferred_source_types=policy.references.preferred_source_types,
        discouraged_source_types=policy.references.discouraged_source_types,
        citation_order=policy.references.citation_order,
        in_text_style=policy.references.in_text_style,
        max_references_per_citation_cluster=(
            policy.references.max_references_per_citation_cluster
        ),
        bibliography_style=policy.references.bibliography_style,
        style_examples=policy.references.style_examples,
        required_management_tools=policy.references.required_management_tools,
        restriction_rules=policy.references.source_restriction_rules,
        all_bibliography_items_must_be_cited_and_discussed=(
            policy.references.all_bibliography_items_must_be_cited_and_discussed
        ),
    )
    requirement = RequirementSpec(
        document_type=policy.document_type,
        institution=policy.institution,
        school_or_department=policy.school_or_department,
        course_name=policy.course_name,
        output_language=policy.output_language,
        topic=policy.topic,
        topic_source="explicit",
        required_theme_elements=policy.required_theme_elements,
        deliverables=policy.deliverables,
        length=length,
        structure=StructureRequirement(
            required_or_recommended_sections=policy.structure.required_sections,
            must_include_original_analysis=policy.structure.must_include_original_analysis,
            must_not_list_titles_or_abstracts_only=(
                policy.structure.must_not_list_titles_or_abstracts_only
            ),
        ),
        references=references,
        formatting=policy.formatting,
        workflow_conditions=policy.workflow_conditions,
        policy_rules=policy.policy_rules,
        selection_policy=policy.selection_policy,
        submission=policy.submission,
        ai_policy=policy.ai_usage,
        ambiguities=policy.unresolved_requirements,
    )
    return ConfirmedRequirementSpec(
        confirmed_by=policy.confirmed_by,
        requirement=requirement,
        acknowledged_issue_ids=policy.acknowledged_issue_ids,
    )
