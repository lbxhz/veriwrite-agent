"""Compile V0.1 requirements into deterministic downstream policy."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from math import ceil

from veriwrite_agent.models.executable_policy import (
    ExecutableLengthPolicy,
    ExecutableReferencePolicy,
    ExecutableRequirementPolicy,
    ExecutableStructurePolicy,
    PolicyCoverageItem,
)
from veriwrite_agent.models.literature_discovery import LiteratureCandidate
from veriwrite_agent.models.requirement_workflow import ConfirmedRequirementSpec


class RequirementPolicyCompilationError(ValueError):
    """Raised when a confirmed requirement still cannot become executable."""


class RequirementPolicyCompiler:
    """Turn descriptive V0.1 fields into one versioned runtime contract."""

    def __init__(self, *, current_year: int | None = None) -> None:
        self._current_year = current_year or datetime.now().year

    def compile(
        self,
        confirmed: ConfirmedRequirementSpec,
    ) -> ExecutableRequirementPolicy:
        requirement = confirmed.requirement
        if not requirement.topic:
            raise RequirementPolicyCompilationError(
                "confirmed requirement has no executable research topic"
            )

        counting_policy = requirement.length.counting_policy
        unresolved = list(requirement.ambiguities)
        if counting_policy == "pending_confirmation":
            if requirement.length.minimum_chars or requirement.length.target_chars:
                counting_policy = "chinese_chars_and_english_words"
                unresolved.append("length.counting_policy was inferred from character-based limits")
            else:
                counting_policy = "words"
                unresolved.append("length.counting_policy was inferred from word-based limits")

        if counting_policy == "chinese_chars_and_english_words":
            minimum_units = requirement.length.minimum_chars
            target_units = (
                requirement.length.target_chars
                or requirement.length.minimum_chars
                or requirement.length.target_words
                or requirement.length.maximum_words
                or requirement.length.minimum_words
                or 4000
            )
            maximum_units = None
        else:
            minimum_units = requirement.length.minimum_words
            target_units = (
                requirement.length.target_words
                or requirement.length.maximum_words
                or requirement.length.minimum_words
                or requirement.length.target_chars
                or requirement.length.minimum_chars
                or 4000
            )
            maximum_units = requirement.length.maximum_words

        references = requirement.references
        minimum_total = references.minimum_total or 1
        target_total = max(
            references.target_total or references.minimum_total or 50,
            minimum_total,
        )
        target_origin = (
            "explicit_target"
            if references.target_total is not None
            else (
                "minimum_only"
                if references.minimum_total is not None
                else "system_default"
            )
        )
        minimum_foreign_count = (
            ceil(target_total * references.minimum_foreign_ratio)
            if references.minimum_foreign_ratio is not None
            else None
        )
        year_from = None
        year_to = None
        if (
            references.recent_year_window is not None
            and references.recent_year_rule_strength == "hard"
        ):
            year_from = self._current_year - references.recent_year_window + 1
            year_to = self._current_year

        if requirement.output_language == "pending_confirmation":
            unresolved.append("output_language remains pending_confirmation")
        if references.bibliography_style == "pending_confirmation":
            unresolved.append("references.bibliography_style remains pending_confirmation")
        for index, rule in enumerate(references.restriction_rules):
            normalized_rule = " ".join(rule.description.split())
            if (
                rule.severity == "hard"
                and ("年发文量" in normalized_rule or "annual publication" in normalized_rule.casefold())
            ):
                unresolved.append(
                    "hard_rule_not_fully_executable:"
                    f"references.restriction_rules.{index}:"
                    "annual journal publication volume is unavailable in current metadata"
                )

        fingerprint = hashlib.sha256(
            json.dumps(
                requirement.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return ExecutableRequirementPolicy(
            requirement_fingerprint=fingerprint,
            confirmed_by=confirmed.confirmed_by,
            document_type=requirement.document_type,
            institution=requirement.institution,
            school_or_department=requirement.school_or_department,
            course_name=requirement.course_name,
            output_language=requirement.output_language,
            topic=requirement.topic,
            required_theme_elements=requirement.required_theme_elements,
            deliverables=requirement.deliverables,
            length=ExecutableLengthPolicy(
                counting_policy=counting_policy,
                minimum_units=minimum_units,
                target_units=target_units,
                maximum_units=maximum_units,
                figures_excluded=requirement.length.figures_excluded,
                excluded_components=requirement.length.excluded_components,
            ),
            references=ExecutableReferencePolicy(
                minimum_total=minimum_total,
                target_total=target_total,
                target_origin=target_origin,
                target_is_approximate=(
                    references.target_is_approximate
                    or target_origin == "system_default"
                ),
                minimum_foreign_ratio=references.minimum_foreign_ratio,
                minimum_foreign_count=minimum_foreign_count,
                recent_year_window=references.recent_year_window,
                recent_year_rule_strength=references.recent_year_rule_strength,
                year_from=year_from,
                year_to=year_to,
                preferred_source_types=references.preferred_source_types,
                discouraged_source_types=references.discouraged_source_types,
                citation_order=references.citation_order,
                in_text_style=references.in_text_style,
                max_references_per_citation_cluster=(
                    references.max_references_per_citation_cluster
                ),
                bibliography_style=references.bibliography_style,
                style_examples=references.style_examples,
                required_management_tools=references.required_management_tools,
                source_restriction_rules=references.restriction_rules,
                all_bibliography_items_must_be_cited_and_discussed=(
                    references.all_bibliography_items_must_be_cited_and_discussed
                ),
            ),
            structure=ExecutableStructurePolicy(
                required_sections=(requirement.structure.required_or_recommended_sections),
                must_include_original_analysis=(
                    requirement.structure.must_include_original_analysis
                ),
                must_not_list_titles_or_abstracts_only=(
                    requirement.structure.must_not_list_titles_or_abstracts_only
                ),
            ),
            formatting=requirement.formatting,
            workflow_conditions=requirement.workflow_conditions,
            policy_rules=requirement.policy_rules,
            selection_policy=requirement.selection_policy,
            submission=requirement.submission,
            ai_usage=requirement.ai_policy,
            acknowledged_issue_ids=confirmed.acknowledged_issue_ids,
            remaining_warning_ids=[issue.issue_id for issue in confirmed.remaining_warnings],
            unresolved_requirements=list(dict.fromkeys(unresolved)),
            coverage=_coverage_map(),
        )


def candidate_source_restriction_reasons(
    policy: ExecutableRequirementPolicy,
    candidate: LiteratureCandidate,
) -> list[str]:
    """Match hard source restrictions against authoritative candidate metadata."""

    return source_restriction_reasons(
        policy,
        publisher=candidate.publisher,
        journal=candidate.journal_title,
        source_type=candidate.source_type,
    )


def source_restriction_reasons(
    policy: ExecutableRequirementPolicy,
    *,
    publisher: str | None,
    journal: str | None,
    source_type: str,
) -> list[str]:
    """Evaluate the same source policy again at final-delivery time."""

    candidate_values = [publisher, journal, source_type]
    normalized_values = [_normalize_source_name(value) for value in candidate_values if value]
    reasons: list[str] = []
    for index, rule in enumerate(policy.references.source_restriction_rules):
        if rule.severity != "hard":
            continue
        rule_text = _normalize_source_name(rule.description)
        exact_match = any(len(value) >= 4 and value in rule_text for value in normalized_values)
        family_match = any(
            family in rule_text
            and any(value.startswith(family) or family in value for value in normalized_values)
            for family in ("mdpi", "frontiersin", "ieeeaccess")
        )
        if exact_match or family_match:
            reasons.append(f"source_restriction_rule_{index}")
    return reasons


def ai_generation_prohibitions(
    policy: ExecutableRequirementPolicy,
) -> list[str]:
    """Return confirmed rules that forbid AI-authored prose."""

    statements = list(policy.ai_usage.prohibited_uses)
    statements.extend(
        rule.description
        for rule in policy.policy_rules
        if rule.category in {"ai_usage", "academic_integrity"}
    )
    pattern = re.compile(
        r"(?:AI|artificial intelligence|人工智能).{0,50}"
        r"(?:generate|draft|write|compose|生成|撰写|写作|代写|句子|段落|正文|洗稿)"
        r"|(?:禁止|不允许).{0,50}(?:AI|人工智能).{0,50}"
        r"(?:生成|撰写|写作|代写|句子|段落|正文|洗稿)",
        re.IGNORECASE,
    )
    return [
        statement.strip()
        for statement in statements
        if statement.strip() and pattern.search(statement)
    ]


def is_foreign_literature(*, language: str | None, title: str) -> bool:
    """Use provider language first, then a transparent title-script fallback."""

    if language:
        normalized = language.strip().casefold().replace("_", "-")
        if normalized.startswith("zh"):
            return False
        return True
    chinese_count = len(re.findall(r"[\u4e00-\u9fff]", title))
    latin_count = len(re.findall(r"[A-Za-z]", title))
    return latin_count > 0 and latin_count >= chinese_count * 2


def _normalize_source_name(value: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", value.casefold())


def _coverage_map() -> list[PolicyCoverageItem]:
    return [
        PolicyCoverageItem(
            requirement_path="topic|required_theme_elements",
            enforcement="enforced",
            consumers=["V0.2 blueprint", "V0.3 page retrieval", "V0.4 outline"],
            note="Controls search themes, retrieval context, and section purposes.",
        ),
        PolicyCoverageItem(
            requirement_path="length",
            enforcement="enforced",
            consumers=["V0.4 word budgets", "final delivery audit"],
            note="One counting policy and one set of numeric bounds are used end to end.",
        ),
        PolicyCoverageItem(
            requirement_path="references",
            enforcement="enforced",
            consumers=["V0.2 filtering/selection", "final bibliography/audit"],
            note="Counts, years, language ratio, restrictions, and citation rules are retained.",
        ),
        PolicyCoverageItem(
            requirement_path="structure|deliverables",
            enforcement="enforced",
            consumers=["V0.4 outline", "final paper assembler"],
            note="Required sections and deliverable components are checked before release.",
        ),
        PolicyCoverageItem(
            requirement_path="formatting",
            enforcement="enforced",
            consumers=["DOCX exporter"],
            note="Paper size, font, font size, and line spacing become document styles.",
        ),
        PolicyCoverageItem(
            requirement_path="ai_policy|policy_rules",
            enforcement="enforced",
            consumers=["V0.4 provider gate", "final AI declaration audit"],
            note="Prohibited generation stops before an LLM call; declaration duties remain visible.",
        ),
        PolicyCoverageItem(
            requirement_path="selection_policy|workflow_conditions|submission",
            enforcement="user_gate",
            consumers=["UI confirmations", "final delivery manifest"],
            note="Administrative and submission requirements remain explicit user-facing gates.",
        ),
        PolicyCoverageItem(
            requirement_path="institution|course_name|output_language",
            enforcement="audited",
            consumers=["final document metadata", "final delivery audit"],
            note="Administrative metadata and language are retained in the final package.",
        ),
    ]
