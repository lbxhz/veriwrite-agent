"""Streamlit workbench for V0.4 evidence-constrained section writing."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, MutableMapping

import streamlit as st

from veriwrite_agent.config.settings import LLMSettings
from veriwrite_agent.llm.deepseek_client import DeepSeekClient
from veriwrite_agent.models.writing import (
    SectionDraftIssue,
    V04WritingProject,
    WritingSectionState,
)
from veriwrite_agent.models.writing_plan import GroundedWritingPlan, WritingSectionPlan
from veriwrite_agent.models.final_delivery import (
    FinalMatterProposal,
    FinalPaperAuditIssue,
    FinalPaperPackage,
)
from veriwrite_agent.models.writing_handoff import V04WritingHandoff
from veriwrite_agent.services.grounded_writing import (
    SectionEvidencePacketBuilder,
    WritingProjectService,
    count_writing_units,
)
from veriwrite_agent.services.writing_planning import (
    GroundedWritingPlanner,
    LLMGroundedParagraphWriter,
    ParagraphWritingRuntimeCache,
    PlannedSectionDraftService,
    WritingPlanRuntimeCache,
    align_writing_plan_language,
    repair_writing_plan_source_coverage,
)
from veriwrite_agent.services.writing_quality import (
    LLMSectionQualityReviewer,
    apply_section_quality_review,
    language_mismatch_detail,
)
from veriwrite_agent.services.final_delivery import (
    FinalPaperAssembler,
    FinalPaperDocxExporter,
    LLMFinalMatterWriter,
)
from veriwrite_agent.ui.workbench import project_root

WRITING_PLAN_KEY = "v04_writing_plan_json"
V04_PROJECT_KEY = "v04_writing_project_json"
FINAL_MATTER_KEY = "mvp_final_matter_json"
FINAL_PACKAGE_KEY = "mvp_final_paper_json"
FINAL_REPAIR_CHECKPOINT_KEY = "mvp_final_repair_checkpoint_json"
FINAL_SEMANTIC_ATTESTATION_KEY = "mvp_final_semantic_review_attestation"
FINAL_REPAIR_AUTO_SUPPRESSION_KEY = "mvp_final_repair_auto_suppressed_id"
SECTION_SELECTION_KEY = "v04_selected_section"
SECTION_SELECTION_REQUEST_KEY = "v04_selected_section_request"
REPAIR_DOWNSTREAM_KEYS = (
    WRITING_PLAN_KEY,
    V04_PROJECT_KEY,
    FINAL_MATTER_KEY,
    FINAL_PACKAGE_KEY,
)
WRITING_REPAIR_CODES = {
    "reference_count_below_minimum",
    "reference_count_below_target",
    "reference_count_above_maximum",
    "uncited_bibliography_item",
    "unknown_citation_key",
}
EDITORIAL_REPAIR_CODES = {
    "paragraph_repetition",
    "topic_drift",
    "coherence_gap",
    "terminology_inconsistent",
    "academic_style_problem",
    "unsupported_claim",
    "overstated_evidence",
}


@dataclass(frozen=True)
class TargetedWritingRepair:
    plan: GroundedWritingPlan
    project: V04WritingProject
    paragraph_numbers: dict[str, tuple[int, ...]]

    @property
    def paragraph_count(self) -> int:
        return sum(len(numbers) for numbers in self.paragraph_numbers.values())


def clear_writing_state() -> None:
    st.session_state.pop(WRITING_PLAN_KEY, None)
    st.session_state.pop(V04_PROJECT_KEY, None)
    st.session_state.pop(FINAL_MATTER_KEY, None)
    st.session_state.pop(FINAL_PACKAGE_KEY, None)
    st.session_state.pop(FINAL_REPAIR_CHECKPOINT_KEY, None)
    st.session_state.pop(FINAL_SEMANTIC_ATTESTATION_KEY, None)
    st.session_state.pop(FINAL_REPAIR_AUTO_SUPPRESSION_KEY, None)
    st.session_state.pop(SECTION_SELECTION_KEY, None)
    st.session_state.pop(SECTION_SELECTION_REQUEST_KEY, None)


def final_delivery_repair_stage(
    package: FinalPaperPackage,
) -> Literal["writing", "delivery"] | None:
    """Route final blockers to the earliest stage that can actually fix them."""

    blocking_codes = {
        issue.code for issue in package.audit.issues if issue.severity == "blocking"
    }
    if not blocking_codes:
        return None
    if any(
        code in WRITING_REPAIR_CODES
        or code.startswith("body_")
        or code.startswith("citation_")
        or code.startswith("length_")
        for code in blocking_codes
    ):
        return "writing"
    return "delivery"


def build_targeted_writing_repair(
    state: MutableMapping[str, Any],
    package: FinalPaperPackage,
) -> TargetedWritingRepair:
    """Convert final audit blockers into paragraph-level V0.4 repair tasks."""

    serialized_plan = state.get(WRITING_PLAN_KEY)
    serialized_project = state.get(V04_PROJECT_KEY)
    if not serialized_plan or not serialized_project:
        raise ValueError("缺少已确认的 V0.4 写作计划或正文检查点，无法定点返修")
    plan = GroundedWritingPlan.model_validate_json(serialized_plan)
    project = V04WritingProject.model_validate_json(serialized_project)
    if plan.status != "confirmed" or project.status != "body_complete":
        raise ValueError("只有已完成的 V0.4 正文才能从最终审计创建定点返修任务")

    blocking = [issue for issue in package.audit.issues if issue.severity == "blocking"]
    coverage_issues = [
        issue
        for issue in blocking
        if issue.code
        in {
            "reference_count_below_minimum",
            "reference_count_below_target",
            "uncited_bibliography_item",
        }
    ]
    target_issues: dict[tuple[str, int], list[FinalPaperAuditIssue]] = {}
    if coverage_issues:
        coverage_repair = repair_writing_plan_source_coverage(project.handoff, plan)
        plan = coverage_repair.plan
        for section_id, numbers in coverage_repair.changed_paragraph_numbers.items():
            for number in numbers:
                target_issues.setdefault((section_id, number), []).extend(
                    coverage_issues
                )

    for issue in blocking:
        if not _is_writing_repair_issue(issue):
            continue
        if issue in coverage_issues and target_issues:
            continue
        mapped = _paragraph_targets_for_final_issue(
            issue,
            plan=plan,
            project=project,
            package=package,
        )
        if not mapped:
            mapped = {_fallback_repair_target(plan)}
        for target in mapped:
            target_issues.setdefault(target, []).append(issue)

    if not target_issues:
        fallback = _fallback_repair_target(plan)
        target_issues[fallback] = blocking
    reopened = _reopen_targeted_paragraphs(project, target_issues)
    ordered_targets = {
        section.section_id: tuple(
            sorted(
                number
                for target_section, number in target_issues
                if target_section == section.section_id
            )
        )
        for section in plan.sections
        if any(target_section == section.section_id for target_section, _ in target_issues)
    }
    return TargetedWritingRepair(
        plan=plan,
        project=reopened,
        paragraph_numbers=ordered_targets,
    )


def _is_writing_repair_issue(issue: FinalPaperAuditIssue) -> bool:
    return (
        issue.code in WRITING_REPAIR_CODES
        or issue.code.startswith("body_")
        or issue.code.startswith("citation_")
        or issue.code.startswith("length_")
    )


def _paragraph_targets_for_final_issue(
    issue: FinalPaperAuditIssue,
    *,
    plan: GroundedWritingPlan,
    project: V04WritingProject,
    package: FinalPaperPackage,
) -> set[tuple[str, int]]:
    if issue.code == "citation_cluster_too_large":
        return {
            (section_id, int(number))
            for section_id, number in re.findall(
                r"([a-z][a-z0-9_]{1,39}):(\d+)=",
                issue.detail,
            )
        }
    if issue.code == "unknown_citation_key":
        unknown_keys = {value.strip() for value in issue.detail.split(",")}
        return {
            (citation.section_id, citation.paragraph_number)
            for section in project.sections
            if section.draft is not None
            for citation in section.draft.citations
            if citation.citation_key in unknown_keys
        }
    if issue.code.startswith("length_"):
        return _length_repair_targets(issue, plan=plan, project=project)
    if issue.code == "body_language_mismatch":
        return {
            (section.section_id, number)
            for section in project.sections
            if section.draft is not None
            for number, paragraph in enumerate(section.draft.paragraphs, 1)
            if language_mismatch_detail(
                paragraph.text,
                output_language=plan.output_language,
            )
        }
    if issue.code in WRITING_REPAIR_CODES:
        cited = {reference.doi for reference in package.references}
        missing = [doi for doi in plan.required_source_dois if doi not in cited]
        return {
            (section.section_id, paragraph.paragraph_number)
            for doi in missing
            for section in plan.sections
            for paragraph in section.paragraphs
            if doi in paragraph.source_dois
        }
    if issue.code.startswith(("body_", "citation_")):
        return {_fallback_repair_target(plan)}
    return set()


def _length_repair_targets(
    issue: FinalPaperAuditIssue,
    *,
    plan: GroundedWritingPlan,
    project: V04WritingProject,
) -> set[tuple[str, int]]:
    plan_sections = {section.section_id: section for section in plan.sections}
    candidates: list[tuple[int, str, int]] = []
    for state in project.sections:
        if state.draft is None:
            continue
        section_plan = plan_sections.get(state.section_id)
        if section_plan is None:
            continue
        for paragraph_plan, paragraph in zip(
            section_plan.paragraphs,
            state.draft.paragraphs,
            strict=False,
        ):
            actual = count_writing_units(
                paragraph.text,
                counting_policy=section_plan.counting_policy,
            )
            delta = paragraph_plan.target_words - actual
            score = delta if issue.code == "length_below_minimum" else -delta
            candidates.append((score, state.section_id, paragraph_plan.paragraph_number))
    if not candidates:
        return set()
    candidates.sort(reverse=True)
    counts = [int(value) for value in re.findall(r"\d+", issue.detail)]
    gap = abs(counts[0] - counts[1]) if len(counts) >= 2 else 1
    selected: set[tuple[str, int]] = set()
    capacity = 0
    for score, section_id, paragraph_number in candidates:
        selected.add((section_id, paragraph_number))
        capacity += max(1, score)
        if capacity >= gap:
            break
    return selected


def _fallback_repair_target(plan: GroundedWritingPlan) -> tuple[str, int]:
    for section in plan.sections:
        synthesis = next(
            (
                paragraph
                for paragraph in reversed(section.paragraphs)
                if paragraph.role == "synthesis"
            ),
            section.paragraphs[-1],
        )
        return section.section_id, synthesis.paragraph_number
    raise ValueError("写作计划没有可返修段落")


def _reopen_targeted_paragraphs(
    project: V04WritingProject,
    target_issues: dict[tuple[str, int], list[FinalPaperAuditIssue]],
) -> V04WritingProject:
    reopened_states: list[WritingSectionState] = []
    for state in project.sections:
        section_targets = {
            number: issues
            for (section_id, number), issues in target_issues.items()
            if section_id == state.section_id
        }
        if not section_targets:
            reopened_states.append(state)
            continue
        if state.draft is None:
            raise ValueError(f"待返修章节缺少原草稿：{state.section_id}")
        retained_issues = [
            issue for issue in state.draft.issues if issue.code != "final_audit_repair"
        ]
        repair_issues: list[SectionDraftIssue] = []
        seen_repair_issues: set[tuple[int, str, str]] = set()
        for number, issues in sorted(section_targets.items()):
            for issue in issues:
                identity = (number, issue.code, issue.detail)
                if identity in seen_repair_issues:
                    continue
                seen_repair_issues.add(identity)
                repair_issues.append(
                    SectionDraftIssue(
                        code="final_audit_repair",
                        severity="blocking",
                        detail=(
                            f"最终审计 {issue.code}：{issue.detail}；"
                            "只重写本段，其他已确认段落保持不变。"
                        ),
                        paragraph_number=number,
                    )
                )
        reopened_draft = state.draft.model_copy(
            update={
                "status": "needs_review",
                "issues": [*retained_issues, *repair_issues],
                "confirmed_by": None,
                "confirmed_at": None,
            }
        )
        reopened_states.append(
            WritingSectionState(
                section_id=state.section_id,
                status="needs_review",
                draft=reopened_draft,
            )
        )
    reopened = project.model_copy(
        update={
            "status": "drafting",
            "sections": reopened_states,
            "updated_at": datetime.now(timezone.utc),
        }
    )
    return V04WritingProject.model_validate(reopened.model_dump(mode="json"))


def upgrade_legacy_full_rebuild_repair(
    state: MutableMapping[str, Any],
) -> bool:
    """Convert the old whole-body rollback checkpoint into targeted repair state."""

    serialized_checkpoint = state.get(FINAL_REPAIR_CHECKPOINT_KEY)
    if not serialized_checkpoint:
        return False
    try:
        checkpoint = json.loads(serialized_checkpoint)
    except (TypeError, json.JSONDecodeError):
        return False
    if checkpoint.get("schema_version") != "mvp-final-repair-1.0":
        return False
    saved_state = checkpoint.get("state")
    if not isinstance(saved_state, dict):
        return False
    try:
        package = FinalPaperPackage.model_validate_json(saved_state[FINAL_PACKAGE_KEY])
        repair = build_targeted_writing_repair(saved_state, package)
    except (KeyError, TypeError, ValueError):
        return False

    current_plan = None
    current_project = None
    try:
        if state.get(WRITING_PLAN_KEY) and state.get(V04_PROJECT_KEY):
            current_plan = GroundedWritingPlan.model_validate_json(
                state[WRITING_PLAN_KEY]
            )
            current_project = V04WritingProject.model_validate_json(
                state[V04_PROJECT_KEY]
            )
    except (TypeError, ValueError):
        current_plan = None
        current_project = None
    repaired_project = repair.project
    if (
        current_plan is not None
        and current_project is not None
        and current_plan.plan_fingerprint == repair.plan.plan_fingerprint
    ):
        repaired_project = _merge_completed_repair_progress(
            repair.project,
            current_project,
            repair.plan,
        )

    state[WRITING_PLAN_KEY] = repair.plan.model_dump_json(indent=2)
    state[V04_PROJECT_KEY] = repaired_project.model_dump_json(indent=2)
    state.pop(FINAL_MATTER_KEY, None)
    state.pop(FINAL_PACKAGE_KEY, None)
    checkpoint["schema_version"] = "mvp-final-repair-1.1"
    checkpoint["migration"] = {
        "mode": "targeted_paragraph_repair",
        "migrated_at": datetime.now(timezone.utc).isoformat(),
        "target_sections": len(repair.paragraph_numbers),
        "target_paragraphs": repair.paragraph_count,
    }
    state[FINAL_REPAIR_CHECKPOINT_KEY] = json.dumps(
        checkpoint,
        ensure_ascii=False,
        indent=2,
    )
    state["mvp_navigation"] = "writing"
    state.pop("mvp_navigation_request", None)
    state[SECTION_SELECTION_REQUEST_KEY] = _next_actionable_section(repaired_project)
    state["mvp_flash"] = (
        "旧版整篇重建任务已升级为定点返修；已经重写或确认的章节继续保留，"
        "其余章节只处理审计定位的段落。"
    )
    return True


def _merge_completed_repair_progress(
    repair_project: V04WritingProject,
    current_project: V04WritingProject,
    plan: GroundedWritingPlan,
) -> V04WritingProject:
    current_states = {state.section_id: state for state in current_project.sections}
    plan_sections = {section.section_id: section for section in plan.sections}
    merged_states: list[WritingSectionState] = []
    for repair_state in repair_project.sections:
        current_state = current_states.get(repair_state.section_id)
        section_plan = plan_sections[repair_state.section_id]
        if (
            current_state is not None
            and current_state.status != "pending"
            and _draft_matches_plan(current_state, section_plan)
        ):
            merged_states.append(current_state)
        else:
            merged_states.append(repair_state)
    merged = repair_project.model_copy(
        update={
            "status": (
                "body_complete"
                if all(state.status == "confirmed" for state in merged_states)
                else "drafting"
            ),
            "sections": merged_states,
            "updated_at": datetime.now(timezone.utc),
        }
    )
    return V04WritingProject.model_validate(merged.model_dump(mode="json"))


def _draft_matches_plan(
    state: WritingSectionState,
    section_plan: WritingSectionPlan,
) -> bool:
    if state.draft is None or len(state.draft.paragraphs) != len(section_plan.paragraphs):
        return False
    return all(
        paragraph.role == planned.role
        and paragraph.evidence_card_ids == planned.evidence_card_ids
        and paragraph.source_dois == planned.source_dois
        for paragraph, planned in zip(
            state.draft.paragraphs,
            section_plan.paragraphs,
            strict=True,
        )
    )


def rollback_blocked_delivery_to_v04(
    state: MutableMapping[str, Any],
) -> bool:
    """Automatically reopen only the V0.4 paragraphs implicated by final audit."""

    serialized = state.get(FINAL_PACKAGE_KEY)
    if not serialized or state.get(FINAL_REPAIR_CHECKPOINT_KEY):
        return False
    try:
        package = FinalPaperPackage.model_validate_json(serialized)
    except Exception:
        return False
    if package.status != "needs_revision" or final_delivery_repair_stage(package) != "writing":
        return False
    repair_id = _final_repair_id(package)
    if state.get(FINAL_REPAIR_AUTO_SUPPRESSION_KEY) == repair_id:
        return False
    blocking_codes = [
        issue.code for issue in package.audit.issues if issue.severity == "blocking"
    ]
    try:
        repair = build_targeted_writing_repair(state, package)
    except (TypeError, ValueError):
        return False
    _create_final_repair_checkpoint(
        blocking_codes,
        state=state,
        package=package,
    )
    state[WRITING_PLAN_KEY] = repair.plan.model_dump_json(indent=2)
    state[V04_PROJECT_KEY] = repair.project.model_dump_json(indent=2)
    state.pop(FINAL_MATTER_KEY, None)
    state.pop(FINAL_PACKAGE_KEY, None)
    state.pop(FINAL_SEMANTIC_ATTESTATION_KEY, None)
    state.pop(FINAL_REPAIR_AUTO_SUPPRESSION_KEY, None)
    state["mvp_navigation"] = "writing"
    state.pop("mvp_navigation_request", None)
    first_section = next(iter(repair.paragraph_numbers))
    state[SECTION_SELECTION_REQUEST_KEY] = first_section
    state["mvp_flash"] = (
        "最终审计已转换为定点返修：保留全部原正文，只重新处理 "
        f"{len(repair.paragraph_numbers)} 个章节中的 {repair.paragraph_count} 个问题段落。"
    )
    return True


def render_grounded_writing_console(
    handoff: V04WritingHandoff,
    *,
    include_final_delivery: bool = True,
) -> None:
    """Render the staged V0.4 body-writing workflow."""

    _render_repair_checkpoint_restore()

    st.divider()
    st.header("V0.4 按证据逐章写作")
    st.caption(
        "DeepSeek只组织段落，不负责创建引用。DOI、引用键、证据卡和PDF页码"
        "由程序绑定；初稿生成后由章节编辑器检查语言、跑题、重复、衔接、"
        "术语和主张强度，只重写被标记的段落；每章确认后才进入正文汇总。"
    )

    writing_plan = _render_writing_plan(handoff)
    if writing_plan is None:
        return

    project = _load_or_start_project(handoff)
    project, language_reopened = _revalidate_project_language(
        project,
        writing_plan,
    )
    if language_reopened:
        _store_project(project)
        st.session_state.pop(FINAL_MATTER_KEY, None)
        st.session_state.pop(FINAL_PACKAGE_KEY, None)
        st.warning(
            f"检测到 {language_reopened} 个正文段落不符合已确认的输出语言；"
            "系统已保留全部原文，仅将这些段落重新打开进行定点修订。"
        )
    confirmed_count = sum(state.status == "confirmed" for state in project.sections)
    columns = st.columns(3)
    columns[0].metric("章节总数", len(project.sections))
    columns[1].metric("已确认章节", confirmed_count)
    columns[2].metric(
        "正文状态",
        "可汇总" if project.status == "body_complete" else "逐章处理中",
    )
    repair_paragraphs = [
        (section.section_id, issue.paragraph_number)
        for section in project.sections
        if section.draft is not None
        for issue in section.draft.issues
        if issue.code == "final_audit_repair" and issue.paragraph_number is not None
    ]
    if repair_paragraphs:
        total_paragraphs = sum(len(section.paragraphs) for section in writing_plan.sections)
        repair_sections = {section_id for section_id, _ in repair_paragraphs}
        st.warning(
            "最终审计定点返修：原正文与写作计划均已保留；"
            f"{total_paragraphs - len(repair_paragraphs)} 个无问题段落不会重写，"
            f"仅需处理 {len(repair_sections)} 个章节中的 "
            f"{len(repair_paragraphs)} 个问题段落。"
        )
    if project.status == "body_complete":
        _render_body_download(project, include_final_delivery=include_final_delivery)
        return

    outline_by_id = {section.section_id: section for section in handoff.outline.outline.sections}
    state_by_id = {state.section_id: state for state in project.sections}
    requested_section = st.session_state.pop(SECTION_SELECTION_REQUEST_KEY, None)
    if requested_section in outline_by_id:
        st.session_state[SECTION_SELECTION_KEY] = requested_section
    if st.session_state.get(SECTION_SELECTION_KEY) not in outline_by_id:
        st.session_state[SECTION_SELECTION_KEY] = _next_actionable_section(project)
    section_id = st.selectbox(
        "选择要处理的章节",
        options=list(outline_by_id),
        format_func=lambda value: (
            f"{outline_by_id[value].title} · {_status_label(state_by_id[value].status)}"
        ),
        key=SECTION_SELECTION_KEY,
    )
    packet = SectionEvidencePacketBuilder().build(handoff, section_id)
    section_plan = next(
        section for section in writing_plan.sections if section.section_id == section_id
    )
    _render_packet(packet)

    if packet.ai_writing_mode == "generation_blocked":
        st.error(
            "确认版课程要求禁止AI生成论文句子或段落，因此系统已在服务层"
            "关闭代写功能。本章仍可导出证据包，供用户自行撰写。"
        )
        for reason in packet.ai_policy_reasons:
            st.markdown(f"- {reason}")
        st.download_button(
            "下载本章人工写作证据包",
            packet.model_dump_json(indent=2),
            file_name=f"{section_id}_manual_writing_evidence.json",
            mime="application/json",
            width="stretch",
        )
        return

    current_state = state_by_id[section_id]
    repairable_issue_numbers = {
        issue.paragraph_number
        for issue in (current_state.draft.issues if current_state.draft else [])
        if issue.paragraph_number is not None
        and (issue.severity == "blocking" or issue.code in EDITORIAL_REPAIR_CODES)
    }
    generate_label = (
        "生成本章草稿"
        if current_state.draft is None
        else (
            "仅重写问题段落"
            if repairable_issue_numbers
            else "重新生成本章草稿"
        )
    )
    if st.button(
        generate_label,
        type="primary",
        width="stretch",
        key=f"v04_generate_{section_id}",
    ):
        try:
            repair_paragraphs = {
                issue.paragraph_number
                for issue in (current_state.draft.issues if current_state.draft else [])
                if issue.paragraph_number is not None
                and (
                    issue.severity == "blocking"
                    or issue.code in EDITORIAL_REPAIR_CODES
                )
            }
            revision_instructions = {
                number: " ".join(
                    issue.detail
                    for issue in (current_state.draft.issues if current_state.draft else [])
                    if issue.paragraph_number == number
                    and (
                        issue.severity == "blocking"
                        or issue.code in EDITORIAL_REPAIR_CODES
                    )
                )
                for number in repair_paragraphs
            }
            with st.spinner(
                "正在按已锁定证据逐段写作；已完成段落会立即保存……"
            ):
                draft = PlannedSectionDraftService().draft(
                    packet,
                    section_plan,
                    LLMGroundedParagraphWriter(
                        DeepSeekClient(LLMSettings().for_structured_output())
                    ),
                    cache=ParagraphWritingRuntimeCache(
                        project_root() / "runtime" / "writing_console",
                        plan_fingerprint=writing_plan.plan_fingerprint,
                    ),
                    existing_draft=current_state.draft,
                    force=(
                        current_state.draft is not None
                        and not repair_paragraphs
                    ),
                    force_paragraph_numbers=repair_paragraphs,
                    revision_instructions=revision_instructions,
                )
                try:
                    quality_review = LLMSectionQualityReviewer(
                        DeepSeekClient(LLMSettings().for_structured_output())
                    ).review(
                        section_plan,
                        draft,
                        packet,
                        output_language=writing_plan.output_language,
                    )
                    draft = apply_section_quality_review(draft, quality_review)
                except Exception as review_exc:
                    retained_issues = [
                        issue
                        for issue in draft.issues
                        if issue.code != "quality_review_failed"
                    ]
                    draft = draft.model_copy(
                        update={
                            "issues": [
                                *retained_issues,
                                SectionDraftIssue(
                                    code="quality_review_failed",
                                    severity="warning",
                                    detail=(
                                        "章节草稿已安全保存，但编辑质量审阅暂未完成："
                                        f"{review_exc}"
                                    ),
                                ),
                            ]
                        }
                    )
                project = WritingProjectService().save_draft(project, draft)
        except Exception as exc:
            st.error(_friendly_writing_error(exc))
            with st.expander("技术详情"):
                st.code(str(exc))
        else:
            _store_project(project)
            st.rerun()

    state = next(item for item in project.sections if item.section_id == section_id)
    if state.draft is None:
        st.info("本章尚未生成草稿。")
    else:
        _render_draft(project, state.draft, section_plan)

def render_final_delivery_console(handoff: V04WritingHandoff) -> None:
    """Render final assembly independently from the V0.4 section workbench."""

    st.divider()
    st.header("最终论文组装与交付")
    project = _load_or_start_project(handoff)
    serialized_plan = st.session_state.get(WRITING_PLAN_KEY)
    if serialized_plan:
        loaded_plan = GroundedWritingPlan.model_validate_json(serialized_plan)
        plan = align_writing_plan_language(
            handoff,
            loaded_plan,
        )
        if plan != loaded_plan:
            st.session_state[WRITING_PLAN_KEY] = plan.model_dump_json(indent=2)
            st.session_state.pop(FINAL_MATTER_KEY, None)
            st.session_state.pop(FINAL_PACKAGE_KEY, None)
            st.rerun()
        project, language_reopened = _revalidate_project_language(project, plan)
        if language_reopened:
            _store_project(project)
            st.session_state.pop(FINAL_MATTER_KEY, None)
            st.session_state.pop(FINAL_PACKAGE_KEY, None)
            first_reopened = next(
                state.section_id
                for state in project.sections
                if state.status == "needs_review"
            )
            st.session_state[SECTION_SELECTION_REQUEST_KEY] = first_reopened
            st.session_state["mvp_navigation_request"] = "writing"
            st.session_state["mvp_flash"] = (
                f"检测到 {language_reopened} 个正文段落不符合已确认的输出语言；"
                "已退回 V0.4 定点修订，其他段落保持不变。"
            )
            st.rerun()
    if project.status != "body_complete":
        st.warning("正文章节尚未全部确认，最终论文组装仍处于锁定状态。")
        return
    body = WritingProjectService().assemble_body(project)
    _render_final_delivery(project, body)


def _render_writing_plan(
    handoff: V04WritingHandoff,
) -> GroundedWritingPlan | None:
    st.subheader("先锁定证据约束写作计划")
    st.caption(
        "系统会结合 V0.2 检索蓝图与 V0.3 实际证据，为每个段落预先分配用途、"
        "字数和允许使用的证据；正文模型不再自行选卡。"
    )
    serialized = st.session_state.get(WRITING_PLAN_KEY)
    if not serialized:
        if st.button(
            "根据实际证据生成写作计划",
            type="primary",
            width="stretch",
            key="v04_generate_writing_plan",
        ):
            force = bool(
                st.session_state.pop("v04_force_writing_plan_regeneration", False)
            )
            try:
                with st.spinner(
                    "正在逐章规划段落与证据；已完成章节会保存为检查点……"
                ):
                    plan = GroundedWritingPlanner(
                        DeepSeekClient(LLMSettings().for_structured_output()),
                        cache=WritingPlanRuntimeCache(
                            project_root() / "runtime" / "writing_plan",
                            handoff=handoff,
                        ),
                        reuse_cache=not force,
                    ).plan(handoff)
            except Exception as exc:
                st.error(
                    "写作计划尚未生成完成。已完成章节已经保存，重试时会继续。"
                )
                with st.expander("技术详情"):
                    st.code(str(exc))
            else:
                st.session_state[WRITING_PLAN_KEY] = plan.model_dump_json(indent=2)
                st.rerun()
        return None

    plan = GroundedWritingPlan.model_validate_json(serialized)
    aligned_plan = align_writing_plan_language(handoff, plan)
    if aligned_plan != plan:
        plan = aligned_plan
        st.session_state[WRITING_PLAN_KEY] = plan.model_dump_json(indent=2)
        st.session_state.pop(FINAL_MATTER_KEY, None)
        st.session_state.pop(FINAL_PACKAGE_KEY, None)
        st.rerun()
    _render_writing_plan_summary(plan)
    if plan.status == "draft":
        existing_project = st.session_state.get(V04_PROJECT_KEY)
        if existing_project:
            st.warning(
                "采用新计划后会重建当前尚未确认的 V0.4 草稿；V0.3 证据库和本地"
                "运行缓存不会删除。"
            )
        left, right = st.columns(2)
        if left.button(
            "采用计划并开始逐段写作",
            type="primary",
            width="stretch",
            key="v04_confirm_writing_plan",
        ):
            confirmed = plan.confirm(
                confirmed_by=handoff.requirement.confirmed_by,
            )
            st.session_state[WRITING_PLAN_KEY] = confirmed.model_dump_json(indent=2)
            st.session_state.pop(V04_PROJECT_KEY, None)
            st.session_state.pop(FINAL_MATTER_KEY, None)
            st.session_state.pop(FINAL_PACKAGE_KEY, None)
            st.session_state.pop(FINAL_SEMANTIC_ATTESTATION_KEY, None)
            st.session_state.pop(FINAL_REPAIR_AUTO_SUPPRESSION_KEY, None)
            st.session_state["mvp_flash"] = (
                "写作计划已锁定，正文将按段落证据包生成。"
            )
            st.rerun()
        if right.button(
            "重新规划",
            width="stretch",
            key="v04_regenerate_writing_plan",
        ):
            st.session_state.pop(WRITING_PLAN_KEY, None)
            st.session_state["v04_force_writing_plan_regeneration"] = True
            st.rerun()
        return None

    st.success(
        f"写作计划已锁定：{len(plan.sections)} 个章节、"
        f"{sum(len(section.paragraphs) for section in plan.sections)} 个段落。"
    )
    return plan


def _render_writing_plan_summary(plan: GroundedWritingPlan) -> None:
    covered_source_dois = {
        doi
        for section in plan.sections
        for paragraph in section.paragraphs
        for doi in paragraph.source_dois
    }
    required_source_dois = set(plan.required_source_dois)
    metrics = st.columns(4)
    metrics[0].metric("章节", len(plan.sections))
    metrics[1].metric(
        "计划段落",
        sum(len(section.paragraphs) for section in plan.sections),
    )
    metrics[2].metric(
        "必引来源覆盖",
        (
            f"{len(required_source_dois & covered_source_dois)}/"
            f"{len(required_source_dois)}"
            if required_source_dois
            else str(len(covered_source_dois))
        ),
    )
    metrics[3].metric(
        "目标字数",
        sum(section.target_words for section in plan.sections),
    )
    with st.expander("查看章节与段落证据分配", expanded=plan.status == "draft"):
        st.dataframe(
            [
                {
                    "章节": section.title,
                    "段落": paragraph.paragraph_number,
                    "角色": paragraph.role,
                    "段落目的": paragraph.purpose,
                    "计划论点": paragraph.claim_focus,
                    "目标字数": paragraph.target_words,
                    "证据卡": len(paragraph.evidence_card_ids),
                    "来源": len(paragraph.source_dois),
                }
                for section in plan.sections
                for paragraph in section.paragraphs
            ],
            hide_index=True,
            width="stretch",
        )


def _load_or_start_project(
    handoff: V04WritingHandoff,
) -> V04WritingProject:
    serialized = st.session_state.get(V04_PROJECT_KEY)
    if serialized:
        project = V04WritingProject.model_validate_json(serialized)
        if project.handoff == handoff:
            return project
    project = WritingProjectService().start(handoff)
    _store_project(project)
    return project


def _revalidate_project_language(
    project: V04WritingProject,
    plan: GroundedWritingPlan,
) -> tuple[V04WritingProject, int]:
    """Reopen only legacy paragraphs that violate the confirmed language."""

    states: list[WritingSectionState] = []
    changed = False
    mismatch_count = 0
    for state in project.sections:
        draft = state.draft
        if draft is None:
            states.append(state)
            continue
        retained = [issue for issue in draft.issues if issue.code != "language_mismatch"]
        language_issues: list[SectionDraftIssue] = []
        for number, paragraph in enumerate(draft.paragraphs, 1):
            detail = language_mismatch_detail(
                paragraph.text,
                output_language=plan.output_language,
            )
            if detail:
                mismatch_count += 1
                language_issues.append(
                    SectionDraftIssue(
                        code="language_mismatch",
                        severity="blocking",
                        paragraph_number=number,
                        detail=detail,
                    )
                )
        issues = [*retained, *language_issues]
        has_blocker = any(issue.severity == "blocking" for issue in issues)
        desired_status = "needs_review" if has_blocker else draft.status
        update: dict[str, Any] = {"issues": issues, "status": desired_status}
        if has_blocker:
            update.update({"confirmed_by": None, "confirmed_at": None})
        revised_draft = draft.model_copy(update=update)
        revised_state = WritingSectionState(
            section_id=state.section_id,
            status=desired_status,
            draft=revised_draft,
        )
        if revised_state != state:
            changed = True
        states.append(revised_state if revised_state != state else state)
    if not changed:
        return project, mismatch_count
    revised = project.model_copy(
        update={
            "status": (
                "body_complete"
                if all(state.status == "confirmed" for state in states)
                else "drafting"
            ),
            "sections": states,
            "updated_at": datetime.now(timezone.utc),
        }
    )
    return V04WritingProject.model_validate(revised.model_dump(mode="json")), mismatch_count


def _store_project(project: V04WritingProject) -> None:
    st.session_state[V04_PROJECT_KEY] = project.model_dump_json(indent=2)


def _friendly_writing_error(exc: Exception) -> str:
    detail = str(exc)
    if "support repair" in detail:
        return (
            "正文已经生成，但证据绑定自动修复后仍未通过审计。"
            "系统没有保存这份草稿，请重新生成本章。"
        )
    if "V0.4 data contract" in detail or "declared source support" in detail:
        return (
            "模型返回的段落结构或证据声明不完整，系统没有保存不合规草稿。"
            "请重新生成本章。"
        )
    return f"本章生成中断：{detail}"


def _render_packet(packet) -> None:
    st.markdown("#### 本章写作依据")
    metrics = st.columns(4)
    metrics[0].metric("目标字数", packet.target_words)
    metrics[1].metric("全文证据卡", len(packet.evidence_items))
    metrics[2].metric(
        "A级核心来源",
        sum(source.evidence_tier == "A_core" for source in packet.sources),
    )
    metrics[3].metric(
        "辅助/背景来源",
        sum(source.evidence_tier != "A_core" for source in packet.sources),
    )
    with st.expander("查看证据明细与导出"):
        if packet.research_questions:
            for question in packet.research_questions:
                st.markdown(f"- {question}")
        st.dataframe(
            [
                {
                    "证据编号": item.evidence_id,
                    "DOI": item.doi,
                    "证据类型": item.evidence_type,
                    "支持强度": item.support_strength,
                    "规范化结论": item.normalized_claim,
                    "页码": ", ".join(str(quote.page_number) for quote in item.supporting_quotes),
                }
                for item in packet.evidence_items
            ],
            hide_index=True,
            width="stretch",
        )
        st.dataframe(
            [
                {
                    "引用键": source.citation_key,
                    "DOI": source.doi,
                    "题名": source.title,
                    "证据等级": source.evidence_tier,
                    "允许用途": source.permitted_use,
                }
                for source in packet.sources
            ],
            hide_index=True,
            width="stretch",
        )
        st.download_button(
            "下载本章证据包",
            packet.model_dump_json(indent=2),
            file_name=f"{packet.section_id}_evidence_packet.json",
            mime="application/json",
            width="stretch",
        )


def _render_draft(
    project: V04WritingProject,
    draft,
    section_plan: WritingSectionPlan,
) -> None:
    st.markdown("#### 本章草稿与审计")
    if draft.status == "needs_review":
        st.error("本章存在阻塞项，不能确认。请修复证据范围或重新生成。")
    elif draft.issues:
        st.warning("草稿可以确认，但仍有非阻塞提醒。")
    else:
        st.success("引用和证据范围检查通过，等待用户确认。")

    if draft.issues:
        st.dataframe(
            [
                {
                    "级别": issue.severity,
                    "问题": issue.code,
                    "段落": issue.paragraph_number or "全章",
                    "说明": issue.detail,
                }
                for issue in draft.issues
            ],
            hide_index=True,
            width="stretch",
        )
    st.markdown(draft.markdown)
    st.caption(
        f"目标 {draft.target_words}；当前统计 {draft.counted_words}。"
        "引用标记由程序生成，不来自模型自由输出。"
    )
    with st.expander("导出本章草稿与引用轨迹"):
        st.download_button(
            "下载草稿审计 JSON",
            json.dumps(
                json.loads(draft.model_dump_json()),
                ensure_ascii=False,
                indent=2,
            ),
            file_name=f"{draft.section_id}_grounded_draft.json",
            mime="application/json",
            width="stretch",
        )

    if draft.status == "confirmed":
        st.success("本章已采用。")
        return
    if st.button(
        "重新运行章节质量审阅",
        width="stretch",
        key=f"v04_quality_review_{draft.section_id}",
    ):
        try:
            with st.spinner("正在只读检查本章论证、语言和证据强度……"):
                packet = SectionEvidencePacketBuilder().build(
                    project.handoff,
                    draft.section_id,
                )
                review = LLMSectionQualityReviewer(
                    DeepSeekClient(LLMSettings().for_structured_output())
                ).review(
                    section_plan,
                    draft,
                    packet,
                    output_language=packet.output_language,
                )
                reviewed = apply_section_quality_review(draft, review)
                updated = WritingProjectService().save_draft(project, reviewed)
        except Exception as exc:
            st.error(f"章节质量审阅未完成：{exc}")
        else:
            _store_project(updated)
            st.rerun()
    st.caption("代码已检查引用键、来源范围和页码；请判断本章论述是否可用。")
    if st.button(
        "采用本章并继续",
        disabled=draft.status == "needs_review",
        type="primary",
        width="stretch",
        key=f"v04_confirm_{draft.section_id}",
    ):
        try:
            updated = WritingProjectService().confirm_section(
                project,
                draft.section_id,
                confirmed_by=project.handoff.requirement.confirmed_by,
            )
        except Exception as exc:
            st.error(f"章节确认失败：{exc}")
        else:
            _store_project(updated)
            next_section = _next_actionable_section(updated)
            st.session_state[SECTION_SELECTION_REQUEST_KEY] = next_section
            st.session_state["mvp_flash"] = (
                "本章已采用，已切换到下一章。"
                if updated.status != "body_complete"
                else "全部正文章节已采用，可以组装最终论文。"
            )
            st.rerun()


def _render_body_download(
    project: V04WritingProject,
    *,
    include_final_delivery: bool = True,
) -> None:
    body = WritingProjectService().assemble_body(project)
    st.divider()
    st.success(
        f"所有正文章节已确认：共 {body.counted_words} 个统计单位，"
        f"使用 {len(body.source_dois)} 篇来源。"
    )
    left, right = st.columns(2)
    left.download_button(
        "下载确认版正文 Markdown",
        body.markdown,
        file_name="v04_confirmed_body.md",
        mime="text/markdown",
        type="primary",
        width="stretch",
    )
    right.download_button(
        "下载正文引用审计包",
        body.model_dump_json(indent=2),
        file_name="v04_body_audit.json",
        mime="application/json",
        width="stretch",
    )
    st.info("正文完成后才能进入摘要、关键词、结论和最终参考文献审计；这些内容不会提前生成。")
    if include_final_delivery:
        _render_final_delivery(project, body)
    elif st.button("进入最终交付", type="primary", width="stretch"):
        st.session_state["mvp_navigation_request"] = "delivery"
        st.rerun()


def _render_final_delivery(project: V04WritingProject, body) -> None:
    st.divider()
    st.subheader("MVP 最终论文交付")
    st.caption(
        "正文确认后才生成标题、摘要、关键词和结论；参考文献、引用格式、"
        "要求合规审计和 DOCX 均由代码组装。"
    )
    policy = project.handoff.requirement_policy
    if policy is None:
        from veriwrite_agent.services.requirement_policy import RequirementPolicyCompiler

        policy = RequirementPolicyCompiler().compile(project.handoff.requirement)
    settings = LLMSettings()
    default_declaration = ""
    if policy.ai_usage.declaration_required:
        default_declaration = (
            "AI tool: DeepSeek; model: "
            f"{settings.structured_model or settings.model}; purpose: "
            "evidence-constrained section drafting and final-matter organization; "
            "citations and bibliography were generated and audited by VeriWrite code."
        )
    st.session_state.setdefault("mvp_ai_declaration", default_declaration)
    if policy.ai_usage.declaration_required:
        ai_declaration = st.text_area(
            "AI 使用声明",
            key="mvp_ai_declaration",
            help="请按课程要求说明工具、版本、修改位置和用途。",
        )
    else:
        ai_declaration = st.session_state["mvp_ai_declaration"]
        with st.expander("可选：添加 AI 使用声明"):
            ai_declaration = st.text_area(
                "AI 使用声明",
                key="mvp_ai_declaration",
                label_visibility="collapsed",
            )

    if FINAL_MATTER_KEY not in st.session_state:
        if st.button(
            "生成最终标题、摘要、关键词和结论",
            type="primary",
            width="stretch",
        ):
            try:
                with st.spinner("正在基于已确认正文生成最终组成部分并执行合规审计……"):
                    matter = LLMFinalMatterWriter(
                        DeepSeekClient(settings.for_structured_output())
                    ).draft(project.handoff, body)
                    package = FinalPaperAssembler().assemble(
                        handoff=project.handoff,
                        body=body,
                        final_matter=matter,
                        ai_declaration=ai_declaration.strip() or None,
                    )
            except Exception as exc:
                st.error(f"最终论文组装失败：{exc}")
            else:
                st.session_state[FINAL_MATTER_KEY] = matter.model_dump_json(indent=2)
                st.session_state[FINAL_PACKAGE_KEY] = package.model_dump_json(indent=2)
                st.rerun()
        return

    matter = FinalMatterProposal.model_validate_json(st.session_state[FINAL_MATTER_KEY])
    package = FinalPaperPackage.model_validate_json(st.session_state[FINAL_PACKAGE_KEY])
    if ai_declaration.strip() != (package.ai_declaration or ""):
        package = FinalPaperAssembler().assemble(
            handoff=project.handoff,
            body=body,
            final_matter=matter,
            ai_declaration=ai_declaration.strip() or None,
        )
        st.session_state[FINAL_PACKAGE_KEY] = package.model_dump_json(indent=2)

    metrics = st.columns(4)
    metrics[0].metric("正文统计单位", package.audit.counted_units)
    metrics[1].metric("实际引用文献", package.audit.reference_count)
    metrics[2].metric("外文文献", package.audit.foreign_reference_count)
    metrics[3].metric("阻塞项", package.audit.blocking_count)
    with st.expander("预览最终论文", expanded=package.status != "confirmed"):
        st.markdown(package.markdown)
    if package.audit.issues:
        with st.expander(
            "查看合规审计问题",
            expanded=package.audit.blocking_count > 0,
        ):
            st.dataframe(
                [
                    {
                        "级别": issue.severity,
                        "代码": issue.code,
                        "要求字段": issue.requirement_path,
                        "说明": issue.detail,
                    }
                    for issue in package.audit.issues
                ],
                hide_index=True,
                width="stretch",
            )
    if package.status == "needs_revision":
        _render_final_audit_repair(package)
        st.error("最终交付审计未通过。系统不会把不符合 V0.1 要求的结果伪装成完成品。")
        return

    if package.status == "ready_for_confirmation":
        st.caption("确认会冻结当前预览并解锁 Markdown、DOCX 与完整审计包。")
        review_warnings = [
            issue
            for issue in package.audit.issues
            if issue.severity == "warning"
            and issue.code
            in {
                "theme_element_requires_user_review",
                "original_analysis_requires_user_review",
                "reference_tool_usage_requires_attestation",
            }
        ]
        review_attested = True
        if review_warnings:
            st.warning(
                "以下语义性要求无法仅靠字符和 DOI 自动判定，需要你在最终预览中一次性确认："
                + "、".join(
                    (
                        f"主题要素“{issue.detail}”已在正文中实质体现"
                        if issue.code == "theme_element_requires_user_review"
                        else (
                            "综合部分包含学生自己的分析、认识和评价"
                            if issue.code == "original_analysis_requires_user_review"
                            else f"已按要求使用参考文献工具：{issue.detail}"
                        )
                    )
                    for issue in review_warnings
                )
            )
            review_attested = st.checkbox(
                "我已检查上述主题体现、原创分析或工具使用要求",
                key=FINAL_SEMANTIC_ATTESTATION_KEY,
            )
        actions = st.columns([2, 1])
        if actions[0].button(
            "确认当前论文并解锁下载",
            type="primary",
            width="stretch",
            disabled=not review_attested,
        ):
            package = package.model_copy(
                update={
                    "user_review_attestations": [
                        issue.code for issue in review_warnings
                    ]
                }
            )
            package = FinalPaperAssembler().confirm(
                package,
                confirmed_by=project.handoff.requirement.confirmed_by,
            )
            st.session_state[FINAL_PACKAGE_KEY] = package.model_dump_json(indent=2)
            st.session_state.pop(FINAL_REPAIR_CHECKPOINT_KEY, None)
            st.session_state.pop(FINAL_REPAIR_AUTO_SUPPRESSION_KEY, None)
            st.rerun()
        if actions[1].button("重新生成组成部分", width="stretch"):
            st.session_state.pop(FINAL_MATTER_KEY, None)
            st.session_state.pop(FINAL_PACKAGE_KEY, None)
            st.session_state.pop(FINAL_SEMANTIC_ATTESTATION_KEY, None)
            st.session_state.pop(FINAL_REPAIR_AUTO_SUPPRESSION_KEY, None)
            st.rerun()
        return

    st.success("最终论文已确认，可下载完整交付物。")
    docx_bytes = FinalPaperDocxExporter().export(package)
    downloads = st.columns(3)
    downloads[0].download_button(
        "下载最终论文 Markdown",
        package.markdown,
        file_name="veriwrite_final_paper.md",
        mime="text/markdown",
        type="primary",
        width="stretch",
    )
    downloads[1].download_button(
        "下载最终论文 DOCX",
        docx_bytes,
        file_name="veriwrite_final_paper.docx",
        mime=("application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
        width="stretch",
    )
    downloads[2].download_button(
        "下载最终合规审计",
        package.model_dump_json(indent=2),
        file_name="veriwrite_final_delivery_audit.json",
        mime="application/json",
        width="stretch",
    )
    st.info(
        "MVP 已验证引用身份、引用绑定和 PDF 页码来源；逐句语义蕴含验证 "
        "（这句话是否真的被引文支持）明确保留为 MVP 后续优化项。"
    )


def _render_final_audit_repair(package: FinalPaperPackage) -> None:
    blocking = [issue for issue in package.audit.issues if issue.severity == "blocking"]
    if not blocking:
        return
    needs_writing_repair = final_delivery_repair_stage(package) == "writing"
    st.subheader("审计修复路由")
    st.caption(
        "正文问题会被定位到具体章节和段落：写作计划、原章节和无问题段落均保留，"
        "只撤销受影响章节的确认。返修前会保存一份可撤销检查点。"
    )
    st.dataframe(
        [
            {
                "问题": issue.code,
                "责任阶段": (
                    "V0.4 写作计划与正文"
                    if issue.code in WRITING_REPAIR_CODES
                    or issue.code.startswith("body_")
                    or issue.code.startswith("citation_")
                    or issue.code.startswith("length_")
                    else "最终结构组装"
                ),
                "修复动作": (
                    "补充缺失来源分配，仅重写受影响段落"
                    if issue.code in WRITING_REPAIR_CODES
                    else (
                        "补齐必需结构并重新审计"
                        if issue.code == "required_section_missing"
                        else "定位问题段落并定点返修"
                    )
                ),
            }
            for issue in blocking
        ],
        hide_index=True,
        width="stretch",
    )
    if needs_writing_repair:
        label = "创建检查点并生成 V0.4 定点返修任务"
        target_stage = "writing"
    else:
        label = "创建检查点并重建最终结构"
        target_stage = "delivery"
    if st.button(label, type="primary", width="stretch", key="final_audit_repair"):
        repair = None
        if needs_writing_repair:
            try:
                repair = build_targeted_writing_repair(st.session_state, package)
            except (TypeError, ValueError) as exc:
                st.error(f"无法创建定点返修任务：{exc}")
                return
        _create_final_repair_checkpoint(
            [issue.code for issue in blocking],
            package=package,
        )
        if repair is not None:
            st.session_state[WRITING_PLAN_KEY] = repair.plan.model_dump_json(indent=2)
            st.session_state[V04_PROJECT_KEY] = repair.project.model_dump_json(indent=2)
            st.session_state[SECTION_SELECTION_REQUEST_KEY] = next(
                iter(repair.paragraph_numbers)
            )
            flash = (
                "已保留全部原正文，仅打开 "
                f"{len(repair.paragraph_numbers)} 个章节中的 "
                f"{repair.paragraph_count} 个问题段落进行返修。"
            )
        else:
            flash = "正文已保留，正在重建最终结构并重新审计。"
        st.session_state.pop(FINAL_MATTER_KEY, None)
        st.session_state.pop(FINAL_PACKAGE_KEY, None)
        st.session_state.pop(FINAL_SEMANTIC_ATTESTATION_KEY, None)
        st.session_state.pop(FINAL_REPAIR_AUTO_SUPPRESSION_KEY, None)
        st.session_state["mvp_navigation_request"] = target_stage
        st.session_state["mvp_flash"] = flash
        st.rerun()


def _create_final_repair_checkpoint(
    reason_codes: list[str],
    *,
    state: MutableMapping[str, Any] | None = None,
    package: FinalPaperPackage | None = None,
) -> None:
    target_state = st.session_state if state is None else state
    if FINAL_REPAIR_CHECKPOINT_KEY in target_state:
        return
    if package is None and target_state.get(FINAL_PACKAGE_KEY):
        try:
            package = FinalPaperPackage.model_validate_json(
                target_state[FINAL_PACKAGE_KEY]
            )
        except Exception:
            package = None
    payload = {
        "schema_version": "mvp-final-repair-1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repair_id": _final_repair_id(package) if package is not None else None,
        "reason_codes": list(dict.fromkeys(reason_codes)),
        "state": {
            key: target_state.get(key) for key in REPAIR_DOWNSTREAM_KEYS
        },
    }
    target_state[FINAL_REPAIR_CHECKPOINT_KEY] = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
    )


def _render_repair_checkpoint_restore() -> None:
    serialized = st.session_state.get(FINAL_REPAIR_CHECKPOINT_KEY)
    if not serialized:
        return
    try:
        payload = json.loads(serialized)
    except (TypeError, json.JSONDecodeError):
        st.session_state.pop(FINAL_REPAIR_CHECKPOINT_KEY, None)
        return
    with st.expander("已保存最终审计修复检查点"):
        st.caption(
            f"创建时间：{payload.get('created_at', '未知')}；"
            f"触发问题：{', '.join(payload.get('reason_codes', [])) or '未记录'}。"
        )
        st.caption("恢复会覆盖本次修复产生的 V0.4 和最终交付状态。")
        if st.button(
            "撤销本次修复并恢复旧版本",
            width="stretch",
            key="restore_final_repair_checkpoint",
        ):
            saved_state = payload.get("state", {})
            for key in REPAIR_DOWNSTREAM_KEYS:
                value = saved_state.get(key)
                if value is None:
                    st.session_state.pop(key, None)
                else:
                    st.session_state[key] = value
            st.session_state.pop(FINAL_REPAIR_CHECKPOINT_KEY, None)
            repair_id = payload.get("repair_id")
            if repair_id:
                st.session_state[FINAL_REPAIR_AUTO_SUPPRESSION_KEY] = repair_id
            st.session_state["mvp_navigation_request"] = "delivery"
            st.session_state["mvp_flash"] = "已恢复审计修复前的版本。"
            st.rerun()


def _final_repair_id(package: FinalPaperPackage) -> str:
    blocking_codes = sorted(
        issue.code for issue in package.audit.issues if issue.severity == "blocking"
    )
    return ":".join(
        [
            package.audit.policy_fingerprint,
            package.generated_at.isoformat(),
            *blocking_codes,
        ]
    )


def _next_actionable_section(project: V04WritingProject) -> str:
    for section in project.sections:
        if section.status != "confirmed":
            return section.section_id
    return project.sections[0].section_id


def _status_label(status: str) -> str:
    return {
        "pending": "待生成",
        "draft": "待确认",
        "needs_review": "存在阻塞",
        "confirmed": "已确认",
    }[status]
