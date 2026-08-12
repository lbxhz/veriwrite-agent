"""Streamlit workbench for V0.4 evidence-constrained section writing."""

from __future__ import annotations

import json
import hashlib
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Literal, MutableMapping
from uuid import uuid4

import streamlit as st

from veriwrite_agent.config.settings import LLMSettings
from veriwrite_agent.llm.deepseek_client import DeepSeekClient
from veriwrite_agent.models.writing import (
    SectionDraftIssue,
    V04WritingProject,
    WritingSectionState,
)
from veriwrite_agent.models.writing_plan import GroundedWritingPlan, WritingSectionPlan
from veriwrite_agent.models.literature_selection import (
    ConfirmedLiteratureSearchBlueprint,
)
from veriwrite_agent.models.final_delivery import (
    FinalMatterProposal,
    FinalPaperAuditIssue,
    FinalPaperPackage,
)
from veriwrite_agent.models.paper_quality import (
    PaperQualityComparison,
    PaperQualityScorecard,
)
from veriwrite_agent.models.external_writing_evaluation import (
    ExternalWritingEvaluation,
)
from veriwrite_agent.models.writing_quality import ManuscriptEditorialCheckpoint
from veriwrite_agent.models.writing_handoff import V04WritingHandoff
from veriwrite_agent.services.grounded_writing import (
    SectionEvidencePacketBuilder,
    WritingProjectService,
    count_writing_units,
)
from veriwrite_agent.services.local_project_store import LocalProjectStore
from veriwrite_agent.services.agent_artifacts import artifact_reference_from_model
from veriwrite_agent.services.agent_runtime_store import AgentRuntimeStore
from veriwrite_agent.services.manuscript_structural_editing import (
    merge_redundant_manuscript_paragraphs,
    semantically_replan_manuscript_sections,
)
from veriwrite_agent.services.writing_planning import (
    GroundedWritingPlanner,
    LLMGroundedParagraphWriter,
    ParagraphWritingRuntimeCache,
    WritingPlanBudgetExceeded,
    WritingPlanDependencyError,
    WritingPlanRuntimeCache,
    _paragraph_requires_rewrite,
    align_writing_plan_language,
    rebase_writing_plan_authority,
    repair_writing_plan_source_coverage,
)
from veriwrite_agent.services.writing_quality import (
    FullManuscriptEditorialService,
    LLMManuscriptQualityReviewer,
    LLMSectionQualityReviewer,
    PLAN_BINDING_DETERMINISTIC_CODES,
    SECTION_QUALITY_REVIEW_CODES,
    false_self_attribution_detail,
    language_mismatch_detail,
    manuscript_body_fingerprint,
    mark_manuscript_editor_targets,
    normalize_manuscript_review_for_current_policy,
    refine_writing_plan_for_manuscript_review,
)
from veriwrite_agent.services.writing_autopilot import (
    ContinuousSectionWritingService,
    ContinuousWritingEvent,
    ContinuousWritingPolicy,
    ContinuousWritingResult,
)
from veriwrite_agent.services.writing_agent_controller import WritingAgentController
from veriwrite_agent.services.writing_agent_runtime import (
    WritingAgentContext,
    WritingAgentRuntimeService,
)
from veriwrite_agent.services.writing_evidence_recovery import (
    WritingEvidenceRecoveryRequest,
    WritingEvidenceRecoveryService,
    downgrade_unresolved_evidence_claims,
    merge_recovery_handoffs,
)
from veriwrite_agent.services.final_delivery import (
    FinalPaperAssembler,
    FinalPaperDocxExporter,
    LLMFinalMatterWriter,
)
from veriwrite_agent.services.paper_quality_evaluation import (
    PaperQualityEvaluationService,
)
from veriwrite_agent.services.external_writing_evaluator import (
    ExternalEvaluatorConfig,
    ExternalEvaluatorError,
    ExternalWritingEvaluatorClient,
    comparable_external_evaluations,
    external_quality_warning,
)
from veriwrite_agent.services.requirement_policy import RequirementPolicyCompiler
from veriwrite_agent.services.topic_admission import audit_topic_admission
from veriwrite_agent.ui.mvp_console import autosave_local_project
from veriwrite_agent.ui.workbench import project_root

WRITING_PLAN_KEY = "v04_writing_plan_json"
V04_PROJECT_KEY = "v04_writing_project_json"
FINAL_MATTER_KEY = "mvp_final_matter_json"
FINAL_PACKAGE_KEY = "mvp_final_paper_json"
FINAL_REPAIR_CHECKPOINT_KEY = "mvp_final_repair_checkpoint_json"
FINAL_SEMANTIC_ATTESTATION_KEY = "mvp_final_semantic_review_attestation"
FINAL_REPAIR_AUTO_SUPPRESSION_KEY = "mvp_final_repair_auto_suppressed_id"
FINAL_DELIVERY_AUTO_RESUME_KEY = "mvp_final_delivery_auto_resume"
MANUSCRIPT_EDITOR_KEY = "mvp_manuscript_editor_checkpoint_json"
GLOBAL_EDITOR_ROUND_KEY = "mvp_global_editor_round"
MAX_GLOBAL_EDITOR_ROUNDS = 3
PAPER_QUALITY_SCORECARD_KEY = "mvp_paper_quality_scorecard_json"
PAPER_QUALITY_BASELINE_KEY = "mvp_paper_quality_baseline_json"
EXTERNAL_WRITING_EVALUATION_KEY = "mvp_external_writing_evaluation_json"
EXTERNAL_WRITING_BASELINE_KEY = "mvp_external_writing_baseline_json"
EXTERNAL_WRITING_FAILURE_KEY = "mvp_external_writing_failure_json"
SECTION_SELECTION_KEY = "v04_selected_section"
SECTION_SELECTION_REQUEST_KEY = "v04_selected_section_request"
WRITING_MODE_KEY = "v04_writing_mode"
EVIDENCE_RECOVERY_REQUEST_KEY = "v04_evidence_recovery_json"
EVIDENCE_RECOVERY_CHECKPOINT_KEY = "v04_evidence_recovery_checkpoint_json"
EVIDENCE_RECOVERY_AUTO_PLAN_KEY = "v04_evidence_recovery_auto_plan"
EVIDENCE_RECOVERY_AUTO_RESUME_KEY = "v04_evidence_recovery_auto_resume"
WRITING_PLAN_AUTO_FAILURES_KEY = "v04_writing_plan_auto_failures"
WRITING_PLAN_PERMISSION_REPAIR_MIGRATION_KEY = (
    "v04_writing_plan_permission_repair_migration_v1"
)
WRITING_PLAN_REPAIR_FEEDBACK_KEY = "v04_writing_plan_repair_feedback_json"
V04_AGENT_RUN_ID_KEY = "v04_agent_run_id"
V04_AUTOPILOT_REQUESTED_KEY = "v04_autopilot_requested"
REPAIR_DOWNSTREAM_KEYS = (
    "literature_blueprint_json",
    "literature_blueprint_editor",
    "literature_confirmed_blueprint_json",
    "literature_result_json",
    "literature_ris",
    "literature_verification_json",
    "literature_run_dir",
    "literature_pool_multiplier",
    "v03_core_dois",
    "v03_download_directory",
    "v03_pdf_inspection_json",
    "v03_evidence_library_json",
    "v03_writing_handoff_json",
    WRITING_PLAN_KEY,
    V04_PROJECT_KEY,
    FINAL_MATTER_KEY,
    FINAL_PACKAGE_KEY,
    PAPER_QUALITY_SCORECARD_KEY,
    PAPER_QUALITY_BASELINE_KEY,
    EXTERNAL_WRITING_EVALUATION_KEY,
    EXTERNAL_WRITING_BASELINE_KEY,
    EXTERNAL_WRITING_FAILURE_KEY,
    MANUSCRIPT_EDITOR_KEY,
    GLOBAL_EDITOR_ROUND_KEY,
    V04_AGENT_RUN_ID_KEY,
)
LITERATURE_REBUILD_CODES = {
    "reference_count_below_minimum",
    "reference_count_below_target",
    "topic_admission_incomplete",
}
EVIDENCE_REBUILD_CODES = {"document_identity_mismatch"}
WRITING_REPAIR_CODES = {
    "reference_count_above_maximum",
    "uncited_bibliography_item",
    "unknown_citation_key",
}
EDITORIAL_REPAIR_CODES = SECTION_QUALITY_REVIEW_CODES


@dataclass(frozen=True)
class TargetedWritingRepair:
    plan: GroundedWritingPlan
    project: V04WritingProject
    paragraph_numbers: dict[str, tuple[int, ...]]

    @property
    def paragraph_count(self) -> int:
        return sum(len(numbers) for numbers in self.paragraph_numbers.values())


_AGENT_STAGE_LABELS = {
    "requirements": "读取已确认要求",
    "literature": "补充与筛选文献",
    "evidence": "核验全文与补充证据",
    "planning": "规划论文结构与证据用途",
    "writing": "逐章写作、审稿与定点返修",
    "editing": "全文去重、衔接与学术表达编辑",
    "delivery": "合规审计、评分与交付组装",
}


def _render_agent_activity(project: V04WritingProject | None = None) -> None:
    """Show a concise, read-only view of the durable Agent run."""

    run_id = st.session_state.get(V04_AGENT_RUN_ID_KEY)
    if not isinstance(run_id, str) or not re.fullmatch(r"run_[0-9a-f]{16}", run_id):
        failures = int(
            st.session_state.get(WRITING_PLAN_AUTO_FAILURES_KEY, 0) or 0
        )
        agent_starting = bool(
            st.session_state.get(V04_AUTOPILOT_REQUESTED_KEY)
            or st.session_state.get(EVIDENCE_RECOVERY_AUTO_PLAN_KEY)
            or st.session_state.get(EVIDENCE_RECOVERY_AUTO_RESUME_KEY)
            or (failures and WRITING_PLAN_KEY not in st.session_state)
        )
        if agent_starting:
            st.info(
                "运行中 · 规划论文结构与证据用途"
                + (f" · 自动修复 {failures} 次" if failures else "")
            )
            st.caption(
                "Agent 正在生成并审计写作计划；已有章节和证据检查点都会保留。"
            )
            return
        st.info(
            "Agent 尚未启动。开始后会自动完成规划、逐章写作、独立审稿、"
            "定点返修、全文编辑、合规审计和评分。"
        )
        return
    try:
        store = AgentRuntimeStore(project_root() / "runtime" / "agent_runs" / run_id)
        state = store.load_state()
        decision = (
            store.load_decision(state.latest_decision_id)
            if state is not None and state.latest_decision_id
            else None
        )
    except (OSError, ValueError) as exc:
        st.warning("Agent 运行记录暂时无法读取；论文阶段产物仍保存在项目检查点中。")
        with st.expander("技术详情"):
            st.code(str(exc))
        return
    if state is None:
        st.info("Agent 正在建立首个可恢复检查点……")
        return

    completed = 0
    total = 0
    if project is not None:
        total = len(project.sections)
        completed = sum(section.status == "confirmed" for section in project.sections)
    stage_label = _AGENT_STAGE_LABELS.get(state.current_stage, state.current_stage)
    lifecycle_label = {
        "running": "运行中",
        "waiting_user": "等待必要的用户输入",
        "completed": "已完成",
        "failed": "执行失败",
        "stopped": "已停止自动循环",
    }[state.lifecycle]
    if state.lifecycle == "running" and not _agent_auto_run_requested(
        st.session_state
    ):
        lifecycle_label = "已暂停"
    headline = f"{lifecycle_label} · {stage_label}"
    if total:
        headline += f" · 章节 {completed}/{total}"
    if state.lifecycle == "completed":
        st.success(headline)
    elif state.lifecycle in {"failed", "stopped"}:
        st.error(headline)
    else:
        st.info(headline)
    if decision is not None:
        st.caption(f"最近判断：{decision.explanation}")
    elif state.blocker_codes:
        st.caption("正在分析阻塞原因：" + "、".join(state.blocker_codes))
    with st.expander("查看 Agent 运行记录（高级）"):
        st.json(
            {
                "run_id": state.run_id,
                "当前阶段": stage_label,
                "运行状态": lifecycle_label,
                "最近决策": decision.decision_type if decision else None,
                "决策原因": decision.reason_code if decision else None,
                "阻塞代码": state.blocker_codes,
                "恢复轮次": state.revision_rounds_by_stage,
                "检查点序号": state.event_sequence,
            }
        )


def _agent_auto_run_requested(state: MutableMapping[str, Any]) -> bool:
    return any(
        bool(state.get(key))
        for key in (
            V04_AUTOPILOT_REQUESTED_KEY,
            EVIDENCE_RECOVERY_AUTO_PLAN_KEY,
            EVIDENCE_RECOVERY_AUTO_RESUME_KEY,
            FINAL_DELIVERY_AUTO_RESUME_KEY,
            "literature_auto_run_requested",
            "literature_auto_advance_requested",
        )
    )


def pause_writing_agent(state: MutableMapping[str, Any]) -> bool:
    """Soft-stop automatic transitions while retaining every durable checkpoint."""

    was_running = _agent_auto_run_requested(state)
    for key in (
        V04_AUTOPILOT_REQUESTED_KEY,
        EVIDENCE_RECOVERY_AUTO_PLAN_KEY,
        EVIDENCE_RECOVERY_AUTO_RESUME_KEY,
        FINAL_DELIVERY_AUTO_RESUME_KEY,
        "literature_auto_run_requested",
        "literature_auto_advance_requested",
    ):
        state.pop(key, None)
    state["mvp_flash"] = (
        "Agent 已暂停。已完成章节、当前草稿、证据恢复请求和检查点均已保留；"
        "下次点击继续时只会恢复尚未完成的节点。"
    )
    return was_running


def _render_agent_pause_control() -> None:
    """Keep the pause action visible on the main V0.4 status surface."""

    if not _agent_auto_run_requested(st.session_state):
        return
    st.caption("需要离开或检查中间结果时，可以随时暂停；已完成内容不会丢失。")
    if st.button(
        "暂停 Agent 自动运行",
        width="stretch",
        key="v04_pause_agent_main",
    ):
        pause_writing_agent(st.session_state)
        _autosave_current_project()
        st.rerun()


def active_writing_recovery_status(
    state: MutableMapping[str, Any],
) -> Literal[
    "pending_full_text",
    "pending_search",
    "ready_to_resume",
    "blocked",
] | None:
    """Return the internal recovery phase without changing visible navigation."""

    raw = state.get(EVIDENCE_RECOVERY_REQUEST_KEY)
    if not raw:
        return None
    try:
        request = WritingEvidenceRecoveryRequest.model_validate_json(raw)
    except (TypeError, ValueError):
        return None
    if request.status == "resolved":
        return None
    return request.status


def _consume_writing_plan_auto_request(
    state: MutableMapping[str, Any],
    *,
    auto_failures: int,
) -> bool:
    """Require an explicit run flag; a resumable checkpoint alone is not consent."""

    return bool(
        state.pop(EVIDENCE_RECOVERY_AUTO_PLAN_KEY, False)
        or (
            state.get(V04_AUTOPILOT_REQUESTED_KEY)
            and auto_failures < 2
        )
    )


def render_writing_agent_recovery_shell() -> None:
    """Keep users in V0.4 while the Agent executes an internal rollback."""

    raw = st.session_state.get(EVIDENCE_RECOVERY_REQUEST_KEY)
    if not raw:
        return
    try:
        request = WritingEvidenceRecoveryRequest.model_validate_json(raw)
    except (TypeError, ValueError):
        st.error("Agent 恢复记录无法读取；已保留此前论文检查点。")
        return
    st.divider()
    st.header("V0.4 Agent 论文生成")
    st.caption(
        "用户始终停留在本页；V0.2 检索与 V0.3 证据处理只是 Agent 的内部工具节点。"
    )
    _render_agent_pause_control()
    labels = {
        "pending_search": ("回退", "正在按缺失论点自动补搜并排除重复 DOI"),
        "pending_full_text": ("执行", "正在检查本地全文并构建可追溯证据"),
        "ready_to_resume": ("规划", "证据已更新，正在重规划受影响章节"),
        "blocked": ("等待协助", "自动替代已用尽，需要一次性补充受限 PDF"),
    }
    phase, detail = labels[request.status]
    st.info(f"当前动作：{phase} · {detail}")
    completed_steps = {
        "pending_search": 1,
        "pending_full_text": 2,
        "ready_to_resume": 3,
        "blocked": 2,
    }[request.status]
    st.progress(completed_steps / 4, text="Agent 循环：计划 → 执行 → 审核 → 判断/回退")
    st.caption(
        f"恢复轮次 {request.recovery_round}/{request.max_recovery_rounds} · "
        f"影响 {len(request.affected_section_ids)} 个章节 · 合格正文保持不变"
    )
    if request.status == "blocked":
        missing = list(
            dict.fromkeys(
                [
                    *request.requested_core_dois,
                    *request.unavailable_full_text_dois,
                    *(
                        doi
                        for gap in request.gaps
                        for doi in gap.missing_full_text_dois
                    ),
                ]
            )
        )
        st.warning(
            "系统已经先尝试本地扫描和替代文献，不会逐轮打断你。"
            "下面是合并后的最终人工下载清单；下载完成后只需重新扫描一次。"
        )
        if missing:
            st.code("\n".join(missing))
        st.caption(
            "如果这些受限全文无法取得，可以放弃无法证明的细节比较；"
            "系统会保留来源，只把相关段落收缩为证据允许的一般背景。"
        )
        if st.button(
            "无法获取这些 PDF，收缩论断并继续",
            width="stretch",
            key="v04_continue_without_restricted_pdf",
        ):
            if continue_without_restricted_full_text():
                _autosave_current_project()
                st.rerun()
            st.error(
                st.session_state.get(
                    "mvp_flash",
                    "无法在不破坏证据约束的前提下收缩这些论断。",
                )
            )
    with st.expander("查看 Agent 恢复依据（高级）"):
        st.json(
            {
                "状态": request.status,
                "影响章节": request.affected_section_ids,
                "缺口数量": len(request.gaps),
                "已判定不可获取全文": request.unavailable_full_text_dois,
                "阻塞原因": request.blocked_reason,
            }
        )


def clear_writing_state(*, preserve_repair_checkpoint: bool = False) -> None:
    preserve_active_agent = bool(
        active_writing_recovery_status(st.session_state) is not None
        or st.session_state.get(EVIDENCE_RECOVERY_AUTO_PLAN_KEY)
        or st.session_state.get(EVIDENCE_RECOVERY_AUTO_RESUME_KEY)
    )
    st.session_state.pop(WRITING_PLAN_KEY, None)
    st.session_state.pop(V04_PROJECT_KEY, None)
    st.session_state.pop(FINAL_MATTER_KEY, None)
    st.session_state.pop(FINAL_PACKAGE_KEY, None)
    if not preserve_repair_checkpoint:
        st.session_state.pop(FINAL_REPAIR_CHECKPOINT_KEY, None)
    st.session_state.pop(FINAL_SEMANTIC_ATTESTATION_KEY, None)
    st.session_state.pop(FINAL_REPAIR_AUTO_SUPPRESSION_KEY, None)
    st.session_state.pop(FINAL_DELIVERY_AUTO_RESUME_KEY, None)
    st.session_state.pop(MANUSCRIPT_EDITOR_KEY, None)
    st.session_state.pop(GLOBAL_EDITOR_ROUND_KEY, None)
    st.session_state.pop(SECTION_SELECTION_KEY, None)
    st.session_state.pop(SECTION_SELECTION_REQUEST_KEY, None)
    st.session_state.pop(WRITING_MODE_KEY, None)
    if not preserve_active_agent:
        st.session_state.pop(V04_AGENT_RUN_ID_KEY, None)
        st.session_state.pop(V04_AUTOPILOT_REQUESTED_KEY, None)
    st.session_state.pop(PAPER_QUALITY_SCORECARD_KEY, None)
    st.session_state.pop(PAPER_QUALITY_BASELINE_KEY, None)
    st.session_state.pop(EXTERNAL_WRITING_EVALUATION_KEY, None)
    st.session_state.pop(EXTERNAL_WRITING_BASELINE_KEY, None)
    st.session_state.pop(EXTERNAL_WRITING_FAILURE_KEY, None)


def queue_evidence_recovery_resume() -> None:
    """Continue a cross-stage evidence repair without another user confirmation."""

    if st.session_state.get(EVIDENCE_RECOVERY_REQUEST_KEY):
        st.session_state[EVIDENCE_RECOVERY_AUTO_PLAN_KEY] = True


def final_delivery_repair_stage(
    package: FinalPaperPackage,
) -> Literal["literature", "evidence", "writing", "delivery"] | None:
    """Route final blockers to the earliest stage that can actually fix them."""

    blocking_codes = {
        issue.code for issue in package.audit.issues if issue.severity == "blocking"
    }
    if not blocking_codes:
        return None
    if blocking_codes & LITERATURE_REBUILD_CODES:
        return "literature"
    if blocking_codes & EVIDENCE_REBUILD_CODES:
        return "evidence"
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
    *,
    structural_planner: GroundedWritingPlanner | None = None,
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

    plan = refine_writing_plan_for_manuscript_review(
        plan,
        package.manuscript_review,
        evidence_doi_by_id={
            card.evidence_id: card.doi
            for card in project.handoff.evidence_library.evidence_cards
        },
    )
    structural_edit = merge_redundant_manuscript_paragraphs(
        plan,
        project,
        package.manuscript_review,
    )
    plan = structural_edit.plan
    project = structural_edit.project
    target_remap = structural_edit.target_remap

    blocking = [issue for issue in package.audit.issues if issue.severity == "blocking"]
    plan = _expand_plan_for_length_shortfall(plan, blocking)
    if structural_planner is not None:
        semantic_sections = _semantic_replan_section_ids(blocking)
        plan = semantically_replan_manuscript_sections(
            plan,
            project,
            section_ids=semantic_sections,
            planner=structural_planner,
        )
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
        original_targets = _paragraph_targets_for_final_issue(
            issue,
            plan=plan,
            project=project,
            package=package,
        )
        if not original_targets:
            original_targets = {_fallback_repair_target(plan)}
        for original_target in original_targets:
            target = target_remap.get(original_target, original_target)
            repair_issue = issue
            if target != original_target:
                repair_issue = issue.model_copy(
                    update={
                        "detail": (
                            "The redundant paragraph has already been removed by the "
                            "structural editor and its unique locked support was migrated "
                            "to this adjacent main paragraph. Rewrite the merged paragraph "
                            "as one submit-ready argument; do not mention the edit process "
                            "or restate material covered elsewhere."
                        )
                    }
                )
            target_issues.setdefault(target, []).append(repair_issue)

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


def build_manuscript_editor_repair(
    state: MutableMapping[str, Any],
    checkpoint: ManuscriptEditorialCheckpoint,
    *,
    structural_planner: GroundedWritingPlanner | None = None,
) -> TargetedWritingRepair:
    """Route an independent full-manuscript review directly to V0.4 repairs.

    Final-matter generation is intentionally not involved: a body defect must be repaired
    before the system spends another call on the abstract, introduction, or conclusion.
    """

    if checkpoint.status != "needs_revision":
        raise ValueError("全文编辑检查点没有需要返修的阻塞问题")
    serialized_plan = state.get(WRITING_PLAN_KEY)
    serialized_project = state.get(V04_PROJECT_KEY)
    if not serialized_plan or not serialized_project:
        raise ValueError("缺少已确认的 V0.4 写作计划或正文检查点，无法执行全文编辑返修")
    plan = GroundedWritingPlan.model_validate_json(serialized_plan)
    project = V04WritingProject.model_validate_json(serialized_project)
    if plan.status != "confirmed" or project.status != "body_complete":
        raise ValueError("只有已完成的 V0.4 正文才能执行全文编辑返修")
    if checkpoint.body_fingerprint != manuscript_body_fingerprint(plan, project):
        raise ValueError("全文编辑结果已过期；正文变化后必须重新审阅")

    review = checkpoint.review
    plan = refine_writing_plan_for_manuscript_review(
        plan,
        review,
        evidence_doi_by_id={
            card.evidence_id: card.doi
            for card in project.handoff.evidence_library.evidence_cards
        },
    )
    structural_edit = merge_redundant_manuscript_paragraphs(plan, project, review)
    plan = structural_edit.plan
    project = structural_edit.project
    target_remap = structural_edit.target_remap

    blocking_findings = [
        finding
        for finding in review.findings
        if finding.severity == "blocking"
        or finding.disposition == "targeted_repair"
    ]
    # Full-manuscript findings request editorial reduction, de-duplication, or style
    # repair. They are not permission to redesign the research question or introduce
    # new sources. A former whole-section semantic replan could select metadata-only
    # DOI records and incorrectly escalate a prose edit into V0.2/V0.3 recovery. Keep
    # the refined, evidence-locked plan and rewrite only the located paragraphs.

    target_issues: dict[tuple[str, int], list[FinalPaperAuditIssue]] = {}
    for finding in blocking_findings:
        original_target = (finding.section_id, finding.paragraph_number)
        target = target_remap.get(original_target, original_target)
        detail = (
            f"{finding.detail} 修订要求：{finding.revision_instruction}"
        )
        if target != original_target:
            detail = (
                "结构编辑器已删除重复段落并把其唯一证据迁移到相邻主段落。"
                "请把迁移后的证据组织为一个中心判断，不得描述编辑过程或复述前文。"
            )
        target_issues.setdefault(target, []).append(
            FinalPaperAuditIssue(
                code=f"body_{finding.code}",
                severity="blocking",
                requirement_path="writing.global_manuscript_editor",
                detail=detail,
            )
        )
    if not target_issues:
        raise ValueError("全文编辑检查点没有可执行的段落级返修目标")

    plan = mark_manuscript_editor_targets(
        plan,
        {
            target: [issue.detail for issue in issues]
            for target, issues in target_issues.items()
        },
    )
    reopened = _reopen_targeted_paragraphs(project, target_issues)
    ordered_targets = {
        section.section_id: tuple(
            sorted(
                paragraph_number
                for target_section, paragraph_number in target_issues
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


def _semantic_replan_section_ids(
    issues: list[FinalPaperAuditIssue],
) -> set[str]:
    semantic_codes = {
        "body_cross_section_repetition",
        "body_section_role_overlap",
        "body_global_coherence_gap",
        "body_oversized_paragraph",
        "citation_cluster_too_large",
    }
    return {
        section_id
        for issue in issues
        if issue.code in semantic_codes
        for section_id, _ in re.findall(
            r"([a-z][a-z0-9_]{1,39}):(\d+)=",
            issue.detail,
        )
    }


def _paragraph_targets_for_final_issue(
    issue: FinalPaperAuditIssue,
    *,
    plan: GroundedWritingPlan,
    project: V04WritingProject,
    package: FinalPaperPackage,
) -> set[tuple[str, int]]:
    if issue.code.startswith("body_"):
        return {
            (section_id, int(number))
            for section_id, number in re.findall(
                r"([a-z][a-z0-9_]{1,39}):(\d+)=",
                issue.detail,
            )
        }
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


def _expand_plan_for_length_shortfall(
    plan: GroundedWritingPlan,
    issues: list[FinalPaperAuditIssue],
) -> GroundedWritingPlan:
    """Turn a real final-length deficit into executable paragraph budgets.

    Merely reopening an under-length paragraph does not help when its compiled target is
    unchanged.  Add the measured shortfall plus a safety margin to ordinary substantive
    paragraphs, while keeping each target below the full-manuscript oversized threshold.
    The existing length router will still select only enough paragraphs to close the gap.
    """

    issue = next((item for item in issues if item.code == "length_below_minimum"), None)
    if issue is None:
        return plan
    values = [int(value) for value in re.findall(r"\d+", issue.detail)]
    if len(values) < 2:
        return plan
    shortfall = max(0, values[0] - values[1])
    if not shortfall:
        return plan
    growth = shortfall + max(500, int(shortfall * 0.25))
    section_payloads = [section.model_dump(mode="json") for section in plan.sections]
    candidates: list[tuple[int, int]] = []
    for section_index, section in enumerate(section_payloads):
        for paragraph_index, paragraph in enumerate(section["paragraphs"]):
            if str(paragraph["purpose"]).startswith("Global manuscript repair:"):
                continue
            if int(paragraph["target_words"]) >= 1100:
                continue
            candidates.append((section_index, paragraph_index))
    if not candidates:
        return plan

    remaining = growth
    while remaining > 0:
        progressed = False
        share = max(1, (remaining + len(candidates) - 1) // len(candidates))
        for section_index, paragraph_index in candidates:
            paragraph = section_payloads[section_index]["paragraphs"][paragraph_index]
            capacity = 1100 - int(paragraph["target_words"])
            if capacity <= 0:
                continue
            increment = min(capacity, share, remaining)
            paragraph["target_words"] = int(paragraph["target_words"]) + increment
            remaining -= increment
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            break
    if remaining == growth:
        return plan
    for section in section_payloads:
        section["target_words"] = sum(
            int(paragraph["target_words"]) for paragraph in section["paragraphs"]
        )
    canonical = json.dumps(
        {
            "pipeline_version": "grounded-writing-plan-v4-length-recovery",
            "topic": plan.topic,
            "output_language": plan.output_language,
            "required_source_dois": plan.required_source_dois,
            "sections": section_payloads,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    payload = plan.model_dump(mode="json")
    payload["sections"] = section_payloads
    payload["plan_fingerprint"] = hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()
    return GroundedWritingPlan.model_validate(payload)


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
                # A final-manuscript finding is a new editorial incident. Old
                # chapter-review rounds must not consume the repair budget before
                # the newly targeted paragraph has had a chance to be revised.
                "quality_review_status": "not_run",
                "quality_review_rounds": 0,
                "quality_reviewed_at": None,
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
    if not serialized:
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
    completed_rounds = int(state.get(GLOBAL_EDITOR_ROUND_KEY, 0) or 0)
    if completed_rounds >= MAX_GLOBAL_EDITOR_ROUNDS:
        state[FINAL_REPAIR_AUTO_SUPPRESSION_KEY] = repair_id
        state["mvp_flash"] = (
            f"全局编辑已完成 {completed_rounds} 轮，仍存在可操作的结构问题。"
            "系统已停止自动改写；这说明写作计划、段落职责或审稿规则需要调整，"
            "继续盲目重试不会可靠解决。"
        )
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
    state[GLOBAL_EDITOR_ROUND_KEY] = completed_rounds + 1
    state.pop(FINAL_MATTER_KEY, None)
    state.pop(FINAL_PACKAGE_KEY, None)
    state.pop(FINAL_SEMANTIC_ATTESTATION_KEY, None)
    state.pop(FINAL_REPAIR_AUTO_SUPPRESSION_KEY, None)
    state["mvp_navigation"] = "writing"
    state.pop("mvp_navigation_request", None)
    first_section = next(iter(repair.paragraph_numbers))
    state[SECTION_SELECTION_REQUEST_KEY] = first_section
    if state.get(WRITING_MODE_KEY, "AI 全托管") == "AI 全托管":
        state[WRITING_MODE_KEY] = "AI 全托管"
        state[EVIDENCE_RECOVERY_AUTO_RESUME_KEY] = True
    state["mvp_flash"] = (
        "最终审计已转换为定点返修：保留全部原正文，只重新处理 "
        f"{len(repair.paragraph_numbers)} 个章节中的 {repair.paragraph_count} 个问题段落。"
        + (
            " AI 全托管将自动继续，不需要再次点击。"
            if state.get(WRITING_MODE_KEY, "AI 全托管") == "AI 全托管"
            else ""
        )
    )
    return True


def rollback_outdated_delivery_to_v04(
    state: MutableMapping[str, Any],
) -> bool:
    """Convert a legacy V0.5 manuscript review into an executable V0.4 repair.

    Older packages stored material repetition and oversized-paragraph findings as
    non-blocking advice.  Reopening them only inside the delivery page leaves the
    sidebar on V0.5 and obscures the actual state transition.  This migration runs
    before Streamlit creates the navigation widget and moves only implicated body
    paragraphs back to V0.4.
    """

    serialized = state.get(FINAL_PACKAGE_KEY)
    if not serialized:
        return False
    try:
        package = FinalPaperPackage.model_validate_json(serialized)
    except (TypeError, ValueError):
        return False
    if package.schema_version == "mvp-2.2" or package.manuscript_review is None:
        return False
    try:
        plan = GroundedWritingPlan.model_validate_json(state[WRITING_PLAN_KEY])
        project = V04WritingProject.model_validate_json(state[V04_PROJECT_KEY])
        review = normalize_manuscript_review_for_current_policy(
            package.manuscript_review,
            plan,
        )
    except (KeyError, TypeError, ValueError):
        return False
    blocking = [
        finding
        for finding in review.findings
        if finding.severity == "blocking"
        or finding.disposition == "targeted_repair"
    ]
    if not blocking:
        return False
    checkpoint = ManuscriptEditorialCheckpoint(
        body_fingerprint=manuscript_body_fingerprint(plan, project),
        status="needs_revision",
        review=review,
        blocking_count=len(blocking),
        warning_count=len(review.findings) - len(blocking),
        completed_at=datetime.now(timezone.utc),
    )
    try:
        repair = build_manuscript_editor_repair(state, checkpoint)
    except (TypeError, ValueError):
        return False
    _create_final_repair_checkpoint(
        [finding.code for finding in blocking],
        state=state,
        package=package,
        replace=True,
    )
    state[WRITING_PLAN_KEY] = repair.plan.model_dump_json(indent=2)
    state[V04_PROJECT_KEY] = repair.project.model_dump_json(indent=2)
    state.pop(FINAL_MATTER_KEY, None)
    state.pop(FINAL_PACKAGE_KEY, None)
    state.pop(MANUSCRIPT_EDITOR_KEY, None)
    state.pop(FINAL_SEMANTIC_ATTESTATION_KEY, None)
    state.pop(FINAL_REPAIR_AUTO_SUPPRESSION_KEY, None)
    state[GLOBAL_EDITOR_ROUND_KEY] = 1
    state[SECTION_SELECTION_REQUEST_KEY] = next(iter(repair.paragraph_numbers))
    state[WRITING_MODE_KEY] = "AI 全托管"
    state[V04_AUTOPILOT_REQUESTED_KEY] = True
    state[EVIDENCE_RECOVERY_AUTO_RESUME_KEY] = True
    state[FINAL_DELIVERY_AUTO_RESUME_KEY] = True
    state["mvp_navigation"] = "writing"
    state.pop("mvp_navigation_request", None)
    state["mvp_flash"] = (
        "新版全文编辑策略发现旧终稿仍有可执行问题；已从 V0.5 回退到 V0.4，"
        f"保留合格正文，只重新生成 {repair.paragraph_count} 个问题段落。"
    )
    return True


def reopen_entire_body_for_regeneration(
    state: MutableMapping[str, Any],
) -> bool:
    """Handle an explicit V0.5 rejection by reopening every V0.4 paragraph."""

    serialized_project = state.get(V04_PROJECT_KEY)
    if not serialized_project:
        return False
    try:
        project = V04WritingProject.model_validate_json(serialized_project)
        package = (
            FinalPaperPackage.model_validate_json(state[FINAL_PACKAGE_KEY])
            if state.get(FINAL_PACKAGE_KEY)
            else None
        )
    except (TypeError, ValueError):
        return False
    _create_final_repair_checkpoint(
        ["user_requested_full_regeneration"],
        state=state,
        package=package,
        replace=True,
    )
    reopened_states: list[WritingSectionState] = []
    for section in project.sections:
        if section.draft is None:
            reopened_states.append(section)
            continue
        issues = [
            SectionDraftIssue(
                code="final_audit_repair",
                severity="blocking",
                paragraph_number=number,
                detail=(
                    "用户在最终交付阶段拒绝当前正文。请在原写作计划和证据范围内"
                    "重新生成本段，不得复用旧措辞或新增未经许可的事实与引用。"
                ),
            )
            for number in range(1, len(section.draft.paragraphs) + 1)
        ]
        draft = section.draft.model_copy(
            update={
                "status": "needs_review",
                "issues": issues,
                "confirmed_by": None,
                "confirmed_at": None,
                "quality_review_status": "not_run",
                "quality_review_rounds": 0,
                "quality_reviewed_at": None,
            }
        )
        reopened_states.append(
            WritingSectionState(
                section_id=section.section_id,
                status="needs_review",
                draft=draft,
            )
        )
    reopened = project.model_copy(
        update={
            "status": "drafting",
            "sections": reopened_states,
            "updated_at": datetime.now(timezone.utc),
        }
    )
    state[V04_PROJECT_KEY] = V04WritingProject.model_validate(
        reopened.model_dump(mode="json")
    ).model_dump_json(indent=2)
    for key in (
        FINAL_MATTER_KEY,
        FINAL_PACKAGE_KEY,
        MANUSCRIPT_EDITOR_KEY,
        FINAL_SEMANTIC_ATTESTATION_KEY,
        FINAL_REPAIR_AUTO_SUPPRESSION_KEY,
        PAPER_QUALITY_SCORECARD_KEY,
        EXTERNAL_WRITING_EVALUATION_KEY,
        EXTERNAL_WRITING_FAILURE_KEY,
    ):
        state.pop(key, None)
    state[GLOBAL_EDITOR_ROUND_KEY] = 0
    state[SECTION_SELECTION_REQUEST_KEY] = project.sections[0].section_id
    state[WRITING_MODE_KEY] = "AI 全托管"
    state[V04_AUTOPILOT_REQUESTED_KEY] = True
    state[EVIDENCE_RECOVERY_AUTO_RESUME_KEY] = True
    state[FINAL_DELIVERY_AUTO_RESUME_KEY] = True
    # The sidebar widget already exists when the V0.5 button is clicked.  Defer
    # the navigation mutation to the next run to avoid StreamlitAPIException.
    state["mvp_navigation_request"] = "writing"
    state["mvp_flash"] = (
        "已保留需求、文献、PDF、证据库和写作计划，并退回 V0.4。"
        "Agent 将重新生成全部正文、重新审稿和全文编辑。旧终稿保存在返修检查点中。"
    )
    return True


def render_grounded_writing_console(
    handoff: V04WritingHandoff,
    *,
    include_final_delivery: bool = True,
) -> None:
    """Render the staged V0.4 body-writing workflow."""

    inspection_raw = st.session_state.get("v03_pdf_inspection_json")
    if inspection_raw:
        try:
            inspection_version = json.loads(inspection_raw).get("schema_version")
        except (TypeError, json.JSONDecodeError):
            inspection_version = None
        if inspection_version != "0.3.2":
            st.session_state["mvp_navigation_request"] = "evidence"
            st.session_state["mvp_flash"] = (
                "检测到旧版 PDF 身份缓存；系统将自动回到 V0.3 用首页题名/DOI "
                "重新核验，避免把正文术语相似的其他论文绑定为核心全文。"
            )
            st.rerun()

    st.divider()
    st.header("V0.4 Agent 论文生成")
    st.caption(
        "一次启动后，Agent 会自动规划、逐章写作、独立审稿、定点返修、"
        "全文编辑、合规审计并生成内部质量评分。"
    )
    if not _render_topic_admission_gate(handoff):
        return

    _restore_unaffected_recovery_progress(handoff)
    st.session_state[WRITING_MODE_KEY] = "AI 全托管"
    existing_project: V04WritingProject | None = None
    serialized_project = st.session_state.get(V04_PROJECT_KEY)
    if serialized_project:
        try:
            candidate = V04WritingProject.model_validate_json(serialized_project)
        except (TypeError, ValueError):
            candidate = None
        if candidate is not None and candidate.handoff == handoff:
            existing_project = candidate
    _render_agent_activity(existing_project)
    _render_agent_pause_control()

    if not st.session_state.get(V04_AUTOPILOT_REQUESTED_KEY):
        button_label = (
            "继续生成成品论文"
            if existing_project is not None
            and existing_project.status == "body_complete"
            else (
                "继续 Agent 写作"
                if existing_project is not None
                or st.session_state.get(V04_AGENT_RUN_ID_KEY)
                else "开始生成论文"
            )
        )
        st.caption(
            "开始后无需逐章点击。只有付费 PDF 无法自动获取、需求发生关键冲突，"
            "或最终成品需要确认时，系统才会请求你的操作。"
        )
        if st.button(
            button_label,
            type="primary",
            width="stretch",
            key="v04_start_agent_paper",
        ):
            st.session_state[V04_AUTOPILOT_REQUESTED_KEY] = True
            st.session_state.pop(WRITING_PLAN_AUTO_FAILURES_KEY, None)
            st.session_state[EVIDENCE_RECOVERY_AUTO_PLAN_KEY] = True
            st.session_state[EVIDENCE_RECOVERY_AUTO_RESUME_KEY] = True
            st.session_state[FINAL_DELIVERY_AUTO_RESUME_KEY] = True
            _autosave_current_project()
            st.rerun()
        return

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
    project, normalized_reviews = _normalize_nonblocking_quality_findings(project)
    if normalized_reviews:
        _store_project(project)
        st.info(
            f"已将 {normalized_reviews} 个仅含普通建议的旧审稿结果恢复为可采用状态；"
            "普通提醒不再阻塞写作。"
        )
    project, quality_reopened = _revalidate_project_quality(project)
    if quality_reopened:
        _store_project(project)
        st.session_state.pop(FINAL_MATTER_KEY, None)
        st.session_state.pop(FINAL_PACKAGE_KEY, None)
        st.warning(
            f"检测到 {quality_reopened} 个旧章节在独立审稿仍有问题时被误标为已采用；"
            "系统已撤销这些章节的确认，正文原文仍保留，只需定点修订问题段落。"
        )
    recovery_request = _exhausted_evidence_recovery_request(project, writing_plan)
    if recovery_request is not None:
        if _begin_evidence_recovery(project, writing_plan, recovery_request):
            st.rerun()
        st.error(st.session_state.get("mvp_flash", "证据恢复已停止。"))
        return
    confirmed_count = sum(state.status == "confirmed" for state in project.sections)
    st.progress(confirmed_count / len(project.sections))
    st.caption(
        f"已完成 {confirmed_count}/{len(project.sections)} 章"
        + (" · 正文可汇总" if project.status == "body_complete" else "")
    )
    # One paragraph may carry several editor findings.  The user-facing repair
    # scope is the number of paragraphs that will be rewritten, not the number
    # of findings attached to those paragraphs.
    repair_paragraphs = {
        (section.section_id, issue.paragraph_number)
        for section in project.sections
        if section.draft is not None
        for issue in section.draft.issues
        if issue.code == "final_audit_repair" and issue.paragraph_number is not None
    }
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
        if _resolve_evidence_recovery():
            _store_project(project)
        _render_body_download(project, include_final_delivery=include_final_delivery)
        return
    project = _render_continuous_writing_control(project, writing_plan)
    if project.status == "body_complete":
        _render_body_download(
            project,
            include_final_delivery=include_final_delivery,
        )


def _resume_after_evidence_recovery(
    handoff: V04WritingHandoff,
    generated_plan: GroundedWritingPlan,
) -> None:
    """Adopt a repaired plan while retaining compatible confirmed chapters."""

    checkpoint_raw = st.session_state.get(EVIDENCE_RECOVERY_CHECKPOINT_KEY)
    request_raw = st.session_state.get(EVIDENCE_RECOVERY_REQUEST_KEY)
    if not checkpoint_raw or not request_raw:
        return
    checkpoint = json.loads(checkpoint_raw)
    request = WritingEvidenceRecoveryRequest.model_validate_json(request_raw)
    old_plan = GroundedWritingPlan.model_validate_json(
        checkpoint["writing_plan_json"]
    )
    old_project = V04WritingProject.model_validate_json(
        checkpoint["writing_project_json"]
    )
    handoff = merge_recovery_handoffs(
        old_project.handoff,
        handoff,
        affected_section_ids=set(request.affected_section_ids),
    )
    old_plans = {section.section_id: section for section in old_plan.sections}
    old_states = {state.section_id: state for state in old_project.sections}
    required_source_dois = list(generated_plan.required_source_dois)
    merged_sections: list[WritingSectionPlan] = []
    preserved_ids: set[str] = set()
    for new_section in generated_plan.sections:
        old_section = old_plans.get(new_section.section_id)
        old_state = old_states.get(new_section.section_id)
        if (
            new_section.section_id not in request.affected_section_ids
            and old_section is not None
            and old_state is not None
            and old_state.status == "confirmed"
            and _recovered_state_is_compatible(old_state, old_section, handoff)
        ):
            merged_sections.append(old_section)
            preserved_ids.add(new_section.section_id)
        else:
            merged_sections.append(new_section)

    merged_sections, preserved_ids = _reopen_minimum_sections_for_source_coverage(
        reference_sections=generated_plan.sections,
        merged_sections=merged_sections,
        preserved_section_ids=preserved_ids,
        required_source_dois=required_source_dois,
    )

    merged_draft = GroundedWritingPlan.model_validate(
        generated_plan.model_copy(
            update={
                "status": "draft",
                "required_source_dois": required_source_dois,
                "sections": merged_sections,
                "confirmed_by": None,
                "confirmed_at": None,
            }
        ).model_dump(mode="json")
    )
    coverage_repair = repair_writing_plan_source_coverage(
        handoff,
        merged_draft,
        required_source_dois=required_source_dois,
    )
    merged_draft = coverage_repair.plan.model_copy(
        update={
            "status": "draft",
            "confirmed_by": None,
            "confirmed_at": None,
        }
    )
    preserved_ids.difference_update(coverage_repair.changed_paragraph_numbers)
    confirmed_plan = GroundedWritingPlan.model_validate(
        merged_draft.model_dump(mode="json")
    ).confirm(confirmed_by=handoff.requirement.confirmed_by)
    new_project = WritingProjectService().start(handoff)
    repaired_sections = {
        section.section_id: section for section in merged_draft.sections
    }
    merged_states: list[WritingSectionState] = []
    for state in new_project.sections:
        if state.section_id in preserved_ids:
            merged_states.append(old_states[state.section_id])
            continue
        old_state = old_states.get(state.section_id)
        old_section = old_plans.get(state.section_id)
        reopened = None
        if (
            state.section_id not in request.affected_section_ids
            and old_state is not None
            and old_section is not None
            and old_state.status == "confirmed"
        ):
            reopened = _reopen_confirmed_state_for_plan_changes(
                old_state,
                previous_plan=old_section,
                current_plan=repaired_sections[state.section_id],
                handoff=handoff,
            )
        merged_states.append(reopened or state)
    resumed_project = V04WritingProject.model_validate(
        new_project.model_copy(
            update={
                "status": (
                    "body_complete"
                    if all(state.status == "confirmed" for state in merged_states)
                    else "drafting"
                ),
                "sections": merged_states,
                "updated_at": datetime.now(timezone.utc),
            }
        ).model_dump(mode="json")
    )
    st.session_state[WRITING_PLAN_KEY] = confirmed_plan.model_dump_json(indent=2)
    st.session_state[V04_PROJECT_KEY] = resumed_project.model_dump_json(indent=2)
    st.session_state["v03_writing_handoff_json"] = handoff.model_dump_json(indent=2)
    st.session_state[EVIDENCE_RECOVERY_REQUEST_KEY] = request.model_copy(
        update={"status": "ready_to_resume"}
    ).model_dump_json(indent=2)
    st.session_state[EVIDENCE_RECOVERY_AUTO_RESUME_KEY] = True
    st.session_state[WRITING_MODE_KEY] = "AI 全托管"
    st.session_state.pop(WRITING_PLAN_REPAIR_FEEDBACK_KEY, None)
    recovery_label = "证据已补齐并重新规划" if request.gaps else "审稿意见已转为规划约束"
    st.session_state["mvp_flash"] = (
        f"{recovery_label}；保留 {len(preserved_ids)} 个已确认章节，"
        f"现在只重写 {len(request.affected_section_ids)} 个受影响章节。"
    )


def _reopen_minimum_sections_for_source_coverage(
    *,
    reference_sections: list[WritingSectionPlan],
    merged_sections: list[WritingSectionPlan],
    preserved_section_ids: set[str],
    required_source_dois: list[str],
) -> tuple[list[WritingSectionPlan], set[str]]:
    """Keep the largest safe set of accepted sections under the new DOI contract."""

    required = set(required_source_dois)

    def covered(sections: list[WritingSectionPlan]) -> set[str]:
        return {
            doi
            for section in sections
            for paragraph in section.paragraphs
            for doi in paragraph.source_dois
        }

    preserved = set(preserved_section_ids)
    if required <= covered(merged_sections):
        return merged_sections, preserved
    reference_by_id = {section.section_id: section for section in reference_sections}
    reopenable = sorted(
        section_id for section_id in preserved if section_id in reference_by_id
    )
    for reopen_count in range(1, len(reopenable) + 1):
        for reopened in combinations(reopenable, reopen_count):
            reopened_ids = set(reopened)
            candidate = [
                reference_by_id[section.section_id]
                if section.section_id in reopened_ids
                else section
                for section in merged_sections
            ]
            if required <= covered(candidate):
                return candidate, preserved.difference(reopened_ids)
    raise ValueError(
        "recovery reference plan does not cover every required source DOI"
    )


def _reopen_confirmed_state_for_plan_changes(
    state: WritingSectionState,
    *,
    previous_plan: WritingSectionPlan,
    current_plan: WritingSectionPlan,
    handoff: V04WritingHandoff,
) -> WritingSectionState | None:
    """Retain accepted prose and reopen only paragraphs whose authority changed."""

    if state.draft is None or len(previous_plan.paragraphs) != len(
        current_plan.paragraphs
    ):
        return None
    changed = [
        current.paragraph_number
        for previous, current in zip(
            previous_plan.paragraphs,
            current_plan.paragraphs,
            strict=True,
        )
        if _paragraph_requires_rewrite(previous, current)
    ]
    try:
        packet = SectionEvidencePacketBuilder().build(handoff, state.section_id)
    except Exception:
        return None
    packet_evidence = {item.evidence_id for item in packet.evidence_items}
    packet_sources = {source.doi for source in packet.sources}
    changed_set = set(changed)
    for draft_paragraph, current in zip(
        state.draft.paragraphs,
        current_plan.paragraphs,
        strict=True,
    ):
        if current.paragraph_number in changed_set:
            continue
        if (
            draft_paragraph.role != current.role
            or draft_paragraph.evidence_card_ids != current.evidence_card_ids
            or draft_paragraph.source_dois != current.source_dois
            or not set(draft_paragraph.evidence_card_ids).issubset(packet_evidence)
            or not set(draft_paragraph.source_dois).issubset(packet_sources)
        ):
            return None
    if not changed:
        return state
    retained = [
        issue for issue in state.draft.issues if issue.code != "final_audit_repair"
    ]
    repair_issues = [
        SectionDraftIssue(
            code="final_audit_repair",
            severity="blocking",
            paragraph_number=number,
            detail=(
                "补证后的来源权限或必引绑定改变；只按新计划重写本段，"
                "本章其他已确认段落保持不变。"
            ),
        )
        for number in changed
    ]
    reopened_draft = state.draft.model_copy(
        update={
            "status": "needs_review",
            "issues": [*retained, *repair_issues],
            "confirmed_by": None,
            "confirmed_at": None,
            "quality_review_status": "not_run",
            "quality_review_rounds": 0,
            "quality_reviewed_at": None,
        }
    )
    return WritingSectionState(
        section_id=state.section_id,
        status="needs_review",
        draft=reopened_draft,
    )


def _restore_unaffected_recovery_progress(handoff: V04WritingHandoff) -> bool:
    """Restore compatible unaffected chapters from an evidence-recovery checkpoint."""

    checkpoint_raw = st.session_state.get(EVIDENCE_RECOVERY_CHECKPOINT_KEY)
    request_raw = st.session_state.get(EVIDENCE_RECOVERY_REQUEST_KEY)
    current_plan_raw = st.session_state.get(WRITING_PLAN_KEY)
    current_project_raw = st.session_state.get(V04_PROJECT_KEY)
    if not all((checkpoint_raw, request_raw, current_plan_raw, current_project_raw)):
        return False
    try:
        checkpoint = json.loads(checkpoint_raw)
        request = WritingEvidenceRecoveryRequest.model_validate_json(request_raw)
        old_plan = GroundedWritingPlan.model_validate_json(
            checkpoint["writing_plan_json"]
        )
        old_project = V04WritingProject.model_validate_json(
            checkpoint["writing_project_json"]
        )
        current_plan = GroundedWritingPlan.model_validate_json(current_plan_raw)
        current_project = V04WritingProject.model_validate_json(current_project_raw)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    if request.status != "ready_to_resume":
        return False

    handoff = merge_recovery_handoffs(
        old_project.handoff,
        handoff,
        affected_section_ids=set(request.affected_section_ids),
    )

    old_sections = {section.section_id: section for section in old_plan.sections}
    old_states = {state.section_id: state for state in old_project.sections}
    current_states = {state.section_id: state for state in current_project.sections}
    affected = set(request.affected_section_ids)
    preserved: set[str] = set()
    merged_sections: list[WritingSectionPlan] = []
    for section in current_plan.sections:
        old_section = old_sections.get(section.section_id)
        old_state = old_states.get(section.section_id)
        current_state = current_states.get(section.section_id)
        can_restore = (
            section.section_id not in affected
            and old_section is not None
            and old_state is not None
            and old_state.status == "confirmed"
            and (current_state is None or current_state.status != "confirmed")
            and _recovered_state_is_compatible(old_state, old_section, handoff)
        )
        if can_restore:
            merged_sections.append(old_section)
            preserved.add(section.section_id)
        else:
            merged_sections.append(section)
    if not preserved:
        return False

    required_source_dois = list(current_plan.required_source_dois)
    merged_sections, preserved = _reopen_minimum_sections_for_source_coverage(
        reference_sections=current_plan.sections,
        merged_sections=merged_sections,
        preserved_section_ids=preserved,
        required_source_dois=required_source_dois,
    )
    candidate = GroundedWritingPlan.model_validate(
        current_plan.model_copy(
            update={
                "required_source_dois": required_source_dois,
                "sections": merged_sections,
            }
        ).model_dump(mode="json")
    )
    coverage_repair = repair_writing_plan_source_coverage(
        handoff,
        candidate,
        required_source_dois=required_source_dois,
    )
    preserved.difference_update(coverage_repair.changed_paragraph_numbers)
    repaired_sections = {
        section.section_id: section for section in coverage_repair.plan.sections
    }
    pending_states = {
        state.section_id: state
        for state in WritingProjectService().start(handoff).sections
    }
    merged_states: list[WritingSectionState] = []
    for section_id, section_plan in repaired_sections.items():
        old_state = old_states.get(section_id)
        current_state = current_states.get(section_id)
        if section_id in preserved and old_state is not None:
            merged_states.append(old_state)
        elif (
            current_state is not None
            and current_state.status != "pending"
            and _draft_matches_plan(current_state, section_plan)
            and _recovered_state_is_compatible(current_state, section_plan, handoff)
        ):
            merged_states.append(current_state)
        else:
            merged_states.append(pending_states[section_id])
    resumed_project = V04WritingProject.model_validate(
        current_project.model_copy(
            update={
                "status": (
                    "body_complete"
                    if all(state.status == "confirmed" for state in merged_states)
                    else "drafting"
                ),
                "handoff": handoff,
                "sections": merged_states,
                "updated_at": datetime.now(timezone.utc),
            }
        ).model_dump(mode="json")
    )
    st.session_state[WRITING_PLAN_KEY] = coverage_repair.plan.model_dump_json(indent=2)
    st.session_state[V04_PROJECT_KEY] = resumed_project.model_dump_json(indent=2)
    st.session_state["v03_writing_handoff_json"] = handoff.model_dump_json(indent=2)
    st.session_state[SECTION_SELECTION_REQUEST_KEY] = _next_actionable_section(
        resumed_project
    )
    st.session_state["mvp_flash"] = (
        f"恢复检查点已重新合并：保留 {len(preserved)} 个未受影响的已确认章节，"
        f"只继续处理 {len(affected)} 个证据或规划受影响章节。"
    )
    _autosave_current_project()
    return True


def _recovered_state_is_compatible(
    state: WritingSectionState,
    section_plan: WritingSectionPlan,
    handoff: V04WritingHandoff,
) -> bool:
    if not _draft_matches_plan(state, section_plan):
        return False
    try:
        packet = SectionEvidencePacketBuilder().build(handoff, state.section_id)
    except Exception:
        return False
    packet_evidence = {item.evidence_id for item in packet.evidence_items}
    packet_sources = {source.doi for source in packet.sources}
    return bool(state.draft) and all(
        set(paragraph.evidence_card_ids).issubset(packet_evidence)
        and set(paragraph.source_dois).issubset(packet_sources)
        for paragraph in state.draft.paragraphs
    )

def render_final_delivery_console(handoff: V04WritingHandoff) -> None:
    """Render final assembly independently from the V0.4 section workbench."""

    inspection_raw = st.session_state.get("v03_pdf_inspection_json")
    if inspection_raw:
        try:
            inspection_version = json.loads(inspection_raw).get("schema_version")
        except (TypeError, json.JSONDecodeError):
            inspection_version = None
        if inspection_version != "0.3.2":
            st.session_state["mvp_navigation_request"] = "evidence"
            st.session_state["mvp_flash"] = (
                "PDF 身份核验规则已升级：系统将自动返回 V0.3，重新检查首页 DOI，"
                "不会沿用可能错绑的证据卡、引文或最终评分。"
            )
            st.rerun()

    st.divider()
    st.header("V0.5 Agent 全文编辑与交付")
    if not _render_topic_admission_gate(handoff):
        st.session_state.pop(FINAL_MATTER_KEY, None)
        st.session_state.pop(FINAL_PACKAGE_KEY, None)
        return
    project = _load_or_start_project(handoff)
    _render_agent_activity(project)
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
        if _legacy_coverage_count(plan):
            st.error(
                "当前正文来自旧版“文献覆盖段落”计划，不能直接进入最终交付；"
                "请回到 V0.4 重建问题驱动的写作计划。"
            )
            st.session_state["mvp_navigation_request"] = "writing"
            st.session_state["mvp_flash"] = (
                "检测到旧版覆盖型计划，已退回 V0.4；原正文仍保留在检查点前状态。"
            )
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
        project, normalized_reviews = _normalize_nonblocking_quality_findings(project)
        if normalized_reviews:
            _store_project(project)
        project, quality_reopened = _revalidate_project_quality(project)
        if quality_reopened:
            _store_project(project)
            st.session_state.pop(FINAL_MATTER_KEY, None)
            st.session_state.pop(FINAL_PACKAGE_KEY, None)
            first_reopened = next(
                state.section_id
                for state in project.sections
                if state.status != "confirmed"
            )
            st.session_state[SECTION_SELECTION_REQUEST_KEY] = first_reopened
            st.session_state["mvp_navigation_request"] = "writing"
            st.session_state["mvp_flash"] = (
                f"检测到 {quality_reopened} 个章节尚未通过独立审稿；"
                "已退回 V0.4 定点修订，原正文和合格章节保持不变。"
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
    serialized = st.session_state.get(WRITING_PLAN_KEY)
    if not serialized:
        if _resume_editorial_evidence_checkpoint_without_search():
            st.rerun()
        st.subheader("写作计划")
        st.caption(
            "系统根据检索蓝图和实际证据，为每个段落分配论点、字数和证据。"
        )
        recovery_raw = st.session_state.get(EVIDENCE_RECOVERY_REQUEST_KEY)
        auto_failures = int(
            st.session_state.get(WRITING_PLAN_AUTO_FAILURES_KEY, 0) or 0
        )
        if (
            auto_failures >= 2
            and recovery_raw
            and not st.session_state.get(
                WRITING_PLAN_PERMISSION_REPAIR_MIGRATION_KEY,
                False,
            )
        ):
            try:
                recovery_status = json.loads(recovery_raw).get("status")
            except (TypeError, json.JSONDecodeError):
                recovery_status = None
            if recovery_status == "resolved":
                # Older checkpoints could mark evidence recovery resolved even though
                # the semantic planner repeated a source-permission mismatch twice.
                # The current planner repairs that mismatch deterministically, so grant
                # exactly one fresh automatic attempt after upgrading the runtime.
                auto_failures = 0
                st.session_state.pop(WRITING_PLAN_AUTO_FAILURES_KEY, None)
                st.session_state[WRITING_PLAN_PERMISSION_REPAIR_MIGRATION_KEY] = True
                st.session_state[EVIDENCE_RECOVERY_AUTO_PLAN_KEY] = True
                st.session_state["mvp_flash"] = (
                    "检测到旧版写作计划因来源权限分配中断；系统已升级确定性修复"
                    "策略，将从现有检查点自动继续，不会重做已完成章节。"
                )
        auto_plan = _consume_writing_plan_auto_request(
            st.session_state,
            auto_failures=auto_failures,
        )
        if (
            st.session_state.get(V04_AUTOPILOT_REQUESTED_KEY)
            and auto_failures >= 2
        ):
            st.error(
                "Agent 连续两次未能生成满足数据合同的写作计划，已停止自动重试。"
                "这通常说明证据边界、章节要求或规划合同冲突；已完成缓存不会丢失。"
            )
            return None
        manual_plan_request = False
        if not auto_plan:
            manual_plan_request = st.button(
                "根据实际证据生成写作计划",
                type="primary",
                width="stretch",
                key="v04_generate_writing_plan",
            )
        if manual_plan_request or auto_plan:
            force = bool(
                st.session_state.pop("v04_force_writing_plan_regeneration", False)
            )
            try:
                with st.spinner(
                    "正在逐章规划段落与证据；已完成章节会保存为检查点……"
                ):
                    planning_settings = LLMSettings().for_structured_output()
                    planning_settings = planning_settings.model_copy(
                        update={
                            "timeout_seconds": min(
                                planning_settings.timeout_seconds,
                                60.0,
                            ),
                            "max_retries": 0,
                        }
                    )
                    plan = GroundedWritingPlanner(
                        DeepSeekClient(planning_settings),
                        cache=WritingPlanRuntimeCache(
                            project_root() / "runtime" / "writing_plan",
                            handoff=handoff,
                        ),
                        reuse_cache=not force,
                        repair_feedback_by_section=_writing_plan_repair_feedback(),
                        max_elapsed_seconds=300.0,
                        max_model_calls=max(
                            1,
                            len(handoff.outline.outline.sections),
                        ),
                    ).plan(handoff)
            except WritingPlanBudgetExceeded:
                pause_writing_agent(st.session_state)
                _autosave_current_project()
                st.error(
                    "本轮写作规划已达到 5 分钟预算，系统已暂停并保留所有"
                    "成功章节缓存。再次继续时只会处理尚未完成的规划节点。"
                )
                return None
            except WritingPlanDependencyError:
                pause_writing_agent(st.session_state)
                _autosave_current_project()
                st.error(
                    "写作计划发现无法由改写解决的来源权限冲突，系统已停止"
                    "自动重试并保留全部规划缓存。该问题需要调整证据路由或"
                    "课程引用策略，不会继续消耗模型调用。"
                )
                return None
            except Exception as exc:
                if auto_plan:
                    failures = int(
                        st.session_state.get(WRITING_PLAN_AUTO_FAILURES_KEY, 0)
                    ) + 1
                    st.session_state[WRITING_PLAN_AUTO_FAILURES_KEY] = failures
                    if failures < 2:
                        st.session_state[EVIDENCE_RECOVERY_AUTO_PLAN_KEY] = True
                        st.session_state["mvp_flash"] = (
                            "写作计划第一次未通过合同检查；系统正在保留已完成章节并"
                            "自动重试，不需要用户点击。"
                        )
                        st.rerun()
                st.error(
                    "写作计划尚未生成完成。已完成章节已经保存，重试时会继续。"
                )
                with st.expander("技术详情"):
                    st.code(str(exc))
            else:
                st.session_state.pop(WRITING_PLAN_AUTO_FAILURES_KEY, None)
                if st.session_state.get(EVIDENCE_RECOVERY_CHECKPOINT_KEY):
                    try:
                        _resume_after_evidence_recovery(handoff, plan)
                    except Exception as exc:
                        pause_writing_agent(st.session_state)
                        st.session_state.pop(WRITING_PLAN_KEY, None)
                        _autosave_current_project()
                        st.error(
                            "补证后的写作计划尚未能安全接回原项目。系统已暂停，"
                            "恢复检查点和已生成的规划缓存均已保留。"
                        )
                        with st.expander("技术详情"):
                            st.code(str(exc))
                        return None
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
    try:
        authority_repair = rebase_writing_plan_authority(handoff, plan)
    except WritingPlanDependencyError as exc:
        pause_writing_agent(st.session_state)
        _autosave_current_project()
        st.error(
            "V0.3 证据交接已变化，但现有写作计划中有段落失去了唯一支撑。"
            "系统已停止正文调用并保留检查点；下一步应只重规划受影响段落。"
        )
        with st.expander("技术详情"):
            st.code(str(exc))
        return None
    cached_project_raw = st.session_state.get(V04_PROJECT_KEY)
    cached_project = None
    if cached_project_raw:
        try:
            cached_project = V04WritingProject.model_validate_json(
                cached_project_raw
            )
        except (TypeError, ValueError):
            cached_project = None
    authority_changed = authority_repair.plan != plan
    handoff_changed = bool(
        cached_project is not None and cached_project.handoff != handoff
    )
    if authority_changed or handoff_changed:
        previous_plan = plan
        plan = authority_repair.plan
        st.session_state[WRITING_PLAN_KEY] = plan.model_dump_json(indent=2)
        if cached_project is not None:
            synchronized = _synchronize_project_handoff(
                cached_project,
                previous_plan=previous_plan,
                current_plan=plan,
                handoff=handoff,
            )
            st.session_state[V04_PROJECT_KEY] = synchronized.model_dump_json(indent=2)
        st.session_state.pop(FINAL_MATTER_KEY, None)
        st.session_state.pop(FINAL_PACKAGE_KEY, None)
        _autosave_current_project()
        st.session_state["mvp_flash"] = (
            "V0.3 证据交接已与写作计划重新对齐；兼容章节和段落均已保留，"
            "仅来源权限实际变化的段落会重新生成。"
        )
        st.rerun()
    legacy_coverage_count = _legacy_coverage_count(plan)
    policy = handoff.requirement_policy or RequirementPolicyCompiler().compile(
        handoff.requirement
    )
    legacy_forced_sources = bool(
        plan.required_source_dois
        and not policy.references.all_bibliography_items_must_be_cited_and_discussed
    )
    if legacy_coverage_count or legacy_forced_sources:
        st.error(
            f"检测到 {legacy_coverage_count} 个旧版覆盖段落，或旧计划将普通文献数量要求"
            "错误解释成了逐篇强制讨论。"
            "这类计划会污染论证，必须重建，原 V0.3 证据库和本地运行文件会保留。"
        )
        auto_replace = bool(st.session_state.get(V04_AUTOPILOT_REQUESTED_KEY))
        replace_requested = auto_replace
        if not auto_replace:
            replace_requested = st.button(
                "废止旧计划并重新生成问题驱动提纲",
                type="primary",
                width="stretch",
                key="replace_legacy_coverage_plan",
            )
        if replace_requested:
            _create_final_repair_checkpoint(
                ["legacy_coverage_plan"],
                replace=True,
            )
            st.session_state.pop(WRITING_PLAN_KEY, None)
            st.session_state.pop(V04_PROJECT_KEY, None)
            st.session_state.pop(FINAL_MATTER_KEY, None)
            st.session_state.pop(FINAL_PACKAGE_KEY, None)
            st.session_state["v04_force_writing_plan_regeneration"] = True
            st.session_state["mvp_flash"] = (
                "旧版覆盖型计划已保存到可撤销检查点；现在将按中心问题重新规划。"
            )
            st.rerun()
        return None
    if plan.status == "draft":
        if st.session_state.get(EVIDENCE_RECOVERY_CHECKPOINT_KEY):
            _resume_after_evidence_recovery(handoff, plan)
            st.rerun()
        auto_confirm = bool(st.session_state.get(V04_AUTOPILOT_REQUESTED_KEY))
        st.subheader("写作计划已生成")
        if not auto_confirm:
            _render_writing_plan_summary(plan)
        existing_project = st.session_state.get(V04_PROJECT_KEY)
        if existing_project and not auto_confirm:
            st.warning(
                "采用新计划后会重建当前尚未确认的 V0.4 草稿；V0.3 证据库和本地"
                "运行缓存不会删除。"
            )
        confirm_requested = auto_confirm
        if not auto_confirm:
            left, right = st.columns(2)
            confirm_requested = left.button(
                "采用计划并开始逐段写作",
                type="primary",
                width="stretch",
                key="v04_confirm_writing_plan",
            )
        if confirm_requested:
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
        if not auto_confirm:
            if right.button(
                "重新规划",
                width="stretch",
                key="v04_regenerate_writing_plan",
            ):
                st.session_state.pop(WRITING_PLAN_KEY, None)
                st.session_state["v04_force_writing_plan_regeneration"] = True
                st.rerun()
        return None

    st.caption(
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
    source_summary = (
        f"必引来源覆盖 {len(required_source_dois & covered_source_dois)}/"
        f"{len(required_source_dois)}"
        if required_source_dois
        else f"计划使用 {len(covered_source_dois)} 篇来源"
    )
    st.caption(
        f"{len(plan.sections)} 个章节 · "
        f"{sum(len(section.paragraphs) for section in plan.sections)} 个段落 · "
        f"目标 {sum(section.target_words for section in plan.sections)} 字 · "
        f"{source_summary}"
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
        serialized_plan = st.session_state.get(WRITING_PLAN_KEY)
        if serialized_plan:
            previous_plan = GroundedWritingPlan.model_validate_json(serialized_plan)
            current_plan = rebase_writing_plan_authority(
                handoff,
                previous_plan,
            ).plan
            synchronized = _synchronize_project_handoff(
                project,
                previous_plan=previous_plan,
                current_plan=current_plan,
                handoff=handoff,
            )
            st.session_state[WRITING_PLAN_KEY] = current_plan.model_dump_json(indent=2)
            _store_project(synchronized)
            return synchronized
    project = WritingProjectService().start(handoff)
    _store_project(project)
    return project


def _synchronize_project_handoff(
    project: V04WritingProject,
    *,
    previous_plan: GroundedWritingPlan,
    current_plan: GroundedWritingPlan,
    handoff: V04WritingHandoff,
) -> V04WritingProject:
    """Adopt a new evidence handoff without discarding compatible writing progress."""

    pending = {
        state.section_id: state
        for state in WritingProjectService().start(handoff).sections
    }
    old_states = {state.section_id: state for state in project.sections}
    old_plans = {section.section_id: section for section in previous_plan.sections}
    states: list[WritingSectionState] = []
    for section in current_plan.sections:
        state = old_states.get(section.section_id)
        previous = old_plans.get(section.section_id)
        if state is None or previous is None or state.draft is None:
            states.append(pending[section.section_id])
            continue
        if (
            _draft_matches_plan(state, section)
            and _recovered_state_is_compatible(state, section, handoff)
        ):
            states.append(state)
            continue
        reopened = _reopen_confirmed_state_for_plan_changes(
            state,
            previous_plan=previous,
            current_plan=section,
            handoff=handoff,
        )
        states.append(reopened or pending[section.section_id])
    synchronized = project.model_copy(
        update={
            "handoff": handoff,
            "status": (
                "body_complete"
                if all(state.status == "confirmed" for state in states)
                else "drafting"
            ),
            "sections": states,
            "updated_at": datetime.now(timezone.utc),
        }
    )
    return V04WritingProject.model_validate(
        synchronized.model_dump(mode="json")
    )


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


def _revalidate_project_quality(
    project: V04WritingProject,
) -> tuple[V04WritingProject, int]:
    """Reopen legacy confirmations that bypassed the independent reviewer."""

    states: list[WritingSectionState] = []
    reopened_count = 0
    for state in project.sections:
        draft = state.draft
        if (
            draft is None
            or state.status != "confirmed"
            or draft.quality_review_status == "passed"
        ):
            states.append(state)
            continue
        reopened_count += 1
        reopened = draft.model_copy(
            update={
                "status": "draft",
                "confirmed_by": None,
                "confirmed_at": None,
            }
        )
        states.append(
            WritingSectionState(
                section_id=state.section_id,
                status="draft",
                draft=reopened,
            )
        )
    if not reopened_count:
        return project, 0
    revised = project.model_copy(
        update={
            "status": "drafting",
            "sections": states,
            "updated_at": datetime.now(timezone.utc),
        }
    )
    return V04WritingProject.model_validate(revised.model_dump(mode="json")), reopened_count


def _render_topic_admission_gate(handoff: V04WritingHandoff) -> bool:
    """Keep legacy or out-of-scope literature out of every writing surface."""

    policy = handoff.requirement_policy or RequirementPolicyCompiler().compile(
        handoff.requirement
    )
    audit = audit_topic_admission(
        handoff.evidence_library,
        policy,
        valid_section_ids=(
            section.section_id for section in handoff.outline.outline.sections
        ),
    )
    if audit.passed:
        return True
    st.error(
        "当前文献库没有通过主题准入门禁，不能继续写作或交付。"
        "旧流程只验证了 DOI 真实性，未确认每篇文献具体支撑什么论点、用于哪一章、"
        "以及不得扩展到什么范围。"
    )
    st.caption(
        f"技术摘要：{audit.detail}。已生成的原文和本地 PDF 不会在此处删除；"
        "请返回 V0.2，按立题卡重新执行相关性准入。"
    )
    if st.button(
        "返回 V0.2 重新执行文献准入",
        type="primary",
        width="stretch",
        key="v04_route_to_topic_admission",
    ):
        st.session_state["mvp_navigation_request"] = "literature"
        st.session_state["mvp_flash"] = (
            "已定位到 V0.2 文献准入；重新检索不会直接删除 runtime 中的 PDF 文件。"
        )
        st.rerun()
    return False


def _store_project(project: V04WritingProject) -> None:
    st.session_state[V04_PROJECT_KEY] = project.model_dump_json(indent=2)
    _autosave_current_project()


def _autosave_current_project() -> bool:
    """Persist the active snapshot with the same path policy used by the app."""

    autosave_override = os.getenv("VERIWRITE_AUTOSAVE_PATH")
    store = LocalProjectStore(
        Path(autosave_override)
        if autosave_override
        else project_root() / "runtime" / "mvp_projects" / "active_project.json"
    )
    try:
        return autosave_local_project(st.session_state, store)
    except (OSError, ValueError) as exc:
        st.session_state["mvp_flash"] = (
            "正文检查点已保存在当前会话，但本地项目存档暂时写入失败："
            f"{exc}"
        )
        return False


def _writing_agent_runtime(
    project: V04WritingProject,
    plan: GroundedWritingPlan,
) -> tuple[WritingAgentRuntimeService, WritingAgentContext]:
    """Load one durable Agent run without putting its event history in session_state."""

    policy = project.handoff.requirement_policy
    if policy is None:
        policy = RequirementPolicyCompiler().compile(project.handoff.requirement)
    run_id = st.session_state.get(V04_AGENT_RUN_ID_KEY)
    if not isinstance(run_id, str) or not re.fullmatch(r"run_[0-9a-f]{16}", run_id):
        run_id = f"run_{uuid4().hex[:16]}"
        st.session_state[V04_AGENT_RUN_ID_KEY] = run_id
        _autosave_current_project()
    runtime = WritingAgentRuntimeService(
        AgentRuntimeStore(project_root() / "runtime" / "agent_runs" / run_id)
    )
    context = runtime.initialize(
        run_id=run_id,
        project_id=f"paper_{policy.requirement_fingerprint[:16]}",
        policy=policy,
        handoff=project.handoff,
        plan=plan,
        project=project,
    )
    return runtime, context


def _normalize_nonblocking_quality_findings(
    project: V04WritingProject,
) -> tuple[V04WritingProject, int]:
    """Migrate old reviews that incorrectly treated every warning as a failure."""

    states: list[WritingSectionState] = []
    normalized_count = 0
    for state in project.sections:
        draft = state.draft
        if draft is None or draft.quality_review_status != "findings":
            states.append(state)
            continue
        retained_issues = []
        removed_false_attribution = False
        for issue in draft.issues:
            if (
                issue.code == "false_self_attribution"
                and issue.paragraph_number is not None
                and issue.paragraph_number <= len(draft.paragraphs)
                and false_self_attribution_detail(
                    draft.paragraphs[issue.paragraph_number - 1].text
                )
                is None
            ):
                removed_false_attribution = True
                continue
            retained_issues.append(issue)
        has_blocking_editorial = any(
            issue.severity == "blocking" and issue.code in EDITORIAL_REPAIR_CODES
            for issue in retained_issues
        )
        if has_blocking_editorial:
            if removed_false_attribution:
                sanitized = draft.model_copy(update={"issues": retained_issues})
                states.append(state.model_copy(update={"draft": sanitized}))
                normalized_count += 1
            else:
                states.append(state)
            continue
        normalized_count += 1
        normalized = draft.model_copy(
            update={
                "status": "draft",
                "issues": retained_issues,
                "quality_review_status": "passed",
            }
        )
        states.append(
            state.model_copy(
                update={
                    "status": "draft" if state.status != "confirmed" else "confirmed",
                    "draft": normalized,
                }
            )
        )
    if not normalized_count:
        return project, 0
    revised = project.model_copy(
        update={
            "sections": states,
            "updated_at": datetime.now(timezone.utc),
        }
    )
    return V04WritingProject.model_validate(revised.model_dump(mode="json")), normalized_count


def _autopilot_exhaustion_detail(draft) -> str | None:
    if (
        draft is None
        or draft.quality_review_rounds < 3
    ):
        return None
    if draft.quality_review_status == "failed":
        return (
            f"独立审稿已经连续失败 {draft.quality_review_rounds} 轮。"
            "这更可能是模型输出合同或审稿服务问题，而不是正文内容问题。"
        )
    if draft.quality_review_status != "findings":
        return None
    blocking = [
        issue
        for issue in draft.issues
        if issue.severity == "blocking" and issue.code in EDITORIAL_REPAIR_CODES
    ]
    if not blocking:
        return None
    affected = "、".join(
        f"第 {issue.paragraph_number} 段 {_issue_label(issue.code)}"
        for issue in blocking
    )
    return (
        f"本章已经自动审阅 {draft.quality_review_rounds} 轮，仍出现 {affected}。"
        "这不是继续点击重写可以可靠解决的问题。"
    )


def _exhausted_evidence_recovery_request(
    project: V04WritingProject,
    plan: GroundedWritingPlan,
) -> WritingEvidenceRecoveryRequest | None:
    """Convert an exhausted reviewer finding into a stage recovery command."""

    problem_state = next(
        (
            state
            for state in project.sections
            if state.status != "confirmed"
            and _autopilot_exhaustion_detail(state.draft)
        ),
        None,
    )
    if problem_state is None or problem_state.draft is None:
        return None
    section_plan = next(
        section
        for section in plan.sections
        if section.section_id == problem_state.section_id
    )
    packet = SectionEvidencePacketBuilder().build(
        project.handoff,
        problem_state.section_id,
    )
    gaps = WritingEvidenceRecoveryService().audit_section(section_plan, packet)
    blocking_paragraphs = {
        issue.paragraph_number
        for issue in problem_state.draft.issues
        if issue.severity == "blocking"
        and issue.code in {"unsupported_claim", "overstated_evidence"}
        and issue.paragraph_number is not None
    }
    if blocking_paragraphs:
        gaps = tuple(
            gap for gap in gaps if gap.paragraph_number in blocking_paragraphs
        )
    if not gaps:
        return None
    return WritingEvidenceRecoveryService().request(
        plan_fingerprint=plan.plan_fingerprint,
        gaps=gaps,
    )


def _render_manual_section_control(
    project: V04WritingProject,
    plan: GroundedWritingPlan,
) -> None:
    """Keep the manual path linear: generate, read, adopt, then advance."""

    valid_ids = {section.section_id for section in plan.sections}
    requested = st.session_state.pop(SECTION_SELECTION_REQUEST_KEY, None)
    if requested in valid_ids:
        requested_state = next(
            state for state in project.sections if state.section_id == requested
        )
        section_id = (
            requested if requested_state.status != "confirmed" else _next_actionable_section(project)
        )
    else:
        section_id = _next_actionable_section(project)
    section_plan = next(section for section in plan.sections if section.section_id == section_id)
    state = next(item for item in project.sections if item.section_id == section_id)
    packet = SectionEvidencePacketBuilder().build(project.handoff, section_id)

    st.subheader(section_plan.title)
    _render_packet(packet)
    if packet.ai_writing_mode == "generation_blocked":
        st.error("课程要求禁止 AI 生成正文。本章只能导出证据后由用户自行撰写。")
        st.download_button(
            "下载本章人工写作证据包",
            packet.model_dump_json(indent=2),
            file_name=f"{section_id}_manual_writing_evidence.json",
            mime="application/json",
            width="stretch",
        )
        return

    exhausted = _autopilot_exhaustion_detail(state.draft)
    needs_generation = (
        state.draft is None
        or state.draft.status == "needs_review"
        or state.draft.quality_review_status != "passed"
    )
    if needs_generation:
        if exhausted:
            st.error(exhausted)
            st.caption(
                "系统已阻止继续盲目重写。请检查本章写作计划、证据准入或审稿规则；"
                "普通的再次点击不会解决约束冲突。"
            )
        else:
            label = (
                "生成并自动检查本章"
                if state.draft is None
                else "自动修复问题段落并复审"
            )
            if st.button(
                label,
                type="primary",
                width="stretch",
                key=f"v04_manual_generate_{section_id}",
            ):
                updated = _execute_continuous_writing(
                    project,
                    plan,
                    section_id=section_id,
                    auto_confirm=False,
                )
                _store_project(updated)
                st.rerun()

    current = next(item for item in project.sections if item.section_id == section_id)
    if current.draft is not None:
        _render_draft(project, current.draft)


def _render_continuous_writing_control(
    project: V04WritingProject,
    plan: GroundedWritingPlan,
) -> V04WritingProject:
    """Offer one low-friction action for all remaining chapters."""

    if project.status == "body_complete":
        return project
    remaining = sum(state.status != "confirmed" for state in project.sections)
    problem_state = next(
        (
            state
            for state in project.sections
            if state.status != "confirmed"
            and _autopilot_exhaustion_detail(state.draft)
        ),
        None,
    )
    if problem_state is not None:
        section_plan = next(
            section
            for section in plan.sections
            if section.section_id == problem_state.section_id
        )
        packet = SectionEvidencePacketBuilder().build(
            project.handoff,
            problem_state.section_id,
        )
        evidence_gaps = WritingEvidenceRecoveryService().audit_section(
            section_plan,
            packet,
        )
        blocking_evidence_paragraphs = {
            issue.paragraph_number
            for issue in problem_state.draft.issues
            if issue.severity == "blocking"
            and issue.code in {"unsupported_claim", "overstated_evidence"}
            and issue.paragraph_number is not None
        }
        if blocking_evidence_paragraphs:
            evidence_gaps = tuple(
                gap
                for gap in evidence_gaps
                if gap.paragraph_number in blocking_evidence_paragraphs
            )
        if evidence_gaps:
            request = WritingEvidenceRecoveryService().request(
                plan_fingerprint=plan.plan_fingerprint,
                gaps=evidence_gaps,
            )
            if _begin_evidence_recovery(project, plan, request):
                st.rerun()
            st.error(st.session_state.get("mvp_flash", "证据恢复已停止。"))
            return project
        if _begin_reviewer_plan_repair(project, plan, problem_state):
            st.rerun()
        st.error(_autopilot_exhaustion_detail(problem_state.draft))
        st.caption(
            "Agent 已停止盲目重写并保留全部合格内容。该问题需要调整写作计划、"
            "证据边界或审稿规则；运行记录中已保存失败原因和最后检查点。"
        )
        return project

    st.caption(
        f"还有 {remaining} 章待处理。AI 将逐章生成、独立审稿并只返修问题段落；"
        "每章最多审阅 3 轮，重复失败会自动停止并报告系统级原因。"
    )
    auto_resume = bool(
        st.session_state.pop(EVIDENCE_RECOVERY_AUTO_RESUME_KEY, False)
        or st.session_state.get(V04_AUTOPILOT_REQUESTED_KEY)
    )
    manual_resume = False
    if not auto_resume:
        manual_resume = st.button(
            "继续自动生成",
            type="primary",
            width="stretch",
            key="v04_continuous_writing",
        )
    if manual_resume or auto_resume:
        updated = _execute_continuous_writing(project, plan)
        if updated.status == "body_complete":
            # Recompute the sidebar, progress counters and delivery lock from the
            # newly persisted project instead of leaving the just-finished run's
            # stale 4/6 (or similar) header on screen.
            st.rerun()
        return updated
    return project


def _execute_continuous_writing(
    project: V04WritingProject,
    plan: GroundedWritingPlan,
    *,
    section_id: str | None = None,
    auto_confirm: bool = True,
) -> V04WritingProject:
    """Run one manual chapter or the recoverable full-managed flow."""

    progress = st.status(
        "正在处理正文……",
        expanded=True,
    )
    displayed_events: set[tuple[str, str, int]] = set()
    agent_runtime: WritingAgentRuntimeService | None = None
    prepared_agent_action = None

    if auto_confirm:
        try:
            agent_runtime, agent_context = _writing_agent_runtime(project, plan)
            pending_section_ids = (
                [section_id]
                if section_id is not None
                else [
                    state.section_id
                    for state in project.sections
                    if state.status != "confirmed"
                ]
            )
            prepared_agent_action = agent_runtime.prepare_section_action(
                agent_context,
                section_ids=pending_section_ids,
            )
        except Exception as exc:
            # A durable runtime is a prerequisite for paid model work. Clear every
            # automatic transition so refresh cannot immediately retry a failed
            # initialization; paper and recovery artifacts remain untouched.
            pause_writing_agent(st.session_state)
            _autosave_current_project()
            progress.update(label="Agent 运行时初始化失败", state="error")
            st.error(
                "正文尚未开始生成，因为本地 Agent 检查点无法可靠建立。"
                "请先修复运行时存储，系统不会在不可恢复状态下继续消耗模型调用。"
            )
            with st.expander("技术详情"):
                st.code(str(exc))
            return project

    def checkpoint(
        checkpoint_project: V04WritingProject,
        event: ContinuousWritingEvent,
    ) -> None:
        _store_project(checkpoint_project)
        st.session_state.pop(FINAL_MATTER_KEY, None)
        st.session_state.pop(FINAL_PACKAGE_KEY, None)
        identity = (event.section_id, event.stage, event.revision_pass)
        if identity not in displayed_events:
            displayed_events.add(identity)
            stage_label = {
                "generating": "生成",
                "reviewing": "独立审稿",
                "revising": "定向修订",
                "ready": "待采用",
                "confirmed": "通过",
                "stopped": "暂停",
            }[event.stage]
            progress.write(f"{event.section_title} · {stage_label}：{event.detail}")

    try:
        if prepared_agent_action is not None and prepared_agent_action.cached_observation:
            progress.write("检测到相同输入和行动的成功记录，直接复用本地结果。")
            result = ContinuousWritingResult(project=project, events=())
        else:
            settings = LLMSettings()
            result = ContinuousSectionWritingService(
                writer=LLMGroundedParagraphWriter(
                    DeepSeekClient(settings.for_structured_output())
                ),
                reviewer=LLMSectionQualityReviewer(
                    DeepSeekClient(settings.for_quality_review())
                ),
                cache=ParagraphWritingRuntimeCache(
                    project_root() / "runtime" / "writing_console",
                    plan_fingerprint=plan.plan_fingerprint,
                ),
                policy=ContinuousWritingPolicy(
                    max_revision_passes=2,
                    max_total_review_rounds=3,
                ),
            ).run(
                project,
                plan,
                confirmed_by=(
                    f"{project.handoff.requirement.confirmed_by}"
                    "（V0.4 连续写作授权）"
                ),
                section_id=section_id,
                auto_confirm=auto_confirm,
                on_checkpoint=checkpoint,
            )
    except Exception as exc:
        progress.update(label="连续写作启动失败", state="error")
        st.error(_friendly_writing_error(exc))
        with st.expander("技术详情"):
            st.code(str(exc))
        return project

    _store_project(result.project)
    agent_transition = None
    if agent_runtime is not None and prepared_agent_action is not None:
        try:
            agent_transition = agent_runtime.record_section_result(
                prepared_agent_action,
                result,
                plan,
            )
        except Exception as exc:
            progress.update(label="Agent 决策记录失败", state="error")
            st.error(
                "正文执行结果已经保留，但 Critic/Controller 决策未能可靠落盘。"
                "系统已停止自动推进，避免在缺少审计轨迹时继续修改下游产物。"
            )
            with st.expander("技术详情"):
                st.code(str(exc))
            return result.project

    if result.stopped_section_id is not None:
        next_action_kind = (
            agent_transition.assessment.decision.next_action.payload.kind
            if agent_transition is not None
            and agent_transition.assessment.decision.next_action is not None
            else None
        )
        if (
            next_action_kind in {"acquire_full_text", "refine_literature_search"}
            and result.recovery_request is not None
        ):
            routed = _begin_evidence_recovery(
                result.project,
                plan,
                result.recovery_request,
            )
            if routed:
                progress.update(label="正在回退补齐证据", state="complete")
                st.rerun()
            progress.update(label="证据恢复已达到上限", state="error")
            st.error(st.session_state.get("mvp_flash", "证据恢复已停止。"))
            return result.project
        if next_action_kind == "rebuild_evidence" and agent_transition is not None:
            payload = agent_transition.assessment.decision.next_action.payload
            request = WritingEvidenceRecoveryRequest(
                status="pending_full_text",
                source_plan_fingerprint=plan.plan_fingerprint,
                affected_section_ids=[result.stopped_section_id],
                requested_core_dois=list(payload.source_dois),
                repair_feedback_by_section={
                    result.stopped_section_id: [
                        result.stop_reason
                        or "Deterministic evidence integrity validation failed."
                    ]
                },
            )
            try:
                routed = _begin_evidence_recovery(result.project, plan, request)
            except (TypeError, ValueError) as exc:
                progress.update(label="证据完整性回退未完成", state="error")
                st.error(f"无法建立可恢复的 V0.3 证据重建任务：{exc}")
                return result.project
            if routed:
                progress.update(label="正在回退重建受影响证据", state="complete")
                st.rerun()
            return result.project
        if next_action_kind == "revise_writing_plan":
            problem_state = next(
                state
                for state in result.project.sections
                if state.section_id == result.stopped_section_id
            )
            if _begin_reviewer_plan_repair(result.project, plan, problem_state):
                progress.update(label="正在按审稿意见重构问题章节计划", state="complete")
                st.rerun()
        if (
            auto_confirm
            and next_action_kind == "write_or_revise_sections"
            and agent_transition is not None
        ):
            retry_rounds = agent_transition.state.revision_rounds_by_stage.get(
                "writing", 0
            )
            if retry_rounds <= 3:
                st.session_state[EVIDENCE_RECOVERY_AUTO_RESUME_KEY] = True
                progress.update(
                    label=f"局部执行异常，正在自动重试（{retry_rounds}/3）",
                    state="complete",
                )
                st.rerun()
        progress.update(label="已在问题章节停止", state="error")
        st.session_state[SECTION_SELECTION_REQUEST_KEY] = result.stopped_section_id
        # Reaching this fallback means no bounded automatic action was scheduled.
        # Clear every auto-run trigger so a browser refresh or Streamlit source
        # reload cannot silently repeat the same model call.
        pause_writing_agent(st.session_state)
        prefix = (
            "自动修订已达到上限。"
            if result.stop_code == "review_exhausted"
            else "本章自动处理未完成。"
        )
        st.session_state["mvp_flash"] = prefix + (
            result.stop_reason or "需要检查写作约束。"
        )
        st.warning(st.session_state["mvp_flash"])
    elif not auto_confirm:
        progress.update(label="本章已生成并通过自动审计", state="complete")
        st.session_state["mvp_flash"] = "本章可以阅读并采用。"
    else:
        _resolve_evidence_recovery()
        progress.update(label="剩余章节已完成独立审稿", state="complete")
        st.session_state["mvp_flash"] = (
            "所有剩余章节均已通过证据门禁与独立审稿，可以进入正文汇总。"
        )
        st.success(st.session_state["mvp_flash"])
        if result.project.status == "body_complete" and auto_confirm:
            st.session_state[FINAL_DELIVERY_AUTO_RESUME_KEY] = True
            st.session_state["mvp_navigation_request"] = "delivery"
    return result.project


def _begin_evidence_recovery(
    project: V04WritingProject,
    plan: GroundedWritingPlan,
    request: WritingEvidenceRecoveryRequest,
) -> bool:
    """Persist V0.4, then route to the earliest stage able to repair evidence."""

    # A manuscript editor may reduce or reorganize existing prose, but it cannot create
    # a new research-evidence demand. Shrink an over-strong legacy paragraph plan in
    # place instead of entering V0.2/V0.3 acquisition for an editorial repair.
    if _evidence_gaps_are_editorial_targets(project, request):
        return _begin_bounded_claim_downgrade(project, plan, request)

    previous: WritingEvidenceRecoveryRequest | None = None
    previous_raw = st.session_state.get(EVIDENCE_RECOVERY_REQUEST_KEY)
    if previous_raw:
        try:
            previous = WritingEvidenceRecoveryRequest.model_validate_json(previous_raw)
        except (TypeError, ValueError):
            previous = None
        if (
            previous is not None
            and _same_evidence_recovery_incident(previous, request)
        ):
            request = request.model_copy(
                update={
                    "recovery_round": previous.recovery_round + 1,
                    "max_recovery_rounds": max(previous.max_recovery_rounds, 4),
                    "planning_repair_round": previous.planning_repair_round,
                    "max_planning_repair_rounds": (
                        max(previous.max_planning_repair_rounds, 2)
                    ),
                    "unavailable_full_text_dois": list(
                        dict.fromkeys(
                            [
                                *previous.unavailable_full_text_dois,
                                *request.unavailable_full_text_dois,
                            ]
                        )
                    ),
                }
            )
    if request.recovery_round > request.max_recovery_rounds:
        if (
            previous is not None
            and previous.planning_repair_round
            < previous.max_planning_repair_rounds
        ):
            _begin_structural_plan_repair(
                project,
                plan,
                request.model_copy(
                    update={
                        "status": "ready_to_resume",
                        "recovery_round": previous.recovery_round,
                        "planning_repair_round": (
                            previous.planning_repair_round + 1
                        ),
                    }
                ),
            )
            return True
        if request.gaps:
            return _begin_bounded_claim_downgrade(project, plan, request)
        blocked = request.model_copy(
            update={
                "status": "blocked",
                "blocked_reason": (
                    "补充全文与定向补搜后仍出现同类证据缺口；需要人工检查"
                    "立题边界、写作计划或审稿规则。"
                ),
            }
        )
        st.session_state[EVIDENCE_RECOVERY_REQUEST_KEY] = blocked.model_dump_json(
            indent=2
        )
        st.session_state["mvp_flash"] = (
            f"证据恢复已运行 {request.max_recovery_rounds} 轮仍未解决同类缺口。"
            "系统已停止自动循环并保留全部正文与证据检查点；"
            "这属于计划或规则冲突，不再继续消耗模型调用。"
        )
        return False

    checkpoint = _best_evidence_recovery_checkpoint(project, plan, request)
    st.session_state[EVIDENCE_RECOVERY_CHECKPOINT_KEY] = json.dumps(
        checkpoint,
        ensure_ascii=False,
        indent=2,
    )
    st.session_state[EVIDENCE_RECOVERY_REQUEST_KEY] = request.model_dump_json(
        indent=2
    )
    if request.repair_feedback_by_section:
        st.session_state[WRITING_PLAN_REPAIR_FEEDBACK_KEY] = json.dumps(
            request.repair_feedback_by_section,
            ensure_ascii=False,
            indent=2,
        )
    for key in (
        WRITING_PLAN_KEY,
        V04_PROJECT_KEY,
        FINAL_MATTER_KEY,
        FINAL_PACKAGE_KEY,
    ):
        st.session_state.pop(key, None)
    st.session_state["v04_force_writing_plan_regeneration"] = True
    st.session_state[EVIDENCE_RECOVERY_AUTO_PLAN_KEY] = True
    st.session_state["mvp_navigation_request"] = "writing"
    st.session_state["mvp_flash"] = (
        "新增证据仍未解决审稿问题，异常路由已改判为“写作计划与证据错配”。"
        "系统将保留合格章节，重新规划受影响章节的中心论点与证据分配，"
        "不再重复检索或要求用户操作。"
    )
    st.session_state[EVIDENCE_RECOVERY_REQUEST_KEY] = request.model_dump_json(indent=2)
    selected_dois = _selected_literature_dois()
    unavailable_dois = set(request.unavailable_full_text_dois)
    reusable_dois = [
        doi
        for doi in request.requested_core_dois
        if doi in selected_dois and doi not in unavailable_dois
    ]
    if reusable_dois:
        current_core = list(st.session_state.get("v03_core_dois", []))
        st.session_state["v03_core_dois"] = list(
            dict.fromkeys([*current_core, *reusable_dois])
        )
        for key in (
            "v03_pdf_inspection_json",
            "v03_evidence_library_json",
            "v03_writing_handoff_json",
            WRITING_PLAN_KEY,
            V04_PROJECT_KEY,
            FINAL_MATTER_KEY,
            FINAL_PACKAGE_KEY,
        ):
            st.session_state.pop(key, None)
        st.session_state["mvp_navigation_request"] = "writing"
        st.session_state["mvp_flash"] = (
            f"审稿定位到 {len(request.gaps)} 个证据缺口。系统已保留原正文，"
            f"并将 {len(reusable_dois)} 篇已验证相关论文加入全文队列；"
            "系统会先自动获取、扫描、重规划并续写；只有确实受限的付费全文才会"
            "在最终需要时请求用户协助。"
        )
        return True
    route_evidence_recovery_to_search(request)
    return True


def _same_evidence_recovery_incident(
    previous: WritingEvidenceRecoveryRequest,
    current: WritingEvidenceRecoveryRequest,
) -> bool:
    """Only carry retry counters across recovery attempts for the same chapter.

    A completed recovery request may remain durable until the resumed writing run
    finishes. It must not make a later gap in another chapter inherit exhausted
    retry counters or unrelated unavailable-DOI history.
    """

    if previous.status in {"resolved", "blocked"}:
        return False
    return bool(
        set(previous.affected_section_ids) & set(current.affected_section_ids)
    )


def _best_evidence_recovery_checkpoint(
    project: V04WritingProject,
    plan: GroundedWritingPlan,
    request: WritingEvidenceRecoveryRequest,
) -> dict[str, Any]:
    """Union compatible accepted chapters instead of replacing one incident snapshot."""

    candidate: dict[str, Any] = {
        "schema_version": "v04-evidence-recovery-checkpoint.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "writing_plan_json": plan.model_dump_json(indent=2),
        "writing_project_json": project.model_dump_json(indent=2),
        "handoff_json": project.handoff.model_dump_json(indent=2),
    }
    existing_raw = st.session_state.get(EVIDENCE_RECOVERY_CHECKPOINT_KEY)
    existing_request_raw = st.session_state.get(EVIDENCE_RECOVERY_REQUEST_KEY)
    if not existing_raw or not existing_request_raw:
        return candidate
    try:
        existing = json.loads(existing_raw)
        existing_request = WritingEvidenceRecoveryRequest.model_validate_json(
            existing_request_raw
        )
        existing_project = V04WritingProject.model_validate_json(
            existing["writing_project_json"]
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return candidate
    try:
        merged = _merge_recovery_checkpoint_progress(
            existing,
            candidate,
            excluded_section_ids=set(request.affected_section_ids),
        )
    except (KeyError, TypeError, ValueError):
        merged = candidate
    if _recovery_project_score(
        V04WritingProject.model_validate_json(merged["writing_project_json"])
    ) > _recovery_project_score(project):
        return merged
    if not _same_evidence_recovery_incident(existing_request, request):
        return candidate
    if _recovery_project_score(existing_project) > _recovery_project_score(project):
        return existing
    return candidate


def _merge_recovery_checkpoint_progress(
    existing: dict[str, Any],
    candidate: dict[str, Any],
    *,
    excluded_section_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Carry compatible confirmed sections across separate recovery incidents."""

    try:
        old_plan = GroundedWritingPlan.model_validate_json(
            existing["writing_plan_json"]
        )
        old_project = V04WritingProject.model_validate_json(
            existing["writing_project_json"]
        )
        current_plan = GroundedWritingPlan.model_validate_json(
            candidate["writing_plan_json"]
        )
        current_project = V04WritingProject.model_validate_json(
            candidate["writing_project_json"]
        )
    except (KeyError, TypeError, ValueError):
        return candidate
    if (
        old_project.handoff.requirement != current_project.handoff.requirement
        or old_project.handoff.requirement_policy
        != current_project.handoff.requirement_policy
    ):
        return candidate
    old_plans = {section.section_id: section for section in old_plan.sections}
    old_states = {state.section_id: state for state in old_project.sections}
    current_plans = {
        section.section_id: section for section in current_plan.sections
    }
    current_states = {
        state.section_id: state for state in current_project.sections
    }
    if set(old_plans) != set(current_plans) or set(old_states) != set(current_states):
        return candidate

    preserved: set[str] = set()
    excluded = excluded_section_ids or set()
    merged_sections: list[WritingSectionPlan] = []
    for section in current_plan.sections:
        old_section = old_plans[section.section_id]
        old_state = old_states[section.section_id]
        current_state = current_states[section.section_id]
        if (
            section.section_id not in excluded
            and current_state.status != "confirmed"
            and old_state.status == "confirmed"
            and _recovered_state_is_compatible(
                old_state,
                old_section,
                current_project.handoff,
            )
        ):
            merged_sections.append(old_section)
            preserved.add(section.section_id)
        else:
            merged_sections.append(section)
    merged_sections, preserved = _reopen_minimum_sections_for_source_coverage(
        reference_sections=current_plan.sections,
        merged_sections=merged_sections,
        preserved_section_ids=preserved,
        required_source_dois=current_plan.required_source_dois,
    )
    if not preserved:
        return candidate
    merged_seed = GroundedWritingPlan.model_validate(
        current_plan.model_copy(update={"sections": merged_sections}).model_dump(
            mode="json"
        )
    )
    merged_plan = repair_writing_plan_source_coverage(
        current_project.handoff,
        merged_seed,
        required_source_dois=current_plan.required_source_dois,
    ).plan
    merged_plan_sections = {
        section.section_id: section for section in merged_plan.sections
    }
    merged_states = [
        (
            old_states[state.section_id]
            if (
                state.section_id in preserved
                and _draft_matches_plan(
                    old_states[state.section_id],
                    merged_plan_sections[state.section_id],
                )
            )
            else state
        )
        for state in current_project.sections
    ]
    merged_project = V04WritingProject.model_validate(
        current_project.model_copy(
            update={
                "status": (
                    "body_complete"
                    if all(state.status == "confirmed" for state in merged_states)
                    else "drafting"
                ),
                "sections": merged_states,
                "updated_at": datetime.now(timezone.utc),
            }
        ).model_dump(mode="json")
    )
    return {
        **candidate,
        "writing_plan_json": merged_plan.model_dump_json(indent=2),
        "writing_project_json": merged_project.model_dump_json(indent=2),
        "handoff_json": current_project.handoff.model_dump_json(indent=2),
    }


def _recovery_project_score(project: V04WritingProject) -> tuple[int, int, int]:
    return (
        sum(state.status == "confirmed" for state in project.sections),
        sum(state.status != "pending" for state in project.sections),
        sum(
            len(state.draft.paragraphs)
            for state in project.sections
            if state.draft is not None
        ),
    )


def _begin_structural_plan_repair(
    project: V04WritingProject,
    plan: GroundedWritingPlan,
    request: WritingEvidenceRecoveryRequest,
) -> None:
    """Replan an affected section when more evidence did not fix its argument."""

    checkpoint = _best_evidence_recovery_checkpoint(project, plan, request)
    st.session_state[EVIDENCE_RECOVERY_CHECKPOINT_KEY] = json.dumps(
        checkpoint,
        ensure_ascii=False,
        indent=2,
    )
    repair_feedback = {
        section_id: list(messages)
        for section_id, messages in request.repair_feedback_by_section.items()
    }
    for gap in request.gaps:
        repair_feedback.setdefault(gap.section_id, []).append(
            f"第 {gap.paragraph_number} 段的论证强度超过现有全文证据权限："
            f"{gap.detail} 请优先改用本章已有直接证据；如果仍不足，删除精细比较、"
            "性能优劣或具体指标主张，将段落收缩为证据允许的一般背景。"
        )
    repair_feedback = {
        section_id: list(dict.fromkeys(messages))
        for section_id, messages in repair_feedback.items()
    }
    repaired_request = request.model_copy(
        update={
            "status": "ready_to_resume",
            "repair_feedback_by_section": repair_feedback,
        }
    )
    st.session_state[EVIDENCE_RECOVERY_REQUEST_KEY] = (
        repaired_request.model_dump_json(indent=2)
    )
    st.session_state[WRITING_PLAN_REPAIR_FEEDBACK_KEY] = json.dumps(
        repair_feedback,
        ensure_ascii=False,
        indent=2,
    )
    for key in (
        WRITING_PLAN_KEY,
        V04_PROJECT_KEY,
        FINAL_MATTER_KEY,
        FINAL_PACKAGE_KEY,
    ):
        st.session_state.pop(key, None)
    st.session_state["v04_force_writing_plan_regeneration"] = True
    st.session_state[EVIDENCE_RECOVERY_AUTO_PLAN_KEY] = True
    st.session_state["mvp_navigation_request"] = "writing"
    st.session_state["mvp_flash"] = (
        "补充全文后同一论点仍超出证据边界。系统已保留全部合格章节，"
        "并将问题改判为写作计划错配；现在会自动收缩或重分配受影响段落的论点，"
        "无需重新生成整篇正文。"
    )


def _begin_bounded_claim_downgrade(
    project: V04WritingProject,
    plan: GroundedWritingPlan,
    request: WritingEvidenceRecoveryRequest,
) -> bool:
    """Keep citations but remove claims that remain impossible after recovery."""

    try:
        plan = rebase_writing_plan_authority(project.handoff, plan).plan
        downgraded_plan = downgrade_unresolved_evidence_claims(plan, request.gaps)
        packets = [
            SectionEvidencePacketBuilder().build(project.handoff, section_id)
            for section_id in request.affected_section_ids
        ]
        resolution_errors = WritingEvidenceRecoveryService().validate_resolution(
            downgraded_plan,
            packets,
            affected_section_ids=request.affected_section_ids,
        )
    except (TypeError, ValueError) as exc:
        resolution_errors = (str(exc),)
    if resolution_errors:
        blocked = request.model_copy(
            update={
                "status": "blocked",
                "blocked_reason": (
                    "证据降级后的写作计划仍未通过执行前检查："
                    + "；".join(resolution_errors[:3])
                ),
            }
        )
        st.session_state[EVIDENCE_RECOVERY_REQUEST_KEY] = blocked.model_dump_json(
            indent=2
        )
        st.session_state["mvp_flash"] = (
            "自动补证和论点降级后，计划仍没有形成可执行的来源绑定。"
            "系统已保留原计划、正文和证据检查点并停止重试，未把失败状态"
            "误记为完成。"
        )
        return False
    target_issues = {
        (gap.section_id, gap.paragraph_number): [
            FinalPaperAuditIssue(
                code="body_evidence_scope_reduced",
                severity="blocking",
                requirement_path="writing.editorial_evidence_scope",
                detail=(
                    "Retain the paragraph's existing sources, but reduce its comparison "
                    "or detail claim to the level supported by the available evidence. "
                    "Do not add facts, sources, or performance claims."
                ),
            )
        ]
        for gap in request.gaps
    }
    # A legacy structural-replan checkpoint may already have replaced the affected
    # chapter with a pending state.  In that case there is no old paragraph to reopen;
    # keep every surviving draft and let the normal writer create only pending chapters.
    available_section_ids = {
        state.section_id for state in project.sections if state.draft is not None
    }
    reopenable_issues = {
        target: issues
        for target, issues in target_issues.items()
        if target[0] in available_section_ids
    }
    resumed = (
        _reopen_targeted_paragraphs(project, reopenable_issues)
        if reopenable_issues
        else project.model_copy(
            update={
                "status": "drafting",
                "updated_at": datetime.now(timezone.utc),
            }
        )
    )
    st.session_state[WRITING_PLAN_KEY] = downgraded_plan.model_dump_json(indent=2)
    st.session_state[V04_PROJECT_KEY] = resumed.model_dump_json(indent=2)
    st.session_state[EVIDENCE_RECOVERY_REQUEST_KEY] = request.model_copy(
        update={
            "status": "resolved",
            "blocked_reason": (
                "全文补齐与计划重构均未形成足够证据；相关段落已自动收缩为"
                "元数据允许的一般背景，不再输出详细比较或性能结论。"
            ),
        }
    ).model_dump_json(indent=2)
    for key in (
        EVIDENCE_RECOVERY_CHECKPOINT_KEY,
        WRITING_PLAN_REPAIR_FEEDBACK_KEY,
        FINAL_MATTER_KEY,
        FINAL_PACKAGE_KEY,
    ):
        st.session_state.pop(key, None)
    st.session_state[EVIDENCE_RECOVERY_AUTO_RESUME_KEY] = True
    st.session_state["mvp_flash"] = (
        f"自动补证和重规划后仍有 {len(request.gaps)} 个细节主张缺少全文支持。"
        "系统已保留全部合格章节，并将这些段落收缩为可由元数据支持的一般背景；"
        "现在会自动续写和复审，不需要用户处理内部异常。"
    )
    return True


def continue_without_restricted_full_text() -> bool:
    """Resolve one blocked PDF batch by safely shrinking unsupported details."""

    request_raw = st.session_state.get(EVIDENCE_RECOVERY_REQUEST_KEY)
    checkpoint_raw = st.session_state.get(EVIDENCE_RECOVERY_CHECKPOINT_KEY)
    if not request_raw or not checkpoint_raw:
        st.session_state["mvp_flash"] = (
            "缺少可恢复的写作检查点，不能安全跳过受限全文。"
        )
        return False
    try:
        request = WritingEvidenceRecoveryRequest.model_validate_json(request_raw)
        checkpoint = json.loads(checkpoint_raw)
        plan = GroundedWritingPlan.model_validate_json(checkpoint["writing_plan_json"])
        project = V04WritingProject.model_validate_json(
            checkpoint["writing_project_json"]
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        st.session_state["mvp_flash"] = (
            "证据恢复检查点无法读取，系统没有修改原计划或正文。"
        )
        return False
    if request.status != "blocked" or not request.gaps:
        st.session_state["mvp_flash"] = (
            "当前恢复任务尚未达到需要放弃受限全文的阶段。"
        )
        return False
    unavailable = list(
        dict.fromkeys(
            [
                *request.unavailable_full_text_dois,
                *request.requested_core_dois,
                *(
                    doi
                    for gap in request.gaps
                    for doi in gap.missing_full_text_dois
                ),
            ]
        )
    )
    declined = request.model_copy(
        update={"unavailable_full_text_dois": unavailable}
    )
    if not _begin_bounded_claim_downgrade(project, plan, declined):
        return False
    st.session_state["v03_writing_handoff_json"] = project.handoff.model_dump_json(
        indent=2
    )
    st.session_state[V04_AUTOPILOT_REQUESTED_KEY] = True
    st.session_state[EVIDENCE_RECOVERY_AUTO_RESUME_KEY] = True
    return True


def _evidence_gaps_are_editorial_targets(
    project: V04WritingProject,
    request: WritingEvidenceRecoveryRequest,
) -> bool:
    if not request.gaps:
        return False
    editorial_targets = {
        (state.section_id, issue.paragraph_number)
        for state in project.sections
        if state.draft is not None
        for issue in state.draft.issues
        if issue.code == "final_audit_repair"
        and issue.severity == "blocking"
        and issue.paragraph_number is not None
    }
    return all(
        (gap.section_id, gap.paragraph_number) in editorial_targets
        for gap in request.gaps
    )


def _resume_editorial_evidence_checkpoint_without_search() -> bool:
    """Migrate an old editor-created evidence rollback to bounded prose repair."""

    checkpoint_raw = st.session_state.get(EVIDENCE_RECOVERY_CHECKPOINT_KEY)
    request_raw = st.session_state.get(EVIDENCE_RECOVERY_REQUEST_KEY)
    if not checkpoint_raw or not request_raw:
        return False
    try:
        checkpoint = json.loads(checkpoint_raw)
        request = WritingEvidenceRecoveryRequest.model_validate_json(request_raw)
        plan = GroundedWritingPlan.model_validate_json(
            checkpoint["writing_plan_json"]
        )
        project = V04WritingProject.model_validate_json(
            checkpoint["writing_project_json"]
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    if request.status not in {"pending_full_text", "pending_search", "ready_to_resume"}:
        return False
    if not _evidence_recovery_can_downgrade_without_search(project, request):
        return False
    return _begin_bounded_claim_downgrade(project, plan, request)


def _evidence_recovery_can_downgrade_without_search(
    project: V04WritingProject,
    request: WritingEvidenceRecoveryRequest,
) -> bool:
    """Return whether more retrieval cannot improve this saved incident."""

    if _evidence_gaps_are_editorial_targets(project, request):
        return True
    return bool(
        request.gaps
        and request.recovery_round >= request.max_recovery_rounds
        and request.planning_repair_round > 0
    )


def _begin_reviewer_plan_repair(
    project: V04WritingProject,
    plan: GroundedWritingPlan,
    state: WritingSectionState,
) -> bool:
    """Turn repeated reviewer findings into one bounded, issue-guided replan."""

    if state.draft is None:
        return False
    blocking = [
        issue
        for issue in state.draft.issues
        if issue.severity == "blocking"
        and issue.code
        in {*EDITORIAL_REPAIR_CODES, *PLAN_BINDING_DETERMINISTIC_CODES}
    ]
    if not blocking:
        return False
    previous: WritingEvidenceRecoveryRequest | None = None
    previous_raw = st.session_state.get(EVIDENCE_RECOVERY_REQUEST_KEY)
    if previous_raw:
        try:
            previous = WritingEvidenceRecoveryRequest.model_validate_json(previous_raw)
        except (TypeError, ValueError):
            previous = None
    same_incident = _same_reviewer_plan_repair_incident(
        previous,
        state.section_id,
    )
    planning_round = (
        previous.planning_repair_round + 1
        if previous is not None and same_incident
        else 1
    )
    max_planning_rounds = max(
        (
            previous.max_planning_repair_rounds
            if previous is not None and same_incident
            else 0
        ),
        3,
    )
    if planning_round > max_planning_rounds:
        st.session_state["mvp_flash"] = (
            f"审稿意见已驱动写作计划重构 {max_planning_rounds} 轮，仍出现同类问题。"
            "系统已保留合格章节并停止自动循环；这属于证据边界、审稿规则或"
            "章节目标之间的系统级冲突。"
        )
        return False
    section_plan = next(
        section for section in plan.sections if section.section_id == state.section_id
    )
    paragraph_plans = {
        paragraph.paragraph_number: paragraph for paragraph in section_plan.paragraphs
    }
    feedback: list[str] = []
    for issue in blocking:
        paragraph = paragraph_plans.get(issue.paragraph_number or -1)
        planned_claim = paragraph.claim_focus if paragraph is not None else "未知"
        feedback.append(
            f"第 {issue.paragraph_number or '?'} 段 [{issue.code}]：{issue.detail} "
            f"被拒绝的原计划主张：{planned_claim}。请更换该段中心判断与证据分配，"
            "不要仅改写措辞。"
        )
    request = WritingEvidenceRecoveryRequest(
        status="ready_to_resume",
        source_plan_fingerprint=plan.plan_fingerprint,
        affected_section_ids=[state.section_id],
        repair_feedback_by_section={state.section_id: feedback},
        unavailable_full_text_dois=(
            previous.unavailable_full_text_dois
            if previous is not None and same_incident
            else []
        ),
        recovery_round=(
            previous.recovery_round
            if previous is not None and same_incident
            else 1
        ),
        max_recovery_rounds=max(
            (
                previous.max_recovery_rounds
                if previous is not None and same_incident
                else 0
            ),
            4,
        ),
        planning_repair_round=planning_round,
        max_planning_repair_rounds=max_planning_rounds,
    )
    _begin_structural_plan_repair(project, plan, request)
    st.session_state["mvp_flash"] = (
        f"独立审稿连续定位到 {len(blocking)} 个计划级问题。系统已保留合格章节，"
        "正在把审稿意见写入规划约束，只重构受影响章节后自动续写。"
    )
    return True


def _same_reviewer_plan_repair_incident(
    previous: WritingEvidenceRecoveryRequest | None,
    section_id: str,
) -> bool:
    """Keep a replan budget local to one unresolved chapter incident."""

    return bool(
        previous is not None
        and previous.status not in {"resolved", "blocked"}
        and section_id in previous.affected_section_ids
    )


def _writing_plan_repair_feedback() -> dict[str, list[str]]:
    raw = st.session_state.get(WRITING_PLAN_REPAIR_FEEDBACK_KEY)
    if not raw:
        return {}
    try:
        payload: Any = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return {
        str(section_id): [str(item) for item in items if str(item).strip()]
        for section_id, items in payload.items()
        if isinstance(items, list)
    }


def route_evidence_recovery_to_search(
    request: WritingEvidenceRecoveryRequest,
) -> None:
    confirmed_raw = st.session_state.get("literature_confirmed_blueprint_json")
    if not confirmed_raw:
        raise ValueError("证据补搜缺少已确认的 V0.2 检索蓝图")
    confirmed = ConfirmedLiteratureSearchBlueprint.model_validate_json(confirmed_raw)
    enriched = WritingEvidenceRecoveryService().enrich_search_blueprint(
        confirmed,
        request.model_copy(update={"status": "pending_search"}),
    )
    blueprint_json = enriched.blueprint.model_dump_json(indent=2)
    st.session_state["literature_blueprint_json"] = blueprint_json
    st.session_state["literature_blueprint_editor"] = blueprint_json
    st.session_state["literature_confirmed_blueprint_json"] = (
        enriched.model_dump_json(indent=2)
    )
    previous_run_dir = st.session_state.get("literature_run_dir")
    if previous_run_dir:
        st.session_state["literature_recovery_seed_run_dir"] = previous_run_dir
    for key in (
        "literature_result_json",
        "literature_ris",
        "literature_verification_json",
        "literature_run_dir",
        "v03_pdf_inspection_json",
        "v03_evidence_library_json",
        "v03_writing_handoff_json",
        WRITING_PLAN_KEY,
        V04_PROJECT_KEY,
        FINAL_MATTER_KEY,
        FINAL_PACKAGE_KEY,
    ):
        st.session_state.pop(key, None)
    st.session_state[EVIDENCE_RECOVERY_REQUEST_KEY] = request.model_copy(
        update={"status": "pending_search"}
    ).model_dump_json(indent=2)
    st.session_state["literature_auto_run_requested"] = True
    st.session_state["mvp_navigation_request"] = "writing"
    st.session_state["mvp_flash"] = (
        "当前文献池没有足够的可用全文来源；系统已按缺失论点扩写检索式，"
        "将自动补搜并避免重复 DOI。"
    )


def _selected_literature_dois() -> set[str]:
    raw = st.session_state.get("literature_result_json")
    if not raw:
        return set()
    try:
        payload = json.loads(raw)
        return {
            str(record["doi"])
            for record in payload["selection"]["selected"]
            if record.get("doi")
        }
    except (KeyError, TypeError, json.JSONDecodeError):
        return set()


def _resolve_evidence_recovery() -> bool:
    raw = st.session_state.get(EVIDENCE_RECOVERY_REQUEST_KEY)
    if not raw:
        return False
    request = WritingEvidenceRecoveryRequest.model_validate_json(raw)
    changed = request.status != "resolved"
    st.session_state[EVIDENCE_RECOVERY_REQUEST_KEY] = request.model_copy(
        update={"status": "resolved"}
    ).model_dump_json(indent=2)
    st.session_state.pop(EVIDENCE_RECOVERY_CHECKPOINT_KEY, None)
    st.session_state.pop(WRITING_PLAN_REPAIR_FEEDBACK_KEY, None)
    return changed


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
    core_sources = sum(
        source.evidence_tier == "A_core" for source in packet.sources
    )
    st.caption(
        f"目标 {packet.target_words} 字 · {len(packet.evidence_items)} 张全文证据卡 · "
        f"{core_sources} 篇核心来源"
    )
    with st.expander("查看本章证据依据（高级）"):
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
) -> None:
    st.markdown("#### 本章草稿")
    review_label = {
        "not_run": "未运行",
        "passed": "通过",
        "findings": "发现问题",
        "failed": "未完成",
    }[draft.quality_review_status]
    blocking = [issue for issue in draft.issues if issue.severity == "blocking"]
    warnings = [issue for issue in draft.issues if issue.severity == "warning"]
    if blocking:
        st.error(
            f"自动审计未通过：{len(blocking)} 个阻塞问题；"
            f"已审阅 {draft.quality_review_rounds} 轮。"
        )
        st.dataframe(
            [_issue_row(issue) for issue in blocking],
            hide_index=True,
            width="stretch",
        )
    elif draft.quality_review_status == "failed":
        st.error(f"独立审稿未完成；系统已保留草稿和已完成的 {draft.quality_review_rounds} 轮记录。")
    elif draft.quality_review_status == "passed":
        st.success(f"自动审计通过 · 已审阅 {draft.quality_review_rounds} 轮")
    else:
        st.info(f"独立审稿状态：{review_label}")

    if warnings:
        with st.expander(f"查看 {len(warnings)} 条非阻塞写作建议"):
            st.dataframe(
                [_issue_row(issue) for issue in warnings],
                hide_index=True,
                width="stretch",
            )
    st.markdown(draft.markdown)
    st.caption(
        f"目标 {draft.target_words} 字 · 当前 {draft.counted_words} 字"
    )
    with st.expander("导出草稿与审计记录（高级）"):
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
    if draft.status == "needs_review" or draft.quality_review_status != "passed":
        return
    if st.button(
        "采用本章并进入下一章",
        type="primary",
        width="stretch",
        key=f"v04_confirm_manual_{draft.section_id}",
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
            st.session_state[SECTION_SELECTION_REQUEST_KEY] = _next_actionable_section(
                updated
            )
            st.session_state["mvp_flash"] = (
                "本章已采用，已进入下一章。"
                if updated.status != "body_complete"
                else "全部正文章节已采用，可以组装最终论文。"
            )
            st.rerun()


def _issue_row(issue) -> dict[str, object]:
    return {
        "问题": _issue_label(issue.code),
        "段落": issue.paragraph_number or "全章",
        "说明": issue.detail,
    }


def _issue_label(code: str) -> str:
    return {
        "paragraph_repetition": "内容重复",
        "topic_drift": "偏离主题",
        "coherence_gap": "论证衔接不足",
        "terminology_inconsistent": "术语不一致",
        "academic_style_problem": "学术表达问题",
        "unsupported_claim": "主张缺少证据",
        "overstated_evidence": "结论强于证据",
        "false_self_attribution": "错误认领被引研究",
        "oversized_paragraph": "段落承载过多论点",
        "language_mismatch": "语言不符合要求",
        "word_count_low": "字数不足",
        "word_count_high": "字数过多",
        "quality_review_failed": "独立审稿未完成",
        "quality_review_degraded": "独立审稿降级",
        "quality_review_deferred": "局部编辑问题转交全文复核",
    }.get(code, code)


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
    elif st.session_state.get(V04_AUTOPILOT_REQUESTED_KEY):
        st.session_state[FINAL_DELIVERY_AUTO_RESUME_KEY] = True
        st.session_state["mvp_navigation_request"] = "delivery"
        st.rerun()
    elif st.button("进入最终交付", type="primary", width="stretch"):
        st.session_state["mvp_navigation_request"] = "delivery"
        st.rerun()


def _cached_manuscript_editor_checkpoint(
    state: MutableMapping[str, Any],
    plan: GroundedWritingPlan,
    project: V04WritingProject,
) -> ManuscriptEditorialCheckpoint | None:
    serialized = state.get(MANUSCRIPT_EDITOR_KEY)
    if not serialized:
        return None
    try:
        checkpoint = ManuscriptEditorialCheckpoint.model_validate_json(serialized)
    except (TypeError, ValueError):
        state.pop(MANUSCRIPT_EDITOR_KEY, None)
        return None
    if checkpoint.body_fingerprint != manuscript_body_fingerprint(plan, project):
        state.pop(MANUSCRIPT_EDITOR_KEY, None)
        return None
    return checkpoint


def _render_manuscript_editor_result(
    checkpoint: ManuscriptEditorialCheckpoint,
) -> None:
    st.subheader("独立全文编辑")
    st.caption(
        "该阶段只读完整正文，检查跨章节重复、章节职责、超长段落和错误研究归属；"
        "它不能改引用或偷偷重写正文。通过后才生成摘要、引言和结论。"
    )
    metrics = st.columns(3)
    metrics[0].metric(
        "编辑门禁",
        "通过" if checkpoint.status == "passed" else "需定点返修",
    )
    metrics[1].metric("阻塞问题", checkpoint.blocking_count)
    metrics[2].metric("编辑建议", checkpoint.warning_count)
    if checkpoint.review.review_status == "deterministic_fallback":
        st.warning(
            "独立编辑模型的结构化输出异常；本轮仍执行了确定性重复、超长段落"
            "和错误研究归属检查，但不会把模型异常伪装成完整审稿。"
        )
    if checkpoint.review.findings:
        with st.expander("查看全文编辑定位"):
            st.dataframe(
                [
                    {
                        "级别": finding.severity,
                        "问题": finding.code,
                        "位置": f"{finding.section_id}:{finding.paragraph_number}",
                        "处理": (
                            "定点返修"
                            if finding.disposition == "targeted_repair"
                            else "报告建议"
                        ),
                        "说明": finding.detail,
                    }
                    for finding in checkpoint.review.findings
                ],
                hide_index=True,
                width="stretch",
            )


def _render_final_delivery(project: V04WritingProject, body) -> None:
    st.divider()
    st.subheader("全文编辑、审计与成品交付")
    st.caption(
        "Agent 会从全文视角检查重复、章节职责、论证衔接和作者归属，"
        "只返修问题段落；随后由代码完成引用、参考文献、合规审计、评分和 DOCX 组装。"
    )
    policy = project.handoff.requirement_policy
    if policy is None:
        from veriwrite_agent.services.requirement_policy import RequirementPolicyCompiler

        policy = RequirementPolicyCompiler().compile(project.handoff.requirement)
    settings = LLMSettings()
    writing_plan = GroundedWritingPlan.model_validate_json(
        st.session_state[WRITING_PLAN_KEY]
    )
    try:
        agent_runtime, agent_context = _writing_agent_runtime(project, writing_plan)
        agent_state = agent_context.state
    except Exception as exc:
        st.error(
            "最终编辑尚未开始，因为 V0.5 Agent 检查点无法可靠建立。"
            "系统不会在缺少可恢复审计轨迹时继续组装终稿。"
        )
        with st.expander("技术详情"):
            st.code(str(exc))
        return
    agent_controller = WritingAgentController()
    default_declaration = ""
    if policy.ai_usage.declaration_required:
        default_declaration = (
            "AI tool: DeepSeek; writing model: "
            f"{settings.structured_model or settings.model}; independent review model: "
            f"{settings.reviewer_model or settings.structured_model or settings.model}; "
            "purpose: evidence-constrained section drafting, independent chapter review, "
            "targeted revision, and final-matter organization; "
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

    auto_upgrade = False
    if FINAL_PACKAGE_KEY in st.session_state:
        try:
            cached_package = FinalPaperPackage.model_validate_json(
                st.session_state[FINAL_PACKAGE_KEY]
            )
        except (TypeError, ValueError):
            cached_package = None
        if cached_package is None or cached_package.schema_version != "mvp-2.2":
            st.session_state.pop(FINAL_MATTER_KEY, None)
            st.session_state.pop(FINAL_PACKAGE_KEY, None)
            # A new global-editor algorithm gets its own bounded repair budget.
            st.session_state[GLOBAL_EDITOR_ROUND_KEY] = 0
            st.session_state.pop(FINAL_REPAIR_AUTO_SUPPRESSION_KEY, None)
            auto_upgrade = True

    if FINAL_MATTER_KEY not in st.session_state:
        auto_resume_delivery = bool(
            st.session_state.pop(FINAL_DELIVERY_AUTO_RESUME_KEY, False)
            or st.session_state.get(V04_AUTOPILOT_REQUESTED_KEY)
        )
        editor_checkpoint = _cached_manuscript_editor_checkpoint(
            st.session_state,
            writing_plan,
            project,
        )
        run_editor = auto_upgrade or auto_resume_delivery
        if editor_checkpoint is None:
            if not run_editor:
                run_editor = st.button(
                    "运行独立全文编辑并继续",
                    type="primary",
                    width="stretch",
                )
        elif editor_checkpoint.status == "passed":
            _render_manuscript_editor_result(editor_checkpoint)
            if not run_editor:
                run_editor = st.button(
                    "全文编辑已通过，继续生成终稿",
                    type="primary",
                    width="stretch",
                )

        new_editor_checkpoint = False
        if editor_checkpoint is None and run_editor:
            try:
                with st.spinner("正在执行独立全文编辑；正文未通过时不会生成终稿……"):
                    editor_checkpoint = FullManuscriptEditorialService(
                        LLMManuscriptQualityReviewer(
                            DeepSeekClient(settings.for_quality_review())
                        )
                    ).run(writing_plan, project)
            except Exception as exc:
                st.error(f"独立全文编辑失败：{exc}")
                return
            st.session_state[MANUSCRIPT_EDITOR_KEY] = (
                editor_checkpoint.model_dump_json(indent=2)
            )
            new_editor_checkpoint = True
            _render_manuscript_editor_result(editor_checkpoint)
            try:
                editor_assessment = agent_controller.assess_manuscript_editor(
                    editor_checkpoint,
                    project_reference=agent_context.project_reference,
                )
                agent_state = agent_runtime.record_assessment(
                    agent_state,
                    editor_assessment,
                )
            except Exception as exc:
                st.error(
                    "全文编辑结果已经保留，但 Controller 决策未能写入检查点；"
                    "系统已停止自动推进。"
                )
                with st.expander("技术详情"):
                    st.code(str(exc))
                return

        if editor_checkpoint is None:
            return
        if editor_checkpoint.status == "needs_revision":
            if not new_editor_checkpoint:
                _render_manuscript_editor_result(editor_checkpoint)
            completed_rounds = int(
                st.session_state.get(GLOBAL_EDITOR_ROUND_KEY, 0) or 0
            )
            if completed_rounds >= MAX_GLOBAL_EDITOR_ROUNDS:
                st.error(
                    f"全文编辑已达到 {MAX_GLOBAL_EDITOR_ROUNDS} 轮上限。系统不会继续"
                    "盲目改写；请检查章节职责、证据边界或编辑规则。"
                )
                return
            auto_repair = (
                st.session_state.get(WRITING_MODE_KEY, "逐章生成") == "AI 全托管"
            )
            apply_repair = auto_repair or st.button(
                "按全文编辑结果只返修问题段落",
                type="primary",
                width="stretch",
            )
            if not apply_repair:
                return
            try:
                blocking_findings = [
                    finding
                    for finding in editor_checkpoint.review.findings
                    if finding.severity == "blocking"
                    or finding.disposition == "targeted_repair"
                ]
                structural_codes = {
                    "cross_section_repetition",
                    "section_role_overlap",
                    "global_coherence_gap",
                    "oversized_paragraph",
                }
                structural_planner = None
                if any(
                    finding.code in structural_codes
                    for finding in blocking_findings
                ):
                    feedback_by_section: dict[str, list[str]] = {}
                    for finding in blocking_findings:
                        feedback_by_section.setdefault(
                            finding.section_id,
                            [],
                        ).append(
                            f"[{finding.code}] {finding.detail} "
                            f"Required correction: {finding.revision_instruction}"
                        )
                    structural_planner = GroundedWritingPlanner(
                        DeepSeekClient(settings.for_structured_output()),
                        reuse_cache=False,
                        repair_feedback_by_section=feedback_by_section,
                    )
                repair = build_manuscript_editor_repair(
                    st.session_state,
                    editor_checkpoint,
                    structural_planner=structural_planner,
                )
            except Exception as exc:
                st.error(f"无法创建全文编辑返修任务：{exc}")
                return
            _create_final_repair_checkpoint(
                [finding.code for finding in blocking_findings],
                replace=True,
            )
            st.session_state[WRITING_PLAN_KEY] = repair.plan.model_dump_json(indent=2)
            st.session_state[V04_PROJECT_KEY] = repair.project.model_dump_json(indent=2)
            st.session_state[GLOBAL_EDITOR_ROUND_KEY] = completed_rounds + 1
            st.session_state.pop(MANUSCRIPT_EDITOR_KEY, None)
            st.session_state.pop(FINAL_MATTER_KEY, None)
            st.session_state.pop(FINAL_PACKAGE_KEY, None)
            st.session_state[SECTION_SELECTION_REQUEST_KEY] = next(
                iter(repair.paragraph_numbers)
            )
            # The sidebar radio keyed by ``mvp_navigation`` already exists in this
            # Streamlit run. Queue the transition for the next rerun instead of
            # mutating an instantiated widget key.
            st.session_state["mvp_navigation_request"] = "writing"
            if auto_repair:
                st.session_state[EVIDENCE_RECOVERY_AUTO_RESUME_KEY] = True
                st.session_state[FINAL_DELIVERY_AUTO_RESUME_KEY] = True
            st.session_state["mvp_flash"] = (
                "独立全文编辑已定位问题；系统保留全部合格章节，"
                f"只重开 {repair.paragraph_count} 个问题段落。"
            )
            _autosave_current_project()
            st.rerun()

        if editor_checkpoint.status == "passed" and (
            new_editor_checkpoint or run_editor
        ):
            try:
                with st.spinner("全文编辑已通过，正在生成并审计最终结构……"):
                    matter = LLMFinalMatterWriter(
                        DeepSeekClient(settings.for_structured_output())
                    ).draft(project.handoff, body)
                    package = FinalPaperAssembler().assemble(
                        handoff=project.handoff,
                        body=body,
                        final_matter=matter,
                        ai_declaration=ai_declaration.strip() or None,
                        manuscript_review=editor_checkpoint.review,
                    )
                    package_reference = artifact_reference_from_model(
                        package,
                        storage_key="mvp_snapshot.state.mvp_final_paper_json",
                    )
                    final_assessment = agent_controller.assess_final_package(
                        package,
                        writing_plan,
                        package_reference=package_reference,
                    )
                    agent_state = agent_runtime.record_assessment(
                        agent_state,
                        final_assessment,
                        output_reference=package_reference,
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
    if (
        package.schema_version != "mvp-2.2"
        or ai_declaration.strip() != (package.ai_declaration or "")
    ):
        package = FinalPaperAssembler().assemble(
            handoff=project.handoff,
            body=body,
            final_matter=matter,
            ai_declaration=ai_declaration.strip() or None,
            manuscript_review=package.manuscript_review,
        )
        st.session_state[FINAL_PACKAGE_KEY] = package.model_dump_json(indent=2)
    if package.status != "needs_revision":
        st.session_state.pop(GLOBAL_EDITOR_ROUND_KEY, None)

    metrics = st.columns(4)
    metrics[0].metric("正文统计单位", package.audit.counted_units)
    metrics[1].metric("实际引用文献", package.audit.reference_count)
    metrics[2].metric("外文文献", package.audit.foreign_reference_count)
    metrics[3].metric("阻塞项", package.audit.blocking_count)
    scorecard, comparison = _evaluate_paper_quality(
        package,
        project,
        writing_plan,
    )
    _render_paper_quality_scorecard(scorecard, comparison)
    if package.status != "needs_revision":
        external_evaluation, external_baseline = _evaluate_external_writing_quality(
            package.markdown
        )
        if external_evaluation is not None:
            _render_external_writing_quality(
                external_evaluation,
                external_baseline,
            )
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
        if st.button(
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
            _autosave_current_project()
            try:
                confirmed_reference = artifact_reference_from_model(
                    package,
                    storage_key="mvp_snapshot.state.mvp_final_paper_json",
                )
                confirmed_assessment = agent_controller.assess_final_package(
                    package,
                    writing_plan,
                    package_reference=confirmed_reference,
                )
                agent_runtime.record_assessment(
                    agent_state,
                    confirmed_assessment,
                    output_reference=confirmed_reference,
                )
            except Exception as exc:
                st.error(
                    "论文确认结果已经安全保存，但 Agent 完成事件未能写入；"
                    "请保留当前页面并检查运行时存储。"
                )
                with st.expander("技术详情"):
                    st.code(str(exc))
                return
            st.session_state.pop(FINAL_REPAIR_CHECKPOINT_KEY, None)
            st.session_state.pop(FINAL_REPAIR_AUTO_SUPPRESSION_KEY, None)
            st.rerun()
        _render_full_body_regeneration_button(
            key="v05_reject_ready_paper",
        )
        with st.expander("高级：重新生成标题、摘要、关键词和结论"):
            st.caption("仅在最终组成部分明显不合适时使用；正文和已通过章节不会被删除。")
            if st.button("重新生成最终组成部分", width="stretch"):
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
    _render_full_body_regeneration_button(
        key="v05_reject_confirmed_paper",
    )
    st.info(
        "MVP 已验证引用身份、引用绑定和 PDF 页码来源；逐句语义蕴含验证 "
        "（这句话是否真的被引文支持）明确保留为 MVP 后续优化项。"
    )


def _render_full_body_regeneration_button(*, key: str) -> None:
    st.caption(
        "如果正文整体不满意，可保留 V0.1–V0.3、PDF、证据库和写作计划，"
        "退回 V0.4 重新生成全部正文章节。"
    )
    if st.button(
        "不满意：返回 V0.4 重新生成正文",
        key=key,
        width="stretch",
    ):
        if not reopen_entire_body_for_regeneration(st.session_state):
            st.error("当前项目缺少可恢复的 V0.4 正文检查点，无法安全重新生成。")
            return
        _autosave_current_project()
        st.rerun()


def _evaluate_paper_quality(
    package: FinalPaperPackage,
    project: V04WritingProject,
    plan: GroundedWritingPlan,
) -> tuple[PaperQualityScorecard, PaperQualityComparison | None]:
    service = PaperQualityEvaluationService()
    candidate = service.evaluate(package, project, plan)
    current_json = st.session_state.get(PAPER_QUALITY_SCORECARD_KEY)
    baseline_json = st.session_state.get(PAPER_QUALITY_BASELINE_KEY)
    current = (
        PaperQualityScorecard.model_validate_json(current_json)
        if isinstance(current_json, str)
        else None
    )
    if (
        current is not None
        and current.paper_fingerprint != candidate.paper_fingerprint
        and current.evaluation_method == candidate.evaluation_method
    ):
        st.session_state[PAPER_QUALITY_BASELINE_KEY] = current.model_dump_json(indent=2)
        baseline = current
    else:
        baseline = (
            PaperQualityScorecard.model_validate_json(baseline_json)
            if isinstance(baseline_json, str)
            else None
        )
    st.session_state[PAPER_QUALITY_SCORECARD_KEY] = candidate.model_dump_json(indent=2)
    comparison = None
    if (
        baseline is not None
        and baseline.paper_fingerprint != candidate.paper_fingerprint
        and baseline.evaluation_method == candidate.evaluation_method
    ):
        comparison = service.compare(baseline, candidate)
    return candidate, comparison


def _render_paper_quality_scorecard(
    scorecard: PaperQualityScorecard,
    comparison: PaperQualityComparison | None,
) -> None:
    st.subheader("工程可信度与交付合规")
    st.caption(
        "这是流程、证据和合规代理分，不代表论文的独立写作质量。"
        "六维分数仅用于同一任务下比较版本，不会用总分覆盖阻塞项。"
        "背景文献允许只绑定已验证元数据，不会因没有核心 PDF 页码被误扣分。"
    )
    grade_labels = {
        "excellent": "优秀",
        "strong": "良好",
        "acceptable": "可接受",
        "weak": "需改进",
    }
    summary = st.columns(3)
    delta = comparison.overall_delta if comparison is not None else None
    summary[0].metric("工程代理分", f"{scorecard.overall_score:.1f}/100", delta=delta)
    summary[1].metric(
        "交付门禁",
        "通过" if scorecard.release_gate == "passed" else "阻塞",
    )
    summary[2].metric("工程状态", grade_labels[scorecard.grade])

    metric_deltas = comparison.metric_deltas if comparison is not None else {}
    st.dataframe(
        [
            {
                "指标": metric.label,
                "得分": metric.score,
                "较上版": metric_deltas.get(metric.code),
                "权重": f"{metric.weight:.0%}",
                "加权分": metric.weighted_points,
                "计算依据": "；".join(metric.basis),
            }
            for metric in scorecard.metrics
        ],
        hide_index=True,
        width="stretch",
    )
    if scorecard.blocking_issues:
        st.error("当前仍不可交付：" + "；".join(scorecard.blocking_issues))
    with st.expander("评分口径、局限与导出"):
        st.markdown(
            "- **要求符合度**：最终合规审计；\n"
            "- **参考文献完整性**：正文引用能否映射到已验证文献与文后表；\n"
            "- **证据可追溯性**：段落、证据卡及 PDF 页码绑定；\n"
            "- **主题相关性**：独立审稿发现的主题漂移；\n"
            "- **分析与综合**：比较、综合、差异和局限等论证动作；\n"
            "- **结构与表达**：语言、重复、连贯、术语和内部指令泄漏。"
        )
        for limitation in scorecard.limitations:
            st.caption(f"局限：{limitation}")
        st.download_button(
            "下载工程合规评分 JSON",
            scorecard.model_dump_json(indent=2),
            file_name="veriwrite_paper_quality_scorecard.json",
            mime="application/json",
            width="stretch",
        )


def _evaluate_external_writing_quality(
    target_text: str,
) -> tuple[ExternalWritingEvaluation | None, ExternalWritingEvaluation | None]:
    """Run the MCP judge once per paper fingerprint and retain comparable baselines."""

    target_hash = hashlib.sha256(target_text.encode("utf-8")).hexdigest()
    current = _external_evaluation_from_state(EXTERNAL_WRITING_EVALUATION_KEY)
    baseline = _external_evaluation_from_state(EXTERNAL_WRITING_BASELINE_KEY)
    if current is not None and _external_receipt_matches(current, target_hash):
        return current, baseline

    failure_raw = st.session_state.get(EXTERNAL_WRITING_FAILURE_KEY)
    if isinstance(failure_raw, str):
        try:
            failure = json.loads(failure_raw)
        except json.JSONDecodeError:
            failure = {}
        if (
            failure.get("target_hash") == target_hash
            and failure.get("measurement_version") == "v2-full-document"
        ):
            st.warning(
                "独立写作评分暂不可用，本次交付仍按确定性合规审计继续。"
                f"技术原因：{failure.get('detail', '未知错误')}"
            )
            return None, baseline

    try:
        with st.spinner("独立评分 Agent 正在检查全文结构、论证与语言……"):
            evaluator = ExternalWritingEvaluatorClient(
                ExternalEvaluatorConfig.from_veriwrite_environment(
                    cwd=project_root() / "veriwrite-evaluator"
                )
            )
            candidate = evaluator.evaluate_writing(target_text)
    except (ExternalEvaluatorError, ValueError) as exc:
        st.session_state[EXTERNAL_WRITING_FAILURE_KEY] = json.dumps(
            {
                "target_hash": target_hash,
                "measurement_version": "v2-full-document",
                "detail": str(exc),
            },
            ensure_ascii=False,
        )
        st.warning(
            "独立写作评分暂不可用，但不会阻塞论文合规审计和 DOCX 交付。"
        )
        with st.expander("独立评分技术详情"):
            st.code(str(exc))
        return None, baseline

    if (
        current is not None
        and not _external_receipt_matches(current, target_hash)
        and comparable_external_evaluations(current, candidate)
    ):
        baseline = current
        st.session_state[EXTERNAL_WRITING_BASELINE_KEY] = current.model_dump_json(
            indent=2
        )
    elif baseline is not None and not comparable_external_evaluations(
        baseline, candidate
    ):
        baseline = None
        st.session_state.pop(EXTERNAL_WRITING_BASELINE_KEY, None)
    st.session_state[EXTERNAL_WRITING_EVALUATION_KEY] = candidate.model_dump_json(
        indent=2
    )
    st.session_state.pop(EXTERNAL_WRITING_FAILURE_KEY, None)
    _autosave_current_project()
    return candidate, baseline


def _external_evaluation_from_state(key: str) -> ExternalWritingEvaluation | None:
    raw = st.session_state.get(key)
    if not isinstance(raw, str):
        return None
    try:
        return ExternalWritingEvaluation.model_validate_json(raw)
    except ValueError:
        st.session_state.pop(key, None)
        return None


def _external_receipt_matches(
    evaluation: ExternalWritingEvaluation,
    target_hash: str,
) -> bool:
    inputs = evaluation.receipt.get("inputs")
    if not isinstance(inputs, dict):
        return False
    recorded = inputs.get("target_hash_sha256")
    return (
        isinstance(recorded, str)
        and target_hash.startswith(recorded)
        and evaluation.is_full_document_measurement
    )


def _render_external_writing_quality(
    evaluation: ExternalWritingEvaluation,
    baseline: ExternalWritingEvaluation | None,
) -> None:
    """Render the independent semantic score without replacing hard gates."""

    st.subheader("独立写作质量评分")
    st.caption(
        "Hermes-rubric 通过独立 DeepSeek 审稿模型评估结构、论证、衔接、去重和学术表达；"
        "它不评估事实真伪、DOI 或引文绑定，因此不会覆盖上方确定性合规门禁。"
    )
    warning = external_quality_warning(evaluation)
    if warning:
        st.warning(warning)
    delta = None
    if baseline is not None and comparable_external_evaluations(
        baseline, evaluation
    ):
        delta = round(evaluation.aggregate_100 - baseline.aggregate_100, 1)
    evidence_hits = sum(item.evidence_found for item in evaluation.evidence_citations)
    summary = st.columns(3)
    summary[0].metric(
        "写作质量",
        f"{evaluation.aggregate_100:.1f}/100",
        delta=delta,
    )
    summary[1].metric(
        "证据命中",
        f"{evidence_hits}/{len(evaluation.evidence_citations)}",
    )
    summary[2].metric("低置信维度", len(evaluation.hedge_dims))
    pipeline = evaluation.receipt.get("pipeline", {})
    if evaluation.is_full_document_measurement:
        st.caption(
            "评测覆盖：全文 "
            f"{pipeline.get('target_visible_bytes', 0)}/"
            f"{pipeline.get('target_total_bytes', 0)} UTF-8 字节。"
        )
    st.dataframe(
        [
            {
                "维度": item.name,
                "得分": item.score_100,
                "权重": item.weight,
                "低置信": item.hedge,
            }
            for item in evaluation.dim_summaries
        ],
        hide_index=True,
        width="stretch",
    )
    with st.expander("查看评分证据、方法指纹与导出"):
        st.caption(f"evaluation_method：{evaluation.evaluation_method}")
        if evaluation.hedge_note:
            st.caption(evaluation.hedge_note)
        st.download_button(
            "下载独立写作评分 JSON",
            evaluation.model_dump_json(indent=2),
            file_name="veriwrite_external_writing_evaluation.json",
            mime="application/json",
            width="stretch",
        )


def _render_final_audit_repair(package: FinalPaperPackage) -> None:
    blocking = [issue for issue in package.audit.issues if issue.severity == "blocking"]
    if not blocking:
        return
    repair_stage = final_delivery_repair_stage(package)
    needs_literature_rebuild = repair_stage == "literature"
    needs_evidence_rebuild = repair_stage == "evidence"
    needs_writing_repair = repair_stage == "writing"
    completed_rounds = int(st.session_state.get(GLOBAL_EDITOR_ROUND_KEY, 0) or 0)
    st.subheader("审计修复路由")
    st.caption(
        "正文问题会被定位到具体章节和段落：写作计划、原章节和无问题段落均保留，"
        "只撤销受影响章节的确认。若文献池本身不足或选材失控，则回到 V0.2 "
        "重做准入与提纲，不能靠补写段落凑数。返修前会保存一份可撤销检查点。"
    )
    st.dataframe(
        [
            {
                "问题": issue.code,
                "责任阶段": (
                    "V0.2 文献准入与 V0.4 提纲"
                    if issue.code in LITERATURE_REBUILD_CODES
                    else "V0.3 PDF 身份与证据库"
                    if issue.code in EVIDENCE_REBUILD_CODES
                    else "V0.4 写作计划与正文"
                    if issue.code in WRITING_REPAIR_CODES
                    or issue.code.startswith("body_")
                    or issue.code.startswith("citation_")
                    or issue.code.startswith("length_")
                    else "最终结构组装"
                ),
                "修复动作": (
                    "保留原运行文件，重新准入文献并按合格证据重构提纲"
                    if issue.code in LITERATURE_REBUILD_CODES
                    else "重新扫描 PDF 首页 DOI，不沿用受污染的证据卡"
                    if issue.code in EVIDENCE_REBUILD_CODES
                    else "按原论证计划定点修复受影响段落"
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
    if needs_writing_repair and completed_rounds >= MAX_GLOBAL_EDITOR_ROUNDS:
        st.error(
            f"全局编辑已达到 {MAX_GLOBAL_EDITOR_ROUNDS} 轮上限。系统不会继续盲目重写；"
            "请检查写作计划中的段落职责、重复的中心论点或审稿规则。"
        )
        return
    if needs_literature_rebuild:
        label = "保存检查点并回到 V0.2 重建文献准入与提纲"
        target_stage = "literature"
    elif needs_evidence_rebuild:
        label = "保存检查点并回到 V0.3 重新核验 PDF"
        target_stage = "evidence"
    elif needs_writing_repair:
        label = "创建检查点并生成 V0.4 定点返修任务"
        target_stage = "writing"
    else:
        label = "创建检查点并重建最终结构"
        target_stage = "delivery"
    auto_route = bool(st.session_state.get(V04_AUTOPILOT_REQUESTED_KEY))
    route_requested = auto_route
    if not auto_route:
        route_requested = st.button(
            label,
            type="primary",
            width="stretch",
            key="final_audit_repair",
        )
    if route_requested:
        repair = None
        if needs_writing_repair:
            try:
                feedback_by_section: dict[str, list[str]] = {}
                if package.manuscript_review is not None:
                    for finding in package.manuscript_review.findings:
                        if finding.severity != "blocking":
                            continue
                        feedback_by_section.setdefault(
                            finding.section_id,
                            [],
                        ).append(
                            f"[{finding.code}] {finding.detail} "
                            f"Required correction: {finding.revision_instruction}"
                        )
                structural_planner = GroundedWritingPlanner(
                    DeepSeekClient(LLMSettings().for_structured_output()),
                    reuse_cache=False,
                    repair_feedback_by_section=feedback_by_section,
                )
                repair = build_targeted_writing_repair(
                    st.session_state,
                    package,
                    structural_planner=structural_planner,
                )
            except (TypeError, ValueError) as exc:
                st.error(f"无法创建定点返修任务：{exc}")
                return
        _create_final_repair_checkpoint(
            [issue.code for issue in blocking],
            package=package,
            replace=True,
        )
        if repair is not None:
            st.session_state[WRITING_PLAN_KEY] = repair.plan.model_dump_json(indent=2)
            st.session_state[V04_PROJECT_KEY] = repair.project.model_dump_json(indent=2)
            st.session_state[SECTION_SELECTION_REQUEST_KEY] = next(
                iter(repair.paragraph_numbers)
            )
            st.session_state[GLOBAL_EDITOR_ROUND_KEY] = completed_rounds + 1
            flash = (
                "已保留全部原正文，仅打开 "
                f"{len(repair.paragraph_numbers)} 个章节中的 "
                f"{repair.paragraph_count} 个问题段落进行返修。"
            )
            if st.session_state.get(WRITING_MODE_KEY) == "AI 全托管":
                st.session_state[EVIDENCE_RECOVERY_AUTO_RESUME_KEY] = True
                st.session_state[FINAL_DELIVERY_AUTO_RESUME_KEY] = True
                flash += " AI 全托管将自动继续，不需要再次点击。"
        elif needs_literature_rebuild:
            for key in REPAIR_DOWNSTREAM_KEYS:
                st.session_state.pop(key, None)
            if auto_route:
                st.session_state["literature_auto_run_requested"] = True
            flash = (
                "原检索与 PDF 运行文件仍保留在本地；页面状态已回到 V0.2，"
                "Agent 将按立题卡重新准入并补充文献。合格来源将进入新提纲，"
                "不再对旧正文做翻译式修补。"
            )
        elif needs_evidence_rebuild:
            for key in (
                "v03_pdf_inspection_json",
                "v03_evidence_library_json",
                "v03_writing_handoff_json",
                WRITING_PLAN_KEY,
                V04_PROJECT_KEY,
            ):
                st.session_state.pop(key, None)
            flash = (
                "下载目录和原 PDF 均已保留；页面返回 V0.3 重新核验首页 DOI，"
                "身份冲突的证据卡与正文不会继续沿用。"
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
    replace: bool = False,
) -> None:
    target_state = st.session_state if state is None else state
    if FINAL_REPAIR_CHECKPOINT_KEY in target_state and not replace:
        return
    if package is None and target_state.get(FINAL_PACKAGE_KEY):
        try:
            package = FinalPaperPackage.model_validate_json(
                target_state[FINAL_PACKAGE_KEY]
            )
        except Exception:
            package = None
    payload = {
        "schema_version": "mvp-final-repair-1.1",
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


def render_final_repair_checkpoint_restore() -> None:
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
        st.caption("恢复会覆盖本次修复产生的 V0.2–V0.4 和最终交付页面状态。")
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
            st.session_state["mvp_navigation_request"] = (
                "delivery" if saved_state.get(FINAL_PACKAGE_KEY) else "writing"
            )
            st.session_state["mvp_flash"] = "已恢复审计修复前的版本。"
            st.rerun()


def _render_source_selection_rebuild_control() -> None:
    with st.expander("选材或论证主线失控？执行重构式修订"):
        st.caption(
            "如果问题来自不相关文献、章节对象漂移或大纲骨架错误，不应继续润色旧稿。"
            "系统会保存可撤销检查点，返回 V0.2 重新执行文献准入；本地 runtime "
            "中的原检索文件和 PDF 不会删除。"
        )
        if st.button(
            "保存检查点并重新准入文献、重建提纲",
            width="stretch",
            key="rebuild_from_literature_admission",
        ):
            _create_final_repair_checkpoint(
                ["source_selection_rebuild"],
                replace=True,
            )
            for key in REPAIR_DOWNSTREAM_KEYS:
                st.session_state.pop(key, None)
            st.session_state["mvp_navigation_request"] = "literature"
            st.session_state["mvp_flash"] = (
                "已保存旧版本检查点并返回 V0.2。请先重做文献准入；"
                "新提纲只会使用保留文献，随后按问题而非按论文重写正文。"
            )
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


def _legacy_coverage_count(plan: GroundedWritingPlan) -> int:
    return sum(
        paragraph.coverage_only
        for section in plan.sections
        for paragraph in section.paragraphs
    )


def _status_label(status: str) -> str:
    return {
        "pending": "待生成",
        "draft": "待确认",
        "needs_review": "存在阻塞",
        "confirmed": "已确认",
    }[status]
