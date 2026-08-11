"""Project navigation, progress diagnostics, and checkpoints for the MVP UI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, Mapping, MutableMapping
from uuid import uuid4

import streamlit as st
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from veriwrite_agent.models.evidence import EvidenceLibrary, PdfInspectionBatch
from veriwrite_agent.models.final_delivery import FinalPaperPackage
from veriwrite_agent.models.literature_selection import BalancedLiteratureSelection
from veriwrite_agent.models.requirement_workflow import (
    ConfirmedRequirementSpec,
    RequirementReviewPackage,
)
from veriwrite_agent.models.writing import V04WritingProject
from veriwrite_agent.models.writing_plan import GroundedWritingPlan
from veriwrite_agent.models.writing_handoff import V04WritingHandoff
from veriwrite_agent.services.local_project_store import LocalProjectStore

StageState = Literal["locked", "ready", "in_progress", "blocked", "complete"]

STAGE_LABELS = {
    "overview": "MVP 总览",
    "evaluation": "独立论文评测",
    "requirements": "V0.1 需求确认",
    "literature": "V0.2 文献检索",
    "evidence": "V0.3 全文证据",
    "writing": "V0.4 Agent 写作",
    "delivery": "V0.5 全文编辑与交付",
}

MVP_STATE_KEYS = (
    "review_json",
    "source_name",
    "source_format",
    "extracted_text",
    "extraction_method",
    "extraction_warnings",
    "ocr_average_confidence",
    "elapsed_seconds",
    "source_count",
    "analysis_mode",
    "mvp_smoke_test",
    "editable_extracted_text",
    "selected_requirement_profile",
    "confirmed_json",
    "recovered_executable_policy_json",
    "requirement_recovered_from_executable_policy",
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
    "v04_writing_plan_json",
    "v04_writing_project_json",
    "v04_writing_mode",
    "v04_agent_run_id",
    "v04_autopilot_requested",
    "v04_evidence_recovery_json",
    "v04_evidence_recovery_checkpoint_json",
    "v04_evidence_recovery_auto_plan",
    "v04_evidence_recovery_auto_resume",
    "v04_writing_plan_auto_failures",
    "v04_writing_plan_repair_feedback_json",
    "literature_auto_run_requested",
    "literature_auto_advance_requested",
    "literature_recovery_seed_run_dir",
    "mvp_final_matter_json",
    "mvp_final_paper_json",
    "mvp_final_repair_checkpoint_json",
    "mvp_final_repair_auto_suppressed_id",
    "mvp_final_delivery_auto_resume",
    "mvp_global_editor_round",
    "mvp_ai_declaration",
    "mvp_final_semantic_review_attestation",
    "mvp_paper_quality_scorecard_json",
    "mvp_paper_quality_baseline_json",
    "mvp_external_writing_evaluation_json",
    "mvp_external_writing_baseline_json",
    "mvp_external_writing_failure_json",
)


@dataclass(frozen=True)
class MvpStage:
    """One user-visible, inspectable MVP stage."""

    stage_id: str
    title: str
    state: StageState
    summary: str
    next_action: str
    blockers: tuple[str, ...] = ()


@dataclass(frozen=True)
class MvpProjectStatus:
    """Derived project state; no hidden workflow state is introduced here."""

    stages: tuple[MvpStage, ...]

    @property
    def completed_count(self) -> int:
        return sum(stage.state == "complete" for stage in self.stages)

    @property
    def progress(self) -> float:
        return self.completed_count / len(self.stages)

    @property
    def blockers(self) -> tuple[str, ...]:
        return tuple(blocker for stage in self.stages for blocker in stage.blockers)

    @property
    def next_stage_id(self) -> str:
        for stage in self.stages:
            if stage.state not in {"complete", "locked"}:
                return stage.stage_id
        return "delivery" if self.completed_count == len(self.stages) else "requirements"


class MvpProjectSnapshot(BaseModel):
    """Portable UI checkpoint containing only explicit, JSON-safe business state."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["mvp-ui-1.0"] = "mvp-ui-1.0"
    project_id: str = Field(min_length=1)
    project_name: str = Field(min_length=1, max_length=120)
    active_stage: str = "overview"
    exported_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    state: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def reject_unknown_state(self) -> MvpProjectSnapshot:
        unknown = set(self.state).difference(MVP_STATE_KEYS)
        if unknown:
            raise ValueError(f"项目快照包含未知状态字段：{', '.join(sorted(unknown))}")
        if self.active_stage not in STAGE_LABELS:
            raise ValueError("项目快照包含未知界面阶段")
        return self


def inspect_mvp_status(state: Mapping[str, Any]) -> MvpProjectStatus:
    """Derive every visible stage status from validated pipeline artifacts."""

    requirements = _requirements_status(state)
    literature = _literature_status(state, requirements)
    evidence = _evidence_status(state, literature)
    writing = _writing_status(state, evidence)
    delivery = _delivery_status(state, writing)
    return MvpProjectStatus((requirements, literature, evidence, writing, delivery))


def create_project_snapshot(state: Mapping[str, Any]) -> MvpProjectSnapshot:
    """Create a safe checkpoint without uploaded file handles or API credentials."""

    serializable: dict[str, JsonValue] = {}
    for key in MVP_STATE_KEYS:
        if key not in state:
            continue
        try:
            serializable[key] = json.loads(json.dumps(state[key], ensure_ascii=False))
        except (TypeError, ValueError):
            continue
    return MvpProjectSnapshot(
        project_id=str(state.get("mvp_project_id") or uuid4()),
        project_name=str(state.get("mvp_project_name") or "未命名论文项目"),
        active_stage=_normalize_live_stage(state.get("mvp_navigation")),
        state=serializable,
    )


def apply_pending_project_actions(
    *,
    local_store: LocalProjectStore | None = None,
) -> None:
    """Apply navigation/import/reset requests before keyed widgets are created."""

    if st.session_state.pop("mvp_reset_request", False):
        if local_store is not None:
            local_store.clear()
        for key in list(st.session_state):
            st.session_state.pop(key, None)
        return
    restored_meta = st.session_state.pop("mvp_restore_meta", None)
    if restored_meta:
        st.session_state["mvp_project_id"] = restored_meta["project_id"]
        st.session_state["mvp_project_name"] = restored_meta["project_name"]
        st.session_state["mvp_navigation"] = restored_meta["active_stage"]
    requested_stage = st.session_state.pop("mvp_navigation_request", None)
    if requested_stage:
        st.session_state["mvp_navigation"] = requested_stage
    elif "mvp_navigation" in st.session_state:
        st.session_state["mvp_navigation"] = _normalize_live_stage(
            st.session_state["mvp_navigation"]
        )


def restore_local_project_if_needed(
    state: MutableMapping[str, Any],
    store: LocalProjectStore,
) -> bool:
    """Restore an autosave only when the new Streamlit session has no progress."""

    if _has_mvp_progress(state):
        return False
    serialized = store.load()
    if not serialized:
        return False
    snapshot = MvpProjectSnapshot.model_validate_json(serialized)
    for key, value in snapshot.state.items():
        state[key] = value
    state["mvp_project_id"] = snapshot.project_id
    state["mvp_project_name"] = snapshot.project_name
    state["mvp_navigation"] = snapshot.active_stage
    return True


def autosave_local_project(
    state: Mapping[str, Any],
    store: LocalProjectStore,
) -> bool:
    """Persist meaningful MVP state so a browser refresh cannot erase progress."""

    if not _has_mvp_progress(state):
        return False
    snapshot = create_project_snapshot(state)
    store.save(snapshot.model_dump_json(indent=2))
    return True


def _has_mvp_progress(state: Mapping[str, Any]) -> bool:
    return any(
        key in state and state[key] not in (None, "", [], {}, ())
        for key in MVP_STATE_KEYS
    )


def render_project_sidebar(status: MvpProjectStatus) -> str:
    """Render project controls and return the selected stage id."""

    st.session_state.setdefault("mvp_project_id", str(uuid4()))
    st.session_state.setdefault("mvp_project_name", "未命名课程论文")
    st.session_state.setdefault("mvp_navigation", "overview")

    with st.sidebar:
        st.markdown("## VeriWrite MVP")
        st.text_input("项目名称", key="mvp_project_name")
        st.progress(status.progress, text=f"全链路完成 {status.progress:.0%}")
        selected = st.radio(
            "工作阶段",
            options=list(STAGE_LABELS),
            format_func=lambda stage_id: _navigation_label(stage_id, status),
            key="mvp_navigation",
        )

        with st.expander("项目与恢复"):
            st.caption(f"项目编号：{st.session_state['mvp_project_id'][:8]}")
            snapshot = create_project_snapshot(st.session_state)
            st.download_button(
                "导出检查点",
                snapshot.model_dump_json(indent=2),
                file_name="veriwrite_mvp_project.json",
                mime="application/json",
                width="stretch",
                help="保存已提取文本和各阶段 JSON；不会保存 API 密钥或原始上传文件。",
            )
            upload = st.file_uploader(
                "恢复检查点",
                type=["json"],
                key="mvp_snapshot_upload",
            )
            if st.button(
                "载入检查点",
                disabled=upload is None,
                width="stretch",
            ):
                try:
                    payload = upload.getvalue()
                    if len(payload) > 10 * 1024 * 1024:
                        raise ValueError("项目检查点不能超过 10 MB")
                    restored = MvpProjectSnapshot.model_validate_json(payload)
                except Exception as exc:
                    st.error(f"检查点无效：{exc}")
                else:
                    _restore_snapshot(restored)
                    st.rerun()

            st.divider()
            confirmed = st.checkbox("清除当前页面中的全部项目状态")
            if st.button("新建项目", disabled=not confirmed, width="stretch"):
                st.session_state["mvp_reset_request"] = True
                st.rerun()
    return selected


def render_mvp_overview(status: MvpProjectStatus) -> None:
    """Render the cross-stage dashboard and one clear next action."""

    st.header("项目进度")
    st.caption("需求 → 文献 → 全文证据 → 正文 → 最终论文")
    metrics = st.columns(3)
    metrics[0].metric("总体进度", f"{status.progress:.0%}")
    metrics[1].metric("完成阶段", f"{status.completed_count}/{len(status.stages)}")
    metrics[2].metric("当前阻塞", len(status.blockers))
    final_complete = status.stages[-1].state == "complete"
    st.progress(status.progress)

    icons = {
        "locked": "🔒",
        "ready": "○",
        "in_progress": "◐",
        "blocked": "!",
        "complete": "✓",
    }
    labels = {
        "locked": "未解锁",
        "ready": "可开始",
        "in_progress": "进行中",
        "blocked": "有阻塞",
        "complete": "已完成",
    }
    next_stage = status.next_stage_id
    if final_complete:
        st.success("MVP 全链路已完成，可以在“最终交付”下载论文和审计包。")
        if st.button("查看最终交付", type="primary", width="stretch"):
            st.session_state["mvp_navigation_request"] = "delivery"
            st.rerun()
    else:
        current = next(stage for stage in status.stages if stage.stage_id == next_stage)
        with st.container(border=True):
            st.markdown(f"### {icons[current.state]} 现在做：{current.title}")
            st.write(current.summary)
            if current.blockers:
                for blocker in current.blockers:
                    st.error(blocker)
            st.caption(current.next_action)
        if st.button(
            f"继续：{STAGE_LABELS[next_stage]}",
            type="primary",
            width="stretch",
        ):
            st.session_state["mvp_navigation_request"] = next_stage
            st.rerun()

    with st.expander("查看全部阶段"):
        for stage in status.stages:
            left, right = st.columns([4, 1])
            left.markdown(f"**{icons[stage.state]} {stage.title}**")
            left.caption(stage.summary)
            right.caption(labels[stage.state])
            if stage.blockers:
                for blocker in stage.blockers:
                    st.caption(f"! {blocker}")


def render_locked_stage(stage: MvpStage, previous_stage_id: str) -> None:
    """Explain a locked stage instead of silently hiding it."""

    st.header(stage.title)
    st.warning(stage.summary)
    st.caption(f"解锁条件：{stage.next_action}")
    if st.button(
        f"前往 {STAGE_LABELS[previous_stage_id]}",
        type="primary",
        width="stretch",
    ):
        st.session_state["mvp_navigation_request"] = previous_stage_id
        st.rerun()


def _requirements_status(state: Mapping[str, Any]) -> MvpStage:
    if state.get("confirmed_json"):
        try:
            confirmed = ConfirmedRequirementSpec.model_validate_json(state["confirmed_json"])
        except Exception as exc:
            return _invalid_stage("requirements", "V0.1 需求确认", exc)
        topic = confirmed.requirement.topic or "未命名主题"
        return MvpStage(
            "requirements",
            "V0.1 需求确认",
            "complete",
            f"需求合同已确认：{topic}",
            "V0.2 可生成并确认检索蓝图。",
        )
    if not state.get("review_json"):
        return MvpStage(
            "requirements",
            "V0.1 需求确认",
            "ready",
            "尚未导入课程要求。",
            "上传 Word、PDF、图片或文本并执行双路提取。",
        )
    try:
        review = RequirementReviewPackage.model_validate_json(state["review_json"])
    except Exception as exc:
        return _invalid_stage("requirements", "V0.1 需求确认", exc)
    blockers = tuple(issue.message for issue in review.completeness.issues if issue.severity == "blocking")
    if review.reconciliation.conflicts:
        blockers += (f"仍有 {len(review.reconciliation.conflicts)} 个双路字段冲突待确认。",)
    return MvpStage(
        "requirements",
        "V0.1 需求确认",
        "blocked" if blockers else "in_progress",
        "文本已提取，正在校对并确认 RequirementSpec。",
        "处理关键冲突与阻塞项，生成最终需求合同。",
        blockers,
    )


def _literature_status(state: Mapping[str, Any], dependency: MvpStage) -> MvpStage:
    if dependency.state != "complete":
        return _locked("literature", "V0.2 文献检索", "先完成 V0.1 需求确认。")
    if not state.get("literature_blueprint_json"):
        return MvpStage(
            "literature",
            "V0.2 文献检索",
            "ready",
            "需求已就绪，尚未生成检索蓝图。",
            "生成临时大纲和检索词，并由用户确认后再联网检索。",
        )
    if not state.get("literature_confirmed_blueprint_json"):
        return MvpStage(
            "literature",
            "V0.2 文献检索",
            "in_progress",
            "临时检索蓝图待用户确认。",
            "核对主题覆盖、检索词和各主题配额。",
        )
    if not state.get("literature_result_json"):
        return MvpStage(
            "literature",
            "V0.2 文献检索",
            "in_progress",
            "检索蓝图已确认，尚未得到最终文献池。",
            "执行或继续 Crossref 检索、DOI/RIS 验证与均衡选择。",
        )
    try:
        payload = json.loads(str(state["literature_result_json"]))
        selection = BalancedLiteratureSelection.model_validate(payload["selection"])
    except Exception as exc:
        return _invalid_stage("literature", "V0.2 文献检索", exc)
    if _literature_admission_incomplete(selection):
        return MvpStage(
            "literature",
            "V0.2 文献检索",
            "blocked",
            "文献真实性已验证，但主题相关性和写作用途准入尚未完成。",
            "按立题卡重新执行相关性准入；本地 PDF 文件不会因此删除。",
            ("存在旧版或不完整的文献准入记录，禁止进入 V0.3/V0.4。",),
        )
    blockers = tuple(selection.policy_issues)
    blockers += tuple(f"主题 {theme_id} 还缺 {count} 篇" for theme_id, count in selection.shortages.items())
    if selection.target_reached:
        return MvpStage(
            "literature",
            "V0.2 文献检索",
            "complete",
            f"已获得 {len(selection.selected)} 篇通过真实性验证且满足配额的文献。",
            "选择核心论文并获取全文 PDF。",
        )
    return MvpStage(
        "literature",
        "V0.2 文献检索",
        "blocked",
        f"已选 {len(selection.selected)} 篇，但配额或 V0.1 策略尚未满足。",
        "在V0.2选择扩大候选池自动补搜，或返回修改检索蓝图。",
        blockers or ("文献选择尚未达到目标。",),
    )


def _literature_admission_incomplete(
    selection: BalancedLiteratureSelection,
) -> bool:
    if not selection.blueprint.topic_boundary.is_actionable:
        return True
    return any(
        item.admission_status != "admit"
        or item.centrality not in {"central", "supporting"}
        or not item.supported_claim
        or not item.suitable_section_id
        or not item.use_boundary
        for item in selection.selected
    )


def _evidence_status(state: Mapping[str, Any], dependency: MvpStage) -> MvpStage:
    if dependency.state != "complete":
        return _locked("evidence", "V0.3 全文证据", "先完成 V0.2 真实文献选择。")
    if state.get("v03_writing_handoff_json"):
        try:
            handoff = V04WritingHandoff.model_validate_json(state["v03_writing_handoff_json"])
        except Exception as exc:
            return _invalid_stage("evidence", "V0.3 全文证据", exc)
        return MvpStage(
            "evidence",
            "V0.3 全文证据",
            "complete",
            f"证据库已确认：{len(handoff.evidence_library.evidence_cards)} 张证据卡。",
            "进入 V0.4，根据实际证据生成并确认段落级写作计划。",
        )
    if not state.get("v03_pdf_inspection_json"):
        return MvpStage(
            "evidence",
            "V0.3 全文证据",
            "ready",
            "尚未检查核心论文 PDF。",
            "选择核心论文、下载 PDF，并扫描下载目录。",
        )
    try:
        batch = PdfInspectionBatch.model_validate_json(state["v03_pdf_inspection_json"])
    except Exception as exc:
        return _invalid_stage("evidence", "V0.3 全文证据", exc)
    unavailable = tuple(
        f"{report.expectation.title}：{report.status}"
        for report in batch.reports
        if report.status != "verified"
    )
    if not state.get("v03_evidence_library_json"):
        verified = sum(report.status == "verified" for report in batch.reports)
        return MvpStage(
            "evidence",
            "V0.3 全文证据",
            "in_progress" if verified else "blocked",
            f"核心 PDF 已验证 {verified}/{len(batch.reports)} 篇，证据库尚未确认。",
            "补齐核心 PDF，并从已验证全文生成证据卡。",
            unavailable if not verified else (),
        )
    try:
        library = EvidenceLibrary.model_validate_json(state["v03_evidence_library_json"])
    except Exception as exc:
        return _invalid_stage("evidence", "V0.3 全文证据", exc)
    blockers = tuple(library.unresolved_issues)
    return MvpStage(
        "evidence",
        "V0.3 全文证据",
        "blocked" if blockers else "in_progress",
        f"证据库草案包含 {len(library.evidence_cards)} 张证据卡，等待完整性检查和确认。",
        "处理全文/OCR/章节覆盖问题并生成 V0.4 交接包。",
        blockers,
    )


def _writing_status(state: Mapping[str, Any], dependency: MvpStage) -> MvpStage:
    recovery_status = _active_writing_recovery_status(state)
    if recovery_status is not None:
        labels = {
            "pending_search": "Agent 正在内部回退并补搜缺失论点。",
            "pending_full_text": "Agent 正在检查全文并重建证据。",
            "ready_to_resume": "Agent 已补齐证据，正在恢复受影响章节。",
            "blocked": "自动替代已经用尽，等待一次性补充受限 PDF。",
        }
        return MvpStage(
            "writing",
            "V0.4 Agent 写作",
            "blocked" if recovery_status == "blocked" else "in_progress",
            labels[recovery_status],
            "留在 V0.4 查看 Agent 计划、执行、审核与回退状态。",
            (
                ("需要一次性下载合并清单中的受限 PDF。",)
                if recovery_status == "blocked"
                else ()
            ),
        )
    if dependency.state != "complete":
        return _locked("writing", "V0.4 逐章写作", "先确认 V0.3 证据库和最终写作大纲。")
    legacy_project = None
    if state.get("v04_writing_project_json"):
        try:
            legacy_project = V04WritingProject.model_validate_json(
                state["v04_writing_project_json"]
            )
        except Exception as exc:
            return _invalid_stage("writing", "V0.4 逐章写作", exc)
        if legacy_project.status == "body_complete":
            return MvpStage(
                "writing",
                "V0.4 逐章写作",
                "complete",
                f"{len(legacy_project.sections)}/{len(legacy_project.sections)} 个正文章节已确认。",
                "组装标题、摘要、关键词、结论和参考文献。",
            )
    if not state.get("v04_writing_plan_json"):
        return MvpStage(
            "writing",
            "V0.4 逐章写作",
            "ready",
            "V0.3 写作交接包已就绪，尚未根据实际证据规划段落。",
            "生成并确认一次证据约束写作计划。",
        )
    try:
        plan = GroundedWritingPlan.model_validate_json(state["v04_writing_plan_json"])
    except Exception as exc:
        return _invalid_stage("writing", "V0.4 逐章写作", exc)
    if plan.status == "draft":
        return MvpStage(
            "writing",
            "V0.4 逐章写作",
            "in_progress",
            f"写作计划草案包含 {len(plan.sections)} 个章节，等待一次性确认。",
            "核对章节、段落和证据分配后采用计划。",
        )
    if not state.get("v04_writing_project_json"):
        return MvpStage(
            "writing",
            "V0.4 逐章写作",
            "in_progress",
            "证据约束写作计划已确认。",
            "按锁定的段落证据包生成并确认正文。",
        )
    project = legacy_project or V04WritingProject.model_validate_json(
        state["v04_writing_project_json"]
    )
    confirmed = sum(section.status == "confirmed" for section in project.sections)
    blockers = tuple(
        f"章节 {section.section_id} 的引用或证据审计未通过"
        for section in project.sections
        if section.status == "needs_review"
    )
    if project.status == "body_complete":
        return MvpStage(
            "writing",
            "V0.4 逐章写作",
            "complete",
            f"{confirmed}/{len(project.sections)} 个正文章节已确认。",
            "组装标题、摘要、关键词、结论和参考文献。",
        )
    return MvpStage(
        "writing",
        "V0.4 逐章写作",
        "blocked" if blockers else "in_progress",
        f"已确认 {confirmed}/{len(project.sections)} 个章节。",
        "继续生成、审计并确认未完成章节。",
        blockers,
    )


def _active_writing_recovery_status(state: Mapping[str, Any]) -> str | None:
    raw = state.get("v04_evidence_recovery_json")
    if not raw:
        return None
    try:
        payload = json.loads(str(raw))
    except (TypeError, json.JSONDecodeError):
        return None
    status = payload.get("status")
    return (
        str(status)
        if status in {
            "pending_search",
            "pending_full_text",
            "ready_to_resume",
            "blocked",
        }
        else None
    )


def _delivery_status(state: Mapping[str, Any], dependency: MvpStage) -> MvpStage:
    if dependency.state != "complete":
        return _locked("delivery", "最终交付", "先完成并确认全部正文章节。")
    if not state.get("mvp_final_paper_json"):
        return MvpStage(
            "delivery",
            "最终交付",
            "ready",
            "正文已确认，尚未组装最终论文。",
            "生成最终组成部分并执行要求、引用和参考文献审计。",
        )
    try:
        package = FinalPaperPackage.model_validate_json(state["mvp_final_paper_json"])
    except Exception as exc:
        return _invalid_stage("delivery", "最终交付", exc)
    blockers = tuple(issue.detail for issue in package.audit.issues if issue.severity == "blocking")
    if package.status == "confirmed":
        return MvpStage(
            "delivery",
            "最终交付",
            "complete",
            f"最终论文已确认，引用 {package.audit.reference_count} 篇文献。",
            "下载 Markdown、DOCX 和完整审计包。",
        )
    if package.status == "needs_revision":
        return MvpStage(
            "delivery",
            "最终交付",
            "blocked",
            "最终合规审计未通过。",
            "按审计问题修改正文、声明或文献配置。",
            blockers or ("最终论文需要修订。",),
        )
    return MvpStage(
        "delivery",
        "最终交付",
        "in_progress",
        "最终论文已通过自动审计，等待用户确认。",
        "人工核对并解锁最终交付文件。",
    )


def _locked(stage_id: str, title: str, action: str) -> MvpStage:
    return MvpStage(stage_id, title, "locked", "上一阶段尚未完成。", action)


def _invalid_stage(stage_id: str, title: str, exc: Exception) -> MvpStage:
    return MvpStage(
        stage_id,
        title,
        "blocked",
        "保存的阶段数据无法通过数据合同校验。",
        "重新生成该阶段产物，或载入有效项目检查点。",
        (f"{type(exc).__name__}: {exc}",),
    )


def _navigation_label(stage_id: str, status: MvpProjectStatus) -> str:
    if stage_id in {"overview", "evaluation"}:
        return STAGE_LABELS[stage_id]
    stage = next(item for item in status.stages if item.stage_id == stage_id)
    marker = {
        "locked": "🔒",
        "ready": "○",
        "in_progress": "◐",
        "blocked": "!",
        "complete": "✓",
    }[stage.state]
    return f"{marker} {STAGE_LABELS[stage_id]}"


def _normalize_live_stage(value: Any) -> str:
    """Recover the internal stage id from a Streamlit display label.

    Streamlit can briefly preserve the formatted radio label when its status
    marker changes after a stage mutation. Checkpoints store stable ids only.
    """

    raw = str(value or "overview").strip()
    if raw in STAGE_LABELS:
        return raw
    legacy_labels = {
        "V0.4 逐章写作": "writing",
        "最终交付": "delivery",
    }
    for label, stage_id in legacy_labels.items():
        if raw == label or raw.endswith(f" {label}"):
            return stage_id
    matching_ids = [
        stage_id
        for stage_id, label in STAGE_LABELS.items()
        if raw == label or raw.endswith(f" {label}")
    ]
    return matching_ids[0] if len(matching_ids) == 1 else "overview"


def _restore_snapshot(snapshot: MvpProjectSnapshot) -> None:
    for key in MVP_STATE_KEYS:
        st.session_state.pop(key, None)
    for key, value in snapshot.state.items():
        st.session_state[key] = value
    st.session_state["mvp_restore_meta"] = {
        "project_id": snapshot.project_id,
        "project_name": snapshot.project_name,
        "active_stage": snapshot.active_stage,
    }
