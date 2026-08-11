"""Streamlit entry point for the integrated V0.1 and V0.2 console."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import streamlit as st

from veriwrite_agent.config.settings import LLMSettings
from veriwrite_agent.models.requirement_workflow import (
    ConfirmedRequirementSpec,
    RequirementConfirmation,
    RequirementConflict,
    RequirementReviewPackage,
)
from veriwrite_agent.models.literature_selection import BalancedLiteratureSelection
from veriwrite_agent.models.writing_handoff import V04WritingHandoff
from veriwrite_agent.services.requirement_confirmation import (
    RequirementConfirmationError,
    RequirementConfirmationService,
)
from veriwrite_agent.services.requirement_review_renderer import (
    RequirementReviewRenderer,
)
from veriwrite_agent.services.literature_run_recovery import (
    LiteratureRunRecoveryService,
)
from veriwrite_agent.services.local_project_store import LocalProjectStore
from veriwrite_agent.ui.literature_console import (
    clear_literature_state,
    render_literature_console,
)
from veriwrite_agent.ui.evidence_console import render_pdf_acquisition_console
from veriwrite_agent.ui.mvp_console import (
    apply_pending_project_actions,
    autosave_local_project,
    inspect_mvp_status,
    render_locked_stage,
    render_mvp_overview,
    render_project_sidebar,
    restore_local_project_if_needed,
)
from veriwrite_agent.ui.paper_evaluation_console import (
    render_paper_evaluation_console,
)
from veriwrite_agent.ui.writing_console import (
    active_writing_recovery_status,
    render_final_delivery_console,
    render_final_repair_checkpoint_restore,
    render_grounded_writing_console,
    render_writing_agent_recovery_shell,
    rollback_blocked_delivery_to_v04,
    rollback_outdated_delivery_to_v04,
    upgrade_legacy_full_rebuild_repair,
)
from veriwrite_agent.ui.workbench import (
    WorkbenchResult,
    built_in_samples,
    comparison_rows,
    prepare_review_from_path,
    prepare_review_from_text,
    prepare_review_from_uploads,
    project_root,
)


def run() -> None:
    st.set_page_config(
        page_title="VeriWrite Agent MVP 工作台",
        page_icon="✓",
        layout="wide",
    )
    _inject_styles()
    autosave_override = os.getenv("VERIWRITE_AUTOSAVE_PATH")
    local_store = LocalProjectStore(
        Path(autosave_override)
        if autosave_override
        else project_root() / "runtime" / "mvp_projects" / "active_project.json"
    )
    apply_pending_project_actions(local_store=local_store)
    autosave_error = None
    restored_from_autosave = False
    try:
        restored_from_autosave = restore_local_project_if_needed(
            st.session_state,
            local_store,
        )
        upgrade_legacy_full_rebuild_repair(st.session_state)
        rollback_outdated_delivery_to_v04(st.session_state)
        rollback_blocked_delivery_to_v04(st.session_state)
        if active_writing_recovery_status(st.session_state) is not None:
            # Literature/PDF recovery is an internal Agent tool call.  A restored
            # browser session must return to the V0.4 control surface even when
            # an older checkpoint did not persist the transient autopilot flag.
            st.session_state["mvp_navigation"] = "writing"
        autosave_local_project(st.session_state, local_store)
    except (OSError, ValueError) as exc:
        autosave_error = str(exc)
    status = inspect_mvp_status(st.session_state)
    selected_stage = render_project_sidebar(status)

    st.title("VeriWrite Agent MVP 工作台")
    st.caption(
        "从课程要求到最终 DOCX 的可恢复 Agent 工作流 · "
        "LLM 负责语义任务，确定性代码负责合同、身份、引用与审计"
    )
    flash_message = st.session_state.pop("mvp_flash", None)
    if flash_message:
        st.success(flash_message)
    if st.session_state.get("mvp_smoke_test"):
        st.info(
            "当前是快速全链路联通测试：目标仅为验证 V0.1→V0.4→最终交付能否跑通。"
            "该模式使用 2 篇真实文献、1 篇核心 PDF 和 600–800 英文词，"
            "并将检索主题合并为一个最小正文单元；不能代表真实课程论文的质量或性能。"
        )
    if restored_from_autosave:
        st.success("已从本地自动存档恢复刷新前的项目进度。")
    if autosave_error:
        st.warning(f"本地项目自动存档暂不可用：{autosave_error}")
    _render_runtime_recovery_offer()
    render_final_repair_checkpoint_restore()

    stages = {stage.stage_id: stage for stage in status.stages}
    if selected_stage == "overview":
        render_mvp_overview(status)
    elif selected_stage == "evaluation":
        render_paper_evaluation_console()
    elif selected_stage == "requirements":
        render_requirement_console()
    elif selected_stage == "literature":
        if stages["literature"].state == "locked":
            render_locked_stage(stages["literature"], status.next_stage_id)
        else:
            render_literature_console(include_downstream=False)
    elif selected_stage == "evidence":
        if stages["evidence"].state == "locked":
            render_locked_stage(stages["evidence"], status.next_stage_id)
        else:
            selection = _restore_literature_selection()
            render_pdf_acquisition_console(selection, include_writing=False)
    elif selected_stage == "writing":
        recovery_status = active_writing_recovery_status(st.session_state)
        if recovery_status in {"pending_search", "pending_full_text", "blocked"}:
            render_writing_agent_recovery_shell()
            if recovery_status == "pending_search":
                render_literature_console(
                    include_downstream=False,
                    agent_embedded=True,
                )
            else:
                selection = _restore_literature_selection()
                render_pdf_acquisition_console(
                    selection,
                    include_writing=False,
                    agent_embedded=True,
                )
        elif recovery_status == "ready_to_resume" and not st.session_state.get(
            "v03_writing_handoff_json"
        ):
            render_writing_agent_recovery_shell()
            selection = _restore_literature_selection()
            render_pdf_acquisition_console(
                selection,
                include_writing=False,
                agent_embedded=True,
            )
        elif stages["writing"].state == "locked":
            render_locked_stage(stages["writing"], status.next_stage_id)
        else:
            handoff = V04WritingHandoff.model_validate_json(
                st.session_state["v03_writing_handoff_json"]
            )
            render_grounded_writing_console(handoff, include_final_delivery=False)
    elif selected_stage == "delivery":
        if stages["delivery"].state == "locked":
            render_locked_stage(stages["delivery"], status.next_stage_id)
        else:
            handoff = V04WritingHandoff.model_validate_json(
                st.session_state["v03_writing_handoff_json"]
            )
            render_final_delivery_console(handoff)


def _render_runtime_recovery_offer() -> None:
    """Offer disaster recovery when old sessions predate local autosave support."""

    if st.session_state.get("confirmed_json") or st.session_state.get("review_json"):
        return
    recovery = LiteratureRunRecoveryService().latest(
        project_root() / "runtime" / "literature_console"
    )
    if recovery is None:
        return
    topic = recovery.confirmed_requirement.requirement.topic or "未命名主题"
    st.warning(
        "检测到刷新前保存在本地的V0.2运行结果："
        f"{topic}，已选 {recovery.selected_count}/{recovery.target_total} 篇。"
        "可以恢复，无需重新执行V0.1或重新验证已有论文。"
    )
    if st.button(
        "恢复最近一次本地项目进度",
        type="primary",
        width="stretch",
    ):
        for key, value in recovery.session_state().items():
            st.session_state[key] = value
        st.session_state["mvp_restore_meta"] = {
            "project_id": st.session_state["mvp_project_id"],
            "project_name": f"{topic}（已恢复）",
            "active_stage": "literature",
        }
        st.rerun()


def render_requirement_console() -> None:
    """Render V0.1 as one independently navigable MVP stage."""

    st.header("V0.1 确认课程要求")
    if "confirmed_json" in st.session_state:
        _render_confirmed_requirement()
        return

    has_review = "review_json" in st.session_state
    input_container = st.expander("更换要求文件并重新分析") if has_review else st.container()
    with input_container:
        source_kind, selected_sample, uploaded_files, pasted_text, mode = _render_input_panel()
        if st.button("分析课程要求", type="primary", width="stretch"):
            try:
                with st.spinner("正在提取文本并检查要求…"):
                    if source_kind == "试用示例":
                        result = prepare_review_from_path(
                            selected_sample.path,
                            mode=mode,
                        )
                    elif source_kind == "粘贴文本":
                        result = prepare_review_from_text(
                            pasted_text,
                            mode=mode,
                            source_name="粘贴的课程要求",
                            source_format="text",
                        )
                    else:
                        if not uploaded_files:
                            st.error("请先上传要求文件。")
                            st.stop()
                        result = prepare_review_from_uploads(
                            [
                                (uploaded_file.name, uploaded_file.getvalue())
                                for uploaded_file in uploaded_files
                            ],
                            mode=mode,
                        )
            except Exception as exc:
                st.error(f"分析失败：{exc}")
            else:
                _store_result(result, mode=mode)
                st.session_state["mvp_smoke_test"] = bool(
                    source_kind == "试用示例" and selected_sample.smoke_test
                )
                st.session_state.pop("confirmed_json", None)
                clear_literature_state()
                st.rerun()

    if has_review:
        result = _restore_result()
        _render_result(result)


def _render_confirmed_requirement() -> None:
    confirmed = ConfirmedRequirementSpec.model_validate_json(st.session_state["confirmed_json"])
    st.success(f"课程要求已确认：{confirmed.requirement.topic or '未命名主题'}")
    _render_requirement_facts(confirmed.requirement)
    if confirmed.remaining_warnings:
        with st.expander(f"查看 {len(confirmed.remaining_warnings)} 条已知提醒"):
            for issue in confirmed.remaining_warnings:
                st.markdown(f"- {issue.message}")
    if st.button("进入 V0.2 文献检索", type="primary", width="stretch"):
        st.session_state["mvp_navigation_request"] = "literature"
        st.rerun()
    with st.expander("技术产物"):
        st.download_button(
            "下载确认后的 RequirementSpec",
            st.session_state["confirmed_json"],
            file_name="confirmed_requirement_spec.json",
            mime="application/json",
            width="stretch",
        )
        st.json(json.loads(st.session_state["confirmed_json"]))


def _restore_literature_selection() -> BalancedLiteratureSelection:
    payload = json.loads(st.session_state["literature_result_json"])
    return BalancedLiteratureSelection.model_validate(payload["selection"])


def _render_input_panel() -> tuple[str, Any, Any, str, str]:
    st.subheader("提供课程要求")
    left, right = st.columns([1.4, 1])
    samples = built_in_samples()
    with left:
        source_kind = st.radio(
            "要求来源",
            ["上传文件", "粘贴文本", "试用示例"],
            horizontal=True,
        )
        selected_sample = samples[0]
        uploaded_files = []
        pasted_text = ""
        if source_kind == "试用示例":
            labels = [sample.label for sample in samples]
            label = st.selectbox("测试样例", labels)
            selected_sample = next(sample for sample in samples if sample.label == label)
            st.caption(selected_sample.focus)
        elif source_kind == "粘贴文本":
            pasted_text = st.text_area(
                "粘贴老师发布的课程要求",
                height=220,
                placeholder="粘贴题目、篇幅、参考文献、格式、截止时间等要求……",
            )
        else:
            uploaded_files = (
                st.file_uploader(
                    "上传课程要求（连续截图请按阅读顺序一次选择）",
                    type=[
                        "txt",
                        "md",
                        "docx",
                        "doc",
                        "pdf",
                        "png",
                        "jpg",
                        "jpeg",
                        "tif",
                        "tiff",
                        "bmp",
                        "webp",
                    ],
                    help=(
                        "图片和扫描 PDF 使用本地 OCR；"
                        "旧 DOC 自动调用本机 Word 转换；"
                        "多张滚动截图会自动去重并合并。"
                    ),
                    accept_multiple_files=True,
                )
                or []
            )
            image_uploads = [
                uploaded_file
                for uploaded_file in uploaded_files
                if uploaded_file.name.lower().endswith(
                    (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")
                )
            ]
            if image_uploads:
                st.image(
                    image_uploads,
                    caption=[
                        f"{index}. {item.name}" for index, item in enumerate(image_uploads, 1)
                    ],
                    width=240,
                )
    with right:
        try:
            summary = LLMSettings().public_summary()
        except Exception:
            summary = None
        mode_label = st.radio(
            "理解方式",
            ["仅规则解析", "规则 + DeepSeek 交叉检查"],
            index=1 if summary is not None else 0,
            help="交叉检查更适合真实课程要求；仅规则解析用于离线调试。",
        )
        mode = "dual" if "DeepSeek" in mode_label else "rule"
        if mode == "dual":
            if summary is not None:
                st.success(f"已配置模型：{summary['model']} · 将产生一次 API 调用")
            else:
                st.error("DeepSeek 配置不可用，请先使用仅规则解析或补充环境配置。")
        else:
            st.info("离线解析，不产生 API 费用。")
    return source_kind, selected_sample, uploaded_files, pasted_text, mode


def _render_result(result: WorkbenchResult) -> None:
    review = result.review
    st.divider()
    st.subheader("核对需要执行的要求")
    for warning in result.extraction_warnings:
        st.warning(warning)

    if review.status == "needs_resolution":
        st.warning("还有关键字段需要处理，完成下方项目后才能检索文献。")
    else:
        st.success("没有关键冲突，请核对主题和默认值后继续。")

    _render_summary(review)
    _render_confirmation(review)

    with st.expander("解析详情与校对"):
        extraction_label = {
            "native": "原生文本",
            "ocr": "本地 OCR",
            "mixed": "文本 + OCR",
        }.get(result.extraction_method, result.extraction_method)
        llm_label = "未启用"
        if review.llm_run is not None:
            llm_label = "成功" if review.llm_run.status == "succeeded" else "失败"
        st.caption(
            f"{result.source_format.upper().lstrip('.')} · {extraction_label} · "
            f"{len(result.extracted_text)} 字符 · LLM {llm_label} · "
            f"{result.elapsed_seconds:.1f} 秒"
        )
        if result.source_count > 1:
            st.caption(f"已按顺序合并 {result.source_count} 个输入文件。")
        if result.ocr_average_confidence is not None:
            st.info(
                f"OCR 平均置信度：{result.ocr_average_confidence:.1%}。"
                "如原文有扫描页，请重点校对关键数字和日期。"
            )
        _render_comparison(review)
        _render_issues(review)
        st.markdown("#### 可校对的提取文本")
        st.caption("OCR 置信度高也可能漏字符。可直接修正文字，再按修正版重新执行规则与 LLM。")
        st.session_state.setdefault("editable_extracted_text", result.extracted_text)
        edited_text = st.text_area(
            "文本",
            height=360,
            key="editable_extracted_text",
            label_visibility="collapsed",
        )
        if st.button("按校对文本重新分析", width="stretch"):
            try:
                rerun_result = prepare_review_from_text(
                    edited_text,
                    mode=st.session_state.get("analysis_mode", "rule"),
                    source_name=result.source_name,
                    source_format=result.source_format,
                    extraction_method=result.extraction_method,
                    extraction_warnings=result.extraction_warnings,
                    ocr_average_confidence=result.ocr_average_confidence,
                    source_count=result.source_count,
                )
            except Exception as exc:
                st.error(f"重新分析失败：{exc}")
            else:
                _store_result(
                    rerun_result,
                    mode=st.session_state.get("analysis_mode", "rule"),
                    reset_editable_text=False,
                )
                st.session_state.pop("confirmed_json", None)
                clear_literature_state()
                st.rerun()
        with st.expander("查看完整审查 JSON"):
            st.json(json.loads(review.model_dump_json()))
        _render_downloads(review)


def _render_summary(review: RequirementReviewPackage) -> None:
    _render_requirement_facts(review.reconciliation.merged_spec)


def _render_requirement_facts(spec) -> None:
    length_value = "未规定"
    if spec.length.minimum_words is not None:
        length_value = f"{spec.length.minimum_words}–{spec.length.maximum_words or '待确认'} 单词"
    elif spec.length.minimum_chars is not None:
        length_value = f"至少 {spec.length.minimum_chars} 字"
    reference_value = (
        f"约 {spec.references.target_total} 篇"
        if spec.references.target_total is not None
        else str(spec.references.minimum_total or "未规定")
    )
    rows = [
        {"字段": "文档类型", "值": str(spec.document_type)},
        {"字段": "学校", "值": spec.institution or "待确认"},
        {"字段": "学院/院系", "值": spec.school_or_department or "待确认"},
        {"字段": "研究主题", "值": spec.topic or "待确认"},
        {
            "字段": "篇幅",
            "值": length_value,
        },
        {
            "字段": "参考文献数量",
            "值": reference_value,
        },
        {
            "字段": "最低外文比例",
            "值": str(spec.references.minimum_foreign_ratio or "未规定"),
        },
        {
            "字段": "章节",
            "值": "、".join(spec.structure.required_or_recommended_sections),
        },
        {
            "字段": "核心问题",
            "值": spec.topic_boundary.central_question or "待确认",
        },
        {
            "字段": "主题边界",
            "值": (
                "纳入："
                + ("、".join(spec.topic_boundary.included_objects) or "待确认")
                + "；排除："
                + ("、".join(spec.topic_boundary.excluded_objects) or "无明确项")
                + "；仅作支撑："
                + ("、".join(spec.topic_boundary.contextual_only_topics) or "无")
            ),
        },
    ]
    st.dataframe(rows, width="stretch", hide_index=True)

    if spec.profiles:
        st.markdown("**可选教师 / 方向**")
        st.dataframe(
            [
                {
                    "选项": profile.profile_id,
                    "教师": profile.teacher,
                    "方向": profile.track or "—",
                    "题目范围": profile.topic or "待确认",
                    "语言": profile.output_language,
                    "文献": (
                        f"约 {profile.references.target_total} 篇"
                        if profile.references.target_total
                        else "未规定"
                    ),
                    "专属硬规则": len(profile.policy_rules),
                }
                for profile in spec.profiles
            ],
            width="stretch",
            hide_index=True,
        )

def _render_comparison(review: RequirementReviewPackage) -> None:
    st.markdown("#### 规则与 LLM 候选结果")
    st.caption("“规范化后一致”表示原始表达不同，但已映射到同一业务含义。")
    st.dataframe(
        comparison_rows(review),
        width="stretch",
        hide_index=True,
        column_config={
            "字段": st.column_config.TextColumn(width="medium"),
            "规则结果": st.column_config.TextColumn(width="large"),
            "LLM结果": st.column_config.TextColumn(width="large"),
            "判断": st.column_config.TextColumn(width="small"),
        },
    )


def _render_issues(review: RequirementReviewPackage) -> None:
    if review.llm_run is not None and review.llm_run.status == "failed":
        st.error("LLM 路径执行失败，本次合并结果仅使用规则解析器。")
        with st.expander("查看 LLM 失败详情", expanded=True):
            st.code(review.llm_run.error or "未返回错误详情")
            st.caption("修正配置或等待接口恢复后，点击“开始分析”即可重试。")

    if review.reconciliation.conflicts:
        st.markdown("#### 实质冲突")
        for conflict in review.reconciliation.conflicts:
            with st.expander(conflict.field, expanded=True):
                left, right = st.columns(2)
                left.markdown("**规则结果**")
                left.code(
                    json.dumps(
                        conflict.rule_value,
                        ensure_ascii=False,
                        indent=2,
                    ),
                    language="json",
                )
                right.markdown("**LLM 结果**")
                right.code(
                    json.dumps(
                        conflict.llm_value,
                        ensure_ascii=False,
                        indent=2,
                    ),
                    language="json",
                )
                evidence = [
                    item.source_text
                    for item in review.reconciliation.merged_spec.source_evidence
                    if item.field == conflict.field or conflict.field.startswith(f"{item.field}.")
                ]
                if evidence:
                    st.caption("原文证据：" + "；".join(evidence))
    else:
        st.success("没有实质冲突。")

    st.markdown("#### 完整性检查")
    for issue in review.completeness.issues:
        icon = "🔴" if issue.severity == "blocking" else "🟠"
        st.markdown(f"{icon} `{issue.issue_id}` — {issue.message}")


def _render_confirmation(review: RequirementReviewPackage) -> None:
    st.markdown("#### 需要你决定的内容")
    st.caption("系统只要求确认会改变后续检索或交付结果的字段。")
    spec = review.reconciliation.merged_spec
    selected_profile_id = None
    selected_profile = None
    if spec.profiles:
        profile_labels = {
            (
                f"{profile.teacher}老师"
                f"{f'（{profile.track}）' if profile.track else ''}"
                f" — {profile.topic or '题目待确认'}"
            ): profile.profile_id
            for profile in spec.profiles
        }
        selected_label = st.selectbox(
            "选择本次采用的教师 / 方向",
            list(profile_labels),
            key="selected_requirement_profile",
        )
        selected_profile_id = profile_labels[selected_label]
        selected_profile = next(
            profile for profile in spec.profiles if profile.profile_id == selected_profile_id
        )

    with st.form("requirement_confirmation"):
        topic = st.text_input(
            "研究主题",
            value=(selected_profile.topic if selected_profile is not None else spec.topic or ""),
            key=f"confirmed_topic_{selected_profile_id or 'single'}",
        )
        boundary = (
            selected_profile.topic_boundary
            if selected_profile is not None
            and selected_profile.topic_boundary.central_question
            else spec.topic_boundary
        )
        st.markdown("**立题卡：检索准入边界**")
        st.caption(
            "这不是额外审批；与课程要求一次确认。它会直接排除“关键词相似但研究对象无关”的文献。"
        )
        central_question = st.text_input(
            "本文要回答的核心问题",
            value=(
                boundary.central_question
                or (f"围绕“{topic or spec.topic or '本主题'}”，本文需要回答什么核心问题？")
            ),
        )
        included_objects = st.text_area(
            "必须纳入的研究对象（每行或分号分隔）",
            value="；".join(boundary.included_objects or ([topic] if topic else [])),
            height=80,
        )
        excluded_objects = st.text_area(
            "默认排除的对象（每行或分号分隔）",
            value="；".join(boundary.excluded_objects),
            height=80,
        )
        contextual_only_topics = st.text_area(
            "只能作为支撑技术、不可主导正文的主题",
            value="；".join(boundary.contextual_only_topics),
            height=70,
        )
        bibliography_style = st.text_input(
            "参考文献格式",
            value=_default_bibliography_style(spec),
            help="课程未明确规定时，系统会按论文语言给出可修改的常用默认值。",
        )
        language_options = {
            "中文": "Chinese",
            "英文": "English",
            "中英双语": "bilingual",
        }
        current_language = next(
            (
                label
                for label, value in language_options.items()
                if value == spec.output_language
            ),
            "中文",
        )
        language_label = st.selectbox(
            "成文语言",
            list(language_options),
            index=list(language_options).index(current_language),
        )
        counting_options = ["按原文：英文单词", "中文字符 + 英文单词"]
        current_counting = {
            "words": "按原文：英文单词",
            "chinese_chars_and_english_words": "中文字符 + 英文单词",
        }.get(
            spec.length.counting_policy,
            "按原文：英文单词"
            if language_options[language_label] == "English"
            else "中文字符 + 英文单词",
        )
        counting_label = st.selectbox(
            "字数统计口径",
            counting_options,
            index=counting_options.index(current_counting),
        )

        controlled_fields = {
            "topic",
            "output_language",
            "references.bibliography_style",
            "length.counting_policy",
        }
        conflict_choices: dict[str, tuple[str, str]] = {}
        remaining_conflicts = [
            conflict
            for conflict in review.reconciliation.conflicts
            if conflict.field not in controlled_fields
        ]
        for index, conflict in enumerate(remaining_conflicts):
            st.markdown(f"**冲突：`{conflict.field}`**")
            rule_column, llm_column = st.columns(2)
            rule_column.caption("规则结果")
            rule_column.code(
                json.dumps(
                    conflict.rule_value,
                    ensure_ascii=False,
                    indent=2,
                ),
                language="json",
            )
            llm_column.caption("LLM 结果")
            llm_column.code(
                json.dumps(
                    conflict.llm_value,
                    ensure_ascii=False,
                    indent=2,
                ),
                language="json",
            )
            choice = st.selectbox(
                "处理方式",
                conflict_resolution_options(conflict),
                key=f"choice_v2_{index}_{conflict.field}",
            )
            custom = st.text_area(
                "自定义 JSON（仅选择“自定义 JSON”时使用）",
                key=f"custom_v2_{index}_{conflict.field}",
                height=80,
            )
            conflict_choices[conflict.field] = (choice, custom)

        warnings = [
            issue.message
            for issue in review.completeness.issues
            if issue.severity == "warning" and issue.requires_user_confirmation
        ]
        if warnings:
            st.caption("点击继续表示接受仍未在课程文件中明确的默认项：")
            for message in warnings:
                st.caption(f"• {message}")
        submitted = st.form_submit_button(
            "确认要求并进入文献检索",
            type="primary",
            width="stretch",
        )

    if submitted:
        try:
            updates = _build_updates(
                review,
                topic,
                central_question,
                included_objects,
                excluded_objects,
                contextual_only_topics,
                bibliography_style,
                language_options[language_label],
                counting_label,
                conflict_choices,
                selected_profile_id=selected_profile_id,
            )
            confirmation = RequirementConfirmation(
                confirmed_by="student",
                field_updates=updates,
                acknowledged_issue_ids=[
                    issue.issue_id
                    for issue in review.completeness.issues
                    if issue.requires_user_confirmation
                ],
            )
            confirmed = RequirementConfirmationService().confirm(
                review,
                confirmation,
            )
        except (RequirementConfirmationError, ValueError, json.JSONDecodeError) as exc:
            st.error(f"确认失败：{exc}")
        else:
            clear_literature_state()
            st.session_state["confirmed_json"] = confirmed.model_dump_json(indent=2)
            st.session_state["mvp_flash"] = "课程要求已确认，正在准备检索方案。"
            st.session_state["mvp_navigation_request"] = "literature"
            st.rerun()


def _build_updates(
    review: RequirementReviewPackage,
    topic: str,
    central_question: str,
    included_objects: str,
    excluded_objects: str,
    contextual_only_topics: str,
    bibliography_style: str,
    output_language: str,
    counting_label: str,
    conflict_choices: dict[str, tuple[str, str]],
    *,
    selected_profile_id: str | None = None,
) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    if selected_profile_id is not None:
        updates["selected_profile_id"] = selected_profile_id
    if topic.strip():
        updates["topic"] = topic.strip()
    if not central_question.strip():
        raise ValueError("立题卡必须说明本文要回答的核心问题。")
    included = _parse_boundary_terms(included_objects)
    if not included:
        raise ValueError("立题卡至少需要一个必须纳入的研究对象。")
    updates["topic_boundary.central_question"] = central_question.strip()
    updates["topic_boundary.included_objects"] = included
    updates["topic_boundary.excluded_objects"] = _parse_boundary_terms(
        excluded_objects
    )
    updates["topic_boundary.contextual_only_topics"] = _parse_boundary_terms(
        contextual_only_topics
    )
    updates["topic_boundary.origin"] = "explicit"
    if bibliography_style.strip():
        updates["references.bibliography_style"] = bibliography_style.strip()
    updates["output_language"] = output_language
    if counting_label == "中文字符 + 英文单词":
        updates["length.counting_policy"] = "chinese_chars_and_english_words"
    elif counting_label == "按原文：英文单词":
        updates["length.counting_policy"] = "words"

    conflicts = {conflict.field: conflict for conflict in review.reconciliation.conflicts}
    for field, (choice, custom) in conflict_choices.items():
        conflict = conflicts[field]
        if choice == "采用规则值":
            updates[field] = conflict.rule_value
        elif choice == "采用 LLM 值":
            updates[field] = conflict.llm_value
        elif choice == "采用两边并集":
            if not isinstance(conflict.rule_value, list) or not isinstance(
                conflict.llm_value,
                list,
            ):
                raise ValueError(f"{field} 不是列表，不能使用并集。")
            updates[field] = list(dict.fromkeys([*conflict.rule_value, *conflict.llm_value]))
        else:
            if not custom.strip():
                raise ValueError(f"{field} 需要填写自定义 JSON。")
            updates[field] = json.loads(custom)
    return updates


def _parse_boundary_terms(value: str) -> list[str]:
    return list(
        dict.fromkeys(
            item.strip()
            for item in re.split(r"[\n；;，,]+", value)
            if item.strip()
        )
    )


def _default_bibliography_style(spec) -> str:
    style = spec.references.bibliography_style
    if style != "pending_confirmation":
        return style
    return "APA 7" if spec.output_language == "English" else "GB/T 7714—2015"


def conflict_resolution_options(
    conflict: RequirementConflict,
) -> list[str]:
    """Expose union only when both conflict values are genuine lists."""

    options = ["采用规则值", "采用 LLM 值"]
    if isinstance(conflict.rule_value, list) and isinstance(
        conflict.llm_value,
        list,
    ):
        options.append("采用两边并集")
    options.append("自定义 JSON")
    return options


def _render_downloads(review: RequirementReviewPackage) -> None:
    st.divider()
    st.subheader("3. 下载结果")
    columns = st.columns(3)
    columns[0].download_button(
        "下载审查 JSON",
        review.model_dump_json(indent=2),
        file_name="requirement_review.json",
        mime="application/json",
        width="stretch",
    )
    columns[1].download_button(
        "下载确认单 Markdown",
        RequirementReviewRenderer().render_markdown(review),
        file_name="requirement_review.md",
        mime="text/markdown",
        width="stretch",
    )
    if "confirmed_json" in st.session_state:
        columns[2].download_button(
            "下载最终需求版本",
            st.session_state["confirmed_json"],
            file_name="confirmed_requirement_spec.json",
            mime="application/json",
            type="primary",
            width="stretch",
        )
    else:
        columns[2].button(
            "完成用户确认后可下载最终版本",
            disabled=True,
            width="stretch",
        )


def _store_result(
    result: WorkbenchResult,
    *,
    mode: str,
    reset_editable_text: bool = True,
) -> None:
    st.session_state["review_json"] = result.review.model_dump_json()
    st.session_state["source_name"] = result.source_name
    st.session_state["source_format"] = result.source_format
    st.session_state["extracted_text"] = result.extracted_text
    st.session_state["extraction_method"] = result.extraction_method
    st.session_state["extraction_warnings"] = result.extraction_warnings
    st.session_state["ocr_average_confidence"] = result.ocr_average_confidence
    st.session_state["elapsed_seconds"] = result.elapsed_seconds
    st.session_state["source_count"] = result.source_count
    st.session_state["analysis_mode"] = mode
    if reset_editable_text:
        st.session_state["editable_extracted_text"] = result.extracted_text


def _restore_result() -> WorkbenchResult:
    return WorkbenchResult(
        review=RequirementReviewPackage.model_validate_json(st.session_state["review_json"]),
        source_name=st.session_state["source_name"],
        source_format=st.session_state["source_format"],
        extracted_text=st.session_state["extracted_text"],
        extraction_method=st.session_state.get(
            "extraction_method",
            "native",
        ),
        extraction_warnings=tuple(st.session_state.get("extraction_warnings", ())),
        ocr_average_confidence=st.session_state.get("ocr_average_confidence"),
        elapsed_seconds=st.session_state["elapsed_seconds"],
        source_count=st.session_state.get("source_count", 1),
    )


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #f7f8f5; }
        [data-testid="stMetric"] {
            background: white;
            border: 1px solid #dfe4dc;
            border-radius: 12px;
            padding: 14px;
        }
        .stButton > button[kind="primary"],
        .stDownloadButton > button[kind="primary"] {
            background: #1f5d50;
            border-color: #1f5d50;
        }
        h1, h2, h3 { color: #173d35; }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    run()
