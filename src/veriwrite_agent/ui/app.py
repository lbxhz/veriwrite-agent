"""Streamlit entry point for the local V0.1 evaluation workbench."""

from __future__ import annotations

import json
from typing import Any

import streamlit as st

from veriwrite_agent.config.settings import LLMSettings
from veriwrite_agent.models.requirement_workflow import (
    RequirementConfirmation,
    RequirementReviewPackage,
)
from veriwrite_agent.services.requirement_confirmation import (
    RequirementConfirmationError,
    RequirementConfirmationService,
)
from veriwrite_agent.services.requirement_review_renderer import (
    RequirementReviewRenderer,
)
from veriwrite_agent.ui.workbench import (
    WorkbenchResult,
    built_in_samples,
    comparison_rows,
    diagnostic_messages,
    prepare_review_from_path,
    prepare_review_from_upload,
)


def run() -> None:
    st.set_page_config(
        page_title="VeriWrite V0.1 验证工作台",
        page_icon="✓",
        layout="wide",
    )
    _inject_styles()
    st.title("VeriWrite V0.1 验证工作台")
    st.caption("把课程要求变成可比较、可确认、可交接的数据合同")

    source_kind, selected_sample, uploaded_file, mode = _render_input_panel()
    if st.button("开始分析", type="primary", width="stretch"):
        try:
            with st.spinner("正在提取文本并执行双路检查…"):
                if source_kind == "内置样例":
                    result = prepare_review_from_path(
                        selected_sample.path,
                        mode=mode,
                    )
                else:
                    if uploaded_file is None:
                        st.error("请先上传要求文件。")
                        st.stop()
                    result = prepare_review_from_upload(
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        mode=mode,
                    )
        except Exception as exc:
            st.error(f"分析失败：{exc}")
        else:
            _store_result(result)
            st.session_state.pop("confirmed_json", None)

    if "review_json" in st.session_state:
        result = _restore_result()
        _render_result(result)


def _render_input_panel() -> tuple[str, Any, Any, str]:
    st.subheader("1. 选择输入与解析方式")
    left, right = st.columns([1.2, 1])
    samples = built_in_samples()
    with left:
        source_kind = st.radio(
            "要求来源",
            ["内置样例", "上传文件"],
            horizontal=True,
        )
        selected_sample = samples[0]
        uploaded_file = None
        if source_kind == "内置样例":
            labels = [sample.label for sample in samples]
            label = st.selectbox("测试样例", labels)
            selected_sample = next(
                sample for sample in samples if sample.label == label
            )
            st.caption(selected_sample.focus)
        else:
            uploaded_file = st.file_uploader(
                "上传课程要求",
                type=["txt", "md", "docx", "doc", "pdf"],
                help=(
                    "扫描 PDF 需要先 OCR；旧 DOC 自动调用本机 Word 转换。"
                ),
            )
    with right:
        mode_label = st.radio(
            "解析模式",
            ["规则模式", "规则 + DeepSeek 双路模式"],
        )
        mode = "dual" if "DeepSeek" in mode_label else "rule"
        if mode == "dual":
            try:
                summary = LLMSettings().public_summary()
            except Exception as exc:
                st.error(f"DeepSeek 配置不可用：{exc}")
            else:
                st.success(
                    f"已配置模型：{summary['model']} · 将产生一次 API 调用"
                )
        else:
            st.info("不联网、不产生费用；适合建立稳定基线。")
    return source_kind, selected_sample, uploaded_file, mode


def _render_result(result: WorkbenchResult) -> None:
    review = result.review
    st.divider()
    st.subheader("2. 本次运行概览")
    metrics = st.columns(5)
    metrics[0].metric("输入格式", result.source_format.upper().lstrip("."))
    metrics[1].metric("提取字符", len(result.extracted_text))
    metrics[2].metric("冲突", len(review.reconciliation.conflicts))
    metrics[3].metric("阻塞项", review.completeness.blocking_count)
    metrics[4].metric("耗时", f"{result.elapsed_seconds:.1f}s")

    if review.status == "needs_resolution":
        st.warning("当前结果需要用户处理后才能交给 V0.2。")
    else:
        st.success("当前结果可以进入最终确认。")

    tab_summary, tab_compare, tab_issues, tab_confirm, tab_raw = st.tabs(
        ["关键结果", "双路对照", "冲突与问题", "用户确认", "原文与 JSON"]
    )
    with tab_summary:
        _render_summary(review)
    with tab_compare:
        _render_comparison(review)
    with tab_issues:
        _render_issues(review)
    with tab_confirm:
        _render_confirmation(review)
    with tab_raw:
        st.markdown("#### 提取文本预览")
        st.text_area(
            "文本",
            result.extracted_text,
            height=260,
            disabled=True,
            label_visibility="collapsed",
        )
        with st.expander("查看完整审查 JSON"):
            st.json(json.loads(review.model_dump_json()))

    _render_downloads(review)


def _render_summary(review: RequirementReviewPackage) -> None:
    spec = review.reconciliation.merged_spec
    st.markdown("#### 合并后的临时要求")
    rows = [
        {"字段": "文档类型", "值": str(spec.document_type)},
        {"字段": "学校", "值": spec.institution or "待确认"},
        {"字段": "学院/院系", "值": spec.school_or_department or "待确认"},
        {"字段": "研究主题", "值": spec.topic or "待确认"},
        {
            "字段": "最低字数",
            "值": str(spec.length.minimum_chars or "未规定"),
        },
        {
            "字段": "最低参考文献",
            "值": str(spec.references.minimum_total or "未规定"),
        },
        {
            "字段": "最低外文比例",
            "值": str(spec.references.minimum_foreign_ratio or "未规定"),
        },
        {
            "字段": "章节",
            "值": "、".join(
                spec.structure.required_or_recommended_sections
            ),
        },
    ]
    st.dataframe(rows, width="stretch", hide_index=True)

    advantages, problems = diagnostic_messages(review)
    left, right = st.columns(2)
    with left:
        st.markdown("#### 已体现的优势")
        for item in advantages:
            st.markdown(f"- {item}")
    with right:
        st.markdown("#### 暴露的问题")
        if problems:
            for item in problems:
                st.markdown(f"- {item}")
        else:
            st.markdown("- 本次运行没有发现遗留问题。")


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
                    if item.field == conflict.field
                    or conflict.field.startswith(f"{item.field}.")
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
    st.caption(
        "用户可以选择规则值、LLM 值、两边并集，或填写自定义 JSON。"
    )
    with st.form("requirement_confirmation"):
        confirmed_by = st.text_input("确认人", value="student")
        topic = st.text_input(
            "真实研究主题",
            value=review.reconciliation.merged_spec.topic or "",
        )
        bibliography_style = st.text_input(
            "参考文献著录标准（可留空并确认待定）",
            value=(
                ""
                if review.reconciliation.merged_spec.references.bibliography_style
                == "pending_confirmation"
                else review.reconciliation.merged_spec.references.bibliography_style
            ),
        )
        counting_label = st.selectbox(
            "字数统计口径",
            ["待确认", "中文字符 + 英文单词"],
        )

        conflict_choices: dict[str, tuple[str, str]] = {}
        for index, conflict in enumerate(review.reconciliation.conflicts):
            st.markdown(f"**冲突：`{conflict.field}`**")
            choice = st.selectbox(
                "处理方式",
                ["采用规则值", "采用 LLM 值", "采用两边并集", "自定义 JSON"],
                key=f"choice_{index}_{conflict.field}",
            )
            custom = st.text_area(
                "自定义 JSON（仅选择“自定义 JSON”时使用）",
                key=f"custom_{index}_{conflict.field}",
                height=80,
            )
            conflict_choices[conflict.field] = (choice, custom)

        acknowledgements: dict[str, bool] = {}
        for issue in review.completeness.issues:
            if not issue.requires_user_confirmation:
                continue
            if issue.issue_id == "missing_topic" or issue.issue_id.startswith(
                "unresolved_conflict:"
            ):
                continue
            acknowledgements[issue.issue_id] = st.checkbox(
                f"我已知悉：{issue.message}",
                key=f"ack_{issue.issue_id}",
            )

        note = st.text_area("确认说明（可选）")
        submitted = st.form_submit_button(
            "验证并生成最终版本",
            type="primary",
            width="stretch",
        )

    if submitted:
        try:
            updates = _build_updates(
                review,
                topic,
                bibliography_style,
                counting_label,
                conflict_choices,
            )
            confirmation = RequirementConfirmation(
                confirmed_by=confirmed_by,
                field_updates=updates,
                acknowledged_issue_ids=[
                    issue_id
                    for issue_id, checked in acknowledgements.items()
                    if checked
                ],
                note=note or None,
            )
            confirmed = RequirementConfirmationService().confirm(
                review,
                confirmation,
            )
        except (RequirementConfirmationError, ValueError, json.JSONDecodeError) as exc:
            st.error(f"确认失败：{exc}")
        else:
            st.session_state["confirmed_json"] = confirmed.model_dump_json(
                indent=2
            )
            st.success("最终需求版本已通过数据合同和完整性检查。")
            st.json(json.loads(st.session_state["confirmed_json"]))


def _build_updates(
    review: RequirementReviewPackage,
    topic: str,
    bibliography_style: str,
    counting_label: str,
    conflict_choices: dict[str, tuple[str, str]],
) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    if topic.strip():
        updates["topic"] = topic.strip()
    if bibliography_style.strip():
        updates["references.bibliography_style"] = bibliography_style.strip()
    if counting_label == "中文字符 + 英文单词":
        updates["length.counting_policy"] = (
            "chinese_chars_and_english_words"
        )

    conflicts = {
        conflict.field: conflict
        for conflict in review.reconciliation.conflicts
    }
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
            updates[field] = list(
                dict.fromkeys([*conflict.rule_value, *conflict.llm_value])
            )
        else:
            if not custom.strip():
                raise ValueError(f"{field} 需要填写自定义 JSON。")
            updates[field] = json.loads(custom)
    return updates


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


def _store_result(result: WorkbenchResult) -> None:
    st.session_state["review_json"] = result.review.model_dump_json()
    st.session_state["source_name"] = result.source_name
    st.session_state["source_format"] = result.source_format
    st.session_state["extracted_text"] = result.extracted_text
    st.session_state["elapsed_seconds"] = result.elapsed_seconds


def _restore_result() -> WorkbenchResult:
    return WorkbenchResult(
        review=RequirementReviewPackage.model_validate_json(
            st.session_state["review_json"]
        ),
        source_name=st.session_state["source_name"],
        source_format=st.session_state["source_format"],
        extracted_text=st.session_state["extracted_text"],
        elapsed_seconds=st.session_state["elapsed_seconds"],
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
