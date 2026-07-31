"""Streamlit workbench for V0.4 evidence-constrained section writing."""

from __future__ import annotations

import json

import streamlit as st

from veriwrite_agent.config.settings import LLMSettings
from veriwrite_agent.llm.deepseek_client import DeepSeekClient
from veriwrite_agent.models.writing import V04WritingProject
from veriwrite_agent.models.writing_handoff import V04WritingHandoff
from veriwrite_agent.services.grounded_writing import (
    LLMGroundedSectionWriter,
    SectionEvidencePacketBuilder,
    WritingProjectService,
)

V04_PROJECT_KEY = "v04_writing_project_json"


def clear_writing_state() -> None:
    st.session_state.pop(V04_PROJECT_KEY, None)


def render_grounded_writing_console(
    handoff: V04WritingHandoff,
) -> None:
    """Render the staged V0.4 body-writing workflow."""

    st.divider()
    st.header("V0.4 证据约束的逐章写作")
    st.caption(
        "DeepSeek只组织段落，不负责创建引用。DOI、引用键、证据卡和PDF页码"
        "由程序绑定；每章确认后才进入正文汇总。"
    )

    project = _load_or_start_project(handoff)
    confirmed_count = sum(
        state.status == "confirmed" for state in project.sections
    )
    columns = st.columns(4)
    columns[0].metric("章节总数", len(project.sections))
    columns[1].metric("已确认章节", confirmed_count)
    columns[2].metric(
        "确认版证据卡",
        len(handoff.evidence_library.evidence_cards),
    )
    columns[3].metric(
        "正文状态",
        "可汇总" if project.status == "body_complete" else "逐章处理中",
    )

    outline_by_id = {
        section.section_id: section
        for section in handoff.outline.outline.sections
    }
    state_by_id = {state.section_id: state for state in project.sections}
    section_id = st.selectbox(
        "选择要处理的章节",
        options=list(outline_by_id),
        format_func=lambda value: (
            f"{outline_by_id[value].title} · "
            f"{_status_label(state_by_id[value].status)}"
        ),
    )
    packet = SectionEvidencePacketBuilder().build(handoff, section_id)
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

    if st.button(
        "使用DeepSeek生成本章证据约束草稿",
        type="primary",
        width="stretch",
        key=f"v04_generate_{section_id}",
    ):
        try:
            with st.spinner("正在组织段落并执行引用审计……"):
                draft = LLMGroundedSectionWriter(
                    DeepSeekClient(LLMSettings())
                ).draft(packet)
                project = WritingProjectService().save_draft(project, draft)
        except Exception as exc:
            st.error(f"本章生成失败：{exc}")
        else:
            _store_project(project)
            st.rerun()

    state = next(
        item for item in project.sections if item.section_id == section_id
    )
    if state.draft is None:
        st.info("本章尚未生成草稿。")
    else:
        _render_draft(project, state.draft)

    if project.status == "body_complete":
        _render_body_download(project)


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


def _store_project(project: V04WritingProject) -> None:
    st.session_state[V04_PROJECT_KEY] = project.model_dump_json(indent=2)


def _render_packet(packet) -> None:
    st.markdown("#### 本章证据包")
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
    with st.expander("查看研究问题与证据明细"):
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
                    "页码": ", ".join(
                        str(quote.page_number)
                        for quote in item.supporting_quotes
                    ),
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
    st.download_button(
        "下载本章草稿与引用轨迹",
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
        st.success(f"本章已由 {draft.confirmed_by} 确认。")
        return
    confirmed_by = st.text_input(
        "章节确认人",
        value=project.handoff.outline.confirmed_by,
        key=f"v04_confirmer_{draft.section_id}",
    )
    accepted = st.checkbox(
        "我已核对本章论述、引用来源和证据页码。",
        key=f"v04_accept_{draft.section_id}",
    )
    if st.button(
        "确认本章",
        disabled=not accepted or draft.status == "needs_review",
        width="stretch",
        key=f"v04_confirm_{draft.section_id}",
    ):
        try:
            updated = WritingProjectService().confirm_section(
                project,
                draft.section_id,
                confirmed_by=confirmed_by,
            )
        except Exception as exc:
            st.error(f"章节确认失败：{exc}")
        else:
            _store_project(updated)
            st.rerun()


def _render_body_download(project: V04WritingProject) -> None:
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
    st.info(
        "正文完成后才能进入摘要、关键词、结论和最终参考文献审计；"
        "这些内容不会提前生成。"
    )


def _status_label(status: str) -> str:
    return {
        "pending": "待生成",
        "draft": "待确认",
        "needs_review": "存在阻塞",
        "confirmed": "已确认",
    }[status]

