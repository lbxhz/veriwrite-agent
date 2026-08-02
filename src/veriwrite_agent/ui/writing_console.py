"""Streamlit workbench for V0.4 evidence-constrained section writing."""

from __future__ import annotations

import json

import streamlit as st

from veriwrite_agent.config.settings import LLMSettings
from veriwrite_agent.llm.deepseek_client import DeepSeekClient
from veriwrite_agent.models.writing import V04WritingProject
from veriwrite_agent.models.writing_plan import GroundedWritingPlan
from veriwrite_agent.models.final_delivery import (
    FinalMatterProposal,
    FinalPaperPackage,
)
from veriwrite_agent.models.writing_handoff import V04WritingHandoff
from veriwrite_agent.services.grounded_writing import (
    SectionEvidencePacketBuilder,
    WritingProjectService,
)
from veriwrite_agent.services.writing_planning import (
    GroundedWritingPlanner,
    LLMGroundedParagraphWriter,
    ParagraphWritingRuntimeCache,
    PlannedSectionDraftService,
    WritingPlanRuntimeCache,
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
SECTION_SELECTION_KEY = "v04_selected_section"
SECTION_SELECTION_REQUEST_KEY = "v04_selected_section_request"


def clear_writing_state() -> None:
    st.session_state.pop(WRITING_PLAN_KEY, None)
    st.session_state.pop(V04_PROJECT_KEY, None)
    st.session_state.pop(FINAL_MATTER_KEY, None)
    st.session_state.pop(FINAL_PACKAGE_KEY, None)
    st.session_state.pop(SECTION_SELECTION_KEY, None)
    st.session_state.pop(SECTION_SELECTION_REQUEST_KEY, None)


def render_grounded_writing_console(
    handoff: V04WritingHandoff,
    *,
    include_final_delivery: bool = True,
) -> None:
    """Render the staged V0.4 body-writing workflow."""

    st.divider()
    st.header("V0.4 按证据逐章写作")
    st.caption(
        "DeepSeek只组织段落，不负责创建引用。DOI、引用键、证据卡和PDF页码"
        "由程序绑定；每章确认后才进入正文汇总。"
    )

    legacy_project_json = st.session_state.get(V04_PROJECT_KEY)
    if legacy_project_json:
        legacy_project = V04WritingProject.model_validate_json(legacy_project_json)
        if legacy_project.status == "body_complete":
            _render_body_download(
                legacy_project,
                include_final_delivery=include_final_delivery,
            )
            return

    writing_plan = _render_writing_plan(handoff)
    if writing_plan is None:
        return

    project = _load_or_start_project(handoff)
    confirmed_count = sum(state.status == "confirmed" for state in project.sections)
    columns = st.columns(3)
    columns[0].metric("章节总数", len(project.sections))
    columns[1].metric("已确认章节", confirmed_count)
    columns[2].metric(
        "正文状态",
        "可汇总" if project.status == "body_complete" else "逐章处理中",
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
    generate_label = (
        "生成本章草稿"
        if current_state.draft is None
        else (
            "仅重写阻塞段落"
            if current_state.status == "needs_review"
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
            blocking_paragraphs = {
                issue.paragraph_number
                for issue in (current_state.draft.issues if current_state.draft else [])
                if issue.severity == "blocking" and issue.paragraph_number is not None
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
                    force=(
                        current_state.draft is not None
                        and not blocking_paragraphs
                    ),
                    force_paragraph_numbers=blocking_paragraphs,
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
        _render_draft(project, state.draft)

def render_final_delivery_console(handoff: V04WritingHandoff) -> None:
    """Render final assembly independently from the V0.4 section workbench."""

    st.divider()
    st.header("最终论文组装与交付")
    project = _load_or_start_project(handoff)
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
    metrics = st.columns(4)
    metrics[0].metric("章节", len(plan.sections))
    metrics[1].metric(
        "计划段落",
        sum(len(section.paragraphs) for section in plan.sections),
    )
    metrics[2].metric(
        "证据卡绑定",
        sum(
            len(paragraph.evidence_card_ids)
            for section in plan.sections
            for paragraph in section.paragraphs
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
        st.error("最终交付审计未通过。系统不会把不符合 V0.1 要求的结果伪装成完成品。")
        return

    if package.status == "ready_for_confirmation":
        st.caption("确认会冻结当前预览并解锁 Markdown、DOCX 与完整审计包。")
        actions = st.columns([2, 1])
        if actions[0].button(
            "确认当前论文并解锁下载",
            type="primary",
            width="stretch",
        ):
            package = FinalPaperAssembler().confirm(
                package,
                confirmed_by=project.handoff.requirement.confirmed_by,
            )
            st.session_state[FINAL_PACKAGE_KEY] = package.model_dump_json(indent=2)
            st.rerun()
        if actions[1].button("重新生成组成部分", width="stretch"):
            st.session_state.pop(FINAL_MATTER_KEY, None)
            st.session_state.pop(FINAL_PACKAGE_KEY, None)
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
