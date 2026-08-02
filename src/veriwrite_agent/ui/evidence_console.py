"""Streamlit controls for human-in-the-loop core paper PDF acquisition."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote

import streamlit as st

from veriwrite_agent.config.settings import LLMSettings
from veriwrite_agent.llm.deepseek_client import DeepSeekClient
from veriwrite_agent.models.evidence import (
    CorePaperExpectation,
    EvidenceLibrary,
    LiteratureLibraryRecord,
    PdfInspectionBatch,
)
from veriwrite_agent.models.literature_selection import (
    BalancedLiteratureSelection,
)
from veriwrite_agent.models.literature_verification import (
    LiteratureVerificationBatch,
)
from veriwrite_agent.models.requirement_workflow import ConfirmedRequirementSpec
from veriwrite_agent.models.writing_handoff import V04WritingHandoff
from veriwrite_agent.services.evidence_card_extraction import (
    LLMEvidenceCardExtractor,
)
from veriwrite_agent.services.evidence_library import (
    EvidenceLibraryBuilder,
    EvidenceLibraryConfirmationService,
)
from veriwrite_agent.services.evidence_runtime import (
    EvidencePageRetriever,
    EvidenceRuntimeCache,
)
from veriwrite_agent.services.pdf_acquisition import PdfAcquisitionInspector
from veriwrite_agent.services.pdf_text_extraction import PdfPageExtractor
from veriwrite_agent.services.writing_handoff import (
    WritingHandoffService,
    WritingOutlineBuilder,
)
from veriwrite_agent.services.requirement_policy import RequirementPolicyCompiler
from veriwrite_agent.ui.workbench import project_root
from veriwrite_agent.ui.writing_console import (
    render_grounded_writing_console,
)

PDF_STATE_KEYS = (
    "v03_core_dois",
    "v03_download_directory",
    "v03_pdf_inspection_json",
    "v03_evidence_library_json",
    "v03_writing_handoff_json",
    "v04_writing_project_json",
)


def render_pdf_acquisition_console(
    selection: BalancedLiteratureSelection,
    *,
    include_writing: bool = True,
) -> None:
    """Let users download only core PDFs, then inspect them as a batch."""

    st.divider()
    st.header("V0.3 获取核心全文并建立证据")
    st.caption(
        "Agent 负责确定下载队列、打开权威入口和检查 PDF；用户只处理出版社登录、验证码与下载按钮。"
    )
    if not selection.selected:
        st.info("V0.2 尚无入选论文，不能建立核心论文下载队列。")
        return

    completed_handoff_json = st.session_state.get("v03_writing_handoff_json")
    if completed_handoff_json:
        handoff = V04WritingHandoff.model_validate_json(completed_handoff_json)
        library = handoff.evidence_library
        st.success("核心全文、证据卡和写作章节已准备完成。")
        metrics = st.columns(4)
        metrics[0].metric("核心全文", sum(r.evidence_tier == "A_core" for r in library.records))
        metrics[1].metric("完整页", len(library.pages))
        metrics[2].metric("证据卡", len(library.evidence_cards))
        metrics[3].metric("写作章节", len(handoff.outline.outline.sections))
        if not include_writing:
            if st.button("进入 V0.4 逐章写作", type="primary", width="stretch"):
                st.session_state["mvp_navigation_request"] = "writing"
                st.rerun()
        with st.expander("查看证据详情与交接包"):
            st.caption(
                f"{len(library.records)} 篇文献 · {len(library.pages)} 个完整文本页 · "
                f"{len(library.evidence_cards)} 张证据卡"
            )
            st.dataframe(
                [
                    {
                        "主题": card.theme_id,
                        "类型": card.evidence_type,
                        "结论": card.normalized_claim,
                        "页码": "、".join(
                            str(item.page_number) for item in card.supporting_quotes
                        ),
                        "DOI": card.doi,
                    }
                    for card in library.evidence_cards
                ],
                width="stretch",
                hide_index=True,
            )
            st.download_button(
                "下载 V0.4 写作交接包",
                completed_handoff_json,
                file_name="v04_writing_handoff.json",
                mime="application/json",
                width="stretch",
            )
        if include_writing:
            render_grounded_writing_console(handoff)
        return

    records = {item.doi: item for item in selection.selected}
    default_limit = 1 if st.session_state.get("mvp_smoke_test") else 8
    default_dois = [
        item.doi for item in selection.selected[: min(default_limit, len(records))]
    ]
    previous_dois = [
        doi for doi in st.session_state.get("v03_core_dois", default_dois) if doi in records
    ]
    st.write(
        f"系统已按相关性和主题覆盖选择前 {len(previous_dois)} 篇作为核心论文；"
        "其余文献保留已验证元数据。"
    )
    with st.expander("调整核心论文"):
        selected_dois = st.multiselect(
            "需要全文核验的论文",
            options=list(records),
            default=previous_dois,
            format_func=lambda doi: f"{records[doi].title} · {doi}",
            help=(
                "快速联通测试默认取 1 篇，真实项目默认取排序靠前的 8 篇；"
                "只有确实需要改变核心范围时才调整。"
            ),
        )
    if selected_dois != previous_dois:
        st.session_state.pop("v03_pdf_inspection_json", None)
        st.session_state.pop("v03_evidence_library_json", None)
        st.session_state.pop("v03_writing_handoff_json", None)
    st.session_state["v03_core_dois"] = selected_dois

    if not selected_dois:
        st.warning("请至少选择一篇核心论文。")
        return

    with st.expander("核心论文下载队列", expanded=True):
        for index, doi in enumerate(selected_dois, 1):
            record = records[doi]
            left, right = st.columns([5, 1])
            left.markdown(f"**{index}. {record.title}**  \n`{doi}` · 主题 `{record.theme_id}`")
            right.link_button(
                "打开 DOI 官网",
                f"https://doi.org/{quote(doi, safe='/')}",
                width="stretch",
            )

    download_directory = st.text_input(
        "浏览器下载目录",
        value=st.session_state.get(
            "v03_download_directory",
            str(Path.home() / "Downloads"),
        ),
        help="点击出版社的下载按钮后，文件通常会保存到这里。",
    )
    st.session_state["v03_download_directory"] = download_directory
    st.info("下载完成后点击扫描；系统会自动匹配文件并检查身份、完整性和可提取性。")

    if st.button(
        "我已下载，扫描核心 PDF",
        type="primary",
        width="stretch",
    ):
        expectations = [
            CorePaperExpectation(
                doi=doi,
                title=records[doi].title,
                source_url=f"https://doi.org/{quote(doi, safe='/')}",
                theme_id=records[doi].theme_id,
            )
            for doi in selected_dois
        ]
        with st.spinner("正在匹配并检查 PDF……"):
            batch = PdfAcquisitionInspector().scan_download_directory(
                expectations,
                download_directory,
            )
        st.session_state["v03_pdf_inspection_json"] = batch.model_dump_json(indent=2)
        st.session_state.pop("v03_evidence_library_json", None)
        st.session_state.pop("v03_writing_handoff_json", None)
        st.rerun()

    serialized = st.session_state.get("v03_pdf_inspection_json")
    if not serialized:
        return
    batch = PdfInspectionBatch.model_validate_json(serialized)
    _render_pdf_results(batch)
    _render_evidence_pipeline(selection, batch, include_writing=include_writing)


def clear_pdf_acquisition_state() -> None:
    for key in PDF_STATE_KEYS:
        st.session_state.pop(key, None)


def _render_pdf_results(batch: PdfInspectionBatch) -> None:
    st.subheader("PDF 检查结果")
    counts = {
        status: sum(report.status == status for report in batch.reports)
        for status in ("verified", "needs_review", "invalid", "missing")
    }
    columns = st.columns(5)
    columns[0].metric("扫描 PDF", batch.inspected_file_count)
    columns[1].metric("全文已验证", counts["verified"])
    columns[2].metric("需要复核/OCR", counts["needs_review"])
    columns[3].metric("无效文件", counts["invalid"])
    columns[4].metric("仍缺失", counts["missing"])

    labels = {
        "verified": "已验证",
        "needs_review": "需要复核/OCR",
        "invalid": "无效",
        "missing": "等待用户下载",
    }
    st.dataframe(
        [
            {
                "状态": labels[report.status],
                "题名": report.expectation.title,
                "DOI": report.expectation.doi,
                "页数": report.page_count or "—",
                "可提取页": report.extractable_page_count,
                "身份得分": report.identity_score,
                "匹配依据": "、".join(report.identity_basis) or "—",
                "问题": "；".join(issue.detail for issue in report.issues) or "—",
                "本地文件": report.local_path or "—",
            }
            for report in batch.reports
        ],
        width="stretch",
        hide_index=True,
    )
    if counts["missing"]:
        st.warning("缺失项会保留在队列中；下载后重新扫描即可继续，不需要重跑 V0.1/V0.2。")
    if batch.unmatched_files:
        with st.expander("未能匹配到核心论文的其他 PDF"):
            st.code("\n".join(batch.unmatched_files))
    with st.expander("导出 PDF 检查报告"):
        st.download_button(
            "下载检查报告 JSON",
            json.dumps(
                json.loads(batch.model_dump_json()),
                ensure_ascii=False,
                indent=2,
            ),
            file_name="core_paper_pdf_inspection.json",
            mime="application/json",
            width="stretch",
        )


def _render_evidence_pipeline(
    selection: BalancedLiteratureSelection,
    batch: PdfInspectionBatch,
    *,
    include_writing: bool = True,
) -> None:
    st.subheader("全文证据与写作准备")
    verified_count = sum(report.status == "verified" for report in batch.reports)
    if verified_count == 0:
        st.warning("当前没有通过身份与完整性检查的PDF，不能提取全文证据。")
        return

    st.caption(
        "代码固定 DOI、文件哈希、页码和原文；DeepSeek只对候选页面做语义归类。"
    )
    if "v03_evidence_library_json" not in st.session_state:
        if st.button(
            "提取证据并准备写作",
            type="primary",
            width="stretch",
        ):
            try:
                with st.spinner("正在分页提取PDF、生成证据并检查章节覆盖……"):
                    library = _build_evidence_library(selection, batch)
            except Exception as exc:
                st.error(f"证据提取中断：{exc}")
            else:
                st.session_state["v03_evidence_library_json"] = library.model_dump_json(indent=2)
                st.session_state.pop("v03_writing_handoff_json", None)
                st.rerun()

    serialized = st.session_state.get("v03_evidence_library_json")
    if not serialized:
        return
    library = EvidenceLibrary.model_validate_json(serialized)
    _render_library_summary(library)

    confirmed_requirement = ConfirmedRequirementSpec.model_validate_json(
        st.session_state["confirmed_json"]
    )
    policy = selection.blueprint.requirement_policy or RequirementPolicyCompiler().compile(
        confirmed_requirement
    )
    outline = WritingOutlineBuilder().build(
        selection.blueprint,
        library,
        policy=policy,
        smoke_test=bool(st.session_state.get("mvp_smoke_test")),
    )
    if st.session_state.get("mvp_smoke_test"):
        st.info(
            "快速联通测试会把临时检索主题合并为一个最小正文单元，"
            "仅验证一篇核心PDF能否完成证据提取、V0.4写作与最终交付；"
            "真实项目仍执行逐章节全文证据门禁。"
        )
    st.markdown("#### 写作章节与证据覆盖")
    st.dataframe(
        [
            {
                "章节": section.title,
                "目标字数": section.target_words,
                "核心全文": len(section.core_dois),
                "辅助文献": len(section.supporting_dois),
                "证据卡": len(section.evidence_card_ids),
                "证据缺口": section.evidence_gap,
            }
            for section in outline.sections
        ],
        width="stretch",
        hide_index=True,
    )
    blockers = [*library.unresolved_issues, *outline.unresolved_gaps]
    if blockers:
        extraction_failures = [
            issue
            for issue in library.unresolved_issues
            if issue.startswith("evidence_extraction_failed:")
        ]
        st.error(
            f"当前还有 {len(blockers)} 个问题，暂时不能进入 V0.4。"
            "系统会保留已经成功的全文和证据卡。"
        )
        if extraction_failures:
            st.warning(
                f"其中 {len(extraction_failures)} 篇论文的证据归类输出未通过格式检查。"
                "这通常不是 PDF 问题，可以直接重试失败文献。"
            )
            if st.button(
                "重新提取失败文献",
                type="primary",
                width="stretch",
                key="retry_failed_evidence_extraction",
            ):
                try:
                    with st.spinner("正在复用成功缓存并重新提取失败文献……"):
                        retried_library = _build_evidence_library(selection, batch)
                except Exception as exc:
                    st.error(f"重新提取中断：{exc}")
                else:
                    st.session_state["v03_evidence_library_json"] = (
                        retried_library.model_dump_json(indent=2)
                    )
                    st.session_state.pop("v03_writing_handoff_json", None)
                    st.rerun()
        with st.expander("查看问题详情"):
            for blocker in blockers:
                st.write(f"- {_friendly_evidence_blocker(blocker)}")
        with st.expander("技术详情"):
            st.code("\n".join(blockers))
        return

    handoff_json = st.session_state.get("v03_writing_handoff_json")
    if not handoff_json:
        if st.button(
            "采用这些证据并进入写作",
            type="primary",
            width="stretch",
        ):
            confirmed_library = EvidenceLibraryConfirmationService().confirm(
                library,
                confirmed_by=confirmed_requirement.confirmed_by,
            )
            handoff_service = WritingHandoffService()
            confirmed_outline = handoff_service.confirm_outline(
                outline,
                confirmed_by=confirmed_requirement.confirmed_by,
            )
            handoff = handoff_service.create(
                requirement=confirmed_requirement,
                outline=confirmed_outline,
                evidence_library=confirmed_library,
                policy=policy,
            )
            st.session_state["v03_evidence_library_json"] = (
                confirmed_library.model_dump_json(indent=2)
            )
            st.session_state["v03_writing_handoff_json"] = handoff.model_dump_json(indent=2)
            st.session_state["mvp_flash"] = "证据与写作章节已锁定，可以开始正文。"
            st.session_state["mvp_navigation_request"] = "writing"
            st.rerun()
        with st.expander("导出证据库草案"):
            st.download_button(
                "下载证据库 JSON",
                serialized,
                file_name="v03_evidence_library_draft.json",
                mime="application/json",
                width="stretch",
            )
        return

    handoff = V04WritingHandoff.model_validate_json(handoff_json)
    st.success(
        f"证据已就绪：{len(handoff.evidence_library.records)}篇文献、"
        f"{len(handoff.evidence_library.evidence_cards)}张证据卡。"
    )
    if not include_writing:
        if st.button("进入 V0.4 逐章写作", type="primary", width="stretch"):
            st.session_state["mvp_navigation_request"] = "writing"
            st.rerun()
    with st.expander("导出写作交接包"):
        st.download_button(
            "下载 V0.4 写作交接包",
            handoff_json,
            file_name="v04_writing_handoff.json",
            mime="application/json",
            width="stretch",
        )
    if include_writing:
        render_grounded_writing_console(handoff)


def _build_evidence_library(
    selection: BalancedLiteratureSelection,
    batch: PdfInspectionBatch,
) -> EvidenceLibrary:
    confirmed_requirement = ConfirmedRequirementSpec.model_validate_json(
        st.session_state["confirmed_json"]
    )
    policy = selection.blueprint.requirement_policy or RequirementPolicyCompiler().compile(
        confirmed_requirement
    )
    cache = EvidenceRuntimeCache(
        project_root() / "runtime" / "evidence_console",
        policy_fingerprint=policy.requirement_fingerprint,
    )
    inspector = PdfAcquisitionInspector()
    documents = inspector.to_document_acquisitions(batch)
    available = {document.doi: document for document in documents if document.status == "available"}
    core_dois = {report.expectation.doi for report in batch.reports}
    verifications = LiteratureVerificationBatch.model_validate_json(
        st.session_state["literature_verification_json"]
    )
    verification_by_doi = {
        result.candidate.doi: result for result in verifications.verified_records
    }

    def authority_fields(doi: str) -> tuple[list[str], str | None, str | None, str]:
        verification = verification_by_doi.get(doi)
        if verification is None:
            return [], None, None, f"https://doi.org/{quote(doi, safe='/')}"
        metadata = verification.authority.metadata if verification.authority is not None else None
        return (
            metadata.authors if metadata is not None else verification.candidate.authors,
            (
                metadata.journal_title
                if metadata is not None
                else verification.candidate.journal_title
            ),
            verification.candidate.abstract,
            (
                verification.resolution.final_url
                if verification.resolution is not None and verification.resolution.final_url
                else f"https://doi.org/{quote(doi, safe='/')}"
            ),
        )

    authority_by_doi = {item.doi: authority_fields(item.doi) for item in selection.selected}
    records = [
        LiteratureLibraryRecord(
            doi=item.doi,
            title=item.title,
            authors=item.authors or authority_by_doi[item.doi][0],
            year=item.year,
            journal=item.journal or authority_by_doi[item.doi][1],
            publisher=item.publisher,
            language=item.language,
            source_type=item.source_type,
            is_foreign=item.is_foreign,
            abstract=authority_by_doi[item.doi][2],
            source_url=authority_by_doi[item.doi][3],
            theme_ids=[item.theme_id],
            evidence_tier=(
                "A_core"
                if item.doi in available
                else ("B_supporting" if item.doi in core_dois else "C_background")
            ),
            evidence_status=(
                "full_text_verified" if item.doi in available else "metadata_verified"
            ),
            permitted_use=(
                "detailed_claims"
                if item.doi in available
                else ("section_support" if item.doi in core_dois else "background_only")
            ),
        )
        for item in selection.selected
    ]
    unresolved = [
        f"core_pdf_{report.status}:{report.expectation.doi}"
        for report in batch.reports
        if report.status != "verified"
    ]

    pages = []
    cards = []
    extractions = []
    page_selections = []
    card_extractor = LLMEvidenceCardExtractor(DeepSeekClient(LLMSettings().for_structured_output()))
    themes = {theme.theme_id: theme for theme in selection.blueprint.themes}
    record_by_doi = {record.doi: record for record in records}
    for document in available.values():
        extraction = cache.load_extraction(document)
        if extraction is None:
            extraction = PdfPageExtractor(enable_ocr=True).extract(document)
            cache.save_extraction(extraction)
        extractions.append(extraction)
        pages.extend(extraction.pages)
        if extraction.status != "complete":
            unresolved.append(f"pdf_extraction_{extraction.status}:{document.doi}")
        if not extraction.pages:
            continue
        record = record_by_doi[document.doi]
        theme_id = record.theme_ids[0]
        theme = themes[theme_id]
        query_text = " ".join(
            [
                record.title,
                theme.section_title,
                theme.section_purpose,
                *theme.research_questions,
                *theme.primary_keywords,
                *theme.related_keywords,
            ]
        )
        selection_audit, selected_pages = EvidencePageRetriever().select(
            doi=document.doi,
            theme_id=theme_id,
            query_text=query_text,
            pages=extraction.pages,
        )
        page_selections.append(selection_audit)
        try:
            document_cards = cache.load_cards(
                document,
                title=record.title,
                selection=selection_audit,
            )
            if document_cards is None:
                document_cards = card_extractor.extract(
                    doi=document.doi,
                    title=record.title,
                    theme_id=theme_id,
                    section_purpose=theme.section_purpose,
                    pages=selected_pages,
                )
                cache.save_cards(
                    document,
                    title=record.title,
                    selection=selection_audit,
                    cards=document_cards,
                )
            cards.extend(document_cards)
        except Exception as exc:
            unresolved.append(f"evidence_extraction_failed:{document.doi}:{exc}")

    return EvidenceLibraryBuilder().build(
        records=records,
        documents=documents,
        extractions=extractions,
        page_selections=page_selections,
        pages=pages,
        evidence_cards=cards,
        unresolved_issues=unresolved,
        requirement_policy_fingerprint=policy.requirement_fingerprint,
    )


def _render_library_summary(library: EvidenceLibrary) -> None:
    metrics = st.columns(4)
    metrics[0].metric("文献总数", len(library.records))
    metrics[1].metric(
        "核心全文",
        sum(record.evidence_tier == "A_core" for record in library.records),
    )
    metrics[2].metric("证据卡", len(library.evidence_cards))
    metrics[3].metric("提取问题", len(library.unresolved_issues))
    if library.extractions:
        with st.expander("查看 PDF 全文提取与检索选页审计"):
            st.caption(
                f"完整提取 {sum(item.status == 'complete' for item in library.extractions)} 篇 · "
                f"完整文本 {len(library.pages)} 页 · "
                "送入 LLM "
                f"{sum(len(item.selected_page_numbers) for item in library.page_selections)} 页"
            )
            st.dataframe(
                [
                    {
                        "DOI": extraction.doi,
                        "提取状态": extraction.status,
                        "PDF 页数": extraction.page_count,
                        "已提取文本页": len(extraction.pages),
                        "问题数": len(extraction.issues),
                        "送入 LLM 页": next(
                            (
                                ", ".join(map(str, selection.selected_page_numbers))
                                for selection in library.page_selections
                                if selection.doi == extraction.doi
                            ),
                            "—",
                        ),
                    }
                    for extraction in library.extractions
                ],
                hide_index=True,
                width="stretch",
            )
    if library.evidence_cards:
        with st.expander("查看全部证据卡"):
            st.dataframe(
                [
                    {
                        "主题": card.theme_id,
                        "类型": card.evidence_type,
                        "结论": card.normalized_claim,
                        "页码": "、".join(
                            str(item.page_number) for item in card.supporting_quotes
                        ),
                        "原文": " / ".join(
                            item.exact_text for item in card.supporting_quotes
                        ),
                        "DOI": card.doi,
                    }
                    for card in library.evidence_cards
                ],
                width="stretch",
                hide_index=True,
            )


def _friendly_evidence_blocker(issue: str) -> str:
    if issue.startswith("evidence_extraction_failed:"):
        _, doi, _ = issue.split(":", 2)
        return f"{doi}：模型返回的证据选择格式不合规，可自动重试。"
    if issue.startswith("core_pdf_missing:"):
        return f"{issue.rsplit(':', 1)[-1]}：尚未找到核心 PDF。"
    if issue.startswith("core_pdf_needs_review:"):
        return f"{issue.rsplit(':', 1)[-1]}：PDF 身份或文本需要复核。"
    if issue.startswith("pdf_extraction_"):
        return f"PDF 全文提取未完成：{issue.split(':', 1)[-1]}。"
    if issue.endswith("缺少已验证全文或可追溯证据卡。"):
        return issue
    return issue


def _target_words(confirmed: ConfirmedRequirementSpec) -> int:
    length = confirmed.requirement.length
    return length.target_words or length.maximum_words or length.minimum_words or 4000
