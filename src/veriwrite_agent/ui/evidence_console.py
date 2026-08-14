"""Streamlit controls for human-in-the-loop core paper PDF acquisition."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import quote

import streamlit as st

from veriwrite_agent.models.evidence import (
    CorePaperExpectation,
    EvidenceLibrary,
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
from veriwrite_agent.services.evidence_assembly import build_evidence_library
from veriwrite_agent.services.evidence_library import (
    EvidenceLibraryConfirmationService,
)
from veriwrite_agent.services.pdf_acquisition import PdfAcquisitionInspector
from veriwrite_agent.services.writing_handoff import (
    WritingHandoffService,
    WritingOutlineBuilder,
)
from veriwrite_agent.services.requirement_policy import RequirementPolicyCompiler
from veriwrite_agent.services.writing_evidence_recovery import (
    ParagraphEvidenceGap,
    WritingEvidenceRecoveryRequest,
)
from veriwrite_agent.ui.workbench import project_root
from veriwrite_agent.ui.writing_console import (
    EVIDENCE_RECOVERY_CHECKPOINT_KEY,
    EVIDENCE_RECOVERY_REQUEST_KEY,
    _create_final_repair_checkpoint,
    clear_writing_state,
    queue_evidence_recovery_resume,
    render_grounded_writing_console,
    route_evidence_recovery_to_search,
)

PDF_STATE_KEYS = (
    "v03_core_dois",
    "v03_download_directory",
    "v03_pdf_inspection_json",
    "v03_evidence_library_json",
    "v03_writing_handoff_json",
    "v04_writing_plan_json",
    "v04_writing_project_json",
)

PDF_DIRECTORY_KEY = "v03_download_directory"
EVIDENCE_VAULT_ENV = "VERIWRITE_EVIDENCE_VAULT"
DEFAULT_EVIDENCE_VAULT = Path.home() / "Documents" / "VeriWrite" / "Evidence-Vault"


def project_pdf_directory(state: Mapping[str, object]) -> str:
    """Return the dedicated PDF directory, migrating only the legacy default."""

    configured = os.getenv(EVIDENCE_VAULT_ENV, str(DEFAULT_EVIDENCE_VAULT)).strip()
    configured_path = configured or str(DEFAULT_EVIDENCE_VAULT)
    saved = str(state.get(PDF_DIRECTORY_KEY, "") or "").strip()
    if not saved or _same_local_path(saved, Path.home() / "Downloads"):
        return configured_path
    return saved


def _same_local_path(left: str | Path, right: str | Path) -> bool:
    try:
        left_path = Path(left).expanduser().resolve(strict=False)
        right_path = Path(right).expanduser().resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        return False
    return os.path.normcase(str(left_path)) == os.path.normcase(str(right_path))


def render_pdf_acquisition_console(
    selection: BalancedLiteratureSelection,
    *,
    include_writing: bool = True,
    agent_embedded: bool = False,
) -> None:
    """Let users download only core PDFs, then inspect them as a batch."""

    if not agent_embedded:
        st.divider()
        st.header("V0.3 获取核心全文并建立证据")
        st.caption(
            "Agent 负责确定下载队列、打开权威入口和检查 PDF；"
            "用户只处理出版社登录、验证码与下载按钮。"
        )
    if not selection.selected:
        st.info("V0.2 尚无入选论文，不能建立核心论文下载队列。")
        return

    legacy_inspection_raw = st.session_state.get("v03_pdf_inspection_json")
    if legacy_inspection_raw:
        try:
            legacy_inspection_version = json.loads(legacy_inspection_raw).get(
                "schema_version"
            )
        except (TypeError, json.JSONDecodeError):
            legacy_inspection_version = None
        if legacy_inspection_version != "0.3.2":
            _create_final_repair_checkpoint(
                ["pdf_identity_rule_upgrade"],
            )
            st.session_state.pop("v03_evidence_library_json", None)
            st.session_state.pop("v03_writing_handoff_json", None)
            clear_writing_state(preserve_repair_checkpoint=True)

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
    recovery = _load_evidence_recovery_request(include_blocked=True)
    if recovery is not None:
        unavailable_dois = set(recovery.unavailable_full_text_dois)
        recovery_candidates = [
            item.doi
            for item in selection.selected
            if item.suitable_section_id in recovery.affected_section_ids
            and item.doi not in unavailable_dois
        ]
        default_dois = list(
            dict.fromkeys(
                [
                    *(
                        doi
                        for doi in _recovery_existing_core_dois()
                        if doi not in unavailable_dois
                    ),
                    *(
                        doi
                        for doi in recovery.requested_core_dois
                        if doi not in unavailable_dois
                    ),
                    *recovery_candidates[:4],
                ]
            )
        )
    saved_dois = list(st.session_state.get("v03_core_dois", []))
    previous_dois = [
        doi
        for doi in (
            list(dict.fromkeys([*saved_dois, *default_dois]))
            if recovery is not None
            else (saved_dois or default_dois)
        )
        if doi in records
    ]
    if agent_embedded:
        selected_dois = previous_dois
        st.caption(
            f"正在自动核验 {len(selected_dois)} 篇候选全文；"
            "不可获取来源会先由系统批量寻找替代文献。"
        )
    else:
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
        clear_writing_state()
    st.session_state["v03_core_dois"] = selected_dois

    if not selected_dois:
        st.warning("请至少选择一篇核心论文。")
        return

    # Internal recovery favors the newest downloads and must return control to
    # the V0.4 status surface quickly.  The standalone V0.3 inspector keeps the
    # broader diagnostic window for users who explicitly open that stage.
    pdf_inspector = PdfAcquisitionInspector(max_files=30 if agent_embedded else 100)

    recovery_blocked = bool(
        (recovery := _load_evidence_recovery_request(include_blocked=True))
        and recovery.status == "blocked"
    )
    with st.expander(
        "合并后的核心论文下载清单" if recovery_blocked else "核心论文下载队列",
        expanded=(not agent_embedded or recovery_blocked),
    ):
        for index, doi in enumerate(selected_dois, 1):
            record = records[doi]
            left, right = st.columns([5, 1])
            left.markdown(f"**{index}. {record.title}**  \n`{doi}` · 主题 `{record.theme_id}`")
            right.link_button(
                "打开 DOI 官网",
                f"https://doi.org/{quote(doi, safe='/')}",
                width="stretch",
            )

    st.session_state[PDF_DIRECTORY_KEY] = project_pdf_directory(st.session_state)
    download_directory = st.text_input(
        "当前项目 PDF 专属目录",
        key=PDF_DIRECTORY_KEY,
        help=(
            "V0.3 只扫描这个目录的第一层文件，不再扫描整个浏览器下载目录。"
            "请将本项目需要核验的 PDF 保存到这里。"
        ),
    )
    if not agent_embedded or recovery_blocked:
        st.info(
            "系统只会在当前项目 PDF 专属目录中按下载时间从新到旧检查文件；"
            "找到全部目标的完整 PDF 后立即停止，"
            "并核验 DOI、题名、完整性和文本可提取性。"
        )

    manual_scan = False
    if not agent_embedded or recovery_blocked:
        manual_scan = st.button(
            (
                "我已完成合并清单，重新扫描"
                if recovery_blocked
                else "我已下载，扫描核心 PDF"
            ),
            type="primary",
            width="stretch",
        )
    auto_scan_recovery = (
        recovery is not None
        and recovery.status != "blocked"
        and "v03_pdf_inspection_json" not in st.session_state
    )
    if manual_scan or auto_scan_recovery:
        expectations = [
            CorePaperExpectation(
                doi=doi,
                title=records[doi].title,
                source_url=f"https://doi.org/{quote(doi, safe='/')}",
                theme_id=records[doi].theme_id,
            )
            for doi in selected_dois
        ]
        spinner_text = (
            "正在自动检查已有 PDF；只把确实缺失的下载任务留给你……"
            if auto_scan_recovery
            else "正在匹配并检查 PDF……"
        )
        with st.spinner(spinner_text):
            batch = pdf_inspector.scan_download_directory(
                expectations,
                download_directory,
            )
        st.session_state["v03_pdf_inspection_json"] = batch.model_dump_json(indent=2)
        st.session_state.pop("v03_evidence_library_json", None)
        st.session_state.pop("v03_writing_handoff_json", None)
        clear_writing_state()
        st.rerun()

    serialized = st.session_state.get("v03_pdf_inspection_json")
    if not serialized:
        return
    batch = PdfInspectionBatch.model_validate_json(serialized)
    inspected_dois = {report.expectation.doi for report in batch.reports}
    if inspected_dois != set(selected_dois):
        expectations = [
            CorePaperExpectation(
                doi=doi,
                title=records[doi].title,
                source_url=f"https://doi.org/{quote(doi, safe='/')}",
                theme_id=records[doi].theme_id,
            )
            for doi in selected_dois
        ]
        with st.spinner("核心队列已更新，正在自动复核新增全文……"):
            batch = pdf_inspector.scan_download_directory(
                expectations,
                download_directory,
            )
        st.session_state["v03_pdf_inspection_json"] = batch.model_dump_json(indent=2)
        st.session_state.pop("v03_evidence_library_json", None)
        st.session_state.pop("v03_writing_handoff_json", None)
        clear_writing_state()
        st.rerun()
    if batch.schema_version != "0.3.2":
        expectations = [
            CorePaperExpectation(
                doi=doi,
                title=records[doi].title,
                source_url=f"https://doi.org/{quote(doi, safe='/')}",
                theme_id=records[doi].theme_id,
            )
            for doi in selected_dois
        ]
        with st.spinner("正在用新版首页身份规则复核旧 PDF 缓存……"):
            batch = pdf_inspector.scan_download_directory(
                expectations,
                download_directory,
            )
        st.session_state["v03_pdf_inspection_json"] = batch.model_dump_json(indent=2)
        st.session_state.pop("v03_evidence_library_json", None)
        st.session_state.pop("v03_writing_handoff_json", None)
        clear_writing_state()
        st.session_state["mvp_flash"] = (
            "旧版 PDF 身份缓存已按首页题名/DOI 规则重新核验；"
            "正文通用术语不再能把无关论文误判为目标全文。"
        )
        st.rerun()
    _render_pdf_results(batch)
    _render_evidence_pipeline(selection, batch, include_writing=include_writing)


def clear_pdf_acquisition_state() -> None:
    for key in PDF_STATE_KEYS:
        st.session_state.pop(key, None)


def _load_evidence_recovery_request(
    *,
    include_blocked: bool = False,
) -> WritingEvidenceRecoveryRequest | None:
    raw = st.session_state.get(EVIDENCE_RECOVERY_REQUEST_KEY)
    if not raw:
        return None
    try:
        request = WritingEvidenceRecoveryRequest.model_validate_json(raw)
    except (TypeError, ValueError):
        return None
    if request.status == "resolved":
        return None
    if request.status == "blocked" and not include_blocked:
        return None
    return request


def _recovery_existing_core_dois() -> list[str]:
    raw = st.session_state.get(EVIDENCE_RECOVERY_CHECKPOINT_KEY)
    if not raw:
        return []
    try:
        checkpoint = json.loads(raw)
        handoff = V04WritingHandoff.model_validate_json(checkpoint["handoff_json"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return []
    return [
        record.doi
        for record in handoff.evidence_library.records
        if record.evidence_tier == "A_core"
    ]


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
        recovery = _load_evidence_recovery_request()
        if recovery is not None and st.button(
            "无法获取这些全文，自动补搜替代论文",
            type="primary",
            width="stretch",
            key="v03_recovery_search_replacements",
        ):
            route_evidence_recovery_to_search(recovery)
            st.rerun()
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
    recovery_request = _load_evidence_recovery_request()
    missing_dois = [
        report.expectation.doi
        for report in batch.reports
        if report.status == "missing"
    ]
    if recovery_request is not None and missing_dois:
        unavailable_dois = list(
            dict.fromkeys(
                [
                    *recovery_request.unavailable_full_text_dois,
                    *missing_dois,
                ]
            )
        )
        next_round = recovery_request.recovery_round + 1
        if next_round <= recovery_request.max_recovery_rounds:
            retried_request = recovery_request.model_copy(
                update={
                    "status": "pending_search",
                    "recovery_round": next_round,
                    "unavailable_full_text_dois": unavailable_dois,
                }
            )
            route_evidence_recovery_to_search(retried_request)
            st.session_state["mvp_flash"] = (
                f"本轮 {len(missing_dois)} 篇候选全文仍无法从本地取得；"
                "Agent 已记录这些 DOI 并自动补搜替代论文，不需要用户逐轮下载。"
            )
            st.rerun()
        blocked_request = recovery_request.model_copy(
            update={
                "status": "blocked",
                "unavailable_full_text_dois": unavailable_dois,
                "blocked_reason": (
                    "自动替代全文已达到恢复上限；需要一次性下载合并清单中的"
                    "受限 PDF。"
                ),
            }
        )
        st.session_state[EVIDENCE_RECOVERY_REQUEST_KEY] = (
            blocked_request.model_dump_json(indent=2)
        )
        st.session_state["mvp_navigation_request"] = "writing"
        st.session_state["mvp_flash"] = (
            "Agent 已完成全部自动替代尝试；现在只请求一次人工操作，"
            "所需 PDF 已合并为一张下载清单。"
        )
        st.rerun()
    if verified_count == 0:
        st.warning("当前没有通过身份与完整性检查的PDF，不能提取全文证据。")
        return

    st.caption(
        "代码固定 DOI、文件哈希、页码和原文；DeepSeek只对候选页面做语义归类。"
    )
    if "v03_evidence_library_json" not in st.session_state:
        recovery_extract = _load_evidence_recovery_request() is not None
        if st.button(
            "提取证据并准备写作",
            type="primary",
            width="stretch",
        ) or recovery_extract:
            try:
                with st.spinner("正在分页提取PDF、生成证据并检查章节覆盖……"):
                    library = _build_evidence_library(selection, batch)
            except Exception as exc:
                st.error(f"证据提取中断：{exc}")
            else:
                st.session_state["v03_evidence_library_json"] = library.model_dump_json(indent=2)
                st.session_state.pop("v03_writing_handoff_json", None)
                clear_writing_state()
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
        recovery_request = _load_evidence_recovery_request()
        missing_recovery_pdfs = [
            issue
            for issue in library.unresolved_issues
            if issue.startswith("core_pdf_missing:")
        ]
        if recovery_request is not None and missing_recovery_pdfs:
            unavailable_dois = list(
                dict.fromkeys(
                    [
                        *recovery_request.unavailable_full_text_dois,
                        *(
                            issue.removeprefix("core_pdf_missing:")
                            for issue in missing_recovery_pdfs
                        ),
                    ]
                )
            )
            next_round = recovery_request.recovery_round + 1
            if next_round <= recovery_request.max_recovery_rounds:
                retried_request = recovery_request.model_copy(
                    update={
                        "recovery_round": next_round,
                        "status": "pending_search",
                        "unavailable_full_text_dois": unavailable_dois,
                    }
                )
                route_evidence_recovery_to_search(retried_request)
                st.session_state["mvp_flash"] = (
                    f"已有全文已生成 {len(library.evidence_cards)} 张证据卡；"
                    f"另有 {len(missing_recovery_pdfs)} 篇全文确实无法从本地获得。"
                    "系统正在自动补搜同一研究问题的替代论文，不需要手动调整关键词。"
                )
                st.rerun()
            missing_dois = set(unavailable_dois)
            remaining_core_dois = [
                doi
                for doi in st.session_state.get("v03_core_dois", [])
                if doi not in missing_dois
            ]
            if remaining_core_dois:
                reduced_request = recovery_request.model_copy(
                    update={
                        "status": "pending_full_text",
                        "unavailable_full_text_dois": unavailable_dois,
                    }
                )
                st.session_state[EVIDENCE_RECOVERY_REQUEST_KEY] = (
                    reduced_request.model_dump_json(indent=2)
                )
                st.session_state["v03_core_dois"] = remaining_core_dois
                for key in (
                    "v03_pdf_inspection_json",
                    "v03_evidence_library_json",
                    "v03_writing_handoff_json",
                ):
                    st.session_state.pop(key, None)
                clear_writing_state()
                st.session_state["mvp_flash"] = (
                    f"{len(missing_recovery_pdfs)} 篇候选全文仍不可获得，系统已将其从核心队列剔除；"
                    f"将使用其余 {len(remaining_core_dois)} 篇可验证全文重建证据库。"
                )
                st.rerun()
            blocked_request = recovery_request.model_copy(
                update={
                    "status": "blocked",
                    "blocked_reason": (
                        "定向补搜后仍无法取得满足同一论点的完整全文；"
                        "需要检查题目边界，或由用户下载受限 PDF。"
                    ),
                }
            )
            st.session_state[EVIDENCE_RECOVERY_REQUEST_KEY] = (
                blocked_request.model_dump_json(indent=2)
            )
            st.error(
                "系统已完成两轮自动恢复，但替代全文仍不足。"
                "已停止循环并保留全部已成功证据；现在只需要下载缺失 PDF，"
                "或调整该段的论证范围。"
            )
        if recovery_request is not None and outline.unresolved_gaps:
            gap_sections = [
                section for section in outline.sections if section.evidence_gap
            ]
            gap_ids = {section.section_id for section in gap_sections}
            previous_ids = set(recovery_request.affected_section_ids)
            new_incident = bool(gap_ids - previous_ids)
            next_round = 1 if new_incident else recovery_request.recovery_round + 1
            if next_round <= recovery_request.max_recovery_rounds:
                themes = {
                    theme.theme_id: theme for theme in selection.blueprint.themes
                }
                gaps = [
                    ParagraphEvidenceGap(
                        section_id=section.section_id,
                        section_title=section.title,
                        paragraph_number=1,
                        reason="detailed_claim_requires_full_text",
                        claim_focus=section.purpose,
                        central_question=(
                            section.research_questions[0]
                            if section.research_questions
                            else section.purpose
                        ),
                        missing_full_text_dois=[],
                        available_direct_evidence_dois=section.core_dois,
                        search_queries=(
                            themes[section.section_id].search_queries
                            if section.section_id in themes
                            else [f"{outline.topic} {section.title}"]
                        ),
                        detail=(
                            "PDF 身份复核或证据重建后，本章节仍缺少可追溯全文；"
                            "需要按同一研究问题补搜新的核心来源。"
                        ),
                    )
                    for section in gap_sections
                ]
                expanded_request = recovery_request.model_copy(
                    update={
                        "status": "pending_search",
                        "affected_section_ids": list(
                            dict.fromkeys(gap.section_id for gap in gaps)
                        ),
                        "gaps": gaps,
                        "requested_core_dois": [],
                        "search_queries_by_section": {
                            gap.section_id: gap.search_queries for gap in gaps
                        },
                        "recovery_round": next_round,
                        "planning_repair_round": 0,
                        "blocked_reason": None,
                    }
                )
                route_evidence_recovery_to_search(expanded_request)
                st.session_state["mvp_flash"] = (
                    "PDF 身份复核后出现新的章节证据缺口；系统已把它识别为新的"
                    "恢复事件，正在按该章节研究问题补搜替代全文。"
                )
                st.rerun()
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
                    clear_writing_state()
                    st.rerun()
        with st.expander("查看问题详情"):
            for blocker in blockers:
                st.write(f"- {_friendly_evidence_blocker(blocker)}")
        with st.expander("技术详情"):
            st.code("\n".join(blockers))
        return

    handoff_json = st.session_state.get("v03_writing_handoff_json")
    if not handoff_json:
        recovery_request = _load_evidence_recovery_request(include_blocked=True)
        recovery_adopt = recovery_request is not None
        if st.button(
            "采用这些证据并进入写作",
            type="primary",
            width="stretch",
        ) or recovery_adopt:
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
            if recovery_request is not None:
                st.session_state[EVIDENCE_RECOVERY_REQUEST_KEY] = (
                    recovery_request.model_copy(
                        update={"status": "ready_to_resume", "blocked_reason": None}
                    ).model_dump_json(indent=2)
                )
            clear_writing_state()
            queue_evidence_recovery_resume()
            st.session_state["mvp_flash"] = (
                "证据已补齐，系统将自动更新受影响章节的计划并继续写作。"
                if recovery_adopt
                else "证据与写作章节已锁定，可以开始正文。"
            )
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
    verifications = LiteratureVerificationBatch.model_validate_json(
        st.session_state["literature_verification_json"]
    )
    return build_evidence_library(
        selection,
        batch,
        confirmed_requirement=confirmed_requirement,
        verifications=verifications,
        policy=policy,
        cache_root=project_root() / "runtime" / "evidence_console",
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
    if library.literature_matrix:
        with st.expander("查看文献准入表与写作用途"):
            st.dataframe(
                [
                    {
                        "题名": row.title,
                        "DOI": row.doi,
                        "准入状态": row.admission_status,
                        "中心性": row.centrality,
                        "支撑论点": row.supported_claim or "旧项目待复核",
                        "适用章节": row.suitable_section_id or "待复核",
                        "使用边界": row.use_boundary or "待复核",
                    }
                    for row in library.literature_matrix
                ],
                hide_index=True,
                width="stretch",
            )
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
