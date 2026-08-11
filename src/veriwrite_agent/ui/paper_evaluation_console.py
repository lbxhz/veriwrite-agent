"""Independent paper-quality evaluation UI for uploaded DOCX and PDF files."""

from __future__ import annotations

import json

import streamlit as st
from pydantic import ValidationError

from veriwrite_agent.config.settings import LLMSettings
from veriwrite_agent.llm.deepseek_client import DeepSeekClient
from veriwrite_agent.models.standalone_evaluation import StandalonePaperEvaluation
from veriwrite_agent.services.standalone_paper_evaluation import (
    StandalonePaperEvaluationService,
    extract_uploaded_paper,
)

REPORTS_KEY = "standalone_paper_evaluation_reports_json"


def render_paper_evaluation_console() -> None:
    """Render a workflow-independent evaluator and multi-paper comparison table."""

    st.header("独立论文质量评测")
    st.caption(
        "上传既有论文或他人论文即可评测，不依赖 V0.1–V0.4 项目进度。"
        "DOCX 与可提取文字的 PDF 均受支持。"
    )
    st.info(
        "这里评的是成品论文：要求符合、引文编号覆盖、论断与文后题名匹配、主题聚焦、"
        "分析综合、结构和语言。由于没有原始证据库，论断匹配属于语义抽查，"
        "不会假装完成逐句事实蕴含验证。"
    )

    with st.expander("统一评测条件（比较多篇论文时建议填写）", expanded=True):
        expected_topic = st.text_input(
            "预期题目或核心主题",
            key="standalone_expected_topic",
            placeholder="例如：大气遥感数据获取与人工智能反演技术研究进展",
        )
        requirements = st.text_area(
            "课程要求或评分标准（可选）",
            key="standalone_requirements",
            height=120,
            placeholder="例如：中文；不少于15000字；至少60篇参考文献；包含引言、研究现状、问题、趋势和结论。",
        )
        st.caption("不填写时按通用课程论文规范评价；多篇横向比较必须使用同一组条件。")

    uploads = st.file_uploader(
        "上传论文",
        type=["docx", "pdf"],
        accept_multiple_files=True,
        key="standalone_paper_uploads",
        help="可一次上传多篇并在同一评分口径下比较；扫描版 PDF 请先完成 OCR。",
    )
    if st.button(
        "开始评测",
        type="primary",
        width="stretch",
        disabled=not uploads,
        key="standalone_start_evaluation",
    ):
        try:
            settings = LLMSettings().for_quality_review()
        except ValidationError as exc:
            st.error(f"LLM 配置不可用：{exc}")
        else:
            service = StandalonePaperEvaluationService(
                DeepSeekClient(settings), reviewer_model=settings.model
            )
            completed: list[StandalonePaperEvaluation] = []
            progress = st.progress(0.0, text="准备读取论文")
            for index, upload in enumerate(uploads, 1):
                progress.progress(
                    (index - 1) / len(uploads),
                    text=f"正在评测 {upload.name}（{index}/{len(uploads)}）",
                )
                try:
                    paper = extract_uploaded_paper(upload.name, upload.getvalue())
                    report = service.evaluate(
                        paper,
                        expected_topic=expected_topic,
                        requirements=requirements,
                    )
                except Exception as exc:
                    st.error(f"{upload.name} 评测失败：{exc}")
                    continue
                completed.append(report)
            progress.progress(1.0, text=f"评测完成：{len(completed)}/{len(uploads)} 篇")
            if completed:
                st.session_state[REPORTS_KEY] = json.dumps(
                    [report.model_dump(mode="json") for report in completed],
                    ensure_ascii=False,
                )
                st.rerun()

    reports = _restore_reports()
    if not reports:
        st.caption("尚无评测结果。上传一篇或多篇论文后点击“开始评测”。")
        return

    st.subheader("评测结果")
    if len(reports) > 1:
        st.dataframe(
            [
                {
                    "论文": report.source_filename,
                    "综合分": report.overall_score,
                    **{metric.label: metric.score for metric in report.metrics},
                }
                for report in reports
            ],
            hide_index=True,
            width="stretch",
        )
    selected_name = st.selectbox(
        "查看论文",
        options=[report.source_filename for report in reports],
        key="standalone_selected_report",
    )
    report = next(item for item in reports if item.source_filename == selected_name)
    _render_report(report)


def _restore_reports() -> list[StandalonePaperEvaluation]:
    raw = st.session_state.get(REPORTS_KEY)
    if not isinstance(raw, str):
        return []
    try:
        payload = json.loads(raw)
        return [StandalonePaperEvaluation.model_validate(item) for item in payload]
    except (json.JSONDecodeError, ValidationError, TypeError):
        st.session_state.pop(REPORTS_KEY, None)
        return []


def _render_report(report: StandalonePaperEvaluation) -> None:
    grade_labels = {
        "excellent": "优秀",
        "strong": "良好",
        "acceptable": "可接受",
        "weak": "需改进",
    }
    summary = st.columns(4)
    summary[0].metric("综合质量", f"{report.overall_score:.1f}/100")
    summary[1].metric("质量等级", grade_labels[report.grade])
    summary[2].metric("正文统计单位", report.counted_units)
    summary[3].metric("参考文献", report.reference_count)
    st.caption(
        f"识别题目：{report.inferred_title}　|　格式：{report.source_format.upper()}　|　"
        f"引用标记：{report.citation_marker_count}"
    )
    st.dataframe(
        [
            {
                "指标": metric.label,
                "得分": metric.score,
                "权重": f"{metric.weight:.0%}",
                "加权分": metric.weighted_points,
                "依据": "；".join(metric.basis),
            }
            for metric in report.metrics
        ],
        hide_index=True,
        width="stretch",
    )
    if report.findings:
        st.markdown("#### 优先改进项")
        st.dataframe(
            [
                {
                    "程度": "主要" if finding.severity == "major" else "次要",
                    "维度": finding.dimension,
                    "位置": finding.location,
                    "问题": finding.detail,
                    "建议": finding.recommendation,
                }
                for finding in report.findings
            ],
            hide_index=True,
            width="stretch",
        )
    else:
        st.success("独立评审未发现需要优先修正的明显问题。")
    with st.expander("评测边界与导出"):
        for limitation in report.limitations:
            st.caption(f"局限：{limitation}")
        st.caption(
            f"评测器：{report.evaluation_method}；模型：{report.reviewer_model}；"
            f"文本提取：{report.extraction_method}"
        )
        st.download_button(
            "下载评测 JSON",
            report.model_dump_json(indent=2),
            file_name=f"{report.paper_fingerprint[:10]}_paper_evaluation.json",
            mime="application/json",
            width="stretch",
            key=f"standalone_download_{report.paper_fingerprint}",
        )
