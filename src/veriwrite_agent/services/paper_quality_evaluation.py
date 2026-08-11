"""Transparent paper scorecards inspired by scientific-agent benchmarks."""

from __future__ import annotations

import hashlib

from veriwrite_agent.models.final_delivery import FinalPaperPackage
from veriwrite_agent.models.paper_quality import (
    PaperQualityComparison,
    PaperQualityMetric,
    PaperQualityMetricCode,
    PaperQualityScorecard,
)
from veriwrite_agent.models.writing import SectionDraftIssue, V04WritingProject
from veriwrite_agent.models.writing_plan import GroundedWritingPlan
from veriwrite_agent.services.pdf_acquisition import (
    evidence_document_identity_conflicts,
)


class PaperQualityEvaluationService:
    """Build a reproducible scorecard without pretending proxies are ground truth."""

    _WEIGHTS: dict[PaperQualityMetricCode, float] = {
        "requirement_compliance": 0.20,
        "reference_integrity": 0.15,
        "evidence_traceability": 0.20,
        "topic_relevance": 0.15,
        "analysis_synthesis": 0.20,
        "presentation_quality": 0.10,
    }

    def evaluate(
        self,
        package: FinalPaperPackage,
        project: V04WritingProject,
        plan: GroundedWritingPlan,
    ) -> PaperQualityScorecard:
        paragraph_count = len(_paragraphs(project))
        issues = [
            issue
            for state in project.sections
            if state.draft is not None
            for issue in state.draft.issues
        ]
        blocking_issues = [
            *(f"final:{issue.code}" for issue in package.audit.issues if issue.severity == "blocking"),
            *(
                f"{state.section_id}:{issue.code}"
                for state in project.sections
                if state.draft is not None
                for issue in state.draft.issues
                if issue.severity == "blocking"
            ),
        ]
        if project.status != "body_complete":
            blocking_issues.append("writing:body_incomplete")
        identity_conflicts = evidence_document_identity_conflicts(
            project.handoff.evidence_library
        )
        blocking_issues.extend(
            f"evidence:document_identity_mismatch:{doi}"
            for doi in identity_conflicts
        )

        metrics = [
            self._metric(
                "requirement_compliance",
                "要求符合度",
                _requirement_score(package),
                [
                    f"阻塞项 {package.audit.blocking_count}",
                    f"审计问题 {len(package.audit.issues)}",
                    f"正文统计单位 {package.audit.counted_units}",
                    f"实际参考文献 {package.audit.reference_count}",
                ],
            ),
            self._metric(
                "reference_integrity",
                "参考文献完整性",
                _reference_integrity_score(package, project),
                _reference_integrity_basis(package, project),
            ),
            self._metric(
                "evidence_traceability",
                "证据可追溯性",
                _evidence_traceability_score(project, issues),
                _evidence_traceability_basis(project),
                limitation="当前为页码与绑定代理分；尚未逐句判断语义蕴含。",
            ),
            self._metric(
                "topic_relevance",
                "主题相关性",
                _issue_ratio_score(
                    issues,
                    {"topic_drift"},
                    paragraph_count=max(paragraph_count, 1),
                    maximum_penalty=65,
                ),
                _issue_basis(issues, {"topic_drift"}, "主题漂移"),
            ),
            self._metric(
                "analysis_synthesis",
                "分析与综合",
                _analysis_score(plan, issues),
                _analysis_basis(plan, issues),
                limitation="该分数结合论证动作和审稿问题；仍需金标准或专家盲评校准。",
            ),
            self._metric(
                "presentation_quality",
                "结构与表达",
                _issue_ratio_score(
                    issues,
                    {
                        "language_mismatch",
                        "paragraph_repetition",
                        "coherence_gap",
                        "terminology_inconsistent",
                        "academic_style_problem",
                        "workflow_instruction_leak",
                    },
                    paragraph_count=max(paragraph_count, 1),
                    maximum_penalty=55,
                ),
                _issue_basis(
                    issues,
                    {
                        "language_mismatch",
                        "paragraph_repetition",
                        "coherence_gap",
                        "terminology_inconsistent",
                        "academic_style_problem",
                        "workflow_instruction_leak",
                    },
                    "表达与结构问题",
                ),
            ),
        ]
        overall = round(sum(metric.weighted_points for metric in metrics), 2)
        return PaperQualityScorecard(
            paper_fingerprint=hashlib.sha256(
                package.markdown.encode("utf-8")
            ).hexdigest(),
            release_gate="blocked" if blocking_issues else "passed",
            overall_score=overall,
            grade=_grade(overall),
            metrics=metrics,
            blocking_issues=list(dict.fromkeys(blocking_issues)),
            limitations=[
                "总分用于同一任务、同一要求下的版本比较，不应跨学科直接排名。",
                "证据追溯分只评价绑定完整性，不把允许的元数据背景引用误判为缺页。",
                "主题、论证与表达分使用当前复审问题占比；仍需专家金标准校准。",
                "逐句事实蕴含和专家金标准覆盖率将在后续版本加入。",
            ],
        )

    def compare(
        self,
        baseline: PaperQualityScorecard,
        candidate: PaperQualityScorecard,
    ) -> PaperQualityComparison:
        if baseline.evaluation_method != candidate.evaluation_method:
            raise ValueError("different paper-quality evaluator versions are not comparable")
        baseline_by_code = {metric.code: metric for metric in baseline.metrics}
        candidate_by_code = {metric.code: metric for metric in candidate.metrics}
        deltas = {
            code: round(candidate_by_code[code].score - metric.score, 2)
            for code, metric in baseline_by_code.items()
        }
        return PaperQualityComparison(
            baseline_fingerprint=baseline.paper_fingerprint,
            candidate_fingerprint=candidate.paper_fingerprint,
            overall_delta=round(candidate.overall_score - baseline.overall_score, 2),
            metric_deltas=deltas,
            improved_metrics=[code for code, delta in deltas.items() if delta > 0.01],
            regressed_metrics=[code for code, delta in deltas.items() if delta < -0.01],
        )

    def _metric(
        self,
        code: PaperQualityMetricCode,
        label: str,
        score: float,
        basis: list[str],
        *,
        limitation: str | None = None,
    ) -> PaperQualityMetric:
        rounded_score = round(max(0.0, min(100.0, score)), 2)
        weight = self._WEIGHTS[code]
        return PaperQualityMetric(
            code=code,
            label=label,
            score=rounded_score,
            weight=weight,
            weighted_points=round(rounded_score * weight, 2),
            basis=basis,
            limitation=limitation,
        )


def _requirement_score(package: FinalPaperPackage) -> float:
    blocking = package.audit.blocking_count
    warnings = sum(issue.severity == "warning" for issue in package.audit.issues)
    return 100 - blocking * 25 - warnings * 4


def _reference_integrity_score(
    package: FinalPaperPackage,
    project: V04WritingProject,
) -> float:
    if evidence_document_identity_conflicts(project.handoff.evidence_library):
        return 0.0
    citations = _citations(project)
    reference_dois = {entry.doi for entry in package.references}
    library_dois = {record.doi for record in project.handoff.evidence_library.records}
    if not citations:
        return 0.0
    mapped = sum(citation.doi in reference_dois for citation in citations) / len(citations)
    verified = sum(citation.doi in library_dois for citation in citations) / len(citations)
    unique_references = len(reference_dois) / max(len(package.references), 1)
    return 45 * mapped + 45 * verified + 10 * unique_references


def _reference_integrity_basis(
    package: FinalPaperPackage,
    project: V04WritingProject,
) -> list[str]:
    identity_conflicts = evidence_document_identity_conflicts(
        project.handoff.evidence_library
    )
    citations = _citations(project)
    reference_dois = {entry.doi for entry in package.references}
    mapped = sum(citation.doi in reference_dois for citation in citations)
    basis = [
        f"正文引用绑定 {len(citations)} 条",
        f"可映射至文后表 {mapped}/{len(citations)}",
        f"文后 DOI 去重后 {len(reference_dois)} 篇",
    ]
    if identity_conflicts:
        basis.append(
            "PDF 首页 DOI 身份冲突："
            + "；".join(
                f"{expected} -> {', '.join(detected)}"
                for expected, detected in identity_conflicts.items()
            )
        )
    return basis


def _evidence_traceability_score(
    project: V04WritingProject,
    issues: list[SectionDraftIssue],
) -> float:
    if evidence_document_identity_conflicts(project.handoff.evidence_library):
        return 0.0
    paragraphs = _paragraphs(project)
    citations = _citations(project)
    if not paragraphs or not citations:
        return 0.0
    bound_paragraphs = sum(
        bool(paragraph.evidence_card_ids or paragraph.source_dois)
        for paragraph in paragraphs
    ) / len(paragraphs)
    evidence_required = [citation for citation in citations if citation.evidence_card_ids]
    if evidence_required:
        page_backed = sum(
            bool(citation.page_numbers) for citation in evidence_required
        ) / len(evidence_required)
    else:
        page_backed = 1.0
    # C-background sources are intentionally admitted from verified metadata and
    # therefore do not require an evidence card or PDF locator. This metric only
    # measures whether bindings that *do* claim full-text evidence are complete.
    return 45 * bound_paragraphs + 55 * page_backed


def _evidence_traceability_basis(project: V04WritingProject) -> list[str]:
    paragraphs = _paragraphs(project)
    citations = _citations(project)
    evidence_required = [citation for citation in citations if citation.evidence_card_ids]
    page_backed = sum(bool(citation.page_numbers) for citation in evidence_required)
    metadata_only = len(citations) - len(evidence_required)
    basis = [
        f"段落证据绑定 {sum(bool(p.evidence_card_ids or p.source_dois) for p in paragraphs)}/{len(paragraphs)}",
        f"需全文证据且带页码的引用 {page_backed}/{len(evidence_required)}",
        f"按规则允许仅使用已验证元数据的背景引用 {metadata_only} 条",
    ]
    if evidence_document_identity_conflicts(project.handoff.evidence_library):
        basis.append("存在 PDF—DOI 身份冲突，全部相关证据轨迹按 0 分处理")
    return basis


def _analysis_score(
    plan: GroundedWritingPlan,
    issues: list[SectionDraftIssue],
) -> float:
    paragraphs = [paragraph for section in plan.sections for paragraph in section.paragraphs]
    analytical_moves = {
        "compare_studies",
        "synthesize_consensus",
        "analyze_difference",
        "evaluate_limitation",
        "author_judgment",
    }
    analytical = sum(
        paragraph.argument_move in analytical_moves for paragraph in paragraphs
    )
    ratio = analytical / max(len(paragraphs), 1)
    issue_score = _issue_ratio_score(
        issues,
        {"unsupported_claim", "overstated_evidence", "coherence_gap"},
        paragraph_count=max(len(paragraphs), 1),
        maximum_penalty=55,
    )
    return min(ratio / 0.40, 1.0) * 65 + issue_score * 0.35


def _analysis_basis(
    plan: GroundedWritingPlan,
    issues: list[SectionDraftIssue],
) -> list[str]:
    paragraphs = [paragraph for section in plan.sections for paragraph in section.paragraphs]
    analytical_moves = {
        "compare_studies",
        "synthesize_consensus",
        "analyze_difference",
        "evaluate_limitation",
        "author_judgment",
    }
    analytical = sum(
        paragraph.argument_move in analytical_moves for paragraph in paragraphs
    )
    related = sum(
        issue.code in {"unsupported_claim", "overstated_evidence", "coherence_gap"}
        for issue in issues
    )
    return [
        f"比较/综合/评价型段落 {analytical}/{len(paragraphs)}",
        f"论证相关审稿问题 {related}",
    ]


def _issue_ratio_score(
    issues: list[SectionDraftIssue],
    codes: set[str],
    *,
    paragraph_count: int,
    maximum_penalty: float,
) -> float:
    matching = [issue for issue in issues if issue.code in codes]
    # One finding per (chapter, paragraph, code) is already enforced upstream.
    # Section-local paragraph numbers repeat, so using the raw finding count is
    # more accurate than collapsing identical paragraph numbers across chapters.
    prevalence = min(len(matching) / max(paragraph_count, 1), 1.0)
    blocking = sum(issue.severity == "blocking" for issue in matching)
    return 100 - maximum_penalty * prevalence - min(20, blocking * 10)


def _issue_basis(
    issues: list[SectionDraftIssue],
    codes: set[str],
    label: str,
) -> list[str]:
    matching = [issue for issue in issues if issue.code in codes]
    blocking = sum(issue.severity == "blocking" for issue in matching)
    return [f"{label} {len(matching)} 项", f"其中阻塞项 {blocking} 项"]


def _paragraphs(project: V04WritingProject):
    return [
        paragraph
        for state in project.sections
        if state.draft is not None
        for paragraph in state.draft.paragraphs
    ]


def _citations(project: V04WritingProject):
    return [
        citation
        for state in project.sections
        if state.draft is not None
        for citation in state.draft.citations
    ]


def _grade(score: float) -> str:
    if score >= 90:
        return "excellent"
    if score >= 80:
        return "strong"
    if score >= 70:
        return "acceptable"
    return "weak"
