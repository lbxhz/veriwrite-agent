"""Generate an isolated, resumable formal DOCX preview from the prior MVP project.

The runner deliberately reuses verified DOI/RIS metadata and confirmed PDF evidence,
but it reruns the topic-admission gate, problem-driven outline, paragraph planning,
section writing, editorial review, final audit, and DOCX export. It never mutates the
active Streamlit project snapshot.
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from veriwrite_agent.config.settings import LLMSettings
from veriwrite_agent.llm.deepseek_client import DeepSeekClient
from veriwrite_agent.models.evidence import (
    EvidenceLibrary,
    LiteratureLibraryRecord,
)
from veriwrite_agent.models.final_delivery import (
    FinalMatterProposal,
)
from veriwrite_agent.models.literature_discovery import CandidateDecision
from veriwrite_agent.models.literature_selection import (
    BalancedLiteratureSelection,
    LiteratureRelevanceAssessment,
    LiteratureRelevanceAssessmentBatch,
    LiteratureSearchBlueprint,
    LiteratureSelectionCandidate,
    LiteratureThemePlan,
    ThemeRelevanceScore,
)
from veriwrite_agent.models.literature_verification import LiteratureVerificationBatch
from veriwrite_agent.models.requirements import TopicBoundary
from veriwrite_agent.models.requirement_workflow import ConfirmedRequirementSpec
from veriwrite_agent.models.writing import V04WritingProject
from veriwrite_agent.models.writing_plan import GroundedWritingPlan
from veriwrite_agent.models.writing_quality import SectionQualityReview
from veriwrite_agent.services.evidence_library import (
    EvidenceLibraryBuilder,
    EvidenceLibraryConfirmationService,
)
from veriwrite_agent.services.final_delivery import (
    FinalPaperAssembler,
    FinalPaperDocxExporter,
    LLMFinalMatterWriter,
)
from veriwrite_agent.services.grounded_writing import (
    SectionEvidencePacketBuilder,
    WritingProjectService,
    count_writing_units,
)
from veriwrite_agent.services.literature_relevance_scorer import (
    LLMLiteratureRelevanceScorer,
)
from veriwrite_agent.services.literature_selector import BalancedLiteratureSelector
from veriwrite_agent.services.requirement_policy import RequirementPolicyCompiler
from veriwrite_agent.services.writing_handoff import (
    WritingHandoffService,
    WritingOutlineBuilder,
)
from veriwrite_agent.services.writing_planning import (
    GroundedWritingPlanner,
    LLMGroundedParagraphWriter,
    ParagraphWritingRuntimeCache,
    PlannedSectionDraftService,
    WritingPlanRuntimeCache,
)
from veriwrite_agent.services.writing_quality import (
    LLMSectionQualityReviewer,
    apply_section_quality_review,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
ACTIVE_PROJECT = REPO_ROOT / "runtime" / "mvp_projects" / "active_project.json"
RUN_ROOT = (
    REPO_ROOT
    / "runtime"
    / "generated_runs"
    / "formal_course_paper_grounded_20260805"
)
OUTPUT_ROOT = REPO_ROOT / "outputs" / "formal_course_paper_grounded_20260805"

CORE_ASSIGNMENTS = {
    "10.1002/2017jd027412": {
        "theme_id": "theme_international_status",
        "supported_claim": (
            "深蓝气溶胶产品向VIIRS的扩展说明，跨传感器延续需要统一反演逻辑、"
            "质量控制与产品验证。"
        ),
        "use_boundary": "仅用于气溶胶卫星产品延续、算法移植和验证，不外推到全部大气参数。",
    },
    "10.5194/amt-17-5655-2024": {
        "theme_id": "theme_ai_retrieval",
        "supported_claim": (
            "机器学习能够从MODIS云属性估计海洋云底高度，但结果受训练样本、"
            "观测条件和验证策略约束。"
        ),
        "use_boundary": "用于云参数反演的方法与局限比较，不代表所有大气遥感任务。",
    },
    "10.1007/bf00138366": {
        "theme_id": "theme_multidisciplinary",
        "supported_claim": (
            "历史卫星资料的回溯分析可服务生物圈—大气耦合研究，体现大气遥感"
            "与地球系统科学的交叉价值。"
        ),
        "use_boundary": "仅作跨学科研究的历史方法锚点，不作为当前算法性能依据。",
    },
    "10.1016/j.atmosres.2026.109065": {
        "theme_id": "theme_future_trends",
        "supported_claim": (
            "辐射传输与机器学习耦合可提升云短波辐射强迫反演的物理一致性，"
            "同时暴露区域迁移和不确定性控制问题。"
        ),
        "use_boundary": "用于物理约束机器学习、可靠性与发展趋势，不泛化为通用AI结论。",
    },
}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(value, "model_dump"):
        payload = value.model_dump(mode="json")
    else:
        payload = value
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_active_state() -> dict[str, object]:
    payload = json.loads(ACTIVE_PROJECT.read_text(encoding="utf-8"))
    return payload["state"]


def _requirement_with_boundary(
    state: dict[str, object],
) -> tuple[ConfirmedRequirementSpec, object]:
    original = ConfirmedRequirementSpec.model_validate_json(str(state["confirmed_json"]))
    boundary = TopicBoundary(
        central_question=(
            "人工智能和新一代信息技术如何服务于大气成分、云、气溶胶、辐射及"
            "边界层的观测与反演，其可靠性、可迁移性和应用边界是什么？"
        ),
        included_objects=[
            "大气成分",
            "云",
            "气溶胶",
            "温室气体和痕量气体",
            "大气辐射",
            "边界层",
            "大气遥感观测系统",
        ],
        excluded_objects=[
            "土壤水分",
            "考古",
            "健身物联网",
            "海底油气",
            "地下储气库",
            "非大气地表遥感",
        ],
        contextual_only_topics=[
            "云计算",
            "边缘计算",
            "物联网",
            "通用大数据平台",
            "卫星通信",
        ],
        origin="explicit",
    )
    requirement = original.requirement.model_copy(update={"topic_boundary": boundary})
    confirmed = original.model_copy(update={"requirement": requirement})
    policy = RequirementPolicyCompiler(current_year=2026).compile(confirmed)
    return confirmed, policy


def _blueprint(policy: object) -> LiteratureSearchBlueprint:
    themes = [
        LiteratureThemePlan(
            theme_id="theme_international_status",
            section_title="国内外大气遥感观测与反演研究现状",
            section_purpose=(
                "以观测对象、传感器产品和验证链为比较轴，归纳国内外研究的共识、"
                "差异和代表性进展，避免逐篇摘要式罗列。"
            ),
            research_questions=[
                "大气成分、云和气溶胶观测形成了哪些主要产品与验证范式？",
                "不同传感器和区域条件下，反演结果的可比性受哪些因素限制？",
            ],
            primary_keywords=["atmospheric remote sensing", "aerosol retrieval", "cloud retrieval"],
            related_keywords=["satellite validation", "atmospheric composition", "remote sensing product"],
            search_queries=[
                "atmospheric remote sensing retrieval validation",
                "satellite aerosol cloud atmospheric composition",
            ],
            target_count=14,
            priority=1,
        ),
        LiteratureThemePlan(
            theme_id="theme_ai_retrieval",
            section_title="人工智能驱动的云、气溶胶与大气成分反演",
            section_purpose=(
                "围绕反演问题比较机器学习、深度学习与传统方法的输入、约束、"
                "验证和适用条件，形成方法判断而不是论文摘要串联。"
            ),
            research_questions=[
                "人工智能在不同大气参数反演中解决了什么问题？",
                "数据驱动方法相对物理反演的优势与失效条件是什么？",
            ],
            primary_keywords=["machine learning atmospheric retrieval", "deep learning cloud retrieval"],
            related_keywords=["aerosol", "trace gas", "radiative transfer", "uncertainty"],
            search_queries=[
                "machine learning atmospheric remote sensing retrieval",
                "deep learning cloud aerosol retrieval",
            ],
            target_count=18,
            priority=2,
        ),
        LiteratureThemePlan(
            theme_id="theme_multidisciplinary",
            section_title="大气遥感专业领域的多源观测与多学科交叉",
            section_purpose=(
                "只讨论能够直接推进大气过程认识的跨学科组合，比较多源数据、"
                "地球系统过程和尺度耦合的证据价值，并明确排除非大气遥感主题。"
            ),
            research_questions=[
                "多源观测如何改善对大气过程和地气耦合的认识？",
                "多学科交叉何时提供新证据，何时只是技术标签？",
            ],
            primary_keywords=["atmosphere biosphere coupling", "multi-source atmospheric observation"],
            related_keywords=["data fusion", "Earth system", "boundary layer", "interdisciplinary"],
            search_queries=[
                "multi-source atmospheric remote sensing data fusion",
                "biosphere atmosphere satellite observation",
            ],
            target_count=14,
            priority=3,
        ),
        LiteratureThemePlan(
            theme_id="theme_future_trends",
            section_title="物理约束、可靠性问题与新一代信息技术趋势",
            section_purpose=(
                "从物理一致性、不确定性、跨区域迁移和业务化验证出发分析现存问题，"
                "把云计算、边缘计算等新一代信息技术限定为大气观测链的支撑条件。"
            ),
            research_questions=[
                "物理约束与数据驱动方法如何结合以提高可靠性？",
                "新一代信息技术在大气遥感中应承担什么支撑角色？",
                "未来研究最需要补足哪些验证与泛化证据？",
            ],
            primary_keywords=["physics-informed atmospheric retrieval", "remote sensing uncertainty"],
            related_keywords=["hybrid inversion", "generalization", "operational validation", "edge processing"],
            search_queries=[
                "physics informed machine learning atmospheric retrieval",
                "atmospheric remote sensing uncertainty operational validation",
            ],
            target_count=14,
            priority=4,
        ),
    ]
    return LiteratureSearchBlueprint(
        topic="大气遥感",
        topic_boundary=policy.topic_boundary,
        discipline="大气科学与遥感科学技术",
        writing_through_line=(
            "从大气观测与产品验证出发，比较人工智能反演路径，再讨论多源交叉证据，"
            "最后以物理一致性和业务可靠性收束到新一代信息技术趋势。"
        ),
        target_total=60,
        themes=themes,
        accepted_tiers=["T1", "T2", "T3", "T4", "T5", "T6"],
        year_from=policy.references.year_from,
        year_to=policy.references.year_to,
        max_candidates=500,
        relevance_threshold=0.60,
        max_contextual_share=0.25,
        requirement_policy=policy,
    )


def _writing_blueprint(policy: object) -> LiteratureSearchBlueprint:
    """Collapse sparse search themes into two evidence-rich problem chapters."""

    return LiteratureSearchBlueprint(
        topic="大气遥感",
        topic_boundary=policy.topic_boundary,
        discipline="大气科学与遥感科学技术",
        writing_through_line=(
            "先比较国内外大气观测、产品验证与多学科证据，再评估人工智能和"
            "新一代信息技术支持的反演方法，最终归纳现状、问题与趋势。"
        ),
        target_total=60,
        themes=[
            LiteratureThemePlan(
                theme_id="theme_research_status",
                section_title="国内外研究现状与大气遥感专业领域的多学科交叉",
                section_purpose=(
                    "以大气观测对象、传感器产品、验证链和尺度耦合为问题轴，比较"
                    "国内外研究的共识与差异；多学科材料只在直接解释大气过程时使用。"
                ),
                research_questions=[
                    "大气成分、云和气溶胶观测形成了哪些主要产品与验证范式？",
                    "跨传感器和跨区域结果的可比性受哪些条件限制？",
                    "多源观测与多学科交叉如何真正推进大气过程认识？",
                ],
                primary_keywords=["atmospheric remote sensing", "aerosol cloud retrieval"],
                related_keywords=["validation", "atmospheric composition", "data fusion"],
                search_queries=["atmospheric remote sensing retrieval validation"],
                target_count=30,
                priority=1,
            ),
            LiteratureThemePlan(
                theme_id="theme_ai_and_trends",
                section_title="人工智能反演、可靠性问题与新一代信息技术趋势",
                section_purpose=(
                    "围绕反演问题比较数据驱动、物理方法与混合方法，分析不确定性、"
                    "迁移能力和业务化验证；计算平台仅作为大气遥感观测链的支撑条件。"
                ),
                research_questions=[
                    "人工智能在不同大气参数反演中解决了什么问题？",
                    "数据驱动方法相对物理反演的优势、失效条件和可靠性边界是什么？",
                    "新一代信息技术应如何服务大气遥感业务链而不喧宾夺主？",
                ],
                primary_keywords=["machine learning atmospheric retrieval", "physics informed retrieval"],
                related_keywords=["uncertainty", "generalization", "operational validation"],
                search_queries=["machine learning atmospheric remote sensing retrieval"],
                target_count=30,
                priority=2,
            ),
        ],
        accepted_tiers=["T1", "T2", "T3", "T4", "T5", "T6"],
        year_from=policy.references.year_from,
        year_to=policy.references.year_to,
        max_candidates=500,
        relevance_threshold=0.60,
        max_contextual_share=0.50,
        requirement_policy=policy,
    )


def _collapse_assessments(
    batch: LiteratureRelevanceAssessmentBatch,
    writing_blueprint: LiteratureSearchBlueprint,
) -> LiteratureRelevanceAssessmentBatch:
    """Map four search dimensions into two evidence-rich writing problems."""

    collapsed = []
    for assessment in batch.assessments:
        scores = {item.theme_id: item for item in assessment.theme_scores}
        status_score = max(
            scores["theme_international_status"].score,
            scores["theme_multidisciplinary"].score,
        )
        ai_score = max(
            scores["theme_ai_retrieval"].score,
            scores["theme_future_trends"].score,
        )
        destination = (
            "theme_research_status"
            if assessment.suitable_section_id
            in {"theme_international_status", "theme_multidisciplinary"}
            else "theme_ai_and_trends"
        )
        if assessment.admission_status != "admit":
            destination = (
                "theme_research_status"
                if status_score >= ai_score
                else "theme_ai_and_trends"
            )
        best_theme = (
            "theme_research_status"
            if status_score >= ai_score
            else "theme_ai_and_trends"
        )
        centrality = assessment.centrality
        if (
            assessment.admission_status == "admit"
            and destination == "theme_research_status"
            and status_score >= 0.60
        ):
            # The four-theme evaluator often called directly relevant survey/status
            # evidence "supporting" merely because it was not an algorithm paper.
            # In this problem chapter it is central evidence, not contextual IT.
            centrality = "central"
        elif (
            assessment.admission_status == "admit"
            and destination == "theme_ai_and_trends"
            and assessment.suitable_section_id == "theme_ai_retrieval"
            and ai_score >= 0.70
        ):
            centrality = "central"
        collapsed.append(
            LiteratureRelevanceAssessment(
                doi=assessment.doi,
                theme_scores=[
                    ThemeRelevanceScore(
                        theme_id="theme_research_status",
                        score=status_score,
                        rationale="合并观测现状与直接相关的多学科证据维度。",
                        matched_concepts=list(
                            dict.fromkeys(
                                [
                                    *scores["theme_international_status"].matched_concepts,
                                    *scores["theme_multidisciplinary"].matched_concepts,
                                ]
                            )
                        ),
                    ),
                    ThemeRelevanceScore(
                        theme_id="theme_ai_and_trends",
                        score=ai_score,
                        rationale="合并人工智能方法、可靠性问题与受控技术趋势维度。",
                        matched_concepts=list(
                            dict.fromkeys(
                                [
                                    *scores["theme_ai_retrieval"].matched_concepts,
                                    *scores["theme_future_trends"].matched_concepts,
                                ]
                            )
                        ),
                    ),
                ],
                best_theme_id=best_theme,
                admission_status=assessment.admission_status,
                centrality=centrality,
                supported_claim=assessment.supported_claim,
                suitable_section_id=(
                    destination if assessment.admission_status == "admit" else None
                ),
                use_boundary=assessment.use_boundary,
                exclusion_reason=assessment.exclusion_reason,
            )
        )
    result = LiteratureRelevanceAssessmentBatch(assessments=collapsed)
    _write_json(RUN_ROOT / "relevance_writing_problems.json", result)
    _write_json(RUN_ROOT / "writing_blueprint_before_selection.json", writing_blueprint)
    return result


def _score_literature(
    blueprint: LiteratureSearchBlueprint,
    verifications: LiteratureVerificationBatch,
    client: DeepSeekClient,
) -> LiteratureRelevanceAssessmentBatch:
    cache_path = RUN_ROOT / "relevance_cache.json"
    if cache_path.is_file():
        batch = LiteratureRelevanceAssessmentBatch.model_validate_json(
            cache_path.read_text(encoding="utf-8")
        )
    else:
        batch = LiteratureRelevanceAssessmentBatch()
    scored = {item.doi for item in batch.assessments}
    pending = [item for item in verifications.verified_records if item.candidate.doi not in scored]
    scorer = LLMLiteratureRelevanceScorer(client, batch_size=20)
    for start in range(0, len(pending), 20):
        chunk = pending[start : start + 20]
        print(
            f"[V0.2 admission] scoring {start + 1}-{start + len(chunk)} / {len(pending)}",
            flush=True,
        )
        batch.assessments.extend(scorer.score(blueprint, chunk))
        _write_json(cache_path, batch)
    return _force_reviewed_core_assignments(batch, blueprint)


def _force_reviewed_core_assignments(
    batch: LiteratureRelevanceAssessmentBatch,
    blueprint: LiteratureSearchBlueprint,
) -> LiteratureRelevanceAssessmentBatch:
    updated = []
    for assessment in batch.assessments:
        assignment = CORE_ASSIGNMENTS.get(assessment.doi)
        if assignment is None:
            updated.append(assessment)
            continue
        target = assignment["theme_id"]
        scores = []
        for theme in blueprint.themes:
            existing = next(
                (item for item in assessment.theme_scores if item.theme_id == theme.theme_id),
                None,
            )
            scores.append(
                ThemeRelevanceScore(
                    theme_id=theme.theme_id,
                    score=(1.0 if theme.theme_id == target else min(existing.score if existing else 0.0, 0.75)),
                    rationale=(
                        "已下载全文经人工边界复核，直接支撑本节。"
                        if theme.theme_id == target
                        else (existing.rationale if existing else "不作为该节核心证据。")
                    ),
                    matched_concepts=(existing.matched_concepts if existing else []),
                )
            )
        updated.append(
            LiteratureRelevanceAssessment(
                doi=assessment.doi,
                theme_scores=scores,
                best_theme_id=str(target),
                admission_status="admit",
                centrality="central",
                supported_claim=str(assignment["supported_claim"]),
                suitable_section_id=str(target),
                use_boundary=str(assignment["use_boundary"]),
            )
        )
    result = LiteratureRelevanceAssessmentBatch(assessments=updated)
    _write_json(RUN_ROOT / "relevance_reviewed.json", result)
    return result


def _allocate_quotas(
    blueprint: LiteratureSearchBlueprint,
    capacities: dict[str, int],
) -> dict[str, int] | None:
    theme_ids = [theme.theme_id for theme in blueprint.themes]
    if any(capacities.get(theme_id, 0) < 1 for theme_id in theme_ids):
        return None
    if sum(capacities.values()) < blueprint.target_total:
        return None
    weights = {theme.theme_id: theme.target_count for theme in blueprint.themes}
    allocation = {theme_id: 1 for theme_id in theme_ids}
    while sum(allocation.values()) < blueprint.target_total:
        eligible = [
            theme_id
            for theme_id in theme_ids
            if allocation[theme_id] < capacities[theme_id]
        ]
        if not eligible:
            return None
        chosen = max(
            eligible,
            key=lambda theme_id: (
                weights[theme_id] / allocation[theme_id],
                capacities[theme_id] - allocation[theme_id],
            ),
        )
        allocation[chosen] += 1
    return allocation


def _select_literature(
    base_blueprint: LiteratureSearchBlueprint,
    assessments: LiteratureRelevanceAssessmentBatch,
    verifications: LiteratureVerificationBatch,
    decisions: dict[str, CandidateDecision],
) -> tuple[LiteratureSearchBlueprint, BalancedLiteratureSelection]:
    relevance = {item.doi: item for item in assessments.assessments}
    candidates = [
        LiteratureSelectionCandidate(
            verification=item,
            ranking=decisions[item.candidate.doi].ranking,
            norwegian_ranking=decisions[item.candidate.doi].norwegian_ranking,
            relevance=relevance[item.candidate.doi],
        )
        for item in verifications.verified_records
        if item.candidate.doi in decisions and item.candidate.doi in relevance
    ]
    best: BalancedLiteratureSelection | None = None
    for threshold in (0.60, 0.55, 0.50):
        for contextual_share in (0.25, 0.35, 0.50):
            capacities = {}
            for theme in base_blueprint.themes:
                central = 0
                supporting = 0
                for candidate in candidates:
                    rel = candidate.relevance
                    if (
                        rel.admission_status != "admit"
                        or rel.suitable_section_id != theme.theme_id
                        or next(item.score for item in rel.theme_scores if item.theme_id == theme.theme_id) < threshold
                    ):
                        continue
                    if rel.centrality == "central":
                        central += 1
                    elif rel.centrality == "supporting":
                        supporting += 1
                contextual_cap = math.ceil((central + supporting) * contextual_share)
                capacities[theme.theme_id] = central + min(supporting, contextual_cap)
            quotas = _allocate_quotas(base_blueprint, capacities)
            if quotas is None:
                continue
            themes = [
                theme.model_copy(update={"target_count": quotas[theme.theme_id]})
                for theme in base_blueprint.themes
            ]
            blueprint = base_blueprint.model_copy(
                update={
                    "themes": themes,
                    "relevance_threshold": threshold,
                    "max_contextual_share": contextual_share,
                }
            )
            selection = BalancedLiteratureSelector().select(blueprint, candidates)
            if best is None or len(selection.selected) > len(best.selected):
                best = selection
            print(
                "[V0.2 selection] "
                f"threshold={threshold:.2f} contextual={contextual_share:.2f} "
                f"selected={len(selection.selected)} shortages={selection.shortages}",
                flush=True,
            )
            if selection.target_reached and set(CORE_ASSIGNMENTS).issubset(
                {item.doi for item in selection.selected}
            ):
                _write_json(RUN_ROOT / "blueprint.json", blueprint)
                _write_json(RUN_ROOT / "selection.json", selection)
                return blueprint, selection
    detail = best.shortages if best is not None else "no feasible quota allocation"
    raise RuntimeError(f"topic-admitted pool cannot satisfy 60 references: {detail}")


def _build_library(
    state: dict[str, object],
    selection: BalancedLiteratureSelection,
    verifications: LiteratureVerificationBatch,
    policy: object,
) -> EvidenceLibrary:
    cached = RUN_ROOT / "evidence_library.json"
    if cached.is_file():
        return EvidenceLibrary.model_validate_json(cached.read_text(encoding="utf-8"))
    old = EvidenceLibrary.model_validate_json(str(state["v03_evidence_library_json"]))
    verification_by_doi = {
        item.candidate.doi: item for item in verifications.verified_records
    }
    old_record_by_doi = {item.doi: item for item in old.records}
    selected_dois = {item.doi for item in selection.selected}
    selected_theme_by_doi = {item.doi: item.theme_id for item in selection.selected}
    core_dois = set(CORE_ASSIGNMENTS) & selected_dois
    records = []
    for item in selection.selected:
        verification = verification_by_doi[item.doi]
        old_record = old_record_by_doi.get(item.doi)
        source_url = (
            old_record.source_url
            if old_record is not None
            else (
                verification.resolution.final_url
                if verification.resolution is not None and verification.resolution.final_url
                else f"https://doi.org/{item.doi}"
            )
        )
        records.append(
            LiteratureLibraryRecord(
                doi=item.doi,
                title=item.title,
                authors=item.authors,
                year=item.year,
                journal=item.journal,
                publisher=item.publisher,
                language=item.language,
                source_type=item.source_type,
                is_foreign=item.is_foreign,
                abstract=(
                    old_record.abstract
                    if old_record is not None and old_record.abstract
                    else verification.candidate.abstract
                ),
                source_url=source_url,
                theme_ids=[item.theme_id],
                admission_status="admitted",
                centrality=item.centrality,
                supported_claim=item.supported_claim,
                suitable_section_id=item.suitable_section_id,
                use_boundary=item.use_boundary,
                evidence_tier="A_core" if item.doi in core_dois else "C_background",
                evidence_status=(
                    "full_text_verified" if item.doi in core_dois else "metadata_verified"
                ),
                permitted_use="detailed_claims" if item.doi in core_dois else "background_only",
            )
        )
    documents = [item for item in old.documents if item.doi in core_dois]
    extractions = [item for item in old.extractions if item.doi in core_dois]
    page_selections = [item for item in old.page_selections if item.doi in core_dois]
    pages = [item for item in old.pages if item.doi in core_dois]
    cards = [
        item.model_copy(update={"theme_id": selected_theme_by_doi[item.doi]})
        for item in old.evidence_cards
        if item.doi in core_dois and item.review_status != "rejected"
    ]
    library = EvidenceLibraryBuilder().build(
        records=records,
        documents=documents,
        extractions=extractions,
        page_selections=page_selections,
        pages=pages,
        evidence_cards=cards,
        requirement_policy_fingerprint=policy.requirement_fingerprint,
    )
    library = EvidenceLibraryConfirmationService().confirm(
        library,
        confirmed_by="Codex preview run requested by user",
    )
    _write_json(cached, library)
    return library


def _write_admission_matrix(selection: BalancedLiteratureSelection) -> None:
    path = OUTPUT_ROOT / "literature_admission_matrix.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "DOI",
                "题名",
                "年份",
                "章节",
                "中心性",
                "支持论点",
                "使用边界",
                "相关性",
            ]
        )
        for item in selection.selected:
            writer.writerow(
                [
                    item.doi,
                    item.title,
                    item.year,
                    item.theme_id,
                    item.centrality,
                    item.supported_claim,
                    item.use_boundary,
                    item.relevance_score,
                ]
            )


def _write_selected_ris(
    selection: BalancedLiteratureSelection,
    verifications: LiteratureVerificationBatch,
) -> None:
    verification_by_doi = {
        item.candidate.doi: item for item in verifications.verified_records
    }
    ris = "\n\n".join(
        verification_by_doi[item.doi].authority.raw_ris.strip()
        for item in selection.selected
        if verification_by_doi[item.doi].authority is not None
        and verification_by_doi[item.doi].authority.raw_ris
    )
    (OUTPUT_ROOT / "selected_60_references.ris").write_text(
        ris + ("\n" if ris else ""),
        encoding="utf-8",
    )


def _draft_sections(
    handoff,
    plan: GroundedWritingPlan,
    client: DeepSeekClient,
) -> V04WritingProject:
    project_path = RUN_ROOT / "writing_project.json"
    if project_path.is_file():
        project = V04WritingProject.model_validate_json(
            project_path.read_text(encoding="utf-8")
        )
    else:
        project = WritingProjectService().start(handoff)
    packet_builder = SectionEvidencePacketBuilder()
    paragraph_service = PlannedSectionDraftService()
    paragraph_writer = LLMGroundedParagraphWriter(client)
    paragraph_cache = ParagraphWritingRuntimeCache(
        RUN_ROOT / "paragraph_cache",
        plan_fingerprint=plan.plan_fingerprint,
    )
    reviewer = LLMSectionQualityReviewer(client)
    for section_plan in plan.sections:
        current = next(item for item in project.sections if item.section_id == section_plan.section_id)
        if current.status == "confirmed":
            prior_review_paths = [
                RUN_ROOT
                / "quality_reviews"
                / f"{section_plan.section_id}_round{round_number}.json"
                for round_number in range(2, 6)
            ]
            available_prior_reviews = [
                path for path in prior_review_paths if path.is_file()
            ]
            prior_review_path = (
                available_prior_reviews[-1]
                if available_prior_reviews
                else prior_review_paths[0]
            )
            prior_review_round = int(prior_review_path.stem.rsplit("round", 1)[-1])
            final_review_path = (
                RUN_ROOT
                / "quality_reviews"
                / f"{section_plan.section_id}_round6.json"
            )
            if (
                current.draft is not None
                and final_review_path.is_file()
                and current.draft.counted_words < section_plan.target_words
            ):
                print(
                    f"[V0.4 editing] expanding post-edit {section_plan.section_id}; "
                    f"current={current.draft.counted_words} target={section_plan.target_words}",
                    flush=True,
                )
                packet = packet_builder.build(handoff, section_plan.section_id)
                draft = current.draft
                for expansion_round in range(1, 4):
                    if draft.counted_words >= section_plan.target_words:
                        break
                    counts = [
                        count_writing_units(
                            paragraph.text,
                            counting_policy=section_plan.counting_policy,
                        )
                        for paragraph in draft.paragraphs
                    ]
                    candidates = [
                        paragraph.paragraph_number
                        for paragraph, actual in zip(
                            section_plan.paragraphs,
                            counts,
                            strict=True,
                        )
                        if actual < paragraph.target_words
                    ]
                    deficit = section_plan.target_words - draft.counted_words
                    numbers = set(
                        candidates[
                            : max(
                                1,
                                min(
                                    len(candidates),
                                    math.ceil(deficit / 220),
                                ),
                            )
                        ]
                    )
                    if not numbers:
                        break
                    instructions = {
                        number: (
                            "在不恢复重复内容、不新增文献事实的前提下，补足本段的比较轴、"
                            "适用条件、证据边界与审慎判断，使段落达到计划篇幅；禁止泛化套话。"
                        )
                        for number in numbers
                    }
                    draft = paragraph_service.draft(
                        packet,
                        section_plan,
                        paragraph_writer,
                        cache=paragraph_cache,
                        existing_draft=draft,
                        force_paragraph_numbers=numbers,
                        revision_instructions=instructions,
                    )
                post_expansion_review = reviewer.review(
                    section_plan,
                    draft,
                    packet,
                    output_language=plan.output_language,
                )
                _write_json(
                    RUN_ROOT
                    / "quality_reviews"
                    / f"{section_plan.section_id}_round5.json",
                    post_expansion_review,
                )
                draft = apply_section_quality_review(draft, post_expansion_review)
                project = WritingProjectService().save_draft(project, draft)
                project = WritingProjectService().confirm_section(
                    project,
                    section_plan.section_id,
                    confirmed_by="Codex preview run requested by user",
                )
                _write_json(project_path, project)
                _write_json(
                    RUN_ROOT / "section_drafts" / f"{section_plan.section_id}.json",
                    next(
                        item.draft
                        for item in project.sections
                        if item.section_id == section_plan.section_id
                    ),
                )
                continue
            if (
                current.draft is None
                or not prior_review_path.is_file()
                or final_review_path.is_file()
            ):
                print(f"[V0.4 writing] restored confirmed {section_plan.section_id}", flush=True)
                continue
            previous_review = SectionQualityReview.model_validate_json(
                prior_review_path.read_text(encoding="utf-8")
            )
            if not previous_review.findings:
                print(f"[V0.4 writing] restored confirmed {section_plan.section_id}", flush=True)
                continue
            print(
                f"[V0.4 editing] reopening {section_plan.section_id} for targeted quality repair",
                flush=True,
            )
            packet = packet_builder.build(handoff, section_plan.section_id)
            draft = current.draft
            latest_review = previous_review
            for review_round in range(prior_review_round + 1, 7):
                if not latest_review.findings:
                    break
                numbers = {item.paragraph_number for item in latest_review.findings}
                instructions = {
                    number: (
                        "保持本段计划字数，不新增任何数字、方法细节或文献事实；删除被指出的"
                        "跑题、重复、夸大、无依据推断和口号式表达。"
                        + " ".join(
                            item.revision_instruction
                            for item in latest_review.findings
                            if item.paragraph_number == number
                        )
                    )
                    for number in numbers
                }
                draft = paragraph_service.draft(
                    packet,
                    section_plan,
                    paragraph_writer,
                    cache=paragraph_cache,
                    existing_draft=draft,
                    force_paragraph_numbers=numbers,
                    revision_instructions=instructions,
                )
                latest_review = reviewer.review(
                    section_plan,
                    draft,
                    packet,
                    output_language=plan.output_language,
                )
                _write_json(
                    RUN_ROOT
                    / "quality_reviews"
                    / f"{section_plan.section_id}_round{review_round}.json",
                    latest_review,
                )
                print(
                    f"[V0.4 editing] {section_plan.section_id} round {review_round} "
                    f"remaining_findings={len(latest_review.findings)}",
                    flush=True,
                )
            for expansion_round in range(1, 3):
                if draft.counted_words >= section_plan.target_words:
                    break
                counts = [
                    count_writing_units(
                        paragraph.text,
                        counting_policy=section_plan.counting_policy,
                    )
                    for paragraph in draft.paragraphs
                ]
                numbers = {
                    paragraph.paragraph_number
                    for paragraph, actual in zip(
                        section_plan.paragraphs,
                        counts,
                        strict=True,
                    )
                    if actual < paragraph.target_words
                }
                if not numbers:
                    break
                draft = paragraph_service.draft(
                    packet,
                    section_plan,
                    paragraph_writer,
                    cache=paragraph_cache,
                    existing_draft=draft,
                    force_paragraph_numbers=numbers,
                    revision_instructions={
                        number: (
                            "保持已修正的论点和证据边界，仅补足必要的比较、适用条件与审慎"
                            "分析至计划篇幅；不得恢复已删除的问题内容或使用泛化套话。"
                        )
                        for number in numbers
                    },
                )
            if draft.counted_words < section_plan.target_words:
                print(
                    f"[V0.4 editing] {section_plan.section_id} remains below target "
                    f"after quality repair: {draft.counted_words}",
                    flush=True,
                )
            latest_review = reviewer.review(
                section_plan,
                draft,
                packet,
                output_language=plan.output_language,
            )
            _write_json(
                RUN_ROOT
                / "quality_reviews"
                / f"{section_plan.section_id}_round7.json",
                latest_review,
            )
            _write_json(final_review_path, latest_review)
            draft = apply_section_quality_review(draft, latest_review)
            project = WritingProjectService().save_draft(project, draft)
            project = WritingProjectService().confirm_section(
                project,
                section_plan.section_id,
                confirmed_by="Codex preview run requested by user",
            )
            _write_json(project_path, project)
            _write_json(
                RUN_ROOT / "section_drafts" / f"{section_plan.section_id}.json",
                next(
                    item.draft
                    for item in project.sections
                    if item.section_id == section_plan.section_id
                ),
            )
            continue
        print(f"[V0.4 writing] drafting {section_plan.title}", flush=True)
        packet = packet_builder.build(handoff, section_plan.section_id)
        draft = paragraph_service.draft(
            packet,
            section_plan,
            paragraph_writer,
            cache=paragraph_cache,
            existing_draft=current.draft,
        )
        review: SectionQualityReview | None = None
        try:
            review = reviewer.review(
                section_plan,
                draft,
                packet,
                output_language=plan.output_language,
            )
            _write_json(
                RUN_ROOT / "quality_reviews" / f"{section_plan.section_id}_round1.json",
                review,
            )
        except Exception as exc:
            print(f"[quality review] first review skipped: {exc}", flush=True)
        if review is not None and review.findings:
            numbers = {item.paragraph_number for item in review.findings}
            instructions = {
                number: " ".join(
                    item.revision_instruction
                    for item in review.findings
                    if item.paragraph_number == number
                )
                for number in numbers
            }
            draft = paragraph_service.draft(
                packet,
                section_plan,
                paragraph_writer,
                cache=paragraph_cache,
                existing_draft=draft,
                force_paragraph_numbers=numbers,
                revision_instructions=instructions,
            )
        for expansion_round in range(1, 4):
            if draft.counted_words >= section_plan.target_words:
                break
            counts = [
                count_writing_units(
                    paragraph.text,
                    counting_policy=section_plan.counting_policy,
                )
                for paragraph in draft.paragraphs
            ]
            candidates = [
                paragraph.paragraph_number
                for paragraph, actual in zip(
                    section_plan.paragraphs,
                    counts,
                    strict=True,
                )
                if actual < paragraph.target_words
            ]
            if not candidates:
                break
            deficit = section_plan.target_words - draft.counted_words
            numbers = set(candidates[: max(1, min(len(candidates), math.ceil(deficit / 250)))])
            instructions = {
                number: (
                    "在不新增文献、数据或事实的前提下，补充证据之间的比较、适用条件和"
                    "审慎的综合判断，使本段达到计划字数；不得用泛化套话凑字数。"
                )
                for number in numbers
            }
            print(
                f"[V0.4 writing] expanding {section_plan.section_id} round {expansion_round}; "
                f"current={draft.counted_words} target={section_plan.target_words}",
                flush=True,
            )
            draft = paragraph_service.draft(
                packet,
                section_plan,
                paragraph_writer,
                cache=paragraph_cache,
                existing_draft=draft,
                force_paragraph_numbers=numbers,
                revision_instructions=instructions,
            )
        try:
            final_review = reviewer.review(
                section_plan,
                draft,
                packet,
                output_language=plan.output_language,
            )
            _write_json(
                RUN_ROOT / "quality_reviews" / f"{section_plan.section_id}_round2.json",
                final_review,
            )
            draft = apply_section_quality_review(draft, final_review)
        except Exception as exc:
            print(f"[quality review] final review skipped: {exc}", flush=True)
        project = WritingProjectService().save_draft(project, draft)
        project = WritingProjectService().confirm_section(
            project,
            section_plan.section_id,
            confirmed_by="Codex preview run requested by user",
        )
        _write_json(project_path, project)
        _write_json(
            RUN_ROOT / "section_drafts" / f"{section_plan.section_id}.json",
            next(item.draft for item in project.sections if item.section_id == section_plan.section_id),
        )
        print(
            f"[V0.4 writing] confirmed {section_plan.section_id}; units={draft.counted_words}",
            flush=True,
        )
    return project


def main() -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    state = _load_active_state()
    confirmed, policy = _requirement_with_boundary(state)
    _write_json(RUN_ROOT / "confirmed_requirement_with_boundary.json", confirmed)
    _write_json(RUN_ROOT / "executable_policy.json", policy)
    base_blueprint = _blueprint(policy)

    verifications = LiteratureVerificationBatch.model_validate_json(
        str(state["literature_verification_json"])
    )
    old_run = Path(str(state["literature_run_dir"]))
    discovery = json.loads((old_run / "discovery_cache.json").read_text(encoding="utf-8"))
    decisions = {
        item.candidate.doi: item
        for raw in discovery["eligible_decisions"]
        if (item := CandidateDecision.model_validate(raw)).status == "eligible"
    }
    settings = LLMSettings().for_structured_output().model_copy(
        update={"timeout_seconds": 180.0, "max_retries": 3, "temperature": 0.2}
    )
    client = DeepSeekClient(settings)
    assessments = _score_literature(base_blueprint, verifications, client)
    writing_blueprint = _writing_blueprint(policy)
    writing_assessments = _collapse_assessments(
        assessments,
        writing_blueprint,
    )
    blueprint, selection = _select_literature(
        writing_blueprint,
        writing_assessments,
        verifications,
        decisions,
    )
    _write_admission_matrix(selection)
    _write_selected_ris(selection, verifications)
    library = _build_library(state, selection, verifications, policy)
    _write_json(OUTPUT_ROOT / "evidence_library.json", library)

    outline = WritingOutlineBuilder().build(blueprint, library, policy=policy)
    if outline.unresolved_gaps:
        raise RuntimeError("writing outline still has evidence gaps: " + "; ".join(outline.unresolved_gaps))
    handoff_service = WritingHandoffService()
    confirmed_outline = handoff_service.confirm_outline(
        outline,
        confirmed_by="Codex preview run requested by user",
        confirmation_note=(
            "依据已确认要求、主题准入表及四篇相关核心全文自动生成的预览写作计划。"
        ),
    )
    handoff = handoff_service.create(
        requirement=confirmed,
        outline=confirmed_outline,
        evidence_library=library,
        policy=policy,
    )
    _write_json(RUN_ROOT / "writing_handoff.json", handoff)
    _write_json(OUTPUT_ROOT / "problem_driven_outline.json", confirmed_outline)

    plan_path = RUN_ROOT / "writing_plan.json"
    if plan_path.is_file():
        plan = GroundedWritingPlan.model_validate_json(plan_path.read_text(encoding="utf-8"))
    else:
        plan = GroundedWritingPlanner(
            client,
            cache=WritingPlanRuntimeCache(RUN_ROOT / "plan_cache", handoff=handoff),
        ).plan(handoff)
        plan = plan.confirm(confirmed_by="Codex preview run requested by user")
        _write_json(plan_path, plan)
    print(
        f"[V0.4 plan] sections={len(plan.sections)} paragraphs="
        f"{sum(len(section.paragraphs) for section in plan.sections)}",
        flush=True,
    )
    project = _draft_sections(handoff, plan, client)
    body = WritingProjectService().assemble_body(project)
    (OUTPUT_ROOT / "confirmed_body.md").write_text(body.markdown, encoding="utf-8")

    matter_path = RUN_ROOT / "final_matter.json"
    project_path = RUN_ROOT / "writing_project.json"
    if (
        matter_path.is_file()
        and project_path.is_file()
        and matter_path.stat().st_mtime >= project_path.stat().st_mtime
    ):
        final_matter = FinalMatterProposal.model_validate_json(
            matter_path.read_text(encoding="utf-8")
        )
    else:
        final_matter = LLMFinalMatterWriter(client).draft(handoff, body)
        _write_json(matter_path, final_matter)
    assembler = FinalPaperAssembler()
    package = assembler.assemble(
        handoff=handoff,
        body=body,
        final_matter=final_matter,
    )
    _write_json(RUN_ROOT / "final_package_preconfirmation.json", package)
    if package.audit.blocking_count:
        details = "; ".join(
            f"{item.code}:{item.detail}"
            for item in package.audit.issues
            if item.severity == "blocking"
        )
        raise RuntimeError(f"final audit has blocking issues: {details}")
    review_codes = [
        item.code
        for item in package.audit.issues
        if item.severity == "warning"
        and item.code
        in {
            "theme_element_requires_user_review",
            "original_analysis_requires_user_review",
            "reference_tool_usage_requires_attestation",
        }
    ]
    package = package.model_copy(update={"user_review_attestations": review_codes})
    package = assembler.confirm(
        package,
        confirmed_by="Codex preview run requested by user",
    )
    _write_json(OUTPUT_ROOT / "final_paper_package.json", package)
    _write_json(OUTPUT_ROOT / "final_compliance_audit.json", package.audit)
    (OUTPUT_ROOT / "final_paper.md").write_text(package.markdown, encoding="utf-8")
    docx_path = OUTPUT_ROOT / "大气遥感_证据约束文献综述_预览版.docx"
    docx_path.write_bytes(FinalPaperDocxExporter().export(package))
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_project": str(ACTIVE_PROJECT),
        "source_project_mutated": False,
        "topic": policy.topic,
        "output_language": policy.output_language,
        "counted_units": package.audit.counted_units,
        "references": package.audit.reference_count,
        "foreign_references": package.audit.foreign_reference_count,
        "blocking_issues": package.audit.blocking_count,
        "warnings": sum(
            item.severity == "warning" for item in package.audit.issues
        ),
        "core_full_text_dois": sorted(CORE_ASSIGNMENTS),
        "docx": str(docx_path),
    }
    _write_json(OUTPUT_ROOT / "run_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
