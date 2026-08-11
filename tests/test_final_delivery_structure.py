import json

import pytest

from veriwrite_agent.llm.fake_client import FakeLLMClient
from veriwrite_agent.models.final_delivery import (
    FinalMatterProposal,
    FinalPaperAudit,
    FinalPaperAuditIssue,
    FinalPaperPackage,
)
from veriwrite_agent.models.writing import BodyDraftPackage
from veriwrite_agent.models.writing_quality import (
    ManuscriptQualityFinding,
    ManuscriptQualityReview,
)
from veriwrite_agent.services.final_delivery import (
    FinalDeliveryError,
    FinalPaperAssembler,
    LLMFinalMatterWriter,
    _assemble_markdown,
    _has_introduction_roadmap,
    _manuscript_finding_audit_severity,
    _required_final_matter_fields,
    _section_present,
    _validate_final_matter_editorial_quality,
)
from veriwrite_agent.services.writing_quality import _merge_manuscript_findings


LONG_TEXT = (
    "这是一段仅根据已确认正文进行归纳的结构性文字，用于验证最终论文的必需章节能够"
    "由确定性程序稳定组装，不增加新的文献、数据、方法或结论。所有概括都必须保留原有论证边界，"
    "不得将背景文献改写为细节证据，也不得将审计尚未确认的内容写入最终结论。"
)


def test_introduction_requires_an_explicit_argument_roadmap() -> None:
    no_roadmap = (
        "大气遥感是认识大气状态与变化的重要手段。本文综述相关研究的意义、"
        "范围与核心问题，并关注观测能力和反演方法的协同发展。"
    )
    roadmap = (
        "大气遥感是认识大气状态与变化的重要手段。下文依次讨论数据获取、"
        "参数反演与智能处理，随后综合尚未解决的问题，最后归纳发展方向。"
    )

    assert not _has_introduction_roadmap(no_roadmap, output_language="Chinese")
    assert _has_introduction_roadmap(roadmap, output_language="Chinese")


def test_final_assembler_materializes_required_structure_and_aliases() -> None:
    matter = FinalMatterProposal(
        title="大气遥感研究综述",
        abstract=LONG_TEXT,
        keywords=["大气遥感", "人工智能", "反演"],
        introduction=LONG_TEXT,
        current_status_analysis=LONG_TEXT,
        problems=LONG_TEXT,
        conclusion=LONG_TEXT,
    )
    body = "## 大气遥感技术发展趋势\n\n" + LONG_TEXT
    required_sections = [
        "引言",
        "国内外研究现状",
        "现状分析",
        "存在问题",
        "结语",
        "存在问题和技术发展趋势",
    ]

    markdown = _assemble_markdown(
        matter,
        body,
        [],
        ai_declaration=None,
        output_language="Chinese",
        required_sections=required_sections,
    )

    assert "## 引言" in markdown
    assert "## 国内外研究现状" in markdown
    assert "### 大气遥感技术发展趋势" in markdown
    assert "## 现状分析" in markdown
    assert "## 存在问题" in markdown
    assert "## 结语" in markdown
    assert all(_section_present(section, markdown) for section in required_sections)


def test_final_matter_only_requires_missing_structural_syntheses() -> None:
    body = "## 大气遥感技术发展趋势\n\n" + LONG_TEXT

    fields = _required_final_matter_fields(
        ["引言", "现状分析", "存在问题和技术发展趋势"],
        body,
    )

    assert fields == ["introduction", "current_status_analysis", "problems"]


def test_global_editor_rejects_repeated_roles_and_single_oversized_status_block() -> None:
    repeated = "大气遥感数据获取技术持续发展，星载观测与地基验证共同提升了监测能力。" * 12
    matter = FinalMatterProposal(
        title="大气遥感研究综述",
        abstract=repeated,
        keywords=["大气遥感", "人工智能", "反演"],
        introduction=repeated,
        current_status_analysis=(
            "本文提出了一种新的遥感反演方法，并获得了95%的准确率。" * 60
        ),
        conclusion=LONG_TEXT,
    )

    with pytest.raises(FinalDeliveryError, match="current_status_analysis|overlap|repeats"):
        _validate_final_matter_editorial_quality(
            matter,
            "## 数据获取技术进展\n\n" + repeated,
            output_language="Chinese",
        )


def test_current_status_editor_repairs_one_field_as_three_paragraphs() -> None:
    abstract = (
        "本综述界定大气遥感的研究范围，归纳观测、反演与数据处理的主要进展，"
        "并综合讨论技术能力、证据边界与业务应用之间的关系。现有研究显示多源观测"
        "和智能方法提升了信息获取能力，但数据代表性、物理一致性与跨区域泛化仍是"
        "主要挑战。"
    ) * 3
    proposal = FinalMatterProposal(
        title="大气遥感研究综述",
        abstract=abstract,
        keywords=["大气遥感", "人工智能", "反演"],
        introduction=(
            "大气环境监测需要连续、可靠且可比较的观测。本文据此说明研究问题、"
            "综述范围和章节安排，并明确各技术章节之间的论证关系。"
        )
        * 2,
        current_status_analysis="原来的单段现状分析包含95%的准确率，并重复罗列了多项具体方法与结果。" * 3,
        conclusion=(
            "综上，大气遥感能力的提升需要观测体系、反演方法和质量控制协同推进，"
            "并在统一证据边界下开展可比较验证。"
        )
        * 2,
    )
    paragraphs = [
        "当前技术能力的主要提升体现在观测连续性、多源协同和数据处理自动化方面，研究重心已由单一数据获取逐步转向观测、反演和应用的协同。",
        "尚未解决的问题集中在训练样本代表性、复杂条件下的泛化能力、物理一致性和不确定性表达，这些限制使研究结果距离稳定业务应用仍有差距。",
        "现有正文不足以形成可靠的国内外定量比较，因此不宜人为概括地域差距；下一步应加强可比较评测、多源数据融合和物理约束下的方法验证。",
    ]
    client = FakeLLMClient(json.dumps({"paragraphs": paragraphs}, ensure_ascii=False))
    body = BodyDraftPackage.model_construct(
        markdown="## 技术进展\n\n观测体系与数据处理方法共同推动了大气遥感能力发展。"
    )

    repaired = LLMFinalMatterWriter(client)._repair_current_status_analysis(
        proposal,
        body,
        output_language="Chinese",
    )

    assert repaired.current_status_analysis is not None
    assert repaired.current_status_analysis.split("\n\n") == paragraphs
    assert len(client.calls) == 1


def test_manuscript_review_cannot_invent_false_self_attribution_blocker() -> None:
    llm_finding = ManuscriptQualityFinding(
        section_id="theme3",
        paragraph_number=3,
        code="false_self_attribution",
        severity="blocking",
        detail="The reviewer inferred ownership without an explicit first-person claim.",
        revision_instruction="Attribute the source explicitly.",
    )
    review = ManuscriptQualityReview(findings=[llm_finding])

    merged = _merge_manuscript_findings(review, [])

    assert merged.findings == []


def test_manuscript_editor_targeted_repair_enters_blocking_repair_route() -> None:
    actionable = ManuscriptQualityFinding(
        section_id="theme1",
        paragraph_number=2,
        code="cross_section_repetition",
        severity="warning",
        disposition="targeted_repair",
        detail="The paragraph substantially repeats the preceding chapter.",
        revision_instruction="Keep only the comparison unique to this chapter.",
    )
    advisory = actionable.model_copy(update={"disposition": "report_only"})

    assert _manuscript_finding_audit_severity(actionable) == "blocking"
    assert _manuscript_finding_audit_severity(advisory) == "warning"


def test_semantic_review_warning_requires_one_recorded_attestation() -> None:
    issue = FinalPaperAuditIssue(
        code="original_analysis_requires_user_review",
        severity="warning",
        requirement_path="structure.must_include_original_analysis",
        detail="User review required.",
    )
    audit = FinalPaperAudit(
        policy_fingerprint="a" * 64,
        counted_units=100,
        reference_count=1,
        foreign_reference_count=1,
        issues=[issue],
    )
    package = FinalPaperPackage.model_construct(
        status="ready_for_confirmation",
        audit=audit,
        user_review_attestations=[],
        confirmed_by=None,
        confirmed_at=None,
    )

    with pytest.raises(FinalDeliveryError, match="semantic user review"):
        FinalPaperAssembler().confirm(package, confirmed_by="student")

    attested = package.model_copy(
        update={
            "user_review_attestations": [
                "original_analysis_requires_user_review"
            ]
        }
    )
    confirmed = FinalPaperAssembler().confirm(attested, confirmed_by="student")

    assert confirmed.status == "confirmed"
