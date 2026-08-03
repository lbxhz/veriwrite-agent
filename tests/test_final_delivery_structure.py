import pytest

from veriwrite_agent.models.final_delivery import (
    FinalMatterProposal,
    FinalPaperAudit,
    FinalPaperAuditIssue,
    FinalPaperPackage,
)
from veriwrite_agent.services.final_delivery import (
    FinalDeliveryError,
    FinalPaperAssembler,
    _assemble_markdown,
    _required_final_matter_fields,
    _section_present,
)


LONG_TEXT = (
    "这是一段仅根据已确认正文进行归纳的结构性文字，用于验证最终论文的必需章节能够"
    "由确定性程序稳定组装，不增加新的文献、数据、方法或结论。所有概括都必须保留原有论证边界，"
    "不得将背景文献改写为细节证据，也不得将审计尚未确认的内容写入最终结论。"
)


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
