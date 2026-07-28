from pathlib import Path

import pytest

from veriwrite_agent.services.requirement_parser import RuleBasedRequirementParser


@pytest.fixture
def parsed_spec():
    fixture = Path(__file__).parent / "fixtures" / "course_requirement.txt"
    return RuleBasedRequirementParser().parse(fixture.read_text(encoding="utf-8"))


def test_parses_hard_length_and_reference_constraints(parsed_spec) -> None:
    assert parsed_spec.length.minimum_chars == 15000
    assert parsed_spec.length.figures_excluded is True
    assert parsed_spec.references.minimum_total == 60
    assert parsed_spec.references.minimum_foreign_ratio == pytest.approx(1 / 3)
    assert parsed_spec.references.minimum_foreign_count == 20


def test_parses_citation_and_recency_rules(parsed_spec) -> None:
    assert parsed_spec.references.recent_year_window == 5
    assert parsed_spec.references.recent_year_rule_strength == "soft_preference"
    assert parsed_spec.references.citation_order == "first_appearance"
    assert parsed_spec.references.in_text_style == "numeric_superscript"
    assert parsed_spec.references.max_references_per_citation_cluster == 4


def test_detects_ambiguity_and_conditional_requirement(parsed_spec) -> None:
    assert parsed_spec.ambiguities
    assert "学院审核未通过后再次提交时必须提供修改说明" in (
        parsed_spec.workflow_conditions
    )


def test_template_example_is_not_treated_as_user_topic(parsed_spec) -> None:
    assert parsed_spec.topic is None
    assert parsed_spec.topic_source == "user_confirmation_required"


def test_keeps_source_evidence_for_audit(parsed_spec) -> None:
    evidence_fields = {item.field for item in parsed_spec.source_evidence}
    assert "length" in evidence_fields
    assert "references.minimum_total" in evidence_fields
    assert "references.minimum_foreign_ratio" in evidence_fields


def test_separates_institution_from_school() -> None:
    result = RuleBasedRequirementParser().parse(
        "中国地质大学未来技术学院要求提交研究方向文献综述。"
    )

    assert result.institution == "中国地质大学"
    assert result.school_or_department == "未来技术学院"


def test_detects_bracketed_review_instructions_as_a_deliverable() -> None:
    result = RuleBasedRequirementParser().parse(
        "基本格式：《研究方向文献综述》说明+课程论文封面。"
    )

    assert "研究方向文献综述说明" in result.deliverables


@pytest.mark.parametrize(
    ("fixture_name", "minimum_chars", "minimum_references", "topic"),
    [
        (
            "foreign_ratio_review.txt",
            8000,
            30,
            "人工智能辅助地质灾害识别综述",
        ),
        (
            "specified_sections_review.txt",
            12000,
            40,
            "多源遥感数据融合方法比较",
        ),
        (
            "figures_and_format_review.txt",
            10000,
            35,
            "遥感图像变化检测技术",
        ),
        ("ambiguous_length_review.txt", 12000, 25, None),
    ],
)
def test_parses_additional_gold_fixtures(
    fixture_name: str,
    minimum_chars: int,
    minimum_references: int,
    topic: str | None,
) -> None:
    fixture = Path(__file__).parent / "fixtures" / fixture_name

    result = RuleBasedRequirementParser().parse(
        fixture.read_text(encoding="utf-8")
    )

    assert result.length.minimum_chars == minimum_chars
    assert result.references.minimum_total == minimum_references
    assert result.topic == topic
