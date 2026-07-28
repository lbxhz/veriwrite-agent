import pytest
from pydantic import ValidationError

from veriwrite_agent.models.requirements import (
    LengthRequirement,
    ReferenceRequirement,
    RequirementSpec,
)


def test_foreign_reference_count_rounds_up() -> None:
    requirement = ReferenceRequirement(minimum_total=61, minimum_foreign_ratio=1 / 3)
    assert requirement.minimum_foreign_count == 21


def test_invalid_foreign_ratio_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ReferenceRequirement(minimum_total=60, minimum_foreign_ratio=1.2)


def test_target_cannot_be_below_minimum() -> None:
    with pytest.raises(ValidationError):
        LengthRequirement(minimum_chars=15000, target_chars=12000)


def test_normalizes_known_source_type_aliases() -> None:
    requirement = ReferenceRequirement(
        preferred_source_types=["重要学术期刊论文", "book", "dissertation"],
        discouraged_source_types=["会议论文", "technical_report"],
    )

    assert requirement.preferred_source_types == [
        "journal_article",
        "book",
        "thesis",
    ]
    assert requirement.discouraged_source_types == [
        "conference_paper",
        "technical_report",
    ]


def test_normalizes_known_document_type_alias() -> None:
    spec = RequirementSpec(document_type="文献综述")

    assert spec.document_type == "research_direction_literature_review"


def test_normalizes_combined_theme_and_deliverable_phrases() -> None:
    spec = RequirementSpec(
        document_type="review",
        required_theme_elements=[
            "人工智能、新一代信息技术与专业领域的交叉融合",
            "多学科交叉特色",
        ],
        deliverables=[
            "课程论文封面",
            "文献综述正文和参考文献",
        ],
    )

    assert spec.required_theme_elements == [
        "人工智能",
        "新一代信息技术",
        "专业领域",
        "多学科交叉",
    ]
    assert spec.deliverables == [
        "课程论文封面",
        "文献综述正文",
        "参考文献",
    ]
