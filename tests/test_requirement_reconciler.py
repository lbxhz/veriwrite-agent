from veriwrite_agent.models.requirements import (
    LengthRequirement,
    ReferenceRequirement,
    RequirementSpec,
    SourceEvidence,
)
from veriwrite_agent.services.requirement_reconciler import RequirementReconciler


def test_reconciler_fills_a_value_missing_from_rule_parser() -> None:
    rule_spec = RequirementSpec(document_type="review")
    llm_spec = RequirementSpec(document_type="review", topic="遥感变化检测")

    result = RequirementReconciler().reconcile(rule_spec, llm_spec)

    assert result.merged_spec.topic == "遥感变化检测"
    assert result.conflicts == []


def test_reconciler_keeps_rule_value_and_records_disagreement() -> None:
    rule_spec = RequirementSpec(
        document_type="review",
        length=LengthRequirement(minimum_chars=15000),
    )
    llm_spec = RequirementSpec(
        document_type="review",
        length=LengthRequirement(minimum_chars=12000),
    )

    result = RequirementReconciler().reconcile(rule_spec, llm_spec)

    assert result.merged_spec.length.minimum_chars == 15000
    assert [conflict.field for conflict in result.conflicts] == ["length.minimum_chars"]


def test_reconciler_rejects_a_fill_that_breaks_cross_field_validation() -> None:
    rule_spec = RequirementSpec(
        document_type="review",
        length=LengthRequirement(minimum_chars=15000),
    )
    llm_spec = RequirementSpec(
        document_type="review",
        length=LengthRequirement(target_chars=12000),
    )

    result = RequirementReconciler().reconcile(rule_spec, llm_spec)

    assert result.merged_spec.length.target_chars is None
    assert result.conflicts[0].field == "length.target_chars"
    assert "data contract" in result.conflicts[0].reason


def test_reconciler_unions_audit_evidence_without_duplicates() -> None:
    shared = SourceEvidence(field="topic", source_text="研究主题：遥感")
    extra = SourceEvidence(field="length", source_text="不少于15000字")
    rule_spec = RequirementSpec(
        document_type="review",
        source_evidence=[shared],
    )
    llm_spec = RequirementSpec(
        document_type="review",
        source_evidence=[shared, extra],
    )

    result = RequirementReconciler().reconcile(rule_spec, llm_spec)

    assert result.merged_spec.source_evidence == [shared, extra]


def test_reconciler_treats_rounded_ratios_as_equivalent() -> None:
    rule_spec = RequirementSpec.model_validate(
        {
            "document_type": "review",
            "references": {"minimum_foreign_ratio": 1 / 3},
        }
    )
    llm_spec = RequirementSpec.model_validate(
        {
            "document_type": "review",
            "references": {"minimum_foreign_ratio": 0.3333},
        }
    )

    result = RequirementReconciler().reconcile(rule_spec, llm_spec)

    assert result.conflicts == []
    assert result.merged_spec.references.minimum_foreign_ratio == 1 / 3


def test_reconciler_treats_set_like_lists_as_order_independent() -> None:
    rule_spec = RequirementSpec(
        document_type="review",
        deliverables=["课程论文封面", "参考文献"],
    )
    llm_spec = RequirementSpec(
        document_type="review",
        deliverables=["参考文献", "课程论文封面"],
    )

    result = RequirementReconciler().reconcile(rule_spec, llm_spec)

    assert result.conflicts == []
    assert result.merged_spec.deliverables == ["课程论文封面", "参考文献"]


def test_reconciler_keeps_a_more_complete_compatible_style() -> None:
    rule_spec = RequirementSpec(
        document_type="review",
        references=ReferenceRequirement(
            bibliography_style=("Remote Sensing of Environment 期刊格式 或 IJGIS 期刊格式")
        ),
    )
    llm_spec = RequirementSpec(
        document_type="review",
        references=ReferenceRequirement(bibliography_style="Remote Sensing of Environment"),
    )

    result = RequirementReconciler().reconcile(rule_spec, llm_spec)

    assert result.conflicts == []
    assert "IJGIS" in result.merged_spec.references.bibliography_style
