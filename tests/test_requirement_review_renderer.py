from veriwrite_agent.models.requirement_workflow import (
    CompletenessReport,
    ParserRun,
    ReconciliationResult,
    RequirementReviewPackage,
)
from veriwrite_agent.models.requirements import RequirementSpec
from veriwrite_agent.services.requirement_review_renderer import (
    RequirementReviewRenderer,
)


def test_renders_user_facing_confirmation_form() -> None:
    spec = RequirementSpec(document_type="review", topic="遥感综述")
    review = RequirementReviewPackage(
        parser_mode="rule_only",
        rule_run=ParserRun(
            parser_name="rule_based",
            status="succeeded",
            spec=spec,
        ),
        reconciliation=ReconciliationResult(merged_spec=spec),
        completeness=CompletenessReport(),
        status="ready_for_confirmation",
    )

    markdown = RequirementReviewRenderer().render_markdown(review)

    assert "# 课程要求确认单" in markdown
    assert "| topic | \"遥感综述\" |" in markdown
    assert "## 候选结果对照" in markdown
    assert "本次使用 rule-only 模式" in markdown
    assert "没有待处理事项" in markdown
