from pathlib import Path

import pytest

from veriwrite_agent.llm.fake_client import FakeLLMClient
from veriwrite_agent.models.requirement_workflow import RequirementConfirmation
from veriwrite_agent.models.requirements import LengthRequirement, RequirementSpec
from veriwrite_agent.services.llm_requirement_parser import LLMRequirementParser
from veriwrite_agent.services.requirement_confirmation import (
    RequirementConfirmationError,
    RequirementConfirmationService,
)
from veriwrite_agent.services.requirement_parser import RuleBasedRequirementParser
from veriwrite_agent.services.requirement_pipeline import RequirementReviewPipeline


@pytest.fixture
def requirement_text() -> str:
    fixture = Path(__file__).parent / "fixtures" / "course_requirement.txt"
    return fixture.read_text(encoding="utf-8")


def test_dual_pipeline_keeps_both_candidates_and_builds_review(
    requirement_text: str,
) -> None:
    llm_spec = RuleBasedRequirementParser().parse(requirement_text).model_copy(
        update={"topic": "人工智能与遥感交叉研究"}
    )
    llm_parser = LLMRequirementParser(FakeLLMClient(llm_spec.model_dump_json()))

    review = RequirementReviewPipeline(
        RuleBasedRequirementParser(),
        llm_parser=llm_parser,
    ).prepare(requirement_text)

    assert review.parser_mode == "dual"
    assert review.rule_run.status == "succeeded"
    assert review.llm_run is not None
    assert review.llm_run.status == "succeeded"
    assert review.reconciliation.merged_spec.topic == "人工智能与遥感交叉研究"
    assert review.status == "ready_for_confirmation"


def test_pipeline_preserves_rule_result_when_llm_fails(
    requirement_text: str,
) -> None:
    llm_parser = LLMRequirementParser(FakeLLMClient('{"document_type": 123}'))

    review = RequirementReviewPipeline(
        RuleBasedRequirementParser(),
        llm_parser=llm_parser,
    ).prepare(requirement_text)

    assert review.llm_run is not None
    assert review.llm_run.status == "failed"
    assert review.reconciliation.merged_spec.length.minimum_chars == 15000
    assert "parser_failed:llm" in {
        issue.issue_id for issue in review.completeness.issues
    }


def test_user_confirmation_produces_v02_handoff(
    requirement_text: str,
) -> None:
    review = RequirementReviewPipeline(
        RuleBasedRequirementParser()
    ).prepare(requirement_text)
    confirmation = RequirementConfirmation(
        confirmed_by="student",
        field_updates={
            "topic": "人工智能驱动的遥感变化检测",
            "references.bibliography_style": "GB/T 7714-2015",
            "length.counting_policy": "chinese_chars_and_english_words",
        },
        acknowledged_issue_ids=["source_ambiguity:0"],
    )

    result = RequirementConfirmationService().confirm(review, confirmation)

    assert result.status == "confirmed"
    assert result.requirement.topic == "人工智能驱动的遥感变化检测"
    assert result.requirement.topic_source == "explicit"
    assert result.confirmed_by == "student"
    assert result.remaining_warnings[0].issue_id == "source_ambiguity:0"
    assert result.requirement.source_evidence[-1].source_text == "用户确认：student"


def test_confirmation_rejects_unknown_fields(requirement_text: str) -> None:
    review = RequirementReviewPipeline(
        RuleBasedRequirementParser()
    ).prepare(requirement_text)
    confirmation = RequirementConfirmation(
        field_updates={"not_a_real_field": "value"}
    )

    with pytest.raises(RequirementConfirmationError, match="未知字段"):
        RequirementConfirmationService().confirm(review, confirmation)


def test_confirmation_cannot_skip_blocking_topic(requirement_text: str) -> None:
    review = RequirementReviewPipeline(
        RuleBasedRequirementParser()
    ).prepare(requirement_text)
    confirmation = RequirementConfirmation(
        acknowledged_issue_ids=[
            issue.issue_id for issue in review.completeness.issues
        ]
    )

    with pytest.raises(RequirementConfirmationError, match="missing_topic"):
        RequirementConfirmationService().confirm(review, confirmation)


def test_dual_pipeline_records_a_real_conflict() -> None:
    text = "课程论文15000字以上。研究主题：遥感变化检测。"
    llm_spec = RequirementSpec(
        document_type="research_direction_literature_review",
        topic="遥感变化检测",
        topic_source="explicit",
        length=LengthRequirement(minimum_chars=12000),
    )
    llm_parser = LLMRequirementParser(FakeLLMClient(llm_spec.model_dump_json()))

    review = RequirementReviewPipeline(
        RuleBasedRequirementParser(),
        llm_parser=llm_parser,
    ).prepare(text)

    assert review.status == "needs_resolution"
    assert "length.minimum_chars" in {
        conflict.field for conflict in review.reconciliation.conflicts
    }
