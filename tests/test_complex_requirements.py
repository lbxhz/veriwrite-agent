from pathlib import Path

from veriwrite_agent.models.requirement_workflow import RequirementConfirmation
from veriwrite_agent.models.requirements import (
    AIUsagePolicy,
    RequirementProfile,
    RequirementSpec,
    SubmissionRequirement,
)
from veriwrite_agent.services.requirement_confirmation import (
    RequirementConfirmationService,
)
from veriwrite_agent.services.requirement_input import extract_requirement_texts
from veriwrite_agent.services.requirement_parser import RuleBasedRequirementParser
from veriwrite_agent.services.requirement_pipeline import RequirementReviewPipeline
from veriwrite_agent.services.requirement_reconciler import RequirementReconciler


FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "complex_multi_profile_review.txt"
)


def test_complex_case_separates_profiles_from_shared_requirements() -> None:
    spec = RuleBasedRequirementParser().parse(
        FIXTURE.read_text(encoding="utf-8")
    )

    assert [profile.teacher for profile in spec.profiles] == [
        "李玮",
        "王轶",
        "解清华",
        "姚尧",
    ]
    assert all(
        profile.references.target_total == 30
        and profile.references.target_is_approximate
        for profile in spec.profiles
    )
    assert spec.profiles[2].topic.startswith("SAR/InSAR")
    assert len(spec.profiles[3].policy_rules) == 4

    assert spec.length.minimum_words == 4000
    assert spec.length.maximum_words == 5000
    assert set(spec.length.excluded_components) == {
        "参考文献",
        "摘要",
        "封面",
        "AI声明",
    }
    assert spec.references.required_management_tools == [
        "Mendeley",
        "Endnote",
    ]


def test_complex_case_keeps_selection_submission_and_ai_rules() -> None:
    spec = RuleBasedRequirementParser().parse(
        FIXTURE.read_text(encoding="utf-8")
    )

    assert spec.selection_policy.options_total == 4
    assert spec.selection_policy.required_choices == 1
    assert spec.selection_policy.fallback_teacher == "姚尧"
    assert spec.submission.required_media == ["paper", "electronic"]
    assert spec.submission.deadline_month == 11
    assert spec.submission.deadline_day == 30
    assert spec.submission.deadline_hour == 21
    assert spec.ai_policy.permitted_uses == ["翻译", "润色"]
    assert spec.ai_policy.declaration_required is True
    assert spec.ai_policy.no_ai_statement == (
        "No AI tools were used in the writing of this report."
    )
    assert spec.policy_rules[0].category == "attendance"


def test_selected_profile_is_materialized_for_v02_handoff() -> None:
    review = RequirementReviewPipeline(
        RuleBasedRequirementParser()
    ).prepare(FIXTURE.read_text(encoding="utf-8"))

    issue_ids = {issue.issue_id for issue in review.completeness.issues}
    assert "missing_selected_profile" in issue_ids
    assert "missing_minimum_chars" not in issue_ids
    assert "missing_minimum_references" not in issue_ids

    confirmed = RequirementConfirmationService().confirm(
        review,
        RequirementConfirmation(
            confirmed_by="student",
            field_updates={"selected_profile_id": "option_4"},
            acknowledged_issue_ids=["missing_deadline_year"],
        ),
    )

    assert confirmed.requirement.selected_profile_id == "option_4"
    assert confirmed.requirement.topic.startswith("地理智能 GeoAI")
    assert confirmed.requirement.references.target_total == 30
    assert len(confirmed.requirement.references.restriction_rules) == 2
    assert any(
        rule.category == "academic_integrity"
        for rule in confirmed.requirement.policy_rules
    )


def test_ordered_text_files_remove_long_scroll_overlap(tmp_path) -> None:
    first = tmp_path / "01.txt"
    second = tmp_path / "02.txt"
    overlap = (
        "解清华老师：撰写一篇英文综述，利用文献检索方法从常用数据库"
        "查找相关文献，并采用Mendeley或Endnote管理参考文献。"
    )
    first.write_text(f"李玮老师：第一部分完整内容。\n{overlap}\n遮挡乱码", encoding="utf-8")
    second.write_text(f"{overlap}\n姚尧老师：第二部分完整内容。", encoding="utf-8")

    result = extract_requirement_texts([first, second])

    assert result.source_count == 2
    assert result.text.count("解清华老师") == 1
    assert "遮挡乱码" not in result.text
    assert "姚尧老师" in result.text
    assert any("去除了 1 处" in warning for warning in result.warnings)


def test_reconciler_normalizes_profile_ids_topics_and_consequence_wording() -> None:
    rule = RequirementSpec(
        document_type="review",
        profiles=[
            RequirementProfile(
                profile_id="option_1",
                teacher="姚尧",
                topic="地理智能 GeoAI 在 GIS 及相关方向中的发展与应用",
            )
        ],
        submission=SubmissionRequirement(deadline_hour=21),
        ai_policy=AIUsagePolicy(
            violation_consequence="违规使用 AI 撰写报告，0分",
            missing_declaration_consequence="无 AI 声明作业不及格",
        ),
    )
    llm = RequirementSpec(
        document_type="review",
        profiles=[
            RequirementProfile(
                profile_id="yaoyao",
                teacher="姚尧",
                topic=(
                    "介绍地理智能 GeoAI 在 GIS 中的最新研究方向、"
                    "方法和应用场景，并总结不足。"
                ),
            )
        ],
        submission=SubmissionRequirement(deadline_hour=21),
        ai_policy=AIUsagePolicy(
            violation_consequence="0分处理",
            missing_declaration_consequence="不及格处理",
        ),
    )

    result = RequirementReconciler().reconcile(rule, llm)

    assert result.conflicts == []
    assert result.merged_spec.profiles[0].profile_id == "option_1"
