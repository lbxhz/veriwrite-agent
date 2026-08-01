from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from veriwrite_agent.llm.fake_client import FakeLLMClient
from veriwrite_agent.models.requirement_workflow import RequirementConflict
from veriwrite_agent.services.llm_requirement_parser import LLMRequirementParser
from veriwrite_agent.services.requirement_parser import RuleBasedRequirementParser
from veriwrite_agent.services.requirement_pipeline import RequirementReviewPipeline
from veriwrite_agent.ui.workbench import (
    built_in_samples,
    comparison_rows,
    diagnostic_messages,
    prepare_review_from_path,
    prepare_review_from_upload,
)
from veriwrite_agent.ui.app import conflict_resolution_options


@pytest.mark.parametrize(
    ("filename", "expected_status"),
    [
        ("course_requirement.txt", "needs_resolution"),
        ("foreign_ratio_review.txt", "ready_for_confirmation"),
        ("specified_sections_review.txt", "ready_for_confirmation"),
        ("figures_and_format_review.txt", "ready_for_confirmation"),
        ("ambiguous_length_review.txt", "needs_resolution"),
    ],
)
def test_five_built_in_samples_run_through_workbench(
    filename: str,
    expected_status: str,
) -> None:
    samples = {sample.path.name: sample for sample in built_in_samples()}

    result = prepare_review_from_path(samples[filename].path, mode="rule")

    assert result.review.status == expected_status
    assert result.review.parser_mode == "rule_only"
    assert result.extracted_text
    assert result.source_format == ".txt"


def test_recommended_smoke_sample_is_small_enough_for_full_chain_review() -> None:
    sample = built_in_samples()[0]

    result = prepare_review_from_path(sample.path, mode="rule")
    requirement = result.review.reconciliation.merged_spec

    assert sample.smoke_test is True
    assert requirement.references.minimum_total == 2
    assert requirement.length.minimum_words == 600
    assert requirement.length.maximum_words == 800


def test_uploaded_plain_text_keeps_original_filename() -> None:
    result = prepare_review_from_upload(
        "teacher-requirements.txt",
        "课程综述8000字以上，参考文献不少于30篇。".encode(),
        mode="rule",
    )

    assert result.source_name == "teacher-requirements.txt"
    assert result.review.reconciliation.merged_spec.length.minimum_chars == 8000


def test_comparison_and_diagnostics_explain_rule_only_baseline() -> None:
    result = prepare_review_from_path(
        next(
            sample.path
            for sample in built_in_samples()
            if sample.path.name == "course_requirement.txt"
        ),
        mode="rule",
    )

    rows = comparison_rows(result.review)
    advantages, problems = diagnostic_messages(result.review)

    assert rows
    assert {row["判断"] for row in rows} == {"仅规则模式"}
    assert any("无需 API" in item for item in advantages)
    assert any("规则结果内部" in item for item in advantages)
    assert any("阻塞项" in item for item in problems)


def test_diagnostics_explain_llm_contract_failure() -> None:
    text = "课程综述8000字以上。"
    llm_parser = LLMRequirementParser(
        FakeLLMClient('{"document_type": 123}')
    )
    review = RequirementReviewPipeline(
        RuleBasedRequirementParser(),
        llm_parser=llm_parser,
    ).prepare(text)

    advantages, problems = diagnostic_messages(review)

    assert any("RequirementSpec 校验" in item for item in problems)
    assert any("LLMOutputValidationError" in item for item in problems)
    assert any("无法判断双路" in item for item in problems)
    assert not any("双路比较没有发现" in item for item in advantages)


def test_scalar_conflict_does_not_offer_invalid_union_operation() -> None:
    conflict = RequirementConflict(
        field="course_name",
        rule_value="研究方向文献综述",
        llm_value="研究方向文献综述课程",
        provisional_value="研究方向文献综述",
        reason="different scalar values",
    )

    options = conflict_resolution_options(conflict)

    assert options == [
        "采用规则值",
        "采用 LLM 值",
        "自定义 JSON",
    ]


def test_list_conflict_can_offer_stable_union_operation() -> None:
    conflict = RequirementConflict(
        field="required_theme_elements",
        rule_value=["气溶胶"],
        llm_value=["云"],
        provisional_value=["气溶胶"],
        reason="different list values",
    )

    options = conflict_resolution_options(conflict)

    assert "采用两边并集" in options


def test_streamlit_workbench_starts_and_analyzes_default_sample(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("VERIWRITE_AUTOSAVE_PATH", str(tmp_path / "active_project.json"))
    app_path = Path(__file__).parents[1] / "streamlit_app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=30)

    assert not app.exception
    assert app.title[0].value == "VeriWrite Agent MVP 工作台"
    assert [metric.label for metric in app.metric] == [
        "总体进度",
        "完成阶段",
        "当前阻塞",
        "最终交付",
    ]
    assert "导出项目检查点" in [
        button.label for button in app.download_button
    ]

    # The dashboard sends the user to the first incomplete stage.
    next(button for button in app.button if button.label == "继续：V0.1 需求确认").click().run(
        timeout=30
    )
    assert not app.exception
    assert any(
        subheader.value == "1. 选择输入与解析方式"
        for subheader in app.subheader
    )

    next(button for button in app.button if button.label == "开始分析").click().run(timeout=30)

    assert not app.exception
    assert [metric.label for metric in app.metric] == [
        "输入格式",
        "提取方式",
        "提取字符",
        "LLM 路径",
        "冲突",
        "阻塞项",
    ]

    app.text_input[1].input("遥感数据智能处理")
    for checkbox in app.checkbox:
        checkbox.check()
    next(
        button for button in app.button if button.label == "验证并生成最终版本"
    ).click().run(timeout=30)

    assert not app.exception
    assert any(
        message.value == "最终需求版本已通过数据合同和完整性检查。"
        for message in app.success
    )
    assert "下载最终需求版本" in [
        button.label for button in app.download_button
    ]
