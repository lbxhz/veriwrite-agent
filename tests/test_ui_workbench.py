from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from veriwrite_agent.ui.workbench import (
    built_in_samples,
    comparison_rows,
    diagnostic_messages,
    prepare_review_from_path,
    prepare_review_from_upload,
)


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
        built_in_samples()[0].path,
        mode="rule",
    )

    rows = comparison_rows(result.review)
    advantages, problems = diagnostic_messages(result.review)

    assert rows
    assert {row["判断"] for row in rows} == {"仅规则模式"}
    assert any("无需 API" in item for item in advantages)
    assert any("阻塞项" in item for item in problems)


def test_streamlit_workbench_starts_and_analyzes_default_sample() -> None:
    app_path = Path(__file__).parents[1] / "streamlit_app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=30)

    assert not app.exception
    assert app.title[0].value == "VeriWrite V0.1 验证工作台"

    app.button[0].click().run(timeout=30)

    assert not app.exception
    assert [metric.label for metric in app.metric] == [
        "输入格式",
        "提取字符",
        "冲突",
        "阻塞项",
        "耗时",
    ]

    app.text_input[1].input("遥感数据智能处理")
    for checkbox in app.checkbox:
        checkbox.check()
    app.button[1].click().run(timeout=30)

    assert not app.exception
    assert any(
        message.value == "最终需求版本已通过数据合同和完整性检查。"
        for message in app.success
    )
    assert "下载最终需求版本" in [
        button.label for button in app.download_button
    ]
