import copy
import json
import sys
from pathlib import Path

import pytest

from veriwrite_agent.models.external_writing_evaluation import (
    ExternalWritingEvaluation,
)
from veriwrite_agent.services.external_writing_evaluator import (
    ExternalEvaluatorConfig,
    ExternalEvaluatorError,
    ExternalWritingEvaluatorClient,
    comparable_external_evaluations,
    external_quality_warning,
)

ROOT = Path(__file__).parents[1]
EVALUATOR_ROOT = ROOT / "veriwrite-evaluator"
SAMPLE_TEXT = (EVALUATOR_ROOT / "tmp_sample.md").read_text(encoding="utf-8")
SAMPLE_RESULT = json.loads(
    (EVALUATOR_ROOT / "result2.json").read_text(encoding="utf-8")
)


def _full_document_result() -> dict:
    payload = copy.deepcopy(SAMPLE_RESULT)
    total_bytes = len(SAMPLE_TEXT.encode("utf-8"))
    old_method = payload["evaluation_method"]
    payload["evaluation_method"] = old_method.replace(
        "hermes-rubric-v1|",
        "hermes-rubric-v2|scope=full-document|",
        1,
    )
    payload["receipt"]["pipeline"].update(
        {
            "target_scope": "full_document",
            "target_window_bytes": max(120000, total_bytes),
            "target_total_bytes": total_bytes,
            "target_visible_bytes": total_bytes,
            "target_coverage_ratio": 1.0,
            "target_truncated": False,
        }
    )
    return payload


class StubEvaluatorClient(ExternalWritingEvaluatorClient):
    def __init__(self, payload):
        super().__init__()
        self.payload = payload

    def _call_tool(self, name, arguments):
        del name, arguments
        return self.payload


def test_real_mcp_stdio_server_lists_locked_rubric() -> None:
    client = ExternalWritingEvaluatorClient(
        ExternalEvaluatorConfig(
            command=sys.executable,
            cwd=EVALUATOR_ROOT,
            timeout_seconds=30,
        )
    )

    rubrics = client.list_rubrics()

    assert [item.id for item in rubrics] == ["academic_writing_zh_v1"]
    assert len(rubrics[0].dimensions) == 8
    assert len(rubrics[0].rubric_hash) == 64


def test_evaluate_writing_validates_fingerprint_and_target_receipt() -> None:
    result = StubEvaluatorClient(_full_document_result()).evaluate_writing(SAMPLE_TEXT)

    assert result.aggregate_100 == 80
    assert len(result.evidence_citations) == 8
    assert len(result.rubric_hash) == 64


def test_receipt_for_another_target_is_rejected() -> None:
    with pytest.raises(ExternalEvaluatorError, match="does not match target"):
        StubEvaluatorClient(_full_document_result()).evaluate_writing("different paper")


def test_legacy_partial_measurement_is_parseable_but_not_reusable() -> None:
    historical = ExternalWritingEvaluation.model_validate(SAMPLE_RESULT)

    assert historical.is_full_document_measurement is False
    with pytest.raises(ExternalEvaluatorError, match="complete paper"):
        StubEvaluatorClient(SAMPLE_RESULT).evaluate_writing(SAMPLE_TEXT)


def test_cross_rubric_versions_are_not_comparable() -> None:
    baseline = ExternalWritingEvaluation.model_validate(SAMPLE_RESULT)
    changed_payload = copy.deepcopy(SAMPLE_RESULT)
    changed_hash = "f" * 64
    changed_payload["evaluation_method"] = (
        "hermes-rubric-v1|rubric=academic_writing_zh_v1|"
        f"rubric_hash={changed_hash}|backend=deepseek-openai"
    )
    changed_payload["receipt"]["pipeline"]["stage_1_rubric_hash_sha256"] = (
        changed_hash
    )
    candidate = ExternalWritingEvaluation.model_validate(changed_payload)

    assert comparable_external_evaluations(baseline, candidate) is False


def test_low_external_score_is_a_visible_signal_not_an_internal_gate() -> None:
    payload = _full_document_result()
    payload["aggregate"] = 6.9
    payload["aggregate_100"] = 69
    evaluation = ExternalWritingEvaluation.model_validate(payload)

    warning = external_quality_warning(evaluation)

    assert warning is not None
    assert "69.0/100" in warning
    assert "不改变内部 release gate" in warning


def test_external_score_at_threshold_does_not_warn() -> None:
    payload = _full_document_result()
    payload["aggregate"] = 7.0
    payload["aggregate_100"] = 70
    evaluation = ExternalWritingEvaluation.model_validate(payload)

    assert external_quality_warning(evaluation) is None


def test_veriwrite_environment_pins_judge_model_but_honors_runtime_window(
    monkeypatch,
) -> None:
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("VW_EVAL_MODEL", "deepseek-v4-flash")
    monkeypatch.setenv("VW_EVAL_BATCH", "0")
    monkeypatch.setenv("VW_EVAL_TARGET_WINDOW_BYTES", "24000")

    config = ExternalEvaluatorConfig.from_veriwrite_environment(cwd=ROOT)

    assert config.environment is not None
    assert config.environment["VW_EVAL_MODEL"] == "deepseek-chat"
    assert config.environment["VW_EVAL_TEMPERATURE"] == "0.0"
    assert config.target_window_bytes == 24000
    assert config.batch is False
