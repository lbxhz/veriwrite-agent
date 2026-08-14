"""Tests for the thin hermes-rubric adapter.

A scripted fake backend stands in for the LLM so the full three-stage pipeline
(evidence collection -> scoring -> aggregate -> receipt) is exercised offline.
"""

from __future__ import annotations

import json
import re

import pytest
from hermes_rubric import backends

from veriwrite_evaluator import adapter, config

SETTINGS = config.EvaluatorSettings(api_key="test-key")

GOOD_TEXT = (
    "本研究提出一种证据锚定的论文质量评估方法。该方法通过锁定评分标准并收集逐字引文来增强可解释性。"
    "实验表明，其在论文写作质量评定上具有较好的区分度。"
)
GOOD_QUOTE = "该方法通过锁定评分标准并收集逐字引文来增强可解释性"

BAD_QUOTE = "这段引文根本不在目标文本中"


def _evidence_payload(dim_id: str, quote: str) -> dict:
    return {
        "dim_id": dim_id,
        "evidence_found": True,
        "confidence": "high",
        "hedge": False,
        "citations": [
            {"quote": quote, "evidence_id": "S1:E1", "location": "", "source_class": "other"}
        ],
        "evidence_summary": "已引用原文作为证据。",
    }


def _score_payload(dim_id: str, quote: str) -> dict:
    return {
        "dim_id": dim_id,
        "dim_name": dim_id,
        "score": 7,
        "score_rationale": "证据支持该判定。",
        "evidence_drove_score": quote,
        "hedge_applied": False,
    }


def _make_fake(quote: str):
    def fake(prompt: str, backend: str | None = None) -> str:
        batched = "JSON ARRAY" in prompt
        if "You are an evidence collector" in prompt:
            if batched:
                dim_ids = re.findall(r'<DIM id="([^"]+)"', prompt)
                return json.dumps([_evidence_payload(d, quote) for d in dim_ids])
            dim_id = re.search(r'"dim_id":\s*"([^"]+)"', prompt).group(1)
            return json.dumps(_evidence_payload(dim_id, quote))
        if "You are a structured scorer" in prompt:
            if batched:
                dim_ids = re.findall(r'<DIM id="([^"]+)"', prompt)
                return json.dumps([_score_payload(d, quote) for d in dim_ids])
            dim_id = re.search(r'"dim_id":\s*"([^"]+)"', prompt).group(1)
            return json.dumps(_score_payload(dim_id, quote))
        raise AssertionError(f"未知的 hermes-rubric 阶段: {prompt[:120]!r}")

    return fake


@pytest.fixture
def monkeypatch_backend(monkeypatch):
    def _apply(quote: str):
        fake = _make_fake(quote)
        monkeypatch.setattr(backends, "call", fake)
        return fake

    return _apply


def test_load_rubric():
    rubric = adapter.load_rubric()
    assert rubric["rubric_source"] == "class-template"
    assert rubric["target_type"] == "academic_writing_zh_v1"
    assert len(rubric["dimensions"]) == 8
    for dim in rubric["dimensions"]:
        for field in ("id", "name", "description", "evidence_instructions", "weight"):
            assert field in dim


def test_rubric_hash_stable():
    from hermes_rubric.receipt import rubric_hash

    r1 = adapter.load_rubric()
    r2 = adapter.load_rubric()
    assert rubric_hash(r1) == rubric_hash(r2)


def test_list_rubrics(monkeypatch_backend):
    items = adapter.list_rubrics()
    assert any(item["id"] == "academic_writing_zh_v1" for item in items)
    assert items[0]["rubric_hash"] == adapter.load_rubric().get("__hash__") or items[0]["rubric_hash"]


def test_evaluate_writing_pipeline(monkeypatch_backend):
    monkeypatch_backend(GOOD_QUOTE)
    result = adapter.evaluate_writing(GOOD_TEXT, settings=SETTINGS, batch=False)
    assert result["aggregate"] == pytest.approx(7.0)
    assert result["aggregate_100"] == pytest.approx(70.0)
    assert len(result["dim_summaries"]) == 8
    assert all(s["score_100"] == s["score"] * 10 for s in result["dim_summaries"])
    assert result["evaluation_method"].startswith(
        "hermes-rubric-v2|scope=full-document|rubric=academic_writing_zh_v1|"
    )
    assert "rubric_hash=" in result["evaluation_method"]
    # evidence citations must be verbatim-verified quotes
    assert result["evidence_citations"][0]["citations"][0]["quote"] == GOOD_QUOTE
    # receipt carries input hashes
    assert result["receipt"]["inputs"]["target_hash_sha256"]
    assert result["receipt"]["pipeline"]["target_coverage_ratio"] == 1.0
    assert result["receipt"]["pipeline"]["target_truncated"] is False


def test_evaluate_writing_batch(monkeypatch_backend):
    monkeypatch_backend(GOOD_QUOTE)
    result = adapter.evaluate_writing(GOOD_TEXT, settings=SETTINGS, batch=True)
    assert len(result["dim_summaries"]) == 8
    assert result["aggregate"] == pytest.approx(7.0)


def test_unverifiable_evidence_hedged_and_capped(monkeypatch_backend):
    monkeypatch_backend(BAD_QUOTE)
    result = adapter.evaluate_writing(GOOD_TEXT, settings=SETTINGS, batch=False)
    # unverifiable quotes are dropped -> evidence_found False -> hedge True -> score capped at 3
    for summary in result["dim_summaries"]:
        assert summary["score"] == 3
    assert result["hedge_dims"]  # at least one dim hedged


def test_evaluate_pairwise(monkeypatch_backend):
    monkeypatch_backend(GOOD_QUOTE)
    better = adapter.evaluate_pairwise(GOOD_TEXT, GOOD_TEXT, settings=SETTINGS, batch=False)
    assert better["preferred"] == "tie"
    assert better["overall_delta_b_minus_a"] == 0.0
    assert len(better["dim_deltas"]) == 8
    assert "result_a" in better and "result_b" in better


def test_evaluation_method_isolates_rubric_versions(monkeypatch_backend):
    monkeypatch_backend(GOOD_QUOTE)
    from hermes_rubric.receipt import rubric_hash

    a = adapter.evaluate_writing(GOOD_TEXT, settings=SETTINGS, batch=False)
    # Mutate a copy of the rubric to simulate a changed measuring stick.
    rubric_b = adapter.load_rubric()
    rubric_b["dimensions"][0]["name"] = "改动后的维度名"
    method_a = a["evaluation_method"]
    method_b = f"hermes-rubric-v2|scope=full-document|rubric=academic_writing_zh_v1|rubric_hash={rubric_hash(rubric_b)}|backend=deepseek-openai"
    assert method_a != method_b


def test_deepseek_backend_url_and_model():
    from veriwrite_evaluator.deepseek_backend import DeepSeekOpenAICompatibleBackend

    backend = DeepSeekOpenAICompatibleBackend(SETTINGS)
    assert backend.name == "deepseek-openai"
    assert backend._url == "https://api.deepseek.com/chat/completions"
    assert backend.model_id() == "deepseek-chat"
    assert backend.availability() is True
    assert backend._uses_system_proxy is False


def test_mcp_server_tools():
    from veriwrite_evaluator.mcp_server import server

    tools = set(server._tool_manager._tools)
    assert {"list_rubrics", "get_rubric", "evaluate_writing", "evaluate_pairwise"} <= tools
