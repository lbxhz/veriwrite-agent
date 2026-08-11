"""Thin adapter over the hermes-rubric engine.

Localization vs upstream:
- deterministic Chinese rubric templates (``templates/*.yaml``) loaded through
  ``classes.to_rubric`` (bypasses non-deterministic LLM rubric synthesis);
- an OpenAI-compatible DeepSeek backend registered into hermes-rubric's plugin
  registry (upstream backends hardcode their endpoints);
- academic writing disables hermes-rubric's repo-oriented self-marketing cap
  (see below) because a paper's own prose IS the evidence being scored;
- output is normalized to both 0-10 and 0-100 scales plus an
  ``evaluation_method`` fingerprint compatible with VeriWrite's compare()
  isolation contract.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from hermes_rubric import backends
from hermes_rubric.classes import to_rubric
from hermes_rubric.evidence import collect_evidence
from hermes_rubric.receipt import build_receipt, rubric_hash
from hermes_rubric.score import compute_aggregate, score_dimensions

from veriwrite_evaluator.config import EvaluatorSettings
from veriwrite_evaluator.deepseek_backend import DeepSeekOpenAICompatibleBackend

# 学术写作评估中，论文正文本身就是被评估的证据来源，不存在"代码/测试 vs 营销文案"的权威分级。
# hermes-rubric 面向代码仓库的 self-marketing 封顶（全部引用为 readme/doc 时把分压到 ≤6/10）
# 会错误地把学术正文证据一律压到 ≤60 分，因此在本学术域禁用该封顶。
# 注：这是对上游模块属性的定点覆盖，仅影响该封顶逻辑，其余 hedge/no-evidence 钳制保持不变。
from hermes_rubric import score as _score_module  # noqa: E402

_score_module._only_self_marketing = lambda ev: False  # type: ignore[attr-defined]

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_RUBRICS: dict[str, str] = {
    "academic_writing_zh_v1": "academic_writing_zh_v1.yaml",
}


def list_rubrics() -> list[dict]:
    """Available deterministic rubric templates, each with a locked hash."""
    results = []
    for rubric_id, file_name in _RUBRICS.items():
        rubric = load_rubric(rubric_id)
        results.append(
            {
                "id": rubric_id,
                "file": file_name,
                "dimensions": [d["id"] for d in rubric["dimensions"]],
                "rubric_hash": rubric_hash(rubric),
            }
        )
    return results


def load_rubric(rubric_id: str = "academic_writing_zh_v1") -> dict:
    if rubric_id not in _RUBRICS:
        raise KeyError(f"未知 rubric: {rubric_id}，可用: {sorted(_RUBRICS)}")
    path = _TEMPLATES_DIR / _RUBRICS[rubric_id]
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return to_rubric(data)


def ensure_backend(settings: EvaluatorSettings) -> str:
    """Register the DeepSeek backend once and return its backend name."""
    name = DeepSeekOpenAICompatibleBackend.name
    try:
        backends.get_backend(name)
    except KeyError:
        backends.register(DeepSeekOpenAICompatibleBackend(settings), replace=True)
    return name


def evaluate_writing(
    text: str,
    rubric_id: str = "academic_writing_zh_v1",
    *,
    settings: EvaluatorSettings | None = None,
    target_path: str = "(in-memory)",
    batch: bool | None = None,
    target_window_bytes: int | None = None,
) -> dict:
    """Run the hermes-rubric three-stage pipeline on ``text``.

    Returns rubric, evidence citations, per-dim scores (0-10 and 0-100), the
    weighted aggregate, hedge info, an ``evaluation_method`` fingerprint and a
    reproducibility receipt.
    """
    settings = settings or EvaluatorSettings.from_env()
    backend = ensure_backend(settings)
    rubric = load_rubric(rubric_id)
    use_batch = settings.batch if batch is None else batch
    requested_window = (
        settings.target_window_bytes
        if target_window_bytes is None
        else target_window_bytes
    )
    target_total_bytes = len(text.encode("utf-8"))
    # Academic-paper evaluation is a full-document measurement.  The upstream
    # engine treats ``target_window_bytes`` as a UTF-8 prefix limit, so silently
    # accepting a smaller value would score only the introduction of a long
    # paper.  Expand the window here even when an older caller still sends the
    # historical 20 KB default.
    window = max(requested_window, target_total_bytes)

    evidence_list = collect_evidence(
        rubric=rubric,
        target_content=text,
        target_path=target_path,
        backend=backend,
        batch=use_batch,
        target_window_bytes=window,
    )
    scores = score_dimensions(
        rubric=rubric,
        evidence_list=evidence_list,
        backend=backend,
        batch=use_batch,
    )
    aggregate = compute_aggregate(rubric=rubric, scores=scores)
    receipt = build_receipt(
        intent=rubric["rubric_intent"],
        context_path="(class-template)",
        target_path=target_path,
        backend=backend,
        rubric=rubric,
        evidence_list=evidence_list,
        scores=scores,
        target_content=text,
        context_content="",
    )
    pipeline = receipt.setdefault("pipeline", {})
    pipeline.update(
        {
            "target_scope": "full_document",
            "target_window_bytes": window,
            "target_total_bytes": target_total_bytes,
            "target_visible_bytes": target_total_bytes,
            "target_coverage_ratio": 1.0,
            "target_truncated": False,
        }
    )

    dim_summaries = []
    for summary in aggregate["dim_summaries"]:
        dim_summaries.append(
            {
                **summary,
                "score_100": round(summary["score"] * 10, 1),
            }
        )

    return {
        "rubric_id": rubric_id,
        "rubric": rubric,
        "evidence_citations": evidence_list,
        "per_dim_scores": scores,
        "aggregate": aggregate["aggregate"],
        "aggregate_100": round(aggregate["aggregate"] * 10, 1),
        "max_possible": 10.0,
        "hedge_dims": aggregate["hedge_dims"],
        "hedge_note": aggregate["hedge_note"],
        "dim_summaries": dim_summaries,
        "evaluation_method": (
            f"hermes-rubric-v2|scope=full-document|rubric={rubric_id}|"
            f"rubric_hash={rubric_hash(rubric)}|backend={backend}"
        ),
        "receipt": receipt,
    }


def evaluate_pairwise(
    text_a: str,
    text_b: str,
    rubric_id: str = "academic_writing_zh_v1",
    *,
    settings: EvaluatorSettings | None = None,
    batch: bool | None = None,
    target_window_bytes: int | None = None,
) -> dict:
    """Blind-ish A/B comparison: run the same rubric on both and report deltas.

    Both texts are evaluated with the identical locked rubric and judge backend,
    so per-dimension and overall deltas are comparable (same-task regression,
    not cross-domain ranking).
    """
    settings = settings or EvaluatorSettings.from_env()
    result_a = evaluate_writing(text_a, rubric_id, settings=settings, batch=batch, target_window_bytes=target_window_bytes)
    result_b = evaluate_writing(text_b, rubric_id, settings=settings, batch=batch, target_window_bytes=target_window_bytes)

    dim_a = {s["dim_id"]: s for s in result_a["dim_summaries"]}
    dim_b = {s["dim_id"]: s for s in result_b["dim_summaries"]}
    dim_deltas = {}
    for dim_id in dim_a:
        if dim_id in dim_b:
            dim_deltas[dim_id] = {
                "name": dim_a[dim_id]["name"],
                "a": dim_a[dim_id]["score_100"],
                "b": dim_b[dim_id]["score_100"],
                "delta_b_minus_a": round(dim_b[dim_id]["score_100"] - dim_a[dim_id]["score_100"], 1),
            }

    overall_delta = round(result_b["aggregate_100"] - result_a["aggregate_100"], 1)
    if overall_delta > 0:
        preferred = "B"
    elif overall_delta < 0:
        preferred = "A"
    else:
        preferred = "tie"

    return {
        "rubric_id": rubric_id,
        "overall_delta_b_minus_a": overall_delta,
        "dim_deltas": dim_deltas,
        "preferred": preferred,
        "result_a": result_a,
        "result_b": result_b,
    }
