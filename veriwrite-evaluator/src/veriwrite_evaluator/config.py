"""Runtime configuration for the hermes-rubric adapter.

Environment variables use prefix ``VW_EVAL_``, falling back to VeriWrite's
``LLM_`` prefix so the tool can be spawned by VeriWrite with its existing .env.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip() != "":
            return value.strip()
    return default


@dataclass(frozen=True)
class EvaluatorSettings:
    """OpenAI-compatible judge endpoint for the scoring LLM."""

    api_key: str
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    temperature: float = 0.0
    timeout_seconds: float = 300.0
    max_retries: int = 3
    seed: int = 42
    batch: bool = True
    target_window_bytes: int = 120000

    @classmethod
    def from_env(cls, api_key: str | None = None) -> "EvaluatorSettings":
        resolved = api_key or _env("VW_EVAL_API_KEY", "LLM_API_KEY") or ""
        if not resolved:
            raise ValueError("缺少 API 密钥：请设置 VW_EVAL_API_KEY（或 VeriWrite 的 LLM_API_KEY）")
        return cls(
            api_key=resolved,
            base_url=_env("VW_EVAL_BASE_URL", "LLM_BASE_URL", default="https://api.deepseek.com") or "https://api.deepseek.com",
            model=_env("VW_EVAL_MODEL", "LLM_REVIEWER_MODEL", "LLM_STRUCTURED_MODEL", "LLM_MODEL", default="deepseek-chat") or "deepseek-chat",
            temperature=float(_env("VW_EVAL_TEMPERATURE", default="0.0") or "0.0"),
            timeout_seconds=float(_env("VW_EVAL_TIMEOUT_SECONDS", default="300.0") or "300.0"),
            max_retries=int(_env("VW_EVAL_MAX_RETRIES", default="3") or "3"),
            seed=int(_env("VW_EVAL_SEED", default="42") or "42"),
            batch=os.getenv("VW_EVAL_BATCH", "1") not in ("0", "false", "False"),
            target_window_bytes=int(_env("VW_EVAL_TARGET_WINDOW_BYTES", default="120000") or "120000"),
        )
