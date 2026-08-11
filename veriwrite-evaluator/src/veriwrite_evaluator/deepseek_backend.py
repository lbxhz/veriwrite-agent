"""OpenAI-compatible backend for hermes-rubric (DeepSeek / vLLM / Ollama).

hermes-rubric ships built-in backends whose endpoints are hardcoded (OpenAI
official, DashScope, Gemini, local Ollama). This plugin implements its
``BackendProtocol`` against an arbitrary OpenAI-compatible ``/chat/completions``
endpoint, so the scoring LLM can be DeepSeek (or any local server) without
touching hermes-rubric source.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from veriwrite_evaluator.config import EvaluatorSettings

_RETRYABLE_STATUS = (429, 500, 502, 503, 504)


class DeepSeekOpenAICompatibleBackend:
    """hermes-rubric backend name ``deepseek-openai``."""

    name = "deepseek-openai"

    def __init__(self, settings: EvaluatorSettings):
        self._settings = settings
        self._url = settings.base_url.rstrip("/") + "/chat/completions"

    def call(self, prompt: str, max_tokens: int = 2048) -> str:
        payload = json.dumps(
            {
                "model": self._settings.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self._settings.temperature,
                "seed": self._settings.seed,
                "max_tokens": max_tokens,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self._url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._settings.api_key}",
            },
            method="POST",
        )
        last_error: str | None = None
        for attempt in range(self._settings.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self._settings.timeout_seconds) as resp:
                    data = json.loads(resp.read())
                return data["choices"][0]["message"]["content"].strip()
            except urllib.error.HTTPError as error:
                body = error.read().decode("utf-8", errors="replace")[:400]
                if error.code in _RETRYABLE_STATUS and attempt < self._settings.max_retries:
                    last_error = f"HTTP {error.code}: {body}"
                    time.sleep(min(2**attempt, 10))
                    continue
                raise RuntimeError(f"DeepSeek 调用失败 (HTTP {error.code}): {body}") from error
            except urllib.error.URLError as error:
                if attempt < self._settings.max_retries:
                    last_error = str(error)
                    time.sleep(min(2**attempt, 10))
                    continue
                raise RuntimeError(f"DeepSeek 调用失败: {error}") from error
        raise RuntimeError(f"DeepSeek 调用失败（已重试 {self._settings.max_retries} 次）: {last_error}")

    def model_id(self) -> str:
        return self._settings.model

    def availability(self) -> bool:
        return bool(self._settings.api_key)
