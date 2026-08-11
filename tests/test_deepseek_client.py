from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import SecretStr

from veriwrite_agent.config.settings import LLMSettings
from veriwrite_agent.llm.deepseek_client import DeepSeekClient
from veriwrite_agent.llm.base import LLMOutputTruncatedError


def test_deepseek_adapter_uses_injected_sdk_without_network() -> None:
    sdk = MagicMock()
    sdk.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok":true}'))]
    )
    settings = LLMSettings(
        api_key=SecretStr("fake-test-key"),
        base_url="https://api.deepseek.com",
        model="deepseek-v4-flash",
        _env_file=None,
    )

    content = DeepSeekClient(settings, sdk_client=sdk).complete(
        [{"role": "user", "content": "hello"}],
        response_format={"type": "json_object"},
    )

    assert content == '{"ok":true}'
    request = sdk.chat.completions.create.call_args.kwargs
    assert request["model"] == "deepseek-v4-flash"
    assert request["response_format"] == {"type": "json_object"}
    assert request["temperature"] == 0.2
    assert request["max_tokens"] == 8192


def test_deepseek_adapter_reports_length_truncation_before_json_parsing() -> None:
    sdk = MagicMock()
    sdk.chat.completions.create.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content='{"unfinished":"value'),
                finish_reason="length",
            )
        ]
    )
    settings = LLMSettings(
        api_key=SecretStr("fake-test-key"),
        base_url="https://api.deepseek.com",
        model="deepseek-chat",
        _env_file=None,
    )

    with pytest.raises(LLMOutputTruncatedError, match="max_tokens=8192"):
        DeepSeekClient(settings, sdk_client=sdk).complete(
            [{"role": "user", "content": "return json"}],
            response_format={"type": "json_object"},
        )
