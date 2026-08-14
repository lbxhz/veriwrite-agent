from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import SecretStr

from veriwrite_agent.config.settings import LLMSettings
from veriwrite_agent.llm.deepseek_client import DeepSeekClient
from veriwrite_agent.llm.base import LLMOutputTruncatedError
from veriwrite_agent.llm import deepseek_client


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


def test_deepseek_transport_ignores_system_proxy_by_default(monkeypatch) -> None:
    transport = object()
    transport_factory = MagicMock(return_value=transport)
    sdk_factory = MagicMock()
    monkeypatch.setattr(deepseek_client, "DefaultHttpxClient", transport_factory)
    monkeypatch.setattr(deepseek_client, "OpenAI", sdk_factory)
    settings = LLMSettings(
        api_key=SecretStr("fake-test-key"),
        base_url="https://api.deepseek.com",
        _env_file=None,
    )

    DeepSeekClient(settings)

    transport_factory.assert_called_once_with(trust_env=False)
    assert sdk_factory.call_args.kwargs["http_client"] is transport
