import pytest
from pydantic import ValidationError

from veriwrite_agent.config.settings import LLMSettings


def test_settings_load_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "test-secret-value")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("LLM_MODEL", "deepseek-v4-flash")

    settings = LLMSettings(_env_file=None)

    assert settings.model == "deepseek-v4-flash"
    assert settings.public_summary()["api_key_configured"] is True
    assert "test-secret-value" not in repr(settings)
    assert "test-secret-value" not in str(settings.public_summary())


def test_blank_api_key_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "")

    with pytest.raises(ValidationError):
        LLMSettings(_env_file=None)

