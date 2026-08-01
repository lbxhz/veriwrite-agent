"""Load LLM configuration from environment variables or a local .env file."""

from __future__ import annotations

from pydantic import AnyHttpUrl, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    """Configuration boundary for an OpenAI-compatible LLM provider."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="LLM_",
        case_sensitive=False,
        extra="ignore",
    )

    api_key: SecretStr
    base_url: AnyHttpUrl = "https://api.deepseek.com"
    model: str = Field(default="deepseek-v4-flash", min_length=1)
    structured_model: str | None = Field(default="deepseek-chat", min_length=1)
    timeout_seconds: float = Field(default=60, gt=0, le=600)
    max_retries: int = Field(default=2, ge=0, le=10)

    @field_validator("api_key")
    @classmethod
    def api_key_must_not_be_blank(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("LLM_API_KEY is blank")
        return value

    def public_summary(self) -> dict[str, str | int | float | bool]:
        """Return safe diagnostic data without exposing the API key."""

        return {
            "api_key_configured": bool(self.api_key.get_secret_value()),
            "base_url": str(self.base_url),
            "model": self.model,
            "structured_model": self.structured_model or self.model,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
        }

    def for_structured_output(self) -> LLMSettings:
        """Use the configured stable JSON model for contract-heavy stages."""

        return self.model_copy(update={"model": self.structured_model or self.model})
