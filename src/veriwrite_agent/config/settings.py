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
    reviewer_model: str | None = Field(default=None, min_length=1)
    timeout_seconds: float = Field(default=60, gt=0, le=600)
    max_retries: int = Field(default=2, ge=0, le=10)
    temperature: float = Field(default=0.2, ge=0, le=2)
    max_tokens: int = Field(default=8192, ge=256)

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
            "reviewer_model": (
                self.reviewer_model or self.structured_model or self.model
            ),
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    def for_structured_output(self) -> LLMSettings:
        """Use the configured stable JSON model for contract-heavy stages."""

        return self.model_copy(update={"model": self.structured_model or self.model})

    def for_quality_review(self) -> LLMSettings:
        """Allow the independent reviewer role to use a separately chosen model."""

        return self.model_copy(
            update={
                "model": self.reviewer_model or self.structured_model or self.model,
                "temperature": min(self.temperature, 0.1),
            }
        )
