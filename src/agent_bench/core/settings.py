"""Benchmark settings and configuration management using pydantic-settings."""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BenchSettings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Environment & System
    environment: Literal["development", "testing", "production"] = Field(
        default="development",
        description="Runtime environment mode",
    )
    debug: bool = Field(
        default=False,
        description="Enable verbose debug logging",
    )
    output_dir: Path = Field(
        default=Path("runs"),
        description="Directory where benchmark run artifacts and traces are saved",
    )
    default_seed: int = Field(
        default=42,
        description="Global seed for reproducible evaluations",
    )

    # API Keys & Provider Config
    openai_api_key: str | None = Field(
        default=None,
        description="API key for OpenAI model provider",
    )
    openai_org_id: str | None = Field(
        default=None,
        description="Optional organization ID for OpenAI",
    )
    anthropic_api_key: str | None = Field(
        default=None,
        description="API key for Anthropic model provider",
    )

    # Pricing per 1,000 tokens for default cost calculations (USD)
    cost_per_1k_input_tokens: float = Field(
        default=0.0015,
        description="Default cost per 1k input prompt tokens in USD",
    )
    cost_per_1k_output_tokens: float = Field(
        default=0.0020,
        description="Default cost per 1k output completion tokens in USD",
    )


# Singleton settings instance
settings = BenchSettings()
