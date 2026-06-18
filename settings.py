"""
settings.py
===========
AgniAI — Configuration Safety Layer.

Pydantic Settings with full validation, environment profiles,
and startup-time failure guarantee.

Application FAILS during startup if critical values are missing.
Never fails after first request.

CRITICAL_VARS reflects only what AgniAI actually requires:
  - DOTNET_API_BASE_URL   (required for admin pipeline)
No DATABASE_URL or INTERNAL_SERVICE_KEY required by the base app.
"""

from __future__ import annotations

import re
import sys
from enum import Enum
from functools import lru_cache
from typing import Any, List, Optional
from urllib.parse import urlparse

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


class DotNetAPIConfig(BaseSettings):
    """Settings for the .NET integration layer."""

    model_config = SettingsConfigDict(
        env_prefix="DOTNET_API_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    BASE_URL: str = Field(..., description="Root URL of the .NET API")
    TIMEOUT_CONNECT: float = Field(default=5.0, ge=1.0, le=30.0)
    TIMEOUT_READ: float = Field(default=30.0, ge=5.0, le=120.0)
    MAX_RETRIES: int = Field(default=3, ge=0, le=10)

    @field_validator("BASE_URL", mode="before")
    @classmethod
    def strip_trailing_slash(cls, v: str) -> str:
        v = str(v).rstrip("/")
        parsed = urlparse(v)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(
                f"DOTNET_API_BASE_URL must use http or https, got: {parsed.scheme!r}"
            )
        if not parsed.netloc:
            raise ValueError(f"DOTNET_API_BASE_URL has no host: {v!r}")
        return v


class APIKeysConfig(BaseSettings):
    """Optional secret credentials."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Optional — only required if the app is deployed with WhatsApp integration
    INTERNAL_SERVICE_KEY: Optional[SecretStr] = Field(default=None)
    WHATSAPP_TOKEN: Optional[SecretStr] = Field(default=None)
    WHATSAPP_VERIFY_TOKEN: Optional[SecretStr] = Field(default=None)
    OLLAMA_API_KEY: Optional[SecretStr] = Field(default=None)


class TimeoutConfig(BaseSettings):
    """Global timeout/retry budget for every subsystem."""

    model_config = SettingsConfigDict(
        env_prefix="TIMEOUT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    LLM_INFERENCE: float = Field(default=60.0, ge=10.0, le=300.0)
    RAG_RETRIEVAL: float = Field(default=10.0, ge=1.0, le=60.0)
    DB_QUERY: float = Field(default=15.0, ge=1.0, le=60.0)
    HEALTH_CHECK: float = Field(default=3.0, ge=0.5, le=10.0)


class FeatureFlagConfig(BaseSettings):
    """Runtime toggles — safe to change without code deployment."""

    model_config = SettingsConfigDict(
        env_prefix="FEATURE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ENABLE_RAG: bool = Field(default=True)
    ENABLE_RERANKER: bool = Field(default=True)
    ENABLE_WHATSAPP_BOT: bool = Field(default=False)
    ENABLE_ADMIN_CHATBOT: bool = Field(default=True)
    ENABLE_SWAGGER_UI: bool = Field(default=True)
    ENABLE_DEBUG_LOGGING: bool = Field(default=False)
    STRICT_INTENT_CLASSIFICATION: bool = Field(default=True)


class PrometheusConfig(BaseSettings):
    """Prometheus/metrics configuration."""

    model_config = SettingsConfigDict(
        env_prefix="PROMETHEUS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ENABLED: bool = Field(default=True)
    PORT: int = Field(default=8000, ge=1024, le=65535)
    PATH: str = Field(default="/metrics")


class SentryConfig(BaseSettings):
    """Sentry error monitoring configuration."""

    model_config = SettingsConfigDict(
        env_prefix="SENTRY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    DSN: Optional[str] = Field(default=None)
    TRACES_SAMPLE_RATE: float = Field(default=0.1, ge=0.0, le=1.0)
    ENVIRONMENT: str = Field(default="development")


class OtelConfig(BaseSettings):
    """OpenTelemetry configuration."""

    model_config = SettingsConfigDict(
        env_prefix="OTEL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ENABLED: bool = Field(default=False)
    EXPORTER_OTLP_ENDPOINT: str = Field(default="http://localhost:4317")
    SERVICE_NAME: str = Field(default="agniai")


class AppSettings(BaseSettings):
    """
    Single source of truth for AgniAI.
    Only DOTNET_API_BASE_URL is truly critical for the admin pipeline.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        hide_input_in_errors=True,
    )

    APP_NAME: str = Field(default="AgniAI")
    APP_VERSION: str = Field(default="1.0.0", pattern=r"^\d+\.\d+\.\d+$")
    ENV: Environment = Field(default=Environment.DEVELOPMENT)

    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434")
    OLLAMA_MODEL: str = Field(default="mistral:7b-instruct-q4_K_M")
    MIN_SCORE: float = Field(default=0.35, ge=0.0, le=1.0)

    HOST: str = Field(default="0.0.0.0")
    PORT: int = Field(default=5000, ge=1024, le=65535)
    WORKERS: int = Field(default=1, ge=1, le=32)

    CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173"]
    )

    @field_validator("OLLAMA_BASE_URL", mode="before")
    @classmethod
    def strip_ollama_slash(cls, v: str) -> str:
        return str(v).rstrip("/")

    @model_validator(mode="after")
    def validate_workers_vs_env(self) -> "AppSettings":
        if self.ENV == Environment.TESTING and self.WORKERS > 1:
            raise ValueError(
                "WORKERS must be 1 in testing environment to avoid parallel test interference."
            )
        return self

    def is_production(self) -> bool:
        return self.ENV == Environment.PRODUCTION

    def is_testing(self) -> bool:
        return self.ENV == Environment.TESTING

    def is_development(self) -> bool:
        return self.ENV == Environment.DEVELOPMENT

    def safe_repr(self) -> dict[str, Any]:
        """Return config dict with secrets masked — safe for logs."""
        return self.model_dump()


# ── Critical startup guard ────────────────────────────────────────────────

# Only the vars that AgniAI ACTUALLY needs to function.
# DATABASE_URL and INTERNAL_SERVICE_KEY are not required by the base app.
CRITICAL_VARS = [
    "DOTNET_API_BASE_URL",
]


def validate_critical_env(env_map: dict[str, str | None]) -> None:
    """
    Called BEFORE any Flask/FastAPI app initialisation.
    Terminates the process immediately with a clear error if any
    critical variable is absent.  Never raises — always sys.exit(1).
    """
    missing = [k for k in CRITICAL_VARS if not env_map.get(k)]
    if missing:
        print(
            "\n[FATAL] AgniAI startup aborted — missing critical environment variables:\n"
            + "\n".join(f"  ✗  {k}" for k in missing)
            + "\n\nSet the above variables in your .env file or environment before starting.\n",
            file=sys.stderr,
        )
        sys.exit(1)


# ── Cached singletons ─────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()


@lru_cache(maxsize=1)
def get_dotnet_config() -> DotNetAPIConfig:
    return DotNetAPIConfig()


@lru_cache(maxsize=1)
def get_api_keys() -> APIKeysConfig:
    return APIKeysConfig()


@lru_cache(maxsize=1)
def get_timeouts() -> TimeoutConfig:
    return TimeoutConfig()


@lru_cache(maxsize=1)
def get_feature_flags() -> FeatureFlagConfig:
    return FeatureFlagConfig()


@lru_cache(maxsize=1)
def get_prometheus_config() -> PrometheusConfig:
    return PrometheusConfig()


@lru_cache(maxsize=1)
def get_sentry_config() -> SentryConfig:
    return SentryConfig()


@lru_cache(maxsize=1)
def get_otel_config() -> OtelConfig:
    return OtelConfig()
