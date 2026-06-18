"""
config/settings.py
==================
AgniAI — Configuration Safety Layer (Phase 12).

Pydantic Settings with full validation, environment profiles,
and startup-time failure guarantee.

Application FAILS during startup if critical values are missing.
Never fails after first request.
"""

from __future__ import annotations

import re
import sys
from enum import Enum
from functools import lru_cache
from typing import Any, List, Literal, Optional
from urllib.parse import urlparse

from pydantic import (
    AnyHttpUrl,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)
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
    """Secret credentials — never logged, never serialised to plain text."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    OLLAMA_API_KEY: Optional[SecretStr] = Field(default=None)
    INTERNAL_SERVICE_KEY: SecretStr = Field(
        ..., description="Shared secret for service-to-service auth."
    )
    WHATSAPP_TOKEN: Optional[SecretStr] = Field(default=None)
    WHATSAPP_VERIFY_TOKEN: Optional[SecretStr] = Field(default=None)

    @field_validator("INTERNAL_SERVICE_KEY", mode="before")
    @classmethod
    def validate_key_strength(cls, v: str) -> str:
        raw = v.get_secret_value() if hasattr(v, "get_secret_value") else str(v)
        if len(raw) < 32:
            raise ValueError(
                "INTERNAL_SERVICE_KEY must be at least 32 characters. "
                "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
            )
        return v


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
    Loaded once at startup; any missing CRITICAL variable causes sys.exit(1).
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

    DATABASE_URL: str = Field(..., description="SQLAlchemy connection string.")
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434")
    OLLAMA_MODEL: str = Field(default="llama3")
    MIN_SCORE: float = Field(default=0.35, ge=0.0, le=1.0)

    HOST: str = Field(default="0.0.0.0")
    PORT: int = Field(default=5000, ge=1024, le=65535)
    WORKERS: int = Field(default=1, ge=1, le=32)

    CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173"]
    )

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def validate_db_url(cls, v: str) -> str:
        allowed = ("sqlite", "postgresql", "mssql", "mysql")
        if not any(v.startswith(p) for p in allowed):
            raise ValueError(
                f"DATABASE_URL must start with one of {allowed}. Got: {v[:20]}…"
            )
        return v

    @field_validator("OLLAMA_BASE_URL", mode="before")
    @classmethod
    def strip_ollama_slash(cls, v: str) -> str:
        return str(v).rstrip("/")

    @model_validator(mode="after")
    def validate_feature_flags_for_env(self) -> "AppSettings":
        if self.ENV == Environment.PRODUCTION:
            flags = FeatureFlagConfig()
            errors: list[str] = []

            if flags.ENABLE_SWAGGER_UI:
                errors.append("FEATURE_ENABLE_SWAGGER_UI must be False in production.")
            if flags.ENABLE_DEBUG_LOGGING:
                errors.append("FEATURE_ENABLE_DEBUG_LOGGING must be False in production.")
            if "sqlite" in self.DATABASE_URL:
                errors.append("SQLite is not allowed in production. Use PostgreSQL or MSSQL.")

            parsed = urlparse(str(self.OLLAMA_BASE_URL))
            if parsed.hostname in ("localhost", "127.0.0.1") and flags.ENABLE_RAG:
                errors.append(
                    "OLLAMA_BASE_URL points to localhost in production with RAG enabled."
                )
            if errors:
                raise ValueError(
                    "Production config violations detected:\n"
                    + "\n".join(f"  • {e}" for e in errors)
                )
        return self

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
        d = self.model_dump()
        d["DATABASE_URL"] = re.sub(r"://[^@]+@", "://<redacted>@", d["DATABASE_URL"])
        return d


# ── Critical startup guard ────────────────────────────────────────────────

CRITICAL_VARS = [
    "DATABASE_URL",
    "INTERNAL_SERVICE_KEY",
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