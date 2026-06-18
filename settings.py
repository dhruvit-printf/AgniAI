"""
AgniAI — Configuration Safety Layer
Pydantic Settings with full validation, environment profiles,
and startup-time failure guarantee.
"""

from __future__ import annotations

import re
import sys
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Optional
from urllib.parse import urlparse

from pydantic import (
    AnyHttpUrl,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────

class Environment(str, Enum):
    DEVELOPMENT = "development"
    TESTING     = "testing"
    PRODUCTION  = "production"


# ─────────────────────────────────────────────────────────────────────────────
# Sub-config: .NET API
# ─────────────────────────────────────────────────────────────────────────────

class DotNetAPIConfig(BaseSettings):
    """Settings for the .NET integration layer (GramBook / DOTNET_API_BASE_URL)."""

    model_config = SettingsConfigDict(
        env_prefix="DOTNET_API_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    BASE_URL: str = Field(
        ...,
        description="Root URL of the .NET API (e.g. http://localhost:5000)",
    )
    TIMEOUT_CONNECT: float = Field(
        default=5.0,
        ge=1.0,
        le=30.0,
        description="TCP connect timeout in seconds.",
    )
    TIMEOUT_READ: float = Field(
        default=30.0,
        ge=5.0,
        le=120.0,
        description="HTTP read timeout in seconds.",
    )
    MAX_RETRIES: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Max retry attempts on transient failures.",
    )

    @field_validator("BASE_URL", mode="before")
    @classmethod
    def strip_trailing_slash(cls, v: str) -> str:
        v = str(v).rstrip("/")
        parsed = urlparse(v)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"DOTNET_API_BASE_URL must use http or https, got: {parsed.scheme!r}")
        if not parsed.netloc:
            raise ValueError(f"DOTNET_API_BASE_URL has no host: {v!r}")
        return v




# ─────────────────────────────────────────────────────────────────────────────
# Sub-config: API Keys
# ─────────────────────────────────────────────────────────────────────────────

class APIKeysConfig(BaseSettings):
    """Secret credentials — never logged, never serialised to plain text."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    OLLAMA_API_KEY: Optional[SecretStr] = Field(
        default=None,
        description="Bearer token for Ollama (optional when running locally).",
    )
    INTERNAL_SERVICE_KEY: SecretStr = Field(
        ...,
        description="Shared secret between Flask and .NET for service-to-service auth.",
    )
    WHATSAPP_TOKEN: Optional[SecretStr] = Field(
        default=None,
        description="Meta WhatsApp Cloud API bearer token.",
    )
    WHATSAPP_VERIFY_TOKEN: Optional[SecretStr] = Field(
        default=None,
        description="Webhook verification token configured in Meta console.",
    )

    @field_validator("INTERNAL_SERVICE_KEY", mode="before")
    @classmethod
    def validate_key_strength(cls, v: str) -> str:
        raw = v.get_secret_value() if hasattr(v, "get_secret_value") else str(v)
        if len(raw) < 32:
            raise ValueError(
                "INTERNAL_SERVICE_KEY must be at least 32 characters. "
                "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
            )
        return v  # return original (may already be SecretStr)


# ─────────────────────────────────────────────────────────────────────────────
# Sub-config: Timeouts (global defaults)
# ─────────────────────────────────────────────────────────────────────────────

class TimeoutConfig(BaseSettings):
    """Global timeout/retry budget for every subsystem."""

    model_config = SettingsConfigDict(
        env_prefix="TIMEOUT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    LLM_INFERENCE: float = Field(
        default=60.0,
        ge=10.0,
        le=300.0,
        description="Max seconds to wait for Ollama inference response.",
    )
    RAG_RETRIEVAL: float = Field(
        default=10.0,
        ge=1.0,
        le=60.0,
        description="Max seconds for FAISS/BM25 retrieval pipeline.",
    )
    DB_QUERY: float = Field(
        default=15.0,
        ge=1.0,
        le=60.0,
        description="SQLAlchemy statement timeout.",
    )
    HEALTH_CHECK: float = Field(
        default=3.0,
        ge=0.5,
        le=10.0,
        description="Timeout for /health probe endpoints.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sub-config: Feature Flags
# ─────────────────────────────────────────────────────────────────────────────

class FeatureFlagConfig(BaseSettings):
    """Runtime toggles — safe to change without code deployment."""

    model_config = SettingsConfigDict(
        env_prefix="FEATURE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ENABLE_RAG: bool = Field(
        default=True,
        description="Toggle FAISS/BM25 retrieval-augmented generation.",
    )
    ENABLE_RERANKER: bool = Field(
        default=True,
        description="Toggle cross-encoder reranker on top of retrieved chunks.",
    )
    ENABLE_WHATSAPP_BOT: bool = Field(
        default=False,
        description="Activate WhatsApp webhook routes.",
    )
    ENABLE_ADMIN_CHATBOT: bool = Field(
        default=True,
        description="Activate admin-side chatbot interface.",
    )
    ENABLE_SWAGGER_UI: bool = Field(
        default=True,
        description="Expose /api/docs Swagger UI (disable in production).",
    )
    ENABLE_DEBUG_LOGGING: bool = Field(
        default=False,
        description="Emit verbose SQL/HTTP logs (never enable in production).",
    )
    STRICT_INTENT_CLASSIFICATION: bool = Field(
        default=True,
        description="Reject queries with intent confidence below MIN_SCORE.",
    )

    @model_validator(mode="after")
    def production_flag_guard(self) -> "FeatureFlagConfig":
        # Called from AppSettings after ENV is known — not standalone.
        # Validation deferred to AppSettings.validate_feature_flags_for_env().
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Root AppSettings — composes all sub-configs
# ─────────────────────────────────────────────────────────────────────────────

class AppSettings(BaseSettings):
    """
    Single source of truth for AgniAI.
    Loaded once at startup; any missing CRITICAL variable causes sys.exit(1).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # Never print secrets in repr
        hide_input_in_errors=True,
    )

    # ── Identity ────────────────────────────────────────────────────────────
    APP_NAME: str = Field(default="AgniAI", description="Application display name.")
    APP_VERSION: str = Field(default="1.0.0", pattern=r"^\d+\.\d+\.\d+$")
    ENV: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="Active environment profile.",
    )

    # ── Sub-configs (composed) ───────────────────────────────────────────────
    # These are loaded lazily via @property to avoid nested BaseSettings issues.

    # ── Database ─────────────────────────────────────────────────────────────
    DATABASE_URL: str = Field(
        ...,
        description="SQLAlchemy connection string. Use env-specific credentials.",
    )

    # ── Ollama / LLM ─────────────────────────────────────────────────────────
    OLLAMA_BASE_URL: AnyHttpUrl = Field(
        default="http://localhost:11434",
        description="Ollama server root URL.",
    )
    OLLAMA_MODEL: str = Field(
        default="llama3",
        description="Model tag to use for inference.",
    )
    MIN_SCORE: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity score for RAG chunk inclusion.",
    )

    # ── Server ────────────────────────────────────────────────────────────────
    HOST: str  = Field(default="0.0.0.0")
    PORT: int  = Field(default=5000, ge=1024, le=65535)
    WORKERS: int = Field(default=1, ge=1, le=32)

    # ── CORS ──────────────────────────────────────────────────────────────────
    CORS_ORIGINS: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173"],
        description="Allowed CORS origins. Lock down in production.",
    )

    # ────────────────────────────────────────────────────────────────────────
    # Validators
    # ────────────────────────────────────────────────────────────────────────

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
        """
        Production safety gates — fail fast if config is dangerous.
        """
        if self.ENV == Environment.PRODUCTION:
            flags = FeatureFlagConfig()
            errors: list[str] = []

            if flags.ENABLE_SWAGGER_UI:
                errors.append(
                    "FEATURE_ENABLE_SWAGGER_UI must be False in production."
                )
            if flags.ENABLE_DEBUG_LOGGING:
                errors.append(
                    "FEATURE_ENABLE_DEBUG_LOGGING must be False in production."
                )
            parsed = urlparse(str(self.OLLAMA_BASE_URL))
            if parsed.hostname in ("localhost", "127.0.0.1") and flags.ENABLE_RAG:
                errors.append(
                    "OLLAMA_BASE_URL points to localhost in production with RAG enabled. "
                    "Set a reachable inference server."
                )
            if "sqlite" in self.DATABASE_URL and self.ENV == Environment.PRODUCTION:
                errors.append(
                    "SQLite is not allowed in production. Use PostgreSQL or MSSQL."
                )

            if errors:
                raise ValueError(
                    "Production config violations detected:\n" +
                    "\n".join(f"  • {e}" for e in errors)
                )

        return self

    @model_validator(mode="after")
    def validate_workers_vs_env(self) -> "AppSettings":
        if self.ENV == Environment.TESTING and self.WORKERS > 1:
            raise ValueError(
                "WORKERS must be 1 in testing environment to avoid parallel test interference."
            )
        return self

    # ────────────────────────────────────────────────────────────────────────
    # Helpers
    # ────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Startup guard — called once from app factory
# ─────────────────────────────────────────────────────────────────────────────

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
        # Print to stderr so it appears in systemd/docker logs regardless
        # of logging config (which may not be initialised yet).
        print(
            "\n[FATAL] AgniAI startup aborted — missing critical environment variables:\n"
            + "\n".join(f"  ✗  {k}" for k in missing)
            + "\n\nSet the above variables in your .env file or environment before starting.\n",
            file=sys.stderr,
        )
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Cached singleton — one load per process
# ─────────────────────────────────────────────────────────────────────────────

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