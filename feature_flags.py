"""
feature_flags.py
================
Centralized feature flag system for AgniAI.
All flags are read from environment variables via pydantic-settings.
No if-statements scattered across the codebase — use this module as the
single source of truth.

Usage:
    from feature_flags import flags

    if flags.ENABLE_REPORTS:
        ...

Changing an env var (e.g. ENABLE_REPORTS=false) is sufficient to toggle
a feature at runtime — no code changes required.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class FeatureFlags(BaseSettings):
    """
    All feature flags for AgniAI.
    Prefixed with ENABLE_ in environment variables.
    Defaults are production-safe (disabled where risk exists).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Core pipeline features ─────────────────────────────────────────────
    ENABLE_REPORTS: bool = True
    """Generate LLM-powered analysis/conclusion in report_generator."""

    ENABLE_OLLAMA: bool = True
    """Enable Ollama LLM inference. Disable to return fallback responses."""

    ENABLE_STREAMING: bool = True
    """Enable SSE token streaming on /api/chat."""

    ENABLE_WEBSOCKET: bool = True
    """Enable Socket.IO WebSocket transport for admin chatbot."""

    # ── Observability ──────────────────────────────────────────────────────
    ENABLE_METRICS: bool = True
    """Expose Prometheus metrics at /metrics."""

    ENABLE_HEALTH_ENDPOINT: bool = True
    """Expose /api/health and /api/admin/health endpoints."""

    ENABLE_AUDIT_LOGGING: bool = True
    """Write structured audit logs for every admin query."""

    ENABLE_OPENTELEMETRY: bool = False
    """Instrument pipeline with OpenTelemetry distributed tracing."""

    ENABLE_SENTRY: bool = False
    """Send exceptions and performance data to Sentry."""

    ENABLE_PROMETHEUS: bool = True
    """Register prometheus_client metrics (requires prometheus_client installed)."""

    def degrade_gracefully(self, feature: str) -> bool:
        """
        Return the flag value for *feature*, logging a warning if the flag
        is False so operators know the system is running in degraded mode.
        """
        import logging

        val = getattr(self, feature, False)
        if not val:
            logging.getLogger(__name__).info(
                "Feature %s is DISABLED — running in degraded mode.", feature
            )
        return val


@lru_cache(maxsize=1)
def get_flags() -> FeatureFlags:
    """Singleton accessor — one load per process."""
    return FeatureFlags()


# Module-level convenience alias
flags = get_flags()
