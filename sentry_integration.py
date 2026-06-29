"""
sentry_integration.py
=====================
Sentry error monitoring integration for AgniAI.

Controlled by ENABLE_SENTRY flag. Gracefully no-ops if sentry-sdk
is not installed or DSN is not set.

Features:
  - Captures exceptions with trace_id context
  - Captures performance metrics (transactions)
  - Sanitizes API keys, prompts, raw payloads before sending
  - Sets release version and environment tags
  - Never exposes Sentry details to the frontend

Before-send sanitization removes:
  - API keys / tokens
  - Prompts and raw user queries
  - .NET payloads
  - Internal metadata
  - Stack traces from user-visible responses

Example Sentry event structure (sanitized):
  {
    "event_id": "abc123",
    "level": "error",
    "environment": "production",
    "release": "agniai@1.0.0",
    "tags": {"trace_id": "xyz", "session_id": "abc"},
    "extra": {"query_type": "simple", "duration_ms": 342},
    "exception": {"values": [{"type": "RuntimeError", "value": "..."}]}
  }
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ── Keys that must NEVER reach Sentry ────────────────────────────────────
_SCRUB_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "x-api-key",
        "token",
        "secret",
        "password",
        "prompt",
        "prompts",
        "queryplan",
        "dotnetpayload",
        "rawfragment",
        "intentresult",
        "operations",
        "reasoning",
        "raw_response",
        "rawpayload",
        "internalmetadata",
    }
)

_SCRUB_VALUE_PATTERNS = (
    "sk-",  # OpenAI-style keys
    "Bearer ",  # Authorization headers
)

_INITIALIZED = False


def _sanitize_value(key: str, value: Any) -> Any:
    """Redact sensitive values by key name."""
    if isinstance(key, str) and key.lower() in _SCRUB_KEYS:
        return "[REDACTED]"
    if isinstance(value, str):
        for pattern in _SCRUB_VALUE_PATTERNS:
            if value.startswith(pattern):
                return "[REDACTED]"
    return value


def _sanitize_dict(data: Dict, depth: int = 0) -> Dict:
    """Recursively sanitize a dict, redacting sensitive keys."""
    if depth > 5:
        return data
    result = {}
    for k, v in data.items():
        sanitized_v = _sanitize_value(k, v)
        if isinstance(sanitized_v, dict):
            sanitized_v = _sanitize_dict(sanitized_v, depth + 1)
        elif isinstance(sanitized_v, list):
            sanitized_v = [
                _sanitize_dict(item, depth + 1) if isinstance(item, dict) else item
                for item in sanitized_v
            ]
        result[k] = sanitized_v
    return result


def before_send(event: Dict, hint: Dict) -> Optional[Dict]:
    """
    Sentry before_send hook — sanitizes the event before transmission.
    Returns None to drop the event, or the sanitized event to send.
    """
    try:
        # Sanitize extra/context data
        if "extra" in event:
            event["extra"] = _sanitize_dict(event.get("extra", {}))

        # Sanitize breadcrumbs
        breadcrumbs = event.get("breadcrumbs", {}).get("values", [])
        for crumb in breadcrumbs:
            if "data" in crumb and isinstance(crumb["data"], dict):
                crumb["data"] = _sanitize_dict(crumb["data"])

        # Sanitize request data
        request_data = event.get("request", {})
        if "data" in request_data and isinstance(request_data["data"], dict):
            request_data["data"] = _sanitize_dict(request_data["data"])
        if "headers" in request_data and isinstance(request_data["headers"], dict):
            request_data["headers"] = _sanitize_dict(request_data["headers"])

        # Sanitize user context
        if "user" in event:
            user = event["user"]
            # Keep only safe user fields
            safe_user = {}
            if "id" in user:
                safe_user["id"] = user["id"]
            if "username" in user:
                safe_user["username"] = user["username"]
            event["user"] = safe_user

        return event
    except Exception as exc:
        logger.warning("Sentry before_send sanitization failed: %s", exc)
        return None


def init_sentry() -> bool:
    """
    Initialize Sentry SDK. Called once at startup from app.py.
    Returns True if Sentry was successfully initialized.
    """
    global _INITIALIZED

    try:
        from feature_flags import flags

        if not flags.ENABLE_SENTRY:
            return False
    except ImportError:
        if os.getenv("ENABLE_SENTRY", "false").lower() not in ("1", "true", "yes"):
            return False

    dsn = os.getenv("SENTRY_DSN", "")
    if not dsn:
        logger.info("SENTRY_DSN not set — Sentry disabled.")
        return False

    try:
        import sentry_sdk
        from sentry_sdk.integrations.flask import FlaskIntegration
        from sentry_sdk.integrations.logging import LoggingIntegration

        sentry_sdk.init(
            dsn=dsn,
            environment=os.getenv("ENV", "development"),
            release=f"agniai@{os.getenv('APP_VERSION', '1.0.0')}",
            before_send=before_send,
            integrations=[
                FlaskIntegration(),
                LoggingIntegration(
                    level=logging.WARNING,  # Capture WARNING and above as breadcrumbs
                    event_level=logging.ERROR,  # Send ERROR+ as Sentry events
                ),
            ],
            traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.1")),
            # Never send PII
            send_default_pii=False,
        )
        _INITIALIZED = True
        logger.info("Sentry initialized for environment=%s", os.getenv("ENV"))
        return True
    except ImportError:
        logger.info("sentry-sdk not installed — Sentry disabled.")
        return False
    except Exception as exc:
        logger.warning("Sentry init failed: %s", exc)
        return False


def capture_exception(exc: Exception, *, trace_id: str = "N/A", **context) -> None:
    """
    Capture an exception in Sentry. No-op if Sentry is not initialized.
    Never raises.
    """
    if not _INITIALIZED:
        return
    try:
        import sentry_sdk

        with sentry_sdk.push_scope() as scope:
            scope.set_tag("trace_id", trace_id)
            for k, v in context.items():
                scope.set_extra(k, str(v))
            sentry_sdk.capture_exception(exc)
    except Exception:
        pass


def capture_message(message: str, level: str = "info", **context) -> None:
    """Capture a message in Sentry. No-op if not initialized."""
    if not _INITIALIZED:
        return
    try:
        import sentry_sdk

        with sentry_sdk.push_scope() as scope:
            for k, v in context.items():
                scope.set_extra(k, str(v))
            sentry_sdk.capture_message(message, level=level)
    except Exception:
        pass


# ── Sample sanitized Sentry event ────────────────────────────────────────
SAMPLE_SENTRY_EVENT = {
    "event_id": "abc123def456",
    "level": "error",
    "environment": "production",
    "release": "agniai@1.0.0",
    "tags": {
        "trace_id": "xyz789",
        "query_type": "simple",
    },
    "extra": {
        "query_type": "simple",
        "duration_ms": 342,
        "session_id": "admin-default",
        # NOTE: api_key, prompt, dotnetPayload are ALWAYS redacted
        "api_key": "[REDACTED]",
        "prompt": "[REDACTED]",
    },
    "exception": {
        "values": [
            {
                "type": "RuntimeError",
                "value": "Cannot connect to .NET backend.",
                "module": "dotnet_executor",
            }
        ]
    },
}
