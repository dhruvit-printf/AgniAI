"""
audit_logger.py
===============
Minimal audit logging for AgniAI.

Captures only what matters for query traceability:
  - question         : the raw officer question
  - query_type       : one of simple | multi_independent | cross_filter | comparison
  - dotnet_payload   : the exact payload sent to the .NET backend
  - visualization_type: the viz type resolved for this query

Schema (one JSON line per query):
  {
    "timestamp":          "2025-01-15T10:23:45.123Z",
    "trace_id":           "abc123",
    "query_type":         "simple",
    "question":           "Show attendance for Batch 23",
    "dotnet_payload":     { ... },
    "visualization_type": "table"
  }

Rotation: size-based (10 MB), up to 30 backup files.
Retention: controlled by AUDIT_LOG_RETENTION_DAYS (default 90 days).
"""

from __future__ import annotations

import json
import logging
import os
from contextvars import ContextVar
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Optional

# ── Config ──────────────────────────────────────────────────────────────────
AUDIT_LOG_FILE = os.getenv("AUDIT_LOG_FILE", "audit.log")
AUDIT_LOG_MAX_BYTES = int(
    os.getenv("AUDIT_LOG_MAX_BYTES", str(10 * 1024 * 1024))
)  # 10 MB
AUDIT_LOG_BACKUP_COUNT = int(os.getenv("AUDIT_LOG_BACKUP_COUNT", "30"))
AUDIT_LOG_RETENTION_DAYS = int(os.getenv("AUDIT_LOG_RETENTION_DAYS", "90"))

# Only these query types are audited — anything else is ignored
AUDITED_QUERY_TYPES = frozenset(
    {"simple", "multi_independent", "cross_filter", "comparison"}
)

# ── Internal logger instance ─────────────────────────────────────────────────
_audit_logger: Optional[logging.Logger] = None
_audit_context: ContextVar[Dict[str, Any]] = ContextVar("_audit_context", default={})


def _get_audit_logger() -> logging.Logger:
    global _audit_logger
    if _audit_logger is not None:
        return _audit_logger

    logger = logging.getLogger("agniai.audit")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    handler = RotatingFileHandler(
        AUDIT_LOG_FILE,
        maxBytes=AUDIT_LOG_MAX_BYTES,
        backupCount=AUDIT_LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)

    _audit_logger = logger
    return logger


# ── Public API ───────────────────────────────────────────────────────────────


def write_audit_log(
    *,
    trace_id: str = "",
    query_type: str = "simple",
    question: str = "",
    dotnet_payload: Any = None,
    visualization_type: str = "",
    **kwargs: Any,
) -> None:
    """
    Write one audit log entry.

    Call this once per admin pipeline execution, after the .NET call and
    visualization type have both been resolved.

    Only logs query types in AUDITED_QUERY_TYPES — silently skips others.
    Never raises — failures are logged as warnings to the root logger.
    """
    ctx = _audit_context.get()

    # Merge context if present
    trace_id = trace_id or ctx.get("trace_id") or ""
    query_type = query_type or ctx.get("query_type") or "simple"
    question = question or ctx.get("question") or ""
    dotnet_payload = dotnet_payload or ctx.get("dotnet_payload")
    visualization_type = visualization_type or ctx.get("visualization_type") or ""

    if query_type not in AUDITED_QUERY_TYPES:
        return

    try:
        from feature_flags import flags

        if not flags.ENABLE_AUDIT_LOGGING:
            return
    except ImportError:
        pass

    try:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trace_id": trace_id,
            "query_type": query_type,
            "question": (question or "").strip(),
            "dotnet_payload": dotnet_payload,
            "visualization_type": visualization_type or "",
        }

        # 1. Merge extra keys from arguments (high priority)
        for k, v in kwargs.items():
            if k not in entry or entry[k] in (None, "", [], {}):
                entry[k] = v

        # 2. Merge extra keys from context (lower priority)
        for k, v in ctx.items():
            if k not in entry or entry[k] in (None, "", [], {}):
                entry[k] = v

        # 3. Unpack intent sub-keys to the root level if present
        if "intent" in entry and isinstance(entry["intent"], dict):
            for sub_k, sub_v in entry["intent"].items():
                if sub_k not in entry:
                    entry[sub_k] = sub_v

        _get_audit_logger().info(json.dumps(entry, ensure_ascii=False, default=str))

    except Exception as exc:
        logging.getLogger(__name__).warning(
            "Audit log write failed (trace=%s): %s", trace_id, exc
        )


# ── Retention cleanup ────────────────────────────────────────────────────────


def purge_old_audit_logs() -> int:
    """
    Delete rotated audit log files older than AUDIT_LOG_RETENTION_DAYS.
    Returns number of files deleted.
    Call from a nightly cron or startup hook.
    """
    import glob
    import time

    cutoff = time.time() - (AUDIT_LOG_RETENTION_DAYS * 86400)
    deleted = 0

    for path in glob.glob(f"{AUDIT_LOG_FILE}.*"):
        try:
            if os.path.getmtime(path) < cutoff:
                os.remove(path)
                deleted += 1
        except OSError as exc:
            logging.getLogger(__name__).warning(
                "Failed to delete audit log %s: %s", path, exc
            )

    return deleted


def set_audit_context(**kwargs: Any) -> None:
    current = _audit_context.get().copy()
    current.update(kwargs)
    _audit_context.set(current)


def reset_audit_context() -> None:
    _audit_context.set({})
