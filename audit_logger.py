"""
audit_logger.py
===============
Enterprise audit logging for AgniAI.

Stores per-query metadata to a rotating JSON-lines audit log file.
NEVER stores: prompts, queryPlan, dotnetPayload, raw responses, API keys,
              internal reasoning, stack traces.

Schema (AuditLog):
  {
    "timestamp":      "2025-01-15T10:23:45.123Z",
    "trace_id":       "abc123",
    "session_id":     "admin-default",
    "query_type":     "simple",
    "query_duration": 342.5,          // ms
    "success":        true,
    "error_type":     null,           // e.g. "dotnet_error", "llm_error"
    "username":       "Officer",      // if available
    "admin_name":     "Maj Sharma"    // if available
  }

Retention:
  - AUDIT_LOG_RETENTION_DAYS controls how many days of files are kept.
  - Rotation happens by size (10 MB) with up to AUDIT_LOG_BACKUP_COUNT backups.
  - Files are named: audit.log, audit.log.1, audit.log.2, …

Log rotation policy:
  RotatingFileHandler: maxBytes=10MB, backupCount=AUDIT_LOG_BACKUP_COUNT (default 30)
  Combined with OS-level cron or logrotate for date-based retention.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Optional
from contextvars import ContextVar

# ── Config ─────────────────────────────────────────────────────────────────
AUDIT_LOG_FILE = os.getenv("AUDIT_LOG_FILE", "audit.log")
AUDIT_LOG_MAX_BYTES = int(
    os.getenv("AUDIT_LOG_MAX_BYTES", str(10 * 1024 * 1024))
)  # 10 MB
AUDIT_LOG_BACKUP_COUNT = int(os.getenv("AUDIT_LOG_BACKUP_COUNT", "30"))
AUDIT_LOG_RETENTION_DAYS = int(os.getenv("AUDIT_LOG_RETENTION_DAYS", "90"))

# ── Audit logger instance ──────────────────────────────────────────────────
_audit_logger: Optional[logging.Logger] = None
_audit_question_var: ContextVar[str] = ContextVar("audit_question", default="")
_audit_intent_var: ContextVar[Dict[str, Any]] = ContextVar("audit_intent", default={})
_audit_written_var: ContextVar[bool] = ContextVar("audit_written", default=False)


def _get_audit_logger() -> logging.Logger:
    global _audit_logger
    if _audit_logger is not None:
        return _audit_logger

    logger = logging.getLogger("agniai.audit")
    logger.setLevel(logging.INFO)
    logger.propagate = False  # Don't bubble up to root logger

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


# ── Schema ─────────────────────────────────────────────────────────────────


def set_audit_context(*, question: str, intent: Optional[Dict[str, Any]] = None) -> None:
    _audit_question_var.set((question or "").strip())
    _audit_intent_var.set(dict(intent or {}))
    _audit_written_var.set(False)


def reset_audit_context() -> None:
    _audit_question_var.set("")
    _audit_intent_var.set({})
    _audit_written_var.set(False)


# ── Public API ─────────────────────────────────────────────────────────────


def write_audit_log(
    *,
    question: Optional[str] = None,
    intent: Optional[Dict[str, Any]] = None,
    **_ignored: Any,
) -> None:
    """
    Write one audit log entry.
    Call this at the end of every admin pipeline execution.
    """
    try:
        from feature_flags import flags

        if not flags.ENABLE_AUDIT_LOGGING:
            return
    except ImportError:
        pass  # feature_flags not available, proceed

    try:
        if question is not None or intent is not None:
            set_audit_context(question=question or "", intent=intent or {})
        if _audit_written_var.get():
            return
        current_intent = dict(_audit_intent_var.get() or {})
        entry = {
            "question": _audit_question_var.get() or "",
            "intent": current_intent,
            "type": current_intent.get("type") or "",
        }
        _get_audit_logger().info(json.dumps(entry, ensure_ascii=False))
        _audit_written_var.set(True)
    except Exception as exc:
        logging.getLogger(__name__).warning("Audit log write failed: %s", exc)


def purge_old_audit_logs() -> int:
    """
    Delete audit log backup files older than AUDIT_LOG_RETENTION_DAYS.
    Returns number of files deleted.
    Call this from a nightly cron or a startup hook.
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
        except OSError:
            pass
    return deleted
