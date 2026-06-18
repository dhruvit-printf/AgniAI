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
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Optional


# ── Config ─────────────────────────────────────────────────────────────────
AUDIT_LOG_FILE        = os.getenv("AUDIT_LOG_FILE",         "audit.log")
AUDIT_LOG_MAX_BYTES   = int(os.getenv("AUDIT_LOG_MAX_BYTES", str(10 * 1024 * 1024)))  # 10 MB
AUDIT_LOG_BACKUP_COUNT = int(os.getenv("AUDIT_LOG_BACKUP_COUNT", "30"))
AUDIT_LOG_RETENTION_DAYS = int(os.getenv("AUDIT_LOG_RETENTION_DAYS", "90"))

# ── Audit logger instance ──────────────────────────────────────────────────
_audit_logger: Optional[logging.Logger] = None


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

class AuditLog:
    """
    Value object representing one audit log entry.
    All fields are validated and sanitized on construction.
    """

    # Fields that must NEVER appear in audit logs
    _FORBIDDEN_KEYS = frozenset({
        "prompt", "prompts", "queryPlan", "dotnetPayload", "rawFragment",
        "intentResult", "operations", "reasoning", "raw_response",
        "api_key", "password", "secret", "token",
        "traceback", "stack_trace", "exception_detail",
    })

    __slots__ = (
        "timestamp", "trace_id", "session_id", "query_type",
        "query_duration", "success", "error_type", "username", "admin_name",
    )

    def __init__(
        self,
        *,
        trace_id: str,
        session_id: str,
        query_type: str,
        query_duration: float,
        success: bool,
        error_type: Optional[str] = None,
        username: Optional[str] = None,
        admin_name: Optional[str] = None,
    ) -> None:
        self.timestamp      = datetime.now(timezone.utc).isoformat()
        self.trace_id       = str(trace_id)[:64]
        self.session_id     = str(session_id)[:128]
        self.query_type     = str(query_type)[:64]
        self.query_duration = round(float(query_duration), 2)
        self.success        = bool(success)
        self.error_type     = str(error_type)[:64] if error_type else None
        self.username       = str(username)[:128] if username else None
        self.admin_name     = str(admin_name)[:128] if admin_name else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp":      self.timestamp,
            "trace_id":       self.trace_id,
            "session_id":     self.session_id,
            "query_type":     self.query_type,
            "query_duration": self.query_duration,
            "success":        self.success,
            "error_type":     self.error_type,
            "username":       self.username,
            "admin_name":     self.admin_name,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


# ── Public API ─────────────────────────────────────────────────────────────

def write_audit_log(
    *,
    trace_id: str,
    session_id: str,
    query_type: str,
    query_duration: float,
    success: bool,
    error_type: Optional[str] = None,
    username: Optional[str] = None,
    admin_name: Optional[str] = None,
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
        entry = AuditLog(
            trace_id=trace_id,
            session_id=session_id,
            query_type=query_type,
            query_duration=query_duration,
            success=success,
            error_type=error_type,
            username=username,
            admin_name=admin_name,
        )
        _get_audit_logger().info(entry.to_json())
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