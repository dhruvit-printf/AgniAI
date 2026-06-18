"""
tests/test_audit_logger.py
Unit tests for audit_logger.py — structured audit trail system.
"""

import json
import os
import time
from unittest.mock import MagicMock, patch

import pytest


class TestAuditLogSchema:
    def test_audit_log_to_dict_has_required_fields(self):
        from audit_logger import AuditLog

        entry = AuditLog(
            trace_id="trace-001",
            session_id="sess-001",
            query_type="simple",
            query_duration=123.4,
            success=True,
        )
        d = entry.to_dict()
        for field in (
            "timestamp",
            "trace_id",
            "session_id",
            "query_type",
            "query_duration",
            "success",
        ):
            assert field in d, f"Missing field: {field}"

    def test_audit_log_timestamp_is_iso_format(self):
        from audit_logger import AuditLog

        entry = AuditLog(
            trace_id="t",
            session_id="s",
            query_type="q",
            query_duration=1.0,
            success=True,
        )
        # ISO format: 2025-01-15T10:23:45.123456+00:00
        assert "T" in entry.timestamp
        assert entry.timestamp.endswith("+00:00") or "Z" in entry.timestamp

    def test_audit_log_to_json_is_valid_json(self):
        from audit_logger import AuditLog

        entry = AuditLog(
            trace_id="t",
            session_id="s",
            query_type="simple",
            query_duration=50.0,
            success=True,
        )
        parsed = json.loads(entry.to_json())
        assert parsed["trace_id"] == "t"

    def test_audit_log_error_type_stored(self):
        from audit_logger import AuditLog

        entry = AuditLog(
            trace_id="t",
            session_id="s",
            query_type="error",
            query_duration=10.0,
            success=False,
            error_type="dotnet_error",
        )
        assert entry.to_dict()["error_type"] == "dotnet_error"

    def test_audit_log_none_error_type_when_success(self):
        from audit_logger import AuditLog

        entry = AuditLog(
            trace_id="t",
            session_id="s",
            query_type="simple",
            query_duration=10.0,
            success=True,
        )
        assert entry.to_dict()["error_type"] is None

    def test_audit_log_truncates_long_trace_id(self):
        from audit_logger import AuditLog

        long_id = "x" * 200
        entry = AuditLog(
            trace_id=long_id,
            session_id="s",
            query_type="q",
            query_duration=1.0,
            success=True,
        )
        assert len(entry.trace_id) <= 64

    def test_audit_log_rounds_duration(self):
        from audit_logger import AuditLog

        entry = AuditLog(
            trace_id="t",
            session_id="s",
            query_type="q",
            query_duration=123.456789,
            success=True,
        )
        assert entry.query_duration == 123.46

    def test_audit_log_forbidden_keys_defined(self):
        from audit_logger import AuditLog

        forbidden = AuditLog._FORBIDDEN_KEYS
        assert "prompt" in forbidden
        assert "queryPlan" in forbidden
        assert "dotnetPayload" in forbidden
        assert "api_key" in forbidden
        assert "traceback" in forbidden


class TestWriteAuditLog:
    def test_write_audit_log_calls_logger(self):
        """write_audit_log() must write a JSON line to the audit logger."""
        with patch("audit_logger._get_audit_logger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            from audit_logger import write_audit_log

            write_audit_log(
                trace_id="trace-123",
                session_id="sess-456",
                query_type="simple",
                query_duration=200.0,
                success=True,
            )

        mock_logger.info.assert_called_once()
        logged_json = mock_logger.info.call_args[0][0]
        parsed = json.loads(logged_json)
        assert parsed["trace_id"] == "trace-123"
        assert parsed["success"] is True

    def test_write_audit_log_skipped_when_flag_disabled(self):
        """write_audit_log() must be a no-op when ENABLE_AUDIT_LOGGING=false."""
        with patch("audit_logger._get_audit_logger") as mock_get_logger, patch(
            "feature_flags.flags"
        ) as mock_flags:
            mock_flags.ENABLE_AUDIT_LOGGING = False

            from audit_logger import write_audit_log

            write_audit_log(
                trace_id="t",
                session_id="s",
                query_type="q",
                query_duration=1.0,
                success=True,
            )

        mock_get_logger.assert_not_called()

    def test_write_audit_log_does_not_raise_on_logger_failure(self):
        """write_audit_log() must never raise — audit failures are silent."""
        with patch("audit_logger._get_audit_logger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_logger.info.side_effect = OSError("disk full")
            mock_get_logger.return_value = mock_logger

            from audit_logger import write_audit_log

            # Must not raise
            write_audit_log(
                trace_id="t",
                session_id="s",
                query_type="q",
                query_duration=1.0,
                success=True,
            )

    def test_write_audit_log_includes_error_type(self):
        with patch("audit_logger._get_audit_logger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            from audit_logger import write_audit_log

            write_audit_log(
                trace_id="t",
                session_id="s",
                query_type="error",
                query_duration=50.0,
                success=False,
                error_type="pipeline_exception",
            )

        logged = mock_logger.info.call_args[0][0]
        parsed = json.loads(logged)
        assert parsed["error_type"] == "pipeline_exception"
        assert parsed["success"] is False


class TestPurgeOldAuditLogs:
    def test_purge_returns_integer(self):
        from audit_logger import purge_old_audit_logs

        result = purge_old_audit_logs()
        assert isinstance(result, int)
        assert result >= 0
