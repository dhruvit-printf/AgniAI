"""
tests/test_audit_logger.py
==========================
Unit tests for audit_logger.py — structured audit trail system.
"""

import json
from unittest.mock import MagicMock, patch
import pytest

from audit_logger import (
    set_audit_context,
    reset_audit_context,
    write_audit_log,
    purge_old_audit_logs,
)


def test_set_and_write_audit_log():
    with patch("audit_logger._get_audit_logger") as mock_get_logger:
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        set_audit_context(
            question="Show attendance",
            intent={"category": "Attendance", "type": "Bar Chart"},
        )
        write_audit_log()

        mock_logger.info.assert_called_once()
        logged_json = mock_logger.info.call_args[0][0]
        parsed = json.loads(logged_json)
        assert parsed["question"] == "Show attendance"
        assert parsed["intent"]["category"] == "Attendance"
        assert parsed["type"] == "Bar Chart"


def test_write_audit_log_with_args():
    with patch("audit_logger._get_audit_logger") as mock_get_logger:
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        write_audit_log(
            question="Show performance",
            intent={"category": "Performance", "type": "Tabular"},
        )

        mock_logger.info.assert_called_once()
        logged_json = mock_logger.info.call_args[0][0]
        parsed = json.loads(logged_json)
        assert parsed["question"] == "Show performance"
        assert parsed["intent"]["category"] == "Performance"
        assert parsed["type"] == "Tabular"


def test_purge_old_logs():
    result = purge_old_audit_logs()
    assert isinstance(result, int)
    assert result >= 0
