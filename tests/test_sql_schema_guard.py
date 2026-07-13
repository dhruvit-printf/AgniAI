"""
tests/test_sql_schema_guard.py
================================
Unit tests for sql_schema_guard.py — dynamic allowlist + drift guard
(Task 8). Uses a mocked INFORMATION_SCHEMA result; never connects to a
real database.
"""

from __future__ import annotations

from unittest.mock import patch

from sql_schema_guard import (
    CURATED_TABLES,
    build_allowlist,
    check_schema_drift,
    fetch_live_schema,
    run_schema_guard,
)


def test_build_allowlist_excludes_denied_table_even_if_present_live():
    live_schema = {
        "AgniveerMaster": {"Id", "AgniveerNo", "FullName"},
        "LoginToken": {"Id", "Token", "RefreshToken"},
    }
    allowlist = build_allowlist(live_schema)
    assert "LoginToken" not in allowlist
    assert "AgniveerMaster" in allowlist


def test_build_allowlist_excludes_denied_column_even_if_present_live():
    live_schema = {
        "UserMaster": {"Id", "Username", "Password"},
    }
    allowlist = build_allowlist(live_schema)
    assert "Password" not in allowlist["UserMaster"]
    assert "Username" in allowlist["UserMaster"]


def test_check_schema_drift_warns_on_missing_curated_table():
    live_schema = {t: {"Id"} for t in CURATED_TABLES if t != "EquipmentMaster"}
    warnings = check_schema_drift(live_schema)
    assert any("EquipmentMaster" in w for w in warnings)


def test_check_schema_drift_warns_on_extra_live_table():
    live_schema = {t: {"Id"} for t in CURATED_TABLES}
    live_schema["SomeNewTable"] = {"Id"}
    warnings = check_schema_drift(live_schema)
    assert any("SomeNewTable" in w for w in warnings)


def test_check_schema_drift_no_warnings_when_in_sync():
    live_schema = {t: {"Id"} for t in CURATED_TABLES}
    warnings = check_schema_drift(live_schema)
    assert warnings == []


def test_check_schema_drift_never_raises_on_empty_schema():
    # Must warn (missing every curated table), not raise.
    warnings = check_schema_drift({})
    assert isinstance(warnings, list)
    assert len(warnings) > 0


def test_fetch_live_schema_returns_none_without_connection_string():
    with patch("sql_schema_guard.SQL_READONLY_CONN", ""):
        assert fetch_live_schema() is None


def test_run_schema_guard_never_raises_when_db_unreachable():
    with patch("sql_schema_guard.fetch_live_schema", return_value=None):
        result = run_schema_guard()
    assert result is None


def test_run_schema_guard_builds_allowlist_and_logs_drift(caplog):
    live_schema = {t: {"Id"} for t in CURATED_TABLES if t != "EquipmentMaster"}
    live_schema["LoginToken"] = {"Id", "Token"}

    with patch("sql_schema_guard.fetch_live_schema", return_value=live_schema):
        import logging

        with caplog.at_level(logging.WARNING, logger="sql_schema_guard"):
            allowlist = run_schema_guard()

    assert allowlist is not None
    assert "LoginToken" not in allowlist
    assert any("EquipmentMaster" in rec.message for rec in caplog.records)
