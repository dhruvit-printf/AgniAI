from __future__ import annotations

from unittest.mock import patch
import pytest

from sql_executor import (
    execute_sql_query,
    execute_attendance_query,
    _build_attendance_base_scope,
    _resolve_attendance_dates,
)


def test_attendance_base_scope_builder():
    intent = {"company_id": 2, "platoon_id": 5}
    where_sql, params = _build_attendance_base_scope(intent)
    assert "(a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)" in where_sql
    assert "a.PlatoonId = ?" in where_sql
    assert (
        "EXISTS (SELECT 1 FROM PlatoonMaster p WHERE p.Id = a.PlatoonId AND p.CompanyId = ?)"
        in where_sql
    )
    assert params == [5, 2]


def test_resolve_attendance_dates():
    from_d, to_d = _resolve_attendance_dates(
        "daily", {"from_date": "2026-07-01", "to_date": "2026-07-31"}
    )
    assert from_d == "2026-07-01"
    assert to_d == "2026-07-31"


def test_execute_attendance_summary():
    def mock_run(sql, params=(), max_rows=None):
        assert "TotalActive" in sql
        assert "PresentCount" in sql
        return ([{"TotalActive": 100, "PresentCount": 92}], None)

    intent = {
        "category": "Attendance",
        "operation": "Summary",
        "responseType": "Summary",
    }

    with patch("sql_executor.run_readonly", side_effect=mock_run):
        res, err = execute_sql_query(intent=intent)

    assert err is None
    assert res["success"] is True
    assert res["records"][0]["totalActive"] == 100
    assert res["records"][0]["presentCount"] == 92
    assert res["records"][0]["absentCount"] == 8
    assert res["records"][0]["presentPct"] == 92.0


def test_execute_attendance_daily():
    def mock_run(sql, params=(), max_rows=None):
        assert "AttendanceDate" in sql
        assert "IsPresent" in sql
        return (
            [
                {
                    "AttendanceDate": "2026-07-22",
                    "IsPresent": 1,
                    "AgniveerNo": "A0701882L",
                    "FullName": "HARMAN SINGH",
                }
            ],
            None,
        )

    intent = {
        "category": "Attendance",
        "operation": "Daily",
        "agniveer_no": "A0701882L",
    }

    with patch("sql_executor.run_readonly", side_effect=mock_run):
        res, err = execute_sql_query(intent=intent)

    assert err is None
    assert res["success"] is True
    assert res["records"][0]["agniveerNo"] == "A0701882L"
    assert res["records"][0]["isPresent"] == 1


def test_execute_attendance_weekly():
    def mock_run(sql, params=(), max_rows=None):
        assert "WeekStart" in sql
        assert "Present" in sql
        assert "Absent" in sql
        return (
            [
                {
                    "AgniveerNo": "A0701882L",
                    "FullName": "HARMAN SINGH",
                    "WeekStart": "2026-07-20",
                    "Present": 6,
                    "Absent": 1,
                }
            ],
            None,
        )

    intent = {"category": "Attendance", "operation": "Weekly"}

    with patch("sql_executor.run_readonly", side_effect=mock_run):
        res, err = execute_sql_query(intent=intent)

    assert err is None
    assert res["success"] is True
    assert res["records"][0]["present"] == 6
    assert res["records"][0]["absent"] == 1


def test_execute_attendance_monthly():
    def mock_run(sql, params=(), max_rows=None):
        assert "Month" in sql
        assert "Present" in sql
        return (
            [
                {
                    "AgniveerNo": "A0701882L",
                    "FullName": "HARMAN SINGH",
                    "Month": "07-2026",
                    "Present": 25,
                    "Absent": 2,
                }
            ],
            None,
        )

    intent = {"category": "Attendance", "operation": "Monthly"}

    with patch("sql_executor.run_readonly", side_effect=mock_run):
        res, err = execute_sql_query(intent=intent)

    assert err is None
    assert res["success"] is True
    assert res["records"][0]["month"] == "07-2026"
    assert res["records"][0]["present"] == 25


def test_execute_attendance_individual():
    def mock_run(sql, params=(), max_rows=None):
        assert "DailyStatus" in sql
        return (
            [
                {
                    "Date": "2026-07-22",
                    "IsPresent": 1,
                    "AgniveerNo": "A0701882L",
                    "FullName": "HARMAN SINGH",
                }
            ],
            None,
        )

    intent = {
        "category": "Attendance",
        "operation": "Individual",
        "agniveer_no": "A0701882L",
    }

    with patch("sql_executor.run_readonly", side_effect=mock_run):
        res, err = execute_sql_query(intent=intent)

    assert err is None
    assert res["success"] is True
    assert res["records"][0]["date"] == "2026-07-22"
    assert res["records"][0]["isPresent"] == 1
