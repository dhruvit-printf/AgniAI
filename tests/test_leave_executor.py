from __future__ import annotations

from unittest.mock import patch
import pytest

from sql_executor import execute_sql_query, execute_leave_query, _build_leave_base_query


def test_leave_base_query_builder():
    intent = {
        "agniveer_no": "A0701882L",
        "company_id": 2,
        "from_date": "2026-01-01",
        "to_date": "2026-06-30"
    }
    where_sql, params = _build_leave_base_query(intent)
    assert "(a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)" in where_sql
    assert "a.IsActive = 1" in where_sql
    assert "LOWER(a.AgniveerNo) LIKE" in where_sql
    assert "EXISTS (SELECT 1 FROM PlatoonMaster p WHERE p.Id = a.PlatoonId AND p.CompanyId = ?)" in where_sql
    assert "CAST(l.FromDate AS DATE) >=" in where_sql
    assert "CAST(l.ToDate AS DATE) <=" in where_sql
    assert params == ["A0701882L", 2, "2026-01-01", "2026-06-30"]


def test_execute_leave_current_summary():
    def mock_run(sql, params=(), max_rows=None):
        assert "ISNULL(l.IsAbscondedLeave, 0) != 1" in sql
        return ([{
            "OnLeaveCount": 10,
            "AnnualLeave": 5,
            "MedicalLeave": 3,
            "SickLeave": 2,
            "Hospitalized": 0,
            "ATTNC": 0,
            "EXPPG": 0
        }], None)

    intent = {
        "category": "Leave",
        "operation": "Current",
        "responseType": "Summary"
    }

    with patch("sql_executor.run_readonly", side_effect=mock_run):
        res, err = execute_sql_query(intent=intent)

    assert err is None
    assert res["success"] is True
    assert res["records"][0]["onLeaveCount"] == 10
    assert res["records"][0]["annualLeave"] == 5


def test_execute_leave_most_detailed():
    def mock_run(sql, params=(), max_rows=None):
        assert "SUM(" in sql
        assert "ORDER BY TotalLeaveDays DESC" in sql
        return ([{
            "agniveerNo": "A0701882L",
            "fullName": "HARMAN SINGH",
            "totalLeaveDays": 45
        }], None)

    intent = {
        "category": "Leave",
        "operation": "Most",
        "responseType": "Detailed",
        "number": 5
    }

    with patch("sql_executor.run_readonly", side_effect=mock_run):
        res, err = execute_sql_query(intent=intent)

    assert err is None
    assert res["success"] is True
    assert res["records"][0]["totalLeaveDays"] == 45


def test_execute_leave_least_noleave():
    def mock_run(sql, params=(), max_rows=None):
        assert "NOT EXISTS (" in sql
        return ([{"agniveerNo": "A0701900K", "fullName": "RAHUL KUMAR"}], None)

    intent = {
        "category": "Leave",
        "operation": "Least",
        "leave_type": "noleave",
        "responseType": "Detailed"
    }

    with patch("sql_executor.run_readonly", side_effect=mock_run):
        res, err = execute_sql_query(intent=intent)

    assert err is None
    assert res["success"] is True
    assert res["records"][0]["agniveerNo"] == "A0701900K"


def test_execute_leave_absconded():
    def mock_run(sql, params=(), max_rows=None):
        assert "l.IsAbscondedLeave = 1" in sql
        return ([{"totalAbsconded": 2}], None)

    intent = {
        "category": "Leave",
        "operation": "Absconded",
        "responseType": "Summary"
    }

    with patch("sql_executor.run_readonly", side_effect=mock_run):
        res, err = execute_sql_query(intent=intent)

    assert err is None
    assert res["success"] is True
    assert res["records"][0]["totalAbsconded"] == 2


def test_execute_leave_threshold():
    def mock_run(sql, params=(), max_rows=None):
        assert "ContinuousThreshold AS (" in sql
        assert "TotalThreshold AS (" in sql
        assert "BETWEEN 40 AND 44" in sql
        assert "BETWEEN 55 AND 59" in sql
        # {base_where} is embedded once per CTE (Continuous and Total), so
        # its placeholder(s) appear twice in the compiled SQL — params must
        # match 1:1 with those placeholders or pyodbc rejects the call with
        # a parameter-count mismatch. batch_id below puts exactly one
        # placeholder in base_where, so this fails loudly if a future change
        # goes back to passing it only once.
        assert sql.count("?") == len(params)
        return ([{
            "AgniveerNo": "A0701882L",
            "FullName": "HARMAN SINGH",
            "Reason": "Continuous 40-44 days"
        }], None)

    intent = {
        "category": "Leave",
        "operation": "Threshold",
        "responseType": "Summary",
        "batch_id": 1,
    }

    with patch("sql_executor.run_readonly", side_effect=mock_run):
        res, err = execute_sql_query(intent=intent)

    assert err is None
    assert res["success"] is True
    assert res["records"][0]["continuous40to44Count"] == 1


def test_execute_leave_history():
    def mock_run(sql, params=(), max_rows=None):
        assert "ORDER BY l.FromDate DESC" in sql
        return ([{"id": 101, "leaveType": "Annual", "leaveDays": 10}], None)

    intent = {
        "category": "Leave",
        "operation": "History",
        "agniveer_no": "A0701882L"
    }

    with patch("sql_executor.run_readonly", side_effect=mock_run):
        res, err = execute_sql_query(intent=intent)

    assert err is None
    assert res["success"] is True
    assert res["records"][0]["leaveType"] == "Annual"
