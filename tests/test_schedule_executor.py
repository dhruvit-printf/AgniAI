from __future__ import annotations

from unittest.mock import patch

import pytest

from sql_executor import (
    build_schedule_sql,
    execute_sql_query,
    resolve_company_id_from_agniveer,
    resolve_company_id_from_name,
    resolve_company_id_from_platoon,
)


def test_build_schedule_sql():
    sql, params = build_schedule_sql(company_id=1, date="2026-07-22", top_n=50)
    assert "s.CompanyId = ?" in sql
    assert "CAST(s.ScheduleDate AS DATE) = CAST(? AS DATE)" in sql
    assert params == [1, "2026-07-22"]


def test_schedule_bytoday_execution():
    def mock_run(sql, params=(), max_rows=None):
        return (
            [
                {
                    "ScheduleId": 1,
                    "CompanyId": 10,
                    "CompanyName": "Bravo Company",
                    "ScheduleDate": "2026-07-22",
                    "Pd": 1,
                    "TimeRange": "06:00 - 07:00",
                    "Code": "BPET",
                    "Type": "Physical",
                    "Details": "Morning BPET",
                    "Location": "Ground A",
                    "Resp": "Instructor A",
                }
            ],
            None,
        )

    intent = {
        "category": "Schedule",
        "operation": "bytoday",
        "raw_query": "Show schedule for today",
    }

    with patch("sql_executor.run_readonly", side_effect=mock_run):
        res, err = execute_sql_query(intent=intent)

    assert err is None
    assert res["success"] is True
    assert len(res["records"]) == 1
    record = res["records"][0]
    assert record["scheduleId"] == 1
    assert record["companyName"] == "Bravo Company"
    assert record["code"] == "BPET"


def test_company_resolution_helpers():
    def mock_run(sql, params=(), max_rows=None):
        return ([{"CompanyId": 3}], None)

    with patch("sql_executor.run_readonly", side_effect=mock_run):
        assert resolve_company_id_from_agniveer("A0701882L") == 3
        assert resolve_company_id_from_platoon(5) == 3
        assert resolve_company_id_from_name("Alpha") == 3
