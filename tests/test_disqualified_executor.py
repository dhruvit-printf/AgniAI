from __future__ import annotations

from unittest.mock import patch

import pytest

from sql_executor import execute_disqualified_query, execute_sql_query


def test_disqualified_summary_count():
    def mock_run(sql, params=(), max_rows=None):
        return ([{"TotalDisqualified": 5}], None)

    intent = {
        "category": "disqualified",
        "responseType": "Summary",
        "raw_query": "How many disqualified agniveers are there?",
    }

    with patch("sql_executor.run_readonly", side_effect=mock_run):
        res, err = execute_sql_query(intent=intent)

    assert err is None
    assert res["success"] is True
    assert res["records"] == [{"totalDisqualified": 5}]


def test_disqualified_detailed_list():
    def mock_run(sql, params=(), max_rows=None):
        return (
            [
                {
                    "Id": 1,
                    "AgniveerNo": "A0701882L",
                    "FullName": "HARMAN SINGH",
                    "PlatoonName": "Platoon 1",
                    "CompanyName": "Alpha Company",
                    "BatchName": "Batch 2026",
                    "DisqualifiedDate": "2026-03-15",
                    "Remarks": "Medical Unfit",
                }
            ],
            None,
        )

    intent = {
        "category": "disqualified",
        "responseType": "Detailed",
        "raw_query": "Show disqualified agniveers list",
    }

    with patch("sql_executor.run_readonly", side_effect=mock_run):
        res, err = execute_sql_query(intent=intent)

    assert err is None
    assert res["success"] is True
    assert len(res["records"]) == 1
    assert res["records"][0]["agniveerNo"] == "A0701882L"


def test_disqualified_with_filters():
    def mock_run(sql, params=(), max_rows=None):
        assert "a.BatchId = ?" in sql
        assert "p.CompanyId = ?" in sql
        return ([{"TotalDisqualified": 2}], None)

    intent = {
        "category": "disqualified",
        "responseType": "Summary",
        "batch_id": 1,
        "company_id": 5,
        "raw_query": "Count disqualified in batch 1 company 5",
    }

    with patch("sql_executor.run_readonly", side_effect=mock_run):
        res, err = execute_sql_query(intent=intent)

    assert err is None
    assert res["records"] == [{"totalDisqualified": 2}]


def test_disqualified_with_leave_filter():
    def mock_run(sql, params=(), max_rows=None):
        assert "AgniveerLeaveMaster" in sql
        return ([], None)

    intent = {
        "category": "disqualified",
        "responseType": "Detailed",
        "leave_type": "leave",
        "raw_query": "Show disqualified agniveers on leave",
    }

    with patch("sql_executor.run_readonly", side_effect=mock_run):
        res, err = execute_sql_query(intent=intent)

    assert err is None


def test_disqualified_with_date_range():
    def mock_run(sql, params=(), max_rows=None):
        assert "CAST(a.DisqualifiedDate AS DATE) >=" in sql
        assert "CAST(a.DisqualifiedDate AS DATE) <=" in sql
        return ([], None)

    intent = {
        "category": "disqualified",
        "responseType": "Detailed",
        "from_date": "2026-01-01",
        "to_date": "2026-06-30",
        "raw_query": "Disqualified between 2026-01-01 and 2026-06-30",
    }

    with patch("sql_executor.run_readonly", side_effect=mock_run):
        res, err = execute_sql_query(intent=intent)

    assert err is None
