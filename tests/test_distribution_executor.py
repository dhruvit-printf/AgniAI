from __future__ import annotations

from unittest.mock import patch
import pytest

from sql_executor import (
    execute_sql_query,
    execute_distribution_query,
    _build_distribution_base_scope,
    _get_latest_distribution_id,
)


def test_distribution_base_scope_builder():
    intent = {"batch_id": 1, "company_id": 3}
    where_sql, params = _build_distribution_base_scope(intent)
    assert "(a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)" in where_sql
    assert "a.BatchId = ?" in where_sql
    assert (
        "EXISTS (SELECT 1 FROM PlatoonMaster p WHERE p.Id = a.PlatoonId AND p.CompanyId = ?)"
        in where_sql
    )
    assert params == [1, 3]


def test_get_latest_distribution_id():
    def mock_run(sql, params=(), max_rows=None):
        return ([{"DistributionId": 105}], None)

    with patch("sql_executor.run_readonly", side_effect=mock_run):
        dist_id = _get_latest_distribution_id()
    assert dist_id == 105


def test_execute_distribution_latest_summary():
    def mock_run(sql, params=(), max_rows=None):
        if "MAX(DistributionId)" in sql:
            return ([{"DistributionId": 105}], None)
        if "TOP 1 InsertedDate" in sql:
            return ([{"DistributionDate": "2026-07-01"}], None)
        return (
            [
                {"TeamId": 1, "TeamName": "Alpha Team", "MemberCount": 15},
                {"TeamId": 2, "TeamName": "Bravo Team", "MemberCount": 12},
            ],
            None,
        )

    intent = {
        "category": "Distribution",
        "operation": "Latest",
        "responseType": "Summary",
    }

    with patch("sql_executor.run_readonly", side_effect=mock_run):
        res, err = execute_sql_query(intent=intent)

    assert err is None
    assert res["success"] is True
    assert res["records"][0]["distributionId"] == 105


def test_execute_distribution_by_unit():
    def mock_run(sql, params=(), max_rows=None):
        if "DistributionMaster" in sql:
            return ([{"UnitId": 2}], None)
        return (
            [{"AgniveerNo": "A0701882L", "FullName": "HARMAN SINGH", "Rank": 1}],
            None,
        )

    intent = {
        "category": "Distribution",
        "operation": "ByUnit",
        "unit_name": "Alpha Team",
    }

    with patch("sql_executor.run_readonly", side_effect=mock_run):
        res, err = execute_sql_query(intent=intent)

    assert err is None
    assert res["success"] is True
    assert res["records"][0]["agniveerNo"] == "A0701882L"


def test_execute_distribution_unassigned():
    def mock_run(sql, params=(), max_rows=None):
        assert "NOT EXISTS" in sql
        return ([{"AgniveerNo": "A0701999X", "FullName": "KULDEEP YADAV"}], None)

    intent = {"category": "Distribution", "operation": "Unassigned"}

    with patch("sql_executor.run_readonly", side_effect=mock_run):
        res, err = execute_sql_query(intent=intent)

    assert err is None
    assert res["success"] is True
    assert res["records"][0]["agniveerNo"] == "A0701999X"


def test_execute_distribution_top_unit():
    def mock_run(sql, params=(), max_rows=None):
        if "MAX(DistributionId)" in sql:
            return ([{"DistributionId": 105}], None)
        return ([{"TeamId": 1, "TeamName": "Alpha Team", "AgniveerCount": 28}], None)

    intent = {"category": "Distribution", "operation": "TopUnit"}

    with patch("sql_executor.run_readonly", side_effect=mock_run):
        res, err = execute_sql_query(intent=intent)

    assert err is None
    assert res["success"] is True
    assert res["records"][0]["teamName"] == "Alpha Team"
    assert res["records"][0]["agniveerCount"] == 28


def test_execute_distribution_history():
    def mock_run(sql, params=(), max_rows=None):
        if "AgniveerMaster" in sql:
            return ([{"Id": 50}], None)
        return ([{"DistributionId": 105, "UnitName": "Alpha Team", "Rank": 2}], None)

    intent = {
        "category": "Distribution",
        "operation": "History",
        "agniveer_no": "A0701882L",
    }

    with patch("sql_executor.run_readonly", side_effect=mock_run):
        res, err = execute_sql_query(intent=intent)

    assert err is None
    assert res["success"] is True
    assert res["records"][0]["unitName"] == "Alpha Team"
    assert res["records"][0]["rank"] == 2
