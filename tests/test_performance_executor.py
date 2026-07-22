from __future__ import annotations

from unittest.mock import patch

import pytest

from performance_executor import execute_performance_query


@pytest.mark.parametrize(
    "operation,intent,expected_fragment",
    [
        ("Top", {"section": "BPET"}, "SUM(MarksObtained) AS BestTotal"),
        ("Bottom", {"section": "BPET"}, "ORDER BY BestTotal ASC"),
        ("Average", {"section": "BPET"}, "AVG(CAST(MarksObtained AS DECIMAL(18, 2)))"),
        ("BestAttempt", {"section": "BPET"}, "sa.IsBestAttempt = 1"),
        ("AttemptWise", {"section": "BPET"}, "AttemptTotal"),
        ("Trend", {"section": "BPET"}, "COUNT(DISTINCT AgniveerId) AS AgniveerCount"),
        ("Improvement", {"from_attempt": 1, "to_attempt": 2}, "AS Improvement"),
        # "Drop" is a reserved T-SQL keyword, so the generated column alias
        # is correctly bracket-quoted ("AS [Drop]") — bare "AS Drop" would
        # risk a SQL syntax error against the reserved word.
        ("Drop", {"from_attempt": 1, "to_attempt": 2}, "AS [Drop]"),
        ("Grading", {"section": "BPET"}, "CASE"),
        ("GradingSummary", {"section": "BPET"}, "GROUP BY SectionName, Grade"),
    ],
)
def test_supported_performance_operations_execute_locally(
    operation: str, intent: dict, expected_fragment: str
):
    payload = {"category": "Performance", "operation": operation, **intent}

    with patch("sql_executor.run_readonly") as mock_run:
        mock_run.return_value = ([], None)
        section, err = execute_performance_query(payload)

    assert err is None
    assert section["success"] is True
    assert mock_run.call_count == 1
    sql = mock_run.call_args[0][0]
    assert "not yet fully translated" not in sql
    assert "AgniveerId IN (" not in sql
    assert expected_fragment in sql


def test_attemptwise_single_agniveer_returns_flat_rows():
    payload = {
        "category": "Performance",
        "operation": "AttemptWise",
        "agniveer_no": "A0701882L",
    }
    mock_db_rows = [
        {"agniveerNo": "A0701882L", "fullName": "HARMAN SINGH", "sectionName": "BPET", "attemptNo": 1, "attemptTotal": 100},
        {"agniveerNo": "A0701882L", "fullName": "HARMAN SINGH", "sectionName": "PPT", "attemptNo": 1, "attemptTotal": 88},
    ]

    with patch("sql_executor.run_readonly") as mock_run:
        mock_run.return_value = (mock_db_rows, None)
        section, err = execute_performance_query(payload)

    assert err is None
    assert section["success"] is True
    assert section["data"] == mock_db_rows


def test_attemptwise_multi_agniveer_returns_pivoted_data():
    payload = {
        "category": "Performance",
        "operation": "AttemptWise",
        "section": "BPET",
    }
    mock_db_rows = [
        {"agniveerNo": "A1", "fullName": "User One", "sectionName": "BPET", "attemptNo": 1, "attemptTotal": 100},
        {"agniveerNo": "A1", "fullName": "User One", "sectionName": "PPT", "attemptNo": 1, "attemptTotal": 88},
        {"agniveerNo": "A2", "fullName": "User Two", "sectionName": "BPET", "attemptNo": 1, "attemptTotal": 90},
    ]

    with patch("sql_executor.run_readonly") as mock_run:
        mock_run.return_value = (mock_db_rows, None)
        section, err = execute_performance_query(payload)

    assert err is None
    assert section["success"] is True
    expected_pivoted = [
        {
            "agniveerNo": "A1",
            "fullName": "User One",
            "attempts": {
                "1": {"BPET": 100, "PPT": 88}
            }
        },
        {
            "agniveerNo": "A2",
            "fullName": "User Two",
            "attempts": {
                "1": {"BPET": 90}
            }
        }
    ]
    assert section["data"] == expected_pivoted


def test_no_exceptional_section_union_query():
    payload = {
        "category": "Performance",
        "operation": "Top",
        "section": "BPET",
    }

    with patch("sql_executor.run_readonly") as mock_run:
        mock_run.return_value = ([], None)
        section, err = execute_performance_query(payload)

    assert err is None
    assert section["success"] is True
    sql = mock_run.call_args[0][0]
    assert "AgniveerSectionResult" not in sql
    assert "ISNULL(sec.IsExceptional, 0) = 0" in sql




