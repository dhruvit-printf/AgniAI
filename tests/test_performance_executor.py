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

