from __future__ import annotations

from unittest.mock import patch
import pytest

from sql_executor import execute_sql_query, build_verification_sql, execute_verification_query


def test_verification_summary_sql():
    sql, params = build_verification_sql("Summary", batch_id=1)
    assert "COUNT(*) AS TotalAgniveers" in sql
    assert "PendingCount" in sql
    assert "SentCount" in sql
    assert "NotRespondedCount" in sql
    assert "VerifiedCount" in sql
    assert "RejectedCount" in sql
    assert "a.BatchId = ?" in sql
    assert params == [1]


def test_verification_pending_sql():
    sql, params = build_verification_sql("Pending", company_id=5)
    assert "(pv.Status = 'Rejected' OR pv.AgniveerId IS NULL)" in sql
    assert "EXISTS (SELECT 1 FROM PlatoonMaster p WHERE p.Id = a.PlatoonId AND p.CompanyId = ?)" in sql
    assert params == [5]


def test_verification_sent_sql():
    sql, params = build_verification_sql("Sent")
    assert "LatestVerification AS (" in sql
    assert "lv.Status != 'Pending'" in sql


def test_verification_not_responded_sql():
    sql, params = build_verification_sql("NotResponded")
    assert "DATEDIFF(DAY, lv.SentDate, GETDATE()) AS DaysSinceSent" in sql
    assert "lv.Status = 'Sent'" in sql
    assert "lv.ReceivedDate IS NULL" in sql


def test_verification_verified_sql():
    sql, params = build_verification_sql("Verified")
    assert "DATEDIFF(DAY, lv.SentDate, lv.ReceivedDate) AS DaysToRespond" in sql
    assert "lv.Status = 'Verified'" in sql


def test_verification_rejected_sql():
    sql, params = build_verification_sql("Rejected")
    assert "lv.Status = 'Rejected'" in sql


def test_execute_verification_query_summary():
    def mock_run(sql, params=(), max_rows=None):
        return ([{
            "totalAgniveers": 100,
            "pendingCount": 15,
            "sentCount": 85,
            "notRespondedCount": 10,
            "verifiedCount": 70,
            "rejectedCount": 5
        }], None)

    intent = {
        "category": "Verification",
        "operation": "Summary",
        "responseType": "Summary",
        "raw_query": "Verification summary"
    }

    with patch("sql_executor.run_readonly", side_effect=mock_run):
        res, err = execute_sql_query(intent=intent)

    assert err is None
    assert res["success"] is True
    assert len(res["records"]) == 1
    assert res["records"][0]["verifiedCount"] == 70
