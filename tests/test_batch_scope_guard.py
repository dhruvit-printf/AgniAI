"""
tests/test_batch_scope_guard.py
================================
Covers the batch-mismatch short-circuit in execute_admin_query(): when the
query text names a batch other than the one the frontend has this request
scoped to (id_filters["batchId"]), the pipeline must ask the user to switch
batches instead of silently answering for the frontend's batch.
"""

from __future__ import annotations

import os
import sys
import types
import unittest
from unittest.mock import patch

_STUB_MODS = [
    "flask",
    "flask_cors",
    "flask_limiter",
    "flask_limiter.util",
    "dotenv",
    "requests",
]
for mod in _STUB_MODS:
    try:
        __import__(mod)
    except ImportError:
        if mod not in sys.modules:
            stub = types.ModuleType(mod)
            if mod == "dotenv":
                stub.load_dotenv = lambda *a, **kw: None  # type: ignore[attr-defined]
            sys.modules[mod] = stub

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from admin_pipeline import execute_admin_query


class TestBatchScopeGuard(unittest.TestCase):

    @patch("sql_executor.get_batch_name", return_value=None)
    @patch("admin_pipeline.fetch_sql_results")
    @patch("admin_pipeline.generate_report")
    def test_query_naming_other_batch_asks_to_switch(
        self, mock_generate_report, mock_fetch_sql, mock_get_batch_name
    ):
        result = execute_admin_query(
            "Give me top 10 performers from batch 2",
            {"batchId": 1, "session_id": "test-batch-guard"},
        )

        self.assertEqual(result["type"], "clarification")
        message = result["combined_message"]
        self.assertIn("Batch 1", message)
        self.assertIn("switch to Batch 2", message)
        mock_fetch_sql.assert_not_called()
        mock_generate_report.assert_not_called()

    @patch("admin_pipeline.fetch_sql_results")
    @patch("admin_pipeline.generate_report")
    def test_query_naming_current_batch_proceeds_normally(
        self, mock_generate_report, mock_fetch_sql
    ):
        section = {"success": True, "records": [], "data": [], "count": 0}
        mock_fetch_sql.return_value = ([section], [("Result", section)], None)
        mock_generate_report.return_value = {
            "message": "Report generated.",
            "analysis": {"summary": "s", "observations": [], "insights": []},
            "conclusion": {"summary": "c"},
        }

        result = execute_admin_query(
            "Give me top 10 performers from batch 1",
            {"batchId": 1, "session_id": "test-batch-guard-2"},
        )

        self.assertNotEqual(result["type"], "clarification")
        mock_fetch_sql.assert_called_once()

    @patch("admin_pipeline.fetch_sql_results")
    @patch("admin_pipeline.generate_report")
    def test_no_frontend_batch_scope_does_not_short_circuit(
        self, mock_generate_report, mock_fetch_sql
    ):
        section = {"success": True, "records": [], "data": [], "count": 0}
        mock_fetch_sql.return_value = ([section], [("Result", section)], None)
        mock_generate_report.return_value = {
            "message": "Report generated.",
            "analysis": {"summary": "s", "observations": [], "insights": []},
            "conclusion": {"summary": "c"},
        }

        result = execute_admin_query(
            "Give me top 10 performers from batch 2",
            {"session_id": "test-batch-guard-3"},
        )

        self.assertNotEqual(result["type"], "clarification")
        mock_fetch_sql.assert_called_once()


if __name__ == "__main__":
    unittest.main()
