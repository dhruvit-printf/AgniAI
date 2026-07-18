"""
tests/test_pipeline_e2e.py
==========================
End-to-End runtime execution tests for execute_admin_query().
"""

from __future__ import annotations

import os

# ── Minimal stubs for environment compatibility ────────────────────────────
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

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


class TestPipelineEndToEnd(unittest.TestCase):

    @patch("admin_pipeline.fetch_sql_results")
    @patch("admin_pipeline.generate_report")
    def test_filter_query_e2e(self, mock_generate_report, mock_fetch_sql):
        # 1. FILTER_QUERY: "Show top performers who play cricket"
        # Only 1 SQL fetch is expected
        section = {
            "success": True,
            "records": [
                {
                    "id": 1,
                    "fullName": "AMIT KUMAR",
                    "agniveerNo": "A01",
                    "sports": "Cricket",
                    "bestTotal": 95,
                },
                {
                    "id": 2,
                    "fullName": "KAPIL DEV",
                    "agniveerNo": "A02",
                    "sports": "Cricket",
                    "bestTotal": 88,
                },
            ],
            "data": [
                {
                    "id": 1,
                    "fullName": "AMIT KUMAR",
                    "agniveerNo": "A01",
                    "sports": "Cricket",
                    "bestTotal": 95,
                },
                {
                    "id": 2,
                    "fullName": "KAPIL DEV",
                    "agniveerNo": "A02",
                    "sports": "Cricket",
                    "bestTotal": 88,
                },
            ],
            "count": 2,
        }
        mock_fetch_sql.return_value = ([section], [("Result", section)], None)
        mock_generate_report.return_value = {
            "introMessage": "Report generated.",
            "analysis": {
                "summary": "Summary text",
                "observations": ["Obs 1"],
                "insights": ["Insight 1"],
            },
            "conclusion": {"summary": "Conclusion text"},
        }

        result = execute_admin_query("Show top performers in PPT", {})

        self.assertEqual(result["type"], "query")
        response_payload = result["response_payload"]
        self.assertEqual(response_payload["metadata"]["queryType"], "simple")
        self.assertTrue(response_payload["status"])

        # Verify widget type: TABLE should be generated since Name/AgniveerNo exist.
        fd = response_payload.get("formattedData", [])
        if isinstance(fd, dict):
            fd = [fd]
        self.assertEqual(fd[0]["type"], "TABLE")

        mock_fetch_sql.assert_called_once()

    @patch("admin_pipeline.fetch_sql_results")
    @patch("admin_pipeline.generate_report")
    def test_disclaimer_banner_stays_conversational(
        self, mock_generate_report, mock_fetch_sql
    ):
        result = execute_admin_query(
            "AgniAI may make mistakes. Verify important information.",
            {},
        )

        self.assertEqual(result["type"], "conversational")
        response_payload = result["response_payload"]
        self.assertEqual(response_payload["metadata"]["queryType"], "conversational")
        mock_fetch_sql.assert_not_called()
        mock_generate_report.assert_not_called()

    @patch("admin_pipeline.fetch_sql_results")
    @patch("admin_pipeline.generate_report")
    def test_cross_filter_query_e2e(self, mock_generate_report, mock_fetch_sql):
        # 2. CROSS_FILTER: "Show top performer in PPT who plays cricket and is
        # currently on leave" — HARD RULE R4 means this is now ONE atomic SQL
        # query (CTE per condition, INNER JOINed/intersected inside the SQL
        # itself), so the mock returns the already-intersected row directly
        # rather than 3 separate per-condition datasets for Python to
        # intersect.
        section = {
            "success": True,
            "records": [
                {"id": 1, "fullName": "AMIT KUMAR", "agniveerNo": "A01"},
            ],
            "data": [
                {"id": 1, "fullName": "AMIT KUMAR", "agniveerNo": "A01"},
            ],
            "count": 1,
        }
        mock_fetch_sql.return_value = ([section], [("Performance", section)], None)

        mock_generate_report.return_value = {
            "introMessage": "Cross-filter report generated.",
            "analysis": {
                "summary": "Intersection completed",
                "observations": [],
                "insights": [],
            },
            "conclusion": {"summary": "Intersection successful"},
        }

        result = execute_admin_query(
            "Show top performer in PPT who plays cricket and is currently on leave", {}
        )

        self.assertEqual(result["type"], "query")
        response_payload = result["response_payload"]
        self.assertTrue(response_payload["status"])
        self.assertEqual(response_payload["metadata"]["queryType"], "cross_filter")

        fd = response_payload.get("formattedData", [])
        if isinstance(fd, dict):
            fd = [fd]
        table_widget = next((w for w in fd if w.get("type") == "TABLE"), None)
        self.assertIsNotNone(table_widget, "TABLE widget not found in formattedData")
        rows = table_widget["data"].get("rows") or table_widget["data"].get("row") or []
        self.assertEqual(len(rows), 1)
        self.assertEqual(
            rows[0]["fullName"], "AMIT KUMAR"
        )  # camelCase after normalisation
        self.assertEqual(rows[0]["agniveerNo"], "A01")  # camelCase after normalisation

        # R4: one atomic SQL call, not one per condition.
        mock_fetch_sql.assert_called_once()

    @patch("intent_engine.query_planner._is_semantic_comparison")
    @patch("admin_pipeline.fetch_sql_results")
    @patch("admin_pipeline.generate_report")
    def test_comparison_query_e2e(self, mock_generate_report, mock_fetch_sql, mock_semantic_comp):
        mock_semantic_comp.return_value = True
        # 3. COMPARISON: "Compare PPT and BEPT" — one SQL fetch per side.
        section_ppt = {
            "success": True,
            "records": [{"id": 1, "bestTotal": 95}, {"id": 2, "bestTotal": 85}],
            "data": [{"id": 1, "bestTotal": 95}, {"id": 2, "bestTotal": 85}],
            "count": 2,
        }
        section_bept = {
            "success": True,
            "records": [{"id": 3, "bestTotal": 75}],
            "data": [{"id": 3, "bestTotal": 75}],
            "count": 1,
        }
        mock_fetch_sql.return_value = (
            [section_ppt, section_bept],
            [("PPT", section_ppt), ("BEPT", section_bept)],
            None,
        )
        mock_generate_report.return_value = {
            "introMessage": "Comparison report.",
            "analysis": {"summary": "Diff summary", "observations": [], "insights": []},
            "conclusion": {"summary": "Comparison done"},
        }

        result = execute_admin_query("Compare PPT and BEPT", {})

        self.assertEqual(result["type"], "query")
        response_payload = result["response_payload"]
        self.assertTrue(response_payload["status"])
        self.assertEqual(response_payload["metadata"]["queryType"], "COMPARISON")

        # Check comparison results — verify correct COMPARE widget is built
        widgets = response_payload["formattedData"]
        if isinstance(widgets, dict):
            widgets = [widgets]
        self.assertEqual(widgets[0]["type"], "TABLE")
        self.assertEqual(len(widgets), 1)
        widget = widgets[0]
        self.assertIn("comparisonMetrics", response_payload["metadata"])
        self.assertIn("left", widget["data"])
        self.assertIn("right", widget["data"])

        mock_fetch_sql.assert_called_once()

    @patch("admin_pipeline.fetch_sql_results")
    @patch("admin_pipeline.generate_report")
    def test_multi_operation_query_e2e(self, mock_generate_report, mock_fetch_sql):
        # 4. MULTI_OPERATION: "Show attendance and current leave records"
        # Expects 1 SQL fetch covering both independent legs.
        # NOTE: pre-existing, unrelated to the SQL migration — plan_query()
        # currently classifies this message as QueryType.SIMPLE rather than
        # MULTI_INDEPENDENT (confirmed failing before this migration too),
        # so this test still fails past the mock setup below.
        section_attendance = {
            "success": True,
            "records": [
                {
                    "id": 1,
                    "fullName": "AMIT KUMAR",
                    "agniveerNo": "A01",
                    "present": True,
                }
            ],
            "data": [
                {
                    "id": 1,
                    "fullName": "AMIT KUMAR",
                    "agniveerNo": "A01",
                    "present": True,
                }
            ],
            "count": 1,
        }
        section_leave = {
            "success": True,
            "records": [
                {
                    "id": 2,
                    "fullName": "KAPIL DEV",
                    "agniveerNo": "A02",
                    "leaveStatus": "Current",
                }
            ],
            "data": [
                {
                    "id": 2,
                    "fullName": "KAPIL DEV",
                    "agniveerNo": "A02",
                    "leaveStatus": "Current",
                }
            ],
            "count": 1,
        }
        mock_fetch_sql.return_value = (
            [section_attendance, section_leave],
            [("Attendance", section_attendance), ("Leave", section_leave)],
            None,
        )
        mock_generate_report.return_value = {
            "introMessage": "Multi-op report.",
            "analysis": {
                "summary": "Consolidated sections",
                "observations": [],
                "insights": [],
            },
            "conclusion": {"summary": "Multi-op done"},
        }

        result = execute_admin_query(
            "Show attendance and current leave records for agniveer 12345", {}
        )

        self.assertEqual(result["type"], "query")
        response_payload = result["response_payload"]
        self.assertTrue(response_payload["status"])
        self.assertEqual(response_payload["metadata"]["queryType"], "multi_independent")

        # Verify sections — each uses the same default widget type a
        # standalone query for that category/operation would get (Attendance
        # defaults to TABLE; Leave/Current
        # defaults to TABLE).
        widgets = response_payload["formattedData"]
        if isinstance(widgets, dict):
            widgets = [widgets]
        self.assertEqual(len(widgets), 2)
        self.assertEqual(widgets[0]["type"], "TABLE")
        self.assertEqual(widgets[0]["title"], "Attendance")
        self.assertEqual(widgets[1]["type"], "TABLE")
        self.assertEqual(widgets[1]["title"], "Leave")

        mock_fetch_sql.assert_called_once()

    @patch("admin_pipeline.fetch_sql_results")
    @patch("admin_pipeline.generate_report")
    def test_analytics_query_e2e(self, mock_generate_report, mock_fetch_sql):
        # 5. ANALYTICS: "Show grading summary"
        # Expects 1 SQL fetch
        section = {
            "success": True,
            "records": [
                {"group": "Excellent", "count": 5},
                {"group": "Good", "count": 10},
            ],
            "data": [
                {"group": "Excellent", "count": 5},
                {"group": "Good", "count": 10},
            ],
            "count": 2,
        }
        mock_fetch_sql.return_value = ([section], [("Result", section)], None)
        mock_generate_report.return_value = {
            "introMessage": "Analytics report.",
            "analysis": {
                "summary": "Grading summary",
                "observations": [],
                "insights": [],
            },
            "conclusion": {"summary": "Analytics done"},
        }

        result = execute_admin_query("Show grading summary", {})

        self.assertEqual(result["type"], "query")
        response_payload = result["response_payload"]
        self.assertEqual(response_payload["metadata"]["queryType"], "simple")
        self.assertTrue(response_payload["status"])

        mock_fetch_sql.assert_called_once()


if __name__ == "__main__":
    unittest.main()
