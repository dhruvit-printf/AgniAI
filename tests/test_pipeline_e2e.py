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

    @patch("admin_pipeline._call_dotnet")
    @patch("admin_pipeline.generate_report")
    def test_filter_query_e2e(self, mock_generate_report, mock_call_dotnet):
        # 1. FILTER_QUERY: "Show top performers who play cricket"
        # Only 1 .NET call is expected
        mock_call_dotnet.return_value = (
            [
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
            None,
        )
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
        self.assertEqual(response_payload["formattedData"][0]["type"], "TABLE")

        mock_call_dotnet.assert_called_once()

    @patch("admin_pipeline._call_dotnet")
    @patch("admin_pipeline.generate_report")
    def test_disclaimer_banner_stays_conversational(
        self, mock_generate_report, mock_call_dotnet
    ):
        result = execute_admin_query(
            "AgniAI may make mistakes. Verify important information.",
            {},
        )

        self.assertEqual(result["type"], "conversational")
        response_payload = result["response_payload"]
        self.assertEqual(response_payload["metadata"]["queryType"], "conversational")
        mock_call_dotnet.assert_not_called()
        mock_generate_report.assert_not_called()

    @patch("admin_pipeline._call_dotnet")
    @patch("admin_pipeline.generate_report")
    def test_cross_filter_query_e2e(self, mock_generate_report, mock_call_dotnet):
        # 2. CROSS_FILTER: "Show top performer in PPT who plays cricket and is currently on leave"
        # Expects 3 .NET calls: Performance.Top, Skills.BySport (Cricket), Leave.Current
        # We use a side-effect mock to simulate responses for each of the 3 calls.
        mock_call_dotnet.side_effect = [
            # Performance.Top
            (
                [
                    {
                        "id": 1,
                        "fullName": "AMIT KUMAR",
                        "agniveerNo": "A01",
                        "bestTotal": 95,
                    },
                    {
                        "id": 2,
                        "fullName": "KAPIL DEV",
                        "agniveerNo": "A02",
                        "bestTotal": 88,
                    },
                ],
                None,
            ),
            # Skills.BySport (Cricket)
            (
                [
                    {
                        "id": 1,
                        "fullName": "AMIT KUMAR",
                        "agniveerNo": "A01",
                        "sport": "Cricket",
                    },
                    {
                        "id": 3,
                        "fullName": "RAM SINGH",
                        "agniveerNo": "A03",
                        "sport": "Cricket",
                    },
                ],
                None,
            ),
            # Leave.Current
            (
                [
                    {
                        "id": 1,
                        "fullName": "AMIT KUMAR",
                        "agniveerNo": "A01",
                        "leaveType": "Current",
                    }
                ],
                None,
            ),
        ]

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

        # Verify intersection result (only AMIT KUMAR matches all three sets)
        # formattedData is multi-widget; find the TABLE widget for row inspection
        table_widget = next(
            (w for w in response_payload["formattedData"] if w["type"] == "TABLE"), None
        )
        self.assertIsNotNone(table_widget, "TABLE widget not found in formattedData")
        rows = table_widget["data"]["rows"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(
            rows[0]["fullName"], "AMIT KUMAR"
        )  # camelCase after normalisation
        self.assertEqual(rows[0]["agniveerNo"], "A01")  # camelCase after normalisation

        self.assertEqual(mock_call_dotnet.call_count, 3)

    @patch("admin_pipeline._call_dotnet")
    @patch("admin_pipeline.generate_report")
    def test_comparison_query_e2e(self, mock_generate_report, mock_call_dotnet):
        # 3. COMPARISON: "Compare PPT and BEPT"
        # Expects 2 .NET calls
        mock_call_dotnet.side_effect = [
            ([{"id": 1, "bestTotal": 95}, {"id": 2, "bestTotal": 85}], None),  # PPT
            ([{"id": 3, "bestTotal": 75}], None),  # BEPT
        ]
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
        self.assertEqual(widgets[0]["type"], "COMPARE_CARD")
        self.assertEqual(widgets[1]["type"], "COMPARE_TABLE")
        widget = widgets[1]
        self.assertIn("comparisonMetrics", response_payload["metadata"])
        self.assertIn("left", widget["data"])
        self.assertIn("right", widget["data"])

        self.assertEqual(mock_call_dotnet.call_count, 2)

    @patch("admin_pipeline._call_dotnet")
    @patch("admin_pipeline.generate_report")
    def test_multi_operation_query_e2e(self, mock_generate_report, mock_call_dotnet):
        # 4. MULTI_OPERATION: "Show attendance and current leave records"
        # Expects 2 .NET calls
        mock_call_dotnet.side_effect = [
            (
                [
                    {
                        "id": 1,
                        "fullName": "AMIT KUMAR",
                        "agniveerNo": "A01",
                        "present": True,
                    }
                ],
                None,
            ),
            (
                [
                    {
                        "id": 2,
                        "fullName": "KAPIL DEV",
                        "agniveerNo": "A02",
                        "leaveStatus": "Current",
                    }
                ],
                None,
            ),
        ]
        mock_generate_report.return_value = {
            "introMessage": "Multi-op report.",
            "analysis": {
                "summary": "Consolidated sections",
                "observations": [],
                "insights": [],
            },
            "conclusion": {"summary": "Multi-op done"},
        }

        result = execute_admin_query("Show attendance and current leave records", {})

        self.assertEqual(result["type"], "query")
        response_payload = result["response_payload"]
        self.assertTrue(response_payload["status"])
        self.assertEqual(response_payload["metadata"]["queryType"], "multi_independent")

        # Verify sections (TABLE widgets for Attendance and Leave)
        widgets = response_payload["formattedData"]
        self.assertEqual(len(widgets), 2)
        self.assertEqual(widgets[0]["type"], "TABLE")
        self.assertEqual(widgets[0]["title"], "Attendance")
        self.assertEqual(widgets[1]["type"], "TABLE")
        self.assertEqual(widgets[1]["title"], "Leave")

        self.assertEqual(mock_call_dotnet.call_count, 2)

    @patch("admin_pipeline._call_dotnet")
    @patch("admin_pipeline.generate_report")
    def test_analytics_query_e2e(self, mock_generate_report, mock_call_dotnet):
        # 5. ANALYTICS: "Show grading summary"
        # Expects 1 .NET call
        mock_call_dotnet.return_value = (
            [{"group": "Excellent", "count": 5}, {"group": "Good", "count": 10}],
            None,
        )
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

        self.assertEqual(mock_call_dotnet.call_count, 1)


if __name__ == "__main__":
    unittest.main()
