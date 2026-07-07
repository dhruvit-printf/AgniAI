import unittest
from unittest.mock import MagicMock, patch

from admin_pipeline import execute_admin_query
from intent_engine.query_planner import plan_query
from result_combiner import combine_results


class TestPartialFailure(unittest.TestCase):

    @patch("admin_pipeline._call_dotnet")
    @patch("admin_pipeline.generate_report")
    def test_cross_filter_secondary_failure(
        self, mock_generate_report, mock_call_dotnet
    ):
        # 3-way cross filter: primary succeeds, secondary 1 fails, secondary 2 succeeds
        # Let's say: "Show top performer in PPT who plays cricket and is currently on leave"
        mock_call_dotnet.side_effect = [
            ([{"agniveerNo": "101"}, {"agniveerNo": "102"}], None),  # PPT (Primary)
            (None, "Connection error to Skills"),  # Cricket (Secondary - Fails!)
            ([{"agniveerNo": "102"}, {"agniveerNo": "103"}], None),  # Leave (Secondary)
        ]
        mock_generate_report.return_value = {
            "introMessage": "Report.",
            "analysis": {"summary": "Analysis", "observations": [], "insights": []},
            "conclusion": {"summary": "Conclusion"},
        }

        result = execute_admin_query(
            "Show top performer in PPT who plays cricket and is currently on leave", {}
        )

        self.assertEqual(result["type"], "query")
        response_payload = result["response_payload"]
        self.assertTrue(response_payload["status"])

        # A single widget is returned bare, not wrapped in a list — see
        # response_builder.py's module docstring. degraded/failedFilters
        # live directly inside that widget's data.
        table_widget = response_payload["formattedData"]
        self.assertEqual(table_widget["type"], "TABLE")
        table_data = table_widget["data"]
        self.assertTrue(table_data.get("degraded"))
        self.assertEqual(table_data.get("failedFilters"), ["Skills"])
        self.assertEqual(table_data["matchCount"], 1)
        self.assertEqual(
            table_data["row"][0].get("agniveerNo")
            or table_data["row"][0].get("AgniveerNo"),
            "102",
        )

    @patch("admin_pipeline._call_dotnet")
    @patch("admin_pipeline.generate_report")
    def test_comparison_side_failure(self, mock_generate_report, mock_call_dotnet):
        # Comparison: one side succeeds, one side fails. Both sides run in
        # parallel threads, so the mock must key off the actual payload
        # (which section it's for) rather than assuming call order — a
        # plain sequential side_effect list races and is flaky.
        def side_effect(payload, *args, **kwargs):
            if payload.get("section") == "PPT":
                return [{"id": 1, "bestTotal": 95}], None
            return None, "Timeout on BEPT"

        mock_call_dotnet.side_effect = side_effect
        mock_generate_report.return_value = {
            "introMessage": "Report.",
            "analysis": {"summary": "Analysis", "observations": [], "insights": []},
            "conclusion": {"summary": "Conclusion"},
        }

        result = execute_admin_query("Compare PPT and BEPT", {})

        self.assertEqual(result["type"], "query")
        response_payload = result["response_payload"]
        self.assertTrue(response_payload["status"])

        # A single widget is returned bare, not wrapped in a list — see
        # response_builder.py's module docstring.
        widget = response_payload["formattedData"]
        self.assertEqual(widget["type"], "COMPARE_TABLE")

        # PPT (left side) succeeds and has 1 row
        left_table = widget["data"]["left"]
        self.assertEqual(len(left_table["row"]), 1)

        # BEPT (right side) failed — rendered as a single "unavailable"
        # placeholder row rather than an empty table.
        right_table = widget["data"]["right"]
        self.assertEqual(len(right_table["row"]), 1)
        self.assertTrue(right_table["row"][0].get("unavailable"))

        # Failed sections metadata is populated
        self.assertEqual(response_payload["failedSections"], ["BPET"])
        self.assertTrue(response_payload["partialFailure"])

    @patch("admin_pipeline._call_dotnet")
    def test_all_intents_fail(self, mock_call_dotnet):
        # All intents fail
        mock_call_dotnet.side_effect = [(None, "Error 1"), (None, "Error 2")]
        result = execute_admin_query("Compare PPT and BEPT", {})
        self.assertEqual(result["type"], "error")
        self.assertIn("trouble reaching", result["error_message"].lower())


if __name__ == "__main__":
    unittest.main()
