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

        # Checked processed data has degraded == True and failedFilters
        processed = response_payload["formattedData"][1]["data"]
        self.assertTrue(processed.get("degraded"))
        self.assertEqual(processed.get("failedFilters"), ["Skills"])

        # Find the TABLE widget in formattedData for row and match assertions
        table_widget = next(
            w for w in response_payload["formattedData"] if w["type"] == "TABLE"
        )
        table_data = table_widget["data"]
        self.assertEqual(table_data["matchCount"], 1)
        self.assertEqual(
            table_data["rows"][0].get("agniveerNo")
            or table_data["rows"][0].get("AgniveerNo"),
            "102",
        )

    @patch("admin_pipeline._call_dotnet")
    @patch("admin_pipeline.generate_report")
    def test_comparison_side_failure(self, mock_generate_report, mock_call_dotnet):
        # Comparison: one side succeeds, one side fails
        mock_call_dotnet.side_effect = [
            ([{"id": 1, "bestTotal": 95}], None),  # PPT
            (None, "Timeout on BEPT"),  # BEPT (Fails)
        ]
        mock_generate_report.return_value = {
            "introMessage": "Report.",
            "analysis": {"summary": "Analysis", "observations": [], "insights": []},
            "conclusion": {"summary": "Conclusion"},
        }

        result = execute_admin_query("Compare PPT and BEPT", {})

        self.assertEqual(result["type"], "query")
        response_payload = result["response_payload"]
        self.assertTrue(response_payload["status"])

        widgets = response_payload["formattedData"]
        self.assertEqual(widgets[0]["type"], "COMPARE_TABLE")
        self.assertEqual(len(widgets), 1)

        # PPT (left side) succeeds and has 1 row
        left_table = widgets[0]["data"]["left"]
        self.assertEqual(len(left_table["rows"]), 1)

        # BEPT (right side) failed and has 0 rows
        right_table = widgets[0]["data"]["right"]
        self.assertEqual(len(right_table["rows"]), 0)

        # Failed sections metadata is populated
        self.assertEqual(response_payload["failedSections"], ["BPET"])
        self.assertTrue(response_payload["partialFailure"])

    @patch("admin_pipeline._call_dotnet")
    def test_all_intents_fail(self, mock_call_dotnet):
        # All intents fail
        mock_call_dotnet.side_effect = [(None, "Error 1"), (None, "Error 2")]
        result = execute_admin_query("Compare PPT and BEPT", {})
        self.assertEqual(result["type"], "error")
        self.assertEqual(result["error_message"], "Failed to process request.")


if __name__ == "__main__":
    unittest.main()
