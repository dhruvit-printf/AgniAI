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
        processed = response_payload["formattedData"]["data"]
        self.assertTrue(processed.get("degraded"))
        self.assertEqual(processed.get("failedFilters"), ["Skills"])

        # Intersection matches only 102 (present in both PPT and Leave, since Skills was skipped)
        self.assertEqual(processed["matchCount"], 1)
        self.assertEqual(processed["rows"][0].get("agniveerNo") or processed["rows"][0].get("AgniveerNo"), "102")

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

        sides = response_payload["formattedData"]["data"]["sides"]
        self.assertEqual(len(sides), 2)
        # Side 0 is PPT
        self.assertEqual(sides[0]["label"], "PPT")
        self.assertNotIn("unavailable", sides[0])
        # Side 1 is BEPT (failed)
        self.assertEqual(sides[1]["label"], "BPET")  # BPET due to normalization
        self.assertTrue(sides[1].get("unavailable") or sides[1].get("data", {}).get("unavailable"))

    @patch("admin_pipeline._call_dotnet")
    def test_all_intents_fail(self, mock_call_dotnet):
        # All intents fail
        mock_call_dotnet.side_effect = [(None, "Error 1"), (None, "Error 2")]
        result = execute_admin_query("Compare PPT and BEPT", {})
        self.assertEqual(result["type"], "error")
        self.assertEqual(result["error_message"], "Failed to process request.")


if __name__ == "__main__":
    unittest.main()
