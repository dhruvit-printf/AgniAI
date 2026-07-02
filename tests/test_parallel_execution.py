import time
import unittest
from unittest.mock import MagicMock, patch

from admin_pipeline import execute_admin_query


class TestParallelExecution(unittest.TestCase):

    @patch("admin_pipeline._call_dotnet")
    @patch("admin_pipeline.generate_report")
    def test_parallel_execution_order_preserved(
        self, mock_generate_report, mock_call_dotnet
    ):
        # We simulate slow first call and fast second call to verify order preservation.
        def side_effect(payload, *args, **kwargs):
            # If it's PPT, we sleep a bit
            if payload.get("section") == "PPT":
                time.sleep(0.3)
                return [{"agniveerNo": "PPT-1"}], None
            else:
                return [{"agniveerNo": "BEPT-1"}], None

        mock_call_dotnet.side_effect = side_effect
        mock_generate_report.return_value = {
            "introMessage": "Report.",
            "analysis": {"summary": "Analysis", "observations": [], "insights": []},
            "conclusion": {"summary": "Conclusion"},
        }

        # Compare PPT and BEPT
        result = execute_admin_query("Compare PPT and BEPT", {})

        self.assertEqual(result["type"], "query")
        response_payload = result["response_payload"]
        self.assertTrue(response_payload["status"])

        widgets = response_payload["formattedData"]
        self.assertEqual(widgets[0]["type"], "COMPARE_TABLE")
        self.assertEqual(len(widgets), 1)

        # In the new widget structure, COMPARE_TABLE data has left and right sides
        table_data = widgets[0]["data"]
        left = table_data["left"]
        right = table_data["right"]

        # PPT must be left side, BEPT (normalised to BPET) must be right side
        self.assertEqual(left["rows"][0]["agniveerNo"], "PPT-1")
        self.assertEqual(right["rows"][0]["agniveerNo"], "BEPT-1")
        self.assertEqual(mock_call_dotnet.call_count, 2)


if __name__ == "__main__":
    unittest.main()
