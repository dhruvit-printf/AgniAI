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
        def side_effect(payload, trace_id=None):
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

        sides = response_payload["formattedData"]["data"]["sides"]
        self.assertEqual(len(sides), 2)

        # PPT must be first side, BEPT must be second side
        self.assertEqual(sides[0]["label"], "PPT")
        self.assertEqual(sides[0]["data"], [{"agniveerNo": "PPT-1"}])

        self.assertEqual(sides[1]["label"], "BPET")  # Normalised BEPT -> BPET
        self.assertEqual(sides[1]["data"], [{"agniveerNo": "BEPT-1"}])
        self.assertEqual(mock_call_dotnet.call_count, 2)


if __name__ == "__main__":
    unittest.main()
