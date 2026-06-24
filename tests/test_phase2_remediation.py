from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import requests

from analysis_engine import generate_analysis
from conclusion_engine import generate_conclusion
from dotnet_adapter import normalize_dotnet_response
from metrics import metrics_collector
from result_combiner import intersect_results
from compare_engine import compare_datasets
from admin_pipeline import execute_admin_query


class TestF4DotnetAdapter(unittest.TestCase):
    def test_normalize_dotnet_response_handles_wrapper_shapes(self):
        samples = [
            {"data": [{"id": 1}]},
            {"Data": {"records": [{"id": 2}]}},
            {"records": [{"id": 3}]},
            {"result": {"records": [{"id": 4}]}},
            {"teams": [{"members": [{"id": 5}]}]},
            [{"id": 6}],
            {"nested": {"records": [{"id": 7}]}},
        ]

        ids = []
        for sample in samples:
            normalized = normalize_dotnet_response(sample)
            ids.append(normalized["records"][0]["id"])

        self.assertEqual(ids, [1, 2, 3, 4, 5, 6, 7])


class TestF6CompareShape(unittest.TestCase):
    def test_compare_shape_flows_into_analysis_and_conclusion(self):
        compared = compare_datasets(
            [
                ("PPT", [{"agniveerId": 1, "bestTotal": 95}, {"agniveerId": 2, "bestTotal": 85}]),
                ("BPET", [{"agniveerId": 3, "bestTotal": 70}, {"agniveerId": 4, "bestTotal": 60}]),
            ]
        )

        self.assertIn("averageScore", compared["comparison"])
        self.assertEqual(compared["comparison"]["averageScore"]["higher"], "PPT")
        self.assertEqual(compared["comparison"]["averageScore"]["lower"], "BPET")

        with patch("analysis_engine.get_flags") as mock_flags:
            mock_flags.return_value = MagicMock(ENABLE_REPORTS=False, ENABLE_OLLAMA=False)
            analysis = generate_analysis(
                {
                    "left": compared["left"],
                    "right": compared["right"],
                    "comparison": compared["comparison"],
                },
                "compare",
                {"category": "Performance"},
            )
        with patch("conclusion_engine.get_flags") as mock_flags:
            mock_flags.return_value = MagicMock(ENABLE_REPORTS=False, ENABLE_OLLAMA=False)
            conclusion = generate_conclusion(
                {
                    "left": compared["left"],
                    "right": compared["right"],
                    "comparison": compared["comparison"],
                },
                "compare",
                {"category": "Performance"},
            )

        self.assertIn("Comparison", analysis["summary"])
        self.assertIn("comparative", conclusion["summary"].lower())


class TestF7MetricsCounters(unittest.TestCase):
    def setUp(self) -> None:
        metrics_collector.dotnet_failures = 0
        metrics_collector.timeout_failures = 0
        metrics_collector.llm_failures = 0
        metrics_collector.successful_queries_total.clear()

    @patch("dotnet_executor._dotnet_session.post")
    def test_dotnet_failure_counter(self, mock_post):
        mock_resp = MagicMock(status_code=500)
        mock_resp.json.side_effect = Exception("No JSON")
        mock_resp.text = "boom"
        mock_post.return_value = mock_resp

        from dotnet_executor import _call_dotnet

        data, err = _call_dotnet({"cmd": "fail"})
        self.assertIsNone(data)
        self.assertIsNotNone(err)
        self.assertGreater(metrics_collector.dotnet_failures, 0)

    @patch("dotnet_executor._dotnet_session.post")
    def test_timeout_failure_counter(self, mock_post):
        mock_post.side_effect = requests.Timeout("slow")

        from dotnet_executor import _call_dotnet

        data, err = _call_dotnet({"cmd": "slow"})
        self.assertIsNone(data)
        self.assertIsNotNone(err)
        self.assertGreater(metrics_collector.timeout_failures, 0)

    @patch("analysis_engine.requests.post")
    @patch("analysis_engine.get_flags")
    def test_llm_failure_counter(self, mock_flags, mock_post):
        mock_flags.return_value = MagicMock(ENABLE_REPORTS=True, ENABLE_OLLAMA=True)
        mock_post.side_effect = requests.RequestException("ollama down")

        generate_analysis(
            {"sections": [{"label": "Result", "data": [{"id": 1, "score": 10}]}]},
            "simple",
            {"category": "Performance"},
        )
        self.assertGreater(metrics_collector.llm_failures, 0)

    @patch("admin_pipeline._call_dotnet")
    @patch("admin_pipeline.generate_report")
    def test_success_counter(self, mock_generate_report, mock_call_dotnet):
        mock_call_dotnet.return_value = ({"records": [{"id": 1, "score": 90}]}, None)
        mock_generate_report.return_value = {
            "introMessage": "Intro",
            "analysis": {"summary": "Summary", "observations": [], "insights": []},
            "conclusion": {"summary": "Conclusion"},
        }

        execute_admin_query("Show attendance", {"session_id": "sess-1"})
        self.assertEqual(metrics_collector.successful_queries_total.get("simple"), 1)


class TestF9CrossFilterMerge(unittest.TestCase):
    def test_secondary_attributes_are_preserved_on_intersection(self):
        primary = [
            {"agniveerId": 101, "fullName": "A", "bestTotal": 95},
            {"agniveerId": 102, "fullName": "B", "bestTotal": 88},
        ]
        sport = [
            {"agniveerId": 101, "sport": "Cricket"},
            {"agniveerId": 103, "sport": "Football"},
        ]
        leave = [
            {"agniveerId": 101, "leaveType": "Current"},
            {"agniveerId": 104, "leaveType": "Past"},
        ]

        combined = intersect_results([primary, sport, leave], primary_index=0)
        self.assertEqual(combined["matchCount"], 1)
        record = combined["records"][0]
        self.assertEqual(record["agniveerId"], 101)
        self.assertEqual(record["sport"], "Cricket")
        self.assertEqual(record["leaveType"], "Current")


if __name__ == "__main__":
    unittest.main()
