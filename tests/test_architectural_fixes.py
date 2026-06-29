import unittest

from intent_engine.query_planner import QueryType, plan_query
from response_builder import build_response
from result_combiner import compare_datasets as compare_results
from result_combiner import (
    process_distribution,
    process_trend,
)
from widget_engine import build_formatted_data


class TestArchitecturalFixes(unittest.TestCase):

    def test_compare_results_new_keys(self):
        set_a = [{"agniveerId": 1, "bestTotal": 95}, {"agniveerId": 2, "bestTotal": 85}]
        set_b = [{"agniveerId": 3, "bestTotal": 70}, {"agniveerId": 4, "bestTotal": 80}]

        # averageScore for A is 90.0, averageScore for B is 75.0
        compared = compare_results([("Side A", set_a), ("Side B", set_b)])

        self.assertIn("left", compared)
        self.assertIn("right", compared)
        self.assertIn("comparison", compared)

        # Verify left/right data structure
        self.assertEqual(compared["left"]["label"], "Side A")
        self.assertEqual(compared["right"]["label"], "Side B")

        # Verify metric calculations
        comp_metrics = compared["comparison"]
        self.assertIn("averageScore", comp_metrics)
        self.assertEqual(comp_metrics["averageScore"]["higher"], "Side A")
        self.assertEqual(comp_metrics["averageScore"]["lower"], "Side B")
        self.assertEqual(comp_metrics["averageScore"]["difference"], 15.0)
        self.assertEqual(
            comp_metrics["averageScore"]["percentage"], 20.0
        )  # (15 / 75) * 100

    def test_trend_engine(self):
        records = [
            {"date": "2026-06-01", "score": 80.0},
            {"date": "2026-06-02", "score": 85.0},
            {"date": "2026-06-03", "score": 95.0},
        ]
        trend = process_trend([records], {})

        self.assertEqual(trend["queryType"], "trend")
        self.assertEqual(trend["granularity"], "daily")
        self.assertEqual(trend["trendDirection"], "increase")
        self.assertTrue(trend["increase"])
        self.assertFalse(trend["decrease"])
        self.assertFalse(trend["stable"])

        chart_data = trend["chartData"]
        self.assertEqual(len(chart_data), 3)
        self.assertEqual(chart_data[0]["label"], "2026-06-01")
        self.assertEqual(chart_data[0]["value"], 80.0)

    def test_distribution_engine(self):
        records = [
            {"platoonName": "Platoon 1"},
            {"platoonName": "Platoon 1"},
            {"platoonName": "Platoon 2"},
        ]
        dist = process_distribution([records], {"group_by": "platoon"})

        self.assertEqual(dist["queryType"], "distribution")
        self.assertEqual(dist["groupBy"], "platoon")
        self.assertEqual(dist["labels"], ["Platoon 1", "Platoon 2"])
        self.assertEqual(dist["values"], [2, 1])

    def test_response_builder_answer_key(self):
        resp = build_response(
            message="Intro text",
            formatted_data={"type": "TABLE", "data": {"columns": [], "rows": []}},
            metadata={"sessionId": "session-123"},
            session_id="session-123",
            suggested_questions=[],
            dotnet_payload={},
        )
        self.assertIn("formattedData", resp)
        self.assertIsInstance(resp["formattedData"], list)
        self.assertEqual(resp["metadata"]["sessionId"], "session-123")

    def test_widgets_selection_step_12(self):
        # 1. Single record -> TABLE
        fd_single = build_formatted_data(
            {"sections": [{"label": "Result", "data": [{"id": 1}]}]},
            query_type="simple",
            intent={},
        )
        self.assertEqual(fd_single["type"], "TABLE")

        # 2. Multiple records -> TABLE
        fd_multi = build_formatted_data(
            {"sections": [{"label": "Result", "data": [{"id": 1}, {"id": 2}]}]},
            query_type="simple",
            intent={},
        )
        self.assertEqual(fd_multi["type"], "TABLE")

        # 3. Comparison -> AREA_CHART
        fd_compare = build_formatted_data(
            {
                "left": {"label": "Left", "data": [{"id": 1}]},
                "right": {"label": "Right", "data": [{"id": 2}]},
                "comparison": {"averageScore": {"higher": "Left", "lower": "Right"}},
            },
            query_type="compare",
            intent={},
        )
        self.assertEqual(fd_compare["type"], "AREA_CHART")

        # 4. Trend -> LINE_CHART
        fd_trend = build_formatted_data(
            {"sections": [{"label": "Result", "data": [{"date": "2026-06-20"}]}]},
            query_type="trend",
            intent={},
        )
        self.assertEqual(fd_trend["type"], "LINE_CHART")

        # 5. Distribution -> PIE_CHART
        fd_dist = build_formatted_data(
            {"sections": [{"label": "Result", "data": [{"sport": "Cricket"}]}]},
            query_type="distribution",
            intent={},
        )
        self.assertEqual(fd_dist["type"], "PIE_CHART")


if __name__ == "__main__":
    unittest.main()
