import unittest

from query_planner import QueryType, plan_query
from response_builder import build_response
from result_combiner import compare_results, process_distribution, process_trend
from visualization_engine import generate_widgets


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
        intent = {
            "category": "Performance",
            "subcategory": "TopPerformers",
            "confidence": "high",
        }
        resp = build_response(
            query_type="simple",
            intro_message="Intro text",
            combined_result={"data": 123},
            analysis=None,
            conclusion=None,
            intent=intent,
            raw_results=[],
            confidence=0.9,
            operation_count=1,
            formatted_data="Formatted text",
        )
        self.assertIn("answer", resp)
        self.assertEqual(resp["answer"], "Intro text\n\nFormatted text")

    def test_widgets_selection_step_12(self):
        # 1. Single record -> CARD
        widgets_single = generate_widgets({"records": [{"id": 1}]}, query_plan="simple")
        self.assertEqual(widgets_single[0]["type"], "CARD")

        # 2. Multiple records -> TABLE
        widgets_multi = generate_widgets(
            {"records": [{"id": 1}, {"id": 2}]}, query_plan="simple"
        )
        self.assertEqual(widgets_multi[0]["type"], "TABLE")

        # 3. Comparison -> TABLE and BAR_CHART
        widgets_compare = generate_widgets({}, query_plan="compare")
        types_compare = [w["type"] for w in widgets_compare]
        self.assertIn("TABLE", types_compare)
        self.assertIn("BAR_CHART", types_compare)

        # 4. Trend -> LINE_CHART and AREA_CHART
        widgets_trend = generate_widgets({}, query_plan="trend")
        types_trend = [w["type"] for w in widgets_trend]
        self.assertIn("LINE_CHART", types_trend)
        self.assertIn("AREA_CHART", types_trend)

        # 5. Distribution -> PIE_CHART and BAR_CHART
        widgets_dist = generate_widgets({}, query_plan="distribution")
        types_dist = [w["type"] for w in widgets_dist]
        self.assertIn("PIE_CHART", types_dist)
        self.assertIn("BAR_CHART", types_dist)


if __name__ == "__main__":
    unittest.main()
