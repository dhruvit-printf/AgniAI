"""Unit tests for shipped widget engine query-aware behavior."""

from __future__ import annotations

import unittest

from widget_engine import build_formatted_data


class TestVisualizationQueryAware(unittest.TestCase):

    def test_comparison_guarantee(self):
        combined = {
            "left": {"label": "A", "data": [{"id": 1}]},
            "right": {"label": "B", "data": [{"id": 2}]},
            "comparison": {"averageScore": {"higher": "A", "lower": "B"}},
        }
        res = build_formatted_data(combined, query_type="compare", intent={})
        self.assertEqual(res["type"], "CHART_LINE")

    def test_cross_filter_guarantee(self):
        combined = {
            "sections": [{"label": "Common Records", "data": [{"id": 1}, {"id": 2}]}]
        }
        res = build_formatted_data(combined, query_type="cross_filter", intent={})
        self.assertEqual(res["type"], "TABLE")

    def test_metric_widget_selection(self):
        combined = {"sections": [{"label": "Result", "data": [{"count": 5}]}]}
        res = build_formatted_data(combined, query_type="simple", intent={})
        self.assertEqual(res["type"], "TABLE")

    def test_frontend_widget_override_wins(self):
        combined = {
            "sections": [{"label": "Result", "data": [{"count": 5}, {"count": 7}]}]
        }
        res = build_formatted_data(
            combined,
            query_type="simple",
            intent={},
            # DONUT_CHART is a legacy input alias; it folds to CHART_PIE on output.
            visualization_intent={"requested_widget_type": "DONUT_CHART"},
        )
        self.assertEqual(res["type"], "CHART_PIE")

    def test_summary_override_for_equipment_stats(self):
        combined = {"sections": [{"label": "Result", "data": [{"count": 12}]}]}
        res = build_formatted_data(
            combined,
            query_type="simple",
            intent={
                "category": "Equipment",
                "operation": "Stats",
                "responseType": "Summary",
            },
        )
        self.assertEqual(res["type"], "CHART_BAR")

    def test_detailed_equipment_stats_stays_table(self):
        combined = {
            "sections": [{"label": "Result", "data": [{"count": 12}, {"count": 8}]}]
        }
        res = build_formatted_data(
            combined,
            query_type="simple",
            intent={
                "category": "Equipment",
                "operation": "Stats",
                "responseType": "Detailed",
            },
        )
        self.assertEqual(res["type"], "TABLE")

    def test_summary_override_for_overall_performance(self):
        combined = {"sections": [{"label": "Result", "data": [{"score": 91}]}]}
        res = build_formatted_data(
            combined,
            query_type="simple",
            intent={
                "category": "Overall",
                "operation": "OverallPerformance",
                "responseType": "Summary",
            },
        )
        self.assertEqual(res["type"], "CARD")


if __name__ == "__main__":
    unittest.main()
