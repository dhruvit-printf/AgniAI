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
        self.assertEqual(res["type"], "AREA_CHART")

    def test_cross_filter_guarantee(self):
        combined = {
            "sections": [
                {"label": "Common Records", "data": [{"id": 1}, {"id": 2}]}
            ]
        }
        res = build_formatted_data(combined, query_type="cross_filter", intent={})
        self.assertEqual(res["type"], "TABLE")

    def test_metric_widget_selection(self):
        combined = {
            "sections": [{"label": "Result", "data": [{"count": 5}]}]
        }
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
            visualization_intent={"requested_widget_type": "DONUT_CHART"},
        )
        self.assertEqual(res["type"], "DONUT_CHART")


if __name__ == "__main__":
    unittest.main()
