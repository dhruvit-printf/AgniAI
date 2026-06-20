"""Unit tests for shipped widget engine query-aware behavior."""

from __future__ import annotations

import unittest

from widget_engine import generate_widgets


class TestVisualizationQueryAware(unittest.TestCase):

    def test_comparison_guarantee(self):
        combined = {
            "left": {"label": "A", "data": [{"id": 1}]},
            "right": {"label": "B", "data": [{"id": 2}]},
            "comparison": {"averageScore": {"higher": "A", "lower": "B"}},
        }
        widgets = generate_widgets(combined, query_type="compare", intent={})
        types = [w["type"] for w in widgets]
        self.assertIn("BAR_CHART", types)
        self.assertIn("TABLE", types)

    def test_cross_filter_guarantee(self):
        combined = {
            "sections": [
                {"label": "Common Records", "data": [{"id": 1}, {"id": 2}]}
            ]
        }
        widgets = generate_widgets(combined, query_type="cross_filter", intent={})
        types = [w["type"] for w in widgets]
        self.assertIn("TABLE", types)

    def test_metric_widget_selection(self):
        combined = {
            "sections": [{"label": "Result", "data": [{"count": 5}]}]
        }
        widgets = generate_widgets(combined, query_type="simple", intent={})
        types = [w["type"] for w in widgets]
        self.assertIn("CARD", types)
        self.assertIn("METRIC_CARD", types)


if __name__ == "__main__":
    unittest.main()
