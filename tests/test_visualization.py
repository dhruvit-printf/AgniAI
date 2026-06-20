"""Unit tests for the shipped widget engine."""

from __future__ import annotations

import unittest

from widget_engine import generate_widgets


class TestWidgetEngine(unittest.TestCase):

    def test_simple_record_yields_card(self):
        answer = {"sections": [{"label": "Result", "data": [{"id": 1}]}]}
        widgets = generate_widgets(answer, query_type="simple", intent={})
        self.assertEqual([w["type"] for w in widgets], ["CARD"])

    def test_multiple_records_yield_table(self):
        answer = {
            "sections": [
                {"label": "Result", "data": [{"id": 1}, {"id": 2}, {"id": 3}]}
            ]
        }
        widgets = generate_widgets(answer, query_type="simple", intent={})
        self.assertEqual([w["type"] for w in widgets], ["TABLE"])

    def test_compare_widgets_include_table_and_bar(self):
        answer = {
            "left": {"label": "PPT", "data": [{"id": 1}]},
            "right": {"label": "BPET", "data": [{"id": 2}]},
            "comparison": {"averageScore": {"higher": "PPT", "lower": "BPET"}},
        }
        widgets = generate_widgets(answer, query_type="compare", intent={})
        types = [w["type"] for w in widgets]
        self.assertIn("TABLE", types)
        self.assertIn("BAR_CHART", types)

    def test_distribution_widgets_include_pie_and_bar(self):
        answer = {
            "sections": [
                {
                    "label": "Result",
                    "data": [{"sport": "Cricket"}, {"sport": "Football"}],
                }
            ]
        }
        widgets = generate_widgets(answer, query_type="distribution", intent={})
        types = [w["type"] for w in widgets]
        self.assertIn("PIE_CHART", types)
        self.assertIn("BAR_CHART", types)

    def test_trend_widgets_include_line_and_area(self):
        answer = {
            "sections": [
                {
                    "label": "Result",
                    "data": [{"date": "2026-06-19", "score": 88}],
                }
            ]
        }
        widgets = generate_widgets(answer, query_type="trend", intent={})
        types = [w["type"] for w in widgets]
        self.assertIn("LINE_CHART", types)
        self.assertIn("AREA_CHART", types)


if __name__ == "__main__":
    unittest.main()
