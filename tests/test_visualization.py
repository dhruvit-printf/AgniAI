"""Unit tests for the schema inference engine."""

from __future__ import annotations

import unittest

from widget_engine import build_formatted_data


class TestWidgetEngine(unittest.TestCase):

    def test_simple_record_yields_card(self):
        answer = {"sections": [{"label": "Result", "data": [{"id": 1}]}]}
        res = build_formatted_data(answer, query_type="simple", intent={})
        self.assertEqual(res["type"], "CARD")

    def test_multiple_records_yield_table(self):
        answer = {
            "sections": [
                {"label": "Result", "data": [{"id": 1}, {"id": 2}, {"id": 3}]}
            ]
        }
        res = build_formatted_data(answer, query_type="simple", intent={})
        self.assertEqual(res["type"], "TABLE")

    def test_compare_widgets_yield_bar_chart(self):
        answer = {
            "left": {"label": "PPT", "data": [{"id": 1}]},
            "right": {"label": "BPET", "data": [{"id": 2}]},
            "comparison": {"averageScore": {"higher": "PPT", "lower": "BPET"}},
        }
        res = build_formatted_data(answer, query_type="compare", intent={})
        self.assertEqual(res["type"], "CHART_BAR")

    def test_distribution_widgets_yield_pie_chart(self):
        answer = {
            "sections": [
                {
                    "label": "Result",
                    "data": [{"sport": "Cricket"}, {"sport": "Football"}],
                }
            ]
        }
        res = build_formatted_data(answer, query_type="distribution", intent={})
        self.assertEqual(res["type"], "CHART_PIE")

    def test_trend_widgets_yield_line_chart(self):
        answer = {
            "sections": [
                {
                    "label": "Result",
                    "data": [{"date": "2026-06-19", "score": 88}],
                }
            ]
        }
        res = build_formatted_data(answer, query_type="trend", intent={})
        self.assertEqual(res["type"], "CHART_LINE")


if __name__ == "__main__":
    unittest.main()
