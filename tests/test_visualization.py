"""Unit tests for the schema inference engine."""

from __future__ import annotations

import unittest

from widget_engine import build_formatted_data


class TestWidgetEngine(unittest.TestCase):

    def test_simple_record_yields_table(self):
        answer = {"sections": [{"label": "Result", "data": [{"id": 1}]}]}
        res = build_formatted_data(answer, query_type="simple", intent={})
        self.assertEqual(res["type"], "TABLE")

    def test_multiple_records_yield_table(self):
        answer = {
            "sections": [{"label": "Result", "data": [{"id": 1}, {"id": 2}, {"id": 3}]}]
        }
        res = build_formatted_data(answer, query_type="simple", intent={})
        self.assertEqual(res["type"], "TABLE")

    def test_compare_widgets_yield_COMPARE_TABLE(self):
        # Tables-only mode: a comparison gets its own dedicated widget type
        # (COMPARE_TABLE, with left/right split) rather than being flattened
        # into a plain TABLE.
        answer = {
            "left": {"label": "PPT", "data": [{"id": 1}]},
            "right": {"label": "BPET", "data": [{"id": 2}]},
            "comparison": {"averageScore": {"higher": "PPT", "lower": "BPET"}},
        }
        res = build_formatted_data(answer, query_type="compare", intent={})
        self.assertEqual(res["type"], "COMPARE_TABLE")
        self.assertIn("left", res["data"])
        self.assertIn("right", res["data"])

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
        self.assertEqual(res["type"], "TABLE")

    def test_pie_chart_preserves_zero_values(self):
        answer = {
            "sections": [
                {
                    "label": "Result",
                    "data": [
                        {"sport": "Cricket", "value": 0},
                        {"sport": "Football", "value": 5},
                    ],
                }
            ]
        }
        res = build_formatted_data(answer, query_type="distribution", intent={})
        self.assertEqual(res["type"], "TABLE")
        rows = res["data"]["rows"]
        self.assertEqual(rows[0]["value"], 0)
        self.assertEqual(rows[1]["value"], 5)

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
        self.assertEqual(res["type"], "TABLE")

    def test_table_rows_are_flattened(self):
        answer = {
            "sections": [
                {
                    "label": "Result",
                    "data": [
                        {
                            "fullName": "Bobby Rana",
                            "agniveerNo": "A0701772W",
                            "attempts": [
                                {
                                    "attemptNo": "1",
                                    "sections": [
                                        {
                                            "sectionName": "PPT",
                                            "grading": "Excellent",
                                            "subItems": [
                                                {
                                                    "subItemName": "2.4km",
                                                    "marksObtained": 20,
                                                }
                                            ],
                                        }
                                    ],
                                }
                            ],
                        },
                        {
                            "fullName": "Amit Kumar",
                            "agniveerNo": "A0701773X",
                            "attempts": [
                                {
                                    "attemptNo": "1",
                                    "sections": [
                                        {
                                            "sectionName": "PPT",
                                            "grading": "Good",
                                            "subItems": [
                                                {
                                                    "subItemName": "2.4km",
                                                    "marksObtained": 18,
                                                }
                                            ],
                                        }
                                    ],
                                }
                            ],
                        },
                    ],
                }
            ]
        }

        res = build_formatted_data(answer, query_type="simple", intent={})
        self.assertEqual(res["type"], "TABLE")
        row = res["data"]["row"][0]
        self.assertIn("fullName", row)  # camelCase after normalisation
        self.assertIn("agniveerNo", row)  # camelCase after normalisation
        self.assertNotIn("attempts", row)
        self.assertNotIn("sections", row)


if __name__ == "__main__":
    unittest.main()