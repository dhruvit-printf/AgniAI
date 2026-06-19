"""
tests/test_visualization.py
===========================
Unit tests for the visualization engine widget auto-generation rules.
"""

from __future__ import annotations

import unittest

from visualization_engine import collect_all_keys, generate_widgets


class TestVisualizationEngine(unittest.TestCase):

    def test_collect_all_keys_flat(self):
        data = {"name": "Agniveer A", "sport": "Cricket", "bestTotal": 90}
        keys = collect_all_keys(data)
        self.assertEqual(keys, {"name", "sport", "besttotal"})

    def test_collect_all_keys_nested(self):
        data = {
            "status": "success",
            "data": {
                "records": [
                    {"agniveerNo": "12345", "details": {"bmiCategory": "Normal"}}
                ],
                "summary": {"count": 1},
            },
        }
        keys = collect_all_keys(data)
        expected = {
            "status",
            "data",
            "records",
            "agniveerno",
            "details",
            "bmicategory",
            "summary",
            "count",
        }
        self.assertTrue(expected.issubset(keys))

    def test_table_widget_generation(self):
        # Identifiers present -> TABLE (Priority 100)
        data = {"name": "Agniveer"}
        widgets = generate_widgets(data)
        self.assertEqual(len(widgets), 1)
        self.assertEqual(widgets[0]["type"], "TABLE")

        data_no = {"agniveerNo": "AG123"}
        widgets_no = generate_widgets(data_no)
        self.assertEqual(len(widgets_no), 1)
        self.assertEqual(widgets_no[0]["type"], "TABLE")

    def test_card_widget_generation(self):
        # Metrics present -> CARD (Priority 80)
        data = {"average": 85.5}
        widgets = generate_widgets(data)
        self.assertEqual(len(widgets), 1)
        self.assertEqual(widgets[0]["type"], "CARD")

    def test_pie_bar_widget_generation(self):
        # Categories present -> PIE_CHART & BAR_CHART (Priority 60)
        data = {"sport": "Cricket"}
        widgets = generate_widgets(data)
        types = {w["type"] for w in widgets}
        self.assertEqual(types, {"PIE_CHART", "BAR_CHART"})

    def test_line_area_widget_generation(self):
        # Time columns present -> LINE_CHART & AREA_CHART (Priority 40)
        data = {"date": "2026-06-19"}
        widgets = generate_widgets(data)
        types = {w["type"] for w in widgets}
        self.assertEqual(types, {"LINE_CHART", "AREA_CHART"})

    def test_radial_widget_generation(self):
        # Percentages present -> RADIAL_CHART (Priority 20)
        data = {"passPercentage": 95.0}
        widgets = generate_widgets(data)
        self.assertEqual(len(widgets), 1)
        self.assertEqual(widgets[0]["type"], "RADIAL_CHART")

    def test_widget_priority_sorting(self):
        # Multiple matching keys present -> Sorted by priority descending:
        # TABLE (100) > CARD (80) > PIE_CHART/BAR_CHART (60) > LINE_CHART/AREA_CHART (40) > RADIAL_CHART (20)
        data = {
            "name": "John",
            "count": 10,
            "grade": "A",
            "attempt": 1,
            "completionRate": 85.5,
        }
        widgets = generate_widgets(data)
        expected_types = [
            "TABLE",
            "CARD",
            "PIE_CHART",
            "BAR_CHART",
            "LINE_CHART",
            "AREA_CHART",
            "RADIAL_CHART",
        ]

        # Verify the length matches
        self.assertEqual(len(widgets), len(expected_types))

        # Verify the order of widgets (PIE/BAR and LINE/AREA have same priority respectively, so we check group ordering)
        self.assertEqual(widgets[0]["type"], "TABLE")
        self.assertEqual(widgets[1]["type"], "CARD")
        self.assertTrue(widgets[2]["type"] in ("PIE_CHART", "BAR_CHART"))
        self.assertTrue(widgets[3]["type"] in ("PIE_CHART", "BAR_CHART"))
        self.assertTrue(widgets[4]["type"] in ("LINE_CHART", "AREA_CHART"))
        self.assertTrue(widgets[5]["type"] in ("LINE_CHART", "AREA_CHART"))
        self.assertEqual(widgets[6]["type"], "RADIAL_CHART")

    def test_no_widgets_generated(self):
        # No matching visualizer columns
        data = {"unrelated_key": "some_value"}
        widgets = generate_widgets(data)
        self.assertEqual(widgets, [])


if __name__ == "__main__":
    unittest.main()
