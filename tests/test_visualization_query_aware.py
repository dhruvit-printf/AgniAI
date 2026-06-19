"""
tests/test_visualization_query_aware.py
=======================================
Unit tests for the query-aware visualization widget guarantees.
"""

import unittest

from query_planner import QueryPlan, QueryType, SubOperation
from visualization_engine import generate_widgets


class TestVisualizationQueryAware(unittest.TestCase):

    def test_comparison_guarantee(self):
        # A comparison result with only scalar metrics still yields a BAR_CHART.
        combined = {"metrics": {"A": 10, "B": 20}}
        plan = QueryPlan(
            query_type=QueryType.COMPARISON,
            operations=[],
            confidence=1.0,
            raw_query="compare",
            reasoning="test",
        )
        widgets = generate_widgets(combined, query_plan=plan)
        types = [w["type"] for w in widgets]
        self.assertIn("BAR_CHART", types)

    def test_cross_filter_guarantee(self):
        # A cross-filter result yields a TABLE.
        combined = {"matchCount": 5}
        plan = QueryPlan(
            query_type=QueryType.CROSS_FILTER,
            operations=[],
            confidence=1.0,
            raw_query="cross filter",
            reasoning="test",
        )
        widgets = generate_widgets(combined, query_plan=plan)
        types = [w["type"] for w in widgets]
        self.assertIn("TABLE", types)

    def test_analytics_guarantee(self):
        # Analytics with group_by yields BAR_CHART and PIE_CHART.
        combined = {"something": "else"}
        op = SubOperation(raw_fragment="grouped by section", group_by="section")
        plan = QueryPlan(
            query_type=QueryType.ANALYTICS,
            operations=[op],
            confidence=1.0,
            raw_query="grouped",
            reasoning="test",
        )
        widgets = generate_widgets(combined, query_plan=plan)
        types = [w["type"] for w in widgets]
        self.assertIn("BAR_CHART", types)
        self.assertIn("PIE_CHART", types)

    def test_multi_operation_and_section_hints(self):
        # Multi-operation yields TABLE and supports section hints.
        combined = {
            "queryType": "multi_independent",
            "sections": [
                {"label": "Section A", "data": {}, "widget_hint": "radial_chart"}
            ],
        }
        plan = QueryPlan(
            query_type=QueryType.MULTI_OPERATION,
            operations=[],
            confidence=1.0,
            raw_query="multi",
            reasoning="test",
        )
        widgets = generate_widgets(combined, query_plan=plan)
        types = [w["type"] for w in widgets]
        self.assertIn("TABLE", types)
        self.assertIn("RADIAL_CHART", types)

    def test_key_based_behavior_unchanged_when_query_plan_is_none(self):
        # Key-based behavior unchanged when query_plan is None
        combined = {"sport": "Cricket"}
        widgets = generate_widgets(combined, query_plan=None)
        types = {w["type"] for w in widgets}
        self.assertEqual(types, {"PIE_CHART", "BAR_CHART"})


if __name__ == "__main__":
    unittest.main()
