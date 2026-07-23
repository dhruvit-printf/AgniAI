"""
Guard tests freezing today's widget-alias / chart-override-map behavior
before any consolidation (see visualization pipeline refactor plan).

These pin the CURRENT resolved values by calling the real functions
directly, not hand-copied expectations, so any future consolidation step
that accidentally changes a resolved value is caught immediately.
"""

from __future__ import annotations

import unittest

from visualization_intent import _comparison_widgets
from widget_engine import (
    _build_compare_bar,
    _build_compare_card,
    _build_compare_line,
    _build_compare_pie,
    _build_compare_table,
    _build_widget_data,
    _canonical_widget_type,
    build_attendance_calendar_data,
    build_bar_chart_data,
    build_card_data,
    build_line_chart_data,
    build_pie_chart_data,
    build_table_data,
)
from widget_selector import WidgetSelector, WidgetSpec, _canonical


class TestChartTypeAliasEquivalence(unittest.TestCase):
    """widget_engine._canonical_widget_type and widget_selector._canonical
    must resolve every input identically today (this is Step 1's premise —
    if either of these assertions ever fails, the two are NOT safe to
    consolidate without a behavior change)."""

    ALIASES = {
        "BAR_CHART": "CHART_BAR",
        "LINE_CHART": "CHART_LINE",
        "AREA_CHART": "CHART_LINE",
        "RADIAL_CHART": "CHART_LINE",
        "PIE_CHART": "CHART_PIE",
        "DONUT_CHART": "CHART_PIE",
        "COMPARE_BAR_CHART": "COMPARE_CHART_BAR",
        "COMPARE_LINE_CHART": "COMPARE_CHART_LINE",
        "COMPARE_PIE_CHART": "COMPARE_CHART_PIE",
    }
    CANONICAL_PASSTHROUGH = ["TABLE", "CARD", "CHART_BAR", "CHART_LINE", "CHART_PIE"]

    def test_aliases_resolve_identically(self):
        for alias, expected in self.ALIASES.items():
            with self.subTest(alias=alias):
                self.assertEqual(_canonical_widget_type(alias), expected)
                self.assertEqual(_canonical(alias), expected)

    def test_canonical_passthrough_identical(self):
        for value in self.CANONICAL_PASSTHROUGH:
            with self.subTest(value=value):
                self.assertEqual(_canonical_widget_type(value), value)
                self.assertEqual(_canonical(value), value)

    def test_falsy_input_defaults_to_table_identically(self):
        for value in (None, ""):
            with self.subTest(value=value):
                self.assertEqual(_canonical_widget_type(value), "TABLE")
                self.assertEqual(_canonical(value), "TABLE")

    def test_unrecognized_truthy_passthrough_identical(self):
        self.assertEqual(_canonical_widget_type("SOMETHING_WEIRD"), "SOMETHING_WEIRD")
        self.assertEqual(_canonical("SOMETHING_WEIRD"), "SOMETHING_WEIRD")


class TestCompareOverrideMapCurrentBehavior(unittest.TestCase):
    """Pins today's resolved value for each of the 3 independently-maintained
    "compare chart override" implementations. NOTE: these are NOT proven
    equivalent to each other — visualization_intent._comparison_widgets has
    a confirmed asymmetry (its "bar" case returns a 2-widget list, not a
    single type), and widget_selector's dict only fires as a fallback when
    visualization_intent doesn't already supply a widgets plan. This test
    exists to freeze each one's OWN behavior, not to assert they agree.
    """

    OVERRIDE_KEYS = ("line", "bar", "pie", "donut", "radial", "area")

    def test_visualization_intent_comparison_widgets(self):
        # bar is asymmetric: returns a companion COMPARE_TABLE too.
        self.assertEqual(
            _comparison_widgets("bar"),
            [{"type": "COMPARE_CHART_BAR"}, {"type": "COMPARE_TABLE"}],
        )
        self.assertEqual(_comparison_widgets("line"), [{"type": "COMPARE_CHART_LINE"}])
        self.assertEqual(_comparison_widgets("area"), [{"type": "COMPARE_CHART_LINE"}])
        self.assertEqual(
            _comparison_widgets("radial"), [{"type": "COMPARE_CHART_LINE"}]
        )
        self.assertEqual(_comparison_widgets("pie"), [{"type": "COMPARE_CHART_PIE"}])
        self.assertEqual(_comparison_widgets("donut"), [{"type": "COMPARE_CHART_PIE"}])
        self.assertEqual(_comparison_widgets(None), [{"type": "COMPARE_TABLE"}])
        self.assertEqual(_comparison_widgets("unknown"), [{"type": "COMPARE_TABLE"}])

    def test_widget_selector_comparison_override_fallback_path(self):
        selector = WidgetSelector()
        expected = {
            "line": "COMPARE_CHART_LINE",
            "bar": "COMPARE_CHART_BAR",
            "pie": "COMPARE_CHART_PIE",
            "donut": "COMPARE_CHART_PIE",
            "radial": "COMPARE_CHART_LINE",
            "area": "COMPARE_CHART_LINE",
        }
        for key in self.OVERRIDE_KEYS:
            with self.subTest(key=key):
                specs = selector.select(
                    query_type="compare",
                    intent={"category": "Performance", "operation": "Top"},
                    combined_result={},
                    primary_widget_type="TABLE",
                    comparison_chart_override=key,
                    visualization_intent=None,
                )
                self.assertEqual(len(specs), 1)
                self.assertEqual(specs[0].widget_type, expected[key])


class TestBuildWidgetDataDictDispatch(unittest.TestCase):
    """_build_widget_data (widget_engine.py) now dispatches via BUILDERS
    dicts instead of an if/elif chain (Step 3 of the refactor plan). Every
    canonical widget type must still produce exactly what calling its
    builder function directly produces — this is the "behavioral
    equivalence" check the plan called for, not just "the dispatch table
    exists"."""

    NON_COMPARE_RESULT = {
        "sections": [
            {
                "label": "Result",
                "data": [
                    {"agniveerNo": "A1", "bestTotal": 80, "date": "2026-01-01"},
                    {"agniveerNo": "A2", "bestTotal": 60, "date": "2026-01-02"},
                ],
            }
        ]
    }
    COMPARE_RESULT = {
        "left": {"label": "A", "data": [{"agniveerNo": "A1", "bestTotal": 80}]},
        "right": {"label": "B", "data": [{"agniveerNo": "A2", "bestTotal": 60}]},
    }
    INTENT: dict = {}

    def _spec(self, widget_type: str, source_hint: str = "primary") -> WidgetSpec:
        return WidgetSpec(
            widget_type=widget_type,
            widget_id="test",
            title="Test",
            source_hint=source_hint,
        )

    def test_compare_widget_types_match_direct_builder_call(self):
        cases = {
            "COMPARE_CARD": lambda: _build_compare_card(self.COMPARE_RESULT),
            "COMPARE_TABLE": lambda: _build_compare_table(self.COMPARE_RESULT),
            "COMPARE_CHART_BAR": lambda: _build_compare_bar(self.COMPARE_RESULT),
            "COMPARE_CHART_LINE": lambda: _build_compare_line(self.COMPARE_RESULT),
            "COMPARE_CHART_PIE": lambda: _build_compare_pie(
                self.COMPARE_RESULT, intent=self.INTENT
            ),
        }
        for widget_type, direct_call in cases.items():
            with self.subTest(widget_type=widget_type):
                dispatched = _build_widget_data(
                    self._spec(widget_type),
                    self.COMPARE_RESULT,
                    "compare",
                    self.INTENT,
                    None,
                )
                self.assertEqual(dispatched, direct_call())

    def test_compare_widget_legacy_aliases_fold_to_same_result(self):
        # Legacy alias input must fold to the same canonical builder output.
        for alias, canonical in (
            ("COMPARE_BAR_CHART", "COMPARE_CHART_BAR"),
            ("COMPARE_LINE_CHART", "COMPARE_CHART_LINE"),
            ("COMPARE_PIE_CHART", "COMPARE_CHART_PIE"),
        ):
            with self.subTest(alias=alias):
                via_alias = _build_widget_data(
                    self._spec(alias), self.COMPARE_RESULT, "compare", self.INTENT, None
                )
                via_canonical = _build_widget_data(
                    self._spec(canonical),
                    self.COMPARE_RESULT,
                    "compare",
                    self.INTENT,
                    None,
                )
                self.assertEqual(via_alias, via_canonical)

    def test_chart_widget_types_match_direct_builder_call(self):
        cases = {
            "CHART_BAR": lambda: build_bar_chart_data(self.NON_COMPARE_RESULT),
            "CHART_LINE": lambda: build_line_chart_data(self.NON_COMPARE_RESULT),
            "CHART_PIE": lambda: build_pie_chart_data(
                self.NON_COMPARE_RESULT, intent=self.INTENT
            ),
            "ATTENDANCE_CALENDAR": lambda: build_attendance_calendar_data(
                self.NON_COMPARE_RESULT, self.INTENT
            ),
        }
        for widget_type, direct_call in cases.items():
            with self.subTest(widget_type=widget_type):
                dispatched = _build_widget_data(
                    self._spec(widget_type),
                    self.NON_COMPARE_RESULT,
                    "simple",
                    self.INTENT,
                    None,
                )
                self.assertEqual(dispatched, direct_call())

    def test_chart_bar_falls_back_to_compare_bar_when_query_type_is_compare(self):
        dispatched = _build_widget_data(
            self._spec("CHART_BAR"), self.COMPARE_RESULT, "compare", self.INTENT, None
        )
        self.assertEqual(dispatched, _build_compare_bar(self.COMPARE_RESULT))

    def test_unmatched_type_falls_back_to_table(self):
        from widget_engine import _extract_records

        dispatched = _build_widget_data(
            self._spec("SOMETHING_UNKNOWN"),
            self.NON_COMPARE_RESULT,
            "simple",
            self.INTENT,
            None,
        )
        flat = _extract_records(self.NON_COMPARE_RESULT, deep_flatten=True)
        self.assertEqual(dispatched, build_table_data(flat))


if __name__ == "__main__":
    unittest.main()
