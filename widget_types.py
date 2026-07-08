"""
widget_types.py
===============
Canonical widget-type name resolution — the single source of truth for
legacy widget-type aliases, shared by widget_engine.py and widget_selector.py
(previously two byte-for-byte-identical copies of this alias map, plus a
redundant third subset re-declared inside widget_engine.py itself).

This module is intentionally a leaf: it must not import from widget_engine,
widget_selector, or visualization_intent, so it can be imported by any of
them without creating a cycle.
"""

from __future__ import annotations

from typing import Any, Dict

# DONUT_CHART and RADIAL_CHART are folded — donut into pie, radial into line —
# and AREA_CHART is folded into line as well.
CHART_TYPE_ALIASES: Dict[str, str] = {
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


def canonical_widget_type(value: Any) -> str:
    """Normalise any accepted widget type name (new or legacy) to canonical form."""
    text = str(value or "TABLE").upper()
    return CHART_TYPE_ALIASES.get(text, text)


# Maps a lowercase comparison_chart_override phrase to its COMPARE_CHART_*
# widget type. Shared by visualization_intent.py, widget_selector.py, and
# widget_engine.py — previously three independently-maintained copies of
# this exact key set/value mapping (see refactor plan). Note: this is ONLY
# the type-string mapping — visualization_intent.py's _comparison_widgets
# additionally appends a companion COMPARE_TABLE widget for "bar" overrides,
# which stays local to that caller and is intentionally not encoded here.
COMPARE_CHART_OVERRIDE_MAP: Dict[str, str] = {
    "line": "COMPARE_CHART_LINE",
    "bar": "COMPARE_CHART_BAR",
    "pie": "COMPARE_CHART_PIE",
    "donut": "COMPARE_CHART_PIE",
    "radial": "COMPARE_CHART_LINE",
    "area": "COMPARE_CHART_LINE",
}
