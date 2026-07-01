"""
widget_selector.py
==================
Widget Selection Engine.

Given query context (type, intent, combined-result shape, analysis),
returns an ordered List[WidgetSpec] that describes WHAT to build.
No data construction happens here — only selection.

Widget order contract: CARD → CHART → TABLE
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ── helpers ──────────────────────────────────────────────────────────────────


def _slug(text: str) -> str:
    """Stable lowercase snake_case slug safe for use in widget IDs."""
    t = re.sub(r"[^a-zA-Z0-9\s]", "", str(text or "")).strip()
    return re.sub(r"\s+", "_", t).lower() or "widget"


def _title(category: str, operation: str = "") -> str:
    parts = [p.strip() for p in (category, operation) if p and p.strip()]
    return " ".join(parts) or "Results"


def _canonical(wt: str) -> str:
    """Normalise alias widget types to canonical constants."""
    _ALIASES: Dict[str, str] = {
        "CHART_BAR": "BAR_CHART",  # legacy alias
        "CHART_LINE": "LINE_CHART",  # legacy alias
        "CHART_PIE": "PIE_CHART",  # legacy alias
        "AREA_CHART": "AREA_CHART",
        "DONUT_CHART": "DONUT_CHART",
        "RADIAL_CHART": "RADIAL_CHART",
    }
    return _ALIASES.get((wt or "").upper(), (wt or "TABLE").upper())


# ── data ─────────────────────────────────────────────────────────────────────


@dataclass
class WidgetSpec:
    """Blueprint for one widget.  No data — only build instructions."""

    widget_type: str  # "TABLE" | "CARD" | "BAR_CHART" | "LINE_CHART" | "AREA_CHART" | "PIE_CHART" | "DONUT_CHART" | "RADIAL_CHART"
    widget_id: str  # Deterministic, unique within this response
    title: str  # Human-readable title rendered on the widget
    source_hint: str = "primary"  # "primary"|"summary"|"left"|"right"|"section"
    section_label: str = ""  # Non-empty for multi_independent TABLE specs


# ── engine ───────────────────────────────────────────────────────────────────


class WidgetSelector:
    """
    Selects widget types, IDs, titles, and order for a given query plan.

    All decisions are shape-aware (query_type, record count, presence of
    analysis.statistics) but NOT data-value aware.
    """

    def select(
        self,
        *,
        query_type: str,
        intent: Dict[str, Any],
        combined_result: Any,
        primary_widget_type: str,
        analysis: Optional[Dict[str, Any]] = None,
        frontend_override_type: Optional[str] = None,
        comparison_chart_override: Optional[str] = None,
    ) -> List[WidgetSpec]:
        qt = (query_type or "simple").strip().lower()
        category = (intent.get("category") or "").strip()
        operation = (intent.get("operation") or intent.get("subcategory") or "").strip()
        cat_slug = _slug(category)
        op_slug = _slug(operation)
        has_stats = bool((analysis or {}).get("statistics"))

        # Honour explicit frontend widget-type override
        if frontend_override_type:
            return self._frontend_override(
                frontend_override_type, category, operation, cat_slug, op_slug
            )

        if qt in ("compare", "comparison"):
            return self._comparison(combined_result, cat_slug, comparison_chart_override)
        if qt == "cross_filter":
            return self._cross_filter(category, cat_slug)
        if qt == "multi_independent":
            return self._multi_independent(combined_result)
        if qt == "trend":
            return self._trend(category, operation, cat_slug, op_slug)
        if qt == "distribution":
            return self._distribution(category, operation, cat_slug, op_slug)
        return self._simple(
            primary_widget_type, category, operation, cat_slug, op_slug, has_stats
        )

    # ── per-query-type builders ───────────────────────────────────────────────

    def _frontend_override(
        self,
        override_type: str,
        category: str,
        operation: str,
        cat_slug: str,
        op_slug: str,
    ) -> List[WidgetSpec]:
        wt = _canonical(override_type)
        title = _title(category, operation)
        if wt == "CARD":
            return [WidgetSpec("CARD", f"{cat_slug}_card", title, "primary")]
        if wt == "TABLE":
            tid = f"{cat_slug}_{op_slug}_table" if op_slug else f"{cat_slug}_table"
            return [WidgetSpec("TABLE", tid, title, "primary")]
        chart_suffix = wt.lower().replace("chart_", "").replace("_chart", "")
        cid = f"{cat_slug}_{chart_suffix}" if cat_slug else chart_suffix
        tid = f"{cat_slug}_table" if cat_slug else "table"
        return [
            WidgetSpec(wt, cid, title, "primary"),
            WidgetSpec(
                "TABLE",
                tid,
                f"{category} Records" if category else "Records",
                "primary",
            ),
        ]

    def _simple(
        self,
        primary_wt: str,
        category: str,
        operation: str,
        cat_slug: str,
        op_slug: str,
        has_stats: bool,
    ) -> List[WidgetSpec]:
        title = _title(category, operation)
        specs: List[WidgetSpec] = []

        if primary_wt == "CARD":
            cid = f"{cat_slug}_{op_slug}_card" if op_slug else f"{cat_slug}_card"
            specs.append(WidgetSpec("CARD", cid, title, "primary"))
            return specs

        if has_stats:
            specs.append(
                WidgetSpec(
                    "CARD", f"{cat_slug}_summary", f"{category} Summary", "summary"
                )
            )

        pwt = _canonical(primary_wt)
        if pwt in (
            "BAR_CHART",
            "LINE_CHART",
            "AREA_CHART",
            "PIE_CHART",
            "DONUT_CHART",
            "RADIAL_CHART",
            "CHART_BAR",
            "CHART_LINE",
            "CHART_PIE",
        ):
            suffix = pwt.lower().replace("chart_", "")
            cid = (
                f"{cat_slug}_{op_slug}_{suffix}" if op_slug else f"{cat_slug}_{suffix}"
            )
            specs.append(WidgetSpec(pwt, cid, title, "primary"))
            tid = f"{cat_slug}_{op_slug}_table" if op_slug else f"{cat_slug}_table"
            specs.append(
                WidgetSpec(
                    "TABLE",
                    tid,
                    f"{category} Records" if category else "Records",
                    "primary",
                )
            )
        else:
            tid = f"{cat_slug}_{op_slug}_table" if op_slug else f"{cat_slug}_table"
            specs.append(WidgetSpec("TABLE", tid, title, "primary"))

        return specs

    def _comparison(
        self,
        combined_result: Any,
        cat_slug: str,
        comparison_chart_override: Optional[str] = None,
    ) -> List[WidgetSpec]:
        _ALIASES = {
            "COMPARE_CHART_BAR": "COMPARE_BAR_CHART",
            "COMPARE_CHART_LINE": "COMPARE_LINE_CHART",
            "COMPARE_CHART_PIE": "COMPARE_PIE_CHART",
        }
        raw_viz = "COMPARE_TABLE"
        if isinstance(combined_result, dict) and combined_result.get(
            "visualizationType"
        ):
            raw_viz = combined_result["visualizationType"]
        viz_type = _ALIASES.get(raw_viz, raw_viz)
        if viz_type == "COMPARE_CARD":
            viz_type = "COMPARE_TABLE"

        # Honor user's explicit chart type request
        _CHART_TYPE_TO_COMPARE = {
            "line": "COMPARE_LINE_CHART",
            "bar": "COMPARE_BAR_CHART",
            "pie": "COMPARE_PIE_CHART",
            "donut": "COMPARE_PIE_CHART",
            "radial": "COMPARE_PIE_CHART",
            "area": "COMPARE_LINE_CHART",
        }
        if comparison_chart_override and isinstance(comparison_chart_override, str):
            mapped = _CHART_TYPE_TO_COMPARE.get(comparison_chart_override.lower())
            if mapped:
                viz_type = mapped

        left_label = ""
        right_label = ""
        if isinstance(combined_result, dict):
            left_label = str((combined_result.get("left") or {}).get("label") or "Left")
            right_label = str(
                (combined_result.get("right") or {}).get("label") or "Right"
            )
        vs_title = (
            f"{left_label} vs {right_label}"
            if left_label and right_label
            else "Comparison"
        )

        specs: List[WidgetSpec] = []
        if viz_type == "COMPARE_TABLE":
            specs.append(
                WidgetSpec(
                    "COMPARE_TABLE", "compare_table", f"{vs_title} — Records", "primary"
                )
            )
        elif viz_type == "COMPARE_BAR_CHART":
            specs.append(
                WidgetSpec(
                    "COMPARE_BAR_CHART",
                    "compare_bar",
                    f"{vs_title} — Score Comparison",
                    "primary",
                )
            )
            specs.append(
                WidgetSpec(
                    "COMPARE_TABLE", "compare_table", f"{vs_title} — Records", "primary"
                )
            )
        elif viz_type == "COMPARE_LINE_CHART":
            specs.append(
                WidgetSpec(
                    "COMPARE_LINE_CHART",
                    "compare_line",
                    f"{vs_title} — Trend",
                    "primary",
                )
            )
        elif viz_type == "COMPARE_PIE_CHART":
            specs.append(
                WidgetSpec(
                    "COMPARE_PIE_CHART",
                    "compare_pie",
                    f"{vs_title} — Distribution",
                    "primary",
                )
            )

        return specs

    def _cross_filter(self, category: str, cat_slug: str) -> List[WidgetSpec]:
        tid = f"{cat_slug}_results" if cat_slug else "filter_results"
        return [
            WidgetSpec("CARD", "filter_summary", "Filter Summary", "summary"),
            WidgetSpec("TABLE", tid, "Matching Records", "primary"),
        ]

    def _multi_independent(self, combined_result: Any) -> List[WidgetSpec]:
        specs: List[WidgetSpec] = []
        sections: List[Any] = []
        if isinstance(combined_result, dict):
            sections = combined_result.get("sections") or []
        if sections:
            for sec in sections:
                if isinstance(sec, dict):
                    label = sec.get("label") or "Section"
                    sec_slug = _slug(label)
                    specs.append(
                        WidgetSpec(
                            "TABLE",
                            f"{sec_slug}_table",
                            label,
                            "section",
                            section_label=label,
                        )
                    )
        else:
            specs.append(WidgetSpec("TABLE", "results_table", "Results", "primary"))
        return specs

    def _trend(
        self, category: str, operation: str, cat_slug: str, op_slug: str
    ) -> List[WidgetSpec]:
        return [
            WidgetSpec("CARD", f"{cat_slug}_summary", f"{category} Summary", "summary"),
            WidgetSpec(
                "LINE_CHART",
                f"{cat_slug}_trend",
                _title(category, operation),
                "primary",
            ),
            WidgetSpec(
                "TABLE", f"{cat_slug}_records", f"{category} Records", "primary"
            ),
        ]

    def _distribution(
        self, category: str, operation: str, cat_slug: str, op_slug: str
    ) -> List[WidgetSpec]:
        return [
            WidgetSpec("CARD", f"{cat_slug}_summary", f"{category} Summary", "summary"),
            WidgetSpec(
                "PIE_CHART",
                f"{cat_slug}_distribution",
                _title(category, operation),
                "primary",
            ),
            WidgetSpec(
                "TABLE", f"{cat_slug}_records", f"{category} Records", "primary"
            ),
        ]
