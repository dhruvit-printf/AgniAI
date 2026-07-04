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
from dataclasses import dataclass
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
    """Normalise legacy/alias widget type names to canonical constants."""
    _ALIASES: Dict[str, str] = {
        "BAR_CHART": "CHART_BAR",  # legacy alias
        "LINE_CHART": "CHART_LINE",  # legacy alias
        "AREA_CHART": "CHART_LINE",  # folded into line
        "RADIAL_CHART": "CHART_LINE",  # folded into line
        "PIE_CHART": "CHART_PIE",  # legacy alias
        "DONUT_CHART": "CHART_PIE",  # folded into pie
        "COMPARE_BAR_CHART": "COMPARE_CHART_BAR",  # legacy alias
        "COMPARE_LINE_CHART": "COMPARE_CHART_LINE",  # legacy alias
        "COMPARE_PIE_CHART": "COMPARE_CHART_PIE",  # legacy alias
    }
    return _ALIASES.get((wt or "").upper(), (wt or "TABLE").upper())


# ── data ─────────────────────────────────────────────────────────────────────


@dataclass
class WidgetSpec:
    """Blueprint for one widget.  No data — only build instructions."""

    widget_type: str  # "TABLE" | "CARD" | "CHART_BAR" | "CHART_LINE" | "CHART_PIE" | "COMPARE_TABLE" | "COMPARE_CARD" | "COMPARE_CHART_BAR" | "COMPARE_CHART_LINE" | "COMPARE_CHART_PIE"
    widget_id: str  # Deterministic, unique within this response
    title: str  # Human-readable title rendered on the widget
    source_hint: str = "primary"  # "primary"|"summary"|"left"|"right"|"section"
    section_label: str = ""  # Non-empty for multi_independent TABLE specs


# ── engine ───────────────────────────────────────────────────────────────────


class WidgetSelector:
    """Routes a declarative widget plan into deterministic WidgetSpec objects."""

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
        visualization_intent: Optional[Dict[str, Any]] = None,
    ) -> List[WidgetSpec]:
        category = (intent.get("category") or "").strip()
        operation = (intent.get("operation") or intent.get("subcategory") or "").strip()
        cat_slug = _slug(category)
        op_slug = _slug(operation)

        plan: List[Dict[str, Any]] = []
        if isinstance(visualization_intent, dict):
            widgets = visualization_intent.get("widgets")
            if isinstance(widgets, list):
                plan = [w for w in widgets if isinstance(w, dict)]

        if not plan and frontend_override_type:
            plan = [{"type": frontend_override_type}]
        elif not plan and comparison_chart_override:
            qt = (query_type or "").strip().lower()
            if qt in ("compare", "comparison"):
                override = comparison_chart_override.strip().lower()
                mapped = {
                    "line": "COMPARE_CHART_LINE",
                    "bar": "COMPARE_CHART_BAR",
                    "pie": "COMPARE_CHART_PIE",
                    "donut": "COMPARE_CHART_PIE",
                    "radial": "COMPARE_CHART_LINE",
                    "area": "COMPARE_CHART_LINE",
                }.get(override, "COMPARE_TABLE")
                plan = [{"type": mapped}]

        if not plan and primary_widget_type:
            plan = [{"type": primary_widget_type}]
        if not plan:
            plan = [{"type": "TABLE"}]

        specs: List[WidgetSpec] = []
        for index, descriptor in enumerate(plan):
            widget_type = _canonical(descriptor.get("type") or "TABLE")
            title = str(descriptor.get("title") or _title(category, operation))
            source_hint = str(descriptor.get("source_hint") or "primary")
            section_label = str(descriptor.get("section_label") or "")
            widget_id = str(
                descriptor.get("widget_id")
                or self._widget_id(widget_type, cat_slug, op_slug, section_label, index)
            )
            specs.append(
                WidgetSpec(
                    widget_type=widget_type,
                    widget_id=widget_id,
                    title=title,
                    source_hint=source_hint,
                    section_label=section_label,
                )
            )

        return specs

    @staticmethod
    def _widget_id(
        widget_type: str,
        cat_slug: str,
        op_slug: str,
        section_label: str,
        index: int,
    ) -> str:
        base = _slug(section_label) if section_label else ""
        if widget_type.startswith("COMPARE_"):
            return widget_type.lower()
        if widget_type == "CARD":
            return f"{cat_slug}_{op_slug}_card" if op_slug else f"{cat_slug}_card"
        if widget_type == "TABLE":
            if base:
                return f"{base}_table"
            return f"{cat_slug}_{op_slug}_table" if op_slug else f"{cat_slug}_table"
        suffix = widget_type.lower().replace("chart_", "").replace("_chart", "")
        if base:
            return f"{base}_{suffix}"
        return f"{cat_slug}_{op_slug}_{suffix}" if op_slug else f"{cat_slug}_{suffix}"
