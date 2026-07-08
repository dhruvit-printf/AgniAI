"""
visualization_intent.py
=======================
Selects the most appropriate presentation layer from the query semantics,
intent, and returned data shape.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional

from widget_types import COMPARE_CHART_OVERRIDE_MAP


def _flatten_records(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, dict):
        if isinstance(data.get("records"), list):
            return [r for r in data["records"] if isinstance(r, dict)]
        if isinstance(data.get("sections"), list):
            records: List[Dict[str, Any]] = []
            for section in data["sections"]:
                if isinstance(section, dict) and isinstance(section.get("data"), list):
                    records.extend([r for r in section["data"] if isinstance(r, dict)])
            if records:
                return records
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]
    return []


def _has_numeric_signal(records: Iterable[Dict[str, Any]]) -> bool:
    for record in records:
        for value in record.values():
            if isinstance(value, (int, float)):
                return True
    return False


_EXPLICIT_PRESENTATION_PATTERNS = (
    (r"\bpie chart\b", "chart", "pie"),
    (r"\bbar chart\b", "chart", "bar"),
    (r"\bline chart\b", "chart", "line"),
    (r"\bdonut chart\b", "chart", "pie"),  # donut folded into pie
    (r"\bradial chart\b", "chart", "line"),  # radial folded into line
    (r"\barea chart\b", "chart", "line"),  # area folded into line
    (r"\btabular\b", "table", None),
    (r"\btable\b", "table", None),
    (r"\bcards?\b", "cards", None),
)


def _detect_explicit_presentation(text: str) -> Optional[Dict[str, Any]]:
    for pattern, presentation, chart_type in _EXPLICIT_PRESENTATION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            result: Dict[str, Any] = {
                "frontend_override": True,
                "presentation": presentation,
            }
            if chart_type:
                result["chart_type"] = chart_type
            return result
    return None


def _widget_list(*widget_types: str) -> List[Dict[str, Any]]:
    return [{"type": widget_type} for widget_type in widget_types if widget_type]


_SUMMARY_WIDGET_PLANS: Dict[tuple[str, str], List[str]] = {
    ("Performance", "top"): ["TABLE"],
    ("Performance", "bottom"): ["TABLE"],
    ("Performance", "improvement"): ["CHART_LINE"],
    ("Performance", "drop"): ["CHART_LINE"],
    ("Performance", "grading"): ["TABLE"],
    ("Performance", "gradingsummary"): ["CHART_BAR"],
    ("Performance", "average"): ["CHART_PIE"],
    ("Performance", "attemptwise"): ["CHART_LINE"],
    ("Performance", "bestattempt"): ["TABLE"],
    ("Performance", "trend"): ["CHART_LINE"],
    ("Leave", "most"): ["CHART_PIE"],
    ("Leave", "least"): ["CHART_PIE"],
    ("Leave", "current"): ["CHART_PIE"],
    ("Leave", "absconded"): ["CHART_PIE"],
    ("Medical", "bmi"): ["CHART_PIE"],
    ("Medical", "bloodgroup"): ["CHART_PIE"],
    ("Medical", "disease"): ["CHART_BAR"],
    ("Medical", "individual"): ["TABLE"],
    ("Attendance", "monthly"): ["CHART_BAR"],
    ("Attendance", "weekly"): ["CHART_BAR"],
    ("Attendance", "daily"): ["ATTENDANCE_CALENDAR"],
    ("Attendance", "present"): ["CHART_PIE"],
    ("Attendance", "summary"): ["CHART_LINE"],
    ("Strength", "strengthbreakdown"): ["CHART_LINE"],
    ("Verification", "pending"): ["CARD"],
    ("Verification", "sent"): ["CARD"],
    ("Verification", "notresponded"): ["CARD"],
    ("Verification", "completed"): ["CARD"],
    ("Verification", "verified"): ["CARD"],
    ("Verification", "rejected"): ["CARD"],
    ("Equipment", "stats"): ["TABLE"],
    ("Equipment", "search"): ["TABLE"],
    ("Equipment", "returned"): ["TABLE"],
    ("Equipment", "holding"): ["TABLE"],
    ("Equipment", "agniveerwise"): ["TABLE"],
    ("Distribution", "latest"): ["TABLE"],
    ("Distribution", "byunit"): ["CHART_BAR"],
    ("Distribution", "unassigned"): ["TABLE"],
    ("Distribution", "topunit"): ["CHART_BAR"],
    ("Skills", "bysport"): ["TABLE"],
    ("Skills", "byclass"): ["CHART_BAR"],
    ("Schedule", "today"): ["TABLE"],
    ("Schedule", "company"): ["TABLE"],
    ("Schedule", "date"): ["TABLE"],
    ("Schedule", "agniveer"): ["TABLE"],
    ("personaldetail", "info"): ["TABLE"],
    ("disqualified", "removed"): ["TABLE"],
    ("Overall", "overallperformance"): ["TABLE"],
}


def _detail_widgets_for(summary_widgets: List[str]) -> List[str]:
    return ["TABLE"]


def _comparison_widgets(override: Optional[str]) -> List[Dict[str, Any]]:
    if not override:
        return _widget_list("COMPARE_TABLE")

    normalized = override.strip().lower()
    widget_type = COMPARE_CHART_OVERRIDE_MAP.get(normalized)
    if widget_type is None:
        return _widget_list("COMPARE_TABLE")
    if widget_type == "COMPARE_CHART_BAR":
        # Bar comparisons additionally get a companion table — a decision
        # local to this caller, not shared with widget_selector.py/
        # widget_engine.py's copies of the type-string mapping above.
        return _widget_list(widget_type, "COMPARE_TABLE")
    return _widget_list(widget_type)


def _plan_widgets(
    *,
    text: str,
    intent: Dict[str, Any],
    records: List[Dict[str, Any]],
    explicit_override: Optional[Dict[str, Any]],
    comparison: bool,
    trend: bool,
    group_by: Optional[str],
    metric: str,
    query_type: str,
) -> List[Dict[str, Any]]:
    category = (intent.get("category") or "").strip()
    operation = (
        intent.get("operation")
        or intent.get("subcategory")
        or intent.get("query_type")
        or ""
    ).strip().lower()
    response_type = (intent.get("responseType") or "Summary").strip()

    if explicit_override:
        presentation = explicit_override.get("presentation")
        chart_type = explicit_override.get("chart_type")
        if presentation == "cards":
            return _widget_list("CARD")
        if presentation == "table":
            return _widget_list("TABLE")
        if presentation == "chart":
            if comparison:
                override = explicit_override.get("comparison_chart_override") or chart_type
                return _comparison_widgets(override)
            chart_map = {
                "line": "CHART_LINE",
                "bar": "CHART_BAR",
                "pie": "CHART_PIE",
                "donut": "CHART_PIE",
                "radial": "CHART_LINE",
                "area": "CHART_LINE",
            }
            return _widget_list(chart_map.get(chart_type or "", "CHART_BAR"))

    if comparison or query_type in {"compare", "comparison"}:
        override = intent.get("comparison_chart_override")
        return _comparison_widgets(override)

    if operation in ("improvement", "drop") and response_type == "Detailed":
        return _widget_list("TABLE")

    # `query_type` is a structural classification from the planner — it must
    # win over text-heuristic checks below (e.g. "breakdown"/"distribution"
    # keyword sniffing), otherwise a multi-independent query whose text just
    # happens to contain one of those words (e.g. "... and strength
    # breakdown") gets collapsed into a single combined chart instead of one
    # widget per independent section.
    if query_type == "multi_independent":
        widgets: List[Dict[str, Any]] = []
        sections = []
        if isinstance(intent.get("combined_result"), dict):
            sections = intent["combined_result"].get("sections") or []
        operations = intent.get("operations") or []
        for idx, section in enumerate(sections):
            if isinstance(section, dict):
                label = section.get("label") or "Section"
                op_intent = operations[idx] if idx < len(operations) else {}
                sec_category = (op_intent.get("category") or "").strip()
                sec_operation = (
                    op_intent.get("operation") or op_intent.get("subcategory") or ""
                ).strip().lower()
                # Same default widget-type lookup a standalone query for this
                # category/operation would use — a section doesn't get a
                # hardcoded presentation just because it's part of a
                # multi-independent bundle.
                sec_widgets = _SUMMARY_WIDGET_PLANS.get((sec_category, sec_operation)) or [
                    "TABLE"
                ]
                widgets.append(
                    {
                        "type": sec_widgets[0],
                        "title": label,
                        "section_label": label,
                        "source_hint": "section",
                    }
                )
        return widgets or _widget_list("TABLE")

    if query_type == "cross_filter":
        return _widget_list("TABLE")

    if trend or query_type == "trend" or any(
        token in text for token in ("trend", "timeline", "over months", "over time", "growth")
    ):
        return _widget_list(*(["CHART_LINE", "TABLE"] if response_type == "Detailed" else ["CHART_LINE"]))

    # AttemptWise is always a line chart (attempt-over-attempt progression) —
    # "attempt wise breakdown" is extremely common phrasing for this exact
    # operation, so it must not fall into the generic "breakdown" ->
    # CHART_PIE sniffing below meant for categories with no specific mapping.
    if (category, operation) != ("Performance", "attemptwise") and (
        query_type == "distribution"
        or any(token in text for token in ("distribution", "breakdown", "percentage", "share"))
    ):
        return _widget_list(*(["CHART_PIE", "TABLE"] if response_type == "Detailed" else ["CHART_PIE"]))

    summary_widgets = _SUMMARY_WIDGET_PLANS.get((category, operation))
    if not summary_widgets:
        # Fallbacks for queries that are clearly summary-oriented but whose
        # canonical operation name does not have an explicit row above.
        if response_type == "Detailed":
            return _widget_list("TABLE")
        return _widget_list("TABLE")

    if response_type == "Detailed":
        return _widget_list(*_detail_widgets_for(summary_widgets))
    return _widget_list(*summary_widgets)


def build_visualization_intent(
    question: str,
    intent: Dict[str, Any],
    combined_result: Any = None,
    query_type_override: Optional[str] = None,
) -> Dict[str, Any]:
    text = (question or "").strip().lower()
    operation = (
        intent.get("operation")
        or intent.get("subcategory")
        or intent.get("query_type")
        or ""
    ).lower()
    query_type = (query_type_override or intent.get("query_type") or "").lower()
    records = _flatten_records(combined_result)
    numeric_data = _has_numeric_signal(records)

    presentation = "table"
    chart_type: Optional[str] = None
    comparison = False
    trend = False
    group_by = intent.get("group_by") or intent.get("groupBy")
    metric = intent.get("metric") or ("average_score" if numeric_data else "count")

    explicit_presentation = _detect_explicit_presentation(text)
    intent_with_result = {**intent, "combined_result": combined_result}
    widgets = _plan_widgets(
        text=text,
        intent=intent_with_result,
        records=records,
        explicit_override=explicit_presentation,
        comparison=False,
        trend=False,
        group_by=group_by,
        metric=metric,
        query_type=query_type,
    )
    if explicit_presentation:
        presentation = explicit_presentation["presentation"]
        chart_type = explicit_presentation.get("chart_type")
        # Detect comparison indicators for ALL chart types, not just bar
        if "compare" in text or " vs " in text or "versus" in text:
            comparison = True
        if chart_type == "line":
            trend = "trend" in text or "timeline" in text or "growth" in text
        # If user explicitly requested a chart type AND it's a comparison,
        # keep the chart_type so the comparison widget engine can honor it.
        if comparison and chart_type:
            widgets = _comparison_widgets(chart_type)
            result = {
                "presentation": "chart",
                "chart_type": chart_type,
                "comparison": comparison,
                "comparison_chart_override": chart_type,
                "trend": trend,
                "group_by": group_by,
                "metric": metric,
                "record_count": len(records),
                "numeric_data": numeric_data,
                "frontend_override": True,
                "widgets": widgets,
            }
            return result
        if comparison:
            presentation = None
            chart_type = None
        result = {
            "presentation": presentation,
            "chart_type": chart_type,
            "comparison": comparison,
            "trend": trend,
            "group_by": group_by,
            "metric": metric,
            "record_count": len(records),
            "numeric_data": numeric_data,
            "frontend_override": True,
            "widgets": widgets,
        }
        return result

    if (
        query_type in {"compare", "comparison"}
        or "compare" in text
        or " vs " in text
        or "versus" in text
    ):
        comparison = True
    elif query_type == "trend" or any(
        token in text
        for token in ("trend", "timeline", "over months", "over time", "growth")
    ):
        presentation = "chart"
        chart_type = "line"
        trend = True
    elif query_type == "distribution" or any(
        token in text for token in ("distribution", "breakdown", "percentage", "share")
    ):
        presentation = "chart"
        chart_type = "pie"
    elif any(
        token in text
        for token in (
            "show all",
            "list all",
            "all candidates",
            "all records",
            "show all candidates",
        )
    ):
        presentation = "table"

    if comparison:
        presentation = None
        chart_type = None

    if not chart_type and presentation == "chart":
        chart_type = "bar" if comparison else ("line" if trend else "pie")

    widgets = _plan_widgets(
        text=text,
        intent=intent_with_result,
        records=records,
        explicit_override=None,
        comparison=comparison,
        trend=trend,
        group_by=group_by,
        metric=metric,
        query_type=query_type,
    )

    return {
        "presentation": presentation,
        "chart_type": chart_type,
        "comparison": comparison,
        "trend": trend,
        "group_by": group_by,
        "metric": metric,
        "record_count": len(records),
        "numeric_data": numeric_data,
        "widgets": widgets,
    }
