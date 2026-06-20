"""
widget_engine.py
================
Widget inference engine for the admin pipeline.

This module performs two passes:
1. Business mapping from (category, operation/query_type) to a preferred widget.
2. Shape inference from the actual answer payload.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from normalized_models import extract_records as _extract_records

WIDGET_MAP: Dict[Tuple[str, str], str] = {
    ("Performance", "Top"): "TABLE",
    ("Performance", "Compare"): "AREA_CHART",
    ("Attendance", "Monthly"): "BAR_CHART",
    ("Attendance", "Trend"): "LINE_CHART",
    ("Medical", "BMI"): "DONUT_CHART",
    ("Attendance", "Present"): "PIE_CHART",
    ("Strength", "Overall"): "RADIAL_CHART",
    ("Equipment", "Stats"): "CARD",
}

_OPERATION_ALIASES: Dict[str, str] = {
    "TopPerformers": "Top",
    "LowestPerformers": "Bottom",
    "Comparison": "Compare",
    "MonthlyAttendance": "Monthly",
    "PresentToday": "Present",
    "StrengthBreakdown": "Overall",
    "BMIAnalysis": "BMI",
    "EquipmentSummary": "Stats",
}

_SHAPE_WIDGET_PRIORITIES = {
    "CARD": 110,
    "TABLE": 100,
    "BAR_CHART": 90,
    "LINE_CHART": 85,
    "AREA_CHART": 80,
    "PIE_CHART": 75,
    "DONUT_CHART": 70,
    "RADIAL_CHART": 65,
    "CHART": 60,
}


def _collect_keys(data: Any) -> Set[str]:
    keys: Set[str] = set()
    if isinstance(data, dict):
        for key, value in data.items():
            keys.add(key.lower())
            keys.update(_collect_keys(value))
    elif isinstance(data, list):
        for item in data:
            keys.update(_collect_keys(item))
    return keys


def _count_records(data: Any) -> int:
    return len(_extract_records(data))


def _infer_query_type(query_type: str) -> str:
    return (query_type or "").strip().lower()


def _infer_business_widget(category: str, operation: str, query_type: str) -> Optional[str]:
    operation = _OPERATION_ALIASES.get(operation, operation)
    if category and operation:
        mapped = WIDGET_MAP.get((category, operation))
        if mapped:
            return mapped

    qtype = _infer_query_type(query_type)
    if qtype == "compare":
        return "BAR_CHART"
    if qtype == "trend":
        return "LINE_CHART"
    if qtype == "distribution":
        return "DONUT_CHART"
    if qtype == "cross_filter":
        return "TABLE"
    if qtype in ("multi_independent", "multi_operation"):
        return "TABLE"
    return None


def _infer_shape_widget(label: str, data: Any, query_type: str) -> List[str]:
    keys = _collect_keys(data)
    rec_count = _count_records(data)
    qtype = _infer_query_type(query_type)
    widgets: List[str] = []

    if rec_count == 1:
        widgets.append("CARD")
    elif rec_count > 1:
        widgets.append("TABLE")

    if any(
        key in keys
        for key in (
            "count",
            "average",
            "max",
            "min",
            "score",
            "besttotal",
            "totalmarks",
            "marksobtained",
            "averagescore",
            "topscore",
            "bottomscore",
            "value",
            "values",
            "percentage",
            "rate",
        )
    ):
        widgets.append("METRIC_CARD")
        widgets.append("RADIAL_CHART")

    if any(
        key in keys
        for key in (
            "comparison",
            "difference",
            "variance",
            "winner",
            "metrics",
            "comparisonmetrics",
        )
    ):
        widgets.append("BAR_CHART")

    if qtype in ("distribution",) or any(
        key in keys
        for key in (
            "leavetype",
            "grade",
            "sport",
            "sports",
            "bloodgroup",
            "bloodtype",
            "disease",
            "unitname",
            "teamname",
            "sectionname",
            "classname",
            "category",
        )
    ):
        widgets.extend(["PIE_CHART", "DONUT_CHART", "BAR_CHART"])

    if qtype in ("trend",) or any(
        key in keys
        for key in ("date", "month", "attempt", "year", "attemptno", "fromattempt", "toattempt", "time")
    ):
        widgets.extend(["LINE_CHART", "AREA_CHART"])

    return widgets


def _dedupe_widgets(widgets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: Set[Tuple[str, str]] = set()
    deduped: List[Dict[str, Any]] = []
    for widget in widgets:
        key = (widget["section"], widget["type"])
        if key not in seen:
            seen.add(key)
            deduped.append(widget)
    return deduped


def _make_widget(section: str, widget_type: str, priority: int) -> Dict[str, Any]:
    return {
        "section": section,
        "type": widget_type,
        "widgetType": widget_type,
        "priority": priority,
    }


def generate_widgets(answer: Dict[str, Any], query_type: str, intent: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Infer widget metadata while preserving section names and backward compatibility.
    """
    widgets: List[Dict[str, Any]] = []
    category = (intent.get("category") or "").strip()
    operation = (intent.get("operation") or intent.get("subcategory") or "").strip()
    operation = _OPERATION_ALIASES.get(operation, operation)

    business_widget = _infer_business_widget(category, operation, query_type)

    if query_type == "compare":
        left = answer.get("left") or {}
        right = answer.get("right") or {}
        comparison = answer.get("comparison") or {}
        left_label = left.get("label") or "Left"
        right_label = right.get("label") or "Right"
        widgets.extend(
            [
                _make_widget(left_label, "TABLE", _SHAPE_WIDGET_PRIORITIES["TABLE"]),
                _make_widget(right_label, "TABLE", _SHAPE_WIDGET_PRIORITIES["TABLE"]),
                _make_widget("Comparison", "BAR_CHART", _SHAPE_WIDGET_PRIORITIES["BAR_CHART"]),
            ]
        )
        if business_widget and business_widget not in {"TABLE", "BAR_CHART"}:
            widgets.append(_make_widget("Comparison", business_widget, _SHAPE_WIDGET_PRIORITIES.get(business_widget, 50)))
        for widget_type in _infer_shape_widget("Comparison", comparison, query_type):
            widgets.append(_make_widget("Comparison", widget_type, _SHAPE_WIDGET_PRIORITIES.get(widget_type, 50)))
    else:
        sections = answer.get("sections") or []
        if not sections:
            sections = [{"label": "Result", "data": answer}]

        for section in sections:
            label = section.get("label") or "Result"
            section_data = section.get("data") or section
            section_widgets: List[str] = []
            if business_widget:
                section_widgets.append(business_widget)
            section_widgets.extend(_infer_shape_widget(label, section_data, query_type))

            if not section_widgets:
                section_widgets.append("TABLE")

            for widget_type in section_widgets:
                widgets.append(_make_widget(label, widget_type, _SHAPE_WIDGET_PRIORITIES.get(widget_type, 50)))

    deduped = _dedupe_widgets(widgets)
    deduped.sort(key=lambda item: item.get("priority", 0), reverse=True)

    return [
        {
            "section": widget["section"],
            "type": widget["type"],
            "widgetType": widget["widgetType"],
        }
        for widget in deduped
    ]
