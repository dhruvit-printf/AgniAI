"""
visualization_intent.py
=======================
Selects the most appropriate presentation layer from the query semantics,
intent, and returned data shape.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional


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
    (r"\bdonut chart\b", "chart", "donut"),
    (r"\bradial chart\b", "chart", "radial"),
    (r"\barea chart\b", "chart", "area"),
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


def build_visualization_intent(
    question: str,
    intent: Dict[str, Any],
    combined_result: Any = None,
) -> Dict[str, Any]:
    text = (question or "").strip().lower()
    operation = (
        intent.get("operation")
        or intent.get("subcategory")
        or intent.get("query_type")
        or ""
    ).lower()
    query_type = (intent.get("query_type") or "").lower()
    records = _flatten_records(combined_result)
    numeric_data = _has_numeric_signal(records)

    presentation = "table"
    chart_type: Optional[str] = None
    comparison = False
    trend = False
    group_by = intent.get("group_by") or intent.get("groupBy")
    metric = intent.get("metric") or ("average_score" if numeric_data else "count")

    explicit_presentation = _detect_explicit_presentation(text)
    if explicit_presentation:
        presentation = explicit_presentation["presentation"]
        chart_type = explicit_presentation.get("chart_type")
        if chart_type == "bar":
            comparison = "compare" in text or " vs " in text or "versus" in text
        if chart_type == "line":
            trend = "trend" in text or "timeline" in text or "growth" in text
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
        }
        return result

    if query_type in {"compare", "comparison"} or "compare" in text or " vs " in text or "versus" in text:
        presentation = None
        chart_type = None
        comparison = True
    elif query_type == "trend" or any(token in text for token in ("trend", "timeline", "over months", "over time", "growth")):
        presentation = "chart"
        chart_type = "line"
        trend = True
    elif query_type == "distribution" or any(token in text for token in ("distribution", "breakdown", "percentage", "share")):
        presentation = "chart"
        chart_type = "pie"
    elif any(token in text for token in ("show all", "list all", "all candidates", "all records", "show all candidates")):
        presentation = "table"

    if comparison:
        presentation = None
        chart_type = None

    if not chart_type and presentation == "chart":
        chart_type = "bar" if comparison else ("line" if trend else "pie")

    return {
        "presentation": presentation,
        "chart_type": chart_type,
        "comparison": comparison,
        "trend": trend,
        "group_by": group_by,
        "metric": metric,
        "record_count": len(records),
        "numeric_data": numeric_data,
    }
