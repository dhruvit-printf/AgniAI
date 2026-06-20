"""
visualization_engine.py
=======================
Analyzes combined result, query plan, and reporting analysis to auto-generate
appropriate visualization widgets for the frontend.
"""

from __future__ import annotations

from typing import Any, Dict, List, Set


def collect_all_keys(data: Any) -> Set[str]:
    """Recursively collect all dictionary keys in the response structure."""
    keys: Set[str] = set()
    if isinstance(data, dict):
        for k, v in data.items():
            keys.add(k.lower())
            keys.update(collect_all_keys(v))
    elif isinstance(data, list):
        for item in data:
            keys.update(collect_all_keys(item))
    return keys


def generate_widgets(
    combined_result: Any,
    query_plan: Any = None,
    analysis: Any = None,
) -> List[Dict[str, str]]:
    """
    Generates visualization widget metadata based on fields in data & analysis.

    Supported widgets: TABLE, CARD, PIE_CHART, BAR_CHART, LINE_CHART, AREA_CHART, RADIAL_CHART.
    """
    keys = collect_all_keys(combined_result)
    if isinstance(analysis, dict):
        keys.update(collect_all_keys(analysis))

    widgets: List[Dict[str, Any]] = []

    # 1. Identifiers -> TABLE (Priority = 100)
    identifier_targets = {"name", "fullname", "agniveerno", "registrationno"}
    if any(tk in keys for tk in identifier_targets):
        widgets.append({"type": "TABLE", "priority": 100})

    # 2. Metrics -> CARD (Priority = 80)
    metric_targets = {
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
    }
    if any(tk in keys for tk in metric_targets):
        widgets.append({"type": "CARD", "priority": 80})

    # 3. Categories -> PIE_CHART / BAR_CHART (Priority = 60)
    category_targets = {
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
    }
    if any(tk in keys for tk in category_targets):
        widgets.append({"type": "PIE_CHART", "priority": 60})
        widgets.append({"type": "BAR_CHART", "priority": 60})

    # 4. Time Columns -> LINE_CHART / AREA_CHART (Priority = 40)
    time_targets = {
        "date",
        "month",
        "attempt",
        "year",
        "attemptno",
        "fromattempt",
        "toattempt",
    }
    if any(tk in keys for tk in time_targets):
        widgets.append({"type": "LINE_CHART", "priority": 40})
        widgets.append({"type": "AREA_CHART", "priority": 40})

    # 5. Percentages -> RADIAL_CHART (Priority = 20)
    percentage_targets = {
        "passpercentage",
        "failpercentage",
        "completionrate",
        "passrate",
        "failrate",
    }
    if any(tk in keys for tk in percentage_targets):
        widgets.append({"type": "RADIAL_CHART", "priority": 20})

    # 6. Query-type and plan aware widgets (additive overrides/guarantees)
    qtype = None
    if query_plan is not None:
        if isinstance(query_plan, str):
            qtype = query_plan
        elif hasattr(query_plan, "query_type") and query_plan.query_type is not None:
            qtype = getattr(query_plan.query_type, "value", query_plan.query_type)
        elif isinstance(query_plan, dict):
            qtype = query_plan.get("queryType") or query_plan.get("query_type")

    if isinstance(qtype, str):
        qtype = qtype.lower()

    def count_records(data):
        if isinstance(data, list):
            return len(data)
        if isinstance(data, dict):
            for key in ("records", "Records", "data", "Data", "result", "Result"):
                val = data.get(key)
                if isinstance(val, list):
                    return len(val)
                if isinstance(val, dict):
                    return count_records(val)
            if "sides" in data and isinstance(data["sides"], list):
                total = 0
                for side in data["sides"]:
                    total += count_records(side)
                return total
            if "sections" in data and isinstance(data["sections"], list):
                total = 0
                for sec in data["sections"]:
                    total += count_records(sec)
                return total
        return 0

    rec_count = count_records(combined_result)

    if qtype in ("comparison", "compare"):
        widgets.append({"type": "TABLE", "priority": 110})
        widgets.append({"type": "BAR_CHART", "priority": 105})
    elif qtype == "trend":
        widgets.append({"type": "LINE_CHART", "priority": 110})
        widgets.append({"type": "AREA_CHART", "priority": 105})
    elif qtype == "distribution":
        widgets.append({"type": "PIE_CHART", "priority": 110})
        widgets.append({"type": "BAR_CHART", "priority": 105})
    elif qtype == "cross_filter":
        widgets.append({"type": "TABLE", "priority": 110})
    elif qtype in ("multi_independent", "multi_operation"):
        widgets.append({"type": "TABLE", "priority": 110})
        if isinstance(combined_result, dict) and "sections" in combined_result:
            for sec in combined_result["sections"]:
                sec_rec_count = count_records(sec)
                if sec_rec_count == 1:
                    widgets.append({"type": "CARD", "priority": 102})
                elif sec_rec_count > 1:
                    widgets.append({"type": "TABLE", "priority": 102})
    has_group_by = False
    if query_plan is not None:
        if hasattr(query_plan, "operations") and query_plan.operations:
            for op in query_plan.operations:
                if getattr(op, "group_by", None) or getattr(op, "groupBy", None):
                    has_group_by = True
        elif isinstance(query_plan, dict) and "operations" in query_plan:
            for op in query_plan["operations"]:
                if isinstance(op, dict) and (op.get("group_by") or op.get("groupBy")):
                    has_group_by = True

    if qtype == "analytics" or (qtype == "simple" and has_group_by):
        widgets.append({"type": "BAR_CHART", "priority": 105})
        widgets.append({"type": "PIE_CHART", "priority": 105})

    else:
        if rec_count == 1:
            widgets.append({"type": "CARD", "priority": 110})
        elif rec_count > 1:
            widgets.append({"type": "TABLE", "priority": 110})

    # Extract per-section or operation widget hints if any
    hints = []
    if hasattr(query_plan, "operations") and query_plan.operations:
        for op in query_plan.operations:
            if hasattr(op, "intent_result") and isinstance(op.intent_result, dict):
                h = op.intent_result.get("widget_hint") or op.intent_result.get(
                    "widgetHint"
                )
                if h:
                    hints.append(h)
            if hasattr(op, "dotnet_payload") and isinstance(op.dotnet_payload, dict):
                h = op.dotnet_payload.get("widget_hint") or op.dotnet_payload.get(
                    "widgetHint"
                )
                if h:
                    hints.append(h)
    elif isinstance(query_plan, dict) and "operations" in query_plan:
        for op in query_plan["operations"]:
            if isinstance(op, dict):
                h = op.get("widget_hint") or op.get("widgetHint")
                if h:
                    hints.append(h)
                for inner_key in (
                    "intent_result",
                    "intentResult",
                    "dotnet_payload",
                    "dotnetPayload",
                ):
                    inner = op.get(inner_key)
                    if isinstance(inner, dict):
                        h_inner = inner.get("widget_hint") or inner.get("widgetHint")
                        if h_inner:
                            hints.append(h_inner)

    if isinstance(combined_result, dict) and "sections" in combined_result:
        for sec in combined_result["sections"]:
            if isinstance(sec, dict):
                h = sec.get("widget_hint") or sec.get("widgetHint")
                if h:
                    hints.append(h)
                sec_data = sec.get("data")
                if isinstance(sec_data, dict):
                    h_data = sec_data.get("widget_hint") or sec_data.get("widgetHint")
                    if h_data:
                        hints.append(h_data)

    _WIDGET_PRIORITIES = {
        "TABLE": 100,
        "CARD": 80,
        "PIE_CHART": 60,
        "BAR_CHART": 60,
        "LINE_CHART": 40,
        "AREA_CHART": 40,
        "RADIAL_CHART": 20,
    }
    for hint in hints:
        h_upper = str(hint).upper()
        if h_upper in _WIDGET_PRIORITIES:
            widgets.append({"type": h_upper, "priority": _WIDGET_PRIORITIES[h_upper]})

    # Sort widgets descending by priority
    widgets.sort(key=lambda w: w["priority"], reverse=True)

    # Deduplicate while preserving priority order
    seen_types = set()
    deduped_widgets = []
    for w in widgets:
        w_type = w["type"]
        if w_type not in seen_types:
            seen_types.add(w_type)
            deduped_widgets.append(w)

    # Return the widget fields for backward compatibility
    return [{"type": w["type"], "widgetType": w["type"], "section": "Result"} for w in deduped_widgets]

