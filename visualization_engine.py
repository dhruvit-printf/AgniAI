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
    metric_targets = {"count", "average", "max", "min", "score", "besttotal", "totalmarks", "marksobtained", "averagescore", "topscore", "bottomscore"}
    if any(tk in keys for tk in metric_targets):
        widgets.append({"type": "CARD", "priority": 80})

    # 3. Categories -> PIE_CHART / BAR_CHART (Priority = 60)
    category_targets = {"leavetype", "grade", "sport", "sports", "bloodgroup", "bloodtype", "disease", "unitname", "teamname", "sectionname", "classname"}
    if any(tk in keys for tk in category_targets):
        widgets.append({"type": "PIE_CHART", "priority": 60})
        widgets.append({"type": "BAR_CHART", "priority": 60})

    # 4. Time Columns -> LINE_CHART / AREA_CHART (Priority = 40)
    time_targets = {"date", "month", "attempt", "year", "attemptno", "fromattempt", "toattempt"}
    if any(tk in keys for tk in time_targets):
        widgets.append({"type": "LINE_CHART", "priority": 40})
        widgets.append({"type": "AREA_CHART", "priority": 40})

    # 5. Percentages -> RADIAL_CHART (Priority = 20)
    percentage_targets = {"passpercentage", "failpercentage", "completionrate", "passrate", "failrate"}
    if any(tk in keys for tk in percentage_targets):
        widgets.append({"type": "RADIAL_CHART", "priority": 20})

    # Sort widgets descending by priority
    widgets.sort(key=lambda w: w["priority"], reverse=True)

    # Return only the type field for each widget
    return [{"type": w["type"]} for w in widgets]
