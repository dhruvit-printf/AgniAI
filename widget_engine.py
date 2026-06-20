"""
widget_engine.py
================
Widget engine for generating visualization metadata based on answer JSON.
"""

from typing import Any, Dict, List

def generate_widgets(
    answer: Dict[str, Any],
    query_type: str,
    intent: Dict[str, Any]
) -> List[Dict[str, str]]:
    """
    Returns visualizer metadata.
    Format:
    [
      {
        "section": "",
        "widgetType": ""
      }
    ]
    """
    widgets: List[Dict[str, str]] = []
    sections = answer.get("sections") or []

    # Helper to count records in data
    def count_records(data: Any) -> int:
        if isinstance(data, list):
            return len(data)
        if isinstance(data, dict):
            for key in ("records", "Records", "data", "Data"):
                val = data.get(key)
                if isinstance(val, list):
                    return len(val)
        return 0

    if query_type == "compare":
        left = answer.get("left") or {}
        right = answer.get("right") or {}
        left_label = left.get("label") or "Left"
        right_label = right.get("label") or "Right"

        widgets.append({"section": left_label, "widgetType": "TABLE"})
        widgets.append({"section": right_label, "widgetType": "TABLE"})
        widgets.append({"section": "Comparison", "widgetType": "BAR_CHART"})
        widgets.append({"section": "Comparison", "widgetType": "TABLE"})

    elif query_type == "trend":
        section_label = sections[0].get("label") if sections else "Trend Analysis"
        widgets.append({"section": section_label, "widgetType": "LINE_CHART"})
        widgets.append({"section": section_label, "widgetType": "AREA_CHART"})

    elif query_type == "distribution":
        section_label = sections[0].get("label") if sections else "Distribution Breakdown"
        widgets.append({"section": section_label, "widgetType": "PIE_CHART"})
        widgets.append({"section": section_label, "widgetType": "BAR_CHART"})

    elif query_type == "multi_independent":
        for sec in sections:
            label = sec.get("label") or "Section"
            rec_count = len(sec.get("data") or [])
            if rec_count == 1:
                widgets.append({"section": label, "widgetType": "CARD"})
            else:
                widgets.append({"section": label, "widgetType": "TABLE"})

    else:
        # simple / cross_filter
        section_label = sections[0].get("label") if sections else "Result"
        records = sections[0].get("data") if sections else []
        rec_count = len(records)

        if query_type == "cross_filter":
            widgets.append({"section": section_label, "widgetType": "TABLE"})
        else:
            if rec_count == 1:
                widgets.append({"section": section_label, "widgetType": "CARD"})
            else:
                widgets.append({"section": section_label, "widgetType": "TABLE"})

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for w in widgets:
        key = (w["section"], w["widgetType"])
        if key not in seen:
            seen.add(key)
            w_copy = dict(w)
            w_copy["type"] = w["widgetType"]
            deduped.append(w_copy)

    return deduped

