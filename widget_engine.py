"""
widget_engine.py
================
Widget engine for generating visualization metadata based on answer JSON.
"""

from typing import Any, Dict, List, Set

from normalized_models import extract_records as _extract_records

def generate_widgets(
    answer: Dict[str, Any],
    query_type: str,
    intent: Dict[str, Any]
) -> List[Dict[str, str]]:
    """
    Dynamically infers and generates widget metadata based on answer JSON structure and contents.
    Never uses hardcoded widget arrays.
    """
    widgets: List[Dict[str, str]] = []

    # Helper to recursively collect all lowercase keys from any data block
    def collect_keys(data: Any) -> Set[str]:
        keys = set()
        if isinstance(data, dict):
            for k, v in data.items():
                keys.add(k.lower())
                keys.update(collect_keys(v))
        elif isinstance(data, list):
            for item in data:
                keys.update(collect_keys(item))
        return keys

    # Helper to count records inside section/data blocks
    def count_records(data: Any) -> int:
        return len(_extract_records(data))

    def infer_for_block(label: str, data: Any):
        keys = collect_keys(data)
        rec_count = count_records(data)

        # 1. Tabular records -> TABLE
        if rec_count > 1:
            widgets.append({"section": label, "widgetType": "TABLE"})
        # 2. Single object -> CARD
        elif rec_count == 1:
            widgets.append({"section": label, "widgetType": "CARD"})

        # 3. Large KPIs -> METRIC_CARD
        metric_targets = {
            "count", "average", "max", "min", "score", "besttotal", "totalmarks",
            "marksobtained", "averagescore", "topscore", "bottomscore", "value", "values"
        }
        if any(tk in keys for tk in metric_targets) or (isinstance(data, dict) and any(k.lower() in metric_targets for k in data)):
            widgets.append({"section": label, "widgetType": "METRIC_CARD"})

        # 4. Comparisons -> BAR_CHART
        if query_type == "compare" and label == "Comparison":
            widgets.append({"section": label, "widgetType": "BAR_CHART"})
        elif any(k in keys for k in ("comparison", "difference", "variance")):
            widgets.append({"section": label, "widgetType": "BAR_CHART"})

        # 5. Category distribution -> PIE_CHART (usually accompanied by BAR_CHART)
        category_targets = {
            "leavetype", "grade", "sport", "sports", "bloodgroup", "bloodtype",
            "disease", "unitname", "teamname", "sectionname", "classname", "category"
        }
        if query_type == "distribution" or any(tk in keys for tk in category_targets):
            widgets.append({"section": label, "widgetType": "PIE_CHART"})
            widgets.append({"section": label, "widgetType": "BAR_CHART"})

        # 6. Trend data -> LINE_CHART
        time_targets = {
            "date", "month", "attempt", "year", "attemptno", "fromattempt", "toattempt", "time"
        }
        if query_type == "trend" or any(tk in keys for tk in time_targets):
            widgets.append({"section": label, "widgetType": "LINE_CHART"})

        # 7. Continuous values -> AREA_CHART
        continuous_targets = {"score", "bmi", "value", "values", "average", "averagescore"}
        if query_type == "trend" or any(tk in keys for tk in continuous_targets):
            widgets.append({"section": label, "widgetType": "AREA_CHART"})

        # 8. Percentage values -> RADIAL_CHART
        percentage_targets = {
            "passpercentage", "failpercentage", "completionrate", "passrate", "failrate", "percentage", "pct", "rate"
        }
        if any(tk in keys for tk in percentage_targets):
            widgets.append({"section": label, "widgetType": "RADIAL_CHART"})

    # Check structure and run inference
    if query_type == "compare":
        left = answer.get("left") or {}
        right = answer.get("right") or {}
        left_label = left.get("label") or "Left"
        right_label = right.get("label") or "Right"

        infer_for_block(left_label, left)
        infer_for_block(right_label, right)
        infer_for_block("Comparison", answer.get("comparison") or {})
        # Keep a guaranteed comparison surface for the frontend.
        widgets.append({"section": "Comparison", "widgetType": "TABLE"})
        widgets.append({"section": "Comparison", "widgetType": "BAR_CHART"})
    else:
        sections = answer.get("sections") or []
        if not sections:
            # Fallback simple inference on the whole answer
            infer_for_block("Result", answer)
        else:
            for sec in sections:
                label = sec.get("label") or "Result"
                infer_for_block(label, sec)

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for w in widgets:
        key = (w["section"], w["widgetType"])
        if key not in seen:
            seen.add(key)
            w_copy = dict(w)
            w_copy["type"] = w["widgetType"]  # For backward compatibility
            deduped.append(w_copy)

    return deduped
