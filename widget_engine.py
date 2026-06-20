"""
widget_engine.py
================
Widget / Schema Inference Engine for the AgniAI admin pipeline.

Converts CombinedResult into a single deterministic FormattedData structure.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import ValidationError

from normalized_models import extract_records as _orig_extract_records

def _extract_records(combined_result: Any) -> List[Dict[str, Any]]:
    if isinstance(combined_result, dict):
        if "sections" in combined_result and isinstance(combined_result["sections"], list):
            records = []
            for sec in combined_result["sections"]:
                if isinstance(sec, dict) and "data" in sec:
                    data_val = sec["data"]
                    if isinstance(data_val, list):
                        records.extend([r for r in data_val if isinstance(r, dict)])
            if records:
                return records
        if "records" in combined_result and isinstance(combined_result["records"], list):
            return [r for r in combined_result["records"] if isinstance(r, dict)]
    elif isinstance(combined_result, list):
        return [r for r in combined_result if isinstance(r, dict)]
    return _orig_extract_records(combined_result)

from schemas import (
    CardItem,
    CardData,
    TableColumn,
    TableData,
    BarChartData,
    SeriesItem,
    LineChartData,
    PieChartItem,
    PieChartData,
    FormattedData,
)

WIDGET_MAP: Dict[Tuple[str, str], str] = {
    ("Performance", "Top"): "TABLE",
    ("Performance", "Compare"): "CHART_BAR",
    ("Attendance", "Monthly"): "CHART_BAR",
    ("Attendance", "Trend"): "CHART_LINE",
    ("Medical", "BMI"): "CHART_PIE",
    ("Attendance", "Present"): "CHART_PIE",
    ("Strength", "Overall"): "CHART_PIE",
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

def _map_to_supported_type(inferred: str) -> str:
    mapped = {
        "TABLE": "TABLE",
        "CARD": "CARD",
        "METRIC_CARD": "CARD",
        "BAR_CHART": "CHART_BAR",
        "LINE_CHART": "CHART_LINE",
        "AREA_CHART": "CHART_LINE",
        "PIE_CHART": "CHART_PIE",
        "DONUT_CHART": "CHART_PIE",
        "RADIAL_CHART": "CHART_PIE",
        "CALENDAR_UI": "TABLE",
    }
    return mapped.get(inferred, "TABLE")

def infer_supported_type(combined_result: Any, query_type: str, intent: Dict[str, Any]) -> str:
    qtype = (query_type or "").strip().lower()
    
    if qtype == "compare" or qtype == "comparison":
        return "CHART_BAR"
    elif qtype == "trend":
        return "CHART_LINE"
    elif qtype == "distribution":
        return "CHART_PIE"
    elif qtype == "cross_filter":
        return "TABLE"
        
    category = (intent.get("category") or "").strip()
    operation = (intent.get("operation") or intent.get("subcategory") or "").strip()
    operation = _OPERATION_ALIASES.get(operation, operation)
    
    business_widget = None
    if category and operation:
        business_widget = WIDGET_MAP.get((category, operation))
        
    if business_widget:
        return _map_to_supported_type(business_widget)
        
    rec_count = len(_extract_records(combined_result))
    if rec_count == 1:
        return "CARD"
    
    return "TABLE"

def build_card_data(records: List[Dict[str, Any]], title: str) -> Dict[str, Any]:
    cards = []
    for r in records:
        title_val = r.get("fullName") or r.get("name") or (f"Record ID: {r.get('id')}" if "id" in r else "Details")
        subtitle_val = r.get("category") or r.get("leaveType") or r.get("sport") or r.get("class") or ""
        value_val = r.get("bestTotal") or r.get("score") or r.get("marksObtained") or r.get("count") or r.get("status") or r.get("leaveStatus") or ""
        desc_val = r.get("description") or r.get("message") or ""
        cards.append({
            "title": str(title_val),
            "subtitle": str(subtitle_val),
            "value": str(value_val),
            "description": str(desc_val)
        })
    if not cards:
        cards.append({
            "title": title,
            "subtitle": "",
            "value": "",
            "description": "No records found."
        })
    return {"cards": cards}

def build_table_data(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {"columns": [], "rows": []}
    
    keys_seen = []
    for r in records:
        for k in r.keys():
            if k not in keys_seen and k != "id":
                keys_seen.append(k)
                
    key_priority = {"fullname": -10, "agniveerno": -9, "name": -8, "agniveerid": -7, "score": -6, "besttotal": -5}
    keys_seen.sort(key=lambda k: key_priority.get(k.lower(), 0))
    
    columns = []
    for k in keys_seen:
        import re
        label = re.sub(r"([A-Z])", r" \1", k).strip().title()
        label = label.replace("Agniveer No", "Agniveer No.").replace("Id", "ID")
        columns.append({
            "key": k,
            "label": label
        })
        
    rows = []
    for r in records:
        row = {}
        for k in keys_seen:
            row[k] = r.get(k)
        rows.append(row)
        
    return {"columns": columns, "rows": rows}

def build_bar_chart_data(combined_result: Any) -> Dict[str, Any]:
    if isinstance(combined_result, dict) and "sides" in combined_result:
        sides = combined_result["sides"]
        y_key = "recordCount"
        if sides and "metrics" in sides[0]:
            metric_keys = list(sides[0]["metrics"].keys())
            if metric_keys:
                y_key = metric_keys[0]
        
        rows = []
        for s in sides:
            row_item = {
                "label": s.get("label", "Section"),
                "recordCount": s.get("recordCount", 0)
            }
            if "metrics" in s:
                row_item.update(s["metrics"])
            rows.append(row_item)
            
        return {
            "xKey": "label",
            "yKey": y_key,
            "rows": rows
        }
        
    records = _extract_records(combined_result)
    if not records:
        return {"xKey": "label", "yKey": "value", "rows": []}
        
    y_key = None
    numeric_candidates = ["count", "score", "value", "percentage", "besttotal", "marksobtained", "rate", "average"]
    for k in numeric_candidates:
        for r in records[:1]:
            for rk in r.keys():
                if rk.lower() == k:
                    y_key = rk
                    break
            if y_key:
                break
        if y_key:
            break
            
    if not y_key:
        for r in records[:1]:
            for rk, rv in r.items():
                if isinstance(rv, (int, float)) and rk != "id":
                    y_key = rk
                    break
    if not y_key:
        y_key = "value"
        
    x_key = None
    x_candidates = ["month", "date", "year", "sport", "grade", "leavetype", "fullname", "name", "label"]
    for k in x_candidates:
        for r in records[:1]:
            for rk in r.keys():
                if rk.lower() == k:
                    x_key = rk
                    break
            if x_key:
                break
        if x_key:
            break
            
    if not x_key:
        for r in records[:1]:
            for rk, rv in r.items():
                if rk != y_key and rk != "id" and isinstance(rv, str):
                    x_key = rk
                    break
    if not x_key:
        x_key = "label"
        
    rows = []
    for r in records:
        row_item = dict(r)
        if x_key not in row_item:
            row_item[x_key] = r.get("fullName") or r.get("name") or "Record"
        if y_key not in row_item:
            row_item[y_key] = 1
        rows.append(row_item)
        
    return {
        "xKey": x_key,
        "yKey": y_key,
        "rows": rows
    }

def build_line_chart_data(combined_result: Any) -> Dict[str, Any]:
    records = _extract_records(combined_result)
    if not records:
        return {"xKey": "month", "series": [], "rows": []}
        
    x_key = None
    time_candidates = ["date", "month", "year", "attemptno", "attempt", "time", "day"]
    for k in time_candidates:
        for r in records[:1]:
            for rk in r.keys():
                if rk.lower() == k:
                    x_key = rk
                    break
            if x_key:
                break
        if x_key:
            break
            
    if not x_key:
        x_key = "month"
        
    series_keys = []
    for r in records[:1]:
        for rk, rv in r.items():
            if isinstance(rv, (int, float)) and rk != "id" and rk.lower() not in time_candidates:
                series_keys.append(rk)
                
    if not series_keys:
        series_keys = ["value"]
        
    series = []
    for sk in series_keys:
        import re
        series_label = re.sub(r"([A-Z])", r" \1", sk).strip().title()
        series.append({
            "key": sk,
            "label": series_label
        })
        
    rows = []
    for r in records:
        row_item = dict(r)
        if x_key not in row_item:
            row_item[x_key] = "N/A"
        for sk in series_keys:
            if sk not in row_item:
                row_item[sk] = 0
        rows.append(row_item)
        
    return {
        "xKey": x_key,
        "series": series,
        "rows": rows
    }

def build_pie_chart_data(combined_result: Any) -> Dict[str, Any]:
    records = _extract_records(combined_result)
    if not records:
        return {"rows": []}
        
    label_key = None
    label_candidates = ["label", "sport", "grade", "leavetype", "bloodgroup", "disease", "status", "category"]
    for k in label_candidates:
        for r in records[:1]:
            for rk in r.keys():
                if rk.lower() == k:
                    label_key = rk
                    break
            if label_key:
                break
        if label_key:
            break
            
    if not label_key:
        for r in records[:1]:
            for rk, rv in r.items():
                if isinstance(rv, str) and rk != "id":
                    label_key = rk
                    break
    if not label_key:
        label_key = "label"
        
    value_key = None
    value_candidates = ["value", "count", "score", "percentage", "besttotal", "marksobtained", "rate"]
    for k in value_candidates:
        for r in records[:1]:
            for rk in r.keys():
                if rk.lower() == k:
                    value_key = rk
                    break
            if value_key:
                break
        if value_key:
            break
            
    if not value_key:
        for r in records[:1]:
            for rk, rv in r.items():
                if isinstance(rv, (int, float)) and rk != "id":
                    value_key = rk
                    break
    if not value_key:
        value_key = "value"
        
    rows = []
    for r in records:
        label_val = r.get(label_key) or r.get("fullName") or r.get("name") or "Category"
        value_val = r.get(value_key) or 1
        rows.append({
            "label": str(label_val),
            "value": value_val
        })
        
    return {"rows": rows}

def validate_payload(inferred_type: str, data: Dict[str, Any]) -> None:
    if inferred_type == "TABLE":
        cols = {c["key"] for c in data.get("columns", [])}
        for row in data.get("rows", []):
            for col in cols:
                if col not in row:
                    row[col] = None
    elif inferred_type == "CHART_BAR":
        x = data.get("xKey")
        y = data.get("yKey")
        rows = data.get("rows", [])
        if rows:
            if x not in rows[0]:
                rows[0][x] = "Category"
            if y not in rows[0]:
                rows[0][y] = 0
    elif inferred_type == "CHART_LINE":
        x = data.get("xKey")
        series = data.get("series", [])
        rows = data.get("rows", [])
        for row in rows:
            if x not in row:
                row[x] = "N/A"
            for s in series:
                if s["key"] not in row:
                    row[s["key"]] = 0
    elif inferred_type == "CHART_PIE":
        rows = data.get("rows", [])
        for row in rows:
            if "label" not in row:
                row["label"] = "Category"
            if "value" not in row:
                row["value"] = 0

def build_formatted_data(
    combined_result: Any,
    query_type: str,
    intent: Dict[str, Any],
    analysis: Optional[Any] = None,
    prediction: Optional[Any] = None,
    conclusion: Optional[Any] = None,
) -> Dict[str, Any]:
    
    inferred_type = infer_supported_type(combined_result, query_type, intent)
    
    from normalized_models import _derive_title
    title = _derive_title(query_type, intent)
    
    records = _extract_records(combined_result)
    
    if inferred_type == "CARD":
        data_payload = build_card_data(records, title)
    elif inferred_type == "CHART_BAR":
        data_payload = build_bar_chart_data(combined_result)
    elif inferred_type == "CHART_LINE":
        data_payload = build_line_chart_data(combined_result)
    elif inferred_type == "CHART_PIE":
        data_payload = build_pie_chart_data(combined_result)
    else:
        data_payload = build_table_data(records)
        
    validate_payload(inferred_type, data_payload)
    
    from normalized_models import (
        combine_analysis_to_string,
        combine_prediction_to_string,
        combine_conclusion_to_string,
    )
    analysis_str = combine_analysis_to_string(analysis) if analysis else ""
    prediction_str = combine_prediction_to_string(prediction) if prediction else ""
    conclusion_str = combine_conclusion_to_string(conclusion) if conclusion else ""
    
    fd = FormattedData(
        type=inferred_type,
        title=title,
        data=data_payload,
        analysis=analysis_str,
        prediction=prediction_str,
        conclusion=conclusion_str,
    )
    return fd.model_dump()
