"""
widget_engine.py
================
Widget / Schema Inference Engine for the AgniAI admin pipeline.

Converts CombinedResult into a single deterministic FormattedData structure.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import ValidationError

from normalized_models import extract_records as _orig_extract_records

def flatten_single_record(r: Dict[str, Any]) -> List[Dict[str, Any]]:
    base = {}
    attempts_key = None
    sections_key = None
    subitems_key = None
    
    for k in r.keys():
        k_lower = k.lower()
        if k_lower == "attempts":
            attempts_key = k
        elif k_lower == "sections":
            sections_key = k
        elif k_lower in ("subitems", "subitems"):
            subitems_key = k
            
    for k, v in r.items():
        if k in (attempts_key, sections_key, subitems_key):
            continue
        base[k] = v
        
    if attempts_key and isinstance(r[attempts_key], list) and r[attempts_key]:
        res = []
        for att in r[attempts_key]:
            if not isinstance(att, dict):
                continue
            att_merged = dict(base)
            
            att_sections_key = None
            att_subitems_key = None
            for k in att.keys():
                if k.lower() == "sections":
                    att_sections_key = k
                elif k.lower() in ("subitems", "subitems"):
                    att_subitems_key = k
            
            for k, v in att.items():
                if k in (att_sections_key, att_subitems_key):
                    continue
                att_merged[k] = v
                
            if att_sections_key and isinstance(att[att_sections_key], list) and att[att_sections_key]:
                for sec in att[att_sections_key]:
                    if not isinstance(sec, dict):
                        continue
                    sec_merged = dict(att_merged)
                    
                    sec_subitems_key = None
                    for k in sec.keys():
                        if k.lower() in ("subitems", "subitems"):
                            sec_subitems_key = k
                            
                    for k, v in sec.items():
                        if k == sec_subitems_key:
                            continue
                        sec_merged[k] = v
                        
                    if sec_subitems_key and isinstance(sec[sec_subitems_key], list) and sec[sec_subitems_key]:
                        for sub in sec[sec_subitems_key]:
                            if not isinstance(sub, dict):
                                continue
                            sub_merged = dict(sec_merged)
                            for k, v in sub.items():
                                sub_merged[k] = v
                            res.append(sub_merged)
                    else:
                        res.append(sec_merged)
            else:
                res.append(att_merged)
        return res
        
    elif sections_key and isinstance(r[sections_key], list) and r[sections_key]:
        res = []
        for sec in r[sections_key]:
            if not isinstance(sec, dict):
                continue
            sec_merged = dict(base)
            
            sec_subitems_key = None
            for k in sec.keys():
                if k.lower() in ("subitems", "subitems"):
                    sec_subitems_key = k
                    
            for k, v in sec.items():
                if k == sec_subitems_key:
                    continue
                sec_merged[k] = v
                
            if sec_subitems_key and isinstance(sec[sec_subitems_key], list) and sec[sec_subitems_key]:
                for sub in sec[sec_subitems_key]:
                    if not isinstance(sub, dict):
                        continue
                    sub_merged = dict(sec_merged)
                    for k, v in sub.items():
                        sub_merged[k] = v
                    res.append(sub_merged)
            else:
                res.append(sec_merged)
        return res
        
    return [r]

def flatten_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    flat_records = []
    for r in records:
        if isinstance(r, dict):
            flat_records.extend(flatten_single_record(r))
        else:
            flat_records.append(r)
    seen = set()
    deduped = []
    for record in flat_records:
        if not isinstance(record, dict):
            continue
        record_id = None
        for key in ("agniveerNo", "agniveerId", "AgniveerId", "AgniVeerId", "id", "Id"):
            val = record.get(key)
            if val is not None:
                record_id = f"id:{val}"
                break
        if record_id is None:
            record_id = "row:" + json.dumps(record, sort_keys=True, ensure_ascii=False, default=str)
        if record_id in seen:
            continue
        seen.add(record_id)
        deduped.append(record)
    return deduped


def _dedupe_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        record_id = None
        for key in ("agniveerNo", "agniveerId", "AgniveerId", "AgniVeerId", "id", "Id"):
            val = record.get(key)
            if val is not None:
                record_id = f"id:{val}"
                break
        if record_id is None:
            record_id = "row:" + json.dumps(record, sort_keys=True, ensure_ascii=False, default=str)
        if record_id in seen:
            continue
        seen.add(record_id)
        deduped.append(record)
    return deduped

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
                return flatten_records(records)
        if "records" in combined_result and isinstance(combined_result["records"], list):
            return flatten_records([r for r in combined_result["records"] if isinstance(r, dict)])
    elif isinstance(combined_result, list):
        return flatten_records([r for r in combined_result if isinstance(r, dict)])
    return flatten_records(_orig_extract_records(combined_result))

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
    ("Performance", "Bottom"): "TABLE",
    ("Performance", "Improvement"): "CHART_LINE",
    ("Performance", "Drop"): "CHART_LINE",
    ("Performance", "Grading"): "TABLE",
    ("Performance", "GradingSummary"): "CHART_BAR",
    ("Performance", "Average"): "CHART_PIE",
    ("Performance", "AttemptWise"): "TABLE",
    ("Performance", "BestAttempt"): "TABLE",
    ("Performance", "Compare"): "AREA_CHART",
    ("Performance", "Summary"): "TABLE",
    ("Performance", "PassPercentage"): "CHART_PIE",
    ("Performance", "FailPercentage"): "CHART_PIE",
    ("Performance", "Overall"): "TABLE",
    ("Leave", "Most"): "TABLE",
    ("Leave", "Least"): "TABLE",
    ("Leave", "Current"): "TABLE",
    ("Leave", "Absconded"): "TABLE",
    ("Leave", "LeaveType"): "TABLE",
    ("Medical", "Active"): "TABLE",
    ("Medical", "BMI"): "DONUT_CHART",
    ("Medical", "Disease"): "TABLE",
    ("Attendance", "Monthly"): "CHART_BAR",
    ("Attendance", "Weekly"): "CHART_BAR",
    ("Attendance", "Daily"): "TABLE",
    ("Attendance", "Present"): "CHART_PIE",
    ("Strength", "Strength"): "RADIAL_CHART",
    ("Verification", "Pending"): "TABLE",
    ("Verification", "Completed"): "TABLE",
    ("Verification", "NotResponded"): "TABLE",
    ("Verification", "Verified"): "TABLE",
    ("Verification", "Rejected"): "TABLE",
    ("Equipment", "Stats"): "CARD",
    ("Equipment", "Overdue"): "TABLE",
    ("Equipment", "Returend"): "TABLE",
    ("Skills", "BySport"): "TABLE",
    ("Skills", "ByClass"): "TABLE",
    ("Roster", "BySport"): "TABLE",
    ("Roster", "ByClass"): "TABLE",
    ("Distribution", "Latest"): "TABLE",
    ("Distribution", "ByUnit"): "TABLE",
    ("Distribution", "Unassigned"): "TABLE",
    ("Distribution", "TopUnit"): "CARD",
    ("Overall", "Overall"): "TABLE",
    ("schedule", "Date"): "TABLE",
}

_OPERATION_ALIASES: Dict[str, str] = {
    "TopPerformers": "Top",
    "LowestPerformers": "Bottom",
    "Comparison": "Compare",
    "MonthlyAttendance": "Monthly",
    "WeeklyAttendance": "Weekly",
    "YearlyAttendance": "Yearly",
    "PresentToday": "Present",
    "StrengthBreakdown": "Strength",
    "BMIAnalysis": "BMI",
    "EquipmentSummary": "Stats",
    "IssuedItems": "Issued",
    "ProcuredItems": "Procured",
    "CompletedVerification": "Verified",
    "SentVerification": "Sent",
    "HoldingEquipment": "Holding",
    "AgniveerWiseEquipment": "AgniveerWise",
    "IndividualMedical": "Individual",
    "AttendanceSummary": "Summary",
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


def _normalize_requested_widget_type(value: Any) -> Optional[str]:
    text = str(value or "").strip().lower()
    if not text:
        return None

    aliases = {
        "table": "TABLE",
        "tabular": "TABLE",
        "grid": "TABLE",
        "card": "CARD",
        "cards": "CARD",
        "bar": "CHART_BAR",
        "bar chart": "CHART_BAR",
        "chart bar": "CHART_BAR",
        "line": "CHART_LINE",
        "trend": "CHART_LINE",
        "trend chart": "CHART_LINE",
        "line chart": "CHART_LINE",
        "area": "AREA_CHART",
        "area chart": "AREA_CHART",
        "pie": "CHART_PIE",
        "pie chart": "CHART_PIE",
        "donut": "DONUT_CHART",
        "donut chart": "DONUT_CHART",
        "radial": "RADIAL_CHART",
        "radial chart": "RADIAL_CHART",
    }
    if text in aliases:
        return aliases[text]

    normalized = text.replace("_", " ").replace("-", " ")
    return aliases.get(normalized) or aliases.get(normalized.strip())


def _default_widget_type_for_intent(
    category: str,
    operation: str,
    query_type: str,
) -> Optional[str]:
    category_key = (category or "").strip()
    operation_key = _OPERATION_ALIASES.get((operation or "").strip(), (operation or "").strip())
    if category_key and operation_key:
        widget_type = WIDGET_MAP.get((category_key, operation_key))
        if widget_type:
            return widget_type

    qtype = (query_type or "").strip().lower()
    if qtype == "compare" or qtype == "comparison":
        return "AREA_CHART"
    if qtype == "trend":
        return "CHART_LINE"
    if qtype == "distribution":
        return "CHART_PIE"
    if qtype == "cross_filter":
        return "TABLE"
    return None

def infer_supported_type(
    combined_result: Any,
    query_type: str,
    intent: Dict[str, Any],
    visualization_intent: Optional[Dict[str, Any]] = None,
) -> str:
    qtype = (query_type or "").strip().lower()

    if isinstance(visualization_intent, dict):
        requested_widget_type = _normalize_requested_widget_type(
            visualization_intent.get("requested_widget_type")
            or visualization_intent.get("widget_type")
        )
        if requested_widget_type:
            return requested_widget_type
        presentation = (visualization_intent.get("presentation") or "").strip().lower()
        chart_type = (visualization_intent.get("chart_type") or "").strip().lower()
        if presentation == "cards":
            return "CARD"
        if presentation == "table":
            return "TABLE"
        if presentation == "chart":
            if chart_type == "line":
                return "CHART_LINE"
            if chart_type == "pie":
                return "CHART_PIE"
            if chart_type == "area":
                return "AREA_CHART"
            return "CHART_BAR"

    category = (intent.get("category") or "").strip()
    operation = (intent.get("operation") or intent.get("subcategory") or "").strip()
    default_widget = _default_widget_type_for_intent(category, operation, query_type)
    if default_widget:
        return default_widget

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

    def _flatten_cell_value(key: str, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value

        key_lower = (key or "").lower()

        if isinstance(value, list):
            if not value:
                return None
            if all(isinstance(item, (str, int, float, bool)) or item is None for item in value):
                return ", ".join(str(item) for item in value if item is not None)
            if key_lower == "attempts":
                return f"{len(value)} attempt(s)"
            if key_lower == "sections":
                return f"{len(value)} section(s)"
            if key_lower == "subitems":
                return f"{len(value)} sub-item(s)"
            sample_labels = []
            for item in value[:5]:
                if isinstance(item, dict):
                    label = (
                        item.get("label")
                        or item.get("sectionName")
                        or item.get("subItemName")
                        or item.get("name")
                        or item.get("title")
                    )
                    if label:
                        sample_labels.append(str(label))
            if sample_labels:
                return ", ".join(sample_labels)
            return f"{len(value)} item(s)"

        if isinstance(value, dict):
            for candidate in ("label", "sectionName", "subItemName", "name", "title", "fullName"):
                nested = value.get(candidate)
                if nested not in (None, ""):
                    return nested
            scalar_parts = []
            for nested_key, nested_val in value.items():
                if isinstance(nested_val, (str, int, float, bool)) or nested_val is None:
                    if nested_val is not None:
                        scalar_parts.append(f"{nested_key}: {nested_val}")
            if scalar_parts:
                return "; ".join(scalar_parts)
            return f"{len(value)} field(s)"

        return str(value)
    
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
            row[k] = _flatten_cell_value(k, r.get(k))
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
                "recordCount": s.get("recordCount", 0),
            }
            if "metrics" in s:
                row_item.update(s["metrics"])
            rows.append(row_item)
            
        return {
            "xKey": "label",
            "yKey": y_key,
            "rows": [
                {
                    f"{'label'}:value": row.get("label", "Section"),
                    f"{y_key}:value": row.get(y_key, row.get("recordCount", 0)),
                    **{
                        k: v
                        for k, v in row.items()
                        if k not in {"label", "recordCount", y_key}
                    },
                }
                for row in rows
            ],
        }
        
    records = _extract_records(combined_result)
    records = _dedupe_records(records)
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
        "rows": [
            {
                f"{x_key}:value": row.get(x_key),
                f"{y_key}:value": row.get(y_key),
            }
            for row in rows
        ],
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
        "rows": [
            {
                f"{x_key}:value": row.get(x_key),
                **{
                    f"series{idx}:value": row.get(sk, 0)
                    for idx, sk in enumerate(series_keys)
                },
            }
            for row in rows
        ],
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
        # Old code used falsy check 'r.get(value_key) or 1' which replaced zero values with 1.
        raw_val = r.get(value_key)
        value_val = raw_val if raw_val is not None else 1
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
    elif inferred_type in {"CHART_BAR", "BAR_CHART"}:
        x = data.get("xKey")
        y = data.get("yKey")
        rows = data.get("rows", [])
        if rows:
            if x not in rows[0]:
                rows[0][x] = "Category"
            if y not in rows[0]:
                rows[0][y] = 0
    elif inferred_type in {"CHART_LINE", "LINE_CHART", "AREA_CHART"}:
        x = data.get("xKey")
        series = data.get("series", [])
        rows = data.get("rows", [])
        for row in rows:
            if x not in row:
                row[x] = "N/A"
            for s in series:
                if s["key"] not in row:
                    row[s["key"]] = 0
    elif inferred_type in {"CHART_PIE", "PIE_CHART", "DONUT_CHART", "RADIAL_CHART"}:
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
    visualization_intent: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    
    inferred_type = infer_supported_type(
        combined_result,
        query_type,
        intent,
        visualization_intent=visualization_intent,
    )
    
    from normalized_models import _derive_title
    title = _derive_title(query_type, intent)
    
    records = _extract_records(combined_result)
    
    if inferred_type == "CARD":
        data_payload = build_card_data(records, title)
    elif inferred_type in {"CHART_BAR", "BAR_CHART"}:
        data_payload = build_bar_chart_data(combined_result)
    elif inferred_type in {"CHART_LINE", "LINE_CHART", "AREA_CHART"}:
        data_payload = build_line_chart_data(combined_result)
    elif inferred_type in {"CHART_PIE", "PIE_CHART", "DONUT_CHART", "RADIAL_CHART"}:
        data_payload = build_pie_chart_data(combined_result)
    else:
        data_payload = build_table_data(records)

    validate_payload(inferred_type, data_payload)
    
    fd = FormattedData(
        type=inferred_type,
        title=title,
        data=data_payload,
        presentation=(visualization_intent or {}).get("presentation"),
        chart_type=(visualization_intent or {}).get("chart_type"),
        comparison=(visualization_intent or {}).get("comparison"),
        trend=(visualization_intent or {}).get("trend"),
        group_by=(visualization_intent or {}).get("group_by"),
        metric=(visualization_intent or {}).get("metric"),
    )
    return fd.model_dump()
