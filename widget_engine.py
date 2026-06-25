"""
widget_engine.py
================
Widget / Schema Inference Engine for the AgniAI admin pipeline.

Converts CombinedResult into a single deterministic FormattedData structure.

Widget selection priority:
  1. visualization_intent["requested_widget_type"]  — frontend override (user chose a view)
  2. WIDGET_MAP[(category, operation)]              — intent-keyed defaults
  3. query_type heuristic                           — comparison/trend/distribution
  4. record count heuristic                        — 1 record → CARD
  5. TABLE                                         — last resort
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import ValidationError

from normalized_models import extract_records as _orig_extract_records

def capitalize_segment(s: str) -> str:
    if not s:
        return ""
    cleaned = "".join(part[0].upper() + part[1:] for part in s.split() if part)
    if cleaned.lower() in ("bmi", "bpet", "id"):
        return cleaned.upper()
    if cleaned.isupper():
        return cleaned
    return cleaned


def get_singular_key(key: str) -> str:
    if key.lower().endswith("s"):
        return key[:-1]
    return key


def recursive_flatten(val: Any, path_prefix: List[str], flat_dict: Dict[str, Any]) -> None:
    if isinstance(val, dict):
        if not path_prefix:
            for k, v in val.items():
                k_seg = capitalize_segment(k)
                recursive_flatten(v, [k_seg], flat_dict)
            return

        name_key = None
        for k in val.keys():
            k_lower = k.lower()
            if k_lower in ("name", "label", "sectionname", "subitemname", "itemname") or k_lower.endswith("name") or k_lower.endswith("label"):
                name_key = k
                break
        
        value_key = None
        for k in val.keys():
            k_lower = k.lower()
            if k_lower in ("marks", "marksobtained", "score", "value", "quantity"):
                if isinstance(val[k], (str, int, float, bool)) or val[k] is None:
                    value_key = k
                    break
        
        label_already_in_path = False
        if name_key and path_prefix:
            label_val = str(val[name_key])
            if label_val:
                label_seg = capitalize_segment(label_val)
                if path_prefix[-1].lower() == label_seg.lower():
                    label_already_in_path = True
        
        if label_already_in_path:
            if value_key:
                flat_dict["_".join(path_prefix)] = val[value_key]
            for k, v in val.items():
                if k == name_key or (value_key and k == value_key):
                    continue
                k_seg = capitalize_segment(k)
                recursive_flatten(v, path_prefix + [k_seg], flat_dict)
        else:
            if name_key and value_key:
                label_val = str(val[name_key])
                label_seg = capitalize_segment(label_val)
                flat_dict["_".join(path_prefix + [label_seg])] = val[value_key]
                for k, v in val.items():
                    if k in (name_key, value_key):
                        continue
                    k_seg = capitalize_segment(k)
                    recursive_flatten(v, path_prefix + [label_seg, k_seg], flat_dict)
            elif name_key:
                label_val = str(val[name_key])
                label_seg = capitalize_segment(label_val)
                for k, v in val.items():
                    if k == name_key:
                        continue
                    k_seg = capitalize_segment(k)
                    recursive_flatten(v, path_prefix + [label_seg, k_seg], flat_dict)
            else:
                for k, v in val.items():
                    k_seg = capitalize_segment(k)
                    recursive_flatten(v, path_prefix + [k_seg], flat_dict)
                    
    elif isinstance(val, list):
        parent_key = path_prefix[-1] if path_prefix else "Item"
        singular = get_singular_key(parent_key)
        base_prefix = capitalize_segment(singular)
        
        for idx, item in enumerate(val):
            segment_name = None
            if isinstance(item, dict):
                name_key = None
                for k in item.keys():
                    k_lower = k.lower()
                    if k_lower in ("name", "label", "sectionname", "subitemname", "itemname") or k_lower.endswith("name") or k_lower.endswith("label"):
                        name_key = k
                        break
                if name_key:
                    label_val = str(item[name_key])
                    if label_val:
                        segment_name = capitalize_segment(label_val)
            
            if not segment_name:
                segment_name = f"{base_prefix}{idx + 1}"
            
            new_path = list(path_prefix)
            if new_path:
                new_path[-1] = segment_name
            else:
                new_path = [segment_name]
            
            recursive_flatten(item, new_path, flat_dict)
            
    else:
        if val is None or isinstance(val, (str, int, float, bool)):
            col_key = "_".join(path_prefix)
            if col_key:
                flat_dict[col_key] = val
        else:
            col_key = "_".join(path_prefix)
            if col_key:
                flat_dict[col_key] = str(val)


def deep_flatten_record(r: Dict[str, Any]) -> Dict[str, Any]:
    flat = {}
    recursive_flatten(r, [], flat)
    return flat


def flatten_records(records: List[Dict[str, Any]], deep_flatten: bool = False) -> List[Dict[str, Any]]:
    flat_records = []
    for r in records:
        if isinstance(r, dict):
            if deep_flatten:
                flat_records.append(deep_flatten_record(r))
            else:
                flat_records.append(r)
        else:
            flat_records.append(r)
    return flat_records


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


def _extract_records(combined_result: Any, deep_flatten: bool = False) -> List[Dict[str, Any]]:
    if isinstance(combined_result, dict):
        if "sections" in combined_result and isinstance(combined_result["sections"], list):
            records = []
            for sec in combined_result["sections"]:
                if isinstance(sec, dict) and "data" in sec:
                    data_val = sec["data"]
                    if isinstance(data_val, list):
                        records.extend([r for r in data_val if isinstance(r, dict)])
            if records:
                return flatten_records(records, deep_flatten=deep_flatten)
        if "records" in combined_result and isinstance(combined_result["records"], list):
            return flatten_records([r for r in combined_result["records"] if isinstance(r, dict)], deep_flatten=deep_flatten)
    elif isinstance(combined_result, list):
        return flatten_records([r for r in combined_result if isinstance(r, dict)], deep_flatten=deep_flatten)
    return flatten_records(_orig_extract_records(combined_result), deep_flatten=deep_flatten)

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

# ---------------------------------------------------------------------------
# WIDGET_MAP — (category, operation) → widget type constant
#
# Keys must match exactly what classify_admin_intent() returns for
# intent["category"] and intent["operation"] / intent["subcategory"].
#
# Widget type constants:
#   TABLE        — tabular grid
#   CARD         — stat cards
#   CHART_BAR    — vertical bar chart
#   CHART_LINE   — line / trend chart
#   AREA_CHART   — area chart (comparison)
#   CHART_PIE    — pie chart
#   DONUT_CHART  — donut chart
#   RADIAL_CHART — radial / gauge chart
# ---------------------------------------------------------------------------
WIDGET_MAP: Dict[Tuple[str, str], str] = {
    # ── Performance ──────────────────────────────────────────────────────────
    ("Performance", "Top"):             "TABLE",
    ("Performance", "Bottom"):          "TABLE",
    ("Performance", "Improvement"):     "CHART_LINE",
    ("Performance", "ImprovementTrend"):"CHART_LINE",
    ("Performance", "Drop"):            "CHART_LINE",
    ("Performance", "DropTrend"):       "CHART_LINE",
    ("Performance", "Grading"):         "TABLE",
    ("Performance", "GradingSummary"):  "CHART_BAR",
    ("Performance", "Average"):         "CHART_PIE",
    ("Performance", "AttemptWise"):     "TABLE",
    ("Performance", "BestAttempt"):     "TABLE",
    ("Performance", "Compare"):         "AREA_CHART",
    ("Performance", "Comparison"):      "AREA_CHART",
    ("Performance", "Summary"):         "TABLE",
    ("Performance", "PassPercentage"):  "CHART_PIE",
    ("Performance", "FailPercentage"):  "CHART_PIE",
    ("Performance", "Overall"):         "TABLE",
    # ── Leave ─────────────────────────────────────────────────────────────────
    ("Leave", "Most"):                  "TABLE",
    ("Leave", "Least"):                 "TABLE",
    ("Leave", "Current"):               "TABLE",
    ("Leave", "Absconded"):             "TABLE",
    ("Leave", "LeaveType"):             "TABLE",
    # ── Medical ──────────────────────────────────────────────────────────────
    ("Medical", "Active"):              "TABLE",
    ("Medical", "BMI"):                 "DONUT_CHART",
    ("Medical", "BMIAnalysis"):         "DONUT_CHART",
    ("Medical", "Disease"):             "TABLE",
    # ── Attendance ───────────────────────────────────────────────────────────
    ("Attendance", "Monthly"):          "CHART_BAR",
    ("Attendance", "MonthlyAttendance"):"CHART_BAR",
    ("Attendance", "Weekly"):           "CHART_BAR",
    ("Attendance", "WeeklyAttendance"): "CHART_BAR",
    ("Attendance", "Daily"):            "TABLE",
    ("Attendance", "Present"):          "CHART_PIE",
    ("Attendance", "PresentToday"):     "CHART_PIE",
    ("Attendance", "Summary"):          "TABLE",
    ("Attendance", "AttendanceSummary"):"TABLE",
    # ── Strength ─────────────────────────────────────────────────────────────
    ("Strength", "Strength"):           "RADIAL_CHART",
    ("Strength", "StrengthBreakdown"):  "RADIAL_CHART",
    ("Strength", "Overall"):            "RADIAL_CHART",
    # ── Verification ─────────────────────────────────────────────────────────
    ("Verification", "Pending"):        "TABLE",
    ("Verification", "Completed"):      "TABLE",
    ("Verification", "CompletedVerification"): "TABLE",
    ("Verification", "NotResponded"):   "TABLE",
    ("Verification", "Verified"):       "TABLE",
    ("Verification", "Rejected"):       "TABLE",
    ("Verification", "Sent"):           "TABLE",
    ("Verification", "SentVerification"):"TABLE",
    # ── Equipment ────────────────────────────────────────────────────────────
    ("Equipment", "Stats"):             "CARD",
    ("Equipment", "EquipmentSummary"):  "CARD",
    ("Equipment", "Overdue"):           "TABLE",
    ("Equipment", "Returned"):          "TABLE",
    ("Equipment", "Returend"):          "TABLE",   # typo in original, keep for compat
    ("Equipment", "Issued"):            "TABLE",
    ("Equipment", "IssuedItems"):       "TABLE",
    ("Equipment", "Procured"):          "TABLE",
    ("Equipment", "ProcuredItems"):     "TABLE",
    ("Equipment", "Holding"):           "TABLE",
    ("Equipment", "HoldingEquipment"):  "TABLE",
    ("Equipment", "AgniveerWise"):      "TABLE",
    ("Equipment", "AgniveerWiseEquipment"): "TABLE",
    # ── Skills / Roster ──────────────────────────────────────────────────────
    ("Skills", "BySport"):              "TABLE",
    ("Skills", "ByClass"):              "TABLE",
    ("Roster", "BySport"):              "TABLE",
    ("Roster", "ByClass"):              "TABLE",
    # ── Distribution ─────────────────────────────────────────────────────────
    ("Distribution", "Latest"):         "TABLE",
    ("Distribution", "ByUnit"):         "TABLE",
    ("Distribution", "Unassigned"):     "TABLE",
    ("Distribution", "TopUnit"):        "CARD",
    ("Distribution", "Overall"):        "TABLE",
    ("Distribution", "Schedule"):       "TABLE",
    # ── Overall (top-level) ──────────────────────────────────────────────────
    ("Overall", "Overall"):             "TABLE",
    ("Overall", "OverallPerformance"):  "TABLE",
    # ── Schedule ─────────────────────────────────────────────────────────────
    # Keyed by both .NET operation string AND subcategory name
    ("schedule", "Date"):               "TABLE",
    ("schedule", "date"):               "TABLE",
    ("schedule", "company"):            "TABLE",
    ("schedule", "agniveer"):           "TABLE",
    ("schedule", "CompanySchedule"):    "TABLE",
    ("schedule", "AgniveerSchedule"):   "TABLE",
    ("schedule", "DateSchedule"):       "TABLE",
    ("Schedule", "Date"):               "TABLE",
    # ── Medical extras ───────────────────────────────────────────────────────
    ("Medical", "Individual"):          "TABLE",
    ("Medical", "IndividualMedical"):   "TABLE",
    ("Medical", "BloodGroup"):          "TABLE",
    # ── Attendance extras ────────────────────────────────────────────────────
    ("Attendance", "Yearly"):           "CHART_BAR",
    ("Attendance", "YearlyAttendance"): "CHART_BAR",
    # ── Performance subcategory-keyed (direct lookup without alias) ───────────
    ("Performance", "GradeDistribution"):  "TABLE",
    ("Performance", "AverageScore"):       "CHART_PIE",
    ("Performance", "SectionSummary"):     "TABLE",
    ("Performance", "OverallPerformance"): "TABLE",
}

# ---------------------------------------------------------------------------
# Operation aliases — maps .NET command names → canonical operation keys
# ---------------------------------------------------------------------------
_OPERATION_ALIASES: Dict[str, str] = {
    "TopPerformers":          "Top",
    "LowestPerformers":       "Bottom",
    "Comparison":             "Compare",
    "MonthlyAttendance":      "Monthly",
    "WeeklyAttendance":       "Weekly",
    "YearlyAttendance":       "Yearly",
    "PresentToday":           "Present",
    "StrengthBreakdown":      "Strength",
    "BMIAnalysis":            "BMI",
    "EquipmentSummary":       "Stats",
    "IssuedItems":            "Issued",
    "ProcuredItems":          "Procured",
    "CompletedVerification":  "Verified",
    "SentVerification":       "Sent",
    "HoldingEquipment":       "Holding",
    "AgniveerWiseEquipment":  "AgniveerWise",
    "IndividualMedical":      "Individual",
    "AttendanceSummary":      "Summary",
    "ImprovementTrend":       "Improvement",
    "DropTrend":              "Drop",
    "GradingSummary":         "GradingSummary",
    "BestAttempt":            "BestAttempt",
    "AttemptWise":            "AttemptWise",
    "PassPercentage":         "PassPercentage",
    "FailPercentage":         "FailPercentage",
    # admin_intent subcategory names → WIDGET_MAP keys
    "GradeDistribution":      "Grading",        # → ("Performance","Grading") TABLE
    "AverageScore":           "Average",         # → ("Performance","Average") CHART_PIE
    "SectionSummary":         "Summary",         # → ("Performance","Summary") TABLE
    "OverallPerformance":     "Overall",         # → ("Performance","Overall") TABLE
    "MostLeaveTaken":         "Most",
    "LeastLeaveTaken":        "Least",
    "CurrentLeaveStatus":     "Current",
    "AbscondedPerson":        "Absconded",
    "ActiveCases":            "Active",
    "DiseaseStatistics":      "Disease",
    "BloodGroup":             "BloodGroup",      # → ("Medical","BloodGroup") TABLE
    "DailyAttendance":        "Daily",
    "PendingVerification":    "Pending",
    "NotRespondedVerification": "NotResponded",
    "VerifiedVerification":   "Verified",
    "RejectedVerification":   "Rejected",
    "OverdueEquipment":       "Overdue",
    "PoorConditionEquipment": "Returned",
    "LatestDistribution":     "Latest",
    "DistributionByUnit":     "ByUnit",
    "UnassignedItems":        "Unassigned",
    "CompanySchedule":        "company",
    "AgniveerSchedule":       "agniveer",
    "DateSchedule":           "date",
    # Direct subcategory → map key (for cases where alias = key itself)
    "Top":                    "Top",
    "Bottom":                 "Bottom",
    "Grading":                "Grading",
    "Average":                "Average",
    "Summary":                "Summary",
    "Compare":                "Compare",
    "Overall":                "Overall",
    "Most":                   "Most",
    "Least":                  "Least",
    "Current":                "Current",
    "Absconded":              "Absconded",
    "LeaveType":              "LeaveType",
    "Active":                 "Active",
    "Disease":                "Disease",
    "Monthly":                "Monthly",
    "Weekly":                 "Weekly",
    "Daily":                  "Daily",
    "Present":                "Present",
    "Pending":                "Pending",
    "Completed":              "Completed",
    "NotResponded":           "NotResponded",
    "Verified":               "Verified",
    "Rejected":               "Rejected",
    "Overdue":                "Overdue",
    "Returned":               "Returned",
    "Returend":               "Returend",
    "BySport":                "BySport",
    "ByClass":                "ByClass",
    "Latest":                 "Latest",
    "ByUnit":                 "ByUnit",
    "Unassigned":             "Unassigned",
    "TopUnit":                "TopUnit",
    "Schedule":               "Schedule",
}

# ---------------------------------------------------------------------------
# _normalize_requested_widget_type
# Accepts any label the frontend might send and returns the canonical constant.
# This is the frontend override — if set, it ALWAYS wins.
# ---------------------------------------------------------------------------
def _normalize_requested_widget_type(value: Any) -> Optional[str]:
    """
    Map a frontend-supplied widget label to a canonical widget type constant.
    Returns None if the value is empty or unrecognized.

    Accepts:
      - Exact canonical constants: "TABLE", "CHART_BAR", etc.
      - Human-readable labels matching what the frontend dropdown shows:
        "Tabular", "Bar Chart", "Improvement Trend Chart", etc.
    """
    text = str(value or "").strip()
    if not text:
        return None

    # Direct canonical match (frontend already sending internal constant)
    _CANONICAL = {
        "TABLE", "CARD", "CHART_BAR", "CHART_LINE",
        "AREA_CHART", "CHART_PIE", "DONUT_CHART", "RADIAL_CHART",
    }
    if text in _CANONICAL:
        return text

    # Case-insensitive label lookup covering all labels from the widget menu
    _LABEL_MAP: Dict[str, str] = {
        # ── Tabular ───────────────────────────────────────────────────────────
        "tabular":                      "TABLE",
        "table":                        "TABLE",
        "grid":                         "TABLE",
        # ── Card ──────────────────────────────────────────────────────────────
        "card":                         "CARD",
        "cards":                        "CARD",
        "stats card":                   "CARD",
        # ── Bar Chart ─────────────────────────────────────────────────────────
        "bar chart":                    "CHART_BAR",
        "bar":                          "CHART_BAR",
        "monthly bar chart":            "CHART_BAR",
        "weekly bar chart":             "CHART_BAR",
        "gradingsummary bar chart":     "CHART_BAR",
        # ── Line / Trend Chart ────────────────────────────────────────────────
        "line chart":                   "CHART_LINE",
        "line":                         "CHART_LINE",
        "trend chart":                  "CHART_LINE",
        "trend":                        "CHART_LINE",
        "improvement trend chart":      "CHART_LINE",
        "drop trend chart":             "CHART_LINE",
        # ── Area Chart ────────────────────────────────────────────────────────
        "area chart":                   "AREA_CHART",
        "area":                         "AREA_CHART",
        "compare area chart":           "AREA_CHART",
        # ── Pie Chart ─────────────────────────────────────────────────────────
        "pie chart":                    "CHART_PIE",
        "pie":                          "CHART_PIE",
        "average pie chart":            "CHART_PIE",
        "present pie chart":            "CHART_PIE",
        "passpercentage pie chart":     "CHART_PIE",
        "failpercentage pie chart":     "CHART_PIE",
        # ── Donut Chart ───────────────────────────────────────────────────────
        "donut chart":                  "DONUT_CHART",
        "donut":                        "DONUT_CHART",
        "bmi donut chart":              "DONUT_CHART",
        # ── Radial Chart ──────────────────────────────────────────────────────
        "radial chart":                 "RADIAL_CHART",
        "radial":                       "RADIAL_CHART",
        "strength radial chart":        "RADIAL_CHART",
    }

    lower = text.lower()
    result = _LABEL_MAP.get(lower)
    if result:
        return result

    # Normalise separators and retry
    normalised = lower.replace("_", " ").replace("-", " ")
    return _LABEL_MAP.get(normalised)


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
        "TABLE":        "TABLE",
        "CARD":         "CARD",
        "METRIC_CARD":  "CARD",
        "BAR_CHART":    "CHART_BAR",
        "CHART_BAR":    "CHART_BAR",
        "LINE_CHART":   "CHART_LINE",
        "CHART_LINE":   "CHART_LINE",
        "AREA_CHART":   "CHART_LINE",   # area renders as line in current frontend
        "PIE_CHART":    "CHART_PIE",
        "CHART_PIE":    "CHART_PIE",
        "DONUT_CHART":  "CHART_PIE",
        "RADIAL_CHART": "CHART_PIE",
        "CALENDAR_UI":  "TABLE",
    }
    return mapped.get(inferred, "TABLE")


def _default_widget_type_for_intent(
    category: str,
    operation: str,
    query_type: str,
) -> Optional[str]:
    category_key = (category or "").strip()
    operation_key = _OPERATION_ALIASES.get(
        (operation or "").strip(), (operation or "").strip()
    )

    # Try exact (category, operation) lookup first
    if category_key and operation_key:
        widget_type = WIDGET_MAP.get((category_key, operation_key))
        if widget_type:
            return widget_type
        # Also try the raw operation without alias resolution
        widget_type = WIDGET_MAP.get((category_key, (operation or "").strip()))
        if widget_type:
            return widget_type

    # Query-type heuristics when no map entry matches
    qtype = (query_type or "").strip().lower()
    if qtype in ("compare", "comparison"):
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
    """
    Resolve the widget type using priority order:
      1. visualization_intent["requested_widget_type"]  — frontend override
      2. visualization_intent presentation/chart_type hints (non-override path)
      3. WIDGET_MAP[(category, operation)]
      4. query_type heuristic
      5. Record-count heuristic (1 record → CARD)
      6. TABLE
    """
    qtype = (query_type or "").strip().lower()

    if isinstance(visualization_intent, dict):
        # ── Priority 1: explicit frontend override ───────────────────────────
        # Honour this when user chose a different widget type or requested it.
        if visualization_intent.get("frontend_override") or visualization_intent.get("requested_widget_type") or visualization_intent.get("widget_type"):
            raw_requested = (
                visualization_intent.get("requested_widget_type")
                or visualization_intent.get("widget_type")
            )
            normalized = _normalize_requested_widget_type(raw_requested)
            if normalized:
                return normalized

        # ── Priority 2: presentation/chart_type hints (non-override) ────────
        # These are soft hints — they lose to WIDGET_MAP if a map entry exists.
        presentation = (visualization_intent.get("presentation") or "").strip().lower()
        chart_type   = (visualization_intent.get("chart_type")   or "").strip().lower()

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
            if chart_type == "donut":
                return "DONUT_CHART"
            if chart_type == "radial":
                return "RADIAL_CHART"
            # generic "chart" hint without chart_type — fall through to map

    # ── Priority 3: WIDGET_MAP ───────────────────────────────────────────────
    # Use subcategory first because `operation` holds query mechanics
    # ("ranking", "filter") which have no map entries, while subcategory
    # holds the semantic command name ("TopPerformers", "BMIAnalysis", etc.).
    category   = (intent.get("category")   or "").strip()
    subcategory = (intent.get("subcategory") or "").strip()
    operation   = (intent.get("operation")   or "").strip()

    # Try subcategory first, then fall back to operation
    default_widget = _default_widget_type_for_intent(category, subcategory, query_type)
    if not default_widget and operation:
        default_widget = _default_widget_type_for_intent(category, operation, query_type)
    if default_widget:
        return default_widget

    # ── Priority 4 / 5: record count / fallback ────────────────────────────
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

# ---------------------------------------------------------------------------
# Column key normalisation helpers
# ---------------------------------------------------------------------------

# Suffixes that mark internal .NET metadata fields — never shown in the table.
_EXCLUDED_COLUMN_SUFFIXES = (
    "_SubItemId",
    "_MaxMarks",
    "_IsBestAttempt",
    "_SectionId",
    "_DisplayOrder",
)

# Top-level fields that are internal and should not appear as columns.
_EXCLUDED_COLUMN_KEYS_EXACT = {
    "IsActive", "isActive",
    "ID", "id",
    "DisplayOrder", "displayOrder",
    "SectionId", "sectionId",
}


def _pascal_to_camel(key: str) -> str:
    """Convert a PascalCase or mixed-case .NET key to camelCase.

    Examples:
        FullName          -> fullName
        AgniveerNo        -> agniveerNo
        BestTotal         -> bestTotal
        Attempt1_BPET_5km -> attempt1_BPET_5km  (prefix only lowercased)
    """
    if not key:
        return key
    # For compound keys like "Attempt1_BPET_5km" only lower the very first char
    # so that the uppercase section acronyms (BPET, PPT, FIRING, DRILL) are preserved.
    return key[0].lower() + key[1:]


def _should_exclude_column(key: str) -> bool:
    """Return True for internal .NET metadata fields that should be hidden."""
    if key in _EXCLUDED_COLUMN_KEYS_EXACT:
        return True
    for suffix in _EXCLUDED_COLUMN_SUFFIXES:
        if key.endswith(suffix):
            return True
    return False


def make_readable_label(k: str) -> str:
    import re
    parts = k.split("_")
    readable_parts = []
    for part in parts:
        if part.isupper():
            readable_parts.append(part)
        else:
            s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", part)
            s = re.sub(r"([A-Z])([A-Z][a-z])", r"\1 \2", s)
            readable_parts.append(s.strip().title())
            
    label = " ".join(readable_parts)
    label = label.replace("Agniveer No", "Agniveer No.").replace("Id", "ID").replace("Bpet", "BPET").replace("Bmi", "BMI")
    return label


def build_table_data(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {"columns": [], "rows": []}

    # Collect all keys, preserving insertion order across all records.
    keys_seen: List[str] = []
    for r in records:
        if isinstance(r, dict):
            for k in r.keys():
                if k not in keys_seen:
                    keys_seen.append(k)

    # Filter out internal metadata columns before building anything else.
    keys_seen = [k for k in keys_seen if not _should_exclude_column(k)]

    # Sort with priority fields first.
    key_priority = {
        "fullname": -10, "agniveerno": -9, "name": -8,
        "agniveerid": -7, "score": -6, "besttotal": -5,
    }
    keys_seen.sort(key=lambda k: key_priority.get(k.lower(), 0))

    # Build normalised column specs: camelCase key, human-readable label.
    columns = []
    key_map: Dict[str, str] = {}   # original_key -> camelCase_key
    for k in keys_seen:
        camel = _pascal_to_camel(k)
        key_map[k] = camel
        label = make_readable_label(k)
        columns.append({"key": camel, "label": label})

    # Build rows using camelCase keys to match the column specs.
    rows = []
    for r in records:
        row: Dict[str, Any] = {}
        for orig_k, camel_k in key_map.items():
            val = r.get(orig_k)
            if isinstance(val, (dict, list)):
                row[camel_k] = json.dumps(val, ensure_ascii=False, default=str)
            else:
                row[camel_k] = val
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
                    "label": row.get("label", "Section"),
                    y_key: row.get(y_key, row.get("recordCount", 0)),
                }
                for row in rows
            ],
        }
        
    records = _extract_records(combined_result, deep_flatten=False)
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
                x_key: row.get(x_key),
                y_key: row.get(y_key),
            }
            for row in rows
        ],
    }

def build_line_chart_data(combined_result: Any) -> Dict[str, Any]:
    records = _extract_records(combined_result, deep_flatten=False)
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
                x_key: row.get(x_key),
                **{sk: row.get(sk, 0) for sk in series_keys},
            }
            for row in rows
        ],
    }

def build_pie_chart_data(combined_result: Any) -> Dict[str, Any]:
    records = _extract_records(combined_result, deep_flatten=False)
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
        raw_val = r.get(value_key)
        value_val = raw_val if raw_val is not None else 1
        rows.append({
            "label": str(label_val),
            "value": value_val
        })
        
    return {"rows": rows}

def validate_payload(inferred_type: str, data: Dict[str, Any]) -> None:
    if inferred_type == "TABLE":
        if "sides" in data:
            return
        if "sections" in data:
            for sec in data["sections"]:
                if isinstance(sec, dict):
                    validate_payload("TABLE", sec)
            return
        if "left" in data and "right" in data:
            if isinstance(data["left"], dict):
                validate_payload("TABLE", data["left"])
            if isinstance(data["right"], dict):
                validate_payload("TABLE", data["right"])
            return
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
    source_result = combined_result

    inferred_type = infer_supported_type(
        source_result,
        query_type,
        intent,
        visualization_intent=visualization_intent,
    )
    
    from normalized_models import _derive_title
    title = _derive_title(query_type, intent)
    
    # Check for multi_independent and comparison early to bypass extract/flatten
    if query_type == "multi_independent" and isinstance(source_result, dict) and "sections" in source_result:
        sections_list = []
        for sec in source_result["sections"]:
            if isinstance(sec, dict):
                label = sec.get("label", "")
                sec_data = sec.get("data", [])
                if not isinstance(sec_data, list):
                    sec_data = [sec_data] if sec_data else []
                flat_sec_records = flatten_records(sec_data, deep_flatten=True)
                sec_table = build_table_data(flat_sec_records)
                sec_payload = {
                    "label": label,
                    "columns": sec_table.get("columns", []),
                    "rows": sec_table.get("rows", [])
                }
                for k, v in sec.items():
                    if k not in ("label", "data", "columns", "rows"):
                        sec_payload[k] = v
                sections_list.append(sec_payload)
        data_payload = {"sections": sections_list}
        
    elif isinstance(source_result, dict) and "left" in source_result and "right" in source_result:
        left_section = source_result["left"]
        right_section = source_result["right"]
        
        # Format left
        left_label = ""
        left_flat_records = []
        left_extra = {}
        if isinstance(left_section, dict):
            left_label = left_section.get("label", "")
            left_data = left_section.get("data", [])
            if not isinstance(left_data, list):
                left_data = [left_data] if left_data else []
            left_flat_records = flatten_records(left_data, deep_flatten=True)
            for k, v in left_section.items():
                if k not in ("label", "data", "columns", "rows"):
                    left_extra[k] = v
        else:
            left_data = left_section if isinstance(left_section, list) else []
            left_flat_records = flatten_records(left_data, deep_flatten=True)
        
        left_table = build_table_data(left_flat_records)
        left_payload = {
            "label": left_label,
            "columns": left_table.get("columns", []),
            "rows": left_table.get("rows", []),
            **left_extra
        }
        
        # Format right
        right_label = ""
        right_flat_records = []
        right_extra = {}
        if isinstance(right_section, dict):
            right_label = right_section.get("label", "")
            right_data = right_section.get("data", [])
            if not isinstance(right_data, list):
                right_data = [right_data] if right_data else []
            right_flat_records = flatten_records(right_data, deep_flatten=True)
            for k, v in right_section.items():
                if k not in ("label", "data", "columns", "rows"):
                    right_extra[k] = v
        else:
            right_data = right_section if isinstance(right_section, list) else []
            right_flat_records = flatten_records(right_data, deep_flatten=True)
            
        right_table = build_table_data(right_flat_records)
        right_payload = {
            "label": right_label,
            "columns": right_table.get("columns", []),
            "rows": right_table.get("rows", []),
            **right_extra
        }
        
        comparison_payload = source_result.get("comparison", {})
        
        data_payload = {
            "left": left_payload,
            "right": right_payload,
            "comparison": comparison_payload
        }

    elif inferred_type == "CARD":
        records = _extract_records(source_result, deep_flatten=False)
        data_payload = build_card_data(records, title)
    elif inferred_type in {"CHART_BAR", "BAR_CHART"}:
        data_payload = build_bar_chart_data(source_result)
    elif inferred_type in {"CHART_LINE", "LINE_CHART", "AREA_CHART"}:
        data_payload = build_line_chart_data(source_result)
    elif inferred_type in {"CHART_PIE", "PIE_CHART", "DONUT_CHART", "RADIAL_CHART"}:
        data_payload = build_pie_chart_data(source_result)
    else:
        if isinstance(source_result, dict) and "sides" in source_result:
            sides_data = []
            for side in source_result["sides"]:
                s_data = side.get("data")
                if isinstance(s_data, list):
                    flattened_data = [r for r in s_data if isinstance(r, dict)]
                else:
                    flattened_data = s_data
                side_payload = {
                    "label": side.get("label"),
                    "data": flattened_data
                }
                for k, v in side.items():
                    if k not in ("label", "data"):
                        side_payload[k] = v
                sides_data.append(side_payload)
            data_payload = {"sides": sides_data}
        else:
            table_records = _extract_records(source_result, deep_flatten=True)
            data_payload = build_table_data(table_records)

    # Preserve 100% of .NET data and unknown keys
    if isinstance(source_result, dict) and isinstance(data_payload, dict):
        for k, v in source_result.items():
            if k not in data_payload:
                data_payload[k] = v

    validate_payload(inferred_type, data_payload)
    
    fd = FormattedData(
        type=inferred_type,
        title=title,
        data=data_payload,
        analysis=analysis or {},
        prediction=prediction or {},
        conclusion=conclusion or {},
        presentation=(visualization_intent or {}).get("presentation"),
        chart_type=(visualization_intent or {}).get("chart_type"),
        comparison=(visualization_intent or {}).get("comparison"),
        trend=(visualization_intent or {}).get("trend"),
        group_by=(visualization_intent or {}).get("group_by"),
        metric=(visualization_intent or {}).get("metric"),
    )
    return fd.model_dump()


# =============================================================================
# MULTI-WIDGET API  (new contract: formattedData is always a list)
# =============================================================================

def build_summary_card_from_analysis(
    analysis: Optional[Dict[str, Any]],
    query_type: str,
    intent: Dict[str, Any],
) -> Dict[str, Any]:
    """Build CARD data from analysis.statistics for use as a summary widget."""
    stats    = (analysis or {}).get("statistics") or {}
    category = (intent or {}).get("category") or "Results"
    cards: List[Dict[str, Any]] = []

    def _card(title: str, value: Any) -> Dict[str, Any]:
        return {"title": title, "subtitle": "", "value": str(value), "description": ""}

    rc = stats.get("record_count")
    if rc is not None:
        cards.append(_card("Total Records", rc))

    avg = stats.get("average_score")
    if avg is not None:
        cards.append(_card("Average Score", avg))

    lc = stats.get("left_count")
    rc2 = stats.get("right_count")
    if lc is not None:
        left_label  = stats.get("left_label")  or "Side 1"
        right_label = stats.get("right_label") or "Side 2"
        cards.append(_card(left_label,  lc))
        if rc2 is not None:
            cards.append(_card(right_label, rc2))

    mc = stats.get("match_count")
    if mc is not None and lc is None:
        cards.append(_card("Matched Records", mc))

    sc = stats.get("section_count")
    if sc is not None:
        cards.append(_card("Sections", sc))

    if not cards:
        cards.append(_card(category, "–"))

    return {"cards": cards}


def _build_comparison_bar_chart(combined_result: Any) -> Dict[str, Any]:
    """Build a side-by-side bar chart from a comparison combined_result."""
    if not isinstance(combined_result, dict):
        return build_bar_chart_data(combined_result)
    left  = combined_result.get("left")  or {}
    right = combined_result.get("right") or {}
    left_label  = (left.get("label")  if isinstance(left, dict) else None)  or "Side 1"
    right_label = (right.get("label") if isinstance(right, dict) else None) or "Side 2"
    left_count  = len(left.get("data")  or []) if isinstance(left, dict) else 0
    right_count = len(right.get("data") or []) if isinstance(right, dict) else 0
    return {
        "xKey": "label",
        "yKey": "count",
        "rows": [
            {"label": left_label,  "count": left_count},
            {"label": right_label, "count": right_count},
        ],
    }


def _build_widget_data(
    spec: Any,   # WidgetSpec — typed loosely to avoid circular import
    combined_result: Any,
    query_type: str,
    intent: Dict[str, Any],
    analysis: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Dispatch to the appropriate builder for a single WidgetSpec."""
    wt   = spec.widget_type
    hint = spec.source_hint

    # ── Summary CARD (from analysis.statistics) ──────────────────────────────
    if hint == "summary":
        return build_summary_card_from_analysis(analysis, query_type, intent)

    # ── Primary CARD (from raw records) ─────────────────────────────────────
    if wt == "CARD":
        records = _extract_records(combined_result, deep_flatten=False)
        return build_card_data(records, spec.title)

    # ── TABLE — left/right/section/primary ──────────────────────────────────
    if wt == "TABLE":
        if hint == "left" and isinstance(combined_result, dict):
            left = combined_result.get("left") or {}
            raw  = left.get("data") if isinstance(left, dict) else []
            flat = flatten_records(raw or [], deep_flatten=True)
        elif hint == "right" and isinstance(combined_result, dict):
            right = combined_result.get("right") or {}
            raw   = right.get("data") if isinstance(right, dict) else []
            flat  = flatten_records(raw or [], deep_flatten=True)
        elif hint == "section" and isinstance(combined_result, dict):
            flat = []
            for sec in (combined_result.get("sections") or []):
                if isinstance(sec, dict) and sec.get("label") == spec.section_label:
                    flat = flatten_records(sec.get("data") or [], deep_flatten=True)
                    break
        else:
            flat = _extract_records(combined_result, deep_flatten=True)
        return build_table_data(flat)

    # ── CHART_BAR ────────────────────────────────────────────────────────────
    if wt == "CHART_BAR":
        if query_type in ("compare", "comparison"):
            return _build_comparison_bar_chart(combined_result)
        return build_bar_chart_data(combined_result)

    # ── CHART_LINE ───────────────────────────────────────────────────────────
    if wt == "CHART_LINE":
        return build_line_chart_data(combined_result)

    # ── CHART_PIE ────────────────────────────────────────────────────────────
    if wt == "CHART_PIE":
        return build_pie_chart_data(combined_result)

    # Fallback — unknown type becomes a TABLE
    flat = _extract_records(combined_result, deep_flatten=True)
    return build_table_data(flat)


def build_widget_list(
    combined_result: Any,
    query_type: str,
    intent: Dict[str, Any],
    analysis: Optional[Dict[str, Any]] = None,
    visualization_intent: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Entry point for multi-widget response assembly.

    Returns an ordered list of widget dicts, each:
        {"id": str, "type": str, "title": str, "data": dict}

    Always returns at least one widget — never an empty list.
    """
    from widget_selector import WidgetSelector

    # Determine the primary widget type (reuses existing priority logic)
    primary_wt = infer_supported_type(
        combined_result, query_type, intent, visualization_intent
    )

    # Extract frontend override (if user explicitly picked a different view)
    frontend_override: Optional[str] = None
    if isinstance(visualization_intent, dict):
        if visualization_intent.get("frontend_override") or visualization_intent.get("requested_widget_type"):
            raw = (
                visualization_intent.get("requested_widget_type")
                or visualization_intent.get("widget_type")
            )
            if raw:
                frontend_override = _normalize_requested_widget_type(raw)

    # Ask the selector for the ordered spec list
    selector = WidgetSelector()
    specs = selector.select(
        query_type=query_type,
        intent=intent,
        combined_result=combined_result,
        primary_widget_type=primary_wt,
        analysis=analysis,
        frontend_override_type=frontend_override,
    )

    widgets: List[Dict[str, Any]] = []
    for spec in specs:
        try:
            data = _build_widget_data(spec, combined_result, query_type, intent, analysis)
        except Exception:
            import logging as _logging
            _logging.getLogger(__name__).exception(
                "build_widget_list: failed building widget %s", spec.widget_id
            )
            data = {}
        widgets.append({
            "id":    spec.widget_id,
            "type":  spec.widget_type,
            "title": spec.title,
            "data":  data,
        })

    # Guard: never return an empty list — fall back to a plain TABLE
    if not widgets:
        flat = _extract_records(combined_result, deep_flatten=True)
        widgets.append({
            "id":    "fallback_table",
            "type":  "TABLE",
            "title": "Results",
            "data":  build_table_data(flat),
        })

    return widgets