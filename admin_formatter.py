"""
admin_formatter.py
==================
Formats raw .NET AiCommand JSON responses into clean plain-text answers
for the admin chatbot UI. No markdown symbols, no emojis, no bold markers.

REAL .NET RESPONSE SHAPES (from actual data):

Leave:
  [ { "agniveerId": 1210, "agniveerNo": "A0701532AA", "fullName": "AMIT KUMAR",
      "photoPath": null, "fromDate": "2026-06-24T06:30:00", "toDate": null, "remarks": "" } ]

Medical BMI:
  [ { "agniveerId": 616, "agniveerNo": "", "fullName": "", "photoPath": null,
      "heightCm": 21, "weightKg": 21, "bmiValue": 476.19, "bmiCategory": "obese" } ]

Pending Verification:
  [ { "verificationId": 10, "agniveerId": 614, "agniveerNo": "A0701528W",
      "fullName": "LAVEPREET", "photoPath": null, "policeStation": "HAJIPUR",
      "sentDate": "2026-06-01T11:48:02.39" } ]

Completed Verification:
  [ { "verificationId": 36, "agniveerId": 822, "agniveerNo": "A0701778X",
      "fullName": "Keshav Rana", "photoPath": null, "policeStation": "BHAWARNA",
      "sentDate": "2026-06-05T11:48:02.39", "receivedDate": "2026-06-09T23:48:02.39",
      "status": "Verified" } ]

Equipment Stats:
  { "success": true, "commandLabel": "Equipment Stats",
    "data": { "totalAssigned": 505, "active": 444, "returned": 61, "poorCondition": 0 } }

Distribution Latest:
  { "success": true, "commandLabel": "Latest Distribution",
    "data": { "distributionId": 5, "distributionDate": "2026-06-12T...",
              "teams": [ { "teamId": 4, "teamName": "14 Punjab", "memberCount": 21,
                           "members": [ { "agniveerId": 639, "agniveerNo": "A0701560L",
                                          "fullName": "PARAMJEET SINGH", "class": "SIKH",
                                          "rank": 1 } ] } ] } }

Distribution By Unit:
  [ { "agniveerId": 946, "agniveerNo": "A0701915L", "fullName": "TARUN KUMAR",
      "photoPath": null, "class": "DOGRA", "rank": 420,
      "distributionEventId": 6, "distributionDate": "2026-06-15T..." } ]

Unassigned:
  [ { "agniveerId": 605, "agniveerNo": "A0701515Y", "fullName": "SHAVAIT CHOUDHARY",
      "photoPath": "uploads/...", "platoonName": "PL-01" } ]

Top Unit:
  { "success": true, "commandLabel": "Top Unit",
    "data": { "distributionEventId": 6, "distributionDate": "2026-06-15T...",
              "teamId": 4, "teamName": "14 Punjab", "agniveerCount": 10 } }

By Sport:
  [ { "agniveerId": 605, "agniveerNo": "A0701515Y", "fullName": "SHAVAIT CHOUDHARY",
      "photoPath": "...", "sports": "Cricket, Kabaddi", "platoonName": "PL-01" } ]

By Class:
  [ { "agniveerId": 680, "agniveerNo": "A0701619A", "fullName": "SULINDER SINGH",
      "photoPath": null, "class": "SIKH", "platoonName": "PL-01", "bloodGroup": "AB-" } ]

Strength Breakdown:
  { "success": true, "commandLabel": "Strength Breakdown",
    "data": { "totalAgniveers": 610, "activeCount": 608, "inactiveCount": 2,
              "presentToday": 0, "absentToday": 0, "onLeave": 10,
              "byPlatoon": [ { "platoonName": "PL-01", "count": 30 }, ... ] } }

NOTE on intent_result keys:
  Receives the original intent_result dict from classify_admin_intent().
  Uses snake_case internally: intent_result["leave_type"], not "leaveType".
"""

from __future__ import annotations

import json
import logging
import re as _re
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# GENERIC HELPERS
# =============================================================================

def _safe_str(value: Any, fallback: str = "N/A") -> str:
    if value is None:
        return fallback
    s = str(value).strip()
    return s if s else fallback


def _camel_to_words(key: str) -> str:
    return _re.sub(r"(?<!^)(?=[A-Z])", " ", key).title()


def _get(obj: Dict, *keys, fallback=None):
    """Try multiple key names, return first non-None hit."""
    for key in keys:
        v = obj.get(key)
        if v is not None:
            return v
    return fallback


def _plain_table(headers: List[str], rows: List[List[Any]]) -> str:
    """Build a plain-text aligned table from headers and rows."""
    if not rows:
        return "No data available."

    all_rows = [headers] + [
        [str(cell) if cell is not None else "" for cell in row]
        for row in rows
    ]
    col_widths = [
        max(len(str(r[i])) for r in all_rows)
        for i in range(len(headers))
    ]

    lines = []
    header_line = "  ".join(
        str(h).ljust(col_widths[i]) for i, h in enumerate(headers)
    )
    lines.append(header_line)
    lines.append("-" * len(header_line))
    for row in rows:
        lines.append(
            "  ".join(
                str(cell if cell is not None else "").ljust(col_widths[i])
                for i, cell in enumerate(row)
            )
        )
    return "\n".join(lines)


def _kv_block(pairs: List[tuple]) -> str:
    """Build a plain key: value block, skipping None / empty values."""
    lines = [
        f"{label}: {value}"
        for label, value in pairs
        if value not in (None, "", "N/A")
    ]
    return "\n".join(lines) if lines else "No data available."


def _rank_label(rank: int) -> str:
    suffixes = {1: "1st", 2: "2nd", 3: "3rd"}
    return suffixes.get(rank, f"{rank}th")


# =============================================================================
# DATE HELPERS
# =============================================================================

def _fmt_date(value: Any) -> str:
    """
    Format an ISO date string like "2026-06-24T06:30:00" → "24 Jun 2026".
    Returns "-" for None / unparseable values.
    """
    if not value:
        return "-"
    raw = str(value).strip()
    # strip fractional seconds and timezone
    date_part = raw.split("T")[0]
    try:
        dt = datetime.strptime(date_part, "%Y-%m-%d")
        return dt.strftime("%d %b %Y")
    except Exception:
        return date_part or "-"


def _fmt_datetime(value: Any) -> str:
    """
    Format an ISO datetime like "2026-06-01T11:48:02.39" → "01 Jun 2026 11:48".
    Returns "-" for None / unparseable values.
    """
    if not value:
        return "-"
    raw = str(value).strip()
    # drop fractional seconds
    raw_clean = raw.split(".")[0]
    try:
        dt = datetime.strptime(raw_clean, "%Y-%m-%dT%H:%M:%S")
        return dt.strftime("%d %b %Y %H:%M")
    except Exception:
        return _fmt_date(value)


def _calc_days(item: Dict) -> Optional[int]:
    """
    Compute leave / absence duration in days from fromDate → toDate.
    If toDate is null (still on leave / absconded), counts up to today.
    Returns None if fromDate is missing or unparseable.
    """
    from_raw = _get(item, "fromDate", "from", "startDate", "sentDate")
    to_raw   = _get(item, "toDate",   "to",   "endDate",   "receivedDate")
    if not from_raw:
        return None
    try:
        fmt = "%Y-%m-%dT%H:%M:%S"
        dt_from = datetime.strptime(str(from_raw).split(".")[0], fmt)
        dt_to   = (
            datetime.strptime(str(to_raw).split(".")[0], fmt)
            if to_raw
            else datetime.now()
        )
        return max(0, (dt_to - dt_from).days)
    except Exception:
        return None


# =============================================================================
# RECORD EXTRACTOR  (handles any .NET wrapper shape)
# =============================================================================

def _extract_records(data: Any) -> List[Dict]:
    """
    Pull the list of records out of whatever shape .NET returned.
    Handles: bare list, dict with data/Data/result/Result/records/persons wrapper.
    """
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in (
            "data", "Data", "result", "Result",
            "records", "Records", "person", "persons", "personnel",
        ):
            val = data.get(key)
            if isinstance(val, list):
                return val
    return []


def _unwrap_data(data: Any) -> Any:
    """
    Unwrap a single-dict wrapper to get the inner data object.
    Returns the inner dict/list or the original if no wrapper found.
    """
    if isinstance(data, dict):
        for key in ("data", "Data", "result", "Result"):
            val = data.get(key)
            if val is not None:
                return val
    return data


# =============================================================================
# PERFORMANCE — NESTED STRUCTURE HANDLERS
# =============================================================================

def _format_single_agniveer(agniveer: Dict, rank: int, intent_result: Dict) -> str:
    """Format one agniveer performance record into clean readable plain text."""
    name        = _safe_str(_get(agniveer, "fullName", "name", "Name"))
    agniveer_no = _safe_str(_get(agniveer, "agniveerNo"), "")
    batch       = _safe_str(_get(agniveer, "batchName"), "")
    platoon     = _safe_str(_get(agniveer, "platoonName"), "")
    cls         = _safe_str(_get(agniveer, "class"), "")
    best_total  = _get(agniveer, "bestTotal", "totalMarks", "score", "Score")
    stored_rank = _get(agniveer, "rank")
    attempts    = agniveer.get("attempts") or []
    exceptional = agniveer.get("exceptionalSections") or []

    display_rank = stored_rank if stored_rank is not None else rank
    rank_label   = _rank_label(int(display_rank))

    lines = [f"{rank_label}. {name}"]

    meta_parts = []
    if agniveer_no: meta_parts.append(f"No: {agniveer_no}")
    if batch:       meta_parts.append(f"Batch: {batch}")
    if platoon:     meta_parts.append(f"Platoon: {platoon}")
    if cls:         meta_parts.append(f"Class: {cls}")
    if meta_parts:
        lines.append("  " + "  |  ".join(meta_parts))

    if best_total is not None:
        lines.append(f"  Best Total: {best_total}")

    section_filter = (intent_result.get("section") or "").upper()
    attempt_filter = intent_result.get("attempt_no")

    for attempt in attempts:
        attempt_no = attempt.get("attemptNo", "?")
        if attempt_filter is not None and str(attempt_no) != str(attempt_filter):
            continue

        sections = attempt.get("sections") or []
        if section_filter:
            sections = [
                s for s in sections
                if (s.get("sectionName") or "").upper() == section_filter
            ]

        has_data = any(
            s.get("omrInputTotal") is not None and s.get("omrInputTotal", 0) > 0
            for s in sections
        )
        if not sections or not has_data:
            continue

        lines.append(f"  Attempt {attempt_no}:")

        for section in sorted(sections, key=lambda s: s.get("displayOrder", 99)):
            s_name  = _safe_str(section.get("sectionName"))
            s_total = section.get("omrInputTotal")
            s_grade = _safe_str(section.get("grading"), "")

            if s_total is None or s_total == 0:
                continue

            grade_str = f"  ({s_grade})" if s_grade else ""
            lines.append(f"    {s_name}: {s_total} pts{grade_str}")

            sub_items = section.get("subItems") or []
            best_sub  = [si for si in sub_items if si.get("isBestAttempt") is True]
            sub_parts = []
            for si in best_sub:
                si_name  = _safe_str(si.get("subItemName"))
                obtained = si.get("marksObtained")
                max_m    = si.get("maxMarks")
                if obtained is not None and max_m is not None:
                    sub_parts.append(f"{si_name}: {obtained}/{max_m}")
            if sub_parts:
                lines.append("      " + ", ".join(sub_parts))

    if exceptional:
        exc_parts = []
        for exc in exceptional:
            exc_name  = _safe_str(exc.get("sectionName"))
            exc_marks = exc.get("marksObtained")
            if exc_marks is not None:
                exc_parts.append(f"{exc_name}: {exc_marks:.2f}")
        if exc_parts:
            lines.append(f"  Exceptional: {', '.join(exc_parts)}")

    return "\n".join(lines)


def _format_performance_list(data: Any, intent_result: Dict) -> str:
    """Handle the real .NET performance list response.

    Supports multiple .NET response shapes:
      1. List of agniveer records (TopPerformers, LowestPerformers, etc.)
      2. Dict with nested lists like "sections", "distribution" (Comparison, BestAttempt)
      3. Dict with flat aggregate fields (AverageScore, etc.)
    """
    command_label = ""
    message_text = ""

    if isinstance(data, dict):
        command_label = data.get("commandLabel", "")
        message_text = _safe_str(data.get("message"), "")
        inner = data.get("data") or data.get("Data") or data.get("result") or data.get("Result") or {}
    elif isinstance(data, list):
        inner = data
    else:
        return "Unexpected data format."

    subcategory    = intent_result.get("subcategory", "")
    section_filter = intent_result.get("section", "")

    label_map = {
        "TopPerformers":      "Top Performers",
        "LowestPerformers":   "Lowest Performers",
        "OverallPerformance": "Overall Performance",
        "Improvement":        "Improvement",
        "Drop":               "Score Drop",
        "SectionSummary":     "Section Summary",
        "AttemptWise":        "Attempt-wise Analysis",
        "BestAttempt":        "Best Attempt",
        "Comparison":         "Comparison",
    }
    label   = command_label or label_map.get(subcategory, subcategory or "Performance Data")
    sec_str = f" - {section_filter}" if section_filter else ""

    # ── Case 1: inner data is a list of records ──────────────────────────────
    if isinstance(inner, list):
        records = inner
        if not records:
            return "No records found for this query."

        # Check if records look like agniveer dicts (have fullName/agniveerNo/attempts)
        first = records[0] if records else {}
        is_agniveer = isinstance(first, dict) and any(
            k in first for k in ("fullName", "agniveerNo", "attempts", "bestTotal", "name")
        )

        if is_agniveer:
            count = len(records)
            lines = [
                f"{label}{sec_str}",
                f"({count} record{'s' if count != 1 else ''})",
                "",
            ]
            for i, record in enumerate(records, start=1):
                lines.append(_format_single_agniveer(record, i, intent_result))
                if i < len(records):
                    lines.append("")
                    lines.append("-" * 40)
                    lines.append("")
            if message_text:
                lines.append("")
                lines.append(message_text)
            return "\n".join(lines).strip()

        # Records are aggregate/summary dicts (sections, distribution entries, etc.)
        # Auto-detect and format as a table
        if records and isinstance(first, dict):
            keys = list(first.keys())
            # Build human-readable headers from camelCase keys
            headers = [_camel_to_words(k) for k in keys]
            rows = []
            for item in records:
                row = []
                for k in keys:
                    val = item.get(k)
                    if isinstance(val, float):
                        row.append(f"{val:.2f}")
                    elif val is not None:
                        row.append(str(val))
                    else:
                        row.append("-")
                rows.append(row)

            lines = [f"{label}{sec_str}", f"({len(records)} record{'s' if len(records) != 1 else ''})", ""]
            lines.append(_plain_table(headers, rows))
            if message_text:
                lines.append("")
                lines.append(message_text)
            return "\n".join(lines).strip()

        # Fallback for list of non-dict items
        lines = [f"{label}{sec_str}", ""]
        for i, item in enumerate(records, 1):
            lines.append(f"{i}. {item}")
        return "\n".join(lines).strip()

    # ── Case 2: inner data is a dict with nested lists ───────────────────────
    if isinstance(inner, dict):
        # Look for any nested list inside the dict (e.g., "sections", "distribution", "grades")
        nested_lists = {k: v for k, v in inner.items() if isinstance(v, list) and v}
        flat_fields  = {k: v for k, v in inner.items() if not isinstance(v, (list, dict))}

        lines = [f"{label}{sec_str}", ""]

        # Show flat summary fields first
        if flat_fields:
            pairs = [(_camel_to_words(k), v) for k, v in flat_fields.items()]
            lines.append(_kv_block(pairs))
            lines.append("")

        # Format each nested list as a table
        for list_key, list_items in nested_lists.items():
            if not list_items or not isinstance(list_items[0], dict):
                continue
            list_label = _camel_to_words(list_key)
            keys = list(list_items[0].keys())
            headers = [_camel_to_words(k) for k in keys]
            rows = []
            for item in list_items:
                row = []
                for k in keys:
                    val = item.get(k)
                    if isinstance(val, float):
                        row.append(f"{val:.2f}")
                    elif val is not None:
                        row.append(str(val))
                    else:
                        row.append("-")
                rows.append(row)
            lines.append(f"{list_label}:")
            lines.append(_plain_table(headers, rows))
            lines.append("")

        if message_text:
            lines.append(message_text)

        result = "\n".join(lines).strip()
        return result if result != f"{label}{sec_str}" else "No records found for this query."

    if not inner:
        return "No records found for this query."

    return str(inner)


# =============================================================================
# LEAVE
# =============================================================================

def _format_leave(subcategory: str, data: Any, intent_result: Dict) -> str:
    """
    Format .NET leave responses.

    Real record shape:
        {
          "agniveerId":  1210,
          "agniveerNo":  "A0701532AA",
          "fullName":    "AMIT KUMAR",
          "photoPath":   null,
          "fromDate":    "2026-06-24T06:30:00",
          "toDate":      null,
          "remarks":     ""
        }

    remarks may be empty — toDate null means still on leave / absconded.
    Days are auto-computed from fromDate → toDate (today if toDate is null).
    """
    leave_type = intent_result.get("leave_type", "")
    lt_str     = f" ({leave_type})" if leave_type else ""

    def _leave_rows(records: List[Dict], numbered: bool = True) -> List[List]:
        rows = []
        for i, item in enumerate(records, 1):
            name        = _safe_str(_get(item, "fullName", "name", "Name"), f"Person {i}")
            agniveer_no = _safe_str(_get(item, "agniveerNo", "AgniveerNo"), "-")
            from_date   = _fmt_datetime(_get(item, "fromDate", "from", "startDate"))
            to_raw      = _get(item, "toDate", "to", "endDate")
            to_date     = _fmt_datetime(to_raw) if to_raw else "Ongoing"
            remarks     = _safe_str(_get(item, "remarks", "leaveType", "type"), "-")
            days        = _calc_days(item)
            days_str    = str(days) if days is not None else "-"
            prefix      = f"{i}. " if numbered else ""
            rows.append([f"{prefix}{name}", agniveer_no, from_date, to_date, days_str, remarks])
        return rows

    # ── MostLeaveTaken ────────────────────────────────────────────────────────
    if subcategory == "MostLeaveTaken":
        records = _extract_records(data)
        if not records:
            return "No leave data found."
        lines = [
            f"Most Leave Taken{lt_str}",
            f"({len(records)} record{'s' if len(records) != 1 else ''})",
            "",
        ]
        lines.append(_plain_table(
            ["Name", "No.", "From", "To", "Days", "Remarks"],
            _leave_rows(records),
        ))
        return "\n".join(lines)

    # ── LeastLeaveTaken ───────────────────────────────────────────────────────
    if subcategory == "LeastLeaveTaken":
        records = _extract_records(data)
        if not records:
            return "No leave data found."
        lines = [
            f"Least Leave Taken{lt_str}",
            f"({len(records)} record{'s' if len(records) != 1 else ''})",
            "",
        ]
        lines.append(_plain_table(
            ["Name", "No.", "From", "To", "Days", "Remarks"],
            _leave_rows(records),
        ))
        return "\n".join(lines)

    # ── CurrentLeaveStatus ────────────────────────────────────────────────────
    if subcategory == "CurrentLeaveStatus":
        records = _extract_records(data)
        if not records:
            return "No person is currently on leave."
        label = (
            f"Currently On Leave "
            f"({len(records)} {'person' if len(records) == 1 else 'persons'})"
        )
        lines = [label, ""]
        lines.append(_plain_table(
            ["Name", "No.", "From", "To", "Days", "Remarks"],
            _leave_rows(records, numbered=False),
        ))
        return "\n".join(lines)

    # ── AbscondedPerson ───────────────────────────────────────────────────────
    if subcategory == "AbscondedPerson":
        records = _extract_records(data)
        if not records:
            return "No absconded person on record."

        rows = []
        for item in records:
            name        = _safe_str(_get(item, "fullName", "name", "Name"), "Unknown")
            agniveer_no = _safe_str(_get(item, "agniveerNo", "AgniveerNo"), "-")
            since       = _fmt_datetime(
                _get(item, "fromDate", "since", "date", "abscondedDate")
            )
            to_raw      = _get(item, "toDate", "to")
            returned    = _fmt_date(to_raw) if to_raw else "Still Absconded"
            days        = _calc_days(item)
            days_str    = str(days) if days is not None else "-"
            remarks     = _safe_str(_get(item, "remarks"), "-")
            rows.append([name, agniveer_no, since, returned, days_str, remarks])

        label = (
            f"Absconded {'Person' if len(records) == 1 else 'Persons'} ({len(records)})"
        )
        lines = [label, ""]
        lines.append(_plain_table(
            ["Name", "No.", "Since", "Returned", "Days", "Remarks"],
            rows,
        ))
        return "\n".join(lines)

    # ── Generic fallback for any unrecognised leave subcategory ───────────────
    records = _extract_records(data)
    if records:
        lines = [
            f"Leave Data{lt_str}",
            f"({len(records)} record{'s' if len(records) != 1 else ''})",
            "",
        ]
        lines.append(_plain_table(
            ["Name", "No.", "From", "To", "Days", "Remarks"],
            _leave_rows(records),
        ))
        return "\n".join(lines)

    if isinstance(data, list):
        rows = [
            [i, _safe_str(_get(item, "fullName", "name", "Name"))]
            for i, item in enumerate(data, 1)
        ]
        return "Leave Data\n\n" + _plain_table(["#", "Name"], rows)

    return str(data)


# =============================================================================
# MEDICAL
# =============================================================================

def _format_medical(subcategory: str, data: Any, intent_result: Dict) -> str:

    if subcategory == "ActiveCases":
        records = _extract_records(data)
        total_from_dict = _get(data, "total", "count") if isinstance(data, dict) else None
        if total_from_dict is not None and not records:
            return f"There are currently {total_from_dict} active medical case(s)."
        if not records:
            return "No active medical cases at the moment."

        rows = [
            [
                _safe_str(_get(item, "fullName", "name", "Name"), "Unknown"),
                _safe_str(_get(item, "disease", "diagnosis", "condition"), "Unknown"),
                _safe_str(_get(item, "ward"), "-"),
            ]
            for item in records
        ]
        lines = [f"Active Medical Cases ({len(records)})", ""]
        lines.append(_plain_table(["Name", "Diagnosis", "Ward"], rows))
        return "\n".join(lines)

    if subcategory == "BMIAnalysis":
        records = _extract_records(data) if not isinstance(data, list) else data
        if not records:
            return "No BMI data found."

        rows = []
        for item in records:
            # real fields: fullName, agniveerNo, heightCm, weightKg, bmiValue, bmiCategory
            name        = _safe_str(_get(item, "fullName", "name", "Name"), "Unknown")
            agniveer_no = _safe_str(_get(item, "agniveerNo", "AgniveerNo"), "-")
            height      = _get(item, "heightCm", "height", "Height")
            weight      = _get(item, "weightKg", "weight", "Weight")
            bmi_val     = _get(item, "bmiValue", "bmi", "BMI")
            bmi_cat     = _safe_str(_get(item, "bmiCategory", "category", "bmiCategory"), "-")

            height_str  = f"{height} cm" if height is not None else "-"
            weight_str  = f"{weight} kg" if weight is not None else "-"
            bmi_str     = f"{bmi_val:.2f}" if isinstance(bmi_val, (int, float)) else _safe_str(bmi_val, "-")

            rows.append([name, agniveer_no, height_str, weight_str, bmi_str, bmi_cat.title()])

        lines = [f"BMI Analysis ({len(records)} record{'s' if len(records) != 1 else ''})", ""]
        lines.append(_plain_table(
            ["Name", "No.", "Height", "Weight", "BMI", "Category"],
            rows,
        ))
        return "\n".join(lines)

    if subcategory == "DiseaseStatistics":
        records = _extract_records(data) if not isinstance(data, list) else data
        if not records:
            return "No disease statistics available."
        rows = [
            [
                i,
                _safe_str(_get(item, "disease", "name", "condition", "diagnosis")),
                _safe_str(_get(item, "count", "cases", "total"), "-"),
            ]
            for i, item in enumerate(records, 1)
        ]
        return "Disease Statistics\n\n" + _plain_table(["#", "Disease", "Cases"], rows)

    if isinstance(data, dict):
        pairs = [
            (_camel_to_words(k), v)
            for k, v in data.items()
            if not isinstance(v, (list, dict))
        ]
        return "Medical Summary\n\n" + _kv_block(pairs)
    return str(data)


# =============================================================================
# ATTENDANCE
# =============================================================================

def _format_attendance(subcategory: str, data: Any, intent_result: Dict) -> str:

    if subcategory == "PresentToday":
        inner = _unwrap_data(data) if isinstance(data, dict) else data
        if isinstance(inner, dict):
            present   = _get(inner, "present", "Present", "count", "Count", "presentToday") or 0
            total     = _get(inner, "total", "Total", "totalAgniveers")
            total_str = f" out of {total}" if total else ""
            return (
                f"{present} {'person is' if present == 1 else 'persons are'} "
                f"present on campus today{total_str}."
            )
        if isinstance(inner, (int, float)):
            return (
                f"{inner} {'person is' if inner == 1 else 'persons are'} "
                f"present on campus today."
            )

    if subcategory == "MonthlyAttendance":
        records = _extract_records(data) if not isinstance(data, list) else data
        if records:
            rows = [
                [
                    _safe_str(_get(item, "month", "Month")),
                    (
                        f"{_get(item, 'percentage', 'attendancePercentage', 'Percentage')}%"
                        if _get(item, "percentage", "attendancePercentage", "Percentage") is not None
                        else "-"
                    ),
                    _safe_str(_get(item, "present", "Present"), "-"),
                ]
                for item in records
            ]
            return "Monthly Attendance\n\n" + _plain_table(
                ["Month", "Attendance %", "Present"], rows
            )
        inner = _unwrap_data(data) if isinstance(data, dict) else data
        if isinstance(inner, dict):
            pairs = [
                (_camel_to_words(k), v)
                for k, v in inner.items()
                if not isinstance(v, (list, dict))
            ]
            return "Monthly Attendance Summary\n\n" + _kv_block(pairs)

    if subcategory == "StrengthBreakdown":
        # Real shape: { "success": true, "data": { "totalAgniveers": 610,
        #   "activeCount": 608, "inactiveCount": 2, "presentToday": 0,
        #   "absentToday": 0, "onLeave": 10,
        #   "byPlatoon": [ { "platoonName": "PL-01", "count": 30 }, ... ] } }
        inner = _unwrap_data(data) if isinstance(data, dict) else data
        if isinstance(inner, dict):
            summary_pairs = []
            for k, v in inner.items():
                if isinstance(v, (list, dict)):
                    continue
                label = {
                    "totalAgniveers": "Total Agniveers",
                    "activeCount":    "Active",
                    "inactiveCount":  "Inactive",
                    "presentToday":   "Present Today",
                    "absentToday":    "Absent Today",
                    "onLeave":        "On Leave",
                }.get(k, _camel_to_words(k))
                summary_pairs.append((label, v))

            by_platoon = inner.get("byPlatoon") or []

            lines = ["Strength Breakdown", ""]
            if summary_pairs:
                lines.append(_kv_block(summary_pairs))
            if by_platoon:
                lines.append("")
                lines.append("Platoon-wise Strength:")
                rows = [
                    [
                        _safe_str(_get(p, "platoonName", "platoon", "name")),
                        str(_get(p, "count", "strength", "total") or 0),
                    ]
                    for p in by_platoon
                ]
                lines.append(_plain_table(["Platoon", "Count"], rows))
            return "\n".join(lines)

    if isinstance(data, dict):
        inner = _unwrap_data(data)
        if isinstance(inner, dict):
            pairs = [
                (_camel_to_words(k), v)
                for k, v in inner.items()
                if not isinstance(v, (list, dict))
            ]
            return "Attendance Summary\n\n" + _kv_block(pairs)
    return str(data)


# =============================================================================
# VERIFICATION
# =============================================================================

def _format_verification(subcategory: str, data: Any, intent_result: Dict) -> str:
    """
    Real shapes:

    Pending:
      [ { "verificationId": 10, "agniveerId": 614, "agniveerNo": "A0701528W",
          "fullName": "LAVEPREET", "photoPath": null,
          "policeStation": "HAJIPUR", "sentDate": "2026-06-01T11:48:02.39" } ]

    Completed:
      [ { "verificationId": 36, "agniveerId": 822, "agniveerNo": "A0701778X",
          "fullName": "Keshav Rana", "photoPath": null,
          "policeStation": "BHAWARNA",
          "sentDate": "2026-06-05T11:48:02.39",
          "receivedDate": "2026-06-09T23:48:02.39", "status": "Verified" } ]
    """
    is_pending  = subcategory == "PendingVerification"
    label       = "Pending" if is_pending else "Completed"
    records     = _extract_records(data)

    total_from_dict = _get(data, "total", "count") if isinstance(data, dict) else None
    if total_from_dict is not None and not records:
        return f"{total_from_dict} {label.lower()} verification(s) found."
    if not records:
        return f"No {label.lower()} verifications found."

    if is_pending:
        # Columns: Name | No. | Police Station | Sent Date | Days Pending
        rows = []
        for item in records:
            name     = _safe_str(_get(item, "fullName", "name", "Name"), "Unknown")
            no_      = _safe_str(_get(item, "agniveerNo", "AgniveerNo"), "-")
            station  = _safe_str(_get(item, "policeStation", "station"), "-")
            sent     = _fmt_datetime(_get(item, "sentDate", "sent", "date"))
            # days pending = sentDate → today
            days     = _calc_days({
                "fromDate": _get(item, "sentDate", "sent"),
                "toDate":   None,
            })
            days_str = str(days) if days is not None else "-"
            rows.append([name, no_, station, sent, days_str])

        lines = [
            f"Pending Verifications ({len(records)})",
            "",
        ]
        lines.append(_plain_table(
            ["Name", "No.", "Police Station", "Sent Date", "Days Pending"],
            rows,
        ))
        return "\n".join(lines)

    else:
        # Columns: Name | No. | Police Station | Sent | Received | Status
        rows = []
        for item in records:
            name     = _safe_str(_get(item, "fullName", "name", "Name"), "Unknown")
            no_      = _safe_str(_get(item, "agniveerNo", "AgniveerNo"), "-")
            station  = _safe_str(_get(item, "policeStation", "station"), "-")
            sent     = _fmt_datetime(_get(item, "sentDate", "sent"))
            received = _fmt_datetime(_get(item, "receivedDate", "received"))
            status   = _safe_str(_get(item, "status", "Status"), "Completed")
            rows.append([name, no_, station, sent, received, status])

        lines = [
            f"Completed Verifications ({len(records)})",
            "",
        ]
        lines.append(_plain_table(
            ["Name", "No.", "Police Station", "Sent", "Received", "Status"],
            rows,
        ))
        return "\n".join(lines)


# =============================================================================
# EQUIPMENT
# =============================================================================

def _format_equipment(subcategory: str, data: Any, intent_result: Dict) -> str:
    if subcategory in ("IssuedItems", "ProcuredItems"):
        return _format_item_list(subcategory, data, intent_result)

    if subcategory == "EquipmentSummary":
        # Real shape: { "success": true, "commandLabel": "Equipment Stats",
        #   "data": { "totalAssigned": 505, "active": 444,
        #             "returned": 61, "poorCondition": 0 } }
        inner = _unwrap_data(data) if isinstance(data, dict) else data
        if isinstance(inner, dict):
            label_map = {
                "totalAssigned": "Total Assigned",
                "active":        "Active",
                "returned":      "Returned",
                "poorCondition": "Poor Condition",
            }
            pairs = []
            for k, v in inner.items():
                if isinstance(v, (list, dict)):
                    continue
                pairs.append((label_map.get(k, _camel_to_words(k)), v))
            return "Equipment Summary\n\n" + _kv_block(pairs)

    if subcategory == "OverdueEquipment":
        records = _extract_records(data) if not isinstance(data, list) else data
        if not records:
            return "No overdue equipment records."
        rows = [
            [
                _safe_str(_get(item, "name", "equipment", "itemName")),
                _safe_str(_get(item, "issuedTo", "holder"), "-"),
                (
                    f"{_get(item, 'overdueDays', 'daysOverdue')} days"
                    if _get(item, "overdueDays", "daysOverdue")
                    else "-"
                ),
            ]
            for item in records
        ]
        return (
            f"Overdue Equipment ({len(records)} items)\n\n"
            + _plain_table(["Equipment", "Issued To", "Overdue By"], rows)
        )

    if subcategory == "PoorConditionEquipment":
        records = _extract_records(data) if not isinstance(data, list) else data
        if not records:
            return "No equipment returned in poor condition."
        rows = [
            [
                _safe_str(_get(item, "name", "equipment", "itemName")),
                _safe_str(_get(item, "condition", "state"), "Poor"),
            ]
            for item in records
        ]
        return (
            f"Poor Condition Equipment ({len(records)} items)\n\n"
            + _plain_table(["Equipment", "Condition"], rows)
        )

    if isinstance(data, dict):
        inner = _unwrap_data(data)
        if isinstance(inner, dict):
            pairs = [
                (_camel_to_words(k), v)
                for k, v in inner.items()
                if not isinstance(v, (list, dict))
            ]
            return "Equipment Data\n\n" + _kv_block(pairs)

    if isinstance(data, list):
        rows = [
            [i, _safe_str(_get(item, "name", "equipment", "itemName"))]
            for i, item in enumerate(data, 1)
        ]
        return "Equipment Records\n\n" + _plain_table(["#", "Item"], rows)

    return str(data)


def _format_item_list(subcategory: str, data: Any, intent_result: Dict) -> str:
    from admin_intent import ISSUED_ITEMS, PROCURED_ITEMS

    is_issued   = subcategory == "IssuedItems"
    label       = "Issued Items" if is_issued else "Procured Items"
    master_list = ISSUED_ITEMS if is_issued else PROCURED_ITEMS
    item_name   = intent_result.get("item_name")

    if item_name:
        in_issued   = item_name in ISSUED_ITEMS
        in_procured = item_name in PROCURED_ITEMS
        category_tag = (
            "Issued" if in_issued
            else ("Procured" if in_procured else "Unknown")
        )
        item_detail = ""
        if isinstance(data, dict):
            item_detail = _safe_str(
                data.get("detail") or data.get("description") or data.get("status"), ""
            )
        elif isinstance(data, list):
            for entry in data:
                if isinstance(entry, dict):
                    name_val = _safe_str(_get(entry, "name", "itemName", "item"))
                    if name_val.lower() == item_name.lower():
                        item_detail = _safe_str(
                            entry.get("detail") or entry.get("status") or
                            entry.get("description"), ""
                        )
                        break

        pairs = [("Category", category_tag)]
        if item_detail:
            pairs.append(("Detail", item_detail))
        return f"Item Lookup - {item_name}\n\n" + _kv_block(pairs)

    records: list = []
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        records = _get(data, "data", "items", "Data", "Items") or []

    items_to_render = records if records else master_list

    rows = []
    for i, entry in enumerate(items_to_render, 1):
        if isinstance(entry, str):
            rows.append([i, entry, "-", "-"])
        elif isinstance(entry, dict):
            rows.append([
                i,
                _safe_str(_get(entry, "name", "itemName", "item", "Name")),
                _safe_str(_get(entry, "status", "Status"), "-"),
                _safe_str(_get(entry, "quantity", "qty", "Quantity"), "-"),
            ])
        else:
            rows.append([i, _safe_str(entry), "-", "-"])

    note = "\n\n(Source: master list)" if not records else ""
    return (
        f"{label} ({len(items_to_render)} items)\n\n"
        + _plain_table(["#", "Item Name", "Status", "Qty"], rows)
        + note
    )


# =============================================================================
# DISTRIBUTION
# =============================================================================

def _format_distribution(subcategory: str, data: Any, intent_result: Dict) -> str:
    """
    Real shapes:

    LatestDistribution:
      { "success": true, "commandLabel": "Latest Distribution",
        "data": { "distributionId": 5, "distributionDate": "2026-06-12T...",
                  "teams": [ { "teamId": 4, "teamName": "14 Punjab",
                               "memberCount": 21,
                               "members": [ { "agniveerId": 639,
                                              "agniveerNo": "A0701560L",
                                              "fullName": "PARAMJEET SINGH",
                                              "class": "SIKH", "rank": 1 } ] } ] } }

    DistributionByUnit:
      [ { "agniveerId": 946, "agniveerNo": "A0701915L", "fullName": "TARUN KUMAR",
          "photoPath": null, "class": "DOGRA", "rank": 420,
          "distributionEventId": 6, "distributionDate": "2026-06-15T..." } ]

    UnassignedItems:
      [ { "agniveerId": 605, "agniveerNo": "A0701515Y",
          "fullName": "SHAVAIT CHOUDHARY", "photoPath": "...", "platoonName": "PL-01" } ]

    TopUnit:
      { "success": true, "commandLabel": "Top Unit",
        "data": { "distributionEventId": 6, "distributionDate": "2026-06-15T...",
                  "teamId": 4, "teamName": "14 Punjab", "agniveerCount": 10 } }
    """

    if subcategory == "LatestDistribution":
        inner = _unwrap_data(data) if isinstance(data, dict) else data
        if isinstance(inner, dict):
            dist_id   = _get(inner, "distributionId", "id")
            dist_date = _fmt_date(_get(inner, "distributionDate", "date"))
            teams     = inner.get("teams") or []

            lines = ["Latest Distribution"]
            meta = []
            if dist_id:   meta.append(f"Distribution ID: {dist_id}")
            if dist_date != "-": meta.append(f"Date: {dist_date}")
            if meta:
                lines.append("  ".join(meta))
            lines.append("")

            if teams:
                for team in teams:
                    team_name    = _safe_str(_get(team, "teamName", "name"), "Unknown Team")
                    member_count = _get(team, "memberCount", "count") or 0
                    members      = team.get("members") or []
                    lines.append(f"Team: {team_name}  ({member_count} members)")
                    if members:
                        rows = []
                        for m in members:
                            rank    = _get(m, "rank") or "-"
                            name    = _safe_str(_get(m, "fullName", "name"), "Unknown")
                            no_     = _safe_str(_get(m, "agniveerNo"), "-")
                            cls     = _safe_str(_get(m, "class"), "-")
                            rows.append([str(rank), name, no_, cls])
                        lines.append(_plain_table(["Rank", "Name", "No.", "Class"], rows))
                    lines.append("")
            else:
                # no teams — show flat key-value
                pairs = [
                    (_camel_to_words(k), v)
                    for k, v in inner.items()
                    if not isinstance(v, (list, dict))
                ]
                lines.append(_kv_block(pairs))

            return "\n".join(lines).strip()

        if isinstance(data, list):
            rows = [
                [i, _safe_str(_get(item, "fullName", "name", "Name"))]
                for i, item in enumerate(data, 1)
            ]
            return "Latest Distribution\n\n" + _plain_table(["#", "Name"], rows)

    if subcategory == "DistributionByUnit":
        records = _extract_records(data) if not isinstance(data, list) else data
        if not records:
            return "No unit distribution data."

        # Records contain individual agniveer entries with class, rank, distributionDate
        rows = []
        for item in records:
            name    = _safe_str(_get(item, "fullName", "name", "Name"), "Unknown")
            no_     = _safe_str(_get(item, "agniveerNo", "AgniveerNo"), "-")
            cls     = _safe_str(_get(item, "class"), "-")
            rank    = _get(item, "rank")
            rank_s  = str(rank) if rank is not None else "-"
            dist_dt = _fmt_date(_get(item, "distributionDate", "date"))
            rows.append([name, no_, cls, rank_s, dist_dt])

        lines = [
            f"Distribution by Unit ({len(records)} member{'s' if len(records) != 1 else ''})",
            "",
        ]
        lines.append(_plain_table(
            ["Name", "No.", "Class", "Rank", "Distribution Date"],
            rows,
        ))
        return "\n".join(lines)

    if subcategory == "UnassignedItems":
        records = _extract_records(data) if not isinstance(data, list) else data
        if not records:
            return "All Agniveers have been assigned to a unit."

        rows = []
        for item in records:
            name    = _safe_str(_get(item, "fullName", "name", "Name"), "Unknown")
            no_     = _safe_str(_get(item, "agniveerNo", "AgniveerNo"), "-")
            platoon = _safe_str(_get(item, "platoonName", "platoon"), "-")
            rows.append([name, no_, platoon])

        lines = [
            f"Unassigned Agniveers ({len(records)})",
            "",
        ]
        lines.append(_plain_table(["Name", "No.", "Platoon"], rows))
        return "\n".join(lines)

    if subcategory == "TopUnit":
        # Real shape: { "success": true, "data": { "teamName": "14 Punjab",
        #   "agniveerCount": 10, "distributionDate": "...", ... } }
        inner = _unwrap_data(data) if isinstance(data, dict) else data
        if isinstance(inner, dict):
            team_name  = _safe_str(_get(inner, "teamName", "unit", "unitName"), "Unknown")
            count      = _get(inner, "agniveerCount", "count", "memberCount")
            dist_date  = _fmt_date(_get(inner, "distributionDate", "date"))
            lines = ["Top Unit"]
            if dist_date != "-":
                lines.append(f"Distribution Date: {dist_date}")
            lines.append("")
            lines.append(f"Unit: {team_name}")
            if count is not None:
                lines.append(f"Agniveer Count: {count}")
            return "\n".join(lines)
        return f"Top unit data: {data}"

    if isinstance(data, dict):
        inner = _unwrap_data(data)
        pairs = [
            (_camel_to_words(k), v)
            for k, v in (inner if isinstance(inner, dict) else {}).items()
            if not isinstance(v, (list, dict))
        ]
        return "Distribution Summary\n\n" + _kv_block(pairs)

    return str(data)


# =============================================================================
# SKILLS
# =============================================================================

def _format_skills(subcategory: str, data: Any, intent_result: Dict) -> str:
    """
    Real shapes:

    BySport:
      [ { "agniveerId": 605, "agniveerNo": "A0701515Y",
          "fullName": "SHAVAIT CHOUDHARY", "photoPath": "...",
          "sports": "Cricket, Kabaddi", "platoonName": "PL-01" } ]

    ByClass:
      [ { "agniveerId": 680, "agniveerNo": "A0701619A",
          "fullName": "SULINDER SINGH", "photoPath": null,
          "class": "SIKH", "platoonName": "PL-01", "bloodGroup": "AB-" } ]

    BloodGroup:
      [ { "bloodGroup": "AB-", "count": 12 } ]   (aggregate)
      OR individual records with "bloodGroup" field
    """

    if subcategory == "BySport":
        records = _extract_records(data) if not isinstance(data, list) else data
        if not records:
            return "No sport data found."

        # Detect if records are individual person records or aggregates
        is_aggregate = all(
            "fullName" not in item and ("sport" in item or "count" in item)
            for item in records
        )

        if is_aggregate:
            rows = [
                [
                    _safe_str(_get(item, "sport", "name", "Sport")),
                    _safe_str(_get(item, "count", "person", "total"), "-"),
                ]
                for item in records
            ]
            return "Roster by Sport\n\n" + _plain_table(["Sport", "Count"], rows)

        # Individual person records with "sports" field (comma-separated string)
        rows = []
        for item in records:
            name    = _safe_str(_get(item, "fullName", "name", "Name"), "Unknown")
            no_     = _safe_str(_get(item, "agniveerNo", "AgniveerNo"), "-")
            sports  = _safe_str(_get(item, "sports", "sport", "Sport"), "-")
            platoon = _safe_str(_get(item, "platoonName", "platoon"), "-")
            rows.append([name, no_, sports, platoon])

        lines = [
            f"Sport Roster ({len(records)} record{'s' if len(records) != 1 else ''})",
            "",
        ]
        lines.append(_plain_table(["Name", "No.", "Sports", "Platoon"], rows))
        return "\n".join(lines)

    if subcategory == "ByClass":
        records = _extract_records(data) if not isinstance(data, list) else data
        if not records:
            return "No class data found."

        is_aggregate = all(
            "fullName" not in item and ("class" in item or "count" in item)
            for item in records
        )

        if is_aggregate:
            rows = [
                [
                    _safe_str(_get(item, "class", "className", "name", "Class")),
                    _safe_str(_get(item, "count", "person", "total"), "-"),
                ]
                for item in records
            ]
            return "Roster by Class\n\n" + _plain_table(["Class", "Count"], rows)

        # Individual person records
        rows = []
        for item in records:
            name       = _safe_str(_get(item, "fullName", "name", "Name"), "Unknown")
            no_        = _safe_str(_get(item, "agniveerNo", "AgniveerNo"), "-")
            cls        = _safe_str(_get(item, "class"), "-")
            platoon    = _safe_str(_get(item, "platoonName", "platoon"), "-")
            blood_grp  = _safe_str(_get(item, "bloodGroup", "blood"), "-")
            rows.append([name, no_, cls, platoon, blood_grp])

        lines = [
            f"Class Roster ({len(records)} record{'s' if len(records) != 1 else ''})",
            "",
        ]
        lines.append(_plain_table(["Name", "No.", "Class", "Platoon", "Blood Group"], rows))
        return "\n".join(lines)

    if subcategory == "BloodGroup":
        records = _extract_records(data) if not isinstance(data, list) else data
        if not records:
            return "No blood group data found."

        is_aggregate = all(
            "fullName" not in item
            for item in records
        )

        if is_aggregate:
            rows = [
                [
                    _safe_str(_get(item, "bloodGroup", "blood", "group")),
                    _safe_str(_get(item, "count", "total"), "-"),
                ]
                for item in records
            ]
            return "Blood Group Distribution\n\n" + _plain_table(["Blood Group", "Count"], rows)

        rows = [
            [
                _safe_str(_get(item, "fullName", "name", "Name"), "Unknown"),
                _safe_str(_get(item, "agniveerNo", "AgniveerNo"), "-"),
                _safe_str(_get(item, "bloodGroup", "blood"), "-"),
                _safe_str(_get(item, "platoonName", "platoon"), "-"),
            ]
            for item in records
        ]
        lines = [
            f"Blood Group Records ({len(records)} record{'s' if len(records) != 1 else ''})",
            "",
        ]
        lines.append(_plain_table(["Name", "No.", "Blood Group", "Platoon"], rows))
        return "\n".join(lines)

    if isinstance(data, list):
        rows = [
            [i, _safe_str(_get(item, "name", "Name", "fullName"))]
            for i, item in enumerate(data, 1)
        ]
        return "Skills / Roster\n\n" + _plain_table(["#", "Name"], rows)
    return str(data)


# =============================================================================
# PERFORMANCE — AGGREGATE
# =============================================================================

_PERFORMANCE_NESTED_SUBCATEGORIES = {
    "TopPerformers", "LowestPerformers", "OverallPerformance",
    "Improvement", "Drop", "AttemptWise", "BestAttempt", "Comparison",
    "SectionSummary",
}

_PERFORMANCE_AGGREGATE_SUBCATEGORIES = {
    "AverageScore", "PassPercentage", "FailPercentage",
    "GradeDistribution", "GradingSummary",
}


def _format_performance_aggregate(subcategory: str, data: Any, intent_result: Dict) -> str:
    section = intent_result.get("section", "")
    sec_str = f" for {section}" if section else ""

    if subcategory == "AverageScore":
        avg = (
            _get(data, "averageScore", "AverageScore", "average", "Average")
            if isinstance(data, dict)
            else data
        )
        return f"The average score{sec_str} is {avg}."

    if subcategory in ("PassPercentage", "FailPercentage"):
        label = "pass" if subcategory == "PassPercentage" else "fail"
        if isinstance(data, dict):
            pct       = _get(data, "percentage", "Percentage", "passPercentage", "failPercentage")
            total     = _get(data, "total", "Total")
            total_str = f" (out of {total} total)" if total else ""
            return f"The {label} percentage{sec_str} is {pct}%{total_str}."
        return f"{label.title()} percentage{sec_str}: {data}%"

    if subcategory in ("GradeDistribution", "GradingSummary"):
        title = f"Grade Distribution{' - ' + section if section else ''}"
        if isinstance(data, dict):
            rows = [
                [k, v]
                for k, v in data.items()
                if not isinstance(v, (list, dict))
            ]
            return f"{title}\n\n" + _plain_table(["Grade", "Count"], rows)
        if isinstance(data, list):
            rows = [
                [
                    _safe_str(_get(item, "grade", "Grade", "grading")),
                    _safe_str(_get(item, "count", "Count", "total"), "-"),
                ]
                for item in data
            ]
            return f"{title}\n\n" + _plain_table(["Grade", "Count"], rows)

    if isinstance(data, dict):
        pairs = [
            (_camel_to_words(k), v)
            for k, v in data.items()
            if not isinstance(v, (list, dict))
        ]
        return f"Performance Data{sec_str}\n\n" + _kv_block(pairs)
    return str(data)


# =============================================================================
# DISPATCH TABLE
# =============================================================================

_FORMATTERS = {
    "Leave":        _format_leave,
    "Medical":      _format_medical,
    "Attendance":   _format_attendance,
    "Verification": _format_verification,
    "Equipment":    _format_equipment,
    "Distribution": _format_distribution,
    "Skills":       _format_skills,
}

def _detect_side_label(side_data: Any) -> str:
    """
    Extract a human-readable label for a comparison side from its
    commandLabel or from the leaveTypeLabel of the first record.
    """
    if isinstance(side_data, dict):
        command_label = (side_data.get("commandLabel") or "").strip()
        if command_label:
            # e.g. "Top 10 Highest Leave — Medical" → use as-is
            return command_label
        # Try leaveTypeLabel from first record in data list
        records = side_data.get("data") or []
        if records and isinstance(records, list) and isinstance(records[0], dict):
            leave_type = records[0].get("leaveTypeLabel") or records[0].get("leaveType")
            if leave_type:
                return str(leave_type)
    return ""


def _format_leave_side_records(records: List[Dict], title: str) -> str:
    """Render leave records from one comparison side as a named table."""
    if not records:
        return f"{title}\\nNo records found."

    rows = []
    for i, item in enumerate(records, 1):
        name     = _safe_str(_get(item, "fullName", "name", "Name"), f"Person {i}")
        no_      = _safe_str(_get(item, "agniveerNo", "AgniveerNo"), "-")
        platoon  = _safe_str(_get(item, "platoonName", "platoon"), "-")
        from_dt  = _fmt_datetime(_get(item, "fromDate", "from"))
        to_dt    = _fmt_datetime(_get(item, "toDate", "to"))
        # Prefer the pre-computed totalDays / totalLeaveDays field
        days = _get(item, "totalDays", "totalLeaveDays")
        if days is not None:
            try:
                days_str = str(int(days))
            except (ValueError, TypeError):
                days_str = str(days)
        else:
            computed = _calc_days(item)
            days_str = str(computed) if computed is not None else "-"
        remarks = _safe_str(_get(item, "remarks"), "-")
        rows.append([f"{i}. {name}", no_, platoon, from_dt, to_dt, days_str, remarks])

    table = _plain_table(
        ["Name", "No.", "Platoon", "From", "To", "Days", "Remarks"],
        rows,
    )
    count = len(records)
    return (
        f"{title}\\n"
        f"({count} record{\'s\' if count != 1 else \'\'})\\n\\n"
        f"{table}"
    )


def _format_performance_side_records(records: List[Dict], title: str) -> str:
    """Render performance records from one comparison side as a named table."""
    if not records:
        return f"{title}\\nNo records found."

    rows = []
    for i, item in enumerate(records, 1):
        name    = _safe_str(_get(item, "fullName", "name", "Name"), f"Person {i}")
        no_     = _safe_str(_get(item, "agniveerNo", "AgniveerNo"), "-")
        score   = _get(item, "bestTotal", "totalMarks", "score", "Score")
        platoon = _safe_str(_get(item, "platoonName", "platoon"), "-")
        rows.append([
            f"{i}. {name}", no_, platoon,
            str(score) if score is not None else "-",
        ])

    table = _plain_table(["Name", "No.", "Platoon", "Score"], rows)
    count = len(records)
    return (
        f"{title}\\n"
        f"({count} record{\'s\' if count != 1 else \'\'})\\n\\n"
        f"{table}"
    )


def _format_generic_side_records(records: List[Dict], title: str) -> str:
    """
    Auto-detect columns from the first record and render a generic table.
    Skips internal/noisy fields that are not useful for display.
    """
    if not records:
        return f"{title}\\nNo records found."

    _SKIP_KEYS = {
        "photoPath", "agniveerId", "isAbscondedLeave", "isHospitalized",
        "onATTNC", "onAnnualLeave", "onEXPPG", "onMedicalLeave", "onSickLeave",
    }
    first = records[0]
    display_keys = [k for k in first.keys() if k not in _SKIP_KEYS][:7]

    def _key_to_header(k: str) -> str:
        import re as _re
        return _re.sub(r"(?<!^)(?=[A-Z])", " ", k).title()

    headers = [_key_to_header(k) for k in display_keys]

    rows = []
    for i, item in enumerate(records, 1):
        row = []
        for k in display_keys:
            val = item.get(k)
            if isinstance(val, bool):
                row.append("Yes" if val else "No")
            elif val is None:
                row.append("-")
            else:
                row.append(str(val))
        rows.append(row)

    table = _plain_table(headers, rows)
    count = len(records)
    return (
        f"{title}\\n"
        f"({count} record{\'s\' if count != 1 else \'\'})\\n\\n"
        f"{table}"
    )


def _format_comparison_result(data: Any, intent_result: Dict) -> str:
    """
    Format a comparison queryType result from result_combiner.compare_results().

    Expected shape:
        {
            "queryType": "comparison",
            "sides": [
                {
                    "label": "Leave",
                    "data": { "commandLabel": "...", "data": [...records...] },
                    "metrics": { "recordCount": N }
                },
                ...
            ],
            "comparedMetrics": [...]
        }

    Renders each side's actual records as a fully labelled table with a
    clear separator between sides — instead of a generic metrics-only table.
    """
    sides = data.get("sides") or []
    if not sides:
        return "No comparison data available."

    section_blocks: List[str] = []

    for side_idx, side in enumerate(sides):
        side_data  = side.get("data") or {}
        side_label = side.get("label", f"Side {side_idx + 1}")
        metrics    = side.get("metrics") or {}

        # Determine the display title for this side
        detected_label = _detect_side_label(side_data)
        title = detected_label if detected_label else side_label

        # Extract records
        records: List[Dict] = []
        if isinstance(side_data, dict):
            inner = side_data.get("data") or []
            if isinstance(inner, list):
                records = inner
        elif isinstance(side_data, list):
            records = side_data

        if not records:
            record_count = metrics.get("recordCount", 0)
            section_blocks.append(
                f"{title}\\n{record_count} record{\'s\' if record_count != 1 else \'\'} found."
            )
            continue

        first = records[0] if records else {}

        # Choose the right renderer based on record shape
        if isinstance(first, dict):
            is_leave = any(k in first for k in (
                "fromDate", "totalDays", "totalLeaveDays",
                "leaveTypeLabel", "onMedicalLeave", "onAnnualLeave",
            ))
            is_performance = any(k in first for k in (
                "bestTotal", "attempts", "score", "totalMarks", "omrInputTotal",
            ))

            if is_leave:
                block = _format_leave_side_records(records, title)
            elif is_performance:
                block = _format_performance_side_records(records, title)
            else:
                block = _format_generic_side_records(records, title)
        else:
            # List of non-dict items
            lines = [title, ""]
            for i, item in enumerate(records, 1):
                lines.append(f"{i}. {item}")
            block = "\\n".join(lines)

        section_blocks.append(block)

    # Join all sides with a clear horizontal separator
    separator = "\\n\\n" + ("\u2500" * 50) + "\\n\\n"
    return separator.join(section_blocks)

# =============================================================================
# MAIN PUBLIC ENTRY POINT
# =============================================================================
# Add this code block at the beginning of `format_dotnet_response` in `admin_formatter.py`

def format_dotnet_response(
    dotnet_response: Any,
    intent_result: Dict,
) -> str:
    """
    [H: Response Formatter]
    Convert raw .NET response + intent into clean plain-text.
    No markdown symbols, no emojis, no bold/italic markers.
    """
    if not isinstance(intent_result, dict):
        try:
            return json.dumps(dotnet_response, indent=2, ensure_ascii=False)
        except Exception:
            return str(dotnet_response)

    if isinstance(dotnet_response, str):
        return dotnet_response

    # ── (NEW) COMPOSITE QUERY PLANNED FORMATTING INTERCEPTION ────────────────
    if isinstance(dotnet_response, dict) and "queryType" in dotnet_response:
        qt = dotnet_response["queryType"]
        
        if qt == "cross_filter":
            records = dotnet_response.get("records") or []
            if not records:
                return "No records matched the cross-filter criteria."
            
            lines = [f"Cross-Filtered Results ({len(records)} matched)", ""]
            first = records[0]
            if isinstance(first, dict):
                headers = [_camel_to_words(k) for k in first.keys() if k not in ("photoPath", "agniveerId")]
                rows = []
                for item in records:
                    rows.append([_safe_str(item.get(k)) for k in first.keys() if k not in ("photoPath", "agniveerId")])
                lines.append(_plain_table(headers, rows))
            else:
                for i, r in enumerate(records, 1):
                    lines.append(f"{i}. {r}")
            return "\n".join(lines)

        elif qt == "multi_independent":
            sections = dotnet_response.get("sections") or []
            lines = ["Combined Diagnostic Report", ""]
            for sec in sections:
                label = sec.get("label", "Section")
                data = sec.get("data")
                lines.append(f"=== {label} ===")
                formatted_sub = format_dotnet_response(data, {"category": label, "subcategory": label})
                lines.append(formatted_sub)
                lines.append("")
            return "\n".join(lines).strip()

        elif qt == "comparison":
            return _format_comparison_result(dotnet_response, intent_result)
            
    # ── END OF COMPOSITE INTERCEPTION ────────────────────────────────────────

    category = intent_result.get("category")
    subcategory = intent_result.get("subcategory")

    if not category:
        try:
            return json.dumps(dotnet_response, indent=2, ensure_ascii=False)
        except Exception:
            return str(dotnet_response)

    if category == "Performance":
        if subcategory in _PERFORMANCE_NESTED_SUBCATEGORIES:
            return _format_performance_list(dotnet_response, intent_result)
        elif subcategory in _PERFORMANCE_AGGREGATE_SUBCATEGORIES:
            return _format_performance_aggregate(subcategory, dotnet_response, intent_result)
        else:
            return _format_performance_list(dotnet_response, intent_result)

    if category in _FORMATTERS:
        return _FORMATTERS[category](subcategory, dotnet_response, intent_result)

    try:
        return json.dumps(dotnet_response, indent=2, ensure_ascii=False)
    except Exception:
        return str(dotnet_response)