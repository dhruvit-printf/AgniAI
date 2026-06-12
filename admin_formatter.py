"""
admin_formatter.py
==================
Formats raw .NET AiCommand JSON responses into clean markdown answers
for the admin chatbot UI. Uses markdown tables for tabular data
(except Performance records, which use a clean line-based layout).

REAL .NET RESPONSE SHAPE (from actual data):
  {
    "success": true,
    "commandLabel": "Top 10 — Overall",
    "data": [
      {
        "id": 816,
        "fullName": "BOBBY RANA",
        "agniveerNo": "A0701772W",
        "mobileNo": "...",
        "bloodGroup": "AB-",
        "class": "DOGRA",
        "batchName": "Batch1",
        "platoonName": "PL-05",
        "isActive": true,
        "attempts": [
          {
            "attemptNo": "1",
            "sections": [
              {
                "sectionId": 1,
                "sectionName": "BPET",
                "displayOrder": 1,
                "subItems": [
                  { "subItemName": "5km", "maxMarks": 40, "marksObtained": 40, "isBestAttempt": true }
                ],
                "omrInputTotal": 100,
                "grading": "Exceptionally Well"
              }
            ]
          }
        ],
        "exceptionalSections": [
          { "sectionName": "Map Reading", "marksObtained": 82.39 }
        ],
        "rank": 1,
        "bestTotal": 311
      }
    ],
    "message": null
  }

NOTE on intent_result keys:
  Receives the original intent_result dict from classify_admin_intent().
  Uses snake_case internally: intent_result["leave_type"], not "leaveType".
"""

from __future__ import annotations

import json
import re as _re
from typing import Any, Dict, List, Optional


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
    """Try multiple key names, return first hit."""
    for key in keys:
        v = obj.get(key)
        if v is not None:
            return v
    return fallback


def _md_escape(text: Any) -> str:
    """Escape pipe characters for use inside markdown table cells."""
    return str(text if text is not None else "").replace("|", "\\|")


def _md_table(headers: List[str], rows: List[List[Any]]) -> str:
    """Build a markdown table from headers and rows."""
    if not rows:
        return "_No data available._"
    header_row = "| " + " | ".join(headers) + " |"
    separator  = "| " + " | ".join(["---"] * len(headers)) + " |"
    data_rows  = [
        "| " + " | ".join(_md_escape(cell) for cell in row) + " |"
        for row in rows
    ]
    return "\n".join([header_row, separator] + data_rows)


def _kv_table(pairs: List[tuple]) -> str:
    """Build a two-column key/value markdown table."""
    rows = [[label, value] for label, value in pairs if value not in (None, "", "N/A")]
    if not rows:
        return "_No data available._"
    return _md_table(["Field", "Value"], rows)


def _rank_medal(rank: int) -> str:
    """Return medal emoji for top 3 ranks."""
    return {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"**{rank}.**")


# =============================================================================
# PERFORMANCE — NESTED STRUCTURE HANDLERS
# =============================================================================

def _grading_emoji(grading: str) -> str:
    g = (grading or "").lower()
    if "exceptional" in g: return "🏆"
    if "excellent"   in g: return "⭐"
    if "good"        in g: return "✅"
    if "sat"         in g: return "🔵"
    if "fail" in g or "unsa" in g: return "❌"
    return "•"


def _format_single_agniveer(agniveer: Dict, rank: int, intent_result: Dict) -> str:
    """Format one agniveer record into a clean readable markdown block."""
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
    medal        = _rank_medal(int(display_rank))

    lines = [f"{medal} **{name}**"]

    # Meta line
    meta_parts = []
    if agniveer_no: meta_parts.append(f"`{agniveer_no}`")
    if batch:       meta_parts.append(f"Batch: **{batch}**")
    if platoon:     meta_parts.append(f"Platoon: **{platoon}**")
    if cls:         meta_parts.append(f"Class: **{cls}**")
    if meta_parts:
        lines.append(" · ".join(meta_parts))

    if best_total is not None:
        lines.append(f"📊 **Best Total: {best_total}**")

    section_filter = (intent_result.get("section") or "").upper()
    attempt_filter = intent_result.get("attempt_no")

    for attempt in attempts:
        attempt_no = attempt.get("attemptNo", "?")
        if attempt_filter is not None and str(attempt_no) != str(attempt_filter):
            continue

        sections = attempt.get("sections") or []
        if section_filter:
            sections = [s for s in sections
                        if (s.get("sectionName") or "").upper() == section_filter]

        has_data = any(
            s.get("omrInputTotal") is not None and s.get("omrInputTotal", 0) > 0
            for s in sections
        )
        if not sections or not has_data:
            continue

        lines.append("")
        lines.append(f"*Attempt {attempt_no}:*")

        for section in sorted(sections, key=lambda s: s.get("displayOrder", 99)):
            s_name  = _safe_str(section.get("sectionName"))
            s_total = section.get("omrInputTotal")
            s_grade = _safe_str(section.get("grading"), "")
            emoji   = _grading_emoji(s_grade)

            if s_total is None or s_total == 0:
                continue

            grade_str = f" — *{s_grade}*" if s_grade else ""
            lines.append(f"&nbsp;&nbsp;{emoji} **{s_name}**: {s_total} pts{grade_str}")

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
                lines.append(f"&nbsp;&nbsp;&nbsp;&nbsp;↳ {', '.join(sub_parts)}")

    # Exceptional sections
    if exceptional:
        exc_parts = []
        for exc in exceptional:
            exc_name  = _safe_str(exc.get("sectionName"))
            exc_marks = exc.get("marksObtained")
            if exc_marks is not None:
                exc_parts.append(f"{exc_name}: **{exc_marks:.2f}**")
        if exc_parts:
            lines.append("")
            lines.append(f"🎖️ *Exceptional:* {' · '.join(exc_parts)}")

    return "\n".join(lines)


def _format_performance_list(data: Any, intent_result: Dict) -> str:
    """Handle the real .NET performance list response."""
    if isinstance(data, dict):
        command_label = data.get("commandLabel", "")
        records = (
            data.get("data") or data.get("Data") or
            data.get("result") or data.get("Result") or []
        )
    elif isinstance(data, list):
        command_label = ""
        records = data
    else:
        return "_Unexpected data format._"

    if not records:
        return "_No records found for this query._"

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
    count   = len(records)
    sec_str = f" — {section_filter}" if section_filter else ""

    lines = [
        f"### {label}{sec_str}",
        f"*{count} record{'s' if count != 1 else ''}*",
        "",
    ]

    for i, record in enumerate(records, start=1):
        lines.append(_format_single_agniveer(record, i, intent_result))
        if i < len(records):
            lines.append("")
            lines.append("---")
            lines.append("")

    return "\n".join(lines).strip()


# =============================================================================
# LEAVE
# =============================================================================

def _format_leave(subcategory: str, data: Any, intent_result: Dict) -> str:
    leave_type = intent_result.get("leave_type", "")
    lt_str = f" ({leave_type})" if leave_type else ""

    if subcategory in ("MostLeaveTaken", "LeastLeaveTaken"):
        label = "Most Leave Taken" if subcategory == "MostLeaveTaken" else "Least Leave Taken"
        records = data if isinstance(data, list) else (
            _get(data, "personnel", "data", "Data") or []
            if isinstance(data, dict) else []
        )
        if not records:
            return "_No leave data found._"

        rows = []
        for i, item in enumerate(records, 1):
            name = _safe_str(_get(item, "name", "Name", "fullName"))
            days = _get(item, "leaveDays", "days", "count", "Days")
            lt   = _safe_str(_get(item, "leaveType", "type"), "—")
            medal = _rank_medal(i)
            rows.append([f"{medal} {name}", days if days is not None else "—", lt])

        return "\n".join([
            f"### {label}{lt_str}",
            "",
            _md_table(["Name", "Days", "Leave Type"], rows),
        ])

    if subcategory == "CurrentLeaveStatus":
        records = data if isinstance(data, list) else (
            _get(data, "personnel", "data") or []
            if isinstance(data, dict) else []
        )
        if not records:
            return "_No personnel are currently on leave._"

        rows = []
        for item in records:
            name = _safe_str(_get(item, "name", "Name", "fullName"))
            lt   = _safe_str(_get(item, "leaveType", "type"), "—")
            date = _safe_str(_get(item, "from", "startDate", "fromDate"), "—")
            rows.append([name, lt, date])

        return "\n".join([
            f"### Currently On Leave ({len(records)} personnel)",
            "",
            _md_table(["Name", "Leave Type", "From"], rows),
        ])

    if subcategory == "AbscondedPersonnel":
        records = data if isinstance(data, list) else []
        if not records:
            return "_No absconded personnel on record._"

        rows = [[f"⚠️ {_safe_str(_get(item, 'name', 'Name', 'fullName'))}",
                 _safe_str(_get(item, "since", "date", "abscondedDate"), "—")]
                for item in records]

        return "\n".join([
            f"### Absconded Personnel ({len(records)})",
            "",
            _md_table(["Name", "Since"], rows),
        ])

    # Generic leave fallback
    if isinstance(data, list):
        rows = [[i, _safe_str(_get(item, "name", "Name", "fullName"))]
                for i, item in enumerate(data, 1)]
        return "\n".join(["### Leave Data", "", _md_table(["#", "Name"], rows)])
    return str(data)


# =============================================================================
# MEDICAL
# =============================================================================

def _format_medical(subcategory: str, data: Any, intent_result: Dict) -> str:
    if subcategory == "ActiveCases":
        records = data if isinstance(data, list) else (
            _get(data, "data", "cases") or []
            if isinstance(data, dict) else []
        )
        total_from_dict = _get(data, "total", "count") if isinstance(data, dict) else None
        if total_from_dict is not None and not records:
            return f"There are currently **{total_from_dict}** active medical case(s)."
        if not records:
            return "_No active medical cases at the moment._"

        rows = [
            [_safe_str(_get(item, "name", "Name", "fullName")),
             _safe_str(_get(item, "disease", "diagnosis", "condition"), "Unknown"),
             _safe_str(_get(item, "ward"), "—")]
            for item in records
        ]
        return "\n".join([
            f"### Active Medical Cases ({len(records)})",
            "",
            _md_table(["Name", "Diagnosis", "Ward"], rows),
        ])

    if subcategory == "BMIAnalysis":
        if isinstance(data, list):
            rows = [
                [_safe_str(_get(item, "name", "Name", "fullName")),
                 _safe_str(_get(item, "bmi", "BMI"), "—"),
                 _safe_str(_get(item, "category", "bmiCategory"), "—")]
                for item in data
            ]
            return "\n".join(["### BMI Analysis", "", _md_table(["Name", "BMI", "Category"], rows)])
        if isinstance(data, dict):
            pairs = [(_camel_to_words(k), v) for k, v in data.items()
                     if not isinstance(v, (list, dict))]
            return "\n".join(["### BMI / Fitness Analysis", "", _kv_table(pairs)])

    if subcategory == "DiseaseStatistics":
        records = data if isinstance(data, list) else []
        if not records:
            return "_No disease statistics available._"
        rows = [
            [i, _safe_str(_get(item, "disease", "name", "condition", "diagnosis")),
             _safe_str(_get(item, "count", "cases", "total"), "—")]
            for i, item in enumerate(records, 1)
        ]
        return "\n".join([
            "### Disease Statistics",
            "",
            _md_table(["#", "Disease", "Cases"], rows),
        ])

    if isinstance(data, dict):
        pairs = [(_camel_to_words(k), v) for k, v in data.items()
                 if not isinstance(v, (list, dict))]
        return "\n".join(["### Medical Summary", "", _kv_table(pairs)])
    return str(data)


# =============================================================================
# ATTENDANCE
# =============================================================================

def _format_attendance(subcategory: str, data: Any, intent_result: Dict) -> str:
    if subcategory == "PresentToday":
        if isinstance(data, dict):
            present   = _get(data, "present", "Present", "count", "Count") or 0
            total     = _get(data, "total", "Total")
            total_str = f" out of **{total}**" if total else ""
            return f"🟢 **{present}** personnel are present on campus today{total_str}."
        if isinstance(data, (int, float)):
            return f"🟢 **{data}** personnel are present on campus today."

    if subcategory == "MonthlyAttendance":
        if isinstance(data, list):
            rows = [
                [_safe_str(_get(item, "month", "Month")),
                 f"{_get(item, 'percentage', 'attendancePercentage', 'Percentage')}%"
                 if _get(item, "percentage", "attendancePercentage", "Percentage") is not None else "—",
                 _safe_str(_get(item, "present", "Present"), "—")]
                for item in data
            ]
            return "\n".join([
                "### Monthly Attendance",
                "",
                _md_table(["Month", "Attendance %", "Present"], rows),
            ])
        if isinstance(data, dict):
            pairs = [(_camel_to_words(k), v) for k, v in data.items()
                     if not isinstance(v, (list, dict))]
            return "\n".join(["### Monthly Attendance Summary", "", _kv_table(pairs)])

    if subcategory == "StrengthBreakdown":
        if isinstance(data, dict):
            pairs = [(_camel_to_words(k), v) for k, v in data.items()
                     if not isinstance(v, (list, dict))]
            return "\n".join(["### Strength Breakdown", "", _kv_table(pairs)])

    if isinstance(data, dict):
        pairs = [(_camel_to_words(k), v) for k, v in data.items()
                 if not isinstance(v, (list, dict))]
        return "\n".join(["### Attendance Summary", "", _kv_table(pairs)])
    return str(data)


# =============================================================================
# VERIFICATION
# =============================================================================

def _format_verification(subcategory: str, data: Any, intent_result: Dict) -> str:
    label   = "Pending" if subcategory == "PendingVerification" else "Completed"
    icon    = "⏳" if label == "Pending" else "✅"
    records = data if isinstance(data, list) else (
        _get(data, "data", "records") or []
        if isinstance(data, dict) else []
    )
    total_from_dict = _get(data, "total", "count") if isinstance(data, dict) else None

    if total_from_dict is not None and not records:
        return f"**{total_from_dict}** {label.lower()} verification(s) found."
    if not records:
        return f"_No {label.lower()} verifications found._"

    rows = [
        [f"{icon} {_safe_str(_get(item, 'name', 'Name', 'fullName'))}",
         _safe_str(_get(item, "documentType", "document", "docType"), "—")]
        for item in records
    ]
    return "\n".join([
        f"### {label} Verifications ({len(records)})",
        "",
        _md_table(["Name", "Document Type"], rows),
    ])


# =============================================================================
# EQUIPMENT
# =============================================================================

def _format_equipment(subcategory: str, data: Any, intent_result: Dict) -> str:
    if subcategory in ("IssuedItems", "ProcuredItems"):
        return _format_item_list(subcategory, data, intent_result)

    if subcategory == "EquipmentSummary":
        if isinstance(data, dict):
            pairs = [(_camel_to_words(k), v) for k, v in data.items()
                     if not isinstance(v, (list, dict))]
            return "\n".join(["### Equipment Summary", "", _kv_table(pairs)])

    if subcategory == "OverdueEquipment":
        records = data if isinstance(data, list) else []
        if not records:
            return "_No overdue equipment records._"
        rows = [
            [f"⚠️ {_safe_str(_get(item, 'name', 'equipment', 'itemName'))}",
             _safe_str(_get(item, "issuedTo", "holder"), "—"),
             f"{_get(item, 'overdueDays', 'daysOverdue')} days"
             if _get(item, "overdueDays", "daysOverdue") else "—"]
            for item in records
        ]
        return "\n".join([
            f"### Overdue Equipment ({len(records)} items)",
            "",
            _md_table(["Equipment", "Issued To", "Overdue By"], rows),
        ])

    if subcategory == "PoorConditionEquipment":
        records = data if isinstance(data, list) else []
        if not records:
            return "_No equipment returned in poor condition._"
        rows = [
            [f"🔴 {_safe_str(_get(item, 'name', 'equipment', 'itemName'))}",
             _safe_str(_get(item, "condition", "state"), "Poor")]
            for item in records
        ]
        return "\n".join([
            f"### Poor Condition Equipment ({len(records)} items)",
            "",
            _md_table(["Equipment", "Condition"], rows),
        ])

    if isinstance(data, dict):
        pairs = [(_camel_to_words(k), v) for k, v in data.items()
                 if not isinstance(v, (list, dict))]
        return "\n".join(["### Equipment Data", "", _kv_table(pairs)])

    if isinstance(data, list):
        rows = [[i, _safe_str(_get(item, "name", "equipment", "itemName"))]
                for i, item in enumerate(data, 1)]
        return "\n".join(["### Equipment Records", "", _md_table(["#", "Item"], rows)])

    return str(data)


def _format_item_list(subcategory: str, data: Any, intent_result: Dict) -> str:
    from admin_intent import ISSUED_ITEMS, PROCURED_ITEMS

    is_issued   = subcategory == "IssuedItems"
    label       = "Issued Items" if is_issued else "Procured Items"
    master_list = ISSUED_ITEMS if is_issued else PROCURED_ITEMS
    item_name   = intent_result.get("item_name")

    # ── Single-item lookup ──────────────────────────────────────────────────
    if item_name:
        in_issued    = item_name in ISSUED_ITEMS
        in_procured  = item_name in PROCURED_ITEMS
        category_tag = "Issued" if in_issued else ("Procured" if in_procured else "Unknown")
        item_detail  = ""
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
        return "\n".join([f"### Item Lookup — {item_name}", "", _kv_table(pairs)])

    # ── List from .NET (or master list fallback) ────────────────────────────
    records: list = []
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        records = _get(data, "data", "items", "Data", "Items") or []

    items_to_render = records if records else master_list

    rows = []
    for i, entry in enumerate(items_to_render, 1):
        if isinstance(entry, str):
            rows.append([i, entry, "—", "—"])
        elif isinstance(entry, dict):
            rows.append([
                i,
                _safe_str(_get(entry, "name", "itemName", "item", "Name")),
                _safe_str(_get(entry, "status", "Status"), "—"),
                _safe_str(_get(entry, "quantity", "qty", "Quantity"), "—"),
            ])
        else:
            rows.append([i, _safe_str(entry), "—", "—"])

    note = "\n\n*Source: master list*" if not records else ""
    return "\n".join([
        f"### {label} ({len(items_to_render)} items)",
        "",
        _md_table(["#", "Item Name", "Status", "Qty"], rows),
    ]) + note


# =============================================================================
# DISTRIBUTION
# =============================================================================

def _format_distribution(subcategory: str, data: Any, intent_result: Dict) -> str:
    if subcategory == "LatestDistribution":
        if isinstance(data, dict):
            pairs = [(_camel_to_words(k), v) for k, v in data.items()
                     if not isinstance(v, (list, dict))]
            return "\n".join(["### Latest Distribution", "", _kv_table(pairs)])
        if isinstance(data, list):
            rows = [[i, _safe_str(_get(item, "name", "Name"))]
                    for i, item in enumerate(data, 1)]
            return "\n".join(["### Latest Distribution", "", _md_table(["#", "Name"], rows)])

    if subcategory == "DistributionByUnit":
        records = data if isinstance(data, list) else []
        if not records:
            return "_No unit distribution data._"
        rows = [
            [_safe_str(_get(item, "unit", "Unit", "unitName")),
             _safe_str(_get(item, "count", "quantity", "total"), "—")]
            for item in records
        ]
        return "\n".join(["### Distribution by Unit", "", _md_table(["Unit", "Count"], rows)])

    if subcategory == "UnassignedItems":
        records = data if isinstance(data, list) else []
        if not records:
            return "_All items have been assigned to units._"
        rows = [[i, _safe_str(_get(item, "name", "Name"))]
                for i, item in enumerate(records, 1)]
        return "\n".join([
            f"### Unassigned Items ({len(records)})",
            "",
            _md_table(["#", "Item"], rows),
        ])

    if subcategory == "TopUnit":
        if isinstance(data, dict):
            unit  = _safe_str(_get(data, "unit", "Unit", "unitName"))
            count = _get(data, "count", "quantity")
            return f"🏆 The top unit is **{unit}** with **{count}** items distributed."

    if isinstance(data, dict):
        pairs = [(_camel_to_words(k), v) for k, v in data.items()
                 if not isinstance(v, (list, dict))]
        return "\n".join(["### Distribution Summary", "", _kv_table(pairs)])
    return str(data)


# =============================================================================
# SKILLS
# =============================================================================

def _format_skills(subcategory: str, data: Any, intent_result: Dict) -> str:
    if subcategory == "BySport":
        records = data if isinstance(data, list) else []
        if not records:
            return "_No sport data found._"
        rows = [
            [f"🏅 {_safe_str(_get(item, 'sport', 'name', 'Sport'))}",
             _safe_str(_get(item, "count", "personnel", "total"), "—")]
            for item in records
        ]
        return "\n".join(["### Roster by Sport", "", _md_table(["Sport", "Personnel"], rows)])

    if subcategory == "ByClass":
        records = data if isinstance(data, list) else []
        if not records:
            return "_No class data found._"
        rows = [
            [_safe_str(_get(item, "class", "className", "name", "Class")),
             _safe_str(_get(item, "count", "personnel", "total"), "—")]
            for item in records
        ]
        return "\n".join(["### Roster by Class", "", _md_table(["Class", "Personnel"], rows)])

    if subcategory == "BloodGroup":
        records = data if isinstance(data, list) else []
        if not records:
            return "_No blood group data found._"
        rows = [
            [f"🩸 **{_safe_str(_get(item, 'bloodGroup', 'blood', 'group'))}**",
             _safe_str(_get(item, "count", "total"), "—")]
            for item in records
        ]
        return "\n".join([
            "### Blood Group Distribution",
            "",
            _md_table(["Blood Group", "Count"], rows),
        ])

    if isinstance(data, list):
        rows = [[i, _safe_str(_get(item, "name", "Name", "fullName"))]
                for i, item in enumerate(data, 1)]
        return "\n".join(["### Skills / Roster", "", _md_table(["#", "Name"], rows)])
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
    sec_str = f" for **{section}**" if section else ""

    if subcategory == "AverageScore":
        avg = (
            _get(data, "averageScore", "AverageScore", "average", "Average")
            if isinstance(data, dict) else data
        )
        return f"📊 The average score{sec_str} is **{avg}**."

    if subcategory in ("PassPercentage", "FailPercentage"):
        label  = "pass" if subcategory == "PassPercentage" else "fail"
        icon   = "✅" if label == "pass" else "❌"
        if isinstance(data, dict):
            pct       = _get(data, "percentage", "Percentage", "passPercentage", "failPercentage")
            total     = _get(data, "total", "Total")
            total_str = f" (out of {total} total)" if total else ""
            return f"{icon} The {label} percentage{sec_str} is **{pct}%**{total_str}."
        return f"{icon} {label.title()} percentage{sec_str}: **{data}%**"

    if subcategory in ("GradeDistribution", "GradingSummary"):
        title = f"### Grade Distribution{' — ' + section if section else ''}"
        if isinstance(data, dict):
            rows = [
                [f"{_grading_emoji(k)} {k}", v]
                for k, v in data.items()
                if not isinstance(v, (list, dict))
            ]
            return "\n".join([title, "", _md_table(["Grade", "Count"], rows)])
        if isinstance(data, list):
            rows = [
                [f"{_grading_emoji(_safe_str(_get(item,'grade','Grade','grading')))} "
                 f"{_safe_str(_get(item,'grade','Grade','grading'))}",
                 _safe_str(_get(item, "count", "Count", "total"), "—")]
                for item in data
            ]
            return "\n".join([title, "", _md_table(["Grade", "Count"], rows)])

    # Generic aggregate fallback
    if isinstance(data, dict):
        pairs = [(_camel_to_words(k), v) for k, v in data.items()
                 if not isinstance(v, (list, dict))]
        return "\n".join([f"### Performance Data{sec_str}", "", _kv_table(pairs)])
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


# =============================================================================
# MAIN PUBLIC ENTRY POINT
# =============================================================================

def format_dotnet_response(
    dotnet_response: Any,
    intent_result: Dict,
) -> str:
    """
    Convert raw .NET response + intent into clean markdown.
    Frontend renders this with react-markdown (or similar).
    """
    category    = intent_result.get("category", "")
    subcategory = intent_result.get("subcategory", "")

    # ── Handle .NET error shapes ───────────────────────────────────────────
    if isinstance(dotnet_response, dict):
        error_msg = _get(dotnet_response, "error", "Error", "errorMessage")
        if error_msg:
            return f"> ⚠️ **Server error:** {error_msg}"
        if dotnet_response.get("success") is False:
            msg = _get(dotnet_response, "message", "Message") or "Unknown error."
            return f"> ⚠️ **Request failed:** {msg}"

    # ── Overall (composite ranking — same shape as Performance) ───────────
    if category == "Overall":
        return _format_performance_list(dotnet_response, intent_result)

    # ── Performance ────────────────────────────────────────────────────────
    if category == "Performance":
        if subcategory in _PERFORMANCE_NESTED_SUBCATEGORIES:
            return _format_performance_list(dotnet_response, intent_result)

        data = (
            _get(dotnet_response, "data", "Data", "result", "Result")
            or dotnet_response
        ) if isinstance(dotnet_response, dict) else dotnet_response

        if subcategory in _PERFORMANCE_AGGREGATE_SUBCATEGORIES:
            return _format_performance_aggregate(subcategory, data, intent_result)

        return _format_performance_list(dotnet_response, intent_result)

    # ── All other modules ──────────────────────────────────────────────────
    data = (
        _get(dotnet_response, "data", "Data", "result", "Result")
        or dotnet_response
    ) if isinstance(dotnet_response, dict) else dotnet_response

    formatter = _FORMATTERS.get(category)
    if formatter and subcategory:
        try:
            return formatter(subcategory, data, intent_result)
        except Exception:
            pass

    # ── Ultimate fallback: pretty JSON ─────────────────────────────────────
    try:
        return "```json\n" + json.dumps(data, indent=2, ensure_ascii=False) + "\n```"
    except Exception:
        return str(data)