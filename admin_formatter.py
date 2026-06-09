"""
admin_formatter.py
==================
Formats raw .NET AiCommand JSON responses into human-readable markdown answers
for the admin chatbot UI.

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
                "sectionName": "BEPT",
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


# =============================================================================
# PERFORMANCE — NESTED STRUCTURE HANDLERS
# =============================================================================

def _grading_emoji(grading: str) -> str:
    g = (grading or "").lower()
    if "exceptional" in g:
        return "🏆"
    if "excellent" in g:
        return "⭐"
    if "good" in g:
        return "✅"
    if "sat" in g:
        return "🔵"
    if "fail" in g or "unsa" in g:
        return "❌"
    return "•"


def _best_attempt_for_section(attempts: List[Dict], section_name: str) -> Optional[Dict]:
    """
    Find the best attempt entry for a given section.
    Priority: attempt with highest omrInputTotal for this section,
    falling back to the first attempt that has data.
    """
    best_section = None
    best_total = -1

    for attempt in (attempts or []):
        for section in (attempt.get("sections") or []):
            if (section.get("sectionName") or "").upper() == section_name.upper():
                total = section.get("omrInputTotal") or 0
                if total > best_total:
                    best_total = total
                    best_section = section

    return best_section


def _format_single_agniveer(agniveer: Dict, rank: int, intent_result: Dict) -> str:
    """Format one agniveer record into a readable block."""
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

    # Header line
    lines = [f"**{display_rank}. {name}**"]

    # Meta line
    meta_parts = []
    if agniveer_no:
        meta_parts.append(f"No: `{agniveer_no}`")
    if batch:
        meta_parts.append(f"Batch: {batch}")
    if platoon:
        meta_parts.append(f"Platoon: {platoon}")
    if cls:
        meta_parts.append(f"Class: {cls}")
    if meta_parts:
        lines.append("   " + " | ".join(meta_parts))

    # Best total
    if best_total is not None:
        lines.append(f"   📊 **Best Total: {best_total}**")

    # Section filter from intent — show only asked section
    section_filter = (intent_result.get("section") or "").upper()

    # Determine which attempts to show
    # Show only best attempts (isBestAttempt sections) unless a specific attemptNo requested
    attempt_filter = intent_result.get("attempt_no")

    for attempt in attempts:
        attempt_no = attempt.get("attemptNo", "?")

        # If a specific attempt was requested, skip others
        if attempt_filter is not None and str(attempt_no) != str(attempt_filter):
            continue

        sections = attempt.get("sections") or []

        # If section filter active, show only that section
        if section_filter:
            sections = [s for s in sections if (s.get("sectionName") or "").upper() == section_filter]

        if not sections:
            continue

        # Only show attempts that have at least one graded section
        has_data = any(
            s.get("omrInputTotal") is not None and s.get("omrInputTotal", 0) > 0
            for s in sections
        )
        if not has_data:
            continue

        lines.append(f"   *Attempt {attempt_no}:*")
        for section in sorted(sections, key=lambda s: s.get("displayOrder", 99)):
            s_name   = _safe_str(section.get("sectionName"))
            s_total  = section.get("omrInputTotal")
            s_grade  = _safe_str(section.get("grading"), "")
            emoji    = _grading_emoji(s_grade)

            if s_total is None or s_total == 0:
                continue

            sub_items = section.get("subItems") or []
            best_sub  = [si for si in sub_items if si.get("isBestAttempt") is True]

            # Build subitem summary string
            sub_summary = ""
            if best_sub:
                parts = []
                for si in best_sub:
                    si_name    = _safe_str(si.get("subItemName"))
                    obtained   = si.get("marksObtained")
                    max_marks  = si.get("maxMarks")
                    if obtained is not None and max_marks is not None:
                        parts.append(f"{si_name}: {obtained}/{max_marks}")
                if parts:
                    sub_summary = " | ".join(parts)

            grade_str = f" ({s_grade})" if s_grade else ""
            line = f"     {emoji} **{s_name}**: {s_total} pts{grade_str}"
            if sub_summary:
                line += f"\n       ↳ {sub_summary}"
            lines.append(line)

    # Exceptional sections (bonus marks)
    if exceptional:
        exc_parts = []
        for exc in exceptional:
            exc_name  = _safe_str(exc.get("sectionName"))
            exc_marks = exc.get("marksObtained")
            if exc_marks is not None:
                exc_parts.append(f"{exc_name}: {exc_marks:.2f}")
        if exc_parts:
            lines.append(f"   🎖️ Exceptional: {' | '.join(exc_parts)}")

    return "\n".join(lines)


def _format_performance_list(data: Any, intent_result: Dict) -> str:
    """
    Handle the real .NET performance list response.
    data is either the raw .NET envelope OR the unwrapped data array.
    """
    # Unwrap if still in .NET envelope
    if isinstance(data, dict):
        command_label = data.get("commandLabel", "")
        records = (
            data.get("data") or
            data.get("Data") or
            data.get("result") or
            data.get("Result") or
            []
        )
    elif isinstance(data, list):
        command_label = ""
        records = data
    else:
        return f"Unexpected data format: {type(data).__name__}"

    if not records:
        return "No records found for this query."

    subcategory = intent_result.get("subcategory", "")
    section_filter = intent_result.get("section", "")

    # Build header
    label_map = {
        "TopPerformers":      "Top Performers",
        "LowestPerformers":   "Lowest Performers",
        "OverallPerformance": "Overall Performance",
        "AverageScore":       "Average Score",
        "PassPercentage":     "Pass Percentage",
        "FailPercentage":     "Fail Percentage",
        "GradeDistribution":  "Grade Distribution",
        "GradingSummary":     "Grading Summary",
        "Improvement":        "Improvement",
        "Drop":               "Score Drop",
        "SectionSummary":     "Section Summary",
        "AttemptWise":        "Attempt-wise Analysis",
        "BestAttempt":        "Best Attempt",
        "Comparison":         "Comparison",
    }
    label    = command_label or label_map.get(subcategory, subcategory or "Performance Data")
    count    = len(records)
    sec_str  = f" — {section_filter}" if section_filter else ""

    header = f"### {label}{sec_str} ({count} records)\n"
    lines  = [header]

    for i, record in enumerate(records, start=1):
        lines.append(_format_single_agniveer(record, i, intent_result))
        lines.append("")  # blank line between records

    return "\n".join(lines).strip()


# =============================================================================
# LEAVE
# =============================================================================

def _format_leave(subcategory: str, data: Any, intent_result: Dict) -> str:
    leave_type = intent_result.get("leave_type", "")
    lt_str = f" ({leave_type})" if leave_type else ""

    if subcategory in ("MostLeaveTaken", "LeastLeaveTaken"):
        label = "Most" if subcategory == "MostLeaveTaken" else "Least"
        records = data if isinstance(data, list) else (
            _get(data, "personnel", "data", "Data") or [] if isinstance(data, dict) else []
        )
        if not records:
            return f"No leave data found."
        lines = [f"### Personnel with {label} Leave{lt_str}\n"]
        for i, item in enumerate(records, 1):
            name  = _safe_str(_get(item, "name", "Name", "fullName"))
            days  = _get(item, "leaveDays", "days", "count", "Days")
            lt    = _safe_str(_get(item, "leaveType", "type"), "")
            line  = f"{i}. **{name}**"
            if days is not None:
                line += f" — {days} days"
            if lt:
                line += f" ({lt})"
            lines.append(line)
        return "\n".join(lines)

    if subcategory == "CurrentLeaveStatus":
        records = data if isinstance(data, list) else (
            _get(data, "personnel", "data") or [] if isinstance(data, dict) else []
        )
        if not records:
            return "No personnel are currently on leave."
        lines = [f"### Currently On Leave ({len(records)} personnel)\n"]
        for item in records:
            name  = _safe_str(_get(item, "name", "Name", "fullName"))
            lt    = _safe_str(_get(item, "leaveType", "type"), "")
            date  = _safe_str(_get(item, "from", "startDate", "fromDate"), "")
            line  = f"• **{name}**"
            if lt:
                line += f" — {lt}"
            if date:
                line += f" (from {date})"
            lines.append(line)
        return "\n".join(lines)

    if subcategory == "AbscondedPersonnel":
        records = data if isinstance(data, list) else []
        if not records:
            return "No absconded personnel on record."
        lines = [f"### Absconded Personnel ({len(records)})\n"]
        for item in records:
            name  = _safe_str(_get(item, "name", "Name", "fullName"))
            since = _safe_str(_get(item, "since", "date", "abscondedDate"), "")
            line  = f"• **{name}**"
            if since:
                line += f" — since {since}"
            lines.append(line)
        return "\n".join(lines)

    # Generic leave fallback
    if isinstance(data, list):
        lines = ["### Leave Data\n"]
        for i, item in enumerate(data, 1):
            name = _safe_str(_get(item, "name", "Name", "fullName"))
            lines.append(f"{i}. {name}")
        return "\n".join(lines)
    return str(data)


# =============================================================================
# MEDICAL
# =============================================================================

def _format_medical(subcategory: str, data: Any, intent_result: Dict) -> str:
    if subcategory == "ActiveCases":
        records = data if isinstance(data, list) else (
            _get(data, "data", "cases") or [] if isinstance(data, dict) else []
        )
        total_from_dict = _get(data, "total", "count") if isinstance(data, dict) else None
        if total_from_dict is not None and not records:
            return f"There are currently **{total_from_dict}** active medical case(s)."
        if not records:
            return "No active medical cases at the moment."
        lines = [f"### Active Medical Cases ({len(records)})\n"]
        for item in records:
            name    = _safe_str(_get(item, "name", "Name", "fullName"))
            disease = _safe_str(_get(item, "disease", "diagnosis", "condition"), "Unknown")
            ward    = _safe_str(_get(item, "ward"), "")
            line    = f"• **{name}** — {disease}"
            if ward:
                line += f" (Ward: {ward})"
            lines.append(line)
        return "\n".join(lines)

    if subcategory == "BMIAnalysis":
        if isinstance(data, list):
            lines = ["### BMI Analysis\n"]
            for item in data:
                name = _safe_str(_get(item, "name", "Name", "fullName"))
                bmi  = _get(item, "bmi", "BMI")
                cat  = _safe_str(_get(item, "category", "bmiCategory"), "")
                line = f"• **{name}** — BMI: {bmi}"
                if cat:
                    line += f" ({cat})"
                lines.append(line)
            return "\n".join(lines)
        if isinstance(data, dict):
            lines = ["### BMI / Fitness Analysis\n"]
            for key, val in data.items():
                lines.append(f"• **{_camel_to_words(key)}:** {val}")
            return "\n".join(lines)

    if subcategory == "DiseaseStatistics":
        records = data if isinstance(data, list) else []
        if not records:
            return "No disease statistics available."
        lines = ["### Disease Statistics\n"]
        for i, item in enumerate(records, 1):
            disease = _safe_str(_get(item, "disease", "name", "condition", "diagnosis"))
            count   = _get(item, "count", "cases", "total")
            lines.append(f"{i}. **{disease}** — {count} cases")
        return "\n".join(lines)

    if isinstance(data, dict):
        lines = ["### Medical Summary\n"]
        for k, v in data.items():
            if not isinstance(v, (list, dict)):
                lines.append(f"• **{_camel_to_words(k)}:** {v}")
        return "\n".join(lines)
    return str(data)


# =============================================================================
# ATTENDANCE
# =============================================================================

def _format_attendance(subcategory: str, data: Any, intent_result: Dict) -> str:
    if subcategory == "PresentToday":
        if isinstance(data, dict):
            present = _get(data, "present", "Present", "count", "Count") or 0
            total   = _get(data, "total", "Total")
            total_str = f" out of {total}" if total else ""
            return f"**{present}** personnel are present on campus today{total_str}."
        if isinstance(data, (int, float)):
            return f"**{data}** personnel are present on campus today."

    if subcategory == "MonthlyAttendance":
        if isinstance(data, list):
            lines = ["### Monthly Attendance\n"]
            for item in data:
                month = _safe_str(_get(item, "month", "Month"))
                pct   = _get(item, "percentage", "attendancePercentage", "Percentage")
                present = _get(item, "present", "Present")
                line  = f"• **{month}:**"
                if pct is not None:
                    line += f" {pct}%"
                if present is not None:
                    line += f" ({present} present)"
                lines.append(line)
            return "\n".join(lines)
        if isinstance(data, dict):
            lines = ["### Monthly Attendance Summary\n"]
            for k, v in data.items():
                if not isinstance(v, (list, dict)):
                    lines.append(f"• **{_camel_to_words(k)}:** {v}")
            return "\n".join(lines)

    if subcategory == "StrengthBreakdown":
        if isinstance(data, dict):
            lines = ["### Strength Breakdown\n"]
            for k, v in data.items():
                if not isinstance(v, (list, dict)):
                    lines.append(f"• **{_camel_to_words(k)}:** {v}")
            return "\n".join(lines)

    if isinstance(data, dict):
        lines = ["### Attendance Summary\n"]
        for k, v in data.items():
            if not isinstance(v, (list, dict)):
                lines.append(f"• **{_camel_to_words(k)}:** {v}")
        return "\n".join(lines)
    return str(data)


# =============================================================================
# VERIFICATION
# =============================================================================

def _format_verification(subcategory: str, data: Any, intent_result: Dict) -> str:
    label = "Pending" if subcategory == "PendingVerification" else "Completed"
    records = data if isinstance(data, list) else (
        _get(data, "data", "records") or [] if isinstance(data, dict) else []
    )
    total_from_dict = _get(data, "total", "count") if isinstance(data, dict) else None

    if total_from_dict is not None and not records:
        return f"**{total_from_dict}** {label.lower()} verification(s) found."
    if not records:
        return f"No {label.lower()} verifications found."

    lines = [f"### {label} Verifications ({len(records)})\n"]
    for item in records:
        name = _safe_str(_get(item, "name", "Name", "fullName"))
        doc  = _safe_str(_get(item, "documentType", "document", "docType"), "")
        line = f"• **{name}**"
        if doc:
            line += f" — {doc}"
        lines.append(line)
    return "\n".join(lines)


# =============================================================================
# EQUIPMENT
# =============================================================================

def _format_equipment(subcategory: str, data: Any, intent_result: Dict) -> str:
    if subcategory == "EquipmentSummary":
        if isinstance(data, dict):
            lines = ["### Equipment Summary\n"]
            for k, v in data.items():
                if not isinstance(v, (list, dict)):
                    lines.append(f"• **{_camel_to_words(k)}:** {v}")
            return "\n".join(lines)

    if subcategory == "OverdueEquipment":
        records = data if isinstance(data, list) else []
        if not records:
            return "No overdue equipment records."
        lines = [f"### Overdue Equipment ({len(records)} items)\n"]
        for item in records:
            name   = _safe_str(_get(item, "name", "equipment", "itemName"))
            person = _safe_str(_get(item, "issuedTo", "holder"), "")
            days   = _get(item, "overdueDays", "daysOverdue")
            line   = f"• **{name}**"
            if person:
                line += f" — Issued to: {person}"
            if days:
                line += f" ({days} days overdue)"
            lines.append(line)
        return "\n".join(lines)

    if subcategory == "PoorConditionEquipment":
        records = data if isinstance(data, list) else []
        if not records:
            return "No equipment returned in poor condition."
        lines = [f"### Poor Condition Equipment ({len(records)} items)\n"]
        for item in records:
            name      = _safe_str(_get(item, "name", "equipment", "itemName"))
            condition = _safe_str(_get(item, "condition", "state"), "Poor")
            lines.append(f"• **{name}** — Condition: {condition}")
        return "\n".join(lines)

    if isinstance(data, dict):
        lines = ["### Equipment Data\n"]
        for k, v in data.items():
            if not isinstance(v, (list, dict)):
                lines.append(f"• **{_camel_to_words(k)}:** {v}")
        return "\n".join(lines)
    if isinstance(data, list):
        lines = ["### Equipment Records\n"]
        for i, item in enumerate(data, 1):
            name = _safe_str(_get(item, "name", "equipment", "itemName"))
            lines.append(f"{i}. {name}")
        return "\n".join(lines)
    return str(data)


# =============================================================================
# DISTRIBUTION
# =============================================================================

def _format_distribution(subcategory: str, data: Any, intent_result: Dict) -> str:
    if subcategory == "LatestDistribution":
        if isinstance(data, dict):
            lines = ["### Latest Distribution\n"]
            for k, v in data.items():
                if not isinstance(v, (list, dict)):
                    lines.append(f"• **{_camel_to_words(k)}:** {v}")
            return "\n".join(lines)
        if isinstance(data, list):
            lines = ["### Latest Distribution Records\n"]
            for i, item in enumerate(data, 1):
                name = _safe_str(_get(item, "name", "Name"))
                lines.append(f"{i}. {name}")
            return "\n".join(lines)

    if subcategory == "DistributionByUnit":
        records = data if isinstance(data, list) else []
        if not records:
            return "No unit distribution data."
        lines = ["### Distribution by Unit\n"]
        for item in records:
            unit  = _safe_str(_get(item, "unit", "Unit", "unitName"))
            count = _get(item, "count", "quantity", "total")
            lines.append(f"• **{unit}:** {count}")
        return "\n".join(lines)

    if subcategory == "UnassignedItems":
        records = data if isinstance(data, list) else []
        if not records:
            return "All items have been assigned to units."
        lines = [f"### Unassigned Items ({len(records)})\n"]
        for item in records:
            name = _safe_str(_get(item, "name", "Name"))
            lines.append(f"• {name}")
        return "\n".join(lines)

    if subcategory == "TopUnit":
        if isinstance(data, dict):
            unit  = _safe_str(_get(data, "unit", "Unit", "unitName"))
            count = _get(data, "count", "quantity")
            return f"The **top unit** is **{unit}** with **{count}** items."

    if isinstance(data, dict):
        lines = ["### Distribution Summary\n"]
        for k, v in data.items():
            if not isinstance(v, (list, dict)):
                lines.append(f"• **{_camel_to_words(k)}:** {v}")
        return "\n".join(lines)
    return str(data)


# =============================================================================
# SKILLS
# =============================================================================

def _format_skills(subcategory: str, data: Any, intent_result: Dict) -> str:
    if subcategory == "BySport":
        records = data if isinstance(data, list) else []
        if not records:
            return "No sport data found."
        lines = ["### Roster by Sport\n"]
        for item in records:
            sport = _safe_str(_get(item, "sport", "name", "Sport"))
            count = _get(item, "count", "personnel", "total")
            lines.append(f"• **{sport}:** {count}")
        return "\n".join(lines)

    if subcategory == "ByClass":
        records = data if isinstance(data, list) else []
        if not records:
            return "No class data found."
        lines = ["### Roster by Class\n"]
        for item in records:
            cls   = _safe_str(_get(item, "class", "className", "name", "Class"))
            count = _get(item, "count", "personnel", "total")
            lines.append(f"• **{cls}:** {count}")
        return "\n".join(lines)

    if subcategory == "BloodGroup":
        records = data if isinstance(data, list) else []
        if not records:
            return "No blood group data found."
        lines = ["### Blood Group Distribution\n"]
        for item in records:
            bg    = _safe_str(_get(item, "bloodGroup", "blood", "group"))
            count = _get(item, "count", "total")
            lines.append(f"• **{bg}:** {count}")
        return "\n".join(lines)

    if isinstance(data, list):
        lines = ["### Skills / Roster\n"]
        for i, item in enumerate(data, 1):
            name = _safe_str(_get(item, "name", "Name", "fullName"))
            lines.append(f"{i}. {name}")
        return "\n".join(lines)
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

# Performance subcategories that contain per-agniveer nested data
_PERFORMANCE_NESTED_SUBCATEGORIES = {
    "TopPerformers", "LowestPerformers", "OverallPerformance",
    "Improvement", "Drop", "AttemptWise", "BestAttempt", "Comparison",
    "SectionSummary",
}

# Performance subcategories that return aggregate/simple data
_PERFORMANCE_AGGREGATE_SUBCATEGORIES = {
    "AverageScore", "PassPercentage", "FailPercentage",
    "GradeDistribution", "GradingSummary",
}


def _format_performance_aggregate(subcategory: str, data: Any, intent_result: Dict) -> str:
    """Handle simple aggregate performance responses (not per-person lists)."""
    section = intent_result.get("section", "")
    sec_str = f" for **{section}**" if section else ""

    if subcategory == "AverageScore":
        avg = (
            _get(data, "averageScore", "AverageScore", "average", "Average")
            if isinstance(data, dict) else data
        )
        return f"The average score{sec_str} is **{avg}**."

    if subcategory in ("PassPercentage", "FailPercentage"):
        label = "pass" if subcategory == "PassPercentage" else "fail"
        if isinstance(data, dict):
            pct   = _get(data, "percentage", "Percentage", "passPercentage", "failPercentage")
            total = _get(data, "total", "Total")
            total_str = f" (out of {total} total)" if total else ""
            return f"The {label} percentage{sec_str} is **{pct}%**{total_str}."
        return f"{label.title()} percentage{sec_str}: **{data}%**"

    if subcategory in ("GradeDistribution", "GradingSummary"):
        if isinstance(data, dict):
            lines = [f"### Grade Distribution{sec_str}\n"]
            for grade_key, count in data.items():
                if not isinstance(count, (list, dict)):
                    emoji = _grading_emoji(grade_key)
                    lines.append(f"{emoji} **{grade_key}:** {count}")
            return "\n".join(lines)
        if isinstance(data, list):
            lines = [f"### Grade Distribution{sec_str}\n"]
            for item in data:
                grade = _safe_str(_get(item, "grade", "Grade", "grading"))
                count = _get(item, "count", "Count", "total")
                emoji = _grading_emoji(grade)
                lines.append(f"{emoji} **{grade}:** {count}")
            return "\n".join(lines)

    # Fallback for aggregate data
    if isinstance(data, dict):
        lines = [f"### Performance Data{sec_str}\n"]
        for k, v in data.items():
            if not isinstance(v, (list, dict)):
                lines.append(f"• **{_camel_to_words(k)}:** {v}")
        return "\n".join(lines)
    return str(data)


# =============================================================================
# MAIN PUBLIC ENTRY POINT
# =============================================================================

def format_dotnet_response(
    dotnet_response: Any,
    intent_result: Dict,
) -> str:
    """
    Take the raw .NET response and the intent_result dict (Python snake_case keys),
    and return a structured human-readable markdown string.

    Handles both:
    1. The full .NET envelope: { "success": true, "commandLabel": "...", "data": [...] }
    2. Pre-unwrapped data (list or dict)
    """
    category    = intent_result.get("category", "")
    subcategory = intent_result.get("subcategory", "")

    # ── Handle .NET error shapes ───────────────────────────────────────────
    if isinstance(dotnet_response, dict):
        error_msg = _get(dotnet_response, "error", "Error", "errorMessage")
        if error_msg:
            return f"⚠️ The server returned an error: **{error_msg}**"

        # Check for .NET success=false
        if dotnet_response.get("success") is False:
            msg = _get(dotnet_response, "message", "Message") or "Unknown error."
            return f"⚠️ Request failed: **{msg}**"

    # ── Performance — handle the real nested structure ─────────────────────
    if category == "Performance":
        if subcategory in _PERFORMANCE_NESTED_SUBCATEGORIES:
            # Pass the full envelope — _format_performance_list handles unwrapping
            return _format_performance_list(dotnet_response, intent_result)

        # For aggregate subcategories, unwrap data first
        if isinstance(dotnet_response, dict):
            data = (
                _get(dotnet_response, "data", "Data", "result", "Result")
                or dotnet_response
            )
        else:
            data = dotnet_response

        if subcategory in _PERFORMANCE_AGGREGATE_SUBCATEGORIES:
            return _format_performance_aggregate(subcategory, data, intent_result)

        # Unknown performance subcategory — try nested then aggregate
        return _format_performance_list(dotnet_response, intent_result)

    # ── All other modules — unwrap data first ─────────────────────────────
    if isinstance(dotnet_response, dict):
        data = (
            _get(dotnet_response, "data", "Data", "result", "Result")
            or dotnet_response
        )
    else:
        data = dotnet_response

    formatter = _FORMATTERS.get(category)
    if formatter and subcategory:
        try:
            return formatter(subcategory, data, intent_result)
        except Exception as exc:
            # Don't crash — fall through to JSON dump
            pass

    # ── Ultimate fallback: pretty JSON ────────────────────────────────────
    try:
        return "```json\n" + json.dumps(data, indent=2, ensure_ascii=False) + "\n```"
    except Exception:
        return str(data)