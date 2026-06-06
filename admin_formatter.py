"""
admin_formatter.py
==================
Formats raw .NET AiCommand JSON responses into human-readable answers
for the admin chatbot UI.

The .NET API returns structured JSON. This module:
  1. Detects the response shape based on category/subcategory
  2. Converts numbers, lists and tables into clear prose or markdown
  3. Returns a formatted string ready to send to the admin frontend
"""

from __future__ import annotations

import json
import re as _re
from typing import Any, Dict, List, Optional


# =============================================================================
# HELPERS
# =============================================================================

def _safe_str(value: Any, fallback: str = "N/A") -> str:
    if value is None:
        return fallback
    return str(value).strip() or fallback


def _plural(n: int, singular: str, plural: Optional[str] = None) -> str:
    label = plural if (plural and n != 1) else singular
    return f"{n} {label}"


def _rank_list(items: List[Dict], name_key: str, score_key: str, label: str = "Score") -> str:
    if not items:
        return "No data found."
    lines = []
    for i, item in enumerate(items, start=1):
        name = _safe_str(item.get(name_key) or item.get("name") or item.get("Name"))
        score = _safe_str(item.get(score_key) or item.get("score") or item.get("Score"))
        lines.append(f"{i}. **{name}** — {label}: {score}")
    return "\n".join(lines)


def _key_value_block(data: Dict, keys: List[str]) -> str:
    lines = []
    for key in keys:
        value = data.get(key) or data.get(key.lower()) or data.get(key.upper())
        if value is not None:
            label = _re.sub(r"(?<!^)(?=[A-Z])", " ", key).title()
            lines.append(f"**{label}:** {value}")
    return "\n".join(lines) if lines else ""


def _camel_to_words(key: str) -> str:
    return _re.sub(r"(?<!^)(?=[A-Z])", " ", key).title()


def _generic_dict_format(data: Dict) -> str:
    lines = []
    for key, value in data.items():
        if isinstance(value, (list, dict)):
            continue
        label = _camel_to_words(str(key))
        lines.append(f"**{label}:** {value}")
    return "\n".join(lines)


def _generic_list_format(items: List, title: str = "") -> str:
    if not items:
        return "No records found."
    lines = [f"**{title}**" if title else ""]
    for i, item in enumerate(items, start=1):
        if isinstance(item, dict):
            name_candidates = (
                item.get("name") or item.get("Name") or
                item.get("personnelName") or item.get("PersonnelName") or
                item.get("agniveerName") or item.get("AgniveerName") or
                item.get("candidateName") or item.get("CandidateName") or
                str(i)
            )
            score_candidates = (
                item.get("score") or item.get("Score") or
                item.get("marks") or item.get("Marks") or
                item.get("percentage") or item.get("Percentage") or ""
            )
            if score_candidates:
                lines.append(f"{i}. **{name_candidates}** — {score_candidates}")
            else:
                lines.append(f"{i}. **{name_candidates}**")
        else:
            lines.append(f"{i}. {item}")
    return "\n".join(l for l in lines if l)


# =============================================================================
# MODULE-SPECIFIC FORMATTERS
# =============================================================================

def _format_performance(subcategory: str, data: Any, intent_result: Dict) -> str:
    n = intent_result.get("number") or 10

    if subcategory in ("TopPerformers", "LowestPerformers"):
        label = "Top" if subcategory == "TopPerformers" else "Lowest"
        if isinstance(data, list):
            intro = f"Here are the **{label} {len(data)} performers**:\n"
            return intro + _rank_list(data, "name", "score", "Score")
        if isinstance(data, dict):
            items = (
                data.get("performers") or data.get("Performers") or
                data.get("data") or data.get("Data") or []
            )
            intro = f"Here are the **{label} performers**:\n"
            return intro + _rank_list(items, "name", "score", "Score")

    if subcategory == "AverageScore":
        if isinstance(data, dict):
            avg = (
                data.get("averageScore") or data.get("AverageScore") or
                data.get("average") or data.get("Average") or "N/A"
            )
            section = intent_result.get("section", "")
            section_str = f" for section **{section}**" if section else ""
            return f"The average score{section_str} is **{avg}**."
        return f"Average score: **{data}**"

    if subcategory in ("PassPercentage", "FailPercentage"):
        label = "pass" if subcategory == "PassPercentage" else "fail"
        if isinstance(data, dict):
            pct = (
                data.get("percentage") or data.get("Percentage") or
                data.get("passPercentage") or data.get("failPercentage") or "N/A"
            )
            total = data.get("total") or data.get("Total") or ""
            total_str = f" (out of {total} total)" if total else ""
            return f"The {label} percentage is **{pct}%**{total_str}."
        return f"{label.title()} percentage: **{data}%**"

    if subcategory in ("GradeDistribution", "GradingSummary"):
        if isinstance(data, dict):
            lines = ["**Grade Distribution:**\n"]
            for grade_key, count in data.items():
                lines.append(f"  • **{grade_key}:** {count}")
            return "\n".join(lines)
        if isinstance(data, list):
            lines = ["**Grade Distribution:**\n"]
            for item in data:
                grade = item.get("grade") or item.get("Grade") or "Unknown"
                count = item.get("count") or item.get("Count") or 0
                lines.append(f"  • **{grade}:** {count}")
            return "\n".join(lines)

    if subcategory == "SectionSummary":
        if isinstance(data, list):
            lines = ["**Section Summary:**\n"]
            for item in data:
                section = item.get("section") or item.get("Section") or "N/A"
                avg = item.get("averageScore") or item.get("average") or item.get("score") or "N/A"
                lines.append(f"  • **{section}:** Avg {avg}")
            return "\n".join(lines)

    if subcategory == "Comparison":
        if isinstance(data, list):
            return "**Performance Comparison:**\n" + _generic_list_format(data)

    if subcategory == "OverallPerformance":
        if isinstance(data, dict):
            return "**Overall Performance Summary:**\n\n" + _generic_dict_format(data)

    # Fallback
    if isinstance(data, list):
        return _generic_list_format(data, title="Performance Data")
    if isinstance(data, dict):
        return "**Performance Summary:**\n\n" + _generic_dict_format(data)
    return str(data)


def _format_leave(subcategory: str, data: Any, intent_result: Dict) -> str:
    if subcategory in ("MostLeaveTaken", "LeastLeaveTaken"):
        label = "most" if subcategory == "MostLeaveTaken" else "least"
        leave_type = intent_result.get("leave_type", "")
        leave_str = f" ({leave_type})" if leave_type else ""
        if isinstance(data, list):
            intro = f"Personnel who have taken the **{label} leave{leave_str}**:\n"
            return intro + _rank_list(data, "name", "leaveDays", "Days")
        if isinstance(data, dict):
            items = data.get("personnel") or data.get("data") or []
            intro = f"Personnel who have taken the **{label} leave{leave_str}**:\n"
            return intro + _rank_list(items, "name", "leaveDays", "Days")

    if subcategory == "CurrentLeaveStatus":
        if isinstance(data, list):
            if not data:
                return "No personnel are currently on leave."
            lines = [f"**{len(data)} personnel currently on leave:**\n"]
            for item in data:
                name = _safe_str(item.get("name") or item.get("Name"))
                lt = _safe_str(item.get("leaveType") or item.get("type") or "")
                date = _safe_str(item.get("from") or item.get("startDate") or "")
                line = f"  • **{name}**"
                if lt:
                    line += f" — {lt}"
                if date:
                    line += f" (from {date})"
                lines.append(line)
            return "\n".join(lines)
        if isinstance(data, dict):
            total = data.get("total") or data.get("count") or 0
            return f"**{total}** personnel are currently on leave."

    if subcategory == "AbscondedPersonnel":
        if isinstance(data, list):
            if not data:
                return "No absconded personnel on record."
            lines = [f"**{len(data)} absconded personnel:**\n"]
            for item in data:
                name = _safe_str(item.get("name") or item.get("Name"))
                since = _safe_str(item.get("since") or item.get("date") or "")
                line = f"  • **{name}**"
                if since:
                    line += f" — since {since}"
                lines.append(line)
            return "\n".join(lines)

    if isinstance(data, list):
        return _generic_list_format(data, title="Leave Data")
    if isinstance(data, dict):
        return "**Leave Summary:**\n\n" + _generic_dict_format(data)
    return str(data)


def _format_medical(subcategory: str, data: Any, intent_result: Dict) -> str:
    if subcategory == "ActiveCases":
        if isinstance(data, list):
            if not data:
                return "No active medical cases at the moment."
            lines = [f"**{len(data)} active medical case(s):**\n"]
            for item in data:
                name = _safe_str(item.get("name") or item.get("Name"))
                disease = _safe_str(item.get("disease") or item.get("diagnosis") or item.get("condition") or "Unknown")
                ward = _safe_str(item.get("ward") or "")
                line = f"  • **{name}** — {disease}"
                if ward:
                    line += f" (Ward: {ward})"
                lines.append(line)
            return "\n".join(lines)
        if isinstance(data, dict):
            total = data.get("total") or data.get("count") or 0
            return f"There are currently **{total}** active medical case(s)."

    if subcategory == "BMIAnalysis":
        if isinstance(data, dict):
            lines = ["**BMI / Fitness Analysis:**\n"]
            for key, val in data.items():
                lines.append(f"  • **{_camel_to_words(key)}:** {val}")
            return "\n".join(lines)
        if isinstance(data, list):
            return "**BMI Analysis:**\n" + _rank_list(data, "name", "bmi", "BMI")

    if subcategory == "DiseaseStatistics":
        if isinstance(data, list):
            lines = ["**Top Disease Statistics:**\n"]
            for i, item in enumerate(data, 1):
                disease = _safe_str(item.get("disease") or item.get("name") or item.get("condition"))
                count = _safe_str(item.get("count") or item.get("cases"))
                lines.append(f"  {i}. **{disease}** — {count} cases")
            return "\n".join(lines)

    if isinstance(data, list):
        return _generic_list_format(data, title="Medical Data")
    if isinstance(data, dict):
        return "**Medical Summary:**\n\n" + _generic_dict_format(data)
    return str(data)


def _format_attendance(subcategory: str, data: Any, intent_result: Dict) -> str:
    if subcategory == "MonthlyAttendance":
        if isinstance(data, dict):
            lines = ["**Monthly Attendance Summary:**\n"]
            for key, val in data.items():
                lines.append(f"  • **{_camel_to_words(key)}:** {val}")
            return "\n".join(lines)
        if isinstance(data, list):
            lines = ["**Monthly Attendance:**\n"]
            for item in data:
                month = _safe_str(item.get("month") or item.get("Month"))
                pct = _safe_str(item.get("percentage") or item.get("attendancePercentage"))
                lines.append(f"  • **{month}:** {pct}%")
            return "\n".join(lines)

    if subcategory == "PresentToday":
        if isinstance(data, dict):
            present = data.get("present") or data.get("Present") or data.get("count") or 0
            total = data.get("total") or data.get("Total") or ""
            total_str = f" out of {total}" if total else ""
            return f"**{present}** personnel are present on campus today{total_str}."
        if isinstance(data, (int, float)):
            return f"**{data}** personnel are present on campus today."

    if subcategory == "StrengthBreakdown":
        if isinstance(data, dict):
            lines = ["**Strength Breakdown:**\n"]
            for key, val in data.items():
                lines.append(f"  • **{_camel_to_words(key)}:** {val}")
            return "\n".join(lines)

    if isinstance(data, dict):
        return "**Attendance Summary:**\n\n" + _generic_dict_format(data)
    return str(data)


def _format_verification(subcategory: str, data: Any, intent_result: Dict) -> str:
    label = "Pending" if subcategory == "PendingVerification" else "Completed"
    if isinstance(data, list):
        if not data:
            return f"No {label.lower()} verifications found."
        lines = [f"**{len(data)} {label} verification(s):**\n"]
        for item in data:
            name = _safe_str(item.get("name") or item.get("Name"))
            doc = _safe_str(item.get("documentType") or item.get("document") or "")
            line = f"  • **{name}**"
            if doc:
                line += f" — {doc}"
            lines.append(line)
        return "\n".join(lines)
    if isinstance(data, dict):
        total = data.get("total") or data.get("count") or 0
        return f"**{total}** {label.lower()} verification(s) found."
    return str(data)


def _format_equipment(subcategory: str, data: Any, intent_result: Dict) -> str:
    if subcategory == "EquipmentSummary":
        if isinstance(data, dict):
            lines = ["**Equipment Summary:**\n"]
            for key, val in data.items():
                lines.append(f"  • **{_camel_to_words(key)}:** {val}")
            return "\n".join(lines)

    if subcategory == "OverdueEquipment":
        if isinstance(data, list):
            if not data:
                return "No overdue equipment records."
            lines = [f"**{len(data)} overdue equipment item(s):**\n"]
            for item in data:
                name = _safe_str(item.get("name") or item.get("equipment") or item.get("itemName"))
                person = _safe_str(item.get("issuedTo") or item.get("holder") or "")
                days = _safe_str(item.get("overdueDays") or item.get("daysOverdue") or "")
                line = f"  • **{name}**"
                if person:
                    line += f" — Issued to: {person}"
                if days:
                    line += f" ({days} days overdue)"
                lines.append(line)
            return "\n".join(lines)

    if subcategory == "PoorConditionEquipment":
        if isinstance(data, list):
            if not data:
                return "No equipment returned in poor condition."
            lines = [f"**{len(data)} item(s) in poor condition:**\n"]
            for item in data:
                name = _safe_str(item.get("name") or item.get("equipment") or item.get("itemName"))
                condition = _safe_str(item.get("condition") or item.get("state") or "Poor")
                lines.append(f"  • **{name}** — Condition: {condition}")
            return "\n".join(lines)

    if isinstance(data, dict):
        return "**Equipment Data:**\n\n" + _generic_dict_format(data)
    if isinstance(data, list):
        return _generic_list_format(data, title="Equipment Records")
    return str(data)


def _format_distribution(subcategory: str, data: Any, intent_result: Dict) -> str:
    if subcategory == "LatestDistribution":
        if isinstance(data, dict):
            return "**Latest Distribution:**\n\n" + _generic_dict_format(data)
        if isinstance(data, list):
            return "**Latest Distribution Records:**\n" + _generic_list_format(data)

    if subcategory == "DistributionByUnit":
        if isinstance(data, list):
            lines = ["**Distribution by Unit:**\n"]
            for item in data:
                unit = _safe_str(item.get("unit") or item.get("Unit"))
                count = _safe_str(item.get("count") or item.get("quantity") or "")
                lines.append(f"  • **{unit}:** {count}")
            return "\n".join(lines)

    if subcategory == "UnassignedItems":
        if isinstance(data, list):
            if not data:
                return "All items have been assigned to units."
            return f"**{len(data)} unassigned item(s):**\n" + _generic_list_format(data)

    if subcategory == "TopUnit":
        if isinstance(data, dict):
            unit = _safe_str(data.get("unit") or data.get("Unit"))
            count = _safe_str(data.get("count") or data.get("quantity") or "")
            return f"The **top unit** for distribution is **{unit}** with **{count}** items."

    if isinstance(data, dict):
        return "**Distribution Summary:**\n\n" + _generic_dict_format(data)
    if isinstance(data, list):
        return _generic_list_format(data, title="Distribution Records")
    return str(data)


def _format_skills(subcategory: str, data: Any, intent_result: Dict) -> str:
    if subcategory == "BySport":
        if isinstance(data, list):
            lines = ["**Skills by Sport:**\n"]
            for item in data:
                sport = _safe_str(item.get("sport") or item.get("name"))
                count = _safe_str(item.get("count") or item.get("personnel") or "")
                lines.append(f"  • **{sport}:** {count}")
            return "\n".join(lines)

    if subcategory == "ByClass":
        if isinstance(data, list):
            lines = ["**Skills by Class:**\n"]
            for item in data:
                cls = _safe_str(item.get("class") or item.get("className") or item.get("name"))
                count = _safe_str(item.get("count") or item.get("personnel") or "")
                lines.append(f"  • **{cls}:** {count}")
            return "\n".join(lines)

    if isinstance(data, dict):
        return "**Skills / Roster Data:**\n\n" + _generic_dict_format(data)
    if isinstance(data, list):
        return _generic_list_format(data, title="Skills / Roster")
    return str(data)


# =============================================================================
# DISPATCH
# =============================================================================

_FORMATTERS = {
    "Performance":  _format_performance,
    "Leave":        _format_leave,
    "Medical":      _format_medical,
    "Attendance":   _format_attendance,
    "Verification": _format_verification,
    "Equipment":    _format_equipment,
    "Distribution": _format_distribution,
    "Skills":       _format_skills,
}


def format_dotnet_response(
    dotnet_response: Any,
    intent_result: Dict,
) -> str:
    """
    Take the raw .NET response (dict, list, or primitive) and the original
    intent_result dict, and return a human-readable answer string.
    """
    category    = intent_result.get("category", "")
    subcategory = intent_result.get("subcategory", "")

    # Handle error shapes from .NET
    if isinstance(dotnet_response, dict):
        if dotnet_response.get("error") or dotnet_response.get("Error"):
            err_msg = (
                dotnet_response.get("error") or dotnet_response.get("Error") or
                dotnet_response.get("message") or "Unknown error from server."
            )
            return f"The server returned an error: **{err_msg}**"
        # Unwrap common wrapper shapes
        data = (
            dotnet_response.get("data") or
            dotnet_response.get("Data") or
            dotnet_response.get("result") or
            dotnet_response.get("Result") or
            dotnet_response
        )
    else:
        data = dotnet_response

    formatter = _FORMATTERS.get(category)
    if formatter and subcategory:
        try:
            return formatter(subcategory, data, intent_result)
        except Exception:
            pass

    # Ultimate fallback: pretty-print JSON
    try:
        return "```json\n" + json.dumps(data, indent=2, ensure_ascii=False) + "\n```"
    except Exception:
        return str(data)