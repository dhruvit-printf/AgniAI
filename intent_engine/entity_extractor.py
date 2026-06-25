"""
entity_extractor.py
===================

Single responsibility: extract entities from the user query.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

from .intent_schema import (
    BLOOD_GROUPS,
    BMI_CATEGORIES,
    CLASSES,
    GRADING_CATEGORIES,
    ISSUED_EQUIPMENT_ITEMS,
    LEAVE_TYPES,
    PROCURED_EQUIPMENT_ITEMS,
    SECTION,
    SPORTS,
    SUBSECTIONS_BY_SECTION,
)
from query_understanding_engine import understand_query


def _extract_number(query: str) -> Optional[int]:
    match = re.search(r"\b(?:top|bottom|highest|lowest|worst|best)\s+(\d+)\b", query.lower())
    if match:
        return int(match.group(1))
    return None


def _extract_section(query: str) -> Optional[str]:
    query_lower = query.lower()
    for section_name, section_data in SECTION.items():
        for alias in section_data["aliases"]:
            pattern = r"\b" + re.escape(alias).replace(r"\ ", r"\s+") + r"\b"
            if re.search(pattern, query_lower, re.IGNORECASE):
                return section_name
    return None


def _extract_subsection(query: str, section: Optional[str]) -> Optional[str]:
    if not section or section not in SUBSECTIONS_BY_SECTION:
        return None
    query_lower = query.lower()
    for subsection in SUBSECTIONS_BY_SECTION[section]:
        if subsection.lower() in query_lower:
            return subsection
    return None


def _extract_grading(query: str) -> Optional[str]:
    query_lower = query.lower()
    for key, value in GRADING_CATEGORIES.items():
        if key in query_lower:
            return value
    return None


def _extract_leave_type(query: str) -> Optional[str]:
    query_lower = query.lower()
    if "leave" not in query_lower and "abscond" not in query_lower and "absent" not in query_lower and "status" not in query_lower and "medical leave" not in query_lower:
        return None
    for key, value in LEAVE_TYPES.items():
        if key in query_lower:
            if key == "medical" and "medical leave" not in query_lower:
                continue
            return value
    return None


def _extract_bmi_category(query: str) -> Optional[str]:
    query_lower = query.lower()
    for key, value in BMI_CATEGORIES.items():
        if key in query_lower:
            return value
    return None


def _extract_blood_group(query: str) -> Optional[str]:
    query_lower = query.lower()
    for blood_group in BLOOD_GROUPS:
        variants = [
            blood_group,
            blood_group.replace("+", " positive"),
            blood_group.replace("-", " negative"),
        ]
        if any(variant.lower() in query_lower for variant in variants):
            return blood_group
    return None


def _extract_sport(query: str) -> Optional[str]:
    query_lower = query.lower()
    for key, value in SPORTS.items():
        if key in query_lower:
            return value
    return None


def _extract_class(query: str) -> Optional[str]:
    query_lower = query.lower()
    for key, value in CLASSES.items():
        if key in query_lower:
            return value
    return None


def _extract_equipment_item(query: str) -> Optional[str]:
    query_lower = query.lower()
    for item in ISSUED_EQUIPMENT_ITEMS:
        if item.lower() in query_lower:
            return item
    for item in PROCURED_EQUIPMENT_ITEMS:
        if item.lower() in query_lower:
            return item
    return None


def _extract_date_patterns(query: str) -> Optional[str]:
    match = re.search(r"(\d{4}-\d{2}-\d{2})", query)
    if match:
        return match.group(1)

    match = re.search(r"(\d{1,2}[/-]\d{1,2}[/-]\d{4})", query)
    if match:
        return match.group(1)

    month_pattern = (
        r"\b(January|February|March|April|May|June|July|August|September|October|"
        r"November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{4})\b"
    )
    match = re.search(month_pattern, query, re.IGNORECASE)
    if match:
        return f"{match.group(1)} {match.group(2)}"

    match = re.search(r"\bdate\s+(\d{1,2})\b", query, re.IGNORECASE)
    if match:
        return match.group(1)

    return None


def _extract_date_range(query: str) -> Tuple[Optional[str], Optional[str]]:
    query_lower = query.lower()
    match = re.search(
        r"\bfrom\s+(.+?)\s+(?:to|until)\s+(.+?)(?:$|[,.?])",
        query_lower,
        re.IGNORECASE,
    )
    if not match:
        return None, None
    return match.group(1).strip(), match.group(2).strip()


def _extract_attempt_no(query: str) -> Optional[int]:
    match = re.search(r"\battempt\s+(\d+)\b", query.lower())
    if match:
        return int(match.group(1))
    ordinals = {"first": 1, "second": 2, "third": 3}
    for ordinal, num in ordinals.items():
        if f"{ordinal} attempt" in query.lower():
            return num
    return None


def _extract_from_attempt(query: str) -> Optional[int]:
    match = re.search(r"(?:from|between)\s+attempt\s+(\d+)", query.lower())
    if match:
        return int(match.group(1))
    match = re.search(r"\battempt\s+(\d+)\s+to\s+attempt", query.lower())
    if match:
        return int(match.group(1))
    return None


def _extract_to_attempt(query: str) -> Optional[int]:
    match = re.search(r"(?:to|until)\s+attempt\s+(\d+)", query.lower())
    if match:
        return int(match.group(1))
    match = re.search(r"\battempt\s+\d+\s+to\s+(\d+)", query.lower())
    if match:
        return int(match.group(1))
    return None


def _extract_unit_name(query: str) -> Optional[str]:
    unit_map = {
        "alpha": "Alpha Unit",
        "bravo": "Bravo Unit",
        "charlie": "Charlie Unit",
        "delta": "Delta Unit",
        "echo": "Echo Unit",
        "foxtrot": "Foxtrot Unit",
        "golf": "Golf Unit",
        "hotel": "Hotel Unit",
        "india": "India Unit",
        "juliet": "Juliet Unit",
        "kilo": "Kilo Unit",
        "lima": "Lima Unit",
        "mike": "Mike Unit",
        "november": "November Unit",
        "oscar": "Oscar Unit",
        "papa": "Papa Unit",
        "quebec": "Quebec Unit",
        "romeo": "Romeo Unit",
        "sierra": "Sierra Unit",
        "tango": "Tango Unit",
        "uniform": "Uniform Unit",
        "victor": "Victor Unit",
        "whiskey": "Whiskey Unit",
        "xray": "Xray Unit",
        "yankee": "Yankee Unit",
        "zulu": "Zulu Unit",
    }

    match = re.search(
        r"(?:\b(?:in|from|for)\s+unit\s+([A-Za-z]+)\b|\bunit\s+([A-Za-z]+)\b|\b([A-Za-z]+)\s+unit\b)",
        query,
        re.IGNORECASE,
    )
    if match:
        token = (match.group(1) or match.group(2) or match.group(3) or "").lower()
        if token in unit_map:
            return unit_map[token]
        if len(token) == 1:
            return f"Unit {token.upper()}"
        return f"{token.capitalize()} Unit"
    return None


def _extract_numeric_id(query: str, id_pattern: str) -> Optional[int]:
    match = re.search(rf"\b(?:{id_pattern})\s+(\d+)\b", query, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def _extract_company_id(query: str) -> Optional[int]:
    return _extract_numeric_id(query, "company|coy")


def _extract_platoon_id(query: str) -> Optional[int]:
    return _extract_numeric_id(query, "platoon|plt")


def _extract_batch_id(query: str) -> Optional[int]:
    return _extract_numeric_id(query, "batch")


def _extract_agniveer_no(query: str) -> Optional[str]:
    match = re.search(r"agniveer\s+(?:no\.?|no\.?\s+)?(\d+|[A-Z]\d+)", query, re.IGNORECASE)
    if match:
        return match.group(1)
    match = re.search(r"\b([A-Z]\d{5,8}[A-Z]?)\b", query)
    if match:
        return match.group(1).upper()
    return None


def _extract_medical_status(query: str) -> Optional[str]:
    query_lower = query.lower()
    if "active medical" in query_lower or "active case" in query_lower or "active cases" in query_lower:
        return "Active"
    if any(token in query_lower for token in ("admitted", "under treatment", "in hospital")):
        return "Active"
    return None


def extract_entities(
    query: str,
    resolved_entities: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    query = str(query).strip()
    resolved_entities = resolved_entities or {}
    semantic = understand_query(query)

    result: Dict[str, Any] = {
        "batchId": None,
        "platoonId": None,
        "companyId": None,
        "agniveerNo": None,
        "section": None,
        "subSection": None,
        "attemptNo": None,
        "fromAttempt": None,
        "toAttempt": None,
        "leaveType": None,
        "grading": None,
        "bmiCategory": None,
        "bloodGroup": None,
        "equipmentName": None,
        "sport": None,
        "class": None,
        "unitName": None,
        "n": None,
        "date": None,
        "fromDate": None,
        "toDate": None,
        "medical_status": None,
    }

    result["n"] = _extract_number(query)
    result["section"] = _extract_section(query)
    if result["section"]:
        result["subSection"] = _extract_subsection(query, result["section"])
    result["grading"] = _extract_grading(query)
    result["leaveType"] = _extract_leave_type(query)
    result["bmiCategory"] = _extract_bmi_category(query)
    result["bloodGroup"] = _extract_blood_group(query)
    result["sport"] = _extract_sport(query)
    result["class"] = _extract_class(query)
    result["equipmentName"] = _extract_equipment_item(query)
    result["unitName"] = _extract_unit_name(query)
    result["attemptNo"] = _extract_attempt_no(query)
    result["fromAttempt"] = _extract_from_attempt(query)
    result["toAttempt"] = _extract_to_attempt(query)
    result["date"] = _extract_date_patterns(query)
    result["fromDate"], result["toDate"] = _extract_date_range(query)
    result["medical_status"] = _extract_medical_status(query)

    result["companyId"] = (
        resolved_entities.get("companyId")
        or resolved_entities.get("company_id")
        or semantic.get("company_id")
        or _extract_company_id(query)
    )
    result["platoonId"] = (
        resolved_entities.get("platoonId")
        or resolved_entities.get("platoon_id")
        or semantic.get("platoon_id")
        or _extract_platoon_id(query)
    )
    result["batchId"] = (
        resolved_entities.get("batchId")
        or resolved_entities.get("batch_id")
        or semantic.get("batch_id")
        or _extract_batch_id(query)
    )
    result["agniveerNo"] = (
        resolved_entities.get("agniveerNo")
        or resolved_entities.get("agniveer_no")
        or _extract_agniveer_no(query)
    )

    return result
