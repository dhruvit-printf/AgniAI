"""
utils.py
========
Shared micro-utilities for normalization and safe scalar parsing.

This module is intentionally small and dependency-light so core pipeline
modules can import it without creating cycles.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

_WRAPPER_KEY_CANDIDATES = (
    "data",
    "Data",
    "result",
    "Result",
    "records",
    "Records",
    "persons",
    "personnel",
)

_RECORD_KEY_CANDIDATES = (
    "agniveerNo",
    "agniveerId",
    "AgniveerId",
    "AgniVeerId",
    "id",
    "Id",
)

_SCORE_FIELDS = (
    "bestTotal",
    "totalMarks",
    "score",
    "Score",
    "omrInputTotal",
    "marksObtained",
)


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_confidence(value: Any, fallback: float = 0.85) -> float:
    if isinstance(value, str):
        lowered = value.lower()
        if "high" in lowered:
            return 0.95
        if "medium" in lowered:
            return 0.70
        if "low" in lowered:
            return 0.30
        parsed = safe_float(value)
        return parsed if parsed is not None else float(fallback)
    if isinstance(value, (int, float)):
        return float(value)
    return float(fallback)


def extract_records(data: Any) -> List[Dict]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if not isinstance(data, dict):
        return []

    # Check comparison sides
    left = data.get("left")
    right = data.get("right")
    if isinstance(left, dict) or isinstance(right, dict):
        res = []
        if isinstance(left, dict):
            res.extend(extract_records(left))
        if isinstance(right, dict):
            res.extend(extract_records(right))
        if res:
            return res

    # Check multi independent sections
    sections = data.get("sections")
    if isinstance(sections, list):
        res = []
        for sec in sections:
            if isinstance(sec, dict):
                res.extend(extract_records(sec))
        if res:
            return res

    # Check standard wrapper candidates
    for key in _WRAPPER_KEY_CANDIDATES:
        val = data.get(key)
        if isinstance(val, list):
            res = [item for item in val if isinstance(item, dict)]
            if res:
                return res
        if isinstance(val, dict):
            nested = extract_records(val)
            if nested:
                return nested

    teams = data.get("teams") or data.get("Teams")
    if isinstance(teams, list):
        members: List[Dict] = []
        for team in teams:
            if isinstance(team, dict):
                team_members = team.get("members") or team.get("Members") or []
                if isinstance(team_members, list):
                    members.extend(
                        member for member in team_members if isinstance(member, dict)
                    )
        if members:
            return members

    # Fallback recursive check on other values
    for value in data.values():
        if isinstance(value, (dict, list)):
            nested = extract_records(value)
            if nested:
                return nested

    return []


def get_score(record: Dict[str, Any]) -> Optional[float]:
    for field in _SCORE_FIELDS:
        score = safe_float(record.get(field))
        if score is not None:
            return score
    return None


def extract_record_id(record: Dict[str, Any]) -> Optional[str]:
    for key in _RECORD_KEY_CANDIDATES:
        value = record.get(key)
        if value is not None:
            return safe_str(value)
    return None


def build_filters_from_entities(entities: Dict[str, Any]) -> Dict[str, Any]:
    """Build a filters dictionary from entities dictionary using camelCase keys."""
    filters = {}
    keys = (
        "section", "subSection", "grading", "leaveType", "sport", "class",
        "unitName", "attemptNo", "fromAttempt", "toAttempt", "date",
        "companyId", "platoonId", "batchId", "agniveerNo", "bmiCategory",
        "bloodGroup", "equipmentName"
    )
    for key in keys:
        val = entities.get(key)
        if val is not None:
            filters[key] = val

    if entities.get("leaveType") == "Current":
        filters["leaveStatus"] = "Current"

    return filters


def has_any_data(records: List[Dict[str, Any]]) -> bool:
    """Check if the list of records contains any non-empty data."""
    if not records:
        return False
    for r in records:
        if any(v is not None for v in r.values()):
            return True
    return False
