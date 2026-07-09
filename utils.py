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
    "present",
    "absent",
    "leaveDays",
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

    # Single-entity, multi-section shape: one sibling dict IS the record
    # (has its own canonical ID, e.g. "profile") while other dict siblings
    # are scalar-only descriptive sections (e.g. "bmi", "stats") and any
    # list sibling is an unrelated collection with a different cardinality
    # (e.g. "medicalHistory" — one row per visit, not one row per person).
    # Without this check, the "direct list of dicts anywhere" scan below
    # would grab that unrelated list and silently discard the entity's own
    # fields — merge the entity + its sections into one row instead, and
    # summarize any sibling list as a `<key>Count` column.
    primary_key = next(
        (
            key
            for key, value in data.items()
            if isinstance(value, dict) and extract_record_id(value) is not None
        ),
        None,
    )
    if primary_key is not None:
        merged: Dict[str, Any] = {}
        for key, value in data.items():
            if key == primary_key and isinstance(value, dict):
                merged.update(value)
            elif isinstance(value, dict) and not _contains_record_list(value):
                merged.update(value)
            elif isinstance(value, list):
                list_records = [item for item in value if isinstance(item, dict)]
                if list_records:
                    merged[f"{key}Count"] = len(list_records)
        if merged and has_any_data([merged]):
            return [merged]

    # A direct list of dicts anywhere in this object is a genuine multi-row
    # result (e.g. Schedule's "byCompany") — check every value for one
    # before considering any dict sibling, since descending into just the
    # first dict found (below) would discard the list otherwise.
    for value in data.values():
        if isinstance(value, list):
            res = [item for item in value if isinstance(item, dict)]
            if res:
                return res

    # Exactly one dict sibling actually contains a record list buried
    # deeper inside it → genuine envelope, safe to unwrap by recursing.
    # Sibling dicts made purely of scalars (e.g. Equipment Stats'
    # "issued"/"procured" groups) do NOT count: recursing into just one of
    # those would silently discard every other sibling group (and any
    # top-level scalars like "totalAssigned") — exactly the bug where
    # "procured" and "totalAssigned" vanished and only "issued" survived.
    dict_children_with_lists = [
        v for v in data.values() if isinstance(v, dict) and _contains_record_list(v)
    ]
    if len(dict_children_with_lists) == 1:
        nested = extract_records(dict_children_with_lists[0])
        if nested:
            return nested

    # No genuine record list anywhere, but this object has one or more
    # nested dicts (e.g. "issued": {...}, "procured": {...}) — it's a
    # single aggregate/summary row made of named sub-groups, not a list of
    # many similar records. Keep the whole object — every sibling group
    # intact — so the widget flattener can turn each one into its own
    # prefixed columns instead of losing all but the first.
    if any(isinstance(value, dict) for value in data.values()) and has_any_data([data]):
        return [data]

    # Flat aggregate/summary dict — no nested list/dict found anywhere in
    # this object, but it carries real scalar content (e.g. a Summary-shaped
    # .NET response like {"totalRecords": 42, "averageScore": 71.3}).
    # Treat it as a single record instead of silently returning [].
    if not any(isinstance(value, (dict, list)) for value in data.values()) and has_any_data([data]):
        return [data]

    return []


def _contains_record_list(value: Any) -> bool:
    """Whether a record list (a list containing at least one dict) exists
    anywhere inside this value, however deeply nested."""
    if isinstance(value, list):
        return any(isinstance(item, dict) for item in value)
    if isinstance(value, dict):
        return any(_contains_record_list(v) for v in value.values())
    return False


def get_score(record: Dict[str, Any]) -> Optional[float]:
    for field in _SCORE_FIELDS:
        score = safe_float(record.get(field))
        if score is not None:
            return score
    return None


def extract_chronological_key(record: Dict[str, Any]) -> Any:
    """Resolve a sortable chronological key from a record's date/attempt/month-year fields."""
    for k in ("date", "Date", "createdDate", "CreatedDate", "timestamp", "Timestamp"):
        val = record.get(k)
        if val:
            return str(val)
    for k in ("attempt", "attemptNo", "Attempt", "AttemptNo"):
        val = safe_float(record.get(k))
        if val is not None:
            return val
    month = record.get("month") or record.get("Month")
    year = record.get("year") or record.get("Year")
    if year is not None:
        if month is not None:
            return (year, month)
        return year
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
        "section",
        "subSection",
        "grading",
        "leaveType",
        "sport",
        "class",
        "unitName",
        "attemptNo",
        "fromAttempt",
        "toAttempt",
        "date",
        "companyId",
        "platoonId",
        "batchId",
        "agniveerNo",
        "bmiCategory",
        "bloodGroup",
        "equipmentName",
    )
    for key in keys:
        val = entities.get(key)
        if val is not None:
            filters[key] = val

    if entities.get("leaveType") == "Current":
        filters["leaveStatus"] = "Current"

    return filters


_ATTENTION_KEYWORDS = (
    "overdue", "pending", "rejected", "disqualified", "absconded", "poor",
    "damaged", "lost", "expired", "failed", "not responded", "notresponded",
    "awol", "missing", "unassigned", "declined",
)

_CATEGORICAL_SKIP_HINTS = (
    "id", "no", "number", "name", "date", "time", "url", "link", "guid",
    "email", "phone", "remark", "comment", "description", "address",
)


def categorical_breakdown(
    records: List[Dict[str, Any]],
    limit: int = 5,
) -> Optional[Dict[str, Any]]:
    """
    Find the most informative low-cardinality string field across *records*
    (e.g. status/condition/grade) and return its value distribution.

    Schema-agnostic — makes no assumption about field names, so it works for
    any category (equipment, skills, medical, ...) not just score-bearing
    ones. Returns None if no informative field is found.
    """
    if not records:
        return None
    n = len(records)
    max_card = max(2, min(8, n))

    field_values: Dict[str, List[str]] = {}
    for r in records:
        if not isinstance(r, dict):
            continue
        for k, v in r.items():
            key = k.lower()
            if any(hint in key for hint in _CATEGORICAL_SKIP_HINTS):
                continue
            if isinstance(v, bool) or v is None:
                continue
            if isinstance(v, str) and v.strip():
                field_values.setdefault(k, []).append(v.strip())

    best_field: Optional[str] = None
    best_values: List[str] = []
    best_card: Optional[int] = None
    for field, values in field_values.items():
        if len(values) < max(2, n * 0.6):
            continue  # too sparse to be meaningful
        distinct = set(values)
        card = len(distinct)
        if card < 2 or card > max_card:
            continue
        if best_card is None or card < best_card:
            best_field, best_values, best_card = field, values, card

    if not best_field:
        return None

    counts: Dict[str, int] = {}
    for v in best_values:
        counts[v] = counts.get(v, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])[:limit]

    attention = [
        {"value": val, "count": cnt}
        for val, cnt in ranked
        if any(kw in val.lower() for kw in _ATTENTION_KEYWORDS)
    ]

    return {
        "field": best_field,
        "breakdown": [{"value": v, "count": c} for v, c in ranked],
        "total": len(best_values),
        "attention": attention,
    }


def numeric_distribution_breakdown(
    records: List[Dict[str, Any]],
    limit: int = 8,
) -> Optional[Dict[str, Any]]:
    """
    Detect a pre-aggregated category->count shape, e.g. a Grading Summary
    like {"DRILL (AMT)": {"Excellent": 188, "Good": 393, "SAT": 1}}.

    `categorical_breakdown` only looks at string-valued fields across many
    records, so it can't see this — here a single record's field holds a
    dict whose *values* are already numeric counts per category. Sum those
    leaves to get the real total (e.g. 582 graded, not "1 record") instead
    of letting callers fall through to `len(records)`.
    """
    label: Optional[str] = None
    counts: Dict[str, float] = {}

    for r in records:
        if not isinstance(r, dict):
            continue
        for k, v in r.items():
            if not isinstance(v, dict) or not v:
                continue
            leaves = {
                str(sub_k): sub_v
                for sub_k, sub_v in v.items()
                if isinstance(sub_v, (int, float)) and not isinstance(sub_v, bool)
            }
            if not leaves or len(leaves) != len(v):
                continue  # not a pure category->count dict
            label = label or k
            for sub_k, sub_v in leaves.items():
                counts[sub_k] = counts.get(sub_k, 0) + sub_v

    if not counts:
        return None

    ranked = sorted(counts.items(), key=lambda kv: -kv[1])[:limit]
    total = sum(counts.values())
    total = int(total) if float(total).is_integer() else round(total, 2)

    attention = [
        {"value": val, "count": (int(cnt) if float(cnt).is_integer() else cnt)}
        for val, cnt in ranked
        if any(kw in val.lower() for kw in _ATTENTION_KEYWORDS)
    ]

    return {
        "field": label,
        "breakdown": [
            {"value": v, "count": (int(c) if float(c).is_integer() else c)}
            for v, c in ranked
        ],
        "total": total,
        "attention": attention,
    }


def has_any_data(records: List[Dict[str, Any]]) -> bool:
    """Check if the list of records contains any non-empty data."""
    if not records:
        return False
    for r in records:
        if any(v is not None for v in r.values()):
            return True
    return False
