"""
widget_field_audit.py
=====================
Field-coverage audit for the AgniAI widget pipeline.

For every widget type in the contract:

    TABLE | CARD | CHART_BAR | CHART_LINE | CHART_PIE
    COMPARE_CARD | COMPARE_TABLE | COMPARE_CHART_BAR
    COMPARE_CHART_LINE | COMPARE_CHART_PIE

this module answers two questions:

1. FIELD COVERAGE — given the source records, which fields will the builder
   in widget_engine.py actually surface, which fields get dropped, and WHY
   (excluded-by-rule, not in candidate list, non-numeric, aggregated away).

2. PAYLOAD VALIDATION — given the built widget dict {type, title, data},
   is the payload structurally consistent (columns match rows, series keys
   match row keys, no empty charts, card slots populated)?

Usage:
    from widget_field_audit import audit_widget, print_report

    report = audit_widget(widget_dict, source_records)
    print_report(report)

No imports from widget_engine — the candidate lists below are mirrored from
it so the audit can run standalone. If you change the lists in
widget_engine.py, update _CANDIDATES here to match.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Candidate lists — mirrored 1:1 from widget_engine.py
# ---------------------------------------------------------------------------

_CANDIDATES = {
    "CARD": {
        "title": ["fullName", "name", "id"],
        "subtitle": ["subtitle", "subTitle", "label", "grade"],
        "value": ["bestTotal", "score", "marksObtained", "count", "status", "leaveStatus"],
        "description": ["description", "details"],
    },
    "CHART_BAR": {
        "x": ["fullName", "name", "month", "date", "year", "sport", "grade",
              "leaveType", "label", "category", "sectionName"],
        "y": ["bestTotal", "totalMarks", "score", "marksObtained", "present",
              "count", "value", "percentage", "averageScore", "absent"],
    },
    "CHART_LINE": {
        "x": ["date", "month", "year", "attemptNo", "attempt", "time", "day"],
    },
    "CHART_PIE": {
        "label": ["label", "sport", "grade", "leaveType", "bloodGroup",
                  "disease", "status", "category", "name"],
        "value": ["value", "present", "count", "score", "percentage",
                  "bestTotal", "marksObtained", "absent"],
    },
    "COMPARE_CHART_BAR": {
        "x": ["fullName", "name", "company", "platoon", "batch", "unit",
              "section", "group", "label", "category", "sport"],
        "y": ["bestTotal", "totalMarks", "score", "marksObtained", "count",
              "value", "percentage", "averageScore"],
    },
    "COMPARE_CHART_PIE": {
        "label": ["label", "sport", "grade", "leaveType", "bloodGroup",
                  "disease", "status", "category", "name", "leavetype",
                  "bmiCategory", "grading", "type"],
        "value": ["value", "count", "score", "percentage", "bestTotal",
                  "marksObtained"],
    },
    "COMPARE_CARD_METRICS": ["recordCount", "averageScore", "topScore", "bottomScore"],
}

_TABLE_EXCLUDED_SUFFIXES = (
    "_DisplayOrder",
)
_TABLE_EXCLUDED_EXACT = {
    "ID", "id", "DisplayOrder", "displayOrder",
}
_TIME_KEYS = ["date", "month", "year", "attemptNo", "attempt", "time", "day"]


# ---------------------------------------------------------------------------
# Report structures
# ---------------------------------------------------------------------------

@dataclass
class FieldAudit:
    widget_type: str
    side: str = ""                                  # "", "left", "right"
    shown: Dict[str, str] = field(default_factory=dict)     # field -> role
    dropped: Dict[str, str] = field(default_factory=dict)   # field -> reason
    issues: List[str] = field(default_factory=list)          # structural problems


# ---------------------------------------------------------------------------
# Helpers (mirroring widget_engine behaviour)
# ---------------------------------------------------------------------------

def _source_fields(records: List[Dict[str, Any]]) -> List[str]:
    seen: List[str] = []
    for r in records:
        if isinstance(r, dict):
            for k in r.keys():
                if k not in seen:
                    seen.append(k)
    return seen


def _find_key(records: List[Dict], candidates: List[str]) -> Optional[str]:
    for c in candidates:
        for r in records[:1]:
            for k in r.keys():
                if k.lower() == c.lower():
                    return k
    return None


def _is_identity_key(key: str) -> bool:
    return key.lower().endswith("id")


def _find_numeric_key(records: List[Dict], exclude: List[str]) -> Optional[str]:
    exclude_lower = {e.lower() for e in exclude}
    for r in records[:1]:
        for k, v in r.items():
            if isinstance(v, (int, float)) and k.lower() not in exclude_lower \
                    and not _is_identity_key(k):
                return k
    return None


def _table_excluded(key: str) -> bool:
    if key in _TABLE_EXCLUDED_EXACT:
        return True
    return any(key.endswith(s) for s in _TABLE_EXCLUDED_SUFFIXES)


# ---------------------------------------------------------------------------
# Per-type field coverage
# ---------------------------------------------------------------------------

def _audit_table_fields(records: List[Dict], audit: FieldAudit) -> None:
    for f in _source_fields(records):
        if _table_excluded(f):
            audit.dropped[f] = "excluded by TABLE metadata rule (exact key or suffix)"
        else:
            audit.shown[f] = "column"


def _audit_card_fields(records: List[Dict], audit: FieldAudit) -> None:
    c = _CANDIDATES["CARD"]
    chosen: Dict[str, Optional[str]] = {}
    for slot, cands in c.items():
        # build_card_data uses r.get(...) with exact-case keys, in order
        key = None
        for cand in cands:
            if records and cand in records[0]:
                key = cand
                break
        chosen[slot] = key
        if key:
            audit.shown[key] = f"card.{slot}"
    for f in _source_fields(records):
        if f not in audit.shown:
            if not str(f).lower().endswith("id"):
                audit.shown[f] = "card.description (folded as leftover)"
            else:
                audit.dropped[f] = "not in CARD candidate lists (title/subtitle/value/description)"
    for slot, key in chosen.items():
        if key is None and slot in ("title", "value"):
            audit.issues.append(
                f"CARD slot '{slot}' has no matching field — will render "
                f"fallback text ('Details' / empty)"
            )


def _audit_bar_fields(records: List[Dict], audit: FieldAudit,
                      candidates_key: str = "CHART_BAR") -> None:
    c = _CANDIDATES[candidates_key]
    x_key = _find_key(records, c["x"])
    if not x_key:
        for r in records[:1]:
            for k, v in r.items():
                if isinstance(v, str) and k.lower() != "id":
                    x_key = k
                    break
    y_key = _find_key(records, c["y"]) or _find_numeric_key(records, ["id"])
    if x_key:
        audit.shown[x_key] = "xKey (category axis)"
    else:
        audit.issues.append("no x-axis field found — falls back to 'label'")
    if y_key:
        audit.shown[y_key] = "yKey (value, summed per category)"
    else:
        audit.issues.append("no numeric value field found — chart will be empty/zero")
    for f in _source_fields(records):
        if f in audit.shown:
            continue
        if _is_identity_key(f):
            audit.dropped[f] = "identity key (endswith 'id') — never picked as metric"
        else:
            audit.shown[f] = "row.data (folded as tooltip)"


def _audit_line_fields(records: List[Dict], audit: FieldAudit) -> None:
    x_key = _find_key(records, _TIME_KEYS) or "date"
    audit.shown[x_key] = "xKey (time axis)"
    time_lower = {t.lower() for t in _TIME_KEYS + ["id"]}
    numeric = []
    for r in records[:1]:
        for k, v in r.items():
            if isinstance(v, (int, float)) and k.lower() not in time_lower \
                    and not _is_identity_key(k):
                numeric.append(k)
    for idx, k in enumerate(numeric):
        audit.shown[k] = f"series{idx}"
    if not numeric:
        audit.issues.append("no numeric series fields — line chart will be empty")
    for f in _source_fields(records):
        if f in audit.shown:
            continue
        if _is_identity_key(f):
            audit.dropped[f] = "identity key — excluded from series"
        else:
            audit.shown[f] = "row.data (folded as tooltip)"


def _audit_pie_fields(records: List[Dict], audit: FieldAudit,
                      candidates_key: str = "CHART_PIE") -> None:
    c = _CANDIDATES[candidates_key]
    label_key = _find_key(records, c["label"])
    if not label_key:
        for r in records[:1]:
            for k, v in r.items():
                if isinstance(v, str) and k.lower() != "id":
                    label_key = k
                    break
    value_key = _find_key(records, c["value"]) or _find_numeric_key(records, ["id"])
    if label_key:
        audit.shown[label_key] = "slice label (grouped)"
    if value_key:
        audit.shown[value_key] = "slice value (summed per label)"
    else:
        audit.issues.append("no numeric value field — every record counts as 1")
    for f in _source_fields(records):
        if f not in audit.shown:
            if _is_identity_key(f):
                audit.dropped[f] = "identity key (endswith 'id') — never picked as metric"
            else:
                audit.shown[f] = "row.data (folded as tooltip)"


def _audit_compare_card_fields(side: Dict[str, Any], audit: FieldAudit) -> None:
    metrics = side.get("metrics") or {}
    allowed = _CANDIDATES["COMPARE_CARD_METRICS"]
    for m in metrics:
        if m in allowed:
            audit.shown[m] = "metric card"
        else:
            audit.dropped[m] = "COMPARE_CARD shows only recordCount/averageScore/topScore/bottomScore"
    records = side.get("data") or []
    for f in _source_fields(records):
        audit.dropped.setdefault(f, "record fields never shown in COMPARE_CARD (metrics only)")
    if not any(m in allowed for m in metrics):
        audit.issues.append("no allowed metrics present — falls back to plain record count")


# ---------------------------------------------------------------------------
# Payload structural validation
# ---------------------------------------------------------------------------

def _validate_table_payload(data: Dict, audit: FieldAudit) -> None:
    cols = data.get("columns", [])
    rows = data.get("row", [])
    col_keys = {c.get("key") for c in cols if isinstance(c, dict)}
    if not cols:
        audit.issues.append("TABLE payload has no columns")
    for i, row in enumerate(rows[:50]):
        missing = col_keys - set(row.keys())
        if missing:
            audit.issues.append(f"row {i} missing column keys: {sorted(missing)}")
            break
    extra = set()
    for row in rows[:50]:
        extra |= set(row.keys()) - col_keys
    if extra:
        audit.issues.append(f"row keys with no column spec (invisible on frontend): {sorted(extra)}")


def _validate_bar_payload(data: Dict, audit: FieldAudit) -> None:
    x, y, rows = data.get("xKey"), data.get("yKey"), data.get("rows", [])
    if not rows:
        audit.issues.append("bar chart has 0 rows")
        return
    for row in rows[:20]:
        if x not in row or y not in row:
            audit.issues.append(f"row missing xKey/yKey ({x}/{y}): {row}")
            break
        if not isinstance(row.get(y), (int, float)):
            audit.issues.append(f"non-numeric y value for '{y}': {row.get(y)!r}")
            break


def _validate_line_payload(data: Dict, audit: FieldAudit) -> None:
    x = data.get("xKey")
    series = data.get("series", [])
    rows = data.get("rows", [])
    if not rows:
        audit.issues.append("line chart has 0 rows")
        return
    if not series:
        audit.issues.append("line chart has no series definitions")
        return
    series_keys = [s.get("key") for s in series if isinstance(s, dict)]
    sample = rows[0]
    for sk in series_keys:
        if sk not in sample:
            audit.issues.append(
                f"SERIES KEY MISMATCH: series declares '{sk}' but rows contain "
                f"{sorted(k for k in sample.keys() if k != x)} — frontend will "
                f"plot nothing for this series"
            )


def _validate_pie_payload(data: Dict, audit: FieldAudit) -> None:
    rows = data.get("rows", [])
    if not rows:
        audit.issues.append("pie chart has 0 rows")
        return
    for row in rows[:20]:
        if "label" not in row or "value" not in row:
            audit.issues.append(f"pie row missing label/value: {row}")
            break
        if not isinstance(row.get("value"), (int, float)):
            audit.issues.append(f"non-numeric pie value: {row.get('value')!r}")
            break


def _validate_card_payload(data: Dict, audit: FieldAudit) -> None:
    cards = data.get("cards", [])
    if not cards:
        audit.issues.append("CARD payload has no cards")
        return
    empty_values = sum(1 for c in cards if not str(c.get("value", "")).strip())
    if empty_values:
        audit.issues.append(f"{empty_values}/{len(cards)} cards have an empty value slot")
    empty_subtitles = sum(1 for c in cards if not str(c.get("subtitle", "")).strip())
    if empty_subtitles == len(cards):
        audit.issues.append("all cards have empty subtitles (no matching source field)")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def audit_widget(widget: Dict[str, Any],
                 source: Any = None) -> List[FieldAudit]:
    """
    Audit one built widget {type, title, data} against its source.

    `source` for simple types: List[Dict] records.
    `source` for COMPARE_* types: the combined_result dict with left/right.
    Returns one FieldAudit per side (single-sided types return one).
    """
    wt = str(widget.get("type") or "").upper()
    data = widget.get("data") or {}
    reports: List[FieldAudit] = []

    def _records(x: Any) -> List[Dict]:
        if isinstance(x, list):
            return [r for r in x if isinstance(r, dict)]
        return []

    if wt == "TABLE":
        a = FieldAudit(wt)
        _audit_table_fields(_records(source), a)
        _validate_table_payload(data, a)
        reports.append(a)

    elif wt == "CARD":
        a = FieldAudit(wt)
        _audit_card_fields(_records(source), a)
        _validate_card_payload(data, a)
        reports.append(a)

    elif wt == "CHART_BAR":
        a = FieldAudit(wt)
        _audit_bar_fields(_records(source), a)
        _validate_bar_payload(data, a)
        reports.append(a)

    elif wt == "CHART_LINE":
        a = FieldAudit(wt)
        _audit_line_fields(_records(source), a)
        _validate_line_payload(data, a)
        reports.append(a)

    elif wt == "CHART_PIE":
        a = FieldAudit(wt)
        _audit_pie_fields(_records(source), a)
        _validate_pie_payload(data, a)
        reports.append(a)

    elif wt.startswith("COMPARE_"):
        combined = source if isinstance(source, dict) else {}
        for side_name in ("left", "right"):
            side = combined.get(side_name) or {}
            recs = _records(side.get("data"))
            a = FieldAudit(wt, side=side_name)
            side_payload = data.get(side_name) or {}

            if wt == "COMPARE_TABLE":
                _audit_table_fields(recs, a)
                _validate_table_payload(side_payload, a)
            elif wt == "COMPARE_CARD":
                _audit_compare_card_fields(side, a)
            elif wt == "COMPARE_CHART_BAR":
                _audit_bar_fields(recs, a, candidates_key="COMPARE_CHART_BAR")
                _validate_bar_payload(side_payload, a)
            elif wt == "COMPARE_CHART_LINE":
                _audit_line_fields(recs, a)
                _validate_line_payload(side_payload, a)
            elif wt == "COMPARE_CHART_PIE":
                _audit_pie_fields(recs, a, candidates_key="COMPARE_CHART_PIE")
                _validate_pie_payload(side_payload, a)
            reports.append(a)

    else:
        a = FieldAudit(wt or "UNKNOWN")
        a.issues.append(f"unknown widget type '{wt}' — no audit rules")
        reports.append(a)

    return reports


def audit_response(response: Dict[str, Any],
                   combined_result: Any = None,
                   source_records: Optional[List[Dict]] = None) -> List[FieldAudit]:
    """
    Audit a full API response. formattedData may be a single widget dict or a list.
    Pass `combined_result` for COMPARE_* widgets, `source_records` for the rest.
    """
    fd = response.get("formattedData")
    widgets = fd if isinstance(fd, list) else [fd] if isinstance(fd, dict) else []
    out: List[FieldAudit] = []
    for w in widgets:
        wt = str(w.get("type") or "").upper()
        src = combined_result if wt.startswith("COMPARE_") else source_records
        out.extend(audit_widget(w, src))
    return out


def print_report(reports: List[FieldAudit]) -> None:
    for a in reports:
        head = a.widget_type + (f" [{a.side}]" if a.side else "")
        print(f"\n{'=' * 62}\n{head}\n{'=' * 62}")
        if a.shown:
            print("  SHOWN:")
            for f, role in a.shown.items():
                print(f"    ✓ {f:<28} -> {role}")
        else:
            print("  SHOWN: (none)")
        if a.dropped:
            print("  DROPPED:")
            for f, why in a.dropped.items():
                print(f"    ✗ {f:<28} — {why}")
        if a.issues:
            print("  ⚠ ISSUES:")
            for i in a.issues:
                print(f"    ! {i}")
        else:
            print("  ✔ payload structurally valid")