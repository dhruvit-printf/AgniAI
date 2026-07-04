"""
widget_engine.py
================
Widget / Schema Inference Engine for the AgniAI admin pipeline.

Converts CombinedResult into a single deterministic FormattedData structure.

Widget selection is owned by visualization_intent.py and routed by
widget_selector.py. This module only builds the requested widget data.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import ValidationError

from normalized_models import extract_records as _orig_extract_records

logger = logging.getLogger(__name__)


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


def recursive_flatten(
    val: Any, path_prefix: List[str], flat_dict: Dict[str, Any]
) -> None:
    if isinstance(val, dict):
        if not path_prefix:
            for k, v in val.items():
                k_seg = capitalize_segment(k)
                recursive_flatten(v, [k_seg], flat_dict)
            return

        name_key = None
        for k in val.keys():
            k_lower = k.lower()
            if (
                k_lower in ("name", "label", "sectionname", "subitemname", "itemname")
                or k_lower.endswith("name")
                or k_lower.endswith("label")
            ):
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
                    if (
                        k_lower
                        in ("name", "label", "sectionname", "subitemname", "itemname")
                        or k_lower.endswith("name")
                        or k_lower.endswith("label")
                    ):
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


def flatten_records(
    records: List[Dict[str, Any]], deep_flatten: bool = False
) -> List[Dict[str, Any]]:
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
            import hashlib
            row_str = json.dumps(
                record, sort_keys=True, ensure_ascii=False, default=str
            )
            record_id = "row:" + hashlib.md5(row_str.encode("utf-8")).hexdigest()
        if record_id in seen:
            continue
        seen.add(record_id)
        deduped.append(record)
    return deduped


def _extract_records(
    combined_result: Any, deep_flatten: bool = False
) -> List[Dict[str, Any]]:
    return flatten_records(
        _orig_extract_records(combined_result), deep_flatten=deep_flatten
    )


# Legacy/alias widget type names accepted as input, normalised to the new
# canonical names (CHART_BAR / CHART_LINE / CHART_PIE / COMPARE_CHART_*).
# DONUT_CHART and RADIAL_CHART are folded — donut into pie, radial into line —
# and AREA_CHART is folded into line as well.
_CHART_TYPE_ALIASES: Dict[str, str] = {
    "BAR_CHART": "CHART_BAR",
    "LINE_CHART": "CHART_LINE",
    "AREA_CHART": "CHART_LINE",
    "RADIAL_CHART": "CHART_LINE",
    "PIE_CHART": "CHART_PIE",
    "DONUT_CHART": "CHART_PIE",
    "COMPARE_BAR_CHART": "COMPARE_CHART_BAR",
    "COMPARE_LINE_CHART": "COMPARE_CHART_LINE",
    "COMPARE_PIE_CHART": "COMPARE_CHART_PIE",
}


def _canonical_widget_type(value: Any) -> str:
    """Normalise any accepted widget type name (new or legacy) to canonical form."""
    text = str(value or "TABLE").upper()
    return _CHART_TYPE_ALIASES.get(text, text)


def _planned_widget_types(
    visualization_intent: Optional[Dict[str, Any]],
) -> List[str]:
    if not isinstance(visualization_intent, dict):
        return []
    widgets = visualization_intent.get("widgets")
    if not isinstance(widgets, list):
        return []
    planned: List[str] = []
    for widget in widgets:
        if isinstance(widget, dict) and widget.get("type"):
            planned.append(_canonical_widget_type(widget["type"]))
    return planned


def _effective_visualization_intent(
    query_type: str,
    intent: Dict[str, Any],
    combined_result: Any,
    visualization_intent: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if isinstance(visualization_intent, dict):
        planned = _planned_widget_types(visualization_intent)
        if planned:
            return visualization_intent
        override = (
            visualization_intent.get("requested_widget_type")
            or visualization_intent.get("widget_type")
        )
        if override:
            from visualization_intent import build_visualization_intent

            resolved = build_visualization_intent(
                "",
                {**intent, "query_type": query_type},
                combined_result,
                query_type_override=query_type,
            )
            resolved["widgets"] = [
                {"type": _normalize_requested_widget_type(override) or _canonical_widget_type(override)}
            ]
            return {**resolved, **visualization_intent}

    from visualization_intent import build_visualization_intent

    resolved = build_visualization_intent(
        "",
        {**intent, "query_type": query_type},
        combined_result,
        query_type_override=query_type,
    )
    if isinstance(visualization_intent, dict):
        resolved = {**resolved, **visualization_intent}
    return resolved


from schemas import (
    BarChartData,
    CardData,
    CardItem,
    FormattedData,
    LineChartData,
    PieChartData,
    PieChartItem,
    SeriesItem,
    TableColumn,
    TableData,
)

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

    # Direct match: canonical constants, plus legacy constants accepted as
    # input aliases (BAR_CHART, DONUT_CHART, RADIAL_CHART, AREA_CHART, ...).
    _CANONICAL = {
        "TABLE",
        "CARD",
        "CHART_BAR",
        "CHART_LINE",
        "CHART_PIE",
        "BAR_CHART",
        "LINE_CHART",
        "AREA_CHART",
        "PIE_CHART",
        "DONUT_CHART",
        "RADIAL_CHART",
    }
    if text.upper() in _CANONICAL:
        return _canonical_widget_type(text.upper())

    # Case-insensitive label lookup covering all labels from the widget menu
    _LABEL_MAP: Dict[str, str] = {
        # ── Tabular ───────────────────────────────────────────────────────────
        "tabular": "TABLE",
        "table": "TABLE",
        "grid": "TABLE",
        # ── Card ──────────────────────────────────────────────────────────────
        "card": "CARD",
        "cards": "CARD",
        "stats card": "CARD",
        # ── Bar Chart ─────────────────────────────────────────────────────────
        "bar chart": "CHART_BAR",
        "bar": "CHART_BAR",
        "monthly bar chart": "CHART_BAR",
        "weekly bar chart": "CHART_BAR",
        "gradingsummary bar chart": "CHART_BAR",
        # ── Line / Trend Chart ────────────────────────────────────────────────
        "line chart": "CHART_LINE",
        "line": "CHART_LINE",
        "trend chart": "CHART_LINE",
        "trend": "CHART_LINE",
        "improvement trend chart": "CHART_LINE",
        "drop trend chart": "CHART_LINE",
        # ── Area Chart (folded into line) ───────────────────────────────────────
        "area chart": "CHART_LINE",
        "area": "CHART_LINE",
        "compare area chart": "CHART_LINE",
        # ── Pie Chart ─────────────────────────────────────────────────────────
        "pie chart": "CHART_PIE",
        "pie": "CHART_PIE",
        "average pie chart": "CHART_PIE",
        "present pie chart": "CHART_PIE",
        # ── Donut Chart (folded into pie) ───────────────────────────────────────
        "donut chart": "CHART_PIE",
        "donut": "CHART_PIE",
        "bmi donut chart": "CHART_PIE",
        # ── Radial Chart (folded into line) ─────────────────────────────────────
        "radial chart": "CHART_LINE",
        "radial": "CHART_LINE",
        "strength radial chart": "CHART_LINE",
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
    text = str(inferred or "").upper()
    if text in ("TABLE", "CARD", "METRIC_CARD", "CALENDAR_UI"):
        return "TABLE" if text in ("TABLE", "CALENDAR_UI") else "CARD"
    if text in ("CHART_BAR", "CHART_LINE", "CHART_PIE"):
        return text
    return _CHART_TYPE_ALIASES.get(text, "TABLE")


def infer_supported_type(
    combined_result: Any,
    query_type: str,
    intent: Dict[str, Any],
    visualization_intent: Optional[Dict[str, Any]] = None,
) -> str:
    planned = _planned_widget_types(visualization_intent)
    return planned[0] if planned else "TABLE"


def build_card_data(records: List[Dict[str, Any]], title: str) -> Dict[str, Any]:
    """CARD schema: { cards: [{title, subtitle, value, description}] }"""
    cards = []
    for r in records:
        card_title = (
            r.get("fullName")
            or r.get("name")
            or (f"Record {r.get('id', '')}" if "id" in r else "Details")
        )
        subtitle = (
            r.get("subtitle")
            or r.get("subTitle")
            or r.get("label")
            or r.get("grade")
            or ""
        )
        card_value = (
            r.get("bestTotal")
            or r.get("score")
            or r.get("marksObtained")
            or r.get("count")
            or r.get("status")
            or r.get("leaveStatus")
            or ""
        )
        description = r.get("description") or r.get("details") or ""
        cards.append(
            {
                "title": str(card_title),
                "subtitle": str(subtitle),
                "value": str(card_value),
                "description": str(description),
            }
        )
    if not cards:
        cards.append(
            {
                "title": title,
                "subtitle": "",
                "value": "No records found.",
                "description": "",
            }
        )
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
    "IsActive",
    "isActive",
    "ID",
    "id",
    "DisplayOrder",
    "displayOrder",
    "SectionId",
    "sectionId",
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
    label = (
        label.replace("Agniveer No", "Agniveer No.")
        .replace("Id", "ID")
        .replace("Bpet", "BPET")
        .replace("Bmi", "BMI")
    )
    return label


def build_table_data(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {"columns": [], "row": []}

    # Collect all keys, preserving insertion order across all records.
    keys_seen: List[str] = []
    for r in records:
        if isinstance(r, dict):
            for k in r.keys():
                if k not in keys_seen:
                    keys_seen.append(k)

    # Filter out internal metadata columns before building anything else.
    _excluded = [k for k in keys_seen if _should_exclude_column(k)]
    if _excluded:
        logger.debug(
            "build_table_data: excluded %d internal columns: %s",
            len(_excluded),
            _excluded,
        )
    keys_seen = [k for k in keys_seen if not _should_exclude_column(k)]

    # Sort with priority fields first.
    key_priority = {
        "fullname": -10,
        "agniveerno": -9,
        "name": -8,
        "agniveerid": -7,
        "score": -6,
        "besttotal": -5,
    }
    keys_seen.sort(key=lambda k: key_priority.get(k.lower(), 0))

    # Build normalised column specs: camelCase key, human-readable label.
    columns = []
    key_map: Dict[str, str] = {}  # original_key -> camelCase_key
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

    return {"columns": columns, "row": rows}


def _find_key(records: List[Dict], candidates: List[str]) -> Optional[str]:
    """Return the first field key matching any candidate (case-insensitive)."""
    for c in candidates:
        for r in records[:1]:
            for k in r.keys():
                if k.lower() == c.lower():
                    return k
    return None


def _is_identity_key(key: str) -> bool:
    """True for ID-like fields (agniveerId, companyId, batchId, ...) that must
    never be auto-picked as a chart's numeric value — they identify a record,
    they aren't a metric."""
    return key.lower().endswith("id")


def _find_numeric_key(records: List[Dict], exclude: List[str]) -> Optional[str]:
    exclude_lower = {e.lower() for e in exclude}
    for r in records[:1]:
        for k, v in r.items():
            if not isinstance(v, (int, float)):
                continue
            if k.lower() in exclude_lower or _is_identity_key(k):
                continue
            return k
    return None


# Attendance-style responses often nest a per-period breakdown inside a list
# field, e.g. "months": [{"month": "06-2026", "present": 30, "absent": 0}].
_PERIOD_LIST_FIELDS = ("months", "weeks", "days")
_PERIOD_LABEL_FIELDS = ("month", "week", "date")


def _expand_period_records(
    records: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Explode a nested per-period list (months/weeks/days) into flat rows so
    charts can plot the real metrics inside it (present/absent/...) instead
    of falling back to an identity field like agniveerId.

    Returns (records, period_label_key) — period_label_key is the field that
    labels each exploded period (e.g. "month"), or None if nothing to explode.
    """
    period_list_key = None
    for candidate in _PERIOD_LIST_FIELDS:
        if records and all(isinstance(r.get(candidate), list) for r in records):
            period_list_key = candidate
            break
    if not period_list_key:
        return records, None

    exploded: List[Dict[str, Any]] = []
    period_label_key: Optional[str] = None
    for r in records:
        identity = {k: v for k, v in r.items() if k != period_list_key}
        periods = r.get(period_list_key) or []
        if not periods:
            exploded.append(identity)
            continue
        for period in periods:
            if isinstance(period, dict):
                exploded.append({**identity, **period})
                if period_label_key is None:
                    for label_field in _PERIOD_LABEL_FIELDS:
                        if label_field in period:
                            period_label_key = label_field
                            break
            else:
                exploded.append(identity)
    return exploded, period_label_key


def build_bar_chart_data(
    combined_result: Any, series_label: str = ""
) -> Dict[str, Any]:
    """
    BAR_CHART schema:
    {
        "xKey": "section",
        "yKey": "count",
        "rows": [{ "section": "...", "count": 0 }]
    }
    """
    records = _extract_records(combined_result, deep_flatten=False)
    records, period_key = _expand_period_records(records)
    if not period_key:
        # Dedup keys off identity fields (agniveerId/id/...), which would
        # wrongly collapse multiple exploded periods for the same subject.
        records = _dedupe_records(records)
    if not records:
        return {
            "xKey": "",
            "yKey": "",
            "rows": [],
        }

    identity_key = _find_key(records, ["agniveerName", "fullName", "name"])

    if period_key and identity_key:
        distinct_identities = {r.get(identity_key) for r in records}
        # One subject across multiple periods -> chart across time.
        # Multiple subjects -> chart across subjects (one bar per subject),
        # since a single flat bar chart can't show both dimensions at once.
        x_key = period_key if len(distinct_identities) <= 1 else identity_key
    else:
        x_key = _find_key(
            records,
            [
                "fullName",
                "name",
                "month",
                "date",
                "year",
                "sport",
                "grade",
                "leaveType",
                "label",
                "category",
                "sectionName",
            ],
        )
        if not x_key:
            for r in records[:1]:
                for k, v in r.items():
                    if isinstance(v, str) and k.lower() not in ("id",):
                        x_key = k
                        break
        x_key = x_key or "label"

    y_key = (
        _find_key(
            records,
            [
                "bestTotal",
                "totalMarks",
                "score",
                "marksObtained",
                "present",
                "count",
                "value",
                "percentage",
                "averageScore",
                "absent",
            ],
        )
        or _find_numeric_key(records, ["id"])
        or "value"
    )

    rows = []
    for r in records:
        x_val = r.get(x_key) or r.get("fullName") or "Category"
        y_val = r.get(y_key) if r.get(y_key) is not None else 0
        rows.append({x_key: x_val, y_key: y_val})

    return {"xKey": x_key, "yKey": y_key, "rows": rows}


def build_line_chart_data(combined_result: Any) -> Dict[str, Any]:
    """
    LINE_CHART / AREA_CHART schema:
    {
        "xKey": "date",
        "series": [{ "key": "series0", "label": "Score" }],
        "rows": [{ "date": "...", "series0": 0 }]
    }
    """
    records = _extract_records(combined_result, deep_flatten=False)
    records, _period_key = _expand_period_records(records)
    if not records:
        return {
            "xKey": "time",
            "series": [],
            "rows": [],
        }

    time_keys = ["date", "month", "year", "attemptNo", "attempt", "time", "day"]
    x_key = _find_key(records, time_keys) or "date"

    numeric_keys = [
        k
        for r in records[:1]
        for k, v in r.items()
        if isinstance(v, (int, float))
        and k.lower() not in {t.lower() for t in time_keys + ["id"]}
        and not _is_identity_key(k)
    ]
    if not numeric_keys:
        numeric_keys = ["value"]

    series = [
        {"key": f"series{idx}", "label": make_readable_label(sk)}
        for idx, sk in enumerate(numeric_keys)
    ]

    rows = []
    for r in records:
        row: Dict[str, Any] = {x_key: r.get(x_key, "")}
        for idx, sk in enumerate(numeric_keys):
            row[f"series{idx}"] = r.get(sk, 0)
        rows.append(row)

    return {"xKey": x_key, "series": series, "rows": rows}


def build_pie_chart_data(
    combined_result: Any, series_label: str = "Distribution"
) -> Dict[str, Any]:
    """
    PIE_CHART / DONUT_CHART schema:
    {
        "rows": [{ "label": "", "value": 0 }]
    }
    """
    records = _extract_records(combined_result, deep_flatten=False)
    records, _period_key = _expand_period_records(records)
    if not records:
        return {"rows": []}

    label_key = _find_key(
        records,
        [
            "label",
            "sport",
            "grade",
            "leaveType",
            "bloodGroup",
            "disease",
            "status",
            "category",
            "name",
        ],
    )
    if not label_key:
        for r in records[:1]:
            for k, v in r.items():
                if isinstance(v, str) and k.lower() not in ("id",):
                    label_key = k
                    break
    label_key = label_key or "label"

    value_key = (
        _find_key(
            records,
            [
                "value",
                "present",
                "count",
                "score",
                "percentage",
                "bestTotal",
                "marksObtained",
                "absent",
            ],
        )
        or _find_numeric_key(records, ["id"])
        or "value"
    )

    rows = [
        {
            "label": str(r.get(label_key) or r.get("fullName") or "Category"),
            "value": r.get(value_key) if r.get(value_key) is not None else 1,
        }
        for r in records
    ]

    return {"rows": rows}


_PRESENT_TRUE_VALUES = {"present", "p", "yes", "true", "1", "attended"}
_PRESENT_FALSE_VALUES = {"absent", "a", "no", "false", "0", "missed"}


def _coerce_is_present(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value or "").strip().lower()
    if text in _PRESENT_TRUE_VALUES:
        return True
    if text in _PRESENT_FALSE_VALUES:
        return False
    return False


def _parse_calendar_date(value: Any) -> Optional["datetime"]:
    from datetime import datetime as _datetime

    text = str(value or "").strip()
    if not text:
        return None
    # Accept "YYYY-MM-DD", full ISO date-times, and trailing "Z" (UTC marker).
    try:
        return _datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            return _datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _calendar_period_from_intent(intent: Dict[str, Any]) -> Tuple[int, int]:
    from datetime import datetime as _datetime

    for field in ("date", "from_date", "fromDate", "to_date", "toDate"):
        parsed = _parse_calendar_date(intent.get(field))
        if parsed:
            return parsed.year, parsed.month
    now = _datetime.now()
    return now.year, now.month


def _find_first_value(records: List[Dict], candidates: List[str]) -> Optional[Any]:
    for r in records[:1]:
        for c in candidates:
            for k in r.keys():
                if k.lower() == c.lower() and r.get(k) not in (None, ""):
                    return r.get(k)
    return None


def build_attendance_calendar_data(
    combined_result: Any, intent: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    ATTENDANCE_CALENDAR schema:
    {
        "year": int, "month": int,
        "agniveerNo": str, "agniveerName": str, "photoPath": str,
        "days": [{"date": str, "isPresent": bool}, ...]
    }
    """
    intent = intent or {}
    records = _extract_records(combined_result, deep_flatten=False)

    date_key = _find_key(records, ["date", "attendanceDate", "day"])
    present_key = _find_key(
        records, ["isPresent", "present", "attended", "attendanceStatus", "status"]
    )

    year: Optional[int] = None
    month: Optional[int] = None
    days: List[Dict[str, Any]] = []

    for r in records:
        raw_date = r.get(date_key) if date_key else None
        parsed = _parse_calendar_date(raw_date)
        if parsed and year is None:
            year, month = parsed.year, parsed.month

        raw_present = r.get(present_key) if present_key else None
        days.append(
            {
                "date": str(raw_date) if raw_date is not None else "",
                "isPresent": _coerce_is_present(raw_present),
            }
        )

    if year is None or month is None:
        year, month = _calendar_period_from_intent(intent)

    agniveer_no = (
        _find_first_value(records, ["agniveerNo", "AgniveerNo", "AgniVeerNo"])
        or intent.get("agniveer_no")
        or intent.get("agniveerNo")
        or ""
    )
    agniveer_name = (
        _find_first_value(records, ["agniveerName", "fullName", "name"]) or ""
    )
    photo_path = (
        _find_first_value(records, ["photoPath", "photoUrl", "photo"]) or ""
    )

    return {
        "year": year,
        "month": month,
        "agniveerNo": str(agniveer_no),
        "agniveerName": str(agniveer_name),
        "photoPath": str(photo_path),
        "days": days,
    }


def validate_payload(inferred_type: str, data: Dict[str, Any]) -> None:
    if inferred_type == "TABLE":
        if "sides" in data or "sections" in data:
            return
        if "left" in data and "right" in data:
            return
        cols = {c["key"] for c in data.get("columns", [])}
        for row in data.get("row", []):
            for col in cols:
                if col not in row:
                    row[col] = None
    elif inferred_type in {
        "CHART_BAR",
        "CHART_LINE",
        "BAR_CHART",
        "LINE_CHART",
        "AREA_CHART",
        "RADIAL_CHART",
    }:  # includes legacy aliases (RADIAL_CHART folded into line)
        rows = data.get("rows", [])
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict):
                    if not row:
                        row.setdefault("xKey", "")
    elif inferred_type in {"CHART_PIE", "PIE_CHART", "DONUT_CHART"}:  # legacy aliases
        for row in data.get("rows", []):
            if isinstance(row, dict):
                row.setdefault("label", "Category")
                row.setdefault("value", 0)
    elif inferred_type == "ATTENDANCE_CALENDAR":
        data.setdefault("year", None)
        data.setdefault("month", None)
        data.setdefault("agniveerNo", "")
        data.setdefault("agniveerName", "")
        data.setdefault("photoPath", "")
        data.setdefault("days", [])


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
    effective_visualization_intent = _effective_visualization_intent(
        query_type, intent, source_result, visualization_intent
    )

    inferred_type = infer_supported_type(
        source_result,
        query_type,
        intent,
        visualization_intent=effective_visualization_intent,
    )

    from normalized_models import _derive_title

    title = _derive_title(query_type, intent)

    if (
        query_type in ("compare", "comparison")
        and isinstance(source_result, dict)
        and "sides" in source_result
    ):
        from widget_selector import WidgetSelector

        selector = WidgetSelector()
        specs = selector.select(
            query_type=query_type,
            intent=intent,
            combined_result=source_result,
            primary_widget_type=infer_supported_type(
                source_result, query_type, intent, effective_visualization_intent
            ),
            analysis=analysis,
            comparison_chart_override=(
                effective_visualization_intent.get("comparison_chart_override")
                if isinstance(effective_visualization_intent, dict)
                else None
            ),
            visualization_intent=effective_visualization_intent,
        )
        if specs:
            spec = specs[0]
            data = _build_widget_data(spec, source_result, query_type, intent, analysis)
            if isinstance(source_result, dict):
                for k, v in source_result.items():
                    if (
                        k not in ("records", "data", "sections", "columns", "row")
                        and k not in data
                    ):
                        data[k] = v
            return {
                "type": spec.widget_type,
                "title": spec.title or title,
                "data": data,
            }

    # Check for multi_independent and comparison early to bypass extract/flatten
    if (
        query_type == "multi_independent"
        and isinstance(source_result, dict)
        and "sections" in source_result
    ):
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
                    "row": sec_table.get("row", []),
                }
                for k, v in sec.items():
                    if k not in ("label", "data", "columns", "row"):
                        sec_payload[k] = v
                sections_list.append(sec_payload)
        data_payload = {"sections": sections_list}

    elif (
        isinstance(source_result, dict)
        and "left" in source_result
        and "right" in source_result
    ):
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
                if k not in ("label", "data", "columns", "row"):
                    left_extra[k] = v
        else:
            left_data = left_section if isinstance(left_section, list) else []
            left_flat_records = flatten_records(left_data, deep_flatten=True)

        left_table = build_table_data(left_flat_records)
        left_payload = {
            "label": left_label,
            "columns": left_table.get("columns", []),
            "row": left_table.get("row", []),
            **left_extra,
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
                if k not in ("label", "data", "columns", "row"):
                    right_extra[k] = v
        else:
            right_data = right_section if isinstance(right_section, list) else []
            right_flat_records = flatten_records(right_data, deep_flatten=True)

        right_table = build_table_data(right_flat_records)
        right_payload = {
            "label": right_label,
            "columns": right_table.get("columns", []),
            "row": right_table.get("row", []),
            **right_extra,
        }

        comparison_payload = source_result.get("comparison", {})

        data_payload = {
            "left": left_payload,
            "right": right_payload,
            "comparison": comparison_payload,
        }

    elif inferred_type == "CARD":
        records = _extract_records(source_result, deep_flatten=False)
        data_payload = build_card_data(records, title)
    elif inferred_type in {"CHART_BAR", "BAR_CHART"}:
        data_payload = build_bar_chart_data(source_result)
    elif inferred_type in {"CHART_LINE", "LINE_CHART", "AREA_CHART", "RADIAL_CHART"}:
        data_payload = build_line_chart_data(source_result)
    elif inferred_type in {"CHART_PIE", "PIE_CHART", "DONUT_CHART"}:
        data_payload = build_pie_chart_data(source_result)
    elif inferred_type == "ATTENDANCE_CALENDAR":
        data_payload = build_attendance_calendar_data(source_result, intent)
    else:
        if isinstance(source_result, dict) and "sides" in source_result:
            sides_data = []
            for side in source_result["sides"]:
                s_data = side.get("data")
                if isinstance(s_data, list):
                    flattened_data = [r for r in s_data if isinstance(r, dict)]
                else:
                    flattened_data = s_data
                side_payload = {"label": side.get("label"), "data": flattened_data}
                for k, v in side.items():
                    if k not in ("label", "data"):
                        side_payload[k] = v
                sides_data.append(side_payload)
            data_payload = {"sides": sides_data}
        else:
            table_records = _extract_records(source_result, deep_flatten=True)
            data_payload = build_table_data(table_records)

    # Preserve unknown .NET keys for non-comparison queries so cross-filter metadata
    # (matchCount, totalBeforeFilter, etc.) and other backend context reaches the frontend.
    # Comparison widgets and ATTENDANCE_CALENDAR are fully self-contained — never merge
    # extra keys into them.
    if (
        query_type not in ("compare", "comparison")
        and inferred_type != "ATTENDANCE_CALENDAR"
        and isinstance(source_result, dict)
        and isinstance(data_payload, dict)
    ):
        for k, v in source_result.items():
            if k not in data_payload:
                data_payload[k] = v

    # Fallback to TABLE if invalid chart fields are detected (FIX 30)
    is_invalid_chart = False
    if inferred_type in (
        "CHART_BAR",
        "CHART_LINE",
        "CHART_PIE",
        "BAR_CHART",
        "LINE_CHART",
        "AREA_CHART",
        "PIE_CHART",
        "DONUT_CHART",
        "RADIAL_CHART",
    ):
        if "left" in data_payload and "right" in data_payload:
            pass
        elif "sections" in data_payload:
            pass
        else:
            rows = data_payload.get("rows")
            if not rows or not isinstance(rows, list):
                is_invalid_chart = True
            elif not any(isinstance(row, dict) and row for row in rows):
                is_invalid_chart = True

    if is_invalid_chart:
        inferred_type = "TABLE"
        table_records = _extract_records(source_result, deep_flatten=True)
        data_payload = build_table_data(table_records)

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
    stats = (analysis or {}).get("statistics") or {}
    category = (intent or {}).get("category") or "Results"
    cards: List[Dict[str, Any]] = []

    def _card(title: str, value: Any) -> Dict[str, Any]:
        return {
            "title": title,
            "subtitle": "",
            "value": str(value),
            "description": "",
        }

    rc = stats.get("record_count")
    if rc is not None:
        cards.append(_card("Total Records", rc))

    avg = stats.get("average_score")
    if avg is not None:
        cards.append(_card("Average Score", avg))

    lc = stats.get("left_count")
    rc2 = stats.get("right_count")
    if lc is not None:
        left_label = stats.get("left_label") or "Side 1"
        right_label = stats.get("right_label") or "Side 2"
        cards.append(_card(left_label, lc))
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


def _get_score(record: Dict[str, Any]) -> Optional[float]:
    for key in (
        "bestTotal",
        "totalMarks",
        "score",
        "Score",
        "omrInputTotal",
        "marksObtained",
    ):
        val = record.get(key)
        if val is not None:
            try:
                return float(val)
            except (ValueError, TypeError):
                pass
    return None


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _extract_chronological_key(record: Dict[str, Any]) -> Any:
    for k in ("date", "Date", "createdDate", "CreatedDate", "timestamp", "Timestamp"):
        val = record.get(k)
        if val:
            return str(val)
    for k in ("attempt", "attemptNo", "Attempt", "AttemptNo"):
        val = _safe_float(record.get(k))
        if val is not None:
            return val
    month = record.get("month") or record.get("Month")
    year = record.get("year") or record.get("Year")
    if year is not None:
        if month is not None:
            return (year, month)
        return year
    return None


def _build_compare_card(combined_result: Dict[str, Any]) -> Dict[str, Any]:
    """COMPARE_CARD: {left: [{label, value}], right: [{label, value}]}"""
    left_side = combined_result.get("left") or {}
    right_side = combined_result.get("right") or {}

    _LABELS = {
        "recordCount": "Total Records",
        "averageScore": "Average Score",
        "topScore": "Top Score",
        "bottomScore": "Bottom Score",
    }

    def _side_cards(side: Dict[str, Any]) -> List[Dict[str, str]]:
        metrics = side.get("metrics") or {}
        records = side.get("data") or []
        cards: List[Dict[str, str]] = []
        for k in ("recordCount", "averageScore", "topScore", "bottomScore"):
            val = metrics.get(k)
            if val is not None:
                display = (
                    str(round(float(val), 2)) if isinstance(val, float) else str(val)
                )
                cards.append({"label": _LABELS[k], "value": display})
        if not cards:
            cards.append({"label": "Records", "value": str(len(records))})
        return cards

    return {
        "left": _side_cards(left_side),
        "right": _side_cards(right_side),
    }


def _build_compare_table(combined_result: Dict[str, Any]) -> Dict[str, Any]:
    """COMPARE_TABLE: {left: {columns, row}, right: {columns, row}}"""
    left_side = combined_result.get("left") or {}
    right_side = combined_result.get("right") or {}

    def _side_heading(side: Dict[str, Any]) -> str:
        return str(side.get("label") or "")

    def _side_table(side: Dict[str, Any]) -> Dict[str, Any]:
        records = side.get("data") or []
        flat = flatten_records(records, deep_flatten=True)
        table = build_table_data(flat)
        table["heading"] = _side_heading(side)
        return table

    return {
        "left": _side_table(left_side),
        "right": _side_table(right_side),
    }


def _build_compare_bar(combined_result: Dict[str, Any]) -> Dict[str, Any]:
    """COMPARE_BAR_CHART: {left: {xKey, yKey, rows}, right: {xKey, yKey, rows}}"""
    left_side = combined_result.get("left") or {}
    right_side = combined_result.get("right") or {}

    def _side_heading(side: Dict[str, Any]) -> str:
        return str(side.get("label") or "")

    def _side_bar(side: Dict[str, Any]) -> Dict[str, Any]:
        records = side.get("data") or []
        if not records:
            return {
                "heading": _side_heading(side),
                "xKey": "label",
                "yKey": "value",
                "rows": [],
            }
        x_key = (
            _find_key(
                records,
                [
                    "fullName",
                    "name",
                    "company",
                    "platoon",
                    "batch",
                    "unit",
                    "section",
                    "group",
                    "label",
                    "category",
                    "sport",
                ],
            )
            or "label"
        )
        y_key = (
            _find_key(
                records,
                [
                    "bestTotal",
                    "totalMarks",
                    "score",
                    "marksObtained",
                    "count",
                    "value",
                    "percentage",
                    "averageScore",
                ],
            )
            or _find_numeric_key(records, ["id"])
            or "value"
        )
        if x_key == "label":
            sample = records[0] if records else {}
            for candidate_key, candidate_value in sample.items():
                if candidate_key == y_key or candidate_key.lower() == "id":
                    continue
                if isinstance(candidate_value, str) and candidate_value.strip():
                    x_key = candidate_key
                    break
        rows = []
        for r in records:
            x_val = r.get(x_key)
            y_val = r.get(y_key)
            rows.append({x_key: x_val, y_key: y_val})
        return {
            "heading": _side_heading(side),
            "xKey": x_key,
            "yKey": y_key,
            "rows": rows,
        }

    return {
        "left": _side_bar(left_side),
        "right": _side_bar(right_side),
    }


def _build_compare_line(combined_result: Dict[str, Any]) -> Dict[str, Any]:
    """COMPARE_LINE_CHART: {left: {xKey, series, rows}, right: {xKey, series, rows}}"""
    left_side = combined_result.get("left") or {}
    right_side = combined_result.get("right") or {}

    _TIME_KEYS = ["date", "month", "year", "attemptNo", "attempt", "time", "day"]

    def _side_heading(side: Dict[str, Any]) -> str:
        return str(side.get("label") or "")

    def _side_line(side: Dict[str, Any]) -> Dict[str, Any]:
        records = side.get("data") or []
        if not records:
            return {
                "heading": _side_heading(side),
                "xKey": "date",
                "series": [],
                "rows": [],
            }
        x_key = _find_key(records, _TIME_KEYS) or "date"
        numeric_keys = [
            k
            for r in records[:1]
            for k, v in r.items()
            if isinstance(v, (int, float))
            and k.lower() not in {t.lower() for t in _TIME_KEYS + ["id"]}
        ] or ["value"]
        series = [
            {"key": key, "label": str(key).replace("_", " ").title()}
            for key in numeric_keys
        ]
        rows = []
        for r in records:
            row: Dict[str, Any] = {x_key: r.get(x_key)}
            for idx, sk in enumerate(numeric_keys):
                value = r.get(sk)
                row[f"series{idx}"] = value
            rows.append(row)
        return {
            "heading": _side_heading(side),
            "xKey": x_key,
            "series": series,
            "rows": rows,
        }

    return {
        "left": _side_line(left_side),
        "right": _side_line(right_side),
    }


def _build_compare_pie(combined_result: Dict[str, Any]) -> Dict[str, Any]:
    """COMPARE_PIE_CHART: {left: {rows: [{label, value}]}, right: {rows: [{label, value}]}}"""
    left_side = combined_result.get("left") or {}
    right_side = combined_result.get("right") or {}

    def _side_heading(side: Dict[str, Any]) -> str:
        return str(side.get("label") or "")

    _PIE_LABEL_CANDIDATES = [
        "label",
        "sport",
        "grade",
        "leaveType",
        "bloodGroup",
        "disease",
        "status",
        "category",
        "name",
        "leavetype",
        "bmiCategory",
        "grading",
        "type",
    ]

    def _side_pie(side: Dict[str, Any]) -> Dict[str, Any]:
        records = side.get("data") or []
        if not records:
            return {"heading": _side_heading(side), "rows": []}
        label_key = _find_key(records, _PIE_LABEL_CANDIDATES)
        if not label_key:
            for r in records[:1]:
                for k, v in r.items():
                    if isinstance(v, str) and k.lower() not in ("id",):
                        label_key = k
                        break
        label_key = label_key or "label"
        value_key = (
            _find_key(
                records,
                [
                    "value",
                    "count",
                    "score",
                    "percentage",
                    "bestTotal",
                    "marksObtained",
                ],
            )
            or _find_numeric_key(records, ["id"])
            or "value"
        )
        rows = [
            {
                "label": str(r.get(label_key) or ""),
                "value": r.get(value_key) if r.get(value_key) is not None else 0,
            }
            for r in records
        ]
        return {"heading": _side_heading(side), "rows": rows}

    return {
        "left": _side_pie(left_side),
        "right": _side_pie(right_side),
    }


# Canonical type names coming from compare_engine — handle any legacy aliases.
_COMPARE_TYPE_ALIASES: Dict[str, str] = {
    "COMPARE_BAR_CHART": "COMPARE_CHART_BAR",
    "COMPARE_LINE_CHART": "COMPARE_CHART_LINE",
    "COMPARE_PIE_CHART": "COMPARE_CHART_PIE",
}


def build_comparison_widgets(
    combined_result: Dict[str, Any],
    base_title: str = "",
    visualization_intent: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Entry point for comparison visualization assembly.

    Returns an ordered list:
      [<primary_viz>, COMPARE_TABLE?]

    All widgets follow {type, title, data} with left/right structure.
    No internal fields (intent, dotnetPayload, metadata, endpoint) are included.
    """
    left_side = combined_result.get("left") or {}
    right_side = combined_result.get("right") or {}
    left_label = str(left_side.get("label") or "Left")
    right_label = str(right_side.get("label") or "Right")
    vs_title = f"{left_label} vs {right_label}"

    planned = _planned_widget_types(visualization_intent)
    viz_type = planned[0] if planned else "COMPARE_TABLE"
    if viz_type in _COMPARE_TYPE_ALIASES:
        viz_type = _COMPARE_TYPE_ALIASES[viz_type]
    if viz_type == "COMPARE_CARD":
        viz_type = "COMPARE_TABLE"

    # Preserve legacy fallback when no visualization plan is available.
    if not planned:
        raw_viz = combined_result.get("visualizationType") or "COMPARE_TABLE"
        viz_type = _COMPARE_TYPE_ALIASES.get(raw_viz, raw_viz)
        if viz_type == "COMPARE_CARD":
            viz_type = "COMPARE_TABLE"
        if isinstance(visualization_intent, dict):
            override = visualization_intent.get("comparison_chart_override")
            if override and isinstance(override, str):
                mapped = {
                    "line": "COMPARE_CHART_LINE",
                    "bar": "COMPARE_CHART_BAR",
                    "pie": "COMPARE_CHART_PIE",
                    "donut": "COMPARE_CHART_PIE",
                    "radial": "COMPARE_CHART_LINE",
                    "area": "COMPARE_CHART_LINE",
                }.get(override.lower())
                if mapped:
                    viz_type = mapped

    widgets: List[Dict[str, Any]] = []

    # Primary visualization first; no always-on summary card.
    if viz_type == "COMPARE_TABLE":
        widgets.append(
            {
                "type": "COMPARE_TABLE",
                "title": f"{vs_title} — Records",
                "data": _build_compare_table(combined_result),
            }
        )
    elif viz_type == "COMPARE_CHART_BAR":
        widgets.append(
            {
                "type": "COMPARE_CHART_BAR",
                "title": f"{vs_title} — Score Comparison",
                "data": _build_compare_bar(combined_result),
            }
        )
        widgets.append(
            {
                "type": "COMPARE_TABLE",
                "title": f"{vs_title} — Records",
                "data": _build_compare_table(combined_result),
            }
        )
    elif viz_type == "COMPARE_CHART_LINE":
        widgets.append(
            {
                "type": "COMPARE_CHART_LINE",
                "title": f"{vs_title} — Trend",
                "data": _build_compare_line(combined_result),
            }
        )
    elif viz_type == "COMPARE_CHART_PIE":
        widgets.append(
            {
                "type": "COMPARE_CHART_PIE",
                "title": f"{vs_title} — Distribution",
                "data": _build_compare_pie(combined_result),
            }
        )
    return widgets


def _build_widget_data(
    spec: Any,  # WidgetSpec — typed loosely to avoid circular import
    combined_result: Any,
    query_type: str,
    intent: Dict[str, Any],
    analysis: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Dispatch to the appropriate builder for a single WidgetSpec."""
    wt = spec.widget_type
    hint = spec.source_hint

    if wt == "COMPARE_CARD":
        return _build_compare_card(combined_result)
    elif wt == "COMPARE_TABLE":
        return _build_compare_table(combined_result)
    elif wt in ("COMPARE_BAR_CHART", "COMPARE_CHART_BAR"):
        return _build_compare_bar(combined_result)
    elif wt in ("COMPARE_LINE_CHART", "COMPARE_CHART_LINE"):
        return _build_compare_line(combined_result)
    elif wt in ("COMPARE_PIE_CHART", "COMPARE_CHART_PIE"):
        return _build_compare_pie(combined_result)

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
            raw = left.get("data") if isinstance(left, dict) else []
            flat = flatten_records(raw or [], deep_flatten=True)
        elif hint == "right" and isinstance(combined_result, dict):
            right = combined_result.get("right") or {}
            raw = right.get("data") if isinstance(right, dict) else []
            flat = flatten_records(raw or [], deep_flatten=True)
        elif hint == "section" and isinstance(combined_result, dict):
            flat = []
            for sec in combined_result.get("sections") or []:
                if isinstance(sec, dict) and sec.get("label") == spec.section_label:
                    flat = flatten_records(sec.get("data") or [], deep_flatten=True)
                    break
        else:
            flat = _extract_records(combined_result, deep_flatten=True)
        return build_table_data(flat)

    # ── BAR_CHART ────────────────────────────────────────────────────────────
    if wt in ("BAR_CHART", "CHART_BAR"):
        if query_type in ("compare", "comparison"):
            # Fallback for dynamic schema
            return _build_compare_bar(combined_result)
        return build_bar_chart_data(combined_result)

    # ── LINE_CHART / AREA_CHART / RADIAL_CHART (folded into line) ────────────
    if wt in ("LINE_CHART", "AREA_CHART", "RADIAL_CHART", "CHART_LINE"):
        return build_line_chart_data(combined_result)

    # ── PIE_CHART / DONUT_CHART (folded into pie) ─────────────────────────────
    if wt in ("PIE_CHART", "DONUT_CHART", "CHART_PIE"):
        return build_pie_chart_data(combined_result)

    # ── ATTENDANCE_CALENDAR ──────────────────────────────────────────────────
    if wt == "ATTENDANCE_CALENDAR":
        return build_attendance_calendar_data(combined_result, intent)

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
    effective_visualization_intent = _effective_visualization_intent(
        query_type, intent, combined_result, visualization_intent
    )

    from widget_selector import WidgetSelector

    # Ask the selector for the ordered spec list
    selector = WidgetSelector()
    specs = selector.select(
        query_type=query_type,
        intent=intent,
        combined_result=combined_result,
        primary_widget_type=infer_supported_type(
            combined_result, query_type, intent, effective_visualization_intent
        ),
        analysis=analysis,
        visualization_intent=effective_visualization_intent,
    )

    widgets: List[Dict[str, Any]] = []
    for spec in specs:
        try:
            data = _build_widget_data(
                spec, combined_result, query_type, intent, analysis
            )
            if (
                spec.widget_type == "TABLE"
                and isinstance(data, dict)
                and "row" not in data
            ):
                flat = _extract_records(combined_result, deep_flatten=True)
                table_data = build_table_data(flat)
                data["row"] = table_data.get("row") or []
                data["columns"] = table_data.get("columns") or []
            if (
                spec.widget_type != "ATTENDANCE_CALENDAR"
                and isinstance(combined_result, dict)
                and isinstance(data, dict)
            ):
                for key in ("degraded", "failedFilters", "matchCount"):
                    if key in combined_result:
                        data[key] = combined_result[key]
        except Exception:
            import logging as _logging

            _logging.getLogger(__name__).exception(
                "build_widget_list: failed building widget %s", spec.widget_id
            )
            data = {"_error": True}
        widgets.append(
            {
                "id": spec.widget_id,
                "type": spec.widget_type,
                "title": spec.title,
                "data": data,
            }
        )

    # Guard: never return an empty list — fall back to a plain TABLE
    if not widgets:
        flat = _extract_records(combined_result, deep_flatten=True)
        widgets.append(
            {
                "id": "fallback_table",
                "type": "TABLE",
                "title": "Results",
                "data": build_table_data(flat),
            }
        )

    # Validate: every widget must have non-None type, title, and data.
    # Any widget missing these fields is a bug — replace it with a safe fallback
    # rather than propagating nulls downstream.
    import logging as _logging

    _wlog = _logging.getLogger(__name__)
    validated: List[Dict[str, Any]] = []
    for w in widgets:
        if (
            not isinstance(w, dict)
            or not w.get("type")
            or w.get("title") is None
            or w.get("data") is None
        ):
            _wlog.error(
                "build_widget_list: null-field widget detected and replaced with fallback — "
                "widget=%r",
                w,
            )
            flat = _extract_records(combined_result, deep_flatten=True)
            validated.append(
                {
                    "id": w.get("id") or "fallback_table",
                    "type": "TABLE",
                    "title": w.get("title") or "Results",
                    "data": build_table_data(flat),
                }
            )
        else:
            validated.append(w)

    return validated

