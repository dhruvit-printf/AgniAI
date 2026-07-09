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
from datetime import datetime as _datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import ValidationError

from normalized_models import extract_records as _orig_extract_records
from widget_types import CHART_TYPE_ALIASES as _CHART_TYPE_ALIASES
from widget_types import COMPARE_CHART_OVERRIDE_MAP as _COMPARE_CHART_OVERRIDE_MAP
from widget_types import canonical_widget_type as _canonical_widget_type

logger = logging.getLogger(__name__)

# Frontend widget-type-override dropdown never offers compare-chart types or
# ATTENDANCE_CALENDAR directly, so this excludes _CHART_TYPE_ALIASES' 3
# COMPARE_* keys — computed once from the shared alias map instead of being
# hand-typed a second time.
_NON_COMPARE_CANONICAL_TYPES = {"TABLE", "CARD", "CHART_BAR", "CHART_LINE", "CHART_PIE"} | {
    k for k in _CHART_TYPE_ALIASES if not k.startswith("COMPARE_")
}


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
    # Deliberately excludes the COMPARE_* aliases and ATTENDANCE_CALENDAR —
    # the frontend widget-type override dropdown never offers those.
    if text.upper() in _NON_COMPARE_CANONICAL_TYPES:
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
        used_keys = set()

        # 1. Title
        title_cands = ["fullName", "name"]
        id_key = _find_key([r], ["id"])
        card_title = title if title else (f"Record {r.get(id_key, '')}" if id_key else "Details")
        k = _find_key([r], title_cands)
        if k:
            card_title = r[k]
            used_keys.add(k)

        # 2. Value
        card_value = ""
        count_key = next((k for k in r.keys() if k.lower().endswith("count") and k.lower() != "count"), None)
        
        if count_key:
            card_value = r[count_key]
            used_keys.add(count_key)
        else:
            val_cands = ["bestTotal", "score", "marksObtained", "count", "status", "leaveStatus", "totalAgniveers"]
            k = _find_key([r], val_cands)
            if k:
                card_value = r[k]
                used_keys.add(k)

        # 3. Description
        desc_key = _find_key([r], ["description", "details"])
        description = str(r.get(desc_key, "") or "") if desc_key else ""
        if desc_key:
            used_keys.add(desc_key)
        if id_key:
            used_keys.add(id_key)

        card_obj = {
            "title": str(card_title),
            "value": str(card_value),
            "description": str(description),
        }
        for k, v in r.items():
            if k not in used_keys and not str(k).lower().endswith("id"):
                card_obj[k] = v
                
        cards.append(card_obj)
    if not cards:
        cards.append(
            {
                "title": title,
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
    "_DisplayOrder",
)

# Top-level fields that are internal and should not appear as columns.
_EXCLUDED_COLUMN_KEYS_EXACT = {
    "ID",
    "id",
    "DisplayOrder",
    "displayOrder",
    "success",
    "Success",
    "commandLabel",
    "CommandLabel",
    "message",
    "Message",
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
    """Return the first field key matching any candidate (case-insensitive).

    Also matches the trailing segment of a flattened/nested key (e.g.
    'Performance_BestTotal' matches candidate 'bestTotal'), so fields that
    arrive nested from .NET are still found once the record has been
    deep-flattened.
    """
    for c in candidates:
        cl = c.lower()
        for r in records[:1]:
            for k in r.keys():
                if k.lower() == cl:
                    return k
    for c in candidates:
        cl = c.lower()
        for r in records[:1]:
            for k in r.keys():
                if "_" in k and k.rsplit("_", 1)[-1].lower() == cl:
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


def _pivot_distribution(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not records or len(records) != 1:
        return records
    rec = records[0]
    num_keys = [k for k, v in rec.items() if isinstance(v, (int, float))]
    other_keys = [k for k in rec.keys() if k != 'label']
    
    if len(num_keys) >= 2 and len(num_keys) == len(other_keys):
        pivoted = []
        for k in num_keys:
            pivoted.append({'label': k, 'value': rec[k]})
        return pivoted
    return records


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
    # Flatten any remaining nested sub-objects (e.g. a "performance": {...}
    # block) now that period-list explosion has already run — otherwise
    # metrics nested by .NET would never be found below and the chart would
    # render as all-zero.
    records = flatten_records(records, deep_flatten=True)
    records = _pivot_distribution(records)
    # Flattening re-cases top-level keys (e.g. "month" -> "Month"), so
    # re-resolve period_key to whatever casing actually survived.
    if period_key:
        period_key = _find_key(records, [period_key]) or period_key

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

    grouped_rows = {}
    for r in records:
        x_val = r.get(x_key) or (r.get(identity_key) if identity_key else None) or "Category"
        y_val = r.get(y_key) if r.get(y_key) is not None else 0
        if not isinstance(y_val, (int, float)):
            try: y_val = float(y_val)
            except: y_val = 0
        if isinstance(y_val, float) and y_val.is_integer(): y_val = int(y_val)

        if x_val not in grouped_rows:
            grouped_rows[x_val] = {x_key: x_val, y_key: 0}
            for k, v in r.items():
                if k not in (x_key, y_key) and not str(k).lower().endswith("id"):
                    grouped_rows[x_val].setdefault(k, v)
        grouped_rows[x_val][y_key] += y_val

    rows = list(grouped_rows.values())

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
    # Flatten remaining nested sub-objects (post period-explosion) so nested
    # numeric metrics (e.g. "performance": {"bestTotal": 92}) surface as
    # plottable series instead of vanishing.
    records = flatten_records(records, deep_flatten=True)

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

    grouped_rows = {}
    for r in records:
        x_val = r.get(x_key, "")
        if x_val not in grouped_rows:
            grouped_rows[x_val] = {x_key: x_val}
            for idx, sk in enumerate(numeric_keys):
                grouped_rows[x_val][f"series{idx}"] = 0
            for k, v in r.items():
                if k != x_key and k not in numeric_keys and not str(k).lower().endswith("id"):
                    grouped_rows[x_val].setdefault(k, v)
                
        for idx, sk in enumerate(numeric_keys):
            val = r.get(sk, 0)
            if not isinstance(val, (int, float)):
                try: val = float(val)
                except: val = 0
            if isinstance(val, float) and val.is_integer(): val = int(val)
            grouped_rows[x_val][f"series{idx}"] += val

    rows = list(grouped_rows.values())

    return {"xKey": x_key, "series": series, "rows": rows}


def build_pie_chart_data(
    combined_result: Any, series_label: str = "Distribution", intent: Optional[Dict[str, Any]] = None
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
    # Flatten remaining nested sub-objects (post period-explosion) so nested
    # fields (e.g. "performance": {"grade": "A", "bestTotal": 92}) are
    # discoverable as slice label/value instead of vanishing.
    records = flatten_records(records, deep_flatten=True)
    records = _pivot_distribution(records)

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
                "totalLeave",
                "totalLeaves",
                "leaveTaken",
                "days",
            ],
        )
        or _find_numeric_key(records, ["id"])
        or "value"
    )

    grouped_rows = {}
    for r in records:
        lbl = str(r.get(label_key) or r.get("fullName") or "Category")
        val = r.get(value_key) if r.get(value_key) is not None else 1
        if not isinstance(val, (int, float)):
            try: val = float(val)
            except: val = 0
        if isinstance(val, float) and val.is_integer(): val = int(val)
        
        if lbl not in grouped_rows:
            grouped_rows[lbl] = {"value": 0}
            for k, v in r.items():
                if k not in (label_key, value_key) and not str(k).lower().endswith("id"):
                    grouped_rows[lbl].setdefault(k, v)
        grouped_rows[lbl]["value"] += val

    is_perf_avg = False
    if intent:
        mod = str(intent.get("module") or "").lower()
        op = str(intent.get("operation") or "").lower()
        if mod == "performance" and op == "average":
            is_perf_avg = True

    rows = []
    for k, grp in grouped_rows.items():
        row = {"label": k}
        if is_perf_avg and "value" in grp:
            grp["Average Score"] = grp.pop("value")
        row.update(grp)
        rows.append(row)

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


def _parse_calendar_date(value: Any) -> Optional["_datetime"]:
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
        records = _extract_records(source_result, deep_flatten=True)
        data_payload = build_card_data(records, title)
    elif inferred_type in {"CHART_BAR", "BAR_CHART"}:
        data_payload = build_bar_chart_data(source_result)
    elif inferred_type in {"CHART_LINE", "LINE_CHART", "AREA_CHART", "RADIAL_CHART"}:
        data_payload = build_line_chart_data(source_result)
    elif inferred_type in {"CHART_PIE", "PIE_CHART", "DONUT_CHART"}:
        data_payload = build_pie_chart_data(source_result, intent=intent)
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

    def _is_meaningful(payload: Any) -> bool:
        if not payload or not isinstance(payload, dict):
            return False
        if "rows" in payload:
            return bool(payload.get("rows"))
        if "row" in payload:
            return bool(payload.get("row"))
        if "sections" in payload:
            return any(
                sec.get("row") or sec.get("rows")
                for sec in payload.get("sections", [])
                if isinstance(sec, dict)
            )
        if "left" in payload and "right" in payload:
            left_has = bool(
                isinstance(payload.get("left"), dict)
                and (payload["left"].get("row") or payload["left"].get("rows"))
            )
            right_has = bool(
                isinstance(payload.get("right"), dict)
                and (payload["right"].get("row") or payload["right"].get("rows"))
            )
            return left_has or right_has
        if "sides" in payload:
            return any(
                side.get("data")
                for side in payload.get("sides", [])
                if isinstance(side, dict)
            )
        if "dates" in payload:
            return bool(payload.get("dates"))
        if "cards" in payload:
            return bool(payload.get("cards"))
        # Fallback for CARD widget which has "value" and "description"
        if "value" in payload:
            return payload.get("value") is not None
        # Allow pass-through if we have unknown keys and some values
        return any(v is not None for v in payload.values())

    if not _is_meaningful(data_payload):
        data_payload = None

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
    dumped = fd.model_dump()
    if dumped.get("data") is None:
        del dumped["data"]
    elif isinstance(dumped.get("data"), dict) and not dumped["data"]:
        del dumped["data"]
        
    return dumped


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
        records = flatten_records(side.get("data") or [], deep_flatten=True)
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
        grouped_rows = {}
        for r in records:
            x_val = r.get(x_key)
            y_val = r.get(y_key)
            if not isinstance(y_val, (int, float)):
                try: y_val = float(y_val)
                except: y_val = 0
            if isinstance(y_val, float) and y_val.is_integer(): y_val = int(y_val)

            if x_val not in grouped_rows:
                grouped_rows[x_val] = {x_key: x_val, y_key: 0}
                for k, v in r.items():
                    if k not in (x_key, y_key) and not str(k).lower().endswith("id"):
                        grouped_rows[x_val].setdefault(k, v)
            grouped_rows[x_val][y_key] += y_val

        rows = list(grouped_rows.values())
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
        records = flatten_records(side.get("data") or [], deep_flatten=True)
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
            and not _is_identity_key(k)
        ] or ["value"]
        series = [
            {"key": f"series{idx}", "label": str(key).replace("_", " ").title()}
            for idx, key in enumerate(numeric_keys)
        ]
        grouped_rows = {}
        for r in records:
            x_val = r.get(x_key)
            if x_val not in grouped_rows:
                grouped_rows[x_val] = {x_key: x_val}
                for idx, sk in enumerate(numeric_keys):
                    grouped_rows[x_val][f"series{idx}"] = 0
                for k, v in r.items():
                    if k != x_key and k not in numeric_keys and not str(k).lower().endswith("id"):
                        grouped_rows[x_val].setdefault(k, v)
                    
            for idx, sk in enumerate(numeric_keys):
                val = r.get(sk, 0)
                if not isinstance(val, (int, float)):
                    try: val = float(val)
                    except: val = 0
                if isinstance(val, float) and val.is_integer(): val = int(val)
                grouped_rows[x_val][f"series{idx}"] += val

        rows = list(grouped_rows.values())
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


def _build_compare_pie(combined_result: Dict[str, Any], intent: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
        records = flatten_records(side.get("data") or [], deep_flatten=True)
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
        grouped_rows = {}
        for r in records:
            lbl = str(r.get(label_key) or "")
            val = r.get(value_key) if r.get(value_key) is not None else 0
            if not isinstance(val, (int, float)):
                try: val = float(val)
                except: val = 0
            if isinstance(val, float) and val.is_integer(): val = int(val)
            
            if lbl not in grouped_rows:
                grouped_rows[lbl] = {"value": 0}
                for k, v in r.items():
                    if k not in (label_key, value_key) and not str(k).lower().endswith("id"):
                        grouped_rows[lbl].setdefault(k, v)
            grouped_rows[lbl]["value"] += val

        is_perf_avg = False
        if intent:
            mod = str(intent.get("module") or "").lower()
            op = str(intent.get("operation") or "").lower()
            if mod == "performance" and op == "average":
                is_perf_avg = True

        rows = []
        for k, grp in grouped_rows.items():
            row = {"label": k}
            if is_perf_avg and "value" in grp:
                grp["Average Score"] = grp.pop("value")
            row.update(grp)
            rows.append(row)
        return {"heading": _side_heading(side), "rows": rows}

    return {
        "left": _side_pie(left_side),
        "right": _side_pie(right_side),
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
    if viz_type in _CHART_TYPE_ALIASES:
        viz_type = _CHART_TYPE_ALIASES[viz_type]
    if viz_type == "COMPARE_CARD":
        viz_type = "COMPARE_TABLE"

    # Preserve legacy fallback when no visualization plan is available.
    if not planned:
        raw_viz = combined_result.get("visualizationType") or "COMPARE_TABLE"
        viz_type = _CHART_TYPE_ALIASES.get(raw_viz, raw_viz)
        if viz_type == "COMPARE_CARD":
            viz_type = "COMPARE_TABLE"
        if isinstance(visualization_intent, dict):
            override = visualization_intent.get("comparison_chart_override")
            if override and isinstance(override, str):
                mapped = _COMPARE_CHART_OVERRIDE_MAP.get(override.lower())
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
    """Dispatch to the appropriate builder for a single WidgetSpec.

    `wt` is re-canonicalized defensively (WidgetSelector already canonicalizes
    every spec it produces, so this is normally a no-op) so BUILDERS below
    only ever needs canonical keys, never legacy aliases.
    """
    wt = _canonical_widget_type(spec.widget_type)
    hint = spec.source_hint

    # ── Compare widgets — pure type -> builder dispatch, checked first so a
    # COMPARE_* type always wins regardless of hint/query_type below. ───────
    compare_builders: Dict[str, Any] = {
        "COMPARE_CARD": lambda: _build_compare_card(combined_result),
        "COMPARE_TABLE": lambda: _build_compare_table(combined_result),
        "COMPARE_CHART_BAR": lambda: _build_compare_bar(combined_result),
        "COMPARE_CHART_LINE": lambda: _build_compare_line(combined_result),
        "COMPARE_CHART_PIE": lambda: _build_compare_pie(combined_result, intent=intent),
    }
    if wt in compare_builders:
        return compare_builders[wt]()

    # ── Summary CARD (from analysis.statistics) — an unconditional hint
    # pre-empt, not a type-based branch, so it stays outside the dict. ──────
    if hint == "summary":
        return build_summary_card_from_analysis(analysis, query_type, intent)

    # ── Primary CARD (from raw records) ─────────────────────────────────────
    if wt == "CARD":
        if hint == "section" and isinstance(combined_result, dict):
            records: List[Dict[str, Any]] = []
            for sec in combined_result.get("sections") or []:
                if isinstance(sec, dict) and sec.get("label") == spec.section_label:
                    records = flatten_records(sec.get("data") or [], deep_flatten=True)
                    break
        else:
            records = _extract_records(combined_result, deep_flatten=True)
            
        raw_query = str(intent.get("raw_query") or "").strip().lower()
        if raw_query.startswith("how many ") or raw_query.startswith("count "):
            val = str(len(records))
            if len(records) == 1:
                r = records[0]
                # If the single record is an aggregate response from .NET, use the count value instead.
                count_key = next((k for k in r.keys() if k.lower().endswith("count") and k.lower() != "count"), None)
                if count_key:
                    val = str(r[count_key])
                else:
                    k = _find_key([r], ["count", "totalAgniveers", "value"])
                    if k:
                        val = str(r[k])
                    else:
                        # Category->count distribution (e.g. Grading Summary
                        # {"DRILL (AMT)": {"Excellent": 188, "Good": 393, "SAT": 1}})
                        # — sum the nested numeric leaves for the real total
                        # instead of reporting "1" for the single wrapper record.
                        leaves = [
                            sub_v
                            for v in r.values()
                            if isinstance(v, dict)
                            for sub_v in v.values()
                            if isinstance(sub_v, (int, float)) and not isinstance(sub_v, bool)
                        ]
                        if leaves:
                            total = sum(leaves)
                            val = str(int(total) if float(total).is_integer() else total)
            return {"cards": [{"title": spec.title or "Total Count", "value": val, "description": ""}]}

        return build_card_data(records, spec.title)

    # ── TABLE — left/right/section/primary (hint-dependent data selection,
    # not a pure type dispatch, so it stays its own branch) ─────────────────
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

    # For a multi_independent section widget, only that section's own
    # records should feed the chart builder below — otherwise records from
    # every other independent section get flattened together and blend into
    # the same bars/slices (e.g. a Leave pie chart picking up Performance
    # rows' bestTotal/grade fields from a sibling section).
    chart_source = combined_result
    if hint == "section" and isinstance(combined_result, dict):
        chart_source = []
        for sec in combined_result.get("sections") or []:
            if isinstance(sec, dict) and sec.get("label") == spec.section_label:
                chart_source = sec.get("data") or []
                break

    # ── CHART_BAR has a query_type-dependent special case (falls back to the
    # compare builder for a dynamic/legacy schema), so it's resolved before
    # the pure chart-type dict dispatch below. ──────────────────────────────
    if wt == "CHART_BAR" and query_type in ("compare", "comparison"):
        return _build_compare_bar(combined_result)

    chart_builders: Dict[str, Any] = {
        "CHART_BAR": lambda: build_bar_chart_data(chart_source),
        "CHART_LINE": lambda: build_line_chart_data(chart_source),
        "CHART_PIE": lambda: build_pie_chart_data(chart_source, intent=intent),
        "ATTENDANCE_CALENDAR": lambda: build_attendance_calendar_data(
            chart_source, intent
        ),
    }
    if wt in chart_builders:
        return chart_builders[wt]()

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
    section_labels: List[str] = []
    if query_type == "multi_independent" and isinstance(combined_result, dict):
        sections = combined_result.get("sections") or []
        if isinstance(sections, list):
            section_labels = [
                str(sec.get("label") or "") if isinstance(sec, dict) else ""
                for sec in sections
            ]

    for index, spec in enumerate(specs):
        resolved_section_label = spec.section_label
        if (
            query_type == "multi_independent"
            and isinstance(combined_result, dict)
            and not resolved_section_label
            and index < len(section_labels)
            and section_labels[index]
        ):
            resolved_section_label = section_labels[index]
            spec.section_label = resolved_section_label
            spec.source_hint = "section"

        if (
            query_type == "multi_independent"
            and isinstance(combined_result, dict)
            and resolved_section_label
        ):
            spec.widget_id = WidgetSelector._widget_id(
                spec.widget_type,
                "",
                "",
                resolved_section_label,
                index,
            )

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

