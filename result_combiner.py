"""
result_combiner.py
==================

"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, cast

from compare_engine import compare_datasets
from cross_filter_engine import cross_filter_datasets
from utils import extract_chronological_key as _extract_chronological_key
from utils import extract_records as _normalize_records
from utils import get_score as _utils_get_score

logger = logging.getLogger(__name__)


# =============================================================================
# INTERNAL RECORD UTILITIES
# =============================================================================


def _extract_records(data: Any) -> List[Dict]:
    """Pull the list of records out of any .NET wrapper shape."""
    return _normalize_records(data)


# =============================================================================
# AGGREGATE HELPER
# =============================================================================

_GROUP_FIELD_MAP: Dict[str, List[str]] = {
    "section": ["sectionName", "section", "Section"],
    "sport": ["sports", "sport", "Sport"],
    "unit": ["teamName", "unitName", "unit", "Unit"],
    "class": ["class", "className", "Class"],
    "platoon": ["platoonName", "platoon"],
    "batch": ["batchName", "batch"],
}

def _get_score(record: Dict) -> Optional[float]:
    return _utils_get_score(record)


def aggregate_records(
    records: List[Dict],
    group_by: str,
    metric: str = "average_score",
) -> List[Dict]:
    """
    Group records by a dimension and compute an aggregate metric.

    Parameters
    ----------
    records  : flat list of agniveer dicts from .NET
    group_by : "section" | "sport" | "unit" | "class" | "platoon" | "batch"
    metric   : "average_score" | "count"

    Returns
    -------
    List of dicts sorted descending by the computed metric value.
    e.g. [{"group": "PPT", "count": 120, "averageScore": 74.3}, ...]
    """
    if not records or group_by not in _GROUP_FIELD_MAP:
        return []

    field_candidates = _GROUP_FIELD_MAP[group_by]
    buckets: Dict[str, List[float]] = {}

    for record in records:
        group_key: Optional[str] = None
        for field_name in field_candidates:
            raw = record.get(field_name)
            if raw is not None:
                raw_str = str(raw).strip()
                if "," in raw_str:
                    # comma-separated (e.g. sports field) — add to each
                    for part in raw_str.split(","):
                        part = part.strip()
                        if part:
                            buckets.setdefault(part, [])
                            score = _get_score(record)
                            if score is not None:
                                buckets[part].append(score)
                    group_key = None
                else:
                    group_key = raw_str
                break

        if group_key:
            score = _get_score(record)
            buckets.setdefault(group_key, [])
            if score is not None:
                buckets[group_key].append(score)

    if not buckets:
        return []

    results: List[Dict] = []
    for group_name, scores in buckets.items():
        row: Dict[str, Any] = {"group": group_name, "count": len(scores)}
        if scores and metric in ("average_score", "averageScore"):
            row["averageScore"] = round(sum(scores) / len(scores), 2)
        results.append(row)

    results.sort(
        key=lambda r: (r.get("averageScore", 0), r.get("count", 0)),
        reverse=True,
    )
    return results


# =============================================================================
# MERGE (MULTI_INDEPENDENT)
# =============================================================================


def merge_results(labeled_results: List[Tuple[str, Any]]) -> Dict[str, Any]:
    """
    Combine independent query results into a multi-section response.
    Each section retains its label and flat data array.
    """
    sections: List[Dict] = []
    for label, data in labeled_results:
        records = _extract_records(data)
        sections.append(
            {
                "label": label,
                "type": "multi_independent",
                "data": records,
            }
        )

    logger.info("merge_results: %d sections", len(sections))

    return {
        "queryType": "multi_independent",
        "sectionCount": len(sections),
        "sections": sections,
    }


# =============================================================================
# COMPARISON
# =============================================================================


def process_trend(raw_results: List[Any], intent: Dict[str, Any]) -> Dict[str, Any]:
    records = []
    for res in raw_results:
        records.extend(_extract_records(res))

    if any(k in r for r in records for k in ("date", "Date")):
        granularity = "daily"
    elif any(k in r for r in records for k in ("month", "Month")):
        granularity = "monthly"
    else:
        granularity = "daily"

    points: Dict[Any, List[float]] = {}
    for r in records:
        key = _extract_chronological_key(r)
        if key is not None:
            score = _get_score(r)
            if score is None:
                # Old code used falsy check 'r.get("present") or r.get("Present")' which evaluated False as None when the other was missing.
                present = None
                if "present" in r:
                    present = r["present"]
                elif "Present" in r:
                    present = r["Present"]
                if present is not None:
                    score = 1.0 if present else 0.0
                else:
                    score = 1.0
            points.setdefault(key, []).append(score)

    sorted_keys = sorted(points.keys())
    chart_data = []
    for k in sorted_keys:
        vals = points[k]
        avg_val = round(sum(vals) / len(vals), 2) if vals else 0.0
        if isinstance(k, tuple):
            label = f"{k[0]}-{k[1]}"
        else:
            label = str(k)
        chart_data.append({"label": label, "value": avg_val})

    trend_direction = "stable"
    increase = False
    decrease = False
    stable = True

    if len(chart_data) >= 2:
        first_val = cast(float, chart_data[0]["value"])
        last_val = cast(float, chart_data[-1]["value"])
        diff = last_val - first_val
        threshold = max(2.0, abs(first_val) * 0.05)
        if abs(diff) > threshold:
            stable = False
            if diff > 0:
                trend_direction = "increase"
                increase = True
            else:
                trend_direction = "decrease"
                decrease = True

    return {
        "queryType": "trend",
        "granularity": granularity,
        "trendDirection": trend_direction,
        "increase": increase,
        "decrease": decrease,
        "stable": stable,
        "chartData": chart_data,
        "records": records,
    }


def process_distribution(
    raw_results: List[Any], intent: Dict[str, Any]
) -> Dict[str, Any]:
    records = []
    for res in raw_results:
        records.extend(_extract_records(res))

    group_by = intent.get("group_by") or intent.get("groupBy")
    if not group_by:
        sample = records[0] if records else {}
        for dim, fields in _GROUP_FIELD_MAP.items():
            if any(f in sample for f in fields):
                group_by = dim
                break
        if not group_by:
            for k in (
                "leaveType",
                "leave_type",
                "leaveStatus",
                "category",
                "sport",
                "class",
                "platoon",
                "batch",
            ):
                if any(k in r for r in records):
                    group_by = k
                    break
            if not group_by:
                group_by = "category"

    buckets: Dict[str, int] = {}
    field_candidates = _GROUP_FIELD_MAP.get(group_by, [group_by])

    for r in records:
        group_val = None
        for field_candidate in field_candidates:
            val = r.get(field_candidate)
            if val is not None:
                group_val = str(val).strip()
                break

        if group_val is None:
            for k in (group_by, group_by.lower(), group_by.capitalize()):
                val = r.get(k)
                if val is not None:
                    group_val = str(val).strip()
                    break

        if not group_val:
            group_val = "Unknown"

        if "," in group_val:
            for part in group_val.split(","):
                part = part.strip()
                if part:
                    buckets[part] = buckets.get(part, 0) + 1
        else:
            buckets[group_val] = buckets.get(group_val, 0) + 1

    sorted_buckets = sorted(buckets.items(), key=lambda x: x[1], reverse=True)
    labels = [item[0] for item in sorted_buckets]
    values = [item[1] for item in sorted_buckets]

    return {
        "queryType": "distribution",
        "groupBy": group_by,
        "labels": labels,
        "values": values,
        "records": records,
    }


def combine_results(
    raw_results: List[Any],
    labeled_results: List[Tuple[str, Any]],
    qtype_str: str,
    primary_intent: Dict[str, Any],
    comparison_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Perform the result combination phase.
    Delegates to appropriate intersection/comparison/merge/trend/distribution strategy based on qtype_str.
    """
    if qtype_str == "cross_filter":
        logger.info(
            "result_combiner: cross_filter_datasets across %d sets", len(raw_results)
        )
        labels = [label for label, _ in labeled_results] if labeled_results else None
        return cross_filter_datasets(raw_results, primary_index=0, labels=labels)
    elif qtype_str in ("comparison", "compare"):
        logger.info(
            "result_combiner: compare_datasets across %d sides", len(labeled_results)
        )
        res = compare_datasets(labeled_results, comparison_context=comparison_context)
        res["queryType"] = "comparison"
        return res

    elif qtype_str in ("multi_independent", "multi_operation"):
        logger.info(
            "result_combiner: merge_results across %d sections", len(labeled_results)
        )
        return merge_results(labeled_results)
    elif qtype_str == "trend":
        logger.info("result_combiner: process_trend")
        return process_trend(raw_results, primary_intent)
    elif qtype_str == "distribution":
        logger.info("result_combiner: process_distribution")
        return process_distribution(raw_results, primary_intent)
    else:
        logger.info("result_combiner: simple passthrough")
        return raw_results[0] if raw_results else {}


# Backward-compatibility aliases (old names used by some tests)
intersect_results = cross_filter_datasets
compare_results = compare_datasets
