"""
result_combiner.py
==================

"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple, cast

from dotnet_adapter import extract_records as _normalize_records
from cross_filter_engine import cross_filter_datasets
from compare_engine import compare_datasets

logger = logging.getLogger(__name__)


# =============================================================================
# INTERNAL RECORD UTILITIES
# =============================================================================


def _extract_records(data: Any) -> List[Dict]:
    """Pull the list of records out of any .NET wrapper shape."""
    return _normalize_records(data)


def _extract_agniveer_ids(records: List[Dict]) -> Set[str]:
    ids: Set[str] = set()
    for record in records:
        val = record.get("agniveerNo")
        if val is not None:
            ids.add(str(val).strip())
        else:
            for key in ("agniveerId", "AgniveerId", "AgniVeerId", "id", "Id"):
                fallback_val = record.get(key)
                if fallback_val is not None:
                    ids.add(str(fallback_val).strip())
                    break
    return ids


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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

_SCORE_FIELDS = [
    "bestTotal",
    "totalMarks",
    "score",
    "Score",
    "omrInputTotal",
    "marksObtained",
]


def _get_field(record: Dict, *keys) -> Any:
    for key in keys:
        val = record.get(key)
        if val is not None:
            return val
    return None


def _get_score(record: Dict) -> Optional[float]:
    for field in _SCORE_FIELDS:
        v = _safe_float(record.get(field))
        if v is not None:
            return v
    return None


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
# N-WAY INTERSECTION (CROSS_FILTER)
# =============================================================================


def intersect_results(
    result_sets: List[Any],
    primary_index: int = 0,
) -> Dict[str, Any]:
    """
    Compute the N-way intersection of result sets by agniveerNo (with agniveerId fallback).

    The primary_index set supplies the full record objects for the filtered list.
    All other sets are used only for their ID sets.
    """
    if not result_sets:
        return {
            "queryType": "cross_filter",
            "filterDepth": 0,
            "matchCount": 0,
            "totalBeforeFilter": 0,
            "records": [],
        }

    all_record_sets = [_extract_records(rs) for rs in result_sets]
    all_id_sets = [_extract_agniveer_ids(recs) for recs in all_record_sets]

    if not all_id_sets or any(len(ids) == 0 for ids in all_id_sets):
        common_ids: Set[str] = set()
    else:
        common_ids = all_id_sets[0]
        for id_set in all_id_sets[1:]:
            common_ids = common_ids & id_set

    primary_index = min(primary_index, len(all_record_sets) - 1)
    primary_records = all_record_sets[primary_index]
    total_before = len(primary_records)

    records_by_id = []
    for recs in all_record_sets:
        lookup = {}
        for record in recs:
            record_id = record.get("agniveerNo")
            if record_id is None:
                for key in ("agniveerId", "AgniveerId", "AgniVeerId", "id", "Id"):
                    val = record.get(key)
                    if val is not None:
                        record_id = str(val).strip()
                        break
            if record_id is not None:
                lookup[str(record_id).strip()] = record
        records_by_id.append(lookup)

    filtered: List[Dict] = []
    for record in primary_records:
        record_id: Optional[str] = record.get("agniveerNo")
        if record_id is None:
            for key in ("agniveerId", "AgniveerId", "AgniVeerId", "id", "Id"):
                val = record.get(key)
                if val is not None:
                    record_id = str(val).strip()
                    break
        if record_id is None or str(record_id).strip() not in common_ids:
            continue

        merged = dict(record)
        normalized_id = str(record_id).strip()
        for lookup in records_by_id:
            match = lookup.get(normalized_id)
            if not match:
                continue
            for key, value in match.items():
                if key not in merged or merged.get(key) in (None, ""):
                    merged[key] = value
        filtered.append(merged)

    logger.info(
        "intersect_results: depth=%d total_before=%d matched=%d",
        len(result_sets),
        total_before,
        len(filtered),
    )

    return {
        "queryType": "cross_filter",
        "filterDepth": len(result_sets),
        "matchCount": len(filtered),
        "totalBeforeFilter": total_before,
        "records": filtered,
    }


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


def _extract_summary_metrics(data: Any) -> Dict[str, Any]:
    """
    Extract comparable scalar metrics from any .NET response shape.
    Works for plain dicts, wrapped dicts, and record lists.
    """
    metrics: Dict[str, Any] = {}

    # Scalar fields from dict
    if isinstance(data, dict):
        inner = data
        for key in ("data", "Data", "result", "Result"):
            val = data.get(key)
            if isinstance(val, dict):
                inner = val
                break

        for k, v in inner.items():
            if isinstance(v, (int, float)):
                metrics[k] = v
            elif isinstance(v, str):
                try:
                    metrics[k] = float(v)
                except ValueError:
                    pass

    # Record-list metrics
    records = _extract_records(data)
    if records:
        metrics["recordCount"] = len(records)

        scores = [s for s in (_get_score(r) for r in records) if s is not None]
        if scores:
            metrics["averageScore"] = round(sum(scores) / len(scores), 2)
            metrics["topScore"] = max(scores)
            metrics["bottomScore"] = min(scores)

    return metrics


def compare_results(labeled_results: List[Tuple[str, Any]]) -> Dict[str, Any]:
    """
    Compare two or more .NET results side by side.

    Each side retains its full data (for the formatter) and extracted metrics
    (for display in analysis/conclusion).
    """
    sides: List[Dict] = []
    all_metric_keys: Set[str] = set()

    for label, data in labeled_results:
        is_unavailable = False
        if isinstance(data, dict) and data.get("unavailable") is True:
            is_unavailable = True
            metrics = {}
        else:
            metrics = _extract_summary_metrics(data)
            all_metric_keys.update(metrics.keys())

        side = {
            "label": label,
            "data": data,
            "metrics": metrics,
        }
        if is_unavailable:
            side["unavailable"] = True

        sides.append(side)

    logger.info(
        "compare_results: %d sides, metrics=%s",
        len(sides),
        sorted(all_metric_keys),
    )

    combined: Dict[str, Any] = {
        "queryType": "comparison",
        "sides": sides,
        "comparedMetrics": sorted(all_metric_keys),
    }

    if len(sides) >= 1:
        combined["left"] = sides[0]
    if len(sides) >= 2:
        combined["right"] = sides[1]

    comparison_metrics = {}
    for metric in all_metric_keys:
        valid_sides_with_metric = []
        for side in sides:
            val = _safe_float(side["metrics"].get(metric))
            if val is not None:
                valid_sides_with_metric.append((side["label"], val))

        if len(valid_sides_with_metric) >= 2:
            valid_sides_with_metric.sort(key=lambda x: x[1])
            lowest_label, lowest_val = valid_sides_with_metric[0]
            highest_label, highest_val = valid_sides_with_metric[-1]

            diff = highest_val - lowest_val
            pct = (diff / lowest_val * 100.0) if lowest_val != 0 else 0.0

            comparison_metrics[metric] = {
                "higher": highest_label,
                "lower": lowest_label,
                "difference": round(diff, 2),
                "percentage": round(pct, 2),
            }
        elif len(valid_sides_with_metric) == 1:
            label, val = valid_sides_with_metric[0]
            comparison_metrics[metric] = {
                "higher": label,
                "lower": "N/A",
                "difference": 0.0,
                "percentage": 0.0,
            }

    combined["comparison"] = comparison_metrics
    return combined


def _extract_chronological_key(record: Dict) -> Any:
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


def process_trend(raw_results: List[Any], intent: Dict[str, Any]) -> Dict[str, Any]:
    records = []
    for res in raw_results:
        records.extend(_extract_records(res))

    granularity = "daily"
    for k in ("date", "Date"):
        if any(k in r for r in records):
            granularity = "daily"
    for k in ("month", "Month"):
        if any(k in r for r in records):
            granularity = "monthly"

    points: Dict[Any, List[float]] = {}
    for r in records:
        key = _extract_chronological_key(r)
        if key is not None:
            score = _get_score(r)
            if score is None:
                present = r.get("present") or r.get("Present")
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
) -> Dict[str, Any]:
    """
    Perform the result combination phase.
    Delegates to appropriate intersection/comparison/merge/trend/distribution strategy based on qtype_str.
    """
    if qtype_str == "cross_filter":
        logger.info(
            "result_combiner: cross_filter_datasets across %d sets", len(raw_results)
        )
        return cross_filter_datasets(raw_results, primary_index=0)
    elif qtype_str in ("comparison", "compare"):
        logger.info(
            "result_combiner: compare_datasets across %d sides", len(labeled_results)
        )
        return compare_datasets(labeled_results)
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
