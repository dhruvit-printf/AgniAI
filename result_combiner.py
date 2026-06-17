"""
result_combiner.py
==================
Combines results from multiple .NET API calls for the Query Planning Layer.

This is Step 4 in the AgniAI intelligence pipeline.
The output — finalResult — is the SOURCE OF TRUTH passed to the Report Generator.
The Report Generator must never modify it.

Supported combination strategies:
  CROSS_FILTER      → intersect_results   (N-way ID intersection)
  COMPARISON        → compare_results     (side-by-side metric extraction)
  MULTI_INDEPENDENT → merge_results       (combine independent sections)
  SIMPLE / other    → caller passes through unchanged (no combiner needed)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# INTERNAL RECORD UTILITIES
# =============================================================================

def _extract_records(data: Any) -> List[Dict]:
    """Pull the list of records out of any .NET wrapper shape."""
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in (
            "data", "Data", "result", "Result",
            "records", "Records", "persons", "personnel",
        ):
            val = data.get(key)
            if isinstance(val, list):
                return val
            if isinstance(val, dict):
                return _extract_records(val)

        # Distribution "teams" → flatten members
        teams = data.get("teams") or data.get("Teams")
        if isinstance(teams, list):
            members: List[Dict] = []
            for team in teams:
                team_members = team.get("members") or team.get("Members") or []
                if isinstance(team_members, list):
                    members.extend(team_members)
            if members:
                return members

    return []


def _extract_agniveer_ids(records: List[Dict]) -> Set[int]:
    ids: Set[int] = set()
    for record in records:
        for key in ("agniveerId", "AgniveerId", "AgniVeerId", "id", "Id"):
            val = record.get(key)
            if val is not None:
                try:
                    ids.add(int(val))
                except (ValueError, TypeError):
                    pass
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
    "section":  ["sectionName", "section", "Section"],
    "sport":    ["sports", "sport", "Sport"],
    "unit":     ["teamName", "unitName", "unit", "Unit"],
    "class":    ["class", "className", "Class"],
    "platoon":  ["platoonName", "platoon"],
    "batch":    ["batchName", "batch"],
}

_SCORE_FIELDS = [
    "bestTotal", "totalMarks", "score", "Score",
    "omrInputTotal", "marksObtained",
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
    Compute the N-way intersection of result sets by agniveerId.

    The primary_index set supplies the full record objects for the filtered list.
    All other sets are used only for their ID sets.
    """
    if not result_sets:
        return {
            "queryType":         "cross_filter",
            "filterDepth":       0,
            "matchCount":        0,
            "totalBeforeFilter": 0,
            "records":           [],
        }

    all_record_sets = [_extract_records(rs) for rs in result_sets]
    all_id_sets     = [_extract_agniveer_ids(recs) for recs in all_record_sets]

    if not all_id_sets or any(len(ids) == 0 for ids in all_id_sets):
        common_ids: Set[int] = set()
    else:
        common_ids = all_id_sets[0]
        for id_set in all_id_sets[1:]:
            common_ids = common_ids & id_set

    primary_index   = min(primary_index, len(all_record_sets) - 1)
    primary_records = all_record_sets[primary_index]
    total_before    = len(primary_records)

    filtered: List[Dict] = []
    for record in primary_records:
        record_id: Optional[int] = None
        for key in ("agniveerId", "AgniveerId", "AgniVeerId", "id", "Id"):
            val = record.get(key)
            if val is not None:
                try:
                    record_id = int(val)
                except (ValueError, TypeError):
                    pass
                break
        if record_id is not None and record_id in common_ids:
            filtered.append(record)

    logger.info(
        "intersect_results: depth=%d total_before=%d matched=%d",
        len(result_sets), total_before, len(filtered),
    )

    return {
        "queryType":         "cross_filter",
        "filterDepth":       len(result_sets),
        "matchCount":        len(filtered),
        "totalBeforeFilter": total_before,
        "records":           filtered,
    }


# =============================================================================
# MERGE (MULTI_INDEPENDENT)
# =============================================================================

def merge_results(labeled_results: List[Tuple[str, Any]]) -> Dict[str, Any]:
    """
    Combine independent query results into a multi-section response.
    Each section retains its label and full data for the formatter.
    """
    sections: List[Dict] = []
    for label, data in labeled_results:
        records = _extract_records(data)
        sections.append({
            "label":       label,
            "data":        data,
            "recordCount": len(records),
        })

    logger.info("merge_results: %d sections", len(sections))

    return {
        "queryType":    "multi_independent",
        "sectionCount": len(sections),
        "sections":     sections,
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
            metrics["topScore"]     = max(scores)
            metrics["bottomScore"]  = min(scores)

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
        metrics = _extract_summary_metrics(data)
        all_metric_keys.update(metrics.keys())
        sides.append({
            "label":   label,
            "data":    data,
            "metrics": metrics,
        })

    logger.info(
        "compare_results: %d sides, metrics=%s",
        len(sides), sorted(all_metric_keys),
    )

    return {
        "queryType":       "comparison",
        "sides":           sides,
        "comparedMetrics": sorted(all_metric_keys),
    }