"""
result_combiner.py
==================
Combines results from multiple .NET API calls for the Query Planning Layer.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


def _extract_records(data: Any) -> List[Dict]:
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ("data", "Data", "result", "Result", "records", "Records", "persons", "personnel"):
            val = data.get(key)
            if isinstance(val, list):
                return val
            if isinstance(val, dict):
                return _extract_records(val)

        teams = data.get("teams") or data.get("Teams")
        if isinstance(teams, list):
            members = []
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


def intersect_results(result_sets: List[Any], primary_index: int = 0) -> Dict[str, Any]:
    if not result_sets:
        return {"queryType": "cross_filter", "matchCount": 0, "totalBeforeFilter": 0, "records": []}

    all_record_sets = [_extract_records(rs) for rs in result_sets]
    all_id_sets = [_extract_agniveer_ids(recs) for recs in all_record_sets]

    if not all_id_sets or any(len(ids) == 0 for ids in all_id_sets):
        common_ids: Set[int] = set()
    else:
        common_ids = all_id_sets[0]
        for id_set in all_id_sets[1:]:
            common_ids = common_ids & id_set

    primary_index = min(primary_index, len(all_record_sets) - 1)
    primary_records = all_record_sets[primary_index]
    total_before = len(primary_records)

    filtered = []
    for record in primary_records:
        record_id = None
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

    return {
        "queryType": "cross_filter",
        "matchCount": len(filtered),
        "totalBeforeFilter": total_before,
        "records": filtered,
    }


def merge_results(labeled_results: List[Tuple[str, Any]]) -> Dict[str, Any]:
    sections = []
    for label, data in labeled_results:
        records = _extract_records(data)
        sections.append({
            "label": label,
            "data": data,
            "recordCount": len(records),
        })

    return {
        "queryType": "multi_independent",
        "sectionCount": len(sections),
        "sections": sections,
    }


def _extract_summary_metrics(data: Any) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
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

    records = _extract_records(data)
    if records:
        metrics["recordCount"] = len(records)
    return metrics


def compare_results(labeled_results: List[Tuple[str, Any]]) -> Dict[str, Any]:
    sides = []
    all_metric_keys: Set[str] = set()

    for label, data in labeled_results:
        metrics = _extract_summary_metrics(data)
        all_metric_keys.update(metrics.keys())
        sides.append({
            "label": label,
            "data": data,
            "metrics": metrics,
        })

    return {
        "queryType": "comparison",
        "sides": sides,
        "comparedMetrics": sorted(all_metric_keys),
    }