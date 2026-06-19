"""
cross_filter_engine.py
======================
Cross-filter engine for performing N-way intersections of datasets.
"""

import logging
from typing import Any, Dict, List, Set, Optional

logger = logging.getLogger(__name__)

def _extract_records(data: Any) -> List[Dict]:
    """Pull the list of records out of any .NET wrapper shape."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "Data", "result", "Result", "records", "Records", "persons", "personnel"):
            val = data.get(key)
            if isinstance(val, list):
                return val
            if isinstance(val, dict):
                return _extract_records(val)
        # Distribution "teams" -> flatten members
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

def cross_filter_datasets(result_sets: List[Any], primary_index: int = 0) -> Dict[str, Any]:
    """
    Find common records across N datasets (Intersection).
    """
    if not result_sets:
        return {
            "status": False,
            "message": "No matching records found"
        }

    all_record_sets = [_extract_records(rs) for rs in result_sets]
    all_id_sets = [_extract_agniveer_ids(recs) for recs in all_record_sets]

    if not all_id_sets or any(len(ids) == 0 for ids in all_id_sets):
        return {
            "status": False,
            "message": "No matching records found"
        }

    # Intersect all ID sets
    common_ids = all_id_sets[0]
    for id_set in all_id_sets[1:]:
        common_ids = common_ids & id_set

    if not common_ids:
        return {
            "status": False,
            "message": "No matching records found"
        }

    primary_index = min(primary_index, len(all_record_sets) - 1)
    primary_records = all_record_sets[primary_index]

    filtered: List[Dict] = []
    for record in primary_records:
        record_id = record.get("agniveerNo")
        if record_id is None:
            for key in ("agniveerId", "AgniveerId", "AgniVeerId", "id", "Id"):
                val = record.get(key)
                if val is not None:
                    record_id = str(val).strip()
                    break
        if record_id is not None and str(record_id).strip() in common_ids:
            filtered.append(record)

    if not filtered:
        return {
            "status": False,
            "message": "No matching records found"
        }

    return {
        "status": True,
        "records": filtered,
        "matchCount": len(filtered),
        "totalBeforeFilter": len(primary_records),
        "filterDepth": len(result_sets)
    }
