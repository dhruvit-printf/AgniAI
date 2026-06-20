"""
cross_filter_engine.py
======================
Cross-filter engine for performing N-way intersections of datasets.
"""

import logging
from typing import Any, Dict, List, Set, Optional

from utils import extract_records as _normalize_records

logger = logging.getLogger(__name__)

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
        record_id = record.get("agniveerNo")
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
