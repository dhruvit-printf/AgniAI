"""
cross_filter_engine.py
======================
Cross-filter engine for performing N-way intersections of datasets.

Finds Agniveers that appear across ALL provided result sets and merges
their fields from each dataset into a single unified record.
"""

import logging
from typing import Any, Dict, List, Optional, Set

from utils import extract_records as _normalize_records

logger = logging.getLogger(__name__)

# ID field resolution order — update here only if new fields are added
_ID_FIELD_PRIORITY = (
    "agniveerNo",
    "agniveerId",
    "AgniveerId",
    "AgniVeerId",
    "id",
    "Id",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_records(data: Any) -> List[Dict]:
    """Pull the list of records out of any .NET wrapper shape."""
    return _normalize_records(data)


def _get_record_id(record: Dict) -> Optional[str]:
    """
    Resolve the canonical Agniveer ID from a record using priority field order.
    Returns None if no ID field is found.
    """
    for key in _ID_FIELD_PRIORITY:
        val = record.get(key)
        if val is not None:
            return str(val).strip()
    return None


def _extract_agniveer_ids(records: List[Dict]) -> Set[str]:
    """Return the set of all resolvable Agniveer IDs from a list of records."""
    ids: Set[str] = set()
    for record in records:
        record_id = _get_record_id(record)
        if record_id is not None:
            ids.add(record_id)
    return ids


def _build_lookup(records: List[Dict]) -> Dict[str, Dict]:
    """Build an ID → record lookup dict from a list of records."""
    lookup: Dict[str, Dict] = {}
    for record in records:
        record_id = _get_record_id(record)
        if record_id is not None:
            lookup[record_id] = record
    return lookup


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def cross_filter_datasets(
    result_sets: List[Any],
    primary_index: int = 0,
) -> Dict[str, Any]:
    """
    Find Agniveers common across N datasets (set intersection) and return
    merged records.

    Args:
        result_sets:   List of raw .NET responses (any wrapper shape).
        primary_index: Index of the dataset whose records are used as the
                       base for iteration and field ordering. Defaults to 0
                       (first dataset). Use the dataset with the most
                       comprehensive base fields as primary.

    Returns:
        {
            "status":            bool,
            "records":           List[Dict],   # merged, intersection records
            "matchCount":        int,
            "totalBeforeFilter": int,          # size of the primary dataset
            "filterDepth":       int,          # number of datasets intersected
            "message":           str           # only present when status=False
        }
    """
    if not result_sets:
        return _empty_result(filter_depth=0, total_before=0)

    # Extract records from every result set
    all_record_sets: List[List[Dict]] = [_extract_records(rs) for rs in result_sets]
    filter_depth = len(all_record_sets)

    # Clamp primary_index to valid range
    primary_index = max(0, min(primary_index, filter_depth - 1))
    primary_records = all_record_sets[primary_index]
    total_before = len(primary_records)

    # Compute intersection of ID sets across all datasets
    # NOTE: We do NOT short-circuit on empty ID sets here — an empty set will
    # simply produce an empty intersection, which is the correct behaviour.
    all_id_sets = [_extract_agniveer_ids(recs) for recs in all_record_sets]

    if not all_id_sets:
        return _empty_result(filter_depth=filter_depth, total_before=total_before)

    common_ids: Set[str] = all_id_sets[0]
    for id_set in all_id_sets[1:]:
        common_ids = common_ids & id_set

    if not common_ids:
        return _no_match_result(filter_depth=filter_depth, total_before=total_before)

    # Build lookup dicts once — O(N * M) total, not O(N * M * K)
    all_lookups: List[Dict[str, Dict]] = [_build_lookup(recs) for recs in all_record_sets]

    # Iterate primary records, keep only those in the intersection,
    # and merge in fields from every other dataset
    filtered: List[Dict] = []
    for record in primary_records:
        record_id = _get_record_id(record)
        if record_id is None or record_id not in common_ids:
            continue

        merged = dict(record)
        for lookup in all_lookups:
            match = lookup.get(record_id)
            if not match:
                continue
            # Only backfill — primary record fields take precedence
            for key, value in match.items():
                if key not in merged or merged[key] in (None, ""):
                    merged[key] = value

        filtered.append(merged)

    if not filtered:
        return _no_match_result(filter_depth=filter_depth, total_before=total_before)

    logger.debug(
        "cross_filter: %d common IDs across %d datasets, %d records returned",
        len(common_ids),
        filter_depth,
        len(filtered),
    )

    return {
        "status": True,
        "records": filtered,
        "matchCount": len(filtered),
        "totalBeforeFilter": total_before,
        "filterDepth": filter_depth,
    }


# ---------------------------------------------------------------------------
# Private result constructors
# ---------------------------------------------------------------------------

def _empty_result(*, filter_depth: int, total_before: int) -> Dict[str, Any]:
    return {
        "status": False,
        "message": "No datasets provided for cross-filter.",
        "records": [],
        "matchCount": 0,
        "totalBeforeFilter": total_before,
        "filterDepth": filter_depth,
    }


def _no_match_result(*, filter_depth: int, total_before: int) -> Dict[str, Any]:
    return {
        "status": False,
        "message": "No matching records found after cross-filter intersection.",
        "records": [],
        "matchCount": 0,
        "totalBeforeFilter": total_before,
        "filterDepth": filter_depth,
    }