"""
cross_filter_engine.py
======================
Cross-filter engine for performing N-way intersections of datasets.

Finds Agniveers that appear across ALL provided result sets and merges
their fields from each dataset into a single unified record.
"""

import logging
from typing import Any, Dict, List, Optional, Set

from universal_normalizer import _ID_FIELD_PRIORITY, normalize_response

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_records(data: Any) -> List[Dict]:
    """Pull the list of records out of any .NET wrapper shape."""
    return normalize_response(data)


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
    labels: Optional[List[str]] = None,
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
        labels:        Optional per-dataset labels (e.g. "disqualified",
                       "Performance") used only to phrase a specific
                       no-overlap message — does not affect matching.

    Returns:
        {
            "status":            bool,
            "records":           List[Dict],   # merged, intersection records
            "matchCount":        int,
            "totalBeforeFilter": int,          # size of the primary dataset
            "filterDepth":       int,          # number of datasets intersected
            "perDatasetCounts":  List[int],    # record count of each input dataset
            "datasetLabels":     List[str],    # labels for perDatasetCounts, if given
            "message":           str           # only present when status=False
        }
    """
    if not result_sets:
        return _empty_result(filter_depth=0, total_before=0)

    # Extract records from every result set
    all_record_sets: List[List[Dict]] = [_extract_records(rs) for rs in result_sets]
    filter_depth = len(all_record_sets)
    per_dataset_counts = [len(recs) for recs in all_record_sets]

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
        return _no_match_result(
            filter_depth=filter_depth,
            total_before=total_before,
            per_dataset_counts=per_dataset_counts,
            labels=labels,
        )

    # Build lookup dicts once — O(N * M) total, not O(N * M * K)
    all_lookups: List[Dict[str, Dict]] = [
        _build_lookup(recs) for recs in all_record_sets
    ]

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
        return _no_match_result(
            filter_depth=filter_depth,
            total_before=total_before,
            per_dataset_counts=per_dataset_counts,
            labels=labels,
        )

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
        "perDatasetCounts": per_dataset_counts,
        "datasetLabels": labels or [],
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
        "perDatasetCounts": [],
        "datasetLabels": [],
    }


def _no_match_result(
    *,
    filter_depth: int,
    total_before: int,
    per_dataset_counts: Optional[List[int]] = None,
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    counts = per_dataset_counts or []
    message = _build_no_overlap_message(counts, labels)
    return {
        "status": False,
        "message": message,
        "records": [],
        "matchCount": 0,
        "totalBeforeFilter": total_before,
        "filterDepth": filter_depth,
        "perDatasetCounts": counts,
        "datasetLabels": labels or [],
    }


def _build_no_overlap_message(
    counts: List[int], labels: Optional[List[str]]
) -> str:
    """
    Distinguish "a condition genuinely returned nothing" from
    "every condition matched records on its own, they just don't overlap".
    """
    if not counts:
        return "I couldn't find any records matching that request."

    if any(c == 0 for c in counts):
        if labels and len(labels) == len(counts):
            empty_labels = [label for label, count in zip(labels, counts) if count == 0]
            parts = " and ".join(empty_labels)
            return (
                f"I couldn't find any records for '{parts}', so there's nothing "
                f"to combine with the rest of your request. You might try "
                f"adjusting that part of the search."
            )
        return (
            "One or more of the conditions in your request didn't match any "
            "records, so there's nothing to combine. You might try adjusting "
            "the filters and searching again."
        )

    if labels and len(labels) == len(counts):
        parts = ", ".join(
            f"{count} {label}" for label, count in zip(labels, counts)
        )
        return (
            f"I found {parts} individually, but none of them meet all of "
            f"those conditions at the same time. You might try relaxing one "
            f"of the filters to see more results."
        )

    return (
        "Each condition matched some records on its own, but none satisfy "
        "every condition together. You might try relaxing one of the "
        "filters to see more results."
    )
