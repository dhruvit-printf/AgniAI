"""
result_combiner.py
==================
Combines results from multiple .NET API calls for the AgniAI Admin Chatbot
Query Planning Layer.

Supports three combination strategies:

  intersect_results()  — CROSS_FILTER: records whose Agniveer ID appears in ALL result sets
  merge_results()      — MULTI_INDEPENDENT: concatenate results, tagged by category
  compare_results()    — COMPARISON: side-by-side metric comparison

CRITICAL RULE:
  Intersection is ALWAYS by agniveerId (integer).
  NEVER match by name — names may be duplicated or formatted differently.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# RECORD EXTRACTION
# =============================================================================

def _extract_records(data: Any) -> List[Dict]:
    """
    Pull the list of individual records from whatever shape .NET returned.

    Handles:
      - Bare list: [{ agniveerId: 1, ... }, ...]
      - Wrapper dict: { success: true, data: [...] }
      - Wrapper dict: { data: { ... nested ... } }
      - Nested teams: { data: { teams: [{ members: [...] }] } }
    """
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        # Try standard wrapper keys
        for key in (
            "data", "Data", "result", "Result",
            "records", "Records", "persons", "personnel",
        ):
            val = data.get(key)
            if isinstance(val, list):
                return val
            if isinstance(val, dict):
                # Recurse one more level
                return _extract_records(val)

        # Try teams/members nesting (Distribution Latest)
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
    """
    Extract the set of unique Agniveer IDs from a list of records.

    Tries multiple field name casings for robustness.
    """
    ids: Set[int] = set()
    for record in records:
        for key in ("agniveerId", "AgniveerId", "AgniVeerId", "id", "Id"):
            val = record.get(key)
            if val is not None:
                try:
                    ids.add(int(val))
                except (ValueError, TypeError):
                    pass
                break  # Use the first key that exists
    return ids


# =============================================================================
# INTERSECT (CROSS_FILTER)
# =============================================================================

def intersect_results(
    result_sets: List[Any],
    primary_index: int = 0,
) -> Dict[str, Any]:
    """
    CROSS_FILTER intersection: return only records whose Agniveer ID
    appears in ALL result sets.

    Parameters
    ----------
    result_sets : list
        Raw .NET responses (one per sub-operation).
    primary_index : int
        Which result set to use as the base for returned records.
        The full record data comes from this set; other sets are used
        only for ID filtering.

    Returns
    -------
    dict
        {
            "queryType": "cross_filter",
            "matchCount": int,
            "totalBeforeFilter": int,
            "records": list,         # Filtered records from primary set
        }
    """
    if not result_sets:
        return {
            "queryType": "cross_filter",
            "matchCount": 0,
            "totalBeforeFilter": 0,
            "records": [],
        }

    # Extract records and IDs from each result set
    all_record_sets = [_extract_records(rs) for rs in result_sets]
    all_id_sets = [_extract_agniveer_ids(recs) for recs in all_record_sets]

    logger.debug(
        "Cross-filter: %d result sets, ID counts: %s",
        len(all_id_sets),
        [len(ids) for ids in all_id_sets],
    )

    # Find IDs present in ALL sets
    if not all_id_sets or any(len(ids) == 0 for ids in all_id_sets):
        # If any set has no IDs, intersection is empty
        common_ids: Set[int] = set()
    else:
        common_ids = all_id_sets[0]
        for id_set in all_id_sets[1:]:
            common_ids = common_ids & id_set

    logger.info(
        "Cross-filter intersection: %d common IDs from %d sets",
        len(common_ids), len(all_id_sets),
    )

    # Filter primary set to only matching IDs
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


# =============================================================================
# MERGE (MULTI_INDEPENDENT)
# =============================================================================

def merge_results(
    labeled_results: List[Tuple[str, Any]],
) -> Dict[str, Any]:
    """
    MULTI_INDEPENDENT merge: concatenate results from independent queries.
    Each result is tagged with its category label.

    Parameters
    ----------
    labeled_results : list of (label, data)
        Each tuple is (category_label: str, raw_dotnet_response: Any).

    Returns
    -------
    dict
        {
            "queryType": "multi_independent",
            "sectionCount": int,
            "sections": [
                {
                    "label": "Attendance",
                    "data": <raw .NET response>,
                    "recordCount": int,
                },
                ...
            ]
        }
    """
    sections = []
    for label, data in labeled_results:
        records = _extract_records(data)
        sections.append({
            "label": label,
            "data": data,
            "recordCount": len(records),
        })

    logger.info(
        "Multi-independent merge: %d sections, record counts: %s",
        len(sections),
        [s["recordCount"] for s in sections],
    )

    return {
        "queryType": "multi_independent",
        "sectionCount": len(sections),
        "sections": sections,
    }


# =============================================================================
# COMPARE (COMPARISON)
# =============================================================================

def _extract_summary_metrics(data: Any) -> Dict[str, Any]:
    """
    Extract numeric metrics from a .NET response for comparison.
    Handles both flat dicts and nested data wrappers.
    """
    metrics: Dict[str, Any] = {}

    if isinstance(data, dict):
        inner = data
        # Unwrap standard wrappers
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


def compare_results(
    labeled_results: List[Tuple[str, Any]],
) -> Dict[str, Any]:
    """
    COMPARISON: build a side-by-side comparison structure.

    Parameters
    ----------
    labeled_results : list of (label, data)
        Each tuple is (category_label: str, raw_dotnet_response: Any).
        Typically exactly 2 entries.

    Returns
    -------
    dict
        {
            "queryType": "comparison",
            "sides": [
                {
                    "label": "PPT Performance",
                    "data": <raw .NET response>,
                    "metrics": { "recordCount": 10, ... },
                },
                ...
            ],
            "comparedMetrics": ["recordCount", "average", ...],
        }
    """
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

    # Sort metric keys for consistent output
    compared_metrics = sorted(all_metric_keys)

    logger.info(
        "Comparison: %d sides, metrics compared: %s",
        len(sides), compared_metrics,
    )

    return {
        "queryType": "comparison",
        "sides": sides,
        "comparedMetrics": compared_metrics,
    }
