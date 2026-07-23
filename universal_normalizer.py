"""
universal_normalizer.py
=======================
A universal recursive JSON walker that flattens any arbitrary .NET response
structure into a single list of canonical Agniveer records.

It uses structural heuristics to detect Agniveer objects and inherits scalar
metadata from parent objects (ignoring aggregates) to ensure no context is lost
no matter how deeply nested the Agniveer is.
"""

from typing import Any, Dict, List, Optional, Set

# Priority for determining if a node is an Agniveer record.
# "recordId" comes first: it's a per-row DB primary key stamped onto rows
# that are genuinely one-row-per-database-record (e.g. each MedicalRecordMaster
# visit) rather than fragments of a single Agniveer profile. Without it,
# multiple real rows sharing the same agniveerNo (a person with several
# medical visits) would resolve to the same canonical ID and get folded
# together by _resolve_duplicates below, silently discarding every visit
# but the one whose fields happened to be picked first.
_ID_FIELD_PRIORITY = (
    "recordId",
    "agniveerNo",
    "AgniveerNo",
    "agniveerId",
    "AgniveerId",
    "AgniVeerId",
    "AgniVeerNo",
    "agniveerNumber",
)

# Fields that should never be inherited from a parent to a child
_AGGREGATE_KEYWORDS = frozenset(
    {
        "count",
        "total",
        "average",
        "summary",
        "statistics",
        "issuedcount",
        "grandtotal",
        "minimum",
        "maximum",
        "percentage",
    }
)


def _get_canonical_id(record: Dict[str, Any]) -> Optional[str]:
    """Return the primary ID string for the record based on priority."""
    for key in _ID_FIELD_PRIORITY:
        if key in record and record[key] is not None:
            return str(record[key]).strip()

    # Fallback: id + fullName (as per Rule 4)
    if "id" in record and "fullName" in record:
        if record["id"] is not None and record["fullName"] is not None:
            return str(record["id"]).strip()

    return None


def _is_agniveer_record(node: Any) -> bool:
    """Detect if a dictionary structurally represents an Agniveer record."""
    if not isinstance(node, dict):
        return False
    return _get_canonical_id(node) is not None


def _is_aggregate_field(field_name: str) -> bool:
    """Check if a field name represents a summary/aggregate value."""
    lowered = field_name.lower()
    return any(agg in lowered for agg in _AGGREGATE_KEYWORDS)


def _walk(
    node: Any, parent_context: Dict[str, Any], records: List[Dict[str, Any]]
) -> None:
    """Recursively walk the JSON tree."""
    if isinstance(node, dict):
        if _is_agniveer_record(node):
            # Merge context. Child wins over parent (Rule 6).
            merged = {**parent_context, **node}
            records.append(merged)
        else:
            # Build current context by inheriting non-aggregate scalars from this structural node
            current_context = {**parent_context}
            for k, v in node.items():
                if k in ("sql", "success"):
                    # "sql" is the internal envelope's raw query string and
                    # "success" is its execution-status flag (from
                    # sql_executor._to_section) — both are bookkeeping on the
                    # wrapper, not Agniveer data, and must never be inherited
                    # into every row record via context propagation.
                    continue
                if _is_aggregate_field(k):
                    continue
                # Only inherit scalars (string, int, float, bool)
                if isinstance(v, (str, int, float, bool)) or v is None:
                    current_context[k] = v

            # Recurse into children
            for k, v in node.items():
                # Avoid unnecessary copies or recursion on scalars we already grabbed
                if not isinstance(v, (str, int, float, bool)) and v is not None:
                    _walk(v, current_context, records)

    elif isinstance(node, list):
        for item in node:
            _walk(item, parent_context, records)


def _records_conflict(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """True if `a` and `b` disagree on any field both have populated.

    Two fragments of the SAME underlying entity (e.g. one leg returning
    fullName/status, another returning bmi for the same AgniveerNo) never
    conflict — every shared field either agrees or one side just hasn't
    populated it yet. Distinct rows that merely share an AgniveerNo (e.g.
    AttemptWise's per-section-per-attempt rows, or Grading's per-section
    rows for one Agniveer) DO conflict — same AgniveerNo, different
    sectionName/attemptNo/attemptTotal — and must not be collapsed into one.
    """
    for k in a.keys() & b.keys():
        va, vb = a[k], b[k]
        if va is None or va == "" or vb is None or vb == "":
            continue
        if va != vb:
            return True
    return False


def _resolve_duplicates(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge duplicate Agniveer records intelligently, filling missing fields
    (Rule 12) — but only when they're actually fragments of the same record.
    Records sharing an AgniveerNo that disagree on some other populated
    field are distinct rows (e.g. one per section/attempt) and are kept
    separate rather than having all but the first silently discarded."""
    resolved: Dict[str, List[Dict[str, Any]]] = {}

    for record in records:
        canonical_id = _get_canonical_id(record)
        if not canonical_id:
            continue

        bucket = resolved.setdefault(canonical_id, [])
        merge_target = next(
            (
                existing
                for existing in bucket
                if not _records_conflict(existing, record)
            ),
            None,
        )
        if merge_target is None:
            bucket.append(dict(record))
        else:
            # Intelligent merge: don't overwrite populated fields, just fill missing
            for k, v in record.items():
                if (
                    k not in merge_target
                    or merge_target[k] is None
                    or merge_target[k] == ""
                ):
                    merge_target[k] = v

    out: List[Dict[str, Any]] = []
    for bucket in resolved.values():
        out.extend(bucket)
    return out


def _find_first_list_of_dicts(node: Any) -> Optional[List[Dict[str, Any]]]:
    """Recursively search a JSON tree for the first list containing dict rows."""
    if isinstance(node, list):
        rows = [item for item in node if isinstance(item, dict)]
        if rows:
            return rows
        return None
    if isinstance(node, dict):
        for value in node.values():
            found = _find_first_list_of_dicts(value)
            if found is not None:
                return found
    return None


def _fallback_raw_rows(response: Any) -> List[Dict[str, Any]]:
    """Used only when no agniveer-shaped record was found anywhere in the tree.

    Some responses are legitimately row-per-something-other-than-agniveer
    (per-date attendance, per-section score averages, per-blood-group
    counts, ...) — every row lacks an agniveerNo, so `_walk` correctly finds
    zero Agniveer records, but the rows themselves are still the real answer
    and must not be silently discarded. Fall back to the raw list of dict
    rows (unwrapping one "data" envelope if present) instead of returning [].

    The rows aren't always directly at `response["data"]` — aggregate shapes
    like `data: {"attemptSummary": [...]}` nest the real list one or more
    levels deeper, so search the whole subtree for the first list of dicts.
    """
    candidate = response
    if isinstance(response, dict) and "data" in response:
        candidate = response["data"]
    if isinstance(candidate, list):
        return [dict(item) for item in candidate if isinstance(item, dict)]
    found = _find_first_list_of_dicts(candidate)
    if found is not None:
        return [dict(item) for item in found]
    return []


def _merge_single_entity_response(response: Any) -> Optional[List[Dict[str, Any]]]:
    """Handle "one specific agniveer, several named sections" shapes — e.g.
    Equipment/AgniveerWise ({profile, summary, currentlyHolding, returnedHistory})
    or Medical/Individual ({profile, bmi, stats, medicalHistory}).

    `_walk` already finds the one agniveer record (from the "profile" section)
    but, by design, skips every sibling section/list because none of their
    rows carry their own agniveerNo — so the summary stats and the actual
    equipment/medical-history rows never make it into the result. Since there
    is exactly one agniveer in play here, merge every sibling section/list
    into that single record instead of losing them.
    """
    candidate = response.get("data") if isinstance(response, dict) else response
    if not isinstance(candidate, dict):
        return None

    primary_key = next(
        (
            key
            for key, value in candidate.items()
            if isinstance(value, dict) and _is_agniveer_record(value)
        ),
        None,
    )
    if primary_key is None:
        return None

    merged: Dict[str, Any] = dict(candidate[primary_key])
    for key, value in candidate.items():
        if key == primary_key:
            continue
        if isinstance(value, list):
            rows = [item for item in value if isinstance(item, dict)]
            if rows:
                merged[key] = rows
        elif value is not None or key not in merged:
            merged[key] = value

    return [merged]


def normalize_response(
    response: Any, base_metadata: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Universally normalize any .NET API response into a flat list of canonical Agniveer records.

    Args:
        response: The arbitrary JSON structure from the API.
        base_metadata: Optional dictionary of metadata (like category, operation, index)
                       to stamp onto every extracted record.
    """
    raw_records: List[Dict[str, Any]] = []
    _walk(response, parent_context={}, records=raw_records)

    resolved = _resolve_duplicates(raw_records)

    if not resolved:
        resolved = _fallback_raw_rows(response)
    elif len(resolved) == 1:
        enriched = _merge_single_entity_response(response)
        if enriched is not None:
            resolved = enriched

    if base_metadata:
        for record in resolved:
            for k, v in base_metadata.items():
                # Only add metadata if it doesn't collide, or just stamp it
                record[f"__{k}"] = v

    return resolved
