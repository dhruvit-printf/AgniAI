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

# Priority for determining if a node is an Agniveer record
_ID_FIELD_PRIORITY = (
    "agniveerNo",
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


def _walk(node: Any, parent_context: Dict[str, Any], records: List[Dict[str, Any]]) -> None:
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


def _resolve_duplicates(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge duplicate Agniveer records intelligently, filling missing fields (Rule 12)."""
    resolved: Dict[str, Dict[str, Any]] = {}
    
    for record in records:
        canonical_id = _get_canonical_id(record)
        if not canonical_id:
            continue
            
        if canonical_id not in resolved:
            resolved[canonical_id] = dict(record)
        else:
            # Intelligent merge: don't overwrite populated fields, just fill missing
            existing = resolved[canonical_id]
            for k, v in record.items():
                if k not in existing or existing[k] is None or existing[k] == "":
                    existing[k] = v
                    
    return list(resolved.values())


def normalize_response(response: Any, base_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
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
    
    if base_metadata:
        for record in resolved:
            for k, v in base_metadata.items():
                # Only add metadata if it doesn't collide, or just stamp it
                record[f"__{k}"] = v
                
    return resolved
