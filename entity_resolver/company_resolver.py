"""Company entity resolution."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .entity_cache import fetch_companies
from .entity_matcher import match_entity


def _get(record: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = record.get(key)
        if value not in (None, "", [], {}):
            return value
    return None


def resolve_company(
    query: str,
    *,
    trace_id: Optional[str] = None,
    session_id: Optional[str] = None,
    threshold: float = 0.85,
) -> Dict[str, Any]:
    records = fetch_companies(trace_id=trace_id)
    match = match_entity(
        query,
        records,
        value_fields=("companyId", "CompanyId", "id", "Id"),
        label_fields=("companyName", "CompanyName", "name", "Name"),
        alias_fields=("companyName", "CompanyName", "name", "Name"),
        threshold=threshold * 100.0,
    )
    record = match.get("record") or {}
    return {
        "CompanyId": match.get("value"),
        "name": _get(record, "companyName", "CompanyName", "name", "Name"),
        "confidence": match.get("confidence", 0.0),
        "match_type": match.get("match_type", ""),
        "matched_text": match.get("matched_text", ""),
        "needs_clarification": bool(match.get("needs_clarification")),
        "candidates": match.get("candidates", []),
    }
