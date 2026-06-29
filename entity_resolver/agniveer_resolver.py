"""Agniveer entity resolution."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .entity_cache import fetch_agniveers
from .entity_matcher import match_entity, normalize_text


def _get(record: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = record.get(key)
        if value not in (None, "", [], {}):
            return value
    return None


def _canonical_no(text: str) -> Optional[str]:
    raw = text or ""
    explicit = re.search(r"\bag[-\s]?(\d{3,10})\b", raw, re.IGNORECASE)
    if explicit:
        return f"AG{explicit.group(1)}"
    alnum = re.search(r"\b([A-Za-z]\d{5,8}[A-Za-z]?)\b", raw)
    if alnum:
        return alnum.group(1).upper()
    return None


def resolve_agniveer(
    query: str,
    *,
    trace_id: Optional[str] = None,
    session_id: Optional[str] = None,
    threshold: float = 0.85,
) -> Dict[str, Any]:
    records = fetch_agniveers(trace_id=trace_id)
    explicit_no = _canonical_no(query)

    if explicit_no:
        for record in records:
            agniveer_no = str(
                _get(record, "agniveerNo", "AgniveerNo", "agniveer_no", "AgniVeerNo")
                or ""
            ).upper()
            if agniveer_no == explicit_no:
                return {
                    "AgniveerNo": agniveer_no,
                    "AgniveerId": _get(record, "id", "Id", "agniveerId", "AgniveerId"),
                    "fullName": _get(record, "fullName", "FullName", "name", "Name"),
                    "confidence": 0.99,
                    "match_type": "exact",
                    "matched_text": explicit_no,
                    "needs_clarification": False,
                    "candidates": [],
                }

    match = match_entity(
        query,
        records,
        value_fields=(
            "agniveerNo",
            "AgniveerNo",
            "id",
            "Id",
            "agniveerId",
            "AgniveerId",
        ),
        label_fields=("fullName", "FullName", "name", "Name"),
        alias_fields=("agniveerNo", "AgniveerNo", "fullName", "FullName"),
        threshold=threshold * 100.0,
    )

    if match.get("needs_clarification"):
        return {
            "AgniveerNo": None,
            "AgniveerId": None,
            "fullName": None,
            "confidence": match.get("confidence", 0.0),
            "match_type": "ambiguous",
            "matched_text": match.get("matched_text", ""),
            "needs_clarification": True,
            "candidates": match.get("candidates", []),
        }

    record = match.get("record") or {}
    value = match.get("value")
    agniveer_no = (
        value
        if isinstance(value, str) and value
        else _get(record, "agniveerNo", "AgniveerNo", "agniveer_no", "AgniVeerNo")
    )
    if isinstance(agniveer_no, str):
        agniveer_no = agniveer_no.upper()
    return {
        "AgniveerNo": agniveer_no,
        "AgniveerId": _get(record, "id", "Id", "agniveerId", "AgniveerId"),
        "fullName": _get(record, "fullName", "FullName", "name", "Name"),
        "confidence": match.get("confidence", 0.0),
        "match_type": match.get("match_type", ""),
        "matched_text": match.get("matched_text", ""),
        "needs_clarification": False,
        "candidates": match.get("candidates", []),
    }
