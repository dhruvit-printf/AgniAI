"""Platoon entity resolution."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .entity_cache import fetch_platoons
from .entity_matcher import match_entity, normalize_text


def _get(record: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = record.get(key)
        if value not in (None, "", [], {}):
            return value
    return None


def _alias_builder(record: Dict[str, Any]) -> List[str]:
    aliases: List[str] = []
    name = str(_get(record, "platoonName", "PlatoonName", "name", "Name") or "")
    platoon_no = _get(record, "platoonNo", "PlatoonNo", "no", "No")
    if name:
        aliases.extend([name])
        compact = normalize_text(name)
        if compact:
            aliases.append(compact)
        digits = re.sub(r"\D", "", name)
        if digits:
            aliases.extend(
                [
                    f"platoon {digits}",
                    f"pl {digits}",
                    f"pl-{digits}",
                    f"pl{digits}",
                    f"p{digits}",
                    f"{digits}th platoon",
                ]
            )
    if platoon_no not in (None, ""):
        digits = re.sub(r"\D", "", str(platoon_no))
        if digits:
            aliases.extend(
                [
                    f"platoon {digits}",
                    f"pl {digits}",
                    f"pl-{digits}",
                    f"pl{digits}",
                    f"p{digits}",
                    f"{digits}th platoon",
                ]
            )
    return list(dict.fromkeys([alias for alias in aliases if alias]))


def _extract_platoon_token(query: str) -> Optional[str]:
    patterns = [
        r"\bplatoon\s+no\.?\s*(\d+)\b",
        r"\bplatoon\s+(\d+)(?:st|nd|rd|th)?\b",
        r"\bpl[-\s]?(\d+)\b",
        r"\bp(\d+)\b",
        r"\b(\d+)(?:st|nd|rd|th)?\s+platoon\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def resolve_platoon(
    query: str,
    *,
    company_id: Optional[int] = None,
    trace_id: Optional[str] = None,
    session_id: Optional[str] = None,
    threshold: float = 0.85,
) -> Dict[str, Any]:
    records = fetch_platoons(trace_id=trace_id)
    token = _extract_platoon_token(query)
    if token:
        for record in records:
            platoon_no = _get(record, "platoonNo", "PlatoonNo", "no", "No")
            if platoon_no is not None and str(platoon_no) == token:
                record_company_id = _get(record, "companyId", "CompanyId")
                if company_id is None or record_company_id in (None, company_id, str(company_id)):
                    return {
                        "PlatoonId": _get(record, "platoonId", "PlatoonId", "id", "Id"),
                        "PlatoonNo": platoon_no,
                        "name": _get(record, "platoonName", "PlatoonName", "name", "Name"),
                        "CompanyId": record_company_id,
                        "companyName": _get(record, "companyName", "CompanyName"),
                        "confidence": 0.99,
                        "match_type": "exact",
                        "matched_text": token,
                        "needs_clarification": False,
                        "candidates": [],
                    }

    match = match_entity(
        query,
        records,
        value_fields=("platoonId", "PlatoonId", "id", "Id"),
        label_fields=("platoonName", "PlatoonName", "name", "Name"),
        alias_fields=("platoonName", "PlatoonName", "name", "Name", "platoonNo", "PlatoonNo"),
        alias_builder=_alias_builder,
        threshold=threshold * 100.0,
    )

    record = match.get("record") or {}
    if company_id is not None:
        record_company_id = _get(record, "companyId", "CompanyId")
        if record_company_id not in (None, company_id, str(company_id)) and not match.get("needs_clarification"):
            match = {
                "value": None,
                "confidence": match.get("confidence", 0.0),
                "matched_text": match.get("matched_text", ""),
                "source": match.get("source", "cache"),
                "match_type": match.get("match_type", ""),
                "needs_clarification": False,
                "candidates": match.get("candidates", []),
            }

    record = match.get("record") or {}
    return {
        "PlatoonId": match.get("value"),
        "PlatoonNo": _get(record, "platoonNo", "PlatoonNo", "no", "No"),
        "name": _get(record, "platoonName", "PlatoonName", "name", "Name"),
        "CompanyId": _get(record, "companyId", "CompanyId"),
        "companyName": _get(record, "companyName", "CompanyName"),
        "confidence": match.get("confidence", 0.0),
        "match_type": match.get("match_type", ""),
        "matched_text": match.get("matched_text", ""),
        "needs_clarification": bool(match.get("needs_clarification")),
        "candidates": match.get("candidates", []),
    }
