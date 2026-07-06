"""
schedule_enrichment.py
======================
A "Schedule" leg in a multi-operation query (e.g. "top performers in BPET
and their today's schedule") has no company/platoon of its own to call the
.NET Schedule API with — the company only becomes known once the *other*
leg's agniveer records (each carrying a platoonName) are on hand.

This module resolves that: given the peer leg's fetched records, it derives
every distinct platoon named in them, resolves each to its company, fetches
that company's schedule, and merges the results into one response shaped
like the .NET Schedule/Company response (grouped `byCompany`), since the
schedule itself is company-wide with no per-agniveer breakdown.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from dotnet_executor import _call_dotnet
from entity_resolver import resolve_platoon
from normalized_models import extract_records

logger = logging.getLogger(__name__)


def _extract_platoon_names(records: List[Dict[str, Any]]) -> List[str]:
    names: List[str] = []
    seen = set()
    for record in records:
        if not isinstance(record, dict):
            continue
        name = (
            record.get("platoonName")
            or record.get("PlatoonName")
            or record.get("platoon")
        )
        if not name:
            continue
        cleaned = str(name).strip()
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            names.append(cleaned)
    return names


def _resolve_company_ids(
    platoon_names: List[str],
    *,
    trace_id: Optional[str],
    session_id: Optional[str],
) -> List[Any]:
    company_ids: List[Any] = []
    seen = set()
    for name in platoon_names:
        resolved = resolve_platoon(name, trace_id=trace_id, session_id=session_id)
        company_id = resolved.get("CompanyId")
        if company_id is not None and company_id not in seen:
            seen.add(company_id)
            company_ids.append(company_id)
        elif company_id is None:
            logger.info(
                "schedule_enrichment: could not resolve company for platoon %r",
                name,
            )
    return company_ids


def enrich_schedule_by_company(
    schedule_payload: Dict[str, Any],
    peer_data: Any,
    *,
    trace_id: Optional[str] = None,
    session_id: Optional[str] = None,
    query_type: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Resolve companies from a peer leg's records and fetch each one's
    schedule, merging them into a single Schedule/Company-shaped response.

    Returns None (leaving the schedule leg's own, unscoped fetch untouched)
    when no platoon/company could be resolved from the peer data — e.g. the
    peer leg failed, returned no records, or its records don't carry a
    platoon at all.
    """
    peer_records = extract_records(peer_data)
    platoon_names = _extract_platoon_names(peer_records)
    if not platoon_names:
        return None

    company_ids = _resolve_company_ids(
        platoon_names, trace_id=trace_id, session_id=session_id
    )
    if not company_ids:
        return None

    merged_by_company: List[Dict[str, Any]] = []
    merged_date = None
    merged_from_date = None
    merged_to_date = None

    for company_id in company_ids:
        payload = dict(schedule_payload)
        payload["operation"] = "Company"
        payload["companyId"] = company_id
        payload.pop("platoonId", None)

        data, err = _call_dotnet(
            payload,
            trace_id=trace_id,
            session_id=session_id,
            query_type=query_type or "schedule",
        )
        if err or not isinstance(data, dict):
            logger.warning(
                "schedule_enrichment: schedule fetch failed for companyId=%s: %s",
                company_id,
                err,
            )
            continue

        company_payload = data.get("data")
        if not isinstance(company_payload, dict):
            continue

        if merged_date is None:
            merged_date = company_payload.get("date")
            merged_from_date = company_payload.get("fromDate")
            merged_to_date = company_payload.get("toDate")

        for entry in company_payload.get("byCompany") or []:
            merged_by_company.append(entry)

    if not merged_by_company:
        return None

    total_slots = sum(entry.get("totalSlots") or 0 for entry in merged_by_company)
    return {
        "success": True,
        "commandLabel": schedule_payload.get("commandLabel") or "Schedule",
        "data": {
            "date": merged_date,
            "fromDate": merged_from_date,
            "toDate": merged_to_date,
            "totalCompanies": len(merged_by_company),
            "totalSlots": total_slots,
            "byCompany": merged_by_company,
        },
        "message": None,
    }
