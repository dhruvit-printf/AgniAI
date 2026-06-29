"""Entity resolution package for AgniAI."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .agniveer_resolver import resolve_agniveer
from .company_resolver import resolve_company
from .entity_cache import (
    ENTITY_CACHE,
    EntityCache,
    fetch_agniveers,
    fetch_companies,
    fetch_platoons,
    invalidate_cache,
    preload_entities,
    refresh_all_entities,
)
from .entity_matcher import match_entity, normalize_text
from .entity_models import AgniveerEntity, CompanyEntity, EntityMatch, PlatoonEntity
from .entity_refresh_service import (
    EntityRefreshService,
    refresh_entities_now,
    start_entity_refresh_service,
)
from .platoon_resolver import resolve_platoon


def resolve_entities_from_query(
    query: str,
    *,
    existing_company_id: Optional[int] = None,
    existing_platoon_id: Optional[int] = None,
    existing_batch_id: Optional[int] = None,
    trace_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    company = resolve_company(query, trace_id=trace_id, session_id=session_id)
    platoon = resolve_platoon(
        query,
        company_id=company.get("CompanyId") or existing_company_id,
        trace_id=trace_id,
        session_id=session_id,
    )
    agniveer = resolve_agniveer(query, trace_id=trace_id, session_id=session_id)

    return {
        "companyId": (
            existing_company_id
            if existing_company_id is not None
            else company.get("CompanyId")
        ),
        "platoonId": (
            existing_platoon_id
            if existing_platoon_id is not None
            else platoon.get("PlatoonId")
        ),
        "batchId": existing_batch_id,
        "agniveerNo": agniveer.get("AgniveerNo"),
        "companyName": company.get("name"),
        "platoonName": platoon.get("name"),
        "batchName": None,
        "company": company,
        "platoon": platoon,
        "agniveer": agniveer,
        "needs_clarification": bool(
            company.get("needs_clarification")
            or platoon.get("needs_clarification")
            or agniveer.get("needs_clarification")
        ),
        "candidates": {
            "company": company.get("candidates", []),
            "platoon": platoon.get("candidates", []),
            "agniveer": agniveer.get("candidates", []),
        },
    }


__all__ = [
    "AgniveerEntity",
    "CompanyEntity",
    "EntityCache",
    "EntityMatch",
    "EntityRefreshService",
    "ENTITY_CACHE",
    "PlatoonEntity",
    "fetch_agniveers",
    "fetch_companies",
    "fetch_platoons",
    "invalidate_cache",
    "match_entity",
    "normalize_text",
    "preload_entities",
    "refresh_all_entities",
    "refresh_entities_now",
    "resolve_entities_from_query",
    "resolve_agniveer",
    "resolve_company",
    "resolve_platoon",
    "start_entity_refresh_service",
]
