"""
admin_entity_resolver.py
========================
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import requests as _requests

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────
_DOTNET_BASE = os.getenv("DOTNET_API_BASE_URL", "https://localhost:7257")
_COMPANY_URL = f"{_DOTNET_BASE}/api/CompanyDetails/Get"
_PLATOON_URL = f"{_DOTNET_BASE}/api/PlatoonDetails/Get"
_DOTNET_API_KEY = os.getenv("DOTNET_API_KEY", "")
_TIMEOUT = int(os.getenv("DOTNET_TIMEOUT", "15"))
_CACHE_TTL_SECONDS = int(os.getenv("ENTITY_CACHE_TTL", "300"))  # 5-minute cache

_skip_raw = os.getenv("DOTNET_SKIP_SSL_VERIFY", os.getenv("DOTNET_VERIFY_SSL", "0"))
_VERIFY_SSL = _skip_raw.strip() not in {"1", "true", "True"}

_session = _requests.Session()

# ── In-memory cache to avoid hammering the lookup APIs on every query ──────
_cache_lock = threading.RLock()
_company_cache: Optional[Tuple[float, List[Dict]]] = None  # (timestamp, data)
_platoon_cache: Optional[Tuple[float, List[Dict]]] = None


# =============================================================================
# NAME EXTRACTION
# =============================================================================

# Words that appear before/after "company" / "coy" / "platoon" / "pl"
# but are NOT part of the entity name.
_NOISE_WORDS = {
    "in",
    "for",
    "of",
    "the",
    "a",
    "an",
    "this",
    "that",
    "all",
    "and",
    "or",
    "at",
    "to",
    "from",
    "by",
    "on",
    "with",
    "about",
    "top",
    "best",
    "show",
    "get",
    "give",
    "is",
    "are",
    "was",
    "my",
    "their",
    "our",
    "its",
    "status",
    "data",
    "report",
    "attendance",
    "leave",
    "performance",
    "breakdown",
    "strength",
    "scores",
    "score",
    "performers",
    "details",
    "info",
    "information",
}


def _clean_candidate(s: str) -> str:
    """Strip leading / trailing noise words from a regex-captured group."""
    words = s.strip().split()
    while words and words[0] in _NOISE_WORDS:
        words = words[1:]
    while words and words[-1] in _NOISE_WORDS:
        words = words[:-1]
    return " ".join(words)


def extract_company_mention(text: str) -> Optional[str]:
    """
    Return the company name mentioned in *text*, or None.

    Handles patterns like:
      "alpha company"       → "alpha"
      "company alpha"       → "alpha"
      "14 punjab company"   → "14 punjab"
      "bravo coy"           → "bravo"
      "coy bravo"           → "bravo"
    """
    q = text.lower().strip()

    # Pattern A: 1-3 words immediately BEFORE "company" or "coy"
    for m in re.finditer(r"\b((?:\w+\s+){0,2}\w+)\s+(?:company|coy)\b", q):
        candidate = _clean_candidate(m.group(1))
        if candidate:
            return candidate

    # Pattern B: "company/coy" followed by exactly ONE word
    for m in re.finditer(r"\b(?:company|coy)\s+(\w+)\b", q):
        candidate = m.group(1).strip()
        if candidate not in _NOISE_WORDS:
            return candidate

    return None


def extract_platoon_mention(text: str) -> Optional[str]:
    """
    Return the platoon name / number mentioned in *text*, or None.

    Handles patterns like:
      "platoon 3"       → "3"
      "platoon no. 5"   → "5"
      "pl-01"           → "01"
      "pl 2"            → "2"
      "pl03"            → "03"
      "3 platoon"       → "3"
      "PL-01"           → "01"
    """
    q = text.lower().strip()
    patterns = [
        r"\bplatoon\s+no\.?\s*(\w[\w-]*)\b",
        r"\bplatoon\s+(\w[\w-]*)\b",
        r"\bpl[-\s](\w+)\b",  # "pl-2", "pl 3"
        r"\bpl(\d+)\b",  # "pl03"
        r"\b(\d+)\s+platoon\b",
    ]
    for pattern in patterns:
        m = re.search(pattern, q)
        if m:
            return m.group(1).strip().rstrip(".,")
    return None


# =============================================================================
# API FETCH WITH CACHING
# =============================================================================


def _dotnet_headers() -> Dict[str, str]:
    h = {"Content-Type": "application/json", "Accept": "application/json"}
    if _DOTNET_API_KEY:
        h["X-Api-Key"] = _DOTNET_API_KEY
    return h


def _fetch_companies() -> List[Dict]:
    """Fetch all companies from .NET, with 5-minute in-memory cache."""
    global _company_cache
    with _cache_lock:
        if _company_cache is not None:
            ts, data = _company_cache
            if time.time() - ts < _CACHE_TTL_SECONDS:
                return data

    try:
        resp = _session.get(
            _COMPANY_URL,
            headers=_dotnet_headers(),
            timeout=_TIMEOUT,
            verify=_VERIFY_SSL,
        )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict):
            # Some .NET wrappers: { "data": [...], "success": true }
            data = data.get("data") or data.get("Data") or []
        data = data if isinstance(data, list) else []
        with _cache_lock:
            _company_cache = (time.time(), data)
        logger.debug("Fetched %d companies from .NET", len(data))
        return data
    except Exception as exc:
        logger.warning("Failed to fetch companies from .NET: %s", exc)
        return []


def _fetch_platoons() -> List[Dict]:
    """Fetch all platoons from .NET, with 5-minute in-memory cache."""
    global _platoon_cache
    with _cache_lock:
        if _platoon_cache is not None:
            ts, data = _platoon_cache
            if time.time() - ts < _CACHE_TTL_SECONDS:
                return data

    try:
        resp = _session.get(
            _PLATOON_URL,
            headers=_dotnet_headers(),
            timeout=_TIMEOUT,
            verify=_VERIFY_SSL,
        )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict):
            data = data.get("data") or data.get("Data") or []
        data = data if isinstance(data, list) else []
        with _cache_lock:
            _platoon_cache = (time.time(), data)
        logger.debug("Fetched %d platoons from .NET", len(data))
        return data
    except Exception as exc:
        logger.warning("Failed to fetch platoons from .NET: %s", exc)
        return []


def invalidate_cache() -> None:
    """Force-clear the entity cache (e.g. after a roster change)."""
    global _company_cache, _platoon_cache
    with _cache_lock:
        _company_cache = None
        _platoon_cache = None
    logger.info("Entity resolver cache invalidated.")


# =============================================================================
# ID RESOLUTION
# =============================================================================


def _get_field(obj: Dict, *keys) -> Any:
    """Try multiple key casings and return the first non-None hit."""
    for key in keys:
        v = obj.get(key)
        if v is not None:
            return v
    return None


def _normalise_name(name: str) -> str:
    """Lowercase, strip punctuation/spaces for fuzzy comparison."""
    return re.sub(r"[\s\-_]+", "", (name or "").lower())


def _name_matches(stored_name: str, query_name: str) -> bool:
    """
    Return True if *query_name* (from user text) matches *stored_name*
    (from .NET API) using multiple strategies:
      1. Exact case-insensitive match
      2. Normalised (no spaces/dashes) match
      3. The stored name contains the query name as a whole word / vice-versa
    """
    sn = stored_name.lower().strip()
    qn = query_name.lower().strip()
    if sn == qn:
        return True
    if _normalise_name(sn) == _normalise_name(qn):
        return True
    # "pl-01" stored, user says "01"  →  normalised stored contains query
    if _normalise_name(qn) in _normalise_name(sn):
        return True
    # user says "platoon 1" but stored says "1"
    if _normalise_name(sn) in _normalise_name(qn):
        return True
    return False


def resolve_company_id(company_name: str) -> Optional[int]:
    """
    Look up company_name against /api/CompanyDetails/Get and return
    the companyId, or None if not found.
    """
    companies = _fetch_companies()
    for co in companies:
        stored = str(_get_field(co, "companyName", "CompanyName", "name", "Name") or "")
        if stored and _name_matches(stored, company_name):
            cid = _get_field(co, "companyId", "CompanyId", "id", "Id")
            if cid is not None:
                logger.debug(
                    "Resolved company %r → %r (id=%s)", company_name, stored, cid
                )
                return int(cid)
    logger.debug("Company not found: %r", company_name)
    return None


def resolve_platoon_id(
    platoon_name: str,
    company_id: Optional[int] = None,
) -> Optional[int]:
    """
    Look up platoon_name against /api/PlatoonDetails/Get and return
    the platoonId, or None if not found.

    If company_id is provided, only platoons belonging to that company
    are considered (avoids cross-company name collisions).
    """
    platoons = _fetch_platoons()
    candidates = []
    for pl in platoons:
        stored = str(_get_field(pl, "platoonName", "PlatoonName", "name", "Name") or "")
        if stored and _name_matches(stored, platoon_name):
            pid = _get_field(pl, "platoonId", "PlatoonId", "id", "Id")
            cid = _get_field(pl, "companyId", "CompanyId")
            if pid is not None:
                candidates.append((int(pid), cid))

    if not candidates:
        logger.debug("Platoon not found: %r", platoon_name)
        return None

    # If we have a company filter, prefer matches within that company
    if company_id is not None:
        scoped = [(pid, cid) for pid, cid in candidates if cid == company_id]
        if scoped:
            pid = scoped[0][0]
            logger.debug(
                "Resolved platoon %r (in company %s) → id=%s",
                platoon_name,
                company_id,
                pid,
            )
            return pid

    pid = candidates[0][0]
    logger.debug("Resolved platoon %r → id=%s", platoon_name, pid)
    return pid


# =============================================================================
# PUBLIC ENTRY POINT
# =============================================================================


def resolve_entities_from_query(
    query: str,
    existing_company_id: Optional[int] = None,
    existing_platoon_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Extract company / platoon mentions from *query*, resolve them to IDs,
    and return a dict with keys:

      {
        "companyId":      int | None,
        "platoonId":      int | None,
        "companyName":    str | None,   (the name extracted from the query)
        "platoonName":    str | None,   (the name extracted from the query)
      }

    Already-resolved IDs passed in via *existing_company_id* /
    *existing_platoon_id* are NOT overwritten — they take priority.
    This lets the frontend still pass explicit IDs when it has them.
    """
    result: Dict[str, Any] = {
        "companyId": existing_company_id,
        "platoonId": existing_platoon_id,
        "companyName": None,
        "platoonName": None,
    }

    company_mention = extract_company_mention(query)
    platoon_mention = extract_platoon_mention(query)

    result["companyName"] = company_mention
    result["platoonName"] = platoon_mention

    # Resolve company first (platoon resolution may use companyId)
    if company_mention and result["companyId"] is None:
        result["companyId"] = resolve_company_id(company_mention)

    # Resolve platoon (scoped to company if known)
    if platoon_mention and result["platoonId"] is None:
        result["platoonId"] = resolve_platoon_id(
            platoon_mention,
            company_id=result["companyId"],
        )

    return result
