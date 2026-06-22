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

from settings import get_dotnet_config
from dotnet_security import resolve_dotnet_verify_ssl

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────
_DOTNET_BASE = get_dotnet_config().BASE_URL
_COMPANY_URL = f"{_DOTNET_BASE}/api/CompanyDetails/Get"
_PLATOON_URL = f"{_DOTNET_BASE}/api/PlatoonDetails/Get"
_DOTNET_API_KEY = os.getenv("DOTNET_API_KEY", "")
_TIMEOUT = int(os.getenv("DOTNET_TIMEOUT", "15"))
_CACHE_TTL_SECONDS = int(os.getenv("ENTITY_CACHE_TTL", "300"))  # 5-minute cache

_VERIFY_SSL = resolve_dotnet_verify_ssl(logger)

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


def _log_dotnet_request(
    *,
    trace_id: Optional[str],
    session_id: Optional[str],
    query_type: str,
    payload: Dict[str, Any],
) -> None:
    logger.info(
        {
            "trace_id": trace_id or "N/A",
            "session_id": session_id or "N/A",
            "query_type": query_type,
            "payload": payload,
        }
    )


def _fetch_companies(
    trace_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> List[Dict]:
    """Fetch all companies from .NET, with 5-minute in-memory cache."""
    global _company_cache
    with _cache_lock:
        if _company_cache is not None:
            ts, data = _company_cache
            if time.time() - ts < _CACHE_TTL_SECONDS:
                return data

    start = time.time()
    _log_dotnet_request(
        trace_id=trace_id,
        session_id=session_id,
        query_type="entity_resolution.company",
        payload={"endpoint": _COMPANY_URL, "method": "GET"},
    )
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
        logger.info(
            {
                "trace_id": trace_id or "N/A",
                "session_id": session_id or "N/A",
                "query_type": "entity_resolution.company",
                "status_code": resp.status_code,
                "duration_ms": round((time.time() - start) * 1000, 2),
                "record_count": len(data),
            }
        )
        return data
    except Exception as exc:
        with _cache_lock:
            cached = list(_company_cache[1]) if _company_cache else []
        logger.warning("Failed to fetch companies from .NET: %s", exc)
        if cached:
            logger.info(
                {
                    "trace_id": trace_id or "N/A",
                    "session_id": session_id or "N/A",
                    "query_type": "entity_resolution.company",
                    "status_code": "cache_fallback",
                    "duration_ms": round((time.time() - start) * 1000, 2),
                    "record_count": len(cached),
                }
            )
            return cached
        return []


def _fetch_platoons(
    trace_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> List[Dict]:
    """Fetch all platoons from .NET, with 5-minute in-memory cache."""
    global _platoon_cache
    with _cache_lock:
        if _platoon_cache is not None:
            ts, data = _platoon_cache
            if time.time() - ts < _CACHE_TTL_SECONDS:
                return data

    start = time.time()
    _log_dotnet_request(
        trace_id=trace_id,
        session_id=session_id,
        query_type="entity_resolution.platoon",
        payload={"endpoint": _PLATOON_URL, "method": "GET"},
    )
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
        logger.info(
            {
                "trace_id": trace_id or "N/A",
                "session_id": session_id or "N/A",
                "query_type": "entity_resolution.platoon",
                "status_code": resp.status_code,
                "duration_ms": round((time.time() - start) * 1000, 2),
                "record_count": len(data),
            }
        )
        return data
    except Exception as exc:
        with _cache_lock:
            cached = list(_platoon_cache[1]) if _platoon_cache else []
        logger.warning("Failed to fetch platoons from .NET: %s", exc)
        if cached:
            logger.info(
                {
                    "trace_id": trace_id or "N/A",
                    "session_id": session_id or "N/A",
                    "query_type": "entity_resolution.platoon",
                    "status_code": "cache_fallback",
                    "duration_ms": round((time.time() - start) * 1000, 2),
                    "record_count": len(cached),
                }
            )
            return cached
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


def resolve_company_id(
    company_name: str,
    trace_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Optional[int]:
    """
    Look up company_name against /api/CompanyDetails/Get and return
    the companyId, or None if not found.
    """
    companies = _fetch_companies(trace_id=trace_id, session_id=session_id)
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
    trace_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Optional[int]:
    """
    Look up platoon_name against /api/PlatoonDetails/Get and return
    the platoonId, or None if not found.

    If company_id is provided, only platoons belonging to that company
    are considered (avoids cross-company name collisions).
    """
    platoons = _fetch_platoons(trace_id=trace_id, session_id=session_id)
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
    trace_id: Optional[str] = None,
    session_id: Optional[str] = None,
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
        result["companyId"] = resolve_company_id(
            company_mention,
            trace_id=trace_id,
            session_id=session_id,
        )

    # Resolve platoon (scoped to company if known)
    if platoon_mention and result["platoonId"] is None:
        result["platoonId"] = resolve_platoon_id(
            platoon_mention,
            company_id=result["companyId"],
            trace_id=trace_id,
            session_id=session_id,
        )

    # ── Fallback 1: Dynamic lookup of Company Names from API ───────────
    if result["companyId"] is None:
        companies = _fetch_companies(trace_id=trace_id, session_id=session_id)
        for co in companies:
            stored = str(_get_field(co, "companyName", "CompanyName", "name", "Name") or "")
            if not stored:
                continue
            # Match stored name as whole words/phrase
            pattern = r"\b" + re.escape(stored.lower()) + r"\b"
            if re.search(pattern, query.lower()):
                result["companyId"] = int(_get_field(co, "companyId", "CompanyId", "id", "Id"))
                result["companyName"] = stored
                break
            
            # Match short name (without company/coy suffix)
            short_name = re.sub(r"\b(?:company|coy)\b", "", stored.lower()).strip()
            if short_name and short_name not in _NOISE_WORDS:
                pattern_short = r"\b" + re.escape(short_name) + r"\b"
                if re.search(pattern_short, query.lower()):
                    result["companyId"] = int(_get_field(co, "companyId", "CompanyId", "id", "Id"))
                    result["companyName"] = stored
                    break

    # ── Fallback 2: Dynamic lookup of Platoon Names from API ───────────
    if result["platoonId"] is None:
        platoons = _fetch_platoons(trace_id=trace_id, session_id=session_id)
        for pl in platoons:
            stored = str(_get_field(pl, "platoonName", "PlatoonName", "name", "Name") or "")
            if not stored:
                continue
            # Match stored name as whole words/phrase (e.g. "PL-01")
            pattern = r"\b" + re.escape(stored.lower()) + r"\b"
            if re.search(pattern, query.lower()):
                result["platoonId"] = int(_get_field(pl, "platoonId", "PlatoonId", "id", "Id"))
                result["platoonName"] = stored
                break
            
            # Match short name (without platoon/pl prefix)
            short_name = re.sub(r"\b(?:platoon|pl)[-\s]?\b", "", stored.lower()).strip()
            if short_name and short_name not in _NOISE_WORDS:
                pattern_short = r"\b" + re.escape(short_name) + r"\b"
                if re.search(pattern_short, query.lower()):
                    # Prefer matching company if company is resolved
                    cid = _get_field(pl, "companyId", "CompanyId")
                    if result["companyId"] is None or cid == result["companyId"]:
                        result["platoonId"] = int(_get_field(pl, "platoonId", "PlatoonId", "id", "Id"))
                        result["platoonName"] = stored
                        break

    # ── Map Resolved IDs back to Canonical Names ───────────────────────
    if result["companyId"] is not None:
        companies = _fetch_companies(trace_id=trace_id, session_id=session_id)
        for co in companies:
            cid = _get_field(co, "companyId", "CompanyId", "id", "Id")
            if cid is not None and int(cid) == result["companyId"]:
                result["companyName"] = str(_get_field(co, "companyName", "CompanyName", "name", "Name") or "")
                break

    if result["platoonId"] is not None:
        platoons = _fetch_platoons(trace_id=trace_id, session_id=session_id)
        for pl in platoons:
            pid = _get_field(pl, "platoonId", "PlatoonId", "id", "Id")
            if pid is not None and int(pid) == result["platoonId"]:
                result["platoonName"] = str(_get_field(pl, "platoonName", "PlatoonName", "name", "Name") or "")
                break

    return result
