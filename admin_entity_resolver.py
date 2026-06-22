"""
admin_entity_resolver.py
========================
Resolves company, platoon, batch, and agniveer IDs from natural language queries.
Fixed version — handles fuzzy name matching, partial names, numeric IDs, and
natural language patterns more robustly.
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

# ── In-memory cache ────────────────────────────────────────────────────────
_cache_lock = threading.RLock()
_company_cache: Optional[Tuple[float, List[Dict]]] = None
_platoon_cache: Optional[Tuple[float, List[Dict]]] = None


# =============================================================================
# NAME EXTRACTION — improved patterns
# =============================================================================

_NOISE_WORDS = {
    "in", "for", "of", "the", "a", "an", "this", "that", "all", "and", "or",
    "at", "to", "from", "by", "on", "with", "about", "top", "best", "show",
    "get", "give", "is", "are", "was", "my", "their", "our", "its", "status",
    "data", "report", "attendance", "leave", "performance", "breakdown",
    "strength", "scores", "score", "performers", "details", "info", "information",
    "today", "current", "monthly", "weekly", "daily",
}


def _clean_candidate(s: str) -> str:
    words = s.strip().split()
    while words and words[0].lower() in _NOISE_WORDS:
        words = words[1:]
    while words and words[-1].lower() in _NOISE_WORDS:
        words = words[:-1]
    return " ".join(words)


def extract_company_mention(text: str) -> Optional[str]:
    """
    Extract company name from query text.

    Handles:
      "alpha company", "company alpha", "alpha coy", "coy bravo",
      "14 punjab company", "A company", "B coy", "company 2" (numeric)
    """
    q = text.lower().strip()

    # Pattern A: 1-3 words BEFORE "company" or "coy"
    for m in re.finditer(r"\b((?:\w+\s+){0,2}\w+)\s+(?:company|coy)\b", q):
        candidate = _clean_candidate(m.group(1))
        if candidate and candidate.lower() not in _NOISE_WORDS:
            return candidate

    # Pattern B: "company/coy" followed by 1-2 words or a number
    for m in re.finditer(r"\b(?:company|coy)\s+([\w]+(?:\s+[\w]+)?)\b", q):
        candidate = m.group(1).strip()
        if candidate.lower() not in _NOISE_WORDS:
            return candidate

    return None


def extract_platoon_mention(text: str) -> Optional[str]:
    """
    Extract platoon name/number from query text.

    Handles:
      "platoon 3", "platoon no. 5", "pl-01", "pl 2", "pl03",
      "3 platoon", "PL-01", "platoon one"
    """
    q = text.lower().strip()

    _WORD_TO_NUM = {
        "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
        "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    }

    patterns = [
        r"\bplatoon\s+no\.?\s*(\w[\w-]*)\b",
        r"\bplatoon\s+(\w[\w-]*)\b",
        r"\bpl[-\s](\w+)\b",
        r"\bpl(\d+)\b",
        r"\b(\d+)\s+platoon\b",
    ]
    for pattern in patterns:
        m = re.search(pattern, q)
        if m:
            val = m.group(1).strip().rstrip(".,")
            return _WORD_TO_NUM.get(val, val)

    return None


def extract_agniveer_mention(text: str) -> Optional[str]:
    """
    Extract agniveer number/ID from query text.

    Handles:
      "AG12345", "ag 12345", "agniveer no AG001", "A0701515Y",
      numeric IDs like "12345", "agniveer 501"
    """
    # Pattern 1: Explicit AG prefix (case-insensitive)
    m = re.search(r"\bag[-\s]?(\d{3,8})\b", text, re.IGNORECASE)
    if m:
        return f"AG{m.group(1)}"

    # Pattern 2: Letter+digits alphanumeric ID (e.g. A0701515Y)
    m = re.search(r"\b([A-Za-z]\d{5,8}[A-Za-z]?)\b", text)
    if m:
        candidate = m.group(1)
        # Avoid matching common words
        if not re.match(r"^[a-z]+$", candidate, re.IGNORECASE):
            return candidate.upper()

    # Pattern 3: "agniveer no/number <digits>"
    m = re.search(
        r"\bagniveer\s+(?:no\.?|number|#)?\s*(\w{3,10})\b",
        text, re.IGNORECASE
    )
    if m:
        return m.group(1).upper()

    return None


def extract_batch_mention(text: str) -> Optional[str]:
    """
    Extract batch number from query text.

    Handles: "batch 5", "batch no 3", "bt 2", "batch-01"
    """
    q = text.lower().strip()
    patterns = [
        r"\bbatch\s+no\.?\s*(\w[\w-]*)\b",
        r"\bbatch\s+(\w[\w-]*)\b",
        r"\bbt[-\s]?(\w+)\b",
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


def _fetch_companies(
    trace_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> List[Dict]:
    global _company_cache
    with _cache_lock:
        if _company_cache is not None:
            ts, data = _company_cache
            if time.time() - ts < _CACHE_TTL_SECONDS:
                return data

    start = time.time()
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
            data = data.get("data") or data.get("Data") or []
        data = data if isinstance(data, list) else []
        with _cache_lock:
            _company_cache = (time.time(), data)
        logger.info(
            {"trace_id": trace_id or "N/A", "query_type": "entity_resolution.company",
             "status_code": resp.status_code,
             "duration_ms": round((time.time() - start) * 1000, 2),
             "record_count": len(data)}
        )
        return data
    except Exception as exc:
        with _cache_lock:
            cached = list(_company_cache[1]) if _company_cache else []
        logger.warning("Failed to fetch companies from .NET: %s", exc)
        return cached


def _fetch_platoons(
    trace_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> List[Dict]:
    global _platoon_cache
    with _cache_lock:
        if _platoon_cache is not None:
            ts, data = _platoon_cache
            if time.time() - ts < _CACHE_TTL_SECONDS:
                return data

    start = time.time()
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
            {"trace_id": trace_id or "N/A", "query_type": "entity_resolution.platoon",
             "status_code": resp.status_code,
             "duration_ms": round((time.time() - start) * 1000, 2),
             "record_count": len(data)}
        )
        return data
    except Exception as exc:
        with _cache_lock:
            cached = list(_platoon_cache[1]) if _platoon_cache else []
        logger.warning("Failed to fetch platoons from .NET: %s", exc)
        return cached


def invalidate_cache() -> None:
    global _company_cache, _platoon_cache
    with _cache_lock:
        _company_cache = None
        _platoon_cache = None
    logger.info("Entity resolver cache invalidated.")


# =============================================================================
# ID RESOLUTION — improved fuzzy matching
# =============================================================================


def _get_field(obj: Dict, *keys) -> Any:
    for key in keys:
        v = obj.get(key)
        if v is not None:
            return v
    return None


def _normalise_name(name: str) -> str:
    """Lowercase, strip punctuation/spaces/hyphens for fuzzy comparison."""
    return re.sub(r"[\s\-_./]+", "", (name or "").lower())


def _name_matches(stored_name: str, query_name: str) -> bool:
    """
    Return True if query_name matches stored_name using multiple strategies:
      1. Exact case-insensitive match
      2. Normalised (no spaces/dashes) match
      3. Stored name contains query (e.g. "Alpha Company" matches "alpha")
      4. Query contains stored name
      5. Stored short name (without suffix words) matches query
    """
    sn = stored_name.lower().strip()
    qn = query_name.lower().strip()

    if not sn or not qn:
        return False

    if sn == qn:
        return True

    sn_norm = _normalise_name(sn)
    qn_norm = _normalise_name(qn)

    if sn_norm == qn_norm:
        return True

    # Stored contains query (e.g. "Alpha Company" stored, user said "alpha")
    if qn_norm and qn_norm in sn_norm:
        return True

    # Query contains stored (e.g. user said "alpha coy", stored is "alpha")
    if sn_norm and sn_norm in qn_norm:
        return True

    # Strip common suffixes from stored name and compare
    suffixes = r"\b(?:company|coy|platoon|pl|unit|battalion|bat)\b"
    sn_stripped = re.sub(suffixes, "", sn).strip()
    sn_stripped_norm = _normalise_name(sn_stripped)
    if sn_stripped_norm and sn_stripped_norm == qn_norm:
        return True

    # Numeric match: stored "PL-01", query "1" → normalise both to digits
    sn_digits = re.sub(r"\D", "", sn_norm)
    qn_digits = re.sub(r"\D", "", qn_norm)
    if sn_digits and qn_digits and sn_digits == qn_digits:
        # Only match if the non-digit part is a prefix indicator (pl, platoon)
        if re.match(r"^(?:pl|platoon|co|company|coy|bat|battalion)", sn_norm):
            return True

    return False


def resolve_company_id(
    company_name: str,
    trace_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Optional[int]:
    companies = _fetch_companies(trace_id=trace_id, session_id=session_id)
    for co in companies:
        stored = str(_get_field(co, "companyName", "CompanyName", "name", "Name") or "")
        if stored and _name_matches(stored, company_name):
            cid = _get_field(co, "companyId", "CompanyId", "id", "Id")
            if cid is not None:
                logger.debug("Resolved company %r → %r (id=%s)", company_name, stored, cid)
                return int(cid)
    logger.debug("Company not found: %r", company_name)
    return None


def resolve_platoon_id(
    platoon_name: str,
    company_id: Optional[int] = None,
    trace_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Optional[int]:
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

    if company_id is not None:
        scoped = [(pid, cid) for pid, cid in candidates if cid == company_id]
        if scoped:
            return scoped[0][0]

    return candidates[0][0]


def resolve_batch_id(
    batch_name: str,
    trace_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Optional[int]:
    """
    Resolve batch by name/number. Tries numeric parse first, then API lookup
    if a batch endpoint is available.
    """
    # If it's purely numeric, return as int directly
    if re.match(r"^\d+$", batch_name.strip()):
        return int(batch_name.strip())

    # Normalised numeric (e.g. "01" → 1)
    digits = re.sub(r"\D", "", batch_name)
    if digits:
        return int(digits)

    return None


# =============================================================================
# PUBLIC ENTRY POINT — enhanced
# =============================================================================


def resolve_entities_from_query(
    query: str,
    existing_company_id: Optional[int] = None,
    existing_platoon_id: Optional[int] = None,
    existing_batch_id: Optional[int] = None,
    trace_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Extract company / platoon / batch / agniveer mentions from *query*,
    resolve them to IDs, and return a dict with keys:

      {
        "companyId":      int | None,
        "platoonId":      int | None,
        "batchId":        int | None,
        "agniveerNo":     str | None,
        "companyName":    str | None,
        "platoonName":    str | None,
        "batchName":      str | None,
      }

    Already-resolved IDs passed in via *existing_** are NOT overwritten.
    """
    result: Dict[str, Any] = {
        "companyId": existing_company_id,
        "platoonId": existing_platoon_id,
        "batchId": existing_batch_id,
        "agniveerNo": None,
        "companyName": None,
        "platoonName": None,
        "batchName": None,
    }

    # ── Extract mentions from query ────────────────────────────────────────
    company_mention = extract_company_mention(query)
    platoon_mention = extract_platoon_mention(query)
    batch_mention = extract_batch_mention(query)
    agniveer_mention = extract_agniveer_mention(query)

    result["companyName"] = company_mention
    result["platoonName"] = platoon_mention
    result["batchName"] = batch_mention
    result["agniveerNo"] = agniveer_mention

    # ── Resolve company first (platoon resolution may use companyId) ────────
    if company_mention and result["companyId"] is None:
        result["companyId"] = resolve_company_id(
            company_mention, trace_id=trace_id, session_id=session_id
        )

    # ── Resolve platoon (scoped to company if known) ────────────────────────
    if platoon_mention and result["platoonId"] is None:
        result["platoonId"] = resolve_platoon_id(
            platoon_mention,
            company_id=result["companyId"],
            trace_id=trace_id,
            session_id=session_id,
        )

    # ── Resolve batch ───────────────────────────────────────────────────────
    if batch_mention and result["batchId"] is None:
        result["batchId"] = resolve_batch_id(
            batch_mention, trace_id=trace_id, session_id=session_id
        )

    # ── Fallback 1: Scan all company names from API against full query ──────
    if result["companyId"] is None:
        companies = _fetch_companies(trace_id=trace_id, session_id=session_id)
        query_norm = _normalise_name(query)
        best_match_len = 0
        for co in companies:
            stored = str(_get_field(co, "companyName", "CompanyName", "name", "Name") or "")
            if not stored:
                continue
            stored_norm = _normalise_name(stored)
            # Check if stored name (or its core without suffix) appears in query
            stored_core = re.sub(r"(?:company|coy|unit)", "", stored.lower()).strip()
            stored_core_norm = _normalise_name(stored_core)
            # Match stored or core against query
            for candidate_norm in filter(None, [stored_norm, stored_core_norm]):
                if len(candidate_norm) >= 2 and candidate_norm in query_norm:
                    if len(candidate_norm) > best_match_len:
                        cid = _get_field(co, "companyId", "CompanyId", "id", "Id")
                        if cid is not None:
                            result["companyId"] = int(cid)
                            result["companyName"] = stored
                            best_match_len = len(candidate_norm)

    # ── Fallback 2: Scan all platoon names from API against full query ──────
    if result["platoonId"] is None:
        platoons = _fetch_platoons(trace_id=trace_id, session_id=session_id)
        query_norm = _normalise_name(query)
        best_match_len = 0
        for pl in platoons:
            stored = str(_get_field(pl, "platoonName", "PlatoonName", "name", "Name") or "")
            if not stored:
                continue
            stored_norm = _normalise_name(stored)
            stored_core = re.sub(r"(?:platoon|pl)", "", stored.lower()).strip()
            stored_core_norm = _normalise_name(stored_core)
            for candidate_norm in filter(None, [stored_norm, stored_core_norm]):
                if len(candidate_norm) >= 2 and candidate_norm in query_norm:
                    if len(candidate_norm) > best_match_len:
                        pid = _get_field(pl, "platoonId", "PlatoonId", "id", "Id")
                        cid = _get_field(pl, "companyId", "CompanyId")
                        if pid is not None:
                            # Prefer scoped match if company is known
                            if result["companyId"] is None or cid == result["companyId"]:
                                result["platoonId"] = int(pid)
                                result["platoonName"] = stored
                                best_match_len = len(candidate_norm)

    # ── Back-fill canonical names from resolved IDs ──────────────────────────
    if result["companyId"] is not None and not result["companyName"]:
        companies = _fetch_companies(trace_id=trace_id, session_id=session_id)
        for co in companies:
            cid = _get_field(co, "companyId", "CompanyId", "id", "Id")
            if cid is not None and int(cid) == result["companyId"]:
                result["companyName"] = str(
                    _get_field(co, "companyName", "CompanyName", "name", "Name") or ""
                )
                break

    if result["platoonId"] is not None and not result["platoonName"]:
        platoons = _fetch_platoons(trace_id=trace_id, session_id=session_id)
        for pl in platoons:
            pid = _get_field(pl, "platoonId", "PlatoonId", "id", "Id")
            if pid is not None and int(pid) == result["platoonId"]:
                result["platoonName"] = str(
                    _get_field(pl, "platoonName", "PlatoonName", "name", "Name") or ""
                )
                break

    return result