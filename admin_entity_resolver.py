"""
admin_entity_resolver.py
========================
Compatibility layer for legacy entity resolution helpers.

The new resolver package lives in ``entity_resolver/``. This module keeps the
existing public surface used by the current tests and pipeline while delegating
all collection fetching to the shared entity cache.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from entity_resolver.entity_cache import fetch_companies as _cache_fetch_companies
from entity_resolver.entity_cache import fetch_platoons as _cache_fetch_platoons
from entity_resolver.entity_matcher import normalize_text as _normalize_text

logger = logging.getLogger(__name__)


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
    "today",
    "current",
    "monthly",
    "weekly",
    "daily",
}


def _clean_candidate(s: str) -> str:
    words = s.strip().split()
    while words and words[0].lower() in _NOISE_WORDS:
        words = words[1:]
    while words and words[-1].lower() in _NOISE_WORDS:
        words = words[:-1]
    return " ".join(words)


_COMPANY_TOKEN_RE = re.compile(r"[a-z0-9]+")
_PLATOON_PATTERNS = [
    re.compile(r"\bplatoon\s+no\.?\s*(\w[\w-]*)\b"),
    re.compile(r"\bplatoon\s+(\w[\w-]*)\b"),
    re.compile(r"\bpl[-\s](\w+)\b"),
    re.compile(r"\bpl(\d+)\b"),
    re.compile(r"\b(\d+)\s+platoon\b"),
]
_AGNIVEER_NUM_RE = re.compile(r"\bag[-\s]?(\d{3,8})\b", re.IGNORECASE)
_AGNIVEER_ALPHANUM_RE = re.compile(r"\b([A-Za-z]\d{5,8}[A-Za-z]?)\b")
_AGNIVEER_WORD_RE = re.compile(r"\bagniveer\s+(?:no\.?|number|#)?\s*(\w{3,10})\b", re.IGNORECASE)
_BATCH_PATTERNS = [
    re.compile(r"\bbatch\s+no\.?\s*(\w[\w-]*)\b"),
    re.compile(r"\bbatch\s+(\w[\w-]*)\b"),
    re.compile(r"\bbt[-\s]?(\w+)\b"),
]


def extract_company_mention(text: str) -> Optional[str]:
    q = text.lower().strip()
    tokens = _COMPANY_TOKEN_RE.findall(q)
    if not tokens:
        return None

    for idx, token in enumerate(tokens):
        if token not in ("company", "coy"):
            continue

        before: List[str] = []
        for candidate in reversed(tokens[max(0, idx - 3) : idx]):
            if candidate in _NOISE_WORDS and not (
                len(candidate) == 1 and candidate.isalpha()
            ):
                continue
            before.append(candidate)
        if before:
            return " ".join(reversed(before)).strip()

        after: List[str] = []
        for candidate in tokens[idx + 1 : idx + 4]:
            if candidate in _NOISE_WORDS and not (
                len(candidate) == 1 and candidate.isalpha()
            ):
                break
            after.append(candidate)
        if after:
            return " ".join(after).strip()

    return None


def extract_platoon_mention(text: str) -> Optional[str]:
    q = text.lower().strip()
    _WORD_TO_NUM = {
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
    }

    for pattern in _PLATOON_PATTERNS:
        m = pattern.search(q)
        if m:
            val = m.group(1).strip().rstrip(".,")
            return _WORD_TO_NUM.get(val, val)

    return None


def extract_agniveer_mention(text: str) -> Optional[str]:
    m = _AGNIVEER_NUM_RE.search(text)
    if m:
        return f"AG{m.group(1)}"

    m = _AGNIVEER_ALPHANUM_RE.search(text)
    if m:
        candidate = m.group(1)
        if not re.match(r"^[a-z]+$", candidate, re.IGNORECASE):
            return candidate.upper()

    m = _AGNIVEER_WORD_RE.search(text)
    if m:
        return m.group(1).upper()

    return None


def extract_batch_mention(text: str) -> Optional[str]:
    q = text.lower().strip()
    for pattern in _BATCH_PATTERNS:
        m = pattern.search(q)
        if m:
            return m.group(1).strip().rstrip(".,")
    return None


def _fetch_companies(
    trace_id: Optional[str] = None,
) -> List[Dict]:
    return _cache_fetch_companies(trace_id=trace_id)


def _fetch_platoons(
    trace_id: Optional[str] = None,
) -> List[Dict]:
    return _cache_fetch_platoons(trace_id=trace_id)


def invalidate_cache() -> None:
    from entity_resolver.entity_cache import invalidate_cache as _invalidate

    _invalidate()


def _get_field(obj: Dict, *keys) -> Any:
    for key in keys:
        v = obj.get(key)
        if v is not None:
            return v
    return None


def _normalise_name(name: str) -> str:
    return re.sub(r"[\s\-_./]+", "", (name or "").lower())


def _name_matches(stored_name: str, query_name: str) -> bool:
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
    if qn_norm and qn_norm in sn_norm:
        return True
    if sn_norm and sn_norm in qn_norm:
        return True

    suffixes = r"\b(?:company|coy|platoon|pl|unit|battalion|bat)\b"
    sn_stripped = re.sub(suffixes, "", sn).strip()
    sn_stripped_norm = _normalise_name(sn_stripped)
    if sn_stripped_norm and sn_stripped_norm == qn_norm:
        return True

    sn_digits = re.sub(r"\D", "", sn_norm)
    qn_digits = re.sub(r"\D", "", qn_norm)
    if sn_digits and qn_digits and sn_digits == qn_digits:
        if re.match(r"^(?:pl|platoon|co|company|coy|bat|battalion)", sn_norm):
            return True

    return False


def resolve_company_id(
    company_name: str,
    trace_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Optional[int]:
    companies = _fetch_companies(trace_id=trace_id)
    for co in companies:
        stored = str(_get_field(co, "companyName", "CompanyName", "name", "Name") or "")
        if stored and _name_matches(stored, company_name):
            cid = _get_field(co, "companyId", "CompanyId", "id", "Id")
            if cid is not None:
                return int(cid)
    return None


def resolve_platoon_id(
    platoon_name: str,
    company_id: Optional[int] = None,
    trace_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Optional[int]:
    platoons = _fetch_platoons(trace_id=trace_id)
    candidates = []
    for pl in platoons:
        stored = str(_get_field(pl, "platoonName", "PlatoonName", "name", "Name") or "")
        if stored and _name_matches(stored, platoon_name):
            pid = _get_field(pl, "platoonId", "PlatoonId", "id", "Id")
            cid = _get_field(pl, "companyId", "CompanyId")
            if pid is not None:
                candidates.append((int(pid), cid))

    if not candidates:
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
    if re.match(r"^\d+$", batch_name.strip()):
        return int(batch_name.strip())
    digits = re.sub(r"\D", "", batch_name)
    if digits:
        return int(digits)
    return None


def resolve_entities_from_query(
    query: str,
    existing_company_id: Optional[int] = None,
    existing_platoon_id: Optional[int] = None,
    existing_batch_id: Optional[int] = None,
    trace_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "companyId": existing_company_id,
        "platoonId": existing_platoon_id,
        "batchId": existing_batch_id,
        "agniveerNo": None,
        "companyName": None,
        "platoonName": None,
        "batchName": None,
    }

    company_mention = extract_company_mention(query)
    platoon_mention = extract_platoon_mention(query)
    batch_mention = extract_batch_mention(query)
    agniveer_mention = extract_agniveer_mention(query)

    result["companyName"] = company_mention
    result["platoonName"] = platoon_mention
    result["batchName"] = batch_mention
    result["agniveerNo"] = agniveer_mention

    if company_mention and result["companyId"] is None:
        result["companyId"] = resolve_company_id(
            company_mention, trace_id=trace_id, session_id=session_id
        )

    if platoon_mention and result["platoonId"] is None:
        result["platoonId"] = resolve_platoon_id(
            platoon_mention,
            company_id=result["companyId"],
            trace_id=trace_id,
            session_id=session_id,
        )

    if batch_mention and result["batchId"] is None:
        result["batchId"] = resolve_batch_id(
            batch_mention, trace_id=trace_id, session_id=session_id
        )

    # ── Helper for whole-word / phrase matching with boundary checks ──
    from query_normalizer import clean_query

    def _is_mention_in_query(query_text: str, candidate_name: str) -> bool:
        if not candidate_name:
            return False
        # Ignore pure digits in fallback lookup to avoid matching counts/dates/attempts
        cand_clean = clean_query(candidate_name).lower().strip()
        if not cand_clean or cand_clean.isdigit() or len(cand_clean) < 2:
            return False
        q_clean = clean_query(query_text).lower()
        pos = q_clean.find(cand_clean)
        while pos != -1:
            before_ok = pos == 0 or not q_clean[pos - 1].isalnum()
            after_ok = (
                pos + len(cand_clean) == len(q_clean)
                or not q_clean[pos + len(cand_clean)].isalnum()
            )
            if before_ok and after_ok:
                return True
            pos = q_clean.find(cand_clean, pos + 1)
        return False

    if result["companyId"] is None:
        companies = _fetch_companies(trace_id=trace_id)
        best_match_len = 0
        for co in companies:
            stored = str(
                _get_field(co, "companyName", "CompanyName", "name", "Name") or ""
            )
            if not stored:
                continue
            stored_core = re.sub(r"(?:company|coy|unit)", "", stored.lower()).strip()
            for candidate in (stored, stored_core):
                if _is_mention_in_query(query, candidate):
                    candidate_len = len(clean_query(candidate).strip())
                    if candidate_len > best_match_len:
                        cid = _get_field(co, "companyId", "CompanyId", "id", "Id")
                        if cid is not None:
                            result["companyId"] = int(cid)
                            result["companyName"] = stored
                            best_match_len = candidate_len

    if result["platoonId"] is None:
        platoons = _fetch_platoons(trace_id=trace_id)
        best_match_len = 0
        for pl in platoons:
            stored = str(
                _get_field(pl, "platoonName", "PlatoonName", "name", "Name") or ""
            )
            if not stored:
                continue
            stored_core = re.sub(r"(?:platoon|pl)", "", stored.lower()).strip()
            for candidate in (stored, stored_core):
                if _is_mention_in_query(query, candidate):
                    candidate_len = len(clean_query(candidate).strip())
                    if candidate_len > best_match_len:
                        pid = _get_field(pl, "platoonId", "PlatoonId", "id", "Id")
                        cid = _get_field(pl, "companyId", "CompanyId")
                        if pid is not None:
                            if (
                                result["companyId"] is None
                                or cid == result["companyId"]
                            ):
                                result["platoonId"] = int(pid)
                                result["platoonName"] = stored
                                best_match_len = candidate_len

    if result["companyId"] is not None:
        companies = _fetch_companies(trace_id=trace_id)
        for co in companies:
            cid = _get_field(co, "companyId", "CompanyId", "id", "Id")
            if cid is not None and int(cid) == result["companyId"]:
                result["companyName"] = str(
                    _get_field(co, "companyName", "CompanyName", "name", "Name") or ""
                )
                break

    if result["platoonId"] is not None:
        platoons = _fetch_platoons(trace_id=trace_id)
        for pl in platoons:
            pid = _get_field(pl, "platoonId", "PlatoonId", "id", "Id")
            if pid is not None and int(pid) == result["platoonId"]:
                result["platoonName"] = str(
                    _get_field(pl, "platoonName", "PlatoonName", "name", "Name") or ""
                )
                break

    return result
