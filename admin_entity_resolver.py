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
from intent_engine.intent_schema import COMPANY_CANONICAL_NAMES

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
    "summary",
    "summaries",
    "grading",
    "grade",
    "grades",
    "count",
    "counts",
    "total",
    "totals",
    "distribution",
    "compare",
    "comparison",
    "trend",
    "trends",
    "improvement",
    "drop",
    "drops",
    "bpet",
    "bept",
    "betp",
    "ppt",
    "firing",
    "drill",
    "topper",
    "toppers",
    "topped",
    "highest",
    "lowest",
    "worst",
    "excellent",
    "good",
    "sat",
    "fail",
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
_AGNIVEER_WORD_RE = re.compile(
    # (?=\w*\d) requires the captured token to contain a digit — without it,
    # the (?:no.|number|#)? prefix being OPTIONAL meant this matched "the
    # next word after 'agniveer'" full stop, so "every Agniveer BELONGING
    # to Batch 3" captured "belonging" as if it were an AgniveerNo.
    r"\bagniveer\s+(?:no\.?|number|#)?\s*(?=\w*\d)(\w{3,10})\b",
    re.IGNORECASE,
)
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
                # Stop at the first noise word walking outward from
                # "company" — it marks the boundary of the actual name.
                # Skipping past it (the old behaviour) let words from an
                # unrelated earlier clause ("show agniveers OF alpha
                # company") get glued onto the name as "agniveers of
                # alpha" instead of just "alpha".
                break
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


def _normalise_company_or_platoon_name(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\b(?:company|coy|platoon|pl|unit)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[\s\-_./]+", " ", text)
    return text.strip()


_NAME_WORD = r"[a-z0-9][a-z0-9\-_./]*"
# Bounded to at most 2 words (the longest real unit alias in UNIT_ALIASES is
# 2 words, e.g. "golf zulu") — the old unbounded `[a-z0-9\s\-_./]*` let the
# capture run all the way back to the start of the sentence for phrasing like
# "show agniveers of alpha company", swallowing "agniveers of" into the
# "company name" as well. Bounding the match keeps any stray noise word
# (like "of") at the very edge of the capture, where _clean_candidate can
# still trim it off.
_COMPANY_NAME_WORDS = rf"(?:{_NAME_WORD}\s+){{0,1}}{_NAME_WORD}"


# CompanyMaster stores some rows as "<Abbr> - <FullName>" (e.g.
# "Lak - Lakhwinder", "Jas - Jaswant") rather than the bare spoken name, so a
# query saying "Lakhwinder company" only ever yields the single token
# "lakhwinder". The live .NET-backed resolve_company_id() already tolerates
# this via substring matching in _name_matches(), but sql_executor.py's own
# SQL-side exact-match lookups (resolve_company_id_from_name, and every
# `LOWER(c.Name) = LOWER(?)` WHERE clause) do not — they need the literal
# stored Name. Canonicalizing here means whichever path ends up consuming
# this return value gets a name that matches either way.
# Alias map itself lives in intent_schema.COMPANY_CANONICAL_NAMES (shared
# with intent_engine/entity_extractor.py) so it's maintained in one place.
_COMPANY_CANONICAL_NAMES = COMPANY_CANONICAL_NAMES


def extract_company_name(text: str) -> Optional[str]:
    q = text.lower().strip()
    if not q:
        return None

    for pattern in (
        re.compile(rf"\bcompany\s+({_COMPANY_NAME_WORDS})\b", re.IGNORECASE),
        re.compile(rf"\b({_COMPANY_NAME_WORDS})\s+company\b", re.IGNORECASE),
        re.compile(rf"\bcoy\s+({_COMPANY_NAME_WORDS})\b", re.IGNORECASE),
        re.compile(rf"\b({_COMPANY_NAME_WORDS})\s+coy\b", re.IGNORECASE),
    ):
        m = pattern.search(q)
        if m:
            candidate = _clean_candidate(_normalise_company_or_platoon_name(m.group(1)))
            if candidate:
                return _COMPANY_CANONICAL_NAMES.get(candidate.lower(), candidate)

    tokens = _COMPANY_TOKEN_RE.findall(q)
    for idx, token in enumerate(tokens):
        if token not in ("company", "coy"):
            continue
        before = [t for t in tokens[max(0, idx - 3) : idx] if t not in _NOISE_WORDS]
        after = [t for t in tokens[idx + 1 : idx + 4] if t not in _NOISE_WORDS]
        if before:
            candidate = _clean_candidate(
                _normalise_company_or_platoon_name(" ".join(before))
            )
            if candidate:
                return _COMPANY_CANONICAL_NAMES.get(candidate.lower(), candidate)
        if after:
            candidate = _clean_candidate(
                _normalise_company_or_platoon_name(" ".join(after))
            )
            if candidate:
                return _COMPANY_CANONICAL_NAMES.get(candidate.lower(), candidate)

    return None


def extract_platoon_name(text: str) -> Optional[str]:
    q = text.lower().strip()
    if not q:
        return None

    for pattern in (
        # Bounded to 2 words for the same reason as _COMPANY_NAME_WORDS above
        # — the old unbounded `[a-z0-9\s\-_./]*` let "Platoon 2 before the
        # current commander" capture "2 before the current commander" whole.
        re.compile(rf"\bplatoon\s+({_COMPANY_NAME_WORDS})\b", re.IGNORECASE),
        re.compile(r"\bpl\s*[- ]?\s*([a-z0-9][a-z0-9\-./]*)\b", re.IGNORECASE),
    ):
        m = pattern.search(q)
        if m:
            candidate = _clean_candidate(_normalise_company_or_platoon_name(m.group(1)))
            if candidate:
                return candidate

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


# Words that legitimately appear capitalised (sentence-start, or as a proper
# category name in Title Case) but are never part of a person's name — used
# to reject false-positive "name" matches like "Show Agniveers" or "Which
# Platoon". Deliberately conservative: better to miss an unusual name than
# to send a random 2-word phrase into a FullName LIKE lookup.
_AGNIVEER_NAME_STOPWORDS = frozenset(_NOISE_WORDS) | {
    "agniveer",
    "agniveers",
    "police",
    "medical",
    "equipment",
    "verification",
    "district",
    "state",
    "village",
    "tehsil",
    "which",
    "who",
    "what",
    "when",
    "where",
    "why",
    "how",
    "compare",
    "list",
    "give",
    "show",
    "does",
    "did",
    # Org-hierarchy nouns — "Lakhwinder Company", "Current Commander" are
    # unit/role references, not a person's own name, even though both
    # words are capitalised.
    "company",
    "companies",
    "coy",
    "platoon",
    "battalion",
    "commander",
    "commanding",
    "officer",
    "current",
    "user",
    "users",
    "role",
    "admin",
}

_AGNIVEER_NAME_RE = re.compile(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)+\b")


def extract_agniveer_name_mention(text: str) -> Optional[str]:
    """Find a person-name-like phrase — 2+ consecutive Title-Case words
    (e.g. "Harminder Singh", "Aditya Kanwar") — for when the user identifies
    an Agniveer by name instead of by AgniveerNo. Only called as a fallback
    when no AgniveerNo pattern was found (see extract_agniveer_mention)."""
    for candidate in _AGNIVEER_NAME_RE.findall(text or ""):
        words = candidate.split()
        if any(w.lower() in _AGNIVEER_NAME_STOPWORDS for w in words):
            continue
        return candidate
    return None


def resolve_agniveer_nos_by_name(name: str) -> List[Dict[str, str]]:
    """Look up every Agniveer whose FullName matches `name` (substring,
    case-insensitive). Returns [] on no match, and — deliberately — every
    match when more than one Agniveer shares the name, so the caller can
    decide how to fan the query out rather than silently picking one.
    """
    if not name:
        return []
    from sql_executor import run_readonly

    rows, err = run_readonly(
        "SELECT AgniveerNo, FullName FROM AgniveerMaster "
        "WHERE ISNULL(IsDisqualified,0) = 0 AND LOWER(FullName) LIKE '%' + LOWER(?) + '%'",
        [name],
    )
    if err or not rows:
        return []
    return [
        {"agniveerNo": r["AgniveerNo"], "fullName": r["FullName"]}
        for r in rows
        if r.get("AgniveerNo")
    ]


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


def _max_typo_distance(word_len: int) -> int:
    """Slightly more permissive than query_normalizer's general vocabulary
    corrector (1 edit for words <=5 chars): company/platoon names are matched
    against a small closed roster fetched from the .NET directory, not the
    whole English vocabulary, so a second edit on longer names is safe from
    false-positive collisions between unrelated real names."""
    return 1 if word_len <= 4 else 2


def _fuzzy_token_match(word: str, tokens: "Any") -> bool:
    """True if `word` is a plausible misspelling of one of `tokens`
    (e.g. "alfa"/"bravoo" for "Alpha"/"Bravo"). Guarded so ordinary English
    query words never get mistaken for a mistyped name:
      - both sides must be >= 4 chars (numeric IDs like platoon "101" are
        excluded, so distinct short numbers never fuzzy-collide)
      - the word must not already be a recognized application keyword
        (e.g. "compare", "schedule") — those are real words, not typos.
    """
    word = (word or "").lower()
    if len(word) < 4 or word in _NOISE_WORDS:
        return False
    try:
        from intent_engine.intent_classifier import _damerau_levenshtein
        from intent_engine.vocabulary_manager import vocab_manager

        if word in vocab_manager.get_domain_vocab():
            return False
    except Exception:
        return False
    max_dist = _max_typo_distance(len(word))
    for token in tokens:
        token = (token or "").lower()
        if len(token) < 4:
            continue
        if abs(len(word) - len(token)) > max_dist:
            continue
        if _damerau_levenshtein(word, token) <= max_dist:
            return True
    return False


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

    # Typo tolerance: nothing exact/substring/suffix/digit matched, so check
    # whether the query name is a plausible misspelling of one of the
    # stored name's words (e.g. "alfa"/"bravoo" for "Alpha"/"Bravo").
    sn_tokens = re.findall(r"[a-z0-9]+", sn)
    qn_tokens = re.findall(r"[a-z0-9]+", qn) or [qn]
    for qt in qn_tokens:
        if _fuzzy_token_match(qt, sn_tokens):
            return True

    return False


def resolve_company_id(
    company_name: str,
    trace_id: Optional[str] = None,
    session_id: Optional[str] = None,
    *,
    companies: Optional[List[Dict]] = None,
) -> Optional[int]:
    if companies is None:
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
    *,
    platoons: Optional[List[Dict]] = None,
) -> Optional[int]:
    if platoons is None:
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
        # No directory match — usually because the platoon-fetch call
        # itself failed/timed out (seen in practice: "Entity fetch failed
        # for platoons: ... Read timed out"), not because the platoon
        # doesn't exist. A bare numeric mention ("Platoon 2") is
        # unambiguous enough to use directly, mirroring resolve_batch_id's
        # existing same-shaped fallback below.
        if re.match(r"^\d+$", platoon_name.strip()):
            return int(platoon_name.strip())
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
        "companyId": None,
        "platoonId": None,
        "batchId": None,
        "agniveerNo": None,
        "companyName": None,
        "platoonName": None,
        "batchName": None,
    }

    company_mention = extract_company_mention(query)
    platoon_mention = extract_platoon_mention(query)
    company_name_mention = extract_company_name(query)
    platoon_name_mention = extract_platoon_name(query)
    batch_mention = extract_batch_mention(query)
    agniveer_mention = extract_agniveer_mention(query)

    result["companyName"] = company_mention
    result["platoonName"] = platoon_mention
    result["batchName"] = batch_mention
    result["agniveerNo"] = agniveer_mention
    result["agniveerMatches"] = []

    # No AgniveerNo pattern (e.g. "A0701749H") in the query — try resolving
    # by person NAME instead (e.g. "Who is Harminder Singh...", "What is
    # Aditya Kanwar's date of birth"). Unlike company/platoon, Agniveers
    # aren't a small enough set to prefetch into a directory, so this goes
    # straight to the DB. If more than one Agniveer shares the name, every
    # match is returned via agniveerMatches — the caller decides how to fan
    # the query out across them, rather than this silently guessing one.
    if not agniveer_mention:
        name_mention = extract_agniveer_name_mention(query)
        if name_mention:
            matches = resolve_agniveer_nos_by_name(name_mention)
            result["agniveerMatches"] = matches
            if len(matches) == 1:
                result["agniveerNo"] = matches[0]["agniveerNo"]

    # Fetched once and reused for every lookup below (resolve_*_id calls,
    # the authoritative directory scan, and the name backfill) instead of
    # each one re-fetching the same directory independently.
    # Fetch companies and platoons in parallel so a slow/failing .NET tunnel
    # only blocks for _TIMEOUT seconds ONCE, not twice sequentially.
    from concurrent.futures import ThreadPoolExecutor, as_completed

    companies: List[Dict[str, Any]] = []
    platoons: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=2) as _pool:
        _f_co = _pool.submit(_fetch_companies, trace_id=trace_id)
        _f_pl = _pool.submit(_fetch_platoons, trace_id=trace_id)
        try:
            companies = _f_co.result(timeout=5)
        except Exception:
            companies = []
        try:
            platoons = _f_pl.result(timeout=5)
        except Exception:
            platoons = []

    # `extract_company_mention`/`extract_platoon_mention` are a positional
    # heuristic (the word(s) next to "company"/"platoon") — they can pick up
    # stray query vocabulary as part of the name (see _NOISE_WORDS) and they
    # only fire when that keyword is actually present in the text. The .NET
    # company/platoon directory (fetched below) is the authoritative source
    # of real names — matching the query against it directly is checked
    # first and, when it finds anything, wins over the positional guess.
    if company_mention:
        result["companyId"] = resolve_company_id(
            company_mention,
            trace_id=trace_id,
            session_id=session_id,
            companies=companies,
        )
    if company_name_mention:
        if result["companyId"] is None:
            result["companyId"] = resolve_company_id(
                company_name_mention,
                trace_id=trace_id,
                session_id=session_id,
                companies=companies,
            )
        result["companyName"] = company_name_mention
    elif result["companyId"] is None:
        result["companyId"] = existing_company_id

    if platoon_mention:
        result["platoonId"] = resolve_platoon_id(
            platoon_mention,
            company_id=result["companyId"],
            trace_id=trace_id,
            session_id=session_id,
            platoons=platoons,
        )
    if platoon_name_mention:
        if result["platoonId"] is None:
            result["platoonId"] = resolve_platoon_id(
                platoon_name_mention,
                company_id=result["companyId"],
                trace_id=trace_id,
                session_id=session_id,
                platoons=platoons,
            )
        result["platoonName"] = platoon_name_mention
    elif result["platoonId"] is None:
        result["platoonId"] = existing_platoon_id

    if batch_mention:
        result["batchId"] = resolve_batch_id(
            batch_mention, trace_id=trace_id, session_id=session_id
        )
    elif result["batchId"] is None:
        result["batchId"] = existing_batch_id
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

    _MIN_PARTIAL_MENTION_LEN = 3

    def _matching_word(query_text: str, candidate_name: str) -> Optional[str]:
        """Return the query word that partially or fuzzily matches this
        candidate name, or None. A truthy return drives real matching at the
        call site below (`if full_hit or partial_word:`), not just logging.
        """
        cand_compact = re.sub(r"[^a-z0-9]", "", candidate_name.lower())
        if not cand_compact or len(cand_compact) < _MIN_PARTIAL_MENTION_LEN:
            return None
        query_words = _COMPANY_TOKEN_RE.findall(query_text.lower())
        for word in query_words:
            if len(word) < _MIN_PARTIAL_MENTION_LEN or word in _NOISE_WORDS:
                continue
            if (
                cand_compact.startswith(word)
                or word.startswith(cand_compact)
                or word in cand_compact
            ):
                return word
        # Typo tolerance: no exact/prefix hit, so check whether some query
        # word is a plausible misspelling of one of the candidate's words
        # (e.g. "alfa"/"bravoo" for "Alpha"/"Bravo").
        cand_tokens = re.findall(r"[a-z0-9]+", candidate_name.lower())
        for word in query_words:
            if _fuzzy_token_match(word, cand_tokens):
                return word
        return None

    def _is_partial_prefix_mention(query_text: str, candidate_name: str) -> bool:
        """True if some word in the query is a genuine partial match of
        candidate_name — catches a half-typed name like "Maha" for
        "Mahadev" (prefix), or one word of a multi-word name like
        "Lakhwinder" for "Lak - Lakhwinder" (embedded, not just a prefix).
        `_is_mention_in_query` only checks whether the FULL candidate name
        occurs inside the query, so it never fires when the user typed only
        part of it.
        """
        cand_compact = re.sub(r"[^a-z0-9]", "", candidate_name.lower())
        if not cand_compact or len(cand_compact) < _MIN_PARTIAL_MENTION_LEN:
            return False
        for word in _COMPANY_TOKEN_RE.findall(query_text.lower()):
            if len(word) < _MIN_PARTIAL_MENTION_LEN or word in _NOISE_WORDS:
                continue
            if (
                cand_compact.startswith(word)
                or word.startswith(cand_compact)
                or word in cand_compact
            ):
                return True
        return False

    # Authoritative lookup: scan the real .NET company directory directly
    # for a name match, regardless of whether the positional heuristic above
    # found anything. This always runs — and wins when it finds a match —
    # because it can only ever match a name that genuinely exists, whereas
    # the heuristic above can misfire on stray words near "company"/"platoon".
    best_match_len = 0
    directory_company_id = None
    for co in companies:
        stored = str(_get_field(co, "companyName", "CompanyName", "name", "Name") or "")
        if not stored:
            continue
        stored_core = re.sub(r"(?:company|coy|unit)", "", stored.lower()).strip()
        for candidate in (stored, stored_core):
            full_hit = _is_mention_in_query(query, candidate)
            partial_word = None if full_hit else _matching_word(query, candidate)
            if full_hit or partial_word:
                candidate_len = len(clean_query(candidate).strip())
                if candidate_len > best_match_len:
                    cid = _get_field(co, "companyId", "CompanyId", "id", "Id")
                    if cid is not None:
                        if not company_mention:
                            logger.info(
                                "resolve_entities_from_query: silent companyId match "
                                "(no 'company'/'coy' keyword in query) | query=%r | "
                                "matched_company=%r | company_id=%s | "
                                "match_type=%s | matched_word=%r",
                                query,
                                stored,
                                cid,
                                "full_name" if full_hit else "partial_prefix",
                                partial_word,
                            )
                        directory_company_id = int(cid)
                        result["companyName"] = stored
                        best_match_len = candidate_len
    if directory_company_id is not None:
        result["companyId"] = directory_company_id

    best_match_len = 0
    directory_platoon_id = None
    for pl in platoons:
        stored = str(_get_field(pl, "platoonName", "PlatoonName", "name", "Name") or "")
        if not stored:
            continue
        stored_core = re.sub(r"(?:platoon|pl)", "", stored.lower()).strip()
        for candidate in (stored, stored_core):
            full_hit = _is_mention_in_query(query, candidate)
            partial_word = None if full_hit else _matching_word(query, candidate)
            if full_hit or partial_word:
                candidate_len = len(clean_query(candidate).strip())
                if candidate_len > best_match_len:
                    pid = _get_field(pl, "platoonId", "PlatoonId", "id", "Id")
                    cid = _get_field(pl, "companyId", "CompanyId")
                    if pid is not None and (
                        result["companyId"] is None or cid == result["companyId"]
                    ):
                        if not platoon_mention:
                            logger.info(
                                "resolve_entities_from_query: silent platoonId match "
                                "(no 'platoon' keyword in query) | query=%r | "
                                "matched_platoon=%r | platoon_id=%s | "
                                "match_type=%s | matched_word=%r",
                                query,
                                stored,
                                pid,
                                "full_name" if full_hit else "partial_prefix",
                                partial_word,
                            )
                        directory_platoon_id = int(pid)
                        result["platoonName"] = stored
                        best_match_len = candidate_len
    if directory_platoon_id is not None:
        result["platoonId"] = directory_platoon_id

    if result["companyId"] is not None:
        for co in companies:
            cid = _get_field(co, "companyId", "CompanyId", "id", "Id")
            if cid is not None and int(cid) == result["companyId"]:
                result["companyName"] = str(
                    _get_field(co, "companyName", "CompanyName", "name", "Name") or ""
                )
                break

    if result["platoonId"] is not None:
        for pl in platoons:
            pid = _get_field(pl, "platoonId", "PlatoonId", "id", "Id")
            if pid is not None and int(pid) == result["platoonId"]:
                result["platoonName"] = str(
                    _get_field(pl, "platoonName", "PlatoonName", "name", "Name") or ""
                )
                break

    return result
