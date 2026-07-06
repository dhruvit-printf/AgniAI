"""Generic entity matching helpers with exact, alias, regex, and fuzzy matching."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # Optional dependency.
    from rapidfuzz import fuzz as _rf_fuzz  # type: ignore
except Exception:  # pragma: no cover
    _rf_fuzz = None

# Shortest query we'll treat as a deliberate name prefix (e.g. "Maha" for
# "Mahadev"). Below this, a bare few letters is too likely to prefix-match
# several unrelated names.
_MIN_PREFIX_LEN = 3

# An alias/identifier value that happens to equal a common interrogative or
# filler word (e.g. a record whose agniveerNo or name field is literally
# "who") must never be used as a matchable alias — otherwise ordinary
# questions like "who is on leave" would false-match against it via the
# boundary check below. This is a data-quality guard, not a query filter.
_ALIAS_STOPWORDS: frozenset = frozenset(
    {
        "who", "whom", "whose", "what", "which",
        "is", "are", "was", "were", "am", "be",
        "a", "an", "the",
        "of", "on", "in", "at", "to", "for", "with", "and", "or",
        "show", "list", "give", "tell", "find",
    }
)


def normalize_text(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _compact(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def _similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    if _rf_fuzz is not None:
        return float(_rf_fuzz.ratio(left, right))
    return SequenceMatcher(None, left, right).ratio() * 100.0


def _value_from_record(record: Dict[str, Any], fields: Sequence[str]) -> Any:
    for field in fields:
        if field in record and record[field] not in (None, "", [], {}):
            return record[field]
    return None


def _aliases_for_record(
    record: Dict[str, Any], label_fields: Sequence[str], alias_fields: Sequence[str]
) -> List[str]:
    aliases: List[str] = []
    for field in list(label_fields) + list(alias_fields):
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            candidate = value.strip()
            if candidate.lower() in _ALIAS_STOPWORDS:
                continue
            aliases.append(candidate)
    return aliases


def match_entity(
    query: str,
    records: Sequence[Dict[str, Any]],
    *,
    value_fields: Sequence[str],
    label_fields: Sequence[str],
    alias_fields: Sequence[str] = (),
    regex_patterns: Sequence[str] = (),
    threshold: float = 85.0,
    alias_builder: Optional[Callable[[Dict[str, Any]], Sequence[str]]] = None,
) -> Dict[str, Any]:
    query = query or ""
    normalized_query = normalize_text(query)
    compact_query = _compact(query)

    scored: List[Tuple[float, Dict[str, Any], str, str]] = []
    for record in records:
        value = _value_from_record(record, value_fields)
        if value in (None, "", [], {}):
            continue

        aliases = _aliases_for_record(record, label_fields, alias_fields)
        if alias_builder is not None:
            aliases.extend([alias for alias in alias_builder(record) if alias])

        best_score = 0.0
        best_alias = ""
        match_type = "fuzzy"

        for alias in aliases:
            alias_text = str(alias).strip()
            if not alias_text:
                continue
            alias_norm = normalize_text(alias_text)
            alias_compact = _compact(alias_text)
            if normalized_query == alias_norm or compact_query == alias_compact:
                best_score = 99.0
                best_alias = alias_text
                match_type = "exact"
                break

            def _has_boundary(pattern: str, query: str) -> bool:
                idx = query.find(pattern)
                if idx == -1:
                    return False
                if idx > 0:
                    if pattern[0].isdigit() and query[idx - 1].isdigit():
                        return False
                    if pattern[0].isalpha() and query[idx - 1].isalnum():
                        return False
                end_idx = idx + len(pattern)
                if end_idx < len(query):
                    if pattern[-1].isdigit() and query[end_idx].isdigit():
                        return False
                    if pattern[-1].isalpha() and query[end_idx].isalnum():
                        return False
                return True

            if alias_norm and (
                _has_boundary(alias_norm, normalized_query)
                or _has_boundary(normalized_query, alias_norm)
            ):
                score = 97.0
                if score > best_score:
                    best_score = score
                    best_alias = alias_text
                    match_type = "alias"
            if alias_compact and (
                _has_boundary(alias_compact, compact_query)
                or _has_boundary(compact_query, alias_compact)
            ):
                score = 96.0
                if score > best_score:
                    best_score = score
                    best_alias = alias_text
                    match_type = "alias"
            if (
                alias_compact
                and compact_query
                and len(compact_query) >= _MIN_PREFIX_LEN
                and (
                    alias_compact.startswith(compact_query)
                    or compact_query.startswith(alias_compact)
                )
            ):
                # A genuine "half the name" prefix (e.g. "Maha" for
                # "Mahadev") has no token boundary to trigger the alias
                # check above, and its fuzzy ratio can fall under threshold
                # for short prefixes of long names. Score it directly off
                # how much of the shorter string the prefix covers so it
                # reliably clears the match threshold.
                shorter = min(len(alias_compact), len(compact_query))
                longer = max(len(alias_compact), len(compact_query))
                score = 90.0 + 5.0 * (shorter / longer)
                if score > best_score:
                    best_score = score
                    best_alias = alias_text
                    match_type = "prefix"
            fuzzy_score = _similarity(normalized_query, alias_norm)
            if fuzzy_score > best_score:
                best_score = fuzzy_score
                best_alias = alias_text
                match_type = "fuzzy"

        for pattern in regex_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                best_score = max(best_score, 98.0)
                best_alias = query
                match_type = "regex"

        if best_score > 0:
            scored.append((best_score, dict(record), best_alias, match_type))

    if not scored:
        return {
            "value": None,
            "confidence": 0.0,
            "matched_text": "",
            "source": "cache",
            "match_type": "",
            "needs_clarification": False,
            "candidates": [],
        }

    scored.sort(
        key=lambda item: (
            item[0],
            len(str(_value_from_record(item[1], value_fields) or "")),
        ),
        reverse=True,
    )
    top_score, top_record, top_alias, top_type = scored[0]
    top_matches = [item for item in scored if abs(item[0] - top_score) < 0.01]

    candidate_list = []
    for score, record, alias, match_type in scored[:5]:
        candidate_list.append(
            {
                "value": _value_from_record(record, value_fields),
                "label": _value_from_record(record, label_fields),
                "confidence": round(score / 100.0, 4),
                "matched_text": alias,
                "match_type": match_type,
            }
        )

    if top_score < threshold:
        return {
            "value": None,
            "confidence": round(top_score / 100.0, 4),
            "matched_text": top_alias,
            "source": "cache",
            "match_type": top_type,
            "needs_clarification": False,
            "candidates": candidate_list,
        }

    if len(top_matches) > 1:
        return {
            "value": None,
            "confidence": round(top_score / 100.0, 4),
            "matched_text": top_alias,
            "source": "cache",
            "match_type": "ambiguous",
            "needs_clarification": True,
            "candidates": candidate_list,
        }

    return {
        "value": _value_from_record(top_record, value_fields),
        "confidence": round(top_score / 100.0, 4),
        "matched_text": top_alias,
        "source": "cache",
        "match_type": top_type,
        "needs_clarification": False,
        "candidates": candidate_list,
        "record": top_record,
    }
