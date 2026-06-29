"""
query_planner.py
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .admin_intent import classify_admin_intent, format_admin_payload
from query_understanding_engine import understand_query
from utils import build_filters_from_entities

logger = logging.getLogger(__name__)


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", (text or "").lower())).strip()


class QueryType(Enum):
    SIMPLE = "simple"
    MULTI_INDEPENDENT = "multi_independent"
    CROSS_FILTER = "cross_filter"
    COMPARE = "compare"
    TREND = "trend"
    DISTRIBUTION = "distribution"

    # Backward compatibility aliases
    FILTER_QUERY = "simple"
    ANALYTICS = "simple"
    COMPARISON = "compare"
    MULTI_OPERATION = "multi_independent"



@dataclass
class SubOperation:
    raw_fragment: str
    intent_result: Dict[str, Any] = field(default_factory=dict)
    dotnet_payload: Dict[str, Any] = field(default_factory=dict)
    group_by: Optional[str] = None
    filter_fragment: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "rawFragment": self.raw_fragment,
            "intentResult": self.intent_result,
            "dotnetPayload": self.dotnet_payload,
        }
        if self.group_by:
            d["groupBy"] = self.group_by
        if self.filter_fragment:
            d["filterFragment"] = self.filter_fragment
        return d


@dataclass
class QueryPlan:
    query_type: QueryType
    operations: List[SubOperation]
    confidence: float
    raw_query: str
    reasoning: str
    filters: Dict[str, Any] = field(default_factory=dict)
    analytics_hint: Optional[str] = None
    comparison_execution_plan: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "queryType": self.query_type.value,
            "confidence": round(self.confidence, 2),
            "operationCount": len(self.operations),
            "reasoning": self.reasoning,
            "operations": [op.to_dict() for op in self.operations],
            "filters": self.filters,
        }
        if self.analytics_hint:
            d["analyticsHint"] = self.analytics_hint
        if self.comparison_execution_plan:
            d["comparisonExecutionPlan"] = self.comparison_execution_plan
        return d



_COMPARISON_KEYWORDS: List[str] = [
    "compare",
    "comparison",
    " vs ",
    "versus",
    "compared to",
    "compared with",
    "difference between",
    "contrast",
]

_CROSS_FILTER_KEYWORDS: List[str] = [
    "who plays",
    "who play",
    "that play",
    "that plays",
    "which play",
    "which plays",
    "having sport",
    "with sport",
    "on leave",
    "currently on leave",
    "who is on leave",
    "currently absent",
    "with medical",
    "with active medical",
    "with medical case",
    "active medical",
    "in hospital",
    "under treatment",
    "attending today",
    "present today",
    "on campus today",
    "who are in",
    "who is in",
    "among",
    "within",
]

_CROSS_FILTER_CONNECTORS: List[str] = [
    "who",
    "that",
    "which",
    "having",
    "among",
    "within",
    "currently",
    "with",
]

_MULTI_INDEPENDENT_CONNECTORS: List[str] = [
    "along with",
    "as well as",
    "together with",
    "additionally show",
    "also show",
    "and also",
]

_NO_SPLIT_PHRASES: List[str] = [
    "approved and pending leave",
    "approved and pending",
    "annual and medical leave",
    "annual and sick leave",
    "medical and sick leave",
    "top and bottom",
    "top and worst",
    "best and worst",
    "highest and lowest",
    "pass and fail",
    "pass percentage and fail percentage",
    "pass rate and fail rate",
    "improvement and drop",
    "improvement and decline",
    "grade distribution",
    "grading distribution",
    "issued and procured",
    "overdue and returned",
    "pending and completed verification",
    "pending and completed",
]

_GROUP_BY_MAP: Dict[str, str] = {
    "section": "section",
    "unit": "unit",
    "class": "class",
    "sport": "sport",
    "sports": "sport",
    "platoon": "platoon",
    "batch": "batch",
}

_CATEGORY_SIGNALS: Dict[str, List[str]] = {
    "Performance": [
        "performance",
        "performer",
        "performers",
        "score",
        "marks",
        "bpet",
        "ppt",
        "firing",
        "drill",
        "grading",
        "grade",
        "top performer",
        "bottom performer",
        "average score",
        "pass percentage",
        "fail percentage",
        "improvement",
        "drop",
        "attempt",
        "section summary",
    ],
    "Leave": [
        "leave",
        "absent",
        "absentee",
        "absconded",
        "awol",
        "annual leave",
        "medical leave",
        "sick leave",
        "on leave",
        "currently on leave",
    ],
    "Medical": [
        "medical",
        "hospital",
        "bmi",
        "disease",
        "diseases",
        "health",
        "admitted",
        "patient",
        "ward",
        "illness",
        "active medical",
        "with medical",
        "fever",
        "injury",
        "injured",
        "sick",
        "ill",
        "cough",
        "cold",
        "infection",
        "fracture",
        "wound",
        "pain",
        "flu",
        "malaria",
        "dengue",
        "typhoid",
        "blood group",
        "blood type",
    ],
    "Attendance": [
        "attendance",
        "present",
        "campus",
        "monthly attendance",
        "present today",
    ],
    "Verification": [
        "verification",
        "verified",
        "pending verification",
        "completed verification",
    ],
    "Equipment": [
        "equipment",
        "gear",
        "overdue",
        "inventory",
        "issued items",
        "procured items",
        "damaged",
    ],
    "Distribution": [
        "distribution",
        "unit",
        "unassigned",
        "distributed",
    ],
    "Skills": [
        "sport",
        "sports",
        "cricket",
        "football",
        "hockey",
        "basketball",
        "volleyball",
        "kabaddi",
        "running",
        "class",
        "sikh",
        "dogra",
        "jat",
        "gurkha",
        "rajput",
        "punjabi",
    ],
    "Roster": [
        "roster",
        "roster by sport",
        "roster by class",
        "roster by community",
        "sports roster",
        "class wise roster",
    ],
    "Strength": [
        "strength",
        "breakdown",
        "strength breakdown",
        "headcount",
        "headcount breakdown",
    ],
    "Overall": [
        "overall",
        "overall performance",
        "composite",
        "allcriteria",
    ],
}

_SPORT_NAMES = {
    "cricket",
    "football",
    "hockey",
    "basketball",
    "volleyball",
    "kabaddi",
    "running",
}


def _detect_categories(text_lower: str) -> List[str]:
    scores: Dict[str, int] = {}
    for category, signals in _CATEGORY_SIGNALS.items():
        score = 0
        for sig in signals:
            if not sig:
                continue
            idx = text_lower.find(sig)
            while idx != -1:
                before_ok = True
                if sig[0].isalnum():
                    before_ok = idx == 0 or not text_lower[idx - 1].isalnum()
                after_ok = True
                if sig[-1].isalnum():
                    after_ok = (
                        idx + len(sig) == len(text_lower)
                        or not text_lower[idx + len(sig)].isalnum()
                    )
                if before_ok and after_ok:
                    score += len(sig.split())
                    break
                idx = text_lower.find(sig, idx + 1)
        if score > 0:
            scores[category] = score
    return sorted(scores.keys(), key=lambda c: scores[c], reverse=True)


def _has_cross_category_signal(text_lower: str, categories: List[str]) -> bool:
    if len(categories) < 2:
        return False
    for phrase in _CROSS_FILTER_KEYWORDS:
        if phrase in text_lower:
            return True
    for connector in _CROSS_FILTER_CONNECTORS:
        if re.search(r"\b" + re.escape(connector) + r"\b", text_lower):
            return True
    return False


def _detect_comparison(
    text_lower: str, categories: List[str]
) -> Optional[Tuple[str, str]]:
    if not any(kw in text_lower for kw in _COMPARISON_KEYWORDS):
        return None
    fragments = _extract_comparison_fragments(text_lower, categories)
    if fragments and len(fragments) >= 2:
        return ("comparison", "comparison")
    sections_found = [
        s for s in {"bpet", "bept", "ppt", "firing", "drill"} if s in text_lower
    ]
    if len(sections_found) >= 2:
        return ("Performance", "Performance")
    if len(categories) >= 2:
        return (categories[0], categories[1])
    return None


def _detect_multi_independent(
    text_lower: str, categories: List[str]
) -> Optional[List[str]]:
    for connector in _MULTI_INDEPENDENT_CONNECTORS:
        if connector in text_lower:
            parts = text_lower.split(connector, 1)
            if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                left_cats = _detect_categories(parts[0])
                right_cats = _detect_categories(parts[1])
                if left_cats and right_cats:
                    return [parts[0].strip(), parts[1].strip()]

    if " and " in text_lower and len(categories) >= 2:
        for m in re.finditer(r"\band\b", text_lower):
            left = text_lower[: m.start()].strip()
            right = text_lower[m.end() :].strip()
            if left and right:
                # The old code was wrong because it did not check for 'suffer', 'suffered', or 'suffering' prefixes, leading to premature query splitting on cross-filter medical condition connectors.
                if re.match(
                    r"^(?:is|are|has|have|had|having|was|were|play|plays|who|which|that|with|on|in|under|currently|suffer|suffered|suffering)\b",
                    right,
                ):
                    continue
                lc = _detect_categories(left)
                rc = _detect_categories(right)
                if lc and rc and lc[0] != rc[0]:
                    return [left, right]
    return None


def _is_no_split_phrase(text_lower: str) -> bool:
    return any(phrase in text_lower for phrase in _NO_SPLIT_PHRASES)


def _extract_group_by(text_lower: str) -> Optional[str]:
    patterns = [
        r"\bby\s+(\w+)\b",
        r"\bper\s+(\w+)\b",
        r"\bgroup(?:ed)?\s+by\s+(\w+)\b",
        r"\b(\w+)[- ]wise\b",
    ]
    for pattern in patterns:
        m = re.search(pattern, text_lower)
        if m:
            candidate = m.group(1).lower()
            if candidate in _GROUP_BY_MAP:
                return _GROUP_BY_MAP[candidate]
    return None


def _enrich_right(right_fragment: str) -> str:
    for sport in _SPORT_NAMES:
        if sport in right_fragment:
            return f"sport {right_fragment}"
    return right_fragment


def _extract_cross_filter_fragments(
    text_lower: str, categories: List[str]
) -> Optional[List[str]]:
    nested_patterns = [
        r"\band is\b",
        r"\band currently\b",
        r"\band also\b",
        r"\band are\b",
        r"\band suffered\b",
        r"\band suffering\b",
        r"\band had\b",
        r"\band has\b",
        r"\band with\b",
        r"\band who\b",
        r"\band plays?\b",
        r"\band on\b",
    ]

    primary_split: Optional[Tuple[str, str, str]] = None

    # 1. Keywords first
    for kw in _CROSS_FILTER_KEYWORDS:
        if kw in text_lower:
            idx = text_lower.index(kw)
            left = text_lower[:idx].strip()
            right = text_lower[idx:].strip()
            if left and right:
                primary_split = (left, right, kw)
                break

    # 2. Connectors second (leftmost preferred)
    if primary_split is None:
        best_match = None
        for connector in _CROSS_FILTER_CONNECTORS:
            m = re.search(r"\b" + re.escape(connector) + r"\b", text_lower)
            if m:
                if best_match is None or m.start() < best_match.start():
                    left = text_lower[: m.start()].strip()
                    right = text_lower[m.start() :].strip()
                    if left and right:
                        lc = _detect_categories(left)
                        rc = _detect_categories(_enrich_right(right))
                        if lc and rc and lc[0] != rc[0]:
                            best_match = m
        if best_match is not None:
            primary_split = (
                text_lower[: best_match.start()].strip(),
                text_lower[best_match.start() :].strip(),
                best_match.group(0),
            )

    # 3. Fallback to nested patterns for primary split if still None (leftmost preferred)
    if primary_split is None:
        best_match = None
        for pat in nested_patterns:
            m = re.search(pat, text_lower)
            if m:
                if best_match is None or m.start() < best_match.start():
                    left = text_lower[: m.start()].strip()
                    right = text_lower[m.end() :].strip()
                    if left and right:
                        lc = _detect_categories(left)
                        rc = _detect_categories(_enrich_right(right))
                        if lc and rc and lc[0] != rc[0]:
                            best_match = m
        if best_match is not None:
            primary_split = (
                text_lower[: best_match.start()].strip(),
                text_lower[best_match.end() :].strip(),
                best_match.group(0),
            )

    if primary_split is None:
        return None

    left, right, matched_conn = primary_split
    fragments: List[str] = [left]

    # Iterative/recursive splits on remainder (right part)
    remainder = right
    while True:
        first_match = None
        for pat in nested_patterns:
            m = re.search(pat, remainder)
            if m:
                if first_match is None or m.start() < first_match.start():
                    first_match = m
        if first_match:
            sub_left = remainder[: first_match.start()].strip()
            sub_right = remainder[first_match.end() :].strip()
            if sub_left:
                fragments.append(_enrich_right(sub_left))
            remainder = sub_right
        else:
            if remainder:
                fragments.append(_enrich_right(remainder))
            break

    return fragments


def _extract_comparison_fragments(
    text_lower: str, categories: List[str]
) -> Optional[List[str]]:
    category_tokens = {cat.lower() for cat in _CATEGORY_SIGNALS.keys()}

    def _normalize_shared_category(left: str, right: str) -> Tuple[str, str]:
        tail_match = re.search(
            r"\b(" + "|".join(re.escape(tok) for tok in sorted(category_tokens)) + r")\b\s*$",
            right,
        )
        if tail_match:
            shared_category = tail_match.group(1)
            right_core = right[: tail_match.start()].strip()
            left_core = left.strip()
            return f"{left_core} {shared_category}".strip(), f"{right_core} {shared_category}".strip()
        return left, right

    for sep in [" vs ", " versus "]:
        if sep in text_lower:
            parts = text_lower.split(sep, 1)
            if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                left, right = parts[0].strip(), parts[1].strip()
                for kw in ["compare", "comparison"]:
                    left = re.sub(r"^" + kw + r"\s+", "", left).strip()
                return list(_normalize_shared_category(left, right))

    m = re.search(r"\bcompare\s+(.+?)\s+and\s+(.+)", text_lower)
    if m:
        return list(_normalize_shared_category(m.group(1).strip(), m.group(2).strip()))

    m = re.search(r"(.+?)\s+compared\s+(?:to|with)\s+(.+)", text_lower)
    if m:
        return list(_normalize_shared_category(m.group(1).strip(), m.group(2).strip()))

    m = re.search(r"(?:difference between|difference\s+between)\s+(.+?)\s+and\s+(.+)", text_lower)
    if m:
        return list(_normalize_shared_category(m.group(1).strip(), m.group(2).strip()))

    return None


def _extract_filtered_comparison_fragments(
    text_lower: str,
) -> Optional[List[str]]:
    filter_m = re.search(r"\b(?:among|for|with|within|by)\s+(.+)$", text_lower)
    if filter_m is None:
        return None

    filter_text = filter_m.group(1).strip()
    filter_text = re.sub(
        r"\b(?:players?|person|agniveers?|trainees?)\b", "", filter_text
    ).strip()
    if not filter_text:
        return None

    body = text_lower[: filter_m.start()].strip()

    comparison_frags = _extract_comparison_fragments(body, [])
    if comparison_frags and len(comparison_frags) == 2:
        return [
            f"{comparison_frags[0]} {filter_text}".strip(),
            f"{comparison_frags[1]} {filter_text}".strip(),
        ]

    m = re.search(r"\bcompare\s+(.+?)\s+and\s+(.+)", body)
    if m:
        return [
            f"{m.group(1).strip()} {filter_text}".strip(),
            f"{m.group(2).strip()} {filter_text}".strip(),
        ]

    return None


def _build_sub_operation(
    fragment: str,
    group_by: Optional[str] = None,
    filter_fragment: Optional[str] = None,
) -> SubOperation:
    intent_result = classify_admin_intent(fragment)
    dotnet_payload = format_admin_payload(intent_result)
    return SubOperation(
        raw_fragment=fragment,
        intent_result=intent_result,
        dotnet_payload=dotnet_payload,
        group_by=group_by,
        filter_fragment=filter_fragment,
    )

def _is_semantic_comparison(text_lower: str, categories: List[str], semantic: Dict[str, Any]) -> bool:
    # Direct keywords
    if any(kw in text_lower for kw in _COMPARISON_KEYWORDS):
        return True
    
    # Semantic query type from understanding engine
    if semantic and (semantic.get("query_type") in ("compare", "comparison") or semantic.get("complexity") == "comparison"):
        return True

    # Adjectives / comparative words
    comparatives = ["better", "worse", "higher", "lower", "faster", "slower", "more", "less", "compare", "comparison", "difference", "vs", "versus"]
    if any(re.search(r"\b" + re.escape(comp) + r"\b", text_lower) for comp in comparatives):
        return True

    # Multiple sections
    sections_found = {s for s in {"bpet", "bept", "ppt", "firing", "drill"} if re.search(r"\b" + re.escape(s) + r"\b", text_lower)}
    if len(sections_found) >= 2:
        return True

    # Multiple companies/units
    companies_found = {u for u in {"alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "vanguard"} if re.search(r"\b" + re.escape(u) + r"\b", text_lower)}
    if len(companies_found) >= 2:
        return True

    # Multiple platoons
    platoons_found = set(re.findall(r"\bplatoon\s*\d+\b|\bpl\s*\d+\b|\b\d+\s*platoon\b", text_lower))
    if len(platoons_found) >= 2:
        return True

    # Multiple batches
    batches_found = set(re.findall(r"\bbatch\s*[a-z0-9]+\b", text_lower))
    if len(batches_found) >= 2:
        return True

    # Multiple sports
    from .intent_schema import SPORTS
    sports_found = {s for s in SPORTS if re.search(r"\b" + re.escape(s.lower()) + r"\b", text_lower)}
    if len(sports_found) >= 2:
        return True

    # Multiple months
    months = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    months_found = {m for m in months if re.search(r"\b" + re.escape(m) + r"\b", text_lower)}
    if len(months_found) >= 2:
        return True

    # Multiple years
    years_found = set(re.findall(r"\b(19\d{2}|20\d{2})\b", text_lower))
    if len(years_found) >= 2:
        return True

    return False


def _normalize_n_parts(parts: List[str]) -> List[Tuple[str, str]]:
    cleaned_parts = []
    for part in parts:
        p = part.strip()
        for kw in ("compare", "comparison", "difference", "versus", "vs"):
            p = re.sub(r"\b" + re.escape(kw) + r"\b", "", p, flags=re.IGNORECASE).strip()
        p = re.sub(r"\b(results|stats|data|records|performance|score|marks)\b.*$", "", p, flags=re.IGNORECASE).strip()
        p = re.sub(r"\bfor\s+\w+(?:\s+\d+)?$", "", p, flags=re.IGNORECASE).strip()
        cleaned_parts.append(p)
    
    # Check if there is a shared trailing category/keyword in the last part
    last_part = cleaned_parts[-1]
    last_tokens = last_part.split()
    if len(last_tokens) > 1:
        last_word = last_tokens[-1]
        word_lower = last_word.lower()
        category_tokens = {cat.lower() for cat in _CATEGORY_SIGNALS.keys()}
        common_nouns = {"attendance", "performance", "records", "status", "cases", "score", "marks", "grade", "stats", "cases"}
        if word_lower in category_tokens or word_lower in common_nouns:
            new_parts = []
            for i in range(len(cleaned_parts) - 1):
                part = cleaned_parts[i]
                if not part.lower().endswith(word_lower):
                    new_parts.append(f"{part} {last_word}")
                else:
                    new_parts.append(part)
            new_parts.append(last_part)
            cleaned_parts = new_parts

    return [(p, p) for p in cleaned_parts if p]


def _extract_comparison_components(query_text: str) -> List[Tuple[str, str]]:
    text_lower = query_text.lower().strip()
    for sep in (" vs ", " versus "):
        if sep in text_lower:
            temp_text = query_text
            temp_lower = text_lower
            for prefix in ("compare ", "comparison of ", "comparison between "):
                if temp_lower.startswith(prefix):
                    temp_text = temp_text[len(prefix):].strip()
                    temp_lower = temp_text.lower().strip()
            parts = re.split(re.escape(sep), temp_text, flags=re.IGNORECASE)
            return _normalize_n_parts(parts)

    diff_match = re.search(r"difference\s+between\s+(.+?)\s+and\s+(.+)", query_text, re.IGNORECASE)
    if diff_match:
        parts = [diff_match.group(1).strip(), diff_match.group(2).strip()]
        return _normalize_n_parts(parts)

    from .intent_schema import SPORTS
    
    def find_matches(pattern_list, text):
        found = []
        for pat in pattern_list:
            for m in re.finditer(r"\b" + re.escape(pat) + r"\b", text, re.IGNORECASE):
                found.append((m.start(), m.end(), pat))
        found.sort()
        return found

    def split_on_matches(text, matches):
        prefix = text[:matches[0][0]].strip()
        # Strip comparison lead words regardless of trailing whitespace
        prefix = re.sub(
            r"^(?:compare|comparison\s+of|comparison\s+between|comparison)\b\s*",
            "",
            prefix,
            flags=re.IGNORECASE,
        ).strip()
        prefix_lower = prefix.lower()

        prefix = re.sub(r"\b(and|or)\b\s*$", "", prefix, flags=re.IGNORECASE).strip()
        
        suffix = text[matches[-1][1]:].strip()
        suffix = re.sub(r"^\s*\b(and|or)\b", "", suffix, flags=re.IGNORECASE).strip()
        
        components = []
        for start, end, val in matches:
            label = text[start:end]
            frag_parts = []
            if prefix:
                frag_parts.append(prefix)
            frag_parts.append(text[start:end])
            if suffix:
                frag_parts.append(suffix)
            frag = " ".join(frag_parts)
            components.append((label, frag))
        return components

    # 1. Sections
    sections = ["bpet", "bept", "ppt", "firing", "drill"]
    sec_matches = find_matches(sections, text_lower)
    if len({m[2] for m in sec_matches}) >= 2:
        return split_on_matches(query_text, sec_matches)

    # 2. Company/Units
    companies = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "vanguard"]
    coy_matches = find_matches(companies, text_lower)
    if len({m[2] for m in coy_matches}) >= 2:
        return split_on_matches(query_text, coy_matches)

    # 3. Platoons
    platoon_matches = [(m.start(), m.end(), m.group(0)) for m in re.finditer(r"\bplatoon\s*\d+\b|\bpl\s*\d+\b|\b\d+\s*platoon\b", text_lower)]
    if len({m[2] for m in platoon_matches}) >= 2:
        return split_on_matches(query_text, platoon_matches)

    # 4. Batches
    batch_matches = [(m.start(), m.end(), m.group(0)) for m in re.finditer(r"\bbatch\s*[a-z0-9]+\b", text_lower)]
    if len({m[2] for m in batch_matches}) >= 2:
        return split_on_matches(query_text, batch_matches)

    # 5. Sports
    sport_matches = find_matches([s.lower() for s in SPORTS], text_lower)
    if len({m[2] for m in sport_matches}) >= 2:
        return split_on_matches(query_text, sport_matches)

    # 6. Months
    months = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    month_matches = find_matches(months, text_lower)
    if len({m[2] for m in month_matches}) >= 2:
        return split_on_matches(query_text, month_matches)

    # 7. Years
    year_matches = [(m.start(), m.end(), m.group(0)) for m in re.finditer(r"\b(19\d{2}|20\d{2})\b", text_lower)]
    if len({m[2] for m in year_matches}) >= 2:
        return split_on_matches(query_text, year_matches)

    # Fallback to simple split by "and"
    if " and " in text_lower:
        parts = re.split(r"\s+and\s+", query_text, flags=re.IGNORECASE)
        return _normalize_n_parts(parts)

    return [(query_text, query_text)]


def plan_query(query: str, semantic: Optional[Dict[str, Any]] = None) -> QueryPlan:
    raw_query = (query or "").strip()
    q = _normalise(raw_query)
    if semantic is None:
        semantic = understand_query(raw_query)

    if not q:
        return QueryPlan(QueryType.SIMPLE, [], 0.0, raw_query, "Empty query")

    if semantic.get("conversational"):
        op = _build_sub_operation(q)
        return QueryPlan(
            QueryType.SIMPLE,
            [op],
            0.0,
            raw_query,
            "Conversational query detected",
            filters={},
        )

    categories = _detect_categories(q)
    is_compare = _is_semantic_comparison(q, categories, semantic)
    
    if is_compare:
        components = _extract_comparison_components(raw_query)
        ops = []
        comparison_execution_plan = []
        combined_filters = {}
        for idx, (label, fragment) in enumerate(components):
            op = _build_sub_operation(fragment)
            ops.append(op)
            combined_filters.update(build_filters_from_entities(op.intent_result.get("filters", {})))
            comparison_execution_plan.append({
                "id": f"dataset_{idx + 1}",
                "label": label,
                "intent": op.intent_result,
                "filters": build_filters_from_entities(op.intent_result.get("filters", {})),
                "payloadContext": {
                    "endpoint": "api/AiCommand/execute",
                    "category": op.intent_result.get("category"),
                    "operation": op.intent_result.get("operation")
                }
            })

        if len(ops) >= 2:
            logger.info(
                "plan_query: COMPARE plan | query_type=compare | operation_count=%d | "
                "operations=%s",
                len(ops),
                [
                    {
                        "label": label,
                        "category": ops[i].intent_result.get("category"),
                        "operation": ops[i].intent_result.get("operation"),
                        "section": ops[i].intent_result.get("section"),
                    }
                    for i, (label, _) in enumerate(components)
                ],
            )
            return QueryPlan(
                query_type=QueryType.COMPARE,
                operations=ops,
                confidence=max(float(semantic.get("confidence") or 0.85), 0.85),
                raw_query=raw_query,
                reasoning="Comparison query detected semantically",
                filters=combined_filters,
                comparison_execution_plan=comparison_execution_plan
            )
        else:
            logger.warning(
                "plan_query: comparison detected but could not decompose into "
                ">= 2 independent operations | raw_query=%r | components_found=%d | "
                "falling through to semantic analysis. "
                "Note: 'Compare' operation will NOT be sent to .NET.",
                raw_query,
                len(ops),
            )

    qtype = (semantic.get("query_type") or "simple").strip().lower()

    def _ops_from_semantic_fragments(default_fragment: str) -> List[SubOperation]:
        fragments = semantic.get("sub_requests")
        ops: List[SubOperation] = []
        if isinstance(fragments, list) and fragments:
            for fragment in fragments:
                frag_text = fragment.get("fragment") if isinstance(fragment, dict) else None
                if not frag_text:
                    continue
                ops.append(_build_sub_operation(frag_text))
        if not ops:
            ops = [_build_sub_operation(default_fragment)]
        return ops

    if qtype == "comparison":
        ops = _ops_from_semantic_fragments(q)
        valid_ops = [op for op in ops if op.intent_result.get("category") or op.intent_result.get("section") or op.intent_result.get("sport")]
        if len(valid_ops) >= 2:
            combined_filters = {}
            for op in valid_ops:
                combined_filters.update(build_filters_from_entities(op.intent_result.get("filters", {})))
            logger.info(
                "plan_query: COMPARE plan (semantic fallback) | query_type=compare | "
                "operation_count=%d",
                len(valid_ops),
            )
            return QueryPlan(
                QueryType.COMPARE,
                valid_ops,
                max(float(semantic.get("confidence") or 0.85), 0.85),
                raw_query,
                "Comparison query detected from semantic understanding",
                filters=combined_filters,
            )
        else:
            logger.warning(
                "plan_query: semantic comparison fallback also produced < 2 valid ops "
                "| raw_query=%r | valid_ops=%d | continuing to single-op fallback. "
                "Note: 'Compare' operation will NOT be sent to .NET.",
                raw_query,
                len(valid_ops),
            )


    if qtype == "multi_independent":
        ops = _ops_from_semantic_fragments(q)
        valid_ops = [op for op in ops if op.intent_result.get("category")]
        if len(valid_ops) >= 2:
            combined_filters = {}
            categories = {op.intent_result["category"] for op in valid_ops if op.intent_result.get("category")}
            for op in valid_ops:
                combined_filters.update(build_filters_from_entities(op.intent_result.get("filters", {})))
            return QueryPlan(
                QueryType.MULTI_INDEPENDENT,
                valid_ops,
                max(float(semantic.get("confidence") or 0.8), 0.8),
                raw_query,
                f"Multi-independent semantic query: {', '.join(sorted(categories))}",
                filters=combined_filters,
            )

    if qtype == "cross_filter":
        ops = _ops_from_semantic_fragments(q)
        # Schedule is a standalone category (timetable/agenda), never a filter condition
        valid_ops = [
            op for op in ops
            if op.intent_result.get("category") and op.intent_result.get("category") != "Schedule"
        ]
        if len(valid_ops) >= 2:
            for op in valid_ops:
                if op.intent_result.get("category") == "Roster" and op.intent_result.get("sport"):
                    op.intent_result["category"] = "Skills"
                    op.dotnet_payload = format_admin_payload(op.intent_result)
            combined_filters = {}
            for op in valid_ops:
                combined_filters.update(build_filters_from_entities(op.intent_result.get("filters", {})))
            return QueryPlan(
                QueryType.CROSS_FILTER,
                valid_ops,
                max(float(semantic.get("confidence") or 0.8), 0.8),
                raw_query,
                "Cross-filter semantic query detected",
                filters=combined_filters,
            )

    if qtype == "trend":
        op = _build_sub_operation(q)
        filters = build_filters_from_entities(op.intent_result.get("filters", {}))
        return QueryPlan(
            QueryType.TREND,
            [op],
            max(float(semantic.get("confidence") or 0.85), 0.85),
            raw_query,
            "Trend query detected from semantic understanding",
            filters=filters,
        )

    if qtype == "distribution":
        op = _build_sub_operation(q)
        filters = build_filters_from_entities(op.intent_result.get("filters", {}))
        return QueryPlan(
            QueryType.DISTRIBUTION,
            [op],
            max(float(semantic.get("confidence") or 0.85), 0.85),
            raw_query,
            "Distribution query detected from semantic understanding",
            filters=filters,
        )

    op = _build_sub_operation(q)
    filters = build_filters_from_entities(op.intent_result.get("filters", {}))
    confidence = max(0.3, float(semantic.get("confidence") or 0.0))
    if semantic.get("operation") == "ranking" or semantic.get("query_type") == "ranking":
        return QueryPlan(
            QueryType.ANALYTICS,
            [op],
            max(confidence, 0.75),
            raw_query,
            "Semantic ranking query detected",
            filters=filters,
            analytics_hint="rank",
        )
    return QueryPlan(
        QueryType.SIMPLE,
        [op],
        max(confidence, 0.5 if filters else 0.3),
        raw_query,
        "Single-intent query with filters",
        filters=filters,
    )
