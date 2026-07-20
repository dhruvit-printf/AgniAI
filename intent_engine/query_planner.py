"""
query_planner.py
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from query_understanding_engine import (
    _COMPARISON_MARKERS as _COMPARISON_KEYWORDS,
    propagate_lead_in_across_parts,
    understand_query,
)
from utils import build_filters_from_entities

from .admin_intent import classify_admin_intent, format_admin_payload
from .entity_extractor import detect_query_number_override
from .intent_classifier import detect_query_response_type_override
from .intent_schema import UNIT_ALIASES

logger = logging.getLogger(__name__)

# Canonical section/unit vocabulary shared by _is_semantic_comparison and
# _extract_comparison_components (previously two independently hand-typed,
# drifting copies each — one of which was stale: missing golf-zulu and
# including a non-canonical "vanguard" entry).
_COMPARISON_SECTIONS: Tuple[str, ...] = ("bpet", "bept", "ppt", "firing", "drill")
_COMPARISON_UNITS: Tuple[str, ...] = tuple(UNIT_ALIASES.keys())


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", (text or "").lower())).strip()


def _clamp_confidence(value: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def _should_treat_cross_filter_as_multi_independent(semantic: Dict[str, Any]) -> bool:
    """Promote same-subject, multi-section requests and independent
    multi-category report requests to multi-independent.

    Queries like "show attendance and current leave records for agniveer 12345"
    are not intersections between unrelated filters. They are two independent
    facts requested for the same agniveer, so the pipeline should keep them as
    separate sections instead of forcing cross-filter execution.
    Likewise, unit-wide requests for distinct categories with no shared entity
    to intersect on should also be multi-independent.
    """
    if not semantic or semantic.get("dependent_intent"):
        return False

    sub_requests = semantic.get("sub_requests")
    if not isinstance(sub_requests, list) or len(sub_requests) < 2:
        return False

    categories = set()
    shared_agniveer_no = None
    has_any_agniveer_no = False
    all_have_agniveer_no = True
    has_any_discriminating_filter = False

    for sub_request in sub_requests:
        if not isinstance(sub_request, dict):
            return False

        category = sub_request.get("category")
        if not category:
            return False
        categories.add(category)

        entities = sub_request.get("entities")
        if not isinstance(entities, dict):
            return False

        discriminating = {
            k: v
            for k, v in entities.items()
            if k
            not in (
                "category",
                "operation",
                "responseType",
                "agniveerNo",
                "agniveer_no",
            )
            and v not in (None, "", [], {})
        }
        if discriminating:
            has_any_discriminating_filter = True

        agniveer_no = entities.get("agniveerNo") or entities.get("agniveer_no")
        if agniveer_no in (None, "", [], {}):
            all_have_agniveer_no = False
            continue

        agniveer_no = str(agniveer_no).strip()
        if not agniveer_no:
            all_have_agniveer_no = False
            continue

        has_any_agniveer_no = True
        if shared_agniveer_no is None:
            shared_agniveer_no = agniveer_no
        elif agniveer_no != shared_agniveer_no:
            return False

    # Case 1: Every sub-request has the exact same non-empty agniveer_no (the original logic)
    if all_have_agniveer_no and len(categories) >= 2 and shared_agniveer_no is not None:
        return True

    # Case 2: No agniveer_no exists in any leg, but categories are distinct.
    if (
        not has_any_agniveer_no
        and len(categories) == len(sub_requests)
        and len(categories) >= 2
        and not has_any_discriminating_filter
    ):
        return True

    return False


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


_GROUPING_OPERATIONS = frozenset({"ByName", "ByUnit", "BySport", "ByClass"})


def _is_leftover_subject_op(op: "SubOperation") -> bool:
    """True for a bare-subject fragment left over from splitting a compound
    query (e.g. "show agniveers" split off "...who failed Firing and still
    have issued equipment"). Such a fragment carries no real entity, so it
    lands either on the literal "AgniveerWise" operation or, via the
    low-confidence semantic fallback, on a "By*" grouping operation
    (ByClass/BySport/ByUnit/ByName) that is structurally meaningless without
    its target entity (e.g. "Skills/ByClass" with no class value). Only
    "By*" operations are checked for a missing entity — other operations
    (e.g. "Medical/Individual" from "suffered from fever") are legitimate
    even when the entity extractor didn't capture a distinct filter value,
    so they must not be swept up by this check.
    """
    operation = op.intent_result.get("operation")
    if operation == "AgniveerWise":
        return True
    if operation not in _GROUPING_OPERATIONS:
        return False
    filters = op.intent_result.get("filters") or {}
    discriminating = {
        key: value
        for key, value in filters.items()
        if key != "operation" and value not in (None, "", [], {})
    }
    return not discriminating


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
        "top",
        "best",
        "worst",
        "bottom",
        "highest",
        "lowest",
        "rank",
        "top performer",
        "bottom performer",
        "average score",
        "improvement",
        "drop",
        "attempt",
    ],
    "Leave": [
        "leave",
        "leaves",
        "absent",
        "absentee",
        "absconded",
        "awol",
        "away",
        "missing",
        "unaccounted",
        "untraceable",
        "annual leave",
        "medical leave",
        "sick leave",
        "on leave",
        "currently on leave",
        "away right now",
        "who is away",
        "absent right now",
        "out today",
        "out right now",
    ],
    "Medical": [
        "medical",
        "medically",
        "hospital",
        "bmi",
        "disease",
        "diseases",
        "health",
        "admitted",
        "patient",
        "ward",
        "illness",
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
        "blood groups",
        "blood type",
        "o+",
        "o-",
        "a+",
        "a-",
        "b+",
        "b-",
        "ab+",
        "ab-",
        "overweight",
        "underweight",
        "obese",
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
        "verifications",
        "verified",
        "pending verification",
        "pending verifications",
        "completed verification",
        "completed verifications",
        "rejected verification",
        "rejected verifications",
        "sent verification",
        "sent verifications",
        "not responded",
        "rejected",
        "unverified",
    ],
    "Equipment": [
        "equipment",
        "gear",
        "search",
        "find equipment",
        "lookup equipment",
        "inventory",
        "damaged",
        "returned",
        "holding",
        "held",
        "overdue",
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
        "skill",
        "who plays",
        "player",
        "players",
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
    "Schedule": [
        "schedule",
        "training",
        "company schedule",
        "today schedule",
        "today's schedule",
        "current schedule",
    ],
    "Overall": [
        "overall",
        "overall performance",
        "composite",
        "allcriteria",
    ],
    "personaldetail": [
        "personal detail",
        "personal details",
        "personaldetail",
        "personal info",
        "profile",
        "profiles",
        "biodata",
        "bio data",
        "contact",
        "education",
        "qualification",
        "family details",
        "next of kin",
    ],
    "disqualified": [
        "disqualified",
        "disqualification",
        "disqualified agniveer",
        "disqualified agniveers",
        "removed agniveer",
        "removed agniveers",
        "expelled agniveer",
        "expelled agniveers",
    ],
}


def _extend_skills_category_signals() -> None:
    """Extend _CATEGORY_SIGNALS["Skills"] with every sport name so that
    mentioning ANY sport (not just the half-dozen hardcoded here) registers
    as a Skills-category signal for cross-filter's ">= 2 categories" gate —
    e.g. "who got excellent in firing among badminton players" needs both
    Performance and Skills detected to split at all.
    """
    from .intent_schema import SPORTS

    base = _CATEGORY_SIGNALS["Skills"]
    additions = [name for name in SPORTS.keys() if name not in base]
    _CATEGORY_SIGNALS["Skills"] = base + additions


def _extend_equipment_category_signals() -> None:
    """Extend _CATEGORY_SIGNALS["Equipment"] with every named equipment item
    (Combat Coat, Kit Bag, Blanket, ...), mirroring the sports extension
    above — a query naming a specific item ("Kit Bag holders who are
    disqualified") otherwise registers zero Equipment signal, since only the
    generic word "equipment" was in the list, and the cross-filter gate
    never sees the 2 categories it needs to split.
    """
    from .intent_schema import ISSUED_EQUIPMENT_ITEMS, PROCURED_EQUIPMENT_ITEMS

    base = _CATEGORY_SIGNALS["Equipment"]
    names = [item.lower() for item in ISSUED_EQUIPMENT_ITEMS + PROCURED_EQUIPMENT_ITEMS]
    additions = [name for name in names if name not in base]
    _CATEGORY_SIGNALS["Equipment"] = base + additions


def _extend_medical_category_signals() -> None:
    """Extend _CATEGORY_SIGNALS["Medical"] with every curated disease name
    (see entity_extractor's `_KNOWN_DISEASES`), mirroring the sports/
    equipment extensions above — a query naming a disease not in the
    hand-picked subset here ("kidney stone", "chicken pox", "food
    poisoning", ...) otherwise registers zero Medical signal, so the
    cross-filter gate never sees the 2 categories it needs to split, and the
    diagnose filter that WAS correctly extracted ends up silently dropped
    against whatever single category the query fell back to.
    """
    from .entity_extractor import _KNOWN_DISEASES

    base = _CATEGORY_SIGNALS["Medical"]
    additions = [name for name in _KNOWN_DISEASES if name not in base]
    _CATEGORY_SIGNALS["Medical"] = base + additions


_extend_skills_category_signals()
_extend_equipment_category_signals()
_extend_medical_category_signals()


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


def _build_sub_operation(
    fragment: str,
    group_by: Optional[str] = None,
    filter_fragment: Optional[str] = None,
) -> SubOperation:
    intent_result = classify_admin_intent(fragment)
    try:
        dotnet_payload = format_admin_payload(intent_result)
    except Exception:
        dotnet_payload = {}
    return SubOperation(
        raw_fragment=fragment,
        intent_result=intent_result,
        dotnet_payload=dotnet_payload,
        group_by=group_by,
        filter_fragment=filter_fragment,
    )


def _apply_response_type_override(ops: List[SubOperation], raw_query: str) -> None:
    """A multi-part query is split into per-category fragments before each
    is classified, so "in detail"/"summary" only lands in whichever fragment
    contains it — e.g. "show bpet performers who returned equipment in
    detail" would otherwise leave the performers leg on the Summary default.
    One "in detail"/"summary" in the query governs the whole answer, so
    detect it against the full raw query and force it onto every leg.
    """
    override = detect_query_response_type_override(raw_query)
    if not override:
        return
    for op in ops:
        if op.intent_result.get("responseType") != override:
            op.intent_result["responseType"] = override
            op.dotnet_payload = format_admin_payload(op.intent_result)


# Operations where a count genuinely means "limit to the top/bottom N" —
# not filter/condition operations (BySport, Search, Holding, Individual, ...)
# where capping the underlying fetch could hide real cross-filter matches
# (e.g. capping a "who plays cricket" leg to 5 could exclude the actual
# agniveer a "top 5 performers" leg needs to intersect against).
_RANKABLE_OPERATIONS = frozenset(
    {"Top", "Bottom", "Most", "Least", "TopUnit", "BestAttempt"}
)

# Default number of rows a "Top N"/"Bottom N" style query should ever display
# when the user didn't state a count. Used to re-trim cross-filter results
# after the fetch-side uncap-to-1000 below (see its comment for why the
# fetch itself must stay uncapped).
_CROSS_FILTER_DISPLAY_DEFAULT = 10


def _apply_number_override(ops: List[SubOperation], raw_query: str) -> None:
    """A multi-part query's "top N"/"bottom N" only lands in whichever
    fragment's own text mentioned it — e.g. "top 5 BPET performers and best
    drill performers" would otherwise leave the second leg with no limit at
    all. Detect the count from the whole raw query once and apply it to
    every rankable operation that doesn't already have its own number.
    """
    override = detect_query_number_override(raw_query)
    if override is None:
        return
    for op in ops:
        if (
            op.intent_result.get("operation") in _RANKABLE_OPERATIONS
            and op.intent_result.get("number") is None
        ):
            op.intent_result["number"] = override
            op.dotnet_payload = format_admin_payload(op.intent_result)


def _is_semantic_comparison(
    text_lower: str, categories: List[str], semantic: Dict[str, Any]
) -> bool:
    # Direct keywords
    if any(kw in text_lower for kw in _COMPARISON_KEYWORDS):
        return True

    # Semantic query type from understanding engine
    if semantic and (
        semantic.get("query_type") in ("compare", "comparison")
        or semantic.get("complexity") == "comparison"
    ):
        return True

    # The understanding engine already ran its own gated cross_filter/
    # multi_independent checks. If it settled on one of those, the weaker
    # heuristics below (e.g. "two sections mentioned") must not silently
    # override that decision back to compare — e.g. "Show BPET report, also
    # show firing report and leave status" names two sections but is three
    # independent requests, not a comparison.
    if semantic and semantic.get("query_type") in ("cross_filter", "multi_independent"):
        return False

    # Adjectives / comparative words
    comparatives = [
        "better",
        "worse",
        "higher",
        "lower",
        "faster",
        "slower",
        "stronger",
        "weaker",
        "superior",
        "inferior",
        "compare",
        "comparison",
        "difference",
        "vs",
        "versus",
    ]
    if any(
        re.search(r"\b" + re.escape(comp) + r"\b", text_lower) for comp in comparatives
    ):
        return True

    # "more than / less than <number>" is a numeric filter, NOT a comparison
    # (only a comparison when there's a distinct comparison keyword present)

    # A section immediately followed by its own grading value ("BPET
    # Excellent", "Firing Good") is a filter condition on each section, not a
    # request to compare the sections against each other — e.g. "candidates
    # with BPET Excellent and Firing Good" wants one filtered result set
    # (CROSS_FILTER), not a side-by-side comparison. Only trust the "multiple
    # sections mentioned" signal below when this qualifier pattern is absent.
    _section_with_grading = re.search(
        r"\b(bpet|bept|ppt|firing|drill)\b\s*\w*\s*\b(excellent|good|sat|fail|unsat)\b",
        text_lower,
    )

    # Multiple sections
    sections_found = {
        s
        for s in _COMPARISON_SECTIONS
        if re.search(r"\b" + re.escape(s) + r"\b", text_lower)
    }
    if len(sections_found) >= 2 and not _section_with_grading:
        return True

    # Multiple companies/units
    companies_found = {
        u
        for u in _COMPARISON_UNITS
        if re.search(r"\b" + re.escape(u) + r"\b", text_lower)
    }
    if len(companies_found) >= 2:
        return True

    # Multiple platoons
    platoons_found = set(
        re.findall(r"\bplatoon\s*\d+\b|\bpl\s*\d+\b|\b\d+\s*platoon\b", text_lower)
    )
    if len(platoons_found) >= 2:
        return True

    # Multiple batches
    batches_found = set(re.findall(r"\bbatch\s*[a-z0-9]+\b", text_lower))
    if len(batches_found) >= 2:
        return True

    # Multiple sports
    from .intent_schema import SPORTS

    sports_found = {
        s for s in SPORTS if re.search(r"\b" + re.escape(s.lower()) + r"\b", text_lower)
    }
    if len(sports_found) >= 2:
        return True

    # Date-range pattern ("from X to Y") — two months/years in a range are NOT a comparison
    _date_range_pattern = re.compile(
        r"\bfrom\s+\S+\s+(?:to|until)\s+\S+", re.IGNORECASE
    )
    if _date_range_pattern.search(text_lower):
        return False

    # Multiple months — only a comparison when NOT in a "from X to Y" range
    months = [
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
        "jan",
        "feb",
        "mar",
        "apr",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec",
    ]
    months_found = {
        m for m in months if re.search(r"\b" + re.escape(m) + r"\b", text_lower)
    }
    if len(months_found) >= 2:
        # Require an explicit comparison keyword alongside the two months
        if any(
            kw in text_lower for kw in ("compare", "vs", "versus", "difference between")
        ):
            return True

    # Multiple years — same guard
    years_found = set(re.findall(r"\b(19\d{2}|20\d{2})\b", text_lower))
    if len(years_found) >= 2:
        if any(
            kw in text_lower for kw in ("compare", "vs", "versus", "difference between")
        ):
            return True

    return False


def _normalize_n_parts(parts: List[str]) -> List[Tuple[str, str]]:
    cleaned_parts = []
    for part in parts:
        p = part.strip()
        for kw in ("compare", "comparison", "difference", "versus", "vs"):
            p = re.sub(
                r"\b" + re.escape(kw) + r"\b", "", p, flags=re.IGNORECASE
            ).strip()
        p = re.sub(
            r"\b(results|stats|data|records|performance|score|marks)\b.*$",
            "",
            p,
            flags=re.IGNORECASE,
        ).strip()
        p = re.sub(r"\bfor\s+\w+(?:\s+\d+)?$", "", p, flags=re.IGNORECASE).strip()
        cleaned_parts.append(p)

    # Check if there is a shared trailing category/keyword in the last part
    last_part = cleaned_parts[-1]
    last_tokens = last_part.split()
    if len(last_tokens) > 1:
        last_word = last_tokens[-1]
        word_lower = last_word.lower()
        category_tokens = {cat.lower() for cat in _CATEGORY_SIGNALS.keys()}
        common_nouns = {
            "attendance",
            "performance",
            "records",
            "status",
            "cases",
            "score",
            "marks",
            "grade",
            "stats",
            "cases",
        }
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
                    temp_text = temp_text[len(prefix) :].strip()
                    temp_lower = temp_text.lower().strip()
            parts = re.split(re.escape(sep), temp_text, flags=re.IGNORECASE)
            parts = propagate_lead_in_across_parts(parts)
            return _normalize_n_parts(parts)

    diff_match = re.search(
        r"difference\s+between\s+(.+?)\s+and\s+(.+)", query_text, re.IGNORECASE
    )
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
        prefix = text[: matches[0][0]].strip()
        # Strip comparison lead words regardless of trailing whitespace
        prefix = re.sub(
            r"^(?:compare|comparison\s+of|comparison\s+between|comparison)\b\s*",
            "",
            prefix,
            flags=re.IGNORECASE,
        ).strip()
        prefix_lower = prefix.lower()

        prefix = re.sub(r"\b(and|or)\b\s*$", "", prefix, flags=re.IGNORECASE).strip()

        suffix = text[matches[-1][1] :].strip()
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
    sec_matches = find_matches(list(_COMPARISON_SECTIONS), text_lower)
    if len({m[2] for m in sec_matches}) >= 2:
        return split_on_matches(query_text, sec_matches)

    # 2. Company/Units
    coy_matches = find_matches(list(_COMPARISON_UNITS), text_lower)
    if len({m[2] for m in coy_matches}) >= 2:
        return split_on_matches(query_text, coy_matches)

    # 3. Platoons
    platoon_matches = [
        (m.start(), m.end(), m.group(0))
        for m in re.finditer(
            r"\bplatoon\s*\d+\b|\bpl\s*\d+\b|\b\d+\s*platoon\b", text_lower
        )
    ]
    if len({m[2] for m in platoon_matches}) >= 2:
        return split_on_matches(query_text, platoon_matches)

    # 4. Batches
    batch_matches = [
        (m.start(), m.end(), m.group(0))
        for m in re.finditer(r"\bbatch\s*[a-z0-9]+\b", text_lower)
    ]
    if len({m[2] for m in batch_matches}) >= 2:
        return split_on_matches(query_text, batch_matches)

    # 5. Sports
    sport_matches = find_matches([s.lower() for s in SPORTS], text_lower)
    if len({m[2] for m in sport_matches}) >= 2:
        return split_on_matches(query_text, sport_matches)

    # 6. Months
    months = [
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
        "jan",
        "feb",
        "mar",
        "apr",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec",
    ]
    month_matches = find_matches(months, text_lower)
    if len({m[2] for m in month_matches}) >= 2:
        return split_on_matches(query_text, month_matches)

    # 7. Years
    year_matches = [
        (m.start(), m.end(), m.group(0))
        for m in re.finditer(r"\b(19\d{2}|20\d{2})\b", text_lower)
    ]
    if len({m[2] for m in year_matches}) >= 2:
        return split_on_matches(query_text, year_matches)

    # Fallback for single target comparison with separator: "compare BPET with Platoon 15"
    for sep in (" with ", " against ", " versus ", " vs "):
        if sep in text_lower:
            idx = text_lower.find(sep)
            left_part = query_text[:idx].strip()
            right_part = query_text[idx + len(sep) :].strip()
            left_clean = re.sub(
                r"^(?:compare|comparison\s+of|comparison\s+between|comparison)\b\s*",
                "",
                left_part,
                flags=re.IGNORECASE,
            ).strip()
            if left_clean and right_part:
                return [
                    (left_clean, left_clean),
                    (right_part, f"{left_clean} for {right_part}"),
                ]

    return [(query_text, query_text)]


def plan_query(query: str, semantic: Optional[Dict[str, Any]] = None) -> QueryPlan:
    raw_query = (query or "").strip()
    q = _normalise(raw_query)
    if semantic is None:
        semantic = understand_query(raw_query)

    if not q:
        return QueryPlan(QueryType.SIMPLE, [], 0.0, raw_query, "Empty query")

    if semantic.get("conversational"):
        op = _build_sub_operation(raw_query)
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
        if len(components) >= 2:
            ops = []
            comparison_execution_plan = []
            combined_filters = {}
            for idx, (label, fragment) in enumerate(components):
                op = _build_sub_operation(fragment)
                ops.append(op)
                combined_filters.update(
                    build_filters_from_entities(op.intent_result.get("filters", {}))
                )
                comparison_execution_plan.append(
                    {
                        "id": f"dataset_{idx + 1}",
                        "label": label,
                        "intent": op.intent_result,
                        "filters": build_filters_from_entities(
                            op.intent_result.get("filters", {})
                        ),
                        "payloadContext": {
                            "endpoint": "api/AiCommand/execute",
                            "category": op.intent_result.get("category"),
                            "operation": op.intent_result.get("operation"),
                        },
                    }
                )

            _apply_response_type_override(ops, raw_query)
            _apply_number_override(ops, raw_query)

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
                confidence=_clamp_confidence(
                    max(float(semantic.get("confidence") or 0.85), 0.85)
                ),
                raw_query=raw_query,
                reasoning="Comparison query detected semantically",
                filters=combined_filters,
                comparison_execution_plan=comparison_execution_plan,
            )
        else:
            logger.warning(
                "plan_query: comparison detected but could not decompose into "
                ">= 2 independent operations | raw_query=%r | components_found=%d | "
                "falling through to semantic analysis. "
                "Note: 'Compare' operation will NOT be sent to .NET.",
                raw_query,
                len(components),
            )

    qtype = (semantic.get("query_type") or "simple").strip().lower()
    if qtype == "cross_filter" and _should_treat_cross_filter_as_multi_independent(
        semantic
    ):
        qtype = "multi_independent"

    def _ops_from_semantic_fragments(default_fragment: str) -> List[SubOperation]:
        fragments = semantic.get("sub_requests")
        ops: List[SubOperation] = []
        if isinstance(fragments, list) and fragments:
            for fragment in fragments:
                frag_text = (
                    fragment.get("fragment") if isinstance(fragment, dict) else None
                )
                if not frag_text:
                    continue
                ops.append(_build_sub_operation(frag_text))
        if not ops:
            ops = [_build_sub_operation(default_fragment)]
        return ops

    if qtype == "comparison":
        ops = _ops_from_semantic_fragments(raw_query)
        valid_ops = [
            op
            for op in ops
            if op.intent_result.get("category")
            or op.intent_result.get("section")
            or op.intent_result.get("sport")
        ]
        if len([op for op in valid_ops if not _is_leftover_subject_op(op)]) >= 2:
            valid_ops = [op for op in valid_ops if not _is_leftover_subject_op(op)]

        if len(valid_ops) >= 2:
            combined_filters = {}
            for op in valid_ops:
                combined_filters.update(
                    build_filters_from_entities(op.intent_result.get("filters", {}))
                )
            _apply_response_type_override(valid_ops, raw_query)
            _apply_number_override(valid_ops, raw_query)
            logger.info(
                "plan_query: COMPARE plan (semantic fallback) | query_type=compare | "
                "operation_count=%d",
                len(valid_ops),
            )
            return QueryPlan(
                QueryType.COMPARE,
                valid_ops,
                _clamp_confidence(max(float(semantic.get("confidence") or 0.85), 0.85)),
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
        ops = _ops_from_semantic_fragments(raw_query)
        valid_ops = [op for op in ops if op.intent_result.get("category")]
        if len(valid_ops) >= 2:
            combined_filters = {}
            categories = {
                op.intent_result["category"]
                for op in valid_ops
                if op.intent_result.get("category")
            }
            for op in valid_ops:
                combined_filters.update(
                    build_filters_from_entities(op.intent_result.get("filters", {}))
                )
            _apply_response_type_override(valid_ops, raw_query)
            _apply_number_override(valid_ops, raw_query)
            return QueryPlan(
                QueryType.MULTI_INDEPENDENT,
                valid_ops,
                _clamp_confidence(max(float(semantic.get("confidence") or 0.8), 0.8)),
                raw_query,
                f"Multi-independent semantic query: {', '.join(sorted(categories))}",
                filters=combined_filters,
            )

    if qtype == "cross_filter":
        ops = _ops_from_semantic_fragments(raw_query)
        # Schedule is a standalone category (timetable/agenda), never a filter condition
        valid_ops = [
            op
            for op in ops
            if op.intent_result.get("category")
            and op.intent_result.get("category") != "Schedule"
        ]
        if len([op for op in valid_ops if not _is_leftover_subject_op(op)]) >= 2:
            valid_ops = [op for op in valid_ops if not _is_leftover_subject_op(op)]

        if len(valid_ops) >= 2:
            for op in valid_ops:
                if op.intent_result.get(
                    "category"
                ) == "Roster" and op.intent_result.get("sport"):
                    op.intent_result["category"] = "Skills"
                    op.dotnet_payload = format_admin_payload(op.intent_result)
            # Cross-filter intersects individual agniveer records by
            # agniveerNo. A "Summary" responseType returns aggregate counts
            # (e.g. improvedCount, totalAgniveers) with no per-agniveer rows
            # at all, so intersection silently finds nothing. Every leg needs
            # the Detailed, per-agniveer response for the intersection to
            # have anything to match on.
            for op in valid_ops:
                if op.intent_result.get("responseType") != "Detailed":
                    op.intent_result["responseType"] = "Detailed"
                    op.dotnet_payload = format_admin_payload(op.intent_result)
            # An explicit count ("top 5") in the query but mentioned in a
            # different fragment than the ranking leg — apply it before the
            # generic uncap-to-1000 below, so a real user-stated number
            # always wins over that fallback.
            _apply_number_override(valid_ops, raw_query)
            # Ranking/trend operations return only the top N (default 10)
            # unless "n" is set explicitly. Left uncapped, a cross-filter leg
            # like "who improved in BPET" only ever considers the top 10
            # improvers — a real match sitting at rank 11+ is silently
            # missed. Uncap to the full candidate set unless the user asked
            # for a specific count (e.g. "top 5 who improved...").
            for op in valid_ops:
                if (
                    op.intent_result.get("category") == "Performance"
                    and op.intent_result.get("operation")
                    in ("Top", "Bottom", "Improvement", "Drop")
                    and op.intent_result.get("number") is None
                ):
                    # Remember the display cap separately from the fetch-side
                    # "number" sent to .NET — the combiner trims the final
                    # intersected rows back down to this after matching, since
                    # the 1000 below exists only to widen the candidate pool.
                    op.intent_result["_displayLimit"] = _CROSS_FILTER_DISPLAY_DEFAULT
                    op.intent_result["number"] = 1000
                    op.dotnet_payload = format_admin_payload(op.intent_result)
            combined_filters = {}
            for op in valid_ops:
                combined_filters.update(
                    build_filters_from_entities(op.intent_result.get("filters", {}))
                )
            return QueryPlan(
                QueryType.CROSS_FILTER,
                valid_ops,
                _clamp_confidence(max(float(semantic.get("confidence") or 0.8), 0.8)),
                raw_query,
                "Cross-filter semantic query detected",
                filters=combined_filters,
            )

    if qtype == "trend":
        op = _build_sub_operation(raw_query)
        filters = build_filters_from_entities(op.intent_result.get("filters", {}))
        return QueryPlan(
            QueryType.TREND,
            [op],
            _clamp_confidence(max(float(semantic.get("confidence") or 0.85), 0.85)),
            raw_query,
            "Trend query detected from semantic understanding",
            filters=filters,
        )

    if qtype == "distribution":
        op = _build_sub_operation(raw_query)
        filters = build_filters_from_entities(op.intent_result.get("filters", {}))
        return QueryPlan(
            QueryType.DISTRIBUTION,
            [op],
            _clamp_confidence(max(float(semantic.get("confidence") or 0.85), 0.85)),
            raw_query,
            "Distribution query detected from semantic understanding",
            filters=filters,
        )

    op = _build_sub_operation(raw_query)
    filters = build_filters_from_entities(op.intent_result.get("filters", {}))
    # semantic.get("confidence") alone badly under-represents queries that
    # classify_intent() already resolved correctly and confidently (e.g.
    # "give details of A0701763P" -> personaldetail/info at 0.52) — the
    # semantic-understanding score is a separate, more conservative signal
    # and must not silently override a real classification result.
    confidence = max(
        0.3,
        float(semantic.get("confidence") or 0.0),
        float(op.intent_result.get("confidence_score") or 0.0),
    )
    if (
        semantic.get("operation") == "ranking"
        or semantic.get("query_type") == "ranking"
    ):
        return QueryPlan(
            QueryType.ANALYTICS,
            [op],
            _clamp_confidence(max(confidence, 0.75)),
            raw_query,
            "Semantic ranking query detected",
            filters=filters,
            analytics_hint="rank",
        )
    return QueryPlan(
        QueryType.SIMPLE,
        [op],
        _clamp_confidence(max(confidence, 0.5 if filters else 0.3)),
        raw_query,
        "Single-intent query with filters",
        filters=filters,
    )
