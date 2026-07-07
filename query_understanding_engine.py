"""
query_understanding_engine.py
=============================
Semantic query understanding for the admin pipeline.

The goal is to infer user intent from meaning, not from isolated keywords.
The engine remains deterministic and lightweight so it can run in the main
request path without an external model.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from conversation_detector import is_conversational_query, normalize_text

_SECTION_ALIASES = {
    "bpet": "BPET",
    "bept": "BPET",
    "ppt": "PPT",
    "firing": "Firing",
    "drill": "Drill",
}

_PERFORMANCE_SECTIONS = {"BPET", "PPT", "Firing", "Drill"}
_GRADING_VALUES = ("excellent", "good", "sat", "fail", "unsat")

# Strong, mostly-unambiguous compare verbs/phrases. Deliberately excludes bare
# "against"/"between"/"across"/"among" (used as range/membership words
# elsewhere) and "more than"/"less than"/"greater than"/"fewer than" (numeric
# filters, e.g. "agniveers with more than 5 days leave" must NOT become a
# comparison — see test_09_more_than_5_days_leave_not_comparison).
_COMPARISON_MARKERS = (
    "compare",
    "comparison",
    "versus",
    "difference between",
    "vs",
    "compared to",
    "compared with",
    "compare with",
    "compare against",
    "contrast",
    "contrasting",
    "differentiate",
    "different from",
    "in comparison with",
    "in comparison to",
    "in contrast to",
    "rank against",
    "relative performance",
    "head to head",
    "side by side",
    "similar to",
    "unlike",
    "identical to",
    "equivalent to",
    "same as",
)

# Multi-independent markers. Every hit here is still gated in understand_query
# by requiring the clauses split on that marker to reference >= 2 distinct
# categories — several of these words (also, plus, then) are common enough
# that using them unconditionally would misfire on ordinary single-request
# queries, so the gate does the real work, not the word list.
_MULTI_INDEPENDENT_MARKERS = (
    "as well as",
    "along with",
    "together with",
    "and also",
    "also show",
    "then show",
    "next show",
    "in addition to",
    "in addition",
    "additionally",
    "furthermore",
    "moreover",
    "besides",
    "after that",
    "followed by",
    "subsequently",
    "at the same time",
    "simultaneously",
    "meanwhile",
    "also",
    "plus",
    "then",
    "next",
    "later",
)
_CROSS_FILTER_MARKERS = (
    "who has",
    "who played",
    "who plays",
    "who is on leave",
    "currently on leave",
    "currently absent",
    "with medical",
    "on leave",
    "medical leave",
    "on medical leave",
    "among",
    "within",
    "suffering",
    "suffered",
    "having ",
    "has ",
    "have ",
    "without ",
    "where ",
    "matching ",
    "meeting ",
    "satisfying ",
    "fulfilling ",
    "belonging to",
    "including ",
    "excluding ",
    "containing ",
    "filtered by",
    "based on",
    "according to",
    "subject to",
    "before ",
    "after ",
    "during ",
    "since ",
    "until ",
    "at least",
    "at most",
    "over ",
    "under ",
    "except ",
    "except for",
    "other than",
    "missing ",
    "part of",
    "member of",
    "unless ",
    "provided ",
    "whenever ",
    "depending on",
)

# Subset of the markers above with a dedicated fallback splitter one level up
# in understand_query (keyed off this exact list). Kept as its own tuple so
# other splitting paths can check "is this a leave-status phrase" without
# duplicating the list.
_LEAVE_STATUS_MARKERS = (
    "who plays",
    "who is on leave",
    "currently on leave",
    "currently absent",
    "with medical",
    "on leave",
    "medical leave",
    "on medical leave",
)

# Generic relative-clause connectors ("who", "with", "whose", "which", "that
# has/have") also signal a filter relationship between two categories, even
# when the exact verb/phrase isn't in the fixed marker list above — e.g. "top
# 10 bpet performers who have volleyball in their skills" instead of "... who
# plays volleyball". The ">= 2 categories" gate applied wherever this is used
# keeps it from firing on ordinary single-category queries that happen to
# contain one of these very common words.
_CROSS_FILTER_GENERIC_CONNECTORS = re.compile(
    r"\bwho\b|\bwith\b|\bwhose\b|\bwhich\b|\bhaving\b|\bhas\b|\bhave\b"
)

# "... for Dogra class agniveers" / "... among Dogra class" — a ranking/trend
# query scoped to a community/class roster. Matched separately from the
# generic markers above because the cutpoint for splitting is the class name
# itself, not a fixed connector word.
_CLASS_NAMES = ("sikh", "dogra", "jat", "gurkha", "gorkha", "rajput", "punjabi", "oic")
_CLASS_FILTER_RE = re.compile(
    rf"\b({'|'.join(_CLASS_NAMES)})\s+class\b", re.IGNORECASE
)


def _build_sport_filter_re() -> "re.Pattern[str]":
    """"... for football players" / "... among cricket players" — a
    performance/ranking query scoped to a sport roster. Matched the same way
    as `_CLASS_FILTER_RE`: the cutpoint for splitting is the sport name
    itself, not a fixed connector word, so "for"/"among"/"from" all work.
    """
    from intent_engine.intent_schema import SPORTS

    names = sorted({re.escape(name) for name in SPORTS.keys()}, key=len, reverse=True)
    return re.compile(rf"\b({'|'.join(names)})\s+players?\b", re.IGNORECASE)


_SPORT_FILTER_RE = _build_sport_filter_re()


def _has_cross_filter_marker(text: str) -> bool:
    if any(marker in text for marker in _CROSS_FILTER_MARKERS):
        return True
    if _CLASS_FILTER_RE.search(text):
        return True
    if _SPORT_FILTER_RE.search(text):
        return True
    return bool(_CROSS_FILTER_GENERIC_CONNECTORS.search(text))
_RANKING_MARKERS = (
    "rank",
    "top",
    "highest",
    "best",
    "maximum",
    "most",
    "leading",
    "lowest",
    "worst",
    "minimum",
    "least",
    "bottom",
)
_THRESHOLD_MARKERS = (
    "threshold",
    "limit",
    "quota",
    "allowance",
    "cap",
    "warning level",
    "critical level",
    "danger zone",
    "safe limit",
    "ceiling",
    "boundary",
    "cutoff",
    "benchmark",
    "above",
    "below",
    "over",
    "under",
    "greater than",
    "less than",
    "more than",
    "higher than",
    "lower than",
    "equal to",
    "equals",
    "at least",
    "at most",
    "between",
    "within",
    "outside",
    "inside",
    "greater",
    "smaller",
    "surpasses",
    "exceeds",
    "crosses",
    "beyond",
    "past",
    "near",
    "nearing",
    "close to",
    "almost",
    "approximately",
    "about to",
    "approaching",
    "on the verge of",
    "just below",
    "nearly",
    "around",
    "close enough",
    "not far from",
    "reaching",
    "heading toward",
    "exhaust",
    "exhaustion",
    "running out",
    "balance",
    "remaining",
    "unused",
    "percentage",
    "ratio",
    "90%",
    "90 %"

)
_DISTRIBUTION_MARKERS = (
    "distribution",
    "breakdown",
    "share",
    "composition",
    "by unit",
    "unit wise",
)
_TREND_MARKERS = (
    "trend",
    "over time",
    "growth",
    "increase",
    "decrease",
    "decline",
    "drop",
)


@dataclass
class QueryUnderstanding:
    mode: str = "admin"
    intent_kind: str = "simple"
    complexity: str = "simple"
    user_goal: str = ""
    operation: str = "lookup"
    category: Optional[str] = None
    section: Optional[str] = None
    metric: Optional[str] = None
    sort: Optional[str] = None
    query_type: str = "analytical"
    confidence: float = 0.0
    group_by: Optional[str] = None
    filters: Dict[str, Any] = field(default_factory=dict)
    entities: Dict[str, Any] = field(default_factory=dict)
    comparison_intent: bool = False
    cross_filter_intent: bool = False
    sub_requests: List[Dict[str, Any]] = field(default_factory=list)
    conversational: bool = False

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["confidence"] = round(float(self.confidence), 2)
        return payload


def _extract_section(text: str) -> Optional[str]:
    for token, label in _SECTION_ALIASES.items():
        if re.search(rf"\b{re.escape(token)}\b", text):
            return label
    return None


def _extract_grading(text: str) -> Optional[str]:
    for phrase in _GRADING_VALUES:
        if re.search(rf"\b{phrase}\b", text):
            return phrase.title() if phrase != "unsat" else "UNSAT"
    return None


def _extract_bmi_category(text: str) -> Optional[str]:
    for token in ("obese", "overweight", "underweight", "normal"):
        if re.search(rf"\b{token}\b", text):
            return token.title()
    return None


def _extract_blood_group(text: str) -> Optional[str]:
    match = re.search(
        r"\b(?:blood\s*group|bg)\s*((?:AB|A|B|O)[+-])\b", text, re.IGNORECASE
    )
    if match:
        return match.group(1).upper()
    for token in ("ab+", "ab-", "a+", "a-", "b+", "b-", "o+", "o-"):
        if re.search(rf"\b{re.escape(token)}\b", text, re.IGNORECASE):
            return token.upper()
    return None


def _extract_number(text: str) -> Optional[int]:
    match = re.search(r"\b(\d+)\b", text)
    return int(match.group(1)) if match else None


def _has_phrase(text: str, phrase: str) -> bool:
    return phrase in text


def _infer_category(text: str, entities: Dict[str, Any]) -> Optional[str]:
    if any(
        token in text
        for token in (
            "disqualified",
            "disqualify",
            "disqualifying",
            "disqualifies",
            "disqualification",
            "disqualified agniveer",
            "disqualified agniveers",
            "removed agniveer",
            "expelled agniveer",
        )
    ):
        return "disqualified"
    if any(
        token in text
        for token in (
            "personal detail",
            "personal details",
            "personaldetail",
            "personal info",
            "profile",
            "biodata",
            "bio data",
            "contact",
            "education",
            "qualification",
            "family details",
            "next of kin",
        )
    ):
        return "personaldetail"
    if entities.get("grading"):
        return "Performance"
    if entities.get("bmi_category") or entities.get("blood_group"):
        return "Medical"
    if entities.get("leave_type"):
        return "Leave"
    if "leave" in text or "absconded" in text or "absent" in text:
        return "Leave"
    if entities.get("sport") and any(
        token in text
        for token in (
            "roster",
            "player",
            "players",
            "play",
            "plays",
            "played",
            "which sport",
            "skills",
        )
    ):
        return "Skills"
    if "class" in text and any(
        token in text for token in ("skills", "roster", "sport", "sports")
    ):
        return "Skills"
    if entities.get("unit_name") and any(
        token in text
        for token in ("distribution", "equipment", "leave", "attendance", "performance")
    ):
        return "Distribution"
    if entities.get("section") in _PERFORMANCE_SECTIONS:
        return "Performance"
    if any(
        token in text
        for token in (
            "grade summary",
            "grading summary",
            "grade distribution",
            "grading distribution",
        )
    ):
        return "Performance"
    if any(
        token in text
        for token in (
            "schedule",
            "training schedule",
            "company schedule",
            "today's schedule",
            "today schedule",
            "current schedule",
            "today's training",
            "training",
        )
    ):
        return "Schedule"
    if any(
        token in text
        for token in (
            "equipment",
            "equipment search",
            "search equipment",
            "find equipment",
            "lookup equipment",
            "item name",
            "category",
            "returned",
            "poor condition",
            "damaged",
            "broken",
            "holding",
        )
    ):
        return "Equipment"
    if any(
        token in text
        for token in (
            "attendance",
            "present",
            "absent",
            "campus",
            "headcount",
            "strength",
        )
    ):
        if "strength" in text or "headcount" in text:
            return "Strength"
        return "Attendance"
    if any(
        token in text
        for token in (
            "medical",
            "bmi",
            "blood group",
            "blood",
            "hospital",
            "disease",
            "fever",
            "malaria",
            "injury",
            "illness",
            "sick",
        )
    ):
        return "Medical"
    if any(token in text for token in ("verification", "verified", "pending")):
        return "Verification"
    if any(
        token in text
        for token in (
            "equipment",
            "issued",
            "procured",
            "overdue",
            "returned",
            "holding",
        )
    ):
        return "Equipment"
    if any(
        token in text
        for token in ("distribution", "breakdown", "assigned", "unassigned")
    ):
        return "Distribution"
    if any(token in text for token in ("overall", "composite")):
        return "Overall"
    if any(token in text for token in ("improvement", "decline", "drop")):
        return "Performance"
    if any(
        token in text
        for token in (
            "performance",
            "score",
            "marks",
            "grade",
            "grading",
            "top",
            "highest",
            "best",
            "lowest",
            "worst",
            "rank",
        )
    ):
        return "Performance"
    return None


def _infer_operation(text: str, entities: Dict[str, Any]) -> str:
    category = _infer_category(text, entities)
    if category == "Equipment" and (entities.get("agniveerNo") or entities.get("agniveer_no")):
        return "byagniveer"
    if any(marker in text for marker in _COMPARISON_MARKERS):
        return "compare"
    if any(marker in text for marker in _RANKING_MARKERS):
        return "ranking"
    if entities.get("grading"):
        return "grading"
    if entities.get("bmi_category"):
        return "bmi"
    if entities.get("blood_group"):
        return "bloodgroup"
    if "attempt" in text:
        return "attemptwise"
    if "best attempt" in text:
        return "bestattempt"
    if any(token in text for token in ("strength breakdown", "headcount", "current strength", "strength")):
        return "strengthbreakdown"
    if any(
        token in text
        for token in (
            "search equipment",
            "find equipment",
            "lookup equipment",
            "search by category",
            "search by name",
            "equipment search",
        )
    ):
        return "search"
    if any(token in text for token in ("returned", "poor condition", "damaged", "broken")):
        return "returned"
    if any(token in text for token in ("overdue", "currently issued", "currently holding", "holding", "issued")):
        return "holding"
    if "current leave" in text or "leave today" in text or "on leave" in text:
        return "current"
    if "absconded" in text:
        return "absconded"
    if any(marker in text for marker in _TREND_MARKERS):
        return "trend"
    if any(token in text for token in ("today's schedule", "today schedule", "current schedule", "training schedule", "company schedule")):
        return "today"
    if any(marker in text for marker in _DISTRIBUTION_MARKERS):
        return "distribution"
    if "average" in text or "mean" in text or "avg" in text:
        return "average"
    if "count" in text or "how many" in text or "number of" in text:
        return "count"
    return "lookup"


def _infer_sort(operation: str, text: str) -> Optional[str]:
    if operation != "ranking":
        return None
    if any(
        marker in text for marker in ("lowest", "worst", "least", "bottom", "minimum")
    ):
        return "ascending"
    if any(
        marker in text
        for marker in ("top", "highest", "best", "most", "maximum", "leading")
    ):
        return "descending"
    return None


def _infer_metric(category: Optional[str], operation: str) -> Optional[str]:
    if operation == "count":
        return "count"
    if operation == "average":
        return "average_score" if category == "Performance" else "average"
    if operation == "trend":
        return "trend_value"
    if operation == "search":
        return "search_term"
    if operation == "compare":
        return "average_score" if category == "Performance" else "count"
    return None


def _infer_group_by(text: str) -> Optional[str]:
    for candidate in (
        "platoon",
        "class",
        "batch",
        "company",
        "section",
        "sport",
        "unit",
    ):
        if re.search(rf"\b{candidate}\b", text):
            return candidate
    return None


def _build_user_goal(
    category: Optional[str], operation: str, entities: Dict[str, Any]
) -> str:
    if operation == "compare":
        return "compare the requested entities"
    if operation == "grading":
        return "review grading results"
    if operation == "attemptwise":
        return "analyze attempts"
    if operation == "bmi":
        return "review medical BMI records"
    if operation == "bloodgroup":
        return "review blood group records"
    if operation == "strengthbreakdown":
        return "review strength breakdown"
    if operation == "search":
        return "search equipment records"
    if operation == "returned":
        return "review returned equipment"
    if operation == "holding":
        return "review holding equipment"
    if operation == "today":
        return "review schedule for today"
    if operation == "current":
        return "show current leave status"
    if operation == "absconded":
        return "find absconded records"
    if operation == "Verify":
        return "verify admin access"
    if category:
        return f"review {category.lower()} data"
    if entities:
        return "understand the request"
    return "understand the request"


def _split_on_connectors(text: str, markers: List[str]) -> List[str]:
    for marker in markers:
        if marker in text:
            parts = [
                part.strip(" ,") for part in text.split(marker) if part.strip(" ,")
            ]
            if len(parts) >= 2:
                return parts
    return [text]


_LEAD_IN_PREPOSITIONS = ("of", "in", "for", "among", "between", "during", "on")


def _apply_shared_lead_in(left: str, right: str) -> str:
    """Propagate a shared lead-in phrase from the left comparison fragment to
    the right one.

    "top performers of Lak-Lakhwinder and Jas-Jaswant company in BPET" splits
    on " and " into "top performers of Lak-Lakhwinder" / "Jas-Jaswant company
    in BPET" — the right side loses "top performers of", so on its own it has
    no operation/category signal and fails classification. "Lowest scorers in
    drill vs firing" has the same problem with "vs": the right side ("firing")
    loses "lowest scorers in" entirely. When the left fragment has a
    "<lead-in> <of/in/for/...> <target>" shape, prepend the same lead-in to
    the right fragment.
    """
    matches = list(
        re.finditer(
            rf"\b({'|'.join(_LEAD_IN_PREPOSITIONS)})\b", left, flags=re.IGNORECASE
        )
    )
    if not matches:
        return right
    prefix = left[: matches[-1].end()].strip()
    if not prefix or prefix.lower() in right.lower():
        return right
    return f"{prefix} {right}".strip()


def propagate_lead_in_across_parts(parts: List[str]) -> List[str]:
    """N-way version of `_apply_shared_lead_in` for raw "A vs B vs C" splits.

    "Lowest scorers in drill vs firing" splits into ["Lowest scorers in
    drill", "firing"] — only the first part keeps the "lowest scorers in"
    lead-in, so "firing" alone has no operation/category signal. Propagate
    the first part's lead-in phrase to every other part that doesn't already
    carry it.
    """
    if len(parts) < 2:
        return list(parts)
    first = parts[0].strip()
    matches = list(
        re.finditer(
            rf"\b({'|'.join(_LEAD_IN_PREPOSITIONS)})\b", first, flags=re.IGNORECASE
        )
    )
    if not matches:
        return [first] + [p.strip() for p in parts[1:]]
    prefix = first[: matches[-1].end()].strip()
    if not prefix:
        return [first] + [p.strip() for p in parts[1:]]
    result = [first]
    for p in parts[1:]:
        p = p.strip()
        if prefix.lower() in p.lower():
            result.append(p)
        else:
            result.append(f"{prefix} {p}".strip())
    return result


_ORG_NOUNS = ("company", "coy", "platoon", "unit")


def _apply_shared_trailing_suffix(left: str, right: str) -> str:
    """Propagate a shared trailing filter clause — and a shared organizational
    noun — from the right comparison fragment to the left one.

    "... of Arora and Thorat company in BPET" splits on " and " into
    "of Arora" / "Thorat company in BPET". Two things qualify both sides but
    only survive on the right: the trailing "in BPET" filter, and the word
    "company" that marks "Thorat" as an org name (needed later to resolve
    "Arora" and "Thorat" as two distinct companies instead of one blended,
    mis-resolved mention). Propagate both to the left fragment.
    """
    prep_match = re.search(
        r"\b(in|for|during|among|between|on)\b", right, flags=re.IGNORECASE
    )
    if prep_match:
        target_part = right[: prep_match.start()].strip()
        suffix = right[prep_match.start():].strip().rstrip(".").strip()
    else:
        target_part = right.strip()
        suffix = ""

    pieces = [left]

    noun_match = re.search(
        rf"\b({'|'.join(_ORG_NOUNS)})\b\s*$", target_part, flags=re.IGNORECASE
    )
    if noun_match and not re.search(
        rf"\b{re.escape(noun_match.group(1))}\b", left, flags=re.IGNORECASE
    ):
        pieces.append(noun_match.group(1))

    if suffix and suffix.lower() not in left.lower():
        pieces.append(suffix)

    return " ".join(p for p in pieces if p).strip()


def _extract_sub_requests(
    text: str,
    category: Optional[str],
    operation: str,
    entities: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if operation == "compare":
        comparator_split = re.split(
            r"\b(?:compare|comparison)\b", text, maxsplit=1, flags=re.IGNORECASE
        )
        body = comparator_split[-1].strip() if comparator_split else text
        for separator in (" and ", " vs ", " versus "):
            if separator in body:
                left, right = body.split(separator, 1)
                left = left.strip()
                right = right.strip()
                left, right = (
                    _apply_shared_trailing_suffix(left, right),
                    _apply_shared_lead_in(left, right),
                )
                return [
                    {
                        "fragment": left,
                        "category": category,
                        "operation": operation,
                        "entities": entities,
                    },
                    {
                        "fragment": right,
                        "category": category,
                        "operation": operation,
                        "entities": entities,
                    },
                ]
        return [
            {
                "fragment": body or text,
                "category": category,
                "operation": operation,
                "entities": entities,
            }
        ]

    if _has_cross_filter_marker(text):
        parts = []
        current = text
        for sep in (
            r"\bwho\b",
            r"\bwith\b",
            r"\bamong\b",
            r"\bwithin\b",
            r"\bsuffering\b",
            r"\bsuffered\b",
        ):
            split_parts = re.split(sep, current, maxsplit=1, flags=re.IGNORECASE)
            # A leading interrogative ("Who improved... among the badminton
            # players?") matches "who" at position 0, leaving an empty lead
            # fragment that then gets dropped entirely — collapsing the
            # query back to one unsplit fragment. Skip a match that produces
            # an empty lead and keep trying the remaining separators (e.g.
            # "among") instead of accepting the degenerate split.
            if len(split_parts) > 1 and split_parts[0].strip(" ,"):
                parts.append(split_parts[0].strip(" ,"))
                current = " ".join(split_parts[1:])
                break
        else:
            # "currently on leave"-style phrases are handled by a dedicated
            # fallback splitter one level up (in understand_query, keyed off
            # this same marker list) once it sees this branch produced only
            # one fragment. Don't let the class/sport cutpoint below claim
            # the split first — "cricket" in "cricket players currently on
            # leave" would cut in the wrong place, leaving "show" as a
            # near-empty lead fragment and blending the leave clause into
            # the sport fragment.
            has_leave_marker = any(
                marker in current.lower() for marker in _LEAVE_STATUS_MARKERS
            )
            roster_matches = (
                []
                if has_leave_marker
                else [
                    m
                    for m in (
                        _CLASS_FILTER_RE.search(current),
                        _SPORT_FILTER_RE.search(current),
                    )
                    if m
                ]
            )
            roster_match = min(roster_matches, key=lambda m: m.start()) if roster_matches else None
            if roster_match:
                lead = current[: roster_match.start()]
                lead = re.sub(
                    r"\b(for|from|among|within|of|in)\s*$",
                    "",
                    lead,
                    flags=re.IGNORECASE,
                ).strip(" ,")
                if lead:
                    parts.append(lead)
                    current = current[roster_match.start() :].strip()
                else:
                    # Roster mention sits at the very start ("Dogra class
                    # agniveers present today") — there's no lead text
                    # before it, so split the roster phrase itself from
                    # what follows instead of collapsing back into one
                    # fragment with an empty lead.
                    parts.append(current[roster_match.start() : roster_match.end()].strip())
                    current = current[roster_match.end() :].strip()
            else:
                parts = [current]
                current = ""

        if current:
            and_parts = [
                p.strip(" ,")
                for p in re.split(r"\band\b", current, flags=re.IGNORECASE)
                if p.strip(" ,")
            ]
            parts.extend(and_parts)
        else:
            new_parts = []
            for p in parts:
                and_parts = [
                    ap.strip(" ,")
                    for ap in re.split(r"\band\b", p, flags=re.IGNORECASE)
                    if ap.strip(" ,")
                ]
                new_parts.extend(and_parts)
            parts = new_parts

        final_parts = []
        for p in parts:
            p_clean = p.strip(" ,")
            if p_clean and p_clean not in (
                "who",
                "with",
                "and",
                "plays",
                "suffering",
                "suffered",
            ):
                final_parts.append(p_clean)

        return [
            (
                {
                    "fragment": p,
                    "category": category,
                    "operation": operation,
                    "entities": entities,
                }
                if idx == 0
                else {
                    "fragment": p,
                    "category": _infer_category(p, {}),
                    "operation": _infer_operation(p, {}),
                    "entities": entities,
                }
            )
            for idx, p in enumerate(final_parts)
        ]
    if any(marker in text for marker in _MULTI_INDEPENDENT_MARKERS) or " and " in text:
        parts = []
        if "," in text:
            parts = [
                part.strip(" ,")
                for part in re.split(r"\s*,\s*|\s+\band\b\s+", text)
                if part.strip(" ,")
            ]
        if len(parts) < 2:
            parts = _split_on_connectors(text, list(_MULTI_INDEPENDENT_MARKERS))
        if len(parts) == 1 and " and " in text:
            parts = [
                part.strip(" ,") for part in text.split(" and ") if part.strip(" ,")
            ]
        if len(parts) >= 2:
            return [
                {
                    "fragment": part,
                    "category": _infer_category(part, {}),
                    "operation": _infer_operation(part, {}),
                    "entities": entities,
                }
                for part in parts
            ]
    return [
        {
            "fragment": text,
            "category": category,
            "operation": operation,
            "entities": entities,
        }
    ]


def understand_query(query: str) -> Dict[str, Any]:
    text = normalize_text(query)
    conversational = is_conversational_query(text)
    if conversational:
        result = QueryUnderstanding(
            mode="conversation",
            intent_kind="conversation",
            complexity="conversation",
            user_goal="conversational",
            operation="conversation",
            query_type="conversational",
            confidence=0.0 if not text else 0.99,
            conversational=True,
        )
        return result.to_dict()

    from intent_engine.entity_extractor import extract_entities

    entities = extract_entities(text, semantic={})
    section = entities.get("section")
    category = _infer_category(text, entities)
    operation = _infer_operation(text, entities)
    sort = _infer_sort(operation, text)
    metric = _infer_metric(category, operation)
    group_by = _infer_group_by(text)

    comparison_intent = any(marker in text for marker in _COMPARISON_MARKERS)

    # cross_filter requires a marker AND at least 2 distinct inferred categories
    _cross_marker_hit = _has_cross_filter_marker(text)
    cross_filter_intent = False
    if _cross_marker_hit:
        from intent_engine.query_planner import _detect_categories as _dc
        _cf_cats = _dc(text)
        if len(set(_cf_cats[:3])) >= 2:
            cross_filter_intent = True

    def _distinct_category_count(clause_parts: List[str]) -> int:
        clause_categories = []
        for part in clause_parts:
            clause_entities = extract_entities(part, semantic={})
            clause_categories.append(_infer_category(part, clause_entities))
        return len({cat for cat in clause_categories if cat})

    # Every multi-independent marker — even weak ones like "also"/"then" — is
    # gated by requiring the clauses split on it to reference >= 2 distinct
    # categories. Words like "also"/"plus"/"as well as" are common enough in
    # single-request phrasing that the word alone proves nothing; the gate is
    # what makes this reliable, not the vocabulary.
    multi_intent = False
    for marker in _MULTI_INDEPENDENT_MARKERS:
        if marker not in text:
            continue
        clause_parts = [
            part.strip(" ,") for part in text.split(marker) if part.strip(" ,")
        ]
        if len(clause_parts) >= 2 and _distinct_category_count(clause_parts) >= 2:
            multi_intent = True
            break

    if not multi_intent and " and " in text:
        clause_parts = [
            part.strip(" ,") for part in text.split(" and ") if part.strip(" ,")
        ]
        if len(clause_parts) >= 2 and _distinct_category_count(clause_parts) >= 2:
            multi_intent = True

    confidence = 0.18
    if operation != "lookup":
        confidence += 0.28
    if category:
        confidence += 0.22
    if section:
        confidence += 0.16
    if metric:
        confidence += 0.08
    if sort:
        confidence += 0.06
    if group_by:
        confidence += 0.05
    if len(text.split()) >= 5:
        confidence += 0.05

    query_type = "simple"
    complexity = "simple"
    intent_kind = "simple"
    if comparison_intent:
        query_type = "comparison"
        complexity = "comparison"
        intent_kind = "comparison"
    elif cross_filter_intent:
        # cross_filter_intent already required a marker hit (including the
        # sport/class roster patterns) AND >= 2 distinct categories, so it's
        # the single source of truth here — no need to re-check a narrower,
        # separately-maintained token list that could drift out of sync with
        # _CROSS_FILTER_MARKERS.
        query_type = "cross_filter"
        complexity = "cross_filter"
        intent_kind = "cross_filter"
    elif multi_intent:
        query_type = "multi_independent"
        complexity = "multi_independent"
        intent_kind = "multi_independent"
    elif any(token in text for token in ("by unit", "unit wise")):
        query_type = "distribution"
        complexity = "distribution"
        intent_kind = "distribution"
    elif any(
        token in text for token in ("monthly", "month wise", "per month", "by month")
    ):
        query_type = "trend"
        complexity = "trend"
        intent_kind = "trend"
    elif any(
        token in text
        for token in ("weekly", "week wise", "this week", "per week", "by week")
    ):
        query_type = "trend"
        complexity = "trend"
        intent_kind = "trend"
    elif "unit" in text and any(
        token in text
        for token in ("most", "highest", "lowest", "least", "absconded", "leave")
    ):
        query_type = "distribution"
        complexity = "distribution"
        intent_kind = "distribution"
    elif operation == "trend":
        query_type = "trend"
        complexity = "trend"
        intent_kind = "trend"
    elif operation == "distribution":
        query_type = "distribution"
        complexity = "distribution"
        intent_kind = "distribution"

    filters: Dict[str, Any] = {}
    if section:
        filters["section"] = section
    if group_by:
        filters["group_by"] = group_by
    for key in (
        "grading",
        "bmi_category",
        "blood_group",
        "platoon_id",
        "company_id",
        "batch_id",
        "leave_type",
        "sport",
        "class",
        "unit_name",
        "date",
    ):
        if entities.get(key) is not None:
            filters[key] = entities[key]

    sub_requests = _extract_sub_requests(text, category, operation, entities)
    if cross_filter_intent and len(sub_requests) == 1:
        for marker in _LEAVE_STATUS_MARKERS:
            if marker in text:
                head, tail = text.split(marker, 1)
                head_clean = head.strip(" ,")
                tail_clean = tail.strip(" ,")
                if head_clean:
                    first_frag = head_clean
                    second_frag = f"{marker} {tail_clean}" if tail_clean else marker
                else:
                    # Marker sits at the very start ("Currently absent
                    # cricket players") — there's no lead text before it to
                    # form a first fragment, so split the marker phrase
                    # itself from what follows instead of reconstructing the
                    # whole original text as one fragment with an empty
                    # sibling.
                    first_frag = marker
                    second_frag = tail_clean or marker
                sub_requests = [
                    {
                        "fragment": first_frag,
                        "category": category,
                        "operation": operation,
                        "entities": entities,
                    },
                    {
                        "fragment": second_frag,
                        "category": None,
                        "operation": "lookup",
                        "entities": entities,
                    },
                ]
                break

    result = QueryUnderstanding(
        mode="admin",
        intent_kind=intent_kind,
        complexity=complexity,
        user_goal=_build_user_goal(category, operation, entities),
        operation=operation,
        category=category,
        section=section,
        metric=metric,
        sort=sort,
        query_type=query_type,
        confidence=min(0.99, round(confidence, 2)),
        group_by=group_by,
        filters=filters,
        entities=entities,
        comparison_intent=comparison_intent,
        cross_filter_intent=cross_filter_intent,
        sub_requests=sub_requests,
        conversational=False,
    )
    return result.to_dict()
