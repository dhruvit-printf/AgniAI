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
from typing import Any, Dict, List, Optional, Tuple

from conversation_detector import is_conversational_query, normalize_text

_PERFORMANCE_SECTIONS = {"BPET", "PPT", "Firing", "Drill"}

# BPET/PPT/Firing/Drill all share the single "Performance" entry in
# _CATEGORY_SIGNALS (intent_engine/query_planner.py), so a query naming two of
# them ("whose PPT grade is Excellent and BPET grade is Good") only ever
# registers as ONE detected category even though it states two distinct,
# independently-fetched conditions. This lets the cross_filter/multi_independent
# gates below recognize that case without conflating it with a genuinely
# single-condition Performance query ("show top BPET performers").
_PERFORMANCE_SECTION_TOKENS = tuple(name.lower() for name in _PERFORMANCE_SECTIONS)


def _distinct_performance_sections(text: str) -> int:
    return len({tok for tok in _PERFORMANCE_SECTION_TOKENS if tok in text})


# Strong, mostly-unambiguous compare verbs/phrases. Deliberately excludes bare
# "against"/"between"/"across"/"among" (used as range/membership words
# elsewhere) and "more than"/"less than"/"greater than"/"fewer than" (numeric
# filters, e.g. "agniveers with more than 5 days leave" must NOT become a
# comparison — see test_09_more_than_5_days_leave_not_comparison).
# " vs " is padded (not bare "vs") so it only matches as a standalone word
# surrounded by spaces — avoids false positives on "vs." at a sentence end,
# "vs,", or "vs" appearing inside an unrelated token. Shared with
# intent_engine/query_planner.py — this is the single canonical source, do
# not redefine a second copy there.
_COMPARISON_MARKERS = (
    "compare",
    "comparison",
    "versus",
    "difference between",
    " vs ",
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
    "relative to",
    "head to head",
    "side by side",
    "side-by-side",
    "similar to",
    "unlike",
    "identical to",
    "equivalent to",
    "same as",
    "vs.",
    "v/s",
    "as compared to",
    "as compared with",
    "differ from",
    "contrasted with",
    "superior to",
    "inferior to",
    "preferable to",
    # "Improvement FROM attempt 1 TO attempt 3" names two specific attempts
    "improvement from attempt",
    "improvement from",
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
    "combined with",
    "and separately",
    "show both",
    "display both",
    "provide both",
    "give both",
    "both",
    "separately",
    "independently",
    "alongside",
    "together",
    "respectively",
    "individually",
    "one by one",
    "in the meantime",
    "lastly",
)
_CROSS_FILTER_MARKERS = (
    # NOTE: status/eligibility adjectives (pending, issued, holding, present,
    # absent, hospitalized, verified, completed, rejected, returned,
    # assigned, waiting, awaiting, currently, still, already, yet,
    # "classified as", "falling under", ...) must NOT be added here — they
    # belong exclusively in _CROSS_FILTER_STATUS_MARKERS below, which exists
    # specifically because they're weak, common-word evidence (see that
    # tuple's docstring). They previously got duplicated into this
    # supposedly-"strong"/unambiguous list too, which meant
    # _has_strong_cross_filter_marker() returned True for almost any
    # multi-category question containing an everyday status word — e.g.
    # "Show me all pending police verification cases and, separately, all
    # hospitalized Agniveers" (two independent report requests) matched
    # "pending" + "hospitalized" here and was forced into cross_filter,
    # silently discarding one of the two requested lists.
    #
    # Concession connectors ("but still", "yet still", "despite", "on top
    # of that") are the opposite case — safe to trust as strong/unambiguous
    # even as full phrases, because they explicitly frame the second clause
    # as an ADDITIONAL condition on the same subject rather than a separate
    # request ("...failed Firing but still have issued equipment").
    "but still",
    "yet still",
    "despite",
    "on top of that",
    "who has",
    "who played",
    "who plays",
    "who is on leave",
    "whose ",
    "who are ",
    "who is ",
    "who were ",
    "who was ",
    "who ",
    "that are ",
    "that is ",
    "which are ",
    "which is ",
    "whose ",
    "that ",
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
    "belonged to",
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
    "from ",
    "who have",
    "with",
    "among them",
    "among those",
    "among these",
    "from them",
    "from those",
    "from these",
    "out of them",
    "within them",
    "inside them",
    "coming under",
    "filter",
    "filtered",
    "restrict",
    "limit",
    "only",
    "exclude",
    "apart from",
    "matched",
    "related",
    "associated",
    "linked",
    "connected",
    "mapped",
    "and among them",
    "and who",
    "and whose",
    "who also",
    "who still",
    "who already",
    "who currently",
    "while having",
    "while being",
    "following that",
    "only",
    "strictly",
    "specifically",
    "whom",
    "which",
    "wherein",
    "whereby",
    "whereupon",
    "whereof",
    "had",
    "possessing",
    "possesses ",
    "carrying",
    "carries ",
    "keeping",
    "keeps ",
    "owning",
    "issued with",
    "allocated",
    "given",
    "them",
    "those",
    "these",
    "they",
    "their",
    "theirs",
    "the same",
    "same",
    "same group",
    "same personnel",
    "same agniveers",
    "same candidates",
    "previous",
    "previously found",
    "previous result",
    "returned result",
    "filtered result",
    "matching records",
    "selected records",
    "matching ones",
    "selected ones",
    "returned ones",
    "within those",
    "within these",
    "inside",
    "inside those",
    "out of",
    "out of those",
    "out of these",
    "of",
    "of them",
    "of those",
    "classified under",
    "categorized under",
    "included in",
    "contained in",
    "listed in",
    "present in",
    "existing in",
    "appearing in",
    "and among",
    "and among those",
    "and among these",
    "and that",
    "and which",
    "but who",
    "but whose",
    "but having",
    "while also",
    "plus those",
    "including those",
    "including only",
    "presently",
    "now",
    "even",
    "even now",
    "further",
    "likewise",
    "similarly",
    "concurrently",
    "only those",
    "only them",
    "just",
    "merely",
    "exactly",
    "particularly",
    "especially",
    "exclusively",
    "solely",
    "limited to",
    "restricted to",
    "confined to",
    "restricted",
    "limited",
    "narrow",
    "narrowed",
    "refine",
    "refined",
    "remove",
    "removed",
    "ignore",
    "ignoring",
    "skip",
    "minus",
    "corresponding",
    "corresponds",
    "related to",
    "associated with",
    "linked with",
    "connected to",
    "mapped to",
    "joined",
    "intersecting",
    "intersected",
    "common",
    "shared",
    "overlapping",
    "if",
    "when",
    "wherever",
    "provided that",
    "assuming",
    "assuming that",
    "given",
    "given that",
    "provided they",
    "provided who",
    "provided whose",
    "then among them",
    "then from them",
    "afterwards",
    "following",
    "thereafter",
    "finally",
    "more than",
    "less than",
    "greater than",
    "smaller than",
    "equal to",
    "above",
    "below",
    "between",
    "outside",
    "inside",
    "approved",
    "admitted",
    "overweight",
    "underweight",
    "obese",
    "excellent",
    "good",
    "sat",
    "fail",
    "current",
    "active",
    "inactive",
    "latest",
    "who yet",
    "who later",
    "who eventually",
    "that also",
    "that still",
    "that already",
    "including only",
    "including those",
    "related to",
    "associated with",
    "connected to",
    "linked to",
    "waiting for",
    "awaiting for",
    # State and membership phrases
    "included in",
    "present in",
    "listed in",
    "identified as",
    "recognized as",
    "marked as",
    "registered as",
    "recorded as",
    # Relative pronouns with nouns
    "those who",
    "people who",
    "candidates who",
    "personnel who",
)

# Status/eligibility adjectives — split out from _CROSS_FILTER_MARKERS above
# because they're weaker evidence: each is a common standalone word (unlike
# "whose "/"having "/"belonging to", which are unambiguously relative-clause
# grammar). Like every marker above, they still rely on the >= 2 categories /
# performance-sections gate for safety — "Show pending verification." stays
# simple because only one category is present. But being weaker, a hit here
# (and only here — see the trailing-report-noun override below) can be
# overridden when the later clause names its own report/analytics noun,
# e.g. "Show rejected verification and equipment summary." is two independent
# reports despite "rejected" matching here, whereas "Show rejected
# verification candidates and issued equipment." has no such trailing report
# noun and stays cross-filter.
_CROSS_FILTER_STATUS_MARKERS = (
    "inside ",
    "falling under",
    "classified as",
    "qualifying",
    "eligible",
    "not eligible",
    "pending",
    "rejected",
    "verified",
    "issued",
    "holding",
    "returned",
    "assigned",
    "completed",
    "failed",
    "passed",
    "present",
    "absent",
    "hospitalized",
    "hospitalised",
    "diagnosed",
    "disqualified",
    "selected",
    "remaining",
    "existing",
    "available",
    "currently",
    "still",
    "already",
    "yet",
    "but not",
    "instead of",
    "waiting",
    "awaiting",
    "treated for",
    "under treatment",
    "medically unfit",
    "unfit",
    "sick",
    "ill",
    "injured",
    "wounded",
    "on annual leave",
    "on sick leave",
    "absconded",
    "off duty",
    "cleared",
    "not responded",
    "on campus",
    "overdue",
    "damaged",
    "poor condition",
    "improved",
    "improvement",
    "declined",
)

# A later clause naming its own report/analytics noun signals an independent
# output request, not a filter condition on the first clause's subject — used
# to override a _CROSS_FILTER_STATUS_MARKERS-only hit (see above).
_REPORT_OUTPUT_MARKERS = (
    "summary",
    "statistics",
    "count",
    "average",
    "report",
    "overview",
    "breakdown",
    "distribution",
    "analysis",
    "trend",
)

# A clause introduced by a multi-independent marker ("additionally", "then",
# "also", ...) that refers back to the PRIMARY request's result set — rather
# than naming its own independent subject — is a dependent/nested request,
# not a second independent one. E.g. "top 10 BPET performers, additionally
# check whether any of them have rejected verification" must intersect the
# verification check against those same 10 performers (cross_filter-style
# execution), not run "top 10 performers" and "rejected verification" as two
# unrelated legs. Anaphoric back-references ("them", "these", "those
# selected", "the above") and bare validation verbs ("check/confirm/verify
# whether", "see/find if") without their own named subject are the tell.
_DEPENDENT_BACKREF_MARKERS = (
    "any of them",
    "of them",
    "among them",
    "among those",
    "between them",
    "out of them",
    "from them",
    "within them",
    "for those",
    "of these",
    "only those",
    "only these",
    "only the",
    "from the above",
    "using these",
    "these selected",
    "those selected",
    "the selected",
    "the same agniveers",
    "same set",
    "who also",
    "who are also",
    "who have also",
    "who still",
    "who currently",
    "who already",
    "who yet",
)

_VALIDATION_VERB_RE = re.compile(
    r"\b(?:check|confirm|verify|see|find)\s+(?:whether|if)\b"
)


def _has_dependent_backref(text: str) -> bool:
    """True when `text` anaphorically refers to a previously-named result set
    (a dependent clause) rather than introducing its own independent subject.
    """
    if any(marker in text for marker in _DEPENDENT_BACKREF_MARKERS):
        return True
    return bool(_VALIDATION_VERB_RE.search(text))


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
    "away right now",
    "who is away",
    "who are away right now",
    "currently away",
    "away today",
    "absent right now",
    "absent today",
    "out right now",
    "out today",
)

# Generic relative-clause connectors ("who", "with", "whose", "which", "that
# has/have") also signal a filter relationship between two categories, even
# when the exact verb/phrase isn't in the fixed marker list above — e.g. "top
# 10 bpet performers who have volleyball in their skills" instead of "... who
# plays volleyball". The ">= 2 categories" gate applied wherever this is used
# keeps it from firing on ordinary single-category queries that happen to
# contain one of these very common words.
_CROSS_FILTER_GENERIC_CONNECTORS = re.compile(
    r"\bwho\b|\bwith\b|\bwhose\b|\bwhich\b|\bhaving\b|\bhas\b|\bhave\b|\bhad\b|\bwhom\b|\bthat\b|\bwhere\b|\bwherein\b|\bwhereby\b|\bbelonging\b|\bbelongs?\b|\bbelong\b|\bholding\b|\bholds?\b|\bpossessing\b|\bcarrying\b|\bkeeping\b|\bowning\b|\bassigned\b|\bissued\b|\ballocated\b|\bgiven\b|\bsuffering\b|\bsuffered\b|\bdiagnosed\b|\badmitted\b|\bhospitalized\b|\bscored\b|\bscoring\b|\bpassed\b|\bpassing\b|\bfailed\b|\bfailing\b|\bqualified\b|\bqualifying\b|\bdisqualified\b|\bverified\b|\bapproved\b|\brejected\b|\bon\s+leave\b|\babsent\b|\bpresent\b",
    re.IGNORECASE,
)

_GENERIC_INTRO_RE = re.compile(
    r"^(?:show|give\s+me|list|display|find|get|fetch|filter|search\s+for|view|provide|tell\s+me\s+about|check|see)?\s*(?:all\s+)?(?:agniveer|agniveers|candidates|personnel|soldiers|trainees)?\s*$",
    re.IGNORECASE,
)


def _is_generic_lead(text: str) -> bool:
    clean = text.strip(" ,.?!")
    if not clean:
        return True
    return bool(_GENERIC_INTRO_RE.match(clean))


# "... for Dogra class agniveers" / "... among Dogra class" — a ranking/trend
# query scoped to a community/class roster. Matched separately from the
# generic markers above because the cutpoint for splitting is the class name
# itself, not a fixed connector word.
_CLASS_NAMES = ("sikh", "dogra", "oic")
_CLASS_FILTER_RE = re.compile(rf"\b({'|'.join(_CLASS_NAMES)})\s+class\b", re.IGNORECASE)


def _build_sport_filter_re() -> "re.Pattern[str]":
    """ "... for football players" / "... among cricket players" — a
    performance/ranking query scoped to a sport roster. Matched the same way
    as `_CLASS_FILTER_RE`: the cutpoint for splitting is the sport name
    itself, not a fixed connector word, so "for"/"among"/"from" all work.
    """
    from intent_engine.intent_schema import SPORTS

    names = sorted({re.escape(name) for name in SPORTS.keys()}, key=len, reverse=True)
    return re.compile(rf"\b({'|'.join(names)})\s+players?\b", re.IGNORECASE)


_SPORT_FILTER_RE = _build_sport_filter_re()


_KNOWN_EQUIPMENT_ITEMS_LOWER: Optional[Tuple[str, ...]] = None


def _mentions_known_equipment_item(text: str) -> bool:
    """Whether `text` names a curated equipment item ("Kit Bag", "DMS Boot
    GP", ...) — see entity_extractor's ISSUED_EQUIPMENT_ITEMS /
    PROCURED_EQUIPMENT_ITEMS, the same lists it uses to extract equipmentName.
    A specific item name is unambiguous evidence of the Equipment category
    even when no generic keyword ("equipment", "holding", "issued", ...) is
    also present.
    """
    global _KNOWN_EQUIPMENT_ITEMS_LOWER
    if _KNOWN_EQUIPMENT_ITEMS_LOWER is None:
        from intent_engine.intent_schema import (
            ISSUED_EQUIPMENT_ITEMS,
            PROCURED_EQUIPMENT_ITEMS,
        )

        _KNOWN_EQUIPMENT_ITEMS_LOWER = tuple(
            item.lower() for item in ISSUED_EQUIPMENT_ITEMS + PROCURED_EQUIPMENT_ITEMS
        )
    return any(item in text for item in _KNOWN_EQUIPMENT_ITEMS_LOWER)


def _has_strong_cross_filter_marker(text: str) -> bool:
    """True for unambiguous relative-clause grammar (whose/having/belonging
    to/...) as opposed to the weaker standalone status words in
    _CROSS_FILTER_STATUS_MARKERS — see that tuple's docstring comment.
    """
    cross_marker_hits = [marker for marker in _CROSS_FILTER_MARKERS if marker in text]
    if cross_marker_hits:
        if all(marker in {"current", "if"} for marker in cross_marker_hits):
            return False
        return True
    if _CLASS_FILTER_RE.search(text):
        return True
    if _SPORT_FILTER_RE.search(text):
        return True
    return bool(_CROSS_FILTER_GENERIC_CONNECTORS.search(text))


def _has_cross_filter_marker(text: str) -> bool:
    if _has_strong_cross_filter_marker(text):
        return True
    return any(marker in text for marker in _CROSS_FILTER_STATUS_MARKERS)


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
    "90 %",
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
    dependent_intent: bool = False
    sub_requests: List[Dict[str, Any]] = field(default_factory=list)
    conversational: bool = False

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["confidence"] = round(float(self.confidence), 2)
        return payload


def _infer_category(text: str, entities: Dict[str, Any]) -> Optional[str]:
    if any(
        token in text
        for token in ("overweight", "underweight", "obese", "medically unfit")
    ):
        # Checked before the personaldetail block below — "overweight" and
        # "underweight" contain "weight" as a substring, which would
        # otherwise match personaldetail's bare "weight" keyword first and
        # misclassify a BMI-category clause as PersonalDetail instead of
        # Medical.
        return "Medical"
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
            "height",
            "weight",
            "hobby",
            "hobbies",
            "skill",
            "skills",
            "sports",
            "cricket",
            "basketball",
            "football",
            "volleyball",
            "hockey",
        )
    ):
        return "personaldetail"
    if entities.get("grading"):
        return "Performance"
    if any(
        token in text
        for token in (
            "topper",
            "toppers",
            "top performer",
            "top performers",
            "highest performer",
            "best performer",
            "top scorer",
            "highest scorer",
            "best scorer",
            "top scoring",
            "highest scoring",
            "chart toppers",
        )
    ):
        return "Performance"
    if entities.get("bmi_category") or entities.get("blood_group"):
        return "Medical"
    if entities.get("leave_type"):
        return "Leave"
    if "leave" in text or "absconded" in text or "absent" in text:
        return "Leave"
    if _mentions_known_equipment_item(text):
        # A curated, unambiguous item name ("Kit Bag", "DMS Boot GP", ...) is
        # a strong signal on its own — checked before the "class"+"roster"
        # Skills shortcut below so "Kit Bag holders and the Sikh class
        # roster" resolves this clause as Equipment instead of falling
        # through to None (and, via the sub-request splitter's fallback,
        # inheriting the *other* clause's Skills category).
        return "Equipment"
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
            "holder",
            "holders",
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
            "parade",
            "roll call",
            "muster",
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
            "scoring",
            "scored",
        )
    ):
        return "Performance"
    if any(token in text for token in _PERFORMANCE_SECTION_TOKENS):
        # Bare section name ("BPET"/"PPT"/"Firing"/"Drill") with no other
        # Performance keyword — happens often once a query gets split into
        # fragments, e.g. "scoring excellent in BPET" loses "score" itself
        # to a different check but "bpet" alone is still unambiguous.
        return "Performance"
    if any(token in text for token in ("platoon", "batch", "company", "class", "unit")):
        return "PersonalDetails"
    return None



def _infer_operation(text: str, entities: Dict[str, Any]) -> str:
    category = _infer_category(text, entities)
    if category == "Equipment" and (
        entities.get("agniveerNo") or entities.get("agniveer_no")
    ):
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
    if any(
        token in text for token in ("returned", "poor condition", "damaged", "broken")
    ):
        return "returned"
    if any(
        token in text
        for token in (
            "overdue",
            "currently issued",
            "currently holding",
            "holding",
            "issued",
            "possessing",
            "carrying",
            "keeping",
            "issued with",
        )
    ):
        return "holding"
    if "current leave" in text or "leave today" in text or "on leave" in text:
        return "current"
    if "absconded" in text:
        return "absconded"
    if any(marker in text for marker in _TREND_MARKERS):
        return "trend"
    if any(
        token in text
        for token in (
            "today's schedule",
            "today schedule",
            "current schedule",
            "training schedule",
            "company schedule",
        )
    ):
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
        suffix = right[prep_match.start() :].strip().rstrip(".").strip()
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


def _match_known_disease_prefix(text: str) -> Optional[str]:
    """Return the longest curated disease name (see entity_extractor's
    `_KNOWN_DISEASES`) that `text` starts with, or None. Multi-word disease
    names ("viral fever", "kidney stone", "chicken pox", ...) must not be
    split across fragments by a naive single-word grab after "with"/"from",
    so the caller uses this to know exactly how many words the diagnosis
    value spans.
    """
    from intent_engine.entity_extractor import _KNOWN_DISEASES

    stripped = text.lstrip()
    stripped_lower = stripped.lower()
    best: Optional[str] = None
    for d in _KNOWN_DISEASES:
        if not stripped_lower.startswith(d):
            continue
        end = len(d)
        if end < len(stripped_lower) and stripped_lower[end].isalnum():
            continue  # partial word match, e.g. "flu" inside "fluster"
        if best is None or len(d) > len(best):
            best = d
    return stripped[: len(best)] if best else None


# Status/state words that determine WHICH operation a category resolves to
# (e.g. Verification/Completed vs Verification/Pending). Used only to pull a
# cutpoint in `_split_by_category_signal` back so this word stays attached to
# the fragment it governs — see that function's docstring.
_STATUS_ADJECTIVES = frozenset(
    {
        "completed",
        "pending",
        "rejected",
        "sent",
        "unverified",
        "overdue",
        "issued",
        "procured",
        "returned",
        "unassigned",
        "distributed",
        "top",
        "bottom",
        "highest",
        "lowest",
        "best",
        "worst",
        "improved",
        "improvement",
        "dropped",
        "drop",
        "failed",
        "fail",
        "passed",
        "pass",
        "medically",
        "medical",
        "currently",
    }
)

_SINGLE_WORD_CATEGORY_SIGNALS: Optional[Dict[str, str]] = None


def _single_word_category_signals() -> Dict[str, str]:
    """Bare single-word category signal -> category, built once from
    `_CATEGORY_SIGNALS` (query_planner.py's category keyword table). A word
    that names more than one category (ambiguous) is dropped rather than
    guessed."""
    global _SINGLE_WORD_CATEGORY_SIGNALS
    if _SINGLE_WORD_CATEGORY_SIGNALS is not None:
        return _SINGLE_WORD_CATEGORY_SIGNALS
    from intent_engine.query_planner import _CATEGORY_SIGNALS

    mapping: Dict[str, str] = {}
    ambiguous: set = set()
    for cat, signals in _CATEGORY_SIGNALS.items():
        if cat == "Schedule":
            continue
        for sig in signals:
            if " " in sig:
                continue
            if sig in mapping and mapping[sig] != cat:
                ambiguous.add(sig)
            else:
                mapping[sig] = cat
    for sig in ambiguous:
        mapping.pop(sig, None)
    _SINGLE_WORD_CATEGORY_SIGNALS = mapping
    return mapping


def _split_by_category_signal(text: str) -> Optional[List[str]]:
    """Fallback split for a cross-filter remainder that still names two
    distinct categories but has no "and"/comma to cut on — e.g. "are present
    today completed police verification?" (Attendance + Verification) or
    "fever are currently on leave?" (Medical + Leave). Without this, the
    whole remainder is classified as a single fragment and whichever
    category scores highest wins, silently dropping the other condition.

    Finds the two categories' earliest single-word signal positions and cuts
    between them, pulling the cutpoint back over any immediately-preceding
    status word ("completed", "pending", ...) within a small window so the
    operation-determining word stays with its own category's fragment
    instead of being stranded on the wrong side of the cut.
    """
    signals = _single_word_category_signals()
    words = list(re.finditer(r"[a-zA-Z0-9']+", text))
    if len(words) < 3:
        return None
    norm_words = [w.group(0).lower() for w in words]

    positions: Dict[str, int] = {}
    for i, w in enumerate(norm_words):
        cat = signals.get(w)
        if cat and cat not in positions:
            positions[cat] = i
    if len(positions) != 2:
        return None

    ordered = sorted(positions.items(), key=lambda kv: kv[1])
    cut_word_idx = ordered[1][1]
    window = 3
    start_scan = max(0, cut_word_idx - window)
    for i in range(cut_word_idx - 1, start_scan - 1, -1):
        if norm_words[i] in _STATUS_ADJECTIVES:
            cut_word_idx = i

    if cut_word_idx <= 0 or cut_word_idx >= len(words):
        return None

    cut_char = words[cut_word_idx].start()
    left = text[:cut_char].strip(" ,")
    right = text[cut_char:].strip(" ,")
    if not left or not right:
        return None
    return [left, right]


def _extract_sub_requests(
    text: str,
    category: Optional[str],
    operation: str,
    entities: Dict[str, Any],
) -> List[Dict[str, Any]]:
    # Dependent/nested clause ("...additionally check whether any of them...")
    # — split on the connecting marker first. None of the generic cross-filter
    # cutpoints below (who/with/among/...) appear in this phrasing, so without
    # this branch the whole query falls through as a single un-split
    # fragment and the dependent leg (e.g. the verification check) is lost.
    if _has_dependent_backref(text):
        for marker in _MULTI_INDEPENDENT_MARKERS:
            if marker not in text:
                continue
            parts = [p.strip(" ,.") for p in text.split(marker) if p.strip(" ,.")]
            if len(parts) >= 2 and _has_dependent_backref(parts[-1]):
                head = parts[0]
                tail = " ".join(parts[1:])
                # A dependent tail can itself contain further siblings —
                # "Then tell me which of them are medically unfit. Also show
                # their attendance." — every clause after the primary request
                # is a dependent child of it, not nested further, so split
                # the tail on any remaining marker into flat sibling
                # fragments instead of leaving them merged into one.
                tail_fragments = _split_on_connectors(
                    tail, list(_MULTI_INDEPENDENT_MARKERS)
                )
                if len(tail_fragments) < 2:
                    tail_fragments = [tail]
                result = [
                    {
                        "fragment": head,
                        "category": category,
                        "operation": operation,
                        "entities": entities,
                    }
                ]
                for frag in tail_fragments:
                    frag_clean = frag.strip(" ,.")
                    if not frag_clean:
                        continue
                    result.append(
                        {
                            "fragment": frag_clean,
                            "category": _infer_category(frag_clean, {}),
                            "operation": _infer_operation(frag_clean, {}),
                            "entities": entities,
                        }
                    )
                if len(result) >= 2:
                    return result

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

    # Also gated on _MULTI_INDEPENDENT_MARKERS, not just cross-filter markers:
    # this splitter is the shared fragment-extraction path both cross_filter
    # AND multi_independent classification build on, but its entry gate used
    # to rely solely on cross_filter_marker words that ALSO happened to be
    # multi-independent connectors (also/plus/together/then/...). Once those
    # duplicates were removed from _CROSS_FILTER_MARKERS (they were false
    # "strong cross-filter evidence" — see that tuple's docstring), a plain
    # multi-independent phrasing with no OTHER cross-filter word ("...status
    # and police verification together?") stopped entering this block at
    # all and fell back to a much weaker split, losing one side of the
    # request. Checking both marker sets here keeps the split working for
    # both intents while the (separate) cross_filter *strength* gate
    # elsewhere still requires a real cross-filter marker.
    if _has_cross_filter_marker(text) or any(
        marker in text for marker in _MULTI_INDEPENDENT_MARKERS
    ):
        parts = []
        current = text
        for sep in (
            r"\band\s+who\b",
            r"\band\s+whose\b",
            r"\band\s+whom\b",
            r"\band\s+that\b",
            r"\band\s+which\b",
            # Possessive-pronoun clause ("...and their police verification is
            # verified") is grammatically the same relative-clause shape as
            # "and whose ..." above, but names the subject with a pronoun
            # instead of "whose". Without this, the only remaining separator
            # match is the trailing status word ("verified"), which leaves
            # the whole first clause (including "and their police
            # verification is") in one fragment that then gets
            # mis-classified by whichever category's words happen to score
            # higher in that leftover text.
            r"\band\s+their\b",
            r"\band\s+his\b",
            r"\band\s+her\b",
            r"\band\s+its\b",
            # Concession connectors ("...failed Firing but still have issued
            # equipment", "...overweight despite scoring Excellent") —
            # unambiguous cross-filter cutpoints (see _CROSS_FILTER_MARKERS),
            # but that list only drives cross_filter_intent DETECTION, not
            # the actual text split; without an entry here too, the whole
            # query stays one un-split fragment and the >= 2 valid_ops
            # cross-filter execution path never gets enough legs to run.
            r"\bbut\s+still\b",
            r"\byet\s+still\b",
            r"\bdespite\b",
            r"\bon\s+top\s+of\s+that\b",
            # Plain "but" as a contrastive clause separator ("...scored
            # Excellent in BPET but overweight and have pending
            # verification") — same cross-filter semantics as "but still"
            # above (an ADDITIONAL condition on the same subject), just
            # without the "still". _CROSS_FILTER_MARKERS already treats
            # "but who"/"but whose"/"but having" as strong DETECTION
            # evidence; without a matching split here the whole clause
            # after "but" stays fused into one fragment and whichever
            # category's keywords happen to score higher wins, silently
            # dropping the other condition.
            r"\bbut\s+who\b",
            r"\bbut\s+whose\b",
            r"\bbut\s+having\b",
            r"\bbut\b",
            r"\band\s+having\b",
            r"\band\s+belonging(?:\s+to)?\b",
            r"\band\s+belongs?\s+to\b",
            r"\band\s+from\b",
            r"\band\s+with\b",
            r"\band\s+also\b",
            r"\bas\s+well\s+as\b",
            r"\balong\s+with\b",
            r"\btogether\s+with\b",
            r"\bplus\b",
            r"\bwho\b",
            r"\bwhose\b",
            r"\bwhom\b",
            r"\bthat\b",
            r"\bwhich\b",
            r"\bwhere\b",
            r"\bwherein\b",
            r"\bwhereby\b",
            r"\bwith\b",
            r"\bwithout\b",
            r"\bamong\b",
            r"\bwithin\b",
            r"\bfrom\s+(?:them|those|these)\b",

            r"\bout\s+of\b",
            r"\binside\b",
            r"\bunder\b",
            r"\bbelonging(?:\s+to)?\b",
            r"\bbelongs?\s+to\b",
            r"\bbelong\s+to\b",
            r"\bbelonged\s+to\b",
            r"\bhaving\b",
            r"\bhas\b",
            r"\bhave\b",
            r"\bhad\b",
            r"\bholding\b",
            r"\bholds?\b",
            r"\bheld\b",
            r"\bpossessing\b",
            r"\bpossesses?\b",
            r"\bcarrying\b",
            r"\bcarries\b",
            r"\bkeeping\b",
            r"\bkeeps?\b",
            r"\bowning\b",
            r"\bowns?\b",
            r"\bassigned(?:\s+to)?\b",
            r"\bissued(?:\s+with)?\b",
            r"\ballocated(?:\s+to)?\b",
            r"\bgiven(?:\s+to)?\b",
            r"\bsuffering(?:\s+from)?\b",
            r"\bsuffered(?:\s+from)?\b",
            r"\bdiagnosed(?:\s+with)?\b",
            r"\badmitted(?:\s+to)?\b",
            r"\bhospitalized(?:\s+for)?\b",
            r"\bscored\b",
            r"\bscoring\b",
            r"\bscores?\b",
            r"\bpassed\b",
            r"\bpassing\b",
            r"\bpasses?\b",
            r"\bfailed\b",
            r"\bfailing\b",
            r"\bfails?\b",
            r"\bqualified\b",
            r"\bqualifying\b",
            r"\bqualifies?\b",
            r"\bdisqualified\b",
            r"\bverified\b",
            r"\bapproved\b",
            r"\brejected\b",
            r"\bon\s+leave\b",
            r"\babsent\b",
            r"\bpresent\b",
            r"\battending\b",
            r"\battended\b",
        ):
            # Try every occurrence of this separator, not just the first —
            # a query can repeat the same relative pronoun ("...whose bmi is
            # normal whose police verification is verified"), and splitting
            # on the first occurrence can leave a generic lead ("give me
            # agniveers") even though a later occurrence of this exact
            # separator is the real cutpoint. Previously a generic-lead
            # first occurrence abandoned this separator entirely and fell
            # through to a different, lower-priority one (here: "verified"
            # instead of the second "whose"), merging both clauses into a
            # single mis-classified fragment instead of splitting them.
            matched_this_sep = False
            for sep_match in re.finditer(sep, current, flags=re.IGNORECASE):
                lead = current[: sep_match.start()].strip(" ,")
                if not lead:
                    continue
                matched_word = current[sep_match.start() : sep_match.end()]
                remainder = current[sep_match.end() :].strip(" ,")
                reattached_to_remainder = True
                if sep == r"\bon\s+top\s+of\s+that\b":
                    # Pure connective phrase, not content — reattaching it
                    # would let "top" (a Performance ranking keyword)
                    # confuse the remainder's own category-signal split,
                    # e.g. "on top of that currently on leave" mis-cutting
                    # into "on top of that" (Performance, from "top") /
                    # "currently on leave" instead of staying one Leave
                    # fragment.
                    reattached_to_remainder = False
                elif sep == r"\bwith\b" and re.search(
                    r"\bdiagnosed\s*$", lead, flags=re.IGNORECASE
                ):
                    disease = _match_known_disease_prefix(remainder)
                    if disease:
                        lead = f"{lead} with {disease}"
                        remainder = remainder[len(disease) :].strip()
                        reattached_to_remainder = False
                    else:
                        value_match = re.match(
                            r"(\S+)\s*(.*)", remainder, flags=re.DOTALL
                        )
                        if value_match and value_match.group(1):
                            lead = f"{lead} with {value_match.group(1)}"
                            remainder = value_match.group(2).strip()
                            reattached_to_remainder = False
                elif sep in (r"\bsuffering\b", r"\bsuffered\b"):
                    from_match = re.match(
                        r"from\s+(.*)", remainder, flags=re.IGNORECASE | re.DOTALL
                    )
                    if from_match:
                        after_from = from_match.group(1)
                        verb = "suffering" if sep == r"\bsuffering\b" else "suffered"
                        disease = _match_known_disease_prefix(after_from)
                        if disease:
                            lead = f"{lead} {verb} from {disease}"
                            remainder = after_from[len(disease) :].strip()
                            reattached_to_remainder = False
                        else:
                            single_match = re.match(
                                r"(\S+)\s*(.*)", after_from, flags=re.DOTALL
                            )
                            if single_match and single_match.group(1):
                                lead = f"{lead} {verb} from {single_match.group(1)}"
                                remainder = single_match.group(2).strip()
                                reattached_to_remainder = False

                if _is_generic_lead(lead):
                    continue

                if reattached_to_remainder:
                    remainder = f"{matched_word} {remainder}".strip()
                parts.append(lead)
                current = remainder
                matched_this_sep = True
                break

            if matched_this_sep:
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
            roster_match = (
                min(roster_matches, key=lambda m: m.start()) if roster_matches else None
            )
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
                    # fragment with an empty lead. A bare noun immediately
                    # trailing the roster phrase ("cricket players LIST")
                    # completes the same request rather than starting a new
                    # one — absorb it too, or it strands as its own
                    # meaningless one-word fragment once the remainder gets
                    # split on "and" below.
                    _roster_end = roster_match.end()
                    _trailing_noun = re.match(
                        r"\s*(?:list|roster|stats|details|summary)\b",
                        current[_roster_end:],
                        flags=re.IGNORECASE,
                    )
                    if _trailing_noun:
                        _roster_end += _trailing_noun.end()
                    parts.append(
                        current[roster_match.start() : _roster_end].strip()
                    )
                    current = current[_roster_end:].strip()
            else:
                parts = [current]
                current = ""

        if current:
            and_parts = [
                p.strip(" ,")
                for p in re.split(
                    r"\s*,\s*(?:and\s+)?|\s+\band\b\s+", current, flags=re.IGNORECASE
                )
                if p.strip(" ,")
            ]
            if len(and_parts) < 2:
                # No explicit "and"/comma, but the remainder can still smuggle
                # a second category's whole clause with no connector at all
                # (e.g. "are present today completed police verification?").
                # Left unsplit, only the highest-scoring category survives
                # classification and the other condition is silently dropped.
                cat_split = _split_by_category_signal(current)
                if cat_split:
                    and_parts = cat_split
            parts.extend(and_parts)
        else:
            new_parts = []
            for p in parts:
                and_parts = [
                    ap.strip(" ,")
                    for ap in re.split(
                        r"\s*,\s*(?:and\s+)?|\s+\band\b\s+", p, flags=re.IGNORECASE
                    )
                    if ap.strip(" ,")
                ]
                new_parts.extend(and_parts)
            parts = new_parts

        # A fragment already committed to `parts` via the main loop's single
        # break can itself still hide a comma/and-joined list — the block
        # above only re-examines `current` (the remainder AFTER that first
        # split), not fragments collected before it. E.g. "who improved in
        # bpet, the disease statistics, and the rejected verifications"
        # splits first on "rejected", leaving "who improved in bpet, the
        # disease statistics, and the" as one whole fragment that still
        # hides two distinct requests. Re-splitting is safe here since a
        # comma/explicit "and" is already trusted as an intentional
        # separator everywhere else in this function.
        _re_split_parts: List[str] = []
        for p in parts:
            if "," in p or re.search(r"\band\b", p, flags=re.IGNORECASE):
                sub_parts = [
                    sp.strip(" ,")
                    for sp in re.split(
                        r"\s*,\s*(?:and\s+)?|\s+\band\b\s+", p, flags=re.IGNORECASE
                    )
                    if sp.strip(" ,")
                ]
                if len(sub_parts) >= 2:
                    _re_split_parts.extend(sub_parts)
                    continue
            _re_split_parts.append(p)
        parts = _re_split_parts

        final_parts = []
        for p in parts:
            p_clean = p.strip(" ,")
            if p_clean and p_clean.lower() not in (
                "who",
                "whose",
                "whom",
                "that",
                "which",
                "with",
                "without",
                "and",
                "plays",
                "suffering",
                "suffered",
                "having",
                "has",
                "have",
                "had",
                "belonging",
                "belongs",
                "belong",
                "from",
                "among",
                "within",
                "also",
                "plus",
                "where",
                "as",
                "is",
                "are",
                "was",
                "were",
                "been",
                "being",
                "the",
                "a",
                "an",
                "or",
                "in",
                "on",
                "at",
                "to",
                "by",
                "for",
                "of",
            ):

                final_parts.append(p_clean)

        return [
            (
                {
                    "fragment": p,
                    "category": _infer_category(p, {}) or category,
                    "operation": _infer_operation(p, {})
                    if _infer_operation(p, {}) != "lookup"
                    else operation,
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
                for part in re.split(r"\s*,\s*(?:and\s+)?|\s+\band\b\s+", text)
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

    sub_requests = _extract_sub_requests(text, category, operation, entities)
    sub_req_cats = {
        sub.get("category")
        for sub in sub_requests
        if isinstance(sub, dict) and sub.get("category")
    }

    # cross_filter requires a marker AND at least 2 distinct inferred categories
    _cross_marker_hit = _has_cross_filter_marker(text)
    _cross_marker_strong = _has_strong_cross_filter_marker(text)
    cross_filter_intent = False
    if _cross_marker_hit:
        from intent_engine.query_planner import _detect_categories as _dc

        _cf_cats = _dc(text)
        _distinct_sections = _distinct_performance_sections(text)
        if (
            len(set(_cf_cats[:3])) >= 2
            or _distinct_sections >= 2
            or len(sub_req_cats) >= 2
        ):
            cross_filter_intent = True

            # _dc(text) is a coarse, whole-text category scan and can find 2
            # categories that the PRECISE per-fragment splitter (sub_requests,
            # computed above) resolves to the same one — "Which sport has the
            # best performers?" scans as {Skills, Performance} from "sport"
            # and "performers" in isolation, but both actual fragments
            # ("which sport" / "has the best performers") classify as
            # Performance once split. That's not 2 filters to intersect, it's
            # one ranking query, so don't commit to cross_filter on the
            # coarse signal alone once the precise one contradicts it.
            if (
                len(set(_cf_cats[:3])) >= 2
                and _distinct_sections < 2
                and len(sub_req_cats) <= 1
                and len(sub_requests) >= 2
            ):
                cross_filter_intent = False



    # A later clause naming its own report/analytics noun ("...and equipment
    # summary") is asking for an independent output, not filtering the first
    # clause's subject — even though a weak status word (rejected/issued/...)
    # matched above. Only overrides a STATUS-only hit; an unambiguous strong
    # marker (whose/having/belonging to/...) is trusted even with a trailing
    # report noun.
    if cross_filter_intent and not _cross_marker_strong and " and " in text:
        _cf_clause_parts = [
            part.strip(" ,") for part in text.split(" and ") if part.strip(" ,")
        ]
        if len(_cf_clause_parts) >= 2 and any(
            marker in _cf_clause_parts[-1] for marker in _REPORT_OUTPUT_MARKERS
        ):
            cross_filter_intent = False

    # Weak status-style cues ("current", "pending", "approved", etc.) can
    # appear in ordinary multi-section requests such as "attendance and
    # current leave records". Those should not be forced down the
    # cross-filter path unless a strong relationship marker is present.
    if cross_filter_intent and not _cross_marker_strong:
        cross_filter_intent = False

    # "and how many"/"and how much" always introduces a second, independent
    # question — unlike "who"/"which"/"that", "how many" has no relative-
    # clause reading, so it can't be filtering the first clause's subject.
    # E.g. "Who topped PPT and how many are present today in Lakhwinder
    # company?" is two unrelated questions, not an intersection — without
    # this, "who" and "present" both match _CROSS_FILTER_GENERIC_CONNECTORS
    # as "strong" markers and the weak-marker guard above doesn't help.
    if cross_filter_intent and re.search(
        r"\band\s+how\s+(?:many|much)\b", text, flags=re.IGNORECASE
    ):
        cross_filter_intent = False

    # "A, B, and C" — an Oxford-comma list (2+ commas, the last one
    # immediately followed by "and") signals list coordination (a run of
    # independent noun phrases), not a relative-clause chain modifying one
    # subject. Requiring ", and " (not just 2+ commas) excludes a
    # parenthetical aside like "...and are, on top of that, currently on
    # leave" — that also has 2 commas, but they set off an inserted phrase,
    # not a third list item, so it must stay cross-filter. Combined with the
    # precise per-fragment splitter also finding 2+ distinct categories
    # (sub_req_cats), trust the list reading over a generic-connector word
    # ("who"/"which"/...) landing inside one of the comma segments — e.g.
    # "Show top performers in BPET, who is on leave today, and the
    # equipment stats." is three independent reports, not an intersection,
    # even though "who" alone reads as a strong cross-filter marker.
    # But not when the relative pronoun governing the whole sentence sits
    # ONLY before the first comma ("Agniveers WHO failed Firing, are
    # overweight, and on leave") — every comma segment after it is a bare
    # verb-phrase continuation of that same "who", not its own independent
    # clause, so this is one 3-way intersection, not three list items. If
    # "who"/"which"/... shows up again in a LATER segment instead ("...,
    # WHO is on leave today, ..."), that later segment is introducing its
    # own new clause, so the list reading still applies.
    _first_comma = text.find(",")
    _rel_pronoun_only_before_first_comma = False
    if _first_comma != -1:
        _rel_pronoun_re = re.compile(
            r"\b(?:who|whose|which|that|with|having)\b", re.IGNORECASE
        )
        # A bare sentence-initial "who" ("Who improved in BPET, ...") is the
        # interrogative question word, not a relative pronoun governing a
        # clause chain — that reading requires a preceding noun ("Agniveers
        # who..."), which is impossible when "who" is the very first word.
        # Other pre-comma matches still count as clause-governing.
        _pre_comma_governing_match = any(
            not (m.start() == 0 and m.group(0).lower() == "who")
            for m in _rel_pronoun_re.finditer(text[:_first_comma])
        )
        if _pre_comma_governing_match and not _rel_pronoun_re.search(
            text[_first_comma:]
        ):
            _rel_pronoun_only_before_first_comma = True

    if (
        cross_filter_intent
        and text.count(",") >= 2
        and re.search(r",\s*and\s+", text, flags=re.IGNORECASE)
        and len(sub_req_cats) >= 2
        and not _rel_pronoun_only_before_first_comma
    ):
        cross_filter_intent = False

    # Plain "A and B" (one bare "and", not "and who"/"and whose" continuing
    # a relative clause) where each side independently names its own
    # category, and the second clause carries no continuation cue
    # ("still"/"already"/"yet"/"too" — the tell that it's ADDING a
    # condition to the first clause's subject rather than asking something
    # new), is two independent requests, not an intersection — even when a
    # strong marker (a bare "who", the class/sport-roster pattern, a bare
    # "of", ...) fires inside one of the two clauses. E.g. "Latest
    # distribution and who got excellent in firing." names Distribution in
    # clause 1 and Performance in clause 2 with nothing tying them together.
    if cross_filter_intent and " and " in text and not _rel_pronoun_only_before_first_comma:
        _ab_parts = [p.strip(" ,?.") for p in text.split(" and ", 1)]
        if (
            len(_ab_parts) == 2
            and all(_ab_parts)
            and not re.search(
                r"\b(?:still|already|yet|too|despite|on\s+top\s+of\s+that)\b",
                _ab_parts[1],
            )
        ):
            # Same "single continuous clause chain" reasoning as
            # _rel_pronoun_only_before_first_comma, applied at this "and"
            # cutpoint instead of a comma: "Agniveers who scored Excellent
            # in BPET but overweight and have pending verification" governs
            # all three legs with one "who" — the "and" here just adds a
            # third leg to the same chain, it doesn't open an unrelated
            # second topic. Only treat this as a real A-and-B split when the
            # governing pronoun ISN'T already sitting in clause 1 alone.
            _ab_pronoun_re = re.compile(
                r"\b(?:who|whose|which|that|with|having)\b", re.IGNORECASE
            )
            # Same bare-sentence-initial-"who" exclusion as above: "Who is
            # on leave today and what is the BMI distribution?" opens with
            # the interrogative "who", not a relative pronoun governing
            # clause 1 — that reading needs a preceding noun ("Agniveers
            # who..."), impossible when "who" is clause 1's first word.
            _ab_governing_match = any(
                not (m.start() == 0 and m.group(0).lower() == "who")
                for m in _ab_pronoun_re.finditer(_ab_parts[0])
            )
            # A repeated pronoun in clause 2 ("...agniveers who have high
            # score in firing and who had malaria") is a second relative
            # clause chained onto the SAME governing subject, not a fresh
            # one — English coordinates "who X and who Y" this way all the
            # time. That's different from clause 1 having no governing
            # match at all (position-0 "who", or no pronoun whatsoever),
            # where a pronoun anywhere in clause 2 doesn't prove anything.
            # A bare "and having ..." with no pronoun in clause 1 at all
            # ("Agniveers treated for fever and having pending
            # verification") is unambiguous the other way: "having" can't
            # open an independent question, so it always continues clause
            # 1's implicit subject.
            _ab_pronoun_governs = _ab_governing_match or bool(
                re.match(r"^having\b", _ab_parts[1], re.IGNORECASE)
            )
            if not _ab_pronoun_governs:
                _ab_cat1 = _infer_category(_ab_parts[0], {})
                _ab_cat2 = _infer_category(_ab_parts[1], {})
                if _ab_cat1 and _ab_cat2 and _ab_cat1 != _ab_cat2:
                    cross_filter_intent = False

    # "which <group> has the most/highest/lowest ..." asks to RANK groups by
    # a single metric (a distribution/group-by question) — not to intersect
    # two independent conditions. Without this guard, the group word (e.g.
    # "unit") and the metric word (e.g. "absconded") register as two
    # distinct categories and wrongly satisfy the cross-filter gate above.
    if re.search(
        r"\bwhich\s+\w+\s+has\s+(?:the\s+)?(?:most|highest|lowest|least)\b", text
    ):
        cross_filter_intent = False

    def _distinct_category_count(clause_parts: List[str]) -> int:
        clause_categories = []
        for part in clause_parts:
            clause_entities = extract_entities(part, semantic={})
            clause_categories.append(_infer_category(part, clause_entities))
        distinct = len({cat for cat in clause_categories if cat})
        if distinct < 2:
            distinct = max(
                distinct, _distinct_performance_sections(" ".join(clause_parts))
            )
        return distinct

    # Every multi-independent marker — even weak ones like "also"/"then" — is
    # gated by requiring the clauses split on it to reference >= 2 distinct
    # categories. Words like "also"/"plus"/"as well as" are common enough in
    # single-request phrasing that the word alone proves nothing; the gate is
    # what makes this reliable, not the vocabulary.
    multi_intent = False
    dependent_intent = False
    # First pass: does splitting on ANY marker reveal a dependent
    # back-reference in a later clause? Checked across every marker (not just
    # the first oyne present) because a 3+ clause chain — "Find the top 20
    # performers. Then tell me which of them are medically unfit. Also show
    # their attendance." — only exposes the "of them" back-reference when
    # split on "then"; splitting on the earlier-matching "also" first would
    # hide it and wrongly fall through to multi_intent classification below.
    for marker in _MULTI_INDEPENDENT_MARKERS:
        if marker not in text:
            continue
        clause_parts = [
            part.strip(" ,") for part in text.split(marker) if part.strip(" ,")
        ]
        if len(clause_parts) >= 2 and any(
            _has_dependent_backref(part) for part in clause_parts[1:]
        ):
            dependent_intent = True
            break

    if not dependent_intent:
        for marker in _MULTI_INDEPENDENT_MARKERS:
            if marker not in text:
                continue
            clause_parts = [
                part.strip(" ,") for part in text.split(marker) if part.strip(" ,")
            ]
            if len(clause_parts) >= 2 and _distinct_category_count(clause_parts) >= 2:
                multi_intent = True
                break

    if not multi_intent and not dependent_intent and " and " in text:
        clause_parts = [
            part.strip(" ,") for part in text.split(" and ") if part.strip(" ,")
        ]
        if len(clause_parts) >= 2:
            if any(_has_dependent_backref(part) for part in clause_parts[1:]):
                dependent_intent = True
            elif _distinct_category_count(clause_parts) >= 2:
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
    elif dependent_intent:
        # A back-referencing clause ("...additionally check whether any of
        # them...") is a nested/dependent request, not an independent second
        # query — it must be intersected against the primary request's result
        # set, exactly like cross_filter. Execution reuses the cross_filter
        # machinery; intent_kind stays "nested" so the distinction is visible
        # in telemetry/canonical output.
        query_type = "cross_filter"
        complexity = "nested"
        intent_kind = "nested"
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
        dependent_intent=dependent_intent,
        sub_requests=sub_requests,
        conversational=False,
    )
    return result.to_dict()
