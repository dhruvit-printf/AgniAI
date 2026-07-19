"""
intent_classifier.py
====================

Single responsibility: determine Category, Operation, and ResponseType.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from query_normalizer import clean_query
from query_understanding_engine import understand_query

from .intent_schema import (
    CATEGORY_ENTITY_HINTS,
    CATEGORY_KEYWORDS,
    DEFAULT_DETAILED_CATEGORIES,
    DETAILED_KEYWORDS,
    OFFICIAL_CATEGORIES,
    OPERATION_SYNONYMS,
    OPERATIONS_BY_CATEGORY,
    RELATIVE_DATE_PHRASES,
    RESPONSE_TYPE_DEFAULT,
)

logger = logging.getLogger(__name__)

_ACTION_HINTS = (
    "show",
    "list",
    "give",
    "find",
    "tell",
    "compare",
    "what",
    "who",
    "which",
    "how",
    "count",
    "top",
    "highest",
    "lowest",
    "best",
    "worst",
    "average",
    "summary",
    "report",
    "details",
    "status",
    "trend",
    "distribution",
    "performance",
    "score",
    "marks",
    "leave",
    "medical",
    "attendance",
    "verification",
    "equipment",
    "skills",
    "overall",
    "schedule",
    "disqualified",
    "personal",
)


def _canonical_text(query: str) -> str:
    return clean_query(query).lower()


def _phrase_score(text: str, phrase: str) -> int:
    if not phrase:
        return 0
    phrase_norm = _canonical_text(phrase)
    if not phrase_norm:
        return 0
    if phrase_norm not in text:
        # Fall back to a whitespace-insensitive match so a hint written as
        # one compact token (e.g. "bycompany") still identifies the
        # naturally spaced phrase in the query text ("by company"). Only
        # reached when the exact spaced phrase wasn't found, so every
        # existing (already-spaced) phrase keeps its original score.
        compact_phrase = phrase_norm.replace(" ", "")
        compact_text = text.replace(" ", "")
        if compact_phrase and compact_phrase in compact_text:
            words = len(phrase_norm.split())
            base = words * 12 + len(compact_phrase)
            occurrences = max(1, compact_text.count(compact_phrase))
            return base * occurrences
        return 0

    words = len(phrase_norm.split())
    base = words * 12 + len(phrase_norm)
    occurrences = max(1, text.count(phrase_norm))
    boundary_bonus = 6 if re.search(rf"\b{re.escape(phrase_norm)}\b", text) else 0
    return base * occurrences + boundary_bonus


def _word_family_score(text_word: str, phrase: str) -> int:
    """Typo/word-family fallback for a single-word keyword against one query token.

    Catches two things a plain substring match misses:
      - Misspellings ("levae", "atendance", "eqipment") via edit distance.
      - Word-family variants sharing a root ("performer"/"performers"/
        "performance" all share "perform") via a long common prefix.
    """
    if len(phrase) < 4 or len(text_word) < 4 or text_word == phrase:
        return 0

    common_prefix = 0
    for a, b in zip(text_word, phrase):
        if a != b:
            break
        common_prefix += 1
    if common_prefix >= 5 or common_prefix >= min(len(text_word), len(phrase)):
        return 12

    # A flat "2 edits allowed" budget here previously let unrelated common
    # words collide (e.g. "right" ~ "weight", both edit-distance 2 under
    # plain Levenshtein) purely because they're a similar length. Using
    # transposition-aware distance and capping at 1 edit fixes that false
    # positive while still catching an adjacent-letter swap typo like
    # "levae" for "leave" (a single transposition costs 1, not 2, here).
    max_distance = 1 if len(phrase) <= 7 else 2
    if (
        abs(len(text_word) - len(phrase)) <= max_distance
        and _damerau_levenshtein(text_word, phrase) <= max_distance
    ):
        return 10
    return 0


def _fuzzy_keyword_bonus(query_text: str, phrases: Iterable[str]) -> int:
    """Typo/word-family credit for a whole keyword list, capped to the single
    best match per query word rather than summed per phrase.

    Dictionaries intentionally list several word-family variants of the same
    root (e.g. distribute/distributed/distribution). Without this cap, one
    query word would earn fuzzy credit against every sibling variant and
    inflate that category's score well past what the same word should be
    worth once.
    """
    # Exclude "by<word>" compact compound tokens ("byagniveer", "bycompany")
    # — these exist purely so _phrase_score's whitespace-insensitive fallback
    # can match the spaced phrase "by agniveer"/"by company"; they are not
    # real words. Fed through word-family fuzzy matching, the "by" prefix is
    # just a 2-character insertion, so any query containing the bare word
    # ("agniveer", present in nearly every query in this system) picks up a
    # false fuzzy-match credit for a category it has nothing to do with.
    single_word_phrases = [
        p
        for p in phrases
        if p and " " not in p and not (p.startswith("by") and len(p) > 3)
    ]
    if not single_word_phrases:
        return 0
    total = 0
    for word in query_text.split():
        best = 0
        for phrase in single_word_phrases:
            matched = _word_family_score(word, phrase)
            if matched > best:
                best = matched
        total += best
    return total


def _semantic_score(candidate: str, semantic_value: Optional[str]) -> int:
    if not semantic_value:
        return 0
    return 8 if candidate == semantic_value else 0


def _has_action_signal(query_text: str) -> bool:
    return any(_phrase_score(query_text, hint) for hint in _ACTION_HINTS) or any(
        phrase in query_text
        for phrase in (
            "vs",
            "versus",
            "compare",
            "comparison",
            "difference between",
            "who plays",
            "who is on leave",
            "currently on leave",
            "currently absent",
            "with medical",
            "on leave",
            "medical leave",
            "trend",
            "distribution",
        )
    )


def _looks_like_noise_query(
    query_text: str,
    category: Optional[str],
    operation: Optional[str],
    entities: Optional[Dict[str, Any]],
) -> bool:
    tokens = re.findall(r"[a-z0-9']+", query_text)
    if len(tokens) < 5:
        return False
    if _has_action_signal(query_text):
        return False

    # A sport/class name is a strong, specific identifier on its own — unlike
    # a bare date or number, it isn't the kind of "stray token in a filler
    # sentence" this heuristic guards against. This matters most for a
    # cross-filter clause split off a larger query (e.g. "who also plays
    # volleyball"), which is inherently short and has no other entity to
    # offer besides the sport/class itself.
    if (entities or {}).get("sport") or (entities or {}).get("class"):
        return False

    filled_entities = [
        key
        for key, value in (entities or {}).items()
        if value not in (None, "", [], {})
    ]

    if category and operation:
        return False

    # Long strings made up mostly of filler words and one stray domain token
    # should be treated as unclear instead of being forced into a category.
    if category and len(filled_entities) <= 1:
        return True
    if not category and len(filled_entities) <= 1:
        return True
    return False


def _entity_present(entities: Optional[Dict[str, Any]], *keys: str) -> bool:
    if not entities:
        return False
    return any(entities.get(key) not in (None, "", [], {}) for key in keys)


def _category_entity_bonus(category: str, entities: Optional[Dict[str, Any]]) -> int:
    if not entities:
        return 0

    bonuses: Dict[str, List[Tuple[Any, int]]] = {
        "Performance": [
            ("section", 30),
            ("grading", 14),
            (("attemptNo", "fromAttempt", "toAttempt"), 14),
        ],
        "Leave": [
            ("leaveType", 30),
        ],
        "Medical": [
            ("bmiCategory", 25),
            ("bloodGroup", 25),
            ("medicalStatus", 20),
        ],
        "Attendance": [
            ("date", 20),
            (("fromDate", "toDate"), 20),
        ],
        "Verification": [
            ("status", 20),
        ],
        "Equipment": [
            ("equipmentName", 28),
        ],
        "Strength": [
            ("section", 10),
        ],
        "Distribution": [
            ("unitName", 18),
        ],
        "Skills": [
            ("sport", 18),
            ("class", 18),
        ],
        "Overall": [
            ("section", 10),
        ],
        "Schedule": [],
    }

    bonus = 0
    for keys, amount in bonuses.get(category, []):
        if isinstance(keys, tuple):
            if _entity_present(entities, *keys):
                bonus += amount
        elif _entity_present(entities, keys):
            bonus += amount
    return bonus


def _score_category(
    query_text: str,
    category: str,
    semantic: Dict[str, Any],
    entities: Optional[Dict[str, Any]],
) -> int:
    score = _semantic_score(category, semantic.get("category"))
    score += _category_entity_bonus(category, entities)

    category_keywords = CATEGORY_KEYWORDS.get(category, ())
    for phrase in category_keywords:
        score += _phrase_score(query_text, phrase)
    score += _fuzzy_keyword_bonus(query_text, category_keywords)

    category_hints = CATEGORY_ENTITY_HINTS.get(category, ())
    for hint in category_hints:
        score += _phrase_score(query_text, hint)
    score += _fuzzy_keyword_bonus(query_text, category_hints)

    for synonyms in OPERATION_SYNONYMS.get(category, {}).values():
        for phrase in synonyms:
            matched = _phrase_score(query_text, phrase)
            if matched:
                score += max(2, matched // 4)
        fuzzy_bonus = _fuzzy_keyword_bonus(query_text, synonyms)
        if fuzzy_bonus:
            score += max(2, fuzzy_bonus // 4)

    # Encourage the natural-language defaults that the old engine used.
    if category == "Skills" and _phrase_score(query_text, "who plays"):
        score += 35
    if category == "Leave" and _phrase_score(query_text, "medical leave"):
        score += 45
    if category == "Distribution" and _phrase_score(query_text, "top unit"):
        score += 35
    if category == "Performance" and _phrase_score(query_text, "grade distribution"):
        score += 25
    if category == "Overall":
        # "overall performance ranking" otherwise loses to Performance —
        # "performance" alone is a strong, common Performance keyword, but
        # "overall performance" as a compound phrase means the composite
        # ranking, not a Performance-category lookup.
        overall_phrases = [
            "overall performance",
            "overall performing",
            "overall performer",
            "best overall",
            "across all activities",
        ]
        for phrase in overall_phrases:
            if _phrase_score(query_text, phrase):
                score += 80
    return score


def _score_operation(
    query_text: str,
    category: Optional[str],
    operation: str,
    semantic: Dict[str, Any],
    entities: Optional[Dict[str, Any]],
) -> int:
    if not category or category not in OPERATIONS_BY_CATEGORY:
        return 0
    if operation not in OPERATIONS_BY_CATEGORY[category]:
        return 0

    score = 0
    semantic_operation = semantic.get("operation")
    if semantic_operation and semantic_operation == operation:
        score += 40

    operation_synonyms = OPERATION_SYNONYMS.get(category, {}).get(operation, ())
    for phrase in operation_synonyms:
        score += _phrase_score(query_text, phrase)
    score += _fuzzy_keyword_bonus(query_text, operation_synonyms)

    if category == "Leave" and operation == "Current":
        score += 20 if _phrase_score(query_text, "currently absent") else 0
    if category == "Performance" and operation == "Top":
        score += 20 if _phrase_score(query_text, "who scored highest") else 0
    if category == "Performance" and operation == "Bottom":
        score += 40 if _phrase_score(query_text, "needs improvement") else 0
        score += 40 if _phrase_score(query_text, "need improvement") else 0
    if category == "Performance" and operation == "Improvement":
        score -= 50 if _phrase_score(query_text, "needs improvement") else 0
        score -= 50 if _phrase_score(query_text, "need improvement") else 0
    if category == "Distribution" and operation == "TopUnit":
        score += 25 if _phrase_score(query_text, "most agniveers") else 0
    if category == "Equipment" and operation == "Search":
        score += 8 if _phrase_score(query_text, "search") else 0
    if category == "Equipment" and operation == "Returned":
        score += 8 if _phrase_score(query_text, "poor condition") else 0
        # If they haven't returned it, they are still holding it
        if (
            _phrase_score(query_text, "haven't returned")
            or _phrase_score(query_text, "not returned")
            or _phrase_score(query_text, "has not returned")
            or _phrase_score(query_text, "hasn't returned")
        ):
            score -= 60
    if category == "Equipment" and operation == "Holding":
        score += 8 if _phrase_score(query_text, "currently holding") else 0
        if (
            _phrase_score(query_text, "haven't returned")
            or _phrase_score(query_text, "not returned")
            or _phrase_score(query_text, "has not returned")
            or _phrase_score(query_text, "hasn't returned")
        ):
            score += 60
        # If the user explicitly asks about issued/procured equipment, they want Holding stats, not ByName
        if _entity_present(entities, "equipmentType"):
            score += 40
        score += 30 if _phrase_score(query_text, "issued") else 0
        score += 30 if _phrase_score(query_text, "procured") else 0
    if category == "Equipment" and operation == "AgniveerWise":
        score += 8 if _entity_present(entities, "equipmentName") else 0
        # Penalize AgniveerWise (ByName) if they are explicitly asking for Issued/Procured counts
        if (
            _entity_present(entities, "equipmentType")
            or _phrase_score(query_text, "issued")
            or _phrase_score(query_text, "procured")
        ):
            score -= 50
    if category == "Medical" and operation == "BMI":
        score += 10 if _entity_present(entities, "bmiCategory") else 0
    if category == "Medical" and operation == "BloodGroup":
        score += 10 if _entity_present(entities, "bloodGroup") else 0
    if category == "Schedule" and operation == "bytoday":
        # Bare "schedule" is a weak, generic signal — don't let it outscore
        # Company/Date/Agniveer when the query actually names one of those
        # (e.g. "schedule for Lakhwinder company" contains "schedule" too,
        # but should resolve to Company, not the bare Today default).
        has_more_specific_signal = _entity_present(
            entities, "agniveerNo", "date"
        ) or bool(re.search(r"\b(company)\b", query_text, re.IGNORECASE))
        score += (
            10
            if _phrase_score(query_text, "schedule") and not has_more_specific_signal
            else 0
        )
    if category == "Attendance" and operation in {
        "Monthly",
        "Weekly",
        "Daily",
        "Yearly",
        "Present",
    }:
        score += 8 if _entity_present(entities, "date", "fromDate", "toDate") else 0

    return score


def _damerau_levenshtein(a: str, b: str) -> int:
    """Edit distance counting an adjacent-letter transposition (e.g.
    "levae" -> "leave") as a single edit rather than two substitutions."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    d = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        d[i][0] = i
    for j in range(lb + 1):
        d[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,
                d[i][j - 1] + 1,
                d[i - 1][j - 1] + cost,
            )
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + cost)
    return d[la][lb]


def _fuzzy_contains_word(
    query_text: str, targets: Iterable[str], max_distance: int = 2
) -> bool:
    # Tolerates typos (e.g. "derail"/"detril" for "detail") so a misspelled
    # request for a detailed/summary response is still recognized. Uses
    # transposition-aware distance so an adjacent-letter swap typo (e.g.
    # "detials" for "details") counts as one edit, not two.
    for word in query_text.split():
        if len(word) < 4:
            continue
        for target in targets:
            if abs(len(word) - len(target)) > max_distance:
                continue
            if _damerau_levenshtein(word, target) <= max_distance:
                return True
    return False


# Single-word DETAILED_KEYWORDS entries are typo-checked directly; multi-word
# phrases ("full report", "deep analysis") aren't — matching a typo'd word
# within a phrase reliably would need per-word phrase alignment, which isn't
# worth the complexity for these rarer, longer phrasings.
_DETAILED_SINGLE_WORDS: Tuple[str, ...] = tuple(
    kw for kw in DETAILED_KEYWORDS if " " not in kw
) + ("details",)


def _detect_explicit_response_type(query_text: str) -> Optional[str]:
    """Whether the query explicitly asks for Detailed or Summary, ignoring
    any category-based default. Returns None when the text carries no such
    signal (typo-tolerant — see _fuzzy_contains_word)."""
    for keyword in DETAILED_KEYWORDS:
        if _phrase_score(query_text, keyword):
            return "Detailed"
    if _fuzzy_contains_word(query_text, _DETAILED_SINGLE_WORDS):
        return "Detailed"
    if (
        _phrase_score(query_text, "summary")
        or _phrase_score(query_text, "summarize")
        or _phrase_score(query_text, "summarise")
    ):
        return "Summary"
    if _fuzzy_contains_word(query_text, ("summary", "summarize", "summarise")):
        return "Summary"
    return None


def _detect_response_type(query_text: str, category: Optional[str] = None) -> str:
    explicit = _detect_explicit_response_type(query_text)
    if explicit:
        return explicit
    if category in DEFAULT_DETAILED_CATEGORIES:
        return "Detailed"
    return RESPONSE_TYPE_DEFAULT


def detect_query_response_type_override(raw_query: str) -> Optional[str]:
    """Public: whether the raw (un-normalised) query explicitly asks for a
    Detailed or Summary response, regardless of category.

    A multi-part query ("show bpet performers who returned equipment in good
    condition in detail") gets split into per-category fragments before each
    one is classified — "in detail" only appears in the last fragment's raw
    text, so only that fragment's responseType would pick it up. Callers
    that assemble multiple sub-operations from one query (query_planner.py)
    use this to detect the signal from the *whole* query once and apply it
    uniformly to every operation, so one "in detail"/"summary" governs the
    entire answer rather than just whichever fragment happened to contain it.
    """
    return _detect_explicit_response_type(_canonical_text(raw_query))


def _choose_category(
    query_text: str,
    semantic: Dict[str, Any],
    entities: Optional[Dict[str, Any]],
) -> Tuple[Optional[str], int]:
    scores: Dict[str, int] = {}
    for category in OFFICIAL_CATEGORIES:
        score = _score_category(query_text, category, semantic, entities)
        if score > 0:
            scores[category] = score

    if not scores:
        return None, 0

    ranked = sorted(
        scores.items(), key=lambda item: (item[1], len(item[0])), reverse=True
    )
    return ranked[0]


def _choose_operation(
    query_text: str,
    category: Optional[str],
    semantic: Dict[str, Any],
    entities: Optional[Dict[str, Any]],
) -> Tuple[Optional[str], int]:
    if not category:
        return None, 0

    scores: Dict[str, int] = {}
    for operation in OPERATIONS_BY_CATEGORY.get(category, ()):  # type: ignore[arg-type]
        score = _score_operation(query_text, category, operation, semantic, entities)
        if score > 0:
            scores[operation] = score

    if not scores:
        return None, 0

    ranked = sorted(
        scores.items(), key=lambda item: (item[1], len(item[0])), reverse=True
    )
    return ranked[0]


def _compute_confidence(
    category: Optional[str],
    category_score: int,
    operation: Optional[str],
    operation_score: int,
    semantic: Dict[str, Any],
    entities: Optional[Dict[str, Any]],
) -> float:
    confidence_score = 0.0
    if category:
        confidence_score += min(category_score / 100.0, 0.5)
    if operation:
        confidence_score += min(operation_score / 120.0, 0.35)
    conf_float = float(semantic.get("confidence") or 0.0)
    if conf_float >= 0.7:
        confidence_score += 0.15
    elif conf_float >= 0.45:
        confidence_score += 0.1
    elif conf_float >= 0.2:
        confidence_score += 0.05
    if entities:
        filled_entities = sum(
            1 for value in entities.values() if value not in (None, "", [], {})
        )
        confidence_score += min(filled_entities / 20.0, 0.1)
    return confidence_score


def _should_entity_override_category(
    entities: Optional[Dict[str, Any]],
    classified_category: Optional[str],
    confidence_score: float,
    query_text: str,
) -> Tuple[bool, Optional[str], str]:
    if not entities:
        return False, None, "no entities present"

    # A named equipment item is matched against a curated, unambiguous list
    # (ISSUED_EQUIPMENT_ITEMS / PROCURED_EQUIPMENT_ITEMS) — unlike the
    # heuristic keyword-score signals below, it's precise enough that it
    # must win even when the classifier is otherwise "confident" about a
    # different category purely because the item's name happens to overlap
    # with an unrelated category's keywords (e.g. "swimming costumes" ~
    # Skills' "swimming", "health card" ~ Medical's "health").
    if (
        _entity_present(entities, "equipmentName")
        and classified_category != "Equipment"
    ):
        return (
            True,
            "Equipment",
            "equipmentName entity present (curated item list — overrides confidence)",
        )

    # Confidence guard — if the classifier is already confident, trust it.
    if confidence_score >= 0.45 and not (
        classified_category == "Leave"
        and _entity_present(entities, "section")
        and any(
            _phrase_score(query_text, phrase)
            for phrase in (
                "topper",
                "toppers",
                "top performer",
                "top performers",
                "highest performer",
                "best performer",
                "top scorer",
                "highest scorer",
                "best scorer",
            )
        )
    ):
        return (
            False,
            None,
            f"confidence sufficient ({confidence_score:.2f}) - classifier wins",
        )

    if _entity_present(entities, "leaveType") and classified_category != "Leave":
        return True, "Leave", "leaveType entity present"
    if (
        _entity_present(
            entities, "bmiCategory", "bloodGroup", "medicalStatus", "diagnose"
        )
        and classified_category != "Medical"
    ):
        return True, "Medical", "medical entity present"

    if (
        _entity_present(entities, "grading", "attemptNo", "fromAttempt", "toAttempt")
        and classified_category != "Performance"
    ):
        return True, "Performance", "performance entity present"
    if _entity_present(
        entities, "section", "subSection"
    ) and classified_category not in ("Performance", "Skills", "Strength"):
        if _phrase_score(query_text, "strength") or _phrase_score(
            query_text, "headcount"
        ):
            return True, "Strength", "strength entity with section present"
        return True, "Performance", "section entity present"

    if _entity_present(entities, "sport") and not _phrase_score(
        query_text, "attendance"
    ):
        if classified_category not in ("Skills",):
            return (
                True,
                "Skills",
                "sport entity present with low confidence — Skills",
            )
    if (
        _entity_present(entities, "class")
        and _phrase_score(query_text, "skills")
        and classified_category != "Skills"
    ):
        return (
            True,
            "Skills",
            "class entity with skills phrase present, classifier confidence low",
        )
    if (
        _entity_present(entities, "unitName")
        and _phrase_score(query_text, "attendance")
        and classified_category != "Attendance"
    ):
        return (
            True,
            "Attendance",
            "unitName entity with attendance phrase present, classifier confidence low",
        )

    return False, None, "no override condition met"


def _should_entity_override_operation(
    entities: Optional[Dict[str, Any]],
    classified_operation: Optional[str],
    category: Optional[str],
    confidence_score: float,
    query_text: str,
) -> Tuple[bool, Optional[str], str]:
    if not entities or not category:
        return False, None, "no entities or category"

    if category == "Performance":
        if _entity_present(
            entities, "attemptNo", "fromAttempt", "toAttempt"
        ) and classified_operation not in (
            "AttemptWise",
            "BestAttempt",
            "Improvement",
            "ImprovementTrend",
            "Drop",
            "DropTrend",
        ):
            return True, "AttemptWise", "attempt entity present"
        if not classified_operation and _entity_present(entities, "grading"):
            return True, "Grading", "grading entity present without operation"
        if classified_operation == "Grading" and not _entity_present(
            entities, "grading"
        ):
            # The .NET "Grading" operation requires an explicit grade value
            # (Good/Excellent/SAT/Fail/...) and 400s without one. A query
            # like "grade distribution" or "compare grading" names no grade,
            # so it means the grade breakdown, not one specific grade.
            return True, "GradingSummary", "Grading operation without a grading value"

    if category == "Medical":
        if _entity_present(entities, "bmiCategory") and classified_operation != "BMI":
            return True, "BMI", "bmiCategory entity present"
        if (
            _entity_present(entities, "bloodGroup")
            and classified_operation != "BloodGroup"
        ):
            return True, "BloodGroup", "bloodGroup entity present"
        if not classified_operation and _entity_present(entities, "medicalStatus"):
            return True, "Summary", "medicalStatus entity present without operation"
        if not classified_operation and _entity_present(entities, "agniveerNo"):
            # .NET rejects Medical calls with no operation at all ("Unknown
            # medical operation ''"). A query naming a specific agniveer
            # ("medical records of agniveer X") means their individual
            # record, even when the phrasing doesn't match a keyword synonym.
            return True, "Individual", "agniveerNo entity present without operation"

    if (
        category == "Leave"
        and entities.get("leaveType") == "Current"
        and classified_operation != "Current"
    ):
        return True, "Current", "leaveType Current present"

    if category == "Attendance" and classified_operation not in (
        "Weekly",
        "Daily",
        "Yearly",
        "Monthly",
    ):
        date_val = str(entities.get("date") or "")
        if re.search(
            r"\b(January|February|March|April|May|June|July|August|September|October|"
            r"November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b",
            date_val,
            re.IGNORECASE,
        ):
            return True, "Monthly", "date value indicates a month"

    if category == "Schedule" and not classified_operation:
        # Company resolution happens later in the pipeline (not in `entities`
        # here), so a company mention can't be caught via _entity_present —
        # fall back to a bare keyword check. Checked before Agniveer/Date so
        # "schedule of agniveer X's company" style text still prefers the
        # more specific agniveerNo/date entity when both are present.
        #
        # A relative phrase ("this week", "today", ...) is a weak, generic
        # time-scope modifier — it must NOT preempt an explicit company/
        # agniveer mention just because it happened to land in the `date`
        # entity (e.g. "company 2 schedule this week" must stay bycompany,
        # not fall through to bydate).
        date_val = str(entities.get("date") or "").lower()
        is_specific_date = bool(date_val) and date_val not in RELATIVE_DATE_PHRASES
        if _entity_present(entities, "agniveerNo"):
            return True, "byagniveer", "Schedule query for a specific agniveer"
        if is_specific_date:
            return True, "bydate", "Schedule query for a specific date"
        if re.search(r"\b(company)\b", query_text, re.IGNORECASE):
            return True, "bycompany", "Schedule query mentions a company"
        return True, "bytoday", "Schedule query default operation"

    if category == "Equipment" and not classified_operation:
        if _entity_present(entities, "equipmentName"):
            return True, "Search", "equipmentName entity present without operation"
        if _phrase_score(query_text, "overdue"):
            return True, "Holding", "equipment overdue phrase present"
        if _phrase_score(query_text, "search") or _phrase_score(query_text, "find"):
            return True, "Search", "equipment search phrase present"

    if category in ("Skills",):
        if not classified_operation and _entity_present(entities, "sport"):
            return True, "BySport", "sport entity present without operation"
        if not classified_operation and _entity_present(entities, "class"):
            return True, "ByClass", "class entity present without operation"

    if category == "Overall" and not classified_operation:
        return True, "OverallPerformance", "Overall query default operation"

    if category == "personaldetail" and not classified_operation:
        return True, "info", f"{category} query default operation"
    if category == "disqualified" and not classified_operation:
        return True, "removed", f"{category} query default operation"

    if (
        category == "Distribution"
        and not classified_operation
        and _entity_present(entities, "unitName")
    ):
        return True, "ByUnit", "unitName entity present without operation"

    return False, None, "no override condition met"


def classify_intent(
    query: str,
    entities: Optional[Dict[str, Any]] = None,
    semantic: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    query = str(query or "").strip()
    query_text = _canonical_text(query)
    semantic = semantic or understand_query(query)

    category, category_score = _choose_category(query_text, semantic, entities)
    operation, operation_score = _choose_operation(
        query_text, category, semantic, entities
    )

    if _looks_like_noise_query(query_text, category, operation, entities):
        return {
            "category": None,
            "operation": None,
            "responseType": _detect_response_type(query_text),
            "raw_query": query,
            "confidence_score": 0.0,
            "confidence": "low",
        }

    confidence_score = _compute_confidence(
        category, category_score, operation, operation_score, semantic, entities
    )

    # Stage 2: semantic fallback when keyword score is low
    if confidence_score < 0.45:
        try:
            from .semantic_classifier import classify_admin_intent_semantic

            sem_result = classify_admin_intent_semantic(query, confidence_score)
            if sem_result is not None:
                if sem_result.get("needs_clarification"):
                    return {
                        "category": None,
                        "operation": None,
                        "responseType": _detect_response_type(query_text),
                        "raw_query": query,
                        "confidence_score": round(sem_result.get("confidence", 0.0), 2),
                        "confidence": "low",
                        "needs_clarification": True,
                        "clarification_question": sem_result.get(
                            "clarification_question"
                        ),
                    }
                sem_cat = sem_result.get("category")
                sem_op = sem_result.get("operation")
                sem_conf = float(sem_result.get("confidence", 0.0))
                if sem_cat:
                    logger.info(
                        "[SEMANTIC STAGE2] %s → %s/%s conf=%.3f",
                        query,
                        sem_cat,
                        sem_op,
                        sem_conf,
                    )
                    category = sem_cat
                    operation = sem_op
                    confidence_score = sem_conf
        except Exception as _sem_exc:
            logger.debug("semantic fallback skipped: %s", _sem_exc)

    should_override_cat, override_cat, cat_reason = _should_entity_override_category(
        entities, category, confidence_score, query_text
    )
    if should_override_cat:
        logger.info(
            f"[CATEGORY OVERRIDE] {category} → {override_cat} | reason: {cat_reason}"
        )
        category = override_cat
        operation, operation_score = _choose_operation(
            query_text, category, semantic, entities
        )
        confidence_score = _compute_confidence(
            category, category_score, operation, operation_score, semantic, entities
        )

    if category == "Leave" and _phrase_score(query_text, "current absent"):
        operation = "Current"
    if category == "Distribution" and _phrase_score(query_text, "top unit"):
        operation = "TopUnit"
    if category == "Medical" and _phrase_score(query_text, "blood group"):
        operation = operation or "BloodGroup"
    if category == "Medical" and _phrase_score(query_text, "bmi"):
        operation = operation or "BMI"
    if category == "Medical" and entities and entities.get("diagnose"):
        operation = operation or "Disease"
    if category == "Attendance" and _phrase_score(query_text, "present today"):
        operation = operation or "Present"

    # Sport-entity strong signal: an exact hit in our SPORTS dictionary with no
    # competing strong category → classify as Skills/BySport at >= 0.6 confidence.
    if (
        entities
        and entities.get("sport")
        and category
        not in ("Leave", "Medical", "Attendance", "Performance", "Equipment")
    ):
        if category != "Skills" or confidence_score < 0.6:
            category = "Skills"
            operation = "BySport"
            confidence_score = max(confidence_score, 0.6)

    should_override_op, override_op, op_reason = _should_entity_override_operation(
        entities, operation, category, confidence_score, query_text
    )
    if should_override_op:
        logger.info(
            f"[OPERATION OVERRIDE] {operation} → {override_op} | reason: {op_reason}"
        )
        operation = override_op
        confidence_score = _compute_confidence(
            category, category_score, operation, operation_score, semantic, entities
        )

    response_type = _detect_response_type(query_text, category)

    return {
        "category": category,
        "operation": operation,
        "responseType": response_type,
        "raw_query": query,
        "confidence_score": round(confidence_score, 2),
        "confidence": (
            "high"
            if confidence_score >= 0.75
            else ("medium" if confidence_score >= 0.45 else "low")
        ),
    }
