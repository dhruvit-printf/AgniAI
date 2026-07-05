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
    DETAILED_KEYWORDS,
    OFFICIAL_CATEGORIES,
    OPERATION_SYNONYMS,
    OPERATIONS_BY_CATEGORY,
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


def _semantic_score(candidate: str, semantic_value: Optional[str]) -> int:
    if not semantic_value:
        return 0
    return 8 if candidate == semantic_value else 0


def _has_action_signal(query_text: str) -> bool:
    return any(
        _phrase_score(query_text, hint) for hint in _ACTION_HINTS
    ) or any(
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
        "Schedule": [
        ],
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

    for phrase in CATEGORY_KEYWORDS.get(category, ()):
        score += _phrase_score(query_text, phrase)

    for hint in CATEGORY_ENTITY_HINTS.get(category, ()):
        score += _phrase_score(query_text, hint)

    for synonyms in OPERATION_SYNONYMS.get(category, {}).values():
        for phrase in synonyms:
            matched = _phrase_score(query_text, phrase)
            if matched:
                score += max(2, matched // 4)

    # Encourage the natural-language defaults that the old engine used.
    if category == "Skills" and _phrase_score(query_text, "who plays"):
        score += 35
    if category == "Leave" and _phrase_score(query_text, "medical leave"):
        score += 45
    if category == "Distribution" and _phrase_score(query_text, "top unit"):
        score += 35
    if category == "Performance" and _phrase_score(query_text, "grade distribution"):
        score += 25
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

    for phrase in OPERATION_SYNONYMS.get(category, {}).get(operation, ()):
        score += _phrase_score(query_text, phrase)

    if category == "Leave" and operation == "Current":
        score += 20 if _phrase_score(query_text, "currently absent") else 0
    if category == "Performance" and operation == "Top":
        score += 20 if _phrase_score(query_text, "who scored highest") else 0
    if category == "Distribution" and operation == "TopUnit":
        score += 25 if _phrase_score(query_text, "most agniveers") else 0
    if category == "Equipment" and operation == "Search":
        score += 8 if _phrase_score(query_text, "search") else 0
    if category == "Equipment" and operation == "Returned":
        score += 8 if _phrase_score(query_text, "poor condition") else 0
    if category == "Equipment" and operation == "Holding":
        score += 8 if _phrase_score(query_text, "currently holding") else 0
    if category == "Equipment" and operation == "AgniveerWise":
        score += 8 if _entity_present(entities, "equipmentName") else 0
    if category == "Medical" and operation == "BMI":
        score += 10 if _entity_present(entities, "bmiCategory") else 0
    if category == "Medical" and operation == "BloodGroup":
        score += 10 if _entity_present(entities, "bloodGroup") else 0
    if category == "Strength" and operation == "StrengthBreakdown":
        score += 10 if _entity_present(entities, "section") else 0
    if category == "Schedule" and operation == "Today":
        score += 10 if _phrase_score(query_text, "schedule") else 0
    if category == "Attendance" and operation in {
        "Monthly",
        "Weekly",
        "Daily",
        "Yearly",
        "Present",
    }:
        score += 8 if _entity_present(entities, "date", "fromDate", "toDate") else 0

    return score


def _detect_response_type(query_text: str) -> str:
    for keyword in DETAILED_KEYWORDS:
        if _phrase_score(query_text, keyword):
            return "Detailed"
    if _phrase_score(query_text, "summary") or _phrase_score(
        query_text, "summarize"
    ) or _phrase_score(query_text, "summarise"):
        return "Summary"
    return RESPONSE_TYPE_DEFAULT


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

    # Confidence guard FIRST — if the classifier is already confident, trust it.
    if confidence_score >= 0.45:
        return (
            False,
            None,
            f"confidence sufficient ({confidence_score:.2f}) — classifier wins",
        )

    if _entity_present(entities, "leaveType") and classified_category != "Leave":
        return True, "Leave", "leaveType entity present"
    if (
        _entity_present(entities, "equipmentName")
        and classified_category != "Equipment"
    ):
        return (
            True,
            "Equipment",
            "equipmentName entity present",
        )
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
    if (
        _entity_present(entities, "section", "subSection")
        and classified_category not in ("Performance", "Skills", "Strength")
    ):
        if _phrase_score(query_text, "strength") or _phrase_score(query_text, "headcount"):
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
        ) and classified_operation not in ("AttemptWise", "BestAttempt", "Improvement", "ImprovementTrend", "Drop", "DropTrend"):
            return True, "AttemptWise", "attempt entity present"
        if not classified_operation and _entity_present(entities, "grading"):
            return True, "Grading", "grading entity present without operation"

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
        return True, "Today", "Schedule query default operation"

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

    if category == "Strength" and not classified_operation:
        return True, "StrengthBreakdown", "Strength query default operation"

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
                        "clarification_question": sem_result.get("clarification_question"),
                    }
                sem_cat = sem_result.get("category")
                sem_op = sem_result.get("operation")
                sem_conf = float(sem_result.get("confidence", 0.0))
                if sem_cat:
                    logger.info(
                        "[SEMANTIC STAGE2] %s → %s/%s conf=%.3f",
                        query, sem_cat, sem_op, sem_conf,
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
        and category not in ("Leave", "Medical", "Attendance", "Performance", "Equipment")
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

    response_type = _detect_response_type(query_text)

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
