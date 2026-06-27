"""
intent_classifier.py
====================

Single responsibility: determine Category, Operation, and ResponseType.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Optional, Tuple

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


def _canonical_text(query: str) -> str:
    return clean_query(query).lower()


def _phrase_score(text: str, phrase: str) -> int:
    if not phrase:
        return 0
    phrase_norm = _canonical_text(phrase)
    if not phrase_norm:
        return 0
    if phrase_norm not in text:
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


def _entity_present(entities: Optional[Dict[str, Any]], *keys: str) -> bool:
    if not entities:
        return False
    return any(entities.get(key) not in (None, "", [], {}) for key in keys)


def _category_entity_bonus(category: str, entities: Optional[Dict[str, Any]]) -> int:
    if not entities:
        return 0

    bonuses = {
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
            ("medical_status", 20),
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
        "Distribution": [
            ("unitName", 18),
        ],
        "Skills": [
            ("sport", 18),
            ("class", 18),
        ],
        "Roster": [
            ("sport", 20),
            ("class", 20),
        ],
        "Strength": [
            ("section", 5),
        ],
        "Overall": [
            ("section", 10),
        ],
        "Schedule": [
            ("date", 12),
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
    if category == "Roster" and _phrase_score(query_text, "who plays"):
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
    if category == "Equipment" and operation in {"Issued", "Procured", "Holding", "Overdue", "Returned"}:
        score += 8 if _entity_present(entities, "equipmentName") else 0
    if category == "Medical" and operation == "BMI":
        score += 10 if _entity_present(entities, "bmiCategory") else 0
    if category == "Medical" and operation == "BloodGroup":
        score += 10 if _entity_present(entities, "bloodGroup") else 0
    if category == "Attendance" and operation in {"Monthly", "Weekly", "Daily", "Yearly", "Present"}:
        score += 8 if _entity_present(entities, "date", "fromDate", "toDate") else 0

    return score


def _detect_response_type(query_text: str) -> str:
    for keyword in DETAILED_KEYWORDS:
        if _phrase_score(query_text, keyword):
            return "Detailed"
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

    ranked = sorted(scores.items(), key=lambda item: (item[1], len(item[0])), reverse=True)
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

    ranked = sorted(scores.items(), key=lambda item: (item[1], len(item[0])), reverse=True)
    return ranked[0]


def classify_intent(
    query: str,
    entities: Optional[Dict[str, Any]] = None,
    semantic: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    query = str(query or "").strip()
    query_text = _canonical_text(query)
    semantic = semantic or understand_query(query)

    category, category_score = _choose_category(query_text, semantic, entities)

    # Contextual tie-breaking for common ambiguous phrases.
    if entities:
        if _entity_present(entities, "leaveType"):
            category = "Leave"
        elif _entity_present(entities, "equipmentName"):
            category = "Equipment"
        elif _entity_present(entities, "bmiCategory", "bloodGroup", "medical_status"):
            category = "Medical"
        elif _entity_present(entities, "section") and category in (None, "Performance"):
            category = "Performance"
        elif _entity_present(entities, "sport") and not _phrase_score(query_text, "attendance"):
            if _phrase_score(query_text, "who plays") or _phrase_score(query_text, "roster"):
                category = "Roster"
            elif category is None:
                category = "Roster"
        elif _entity_present(entities, "class") and _phrase_score(query_text, "skills"):
            category = "Skills"
        elif _entity_present(entities, "unitName") and _phrase_score(query_text, "attendance"):
            category = "Attendance"

    operation, operation_score = _choose_operation(query_text, category, semantic, entities)

    if category == "Leave" and _phrase_score(query_text, "current absent"):
        operation = "Current"
    if category == "Distribution" and _phrase_score(query_text, "top unit"):
        operation = "TopUnit"
    if category == "Medical" and _phrase_score(query_text, "blood group"):
        operation = operation or "BloodGroup"
    if category == "Medical" and _phrase_score(query_text, "bmi"):
        operation = operation or "BMI"
    if category == "Attendance" and _phrase_score(query_text, "present today"):
        operation = operation or "Present"

    # Entity-driven operation overrides — entity presence is authoritative for mappings
    # that pure keyword scoring cannot reliably infer (e.g. "show BPET attempt 2" has
    # an attemptNo entity but no "attempt wise" keyword).
    if entities:
        if category == "Performance":
            if _entity_present(entities, "attemptNo", "fromAttempt", "toAttempt") and operation not in ("AttemptWise", "BestAttempt"):
                operation = "AttemptWise"
            if not operation and _entity_present(entities, "grading"):
                operation = "Grading"
        if category == "Medical":
            if _entity_present(entities, "bmiCategory"):
                operation = "BMI"
            if not operation and _entity_present(entities, "bloodGroup"):
                operation = "BloodGroup"
            if not operation and _entity_present(entities, "medical_status"):
                operation = "Active"
        if category == "Leave" and entities.get("leaveType") == "Current":
            operation = "Current"
        if category == "Attendance" and operation not in ("Weekly", "Daily", "Yearly"):
            date_val = str(entities.get("date") or "")
            if re.search(
                r"\b(January|February|March|April|May|June|July|August|September|October|"
                r"November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}\b",
                date_val,
                re.IGNORECASE,
            ):
                operation = "Monthly"
        if category in ("Roster", "Skills"):
            if not operation and _entity_present(entities, "sport"):
                operation = "BySport"
            if not operation and _entity_present(entities, "class"):
                operation = "ByClass"
        if category == "Strength" and not operation:
            operation = "Summary"
        if category == "Overall" and not operation:
            operation = "OverallPerformance"
        if category == "Distribution" and not operation and _entity_present(entities, "unitName"):
            operation = "ByUnit"

    response_type = _detect_response_type(query_text)

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
        filled_entities = sum(1 for value in entities.values() if value not in (None, "", [], {}))
        confidence_score += min(filled_entities / 20.0, 0.1)

    if operation and category:
        confidence = "high" if confidence_score >= 0.6 else "medium"
    elif category:
        confidence = "medium" if confidence_score >= 0.35 else "low"
    else:
        confidence = "low"

    return {
        "category": category,
        "operation": operation,
        "responseType": response_type,
        "raw_query": query,
        "confidence": confidence,
    }
