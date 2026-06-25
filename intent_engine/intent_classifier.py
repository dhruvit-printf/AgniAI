"""
intent_classifier.py
====================

Single responsibility: determine Category, Operation, and ResponseType.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .intent_schema import (
    CATEGORY_KEYWORDS,
    DETAILED_KEYWORDS,
    OFFICIAL_CATEGORIES,
    OPERATION_SYNONYMS,
    OPERATIONS_BY_CATEGORY,
    RESPONSE_TYPE_DEFAULT,
)
from query_understanding_engine import understand_query


def _detect_response_type(query: str) -> str:
    query_lower = query.lower()
    for keyword in DETAILED_KEYWORDS:
        if keyword in query_lower:
            return "Detailed"
    return RESPONSE_TYPE_DEFAULT


def _detect_category(query: str, semantic: Dict[str, Any]) -> Optional[str]:
    semantic_category = semantic.get("category")
    if semantic_category and semantic_category in OFFICIAL_CATEGORIES:
        return semantic_category

    query_lower = query.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in query_lower:
                return category
    return None


def _detect_operation(
    query: str,
    category: Optional[str],
    semantic: Dict[str, Any],
) -> Optional[str]:
    if not category or category not in OPERATIONS_BY_CATEGORY:
        return None

    semantic_operation = semantic.get("operation")
    if semantic_operation and semantic_operation in OPERATIONS_BY_CATEGORY[category]:
        return semantic_operation

    query_lower = query.lower()
    category_synonyms = OPERATION_SYNONYMS.get(category, {})
    best_match = None
    for operation, synonyms in category_synonyms.items():
        for synonym in synonyms:
            if synonym in query_lower:
                candidate = (len(synonym), operation)
                if best_match is None or candidate[0] > best_match[0]:
                    best_match = candidate
    if best_match is not None:
        return best_match[1]
    return None


def classify_intent(query: str) -> Dict[str, Any]:
    query = str(query).strip()
    semantic = understand_query(query)

    category = _detect_category(query, semantic)
    operation = _detect_operation(query, category, semantic)
    response_type = _detect_response_type(query)

    confidence = "low"
    if category and operation:
        confidence = "high"
    elif category:
        confidence = "medium"

    return {
        "category": category,
        "operation": operation,
        "responseType": response_type,
        "raw_query": query,
        "confidence": confidence,
    }
