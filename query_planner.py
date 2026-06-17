"""
query_planner.py
================
Query Planning Layer for the AgniAI Admin Chatbot.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from admin_intent import (
    _normalise,
    classify_admin_intent,
    format_admin_payload,
)

logger = logging.getLogger(__name__)


class QueryType(Enum):
    SIMPLE = "simple"
    CROSS_FILTER = "cross_filter"
    COMPARISON = "comparison"
    MULTI_INDEPENDENT = "multi_independent"


@dataclass
class SubOperation:
    raw_fragment: str
    intent_result: Dict[str, Any] = field(default_factory=dict)
    dotnet_payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rawFragment": self.raw_fragment,
            "intentResult": self.intent_result,
            "dotnetPayload": self.dotnet_payload,
        }


@dataclass
class QueryPlan:
    query_type: QueryType
    operations: List[SubOperation]
    confidence: float
    raw_query: str
    reasoning: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "queryType": self.query_type.value,
            "confidence": round(self.confidence, 2),
            "operationCount": len(self.operations),
            "reasoning": self.reasoning,
            "operations": [op.to_dict() for op in self.operations],
        }


_COMPARISON_KEYWORDS: List[str] = [
    "compare", "comparison", " vs ", "versus", "compared to", "compared with", "difference between", "contrast"
]

_CROSS_FILTER_KEYWORDS: List[str] = [
    "who plays", "who play", "who are in", "who is in", "among", "that play", "that plays", "which play", "which plays", "having sport", "with sport"
]

_CROSS_FILTER_CONNECTORS: List[str] = [
    "who", "that", "which", "having", "among"
]

_MULTI_INDEPENDENT_CONNECTORS: List[str] = [
    "along with", "as well as", "together with", "additionally show", "also show", "and also"
]

_NO_SPLIT_PHRASES: List[str] = [
    "approved and pending leave", "approved and pending", "annual and medical leave",
    "annual and sick leave", "medical and sick leave", "top and bottom", "top and worst",
    "best and worst", "highest and lowest", "pass and fail", "pass percentage and fail percentage",
    "pass rate and fail rate", "improvement and drop", "improvement and decline",
    "issued and procured", "overdue and returned", "pending and completed verification", "pending and completed"
]

_CATEGORY_SIGNALS: Dict[str, List[str]] = {
    "Performance": [
        "performance", "performer", "performers", "score", "marks", "bpet", "ppt", "firing", "drill",
        "grading", "grade", "top performer", "bottom performer", "average score", "pass percentage",
        "fail percentage", "improvement", "drop", "attempt", "section summary", "overall performance"
    ],
    "Leave": [
        "leave", "absent", "absentee", "absconded", "awol", "annual leave", "medical leave", "sick leave"
    ],
    "Medical": [
        "medical", "hospital", "bmi", "disease", "health", "admitted", "patient", "ward", "illness"
    ],
    "Attendance": [
        "attendance", "present", "campus", "strength", "headcount", "monthly attendance"
    ],
    "Verification": [
        "verification", "verified", "pending verification", "completed verification"
    ],
    "Equipment": [
        "equipment", "gear", "overdue", "inventory", "issued items", "procured items", "damaged"
    ],
    "Distribution": [
        "distribution", "unit", "unassigned", "distributed"
    ],
    "Skills": [
        "sport", "sports", "cricket", "football", "hockey", "basketball", "volleyball", "kabaddi",
        "running", "blood group", "blood type", "class", "roster", "sikh", "dogra", "jat", "gurkha",
        "rajput", "punjabi"
    ]
}


def _detect_categories(text_lower: str) -> List[str]:
    scores: Dict[str, int] = {}
    for category, signals in _CATEGORY_SIGNALS.items():
        score = sum(len(sig.split()) for sig in signals if sig in text_lower)
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


def _detect_comparison(text_lower: str, categories: List[str]) -> Optional[Tuple[str, str]]:
    if not any(kw in text_lower for kw in _COMPARISON_KEYWORDS):
        return None
    
    sections_found = [s for s in {"bpet", "ppt", "firing", "drill"} if s in text_lower]
    if len(sections_found) >= 2 and ("Performance" in categories or len(categories) <= 1):
        logger.debug("Intra-performance comparison detected: %s - using Simple path", sections_found)
        return None

    if len(categories) >= 2:
        return (categories[0], categories[1])
    return None


def _detect_multi_independent(text_lower: str, categories: List[str]) -> Optional[List[str]]:
    for connector in _MULTI_INDEPENDENT_CONNECTORS:
        if connector in text_lower:
            parts = text_lower.split(connector, 1)
            if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                if _detect_categories(parts[0]) and _detect_categories(parts[1]):
                    return [parts[0].strip(), parts[1].strip()]

    if " and " in text_lower and len(categories) >= 2:
        and_positions = [m.start() for m in re.finditer(r"\band\b", text_lower)]
        for pos in and_positions:
            left = text_lower[:pos].strip()
            right = text_lower[pos + 5:].strip()
            if left and right:
                left_cats = _detect_categories(left)
                right_cats = _detect_categories(right)
                if left_cats and right_cats and left_cats[0] != right_cats[0]:
                    return [left, right]
    return None


def _is_no_split_phrase(text_lower: str) -> bool:
    return any(phrase in text_lower for phrase in _NO_SPLIT_PHRASES)


def _extract_cross_filter_fragments(text_lower: str, categories: List[str]) -> Optional[List[str]]:
    _SPORT_NAMES = {"cricket", "football", "hockey", "basketball", "volleyball", "kabaddi", "running"}

    def _enrich_right(right_fragment: str) -> str:
        for sport in _SPORT_NAMES:
            if sport in right_fragment:
                return f"sport {sport}"
        return right_fragment

    for kw in _CROSS_FILTER_KEYWORDS:
        if kw in text_lower:
            idx = text_lower.index(kw)
            left = text_lower[:idx].strip()
            right = text_lower[idx:].strip()
            if left and right:
                return [left, _enrich_right(right)]

    for connector in _CROSS_FILTER_CONNECTORS:
        match = re.search(r"\b" + re.escape(connector) + r"\b", text_lower)
        if match:
            idx = match.start()
            left = text_lower[:idx].strip()
            right = text_lower[idx:].strip()
            if left and right:
                left_cats = _detect_categories(left)
                right_enriched = _enrich_right(right)
                right_cats = _detect_categories(right_enriched)
                if left_cats and right_cats and left_cats[0] != right_cats[0]:
                    return [left, right_enriched]
    return None


def _extract_comparison_fragments(text_lower: str, categories: List[str]) -> Optional[List[str]]:
    for sep in [" vs ", " versus "]:
        if sep in text_lower:
            parts = text_lower.split(sep, 1)
            if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                left, right = parts[0].strip(), parts[1].strip()
                for kw in ["compare", "comparison"]:
                    left = re.sub(r"^" + kw + r"\s+", "", left).strip()
                return [left, right]

    compare_match = re.search(r"\bcompare\s+(.+?)\s+and\s+(.+)", text_lower)
    if compare_match:
        return [compare_match.group(1).strip(), compare_match.group(2).strip()]

    compared_match = re.search(r"(.+?)\s+compared\s+(?:to|with)\s+(.+)", text_lower)
    if compared_match:
        return [compared_match.group(1).strip(), compared_match.group(2).strip()]

    return None


def _build_sub_operation(fragment: str) -> SubOperation:
    intent_result = classify_admin_intent(fragment)
    dotnet_payload = format_admin_payload(intent_result)
    return SubOperation(
        raw_fragment=fragment,
        intent_result=intent_result,
        dotnet_payload=dotnet_payload,
    )


def plan_query(query: str) -> QueryPlan:
    raw_query = (query or "").strip()
    q = _normalise(raw_query)

    if not q:
        return QueryPlan(QueryType.SIMPLE, [], 0.0, raw_query, "Empty query")

    if _is_no_split_phrase(q):
        op = _build_sub_operation(q)
        return QueryPlan(QueryType.SIMPLE, [op], 0.95, raw_query, "Contains split-prevention phrase")

    categories = _detect_categories(q)

    # 1. COMPARISON
    comparison_entities = _detect_comparison(q, categories)
    if comparison_entities is not None:
        fragments = _extract_comparison_fragments(q, categories)
        if fragments and len(fragments) >= 2:
            ops = [_build_sub_operation(f) for f in fragments]
            valid_ops = [op for op in ops if op.intent_result.get("category")]
            if len(valid_ops) >= 2:
                return QueryPlan(
                    QueryType.COMPARISON, valid_ops, 0.85, raw_query,
                    f"Comparison signals between {comparison_entities[0]} and {comparison_entities[1]}"
                )

    # 2. CROSS_FILTER
    if _has_cross_category_signal(q, categories):
        fragments = _extract_cross_filter_fragments(q, categories)
        if fragments and len(fragments) >= 2:
            ops = [_build_sub_operation(f) for f in fragments]
            valid_ops = [op for op in ops if op.intent_result.get("category")]
            if len(valid_ops) >= 2:
                op_categories = {op.intent_result["category"] for op in valid_ops}
                if len(op_categories) >= 2:
                    return QueryPlan(
                        QueryType.CROSS_FILTER, valid_ops, 0.85, raw_query,
                        f"Cross-filtering between: {', '.join(sorted(op_categories))}"
                    )

    # 3. MULTI_INDEPENDENT
    multi_fragments = _detect_multi_independent(q, categories)
    if multi_fragments:
        ops = [_build_sub_operation(f) for f in multi_fragments]
        valid_ops = [op for op in ops if op.intent_result.get("category")]
        if len(valid_ops) >= 2:
            op_categories = {op.intent_result["category"] for op in valid_ops}
            if len(op_categories) >= 2:
                return QueryPlan(
                    QueryType.MULTI_INDEPENDENT, valid_ops, 0.80, raw_query,
                    f"Multi-independent domains: {', '.join(sorted(op_categories))}"
                )

    # Default fallback
    op = _build_sub_operation(q)
    confidence = 0.95 if op.intent_result.get("category") else 0.3
    return QueryPlan(QueryType.SIMPLE, [op], confidence, raw_query, "Vague or single-intent context")