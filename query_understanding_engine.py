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
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional

from conversation_detector import is_conversational_query, normalize_text

_SECTION_ALIASES = {
    "bpet": "BPET",
    "bept": "BPET",
    "ppt": "PPT",
    "firing": "Firing",
    "drill": "Drill",
}

_CATEGORY_SIGNALS = {
    "Performance": (
        "performance",
        "score",
        "marks",
        "performer",
        "grading",
        "top",
        "bottom",
        "highest",
        "lowest",
        "best",
        "worst",
        "weakest",
        "improvement",
        "decline",
        "drop",
    ),
    "Attendance": ("attendance", "present", "campus", "absent"),
    "Medical": ("medical", "bmi", "disease", "hospital", "health", "sick", "blood"),
    "Leave": ("leave", "absconded", "absent", "on leave"),
    "Equipment": ("equipment", "overdue", "inventory", "issued", "procured"),
    "Verification": ("verification", "verified", "pending", "completed"),
    "Distribution": ("distribution", "unassigned", "assigned"),
    "Skills": ("sport", "sports", "roster", "class"),
    "Strength": ("strength", "headcount"),
    "Overall": ("overall", "composite"),
}

_RANKING_WORDS = ("top", "highest", "best", "most", "maximum", "leading")
_LOW_RANKING_WORDS = ("lowest", "worst", "weakest", "minimum", "fewest", "bottom", "least")
_TREND_WORDS = ("trend", "over time", "over months", "over days", "growth", "decline", "increase", "decrease")
_DISTRIBUTION_WORDS = ("distribution", "breakdown", "share", "percentage", "composition")
_COMPARE_WORDS = ("compare", "comparison", "versus", "vs", "difference")
_COUNT_WORDS = ("how many", "count", "number of", "total")
_AVERAGE_WORDS = ("average", "avg", "mean")
_INTERSECTION_WORDS = ("who plays", "who is on leave", "currently on leave", "with medical", "intersection", "common")


@dataclass
class QueryUnderstanding:
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


def _detect_category(text: str, section: Optional[str]) -> Optional[str]:
    if section in {"BPET", "Firing", "Drill"}:
        return "Performance"
    if "strength" in text:
        return "Strength"
    for category, signals in _CATEGORY_SIGNALS.items():
        if any(signal in text for signal in signals):
            return category
    if "cricket" in text or "football" in text:
        if any(term in text for term in ("roster", "players", "list", "show")):
            return "Skills"
    return None


def _detect_operation(text: str) -> str:
    if any(word in text for word in _COMPARE_WORDS):
        return "compare"
    if any(word in text for word in _TREND_WORDS):
        return "trend"
    if any(word in text for word in _DISTRIBUTION_WORDS):
        return "distribution"
    if any(word in text for word in _INTERSECTION_WORDS):
        return "intersection"
    if any(phrase in text for phrase in _COUNT_WORDS):
        return "count"
    if any(phrase in text for phrase in _AVERAGE_WORDS):
        return "average"
    if any(word in text for word in _LOW_RANKING_WORDS):
        return "ranking"
    if any(word in text for word in _RANKING_WORDS):
        return "ranking"
    return "lookup"


def _detect_sort(operation: str, text: str) -> Optional[str]:
    if operation != "ranking":
        return None
    if any(word in text for word in _LOW_RANKING_WORDS):
        return "ascending"
    if any(word in text for word in _RANKING_WORDS):
        return "descending"
    return None


def _detect_metric(category: Optional[str], operation: str, text: str) -> Optional[str]:
    if "percentage" in text or "pass rate" in text or "fail rate" in text:
        return "percentage"
    if operation == "count":
        return "count"
    if operation == "trend":
        return "trend_value"
    if operation in {"compare", "ranking", "average"}:
        if category == "Performance":
            return "average_score"
        return "count" if operation == "compare" else "average_score"
    return None


def _detect_group_by(text: str) -> Optional[str]:
    for candidate in ("platoon", "class", "batch", "company", "section", "sport", "unit"):
        if re.search(rf"\b{candidate}\b", text):
            return candidate
    return None


def _build_user_goal(
    operation: str,
    category: Optional[str],
    section: Optional[str],
    text: str,
    group_by: Optional[str] = None,
) -> str:
    if operation == "ranking":
        if any(word in text for word in _LOW_RANKING_WORDS):
            if group_by:
                return f"find weakest {group_by}"
            if section:
                return f"find weakest {section.lower()}"
            target = category.lower() if category else "records"
            return f"find weakest {target}"
        if any(word in text for word in _RANKING_WORDS):
            if group_by:
                return f"find top {group_by}"
            if section:
                return f"find top {section.lower()}"
            target = category.lower() if category else "records"
            return f"find top {target}"
        return "rank records"
    if operation == "compare":
        return "compare the requested entities"
    if operation == "trend":
        return "analyze the trend over time"
    if operation == "distribution":
        return "show the distribution breakdown"
    if operation == "count":
        return "count the matching records"
    if operation == "average":
        return "compute the average value"
    if operation == "intersection":
        return "find records matching all conditions"
    if category:
        return f"review {category.lower()} data"
    return "understand the request"


def understand_query(query: str) -> Dict[str, Any]:
    text = normalize_text(query)
    conversational = is_conversational_query(text)
    if conversational:
        result = QueryUnderstanding(
            user_goal="conversational",
            operation="conversation",
            query_type="conversational",
            confidence=0.99 if text else 1.0,
            conversational=True,
        )
        return result.to_dict()

    section = _extract_section(text)
    category = _detect_category(text, section)
    operation = _detect_operation(text)
    sort = _detect_sort(operation, text)
    metric = _detect_metric(category, operation, text)
    group_by = _detect_group_by(text)

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

    query_type = "analytical"
    if operation == "compare":
        query_type = "compare"
    elif operation == "trend":
        query_type = "trend"
    elif operation == "distribution":
        query_type = "distribution"
    elif operation == "intersection":
        query_type = "cross_filter"
    elif operation == "ranking" and (category == "Performance" or section):
        query_type = "ranking"
    elif operation == "count" and category is None and not section:
        query_type = "unclear"

    filters: Dict[str, Any] = {}
    if section:
        filters["section"] = section
    if group_by:
        filters["group_by"] = group_by

    result = QueryUnderstanding(
        user_goal=_build_user_goal(operation, category, section, text, group_by),
        operation=operation,
        category=category,
        section=section,
        metric=metric,
        sort=sort,
        query_type=query_type,
        confidence=min(0.99, round(confidence, 2)),
        group_by=group_by,
        filters=filters,
        conversational=False,
    )
    return result.to_dict()
