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
_COMPARISON_MARKERS = ("compare", "comparison", "versus", "difference between", "vs")
_MULTI_INDEPENDENT_MARKERS = (
    "as well as",
    "along with",
    "together with",
    "also",
    "and also",
)
_CROSS_FILTER_MARKERS = (
    "who plays",
    "who is on leave",
    "currently on leave",
    "currently absent",
    "with medical",
    "with active medical",
    "on leave",
    "medical leave",
    "on medical leave",
    "among",
    "within",
    "who",
    "with",
    "suffering",
    "suffered",
    "had",
    "whose",
)
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
        return "Roster"
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
    if "pass percentage" in text or "pass rate" in text:
        return "passpercentage"
    if "fail percentage" in text or "fail rate" in text:
        return "failpercentage"
    if "attempt" in text:
        return "attemptwise"
    if "best attempt" in text:
        return "bestattempt"
    if "current leave" in text or "leave today" in text or "on leave" in text:
        return "current"
    if "absconded" in text:
        return "absconded"
    if any(marker in text for marker in _TREND_MARKERS):
        return "trend"
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
    if operation in {"passpercentage", "failpercentage"}:
        return "percentage"
    if operation == "count":
        return "count"
    if operation == "average":
        return "average_score" if category == "Performance" else "average"
    if operation == "trend":
        return "trend_value"
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
    if operation in {"passpercentage", "failpercentage"}:
        return "calculate percentage results"
    if operation == "attemptwise":
        return "analyze attempts"
    if operation == "bmi":
        return "review medical BMI records"
    if operation == "bloodgroup":
        return "review blood group records"
    if operation == "current":
        return "show current leave status"
    if operation == "absconded":
        return "find absconded records"
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
        if " and " in body:
            left, right = body.split(" and ", 1)
            return [
                {
                    "fragment": left.strip(),
                    "category": category,
                    "operation": operation,
                    "entities": entities,
                },
                {
                    "fragment": right.strip(),
                    "category": category,
                    "operation": operation,
                    "entities": entities,
                },
            ]
        if " vs " in body:
            left, right = body.split(" vs ", 1)
            return [
                {
                    "fragment": left.strip(),
                    "category": category,
                    "operation": operation,
                    "entities": entities,
                },
                {
                    "fragment": right.strip(),
                    "category": category,
                    "operation": operation,
                    "entities": entities,
                },
            ]
        if " versus " in body:
            left, right = body.split(" versus ", 1)
            return [
                {
                    "fragment": left.strip(),
                    "category": category,
                    "operation": operation,
                    "entities": entities,
                },
                {
                    "fragment": right.strip(),
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

    if any(marker in text for marker in _CROSS_FILTER_MARKERS):
        parts = []
        current = text
        for sep in (r"\bwho\b", r"\bwith\b"):
            split_parts = re.split(sep, current, maxsplit=1, flags=re.IGNORECASE)
            if len(split_parts) > 1:
                parts.append(split_parts[0].strip(" ,"))
                current = " ".join(split_parts[1:])
                break
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
    cross_filter_intent = any(marker in text for marker in _CROSS_FILTER_MARKERS)
    multi_intent = any(marker in text for marker in _MULTI_INDEPENDENT_MARKERS)
    if not multi_intent and " and " in text:
        clause_parts = [
            part.strip(" ,") for part in text.split(" and ") if part.strip(" ,")
        ]
        if len(clause_parts) >= 2:
            clause_categories = []
            for part in clause_parts:
                clause_entities = extract_entities(part, semantic={})
                clause_categories.append(_infer_category(part, clause_entities))
            if len({cat for cat in clause_categories if cat}) >= 2:
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
    elif cross_filter_intent and any(
        token in text
        for token in (
            "with ",
            "who ",
            "among ",
            "within ",
            "players",
            "player",
            "plays",
            "currently on leave",
            "currently absent",
        )
    ):
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
        for marker in (
            "who plays",
            "who is on leave",
            "currently on leave",
            "currently absent",
            "with medical",
            "with active medical",
            "on leave",
            "medical leave",
            "on medical leave",
        ):
            if marker in text:
                head, tail = text.split(marker, 1)
                tail_clean = tail.strip(" ,")
                tail_frag = f"{marker} {tail_clean}" if tail_clean else marker
                sub_requests = [
                    {
                        "fragment": head.strip(" ,"),
                        "category": category,
                        "operation": operation,
                        "entities": entities,
                    },
                    {
                        "fragment": tail_frag,
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
