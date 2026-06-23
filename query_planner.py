"""
query_planner.py
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from admin_intent import _normalise, classify_admin_intent, format_admin_payload

logger = logging.getLogger(__name__)


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


@dataclass
class QueryPlan:
    query_type: QueryType
    operations: List[SubOperation]
    confidence: float
    raw_query: str
    reasoning: str
    filters: Dict[str, Any] = field(default_factory=dict)
    analytics_hint: Optional[str] = None

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
        return d


_COMPARISON_KEYWORDS: List[str] = [
    "compare",
    "comparison",
    " vs ",
    "versus",
    "compared to",
    "compared with",
    "difference between",
    "contrast",
]

_CROSS_FILTER_KEYWORDS: List[str] = [
    "who plays",
    "who play",
    "that play",
    "that plays",
    "which play",
    "which plays",
    "having sport",
    "with sport",
    "on leave",
    "currently on leave",
    "who is on leave",
    "currently absent",
    "with medical",
    "with active medical",
    "with medical case",
    "active medical",
    "in hospital",
    "under treatment",
    "attending today",
    "present today",
    "on campus today",
    "who are in",
    "who is in",
    "among",
    "within",
]

_CROSS_FILTER_CONNECTORS: List[str] = [
    "who",
    "that",
    "which",
    "having",
    "among",
    "within",
    "currently",
    "with",
]

_MULTI_INDEPENDENT_CONNECTORS: List[str] = [
    "along with",
    "as well as",
    "together with",
    "additionally show",
    "also show",
    "and also",
]

_NO_SPLIT_PHRASES: List[str] = [
    "approved and pending leave",
    "approved and pending",
    "annual and medical leave",
    "annual and sick leave",
    "medical and sick leave",
    "top and bottom",
    "top and worst",
    "best and worst",
    "highest and lowest",
    "pass and fail",
    "pass percentage and fail percentage",
    "pass rate and fail rate",
    "improvement and drop",
    "improvement and decline",
    "issued and procured",
    "overdue and returned",
    "pending and completed verification",
    "pending and completed",
]

_ANALYTICS_RANKING_KEYWORDS: List[str] = [
    "highest average",
    "lowest average",
    "best average",
    "worst average",
    "which section has the highest",
    "which section has the lowest",
    "which section is best",
    "which section is worst",
    "rank sections",
    "rank units",
    "rank classes",
    "rank sports",
    "section ranking",
    "unit ranking",
    "which unit has most",
    "which unit has the most",
    "which sport has the best",
    "which sport has most",
    "most absconded",
    "most leave",
    "most absent",
    "top category",
    "best category",
    "worst category",
    "highest pass",
    "lowest pass",
    "highest fail",
    "lowest fail",
    "highest score section",
    "lowest score section",
]

_ANALYTICS_AGGREGATE_KEYWORDS: List[str] = [
    "average by section",
    "average per section",
    "average by unit",
    "average per unit",
    "average by class",
    "average per class",
    "average by sport",
    "average per sport",
    "attendance by unit",
    "attendance per unit",
    "pass percentage by",
    "fail percentage by",
    "score by section",
    "marks by section",
    "group by section",
    "group by unit",
    "group by class",
    "group by sport",
    "breakdown by section",
    "breakdown by unit",
    "grading summary",
]

_GROUP_BY_MAP: Dict[str, str] = {
    "section": "section",
    "unit": "unit",
    "class": "class",
    "sport": "sport",
    "sports": "sport",
    "platoon": "platoon",
    "batch": "batch",
}

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
        "top performer",
        "bottom performer",
        "average score",
        "pass percentage",
        "fail percentage",
        "improvement",
        "drop",
        "attempt",
        "section summary",
    ],
    "Leave": [
        "leave",
        "absent",
        "absentee",
        "absconded",
        "awol",
        "annual leave",
        "medical leave",
        "sick leave",
        "on leave",
        "currently on leave",
    ],
    "Medical": [
        "medical",
        "hospital",
        "bmi",
        "disease",
        "diseases",
        "health",
        "admitted",
        "patient",
        "ward",
        "illness",
        "active medical",
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
        "blood type",
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
        "verified",
        "pending verification",
        "completed verification",
    ],
    "Equipment": [
        "equipment",
        "gear",
        "overdue",
        "inventory",
        "issued items",
        "procured items",
        "damaged",
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
    ],
    "Roster": [
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
    "Overall": [
        "overall",
        "overall performance",
        "composite",
        "allcriteria",
    ],
}

_SPORT_NAMES = {
    "cricket",
    "football",
    "hockey",
    "basketball",
    "volleyball",
    "kabaddi",
    "running",
}


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


def _detect_comparison(
    text_lower: str, categories: List[str]
) -> Optional[Tuple[str, str]]:
    if not any(kw in text_lower for kw in _COMPARISON_KEYWORDS):
        return None
    sections_found = [
        s for s in {"bpet", "bept", "ppt", "firing", "drill"} if s in text_lower
    ]
    if len(sections_found) >= 2:
        return ("Performance", "Performance")
    if len(categories) >= 2:
        return (categories[0], categories[1])
    return None


def _detect_multi_independent(
    text_lower: str, categories: List[str]
) -> Optional[List[str]]:
    for connector in _MULTI_INDEPENDENT_CONNECTORS:
        if connector in text_lower:
            parts = text_lower.split(connector, 1)
            if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                left_cats = _detect_categories(parts[0])
                right_cats = _detect_categories(parts[1])
                if left_cats and right_cats:
                    return [parts[0].strip(), parts[1].strip()]

    if " and " in text_lower and len(categories) >= 2:
        for m in re.finditer(r"\band\b", text_lower):
            left = text_lower[: m.start()].strip()
            right = text_lower[m.end() :].strip()
            if left and right:
                # The old code was wrong because it did not check for 'suffer', 'suffered', or 'suffering' prefixes, leading to premature query splitting on cross-filter medical condition connectors.
                if re.match(
                    r"^(?:is|are|has|have|had|having|was|were|play|plays|who|which|that|with|on|in|under|currently|suffer|suffered|suffering)\b",
                    right,
                ):
                    continue
                lc = _detect_categories(left)
                rc = _detect_categories(right)
                if lc and rc and lc[0] != rc[0]:
                    return [left, right]
    return None


def _is_no_split_phrase(text_lower: str) -> bool:
    return any(phrase in text_lower for phrase in _NO_SPLIT_PHRASES)


def _extract_group_by(text_lower: str) -> Optional[str]:
    patterns = [
        r"\bby\s+(\w+)\b",
        r"\bper\s+(\w+)\b",
        r"\bgroup(?:ed)?\s+by\s+(\w+)\b",
        r"\b(\w+)[- ]wise\b",
    ]
    for pattern in patterns:
        m = re.search(pattern, text_lower)
        if m:
            candidate = m.group(1).lower()
            if candidate in _GROUP_BY_MAP:
                return _GROUP_BY_MAP[candidate]
    return None


def _enrich_right(right_fragment: str) -> str:
    for sport in _SPORT_NAMES:
        if sport in right_fragment:
            return f"sport {right_fragment}"
    return right_fragment


def _detect_analytics(text_lower: str) -> Optional[str]:
    for phrase in _ANALYTICS_RANKING_KEYWORDS:
        if phrase in text_lower:
            if "highest" in phrase or "best" in phrase:
                return "highest"
            if "lowest" in phrase or "worst" in phrase:
                return "lowest"
            if "rank" in phrase:
                return "rank"
            if "most" in phrase:
                return "most"
            if "least" in phrase:
                return "least"
            return "aggregate"

    for phrase in _ANALYTICS_AGGREGATE_KEYWORDS:
        if phrase in text_lower:
            return "aggregate"

    return None


def _is_trend_query(text_lower: str) -> bool:
    trend_kws = {
        "trend",
        "trends",
        "over time",
        "chronological",
        "timeline",
        "timeline wise",
        "timelinewise",
        "daily",
        "weekly",
        "monthly",
        "yearly",
        "last 7 days",
        "last 30 days",
        "last 14 days",
    }
    return any(kw in text_lower for kw in trend_kws)


def _is_distribution_query(
    text_lower: str, categories: List[str], group_by: Optional[str]
) -> bool:
    dist_kws = {
        "distribution",
        "distributions",
        "breakdown",
        "distribute",
        "distributed",
    }
    if any(kw in text_lower for kw in dist_kws):
        return True
    if group_by in ("platoon", "class", "batch", "category") and "by" in text_lower:
        return True
    if "Distribution" in categories:
        return True
    return False


def _extract_cross_filter_fragments(
    text_lower: str, categories: List[str]
) -> Optional[List[str]]:
    nested_patterns = [
        r"\band is\b",
        r"\band currently\b",
        r"\band also\b",
        r"\band are\b",
        r"\band suffered\b",
        r"\band suffering\b",
        r"\band had\b",
        r"\band has\b",
        r"\band with\b",
        r"\band who\b",
        r"\band plays?\b",
        r"\band on\b",
    ]

    primary_split: Optional[Tuple[str, str, str]] = None

    # 1. Keywords first
    for kw in _CROSS_FILTER_KEYWORDS:
        if kw in text_lower:
            idx = text_lower.index(kw)
            left = text_lower[:idx].strip()
            right = text_lower[idx:].strip()
            if left and right:
                primary_split = (left, right, kw)
                break

    # 2. Connectors second (leftmost preferred)
    if primary_split is None:
        best_match = None
        for connector in _CROSS_FILTER_CONNECTORS:
            m = re.search(r"\b" + re.escape(connector) + r"\b", text_lower)
            if m:
                if best_match is None or m.start() < best_match.start():
                    left = text_lower[: m.start()].strip()
                    right = text_lower[m.start() :].strip()
                    if left and right:
                        lc = _detect_categories(left)
                        rc = _detect_categories(_enrich_right(right))
                        if lc and rc and lc[0] != rc[0]:
                            best_match = m
        if best_match is not None:
            primary_split = (
                text_lower[: best_match.start()].strip(),
                text_lower[best_match.start() :].strip(),
                best_match.group(0),
            )

    # 3. Fallback to nested patterns for primary split if still None (leftmost preferred)
    if primary_split is None:
        best_match = None
        best_pat = None
        for pat in nested_patterns:
            m = re.search(pat, text_lower)
            if m:
                if best_match is None or m.start() < best_match.start():
                    left = text_lower[: m.start()].strip()
                    right = text_lower[m.end() :].strip()
                    if left and right:
                        lc = _detect_categories(left)
                        rc = _detect_categories(_enrich_right(right))
                        if lc and rc and lc[0] != rc[0]:
                            best_match = m
                            best_pat = pat
        if best_match is not None:
            primary_split = (
                text_lower[: best_match.start()].strip(),
                text_lower[best_match.end() :].strip(),
                best_match.group(0),
            )

    if primary_split is None:
        return None

    left, right, matched_conn = primary_split
    fragments: List[str] = [left]

    # Iterative/recursive splits on remainder (right part)
    remainder = right
    while True:
        first_match = None
        matched_pat = None
        for pat in nested_patterns:
            m = re.search(pat, remainder)
            if m:
                if first_match is None or m.start() < first_match.start():
                    first_match = m
                    matched_pat = pat
        if first_match:
            sub_left = remainder[: first_match.start()].strip()
            sub_right = remainder[first_match.end() :].strip()
            if sub_left:
                fragments.append(_enrich_right(sub_left))
            remainder = sub_right
        else:
            if remainder:
                fragments.append(_enrich_right(remainder))
            break

    return fragments


def _extract_comparison_fragments(
    text_lower: str, categories: List[str]
) -> Optional[List[str]]:
    category_tokens = {cat.lower() for cat in _CATEGORY_SIGNALS.keys()}

    def _normalize_shared_category(left: str, right: str) -> Tuple[str, str]:
        tail_match = re.search(
            r"\b(" + "|".join(re.escape(tok) for tok in sorted(category_tokens)) + r")\b\s*$",
            right,
        )
        if tail_match:
            shared_category = tail_match.group(1)
            right_core = right[: tail_match.start()].strip()
            left_core = left.strip()
            return f"{left_core} {shared_category}".strip(), f"{right_core} {shared_category}".strip()
        return left, right

    for sep in [" vs ", " versus "]:
        if sep in text_lower:
            parts = text_lower.split(sep, 1)
            if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                left, right = parts[0].strip(), parts[1].strip()
                for kw in ["compare", "comparison"]:
                    left = re.sub(r"^" + kw + r"\s+", "", left).strip()
                return list(_normalize_shared_category(left, right))

    m = re.search(r"\bcompare\s+(.+?)\s+and\s+(.+)", text_lower)
    if m:
        return list(_normalize_shared_category(m.group(1).strip(), m.group(2).strip()))

    m = re.search(r"(.+?)\s+compared\s+(?:to|with)\s+(.+)", text_lower)
    if m:
        return list(_normalize_shared_category(m.group(1).strip(), m.group(2).strip()))

    return None


def _extract_filtered_comparison_fragments(
    text_lower: str,
) -> Optional[List[str]]:
    filter_m = re.search(r"\b(?:among|for|with|within|by)\s+(.+)$", text_lower)
    if filter_m is None:
        return None

    filter_text = filter_m.group(1).strip()
    filter_text = re.sub(
        r"\b(?:players?|person|agniveers?|trainees?)\b", "", filter_text
    ).strip()
    if not filter_text:
        return None

    body = text_lower[: filter_m.start()].strip()

    comparison_frags = _extract_comparison_fragments(body, [])
    if comparison_frags and len(comparison_frags) == 2:
        return [
            f"{comparison_frags[0]} {filter_text}".strip(),
            f"{comparison_frags[1]} {filter_text}".strip(),
        ]

    m = re.search(r"\bcompare\s+(.+?)\s+and\s+(.+)", body)
    if m:
        return [
            f"{m.group(1).strip()} {filter_text}".strip(),
            f"{m.group(2).strip()} {filter_text}".strip(),
        ]

    return None


def _extract_filters_dict(intent: Dict[str, Any]) -> Dict[str, Any]:
    filters = {}
    mapping = {
        "sport": "sport",
        "class": "class",
        "section": "section",
        "sub_section": "subSection",
        "attempt_no": "attemptNo",
        "from_attempt": "fromAttempt",
        "to_attempt": "toAttempt",
        "leave_type": "leaveType",
        "company_id": "companyId",
        "platoon_id": "platoonId",
        "batch_id": "batchId",
        "date": "date",
        "from_date": "fromDate",
        "to_date": "toDate",
        "agniveer_no": "agniveerNo",
        "bmi_category": "bmiCategory",
        "medical_status": "medicalStatus",
        "number": "n",
        "blood_group": "bloodGroup",
        "item_name": "equipmentName",
        "unit_name": "unitName",
    }
    for key, filter_name in mapping.items():
        val = intent.get(key)
        if val is not None:
            filters[filter_name] = val

    if intent.get("leave_type") == "Current":
        filters["leaveStatus"] = "Current"

    return filters


def _build_sub_operation(
    fragment: str,
    group_by: Optional[str] = None,
    filter_fragment: Optional[str] = None,
) -> SubOperation:
    intent_result = classify_admin_intent(fragment)
    dotnet_payload = format_admin_payload(intent_result)
    return SubOperation(
        raw_fragment=fragment,
        intent_result=intent_result,
        dotnet_payload=dotnet_payload,
        group_by=group_by,
        filter_fragment=filter_fragment,
    )


def plan_query(query: str) -> QueryPlan:
    raw_query = (query or "").strip()
    q = _normalise(raw_query)

    if not q:
        return QueryPlan(QueryType.SIMPLE, [], 0.0, raw_query, "Empty query")

    if _is_no_split_phrase(q):
        op = _build_sub_operation(q)
        filters = _extract_filters_dict(op.intent_result)
        return QueryPlan(
            QueryType.SIMPLE,
            [op],
            0.95,
            raw_query,
            "Contains split-prevention phrase",
            filters=filters,
        )

    categories = _detect_categories(q)
    group_by = _extract_group_by(q)

    if _is_trend_query(q):
        op = _build_sub_operation(q, group_by=group_by)
        filters = _extract_filters_dict(op.intent_result)
        return QueryPlan(
            QueryType.TREND,
            [op],
            0.85,
            raw_query,
            "Trend/timeline query detected",
            filters=filters,
        )

    multi_fragments = _detect_multi_independent(q, categories)
    if multi_fragments and not any(kw in q for kw in _COMPARISON_KEYWORDS):
        ops = [_build_sub_operation(f) for f in multi_fragments]
        valid_ops = [op for op in ops if op.intent_result.get("category")]
        if len(valid_ops) >= 2:
            op_categories = {op.intent_result["category"] for op in valid_ops}
            if len(op_categories) >= 2:
                combined_filters = {}
                for op in valid_ops:
                    combined_filters.update(_extract_filters_dict(op.intent_result))
                return QueryPlan(
                    QueryType.MULTI_INDEPENDENT,
                    valid_ops,
                    0.80,
                    raw_query,
                    f"Multi-independent: {', '.join(sorted(op_categories))}",
                    filters=combined_filters,
                )

    if _is_distribution_query(q, categories, group_by):
        op = _build_sub_operation(q, group_by=group_by)
        filters = _extract_filters_dict(op.intent_result)
        return QueryPlan(
            QueryType.DISTRIBUTION,
            [op],
            0.85,
            raw_query,
            "Distribution/breakdown query detected",
            filters=filters,
        )

    analytics_hint = _detect_analytics(q)
    if analytics_hint:
        op = _build_sub_operation(q, group_by=group_by)
        filters = _extract_filters_dict(op.intent_result)
        qtype = QueryType.DISTRIBUTION if group_by else QueryType.SIMPLE
        return QueryPlan(
            qtype,
            [op],
            0.85,
            raw_query,
            f"Analytics/ranking query detected: hint={analytics_hint}",
            filters=filters,
            analytics_hint=analytics_hint,
        )

    if any(kw in q for kw in _COMPARISON_KEYWORDS):
        filtered_frags = _extract_filtered_comparison_fragments(q)
        if filtered_frags and len(filtered_frags) >= 2:
            ops = [_build_sub_operation(f) for f in filtered_frags]
            valid_ops = [op for op in ops if op.intent_result.get("category")]
            if len(valid_ops) >= 2:
                combined_filters = {}
                for op in valid_ops:
                    combined_filters.update(_extract_filters_dict(op.intent_result))
                return QueryPlan(
                    QueryType.COMPARE,
                    valid_ops,
                    0.85,
                    raw_query,
                    "Filtered comparison: each side carries the cross-filter",
                    filters=combined_filters,
                )

    comparison_entities = _detect_comparison(q, categories)
    if comparison_entities is not None:
        fragments = _extract_comparison_fragments(q, categories)
        if fragments and len(fragments) >= 2:
            ops = []
            for i, f in enumerate(fragments):
                op = _build_sub_operation(f)
                if not op.intent_result.get("category") and i < len(
                    comparison_entities
                ):
                    op.intent_result["category"] = comparison_entities[i]
                    op.dotnet_payload = format_admin_payload(op.intent_result)
                ops.append(op)
            valid_ops = [op for op in ops if op.intent_result.get("category")]
            if len(valid_ops) >= 2:
                combined_filters = {}
                for op in valid_ops:
                    combined_filters.update(_extract_filters_dict(op.intent_result))
                return QueryPlan(
                    QueryType.COMPARE,
                    valid_ops,
                    0.85,
                    raw_query,
                    f"Comparison between {comparison_entities[0]} and "
                    f"{comparison_entities[1]}",
                    filters=combined_filters,
                )

    if _has_cross_category_signal(q, categories):
        fragments = _extract_cross_filter_fragments(q, categories)
        if fragments and len(fragments) >= 2:
            ops = [_build_sub_operation(f) for f in fragments]
            valid_ops = [op for op in ops if op.intent_result.get("category")]
            if len(valid_ops) >= 2:
                op_categories = {op.intent_result["category"] for op in valid_ops}
                if len(op_categories) >= 2:
                    depth = "3-way" if len(valid_ops) >= 3 else "2-way"
                    combined_filters = {}
                    for op in valid_ops:
                        combined_filters.update(_extract_filters_dict(op.intent_result))
                    return QueryPlan(
                        QueryType.CROSS_FILTER,
                        valid_ops,
                        0.85,
                        raw_query,
                        f"{depth} cross-filter: {', '.join(sorted(op_categories))}",
                        filters=combined_filters,
                    )

    op = _build_sub_operation(q, group_by=group_by)
    confidence = 0.95 if op.intent_result.get("category") else 0.3
    filters = _extract_filters_dict(op.intent_result)
    return QueryPlan(
        QueryType.SIMPLE,
        [op],
        confidence,
        raw_query,
        "Single-intent query with filters",
        filters=filters,
    )
