"""
query_planner.py
================
Query Planning Layer for the AgniAI Admin Chatbot.

Classifies incoming admin queries into one of four types:

  SIMPLE           → Single category, single operation (existing flow)
  CROSS_FILTER     → Two+ categories, intersect results by Agniveer ID
  COMPARISON       → Compare metrics from two queries side-by-side
  MULTI_INDEPENDENT → Two+ independent queries, return both result sets

The planner sits BEFORE intent execution and decides how many API calls
are needed, then delegates to the existing classify_admin_intent() for
each sub-operation.

DESIGN PRINCIPLES:
  1. Rule-based detection (no LLM overhead in the hot path)
  2. Category-aware splitting — only splits on connectors that bridge
     different admin categories
  3. Conservative — defaults to SIMPLE when unsure
  4. Reuses existing admin_intent.py for all intent classification
  5. Logs every decision for debugging
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from admin_intent import (
    _MODULES,
    _match_module,
    _normalise,
    _score_intent,
    classify_admin_intent,
    format_admin_payload,
)

logger = logging.getLogger(__name__)


# =============================================================================
# QUERY TYPES
# =============================================================================

class QueryType(Enum):
    """The four supported query plan types."""
    SIMPLE = "simple"
    CROSS_FILTER = "cross_filter"
    COMPARISON = "comparison"
    MULTI_INDEPENDENT = "multi_independent"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SubOperation:
    """One intent within a multi-operation plan."""
    raw_fragment: str                           # The sub-query text
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
    """Complete plan for executing an admin query."""
    query_type: QueryType
    operations: List[SubOperation]
    confidence: float                           # 0.0 – 1.0
    raw_query: str
    reasoning: str                              # Why this type was chosen

    def to_dict(self) -> Dict[str, Any]:
        return {
            "queryType": self.query_type.value,
            "confidence": round(self.confidence, 2),
            "operationCount": len(self.operations),
            "reasoning": self.reasoning,
            "operations": [op.to_dict() for op in self.operations],
        }


# =============================================================================
# KEYWORD SETS
# =============================================================================

# ── Comparison keywords ────────────────────────────────────────────────────
_COMPARISON_KEYWORDS: List[str] = [
    "compare",
    "comparison",
    " vs ",
    "versus",
    "compared to",
    "compared with",
    "difference between",
    "contrast",
    "head to head",
    "side by side",
]

# ── Cross-filter keywords ─────────────────────────────────────────────────
# These indicate "records from category A that match a condition in category B"
_CROSS_FILTER_KEYWORDS: List[str] = [
    "who plays",
    "who play",
    "who are in",
    "who is in",
    "among",
    "that play",
    "that plays",
    "which play",
    "which plays",
    "having sport",
    "with sport",
    "among football",
    "among cricket",
    "among hockey",
    "among basketball",
    "among volleyball",
    "among kabaddi",
    "among running",
]

# Broader cross-filter connectors (require more context to activate)
_CROSS_FILTER_CONNECTORS: List[str] = [
    "who",
    "that",
    "which",
    "having",
    "among",
]

# ── Multi-independent connectors ──────────────────────────────────────────
_MULTI_INDEPENDENT_CONNECTORS: List[str] = [
    "along with",
    "as well as",
    "together with",
    "additionally show",
    "also show",
    "and also",
]

# ── Phrases that should NEVER be split (same-category compounds) ──────────
_NO_SPLIT_PHRASES: List[str] = [
    # Leave compounds
    "approved and pending leave",
    "approved and pending",
    "annual and medical leave",
    "annual and sick leave",
    "medical and sick leave",
    # Performance compounds
    "top and bottom",
    "top and worst",
    "best and worst",
    "highest and lowest",
    "pass and fail",
    "pass percentage and fail percentage",
    "pass rate and fail rate",
    "improvement and drop",
    "improvement and decline",
    # Equipment compounds
    "issued and procured",
    "overdue and returned",
    # Verification compounds
    "pending and completed verification",
    "pending and completed",
]

# ── Category keywords (for detecting which categories a query references) ──
_CATEGORY_SIGNALS: Dict[str, List[str]] = {
    "Performance": [
        "performance", "performer", "performers", "score", "marks",
        "bpet", "ppt", "firing", "drill", "grading", "grade",
        "top performer", "bottom performer", "average score",
        "pass percentage", "fail percentage", "improvement", "drop",
        "attempt", "section summary", "overall performance",
    ],
    "Leave": [
        "leave", "absent", "absentee", "absconded", "awol",
        "annual leave", "medical leave", "sick leave",
    ],
    "Medical": [
        "medical", "hospital", "bmi", "disease", "health",
        "admitted", "patient", "ward", "illness",
    ],
    "Attendance": [
        "attendance", "present", "campus", "strength",
        "headcount", "monthly attendance",
    ],
    "Verification": [
        "verification", "verified", "pending verification",
        "completed verification",
    ],
    "Equipment": [
        "equipment", "gear", "overdue", "inventory",
        "issued items", "procured items", "damaged",
    ],
    "Distribution": [
        "distribution", "unit", "unassigned", "distributed",
    ],
    "Skills": [
        "sport", "sports", "cricket", "football", "hockey",
        "basketball", "volleyball", "kabaddi", "running",
        "blood group", "blood type", "class", "roster",
        "sikh", "dogra", "jat", "gurkha", "rajput", "punjabi",
    ],
}


# =============================================================================
# CATEGORY DETECTION
# =============================================================================

def _detect_categories(text_lower: str) -> List[str]:
    """
    Detect which admin categories are referenced in the query.
    Returns a list of category names, ordered by signal strength (strongest first).
    """
    scores: Dict[str, int] = {}
    for category, signals in _CATEGORY_SIGNALS.items():
        score = 0
        for signal in signals:
            if signal in text_lower:
                score += len(signal.split())
        if score > 0:
            scores[category] = score

    # Sort by score descending
    sorted_cats = sorted(scores.keys(), key=lambda c: scores[c], reverse=True)
    return sorted_cats


def _has_cross_category_signal(text_lower: str, categories: List[str]) -> bool:
    """
    Return True if the query mentions at least 2 different categories
    AND has a cross-filter connector bridging them.
    """
    if len(categories) < 2:
        return False

    # Check for specific cross-filter phrases first
    for phrase in _CROSS_FILTER_KEYWORDS:
        if phrase in text_lower:
            return True

    # Check for broader connectors only if 2+ categories detected
    for connector in _CROSS_FILTER_CONNECTORS:
        # Match as whole word to avoid false positives
        pattern = r"\b" + re.escape(connector) + r"\b"
        if re.search(pattern, text_lower):
            return True

    return False


# =============================================================================
# COMPARISON DETECTION
# =============================================================================

def _detect_comparison(text_lower: str, categories: List[str]) -> Optional[Tuple[str, str]]:
    """
    Detect if the query is a comparison between two entities.

    Returns a tuple of (entity_a, entity_b) if comparison detected, else None.

    IMPORTANT: If the comparison is within a single Performance section
    (e.g., "compare PPT and BPET"), the existing .NET API handles this
    natively via the Compare operation — so we return None to keep it SIMPLE.
    """
    has_compare_keyword = False
    for kw in _COMPARISON_KEYWORDS:
        if kw in text_lower:
            has_compare_keyword = True
            break

    if not has_compare_keyword:
        return None

    # Check if this is an intra-Performance section comparison
    # The .NET API already handles: "compare PPT and BPET", "PPT vs BPET"
    _SECTIONS = {"bpet", "ppt", "firing", "drill"}
    sections_found = [s for s in _SECTIONS if s in text_lower]

    if len(sections_found) >= 2 and ("Performance" in categories or len(categories) <= 1):
        # This is a standard Performance comparison — let .NET handle it
        logger.debug(
            "Comparison detected within Performance sections %s — keeping SIMPLE",
            sections_found,
        )
        return None

    # Cross-category comparison (e.g., "compare attendance and leave")
    if len(categories) >= 2:
        return (categories[0], categories[1])

    # Intra-category comparison with different entities
    # (e.g., "compare BPET top 5 and BPET bottom 5" — less common)
    return None


# =============================================================================
# MULTI-INDEPENDENT DETECTION
# =============================================================================

def _detect_multi_independent(
    text_lower: str,
    categories: List[str],
) -> Optional[List[str]]:
    """
    Detect if the query asks for multiple independent pieces of data.

    Returns the list of split fragments if detected, else None.
    """
    # Check for explicit multi-independent connectors first
    for connector in _MULTI_INDEPENDENT_CONNECTORS:
        if connector in text_lower:
            parts = text_lower.split(connector, 1)
            if len(parts) == 2:
                left = parts[0].strip()
                right = parts[1].strip()
                if left and right:
                    left_cats = _detect_categories(left)
                    right_cats = _detect_categories(right)
                    if left_cats and right_cats and left_cats[0] != right_cats[0]:
                        return [left, right]

    # Check for "and" connector — most common but most prone to false splits
    if " and " in text_lower and len(categories) >= 2:
        # Find "and" positions and check if they bridge different categories
        and_positions = [m.start() for m in re.finditer(r"\band\b", text_lower)]

        for pos in and_positions:
            left = text_lower[:pos].strip()
            right = text_lower[pos + 4:].strip()  # len(" and") == 4

            if not left or not right:
                continue

            left_cats = _detect_categories(left)
            right_cats = _detect_categories(right)

            # Only split if left and right resolve to DIFFERENT categories
            if (left_cats and right_cats
                    and left_cats[0] != right_cats[0]):
                return [left, right]

    return None


# =============================================================================
# FALSE SPLIT GUARD
# =============================================================================

def _is_no_split_phrase(text_lower: str) -> bool:
    """Return True if the query contains a known compound phrase that should NOT be split."""
    for phrase in _NO_SPLIT_PHRASES:
        if phrase in text_lower:
            return True
    return False


# =============================================================================
# CROSS-FILTER FRAGMENT EXTRACTION
# =============================================================================

def _extract_cross_filter_fragments(
    text_lower: str,
    categories: List[str],
) -> Optional[List[str]]:
    """
    Split a cross-filter query into its component fragments.

    Example: "Show top performer in PPT who plays cricket"
    → ["show top performer in ppt", "sport cricket"]

    The right fragment is enriched with category context words so that
    classify_admin_intent() can resolve it even from a short phrase.
    """
    # ── Sport-enrichment helper ────────────────────────────────────────────
    _SPORT_NAMES = {"cricket", "football", "hockey", "basketball",
                    "volleyball", "kabaddi", "running"}

    def _enrich_right(right_fragment: str) -> str:
        """Add 'sport' prefix if the fragment mentions a sport but lacks
        enough admin keywords for classify_admin_intent() to detect it."""
        for sport in _SPORT_NAMES:
            if sport in right_fragment:
                # Replace vague phrases with a clear Skills-domain query
                return f"sport {sport}"
        return right_fragment

    # Try each cross-filter keyword as a split point
    for kw in _CROSS_FILTER_KEYWORDS:
        if kw in text_lower:
            idx = text_lower.index(kw)
            left = text_lower[:idx].strip()
            right = text_lower[idx:].strip()
            if left and right:
                right = _enrich_right(right)
                return [left, right]

    # Try broader connectors with category verification
    for connector in _CROSS_FILTER_CONNECTORS:
        pattern = r"\b" + re.escape(connector) + r"\b"
        match = re.search(pattern, text_lower)
        if match:
            idx = match.start()
            left = text_lower[:idx].strip()
            right = text_lower[idx:].strip()
            if left and right:
                left_cats = _detect_categories(left)
                right_enriched = _enrich_right(right)
                right_cats = _detect_categories(right_enriched)
                if (left_cats and right_cats
                        and left_cats[0] != right_cats[0]):
                    return [left, right_enriched]

    return None


# =============================================================================
# COMPARISON FRAGMENT EXTRACTION
# =============================================================================

def _extract_comparison_fragments(
    text_lower: str,
    categories: List[str],
) -> Optional[List[str]]:
    """
    Split a comparison query into its component fragments.

    Example: "Compare PPT and BEPT performance"
    → ["ppt performance", "bpet performance"]
    """
    # Try "vs" / "versus"
    for sep in [" vs ", " versus "]:
        if sep in text_lower:
            parts = text_lower.split(sep, 1)
            if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                # Strip comparison keywords from fragments
                left = parts[0].strip()
                right = parts[1].strip()
                for kw in ["compare", "comparison"]:
                    left = re.sub(r"^" + kw + r"\s+", "", left).strip()
                return [left, right]

    # Try "compare X and Y" pattern
    compare_match = re.search(
        r"\bcompare\s+(.+?)\s+and\s+(.+)",
        text_lower,
    )
    if compare_match:
        left = compare_match.group(1).strip()
        right = compare_match.group(2).strip()
        if left and right:
            return [left, right]

    # Try "X compared to/with Y"
    compared_match = re.search(
        r"(.+?)\s+compared\s+(?:to|with)\s+(.+)",
        text_lower,
    )
    if compared_match:
        left = compared_match.group(1).strip()
        right = compared_match.group(2).strip()
        if left and right:
            return [left, right]

    # Try "difference between X and Y"
    diff_match = re.search(
        r"difference\s+between\s+(.+?)\s+and\s+(.+)",
        text_lower,
    )
    if diff_match:
        left = diff_match.group(1).strip()
        right = diff_match.group(2).strip()
        if left and right:
            return [left, right]

    # Fallback: split on "and" if we have 2+ categories
    if " and " in text_lower and len(categories) >= 2:
        for m in re.finditer(r"\band\b", text_lower):
            pos = m.start()
            left = text_lower[:pos].strip()
            right = text_lower[pos + 3:].strip()
            if left and right:
                # Strip comparison keywords
                for kw in ["compare", "comparison"]:
                    left = re.sub(r"^" + kw + r"\s+", "", left).strip()
                    right = re.sub(r"\s+" + kw + r"$", "", right).strip()
                return [left, right]

    return None


# =============================================================================
# BUILD SUB-OPERATIONS
# =============================================================================

def _build_sub_operation(fragment: str) -> SubOperation:
    """
    Build a SubOperation by running the fragment through the existing
    classify_admin_intent() + format_admin_payload() pipeline.
    """
    intent_result = classify_admin_intent(fragment)
    dotnet_payload = format_admin_payload(intent_result)
    return SubOperation(
        raw_fragment=fragment,
        intent_result=intent_result,
        dotnet_payload=dotnet_payload,
    )


# =============================================================================
# PUBLIC API
# =============================================================================

def plan_query(query: str) -> QueryPlan:
    """
    Analyse an admin natural-language question and produce a QueryPlan
    describing how to execute it.

    This function:
      1. Normalises the query
      2. Detects referenced categories
      3. Checks for comparison / cross-filter / multi-independent signals
      4. Builds sub-operations using classify_admin_intent()
      5. Returns a QueryPlan with confidence score

    SIMPLE queries (the vast majority) pass through with minimal overhead.
    """
    raw_query = (query or "").strip()
    q = _normalise(raw_query)

    # ── Guard: empty query ─────────────────────────────────────────────────
    if not q:
        return QueryPlan(
            query_type=QueryType.SIMPLE,
            operations=[],
            confidence=0.0,
            raw_query=raw_query,
            reasoning="Empty query",
        )

    # ── Guard: no-split phrases ────────────────────────────────────────────
    if _is_no_split_phrase(q):
        logger.debug("No-split phrase detected, keeping SIMPLE: %r", q)
        op = _build_sub_operation(q)
        return QueryPlan(
            query_type=QueryType.SIMPLE,
            operations=[op],
            confidence=0.95,
            raw_query=raw_query,
            reasoning=f"Contains no-split compound phrase",
        )

    # ── Detect categories referenced ───────────────────────────────────────
    categories = _detect_categories(q)

    logger.debug("Detected categories for %r: %s", q, categories)

    # ── Check 1: COMPARISON ────────────────────────────────────────────────
    comparison_entities = _detect_comparison(q, categories)
    if comparison_entities is not None:
        fragments = _extract_comparison_fragments(q, categories)
        if fragments and len(fragments) >= 2:
            ops = [_build_sub_operation(f) for f in fragments]
            # Verify that at least some operations resolved
            valid_ops = [op for op in ops if op.intent_result.get("category")]
            if len(valid_ops) >= 2:
                confidence = 0.85
                logger.info(
                    "COMPARISON plan: %d operations, confidence=%.2f, fragments=%s",
                    len(valid_ops), confidence, fragments,
                )
                return QueryPlan(
                    query_type=QueryType.COMPARISON,
                    operations=valid_ops,
                    confidence=confidence,
                    raw_query=raw_query,
                    reasoning=(
                        f"Comparison keywords detected between "
                        f"{comparison_entities[0]} and {comparison_entities[1]}"
                    ),
                )

    # ── Check 2: CROSS_FILTER ──────────────────────────────────────────────
    if _has_cross_category_signal(q, categories):
        fragments = _extract_cross_filter_fragments(q, categories)
        if fragments and len(fragments) >= 2:
            ops = [_build_sub_operation(f) for f in fragments]
            valid_ops = [op for op in ops if op.intent_result.get("category")]
            if len(valid_ops) >= 2:
                # Verify they are different categories (cross-filter requirement)
                op_categories = {op.intent_result["category"] for op in valid_ops}
                if len(op_categories) >= 2:
                    confidence = 0.85
                    logger.info(
                        "CROSS_FILTER plan: %d operations, confidence=%.2f, "
                        "categories=%s, fragments=%s",
                        len(valid_ops), confidence, op_categories, fragments,
                    )
                    return QueryPlan(
                        query_type=QueryType.CROSS_FILTER,
                        operations=valid_ops,
                        confidence=confidence,
                        raw_query=raw_query,
                        reasoning=(
                            f"Cross-filter between categories: "
                            f"{', '.join(sorted(op_categories))}"
                        ),
                    )

    # ── Check 3: MULTI_INDEPENDENT ─────────────────────────────────────────
    multi_fragments = _detect_multi_independent(q, categories)
    if multi_fragments:
        ops = [_build_sub_operation(f) for f in multi_fragments]
        valid_ops = [op for op in ops if op.intent_result.get("category")]
        if len(valid_ops) >= 2:
            op_categories = {op.intent_result["category"] for op in valid_ops}
            if len(op_categories) >= 2:
                confidence = 0.80
                logger.info(
                    "MULTI_INDEPENDENT plan: %d operations, confidence=%.2f, "
                    "categories=%s, fragments=%s",
                    len(valid_ops), confidence, op_categories, multi_fragments,
                )
                return QueryPlan(
                    query_type=QueryType.MULTI_INDEPENDENT,
                    operations=valid_ops,
                    confidence=confidence,
                    raw_query=raw_query,
                    reasoning=(
                        f"Independent queries for categories: "
                        f"{', '.join(sorted(op_categories))}"
                    ),
                )

    # ── Default: SIMPLE ────────────────────────────────────────────────────
    op = _build_sub_operation(q)
    confidence = 0.95 if op.intent_result.get("category") else 0.3

    logger.debug(
        "SIMPLE plan: category=%s, confidence=%.2f",
        op.intent_result.get("category"), confidence,
    )

    return QueryPlan(
        query_type=QueryType.SIMPLE,
        operations=[op],
        confidence=confidence,
        raw_query=raw_query,
        reasoning="Single category query or no multi-operation signal detected",
    )
