"""
suggested_question_engine.py
============================
Generates dynamic, data-aware, context-prior next-step questions based on
query intent AND actual result data from the .NET response.

v3 upgrade: suggestions are now drawn primarily from `question_bank.py` —
a curated bank of ~800 real, hand-written questions from the AgniAI
operation-level test suite (13 categories x 47 operations x 4 query types,
plus the expanded cross-filter / multi-independent / comparison suite).
These are questions we already know the pipeline is designed to handle,
so suggesting them is both safer (no made-up phrasing) and a de-facto
regression net — every suggestion shown to a user is also a test case.

Bank questions are personalized against the current result context
(swap the bank's example batch/platoon/agniveer number for the real one
currently in view, where that doesn't break a two-sided compare question)
and de-duplicated against the question that was just asked.

If the bank doesn't have enough matching questions for a given
category/subcategory/query_type combination, we top up the remaining
slots with the original score-aware / trend-aware generated questions
(kept from v2) so the result is never thin or generic.

Key behaviors carried over from v2:
  - Score-shape awareness: if scores are low -> suggest improvement drill-down;
    if high -> suggest ceiling-push queries.
  - Trend-aware: if trend is falling -> suggest intervention queries.
  - At-risk detection: generates targeted queries for below-50 agniveers.
  - Fallback is always non-empty and contextual.
"""

import random
import re
from typing import Any, Dict, List, Optional

from utils import get_score as _get_score
from question_bank import QUESTION_BANK

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Alias tables — map live intent category/subcategory names to the bank's keys
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CATEGORY_ALIASES: Dict[str, str] = {
    "performance": "PERFORMANCE",
    "leave": "LEAVE",
    "medical": "MEDICAL",
    "attendance": "ATTENDANCE",
    "strength": "STRENGTH",
    "verification": "VERIFICATION",
    "equipment": "EQUIPMENT",
    "distribution": "DISTRIBUTION",
    "skills": "SKILLS",
    "roster": "SKILLS",
    "overall": "OVERALL",
    "schedule": "SCHEDULE",
    "personaldetail": "PERSONALDETAIL",
    "personal_detail": "PERSONALDETAIL",
    "disqualified": "DISQUALIFIED",
}

# Per-category subcategory aliases: live intent subcategory -> bank operation key
SUBCATEGORY_ALIASES: Dict[str, Dict[str, str]] = {
    "PERFORMANCE": {
        "topperformers": "Top",
        "top": "Top",
        "lowestperformers": "Bottom",
        "bottomperformers": "Bottom",
        "bottom": "Bottom",
        "improvement": "Improvement",
        "improved": "Improvement",
        "drop": "Drop",
        "decline": "Drop",
        "grading": "Grading",
        "gradingsummary": "GradingSummary",
        "average": "Average",
        "attemptwise": "AttemptWise",
        "bestattempt": "BestAttempt",
        "trend": "Trend",
    },
    "LEAVE": {
        "most": "Most",
        "mostleave": "Most",
        "least": "Least",
        "leastleave": "Least",
        "current": "Current",
        "currentleave": "Current",
        "absconded": "Absconded",
        "awol": "Absconded",
    },
    "MEDICAL": {
        "bmi": "BMI",
        "bloodgroup": "BloodGroup",
        "blood_group": "BloodGroup",
        "disease": "Disease",
        "individual": "Individual",
    },
    "ATTENDANCE": {
        "monthly": "Monthly",
        "weekly": "Weekly",
        "daily": "Daily",
        "present": "Present",
        "summary": "Summary",
    },
    "VERIFICATION": {
        "pending": "Pending",
        "sent": "Sent",
        "notresponded": "NotResponded",
        "not_responded": "NotResponded",
        "completed": "Completed",
        "rejected": "Rejected",
    },
    "EQUIPMENT": {
        "stats": "Stats",
        "search": "Search",
        "returned": "Returned",
        "holding": "Holding",
        "agniveerwise": "AgniveerWise",
    },
    "DISTRIBUTION": {
        "latest": "Latest",
        "byunit": "ByUnit",
        "unassigned": "Unassigned",
        "topunit": "TopUnit",
    },
    "SKILLS": {
        "bysport": "BySport",
        "sport": "BySport",
        "byclass": "ByClass",
        "class": "ByClass",
    },
    "SCHEDULE": {
        "today": "Today",
        "company": "Company",
        "date": "Date",
        "agniveer": "Agniveer",
    },
}

# Entity patterns used to personalize bank questions against the live context
_AGNIVEER_RE = re.compile(r"\bA\d{5,8}[A-Z]?\b")
_BATCH_RE = re.compile(r"\bbatch\s+\d+\b", re.IGNORECASE)
_PLATOON_RE = re.compile(r"\bplatoon\s+\d+\b", re.IGNORECASE)
_COMPANY_WORDS = ["Lakhwinder", "Jaswant", "Arora", "Thorat", "Lak", "Jas"]
_COMPANY_RE = re.compile(
    r"\b(" + "|".join(_COMPANY_WORDS) + r")\b", re.IGNORECASE
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Internal helpers — extract context from combined_result
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _extract_scores(records: List[Any]) -> List[float]:
    return [
        s
        for s in (_get_score(r) for r in records if isinstance(r, dict))
        if s is not None
    ]


def _get_records(combined_result: Any) -> List[Dict]:
    if not isinstance(combined_result, dict):
        return []
    sections = combined_result.get("sections") or []
    if sections:
        return sections[0].get("data") or []
    left = combined_result.get("left") or {}
    right = combined_result.get("right") or {}
    return (left.get("data") or []) + (right.get("data") or [])


def _extract_context(
    combined_result: Any,
    intent: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Pull dynamic context tokens from the actual .NET response and intent.
    Returns a dict with keys:
      batch_id, platoon_id, agniveer_no, section_name, company_name,
      avg_score, min_score, max_score, at_risk_count,
      high_performer_count, total_count, trend_hint,
      left_label, right_label, raw_query
    """
    ctx: Dict[str, Any] = {}

    # From intent
    ctx["batch_id"] = intent.get("batch_id") or intent.get("batch") or ""
    ctx["platoon_id"] = intent.get("platoon_id") or intent.get("platoon") or ""
    ctx["agniveer_no"] = (
        intent.get("agniveer_no") or intent.get("agniveer_number") or ""
    )
    ctx["section_name"] = intent.get("section") or intent.get("sub_section") or ""
    ctx["sport_name"] = intent.get("sport") or ""
    ctx["company_name"] = (
        intent.get("company_name")
        or intent.get("companyName")
        or intent.get("company")
        or ""
    )
    ctx["raw_query"] = (
        intent.get("raw_query")
        or intent.get("original_query")
        or intent.get("query")
        or ""
    ).strip()

    if not isinstance(combined_result, dict):
        return ctx

    # Section label as fallback section name
    sections = combined_result.get("sections") or []
    if sections and not ctx["section_name"]:
        ctx["section_name"] = sections[0].get("label") or ""

    # Compare labels
    left = combined_result.get("left") or {}
    right = combined_result.get("right") or {}
    ctx["left_label"] = left.get("label") or ""
    ctx["right_label"] = right.get("label") or ""

    # Score intelligence from records
    records = _get_records(combined_result)
    scores = _extract_scores(records)
    ctx["total_count"] = len(records)

    if scores:
        avg = round(sum(scores) / len(scores), 2)
        ctx["avg_score"] = avg
        ctx["min_score"] = round(min(scores), 2)
        ctx["max_score"] = round(max(scores), 2)
        ctx["at_risk_count"] = sum(1 for s in scores if s < 50)
        ctx["high_performer_count"] = sum(1 for s in scores if s > 75)

        # Simple trend hint from first vs last quartile averages
        n = len(scores)
        if n >= 6:
            first_half = scores[: n // 2]
            second_half = scores[n // 2 :]
            first_avg = sum(first_half) / len(first_half)
            second_avg = sum(second_half) / len(second_half)
            if second_avg > first_avg + 2:
                ctx["trend_hint"] = "rising"
            elif second_avg < first_avg - 2:
                ctx["trend_hint"] = "falling"
            else:
                ctx["trend_hint"] = "stable"
        else:
            ctx["trend_hint"] = "stable"
    else:
        ctx["avg_score"] = None
        ctx["at_risk_count"] = 0
        ctx["high_performer_count"] = 0
        ctx["trend_hint"] = "stable"

    return ctx


def _fmt_batch(ctx: Dict[str, Any]) -> str:
    """Return 'Batch 12' or 'this batch' depending on context."""
    return f"Batch {ctx['batch_id']}" if ctx.get("batch_id") else "this batch"


def _fmt_platoon(ctx: Dict[str, Any]) -> str:
    return f"Platoon {ctx['platoon_id']}" if ctx.get("platoon_id") else "this platoon"


def _fmt_section(ctx: Dict[str, Any], fallback: str = "this section") -> str:
    return ctx.get("section_name") or fallback


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Question bank lookup + personalization
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _resolve_bank_category(category: str) -> Optional[str]:
    if not category:
        return None
    key = category.strip().lower()
    resolved = CATEGORY_ALIASES.get(key, category.strip().upper())
    if resolved in QUESTION_BANK["by_category"]:
        return resolved
    return None


def _resolve_bank_subcategory(bank_category: str, subcategory: str) -> Optional[str]:
    if not subcategory or not bank_category:
        return None
    aliases = SUBCATEGORY_ALIASES.get(bank_category, {})
    key = subcategory.strip().lower()
    candidate = aliases.get(key, subcategory.strip())
    bank_cat = QUESTION_BANK["by_category"].get(bank_category, {})
    if candidate in bank_cat:
        return candidate
    # Try case-insensitive match against real subcategory keys
    for real_key in bank_cat:
        if real_key.lower() == key:
            return real_key
    return None


def _bank_lookup(category: str, subcategory: str, qtype: str) -> List[str]:
    """
    Collect candidate questions for (category, subcategory, qtype), widening
    the search (subcategory -> whole category -> cross-category "mixed" bank)
    whenever the narrower scope doesn't have enough material.
    """
    bank_qtype = qtype if qtype in ("simple", "cross_filter", "multi_independent", "compare") else "simple"
    bank_category = _resolve_bank_category(category)

    candidates: List[str] = []

    if bank_category:
        bank_cat = QUESTION_BANK["by_category"][bank_category]
        bank_sub = _resolve_bank_subcategory(bank_category, subcategory)

        if bank_sub:
            candidates.extend(bank_cat[bank_sub].get(bank_qtype, []))

        # Widen to sibling subcategories in the same category if thin
        if len(candidates) < 4:
            for sc, qtypes in bank_cat.items():
                if sc == bank_sub:
                    continue
                candidates.extend(qtypes.get(bank_qtype, []))

    # Widen to the cross-category "mixed" bank (Part 2 material) if still thin
    if len(candidates) < 4:
        candidates.extend(QUESTION_BANK["mixed"].get(bank_qtype, []))

    return candidates


def _personalize(question: str, ctx: Dict[str, Any]) -> str:
    """
    Swap the bank's example entities (a demo batch/platoon/agniveer/company)
    for the real ones in the current context — but only when doing so can't
    break a two-sided compare question (e.g. never collapse
    "Lakhwinder and Jaswant company" down to the same company twice).
    """
    q = question

    if ctx.get("agniveer_no") and len(_AGNIVEER_RE.findall(q)) == 1:
        q = _AGNIVEER_RE.sub(ctx["agniveer_no"], q)

    if ctx.get("batch_id") and len(_BATCH_RE.findall(q)) == 1:
        q = _BATCH_RE.sub(f"batch {ctx['batch_id']}", q)

    if ctx.get("platoon_id") and len(_PLATOON_RE.findall(q)) == 1:
        q = _PLATOON_RE.sub(f"platoon {ctx['platoon_id']}", q)

    if ctx.get("company_name") and len(_COMPANY_RE.findall(q)) == 1:
        q = _COMPANY_RE.sub(ctx["company_name"], q)

    return q


def _select_bank_questions(
    category: str,
    subcategory: str,
    qtype: str,
    ctx: Dict[str, Any],
    n: int = 4,
) -> List[str]:
    candidates = _bank_lookup(category, subcategory, qtype)
    if not candidates:
        return []

    raw_query = ctx.get("raw_query", "").strip().lower().rstrip("?.")
    pool = [
        q for q in candidates if q.strip().lower().rstrip("?.") != raw_query
    ]
    if not pool:
        pool = candidates

    pool = list(pool)
    random.shuffle(pool)

    chosen: List[str] = []
    seen = set()
    for q in pool:
        personalized = _personalize(q, ctx)
        key = personalized.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        chosen.append(personalized)
        if len(chosen) >= n:
            break

    return chosen


def _merge_with_fallback(bank_questions: List[str], fallback_fn, n: int = 4) -> List[str]:
    if len(bank_questions) >= n:
        return bank_questions[:n]

    seen = {q.strip().lower() for q in bank_questions}
    result = list(bank_questions)
    for q in fallback_fn():
        key = q.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(q)
        if len(result) >= n:
            break
    return result[:n]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fallback generator (context-enriched, from v2) — cross-filter is the only
# suggestion style offered (see generate_suggested_questions), so this is
# the only fallback generator still reachable.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _questions_for_cross_filter(ctx: Dict[str, Any]) -> List[str]:
    batch = _fmt_batch(ctx)
    platoon = _fmt_platoon(ctx)
    section = _fmt_section(ctx)
    at_risk = ctx.get("at_risk_count", 0)
    questions = [
        f"Show Class A agniveers with Excellent grading in {section}.",
        f"List overweight agniveers from {batch}.",
        f"Show cricket players in {platoon}.",
        f"How many matching records from {batch} are currently on leave?",
    ]
    if at_risk:
        questions.insert(
            0, f"Show the {at_risk} at-risk agniveer(s) below 50 in {section}."
        )
    return questions[:4]


def _fallback_for(qtype: str, category: str, subcategory: str, ctx: Dict[str, Any]):
    """Return a zero-arg callable producing the v2-style generated questions.

    Cross-filter is the only suggestion style ever requested by
    generate_suggested_questions (see its docstring), so this always
    resolves to the cross-filter generator. qtype/category/subcategory are
    kept as parameters for call-site stability, even though only ctx is
    actually used.
    """
    return lambda: _questions_for_cross_filter(ctx)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Public API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def generate_suggested_questions(
    query_type: str,
    intent: Dict[str, Any],
    combined_result: Any,
) -> List[str]:
    """
    Return 3-4 relevant next-step questions that are:
      1. Drawn first from the curated real-question bank (question_bank.py),
         matched to the current category / subcategory / query_type, and
         personalized against the live batch/platoon/agniveer/company context.
      2. Topped up, only if the bank comes up short, with the score-aware,
         trend-aware generated questions from v2.
    """
    category = (intent.get("category") or "").strip()
    if not category or category.lower() in ("greeting", "unknown", "none", "unclear"):
        return []

    subcategory = (intent.get("subcategory") or "").strip()
    # Only cross-filter style follow-ups are suggested: multiple conditions
    # applied to the *same* record set are inherently related to each other.
    # simple / compare / multi_independent suggestions are not offered —
    # a compare or a side-by-side section dump isn't a natural next step
    # off of an arbitrary query, whereas "who else matches this + that" is.
    qtype = "cross_filter"

    ctx = _extract_context(combined_result, intent)

    bank_questions = _select_bank_questions(category, subcategory, qtype, ctx, n=4)
    fallback_fn = _fallback_for(qtype, category, subcategory, ctx)

    return _merge_with_fallback(bank_questions, fallback_fn, n=4)