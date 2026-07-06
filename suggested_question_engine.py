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

QTYPE_NORMALIZE: Dict[str, str] = {
    "compare": "compare",
    "comparison": "compare",
    "cross_filter": "cross_filter",
    "crossfilter": "cross_filter",
    "multi_independent": "multi_independent",
    "multiindependent": "multi_independent",
    "analytics": "analytics",
    "simple": "simple",
    "filter": "simple",
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
# Query-type level fallback generators (context-enriched, from v2)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _questions_for_compare(ctx: Dict[str, Any]) -> List[str]:
    left = ctx.get("left_label") or "Side A"
    right = ctx.get("right_label") or "Side B"
    batch = _fmt_batch(ctx)
    platoon = _fmt_platoon(ctx)
    questions = [
        f"Compare Drill and Firing performance for {batch}.",
        f"Compare BEPT and PPT scores between {left} and {right}.",
        f"Show score distribution comparison for {platoon}.",
        f"Compare {left} and {right} with the overall {batch} average.",
    ]
    if ctx.get("avg_score") is not None:
        questions.append(f"Which agniveers in {left} are below the {right} average?")
    return questions[:4]


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


def _questions_for_multi_independent(ctx: Dict[str, Any]) -> List[str]:
    batch = _fmt_batch(ctx)
    return [
        f"Show the top 5 performers and the top 5 leave takers in {batch}.",
        "Show active medical cases and current attendance summary.",
        f"List overdue equipment and pending police verifications for {batch}.",
        "Show monthly attendance statistics and blood group distribution.",
    ]


def _questions_for_analytics(ctx: Dict[str, Any], category: str) -> List[str]:
    batch = _fmt_batch(ctx)
    section = _fmt_section(ctx)
    trend = ctx.get("trend_hint", "stable")
    at_risk = ctx.get("at_risk_count", 0)
    high = ctx.get("high_performer_count", 0)

    questions = [
        f"Show trend analysis for {category} in {batch}.",
        f"Predict future performance for {section} based on current scores.",
        f"Show score distribution for {category} across all batches.",
    ]
    if trend == "falling":
        questions.insert(
            0, f"Which agniveers in {batch} show a declining {category} trend?"
        )
    if at_risk:
        questions.insert(
            0,
            f"Show the {at_risk} at-risk agniveer(s) in {batch} below passing threshold.",
        )
    if high:
        questions.append(f"Show the top {high} high-performer(s) (>75) in {section}.")
    return questions[:4]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Category-level fallback generators (context-enriched, from v2)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _questions_performance(ctx: Dict[str, Any], subcategory: str) -> List[str]:
    section = _fmt_section(ctx)
    batch = _fmt_batch(ctx)
    platoon = _fmt_platoon(ctx)
    avg = ctx.get("avg_score")
    at_risk = ctx.get("at_risk_count", 0)
    high = ctx.get("high_performer_count", 0)
    trend = ctx.get("trend_hint", "stable")

    base: List[str] = []

    if subcategory == "TopPerformers":
        base = [
            f"Who are the top 10 performers overall in {batch}?",
            f"Show the lowest scoring agniveers in {platoon}.",
            f"Who are the lowest performers in {section}?",
            f"Compare {section}'s top performers with another section.",
        ]
        if trend == "falling":
            base.insert(1, f"Why is the top performer average declining in {batch}?")

    elif subcategory == "LowestPerformers":
        base = [
            f"Who scored highest in BEPT in {batch}?",
            f"Show the top 5 performers in Firing for {platoon}.",
            f"Who are the top performers in {section}?",
            f"Compare {section}'s lowest performers with another section.",
        ]
        if at_risk:
            base.insert(
                0,
                f"Show {at_risk} below-passing agniveer(s) in {batch} who need intervention.",
            )

    else:
        base = [
            f"Who are the top 10 performers overall in {batch}?",
            f"Show the lowest scoring agniveers in {platoon}.",
            f"Who are the lowest performers in {section}?",
            f"Compare {section} with another section.",
        ]
        if avg is not None:
            base.append(f"Which agniveers are above the {batch} average of {avg}?")
        if high:
            base.append(f"Show all {high} high-performer(s) above 75 in {section}.")

    return base[:4]


def _questions_leave(ctx: Dict[str, Any]) -> List[str]:
    batch = _fmt_batch(ctx)
    section = _fmt_section(ctx)
    return [
        f"Who has taken the most leave in {batch}?",
        f"Which agniveers are currently on leave in {section}?",
        f"Who took the most leaves in {section} this month?",
        "Show all absconded leave records.",
        f"Show leave trend for {batch} over the past 3 months.",
    ][:4]


def _questions_medical(ctx: Dict[str, Any]) -> List[str]:
    batch = _fmt_batch(ctx)
    section = _fmt_section(ctx)
    return [
        f"Show active medical cases in {batch}.",
        "Which agniveers are obese?",
        f"Show BMI outliers and fitness analysis for {section}.",
        "Give blood group distribution.",
        f"Show medical trend for {batch} over the past month.",
    ][:4]


def _questions_attendance(ctx: Dict[str, Any]) -> List[str]:
    batch = _fmt_batch(ctx)
    section = _fmt_section(ctx)
    return [
        f"Show monthly attendance statistics for {batch}.",
        "Give weekly attendance report.",
        f"Show monthly attendance statistics for {section}.",
        "Show today's attendance.",
        f"Which agniveers in {batch} have attendance below 75%?",
    ][:4]


def _questions_equipment(ctx: Dict[str, Any]) -> List[str]:
    batch = _fmt_batch(ctx)
    return [
        f"Show overall equipment statistics for {batch}.",
        "Which equipment is overdue for return?",
        "Show returned equipment.",
        "Show overdue equipment returns.",
    ]


def _questions_verification(ctx: Dict[str, Any]) -> List[str]:
    batch = _fmt_batch(ctx)
    return [
        f"Which agniveers in {batch} are pending verification?",
        "Show sent verification records.",
        "Show verified agniveers.",
        f"Show pending verification list for {batch}.",
    ]


def _questions_skills_roster(ctx: Dict[str, Any]) -> List[str]:
    sport = ctx.get("sport_name") or "the sport"
    platoon = _fmt_platoon(ctx)
    batch = _fmt_batch(ctx)
    return [
        f"Show all sports players in {batch}.",
        "List cricket players.",
        f"Show roster by sport for {sport} in {platoon}.",
        "List all Class A agniveers.",
    ]


def _questions_strength(ctx: Dict[str, Any]) -> List[str]:
    batch = _fmt_batch(ctx)
    platoon = _fmt_platoon(ctx)
    return [
        f"Show complete strength breakdown for {batch}.",
        f"Give company-wise strength report for {batch}.",
        f"Show platoon-wise strength for {platoon}.",
        "Show present and absent counts.",
    ]


def _questions_overall(ctx: Dict[str, Any]) -> List[str]:
    batch = _fmt_batch(ctx)
    avg = ctx.get("avg_score")
    trend = ctx.get("trend_hint", "stable")
    questions = [
        f"Show top overall performers in {batch}.",
        "Rank agniveers using composite score.",
        f"Which batch has the best overall performance compared to {batch}?",
        f"Show top 20 overall agniveers in {batch}.",
    ]
    if avg is not None and trend == "falling":
        questions.insert(0, f"Why is the overall average ({avg}) declining in {batch}?")
    return questions[:4]


def _generic_fallback(ctx: Dict[str, Any], category: str) -> List[str]:
    batch = _fmt_batch(ctx)
    section = _fmt_section(ctx, fallback=category)
    avg = ctx.get("avg_score")

    fallback = [
        f"Show average score for {category} in {batch}.",
        f"List all records in {section}.",
        f"Show section-wise distribution for {category}.",
    ]
    if avg is not None:
        fallback.append(
            f"Which agniveers in {batch} are above the {category} average of {avg}?"
        )
    if ctx.get("at_risk_count"):
        fallback.insert(
            0,
            f"Show {ctx['at_risk_count']} at-risk agniveer(s) below 50 in {section}.",
        )
    return fallback[:4]


def _fallback_for(qtype: str, category: str, subcategory: str, ctx: Dict[str, Any]):
    """Return a zero-arg callable producing the v2-style generated questions."""
    if qtype in ("compare",):
        return lambda: _questions_for_compare(ctx)
    if qtype == "cross_filter":
        return lambda: _questions_for_cross_filter(ctx)
    if qtype == "multi_independent":
        return lambda: _questions_for_multi_independent(ctx)
    if qtype == "analytics":
        return lambda: _questions_for_analytics(ctx, category)

    dispatch = {
        "PERFORMANCE": lambda: _questions_performance(ctx, subcategory),
        "LEAVE": lambda: _questions_leave(ctx),
        "MEDICAL": lambda: _questions_medical(ctx),
        "ATTENDANCE": lambda: _questions_attendance(ctx),
        "EQUIPMENT": lambda: _questions_equipment(ctx),
        "VERIFICATION": lambda: _questions_verification(ctx),
        "SKILLS": lambda: _questions_skills_roster(ctx),
        "STRENGTH": lambda: _questions_strength(ctx),
        "OVERALL": lambda: _questions_overall(ctx),
    }
    bank_category = _resolve_bank_category(category) or (category or "").upper()
    return dispatch.get(bank_category, lambda: _generic_fallback(ctx, category))


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
    qtype = QTYPE_NORMALIZE.get((query_type or "").strip().lower(), "simple")

    ctx = _extract_context(combined_result, intent)

    bank_questions = _select_bank_questions(category, subcategory, qtype, ctx, n=4)
    fallback_fn = _fallback_for(qtype, category, subcategory, ctx)

    return _merge_with_fallback(bank_questions, fallback_fn, n=4)