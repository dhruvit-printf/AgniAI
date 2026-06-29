"""
suggested_question_engine.py
============================
Generates dynamic, data-aware, context-prior next-step questions based on
query intent AND actual result data from the .NET response.

Key upgrades vs v1:
  - Questions are PRIOR to the result — they reference actual batch IDs,
    platoon IDs, score ranges, and section names extracted from combined_result.
  - Score-shape awareness: if scores are low → suggest improvement drill-down;
    if high → suggest ceiling-push queries.
  - Trend-aware: if trend is falling → suggest intervention queries.
  - At-risk detection: generates targeted queries for below-50 agniveers.
  - Category + subcategory + query_type matrix fully covered.
  - Fallback is always non-empty and contextual.
"""

from typing import Any, Dict, List, Optional
from utils import get_score as _get_score


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Internal helpers — extract context from combined_result
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _extract_scores(records: List[Any]) -> List[float]:
    return [s for s in (_get_score(r) for r in records if isinstance(r, dict)) if s is not None]


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
      batch_id, platoon_id, agniveer_no, section_name,
      avg_score, min_score, max_score, at_risk_count,
      high_performer_count, total_count, trend_hint,
      left_label, right_label
    """
    ctx: Dict[str, Any] = {}

    # From intent
    ctx["batch_id"] = intent.get("batch_id") or intent.get("batch") or ""
    ctx["platoon_id"] = intent.get("platoon_id") or intent.get("platoon") or ""
    ctx["agniveer_no"] = intent.get("agniveer_no") or intent.get("agniveer_number") or ""
    ctx["section_name"] = (
        intent.get("section") or intent.get("sub_section") or ""
    )
    ctx["sport_name"] = intent.get("sport") or ""

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
            first_half = scores[:n // 2]
            second_half = scores[n // 2:]
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
# Query-type level overrides (context-enriched)
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
    # If one side's avg was lower, surface that
    if ctx.get("avg_score") is not None:
        questions.append(
            f"Which agniveers in {left} are below the {right} average?"
        )
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
        questions.insert(0, f"Show the {at_risk} at-risk agniveer(s) below 50 in {section}.")
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
    avg = ctx.get("avg_score")
    trend = ctx.get("trend_hint", "stable")
    at_risk = ctx.get("at_risk_count", 0)
    high = ctx.get("high_performer_count", 0)

    questions = [
        f"Show trend analysis for {category} in {batch}.",
        f"Predict future performance for {section} based on current scores.",
        f"Show score distribution for {category} across all batches.",
    ]
    if trend == "falling":
        questions.insert(0, f"Which agniveers in {batch} show a declining {category} trend?")
    if at_risk:
        questions.insert(0, f"Show the {at_risk} at-risk agniveer(s) in {batch} below passing threshold.")
    if high:
        questions.append(f"Show the top {high} high-performer(s) (>75) in {section}.")
    return questions[:4]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Category-level question banks (context-enriched)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _questions_performance(
    ctx: Dict[str, Any],
    subcategory: str,
) -> List[str]:
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
            base.insert(0, f"Show {at_risk} below-passing agniveer(s) in {batch} who need intervention.")

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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Public API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_suggested_questions(
    query_type: str,
    intent: Dict[str, Any],
    combined_result: Any,
) -> List[str]:
    """
    Return 3–5 relevant next-step questions that are:
      1. Prior to the data already shown (reference actual batch/platoon/scores).
      2. Score-aware (at-risk → intervention; high performers → drill deeper).
      3. Trend-aware (falling → suggest investigation queries).
      4. Query-type-aware (compare/cross_filter/analytics get their own banks).
    """
    category = (intent.get("category") or "").strip()
    if not category or category.lower() in ("greeting", "unknown", "none", "unclear"):
        return []

    subcategory = (intent.get("subcategory") or "").strip()
    qtype = (query_type or "").strip().lower()

    # Build rich context from combined_result + intent
    ctx = _extract_context(combined_result, intent)

    # ── Query-type matrix ──────────────────────────────────────────────────────
    if qtype in ("compare", "comparison"):
        return _questions_for_compare(ctx)

    if qtype == "cross_filter":
        return _questions_for_cross_filter(ctx)

    if qtype == "multi_independent":
        return _questions_for_multi_independent(ctx)

    if qtype == "analytics":
        return _questions_for_analytics(ctx, category)

    # ── Category matrix ────────────────────────────────────────────────────────
    if category == "Performance":
        return _questions_performance(ctx, subcategory)

    if category == "Leave":
        return _questions_leave(ctx)

    if category == "Medical":
        return _questions_medical(ctx)

    if category == "Attendance":
        return _questions_attendance(ctx)

    if category == "Equipment":
        return _questions_equipment(ctx)

    if category == "Verification":
        return _questions_verification(ctx)

    if category in ("Skills", "Roster"):
        return _questions_skills_roster(ctx)

    if category == "Strength":
        return _questions_strength(ctx)

    if category == "Overall":
        return _questions_overall(ctx)

    # ── Generic fallback (always contextual, never empty) ──────────────────────
    batch = _fmt_batch(ctx)
    section = _fmt_section(ctx, fallback=category)
    avg = ctx.get("avg_score")

    fallback = [
        f"Show average score for {category} in {batch}.",
        f"List all records in {section}.",
        f"Show section-wise distribution for {category}.",
    ]
    if avg is not None:
        fallback.append(f"Which agniveers in {batch} are above the {category} average of {avg}?")
    if ctx.get("at_risk_count"):
        fallback.insert(
            0,
            f"Show {ctx['at_risk_count']} at-risk agniveer(s) below 50 in {section}.",
        )

    return fallback[:4]