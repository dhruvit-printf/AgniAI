"""
conclusion_engine.py
====================
Generates a concise conclusion with at most three bullet points.
Includes:
  - Trend detection (rising / falling / stable / volatile)
  - Future prediction bullets (at-risk, high-performer ceiling, pass-fail forecast)
  - Analytics query type support
  - Distribution-aware summaries (skew, outliers, clustering)
  - Comparative delta intelligence
Pure Python — no LLM dependency.
"""

import logging
import statistics as _stats_mod
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from utils import get_score as _get_score


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Statistical helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _extract_scores(records: List[Any]) -> List[float]:
    return [s for s in (_get_score(r) for r in records if isinstance(r, dict)) if s is not None]


# ── Named record helpers (who to call out in the conclusion) ──────────────────

_NAME_FIELDS = ("fullName", "name", "agniveerName", "recruiterName")
_ID_FIELDS = ("agniveerNo", "agniveerNumber", "enrollmentNo")


def _record_label(record: Dict[str, Any]) -> Optional[str]:
    for f in _NAME_FIELDS:
        if record.get(f):
            return str(record[f])
    for f in _ID_FIELDS:
        if record.get(f):
            return str(record[f])
    return None


def _named_score_records(records: List[Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in records:
        if not isinstance(r, dict):
            continue
        label = _record_label(r)
        score = _get_score(r)
        if label and score is not None:
            out.append({"label": label, "score": score})
    return out


def _percentile(sorted_data: List[float], pct: float) -> float:
    if not sorted_data:
        return 0.0
    n = len(sorted_data)
    idx = (pct / 100) * (n - 1)
    lo, hi = int(idx), min(int(idx) + 1, n - 1)
    return round(sorted_data[lo] + (idx - lo) * (sorted_data[hi] - sorted_data[lo]), 2)


def _std_dev(scores: List[float]) -> float:
    if len(scores) < 2:
        return 0.0
    return round(_stats_mod.stdev(scores), 2)


def _skewness(scores: List[float]) -> float:
    """
    Pearson's moment coefficient of skewness.
    Positive → tail on high side (right skew), negative → tail on low side.
    """
    n = len(scores)
    if n < 3:
        return 0.0
    mean = sum(scores) / n
    std = _std_dev(scores)
    if std == 0:
        return 0.0
    return round(sum(((x - mean) / std) ** 3 for x in scores) / n, 3)


def _detect_trend(scores: List[float]) -> Tuple[str, float]:
    """
    Fit a simple linear regression over score index to determine direction.
    Returns (label, slope_per_record).
    Labels: 'rising', 'falling', 'stable', 'volatile'
    """
    n = len(scores)
    if n < 3:
        return ("insufficient_data", 0.0)

    std = _std_dev(scores)
    if std > 20:
        return ("volatile", 0.0)

    xs = list(range(n))
    x_mean = (n - 1) / 2
    y_mean = sum(scores) / n
    numerator = sum((xs[i] - x_mean) * (scores[i] - y_mean) for i in range(n))
    denominator = sum((x - x_mean) ** 2 for x in xs)
    slope = round(numerator / denominator, 4) if denominator else 0.0

    if abs(slope) < 0.05:
        label = "stable"
    elif slope > 0:
        label = "rising"
    else:
        label = "falling"
    return (label, slope)


def _predict_future(
    scores: List[float],
    trend_label: str,
    slope: float,
    n_steps: int = 5,
) -> Optional[str]:
    """
    Project score trajectory forward by n_steps records.
    Returns a prediction sentence or None if data is too thin.
    """
    n = len(scores)
    if n < 3 or trend_label in ("insufficient_data", "volatile"):
        return None

    current_avg = round(sum(scores) / n, 2)
    projected = round(current_avg + slope * n_steps, 2)
    projected = max(0.0, min(100.0, projected))

    at_risk = sum(1 for s in scores if s < 50)
    near_boundary = sum(1 for s in scores if 48 <= s <= 52)

    parts = []

    if trend_label == "rising":
        parts.append(
            f"Trend is upward (slope +{round(slope, 2)}/record); "
            f"projected avg in {n_steps} records: {projected}."
        )
    elif trend_label == "falling":
        parts.append(
            f"Trend is declining (slope {round(slope, 2)}/record); "
            f"projected avg in {n_steps} records: {projected}."
        )
        if projected < 50:
            parts.append("Intervention recommended before scores fall below passing threshold.")
    else:
        parts.append(f"Scores are stable; projected avg remains near {projected}.")

    if at_risk:
        parts.append(
            f"{at_risk} agniveer(s) currently below 50 — immediate review advised."
        )
    if near_boundary:
        parts.append(
            f"{near_boundary} agniveer(s) within 2 points of the 50-mark boundary."
        )

    return " ".join(parts) if parts else None


def _distribution_bullet(scores: List[float]) -> Optional[str]:
    """
    Returns a distribution-awareness bullet (skew, clustering, outlier count).
    """
    if len(scores) < 4:
        return None
    skew = _skewness(scores)
    std = _std_dev(scores)
    avg = sum(scores) / len(scores)
    outlier_threshold = 2 * std
    outliers = [s for s in scores if abs(s - avg) > outlier_threshold]

    parts = []
    if skew > 0.5:
        parts.append(f"Distribution is right-skewed ({skew}) — most scores cluster below mean.")
    elif skew < -0.5:
        parts.append(f"Distribution is left-skewed ({skew}) — most scores cluster above mean.")
    else:
        parts.append(f"Score distribution is approximately symmetric (skew: {skew}).")
    if std:
        parts.append(f"Std deviation: {std}.")
    if outliers:
        parts.append(f"{len(outliers)} outlier(s) detected beyond ±2σ.")

    return " ".join(parts) if parts else None


def _score_summary_bullet(scores: List[float], label: str = "") -> Optional[str]:
    if not scores:
        return None
    sorted_scores = sorted(scores)
    avg = round(sum(scores) / len(scores), 2)
    median = round(_stats_mod.median(scores), 2)
    p75 = _percentile(sorted_scores, 75)
    high_cnt = sum(1 for s in scores if s > 75)
    prefix = f"{label} — " if label else ""
    return (
        f"{prefix}avg: {avg}, median: {median}, top-quartile threshold: {p75}"
        + (f", {high_cnt} high performer(s) (>75)" if high_cnt else "")
        + "."
    )


def _build_conclusion_grounding_text(answer: Dict[str, Any], query_type: str) -> str:
    lines = []
    sections = answer.get("sections") or []

    if query_type == "compare":
        left = answer.get("left") or {}
        right = answer.get("right") or {}
        comp = answer.get("comparison") or {}
        if left:
            lines.append(f"{left.get('label')} count is {len(left.get('data', []))}.")
        if right:
            lines.append(f"{right.get('label')} count is {len(right.get('data', []))}.")
        for k, v in comp.items():
            if isinstance(v, dict):
                lines.append(
                    f"{k} difference is {v.get('difference')} and percentage is {v.get('percentage')}."
                )
    else:
        records = sections[0].get("data") if sections else []
        lines.append(f"Count: {len(records)}")
        scores = _extract_scores(records)
        if scores:
            lines.append(f"Average Score: {round(sum(scores) / len(scores), 2)}")
            lines.append(f"Top Score: {max(scores)}")
            lines.append(f"Bottom Score: {min(scores)}")

    return "\n".join(lines)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Analytics helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _analytics_conclusion(
    records: List[Any],
    intent: Dict[str, Any],
    category: str,
) -> Dict[str, Any]:
    """
    Deep analytics conclusion: trend + distribution + prediction bullets.
    Used for query_type == 'analytics'.
    """
    scores = sorted(_extract_scores(records))
    cnt = len(records)
    summary = f"Analytics review of {cnt} {category.lower()} record(s) complete."
    bullets: List[str] = []

    if not scores:
        bullets.append(f"Found {cnt} record(s); no numeric scores available for analysis.")
        return {"summary": summary, "bullets": bullets[:3]}

    avg = round(sum(scores) / len(scores), 2)
    p25 = _percentile(scores, 25)
    p75 = _percentile(scores, 75)
    high = sum(1 for s in scores if s > 75)
    low = sum(1 for s in scores if s < 50)

    # Bullet 1: core stats
    bullets.append(
        f"{cnt} record(s) — avg: {avg}, range: {round(min(scores), 2)}–{round(max(scores), 2)}, "
        f"IQR: {p25}–{p75}. {high} high performer(s) (>75), {low} below passing threshold (<50)."
    )

    # Bullet 2: trend + prediction
    trend_label, slope = _detect_trend(scores)
    pred = _predict_future(scores, trend_label, slope)
    if pred:
        bullets.append(pred)

    # Bullet 3: distribution
    dist = _distribution_bullet(scores)
    if dist:
        bullets.append(dist)

    return {"summary": summary, "bullets": bullets[:3]}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Public API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_conclusion(
    combined_result: Any,
    query_type: str,
    intent: Dict[str, Any],
    trace_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate a short conclusion with at most three data-grounded bullet points.

    Supports query types:
      simple / trend / distribution  →  count + score stats + trend prediction
      compare / comparison            →  delta intelligence + high-performer delta
      cross_filter                    →  filter satisfaction count + score summary
      multi_independent               →  section consolidation
      analytics                       →  full trend + distribution + prediction

    Pure Python — no LLM calls. Never raises.
    """
    try:
        from normalized_models import extract_records as _extract_records

        records = _extract_records(combined_result)

        if isinstance(combined_result, dict):
            sections = combined_result.get("sections") or []
            left = combined_result.get("left") or {}
            right = combined_result.get("right") or {}
            comp = combined_result.get("comparison") or {}
        else:
            sections = []
            left = {}
            right = {}
            comp = {}

        if not sections and not (left or right):
            sections = [{"label": "Result", "data": records}]

        is_empty = True
        if query_type in ("compare", "comparison"):
            left_data = left.get("data") or []
            right_data = right.get("data") or []
            if left_data or right_data or left.get("metrics", {}).get("recordCount", 0) > 0 or right.get("metrics", {}).get("recordCount", 0) > 0:
                is_empty = False
        else:
            for sec in sections:
                if sec.get("data"):
                    is_empty = False
                    break

        category = intent.get("category") or "Agniveer"

        if is_empty:
            return {"summary": "No matching records found.", "bullets": []}

        bullets: List[str] = []
        qtype = (query_type or "").strip().lower()

        # ── Analytics ──────────────────────────────────────────────────────────
        if qtype == "analytics":
            return _analytics_conclusion(records, intent, category)

        # ── Compare ────────────────────────────────────────────────────────────
        elif qtype in ("compare", "comparison"):
            left_label = left.get("label", "Side 1")
            right_label = right.get("label", "Side 2")
            left_data = left.get("data") or []
            right_data = right.get("data") or []
            left_scores = _extract_scores(left_data)
            right_scores = _extract_scores(right_data)
            left_avg = round(sum(left_scores) / len(left_scores), 2) if left_scores else None
            right_avg = round(sum(right_scores) / len(right_scores), 2) if right_scores else None

            left_metrics = left.get("metrics") or {}
            right_metrics = right.get("metrics") or {}
            left_cnt = left_metrics.get("recordCount")
            if left_cnt is None:
                left_cnt = len(left_data)
            else:
                left_cnt = int(left_cnt)
            right_cnt = right_metrics.get("recordCount")
            if right_cnt is None:
                right_cnt = len(right_data)
            else:
                right_cnt = int(right_cnt)

            summary = f"Comparative review of {left_label} and {right_label} is complete."

            # Bullet 1: record counts
            bullets.append(
                f"Evaluated {left_cnt} {left_label} record(s) and "
                f"{right_cnt} {right_label} record(s)."
            )

            # Bullet 2: avg delta + leader + trend signal
            if left_avg is not None and right_avg is not None:
                diff = round(abs(left_avg - right_avg), 2)
                leader = left_label if left_avg > right_avg else right_label
                laggard = right_label if leader == left_label else left_label
                pct_gap = round((diff / max(left_avg, right_avg)) * 100, 1) if max(left_avg, right_avg) else 0
                bullets.append(
                    f"Avg scores — {left_label}: {left_avg}, {right_label}: {right_avg}. "
                    f"{leader} leads by {diff} pts ({pct_gap}%). "
                    f"{laggard} may need targeted improvement."
                )

            # Bullet 3: high-performer delta + prediction note
            left_high = sum(1 for s in left_scores if s > 75)
            right_high = sum(1 for s in right_scores if s > 75)
            left_low = sum(1 for s in left_scores if s < 50)
            right_low = sum(1 for s in right_scores if s < 50)
            if left_high or right_high or left_low or right_low:
                bullets.append(
                    f"High performers (>75): {left_label} {left_high} vs {right_label} {right_high}. "
                    f"Below passing (<50): {left_label} {left_low} vs {right_label} {right_low}."
                )

        # ── Cross-filter ───────────────────────────────────────────────────────
        elif qtype == "cross_filter":
            cnt = len(records)
            scores = sorted(_extract_scores(records))
            summary = "Cross-filter analysis successfully completed."

            bullets.append(
                f"Found {cnt} record(s) satisfying all combined filter conditions."
            )

            score_bullet = _score_summary_bullet(scores)
            if score_bullet:
                bullets.append(score_bullet)

            # Trend within filtered subset
            if len(scores) >= 3:
                trend_label, slope = _detect_trend(scores)
                if trend_label not in ("stable", "insufficient_data"):
                    pred = _predict_future(scores, trend_label, slope, n_steps=3)
                    if pred:
                        bullets.append(pred)

        # ── Multi-independent ──────────────────────────────────────────────────
        elif qtype == "multi_independent":
            summary = "Consolidation of multi-section dataset is complete."
            bullets.append(
                f"Consolidated data covers {len(sections)} independent section(s)."
            )
            for sec in sections[:2]:
                label = sec.get("label", "Section")
                sec_records = sec.get("data", [])
                cnt = len(sec_records)
                sec_scores = _extract_scores(sec_records)
                if sec_scores:
                    sec_avg = round(sum(sec_scores) / len(sec_scores), 2)
                    trend_label, _ = _detect_trend(sorted(sec_scores))
                    trend_note = (
                        f", trend: {trend_label}"
                        if trend_label not in ("insufficient_data",)
                        else ""
                    )
                    bullets.append(
                        f"{label}: {cnt} record(s), avg score {sec_avg}{trend_note}."
                    )
                else:
                    bullets.append(f"{label}: {cnt} record(s).")

        # ── Simple / trend / distribution ──────────────────────────────────────
        else:
            cnt = len(records)
            scores = sorted(_extract_scores(records))
            summary = f"Query lookup for {category.lower()} records is complete."

            bullets.append(f"Found {cnt} matching {category.lower()} record(s).")

            if scores:
                avg = round(sum(scores) / len(scores), 2)
                p75 = _percentile(scores, 75)
                high_cnt = sum(1 for s in scores if s > 75)
                above_avg = sum(1 for s in scores if s > avg)
                pct_above = round((above_avg / len(scores)) * 100, 1)

                bullets.append(
                    f"Avg score: {avg} (range: {round(min(scores), 2)}–{round(max(scores), 2)}, "
                    f"P75: {p75}). {pct_above}% scored above average."
                )

                named = _named_score_records(records)
                if named:
                    best = max(named, key=lambda n: n["score"])
                    worst = min(named, key=lambda n: n["score"])
                    if best["label"] != worst["label"] or best["score"] != worst["score"]:
                        bullets.append(
                            f"Top performer: {best['label']} ({best['score']}). "
                            f"Lowest recorded: {worst['label']} ({worst['score']})"
                            + (", flagged for review." if worst["score"] < 50 else ".")
                        )

                # Trend + prediction
                trend_label, slope = _detect_trend(scores)
                pred = _predict_future(scores, trend_label, slope)
                if pred:
                    bullets.append(pred)
                elif high_cnt:
                    bullets.append(
                        f"{high_cnt} high performer(s) scored above 75 out of {cnt} total."
                    )

                # Closing verdict — ties the numbers to a clear next step
                below_passing = sum(1 for s in scores if s < 50)
                if below_passing:
                    bullets.append(
                        f"Overall verdict: {below_passing} record(s) are below the passing mark "
                        f"and need corrective action — see the prediction guidance below for a "
                        f"recovery plan."
                    )
                else:
                    bullets.append(
                        "Overall verdict: the set is performing at or above standard, with no "
                        "records currently flagged for concern."
                    )

        # Cap at 5 bullets to allow named highlights plus a closing verdict
        bullets = bullets[:5]
        return {"summary": summary, "bullets": bullets}

    except Exception as exc:
        logger.error(
            "conclusion_engine.generate_conclusion failed: %s", exc, exc_info=True
        )
        category = intent.get("category") or "Agniveer"
        fallback = f"The review of {category.lower()} records is complete."
        return {"summary": fallback, "bullets": [fallback]}