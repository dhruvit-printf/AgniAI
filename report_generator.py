"""
report_generator.py
===================

"""

from __future__ import annotations

import json
import logging
import re
import threading
from typing import Any, Dict, List, Optional, Tuple

import requests

from admin_formatter import format_dotnet_response
from config import DEFAULT_MODEL, OLLAMA_URL

logger = logging.getLogger(__name__)

_report_ctx = threading.local()

# =============================================================================
# INTRO TEMPLATES
# =============================================================================

_INTRO_TEMPLATES: Dict[Tuple[str, str], str] = {
    (
        "Performance",
        "TopPerformers",
    ): "These assessment results highlight the strongest performers in the evaluation.",
    (
        "Performance",
        "LowestPerformers",
    ): "These results identify the individuals requiring additional training support.",
    (
        "Performance",
        "AverageScore",
    ): "The average scores outline overall achievement levels across the group.",
    (
        "Performance",
        "PassPercentage",
    ): "Pass rates reflect the percentage of trainees meeting the assessment standards.",
    (
        "Performance",
        "FailPercentage",
    ): "Fail rates identify the proportion of trainees currently below standard.",
    (
        "Performance",
        "GradeDistribution",
    ): "The grade filter results show performance by the selected grading category.",
    (
        "Performance",
        "GradingSummary",
    ): "The grading summary provides a breakdown of performance achievements.",
    (
        "Performance",
        "OverallPerformance",
    ): "Overall performance metrics highlight trainee progress across all criteria.",
    (
        "Performance",
        "Improvement",
    ): "These records highlight the trainees showing positive performance growth.",
    (
        "Performance",
        "Drop",
    ): "These trends identify trainees experiencing a decline in assessment scores.",
    (
        "Performance",
        "SectionSummary",
    ): "The section summary provides a view of performance across individual modules.",
    (
        "Performance",
        "AttemptWise",
    ): "Attempt-wise statistics track trainee progress across successive evaluation cycles.",
    (
        "Performance",
        "BestAttempt",
    ): "Best attempt outcomes reflect peak trainee achievements in this evaluation.",
    (
        "Performance",
        "Comparison",
    ): "This comparison highlights achievement differences across the selected categories.",
    (
        "Leave",
        "MostLeaveTaken",
    ): "Leave patterns highlight the person with the highest absence rate.",
    (
        "Leave",
        "LeastLeaveTaken",
    ): "Leave summaries identify the person with the highest duty presence.",
    (
        "Leave",
        "CurrentLeaveStatus",
    ): "Current leave records outline person availability across the unit.",
    (
        "Leave",
        "AbscondedPerson",
    ): "These records flag persons currently absent without official leave.",
    (
        "Medical",
        "ActiveCases",
    ): "This summary captures current active cases undergoing medical attention.",
    (
        "Medical",
        "BMIAnalysis",
    ): "BMI records outline fitness levels and weight distribution across persons.",
    (
        "Medical",
        "DiseaseStatistics",
    ): "Health records highlight the most common medical cases reported recently.",
    (
        "Attendance",
        "MonthlyAttendance",
    ): "Monthly attendance trends provide a view of person participation.",
    (
        "Attendance",
        "PresentToday",
    ): "Today's attendance records outline current person presence on campus.",
    (
        "Attendance",
        "StrengthBreakdown",
    ): "The strength breakdown captures unit headcount and active person counts.",
    (
        "Verification",
        "PendingVerification",
    ): "Verification files track documents currently awaiting official review.",
    (
        "Verification",
        "CompletedVerification",
    ): "These records confirm files that have cleared the verification process.",
    (
        "Equipment",
        "EquipmentSummary",
    ): "This inventory summary reflects current equipment counts and status.",
    (
        "Equipment",
        "OverdueEquipment",
    ): "These records flag issued gear currently overdue for return.",
    (
        "Equipment",
        "PoorConditionEquipment",
    ): "This quality review highlights equipment returned in sub-standard condition.",
    (
        "Equipment",
        "IssuedItems",
    ): "Here is the complete list of items issued to Agniveers.",
    (
        "Equipment",
        "ProcuredItems",
    ): "Here is the complete list of items procured by Agniveers.",
    (
        "Skills",
        "BySport",
    ): "Sport rosters track athletic participation and team assignments.",
    (
        "Skills",
        "ByClass",
    ): "Class rosters group persons by their administrative designations.",
    (
        "Skills",
        "BloodGroup",
    ): "Medical profiles outline the blood group distribution across the group.",
}

_QUERY_TYPE_INTROS: Dict[str, str] = {
    "cross_filter": "Cross-filter analysis completed — records matched across the selected criteria.",
    "comparison": "Comparison completed between the selected categories.",
    "multi_independent": "Combined data successfully consolidated from multiple modules.",
}


# =============================================================================
# AGGREGATE TEXT BUILDER
# =============================================================================
# Builds a compact, aggregate-only text representation of the combinedResult.
# This text is what gets passed to Ollama AND used for grounding.
# It intentionally EXCLUDES individual record data, attempts, sections, subItems.


def _extract_records_from_combined(data: Any) -> List[Dict]:
    """Pull the list of records out of any .NET wrapper shape."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in (
            "data",
            "Data",
            "result",
            "Result",
            "records",
            "Records",
            "persons",
            "personnel",
        ):
            val = data.get(key)
            if isinstance(val, list):
                return val
            if isinstance(val, dict):
                return _extract_records_from_combined(val)
    return []


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


_SCORE_FIELDS = [
    "bestTotal",
    "totalMarks",
    "score",
    "Score",
    "omrInputTotal",
    "marksObtained",
]


def _get_score(record: Dict) -> Optional[float]:
    for field in _SCORE_FIELDS:
        v = _safe_float(record.get(field))
        if v is not None:
            return v
    return None


def _get_name(record: Dict) -> Optional[str]:
    for key in ("fullName", "name", "Name"):
        val = record.get(key)
        if val:
            return str(val).strip()
    return None


def _build_aggregate_text(
    combined_result: Any,
    query_type: str,
    intent: Dict[str, Any],
) -> str:
    """
    Build a compact aggregate-only text from combinedResult.

    This text is the ONLY data the LLM sees. It contains:
      - Counts, totals, percentages, summary metrics
      - Record names (for simple queries)
      - NO attempts, sections, subItems, marks, grades
    """
    category = intent.get("category") or "Agniveer"
    subcategory = intent.get("subcategory") or ""
    lines: List[str] = []

    if query_type == "cross_filter":
        match_count = (
            combined_result.get("matchCount", 0)
            if isinstance(combined_result, dict)
            else 0
        )
        total_before = (
            combined_result.get("totalBeforeFilter", 0)
            if isinstance(combined_result, dict)
            else 0
        )
        filter_depth = (
            combined_result.get("filterDepth", 2)
            if isinstance(combined_result, dict)
            else 2
        )
        records = (
            combined_result.get("records", [])
            if isinstance(combined_result, dict)
            else []
        )

        lines.append(f"Query Type: cross_filter")
        lines.append(f"Match Count: {match_count}")
        lines.append(f"Total Before Filter: {total_before}")
        lines.append(f"Filter Depth: {filter_depth}")
        lines.append(f"Records Returned: {len(records)}")

        if total_before > 0 and match_count > 0:
            pct = round((match_count / total_before) * 100, 1)
            lines.append(f"Match Percentage: {pct}%")

        # Include names only — no nested data
        names = []
        for r in records:
            name = _get_name(r)
            if name:
                names.append(name)
        if names:
            lines.append(f"Matched Agniveers: {', '.join(names)}")

    elif query_type == "comparison":
        sides = (
            combined_result.get("sides", [])
            if isinstance(combined_result, dict)
            else []
        )
        compared_metrics = (
            combined_result.get("comparedMetrics", [])
            if isinstance(combined_result, dict)
            else []
        )

        lines.append(f"Query Type: comparison")
        lines.append(f"Number of Sides: {len(sides)}")

        for i, side in enumerate(sides):
            label = side.get("label", f"Side {i + 1}")
            metrics = side.get("metrics") or {}
            lines.append(f"")
            lines.append(f"Side {i + 1}: {label}")
            for mk, mv in metrics.items():
                lines.append(f"  {mk}: {mv}")

        if compared_metrics:
            lines.append(f"")
            lines.append(f"Compared Metrics: {', '.join(compared_metrics)}")

    elif query_type == "multi_independent":
        sections = (
            combined_result.get("sections", [])
            if isinstance(combined_result, dict)
            else []
        )
        section_count = (
            combined_result.get("sectionCount", len(sections))
            if isinstance(combined_result, dict)
            else 0
        )

        lines.append(f"Query Type: multi_independent")
        lines.append(f"Section Count: {section_count}")

        for sec in sections:
            label = sec.get("label", "Section")
            rec_count = sec.get("recordCount", 0)
            lines.append(f"  Section: {label} — {rec_count} records")

    else:
        # simple
        records = _extract_records_from_combined(combined_result)
        cnt = len(records)

        lines.append(f"Query Type: simple")
        lines.append(f"Category: {category}")
        if subcategory:
            lines.append(f"Subcategory: {subcategory}")
        lines.append(f"Record Count: {cnt}")

        # Extract aggregate scores if available
        scores = [s for s in (_get_score(r) for r in records) if s is not None]
        if scores:
            lines.append(f"Average Score: {round(sum(scores) / len(scores), 2)}")
            lines.append(f"Top Score: {max(scores)}")
            lines.append(f"Bottom Score: {min(scores)}")

        # Extract scalar summary fields from dict wrapper
        if isinstance(combined_result, dict):
            for k, v in combined_result.items():
                if k in ("data", "Data", "result", "Result", "records", "Records"):
                    continue
                if isinstance(v, (int, float)):
                    lines.append(f"{k}: {v}")
                elif isinstance(v, str) and v.strip():
                    # Only short scalar strings
                    if len(v) < 100:
                        lines.append(f"{k}: {v}")

        # Include names only — no nested data
        names = []
        for r in records:
            name = _get_name(r)
            if name:
                names.append(name)
        if names and len(names) <= 20:
            lines.append(f"Records: {', '.join(names)}")
        elif names:
            lines.append(
                f"Records: {', '.join(names[:20])} (and {len(names) - 20} more)"
            )

    return "\n".join(lines)


# =============================================================================
# GROUNDING GUARD (ENHANCED)
# =============================================================================

_METRIC_KEYWORDS = re.compile(
    r"\b(?:scored|marks|mark|attempt|attempts|grade|grades|grading|"
    r"percentage|percent|total|totals|out\s+of|scored\s+\d|"
    r"marks\s+obtained|section\s+score|sub\s*item)"
    r"\b",
    re.IGNORECASE,
)


def _extract_numbers_from_text(text: str) -> set:
    return set(re.findall(r"\b\d+(?:\.\d+)?\b", text or ""))


def _contains_ungrounded_metrics(sentence: str, grounded_numbers: set) -> bool:
    """
    Returns True if the sentence uses metric keywords but contains numbers
    not found in the grounded text. This catches fabricated scores/marks/grades.
    """
    if not _METRIC_KEYWORDS.search(sentence):
        return False
    sentence_numbers = _extract_numbers_from_text(sentence)
    if not sentence_numbers:
        return False
    return not sentence_numbers.issubset(grounded_numbers)


def _strip_ungrounded_numbers(llm_text: str, grounded_text: str) -> str:
    """
    Remove any sentence from llm_text that contains a number not present
    in grounded_text. Prevents hallucinated totals/percentages/counts.
    Also catches metric-keyword sentences with ungrounded numbers.
    """
    grounded_numbers = _extract_numbers_from_text(grounded_text)

    sentences = re.split(r"(?<=[.!?])\s+", (llm_text or "").strip())
    kept: List[str] = []
    for sentence in sentences:
        sentence_numbers = _extract_numbers_from_text(sentence)

        # Drop sentences with ungrounded numbers
        if sentence_numbers and not sentence_numbers.issubset(grounded_numbers):
            bad = sentence_numbers - grounded_numbers
            logger.info(
                json.dumps(
                    {
                        "message": "Grounding guard: dropped sentence with unverified numbers",
                        "trace_id": getattr(_report_ctx, "trace_id", None) or "N/A",
                        "unverified_numbers": list(bad),
                    }
                )
            )
            continue

        # Extra check: metric keywords with ungrounded numbers
        if _contains_ungrounded_metrics(sentence, grounded_numbers):
            logger.info(
                json.dumps(
                    {
                        "message": "Grounding guard: dropped metric sentence with unverified data",
                        "trace_id": getattr(_report_ctx, "trace_id", None) or "N/A",
                    }
                )
            )
            continue

        kept.append(sentence)
    return " ".join(kept).strip()


def _ground_and_sanitize(text: str, grounded_text: str) -> str:
    if not text:
        return ""
    return _strip_ungrounded_numbers(text, grounded_text)


# =============================================================================
# FALLBACK GENERATOR
# =============================================================================


def get_fallback_report(
    combined_result: Any,
    query_type: str,
    intent: Dict[str, Any],
) -> Dict[str, Any]:
    category = intent.get("category") or "Agniveer"
    subcategory = intent.get("subcategory") or ""

    # Extract records and counts
    records = _extract_records_from_combined(combined_result)
    cnt = len(records)

    # Base fallback values
    if query_type == "cross_filter":
        match_count = (
            combined_result.get("matchCount", cnt)
            if isinstance(combined_result, dict)
            else cnt
        )
        total_before = (
            combined_result.get("totalBeforeFilter", 0)
            if isinstance(combined_result, dict)
            else 0
        )
        intro = f"{match_count} Agniveers matched the requested cross-filter criteria."
        summary = f"Cross-filter intersection completed with {match_count} matches."
        obs = [f"{match_count} records matched out of {total_before} primary records."]
        insights = [
            "Intersection identifies trainees matching all filtered properties simultaneously."
        ]
        conclusion = f"{match_count} trainees have been successfully cross-referenced."
    elif query_type == "comparison":
        sides = (
            combined_result.get("sides", [])
            if isinstance(combined_result, dict)
            else []
        )
        labels = [s.get("label", "Section") for s in sides]
        labels_str = " and ".join(labels) if labels else "selected categories"
        intro = f"Comparison between {labels_str} has been completed."
        summary = f"Side-by-side comparison compiled for {labels_str}."
        obs = [f"Compared {len(sides)} categories: {', '.join(labels)}."]
        insights = ["Comparison highlights metric variances across categories."]
        conclusion = "The comparative analysis of the requested metrics is complete."
    elif query_type == "multi_independent":
        sections = (
            combined_result.get("sections", [])
            if isinstance(combined_result, dict)
            else []
        )
        labels = [s.get("label", "Section") for s in sections]
        labels_str = ", ".join(labels) if labels else "multiple modules"
        intro = f"{labels_str} statistics have been consolidated."
        summary = f"Consolidated dataset generated from {len(sections)} sections."
        obs = [
            f"Successfully loaded {len(sections)} independent data sections: {labels_str}."
        ]
        insights = ["No correlation analysis is performed for independent requests."]
        conclusion = "All requested sections are merged into a single report view."
    else:
        # simple
        intro = f"{cnt} records were identified for {category.lower()}."

        # Check specific category templates if available
        key = (category, subcategory)
        if key in _INTRO_TEMPLATES:
            intro = _INTRO_TEMPLATES[key]
        elif query_type in _QUERY_TYPE_INTROS:
            intro = _QUERY_TYPE_INTROS[query_type]

        summary = (
            f"A total of {cnt} records are present in the {category.lower()} dataset."
        )
        obs = [f"Found {cnt} active {category.lower()} records."]
        insights = ["The dataset matches the requested filter criteria."]
        conclusion = f"The {category.lower()} records remain stable and up to date."

    return {
        "introMessage": intro,
        "analysis": {"summary": summary, "observations": obs, "insights": insights},
        "conclusion": {"summary": conclusion},
    }


# =============================================================================
# OLLAMA HELPERS
# =============================================================================


def _call_ollama(
    prompt: str, temperature: float = 0.3, max_tokens: int = 200
) -> Optional[str]:
    """Call Ollama and return the raw text response, or None on failure."""
    try:
        payload: Dict[str, Any] = {
            "model": DEFAULT_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": 1024,
            },
        }
        resp = requests.post(OLLAMA_URL, json=payload, timeout=(1.0, 5.0))
        resp.raise_for_status()
        raw = resp.json().get("message", {}).get("content", "").strip()
        return raw if raw else None
    except Exception as exc:
        logger.warning(
            json.dumps(
                {
                    "message": "Ollama call failed",
                    "trace_id": getattr(_report_ctx, "trace_id", None) or "N/A",
                    "error": str(exc),
                }
            )
        )
        return None


def parse_analysis_json(text: str) -> Dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            val = json.loads(candidate)
            if isinstance(val, dict) and "summary" in val:
                return {
                    "summary": str(val.get("summary") or ""),
                    "observations": [
                        str(x) for x in val.get("observations") or [] if x
                    ],
                    "insights": [str(x) for x in val.get("insights") or [] if x],
                }
        except Exception:
            pass
    return {}


def parse_analysis_non_json(text: str) -> Dict[str, Any]:
    lines = [l.strip().lstrip("-*• ").strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return {}
    summary = lines[0]
    observations = lines[1:4]
    insights = lines[4:6]
    return {"summary": summary, "observations": observations, "insights": insights}


def _parse_llm_analysis(raw: str) -> Dict[str, Any]:
    """Try to parse LLM analysis response as JSON, then fallback to plain text."""
    parsed = parse_analysis_json(raw)
    if not parsed:
        parsed = parse_analysis_non_json(raw)
    return parsed


# =============================================================================
# QUERY-TYPE-SPECIFIC ANALYSIS ENGINES
# =============================================================================

_ANALYSIS_RULES_COMMON = (
    "STRICT RULES:\n"
    "1. Base your response 100% on the Aggregate Data below. Never hallucinate, never invent details.\n"
    "2. Only mention numbers/metrics that appear verbatim in the Aggregate Data.\n"
    "3. Do NOT mention any person's name unless it appears in the Aggregate Data.\n"
    "4. Do NOT analyze individual records, attempts, sections, subItems, marks, or grades.\n"
    "5. 'summary' must be a single string (1 sentence overview).\n"
    "6. 'observations' must be a list of 1-3 strings representing key data points/metrics.\n"
    "7. 'insights' must be a list of 1-2 strings representing trends or observations.\n"
    "8. Produce valid JSON with keys: 'summary', 'observations', 'insights'.\n"
)


def _generate_simple_analysis(aggregate_text: str, user_query: str) -> Dict[str, Any]:
    """Analysis engine for simple queries — focuses on record count and summary metrics."""
    prompt = (
        "You are AgniAI, an intelligent military assistant.\n"
        "Analyze the AGGREGATE data summary below. Focus on:\n"
        "- Record count\n"
        "- Summary metrics (averages, totals, counts)\n"
        "- Returned record names\n\n"
        "Do NOT focus on any single record. Do NOT mention attempts, sections, or sub-items.\n\n"
        + _ANALYSIS_RULES_COMMON
        + f"\nUser Query: {user_query}\n"
        f"Aggregate Data:\n{aggregate_text}\n\n"
        "Generate only the raw JSON."
    )
    raw = _call_ollama(prompt, temperature=0.3, max_tokens=250)
    if raw:
        parsed = _parse_llm_analysis(raw)
        if parsed and parsed.get("summary"):
            return parsed
    return {}


def _generate_cross_filter_analysis(
    aggregate_text: str, user_query: str
) -> Dict[str, Any]:
    """Analysis engine for cross-filter queries — focuses on matchCount, filterDepth, intersection."""
    prompt = (
        "You are AgniAI, an intelligent military assistant.\n"
        "Analyze the cross-filter AGGREGATE data below. Focus ONLY on:\n"
        "- matchCount (how many records matched)\n"
        "- totalBeforeFilter (original pool size)\n"
        "- filterDepth (number of filter criteria applied)\n"
        "- Match percentage\n"
        "- Intersection results\n\n"
        "CRITICAL: Do NOT analyze attempts, sections, subItems, marks, grades, or scores.\n"
        "Do NOT write about individual candidate performance.\n"
        "Focus on the aggregate intersection result.\n\n"
        + _ANALYSIS_RULES_COMMON
        + f"\nUser Query: {user_query}\n"
        f"Aggregate Data:\n{aggregate_text}\n\n"
        "Generate only the raw JSON."
    )
    raw = _call_ollama(prompt, temperature=0.2, max_tokens=250)
    if raw:
        parsed = _parse_llm_analysis(raw)
        if parsed and parsed.get("summary"):
            return parsed
    return {}


def _generate_comparison_analysis(
    aggregate_text: str, user_query: str
) -> Dict[str, Any]:
    """Analysis engine for comparison queries — focuses on side-by-side metrics."""
    prompt = (
        "You are AgniAI, an intelligent military assistant.\n"
        "Analyze the comparison AGGREGATE data below. Focus ONLY on:\n"
        "- Side labels and their metrics\n"
        "- Differences between sides (record counts, averages, scores)\n"
        "- Percentages and relative comparisons\n\n"
        "CRITICAL: Do NOT analyze individual records, nested attempts, or sub-items.\n"
        "Compare the aggregate metrics between sides.\n\n"
        + _ANALYSIS_RULES_COMMON
        + f"\nUser Query: {user_query}\n"
        f"Aggregate Data:\n{aggregate_text}\n\n"
        "Generate only the raw JSON."
    )
    raw = _call_ollama(prompt, temperature=0.3, max_tokens=250)
    if raw:
        parsed = _parse_llm_analysis(raw)
        if parsed and parsed.get("summary"):
            return parsed
    return {}


def _generate_multi_independent_analysis(
    aggregate_text: str, user_query: str
) -> Dict[str, Any]:
    """Analysis engine for multi-independent queries — focuses on section summaries."""
    prompt = (
        "You are AgniAI, an intelligent military assistant.\n"
        "Analyze the multi-section AGGREGATE data below. Focus ONLY on:\n"
        "- Section labels\n"
        "- Record counts per section\n"
        "- Total sections consolidated\n\n"
        "CRITICAL: Do NOT analyze individual records or nested data within sections.\n"
        "Summarize the section-level data only.\n\n"
        + _ANALYSIS_RULES_COMMON
        + f"\nUser Query: {user_query}\n"
        f"Aggregate Data:\n{aggregate_text}\n\n"
        "Generate only the raw JSON."
    )
    raw = _call_ollama(prompt, temperature=0.3, max_tokens=250)
    if raw:
        parsed = _parse_llm_analysis(raw)
        if parsed and parsed.get("summary"):
            return parsed
    return {}


# =============================================================================
# QUERY-TYPE-SPECIFIC CONCLUSION ENGINES
# =============================================================================

_CONCLUSION_RULES_COMMON = (
    "STRICT RULES:\n"
    "1. Base your conclusion 100% on the Analysis and Aggregate Data below.\n"
    "2. Maximum 2-3 sentences.\n"
    "3. NEVER introduce new facts, numbers, or details not in the Analysis or Aggregate Data.\n"
    "4. NEVER mention attempts, sections, subItems, marks, or grades unless they appear in the Aggregate Data.\n"
    "5. Summarize the analysis findings concisely.\n"
)


def _generate_conclusion(
    aggregate_text: str,
    analysis_data: Dict[str, Any],
    user_query: str,
    query_type_instruction: str,
) -> str:
    """Generic conclusion generator with query-type-specific instructions."""
    analysis_summary = analysis_data.get("summary", "")
    analysis_obs = analysis_data.get("observations", [])
    analysis_ins = analysis_data.get("insights", [])

    analysis_text_parts = []
    if analysis_summary:
        analysis_text_parts.append(f"Summary: {analysis_summary}")
    if analysis_obs:
        analysis_text_parts.append("Observations: " + "; ".join(analysis_obs))
    if analysis_ins:
        analysis_text_parts.append("Insights: " + "; ".join(analysis_ins))
    analysis_text = (
        "\n".join(analysis_text_parts)
        if analysis_text_parts
        else "No analysis available."
    )

    prompt = (
        "You are AgniAI, an intelligent military assistant.\n"
        "Generate a brief conclusion summarizing the analysis findings.\n\n"
        + query_type_instruction
        + "\n\n"
        + _CONCLUSION_RULES_COMMON
        + f"\nUser Query: {user_query}\n"
        f"Aggregate Data:\n{aggregate_text}\n\n"
        f"Analysis:\n{analysis_text}\n\n"
        "Generate only the conclusion text (2-3 sentences)."
    )
    raw = _call_ollama(prompt, temperature=0.3, max_tokens=120)
    if raw:
        # Clean up common LLM prefixes
        cleaned = re.sub(r"^(?:CONCLUSION\s*:\s*)", "", raw, flags=re.IGNORECASE)
        cleaned = re.sub(r"[*_`#]", "", cleaned).strip()
        cleaned = cleaned.strip('"' + "'")
        return cleaned
    return ""


def _generate_simple_conclusion(
    aggregate_text: str, analysis_data: Dict[str, Any], user_query: str
) -> str:
    return _generate_conclusion(
        aggregate_text,
        analysis_data,
        user_query,
        "Focus on summarizing the record count and overall dataset findings.",
    )


def _generate_cross_filter_conclusion(
    aggregate_text: str, analysis_data: Dict[str, Any], user_query: str
) -> str:
    return _generate_conclusion(
        aggregate_text,
        analysis_data,
        user_query,
        "Focus on the cross-filter intersection result: how many matched and out of how many. "
        "Do NOT mention individual candidate performance, attempts, or scores.",
    )


def _generate_comparison_conclusion(
    aggregate_text: str, analysis_data: Dict[str, Any], user_query: str
) -> str:
    return _generate_conclusion(
        aggregate_text,
        analysis_data,
        user_query,
        "Focus on the comparison between sides: which side has more/fewer records, higher/lower metrics. "
        "Do NOT mention individual records.",
    )


def _generate_multi_independent_conclusion(
    aggregate_text: str, analysis_data: Dict[str, Any], user_query: str
) -> str:
    return _generate_conclusion(
        aggregate_text,
        analysis_data,
        user_query,
        "Focus on summarizing how many sections were consolidated and their respective record counts. "
        "Do NOT analyze individual records within sections.",
    )


# =============================================================================
# ANALYSIS / CONCLUSION DISPATCH
# =============================================================================

_ANALYSIS_ENGINES = {
    "simple": _generate_simple_analysis,
    "cross_filter": _generate_cross_filter_analysis,
    "comparison": _generate_comparison_analysis,
    "multi_independent": _generate_multi_independent_analysis,
}

_CONCLUSION_ENGINES = {
    "simple": _generate_simple_conclusion,
    "cross_filter": _generate_cross_filter_conclusion,
    "comparison": _generate_comparison_conclusion,
    "multi_independent": _generate_multi_independent_conclusion,
}


# =============================================================================
# PUBLIC INTERFACE
# =============================================================================


def generate_report(
    combined_result: Any,
    query_type: str,
    intent: Dict[str, Any],
    user_query: str = "",
    trace_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Produce the structured report: introMessage, analysis, and conclusion.

    CRITICAL: All analysis is based on the aggregate summary of combinedResult,
    never on individual records. The LLM only sees aggregate metrics.
    """
    _report_ctx.trace_id = trace_id
    try:
        # Check record count first
        records = _extract_records_from_combined(combined_result)
        if not records:
            logger.info("Report Generator: 0 records found. Skipping LLM.")
            return {
                "summary": "No matching records found.",
                "introMessage": "No matching records found.",
                "analysis": {
                    "summary": "No matching records found.",
                    "observations": [],
                    "insights": []
                },
                "conclusion": {
                    "summary": "No matching records found."
                }
            }

        # Build the aggregate-only text — this is the ONLY data the LLM sees
        aggregate_text = _build_aggregate_text(combined_result, query_type, intent)
        logger.info(
            json.dumps(
                {
                    "message": "Report Generator aggregate text built",
                    "trace_id": trace_id or "N/A",
                    "text_length": len(aggregate_text) if aggregate_text else 0,
                }
            )
        )

        fallback = get_fallback_report(combined_result, query_type, intent)

        if not aggregate_text or len(aggregate_text.strip()) < 10:
            return fallback

        prompt = (
            "You are AgniAI, an intelligent military assistant.\n"
            "Generate a complete query report in JSON format based on the AGGREGATE data summary below.\n\n"
            "STRICT RULES:\n"
            "1. Base your response 100% on the Aggregate Data below. Never hallucinate, never invent details.\n"
            "2. Only mention numbers/metrics that appear verbatim in the Aggregate Data.\n"
            "3. Do NOT mention any person's name unless it appears in the Aggregate Data.\n"
            "4. Do NOT analyze individual records, attempts, sections, subItems, marks, or grades.\n"
            "5. You must output a single JSON object with EXACTLY the following structure:\n"
            "{\n"
            '  "introMessage": "A short 1-2 sentence description of what data was retrieved.",\n'
            '  "analysis": {\n'
            '    "summary": "A single sentence overview of the aggregate metrics.",\n'
            '    "observations": ["1-3 key data points/metrics from the aggregate data"],\n'
            '    "insights": ["1-2 trends or insights based on the aggregate data"]\n'
            '  },\n'
            '  "conclusion": "A brief conclusion summarizing the analysis (1-2 sentences)."\n'
            "}\n\n"
            f"User Query: {user_query}\n"
            f"Query Type: {query_type}\n"
            f"Aggregate Data:\n{aggregate_text}\n\n"
            "Generate only the raw JSON, do not include markdown formatting (such as ```json) or any extra text."
        )

        # Call Ollama exactly once
        raw_response = _call_ollama(prompt, temperature=0.2, max_tokens=350)
        if not raw_response:
            return fallback

        # Parse JSON
        parsed_report = None
        start = raw_response.find("{")
        end = raw_response.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = raw_response[start : end + 1]
            try:
                parsed_report = json.loads(candidate)
            except Exception:
                pass

        if not parsed_report or not isinstance(parsed_report, dict):
            return fallback

        intro_message = parsed_report.get("introMessage")
        if not intro_message or not isinstance(intro_message, str):
            intro_message = fallback["introMessage"]
        intro_message = _ground_and_sanitize(intro_message, aggregate_text) or fallback["introMessage"]

        analysis_dict = parsed_report.get("analysis") or {}
        if not isinstance(analysis_dict, dict):
            analysis_dict = {}

        clean_summary = _ground_and_sanitize(analysis_dict.get("summary", ""), aggregate_text) or fallback["analysis"]["summary"]

        clean_obs = []
        for obs in analysis_dict.get("observations") or []:
            san = _ground_and_sanitize(str(obs), aggregate_text)
            if san:
                clean_obs.append(san)
        if not clean_obs:
            clean_obs = fallback["analysis"]["observations"]

        clean_ins = []
        for ins in analysis_dict.get("insights") or []:
            san = _ground_and_sanitize(str(ins), aggregate_text)
            if san:
                clean_ins.append(san)
        if not clean_ins:
            clean_ins = fallback["analysis"]["insights"]

        grounded_analysis = {
            "summary": clean_summary,
            "observations": clean_obs,
            "insights": clean_ins,
        }

        conclusion_val = parsed_report.get("conclusion")
        if isinstance(conclusion_val, dict):
            conclusion_text = conclusion_val.get("summary") or fallback["conclusion"]["summary"]
        elif isinstance(conclusion_val, str):
            conclusion_text = conclusion_val
        else:
            conclusion_text = fallback["conclusion"]["summary"]

        conclusion_text = _ground_and_sanitize(conclusion_text, aggregate_text) or fallback["conclusion"]["summary"]

        return {
            "introMessage": intro_message,
            "analysis": grounded_analysis,
            "conclusion": {"summary": conclusion_text},
        }

    except Exception as exc:
        logger.error(
            json.dumps(
                {
                    "message": "LLM failure in generate_report, applying graceful degradation",
                    "trace_id": trace_id or "N/A",
                    "error": str(exc),
                }
            )
        )
        try:
            fallback = get_fallback_report(combined_result, query_type, intent)
            intro_msg = fallback.get("introMessage", "")
        except Exception:
            intro_msg = "Report generated with partial metrics."
        return {"introMessage": intro_msg, "analysis": None, "conclusion": None}
    finally:
        _report_ctx.trace_id = None
