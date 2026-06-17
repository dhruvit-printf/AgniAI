"""
report_generator.py
===================
Report Generator Layer for the AgniAI Admin Chatbot Response Pipeline.

CRITICAL CONTRACT:
  - Input: combinedResult, queryType, intent, user_query
  - Output: introMessage (string), analysis (dict), conclusion (dict)
  - Enforces strict grounding guard via formatted plain-text representation of combinedResult.
  - Never consumes raw .NET response directly or query alone.
"""

from __future__ import annotations

import re
import json
import logging
import requests
from typing import Any, Dict, List, Optional, Tuple

from config import OLLAMA_URL, DEFAULT_MODEL
from admin_formatter import format_dotnet_response

logger = logging.getLogger(__name__)

# =============================================================================
# INTRO TEMPLATES
# =============================================================================

_INTRO_TEMPLATES: Dict[Tuple[str, str], str] = {
    ("Performance", "TopPerformers"):      "These assessment results highlight the strongest performers in the evaluation.",
    ("Performance", "LowestPerformers"):   "These results identify the individuals requiring additional training support.",
    ("Performance", "AverageScore"):       "The average scores outline overall achievement levels across the group.",
    ("Performance", "PassPercentage"):     "Pass rates reflect the percentage of trainees meeting the assessment standards.",
    ("Performance", "FailPercentage"):     "Fail rates identify the proportion of trainees currently below standard.",
    ("Performance", "GradeDistribution"):  "The grade filter results show performance by the selected grading category.",
    ("Performance", "GradingSummary"):     "The grading summary provides a breakdown of performance achievements.",
    ("Performance", "OverallPerformance"): "Overall performance metrics highlight trainee progress across all criteria.",
    ("Performance", "Improvement"):        "These records highlight the trainees showing positive performance growth.",
    ("Performance", "Drop"):               "These trends identify trainees experiencing a decline in assessment scores.",
    ("Performance", "SectionSummary"):     "The section summary provides a view of performance across individual modules.",
    ("Performance", "AttemptWise"):        "Attempt-wise statistics track trainee progress across successive evaluation cycles.",
    ("Performance", "BestAttempt"):        "Best attempt outcomes reflect peak trainee achievements in this evaluation.",
    ("Performance", "Comparison"):         "This comparison highlights achievement differences across the selected categories.",
    ("Leave", "MostLeaveTaken"):           "Leave patterns highlight the person with the highest absence rate.",
    ("Leave", "LeastLeaveTaken"):          "Leave summaries identify the person with the highest duty presence.",
    ("Leave", "CurrentLeaveStatus"):       "Current leave records outline person availability across the unit.",
    ("Leave", "AbscondedPerson"):          "These records flag persons currently absent without official leave.",
    ("Medical", "ActiveCases"):            "This summary captures current active cases undergoing medical attention.",
    ("Medical", "BMIAnalysis"):            "BMI records outline fitness levels and weight distribution across persons.",
    ("Medical", "DiseaseStatistics"):      "Health records highlight the most common medical cases reported recently.",
    ("Attendance", "MonthlyAttendance"):   "Monthly attendance trends provide a view of person participation.",
    ("Attendance", "PresentToday"):        "Today's attendance records outline current person presence on campus.",
    ("Attendance", "StrengthBreakdown"):   "The strength breakdown captures unit headcount and active person counts.",
    ("Verification", "PendingVerification"):   "Verification files track documents currently awaiting official review.",
    ("Verification", "CompletedVerification"): "These records confirm files that have cleared the verification process.",
    ("Equipment", "EquipmentSummary"):     "This inventory summary reflects current equipment counts and status.",
    ("Equipment", "OverdueEquipment"):     "These records flag issued gear currently overdue for return.",
    ("Equipment", "PoorConditionEquipment"): "This quality review highlights equipment returned in sub-standard condition.",
    ("Equipment", "IssuedItems"):          "Here is the complete list of items issued to Agniveers.",
    ("Equipment", "ProcuredItems"):        "Here is the complete list of items procured by Agniveers.",
    ("Skills", "BySport"):                   "Sport rosters track athletic participation and team assignments.",
    ("Skills", "ByClass"):                   "Class rosters group persons by their administrative designations.",
    ("Skills", "BloodGroup"):                "Medical profiles outline the blood group distribution across the group.",
}

_QUERY_TYPE_INTROS: Dict[str, str] = {
    "cross_filter":      "Cross-filter analysis completed — records matched across the selected criteria.",
    "comparison":        "Comparison completed between the selected categories.",
    "multi_independent": "Combined data successfully consolidated from multiple modules.",
}


# =============================================================================
# GROUNDING GUARD
# =============================================================================

def _extract_numbers_from_text(text: str) -> set:
    return set(re.findall(r"\b\d+(?:\.\d+)?\b", text or ""))


def _strip_ungrounded_numbers(llm_text: str, grounded_text: str) -> str:
    """
    Remove any sentence from llm_text that contains a number not present
    in grounded_text. Prevents hallucinated totals/percentages/counts.
    """
    grounded_numbers = _extract_numbers_from_text(grounded_text)

    sentences = re.split(r"(?<=[.!?])\s+", (llm_text or "").strip())
    kept: List[str] = []
    for sentence in sentences:
        sentence_numbers = _extract_numbers_from_text(sentence)
        if not sentence_numbers:
            kept.append(sentence)
            continue
        if sentence_numbers.issubset(grounded_numbers):
            kept.append(sentence)
        else:
            bad = sentence_numbers - grounded_numbers
            logger.debug(
                "Grounding guard: dropped sentence with unverified numbers %s: %r",
                bad, sentence,
            )
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
    from result_combiner import _extract_records
    records = _extract_records(combined_result)
    cnt = len(records)
    
    # Base fallback values
    if query_type == "cross_filter":
        match_count = combined_result.get("matchCount", cnt)
        total_before = combined_result.get("totalBeforeFilter", 0)
        intro = f"{match_count} Agniveers matched the requested cross-filter criteria."
        summary = f"Cross-filter intersection completed with {match_count} matches."
        obs = [f"{match_count} records matched out of {total_before} primary records."]
        insights = ["Intersection identifies trainees matching all filtered properties simultaneously."]
        conclusion = f"{match_count} trainees have been successfully cross-referenced."
    elif query_type == "comparison":
        sides = combined_result.get("sides", [])
        labels = [s.get("label", "Section") for s in sides]
        labels_str = " and ".join(labels) if labels else "selected categories"
        intro = f"Comparison between {labels_str} has been completed."
        summary = f"Side-by-side comparison compiled for {labels_str}."
        obs = [f"Compared {len(sides)} categories: {', '.join(labels)}."]
        insights = ["Comparison highlights metric variances across categories."]
        conclusion = "The comparative analysis of the requested metrics is complete."
    elif query_type == "multi_independent":
        sections = combined_result.get("sections", [])
        labels = [s.get("label", "Section") for s in sections]
        labels_str = ", ".join(labels) if labels else "multiple modules"
        intro = "Attendance, equipment, and verification statistics have been consolidated."
        summary = f"Consolidated dataset generated from {labels_str} sections."
        obs = [f"Successfully loaded {len(sections)} independent data sections."]
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

        summary = f"A total of {cnt} records are present in the {category.lower()} dataset."
        obs = [f"Found {cnt} active {category.lower()} records."]
        insights = ["The dataset matches the requested filter criteria."]
        conclusion = f"The {category.lower()} records remain stable and up to date."
        
    return {
        "introMessage": intro,
        "analysis": {
            "summary": summary,
            "observations": obs,
            "insights": insights
        },
        "conclusion": {
            "summary": conclusion
        }
    }


# =============================================================================
# OLLAMA HELPERS
# =============================================================================

def parse_analysis_json(text: str) -> Dict[str, Any]:
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end+1]
        try:
            val = json.loads(candidate)
            if isinstance(val, dict) and "summary" in val:
                return {
                    "summary": str(val.get("summary") or ""),
                    "observations": [str(x) for x in val.get("observations") or [] if x],
                    "insights": [str(x) for x in val.get("insights") or [] if x]
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
    return {
        "summary": summary,
        "observations": observations,
        "insights": insights
    }


# =============================================================================
# PUBLIC INTERFACE
# =============================================================================

def generate_report(
    combined_result: Any,
    query_type: str,
    intent: Dict[str, Any],
    user_query: str = "",
) -> Dict[str, Any]:
    """
    Produce the structured report: introMessage, analysis, and conclusion.
    Enforces strict grounding guard to prevent LLM hallucinations.
    """
    formatted_data = format_dotnet_response(combined_result, intent)
    fallback = get_fallback_report(combined_result, query_type, intent)

    if not formatted_data or "No records found" in formatted_data or "No comparison data" in formatted_data:
        return fallback

    category = intent.get("category") or "Agniveer data"
    
    # 1. Generate Intro Message
    intro_prompt = (
        "You are AgniAI, an intelligent military assistant.\n"
        "Generate a short introductory summary (1-2 sentences) describing what data was retrieved.\n\n"
        "STRICT RULES:\n"
        "1. Maximum 1-2 sentences. End with a period.\n"
        "2. No analysis, recommendations, or assumptions.\n"
        "3. State only what has been identified or fetched.\n"
        "4. You can mention count numbers if they are in the Formatted Data, but do not calculate or guess other numbers.\n\n"
        f"User Query: {user_query}\n"
        f"Query Type: {query_type}\n"
        f"Formatted Data:\n{formatted_data}\n\n"
        "Generate only the introductory sentence."
    )

    intro_message = fallback["introMessage"]
    try:
        payload = {
            "model": DEFAULT_MODEL,
            "messages": [{"role": "user", "content": intro_prompt}],
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_predict": 80,
                "num_ctx": 1024,
            },
        }
        resp = requests.post(OLLAMA_URL, json=payload, timeout=(8, 30))
        resp.raise_for_status()
        raw_intro = resp.json().get("message", {}).get("content", "").strip()
        raw_intro = raw_intro.strip('"' + "'")
        clean_intro = _ground_and_sanitize(raw_intro, formatted_data)
        if clean_intro and len(clean_intro) > 5:
            intro_message = clean_intro
    except Exception as exc:
        logger.debug("Ollama intro generation failed: %s", exc)

    # 2. Generate Analysis
    analysis_prompt = (
        "You are AgniAI, an intelligent military assistant.\n"
        "Analyze the provided training data based on the User Query and Formatted Data.\n"
        "Produce your response in valid JSON format with three keys: 'summary', 'observations', 'insights'.\n\n"
        "STRICT RULES:\n"
        "1. Base your response 100% on the Formatted Data. Never hallucinate, never invent or extrapolate details.\n"
        "2. Do NOT mention any person's name or specific details not in the data.\n"
        "3. Only mention numbers/metrics that appear verbatim in the Formatted Data.\n"
        "4. 'summary' must be a single string (1 sentence overview).\n"
        "5. 'observations' must be a list of 1-3 strings representing key data points/metrics/counts.\n"
        "6. 'insights' must be a list of 1-2 strings representing trends, anomalies, or observations.\n"
        "7. Ensure the JSON is properly formatted.\n\n"
        f"User Query: {user_query}\n"
        f"Query Type: {query_type}\n"
        f"Formatted Data:\n{formatted_data}\n\n"
        "Generate only the raw JSON."
    )

    analysis_data = fallback["analysis"]
    try:
        payload = {
            "model": DEFAULT_MODEL,
            "messages": [{"role": "user", "content": analysis_prompt}],
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 250,
                "num_ctx": 1024,
            },
        }
        resp = requests.post(OLLAMA_URL, json=payload, timeout=(8, 30))
        resp.raise_for_status()
        raw_analysis = resp.json().get("message", {}).get("content", "").strip()
        
        parsed = parse_analysis_json(raw_analysis)
        if not parsed:
            parsed = parse_analysis_non_json(raw_analysis)
            
        if parsed and parsed.get("summary"):
            analysis_data = parsed
    except Exception as exc:
        logger.debug("Ollama analysis generation failed: %s", exc)

    # Apply Grounding Guard to Analysis fields
    clean_summary = _ground_and_sanitize(analysis_data.get("summary", ""), formatted_data)
    if not clean_summary:
        clean_summary = fallback["analysis"]["summary"]

    clean_obs = []
    for obs in analysis_data.get("observations", []):
        san = _ground_and_sanitize(obs, formatted_data)
        if san:
            clean_obs.append(san)
    if not clean_obs:
        clean_obs = fallback["analysis"]["observations"]

    clean_ins = []
    for ins in analysis_data.get("insights", []):
        san = _ground_and_sanitize(ins, formatted_data)
        if san:
            clean_ins.append(san)
    if not clean_ins:
        clean_ins = fallback["analysis"]["insights"]

    # 3. Generate Conclusion
    conclusion_prompt = (
        "You are AgniAI, an intelligent military assistant.\n"
        "Generate a brief conclusion (2-4 sentences) summarizing findings from the analysis.\n\n"
        "STRICT RULES:\n"
        "1. Base your response 100% on the Formatted Data. Do not introduce new information.\n"
        "2. Maximum 2-4 sentences.\n"
        "3. Never hallucinate or extrapolate.\n\n"
        f"User Query: {user_query}\n"
        f"Formatted Data:\n{formatted_data}\n\n"
        "Generate only the conclusion text."
    )

    conclusion_text = fallback["conclusion"]["summary"]
    try:
        payload = {
            "model": DEFAULT_MODEL,
            "messages": [{"role": "user", "content": conclusion_prompt}],
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 120,
                "num_ctx": 1024,
            },
        }
        resp = requests.post(OLLAMA_URL, json=payload, timeout=(8, 30))
        resp.raise_for_status()
        raw_conclusion = resp.json().get("message", {}).get("content", "").strip()
        raw_conclusion = re.sub(r'^(?:CONCLUSION\s*:\s*)', '', raw_conclusion, flags=re.IGNORECASE)
        raw_conclusion = re.sub(r'[*_`#]', '', raw_conclusion).strip()
        clean_conclusion = _ground_and_sanitize(raw_conclusion, formatted_data)
        if clean_conclusion and len(clean_conclusion) > 5:
            conclusion_text = clean_conclusion
    except Exception as exc:
        logger.debug("Ollama conclusion generation failed: %s", exc)

    return {
        "introMessage": intro_message,
        "analysis": {
            "summary": clean_summary,
            "observations": clean_obs,
            "insights": clean_ins
        },
        "conclusion": {
            "summary": conclusion_text
        }
    }
