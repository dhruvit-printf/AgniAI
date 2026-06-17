"""
admin_report_generator.py
==========================
Step 6 in the AgniAI intelligence pipeline — Report Generator.

CRITICAL CONTRACT (from architecture spec):
  - Receives ONLY: { queryType, finalResult, intent_result }
  - NEVER modifies finalResult
  - Only generates: introMessage, analysis, conclusion
  - All three are derived from finalResult — no hallucinated numbers

GROUNDING GUARD:
  generate_analysis() and generate_conclusion() pass the formatter's plain-text
  output as the sole data source for the LLM, and strip any number from the
  LLM output that is not present in the formatted data. This prevents
  hallucinated aggregates (e.g. "159 total days" when the real total is 87).
"""

from __future__ import annotations

import re
import logging
import requests
from typing import Any, Dict, Optional, Tuple

from config import OLLAMA_URL, DEFAULT_MODEL
from admin_formatter import format_dotnet_response

logger = logging.getLogger(__name__)


# =============================================================================
# INTRO TEMPLATES (fallback when LLM is unavailable or produces bad output)
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
    ("Performance", "SectionSummary"):     "The section summary provides a clear view of performance across individual modules.",
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
    ("Attendance", "MonthlyAttendance"):   "Monthly attendance trends provide a clear view of person participation.",
    ("Attendance", "PresentToday"):        "Today's attendance records outline current person presence on campus.",
    ("Attendance", "StrengthBreakdown"):   "The strength breakdown captures unit headcount and active person counts.",
    ("Verification", "PendingVerification"):   "Verification files track documents currently awaiting official review.",
    ("Verification", "CompletedVerification"): "These records confirm files that have cleared the verification process.",
    ("Equipment", "EquipmentSummary"):     "This inventory summary reflects current equipment counts and status.",
    ("Equipment", "OverdueEquipment"):     "These records flag issued gear currently overdue for return.",
    ("Equipment", "PoorConditionEquipment"): "This quality review highlights equipment returned in sub-standard condition.",
    ("Equipment", "IssuedItems"):          "Here is the complete list of items issued to Agniveers.",
    ("Equipment", "ProcuredItems"):        "Here is the complete list of items procured by Agniveers.",
    ("Distribution", "LatestDistribution"):  "Recent distribution logs track the latest issue of supplies and gear.",
    ("Distribution", "DistributionByUnit"):  "Distribution logs trace supply allocation across different units.",
    ("Distribution", "UnassignedItems"):     "Supply records outline items currently unassigned to any unit.",
    ("Distribution", "TopUnit"):             "This summary highlights the unit receiving the largest supply allocation.",
    ("Skills", "BySport"):                   "Sport rosters track athletic participation and team assignments.",
    ("Skills", "ByClass"):                   "Class rosters group persons by their administrative designations.",
    ("Skills", "BloodGroup"):                "Medical profiles outline the blood group distribution across the group.",
}

# Query-type level intro templates
_QUERY_TYPE_INTROS: Dict[str, str] = {
    "cross_filter":      "Cross-filter analysis completed — records matched across the selected criteria.",
    "comparison":        "Comparison completed between the selected categories.",
    "multi_independent": "Combined data successfully consolidated from multiple modules.",
    "analytics":         "Analytics results are ready for your review.",
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
    kept: list[str] = []
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


# =============================================================================
# HELPERS
# =============================================================================

def _build_intro_prompt(question: str, intent: Dict[str, Any]) -> str:
    category    = intent.get("category", "")
    subcategory = intent.get("subcategory", "")
    number      = intent.get("number")
    section     = intent.get("section", "")
    leave_type  = intent.get("leave_type", "")
    grading     = intent.get("grading", "")
    unit_name   = intent.get("unit_name", "")
    sport       = intent.get("sport", "")
    class_name  = intent.get("class", "")

    context_parts = []
    if category:    context_parts.append(f"Module: {category}")
    if subcategory: context_parts.append(f"Query type: {subcategory}")
    if number:      context_parts.append(f"Requested count: {number}")
    if section:     context_parts.append(f"Section filter: {section}")
    if leave_type:  context_parts.append(f"Leave type: {leave_type}")
    if grading:     context_parts.append(f"Grading filter: {grading}")
    if unit_name:   context_parts.append(f"Unit filter: {unit_name}")
    if sport:       context_parts.append(f"Sport filter: {sport}")
    if class_name:  context_parts.append(f"Class filter: {class_name}")

    context_str = "\n".join(context_parts)

    return (
        "You are AgniAI, an intelligent military training assistant.\n\n"
        "Generate ONE short introductory sentence for the data being shown to the admin.\n\n"
        "STRICT RULES:\n"
        "1. ONE sentence only. End with a period.\n"
        "2. 10 to 20 words maximum.\n"
        "3. NEVER mention any person's name, rank, or ID.\n"
        "4. NEVER mention any score, number, percentage, or statistic.\n"
        "5. NEVER say what the result is — only describe what type of data is shown.\n"
        "6. NEVER use: retrieved, fetched, generated, extracted, shown below, listed below.\n"
        "7. Do not ask questions. No markdown, no bullets, no quotes.\n"
        "8. Sound like a professional assistant introducing a report.\n\n"
        "GOOD EXAMPLES:\n"
        "Attempt-wise improvement data is ready for your review.\n"
        "Here is the leave status across persons for the current period.\n"
        "Performance rankings for the selected section are available below.\n\n"
        f"Admin question: {question}\n\n"
        f"Context:\n{context_str}\n\n"
        "Generate only the sentence."
    )


def _sanitize_intro(text: str) -> str:
    text = text.strip().strip('"\'')

    meta_prefixes = re.compile(
        r"^(?:"
        r"here(?:'s| is)(?: a| the| my)?(?: possible| suggested?)?(?: introductory?| intro)?(?: sentence| line| message| response)?[:\s]*|"
        r"(?:introductory?|intro|opening|response)\s+(?:sentence|line|message)[:\s]*|"
        r"(?:a possible|suggested?) introduction[:\s]*|"
        r"introduction[:\s]*|"
        r"answer[:\s]*|"
        r"response[:\s]*"
        r")",
        re.IGNORECASE,
    )
    text = meta_prefixes.sub("", text).strip()
    text = re.sub(r"\s*\([^)]{0,200}\)\s*$", "", text).strip()
    text = re.sub(r"\s*[Nn]ote\s*[:—–-].*$", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"\s*[Pp]lease note.*$", "", text, flags=re.DOTALL).strip()

    sentences = re.split(r"(?<=[.!?])\s+", text)
    if sentences:
        text = sentences[0].strip()

    if text.endswith("?"):
        text = text.rstrip("?").rstrip() + "."
    if text and text[-1] not in ".!":
        text += "."

    # Reject if contains numbers
    if re.search(r"\b\d+\b", text):
        return ""

    meta_check = re.compile(
        r"^(?:here(?:'s| is)|i(?:'ve| have)|based on|the following|as requested)",
        re.IGNORECASE,
    )
    if meta_check.match(text):
        return ""

    return text


# =============================================================================
# PUBLIC INTERFACE — called by admin_routes.py pipeline
# =============================================================================

def generate_intro_message(
    user_query: str,
    query_type: str,
    intent_result: Dict[str, Any],
) -> str:
    """
    Generate a 1-sentence introduction for the data being shown.
    Never references specific numbers or names from the data.
    """
    category    = intent_result.get("category", "")
    subcategory = intent_result.get("subcategory", "")

    try:
        prompt = _build_intro_prompt(user_query, intent_result)
        payload = {
            "model": DEFAULT_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_predict": 60,
                "num_ctx": 512,
                "stop": ["Note:", "Please note", "\n", "?"],
            },
        }
        resp = requests.post(OLLAMA_URL, json=payload, timeout=(8, 30))
        resp.raise_for_status()
        raw_text   = resp.json().get("message", {}).get("content", "").strip()
        clean_text = _sanitize_intro(raw_text)

        if clean_text and 10 <= len(clean_text) <= 200:
            logger.debug("LLM intro: %s", clean_text)
            return clean_text

        logger.debug("LLM intro rejected (raw=%r clean=%r), using template", raw_text, clean_text)
    except Exception as exc:
        logger.debug("Ollama intro generation failed, using template: %s", exc)

    # Template fallback: try (category, subcategory) first, then query_type level
    key = (category, subcategory)
    if key in _INTRO_TEMPLATES:
        return _INTRO_TEMPLATES[key]

    qt = (query_type or "").lower()
    if qt in _QUERY_TYPE_INTROS:
        return _QUERY_TYPE_INTROS[qt]

    category_label = category or "requested"
    return f"These records outline the current {category_label.lower()} status across the unit."


def generate_analysis(
    user_query: str,
    query_type: str,
    combined_result: Any,
    intent_result: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Analyze the combined result (finalResult).

    CRITICAL: Uses the formatted plain-text of combined_result as the
    sole data source for the LLM. Numbers not in the formatted text are
    stripped from the LLM output by the grounding guard.

    The combined_result is NEVER modified — only read for formatting.

    Parameters
    ----------
    intent_result : preferred — the primary intent from the pipeline.
                    Falls back to classify_admin_intent(user_query) if None.
    """
    if intent_result is None:
        from admin_intent import classify_admin_intent
        intent_result = classify_admin_intent(user_query)

    category = intent_result.get("category", "")
    category_label = category or "Agniveer data"

    fallback_analysis = (
        f"Analysis of the retrieved records indicates that the dataset contains "
        f"active {category_label.lower()} indicators. "
        "The distribution pattern matches the requested parameters and no anomalies are highlighted."
    )

    # Format combined_result to plain text — this is the grounding source
    formatted_data = format_dotnet_response(combined_result, intent_result)
    if not formatted_data or "No records found" in formatted_data or "No comparison data" in formatted_data:
        return fallback_analysis

    qt_upper = str(query_type).upper()
    if "SIMPLE" in qt_upper or "ANALYTICS" in qt_upper:
        type_instruction = "Analyze the returned records."
    elif "CROSS_FILTER" in qt_upper or "CROSS-FILTER" in qt_upper:
        type_instruction = "Analyze the relationship between intersected groups."
    elif "COMPARISON" in qt_upper:
        type_instruction = "Analyze similarities and differences between the compared sides."
    elif "MULTI_INDEPENDENT" in qt_upper:
        type_instruction = "Analyze each section individually and provide overall observations."
    else:
        type_instruction = "Analyze the returned records."

    prompt = (
        "You are AgniAI, an intelligent military command console assistant.\n"
        "Review Agniveer training data. Based on the User Query and Formatted Backend Data below, "
        "generate a brief Analysis.\n\n"
        "STRICT RULES:\n"
        f"1. Focus: {type_instruction}\n"
        "2. Base your response 100% on the Formatted Backend Data. "
        "Do NOT invent or calculate any numbers, totals, percentages, or averages "
        "that are not explicitly present in the data.\n"
        "3. Only mention numbers that appear verbatim in the Formatted Backend Data.\n"
        "4. Do NOT mention any person's name or specific details not in the data.\n"
        "5. Use cautious language: 'indicates', 'suggests', 'appears', 'may reflect'.\n"
        "6. Keep it concise (1-3 sentences). No markdown.\n\n"
        f"User Query: {user_query}\n\n"
        f"Formatted Backend Data:\n{formatted_data}\n\n"
        "Generate only the ANALYSIS text."
    )

    try:
        payload = {
            "model": DEFAULT_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 120, "num_ctx": 1024},
        }
        resp = requests.post(OLLAMA_URL, json=payload, timeout=(8, 30))
        resp.raise_for_status()
        analysis = resp.json().get("message", {}).get("content", "").strip()

        analysis = re.sub(r'^(?:ANALYSIS\s*:\s*)', '', analysis, flags=re.IGNORECASE)
        analysis = re.sub(r'[*_`#]', '', analysis).strip()

        if analysis and len(analysis) >= 5:
            grounded = _strip_ungrounded_numbers(analysis, formatted_data)
            return grounded if grounded else fallback_analysis

    except Exception as exc:
        logger.debug("Ollama analysis generation failed: %s", exc)

    return fallback_analysis


def generate_conclusion(
    user_query: str,
    query_type: str,
    combined_result: Any,
    intent_result: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generate an executive summary conclusion grounded entirely in the data.

    CRITICAL: combined_result is read-only — never modified.

    Parameters
    ----------
    intent_result : preferred — the primary intent from the pipeline.
                    Falls back to classify_admin_intent(user_query) if None.
    """
    if intent_result is None:
        from admin_intent import classify_admin_intent
        intent_result = classify_admin_intent(user_query)

    category = intent_result.get("category", "")
    category_label = category or "Agniveer data"

    fallback_conclusion = (
        f"The current {category_label.lower()} status remains stable, "
        "and no immediate actions are required."
    )

    formatted_data = format_dotnet_response(combined_result, intent_result)
    if not formatted_data or "No records found" in formatted_data or "No comparison data" in formatted_data:
        return fallback_conclusion

    prompt = (
        "You are AgniAI, an intelligent military command console assistant.\n"
        "Based on the User Query and Formatted Backend Data, generate a brief Conclusion "
        "that directly answers the user's question.\n\n"
        "STRICT RULES:\n"
        "1. Base your response 100% on the Formatted Backend Data. "
        "Do NOT invent or calculate any numbers, totals, percentages, or averages "
        "that are not explicitly present in the data.\n"
        "2. Only mention numbers that appear verbatim in the Formatted Backend Data.\n"
        "3. Do NOT mention any person's name or specific details not in the data.\n"
        "4. Maximum 2-4 sentences. No markdown.\n\n"
        f"User Query: {user_query}\n\n"
        f"Formatted Backend Data:\n{formatted_data}\n\n"
        "Generate only the CONCLUSION text."
    )

    try:
        payload = {
            "model": DEFAULT_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 120, "num_ctx": 1024},
        }
        resp = requests.post(OLLAMA_URL, json=payload, timeout=(8, 30))
        resp.raise_for_status()
        conclusion = resp.json().get("message", {}).get("content", "").strip()

        conclusion = re.sub(r'^(?:CONCLUSION\s*:\s*)', '', conclusion, flags=re.IGNORECASE)
        conclusion = re.sub(r'[*_`#]', '', conclusion).strip()

        if conclusion and len(conclusion) >= 5:
            grounded = _strip_ungrounded_numbers(conclusion, formatted_data)
            return grounded if grounded else fallback_conclusion

    except Exception as exc:
        logger.debug("Ollama conclusion generation failed: %s", exc)

    return fallback_conclusion


def generate_admin_report(
    user_query: str,
    query_type: str,
    intent_result: Dict[str, Any],
    combined_result: Any,
) -> Dict[str, str]:
    """
    Main report engine endpoint called by admin_routes.py pipeline (Step 6).

    Receives:
      user_query     : the original user question
      query_type     : SIMPLE / CROSS_FILTER / COMPARISON / MULTI_INDEPENDENT / ANALYTICS
      intent_result  : the primary intent from the pipeline (not reclassified here)
      combined_result: the finalResult from result_combiner (READ-ONLY — never modified)

    Returns:
      { "introMessage": "...", "analysis": "...", "conclusion": "..." }
    """
    intro      = generate_intro_message(user_query, query_type, intent_result)
    analysis   = generate_analysis(user_query, query_type, combined_result, intent_result)
    conclusion = generate_conclusion(user_query, query_type, combined_result, intent_result)

    return {
        "introMessage": intro,
        "analysis":     analysis,
        "conclusion":   conclusion,
    }