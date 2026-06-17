"""
admin_report_generator.py
==========================
Dedicated reporting and analysis layer for the AgniAI Admin Chatbot.
Generates:
1. Intro Messages
2. Data Analysis (cautious, data-backed)
3. Executive Conclusions
"""

from __future__ import annotations

import re
import logging
import requests
from typing import Any, Dict, Optional, Tuple

from config import OLLAMA_URL, DEFAULT_MODEL
from admin_intent import classify_admin_intent
from admin_formatter import format_dotnet_response

logger = logging.getLogger(__name__)

# =============================================================================
# INTRO TEMPLATES
# =============================================================================

_INTRO_TEMPLATES: Dict[Tuple[str, str], str] = {
    # Performance
    ("Performance", "TopPerformers"):      "These assessment results highlight the strongest performers in the evaluation.",
    ("Performance", "LowestPerformers"):   "These results identify the individuals requiring additional training support.",
    ("Performance", "AverageScore"):       "The average scores outline overall achievement levels across the group.",
    ("Performance", "PassPercentage"):     "Pass rates reflect the percentage of trainees meeting the assessment standards.",
    ("Performance", "FailPercentage"):     "Fail rates identify the proportion of trainees currently below standard.",
    ("Performance", "GradeFilter"):        "The grade filter results show performance by the selected grading category.",
    ("Performance", "GradingSummary"):     "The grading summary provides a breakdown of performance achievements.",
    ("Performance", "OverallPerformers"):  "Overall performance metrics highlight trainee progress across all criteria.",
    ("Performance", "Improvement"):        "These records highlight the trainees showing positive performance growth.",
    ("Performance", "Drop"):               "These trends identify trainees experiencing a decline in assessment scores.",
    ("Performance", "SectionSummary"):     "The section summary provides a clear view of performance across individual modules.",
    ("Performance", "AttemptWise"):        "Attempt-wise statistics track trainee progress across successive evaluation cycles.",
    ("Performance", "BestAttempt"):        "Best attempt outcomes reflect peak trainee achievements in this evaluation.",
    ("Performance", "Comparison"):         "This comparison highlights achievement differences across the selected categories.",
    # Leave
    ("Leave", "MostLeaveTaken"):           "Leave patterns highlight the person with the highest absence rate.",
    ("Leave", "LeastLeaveTaken"):          "Leave summaries identify the person with the highest duty presence.",
    ("Leave", "CurrentLeave"):             "Current leave records outline person availability across the unit.",
    ("Leave", "AbscondedLeave"):           "These records flag persons currently absent without official leave.",
    # Medical
    ("Medical", "ActiveCases"):            "This summary captures current active cases undergoing medical attention.",
    ("Medical", "BMIAnalysis"):            "BMI records outline fitness levels and weight distribution across persons.",
    ("Medical", "DiseaseStats"):           "Health records highlight the most common medical cases reported recently.",
    # Attendance
    ("Attendance", "MonthlyAttendance"):   "Monthly attendance trends provide a clear view of person participation.",
    ("Attendance", "PresentToday"):        "Today's attendance records outline current person presence on campus.",
    ("Attendance", "StrengthBreakdown"):   "The strength breakdown captures unit headcount and active person counts.",
    # Verification
    ("Verification", "PendingVerification"):   "Verification files track documents currently awaiting official review.",
    ("Verification", "CompletedVerification"): "These records confirm files that have cleared the verification process.",
    # Equipment
    ("Equipment", "EquipmentStats"):           "This inventory summary reflects current equipment counts and status.",
    ("Equipment", "OverdueEquipment"):         "These records flag issued gear currently overdue for return.",
    ("Equipment", "ReturnedEquipment"):        "This quality review highlights equipment returned in sub-standard condition.",
    ("Equipment", "IssuedItems"):              "Here is the complete list of items issued to Agniveers.",
    ("Equipment", "ProcuredItems"):            "Here is the complete list of items procured by Agniveers.",
    # Distribution
    ("Distribution", "LatestDistribution"):    "Recent distribution logs track the latest issue of supplies and gear.",
    ("Distribution", "DistributionByUnit"):    "Distribution logs trace supply allocation across different units.",
    ("Distribution", "UnassignedItems"):       "Supply records outline items currently unassigned to any unit.",
    ("Distribution", "TopUnit"):               "This summary highlights the unit receiving the largest supply allocation.",
    # Skills
    ("Skills", "BySport"):                     "Sport rosters track athletic participation and team assignments.",
    ("Skills", "ByClass"):                     "Class rosters group persons by their administrative designations.",
    ("Skills", "BloodGroup"):                  "Medical profiles outline the blood group distribution across the group.",
}


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
    if category:
        context_parts.append(f"Module: {category}")
    if subcategory:
        context_parts.append(f"Query type: {subcategory}")
    if number:
        context_parts.append(f"Requested count: {number}")
    if section:
        context_parts.append(f"Section filter: {section}")
    if leave_type:
        context_parts.append(f"Leave type: {leave_type}")
    if grading:
        context_parts.append(f"Grading filter: {grading}")
    if unit_name:
        context_parts.append(f"Unit filter: {unit_name}")
    if sport:
        context_parts.append(f"Sport filter: {sport}")
    if class_name:
        context_parts.append(f"Class filter: {class_name}")

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
        "7. Do not ask questions.\n"
        "8. No markdown, no bullets, no quotes.\n"
        "9. Sound like a professional assistant introducing a report.\n\n"
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

    if re.search(r"\b\d+\b", text):
        return ""

    words = text.split()
    for word in words[1:]:
        clean_word = re.sub(r"[^A-Za-z]", "", word)
        if clean_word and clean_word[0].isupper() and clean_word.lower() not in {
            "agniveer", "bpet", "ppt", "drill", "firing", "medical",
            "attendance", "leave", "equipment", "performance", "verification",
            "distribution", "skills", "unit", "platoon", "batch",
        }:
            return ""

    meta_check = re.compile(
        r"^(?:here(?:'s| is)|i(?:'ve| have)|based on|the following|as requested)",
        re.IGNORECASE,
    )
    if meta_check.match(text):
        return ""

    return text


def _get_formatted_data(user_query: str, combined_result: Any) -> str:
    intent_result = classify_admin_intent(user_query)
    return format_dotnet_response(combined_result, intent_result)


# =============================================================================
# PUBLIC INTERFACE
# =============================================================================

def generate_intro_message(
    user_query: str,
    query_type: str,
    intent_result: Dict[str, Any]
) -> str:
    """
    Generate a 1-3 sentence introduction detailing what was requested.
    """
    category = intent_result.get("category", "")
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
        raw_text = resp.json().get("message", {}).get("content", "").strip()
        clean_text = _sanitize_intro(raw_text)

        if clean_text and 10 <= len(clean_text) <= 200:
            logger.debug("LLM intro (sanitized): %s", clean_text)
            return clean_text

        logger.debug("LLM intro rejected after sanitize (raw=%r clean=%r), using template", raw_text, clean_text)
    except Exception as exc:
        logger.debug("Ollama intro generation failed, using template fallback: %s", exc)

    key = (category, subcategory)
    if key in _INTRO_TEMPLATES:
        return _INTRO_TEMPLATES[key]

    category_label = category or "requested"
    return f"These records outline the current {category_label.lower()} status across the unit."


def generate_analysis(
    user_query: str,
    query_type: str,
    combined_result: Any
) -> str:
    """
    Analyze returned backend data according to the query type and query.
    Must be fully data-backed and use cautious, tentative language.
    """
    intent = classify_admin_intent(user_query)
    category = intent.get("category", "")
    category_label = category or "Agniveer data"

    fallback_analysis = (
        f"Analysis of the retrieved records indicates that the dataset contains active {category_label.lower()} indicators. "
        "The distribution pattern matches the requested parameters and no anomalies are highlighted."
    )

    formatted_data = format_dotnet_response(combined_result, intent)
    if not formatted_data or "No records found" in formatted_data or "No comparison data" in formatted_data:
        return fallback_analysis

    qt_upper = str(query_type).upper()
    if "SIMPLE" in qt_upper:
        type_instruction = "Analyze the returned records."
    elif "CROSS_FILTER" in qt_upper or "CROSS-FILTER" in qt_upper:
        type_instruction = "Analyze the relationship between intersected groups."
    elif "COMPARISON" in qt_upper:
        type_instruction = "Analyze similarities and differences."
    elif "MULTI_INDEPENDENT" in qt_upper:
        type_instruction = "Analyze each section individually and provide overall observations."
    else:
        type_instruction = "Analyze the returned records."

    prompt = (
        "You are AgniAI, an intelligent military command console assistant.\n"
        "You are reviewing Agniveer training data in the command center. Based on the User Query and the Formatted Backend Data, "
        "generate a detailed Analysis.\n\n"
        "STRICT RULES:\n"
        f"1. Focus instruction: {type_instruction}\n"
        "2. Base your response 100% on the actual Formatted Backend Data provided. Do NOT invent or make up any names, ranks, IDs, scores, counts, stats, or rankings.\n"
        "3. Analyze actual returned data, and mention trends, patterns, strengths, weaknesses, concentrations, or anomalies where present.\n"
        "4. Do NOT mention any person's name or specific details not present in the data.\n"
        "5. Do NOT contradict the backend data in any way.\n"
        "6. Use cautious/tentative language such as 'indicates', 'suggests', 'appears', or 'may reflect'.\n"
        "7. Keep the Analysis focused and concise (1-3 sentences).\n"
        "8. Do NOT include any 'CONCLUSION' section, recommendations, or executive summaries. Generate only the ANALYSIS text.\n"
        "9. No markdown formatting (no asterisks, no headers, no bold text).\n\n"
        f"User Query: {user_query}\n\n"
        f"Formatted Backend Data:\n{formatted_data}\n\n"
        "Generate only the ANALYSIS text."
    )

    try:
        payload = {
            "model": DEFAULT_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 120,
                "num_ctx": 1024,
            },
        }
        resp = requests.post(OLLAMA_URL, json=payload, timeout=(8, 30))
        resp.raise_for_status()
        analysis = resp.json().get("message", {}).get("content", "").strip()

        # Clean prefix and markdown
        analysis = re.sub(r'^(?:ANALYSIS\s*:\s*)', '', analysis, flags=re.IGNORECASE)
        analysis = re.sub(r'[*_`#]', '', analysis).strip()

        if analysis and len(analysis) >= 5:
            return analysis
    except Exception as exc:
        logger.debug("Ollama analysis generation failed: %s", exc)

    return fallback_analysis


def generate_conclusion(
    user_query: str,
    query_type: str,
    combined_result: Any
) -> str:
    """
    Generate an executive summary conclusion of 2-4 sentences grounded entirely in data.
    """
    intent = classify_admin_intent(user_query)
    category = intent.get("category", "")
    category_label = category or "Agniveer data"

    fallback_conclusion = (
        f"The current {category_label.lower()} status remains stable, and no immediate actions are required."
    )

    formatted_data = format_dotnet_response(combined_result, intent)
    if not formatted_data or "No records found" in formatted_data or "No comparison data" in formatted_data:
        return fallback_conclusion

    prompt = (
        "You are AgniAI, an intelligent military command console assistant.\n"
        "You are reviewing Agniveer training data in the command center. Based on the User Query and the Formatted Backend Data, "
        "generate a brief Conclusion/Executive Summary.\n\n"
        "STRICT RULES:\n"
        "1. Base your response 100% on the actual Formatted Backend Data provided. Do NOT invent or make up any names, ranks, IDs, scores, counts, or stats.\n"
        "2. Do NOT mention any person's name or specific details not present in the data.\n"
        "3. Do NOT contradict the backend data in any way.\n"
        "4. Maximum of 2-4 sentences.\n"
        "5. The conclusion must be grounded entirely in backend data, representing a stable military command summary.\n"
        "6. Do NOT include any 'ANALYSIS' or recommendations. Generate only the CONCLUSION text.\n"
        "7. No markdown formatting (no asterisks, no headers, no bold text).\n\n"
        f"User Query: {user_query}\n\n"
        f"Formatted Backend Data:\n{formatted_data}\n\n"
        "Generate only the CONCLUSION text."
    )

    try:
        payload = {
            "model": DEFAULT_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 120,
                "num_ctx": 1024,
            },
        }
        resp = requests.post(OLLAMA_URL, json=payload, timeout=(8, 30))
        resp.raise_for_status()
        conclusion = resp.json().get("message", {}).get("content", "").strip()

        # Clean prefix and markdown
        conclusion = re.sub(r'^(?:CONCLUSION\s*:\s*)', '', conclusion, flags=re.IGNORECASE)
        conclusion = re.sub(r'[*_`#]', '', conclusion).strip()

        if conclusion and len(conclusion) >= 5:
            return conclusion
    except Exception as exc:
        logger.debug("Ollama conclusion generation failed: %s", exc)

    return fallback_conclusion


def generate_admin_report(
    user_query: str,
    query_type: str,
    intent_result: Dict[str, Any],
    combined_result: Any
) -> Dict[str, str]:
    """
    Main reporting engine endpoint.
    """
    intro = generate_intro_message(user_query, query_type, intent_result)
    analysis = generate_analysis(user_query, query_type, combined_result)
    conclusion = generate_conclusion(user_query, query_type, combined_result)

    return {
        "introMessage": intro,
        "analysis": analysis,
        "conclusion": conclusion,
    }
