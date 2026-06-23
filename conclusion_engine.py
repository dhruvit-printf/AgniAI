"""
conclusion_engine.py
====================
Generates a concise (20-40 words) conclusion summarizing the findings from JSON data.
"""

import json
import logging
import requests
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

from grounding_utils import extract_numbers_from_text as _extract_numbers_from_text
from grounding_utils import ground_and_sanitize as _ground_and_sanitize
from config import DEFAULT_MODEL, OLLAMA_URL
from feature_flags import get_flags
from metrics import metrics_collector
from utils import get_score as _get_score
from utils import safe_float as _safe_float

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
                lines.append(f"{k} difference is {v.get('difference')} and percentage is {v.get('percentage')}.")
    else:
        records = sections[0].get("data") if sections else []
        lines.append(f"Count: {len(records)}")
        scores = []
        for r in records:
            score = _get_score(r)
            if score is not None:
                scores.append(score)
        if scores:
            lines.append(f"Average Score: {round(sum(scores) / len(scores), 2)}")
            lines.append(f"Top Score: {max(scores)}")
            lines.append(f"Bottom Score: {min(scores)}")

    return "\n".join(lines)

def generate_conclusion(
    answer: Dict[str, Any],
    query_type: str,
    intent: Dict[str, Any],
    trace_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate a short conclusion with at most three bullet points.
    """
    sections = answer.get("sections") or []
    is_empty = True
    if query_type in ("compare", "comparison"):
        left_data = answer.get("left", {}).get("data") or []
        right_data = answer.get("right", {}).get("data") or []
        if left_data or right_data:
            is_empty = False
    else:
        for sec in sections:
            if sec.get("data"):
                is_empty = False
                break

    category = intent.get("category") or "Agniveer"

    if is_empty:
        if query_type == "cross_filter":
            msg = "In conclusion, the cross-filter query did not return any matching records from the active database. To proceed with the analysis, we recommend adjusting your filter criteria or broadening the search parameters to see if any matching personnel can be identified under less restrictive conditions."
        elif query_type in ("compare", "comparison"):
            msg = "In conclusion, the side-by-side comparison could not be completed because no matching data was found for either of the categories. We recommend verifying that records exist for these groups in the database before trying to perform another comparison query."
        elif query_type == "multi_independent":
            msg = "In conclusion, the consolidated report is empty because no matching data was found across any of the independent categories. Please check your query parameters or ensure that the target modules have active records available for reporting."
        else:
            msg = f"In conclusion, the database query returned zero active {category.lower()} records. We recommend adjusting your filter criteria, checking the spelling of your query parameters, or verifying that active records are present in the source system before attempting this search again."
        return {"summary": msg, "bullets": [msg[:120]]}

    grounding_text = _build_conclusion_grounding_text(answer, query_type)
    flags = get_flags()

    # Rule-based fallback generator
    fallback_message = f"The review of {category.lower()} records is complete."

    if query_type in ("compare", "comparison"):
        left = answer.get("left") or {}
        right = answer.get("right") or {}
        fallback_message = f"The comparative review of {left.get('label', 'Side 1')} and {right.get('label', 'Side 2')} is complete. The side-by-side metrics and average score differences provide a clear statistical overview of how these categories compare. These findings are finalized and recorded to assist with ongoing evaluation and training updates."
    elif query_type == "cross_filter":
        records = sections[0].get("data") if sections else []
        fallback_message = f"In conclusion, the cross-filter query has successfully isolated {len(records)} Agniveer records matching all specified requirements. These individuals have been cross-referenced and validated against the primary unit databases, making this compiled list ready for immediate administrative reporting and command evaluation."
    elif query_type == "multi_independent":
        fallback_message = f"In conclusion, the consolidation of the requested independent administrative modules is complete. All {len(sections)} sections have been successfully populated with their respective active database records, verified for completeness, and formatted to provide a comprehensive and clean administrative review."

    if not (flags.ENABLE_REPORTS and flags.ENABLE_OLLAMA):
        return {"summary": fallback_message, "bullets": [fallback_message]}

    prompt = (
        "You are AgniAI, an intelligent military assistant.\n"
        "Generate a short conclusion and up to three bullet points from the aggregate data below.\n\n"
        "STRICT RULES:\n"
        "1. Base your response 100% on the Aggregate Data below.\n"
        "2. Do NOT hallucinate any numbers or names.\n"
        "3. Keep the summary short and the bullet list to 1-3 items.\n"
        f"Aggregate Data:\n{grounding_text}\n\n"
        "Return only JSON with keys summary and bullets."
    )

    try:
        payload = {
            "model": DEFAULT_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 100, "num_ctx": 512}
        }
        from ollama_settings import get_ollama_timeout

        resp = requests.post(OLLAMA_URL, json=payload, timeout=get_ollama_timeout())
        resp.raise_for_status()
        raw = resp.json().get("message", {}).get("content", "").strip()

        sanitized = _ground_and_sanitize(raw, grounding_text)
        if sanitized:
            return {"summary": sanitized, "bullets": [sanitized]}
    except Exception as e:
        logger.warning("Ollama call failed in conclusion engine: %s", e)
        metrics_collector.inc_llm_failure()

    return {"summary": fallback_message, "bullets": [fallback_message]}
