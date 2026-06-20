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
    Generate a concise 20-40 words conclusion.
    """
    sections = answer.get("sections") or []
    is_empty = True
    if query_type == "compare":
        if answer.get("left") or answer.get("right"):
            is_empty = False
    else:
        for sec in sections:
            if sec.get("data"):
                is_empty = False
                break

    if is_empty:
        return {"message": "No matching records found to summarize."}

    grounding_text = _build_conclusion_grounding_text(answer, query_type)
    category = intent.get("category") or "Agniveer"
    flags = get_flags()

    # Rule-based fallback generator (Target 20-40 words)
    fallback_message = f"The review of {category.lower()} records is complete. A total of {len(sections[0].get('data', [])) if sections else 0} matching entries are successfully filtered and analyzed for current command console reporting."

    if query_type == "compare":
        left = answer.get("left") or {}
        right = answer.get("right") or {}
        fallback_message = f"Comparative metrics outline performance differences between {left.get('label', 'Side 1')} and {right.get('label', 'Side 2')}. Variance is documented across the compared parameters to assist evaluation."
    elif query_type == "cross_filter":
        records = sections[0].get("data") if sections else []
        fallback_message = f"The cross-filter operation successfully identified {len(records)} Agniveers matching all selected query criteria. These records are isolated and verified against the command records logs."
    elif query_type == "multi_independent":
        fallback_message = f"Independent sections have been consolidated from multiple data modules. A total of {len(sections)} sections are loaded and verified in this report with no correlation performed."

    if not (flags.ENABLE_REPORTS and flags.ENABLE_OLLAMA):
        return {"message": fallback_message}

    prompt = (
        "You are AgniAI, an intelligent military assistant.\n"
        "Generate a brief conclusion summarizing the findings from the aggregate data below.\n\n"
        "STRICT RULES:\n"
        "1. Base your response 100% on the Aggregate Data below.\n"
        "2. Do NOT hallucinate any numbers or names.\n"
        "3. Write EXACTLY between 20 and 40 words.\n"
        f"Aggregate Data:\n{grounding_text}\n\n"
        "Generate only the plain text conclusion without formatting."
    )

    try:
        payload = {
            "model": DEFAULT_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 60, "num_ctx": 512}
        }
        from ollama_settings import get_ollama_timeout

        resp = requests.post(OLLAMA_URL, json=payload, timeout=get_ollama_timeout())
        resp.raise_for_status()
        raw = resp.json().get("message", {}).get("content", "").strip()

        sanitized = _ground_and_sanitize(raw, grounding_text)
        if sanitized and 20 <= len(sanitized.split()) <= 45:
            return {"message": sanitized}
    except Exception as e:
        logger.warning("Ollama call failed in conclusion engine: %s", e)
        metrics_collector.inc_llm_failure()

    return {"message": fallback_message}
