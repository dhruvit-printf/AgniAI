"""
analysis_engine.py
==================
Generates observations and insights from JSON answer data.
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

def _build_aggregate_text(answer: Dict[str, Any], query_type: str, intent: Dict[str, Any]) -> str:
    lines = []
    category = intent.get("category") or "Agniveer"
    sections = answer.get("sections") or []

    if query_type == "cross_filter":
        records = sections[0].get("data") if sections else []
        lines.append(f"Query Type: cross_filter")
        lines.append(f"Match Count: {len(records)}")
        names = [r.get("fullName") or r.get("name") for r in records if r.get("fullName") or r.get("name")]
        if names:
            lines.append(f"Matched Agniveers: {', '.join(names)}")
    elif query_type == "compare":
        left = answer.get("left") or {}
        right = answer.get("right") or {}
        comp = answer.get("comparison") or {}
        lines.append(f"Query Type: comparison")
        if left:
            lines.append(f"Side 1: {left.get('label')} - Count: {len(left.get('data', []))}")
            for k, v in left.get("metrics", {}).items():
                lines.append(f"  Side 1 {k}: {v}")
        if right:
            lines.append(f"Side 2: {right.get('label')} - Count: {len(right.get('data', []))}")
            for k, v in right.get("metrics", {}).items():
                lines.append(f"  Side 2 {k}: {v}")
        for k, v in comp.items():
            if isinstance(v, dict):
                lines.append(f"  Comparison {k}: difference={v.get('difference')}, percentage={v.get('percentage')}, higher={v.get('higher')}, lower={v.get('lower')}")
    elif query_type == "multi_independent":
        lines.append(f"Query Type: multi_independent")
        lines.append(f"Section Count: {len(sections)}")
        for sec in sections:
            lines.append(f"  Section: {sec.get('label')} - {len(sec.get('data', []))} records")
    else:
        # simple/trend/distribution
        records = sections[0].get("data") if sections else []
        lines.append(f"Query Type: simple")
        lines.append(f"Category: {category}")
        lines.append(f"Record Count: {len(records)}")
        scores = []
        for r in records:
            score = _get_score(r)
            if score is not None:
                scores.append(score)
        if scores:
            lines.append(f"Average Score: {round(sum(scores) / len(scores), 2)}")
            lines.append(f"Top Score: {max(scores)}")
            lines.append(f"Bottom Score: {min(scores)}")
        names = [r.get("fullName") or r.get("name") for r in records if r.get("fullName") or r.get("name")]
        if names:
            lines.append(f"Records: {', '.join(names[:20])}")
            if len(names) > 20:
                lines.append(f"...and {len(names) - 20} more")

    return "\n".join(lines)


def _analysis_payload(
    summary: str,
    observations: List[str],
    insights: List[str],
    predictions: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "summary": summary,
        "observations": observations,
        "insights": insights,
        "predictions": predictions or [],
    }

def generate_analysis(
    answer: Dict[str, Any],
    query_type: str,
    intent: Dict[str, Any],
    user_query: str = "",
    trace_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate observations and insights from JSON answer.
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
            return _analysis_payload(
                "A detailed cross-filter query was executed across multiple datasets to identify common personnel matching all filter parameters. The resulting intersection yielded zero common records, demonstrating that the overlapping conditions specified in the query filter out all available Agniveers in the system database.",
                [
                    "A cross-filter search was performed across all selected category sets, but no overlapping records were retrieved."
                ],
                [
                    "This suggests that the filter parameters are mutually exclusive for the current cohort of personnel."
                ],
            )
        elif query_type in ("compare", "comparison"):
            return _analysis_payload(
                "A side-by-side comparison was initiated to evaluate the metrics of the selected categories. However, since the query returned no records for any of the groups, a comparative breakdown cannot be compiled. There is no active data to compare across the selected dimensions.",
                [
                    "Both comparison groups returned empty datasets from the primary database query."
                ],
                [
                    "The lack of records indicates that either no data has been logged yet or the categories do not contain any active personnel."
                ],
            )
        elif query_type == "multi_independent":
            return _analysis_payload(
                "The multi-section consolidation process compiled results from all requested data modules. Unfortunately, none of the query paths returned any active records, meaning that all sections in this report are currently empty and there are no statistics available for analysis.",
                [
                    "All requested independent sections returned zero active records from the system."
                ],
                [
                    "This suggests a widespread absence of matching records across all requested data tables or filters."
                ],
            )
        else:
            return _analysis_payload(
                f"No matching data was found for the requested {category.lower()} query filter criteria. The database search completed successfully but returned an empty dataset, meaning there are no records available to display or analyze at this time.",
                [
                    f"The search returned 0 records matching the {category.lower()} query parameters."
                ],
                [
                    "No active records match the specified filters, indicating either no data has been logged or filters are too restrictive."
                ],
            )

    aggregate_text = _build_aggregate_text(answer, query_type, intent)
    flags = get_flags()

    # Rule-based fallback generator
    fallback_summary = f"Summary of {category.lower()} metrics completed."
    fallback_obs = [f"Retrieved data for {category.lower()} query."]
    fallback_ins = ["Dataset matches the specified parameters."]

    if query_type in ("compare", "comparison"):
        left = answer.get("left") or {}
        right = answer.get("right") or {}
        fallback_summary = f"Comparison completed between {left.get('label', 'Side 1')} and {right.get('label', 'Side 2')}."
        fallback_obs = [
            f"Compared {left.get('label', 'Side 1')} ({len(left.get('data', []))} records) with {right.get('label', 'Side 2')} ({len(right.get('data', []))} records)."
        ]
        fallback_ins = ["Comparison highlights metric variances across categories."]
    elif query_type == "cross_filter":
        records = sections[0].get("data") if sections else []
        fallback_summary = f"Cross-filter analysis matched {len(records)} records."
        fallback_obs = [f"Found {len(records)} common records satisfying all filters."]
        fallback_ins = ["Records satisfy multiple overlapping conditions."]
    elif query_type == "multi_independent":
        fallback_summary = f"Consolidated data from {len(sections)} sections."
        fallback_obs = [f"Merged {len(sections)} independent categories: {', '.join(s.get('label','') for s in sections)}."]
        fallback_ins = ["Sections are presented independently without correlation."]

    fallback_report = {
        "summary": fallback_summary,
        "observations": fallback_obs,
        "insights": fallback_ins
    }

    if not (flags.ENABLE_REPORTS and flags.ENABLE_OLLAMA):
        return fallback_report

    prompt = (
        "You are AgniAI, an intelligent military assistant.\n"
        "Generate a JSON object containing summary, observations, and insights based on the AGGREGATE data below.\n\n"
        "STRICT RULES:\n"
        "1. Base your response 100% on the Aggregate Data below. Never hallucinate, never invent details.\n"
        "2. Only mention numbers/metrics that appear verbatim in the Aggregate Data.\n"
        "3. Do NOT mention any person's name unless it appears in the Aggregate Data.\n"
        "4. Output a single JSON object with EXACTLY this structure:\n"
        "{\n"
        '  "summary": "A detailed, natural language summary of the aggregate metrics (around 50 words).",\n'
        '  "observations": ["1-3 key data points/metrics from the aggregate data"],\n'
        '  "insights": ["1-2 trends or insights based on the aggregate data"]\n'
        "}\n\n"
        f"User Query: {user_query}\n"
        f"Query Type: {query_type}\n"
        f"Aggregate Data:\n{aggregate_text}\n\n"
        "Generate only the raw JSON, no markdown formatting or extra text."
    )

    try:
        payload = {
            "model": DEFAULT_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": 250, "num_ctx": 1024}
        }
        from ollama_settings import get_ollama_timeout

        resp = requests.post(OLLAMA_URL, json=payload, timeout=get_ollama_timeout())
        resp.raise_for_status()
        raw = resp.json().get("message", {}).get("content", "").strip()

        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            parsed = json.loads(raw[start:end+1])
            if isinstance(parsed, dict):
                summary = _ground_and_sanitize(parsed.get("summary", ""), aggregate_text) or fallback_summary
                obs = [_ground_and_sanitize(str(o), aggregate_text) for o in parsed.get("observations", [])]
                obs = [o for o in obs if o] or fallback_obs
                ins = [_ground_and_sanitize(str(i), aggregate_text) for i in parsed.get("insights", [])]
                ins = [i for i in ins if i] or fallback_ins
                return _analysis_payload(summary, obs, ins)
    except Exception as e:
        logger.warning("Ollama call failed in analysis engine: %s", e)
        metrics_collector.inc_llm_failure()

    return _analysis_payload(
        fallback_report["summary"],
        fallback_report["observations"],
        fallback_report["insights"],
    )
