"""
message_engine.py
=================
Generates a natural language conversational response message using Ollama.

The message is the PRIMARY text the user reads — it must:
  - Directly answer their question
  - Reference the actual numbers/records returned
  - Sound like a knowledgeable aide, not a query engine
  - Never be a static template

Falls back to a data-grounded static message if Ollama is unavailable or slow.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Score field candidates (same as utils.get_score)
# ---------------------------------------------------------------------------
_SCORE_FIELDS = (
    "bestTotal", "totalMarks", "score", "Score",
    "omrInputTotal", "marksObtained", "averageScore",
    "percentage", "value", "count",
)


def _safe_float(val: Any) -> Optional[float]:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _extract_flat_records(data: Any) -> List[Dict[str, Any]]:
    """Pull records from any combined_result shape."""
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]
    if not isinstance(data, dict):
        return []
    # Simple / cross_filter
    for key in ("data", "Data", "records", "Records"):
        val = data.get(key)
        if isinstance(val, list) and val:
            return [r for r in val if isinstance(r, dict)]
    # Multi_independent
    sections = data.get("sections") or []
    if sections:
        out = []
        for sec in sections:
            if isinstance(sec, dict):
                out.extend([r for r in (sec.get("data") or []) if isinstance(r, dict)])
        if out:
            return out
    # Compare
    left = (data.get("left") or {}).get("data") or []
    right = (data.get("right") or {}).get("data") or []
    if left or right:
        return [r for r in (left + right) if isinstance(r, dict)]
    return []


def _get_score(record: Dict[str, Any]) -> Optional[float]:
    for field in _SCORE_FIELDS:
        val = _safe_float(record.get(field))
        if val is not None:
            return val
    return None


def _build_data_summary(
    combined_result: Any,
    query_type: str,
    intent: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a compact, number-grounded summary dict for the LLM prompt.
    Only real numbers from the data — no invented values.
    """
    category = intent.get("category") or "Agniveer"
    subcategory = intent.get("subcategory") or ""

    if query_type in ("compare", "comparison"):
        left = (combined_result or {}).get("left") or {} if isinstance(combined_result, dict) else {}
        right = (combined_result or {}).get("right") or {} if isinstance(combined_result, dict) else {}
        comp = (combined_result or {}).get("comparison") or {} if isinstance(combined_result, dict) else {}
        left_data = left.get("data") or []
        right_data = right.get("data") or []
        left_label = left.get("label") or "Side A"
        right_label = right.get("label") or "Side B"

        left_scores = [s for s in (_get_score(r) for r in left_data if isinstance(r, dict)) if s is not None]
        right_scores = [s for s in (_get_score(r) for r in right_data if isinstance(r, dict)) if s is not None]

        summary = {
            "type": "comparison",
            "left_label": left_label,
            "right_label": right_label,
            "left_count": len(left_data),
            "right_count": len(right_data),
        }
        if left_scores:
            summary["left_avg"] = round(sum(left_scores) / len(left_scores), 2)
        if right_scores:
            summary["right_avg"] = round(sum(right_scores) / len(right_scores), 2)
        # Add first metric diff if available
        if comp:
            first_metric_name = next(iter(comp), None)
            if first_metric_name:
                m = comp[first_metric_name]
                summary["metric_name"] = first_metric_name
                summary["higher"] = m.get("higher")
                summary["lower"] = m.get("lower")
                summary["difference"] = m.get("difference")
        return summary

    if query_type == "cross_filter":
        records = _extract_flat_records(combined_result)
        match_count = (combined_result or {}).get("matchCount", len(records)) if isinstance(combined_result, dict) else len(records)
        total_before = (combined_result or {}).get("totalBeforeFilter", 0) if isinstance(combined_result, dict) else 0
        names = [r.get("fullName") or r.get("name") for r in records[:5] if isinstance(r, dict)]
        names = [n for n in names if n]
        return {
            "type": "cross_filter",
            "match_count": match_count,
            "total_before": total_before,
            "sample_names": names,
        }

    if query_type == "multi_independent":
        sections = (combined_result or {}).get("sections") or [] if isinstance(combined_result, dict) else []
        section_info = []
        for sec in sections:
            if isinstance(sec, dict):
                label = sec.get("label") or "Section"
                count = len(sec.get("data") or [])
                section_info.append({"label": label, "count": count})
        return {
            "type": "multi_independent",
            "sections": section_info,
            "section_count": len(section_info),
        }

    # simple / trend / distribution
    records = _extract_flat_records(combined_result)
    scores = [s for s in (_get_score(r) for r in records if isinstance(r, dict)) if s is not None]

    summary: Dict[str, Any] = {
        "type": "simple",
        "category": category,
        "subcategory": subcategory,
        "record_count": len(records),
    }

    if scores:
        summary["avg_score"] = round(sum(scores) / len(scores), 2)
        summary["max_score"] = round(max(scores), 2)
        summary["min_score"] = round(min(scores), 2)

    # For AverageScore queries — the record itself IS the summary
    if records and len(records) <= 10:
        # Extract any numeric fields from the first record for direct reference
        first = records[0]
        numeric_fields = {k: v for k, v in first.items() if isinstance(v, (int, float)) and k != "id"}
        text_fields = {k: v for k, v in first.items() if isinstance(v, str) and v and k not in ("id", "commandLabel")}
        summary["record_fields"] = {**text_fields, **numeric_fields}
        # All records if small set
        if len(records) > 1:
            summary["all_records"] = [
                {k: v for k, v in r.items() if not isinstance(v, (dict, list))}
                for r in records[:20]
            ]

    # Top names for ranking queries
    if records and len(records) > 1:
        names = []
        for r in records[:5]:
            name = r.get("fullName") or r.get("name") or r.get("agniveerNo")
            score = _get_score(r)
            if name:
                names.append(f"{name}" + (f" ({score})" if score is not None else ""))
        if names:
            summary["top_names"] = names

    # Section filter
    section = intent.get("section") or intent.get("sub_section")
    if section:
        summary["section_filter"] = section

    return summary


def _build_llm_prompt(
    user_query: str,
    data_summary: Dict[str, Any],
    query_type: str,
    intent: Dict[str, Any],
) -> str:
    """
    Build the Ollama prompt. Keeps it short and data-grounded.
    The LLM must only reference numbers explicitly given.
    """
    category = intent.get("category") or "Agniveer"
    query_type_label = {
        "simple": "data lookup",
        "compare": "comparison",
        "cross_filter": "cross-filter search",
        "multi_independent": "multi-section report",
        "trend": "trend analysis",
        "distribution": "distribution breakdown",
    }.get(query_type, "query")

    # Serialise data summary compactly
    import json
    data_str = json.dumps(data_summary, ensure_ascii=False, default=str)

    prompt = f"""You are AgniAI — an intelligent military training management assistant for the Indian Army Agniveer program.
A commanding officer asked: "{user_query}"

The system retrieved this data:
{data_str}

Write a natural, conversational response (2-4 sentences) that:
1. Directly answers what the officer asked
2. References the EXACT numbers from the data above (do not invent any numbers)
3. Sounds like a knowledgeable aide briefing a senior officer — professional but clear
4. If record_count is 0, say no records were found and suggest broadening the criteria

Rules:
- Use only numbers present in the data above
- Do not use markdown, bullets, or headers
- Do not say "I retrieved" or "the system found" — just answer directly
- Keep it under 80 words"""

    return prompt


def _static_fallback_message(
    user_query: str,
    data_summary: Dict[str, Any],
    query_type: str,
    intent: Dict[str, Any],
) -> str:
    """
    Data-grounded static message when Ollama is unavailable.
    References actual numbers — never a generic template.
    """
    category = (intent.get("category") or "Agniveer").lower()
    qtype = data_summary.get("type", query_type)

    if qtype == "comparison":
        left = data_summary.get("left_label", "Side A")
        right = data_summary.get("right_label", "Side B")
        lc = data_summary.get("left_count", 0)
        rc = data_summary.get("right_count", 0)
        la = data_summary.get("left_avg")
        ra = data_summary.get("right_avg")
        msg = f"Comparison between {left} ({lc} records) and {right} ({rc} records) is complete."
        if la is not None and ra is not None:
            higher = left if la > ra else right
            msg += f" {higher} has a higher average score ({max(la, ra)} vs {min(la, ra)})."
        return msg

    if qtype == "cross_filter":
        match = data_summary.get("match_count", 0)
        total = data_summary.get("total_before", 0)
        if match == 0:
            return f"No records matched all the selected conditions. Try broadening the search criteria."
        names = data_summary.get("sample_names", [])
        msg = f"Found {match} records matching all conditions"
        if total:
            msg += f" out of {total} candidates"
        msg += "."
        if names:
            msg += f" Includes: {', '.join(names[:3])}" + ("..." if len(names) > 3 else ".")
        return msg

    if qtype == "multi_independent":
        sections = data_summary.get("sections", [])
        if not sections:
            return "The consolidated report is ready."
        parts = [f"{s['label']} ({s['count']} records)" for s in sections[:4]]
        return f"Consolidated report across {len(sections)} sections: {', '.join(parts)}."

    # simple
    count = data_summary.get("record_count", 0)
    if count == 0:
        return f"No matching {category} records were found. Try adjusting the search criteria."

    # If it's a summary-type record (averageScore, etc.)
    fields = data_summary.get("record_fields", {})
    section_filter = data_summary.get("section_filter", "")

    # AverageScore / stats type
    if "averageScore" in fields or "sectionName" in fields:
        section_name = fields.get("sectionName", section_filter or category.title())
        avg = fields.get("averageScore")
        min_s = fields.get("minScore")
        max_s = fields.get("maxScore")
        total_r = fields.get("totalRecords", count)
        max_p = fields.get("maxPossible")

        parts = [f"In {section_name}"]
        if total_r:
            parts[0] += f", across {total_r} agniveers"
        if avg is not None:
            parts.append(f"the average score is {avg}")
        if max_p is not None:
            parts[-1] += f" out of {max_p}"
        if min_s is not None and max_s is not None:
            parts.append(f"ranging from {min_s} to {max_s}")
        return ", ".join(parts) + "."

    # Ranking type
    avg = data_summary.get("avg_score")
    max_s = data_summary.get("max_score")
    top_names = data_summary.get("top_names", [])
    section_label = f" in {section_filter}" if section_filter else ""

    if top_names:
        msg = f"Found {count} {category} records{section_label}. Top performers: {', '.join(top_names[:3])}"
        msg += "..." if count > 3 else "."
        if avg is not None:
            msg += f" Group average: {avg}."
        return msg

    msg = f"Retrieved {count} {category} records{section_label}."
    if avg is not None:
        msg += f" Average score: {avg}."
    if max_s is not None:
        msg += f" Highest: {max_s}."
    return msg


def generate_message(
    user_query: str,
    combined_result: Any,
    query_type: str,
    intent: Dict[str, Any],
    trace_id: Optional[str] = None,
) -> str:
    """
    Generate a natural language response message.

    Priority:
      1. Ollama LLM — data-grounded prompt → natural response
      2. Static fallback — references real numbers, no generic templates

    Never raises — always returns a usable string.
    """
    try:
        data_summary = _build_data_summary(combined_result, query_type, intent)
    except Exception as exc:
        logger.warning("message_engine: _build_data_summary failed: %s", exc)
        data_summary = {"type": query_type, "record_count": 0}

    # ── Try Ollama ──────────────────────────────────────────────────────────
    try:
        import requests as _req
        from config import DEFAULT_MODEL, OLLAMA_URL

        prompt = _build_llm_prompt(user_query, data_summary, query_type, intent)
        payload = {
            "model": DEFAULT_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": 0.3,      # Low temp — factual, not creative
                "num_predict": 150,
                "num_ctx": 1024,
            },
        }
        resp = _req.post(OLLAMA_URL, json=payload, timeout=(5, 20))
        resp.raise_for_status()
        llm_text = resp.json().get("message", {}).get("content", "").strip().strip("\"'")

        if llm_text and 10 <= len(llm_text) <= 500:
            # Sanity check: reject if LLM invented numbers not in data_summary
            llm_clean = _validate_no_invented_numbers(llm_text, data_summary)
            if llm_clean:
                logger.debug(
                    "message_engine: LLM message generated (trace=%s, len=%d)",
                    trace_id or "N/A", len(llm_clean)
                )
                return llm_clean

    except Exception as exc:
        logger.debug("message_engine: Ollama unavailable, using static fallback: %s", exc)

    # ── Static fallback ──────────────────────────────────────────────────────
    msg = _static_fallback_message(user_query, data_summary, query_type, intent)
    logger.debug("message_engine: static fallback message (trace=%s)", trace_id or "N/A")
    return msg


def _validate_no_invented_numbers(text: str, data_summary: Dict[str, Any]) -> Optional[str]:
    """
    Check that numbers in the LLM response are grounded in data_summary.
    If LLM invents a number not in the data, return None (use fallback instead).
    Returns the original text if all numbers check out.
    """
    import json

    # Collect all numbers present in data_summary
    data_str = json.dumps(data_summary, default=str)
    data_numbers = set(re.findall(r'\b\d+(?:\.\d+)?\b', data_str))

    # Find numbers in LLM response
    llm_numbers = set(re.findall(r'\b\d+(?:\.\d+)?\b', text))

    # Allow small integers (1-10) and years as they're likely contextual
    invented = {
        n for n in llm_numbers
        if n not in data_numbers
        and float(n) > 10
        and not (2000 <= float(n) <= 2030)
    }

    if invented:
        logger.warning(
            "message_engine: LLM invented numbers %s not in data — using fallback", invented
        )
        return None

    return text