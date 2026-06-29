"""
message_engine.py
=================
Generates a natural language conversational response message using Ollama.

The message is the PRIMARY text the user reads — it must:
  - Directly answer their question in natural military-aide language
  - Reference the actual numbers/records returned (zero invented values)
  - Sound like a knowledgeable aide, not a query engine or template renderer
  - Gracefully fall back to a data-grounded static message if Ollama is unavailable

Architecture:
  generate_message()
    ├─ _build_data_summary()    → compact, number-grounded dict from combined_result
    ├─ _build_llm_prompt()      → tightly scoped Ollama prompt
    ├─ [Ollama call]            → natural language response
    ├─ _validate_llm_response() → reject hallucinated numbers
    └─ _static_fallback()       → structured but warm fallback prose
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from utils import extract_records as _extract_records

# ---------------------------------------------------------------------------
# Score field candidates (priority order)
# ---------------------------------------------------------------------------
_SCORE_FIELDS = (
    "bestTotal",
    "totalMarks",
    "score",
    "Score",
    "omrInputTotal",
    "marksObtained",
    "averageScore",
    "percentage",
    "value",
    "count",
)

# Fields that carry identity / label info for natural references
_NAME_FIELDS = ("fullName", "name", "agniveerName", "recruiterName")
_ID_FIELDS = ("agniveerNo", "agniveerNumber", "enrollmentNo", "batchId", "platoonId")
_UNIT_FIELDS = ("platoon", "platoonName", "batch", "batchName", "company", "unit", "commandLabel")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(val: Any) -> Optional[float]:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _get_score(record: Dict[str, Any]) -> Optional[float]:
    for field in _SCORE_FIELDS:
        val = _safe_float(record.get(field))
        if val is not None:
            return val
    return None


def _get_label(record: Dict[str, Any]) -> Optional[str]:
    """Extract best human-readable label from a record."""
    for f in _NAME_FIELDS:
        if record.get(f):
            return str(record[f])
    for f in _ID_FIELDS:
        if record.get(f):
            return str(record[f])
    return None


def _get_unit(record: Dict[str, Any]) -> Optional[str]:
    for f in _UNIT_FIELDS:
        if record.get(f):
            return str(record[f])
    return None


def _intent_context(intent: Dict[str, Any]) -> Dict[str, str]:
    """Extract human-readable context strings from intent."""
    ctx: Dict[str, str] = {}

    category = intent.get("category") or ""
    subcategory = intent.get("subcategory") or ""
    section = intent.get("section") or intent.get("sub_section") or ""

    if category:
        ctx["module"] = category
    if subcategory:
        ctx["sub_module"] = subcategory
    if section:
        ctx["section"] = section

    # Filters — batch / platoon / agniveer
    batch_id = intent.get("batch_id") or intent.get("batchId")
    platoon_id = intent.get("platoon_id") or intent.get("platoonId")
    agniveer_no = intent.get("agniveer_no") or intent.get("agniveerNo")

    if batch_id:
        ctx["batch"] = str(batch_id)
    if platoon_id:
        ctx["platoon"] = str(platoon_id)
    if agniveer_no:
        ctx["agniveer"] = str(agniveer_no)

    # Date / period filters
    for key in ("from_date", "to_date", "month", "year", "period", "fromDate", "toDate"):
        val = intent.get(key)
        if val:
            ctx[key] = str(val)

    return ctx


# ---------------------------------------------------------------------------
# Data summary builder
# ---------------------------------------------------------------------------

def _build_data_summary(
    combined_result: Any,
    query_type: str,
    intent: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a compact, number-grounded summary dict for the LLM prompt.
    Only real values from the retrieved data — nothing invented.
    """
    ctx = _intent_context(intent)

    # ── Comparison ───────────────────────────────────────────────────────────
    if query_type in ("compare", "comparison"):
        result = combined_result if isinstance(combined_result, dict) else {}
        left = result.get("left") or {}
        right = result.get("right") or {}
        comp_metrics = result.get("comparison") or {}

        left_data = left.get("data") or []
        right_data = right.get("data") or []
        left_label = left.get("label") or "Group A"
        right_label = right.get("label") or "Group B"

        left_scores = [s for s in (_get_score(r) for r in left_data if isinstance(r, dict)) if s is not None]
        right_scores = [s for s in (_get_score(r) for r in right_data if isinstance(r, dict)) if s is not None]

        summary: Dict[str, Any] = {
            "type": "comparison",
            "left_label": left_label,
            "right_label": right_label,
            "left_count": len(left_data),
            "right_count": len(right_data),
        }

        if left_scores:
            summary["left_avg"] = round(sum(left_scores) / len(left_scores), 2)
            summary["left_max"] = round(max(left_scores), 2)
        if right_scores:
            summary["right_avg"] = round(sum(right_scores) / len(right_scores), 2)
            summary["right_max"] = round(max(right_scores), 2)

        # Summarise top 2 comparison metrics
        metrics_summary: List[Dict[str, Any]] = []
        for metric_name, m in list(comp_metrics.items())[:2]:
            if isinstance(m, dict):
                entry: Dict[str, Any] = {"metric": metric_name}
                if m.get("higher"):
                    entry["higher"] = m["higher"]
                if m.get("lower"):
                    entry["lower"] = m["lower"]
                diff = _safe_float(m.get("difference"))
                if diff is not None:
                    entry["difference"] = round(diff, 2)
                metrics_summary.append(entry)
        if metrics_summary:
            summary["metrics"] = metrics_summary

        if ctx:
            summary["context"] = ctx
        return summary

    # ── Cross-filter ─────────────────────────────────────────────────────────
    if query_type == "cross_filter":
        records = _extract_records(combined_result)
        result = combined_result if isinstance(combined_result, dict) else {}
        match_count = result.get("matchCount", len(records))
        total_before = result.get("totalBeforeFilter", 0)
        filter_desc = result.get("filterDescription") or ""

        sample: List[Dict[str, Any]] = []
        for r in records[:5]:
            if not isinstance(r, dict):
                continue
            entry: Dict[str, Any] = {}
            label = _get_label(r)
            if label:
                entry["name"] = label
            unit = _get_unit(r)
            if unit:
                entry["unit"] = unit
            score = _get_score(r)
            if score is not None:
                entry["score"] = score
            if entry:
                sample.append(entry)

        summary = {
            "type": "cross_filter",
            "match_count": match_count,
        }
        if total_before:
            summary["total_before_filter"] = total_before
        if filter_desc:
            summary["filter_description"] = filter_desc
        if sample:
            summary["sample_records"] = sample
        if ctx:
            summary["context"] = ctx
        return summary

    # ── Multi-independent ────────────────────────────────────────────────────
    if query_type == "multi_independent":
        result = combined_result if isinstance(combined_result, dict) else {}
        sections = result.get("sections") or []
        section_info: List[Dict[str, Any]] = []
        for sec in sections:
            if not isinstance(sec, dict):
                continue
            label = sec.get("label") or "Section"
            data = sec.get("data") or []
            scores = [s for s in (_get_score(r) for r in data if isinstance(r, dict)) if s is not None]
            entry: Dict[str, Any] = {"label": label, "count": len(data)}
            if scores:
                entry["avg_score"] = round(sum(scores) / len(scores), 2)
            section_info.append(entry)

        summary = {
            "type": "multi_independent",
            "section_count": len(section_info),
            "sections": section_info,
        }
        if ctx:
            summary["context"] = ctx
        return summary

    # ── Simple / trend / distribution / ranking ──────────────────────────────
    records = _extract_records(combined_result)
    scores = [s for s in (_get_score(r) for r in records if isinstance(r, dict)) if s is not None]

    summary = {
        "type": query_type or "simple",
        "record_count": len(records),
    }

    if ctx:
        summary["context"] = ctx

    if scores:
        summary["avg_score"] = round(sum(scores) / len(scores), 2)
        summary["max_score"] = round(max(scores), 2)
        summary["min_score"] = round(min(scores), 2)

    # Top performers / named entries (most useful for natural language)
    top_entries: List[str] = []
    for r in records[:5]:
        if not isinstance(r, dict):
            continue
        label = _get_label(r)
        unit = _get_unit(r)
        score = _get_score(r)
        if label:
            parts = [label]
            if unit:
                parts.append(f"({unit})")
            if score is not None:
                parts.append(f"— {score}")
            top_entries.append(" ".join(parts))
    if top_entries:
        summary["top_entries"] = top_entries

    # For small result sets (≤ 10), include flattened record fields directly
    # so the LLM can reference specific values without hallucination
    if 0 < len(records) <= 10:
        flat_records: List[Dict[str, Any]] = []
        for r in records:
            if not isinstance(r, dict):
                continue
            flat = {
                k: v
                for k, v in r.items()
                if not isinstance(v, (dict, list)) and k not in ("id",)
            }
            if flat:
                flat_records.append(flat)
        if flat_records:
            summary["records"] = flat_records

    # Unit breakdown — how many per unique unit
    unit_counts: Dict[str, int] = {}
    for r in records:
        if not isinstance(r, dict):
            continue
        unit = _get_unit(r)
        if unit:
            unit_counts[unit] = unit_counts.get(unit, 0) + 1
    if len(unit_counts) > 1:
        # Only include if it adds value (multiple units)
        summary["unit_breakdown"] = dict(list(unit_counts.items())[:5])

    return summary


# ---------------------------------------------------------------------------
# LLM prompt builder
# ---------------------------------------------------------------------------

def _build_llm_prompt(
    user_query: str,
    data_summary: Dict[str, Any],
    query_type: str,
    intent: Dict[str, Any],
) -> str:
    """
    Build the Ollama prompt. Tight, data-grounded, military-domain aware.
    """
    query_type_label = {
        "simple": "data lookup",
        "compare": "side-by-side comparison",
        "cross_filter": "multi-criteria filter search",
        "multi_independent": "multi-module consolidated report",
        "trend": "trend analysis over time",
        "distribution": "distribution breakdown",
        "ranking": "ranking / leaderboard",
    }.get(query_type, "data query")

    data_str = json.dumps(data_summary, ensure_ascii=False, default=str, indent=2)

    # Determine what the officer is likely looking at
    module = intent.get("category") or "training"
    submodule = intent.get("subcategory") or ""
    scope_label = f"{module} — {submodule}" if submodule else module

    prompt = f"""You are AgniAI, an intelligent command console assistant for the Indian Army Agniveer training programme. You brief commanding officers in clear, professional English — direct, respectful, and grounded only in the data retrieved.

Officer's query: "{user_query}"
Query type: {query_type_label}
Module: {scope_label}

Retrieved data summary:
{data_str}

Write a response of 2 to 4 sentences that:
1. Opens by directly answering what was asked — no preamble like "Sure" or "Of course"
2. States the key numbers from the data above (use ONLY numbers present in the JSON above)
3. If there are named records (top_entries, sample_records), reference 1-2 by name naturally
4. If record_count is 0 (or match_count is 0), politely report no records found and suggest widening the filter
5. Ends with a brief, relevant observation or forward pointer if appropriate (e.g. "The detailed breakdown is available below.")

Style rules:
- Tone: confident aide briefing a senior officer — not a chatbot, not a report template
- Do NOT use markdown, bullets, headers, or numbered lists
- Do NOT say "I retrieved", "the system fetched", "as per the data" — just speak the answer
- Do NOT invent any number, name, or statistic not present in the data summary above
- Maximum 90 words"""

    return prompt


# ---------------------------------------------------------------------------
# Static fallback
# ---------------------------------------------------------------------------

def _static_fallback(
    user_query: str,
    data_summary: Dict[str, Any],
    intent: Dict[str, Any],
) -> str:
    """
    Constructs a natural, data-grounded fallback message without Ollama.
    References real numbers from data_summary — never generic boilerplate.
    """
    qtype = data_summary.get("type", "simple")
    ctx = data_summary.get("context") or {}
    module = ctx.get("module") or intent.get("category") or "records"
    section = ctx.get("section") or ctx.get("sub_module") or ""
    scope = f"{module} — {section}" if section else module

    # ── No results ───────────────────────────────────────────────────────────
    count = data_summary.get("record_count") or data_summary.get("match_count") or 0
    if count == 0 and qtype not in ("comparison", "multi_independent"):
        filter_hints = []
        if ctx.get("batch"):
            filter_hints.append(f"Batch {ctx['batch']}")
        if ctx.get("platoon"):
            filter_hints.append(f"Platoon {ctx['platoon']}")
        if ctx.get("agniveer"):
            filter_hints.append(f"Agniveer {ctx['agniveer']}")
        scope_note = f" under {', '.join(filter_hints)}" if filter_hints else ""
        return (
            f"No {scope} records were found{scope_note}. "
            "You may want to check the filter criteria or broaden the date range."
        )

    # ── Comparison ───────────────────────────────────────────────────────────
    if qtype in ("comparison", "compare"):
        left = data_summary.get("left_label", "Group A")
        right = data_summary.get("right_label", "Group B")
        left_count = data_summary.get("left_count", 0)
        right_count = data_summary.get("right_count", 0)
        left_avg = data_summary.get("left_avg")
        right_avg = data_summary.get("right_avg")

        msg = f"Here is the side-by-side comparison between {left} and {right}."
        if left_count or right_count:
            msg += f" {left} has {left_count} record{'s' if left_count != 1 else ''}"
            msg += f" and {right} has {right_count}."
        if left_avg is not None and right_avg is not None:
            better = left if left_avg >= right_avg else right
            msg += f" {better} leads on average score. The full breakdown is displayed below."
        else:
            msg += " The detailed comparison is displayed below."
        return msg

    # ── Cross-filter ─────────────────────────────────────────────────────────
    if qtype == "cross_filter":
        match_count = data_summary.get("match_count", 0)
        total_before = data_summary.get("total_before_filter")
        sample = data_summary.get("sample_records") or []
        sample_names = [s["name"] for s in sample if s.get("name")]

        msg = f"{match_count} Agniveer{'s' if match_count != 1 else ''} matched all the specified criteria."
        if total_before:
            msg += f" This is filtered from a pool of {total_before} records."
        if sample_names:
            names_str = ", ".join(sample_names[:2])
            msg += f" Top matches include {names_str}."
        msg += " The full filtered list is shown below."
        return msg

    # ── Multi-independent ────────────────────────────────────────────────────
    if qtype == "multi_independent":
        section_count = data_summary.get("section_count", 0)
        sections = data_summary.get("sections") or []
        section_labels = [s["label"] for s in sections if s.get("label")]
        labels_str = ", ".join(section_labels[:3]) if section_labels else "multiple modules"

        msg = f"The consolidated report covers {section_count} independent section{'s' if section_count != 1 else ''}: {labels_str}."
        msg += " Each module's data is presented separately below for your review."
        return msg

    # ── Trend ────────────────────────────────────────────────────────────────
    if qtype == "trend":
        rec_count = data_summary.get("record_count", 0)
        avg = data_summary.get("avg_score")
        msg = f"The {scope} trend data covers {rec_count} data point{'s' if rec_count != 1 else ''}."
        if avg is not None:
            msg += f" The average across this period is {avg}."
        msg += " The trend chart is displayed below."
        return msg

    # ── Distribution ─────────────────────────────────────────────────────────
    if qtype == "distribution":
        rec_count = data_summary.get("record_count", 0)
        unit_breakdown = data_summary.get("unit_breakdown") or {}
        msg = f"The {scope} distribution spans {rec_count} record{'s' if rec_count != 1 else ''}."
        if unit_breakdown:
            top_unit = max(unit_breakdown, key=unit_breakdown.get)  # type: ignore[arg-type]
            msg += f" The largest group is {top_unit} with {unit_breakdown[top_unit]} entries."
        msg += " The full distribution is shown in the table below."
        return msg

    # ── Simple / ranking / default ────────────────────────────────────────────
    rec_count = data_summary.get("record_count", 0)
    avg = data_summary.get("avg_score")
    top_entries = data_summary.get("top_entries") or []

    # Scope label with filter context
    filter_parts = []
    if ctx.get("batch"):
        filter_parts.append(f"Batch {ctx['batch']}")
    if ctx.get("platoon"):
        filter_parts.append(f"Platoon {ctx['platoon']}")
    if ctx.get("agniveer"):
        filter_parts.append(f"Agniveer {ctx['agniveer']}")
    filter_str = " for " + ", ".join(filter_parts) if filter_parts else ""

    if rec_count == 1 and data_summary.get("records"):
        # Single record — be specific
        r = data_summary["records"][0]
        name = r.get("fullName") or r.get("name") or r.get("agniveerNo") or "the Agniveer"
        score_val = _get_score(r)
        msg = f"Here are the {scope} details for {name}{filter_str}."
        if score_val is not None:
            msg += f" Their recorded score is {score_val}."
        return msg

    msg = f"{rec_count} {scope} record{'s' if rec_count != 1 else ''} found{filter_str}."

    if avg is not None:
        msg += f" The average score across this set is {avg}."

    if top_entries:
        first = top_entries[0]
        msg += f" Top result: {first}."

    if qtype == "ranking" and rec_count > 1:
        msg += " The full ranking is listed below."
    else:
        msg += " The complete dataset is formatted below."

    return msg


# ---------------------------------------------------------------------------
# LLM response validator
# ---------------------------------------------------------------------------

def _validate_llm_response(
    text: str,
    data_summary: Dict[str, Any],
) -> Optional[str]:
    """
    Verify the LLM response only references numbers present in the data summary.
    - Small integers (≤ 10) and plausible years (2000-2035) are always allowed
      as they're likely contextual language ("2 sentences", "3rd rank", etc.)
    - Returns cleaned text if valid, None if hallucination detected.
    """
    data_str = json.dumps(data_summary, default=str)

    def _canon(n: str) -> str:
        try:
            return str(float(n))
        except ValueError:
            return n

    data_numbers = {_canon(n) for n in re.findall(r"\b\d+(?:\.\d+)?\b", data_str)}
    llm_numbers = {_canon(n) for n in re.findall(r"\b\d+(?:\.\d+)?\b", text)}

    invented = set()
    for n in llm_numbers:
        try:
            fval = float(n)
        except ValueError:
            continue
        if fval <= 10:
            continue  # Allow small context integers
        if 2000 <= fval <= 2035:
            continue  # Allow years
        if n not in data_numbers:
            invented.add(n)

    if invented:
        logger.warning(
            "message_engine: LLM hallucinated numbers %s — falling back to static",
            invented,
        )
        return None

    # Basic quality check: must be at least a sentence
    stripped = text.strip().strip("\"'")
    if len(stripped) < 15:
        return None

    return stripped


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_message(
    user_query: str,
    combined_result: Any,
    query_type: str,
    intent: Dict[str, Any],
    trace_id: Optional[str] = None,
) -> str:
    """
    Generate a natural language response message for AgniAI.

    Priority:
      1. Ollama LLM  — data-grounded prompt → conversational response
      2. Static fallback — references real numbers, domain-aware prose

    Never raises — always returns a usable string.
    """
    log_prefix = f"[trace={trace_id}]" if trace_id else ""

    # Build data summary (safe — never raises past this point)
    try:
        data_summary = _build_data_summary(combined_result, query_type, intent)
    except Exception as exc:
        logger.warning("%s message_engine: _build_data_summary failed: %s", log_prefix, exc)
        data_summary = {"type": query_type, "record_count": 0}

    # ── Attempt Ollama ────────────────────────────────────────────────────────
    try:
        from config import DEFAULT_MODEL, OLLAMA_URL, ollama_session

        prompt = _build_llm_prompt(user_query, data_summary, query_type, intent)
        payload = {
            "model": DEFAULT_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": 0.25,  # Low temp — factual, grounded
                "num_predict": 180,   # ~3-4 sentences
                "num_ctx": 2048,      # Enough for prompt + data summary
            },
        }

        resp = ollama_session.post(OLLAMA_URL, json=payload, timeout=(5, 25))
        resp.raise_for_status()

        raw_text = resp.json().get("message", {}).get("content", "").strip()
        validated = _validate_llm_response(raw_text, data_summary)

        if validated:
            logger.debug(
                "%s message_engine: LLM message OK (len=%d)",
                log_prefix,
                len(validated),
            )
            return validated

        logger.debug(
            "%s message_engine: LLM response rejected by validator — using fallback",
            log_prefix,
        )

    except Exception as exc:
        logger.debug(
            "%s message_engine: Ollama unavailable — using static fallback: %s",
            log_prefix,
            exc,
        )

    # ── Static fallback ───────────────────────────────────────────────────────
    msg = _static_fallback(user_query, data_summary, intent)
    logger.debug("%s message_engine: static fallback used", log_prefix)
    return msg