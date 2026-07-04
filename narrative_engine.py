"""
narrative_engine.py
===================
Unified natural-language narrative generation for AgniAI.

Takes the FACTS computed by the deterministic engines
(analysis_engine, prediction_engine, conclusion_engine) plus the data
summary, and produces FIVE polished narrative strings in ONE Ollama call:

    message     — direct conversational answer to the officer's query
    analysis    — what the data shows (interpretation, not stat recital)
    prediction  — forward outlook (direction, confidence, what to watch)
    conclusion  — crisp verdict + one recommended action
    summary     — 1-2 sentence executive line (NOT a concatenation)

Design rules (this is what makes output read like Claude/GPT):
  1. The LLM never computes — it only rephrases numbers/names already
     present in the grounding facts. Zero invented values.
  2. Each field has a distinct JOB. No field repeats another field's
     content. The prompt enforces this explicitly.
  3. Every field is validated for hallucinated numbers; a failing field
     falls back to a polished static rendering of the engine dicts —
     never the old raw concatenation.
  4. One Ollama round-trip for all five fields (previously message alone
     cost one call; analysis/prediction/conclusion were template dumps).

Integration (pipeline):

    from narrative_engine import generate_narratives

    narratives = generate_narratives(
        user_query=user_query,
        combined_result=combined_result,
        query_type=query_type,
        intent=intent,
        analysis=analysis_dict,        # from analysis_engine
        prediction=prediction_dict,    # from prediction_engine
        conclusion=conclusion_dict,    # from conclusion_engine
        trace_id=trace_id,
    )

    response = build_response(
        message=narratives["message"],
        ...
        analysis=narratives["analysis"],       # now a string
        prediction=narratives["prediction"],   # now a string
        conclusion=narratives["conclusion"],   # now a string
        summary_override=narratives["summary"],
    )

response_builder._to_str already passes plain strings through unchanged,
so no widget/contract changes are needed.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Grounding facts builder
# ---------------------------------------------------------------------------

def _compact(d: Any, max_len: int = 3500) -> str:
    """JSON-serialise grounding facts, truncated to keep the prompt small."""
    try:
        s = json.dumps(d, ensure_ascii=False, default=str)
    except Exception:
        s = str(d)
    return s[:max_len]


def _build_grounding_facts(
    analysis: Optional[Dict[str, Any]],
    prediction: Optional[Dict[str, Any]],
    conclusion: Optional[Dict[str, Any]],
    data_summary: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Collect ONLY the fields the LLM is allowed to reference.
    Everything numeric the LLM says must exist somewhere in here.
    """
    facts: Dict[str, Any] = {}
    if isinstance(data_summary, dict):
        facts["data"] = data_summary
    if isinstance(analysis, dict):
        facts["analysis_facts"] = {
            "summary": analysis.get("summary"),
            "insights": (analysis.get("insights") or [])[:6],
            "statistics": analysis.get("statistics") or {},
        }
    if isinstance(prediction, dict):
        facts["prediction_facts"] = {
            "trend": prediction.get("trend"),
            "trendConfidence": prediction.get("trendConfidence"),
            "momentum": prediction.get("momentum"),
            "projection": prediction.get("projection"),
            "heuristicEstimate": prediction.get("heuristicEstimate"),
            "futureTrends": (prediction.get("futureTrends") or [])[:4],
        }
    if isinstance(conclusion, dict):
        facts["conclusion_facts"] = {
            "summary": conclusion.get("summary"),
            "bullets": (conclusion.get("bullets") or [])[:4],
        }
    return facts


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SUPER_PROMPT_TEMPLATE = """You are AgniAI, the intelligence aide embedded in an Indian Army Agniveer training command console. You brief commanding officers. Your writing is what a sharp, trusted staff officer sounds like: direct, confident, specific, and economical. You never sound like a database, a report template, or a chatbot.

## THE OFFICER'S QUERY
"{user_query}"

Query type: {query_type}
Module: {scope}

## GROUNDING FACTS — YOUR ONLY SOURCE OF TRUTH
{facts_json}

## YOUR TASK
Return ONE JSON object with exactly these five keys:
"message", "analysis", "prediction", "conclusion", "summary"

Each field is a plain string of complete sentences. No markdown, no bullets, no headers, no line breaks inside a field.

## THE DATA CONTRACT — ABSOLUTE
1. Every number you write must appear in GROUNDING FACTS exactly as written there. Do not re-round, re-compute, or estimate.
2. Every person, unit, batch, or platoon you name must appear in the facts.
3. If a detail is not in the facts, you do not know it. Never fill gaps.
4. If the facts show 0 records: every field states plainly that no records matched and suggests widening the filter or date range. Prediction and conclusion become one sentence each.

## FIELD BLUEPRINTS — EACH FIELD HAS ONE JOB

### "message" (3-6 sentences) — ANSWER THE QUESTION
The first sentence IS the answer to the officer's query — the single fact they asked for, stated plainly. Then support it: the 2-3 most decision-relevant numbers, and 2-3 individuals BY NAME with their scores when names exist. If anything demands attention (below-passing trainees, a widening gap), say so here — do not save bad news for later fields. Close by pointing to the visual: "The full breakdown is charted below." or similar.

### "analysis" (2-4 sentences) — EXPLAIN WHAT THE NUMBERS MEAN
Interpretation only, no recital. Lead with the most significant pattern: where scores cluster, who or which unit leads and who trails, whether the spread is tight or scattered, what the outliers are. Every number you cite must earn its place by supporting a judgement. Name the strong and weak performers if names exist.

### "prediction" (2-3 sentences) — LOOK FORWARD ONLY
No restating of current stats. State: the direction (improving / declining / holding steady), how firm that call is (from trendConfidence), the expected next-cycle level if the facts contain one, and the single most important thing to watch or act on before the next cycle. If momentum is near zero, say the picture is steady — do not invent drama.

### "conclusion" (1-2 sentences) — VERDICT PLUS ONE ACTION
One clear verdict: on track / needs attention / mixed. Then ONE concrete recommended action ONLY if the facts explicitly contain the necessary context. Do NOT invent training plans, schedules, names, or actions that are not present in the facts. Cite a number only if that number IS the verdict.

### "summary" (1-2 sentences) — THE FIVE-SECOND READ
The one thing a commanding officer must retain if they read nothing else. Written fresh — never assembled from the other fields' sentences.

## STYLE DISCIPLINE
- BANNED OPENERS: "Based on", "As per", "According to", "The data shows", "The system", "I retrieved", "Sure", "Certainly", "Overall,", "In summary", "It is important to note", "Here is", "Here are".
- BANNED FORMS: "record(s)" with parentheses — choose "record" or "records"; "utilize"; "delve"; "aforementioned"; "as follows".
- Vary sentence openings across the five fields — if two fields start with the same word, rewrite one.
- Active voice. "Alpha Company leads by 6 points", not "a lead of 6 points is held by Alpha Company".
- Domain terms are natural: Agniveer, platoon, batch, BPET, PPT, firing, drill.
- NO sentence may appear in two fields. NO number may appear in more than two fields unless it is the record count.

## QUERY-TYPE SHARPENING
- compare: "message" must name the leader and the size of the gap in its first two sentences. "prediction" states whether the gap should widen, close, or hold.
- trend: "prediction" must state momentum direction and the projected level.
- cross_filter: "message" opens with how many matched the combined criteria.
- multi_independent: "message" gives a one-line verdict per section, sharpest finding first; "summary" names the section that most needs attention.

## QUALITY BAR — WORKED EXAMPLE
Given facts (abbreviated): 12 performance records, avg 68.4, range 41.0-91.0, tight cluster; Rakesh Kumar 91.0 top; Ramesh Yadav 41.0 and Vikram Joshi 47.5 below the 50 passing mark; trend Stable, confidence Medium, next-cycle estimate ~69.0; recovery path: structured physical conditioning.

{{"message": "Batch 12 is performing at a moderate level — 12 Agniveers assessed with an average of 68.4. Rakesh Kumar leads the batch at 91.0, while Ramesh Yadav at 41.0 and Vikram Joshi at 47.5 sit below the passing mark and need intervention. The rest of the batch is grouped comfortably in the middle band. The full score breakdown is charted below.", "analysis": "Scores cluster tightly around the 68.4 average, which makes the two failing results stand out as individual problems rather than a batch-wide weakness. Rakesh Kumar's 91.0 is well clear of the field and sets the benchmark. The gap between him and the two lowest scorers spans the entire 41.0-91.0 range, so the headline average masks a wide floor-to-ceiling spread.", "prediction": "The batch should hold near 69.0 next cycle — momentum is flat and confidence in that call is medium. The deciding factor is whether Yadav and Joshi respond to conditioning before the next assessment; their recovery alone would lift the batch floor meaningfully.", "conclusion": "The batch is on track overall, but two trainees need attention now. Put Yadav and Joshi on the structured physical conditioning plan and re-test them at the next cycle.", "summary": "Batch 12 averages 68.4 and is stable, but Ramesh Yadav and Vikram Joshi are below the passing mark — conditioning intervention before the next cycle is the priority."}}

Note what the example does: answer first, names early, bad news up front, each field doing a different job, no field copying another, every number traceable to the facts.

## BEFORE YOU RETURN — SELF-CHECK
1. Valid JSON object, exactly five keys, nothing outside the braces.
2. Every number, name, and recommended action MUST exist in GROUNDING FACTS. If the facts lack names or specific problems, do NOT copy names like "Yadav" or "Joshi" from the example, and do NOT invent schedule/training plans.
3. No banned opener or banned form anywhere.
4. No repeated sentence across fields.
5. Every field ends with terminal punctuation.

Return ONLY the JSON object."""


def _build_prompt(
    user_query: str,
    query_type: str,
    intent: Dict[str, Any],
    facts: Dict[str, Any],
) -> str:
    category = intent.get("category") or "training"
    subcategory = intent.get("subcategory") or ""
    scope = f"{category} — {subcategory}" if subcategory else category
    return _SUPER_PROMPT_TEMPLATE.format(
        user_query=user_query,
        query_type=query_type or "simple",
        scope=scope,
        facts_json=_compact(facts),
    )



# ---------------------------------------------------------------------------
# Deterministic scrubber — strips chatbot boilerplate the LLM produces despite
# prompt bans. Runs on every field BEFORE validation, so good content inside a
# badly-wrapped response is salvaged instead of discarded.
# ---------------------------------------------------------------------------

_SCRUB_PATTERNS = [
    # Greetings with or without honorific / time of day
    r"^(?:good\s+(?:morning|afternoon|evening|day)|hello|greetings|namaste|jai\s+hind)[,!.]?\s*(?:sir|ma'?am|officer)?[,!.]?\s*",
    # Retrieval / system narration openers
    r"^(?:i\s+have\s+retrieved|i\s+retrieved|i\s+have\s+fetched|the\s+system\s+(?:has\s+)?(?:retrieved|fetched))[^.]*?(?:\.|:)\s*",
    # Hedging connectors anywhere in the text
    r"\b(?:according\s+to\s+the\s+data|as\s+per\s+the\s+data|based\s+on\s+the\s+(?:data|records|information))(?:\s+(?:provided|retrieved|available))?,?\s*",
    r"\b(?:it(?:'|\u2019)s|it\s+is)\s+(?:important|worth)\s+(?:to\s+note|noting)\s+that\s*,?\s*",
    r"\bplease\s+note\s+that\s*,?\s*",
    # Chatbot sign-offs
    r"\s*(?:if\s+you\s+(?:require|need)\s+(?:any\s+)?(?:further|more|additional)\s+(?:information|details|assistance)[^.]*|should\s+you\s+(?:require|need)[^.]*|feel\s+free\s+to\s+ask[^.]*|please\s+let\s+me\s+know[^.]*|don(?:'|\u2019)t\s+hesitate\s+to[^.]*)\.\s*",
    # "Sure," / "Certainly," / "Of course," openers
    r"^(?:sure|certainly|of\s+course|absolutely)[,!.]?\s*",
]

_SCRUB_COMPILED = [re.compile(p, re.IGNORECASE) for p in _SCRUB_PATTERNS]

# PascalCase internal module names -> spoken form (e.g. "DisqualifiedAgniveer"
# -> "disqualified Agniveer"). Applied only to known internal-looking tokens.
_PASCAL_MODULE_RE = re.compile(
    r"\b((?:[A-Z][a-z]+){2,})\s+module\b"
)


def _humanize_pascal(match: "re.Match") -> str:
    token = match.group(1)
    words = re.findall(r"[A-Z][a-z]+", token)
    spoken = " ".join(words)
    # Keep domain proper nouns capitalised
    spoken = spoken.replace("Agniveer", "Agniveer")
    return spoken[0].lower() + spoken[1:] + " records"


def _scrub(text: str) -> str:
    """Strip banned boilerplate; tidy whitespace; re-capitalise the opener."""
    if not text:
        return ""
    out = text.strip()
    # Iterate until stable (max 3 passes) — stacked openers like
    # "Good afternoon Sir, hello! Sure, ..." need multiple rounds since
    # anchored (^) patterns only match the current start of the text.
    for _ in range(3):
        before = out
        for pat in _SCRUB_COMPILED:
            out = pat.sub("", out).strip()
        if out == before:
            break
    out = _PASCAL_MODULE_RE.sub(_humanize_pascal, out)
    out = re.sub(r"\s{2,}", " ", out).strip()
    # A leading orphan like ", there were" after scrubbing an opener
    out = re.sub(r"^[,;:\s]+", "", out)
    # Re-capitalise the start of every sentence (hedge removal can leave
    # a lowercase word right after a period)
    out = re.sub(
        r"(^|[.!?]\s+)([a-z])",
        lambda m: m.group(1) + m.group(2).upper(),
        out,
    )
    # Restore a space if scrubbing glued two sentences together
    out = re.sub(r"([.!?])([A-Z])", r"\1 \2", out)
    return out

# ---------------------------------------------------------------------------
# Number validation (same philosophy as message_engine._validate_llm_response)
# ---------------------------------------------------------------------------

def _canon(n: str) -> str:
    try:
        return str(float(n))
    except ValueError:
        return n


def _numbers_in(text: str) -> set:
    return {_canon(n) for n in re.findall(r"\b\d+(?:\.\d+)?\b", text)}


def _validate_field(text: str, allowed_numbers: set) -> bool:
    """A field passes if every number > 10 (and not a year) exists in facts."""
    if not text or len(text.strip()) < 10:
        return False
    stripped = text.strip()
    if stripped[-1] not in ".!?":
        return False
    for n in _numbers_in(stripped):
        try:
            f = float(n)
        except ValueError:
            continue
        if f <= 10 or 2000 <= f <= 2035:
            continue
        if n not in allowed_numbers:
            return False
    return True


# ---------------------------------------------------------------------------
# Polished static fallbacks (per field) — used when Ollama is down or a
# field fails validation. These read far better than the old concatenation.
# ---------------------------------------------------------------------------

def _fallback_analysis(analysis: Optional[Dict[str, Any]]) -> str:
    if not isinstance(analysis, dict):
        return ""
    head = str(analysis.get("summary") or "").strip()
    insights = [str(i).strip() for i in (analysis.get("insights") or []) if str(i).strip()]

    def _is_stat_recital(text: str) -> bool:
        # Pipe-separated stat dumps or lines whose numbers all repeat the head
        if " | " in text:
            return True
        nums = _numbers_in(text)
        return bool(nums) and nums.issubset(_numbers_in(head))

    def _priority(text: str) -> int:
        t = text.lower()
        if any(w in t for w in ("attention", "performer", "leads", "trails", "needing")):
            return 0  # named individuals / group comparisons first
        if any(w in t for w in ("cluster", "variab", "skew", "spread", "consistent")):
            return 1  # distribution shape second
        return 2

    candidates = sorted(
        (i for i in insights if not _is_stat_recital(i)), key=_priority
    )
    picked: List[str] = []
    for i in candidates:
        if i not in head and all(i not in p for p in picked):
            picked.append(i)
        if len(picked) == 2:
            break
    parts = [p for p in [head] + picked if p]
    return " ".join(p if p.endswith((".", "!", "?")) else f"{p}." for p in parts)


def _fallback_prediction(prediction: Optional[Dict[str, Any]]) -> str:
    if not isinstance(prediction, dict):
        return ""
    trend = str(prediction.get("trend") or "Stable")
    conf = str(prediction.get("trendConfidence") or "Low")
    est = str(prediction.get("heuristicEstimate") or "").strip()
    trends = [str(t).strip() for t in (prediction.get("futureTrends") or []) if str(t).strip()]
    lead = f"The outlook is {trend.lower()} with {conf.lower()} confidence."
    parts = [lead]
    if est and est not in parts:
        parts.append(est)
    # One action-oriented trend line if available (prefer ones mentioning intervention/recovery)
    action = next(
        (t for t in trends if any(w in t.lower() for w in ("recommend", "advис", "advised", "intervention", "recovery", "monitor"))),
        trends[0] if trends else "",
    )
    if action and action not in parts:
        parts.append(action)
    return " ".join(p if p.endswith((".", "!", "?")) else f"{p}." for p in parts)


def _fallback_conclusion(conclusion: Optional[Dict[str, Any]]) -> str:
    if not isinstance(conclusion, dict):
        return ""
    head = str(conclusion.get("summary") or "").strip()
    bullets = [str(b).strip() for b in (conclusion.get("bullets") or []) if str(b).strip()]
    # Prefer the verdict bullet if the engine produced one
    verdict = next((b for b in bullets if b.lower().startswith("overall verdict")), "")
    parts = [p for p in (head, verdict or (bullets[0] if bullets else "")) if p]
    return " ".join(p if p.endswith((".", "!", "?")) else f"{p}." for p in parts)


def _fallback_summary(
    analysis: Optional[Dict[str, Any]],
    conclusion: Optional[Dict[str, Any]],
) -> str:
    """One executive line: prefer the conclusion verdict, else analysis head."""
    if isinstance(conclusion, dict):
        bullets = [str(b).strip() for b in (conclusion.get("bullets") or [])]
        verdict = next((b for b in bullets if b.lower().startswith("overall verdict")), "")
        if verdict:
            return verdict
        head = str(conclusion.get("summary") or "").strip()
        if head:
            return head
    if isinstance(analysis, dict):
        return str(analysis.get("summary") or "").strip()
    return ""


# ---------------------------------------------------------------------------
# JSON extraction — tolerant of code fences and stray text
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    # Fast path
    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    # Find the outermost {...}
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(cleaned[start : end + 1])
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_FIELDS = ("message", "analysis", "prediction", "conclusion", "summary")


def generate_narratives(
    user_query: str,
    combined_result: Any,
    query_type: str,
    intent: Dict[str, Any],
    analysis: Optional[Dict[str, Any]] = None,
    prediction: Optional[Dict[str, Any]] = None,
    conclusion: Optional[Dict[str, Any]] = None,
    trace_id: Optional[str] = None,
) -> Dict[str, str]:
    """
    Produce all five narrative strings. Never raises.
    Any field the LLM fails on individually falls back to its static version —
    partial LLM success is kept, not discarded wholesale.
    """
    log_prefix = f"[trace={trace_id}]" if trace_id else ""

    # Reuse message_engine's compact grounded summary of the raw data
    try:
        from message_engine import _build_data_summary, _static_fallback
        data_summary = _build_data_summary(combined_result, query_type, intent)
    except Exception as exc:
        logger.warning("%s narrative_engine: data summary failed: %s", log_prefix, exc)
        data_summary = {"type": query_type, "record_count": 0}
        _static_fallback = None  # type: ignore[assignment]

    facts = _build_grounding_facts(analysis, prediction, conclusion, data_summary)
    allowed_numbers = _numbers_in(_compact(facts, max_len=100000))

    # Static fallbacks prepared up-front (also the Ollama-down path)
    fallbacks: Dict[str, str] = {
        "message": (
            _static_fallback(user_query, data_summary, intent)
            if _static_fallback
            else "The requested data is displayed below."
        ),
        "analysis": _fallback_analysis(analysis),
        "prediction": _fallback_prediction(prediction),
        "conclusion": _fallback_conclusion(conclusion),
        "summary": _fallback_summary(analysis, conclusion),
    }

    # ── Single Ollama call for all five fields ────────────────────────────
    llm_fields: Dict[str, str] = {}
    try:
        from config import DEFAULT_MODEL, OLLAMA_URL, ollama_session

        prompt = _build_prompt(user_query, query_type, intent, facts)
        payload = {
            "model": DEFAULT_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "format": "json",  # Ollama structured mode — forces valid JSON
            "options": {
                "temperature": 0.3,
                "num_predict": 1100,
                "num_ctx": 8192,
            },
        }
        resp = ollama_session.post(OLLAMA_URL, json=payload, timeout=(0.5, 30.0))
        resp.raise_for_status()
        raw = resp.json().get("message", {}).get("content", "")
        parsed = _extract_json(raw)

        if isinstance(parsed, dict):
            for f in _FIELDS:
                candidate = _scrub(str(parsed.get(f) or ""))
                if _validate_field(candidate, allowed_numbers):
                    llm_fields[f] = candidate
                else:
                    logger.debug(
                        "%s narrative_engine: field '%s' failed validation — static fallback",
                        log_prefix, f,
                    )
        else:
            logger.debug("%s narrative_engine: LLM returned unparseable JSON", log_prefix)

    except Exception as exc:
        logger.debug(
            "%s narrative_engine: Ollama unavailable — all-static narratives: %s",
            log_prefix, exc,
        )

    # Per-field merge: LLM where valid, polished static otherwise
    return {f: _scrub(llm_fields.get(f) or fallbacks.get(f) or "") for f in _FIELDS}