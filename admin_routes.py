"""
admin_routes.py
===============
Admin chatbot Flask blueprint for AgniAI.

Registers:
  POST /api/admin/chat       — main admin chat endpoint
  POST /api/admin/classify   — classify-only (no .NET call; for debugging)
  GET  /api/admin/health     — admin route health check

Flow for /api/admin/chat:
  1. Receive admin question from frontend
  2. classify_admin_intent()  → structured intent dict (Python internal)
  3. format_admin_payload()   → camelCase JSON payload for .NET
     └─ includes commandId, batchId, platoonId from the frontend request
  4. POST payload to .NET     → https://<DOTNET_API_BASE_URL>/api/AiCommand/execute
  5. generate_intro_message() → LLM generates a natural-language intro sentence
  6. Return unified response  → { status, httpStatus, message: "<LLM intro>", data: { intent, dotnetPayload, result } }

RESPONSE ENVELOPE (single message field — the LLM intro sentence):
  {
    "status": true,
    "httpStatus": 200,
    "message": "The top 5 BEPT performers have been retrieved, ranked by their scores.",
    "data": {
      "intent":        { ...classified intent fields... },
      "dotnetPayload": { ...what was sent to .NET, including commandId/batchId/platoonId... },
      "data":          [ ...raw .NET response records... ],
      "commandLabel":  "...",
      "success":       true,
      "sessionId":     "...",
      "elapsedMs":     142
    }
  }

NOTE: There is only ONE "message" field — at the top level — containing the
LLM-generated intro sentence. The "data" object never contains its own
"message" key. This prevents the frontend from ever seeing two messages.

FRONTEND REQUEST FIELDS:
  message     — (required) natural-language admin question
  session_id  — (optional) session identifier
  commandId   — (optional, default 0) command ID filter for .NET
  batchId     — (optional, default 0) batch ID filter for .NET
  platoonId   — (optional, default 0) platoon ID filter for .NET

PORT SEPARATION:
  - Python / Flask  → port 5000  (python app.py)
  - .NET AiCommand  → port 7257  (set DOTNET_API_BASE_URL in .env)

Configuration (via environment variables):
  DOTNET_API_BASE_URL    — default: https://localhost:7257
  DOTNET_API_KEY         — optional X-Api-Key header
  DOTNET_SKIP_SSL_VERIFY — "1" to skip SSL verification (self-signed cert)
  ADMIN_RATE_LIMIT       — default: "20 per minute"
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, Optional

import requests as _requests
from flask import Blueprint, jsonify, request

from admin_intent import classify_admin_intent, format_admin_payload
from config import _is_greeting 

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────
DOTNET_API_BASE_URL = os.getenv("DOTNET_API_BASE_URL", "https://localhost:7257")
DOTNET_EXECUTE_URL  = f"{DOTNET_API_BASE_URL}/api/AiCommand/execute"
DOTNET_API_KEY      = os.getenv("DOTNET_API_KEY", "")
DOTNET_TIMEOUT      = int(os.getenv("DOTNET_TIMEOUT", "30"))
ADMIN_RATE_LIMIT    = os.getenv("ADMIN_RATE_LIMIT", "20 per minute")

_skip_raw = os.getenv("DOTNET_SKIP_SSL_VERIFY", os.getenv("DOTNET_VERIFY_SSL", "0"))
DOTNET_VERIFY_SSL = _skip_raw.strip() not in {"1", "true", "True"}

# ── Blueprint ──────────────────────────────────────────────────────────────
admin_bp = Blueprint("admin", __name__, url_prefix="/api/admin")

_dotnet_session = _requests.Session()

if not DOTNET_VERIFY_SSL:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# =============================================================================
# RATE LIMITER
# =============================================================================

def _get_limiter():
    try:
        from app import _limiter
        return _limiter
    except (ImportError, AttributeError):
        return None


def _register_rate_limits(app):
    try:
        from app import _limiter
        if _limiter is not None:
            _limiter.limit(
                ADMIN_RATE_LIMIT,
                override_defaults=True,
            )(admin_chat)
            logger.info("Admin rate limit applied: %s", ADMIN_RATE_LIMIT)
    except Exception as exc:
        logger.warning("Could not register admin rate limit: %s", exc)


# =============================================================================
# NATURAL LANGUAGE INTRO GENERATOR
# =============================================================================

_INTRO_TEMPLATES: Dict[tuple, str] = {
    # Performance
    ("Performance", "TopPerformers"):      "Here are the top performers as requested.",
    ("Performance", "LowestPerformers"):   "Here are the lowest performers as requested.",
    ("Performance", "AverageScore"):       "Here is the average score data.",
    ("Performance", "PassPercentage"):     "Here is the pass percentage breakdown.",
    ("Performance", "FailPercentage"):     "Here is the fail percentage breakdown.",
    ("Performance", "GradeDistribution"):  "Here is the grade distribution.",
    ("Performance", "GradingSummary"):     "Here is the grading summary.",
    ("Performance", "OverallPerformance"): "Here is the overall performance report.",
    ("Performance", "Improvement"):        "Here are the improvement details.",
    ("Performance", "Decline"):            "Here are the decline details.",
    ("Performance", "SectionSummary"):     "Here is the section-wise summary.",
    ("Performance", "AttemptWise"):        "Here is the attempt-wise analysis.",
    ("Performance", "BestAttempt"):        "Here is the best attempt data.",
    ("Performance", "Comparison"):         "Here is the performance comparison.",
    # Leave
    ("Leave", "MostLeaveTaken"):           "Here are the personnel who have taken the most leave.",
    ("Leave", "LeastLeaveTaken"):          "Here are the personnel who have taken the least leave.",
    ("Leave", "CurrentLeaveStatus"):       "Here is the current leave status.",
    ("Leave", "AbscondedPersonnel"):       "Here is the list of absconded personnel.",
    # Medical
    ("Medical", "ActiveCases"):            "Here are the active medical cases.",
    ("Medical", "BMIAnalysis"):            "Here is the BMI and fitness analysis.",
    ("Medical", "DiseaseStatistics"):      "Here are the top disease statistics.",
    # Attendance
    ("Attendance", "MonthlyAttendance"):   "Here is the monthly attendance summary.",
    ("Attendance", "PresentToday"):        "Here is today's attendance status.",
    ("Attendance", "StrengthBreakdown"):   "Here is the strength breakdown.",
    # Verification
    ("Verification", "PendingVerification"):   "Here are the pending verifications.",
    ("Verification", "CompletedVerification"): "Here are the completed verifications.",
    # Equipment
    ("Equipment", "EquipmentSummary"):         "Here is the equipment summary.",
    ("Equipment", "OverdueEquipment"):         "Here is the list of overdue equipment.",
    ("Equipment", "PoorConditionEquipment"):   "Here is the equipment returned in poor condition.",
    # Distribution
    ("Distribution", "LatestDistribution"):    "Here is the latest distribution data.",
    ("Distribution", "DistributionByUnit"):    "Here is the distribution broken down by unit.",
    ("Distribution", "UnassignedItems"):       "Here are the unassigned items.",
    ("Distribution", "TopUnit"):               "Here is the top unit for distribution.",
    # Skills
    ("Skills", "BySport"):                     "Here is the roster grouped by sport.",
    ("Skills", "ByClass"):                     "Here is the roster grouped by class.",
    ("Skills", "BloodGroup"):                  "Here is the blood group distribution.",
}


def _build_intro_prompt(
    question: str,
    intent: Dict[str, Any],
    dotnet_data: Any,
) -> str:
    category    = intent.get("category", "")
    subcategory = intent.get("subcategory", "")
    number      = intent.get("number")
    section     = intent.get("section", "")
    leave_type  = intent.get("leave_type", "")
    grading     = intent.get("grading", "")
    unit_name   = intent.get("unit_name", "")
    sport       = intent.get("sport", "")
    class_name  = intent.get("class", "")

    data_summary = ""
    try:
        if isinstance(dotnet_data, list) and dotnet_data:
            data_summary = f"The data contains {len(dotnet_data)} record(s)."
        elif isinstance(dotnet_data, dict):
            count = (
                dotnet_data.get("total") or
                dotnet_data.get("count") or
                dotnet_data.get("Count") or
                dotnet_data.get("Total")
            )
            if count is not None:
                data_summary = f"The data shows a total/count of {count}."
            else:
                keys = list(dotnet_data.keys())[:5]
                data_summary = f"The data contains fields: {', '.join(str(k) for k in keys)}."
        elif isinstance(dotnet_data, (int, float)):
            data_summary = f"The result is: {dotnet_data}."
    except Exception:
        pass

    context_parts = []
    if category:
        context_parts.append(f"Module: {category}")
    if subcategory:
        context_parts.append(f"Query type: {subcategory}")
    if number:
        context_parts.append(f"Requested count: {number}")
    if section:
        context_parts.append(f"Section: {section}")
    if leave_type:
        context_parts.append(f"Leave type: {leave_type}")
    if grading:
        context_parts.append(f"Grading filter: {grading}")
    if unit_name:
        context_parts.append(f"Unit: {unit_name}")
    if sport:
        context_parts.append(f"Sport: {sport}")
    if class_name:
        context_parts.append(f"Class: {class_name}")
    if data_summary:
        context_parts.append(data_summary)

    context_str = "\n".join(context_parts)

    return (
        "You are an assistant for a military training management system. "
        "An admin just asked a question and the system has retrieved the relevant data. "
        "Your job is to write ONE short, natural, professional introductory sentence "
        "that will appear above the data before the admin sees it.\n\n"
        "Rules:\n"
        "1. Write exactly ONE sentence — no more.\n"
        "2. Be specific: mention the module, count, section, or filter if relevant.\n"
        "3. Sound natural and professional — not robotic or template-like.\n"
        "4. Do NOT say 'Here is the data' or 'Here are the results' — be descriptive.\n"
        "5. Do NOT include any JSON, bullet points, or markdown.\n"
        "6. Do NOT ask questions or add follow-ups.\n\n"
        f"Admin question: {question}\n\n"
        f"Context:\n{context_str}\n\n"
        "Write the introductory sentence now:"
    )


def _generate_intro_message(
    question: str,
    intent: Dict[str, Any],
    dotnet_data: Any,
) -> str:
    """
    Generate a natural-language intro sentence using Ollama.
    Falls back to a static template if Ollama is unavailable.
    Returns a single sentence string — no wrapping, no extra keys.
    """
    category    = intent.get("category", "")
    subcategory = intent.get("subcategory", "")

    try:
        import requests as _req
        from config import OLLAMA_URL, DEFAULT_MODEL

        prompt   = _build_intro_prompt(question, intent, dotnet_data)
        messages = [{"role": "user", "content": prompt}]
        payload  = {
            "model":   DEFAULT_MODEL,
            "messages": messages,
            "stream":  False,
            "options": {
                "temperature": 0.4,
                "num_predict": 80,
                "num_ctx":     1024,
            },
        }
        resp = _req.post(OLLAMA_URL, json=payload, timeout=(8, 30))
        resp.raise_for_status()
        text = (
            resp.json()
            .get("message", {})
            .get("content", "")
            .strip()
        )
        text = text.strip('"\'')
        if text and len(text) > 10:
            logger.debug("LLM intro generated: %s", text[:80])
            return text
    except Exception as exc:
        logger.debug("Ollama intro generation failed, using template: %s", exc)

    key = (category, subcategory)
    if key in _INTRO_TEMPLATES:
        return _INTRO_TEMPLATES[key]

    category_label = category or "the requested"
    return f"Here is the {category_label.lower()} data retrieved for your query."


# =============================================================================
# HELPERS
# =============================================================================

def _get_session_id(data: Dict) -> str:
    from config import SESSION_HEADER
    session_id = (
        data.get("session_id") or
        data.get("sessionId") or
        request.headers.get(SESSION_HEADER) or
        request.headers.get("X-Session-Id") or ""
    ).strip()
    return session_id or "admin-default"


def _get_id_filters(data: Dict) -> Dict[str, int]:
    def _safe_int(value) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    filters = {}

    command_id = _safe_int(data.get("commandId", data.get("command_id")))
    batch_id   = _safe_int(data.get("batchId",   data.get("batch_id")))
    platoon_id = _safe_int(data.get("platoonId", data.get("platoon_id")))

    if command_id is not None:
        filters["commandId"] = command_id
    if batch_id is not None:
        filters["batchId"] = batch_id
    if platoon_id is not None:
        filters["platoonId"] = platoon_id

    return filters


def _call_dotnet(payload: Dict) -> tuple[Any, Optional[str]]:
    headers = {"Content-Type": "application/json"}
    if DOTNET_API_KEY:
        headers["X-Api-Key"] = DOTNET_API_KEY

    logger.debug(
        "Calling .NET: URL=%s payload=%s",
        DOTNET_EXECUTE_URL,
        json.dumps(payload),
    )

    try:
        resp = _dotnet_session.post(
            DOTNET_EXECUTE_URL,
            json=payload,
            headers=headers,
            timeout=DOTNET_TIMEOUT,
            verify=DOTNET_VERIFY_SSL,
        )
        if resp.status_code >= 400:
            try:
                err_body = resp.json()
            except Exception:
                err_body = resp.text[:400]
            return None, f"Backend returned HTTP {resp.status_code}: {err_body}"
        return resp.json(), None

    except _requests.ConnectionError as exc:
        return None, (
            f"Cannot connect to .NET backend at {DOTNET_EXECUTE_URL}. "
            f"Is the .NET service running on the correct port? ({exc})\n"
            f"Tip: Python runs on port 5000, .NET should run on a different port "
            f"(set DOTNET_API_BASE_URL in .env)."
        )
    except _requests.Timeout:
        return None, f"Backend timed out after {DOTNET_TIMEOUT}s."
    except _requests.RequestException as exc:
        return None, f"Backend request failed: {exc}"
    except ValueError as exc:
        return None, f"Backend returned invalid JSON: {exc}"


def _success_response(data: Dict, http_status: int = 200, message: str = ""):
    """
    Build the unified success envelope.

    IMPORTANT: `message` is the ONLY message field in the entire response.
    It should always be the LLM-generated intro sentence (or a meaningful
    one-liner for health/classify endpoints).
    The `data` dict must NEVER contain its own 'message' key.
    """
    return jsonify({
        "status":     True,
        "httpStatus": http_status,
        "message":    message,
        "data":       data,
    }), http_status


def _error_response(message: str, http_status: int = 400, data: Optional[Dict] = None):
    """Build the unified error envelope."""
    return jsonify({
        "status":     False,
        "httpStatus": http_status,
        "message":    message,
        "data":       data or {},
    }), http_status


def _handle_general_question(
    *,
    message: str,
    session_id: str,
    elapsed_ms_fn,
):
    """
    Handle questions that are not admin data queries (greetings, Agniveer
    knowledge questions, general conversation).

    Response shape:
      {
        "status":     true,
        "httpStatus": 200,
        "message":    "<answer from LLM>",   ← single message, the LLM answer
        "data":       { "type": "general", "sessionId": "..." }
      }
    """
    try:
        from config import (
            classify_intent,
            detect_answer_style,
            _fuzzy_normalize_query,
            _is_date_query,
            _get_current_date_response,
            CHAT_SYSTEM_PROMPT,
            GENERAL_KNOWLEDGE_FALLBACK_PROMPT,
            MAX_TOKENS_STYLE,
            MAX_TOKENS_DEFAULT,
            REFERENCE_FALLBACK,
            TOP_K,
        )
        from rag import (
            prepare_rag_bundle,
            is_reasoning_query,
            deterministic_policy_answer,
            get_cached_response,
            set_cached_response,
            make_response_cache_key,
            LOW_RETRIEVAL_CONFIDENCE,
            STRICT_TOP_K,
            build_context,
        )
        from ollama_cpu_chat import (
            MODEL_NAME as DEFAULT_MODEL,
            chat_with_fallback,
            PartialResponseError,
        )
        from config import (
            STRICT_RAG_PROMPT,
            STRICT_RAG_PROMPT_COMPUTE,
            style_structure_instruction,
            trim_to_complete_sentence,
        )
        import requests as _req
    except ImportError as exc:
        logger.warning("Could not import chat pipeline for general question: %s", exc)
        return _error_response(
            "I can answer general questions too, but the chat module is currently unavailable.",
            503,
        )

    q = _fuzzy_normalize_query(message)
    cache_key = make_response_cache_key(
        q, style="elaborate", model=DEFAULT_MODEL,
        context="admin_general", session_id=session_id,
    )

    # ── Build response_data WITHOUT a "message" key ────────────────────────
    def _make_data() -> Dict[str, Any]:
        d: Dict[str, Any] = {"type": "general"}
        if session_id and session_id != "admin-default":
            d["sessionId"] = session_id
        return d

    if _is_date_query(q):
        answer = _get_current_date_response()
        # single message at outer level only
        return _success_response(_make_data(), message=answer)

    style_name, _ = detect_answer_style(q)
    intent        = classify_intent(q)
    token_limit   = MAX_TOKENS_STYLE.get(style_name, MAX_TOKENS_DEFAULT)
    session       = _req.Session()

    def _finalize(text: str) -> str:
        final = trim_to_complete_sentence(text or "")
        return final or REFERENCE_FALLBACK

    def _build_rag_messages(query, context, reasoning):
        system = STRICT_RAG_PROMPT_COMPUTE if reasoning else STRICT_RAG_PROMPT
        system = f"{system}\n\n{style_structure_instruction(style_name)}"
        msgs = [{"role": "system", "content": system}]
        if context.strip():
            user_content = (
                f"Reference information:\n{context}\n\n"
                f"Question: {query}\n\n"
                "Using ONLY the reference information above, write a complete answer. "
                "Do not use any knowledge outside the reference information."
            )
        else:
            user_content = query
        msgs.append({"role": "user", "content": user_content})
        return msgs

    def _build_chat_messages(query):
        return [
            {"role": "system", "content": CHAT_SYSTEM_PROMPT},
            {"role": "user",   "content": query},
        ]

    def _build_general_messages(query):
        q_lower = query.lower()
        factual_signals = (
            "what is", "what are", "who is", "who was", "when did", "how does",
            "explain", "define", "how many", "how much", "capital of",
        )
        is_factual = any(s in q_lower for s in factual_signals)
        if is_factual:
            system = (
                f"{CHAT_SYSTEM_PROMPT}\n\n{GENERAL_KNOWLEDGE_FALLBACK_PROMPT}\n\n"
                "Answer from general knowledge if confident. Be conservative with numbers."
            )
        else:
            system = (
                f"{CHAT_SYSTEM_PROMPT}\n\n"
                "Respond naturally like a warm, knowledgeable assistant. "
                "No bullet points for casual replies."
            )
        return [
            {"role": "system", "content": system},
            {"role": "user",   "content": query},
        ]

    try:
        if intent == "rag":
            bundle = prepare_rag_bundle(
                q, top_k=TOP_K, style=style_name, include_points=False,
            )
            context   = bundle.get("context", "") if isinstance(bundle, dict) else ""
            reasoning = bool(bundle.get("reasoning", False)) if isinstance(bundle, dict) else False

            det_answer = deterministic_policy_answer(q, context)
            if det_answer:
                return _success_response(_make_data(), message=det_answer)

            cache_key = make_response_cache_key(
                q, style=style_name, model=DEFAULT_MODEL,
                context=context, session_id=session_id,
            )
            cached = get_cached_response(cache_key)
            if cached:
                return _success_response(_make_data(), message=cached)

            if context.strip():
                messages = _build_rag_messages(q, context, reasoning)
            else:
                messages = _build_general_messages(q)

        elif intent == "chat":
            messages  = _build_chat_messages(q)
            cache_key = make_response_cache_key(
                q, style=style_name, model=DEFAULT_MODEL,
                context="chat", session_id=session_id,
            )
            cached = get_cached_response(cache_key)
            if cached:
                return _success_response(_make_data(), message=cached)

        else:
            messages  = _build_general_messages(q)
            cache_key = make_response_cache_key(
                q, style=style_name, model=DEFAULT_MODEL,
                context="general", session_id=session_id,
            )
            cached = get_cached_response(cache_key)
            if cached:
                return _success_response(_make_data(), message=cached)

        result = chat_with_fallback(
            session, DEFAULT_MODEL, messages,
            stream_tokens=False, max_tokens_override=token_limit,
        )
        answer = _finalize(result.text)

    except PartialResponseError as exc:
        answer = _finalize(exc.partial_text or REFERENCE_FALLBACK)
    except Exception as exc:
        logger.warning("General question LLM call failed: %s", exc)
        answer = REFERENCE_FALLBACK

    try:
        set_cached_response(cache_key, answer)
    except Exception:
        pass

    # single message at outer level only — _make_data() has NO "message" key
    return _success_response(_make_data(), message=answer)


# =============================================================================
# ROUTES
# =============================================================================

@admin_bp.route("/health")
def admin_health():
    """Quick health check for the admin chatbot subsystem."""
    dotnet_ok = True
    try:
        resp = _dotnet_session.get(
            f"{DOTNET_API_BASE_URL}/api/health",
            timeout=5,
            verify=DOTNET_VERIFY_SSL,
        )
        dotnet_ok = resp.status_code < 400
    except Exception:
        dotnet_ok = False

    return _success_response(
        {
            "pythonStatus": "ok",
            "dotnetBackend": "reachable" if dotnet_ok else "unreachable",
            "dotnetUrl":     DOTNET_EXECUTE_URL,
            "pythonPort":    5000,
        },
        message="Admin health check complete.",
    )


@admin_bp.route("/classify", methods=["POST"])
def admin_classify():
    """
    Classify-only endpoint — returns intent JSON and .NET payload
    without calling .NET. Useful for debugging.
    """
    body       = request.get_json(force=True, silent=True) or {}
    message    = (body.get("message") or "").strip()

    if not message:
        return _error_response("message field is required.", 400)

    # ── Greeting short-circuit ─────────────────────────────────────────────────
    session_id = _get_session_id(body)

    # ── Greeting short-circuit ─────────────────────────────────────────────────
    if _is_greeting(message.lower().strip().rstrip("!?.,")):
        commander_name = (
            body.get("commanderName") or body.get("commander_name") or body.get("name") or "Officer"
        ).strip().title()
        commander_rank = (
            body.get("commanderRank") or body.get("commander_rank") or body.get("rank") or ""
        ).strip()

        import datetime as _dt
        hour = _dt.datetime.now().hour
        time_greeting = (
            "Good Morning" if 5 <= hour < 12 else
            "Good Afternoon" if 12 <= hour < 17 else
            "Good Evening"
        )

        _RANK_SHORT = {
            "colonel": "Col", "lieutenant colonel": "Lt Col", "major": "Maj",
            "captain": "Capt", "brigadier": "Brig", "general": "Gen",
            "major general": "Maj Gen", "lieutenant general": "Lt Gen",
            "platoon commander": "Plt Cdr", "commanding officer": "CO",
            "wing commander": "Wg Cdr", "squadron leader": "Sqn Ldr",
        }
        short_rank = _RANK_SHORT.get(commander_rank.lower(), commander_rank)
        salutation = f"{short_rank} {commander_name}".strip() if short_rank else commander_name

        welcome = (
            f"{time_greeting}, {salutation}. "
            f"Welcome to AgniAI Command Intelligence. "
            f"I'm ready to assist you with personnel performance, attendance, "
            f"leave records, medical data, equipment status, and more. "
            f"How can I help you today?"
        )

        response_data: Dict[str, Any] = {"type": "greeting"}
        if session_id and session_id != "admin-default":
            response_data["sessionId"] = session_id

        return _success_response(response_data, message=welcome)

    # ── Step 1: Classify intent ────────────────────────────────────────────────
    intent_result = classify_admin_intent(message)

    id_filters     = _get_id_filters(body)
    intent_result  = classify_admin_intent(message)
    dotnet_payload = format_admin_payload(intent_result)
    dotnet_payload.update(id_filters)

    return _success_response(
        {
            "intent":        intent_result,
            "dotnetPayload": dotnet_payload,
        },
        message="Intent classified successfully.",
    )


@admin_bp.route("/chat", methods=["POST"])
def admin_chat():
    """
    Main admin chat endpoint.

    Full pipeline:
      1. Classify intent from admin's natural-language question
      2. Build .NET payload (camelCase)
      3. POST to .NET AiCommand/execute
      4. Generate natural-language intro via LLM
      5. Return unified structured response

    RESPONSE SHAPE — single message field:
      {
        "status":     true,
        "httpStatus": 200,
        "message":    "<LLM intro sentence>",   ← THE only message
        "data": {
          "intent":        { ... },
          "dotnetPayload": { ... },
          "data":          [ ... ],
          "commandLabel":  "...",
          "success":       true,
          "sessionId":     "..."                 ← only if sent by frontend
        }
      }
    """
    start_time = time.time()
    body       = request.get_json(force=True, silent=True) or {}
    message    = (body.get("message") or "").strip()
    session_id = _get_session_id(body)
    id_filters = _get_id_filters(body)

    if not message:
        return _error_response("message field is required and cannot be empty.", 400)

    # ── Step 1: Classify intent ────────────────────────────────────────────
    intent_result = classify_admin_intent(message)
    logger.info(
        "Admin intent: session=%s category=%s subcategory=%s confidence=%s",
        session_id,
        intent_result.get("category"),
        intent_result.get("subcategory"),
        intent_result.get("confidence"),
    )

    elapsed_ms = lambda: round((time.time() - start_time) * 1000)

    # ── Unrecognised query → RAG / general chat ───────────────────────────
    if intent_result.get("category") is None:
        return _handle_general_question(
            message=message,
            session_id=session_id,
            elapsed_ms_fn=elapsed_ms,
        )

    # ── Step 2: Build .NET payload ─────────────────────────────────────────
    dotnet_payload = format_admin_payload(intent_result)
    dotnet_payload.update(id_filters)

    logger.info("Sending to .NET: %s", json.dumps(dotnet_payload))

    # ── Step 3: Call .NET backend ──────────────────────────────────────────
    dotnet_data, dotnet_error = _call_dotnet(dotnet_payload)

    if dotnet_error:
        logger.warning("Admin .NET call failed: %s", dotnet_error)
        return _error_response(
            "Unable to fetch data at the moment. Please try again shortly.",
            502,
        )

    # ── Step 4: Generate natural-language intro (single message) ──────────
    intro_message = _generate_intro_message(
        question=message,
        intent=intent_result,
        dotnet_data=dotnet_data,
    )

    # ── Step 5: Build response_data ───────────────────────────────────────
    # Shape:
    # {
    #   "dotnetPayload": { ...what was sent to .NET... },
    #   "result":        { ...raw .NET response... },
    #   "sessionId":     "..." (only if frontend sent one)
    # }
    response_data: Dict[str, Any] = {
        "dotnetPayload": dotnet_payload,
        "result":        dotnet_data if dotnet_data is not None else {},
    }

    # Include sessionId only if the frontend sent one
    if session_id and session_id != "admin-default":
        response_data["sessionId"] = session_id

    logger.info(
        "Admin chat complete: session=%s elapsed=%dms",
        session_id,
        elapsed_ms(),
    )

    # Single message at the top level — intro_message is the LLM sentence
    return _success_response(response_data, message=intro_message)