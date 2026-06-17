"""
admin_routes.py
===============
Admin chatbot Flask blueprint for AgniAI.

INTELLIGENCE LAYER PIPELINE (per architecture spec):
  User Query
    → Query Planner (query_planner.py)
    → Query Type Detection (SIMPLE / CROSS_FILTER / COMPARISON / MULTI_INDEPENDENT)
    → Intent Generation (admin_intent.py)
    → .NET API Call(s)
    → Raw .NET Responses   ← NEVER sent to frontend
    → Result Combiner      ← intersect / compare / merge / passthrough
    → Final Combined Result (source of truth)
    → Report Generator     ← intro + analysis + conclusion only, never modifies result
    → Response Builder     ← exact JSON structure per spec
    → Frontend
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import requests as _requests
from flask import Blueprint, jsonify, request

from admin_intent import admin_normalize_query, classify_admin_intent, format_admin_payload
from config import _is_greeting, _is_small_talk, _is_patriotic, GREETING_PHRASES
from query_planner import plan_query, QueryType
from result_combiner import intersect_results, merge_results, compare_results
from admin_context import AdminSessionContext
from admin_entity_resolver import resolve_entities_from_query
from admin_formatter import format_dotnet_response
from admin_report_generator import generate_admin_report

_session_context = AdminSessionContext()

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
            _limiter.limit(ADMIN_RATE_LIMIT, override_defaults=True)(admin_chat)
            logger.info("Admin rate limit applied: %s", ADMIN_RATE_LIMIT)
    except Exception as exc:
        logger.warning("Could not register admin rate limit: %s", exc)


# =============================================================================
# STEP 4 — RESULT COMBINER
# Routes raw .NET responses through the correct combination strategy.
# This is the layer that creates the "Final Combined Result" which is the
# SOURCE OF TRUTH passed to the Report Generator.
# =============================================================================

def _combine_dotnet_results(
    query_type: QueryType,
    raw_results: List[Any],
    labeled_results: List[Tuple[str, Any]],
) -> Any:
    """
    Route raw .NET results through result_combiner based on query type.

    CROSS_FILTER      → intersect_results  (N-way ID intersection)
    COMPARISON        → compare_results    (side-by-side metric comparison)
    MULTI_INDEPENDENT → merge_results      (combine independent sections)
    SIMPLE / other    → passthrough        (single result, no combination)

    The output of this function is the finalResult — the source of truth.
    It is NEVER modified by any downstream step.
    """
    if query_type == QueryType.CROSS_FILTER:
        logger.info("result_combiner: intersect_results across %d sets", len(raw_results))
        return intersect_results(raw_results, primary_index=0)

    elif query_type == QueryType.COMPARISON:
        logger.info("result_combiner: compare_results across %d sides", len(labeled_results))
        return compare_results(labeled_results)

    elif query_type == QueryType.MULTI_INDEPENDENT:
        logger.info("result_combiner: merge_results across %d sections", len(labeled_results))
        return merge_results(labeled_results)

    else:
        # SIMPLE / ANALYTICS — single result, no combination needed
        logger.info("result_combiner: simple passthrough")
        return raw_results[0] if raw_results else {}


# =============================================================================
# STEP 5 — FORMAT COMBINED RESULT
# Converts the combined result into human-readable plain text.
# Uses admin_formatter.py which handles all .NET response shapes.
# =============================================================================

def _format_combined_result(
    combined_result: Any,
    intent_result: Dict[str, Any],
) -> str:
    """
    Pass the combined result (source of truth) through format_dotnet_response()
    to produce clean human-readable plain text for the frontend message bubble.
    """
    try:
        formatted = format_dotnet_response(combined_result, intent_result)
        logger.debug("format_dotnet_response: %d chars", len(formatted or ""))
        return formatted or ""
    except Exception as exc:
        logger.warning("format_dotnet_response failed: %s", exc)
        return ""


# =============================================================================
# STEP 6 — REPORT GENERATOR
# Receives only {queryType, finalResult} and generates:
#   introMessage, analysis, conclusion
# CRITICAL: Report Generator NEVER modifies finalResult.
# =============================================================================

def _generate_report(
    user_query: str,
    query_type: QueryType,
    intent_result: Dict[str, Any],
    combined_result: Any,
) -> Dict[str, str]:
    """
    Call generate_admin_report() with the combined result (not raw .NET data).
    Returns {introMessage, analysis, conclusion}.
    The combined_result is passed read-only — report generator must not modify it.
    """
    return generate_admin_report(
        user_query=user_query,
        query_type=query_type.value,
        intent_result=intent_result,
        combined_result=combined_result,
    )


# =============================================================================
# STEP 7 — RESPONSE BUILDER
# Builds the final JSON response in the exact order specified in the spec:
#   status, queryType, introMessage, result, analysis, conclusion,
#   intent, dotnetResponses, metadata
# The message field contains all text sections joined for the frontend.
# =============================================================================

def _build_response_message(
    report: Dict[str, str],
    formatted_data: str = "",
) -> str:
    """
    Merge introMessage + formatted_data + analysis + conclusion into one
    string that the frontend reads from response.message.

    Order:
      1. introMessage   — what was fetched / what we are showing
      2. formatted_data — the actual records in readable plain-text
      3. Analysis:      — AI analysis of the data
      4. Conclusion:    — AI executive summary

    Empty sections are skipped cleanly.
    """
    parts = []

    intro = (report.get("introMessage") or "").strip()
    if intro:
        parts.append(intro)

    data_text = (formatted_data or "").strip()
    if data_text:
        parts.append(data_text)

    analysis = (report.get("analysis") or "").strip()
    if analysis:
        parts.append(f"Analysis:\n{analysis}")

    conclusion = (report.get("conclusion") or "").strip()
    if conclusion:
        parts.append(f"Conclusion:\n{conclusion}")

    return "\n\n".join(parts)


def _build_final_response(
    *,
    query_type: QueryType,
    confidence: float,
    query_plan,
    intent_result: Dict[str, Any],
    combined_result: Any,
    formatted_data: str,
    report: Dict[str, str],
    raw_dotnet_responses: List[Any],
    session_id: str,
    execution_time_ms: int,
    analytics_hint: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build the final response payload in the exact order from the architecture spec:

    {
      "status": true,
      "queryType": "...",
      "introMessage": "...",
      "result": { ...FINAL RESULT COMBINER OUTPUT... },
      "analysis": "...",
      "conclusion": "...",
      "intent": { ...intent(s)... },
      "dotnetResponses": [ ...raw .NET responses... ],
      "metadata": { "confidence": 0.95, "executionTimeMs": 0 }
    }

    The message field (for frontend text bubble) contains all text joined.
    """
    combined_message = _build_response_message(report, formatted_data)

    payload: Dict[str, Any] = {
        # 1. Status
        "status":       True,
        "httpStatus":   200,

        # 2. Query type
        "queryType":    query_type.value,

        # 3. Intro message
        "introMessage": report.get("introMessage", ""),

        # 4. Result — the final combined result (source of truth)
        "result":       combined_result if combined_result is not None else {},

        # 5. Analysis
        "analysis":     report.get("analysis", ""),

        # 6. Conclusion
        "conclusion":   report.get("conclusion", ""),

        # 7. Intent
        "intent":       intent_result,

        # 8. Raw .NET responses (for debugging / audit)
        "dotnetResponses": raw_dotnet_responses,

        # 9. Metadata
        "metadata": {
            "confidence":     round(confidence, 2),
            "executionTimeMs": execution_time_ms,
            "queryPlan":      query_plan.to_dict() if query_plan else {},
            **({"analyticsHint": analytics_hint} if analytics_hint else {}),
        },

        # Extra fields for frontend convenience
        "formattedData": formatted_data,      # human-readable text of the result
        "message":       combined_message,    # full text for message bubble
    }

    if session_id and session_id != "admin-default":
        payload["sessionId"] = session_id

    return payload


# =============================================================================
# FULL PIPELINE RUNNER
# Runs steps 4 → 5 → 6 → 7 after all .NET data has been collected.
# =============================================================================

def _run_intelligence_pipeline(
    *,
    user_query: str,
    query_type: QueryType,
    query_plan,
    raw_results: List[Any],
    labeled_results: List[Tuple[str, Any]],
    primary_intent: Dict[str, Any],
    session_id: str,
    confidence: float,
    start_time: float,
    analytics_hint: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Runs the full AgniAI intelligence pipeline after .NET data is collected:

    Step 4: result_combiner        — combine/intersect/compare raw .NET responses
    Step 5: format_dotnet_response — format combined result to readable text
    Step 6: generate_admin_report  — produce introMessage + analysis + conclusion
    Step 7: _build_final_response  — build final JSON response payload

    Returns the complete response payload dict ready for jsonify().
    """

    # ── Step 4: Result Combiner ────────────────────────────────────────────
    combined_result = _combine_dotnet_results(query_type, raw_results, labeled_results)
    logger.info(
        "Pipeline step 4 complete: result_combiner produced type=%s",
        type(combined_result).__name__,
    )

    # ── Step 5: Format combined result to human-readable text ──────────────
    formatted_data = _format_combined_result(combined_result, primary_intent)
    logger.info(
        "Pipeline step 5 complete: formatted_data=%d chars",
        len(formatted_data),
    )

    # ── Step 6: Report Generator (intro + analysis + conclusion) ───────────
    # CRITICAL: Report Generator receives combined_result READ-ONLY.
    # It only generates introMessage, analysis, conclusion — never modifies result.
    report = _generate_report(
        user_query=user_query,
        query_type=query_type,
        intent_result=primary_intent,
        combined_result=combined_result,
    )
    logger.info(
        "Pipeline step 6 complete: intro=%d analysis=%d conclusion=%d",
        len(report.get("introMessage", "")),
        len(report.get("analysis", "")),
        len(report.get("conclusion", "")),
    )

    # ── Update session context ─────────────────────────────────────────────
    _session_context.update(session_id, user_query, primary_intent, combined_result)

    # ── Step 7: Response Builder ───────────────────────────────────────────
    execution_time_ms = round((time.time() - start_time) * 1000)

    return _build_final_response(
        query_type=query_type,
        confidence=confidence,
        query_plan=query_plan,
        intent_result=primary_intent,
        combined_result=combined_result,
        formatted_data=formatted_data,
        report=report,
        raw_dotnet_responses=raw_results,   # kept for audit, never sent raw to frontend
        session_id=session_id,
        execution_time_ms=execution_time_ms,
        analytics_hint=analytics_hint,
    )


# =============================================================================
# CONVERSATIONAL DETECTION
# =============================================================================

def _is_admin_conversational(message: str) -> bool:
    cleaned = message.lower().strip().rstrip("!?.,;")

    if _is_greeting(cleaned):
        return True
    if _is_small_talk(cleaned):
        return True
    if _is_patriotic(cleaned):
        return True

    tokens = cleaned.split()
    if len(tokens) <= 3:
        _ADMIN_SIGNAL_WORDS = {
            "performance", "leave", "attendance", "medical", "equipment",
            "verification", "distribution", "skills", "top", "bottom",
            "score", "marks", "grading", "bpet", "ppt", "firing", "drill",
            "performer", "performers", "overdue", "absconded", "bmi",
            "disease", "present", "strength", "issued", "procured",
            "pending", "completed", "sport", "blood", "unit", "overall",
            "average", "pass", "fail", "improvement", "drop", "attempt",
            "comparison", "compare", "summary", "ranking",
        }
        if not any(t in _ADMIN_SIGNAL_WORDS for t in tokens):
            return True

    return False


# =============================================================================
# GREETING / CONVERSATIONAL BUILDERS
# =============================================================================

def _build_greeting_response(body: Dict, session_id: str) -> tuple:
    import datetime as _dt
    import random

    admin_name = (
        body.get("fullName") or body.get("adminName") or body.get("userName")
        or body.get("commanderName") or body.get("commander_name")
        or body.get("name") or "Officer"
    ).strip()

    hour = _dt.datetime.now().hour
    if 5 <= hour < 12:
        time_greeting = "Good Morning"
    elif 12 <= hour < 17:
        time_greeting = "Good Afternoon"
    else:
        time_greeting = "Good Evening"

    greetings = [
        f"{time_greeting}, {admin_name}! Welcome back to AgniAI Command Console. What would you like to review today?",
        f"{time_greeting}, {admin_name}. All systems are ready. How can I assist you today?",
        f"{time_greeting}, {admin_name}. AgniAI is at your service. What would you like to analyze today?",
        f"{time_greeting}, {admin_name}! Good to have you back. What insights can I pull up for you?",
        f"{time_greeting}, {admin_name}. Ready for duty. What would you like to review today?",
    ]

    response_data: Dict[str, Any] = {"type": "greeting"}
    if session_id and session_id != "admin-default":
        response_data["sessionId"] = session_id

    return response_data, random.choice(greetings)


def _build_conversational_response(message: str, body: Dict, session_id: str) -> tuple:
    import random
    import datetime as _dt

    admin_name = (
        body.get("fullName") or body.get("adminName") or body.get("userName")
        or body.get("commanderName") or body.get("commander_name")
        or body.get("name") or "Officer"
    ).strip()

    hour = _dt.datetime.now().hour
    if 5 <= hour < 12:
        time_greeting = "Good Morning"
    elif 12 <= hour < 17:
        time_greeting = "Good Afternoon"
    else:
        time_greeting = "Good Evening"

    try:
        import requests as _req
        from config import OLLAMA_URL, DEFAULT_MODEL

        prompt = (
            f"You are AgniAI Command Console — an intelligent admin assistant that helps "
            f"commanding officers review and analyze Agniveer data.\n"
            f"The admin officer's name/title is \"{admin_name}\". They sent: \"{message}\"\n"
            f"Time greeting: \"{time_greeting}\".\n\n"
            f"Start with \"{time_greeting}, {admin_name}!\" then reply warmly and "
            f"professionally in 1-2 sentences. Offer to help with Agniveer data.\n"
            f"No markdown, no bullets."
        )
        payload = {
            "model":    DEFAULT_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream":   False,
            "options":  {"temperature": 0.7, "num_predict": 80, "num_ctx": 512},
        }
        resp = _req.post(OLLAMA_URL, json=payload, timeout=(8, 30))
        resp.raise_for_status()
        llm_reply = resp.json().get("message", {}).get("content", "").strip().strip('"\'')
        if llm_reply and 5 <= len(llm_reply) <= 300:
            response_data: Dict[str, Any] = {"type": "conversational"}
            if session_id and session_id != "admin-default":
                response_data["sessionId"] = session_id
            return response_data, llm_reply
    except Exception as exc:
        logger.debug("LLM conversational reply failed, using fallback: %s", exc)

    fallbacks = [
        f"{time_greeting}, {admin_name}. I'm here and ready to help. What data would you like to review?",
        f"{time_greeting}, {admin_name}! Let me know if you need any reports or insights.",
        f"{time_greeting}, {admin_name}. All systems are operational. What would you like to look into today?",
    ]

    response_data = {"type": "conversational"}
    if session_id and session_id != "admin-default":
        response_data["sessionId"] = session_id

    return response_data, random.choice(fallbacks)


# =============================================================================
# HELPERS
# =============================================================================

def _get_session_id(data: Dict) -> str:
    from config import SESSION_HEADER
    session_id = (
        data.get("session_id") or data.get("sessionId") or
        request.headers.get(SESSION_HEADER) or
        request.headers.get("X-Session-Id") or ""
    ).strip()
    return session_id or "admin-default"


def _get_id_filters(data: Dict) -> Dict[str, int]:
    def _safe_int(value) -> Optional[int]:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    filters = {}
    command_id = _safe_int(data.get("commandId", data.get("command_id")))
    batch_id   = _safe_int(data.get("batchId",   data.get("batch_id")))
    platoon_id = _safe_int(data.get("platoonId", data.get("platoon_id")))
    company_id = _safe_int(data.get("companyId", data.get("company_id")))

    if command_id is not None: filters["commandId"] = command_id
    if batch_id   is not None: filters["batchId"]   = batch_id
    if platoon_id is not None: filters["platoonId"] = platoon_id
    if company_id is not None: filters["companyId"] = company_id
    return filters


def _get_full_name(data: Dict) -> str:
    return (data.get("fullName") or data.get("full_name") or "").strip()


def _call_dotnet(payload: Dict) -> tuple[Any, Optional[str]]:
    headers = {"Content-Type": "application/json"}
    if DOTNET_API_KEY:
        headers["X-Api-Key"] = DOTNET_API_KEY

    logger.debug("Calling .NET: payload=%s", json.dumps(payload))

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
        return None, f"Cannot connect to .NET backend at {DOTNET_EXECUTE_URL}. ({exc})"
    except _requests.Timeout:
        return None, f"Backend timed out after {DOTNET_TIMEOUT}s."
    except _requests.RequestException as exc:
        return None, f"Backend request failed: {exc}"
    except ValueError as exc:
        return None, f"Backend returned invalid JSON: {exc}"


def _success_response(data: Dict, http_status: int = 200, message: str = ""):
    """Return a flat JSON response. All keys from data are at root level."""
    payload: Dict[str, Any] = {
        "status":     True,
        "httpStatus": http_status,
        "message":    message,
    }
    payload.update(data)
    return jsonify(payload), http_status


def _error_response(message: str, http_status: int = 400, data: Optional[Dict] = None):
    return jsonify({
        "status":     False,
        "httpStatus": http_status,
        "message":    message,
        "data":       data or {},
    }), http_status


# =============================================================================
# ROUTES
# =============================================================================

@admin_bp.route("/health")
def admin_health():
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
            "pythonStatus":  "ok",
            "dotnetBackend": "reachable" if dotnet_ok else "unreachable",
            "dotnetUrl":     DOTNET_EXECUTE_URL,
            "pythonPort":    5000,
        },
        message="Admin health check complete.",
    )


@admin_bp.route("/classify", methods=["POST"])
def admin_classify():
    """Classify-only endpoint — returns intent JSON without calling .NET. For debugging."""
    body    = request.get_json(force=True, silent=True) or {}
    message = (body.get("message") or "").strip()

    if not message:
        return _error_response("message field is required.", 400)

    session_id = _get_session_id(body)

    if _is_admin_conversational(message):
        cleaned = message.lower().strip().rstrip("!?.,;")
        if _is_greeting(cleaned):
            response_data, greeting_message = _build_greeting_response(body, session_id)
        else:
            response_data, greeting_message = _build_conversational_response(message, body, session_id)
        return _success_response(response_data, message=greeting_message)

    id_filters = _get_id_filters(body)
    full_name  = _get_full_name(body)

    resolved_entities = resolve_entities_from_query(
        message,
        existing_company_id=id_filters.get("companyId"),
        existing_platoon_id=id_filters.get("platoonId"),
    )
    if resolved_entities.get("companyId") is not None:
        id_filters["companyId"] = resolved_entities["companyId"]
    if resolved_entities.get("platoonId") is not None:
        id_filters["platoonId"] = resolved_entities["platoonId"]

    message        = admin_normalize_query(message)
    intent_result  = classify_admin_intent(message)
    dotnet_payload = format_admin_payload(intent_result)
    dotnet_payload.update(id_filters)
    if full_name:
        dotnet_payload["fullName"] = full_name

    query_plan = plan_query(message)

    response_data: Dict[str, Any] = {
        "queryType":     query_plan.query_type.value,
        "confidence":    round(query_plan.confidence, 2),
        "queryPlan":     query_plan.to_dict(),
        "intent":        intent_result,
        "dotnetPayload": dotnet_payload,
    }
    if session_id and session_id != "admin-default":
        response_data["sessionId"] = session_id

    return _success_response(response_data, message="Intent classified successfully.")


@admin_bp.route("/chat", methods=["POST"])
def admin_chat():
    """
    Main admin chat endpoint.

    Full intelligence pipeline:
      1. Validate + normalize input
      2. Resolve named entities (company/platoon names → IDs)
      3. Query Planner: detect query type + generate all required intents
      4. Execute .NET API call(s) — one per sub-operation
      5. Result Combiner: combine raw .NET responses → finalResult
      6. Format finalResult → human-readable text
      7. Report Generator: intro + analysis + conclusion from finalResult
      8. Response Builder: structured JSON response
      9. Return response (frontend reads response.message for the text bubble)
    """
    start_time = time.time()
    body       = request.get_json(force=True, silent=True) or {}
    message    = (body.get("message") or "").strip()
    session_id = _get_session_id(body)
    id_filters = _get_id_filters(body)
    full_name  = _get_full_name(body)

    # ── Greeting / conversational short-circuit ─────────────────────────────
    if _is_admin_conversational(message):
        cleaned = message.lower().strip().rstrip("!?.,;")
        if _is_greeting(cleaned):
            response_data, greeting_message = _build_greeting_response(body, session_id)
        else:
            response_data, greeting_message = _build_conversational_response(message, body, session_id)
        return _success_response(response_data, message=greeting_message)

    if not message:
        return _error_response("message field is required and cannot be empty.", 400)

    # ── Step 2: Resolve Named Entities ─────────────────────────────────────
    resolved_entities = resolve_entities_from_query(
        message,
        existing_company_id=id_filters.get("companyId"),
        existing_platoon_id=id_filters.get("platoonId"),
    )
    if resolved_entities.get("companyId") is not None:
        id_filters["companyId"] = resolved_entities["companyId"]
    if resolved_entities.get("platoonId") is not None:
        id_filters["platoonId"] = resolved_entities["platoonId"]

    # ── Step 3: Query Planner ───────────────────────────────────────────────
    message    = admin_normalize_query(message)
    query_plan = plan_query(message)

    logger.info(
        "Query plan: session=%s type=%s confidence=%.2f ops=%d reason=%s",
        session_id,
        query_plan.query_type.value,
        query_plan.confidence,
        len(query_plan.operations),
        query_plan.reasoning,
    )

    # ── Step 4: Execute .NET API Call(s) ───────────────────────────────────
    raw_results:     List[Any]             = []
    labeled_results: List[Tuple[str, Any]] = []
    primary_intent:  Dict[str, Any]        = {}

    if (query_plan.query_type != QueryType.SIMPLE
            and query_plan.confidence >= 0.5
            and len(query_plan.operations) >= 2):

        # MULTI-OP: one .NET call per sub-operation
        logger.info(
            "Multi-op: session=%s type=%s ops=%d",
            session_id, query_plan.query_type.value, len(query_plan.operations),
        )

        for i, op in enumerate(query_plan.operations):
            payload = dict(op.dotnet_payload)
            payload.update(id_filters)
            if full_name:
                payload["fullName"] = full_name

            logger.info(
                "Multi-op %d/%d → .NET: %s",
                i + 1, len(query_plan.operations), json.dumps(payload),
            )

            dotnet_data, dotnet_error = _call_dotnet(payload)
            if dotnet_error:
                logger.warning("Multi-op %d/%d failed: %s", i + 1, len(query_plan.operations), dotnet_error)
                return _error_response(
                    "Unable to fetch data at the moment. Please try again shortly.", 502
                )

            raw_results.append(dotnet_data)
            label = op.intent_result.get("category", f"Query {i + 1}")
            labeled_results.append((label, dotnet_data))

        primary_intent = query_plan.operations[0].intent_result

    else:
        # SIMPLE / ANALYTICS: single .NET call
        # For ANALYTICS, we use the intent from the first (and only) planned operation
        # so that group_by and analyticsHint are preserved in the payload.
        if (query_plan.query_type == QueryType.ANALYTICS
                and query_plan.operations
                and query_plan.operations[0].intent_result.get("category")):
            # Use the planner's pre-built intent — it already has group_by etc.
            primary_intent = query_plan.operations[0].intent_result
            intent_result  = primary_intent
            logger.info(
                "Analytics intent: session=%s category=%s subcategory=%s hint=%s group_by=%s",
                session_id,
                intent_result.get("category"),
                intent_result.get("subcategory"),
                query_plan.analytics_hint,
                getattr(query_plan.operations[0], "group_by", None),
            )
        else:
            intent_result = classify_admin_intent(message)
            primary_intent = intent_result
            logger.info(
                "Intent: session=%s category=%s subcategory=%s confidence=%s",
                session_id,
                intent_result.get("category"),
                intent_result.get("subcategory"),
                intent_result.get("confidence"),
            )

        # Unrecognised query
        if intent_result.get("category") is None:
            unrecognised_msg = (
                "Sorry, I was unable to understand your request. "
                "I can help with Performance, Leave, Attendance, Medical, Equipment, "
                "Verification, Distribution, and Skills information. "
                "Please ask a relevant question."
            )
            response_data = {
                "queryType":      query_plan.query_type.value,
                "introMessage":   unrecognised_msg,
                "result":         {},
                "analysis":       "",
                "conclusion":     "",
                "intent":         intent_result,
                "dotnetResponses": [],
                "formattedData":  "",
                "metadata": {
                    "confidence":     round(query_plan.confidence, 2),
                    "executionTimeMs": round((time.time() - start_time) * 1000),
                    "queryPlan":      query_plan.to_dict(),
                },
            }
            if session_id and session_id != "admin-default":
                response_data["sessionId"] = session_id
            return _success_response(response_data, message=unrecognised_msg)

        dotnet_payload = format_admin_payload(intent_result)
        dotnet_payload.update(id_filters)
        if full_name:
            dotnet_payload["fullName"] = full_name

        # For ANALYTICS queries, pass group_by and analyticsHint to .NET
        # so it can group/aggregate server-side if supported.
        if query_plan.query_type == QueryType.ANALYTICS and query_plan.operations:
            op = query_plan.operations[0]
            if getattr(op, "group_by", None):
                dotnet_payload["groupBy"] = op.group_by
            if query_plan.analytics_hint:
                dotnet_payload["analyticsHint"] = query_plan.analytics_hint

        logger.info("Sending to .NET: %s", json.dumps(dotnet_payload))

        dotnet_data, dotnet_error = _call_dotnet(dotnet_payload)
        if dotnet_error:
            logger.warning("Admin .NET call failed: %s", dotnet_error)
            return _error_response(
                "Unable to fetch data at the moment. Please try again shortly.", 502
            )

        raw_results     = [dotnet_data]
        labeled_results = [(intent_result.get("category", "Result"), dotnet_data)]
        primary_intent  = intent_result

    # ── Steps 5–8: Run Full Intelligence Pipeline ──────────────────────────
    # raw_results and labeled_results contain the raw .NET data.
    # The pipeline processes them through result_combiner → formatter →
    # report_generator → response_builder.
    # Raw .NET data is NEVER sent directly to the frontend.
    pipeline_response = _run_intelligence_pipeline(
        user_query=message,
        query_type=query_plan.query_type,
        query_plan=query_plan,
        raw_results=raw_results,
        labeled_results=labeled_results,
        primary_intent=primary_intent,
        session_id=session_id,
        confidence=query_plan.confidence,
        start_time=start_time,
        analytics_hint=query_plan.analytics_hint if hasattr(query_plan, "analytics_hint") else None,
    )

    logger.info(
        "Admin chat complete: session=%s elapsed=%dms",
        session_id,
        pipeline_response.get("metadata", {}).get("executionTimeMs", 0),
    )

    # ── Return final response ──────────────────────────────────────────────
    # response.message contains the full text bubble for the frontend.
    # response.result contains the structured combined result.
    # response.dotnetResponses contains raw .NET data (for debugging only).
    combined_message = pipeline_response.pop("message", "")
    return _success_response(pipeline_response, message=combined_message)