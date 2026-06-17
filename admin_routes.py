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
  2. Resolve company and platoon mentions from query to numeric IDs
  3. classify_admin_intent()    → structured intent dict (Python internal)
  4. format_admin_payload()     → camelCase JSON payload for .NET
     └─ includes commandId, batchId, platoonId, companyId, fullName from the frontend request
  5. POST payload to .NET       → https://<DOTNET_API_BASE_URL>/api/AiCommand/execute
  6. generate_intro_message()   → LLM generates a single clean intro sentence
  7. Return raw .NET data directly in data.result — NO formatting layer
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, Optional

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
            _limiter.limit(
                ADMIN_RATE_LIMIT,
                override_defaults=True,
            )(admin_chat)
            logger.info("Admin rate limit applied: %s", ADMIN_RATE_LIMIT)
    except Exception as exc:
        logger.warning("Could not register admin rate limit: %s", exc)


# =============================================================================
# CONVERSATIONAL DETECTION
# =============================================================================

def _is_admin_conversational(message: str) -> bool:
    """Return True if the message is casual / conversational and not a data query."""
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
# GREETING BUILDER
# =============================================================================

def _build_greeting_response(body: Dict, session_id: str) -> tuple:
    import datetime as _dt
    import random

    admin_name = (
        body.get("fullName")
        or body.get("adminName")
        or body.get("userName")
        or body.get("commanderName")
        or body.get("commander_name")
        or body.get("name")
        or "Officer"
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


# =============================================================================
# CONVERSATIONAL RESPONSE BUILDER
# =============================================================================

def _build_conversational_response(message: str, body: Dict, session_id: str) -> tuple:
    """Generate a natural conversational reply for non-greeting casual messages."""
    import random
    import datetime as _dt

    admin_name = (
        body.get("fullName")
        or body.get("adminName")
        or body.get("userName")
        or body.get("commanderName")
        or body.get("commander_name")
        or body.get("name")
        or "Officer"
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
            f"commanding officers review and analyze Agniveer data such as Performance, "
            f"Attendance, Leave, Medical, Equipment, Verification, Distribution, and Skills.\n"
            f"The admin officer's name/title is \"{admin_name}\". They sent this casual message: \"{message}\"\n"
            f"The current time of day greeting is \"{time_greeting}\".\n\n"
            f"IMPORTANT: Start your reply with \"{time_greeting}, {admin_name}!\" or \"{time_greeting}, {admin_name}.\" "
            f"then continue naturally.\n"
            f"Reply warmly, professionally, and naturally in 1-2 sentences as a command console assistant would. "
            f"Be respectful and military-professional in tone. If they asked how you are, respond naturally. "
            f"If they said thanks, acknowledge it warmly. If they said something patriotic, match that energy with pride. "
            f"End by offering to help with Agniveer data, reports, or analytics they may need.\n"
            f"Do NOT use markdown, bullets, or headers. Do NOT be robotic. Do NOT mention aspirants or recruitment."
        )
        payload = {
            "model":    DEFAULT_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream":   False,
            "options": {
                "temperature": 0.7,
                "num_predict": 80,
                "num_ctx":     512,
            },
        }
        resp = _req.post(OLLAMA_URL, json=payload, timeout=(8, 30))
        resp.raise_for_status()
        llm_reply = (
            resp.json()
            .get("message", {})
            .get("content", "")
            .strip()
            .strip('"\'')
        )
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
        f"{time_greeting}, {admin_name}. At your service — feel free to ask about Performance, Attendance, Leave, or any other module.",
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
        data.get("session_id") or
        data.get("sessionId") or
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

    if command_id is not None:
        filters["commandId"] = command_id
    if batch_id is not None:
        filters["batchId"] = batch_id
    if platoon_id is not None:
        filters["platoonId"] = platoon_id
    if company_id is not None:
        filters["companyId"] = company_id

    return filters


def _get_full_name(data: Dict) -> str:
    return (
        data.get("fullName") or
        data.get("full_name") or
        ""
    ).strip()


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


def _execute_multi_operation(
    query_plan,
    id_filters: Dict,
    full_name: str,
) -> tuple[Optional[Dict], Optional[str]]:
    """
    Execute a multi-operation query plan.
    """
    results = []
    labeled_results = []

    for i, op in enumerate(query_plan.operations):
        payload = dict(op.dotnet_payload)
        payload.update(id_filters)
        if full_name:
            payload["fullName"] = full_name

        logger.info(
            "Multi-op %d/%d: sending to .NET: %s",
            i + 1, len(query_plan.operations), json.dumps(payload),
        )

        data, error = _call_dotnet(payload)
        if error:
            logger.warning(
                "Multi-op %d/%d failed: %s", i + 1, len(query_plan.operations), error,
            )
            return None, f"Sub-query {i + 1} failed: {error}"

        results.append(data)
        label = op.intent_result.get("category", f"Query {i + 1}")
        labeled_results.append((label, data))

    if query_plan.query_type == QueryType.CROSS_FILTER:
        combined = intersect_results(results, primary_index=0)
    elif query_plan.query_type == QueryType.COMPARISON:
        combined = compare_results(labeled_results)
    elif query_plan.query_type == QueryType.MULTI_INDEPENDENT:
        combined = merge_results(labeled_results)
    else:
        combined = results[0] if results else {}

    return combined, None


def _success_response(data: Dict, http_status: int = 200, message: str = ""):
    """
    Flatten all keys from data directly into the response payload so that
    analysis, conclusion, introMessage, queryType, confidence, data (dotnet records)
    etc. are all accessible at the root level of the JSON response.

    Before (broken):
        { "status": true, "message": "intro", "data": { "analysis": "...", "conclusion": "...", "data": {...} } }

    After (fixed):
        { "status": true, "message": "intro", "analysis": "...", "conclusion": "...", "data": {...} }
    """
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
    """
    Classify-only endpoint — returns intent JSON and .NET payload
    without calling .NET. Useful for debugging.
    """
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
        existing_platoon_id=id_filters.get("platoonId")
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

    return _success_response(
        response_data,
        message="Intent classified successfully.",
    )


@admin_bp.route("/chat", methods=["POST"])
def admin_chat():
    """
    Main admin chat endpoint.
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

    # ── Resolve Named Entities (Company / Platoon) ─────────────────────────
    resolved_entities = resolve_entities_from_query(
        message,
        existing_company_id=id_filters.get("companyId"),
        existing_platoon_id=id_filters.get("platoonId")
    )
    if resolved_entities.get("companyId") is not None:
        id_filters["companyId"] = resolved_entities["companyId"]
    if resolved_entities.get("platoonId") is not None:
        id_filters["platoonId"] = resolved_entities["platoonId"]

    elapsed_ms = lambda: round((time.time() - start_time) * 1000)

    # ── Step 1: Normalize & plan ────────────────────────────────────────────
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

    # ── Multi-operation path ────────────────────────────────────────────────
    if (query_plan.query_type != QueryType.SIMPLE
            and query_plan.confidence >= 0.5
            and len(query_plan.operations) >= 2):

        primary_intent  = query_plan.operations[0].intent_result
        primary_payload = query_plan.operations[0].dotnet_payload

        logger.info(
            "Admin multi-op: session=%s type=%s ops=%d",
            session_id, query_plan.query_type.value, len(query_plan.operations),
        )

        combined_data, multi_error = _execute_multi_operation(
            query_plan, id_filters, full_name,
        )

        if multi_error:
            logger.warning("Admin multi-op failed: %s", multi_error)
            return _error_response(
                "Unable to fetch data at the moment. Please try again shortly.",
                502,
            )

        report = generate_admin_report(
            user_query=message,
            query_type=query_plan.query_type.value,
            intent_result=primary_intent,
            combined_result=combined_data
        )

        if combined_data is not None:
            _session_context.update(session_id, message, primary_intent, combined_data)

        response_data: Dict[str, Any] = {
            "queryType":    query_plan.query_type.value,
            "confidence":   round(query_plan.confidence, 2),
            "introMessage": report["introMessage"],
            "analysis":     report["analysis"],
            "conclusion":   report["conclusion"],
            "data":         combined_data if combined_data is not None else {},
            "queryPlan":    query_plan.to_dict(),
        }

        if session_id and session_id != "admin-default":
            response_data["sessionId"] = session_id

        logger.info(
            "Admin multi-op complete: session=%s elapsed=%dms",
            session_id, elapsed_ms(),
        )

        return _success_response(response_data, message=report["introMessage"])

    # ══════════════════════════════════════════════════════════════════════
    # SIMPLE PATH
    # ══════════════════════════════════════════════════════════════════════

    # ── Step 1: Classify intent ────────────────────────────────────────────
    intent_result = classify_admin_intent(message)
    logger.info(
        "Admin intent: session=%s category=%s subcategory=%s confidence=%s",
        session_id,
        intent_result.get("category"),
        intent_result.get("subcategory"),
        intent_result.get("confidence"),
    )

    # ── Unrecognised query ─────────────────────────────────────────────────
    if intent_result.get("category") is None:
        response_data = {
            "queryType":     query_plan.query_type.value,
            "confidence":    round(query_plan.confidence, 2),
            "queryPlan":     query_plan.to_dict(),
            "dotnetPayload": {},
            "result":        None,
            "intent":        intent_result,
            "introMessage":  "",
            "analysis":      "",
            "conclusion":    "",
            "data":          {},
        }
        if session_id and session_id != "admin-default":
            response_data["sessionId"] = session_id
        return _success_response(
            response_data,
            message=(
                "Sorry, I was unable to understand your request. "
                "I can help with Performance, Leave, Attendance, Medical, Equipment, "
                "Verification, Distribution, and Skills information. "
                "Please ask a relevant question."
            ),
        )

    # ── Step 2: Build .NET payload ─────────────────────────────────────────
    dotnet_payload = format_admin_payload(intent_result)
    dotnet_payload.update(id_filters)

    if full_name:
        dotnet_payload["fullName"] = full_name

    logger.info("Sending to .NET: %s", json.dumps(dotnet_payload))

    # ── Step 3: Call .NET backend ──────────────────────────────────────────
    dotnet_data, dotnet_error = _call_dotnet(dotnet_payload)

    if dotnet_error:
        logger.warning("Admin .NET call failed: %s", dotnet_error)
        return _error_response(
            "Unable to fetch data at the moment. Please try again shortly.",
            502,
        )

    # ── Step 4: Generate report (intro + analysis + conclusion) ────────────
    report = generate_admin_report(
        user_query=message,
        query_type=query_plan.query_type.value,
        intent_result=intent_result,
        combined_result=dotnet_data
    )

    if dotnet_data is not None:
        _session_context.update(session_id, message, intent_result, dotnet_data)

    # ── Step 5: Build response — all fields flat at root level ─────────────
    response_data: Dict[str, Any] = {
        "queryType":    query_plan.query_type.value,
        "confidence":   round(query_plan.confidence, 2),
        "introMessage": report["introMessage"],
        "analysis":     report["analysis"],
        "conclusion":   report["conclusion"],
        "data":         dotnet_data if dotnet_data is not None else {},
        "queryPlan":    query_plan.to_dict(),
    }

    if session_id and session_id != "admin-default":
        response_data["sessionId"] = session_id

    logger.info(
        "Admin chat complete: session=%s elapsed=%dms",
        session_id,
        elapsed_ms(),
    )

    return _success_response(response_data, message=report["introMessage"])