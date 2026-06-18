"""
admin_routes.py
===============

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
from report_generator import generate_report
from response_builder import build_response

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


def map_query_type(qt: QueryType) -> str:
    if qt == QueryType.CROSS_FILTER:
        return "cross_filter"
    elif qt == QueryType.COMPARISON:
        return "comparison"
    elif qt == QueryType.MULTI_INDEPENDENT:
        return "multi_independent"
    else:
        return "simple"


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
    Strict response pipeline processing.
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
    qtype_str:       str                   = "simple"
    operation_count: int                   = 1

    if (query_plan.query_type != QueryType.SIMPLE
            and query_plan.confidence >= 0.5
            and len(query_plan.operations) >= 2):

        # MULTI-OP: one .NET call per sub-operation
        qtype_str = map_query_type(query_plan.query_type)
        operation_count = len(query_plan.operations)
        logger.info(
            "Multi-op: session=%s type=%s ops=%d",
            session_id, qtype_str, operation_count,
        )

        for i, op in enumerate(query_plan.operations):
            payload = dict(op.dotnet_payload)
            payload.update(id_filters)
            if full_name:
                payload["fullName"] = full_name

            logger.info(
                "Multi-op %d/%d → .NET: %s",
                i + 1, operation_count, json.dumps(payload),
            )

            dotnet_data, dotnet_error = _call_dotnet(payload)
            if dotnet_error:
                logger.warning("Multi-op %d/%d failed: %s", i + 1, operation_count, dotnet_error)
                return _error_response(
                    "Unable to fetch data at the moment. Please try again shortly.", 502
                )

            raw_results.append(dotnet_data)
            label = op.intent_result.get("category", f"Query {i + 1}")
            labeled_results.append((label, dotnet_data))

        primary_intent = query_plan.operations[0].intent_result

    else:
        # SIMPLE / ANALYTICS: single .NET call
        qtype_str = "simple"
        operation_count = 1
        
        if (query_plan.query_type == QueryType.ANALYTICS
                and query_plan.operations
                and query_plan.operations[0].intent_result.get("category")):
            primary_intent = query_plan.operations[0].intent_result
            logger.info(
                "Analytics intent: session=%s category=%s subcategory=%s hint=%s",
                session_id,
                primary_intent.get("category"),
                primary_intent.get("subcategory"),
                query_plan.analytics_hint,
            )
        else:
            primary_intent = classify_admin_intent(message)
            logger.info(
                "Intent: session=%s category=%s subcategory=%s confidence=%s",
                session_id,
                primary_intent.get("category"),
                primary_intent.get("subcategory"),
                primary_intent.get("confidence"),
            )

        # Unrecognised query
        if primary_intent.get("category") is None:
            unrecognised_msg = (
                "Sorry, I was unable to understand your request. "
                "I can help with Performance, Leave, Attendance, Medical, Equipment, "
                "Verification, Distribution, and Skills information. "
                "Please ask a relevant question."
            )
            response_payload = build_response(
                query_type=qtype_str,
                intro_message=unrecognised_msg,
                combined_result={},
                analysis={"summary": "", "observations": [], "insights": []},
                conclusion={"summary": ""},
                intent=primary_intent,
                raw_results=[],
                confidence=query_plan.confidence,
                operation_count=0,
                formatted_data="",
                session_id=session_id,
            )
            combined_message = response_payload.pop("message", "")
            return _success_response(response_payload, message=combined_message)

        dotnet_payload = format_admin_payload(primary_intent)
        dotnet_payload.update(id_filters)
        if full_name:
            dotnet_payload["fullName"] = full_name

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
        labeled_results = [(primary_intent.get("category", "Result"), dotnet_data)]

    # ── Step 5: Result Combiner ────────────────────────────────────────────
    # In multi-op, combine results. In simple, pass-through.
    if qtype_str == "cross_filter":
        logger.info("result_combiner: intersect_results across %d sets", len(raw_results))
        combined_result = intersect_results(raw_results, primary_index=0)
    elif qtype_str == "comparison":
        logger.info("result_combiner: compare_results across %d sides", len(labeled_results))
        combined_result = compare_results(labeled_results)
    elif qtype_str == "multi_independent":
        logger.info("result_combiner: merge_results across %d sections", len(labeled_results))
        combined_result = merge_results(labeled_results)
    else:
        logger.info("result_combiner: simple passthrough")
        combined_result = raw_results[0] if raw_results else {}

    # ── Step 6: Format combined result to human-readable plain text ──────────
    formatted_data = format_dotnet_response(combined_result, primary_intent)

    # ── Step 7: Report Generator (intro + analysis + conclusion) ───────────
    report = generate_report(
        combined_result=combined_result,
        query_type=qtype_str,
        intent=primary_intent,
        user_query=message,
    )

    # ── Update session context ─────────────────────────────────────────────
    _session_context.update(session_id, message, primary_intent, combined_result)

    # ── Step 8: Response Builder ───────────────────────────────────────────
    response_payload = build_response(
        query_type=qtype_str,
        intro_message=report.get("introMessage", ""),
        combined_result=combined_result,
        analysis=report.get("analysis") or {},
        conclusion=report.get("conclusion") or {},
        intent=primary_intent,
        raw_results=raw_results,
        confidence=query_plan.confidence,
        operation_count=operation_count,
        formatted_data=formatted_data,
        session_id=session_id,
    )

    execution_time_ms = round((time.time() - start_time) * 1000)
    response_payload["metadata"]["executionTimeMs"] = execution_time_ms

    logger.info(
        "Admin chat complete: session=%s elapsed=%dms",
        session_id,
        execution_time_ms,
    )

    combined_message = response_payload.pop("message", "")
    return _success_response(response_payload, message=combined_message)