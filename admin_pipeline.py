"""
admin_pipeline.py
=================
Single source of truth for the AgniAI admin query execution pipeline.

Both HTTP (admin_routes.py) and WebSocket (websocket_routes.py) call
execute_admin_query() — the ONLY orchestration function.

This module owns:
  - .NET API configuration and communication
  - Conversational / greeting detection
  - The full pipeline: Planner → Intent → .NET → Combiner → Report → Response
  - Session context management

No other module should contain pipeline orchestration logic.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests as _requests

from admin_intent import admin_normalize_query, classify_admin_intent, format_admin_payload
from config import _is_greeting, _is_small_talk, _is_patriotic, GREETING_PHRASES
from query_planner import plan_query, QueryType
from result_combiner import intersect_results, merge_results, compare_results
from admin_context import AdminSessionContext
from admin_entity_resolver import resolve_entities_from_query
from admin_formatter import format_dotnet_response
from report_generator import generate_report
from response_builder import build_response

logger = logging.getLogger(__name__)

_session_context = AdminSessionContext()

# =============================================================================
# .NET CONFIGURATION
# =============================================================================

DOTNET_API_BASE_URL = os.getenv("DOTNET_API_BASE_URL", "https://localhost:7257")
DOTNET_EXECUTE_URL  = f"{DOTNET_API_BASE_URL}/api/AiCommand/execute"
DOTNET_API_KEY      = os.getenv("DOTNET_API_KEY", "")
DOTNET_TIMEOUT      = int(os.getenv("DOTNET_TIMEOUT", "30"))

_skip_raw = os.getenv("DOTNET_SKIP_SSL_VERIFY", os.getenv("DOTNET_VERIFY_SSL", "0"))
DOTNET_VERIFY_SSL = _skip_raw.strip() not in {"1", "true", "True"}

_dotnet_session = _requests.Session()

if not DOTNET_VERIFY_SSL:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# =============================================================================
# HELPERS
# =============================================================================

def map_query_type(qt: QueryType) -> str:
    """Map QueryType enum to string label for the response."""
    if qt == QueryType.CROSS_FILTER:
        return "cross_filter"
    elif qt == QueryType.COMPARISON:
        return "comparison"
    elif qt == QueryType.MULTI_INDEPENDENT:
        return "multi_independent"
    else:
        return "simple"


def _get_session_id(data: Dict) -> str:
    """Extract session ID from request body or headers."""
    session_id = (
        data.get("session_id") or data.get("sessionId") or ""
    ).strip()
    return session_id or "admin-default"


def _get_id_filters(data: Dict) -> Dict[str, int]:
    """Extract numeric ID filters from the request body."""
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
    """Extract the admin's full name from the request body."""
    return (data.get("fullName") or data.get("full_name") or "").strip()


def _call_dotnet(payload: Dict) -> Tuple[Any, Optional[str]]:
    """
    Execute a single .NET API call.

    Returns (data, None) on success, or (None, error_message) on failure.
    """
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


# =============================================================================
# CONVERSATIONAL DETECTION
# =============================================================================

def _is_admin_conversational(message: str) -> bool:
    """Check if the message is a greeting or small talk (not a data query)."""
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

def _build_greeting_response(body: Dict, session_id: str) -> Tuple[Dict, str]:
    """Build a time-aware greeting response."""
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


def _build_conversational_response(message: str, body: Dict, session_id: str) -> Tuple[Dict, str]:
    """Build a conversational response using LLM or fallback."""
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
# MAIN PIPELINE — THE SINGLE SOURCE OF TRUTH
# =============================================================================

def execute_admin_query(
    user_query: str,
    body: Dict[str, Any],
    session_id: Optional[str] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    Execute the complete admin query pipeline.

    This is the ONLY orchestration function. Both HTTP and WebSocket
    transports call this function — they differ only in how they
    deliver the result.

    Parameters
    ----------
    user_query : str
        The raw user query text.
    body : dict
        The full request body (contains filters, names, session info).
    session_id : str, optional
        Override session ID. If None, extracted from body.
    progress_callback : callable, optional
        If provided, called with a stage name string at each pipeline step.
        Used by WebSocket transport to emit real-time progress events.
        HTTP transport passes None (no progress needed).

    Returns
    -------
    dict with keys:
        "type"              : "greeting" | "conversational" | "query" | "unrecognised" | "error"
        "greeting_message"  : str (for greeting/conversational)
        "response_data"     : dict (for greeting/conversational — type/sessionId)
        "response_payload"  : dict (for query — full build_response output)
        "combined_message"  : str (for query — the text bubble)
        "execution_time_ms" : int (for query)
        "error_message"     : str (for error)
    """
    def _notify(stage: str) -> None:
        """Fire progress_callback if one was provided."""
        if progress_callback is not None:
            try:
                progress_callback(stage)
            except Exception as exc:
                logger.debug("progress_callback(%s) failed: %s", stage, exc)
    start_time = time.time()

    message = (user_query or "").strip()
    if session_id is None:
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
        return {
            "type": response_data.get("type", "greeting"),
            "greeting_message": greeting_message,
            "response_data": response_data,
        }

    if not message:
        return {
            "type": "error",
            "error_message": "message field is required and cannot be empty.",
        }

    # ── Step 1: Resolve Named Entities ──────────────────────────────────────
    _notify("planner")
    resolved_entities = resolve_entities_from_query(
        message,
        existing_company_id=id_filters.get("companyId"),
        existing_platoon_id=id_filters.get("platoonId"),
    )
    if resolved_entities.get("companyId") is not None:
        id_filters["companyId"] = resolved_entities["companyId"]
    if resolved_entities.get("platoonId") is not None:
        id_filters["platoonId"] = resolved_entities["platoonId"]

    # ── Step 2: Query Planner ───────────────────────────────────────────────
    message    = admin_normalize_query(message)
    query_plan = plan_query(message)
    _notify("intent")

    logger.info(
        "Query plan: session=%s type=%s confidence=%.2f ops=%d reason=%s",
        session_id,
        query_plan.query_type.value,
        query_plan.confidence,
        len(query_plan.operations),
        query_plan.reasoning,
    )

    # ── Step 3: Execute .NET API Call(s) ────────────────────────────────────
    _notify("dotnet")
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
                return {
                    "type": "error",
                    "error_message": "Unable to fetch data at the moment. Please try again shortly.",
                }

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
            return {
                "type": "unrecognised",
                "response_payload": response_payload,
                "combined_message": combined_message,
            }

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
            return {
                "type": "error",
                "error_message": "Unable to fetch data at the moment. Please try again shortly.",
            }

        raw_results     = [dotnet_data]
        labeled_results = [(primary_intent.get("category", "Result"), dotnet_data)]

    # ── Step 4: Result Combiner ─────────────────────────────────────────────
    _notify("combiner")
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

    # ── Step 5: Format combined result to human-readable plain text ─────────
    formatted_data = format_dotnet_response(combined_result, primary_intent)

    # ── Step 6: Report Generator (intro + analysis + conclusion) ────────────
    _notify("report")
    report = generate_report(
        combined_result=combined_result,
        query_type=qtype_str,
        intent=primary_intent,
        user_query=message,
    )

    # ── Update session context ──────────────────────────────────────────────
    _session_context.update(session_id, message, primary_intent, combined_result)

    # ── Step 7: Response Builder ────────────────────────────────────────────
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
        "Admin pipeline complete: session=%s elapsed=%dms",
        session_id,
        execution_time_ms,
    )

    combined_message = response_payload.pop("message", "")
    return {
        "type": "query",
        "response_payload": response_payload,
        "combined_message": combined_message,
        "execution_time_ms": execution_time_ms,
    }


# =============================================================================
# DEBUG / HEALTH HELPERS (used by admin_routes.py)
# =============================================================================

def classify_for_debug(
    message: str,
    body: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Classify-only helper for the /api/admin/classify debug endpoint.

    Returns intent, dotnet payload, and query plan without executing .NET.
    This keeps all planner/intent logic inside the pipeline module.
    """
    session_id = _get_session_id(body)
    id_filters = _get_id_filters(body)
    full_name  = _get_full_name(body)

    # Greeting / conversational short-circuit
    if _is_admin_conversational(message):
        cleaned = message.lower().strip().rstrip("!?.,;")
        if _is_greeting(cleaned):
            response_data, greeting_message = _build_greeting_response(body, session_id)
        else:
            response_data, greeting_message = _build_conversational_response(message, body, session_id)
        return {"type": "greeting", "response_data": response_data, "greeting_message": greeting_message}

    # Entity resolution
    resolved_entities = resolve_entities_from_query(
        message,
        existing_company_id=id_filters.get("companyId"),
        existing_platoon_id=id_filters.get("platoonId"),
    )
    if resolved_entities.get("companyId") is not None:
        id_filters["companyId"] = resolved_entities["companyId"]
    if resolved_entities.get("platoonId") is not None:
        id_filters["platoonId"] = resolved_entities["platoonId"]

    # Classify
    message        = admin_normalize_query(message)
    intent_result  = classify_admin_intent(message)
    dotnet_payload = format_admin_payload(intent_result)
    dotnet_payload.update(id_filters)
    if full_name:
        dotnet_payload["fullName"] = full_name

    query_plan = plan_query(message)

    result: Dict[str, Any] = {
        "type":          "classify",
        "queryType":     query_plan.query_type.value,
        "confidence":    round(query_plan.confidence, 2),
        "queryPlan":     query_plan.to_dict(),
        "intent":        intent_result,
        "dotnetPayload": dotnet_payload,
    }
    if session_id and session_id != "admin-default":
        result["sessionId"] = session_id

    return result


def check_dotnet_health() -> Dict[str, Any]:
    """
    Check .NET backend connectivity.

    Returns a dict with reachability status and URL info.
    This keeps the .NET session and config encapsulated in the pipeline module.
    """
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

    return {
        "pythonStatus":  "ok",
        "dotnetBackend": "reachable" if dotnet_ok else "unreachable",
        "dotnetUrl":     DOTNET_EXECUTE_URL,
        "pythonPort":    5000,
    }
