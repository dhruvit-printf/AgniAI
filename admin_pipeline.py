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
  - Audit logging
  - OpenTelemetry span instrumentation (when enabled)

No other module should contain pipeline orchestration logic.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests as _requests

from admin_context import AdminSessionContext
from admin_entity_resolver import resolve_entities_from_query
from admin_formatter import format_dotnet_response
from admin_intent import (
    admin_normalize_query,
    classify_admin_intent,
    format_admin_payload,
)
from audit_logger import write_audit_log
from config import GREETING_PHRASES, _is_greeting, _is_patriotic, _is_small_talk
from dotnet_executor import _call_dotnet
from query_planner import QueryType, plan_query
from report_generator import generate_report
from response_builder import build_response
from result_combiner import combine_results
from telemetry import (
    SPAN_BUILD_RESPONSE,
    SPAN_CALL_DOTNET,
    SPAN_CLASSIFY_ADMIN_INTENT,
    SPAN_COMBINE_RESULTS,
    SPAN_GENERATE_REPORT,
    SPAN_PLAN_QUERY,
    span,
)
from visualization_engine import generate_widgets

logger = logging.getLogger(__name__)

SLOW_QUERY_THRESHOLD = float(os.getenv("SLOW_QUERY_THRESHOLD_SEC", "5.0"))
from metrics import metrics_collector

_session_context = AdminSessionContext()


# =============================================================================
# HELPERS
# =============================================================================


def ensure_agniveer_no_in_data(data: Any) -> None:
    """Recursively ensure that all records inside data contain agniveerNo if agniveerId is present."""
    if isinstance(data, list):
        for item in data:
            ensure_agniveer_no_in_data(item)
    elif isinstance(data, dict):
        id_val = None
        for key in ("agniveerId", "AgniveerId", "AgniVeerId", "id", "Id"):
            if key in data and data[key] is not None:
                id_val = data[key]
                break
        if "agniveerNo" not in data and id_val is not None:
            data["agniveerNo"] = str(id_val)
        
        for v in data.values():
            if isinstance(v, (dict, list)):
                ensure_agniveer_no_in_data(v)


def map_query_type(qt: QueryType) -> str:
    """Map QueryType enum to string label for the response."""
    if qt == QueryType.FILTER_QUERY or qt == QueryType.CROSS_FILTER or qt == QueryType.SIMPLE:
        return "filter_query"
    elif qt == QueryType.MULTI_OPERATION or qt == QueryType.MULTI_INDEPENDENT:
        return "multi_operation"
    elif qt == QueryType.COMPARISON:
        return "comparison"
    elif qt == QueryType.ANALYTICS:
        return "analytics"
    else:
        return "filter_query"


def _sanitize_error(err_msg: Any) -> str:
    """Scrub raw database response body or detail dumps from error messages."""
    if not err_msg:
        return ""
    err_str = str(err_msg)
    if "Backend returned HTTP" in err_str:
        # e.g. "Backend returned HTTP 400: <json-body>" -> "Backend returned HTTP 400"
        parts = err_str.split(":", 1)
        if len(parts) > 0:
            return parts[0]
    return err_str


def _get_session_id(data: Dict) -> str:
    """Extract session ID from request body or headers."""
    session_id = (data.get("session_id") or data.get("sessionId") or "").strip()
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
    batch_id = _safe_int(data.get("batchId", data.get("batch_id")))
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
    """Extract the admin's full name from the request body."""
    return (data.get("fullName") or data.get("full_name") or "").strip()


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
            "performance",
            "leave",
            "attendance",
            "medical",
            "equipment",
            "verification",
            "distribution",
            "skills",
            "top",
            "bottom",
            "score",
            "marks",
            "grading",
            "bpet",
            "ppt",
            "firing",
            "drill",
            "performer",
            "performers",
            "overdue",
            "absconded",
            "bmi",
            "disease",
            "present",
            "strength",
            "issued",
            "procured",
            "pending",
            "completed",
            "sport",
            "blood",
            "unit",
            "overall",
            "average",
            "pass",
            "fail",
            "improvement",
            "drop",
            "attempt",
            "comparison",
            "compare",
            "summary",
            "ranking",
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


def _build_conversational_response(
    message: str, body: Dict, session_id: str, trace_id: Optional[str] = None
) -> Tuple[Dict, str]:
    """Build a conversational response using LLM or fallback."""
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

    try:
        import requests as _req

        from config import DEFAULT_MODEL, OLLAMA_URL

        prompt = (
            f"You are AgniAI Command Console — an intelligent admin assistant that helps "
            f"commanding officers review and analyze Agniveer data.\n"
            f'The admin officer\'s name/title is "{admin_name}". They sent: "{message}"\n'
            f'Time greeting: "{time_greeting}".\n\n'
            f'Start with "{time_greeting}, {admin_name}!" then reply warmly and '
            f"professionally in 1-2 sentences. Offer to help with Agniveer data.\n"
            f"No markdown, no bullets."
        )
        payload: Dict[str, Any] = {
            "model": DEFAULT_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.7, "num_predict": 80, "num_ctx": 512},
        }
        resp = _req.post(OLLAMA_URL, json=payload, timeout=(8, 30))
        resp.raise_for_status()
        llm_reply = (
            resp.json().get("message", {}).get("content", "").strip().strip("\"'")
        )
        if llm_reply and 5 <= len(llm_reply) <= 300:
            response_data: Dict[str, Any] = {"type": "conversational"}
            if session_id and session_id != "admin-default":
                response_data["sessionId"] = session_id
            return response_data, llm_reply
    except Exception as exc:
        logger.warning(
            json.dumps(
                {
                    "message": "LLM conversational reply failed, using fallback",
                    "trace_id": trace_id or "N/A",
                    "error": str(exc),
                }
            )
        )

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
    trace_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute the complete admin query pipeline.

    This is the ONLY orchestration function. Both HTTP and WebSocket
    transports call this function.
    """
    if not trace_id:
        trace_id = uuid.uuid4().hex

    def _notify(stage: str) -> None:
        """Fire progress_callback if one was provided."""
        if progress_callback is not None:
            try:
                progress_callback(stage)
            except Exception as exc:
                logger.exception("progress_callback(%s) failed: %s", stage, exc)

    start_time = time.time()

    planner_duration = 0.0
    intent_duration = 0.0
    dotnet_duration = 0.0
    combiner_duration = 0.0
    report_duration = 0.0
    total_duration = 0.0
    qtype_str = "simple"
    audit_success = True
    audit_error_type: Optional[str] = None

    try:
        message = (user_query or "").strip()
        if session_id is None:
            session_id = _get_session_id(body)
        id_filters = _get_id_filters(body)
        full_name = _get_full_name(body)

        # ── Greeting / conversational short-circuit ──────────────────────────
        if _is_admin_conversational(message):
            cleaned = message.lower().strip().rstrip("!?.,;")

            intent_start = time.time()
            if _is_greeting(cleaned):
                response_data, greeting_message = _build_greeting_response(
                    body, session_id
                )
            else:
                response_data, greeting_message = _build_conversational_response(
                    message, body, session_id, trace_id=trace_id
                )
            intent_duration = time.time() - intent_start

            _notify("planner")
            _notify("intent")
            _notify("dotnet")
            _notify("combiner")
            _notify("report")

            qtype = response_data.get("type", "greeting")

            total_duration = time.time() - start_time
            durations = {
                "planner_duration": round(planner_duration * 1000, 2),
                "intent_duration": round(intent_duration * 1000, 2),
                "dotnet_duration": round(dotnet_duration * 1000, 2),
                "combiner_duration": round(combiner_duration * 1000, 2),
                "report_duration": round(report_duration * 1000, 2),
                "total_duration": round(total_duration * 1000, 2),
            }

            response_payload = build_response(
                query_type=qtype,
                intro_message=greeting_message,
                combined_result={},
                analysis={"summary": "", "observations": [], "insights": []},
                conclusion={"summary": ""},
                intent={"category": qtype, "subcategory": "", "confidence": 1.0},
                raw_results=[],
                confidence=1.0,
                operation_count=0,
                formatted_data="",
                session_id=session_id,
                durations=durations,
            )

            execution_time_ms = round(total_duration * 1000)
            response_payload["metadata"]["executionTimeMs"] = execution_time_ms

            logger.info(
                json.dumps(
                    {
                        "message": "Admin pipeline complete",
                        "trace_id": trace_id,
                        "session_id": session_id,
                        "query_type": qtype,
                        "duration": durations["total_duration"],
                        "planner_duration": durations["planner_duration"],
                        "intent_duration": durations["intent_duration"],
                        "dotnet_duration": durations["dotnet_duration"],
                        "combiner_duration": durations["combiner_duration"],
                        "report_duration": durations["report_duration"],
                    }
                )
            )

            metrics_collector.inc_requests(qtype)
            metrics_collector.record_duration(
                "planner_duration", durations["planner_duration"]
            )
            metrics_collector.record_duration(
                "intent_duration", durations["intent_duration"]
            )
            metrics_collector.record_duration(
                "dotnet_duration", durations["dotnet_duration"]
            )
            metrics_collector.record_duration(
                "report_duration", durations["report_duration"]
            )
            metrics_collector.record_duration(
                "pipeline_duration", durations["total_duration"]
            )

            if total_duration > SLOW_QUERY_THRESHOLD:
                logger.warning(
                    json.dumps(
                        {
                            "message": f"Query exceeded {int(SLOW_QUERY_THRESHOLD)} seconds.",
                            "trace_id": trace_id,
                            "session_id": session_id,
                            "query_type": qtype,
                            "duration_ms": round(total_duration * 1000, 2),
                        }
                    )
                )

            write_audit_log(
                trace_id=trace_id,
                session_id=session_id,
                query_type=qtype,
                query_duration=durations["total_duration"],
                success=True,
            )

            combined_message = response_payload.pop("message", "")
            return {
                "type": qtype,
                "response_payload": response_payload,
                "combined_message": combined_message,
                "execution_time_ms": execution_time_ms,
            }

        if not message:
            return {
                "type": "error",
                "error_message": "Failed to process request.",
            }

        # ── Step 1: Resolve Named Entities ───────────────────────────────────
        with span(SPAN_PLAN_QUERY, trace_id=trace_id):
            planner_start = time.time()
            _notify("planner")
            resolved_entities = resolve_entities_from_query(
                message,
                existing_company_id=id_filters.get("companyId"),
                existing_platoon_id=id_filters.get("platoonId"),
            )
            resolved_company = resolved_entities.get("companyId")
            if resolved_company is not None:
                id_filters["companyId"] = int(resolved_company)
            resolved_platoon = resolved_entities.get("platoonId")
            if resolved_platoon is not None:
                id_filters["platoonId"] = int(resolved_platoon)

            message = admin_normalize_query(message)
            query_plan = plan_query(message)
            planner_duration = time.time() - planner_start

        _notify("intent")

        # ── Step 2: Intent Classification ────────────────────────────────────
        with span(SPAN_CLASSIFY_ADMIN_INTENT, trace_id=trace_id):
            intent_start = time.time()
            raw_results: List[Any] = []
            labeled_results: List[Tuple[str, Any]] = []
            primary_intent: Dict[str, Any] = {}
            operation_count: int = 1

            if (
                query_plan.query_type != QueryType.SIMPLE
                and query_plan.confidence >= 0.5
                and len(query_plan.operations) >= 2
            ):

                qtype_str = map_query_type(query_plan.query_type)
                operation_count = len(query_plan.operations)

                logger.info(
                    json.dumps(
                        {
                            "message": "Query plan compiled",
                            "trace_id": trace_id,
                            "session_id": session_id,
                            "query_type": qtype_str,
                            "confidence": query_plan.confidence,
                            "operation_count": operation_count,
                            "reasoning": query_plan.reasoning,
                        }
                    )
                )

                intent_duration = time.time() - intent_start

                # ── Step 3: Execute .NET API Call(s) ─────────────────────────
                with span(SPAN_CALL_DOTNET, trace_id=trace_id):
                    dotnet_start = time.time()
                    _notify("dotnet")
                    for i, op in enumerate(query_plan.operations):
                        payload = dict(op.dotnet_payload)
                        payload.update(id_filters)
                        if full_name:
                            payload["fullName"] = full_name

                        logger.info(
                            json.dumps(
                                {
                                    "message": "Sending multi-op request to .NET",
                                    "trace_id": trace_id,
                                    "session_id": session_id,
                                    "query_type": qtype_str,
                                    "op_index": i + 1,
                                    "total_ops": operation_count,
                                }
                            )
                        )

                        dotnet_data, dotnet_error = _call_dotnet(
                            payload, trace_id=trace_id
                        )
                        if dotnet_error:
                            logger.warning(
                                json.dumps(
                                    {
                                        "message": "Multi-op failed",
                                        "trace_id": trace_id,
                                        "session_id": session_id,
                                        "query_type": qtype_str,
                                        "op_index": i + 1,
                                        "error": _sanitize_error(dotnet_error),
                                    }
                                )
                            )

                            metrics_collector.inc_requests(qtype_str)
                            metrics_collector.inc_errors(qtype_str)
                            total_duration = time.time() - start_time
                            metrics_collector.record_duration(
                                "pipeline_duration", round(total_duration * 1000, 2)
                            )

                            if total_duration > SLOW_QUERY_THRESHOLD:
                                logger.warning(
                                    json.dumps(
                                        {
                                            "message": f"Query exceeded {int(SLOW_QUERY_THRESHOLD)} seconds.",
                                            "trace_id": trace_id,
                                            "session_id": session_id,
                                            "query_type": qtype_str,
                                            "duration_ms": round(
                                                total_duration * 1000, 2
                                            ),
                                        }
                                    )
                                )

                            write_audit_log(
                                trace_id=trace_id,
                                session_id=session_id,
                                query_type=qtype_str,
                                query_duration=round(total_duration * 1000, 2),
                                success=False,
                                error_type="dotnet_error",
                            )

                            return {
                                "type": "error",
                                "error_message": "Failed to process request.",
                            }

                        ensure_agniveer_no_in_data(dotnet_data)
                        raw_results.append(dotnet_data)
                        label = op.intent_result.get("category", f"Query {i + 1}")
                        labeled_results.append((label, dotnet_data))

                    primary_intent = query_plan.operations[0].intent_result
                    dotnet_duration = time.time() - dotnet_start

            else:
                # SIMPLE / ANALYTICS: single .NET call
                qtype_str = "simple"
                operation_count = 1

                if (
                    query_plan.query_type == QueryType.ANALYTICS
                    and query_plan.operations
                    and query_plan.operations[0].intent_result.get("category")
                ):
                    primary_intent = query_plan.operations[0].intent_result
                else:
                    primary_intent = classify_admin_intent(message)

                logger.info(
                    json.dumps(
                        {
                            "message": "Query plan compiled",
                            "trace_id": trace_id,
                            "session_id": session_id,
                            "query_type": qtype_str,
                            "confidence": query_plan.confidence,
                            "operation_count": operation_count,
                            "reasoning": query_plan.reasoning,
                        }
                    )
                )

                # Unrecognised query
                if primary_intent.get("category") is None:
                    intent_duration = time.time() - intent_start
                    unrecognised_msg = (
                        "Sorry, I was unable to understand your request. "
                        "I can help with Performance, Leave, Attendance, Medical, Equipment, "
                        "Verification, Distribution, and Skills information. "
                        "Please ask a relevant question."
                    )

                    total_duration = time.time() - start_time
                    durations = {
                        "planner_duration": round(planner_duration * 1000, 2),
                        "intent_duration": round(intent_duration * 1000, 2),
                        "dotnet_duration": round(dotnet_duration * 1000, 2),
                        "combiner_duration": round(combiner_duration * 1000, 2),
                        "report_duration": round(report_duration * 1000, 2),
                        "total_duration": round(total_duration * 1000, 2),
                    }

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
                        durations=durations,
                    )

                    logger.info(
                        json.dumps(
                            {
                                "message": "Admin pipeline complete",
                                "trace_id": trace_id,
                                "session_id": session_id,
                                "query_type": "unrecognised",
                                "duration": durations["total_duration"],
                                "planner_duration": durations["planner_duration"],
                                "intent_duration": durations["intent_duration"],
                                "dotnet_duration": durations["dotnet_duration"],
                                "combiner_duration": durations["combiner_duration"],
                                "report_duration": durations["report_duration"],
                            }
                        )
                    )

                    metrics_collector.inc_requests("unrecognised")
                    metrics_collector.record_duration(
                        "planner_duration", durations["planner_duration"]
                    )
                    metrics_collector.record_duration(
                        "intent_duration", durations["intent_duration"]
                    )
                    metrics_collector.record_duration(
                        "dotnet_duration", durations["dotnet_duration"]
                    )
                    metrics_collector.record_duration(
                        "report_duration", durations["report_duration"]
                    )
                    metrics_collector.record_duration(
                        "pipeline_duration", durations["total_duration"]
                    )

                    if total_duration > SLOW_QUERY_THRESHOLD:
                        logger.warning(
                            json.dumps(
                                {
                                    "message": f"Query exceeded {int(SLOW_QUERY_THRESHOLD)} seconds.",
                                    "trace_id": trace_id,
                                    "session_id": session_id,
                                    "query_type": "unrecognised",
                                    "duration_ms": round(total_duration * 1000, 2),
                                }
                            )
                        )

                    write_audit_log(
                        trace_id=trace_id,
                        session_id=session_id,
                        query_type="unrecognised",
                        query_duration=durations["total_duration"],
                        success=True,
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

                if (
                    query_plan.query_type == QueryType.ANALYTICS
                    and query_plan.operations
                ):
                    op = query_plan.operations[0]
                    if getattr(op, "group_by", None):
                        dotnet_payload["groupBy"] = op.group_by
                    if query_plan.analytics_hint:
                        dotnet_payload["analyticsHint"] = query_plan.analytics_hint

                intent_duration = time.time() - intent_start

                # ── Step 3: Execute .NET API Call ─────────────────────────────
                with span(SPAN_CALL_DOTNET, trace_id=trace_id):
                    dotnet_start = time.time()
                    _notify("dotnet")

                    logger.info(
                        json.dumps(
                            {
                                "message": "Sending simple request to .NET",
                                "trace_id": trace_id,
                                "session_id": session_id,
                                "query_type": qtype_str,
                            }
                        )
                    )

                    dotnet_data, dotnet_error = _call_dotnet(
                        dotnet_payload, trace_id=trace_id
                    )
                    if dotnet_error:
                        logger.warning(
                            json.dumps(
                                {
                                    "message": "Admin .NET call failed",
                                    "trace_id": trace_id,
                                    "session_id": session_id,
                                    "query_type": qtype_str,
                                    "error": _sanitize_error(dotnet_error),
                                }
                            )
                        )

                        metrics_collector.inc_requests(qtype_str)
                        metrics_collector.inc_errors(qtype_str)
                        total_duration = time.time() - start_time
                        metrics_collector.record_duration(
                            "pipeline_duration", round(total_duration * 1000, 2)
                        )

                        if total_duration > SLOW_QUERY_THRESHOLD:
                            logger.warning(
                                json.dumps(
                                    {
                                        "message": f"Query exceeded {int(SLOW_QUERY_THRESHOLD)} seconds.",
                                        "trace_id": trace_id,
                                        "session_id": session_id,
                                        "query_type": qtype_str,
                                        "duration_ms": round(total_duration * 1000, 2),
                                    }
                                )
                            )

                        write_audit_log(
                            trace_id=trace_id,
                            session_id=session_id,
                            query_type=qtype_str,
                            query_duration=round(total_duration * 1000, 2),
                            success=False,
                            error_type="dotnet_error",
                        )

                        return {
                            "type": "error",
                            "error_message": "Failed to process request.",
                        }

                    ensure_agniveer_no_in_data(dotnet_data)
                    raw_results = [dotnet_data]
                    labeled_results = [
                        (primary_intent.get("category", "Result"), dotnet_data)
                    ]
                    dotnet_duration = time.time() - dotnet_start

        # ── Step 4: Result Combiner & Formatting ──────────────────────────────
        with span(SPAN_COMBINE_RESULTS, trace_id=trace_id):
            combiner_start = time.time()
            _notify("combiner")
            combined_result = combine_results(
                raw_results, labeled_results, qtype_str, primary_intent
            )
            formatted_data = format_dotnet_response(combined_result, primary_intent)
            combiner_duration = time.time() - combiner_start

        # ── Step 5: Report Generator ──────────────────────────────────────────
        with span(SPAN_GENERATE_REPORT, trace_id=trace_id):
            report_start = time.time()
            _notify("report")
            try:
                report = generate_report(
                    combined_result=combined_result,
                    query_type=qtype_str,
                    intent=primary_intent,
                    user_query=message,
                    trace_id=trace_id,
                )
            except Exception as report_exc:
                logger.error(
                    json.dumps(
                        {
                            "message": "Unexpected exception from generate_report",
                            "trace_id": trace_id,
                            "session_id": session_id,
                            "query_type": qtype_str,
                            "error": str(report_exc),
                        }
                    )
                )
                report = {
                    "introMessage": "Report generated with partial metrics.",
                    "analysis": None,
                    "conclusion": None,
                }
            report_duration = time.time() - report_start

        # ── Update session context ────────────────────────────────────────────
        _session_context.update(session_id, message, primary_intent, combined_result)

        total_duration = time.time() - start_time
        durations = {
            "planner_duration": round(planner_duration * 1000, 2),
            "intent_duration": round(intent_duration * 1000, 2),
            "dotnet_duration": round(dotnet_duration * 1000, 2),
            "combiner_duration": round(combiner_duration * 1000, 2),
            "report_duration": round(report_duration * 1000, 2),
            "total_duration": round(total_duration * 1000, 2),
        }

        # ── Step 6: Response Builder & Visualization ──────────────────────────
        widgets = generate_widgets(
            combined_result=combined_result,
            query_plan=query_plan,
            analysis=report.get("analysis"),
        )

        with span(SPAN_BUILD_RESPONSE, trace_id=trace_id):
            response_payload = build_response(
                query_type=qtype_str,
                intro_message=report.get("introMessage", ""),
                combined_result=combined_result,
                analysis=report.get("analysis"),
                conclusion=report.get("conclusion"),
                intent=primary_intent,
                raw_results=raw_results,
                confidence=query_plan.confidence,
                operation_count=operation_count,
                formatted_data=formatted_data,
                session_id=session_id,
                durations=durations,
                widgets=widgets,
            )

        execution_time_ms = round(total_duration * 1000)
        response_payload["metadata"]["executionTimeMs"] = execution_time_ms

        # Extract combiner strategy details
        if qtype_str in ("multi_independent", "multi_operation"):
            combiner_strategy = "merge"
        elif qtype_str == "comparison":
            combiner_strategy = "comparison"
        elif qtype_str == "cross_filter":
            combiner_strategy = "intersect"
        else:
            combiner_strategy = "passthrough"

        # Log all 10 required audit metrics
        logger.info(
            json.dumps(
                {
                    "message": "Admin pipeline complete",
                    "trace_id": trace_id,
                    "session_id": session_id,
                    "main_intent": {
                        "category": primary_intent.get("category", ""),
                        "operation": primary_intent.get("subcategory", "") or primary_intent.get("operation", ""),
                    },
                    "filters": query_plan.filters if query_plan else {},
                    "query_type": qtype_str,
                    "operation_count": operation_count,
                    "payload_sent": [format_admin_payload(op.intent_result) for op in query_plan.operations] if query_plan else [],
                    "combiner_strategy": combiner_strategy,
                    "visualization_decisions": [w["type"] for w in widgets],
                    "widget_count": len(widgets),
                    "report_duration": durations["report_duration"],
                    "dotnet_duration": durations["dotnet_duration"],
                    "total_duration": durations["total_duration"],
                }
            )
        )

        metrics_collector.inc_requests(qtype_str)
        metrics_collector.record_duration(
            "planner_duration", durations["planner_duration"]
        )
        metrics_collector.record_duration(
            "intent_duration", durations["intent_duration"]
        )
        metrics_collector.record_duration(
            "dotnet_duration", durations["dotnet_duration"]
        )
        metrics_collector.record_duration(
            "report_duration", durations["report_duration"]
        )
        metrics_collector.record_duration(
            "pipeline_duration", durations["total_duration"]
        )

        if total_duration > SLOW_QUERY_THRESHOLD:
            logger.warning(
                json.dumps(
                    {
                        "message": f"Query exceeded {int(SLOW_QUERY_THRESHOLD)} seconds.",
                        "trace_id": trace_id,
                        "session_id": session_id,
                        "query_type": qtype_str,
                        "duration_ms": round(total_duration * 1000, 2),
                    }
                )
            )

        write_audit_log(
            trace_id=trace_id,
            session_id=session_id,
            query_type=qtype_str,
            query_duration=durations["total_duration"],
            success=True,
        )

        combined_message = response_payload.pop("message", "")
        return {
            "type": "query",
            "response_payload": response_payload,
            "combined_message": combined_message,
            "execution_time_ms": execution_time_ms,
        }

    except Exception as exc:
        total_duration = time.time() - start_time
        logger.error(
            json.dumps(
                {
                    "message": "Error in execute_admin_query",
                    "trace_id": trace_id,
                    "session_id": session_id or "admin-default",
                    "query_type": "error",
                    "duration_ms": round(total_duration * 1000, 2),
                    "error": str(exc),
                }
            )
        )

        metrics_collector.inc_requests("error")
        metrics_collector.inc_errors("error")
        metrics_collector.record_duration(
            "pipeline_duration", round(total_duration * 1000, 2)
        )

        if total_duration > SLOW_QUERY_THRESHOLD:
            logger.warning(
                json.dumps(
                    {
                        "message": f"Query exceeded {int(SLOW_QUERY_THRESHOLD)} seconds.",
                        "trace_id": trace_id,
                        "session_id": session_id or "admin-default",
                        "query_type": "error",
                        "duration_ms": round(total_duration * 1000, 2),
                    }
                )
            )

        write_audit_log(
            trace_id=trace_id,
            session_id=session_id or "admin-default",
            query_type="error",
            query_duration=round(total_duration * 1000, 2),
            success=False,
            error_type="pipeline_exception",
        )

        return {
            "type": "error",
            "error_message": "Failed to process request.",
        }
