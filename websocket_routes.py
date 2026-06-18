"""
websocket_routes.py
===================
WebSocket transport layer for AgniAI Admin Chatbot.

Responsibilities:
  - Accept WebSocket connections via Socket.IO.
  - Receive user queries from the frontend.
  - Call the existing pipeline in the correct order.
  - Stream progress events and final response sections back.

STRICT CONTRACT — this file MUST NOT:
  - Modify query_planner.py, admin_intent.py, result_combiner.py,
    report_generator.py, or response_builder.py.
  - Contain any query processing logic.
  - Contain any formatting or analysis generation.
  - Make direct .NET API calls outside of what admin_routes already does.
  - Expose stack traces to the frontend.

The pipeline called here is the exact same one used in admin_routes.py.
WebSocket is only a transport mechanism — a thin streaming wrapper.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import requests as _requests
from flask_socketio import SocketIO, emit

from websocket_manager import ws_manager

logger = logging.getLogger(__name__)

# ── .NET config (mirrors admin_routes.py) ────────────────────────────────────
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
# INTERNAL HELPERS  (mirrors admin_routes._call_dotnet)
# =============================================================================

def _call_dotnet(payload: Dict) -> Tuple[Any, Optional[str]]:
    headers = {"Content-Type": "application/json"}
    if DOTNET_API_KEY:
        headers["X-Api-Key"] = DOTNET_API_KEY
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
        return None, f"Cannot connect to .NET backend. ({exc})"
    except _requests.Timeout:
        return None, f"Backend timed out after {DOTNET_TIMEOUT}s."
    except _requests.RequestException as exc:
        return None, f"Backend request failed: {exc}"
    except ValueError as exc:
        return None, f"Backend returned invalid JSON: {exc}"


def _get_id_filters(data: Dict) -> Dict[str, int]:
    def _safe_int(v) -> Optional[int]:
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    filters = {}
    for key_pair in [
        ("commandId",  "command_id"),
        ("batchId",    "batch_id"),
        ("platoonId",  "platoon_id"),
        ("companyId",  "company_id"),
    ]:
        val = _safe_int(data.get(key_pair[0], data.get(key_pair[1])))
        if val is not None:
            filters[key_pair[0]] = val
    return filters


def _progress(sid: str, stage: str, message: str) -> None:
    ws_manager.send_json(sid, {"type": "progress", "stage": stage, "message": message})


# =============================================================================
# PIPELINE RUNNER
# =============================================================================

def _run_pipeline(sid: str, message: str, body: Dict) -> None:
    """
    Execute the complete AgniAI admin pipeline and stream results back.

    Mirrors admin_routes.admin_chat() exactly — only the *transport* differs.
    The pipeline modules are called in the same order with the same arguments.
    """
    from admin_intent import admin_normalize_query, classify_admin_intent, format_admin_payload
    from config import _is_greeting, _is_small_talk, _is_patriotic
    from query_planner import plan_query, QueryType
    from result_combiner import intersect_results, merge_results, compare_results
    from admin_entity_resolver import resolve_entities_from_query
    from admin_formatter import format_dotnet_response
    from report_generator import generate_report
    from response_builder import build_response
    from admin_routes import (
        _is_admin_conversational,
        _build_greeting_response,
        _build_conversational_response,
        map_query_type,
        _get_session_id,
    )

    session_id  = _get_session_id(body)
    id_filters  = _get_id_filters(body)
    full_name   = (body.get("fullName") or body.get("full_name") or "").strip()

    try:
        # ── Greeting / conversational short-circuit ───────────────────────
        if _is_admin_conversational(message):
            cleaned = message.lower().strip().rstrip("!?.,;")
            if _is_greeting(cleaned):
                _data, greeting_msg = _build_greeting_response(body, session_id)
            else:
                _data, greeting_msg = _build_conversational_response(message, body, session_id)
            ws_manager.send_json(sid, {"type": "intro", "data": greeting_msg})
            ws_manager.send_json(sid, {"type": "done"})
            return

        if not message:
            ws_manager.send_json(sid, {"type": "error", "message": "Message cannot be empty."})
            return

        # ── Resolve named entities ─────────────────────────────────────────
        resolved = resolve_entities_from_query(
            message,
            existing_company_id=id_filters.get("companyId"),
            existing_platoon_id=id_filters.get("platoonId"),
        )
        if resolved.get("companyId") is not None:
            id_filters["companyId"] = resolved["companyId"]
        if resolved.get("platoonId") is not None:
            id_filters["platoonId"] = resolved["platoonId"]

        # ── Step 1: Query Planner ──────────────────────────────────────────
        _progress(sid, "planner", "Understanding query...")
        message      = admin_normalize_query(message)
        query_plan   = plan_query(message)

        qtype_str      = "simple"
        operation_count = 1
        raw_results: List[Any]             = []
        labeled_results: List[Tuple]       = []
        primary_intent: Dict[str, Any]     = {}

        if (query_plan.query_type != QueryType.SIMPLE
                and query_plan.confidence >= 0.5
                and len(query_plan.operations) >= 2):

            qtype_str       = map_query_type(query_plan.query_type)
            operation_count = len(query_plan.operations)

            # ── Step 2: Intent Builder (multi-op) ─────────────────────────
            _progress(sid, "intent", "Building intents...")

            # ── Step 3: .NET API Executor ──────────────────────────────────
            _progress(sid, "dotnet", "Fetching records...")

            for i, op in enumerate(query_plan.operations):
                payload = dict(op.dotnet_payload)
                payload.update(id_filters)
                if full_name:
                    payload["fullName"] = full_name

                dotnet_data, dotnet_error = _call_dotnet(payload)
                if dotnet_error:
                    logger.warning("WS multi-op %d/%d failed: %s", i + 1, operation_count, dotnet_error)
                    ws_manager.send_json(sid, {"type": "error", "message": "Failed to process request."})
                    return

                raw_results.append(dotnet_data)
                label = op.intent_result.get("category", f"Query {i + 1}")
                labeled_results.append((label, dotnet_data))

            primary_intent = query_plan.operations[0].intent_result

        else:
            qtype_str       = "simple"
            operation_count = 1

            if (query_plan.query_type.value == "analytics"
                    and query_plan.operations
                    and query_plan.operations[0].intent_result.get("category")):
                primary_intent = query_plan.operations[0].intent_result
            else:
                # ── Step 2: Intent Builder (simple) ───────────────────────
                _progress(sid, "intent", "Building intents...")
                primary_intent = classify_admin_intent(message)

            if primary_intent.get("category") is None:
                unrecognised = (
                    "Sorry, I was unable to understand your request. "
                    "I can help with Performance, Leave, Attendance, Medical, Equipment, "
                    "Verification, Distribution, and Skills information."
                )
                ws_manager.send_json(sid, {"type": "intro",      "data": unrecognised})
                ws_manager.send_json(sid, {"type": "result",     "data": {}})
                ws_manager.send_json(sid, {"type": "analysis",   "data": {}})
                ws_manager.send_json(sid, {"type": "conclusion", "data": {}})
                ws_manager.send_json(sid, {"type": "done"})
                return

            dotnet_payload = format_admin_payload(primary_intent)
            dotnet_payload.update(id_filters)
            if full_name:
                dotnet_payload["fullName"] = full_name

            if query_plan.query_type.value == "analytics" and query_plan.operations:
                op = query_plan.operations[0]
                if getattr(op, "group_by", None):
                    dotnet_payload["groupBy"] = op.group_by
                if query_plan.analytics_hint:
                    dotnet_payload["analyticsHint"] = query_plan.analytics_hint

            # ── Step 3: .NET API Executor ──────────────────────────────────
            _progress(sid, "dotnet", "Fetching records...")
            dotnet_data, dotnet_error = _call_dotnet(dotnet_payload)
            if dotnet_error:
                logger.warning("WS .NET call failed: %s", dotnet_error)
                ws_manager.send_json(sid, {"type": "error", "message": "Failed to process request."})
                return

            raw_results     = [dotnet_data]
            labeled_results = [(primary_intent.get("category", "Result"), dotnet_data)]

        # ── Step 4: Result Combiner ────────────────────────────────────────
        _progress(sid, "combiner", "Combining results...")

        if qtype_str == "cross_filter":
            combined_result = intersect_results(raw_results, primary_index=0)
        elif qtype_str == "comparison":
            combined_result = compare_results(labeled_results)
        elif qtype_str == "multi_independent":
            combined_result = merge_results(labeled_results)
        else:
            combined_result = raw_results[0] if raw_results else {}

        # ── Step 5: Format data ───────────────────────────────────────────
        formatted_data = format_dotnet_response(combined_result, primary_intent)

        # ── Step 6: Report Generator ──────────────────────────────────────
        _progress(sid, "report", "Generating analysis...")
        report = generate_report(
            combined_result=combined_result,
            query_type=qtype_str,
            intent=primary_intent,
            user_query=message,
        )

        # ── Step 7: Response Builder ──────────────────────────────────────
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

        # ── Step 8: Stream sections back to frontend ──────────────────────
        ws_manager.send_json(sid, {
            "type": "intro",
            "data": response_payload.get("introMessage", ""),
        })
        ws_manager.send_json(sid, {
            "type": "result",
            "data": response_payload.get("result", {}),
        })
        ws_manager.send_json(sid, {
            "type": "analysis",
            "data": response_payload.get("analysis", {}),
        })
        ws_manager.send_json(sid, {
            "type": "conclusion",
            "data": response_payload.get("conclusion", {}),
        })
        ws_manager.send_json(sid, {"type": "done"})

    except Exception as exc:
        logger.exception("WebSocket pipeline error for sid=%s: %s", sid, exc)
        ws_manager.send_json(sid, {"type": "error", "message": "Failed to process request."})


# =============================================================================
# SOCKET.IO EVENT HANDLERS
# =============================================================================

def register_socketio_events(socketio: SocketIO) -> None:
    """
    Attach all Socket.IO event handlers to the given SocketIO instance.
    Called once from app.py after socketio is created.
    """

    @socketio.on("connect")
    def on_connect():
        from flask import request as flask_request
        sid = flask_request.sid  # type: ignore[attr-defined]
        ws_manager.register(sid)
        logger.info("WebSocket connected: sid=%s", sid)

    @socketio.on("disconnect")
    def on_disconnect():
        from flask import request as flask_request
        sid = flask_request.sid  # type: ignore[attr-defined]
        ws_manager.unregister(sid)
        logger.info("WebSocket disconnected: sid=%s", sid)

    @socketio.on("query")
    def on_query(data):
        """
        Receive a query from the frontend and stream the pipeline response.

        Expected payload:
            {
                "message":    "<user query>",
                "session_id": "<optional>",
                ... (any other fields forwarded to the pipeline)
            }
        """
        from flask import request as flask_request
        sid = flask_request.sid  # type: ignore[attr-defined]

        if not isinstance(data, dict):
            ws_manager.send_json(sid, {"type": "error", "message": "Invalid payload."})
            return

        message = (data.get("message") or "").strip()
        if not message:
            ws_manager.send_json(sid, {"type": "error", "message": "Message cannot be empty."})
            return

        # ── Immediately acknowledge receipt ────────────────────────────────
        ws_manager.send_json(sid, {"type": "query_received"})

        # ── Run pipeline in background thread to avoid blocking the event loop
        socketio.start_background_task(_run_pipeline, sid, message, data)