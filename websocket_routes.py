"""
websocket_routes.py
===================
WebSocket transport layer for the AgniAI Admin Chatbot.

This file contains ONLY WebSocket routing logic. All business logic lives in
admin_pipeline.py, which is the single source of truth for query execution.

Responsibilities:
  - Accept WebSocket connections via Socket.IO.
  - Receive user queries from the frontend.
  - Emit progress events (UX indicators) via progress_callback.
  - Call execute_admin_query() — the ONLY pipeline function.
  - Stream response sections back to the client.

This file MUST NOT:
  - Contain any query processing logic.
  - Make direct .NET API calls.
  - Duplicate planner, combiner, report, or response builder logic.
  - Expose stack traces to the frontend.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from flask_socketio import SocketIO

from websocket_manager import ws_manager
from admin_pipeline import execute_admin_query

logger = logging.getLogger(__name__)


# =============================================================================
# PROGRESS HELPER
# =============================================================================

_STAGE_MESSAGES: Dict[str, str] = {
    "planner":  "Understanding query...",
    "intent":   "Building intents...",
    "dotnet":   "Fetching records...",
    "combiner": "Combining results...",
    "report":   "Generating analysis...",
}


def _progress(sid: str, stage: str, message: str) -> None:
    """Emit a progress event to the client."""
    ws_manager.send_json(sid, {"type": "progress", "stage": stage, "message": message})


# =============================================================================
# PIPELINE RUNNER
# =============================================================================

def _run_pipeline(sid: str, message: str, body: Dict) -> None:
    """
    Execute the admin pipeline and stream results back via WebSocket.

    This is a thin transport wrapper around execute_admin_query().
    Progress events are emitted at each real pipeline stage via the
    progress_callback parameter — not pre-fired.
    """
    try:
        # ── Build progress callback tied to this WebSocket session ──────────
        def emit_progress(stage: str) -> None:
            """Called by the pipeline at each stage to emit real-time progress."""
            stage_message = _STAGE_MESSAGES.get(stage, "Processing...")
            _progress(sid, stage, stage_message)

        # ── Call the unified pipeline with progress callback ────────────────
        result = execute_admin_query(
            user_query=message,
            body=body,
            progress_callback=emit_progress,
        )

        result_type = result.get("type", "error")

        # ── Greeting / conversational ───────────────────────────────────────
        if result_type in ("greeting", "conversational"):
            ws_manager.send_json(sid, {"type": "intro", "data": result["greeting_message"]})
            ws_manager.send_json(sid, {"type": "done"})
            return

        # ── Error ───────────────────────────────────────────────────────────
        if result_type == "error":
            ws_manager.send_json(sid, {
                "type": "error",
                "message": result.get("error_message", "Failed to process request."),
            })
            return

        # ── Unrecognised query ──────────────────────────────────────────────
        if result_type == "unrecognised":
            unrecognised_msg = result.get("combined_message", "")
            ws_manager.send_json(sid, {"type": "intro",      "data": unrecognised_msg or "Sorry, I was unable to understand your request."})
            ws_manager.send_json(sid, {"type": "result",     "data": {}})
            ws_manager.send_json(sid, {"type": "analysis",   "data": {}})
            ws_manager.send_json(sid, {"type": "conclusion", "data": {}})
            ws_manager.send_json(sid, {"type": "done"})
            return

        # ── Stream response sections ────────────────────────────────────────
        response_payload = result.get("response_payload", {})

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