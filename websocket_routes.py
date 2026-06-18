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

import json
import logging
import time
import uuid
from typing import Any, Dict

from flask_socketio import SocketIO

from admin_pipeline import execute_admin_query
from websocket_manager import ws_manager

logger = logging.getLogger(__name__)


# =============================================================================
# PROGRESS HELPER
# =============================================================================

_STAGE_MESSAGES: Dict[str, str] = {
    "planner": "Understanding query...",
    "intent": "Building intents...",
    "dotnet": "Fetching records...",
    "combiner": "Combining results...",
    "report": "Generating analysis...",
}


def _progress(sid: str, stage: str, message: str) -> None:
    """Emit a progress event to the client."""
    try:
        ws_manager.send_json(
            sid, {"type": "progress", "stage": stage, "message": message}
        )
    except Exception as exc:
        logger.exception("Failed to send progress event to sid=%s: %s", sid, exc)


# =============================================================================
# PIPELINE RUNNER
# =============================================================================


def _run_pipeline(sid: str, message: str, body: Dict, trace_id: str) -> None:
    """
    Execute the admin pipeline and stream results back via WebSocket.

    This is a thin transport wrapper around execute_admin_query().
    Progress events are emitted at each real pipeline stage via the
    progress_callback parameter — not pre-fired.
    """
    start_time = time.time()
    session_id = (
        body.get("session_id") or body.get("sessionId") or ""
    ).strip() or "admin-default"

    # Structured entry log
    logger.info(
        json.dumps(
            {
                "message": "WebSocket admin query entry",
                "trace_id": trace_id,
                "session_id": session_id,
                "query_type": "N/A",
                "duration_ms": None,
            }
        )
    )

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
            trace_id=trace_id,
        )

        result_type = result.get("type", "error")
        duration_ms = round((time.time() - start_time) * 1000, 2)

        # ── Error ───────────────────────────────────────────────────────────
        if result_type == "error":
            logger.error(
                json.dumps(
                    {
                        "message": "WebSocket admin query error response",
                        "trace_id": trace_id,
                        "session_id": session_id,
                        "query_type": "error",
                        "duration_ms": duration_ms,
                    }
                )
            )
            ws_manager.send_json(
                sid,
                {
                    "type": "error",
                    "message": "Failed to process request.",
                },
            )
            ws_manager.send_json(sid, {"type": "done"})
            return

        # ── Stream response sections ────────────────────────────────────────
        response_payload = result.get("response_payload", {})

        logger.info(
            json.dumps(
                {
                    "message": "WebSocket admin query success response",
                    "trace_id": trace_id,
                    "session_id": session_id,
                    "query_type": result_type,
                    "duration_ms": duration_ms,
                }
            )
        )

        ws_manager.send_json(
            sid,
            {
                "type": "intro",
                "data": response_payload.get("introMessage", ""),
            },
        )
        ws_manager.send_json(
            sid,
            {
                "type": "result",
                "data": response_payload.get("result", {}),
            },
        )
        ws_manager.send_json(
            sid,
            {
                "type": "analysis",
                "data": response_payload.get("analysis", {}),
            },
        )
        ws_manager.send_json(
            sid,
            {
                "type": "conclusion",
                "data": response_payload.get("conclusion", {}),
            },
        )
        ws_manager.send_json(sid, {"type": "done"})

    except Exception as exc:
        duration_ms = round((time.time() - start_time) * 1000, 2)
        logger.error(
            json.dumps(
                {
                    "message": "WebSocket pipeline error",
                    "trace_id": trace_id,
                    "session_id": session_id,
                    "query_type": "error",
                    "duration_ms": duration_ms,
                    "error": str(exc),
                }
            )
        )
        ws_manager.send_json(
            sid, {"type": "error", "message": "Failed to process request."}
        )
        ws_manager.send_json(sid, {"type": "done"})


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
        trace_id = uuid.uuid4().hex
        from flask import request as flask_request

        sid = flask_request.sid  # type: ignore[attr-defined]

        if not isinstance(data, dict):
            ws_manager.send_json(sid, {"type": "error", "message": "Invalid payload."})
            return

        message = (data.get("message") or "").strip()
        if not message:
            ws_manager.send_json(
                sid, {"type": "error", "message": "Message cannot be empty."}
            )
            return

        # ── Immediately acknowledge receipt ────────────────────────────────
        ws_manager.send_json(sid, {"type": "query_received"})

        # ── Run pipeline in background thread to avoid blocking the event loop
        socketio.start_background_task(_run_pipeline, sid, message, data, trace_id)
