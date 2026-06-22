"""
admin_routes.py
===============
HTTP transport layer for the AgniAI Admin Chatbot.

This file contains ONLY HTTP routing logic. All business logic lives in
admin_pipeline.py, which is the single source of truth for query execution.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any, Dict, Optional

from flask import Blueprint, jsonify, request

from admin_intent import (
    classify_admin_intent,
    format_admin_intent,
    format_admin_payload,
)
from admin_pipeline import execute_admin_query
from response_builder import public_response_view

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────
ADMIN_RATE_LIMIT = os.getenv("ADMIN_RATE_LIMIT", "20 per minute")
LAST_DOTNET_HEALTH_LATENCY_MS: Optional[float] = None

# ── Blueprint ──────────────────────────────────────────────────────────────
admin_bp = Blueprint("admin", __name__, url_prefix="/api/admin")


# =============================================================================
# RATE LIMITER
# =============================================================================


def _register_rate_limits(app, limiter) -> None:
    try:
        if limiter is not None:
            limiter.limit(ADMIN_RATE_LIMIT, override_defaults=True)(admin_chat)
            logger.info("Admin rate limit applied: %s", ADMIN_RATE_LIMIT)
    except Exception as exc:
        logger.warning("Could not register admin rate limit: %s", exc)


# =============================================================================
# RESPONSE HELPERS
# =============================================================================


def _success_response(data: Dict, http_status: int = 200, message: str = ""):
    """Return a flat JSON response. All keys from data are at root level."""
    payload: Dict[str, Any] = {
        "status": True,
        "httpStatus": http_status,
        "message": message,
    }
    payload.update(data)
    return jsonify(payload), http_status


# =============================================================================
# ROUTES
# =============================================================================


@admin_bp.route("/chat", methods=["POST"])
def admin_chat():
    """
    Main admin chat endpoint.
    Delegates ALL processing to execute_admin_query() in admin_pipeline.py.
    This function is HTTP transport only.
    """
    trace_id = uuid.uuid4().hex
    start_time = time.time()

    body = request.get_json(force=True, silent=True) or {}
    message = (body.get("message") or "").strip()

    # Extract session ID for context (flat default if none found)
    session_id = (
        body.get("session_id") or body.get("sessionId") or ""
    ).strip() or "admin-default"

    # Structured entry log
    logger.info(
        json.dumps(
            {
                "message": "HTTP admin chat entry",
                "question": message,
                "trace_id": trace_id,
                "session_id": session_id,
                "query_type": "N/A",
                "duration_ms": None,
            }
        )
    )

    # ── Call the unified pipeline ───────────────────────────────────────────
    result = execute_admin_query(user_query=message, body=body, trace_id=trace_id)

    result_type = result.get("type", "error")
    duration_ms = round((time.time() - start_time) * 1000, 2)

    # ── Error ───────────────────────────────────────────────────────────────
    if result_type == "error":
        logger.error(
            json.dumps(
                {
                    "message": "HTTP admin chat error response",
                    "question": message,
                    "trace_id": trace_id,
                    "session_id": session_id,
                    "query_type": "error",
                    "duration_ms": duration_ms,
                }
            )
        )
        return jsonify({"type": "error", "message": "Failed to process request."}), 500

    # ── Successful query / greeting / conversational ────────────────────────
    payload = result.get("response_payload") or {}
    logger.info(
        json.dumps(
            {
                "question": message,
                "query_type": payload.get("queryType"),
                "intent_formed": payload.get("intent"),
            }
        )
    )

    return jsonify(public_response_view(result["response_payload"])), 200


# =============================================================================
# SUBSYSTEM HEALTH CHECKS
# =============================================================================


def check_python_health() -> str:
    try:
        from config import FAISS_INDEX_PATH

        if FAISS_INDEX_PATH.exists():
            return "healthy"
        return "unhealthy"
    except Exception:
        return "unhealthy"


def check_dotnet_health() -> str:
    from dotnet_executor import DOTNET_EXECUTE_URL, DOTNET_VERIFY_SSL, _cb

    global LAST_DOTNET_HEALTH_LATENCY_MS
    if _cb.state == "OPEN":
        LAST_DOTNET_HEALTH_LATENCY_MS = None
        return "unhealthy"
    try:
        import requests

        start = time.time()
        resp = requests.options(DOTNET_EXECUTE_URL, timeout=2, verify=DOTNET_VERIFY_SSL)
        LAST_DOTNET_HEALTH_LATENCY_MS = round((time.time() - start) * 1000, 2)
        if resp.status_code < 400:
            return "healthy"
        if resp.status_code == 405:
            return "degraded"
        return "unhealthy"
    except Exception:
        LAST_DOTNET_HEALTH_LATENCY_MS = None
        return "unhealthy"


def check_llm_health() -> str:
    from config import OLLAMA_TAGS_URL

    try:
        import requests

        resp = requests.get(OLLAMA_TAGS_URL, timeout=2)
        if resp.status_code < 400:
            return "healthy"
        return "unhealthy"
    except Exception:
        return "unhealthy"


def check_database_health() -> str:
    try:
        from rag import index_stats, is_ready

        stats = index_stats()
        if stats and stats.get("vectors", 0) > 0 and is_ready():
            return "healthy"
        return "unhealthy"
    except Exception:
        return "unhealthy"


@admin_bp.route("/health", methods=["GET"])
def admin_health():
    py_h = check_python_health()
    dn_h = check_dotnet_health()
    llm_h = check_llm_health()
    db_h = check_database_health()

    payload = {
        "python": py_h,
        "dotnet": dn_h,
        "dotnetLatencyMs": LAST_DOTNET_HEALTH_LATENCY_MS,
        "llm": llm_h,
        "database": db_h,
    }

    status_code = 200
    if any(
        value in ("unhealthy", "unreachable")
        for value in payload.values()
        if isinstance(value, str)
    ):
        status_code = 503

    return jsonify(payload), status_code


@admin_bp.route("/classify", methods=["POST"])
def admin_classify():
    """
    Classify admin intent debug endpoint.
    Runs the classifier without calling the .NET backend.
    """
    body = request.get_json(force=True, silent=True) or {}
    message = (body.get("message") or "").strip()
    intent = classify_admin_intent(message)
    frontend_intent = format_admin_intent(intent)
    dotnet_payload = format_admin_payload(intent)
    logger.info(
        json.dumps(
            {
                "question": message,
                "intent": frontend_intent,
                "type": frontend_intent.get("type") or "",
                "intentFormed": bool(frontend_intent.get("category")),
            },
            ensure_ascii=False,
        )
    )
    return (
        jsonify(
            {
                "success": True,
                "intent": frontend_intent,
                "dotnet_payload": dotnet_payload,
            }
        ),
        200,
    )
