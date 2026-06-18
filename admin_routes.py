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

from admin_pipeline import execute_admin_query

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────
ADMIN_RATE_LIMIT = os.getenv("ADMIN_RATE_LIMIT", "20 per minute")

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
                    "trace_id": trace_id,
                    "session_id": session_id,
                    "query_type": "error",
                    "duration_ms": duration_ms,
                }
            )
        )
        return jsonify({"type": "error", "message": "Failed to process request."}), 500

    # ── Successful query / greeting / conversational ────────────────────────
    combined_message = result.get("combined_message", "")

    logger.info(
        json.dumps(
            {
                "message": "HTTP admin chat success response",
                "trace_id": trace_id,
                "session_id": session_id,
                "query_type": result_type,
                "duration_ms": duration_ms,
            }
        )
    )

    return _success_response(result["response_payload"], message=combined_message)


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
    from dotnet_executor import DOTNET_API_BASE_URL, DOTNET_VERIFY_SSL, _cb

    if _cb.state == "OPEN":
        return "unhealthy"
    try:
        import requests

        # Perform GET on base URL to check reachability
        requests.get(DOTNET_API_BASE_URL, timeout=2, verify=DOTNET_VERIFY_SSL)
        return "healthy"
    except Exception:
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

    payload = {"python": py_h, "dotnet": dn_h, "llm": llm_h, "database": db_h}

    status_code = 200
    if "unhealthy" in payload.values():
        status_code = 503

    return jsonify(payload), status_code
