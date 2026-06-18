"""
admin_routes.py
===============
HTTP transport layer for the AgniAI Admin Chatbot.

This file contains ONLY HTTP routing logic. All business logic lives in
admin_pipeline.py, which is the single source of truth for query execution.
"""

from __future__ import annotations

import logging
import os
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
        "status":     True,
        "httpStatus": http_status,
        "message":    message,
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
    body    = request.get_json(force=True, silent=True) or {}
    message = (body.get("message") or "").strip()

    # ── Call the unified pipeline ───────────────────────────────────────────
    result = execute_admin_query(user_query=message, body=body)

    result_type = result.get("type", "error")

    # ── Error ───────────────────────────────────────────────────────────────
    if result_type == "error":
        return jsonify({
            "type": "error",
            "message": "Failed to process request."
        }), 500

    # ── Successful query / greeting / conversational ────────────────────────
    combined_message = result.get("combined_message", "")
    return _success_response(result["response_payload"], message=combined_message)