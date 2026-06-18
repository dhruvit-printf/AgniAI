"""
admin_routes.py
===============
HTTP transport layer for the AgniAI Admin Chatbot.

This file contains ONLY HTTP routing logic. All business logic lives in
admin_pipeline.py, which is the single source of truth for query execution.

Endpoints:
  /api/admin/health    — Health check (dotnet connectivity)
  /api/admin/classify  — Debug: classify intent without executing
  /api/admin/chat      — Main chat endpoint (calls execute_admin_query)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from flask import Blueprint, jsonify, request

from admin_pipeline import (
    execute_admin_query,
    classify_for_debug,
    check_dotnet_health,
)

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────
ADMIN_RATE_LIMIT = os.getenv("ADMIN_RATE_LIMIT", "20 per minute")

# ── Blueprint ──────────────────────────────────────────────────────────────
admin_bp = Blueprint("admin", __name__, url_prefix="/api/admin")


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
    health_data = check_dotnet_health()
    return _success_response(health_data, message="Admin health check complete.")


@admin_bp.route("/classify", methods=["POST"])
def admin_classify():
    """Classify-only endpoint — returns intent JSON without calling .NET. For debugging."""
    body    = request.get_json(force=True, silent=True) or {}
    message = (body.get("message") or "").strip()

    if not message:
        return _error_response("message field is required.", 400)

    result = classify_for_debug(message, body)

    # Greeting / conversational
    if result.get("type") in ("greeting", "conversational"):
        return _success_response(
            result.get("response_data", {}),
            message=result.get("greeting_message", ""),
        )

    # Classification result — remove internal 'type' key before sending
    response_data = {k: v for k, v in result.items() if k != "type"}
    return _success_response(response_data, message="Intent classified successfully.")


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

    # ── Greeting / conversational ───────────────────────────────────────────
    if result_type in ("greeting", "conversational"):
        return _success_response(
            result["response_data"],
            message=result["greeting_message"],
        )

    # ── Unrecognised query ──────────────────────────────────────────────────
    if result_type == "unrecognised":
        combined_message = result.get("combined_message", "")
        return _success_response(result["response_payload"], message=combined_message)

    # ── Error ───────────────────────────────────────────────────────────────
    if result_type == "error":
        error_msg = result.get("error_message", "Failed to process request.")
        if "required" in error_msg.lower():
            return _error_response(error_msg, 400)
        return _error_response(error_msg, 502)

    # ── Successful query ────────────────────────────────────────────────────
    combined_message = result.get("combined_message", "")
    return _success_response(result["response_payload"], message=combined_message)