"""
admin_routes.py
===============
Admin chatbot Flask blueprint for AgniAI.

Registers:
  POST /api/admin/chat       — main admin chat endpoint
  POST /api/admin/classify   — classify-only (no .NET call; for debugging)
  GET  /api/admin/health     — admin route health check

Flow for /api/admin/chat:
  1. Receive admin question from frontend
  2. classify_admin_intent() → structured intent dict (Python internal)
  3. format_admin_payload()  → camelCase JSON payload for .NET
  4. POST payload to .NET  https://<DOTNET_API_BASE_URL>/api/AiCommand/execute
  5. Forward raw .NET JSON response directly to frontend (no formatting)

NOTE — NO FORMATTER:
  The raw JSON from .NET is forwarded to the frontend as-is under the "data"
  key. The frontend is responsible for rendering the structured data.
  admin_formatter.py is NOT used in this pipeline.

IMPORTANT PORT SEPARATION:
  - Python / Flask runs on port 5000  (python app.py  OR  gunicorn ... :5000)
  - .NET AiCommand API runs on port 7257 (or whatever DOTNET_API_BASE_URL is set to)
  These MUST be different ports. If they are the same, admin chat calls itself
  and fails with a connection error or wrong-route response.

Configuration (via environment variables):
  DOTNET_API_BASE_URL   — default: https://localhost:7257  (.NET app port)
  DOTNET_API_KEY        — optional X-Api-Key header for .NET endpoint
  DOTNET_SKIP_SSL_VERIFY— "1" to skip SSL verification (self-signed localhost cert)
                          "0" (default) to verify SSL normally
  ADMIN_RATE_LIMIT      — default: "20 per minute"
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, Optional

import requests as _requests
from flask import Blueprint, jsonify, request

from admin_intent import classify_admin_intent, format_admin_payload

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────
# Python runs on 5000, .NET runs on 7257 — these MUST differ.
DOTNET_API_BASE_URL = os.getenv("DOTNET_API_BASE_URL", "https://localhost:7257")
DOTNET_EXECUTE_URL  = f"{DOTNET_API_BASE_URL}/api/AiCommand/execute"
DOTNET_API_KEY      = os.getenv("DOTNET_API_KEY", "")
DOTNET_TIMEOUT      = int(os.getenv("DOTNET_TIMEOUT", "30"))
ADMIN_RATE_LIMIT    = os.getenv("ADMIN_RATE_LIMIT", "20 per minute")

# Set DOTNET_SKIP_SSL_VERIFY=1 in .env to skip SSL verification (self-signed
# localhost cert). Default is "0" (verify SSL normally).
_skip_raw = os.getenv("DOTNET_SKIP_SSL_VERIFY", os.getenv("DOTNET_VERIFY_SSL", "0"))
DOTNET_VERIFY_SSL = _skip_raw.strip() not in {"1", "true", "True"}
# DOTNET_VERIFY_SSL=True  → requests verifies the cert (production)
# DOTNET_VERIFY_SSL=False → requests skips verification (localhost self-signed)

# ── Blueprint ──────────────────────────────────────────────────────────────
admin_bp = Blueprint("admin", __name__, url_prefix="/api/admin")

_dotnet_session = _requests.Session()

# Disable SSL warnings for self-signed localhost cert when verification is off
if not DOTNET_VERIFY_SSL:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# =============================================================================
# RATE LIMITER — imported from app.py's shared limiter
# =============================================================================

def _get_limiter():
    """
    Import the shared Flask-Limiter instance from app.py at call time to avoid
    circular imports at module load. Returns None if limiter is unavailable.
    """
    try:
        from app import _limiter
        return _limiter
    except (ImportError, AttributeError):
        return None


def _register_rate_limits(app):
    """
    Called from app.py after the blueprint is registered:
        from admin_routes import _register_rate_limits
        _register_rate_limits(app)

    This wires ADMIN_RATE_LIMIT to the /api/admin/chat route using the
    already-initialised Flask-Limiter from app.py.
    """
    try:
        from app import _limiter
        if _limiter is not None:
            _limiter.limit(
                ADMIN_RATE_LIMIT,
                override_defaults=True,
            )(admin_chat)
            logger.info("Admin rate limit applied: %s", ADMIN_RATE_LIMIT)
    except Exception as exc:
        logger.warning("Could not register admin rate limit: %s", exc)


# =============================================================================
# HELPERS
# =============================================================================

def _get_session_id(data: Dict) -> str:
    from config import SESSION_HEADER
    session_id = (
        data.get("session_id") or
        request.headers.get(SESSION_HEADER) or
        request.headers.get("X-Session-Id") or ""
    ).strip()
    return session_id or "admin-default"


def _call_dotnet(payload: Dict) -> tuple[Any, Optional[str]]:
    """
    POST payload to .NET AiCommand/execute.
    Returns (response_data, error_message).
    error_message is None on success.
    """
    headers = {"Content-Type": "application/json"}
    if DOTNET_API_KEY:
        headers["X-Api-Key"] = DOTNET_API_KEY

    logger.debug(
        "Calling .NET: URL=%s payload=%s",
        DOTNET_EXECUTE_URL,
        json.dumps(payload),
    )

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
        return None, (
            f"Cannot connect to .NET backend at {DOTNET_EXECUTE_URL}. "
            f"Is the .NET service running on the correct port? ({exc})\n"
            f"Tip: Python runs on port 5000, .NET should run on a different port "
            f"(set DOTNET_API_BASE_URL in .env)."
        )
    except _requests.Timeout:
        return None, f"Backend timed out after {DOTNET_TIMEOUT}s."
    except _requests.RequestException as exc:
        return None, f"Backend request failed: {exc}"
    except ValueError as exc:
        return None, f"Backend returned invalid JSON: {exc}"


def _json_error(message: str, status_code: int = 400):
    return jsonify({"success": False, "error": message}), status_code


# =============================================================================
# ROUTES
# =============================================================================

@admin_bp.route("/health")
def admin_health():
    """Quick health check for the admin chatbot subsystem."""
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

    return jsonify({
        "success":         True,
        "status":          "ok",
        "dotnet_backend":  "reachable" if dotnet_ok else "unreachable",
        "dotnet_url":      DOTNET_EXECUTE_URL,
        "python_port":     5000,
    })


@admin_bp.route("/classify", methods=["POST"])
def admin_classify():
    """
    Classify-only endpoint — returns the intent JSON and the exact .NET payload
    without actually calling .NET. Useful for debugging the classifier.

    Request JSON:
      { "message": "Who are the top 5 performers in BEPT?" }

    Response JSON:
      {
        "success": true,
        "intent": {
          "category": "Performance",
          "subcategory": "TopPerformers",
          "number": 5,
          "section": "BEPT",
          "confidence": "high",
          ...
        },
        "dotnet_payload": {
          "category": "Performance",
          "operation": "Top",
          "section": "BEPT",
          "n": 5
        }
      }
    """
    data = request.get_json(force=True, silent=True) or {}
    message = (data.get("message") or "").strip()

    if not message:
        return _json_error("message field is required.", 400)

    intent_result  = classify_admin_intent(message)
    dotnet_payload = format_admin_payload(intent_result)

    return jsonify({
        "success":        True,
        "intent":         intent_result,
        "dotnet_payload": dotnet_payload,
    })


@admin_bp.route("/chat", methods=["POST"])
def admin_chat():
    """
    Main admin chat endpoint.

    Receives a natural-language question, classifies intent, builds the .NET
    payload, calls .NET, and forwards the raw .NET response to the frontend.
    No formatting is applied — the frontend receives the structured JSON as-is.

    Request JSON:
      {
        "message":    "Who are the top 5 performers in BEPT?",
        "session_id": "admin-user-1"   (optional)
      }

    Response JSON (success):
      {
        "success":        true,
        "intent":         { "category": "Performance", "subcategory": "TopPerformers", ... },
        "dotnet_payload": { "category": "Performance", "operation": "Top", "n": 5, "section": "BEPT" },
        "data":           { ...raw JSON from .NET... },
        "session_id":     "admin-user-1",
        "elapsed_ms":     142
      }

    Response JSON (.NET error):
      {
        "success":        false,
        "error":          "Backend returned HTTP 404: ...",
        "intent":         { ... },
        "dotnet_payload": { ... },
        "session_id":     "admin-user-1",
        "elapsed_ms":     30
      }

    Response JSON (unrecognised query):
      {
        "success":    true,
        "recognised": false,
        "message":    "I'm not sure what you're asking about...",
        "intent":     { "category": null, ... },
        "session_id": "admin-user-1"
      }
    """
    start_time = time.time()
    data       = request.get_json(force=True, silent=True) or {}
    message    = (data.get("message") or "").strip()
    session_id = _get_session_id(data)

    if not message:
        return _json_error("message field is required and cannot be empty.", 400)

    # ── Step 1: Classify intent ────────────────────────────────────────────
    intent_result = classify_admin_intent(message)
    logger.info(
        "Admin intent: session=%s category=%s subcategory=%s confidence=%s",
        session_id,
        intent_result.get("category"),
        intent_result.get("subcategory"),
        intent_result.get("confidence"),
    )

    # ── Handle unrecognised queries ────────────────────────────────────────
    if intent_result.get("category") is None:
        elapsed_ms = round((time.time() - start_time) * 1000)
        return jsonify({
            "success":    True,
            "recognised": False,
            "message": (
                "I'm not sure what you're asking about. "
                "You can ask me about Performance, Leave, Medical, "
                "Attendance, Verification, Equipment, Distribution, "
                "or Skills/Roster data.\n\n"
                "For example: \"Show me the top 10 performers in BEPT\" or "
                "\"How many personnel are on leave today?\""
            ),
            "intent":     intent_result,
            "session_id": session_id,
            "elapsed_ms": elapsed_ms,
        })

    # ── Step 2: Build .NET payload ─────────────────────────────────────────
    dotnet_payload = format_admin_payload(intent_result)
    logger.info("Sending to .NET: %s", json.dumps(dotnet_payload))

    # ── Step 3: Call .NET backend ──────────────────────────────────────────
    dotnet_data, dotnet_error = _call_dotnet(dotnet_payload)
    elapsed_ms = round((time.time() - start_time) * 1000)

    if dotnet_error:
        logger.warning("Admin .NET call failed: %s", dotnet_error)
        return jsonify({
            "success":        False,
            "error":          dotnet_error,
            "intent":         intent_result,
            "dotnet_payload": dotnet_payload,
            "session_id":     session_id,
            "elapsed_ms":     elapsed_ms,
        }), 502

    # ── Step 4: Forward raw .NET response to frontend ──────────────────────
    # No formatting is applied. The frontend receives the .NET JSON as-is
    # under the "data" key, along with intent metadata for context.
    logger.info(
        "Admin chat complete: session=%s elapsed=%dms",
        session_id,
        elapsed_ms,
    )

    return jsonify({
        "success":        True,
        "recognised":     True,
        "intent":         intent_result,
        "dotnet_payload": dotnet_payload,
        "data":           dotnet_data,       # raw .NET response, untouched
        "session_id":     session_id,
        "elapsed_ms":     elapsed_ms,
    })