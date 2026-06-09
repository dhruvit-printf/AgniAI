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
  2. classify_admin_intent()    → structured intent dict (Python internal)
  3. format_admin_payload()     → camelCase JSON payload for .NET
     └─ includes commandId, batchId, platoonId, fullName from the frontend request
  4. POST payload to .NET       → https://<DOTNET_API_BASE_URL>/api/AiCommand/execute
  5. generate_intro_message()   → LLM generates a single clean intro sentence
  6. Return unified response

RESPONSE SHAPE:
  {
    "status":     true,
    "httpStatus": 200,
    "message":    "<single LLM intro sentence — no notes, no extra text>",
    "data": {
      "intent":        { ...classified intent fields... },
      "dotnetPayload": { ...sent to .NET... },
      "result":        <raw .NET response — frontend renders this>,
      "sessionId":     "..."   (only if not admin-default)
    }
  }

  message  → one clean sentence, e.g. "Top 5 BEPT performers retrieved."
  data.result → raw .NET JSON for frontend to render as needed
  data.intent → full intent classification (category, subcategory, filters, confidence)
  data.dotnetPayload → exact payload sent to .NET

FRONTEND REQUEST FIELDS:
  message     — (required) natural-language admin question
  session_id  — (optional) session identifier
  commandId   — (optional, default 0) command ID filter for .NET
  batchId     — (optional, default 0) batch ID filter for .NET
  platoonId   — (optional, default 0) platoon ID filter for .NET
  fullName    — (optional) full name of the commanding officer / user

PORT SEPARATION:
  - Python / Flask  → port 5000  (python app.py)
  - .NET AiCommand  → port 7257  (set DOTNET_API_BASE_URL in .env)
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, Optional

import requests as _requests
from flask import Blueprint, jsonify, request

from admin_intent import admin_normalize_query, classify_admin_intent, format_admin_payload
from config import _is_greeting

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────
DOTNET_API_BASE_URL = os.getenv("DOTNET_API_BASE_URL", "https://localhost:7257")
DOTNET_EXECUTE_URL  = f"{DOTNET_API_BASE_URL}/api/AiCommand/execute"
DOTNET_API_KEY      = os.getenv("DOTNET_API_KEY", "")
DOTNET_TIMEOUT      = int(os.getenv("DOTNET_TIMEOUT", "30"))
ADMIN_RATE_LIMIT    = os.getenv("ADMIN_RATE_LIMIT", "20 per minute")

_skip_raw = os.getenv("DOTNET_SKIP_SSL_VERIFY", os.getenv("DOTNET_VERIFY_SSL", "0"))
DOTNET_VERIFY_SSL = _skip_raw.strip() not in {"1", "true", "True"}

# ── Blueprint ──────────────────────────────────────────────────────────────
admin_bp = Blueprint("admin", __name__, url_prefix="/api/admin")

_dotnet_session = _requests.Session()

if not DOTNET_VERIFY_SSL:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


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
            _limiter.limit(
                ADMIN_RATE_LIMIT,
                override_defaults=True,
            )(admin_chat)
            logger.info("Admin rate limit applied: %s", ADMIN_RATE_LIMIT)
    except Exception as exc:
        logger.warning("Could not register admin rate limit: %s", exc)


# =============================================================================
# GREETING BUILDER
# =============================================================================

def _build_greeting_response(body: Dict, session_id: str) -> tuple:
    """
    Single source of truth for all admin greeting responses.
    Priority for name: fullName → adminName → userName → commanderName → name → "Officer"
    """
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


# =============================================================================
# INTRO MESSAGE GENERATOR
# =============================================================================

# Static fallback templates — used when Ollama is unavailable.
# One clean sentence per (category, subcategory) pair.
# No "Note:", no qualifiers, no follow-up questions.
_INTRO_TEMPLATES: Dict[tuple, str] = {
    # Performance
    ("Performance", "TopPerformers"):      "These assessment results highlight the strongest performers in the evaluation.",
    ("Performance", "LowestPerformers"):   "These results identify the individuals requiring additional training support.",
    ("Performance", "AverageScore"):       "The average scores outline overall achievement levels across the group.",
    ("Performance", "PassPercentage"):     "Pass rates reflect the percentage of trainees meeting the assessment standards.",
    ("Performance", "FailPercentage"):     "Fail rates identify the proportion of trainees currently below standard.",
    ("Performance", "GradeDistribution"):  "The grade distribution shows overall qualification levels across the group.",
    ("Performance", "GradingSummary"):     "The grading summary provides a breakdown of performance achievements.",
    ("Performance", "OverallPerformance"): "Overall performance metrics highlight trainee progress and evaluation outcomes.",
    ("Performance", "Improvement"):        "These records highlight the trainees showing positive performance growth.",
    ("Performance", "Drop"):               "These trends identify trainees experiencing a decline in assessment scores.",
    ("Performance", "SectionSummary"):     "The section summary provides a clear view of performance across individual modules.",
    ("Performance", "AttemptWise"):        "Attempt-wise statistics track trainee progress across successive evaluation cycles.",
    ("Performance", "BestAttempt"):        "Best attempt outcomes reflect peak trainee achievements in this evaluation.",
    ("Performance", "Comparison"):         "This comparison highlights achievement differences across the selected categories.",
    # Leave
    ("Leave", "MostLeaveTaken"):           "Leave patterns highlight the personnel with the highest absence rate.",
    ("Leave", "LeastLeaveTaken"):          "Leave summaries identify the personnel with the highest duty presence.",
    ("Leave", "CurrentLeaveStatus"):       "Current leave records outline personnel availability across the unit.",
    ("Leave", "AbscondedPersonnel"):       "These records flag personnel currently absent without official leave.",
    # Medical
    ("Medical", "ActiveCases"):            "This summary captures current active cases undergoing medical attention.",
    ("Medical", "BMIAnalysis"):            "BMI records outline fitness levels and weight distribution across personnel.",
    ("Medical", "DiseaseStatistics"):      "Health records highlight the most common medical cases reported recently.",
    # Attendance
    ("Attendance", "MonthlyAttendance"):   "Monthly attendance trends provide a clear view of personnel participation.",
    ("Attendance", "PresentToday"):        "Today's attendance records outline current personnel presence on campus.",
    ("Attendance", "StrengthBreakdown"):   "The strength breakdown captures unit headcount and active personnel counts.",
    # Verification
    ("Verification", "PendingVerification"):   "Verification files track documents currently awaiting official review.",
    ("Verification", "CompletedVerification"): "These records confirm files that have cleared the verification process.",
    # Equipment
    ("Equipment", "EquipmentSummary"):         "This inventory summary reflects current equipment counts and status.",
    ("Equipment", "OverdueEquipment"):         "These records flag issued gear currently overdue for return.",
    ("Equipment", "PoorConditionEquipment"):   "This quality review highlights equipment returned in sub-standard condition.",
    # Distribution
    ("Distribution", "LatestDistribution"):    "Recent distribution logs track the latest issue of supplies and gear.",
    ("Distribution", "DistributionByUnit"):    "Distribution logs trace supply allocation across different units.",
    ("Distribution", "UnassignedItems"):       "Supply records outline items currently unassigned to any unit.",
    ("Distribution", "TopUnit"):               "This summary highlights the unit receiving the largest supply allocation.",
    # Skills
    ("Skills", "BySport"):                     "Sport rosters track athletic participation and team assignments.",
    ("Skills", "ByClass"):                     "Class rosters group personnel by their administrative designations.",
    ("Skills", "BloodGroup"):                  "Medical profiles outline the blood group distribution across the group.",
    # Overall
    ("Overall", "OverallRanking"):             "The overall rankings reflect those leading the composite evaluations.",
}


def _build_intro_prompt(
    question: str,
    intent: Dict[str, Any],
    dotnet_data: Any,
) -> str:
    """
    Build the LLM prompt for generating a single clean intro sentence.

    Rules enforced in the prompt:
    - Exactly ONE sentence, ending with a period.
    - No "Note:", no "Please note", no qualifiers, no follow-up questions.
    - No markdown, no bullet points, no JSON.
    - Specific: mention count, section, or filter when present.
    - Do NOT start with "Here is" or "Here are".
    """
    category    = intent.get("category", "")
    subcategory = intent.get("subcategory", "")
    number      = intent.get("number")
    section     = intent.get("section", "")
    leave_type  = intent.get("leave_type", "")
    grading     = intent.get("grading", "")
    unit_name   = intent.get("unit_name", "")
    sport       = intent.get("sport", "")
    class_name  = intent.get("class", "")

    # Build a compact data summary for context
    data_summary = ""
    try:
        actual_data = dotnet_data
        if isinstance(dotnet_data, dict):
            actual_data = (
                dotnet_data.get("data") or
                dotnet_data.get("Data") or
                dotnet_data.get("result") or
                dotnet_data
            )
        if isinstance(actual_data, list) and actual_data:
            data_summary = f"{len(actual_data)} record(s) returned."
        elif isinstance(actual_data, dict):
            count = (
                actual_data.get("total") or
                actual_data.get("count") or
                actual_data.get("Count") or
                actual_data.get("Total")
            )
            if count is not None:
                data_summary = f"Total count: {count}."
        elif isinstance(actual_data, (int, float)):
            data_summary = f"Result value: {actual_data}."
    except Exception:
        pass

    context_parts = []
    if category:
        context_parts.append(f"Module: {category}")
    if subcategory:
        context_parts.append(f"Query type: {subcategory}")
    if number:
        context_parts.append(f"Requested count: {number}")
    if section:
        context_parts.append(f"Section filter: {section}")
    if leave_type:
        context_parts.append(f"Leave type filter: {leave_type}")
    if grading:
        context_parts.append(f"Grading filter: {grading}")
    if unit_name:
        context_parts.append(f"Unit filter: {unit_name}")
    if sport:
        context_parts.append(f"Sport filter: {sport}")
    if class_name:
        context_parts.append(f"Class filter: {class_name}")
    if data_summary:
        context_parts.append(data_summary)

    context_str = "\n".join(context_parts)

    return (
        "You are AgniAI, an intelligent military training and administration assistant.\n\n"
        "Your task is to generate a short introductory sentence for the requested data.\n\n"
        "IMPORTANT RULES:\n"
        "1. Return EXACTLY ONE sentence.\n"
        "2. Maximum 8-20 words.\n"
        "3. Sound like a professional assistant speaking to a human administrator.\n"
        "4. Never sound like a database, API, report generator, or system log.\n"
        "5. Never use:\n"
        "   - retrieved\n"
        "   - fetched\n"
        "   - generated\n"
        "   - extracted\n"
        "   - pulled from database\n"
        "   - shown below\n"
        "   - listed below\n"
        "   - available below\n"
        "   - successfully\n"
        "6. Do not mention technical operations.\n"
        "7. Do not ask questions.\n"
        "8. Do not use markdown, bullets, quotes, or explanations.\n"
        "9. Each response should feel natural and conversational.\n"
        "10. Use different phrasing every time while keeping the same meaning.\n"
        "11. Acknowledge the user's request and provide context about what they are viewing.\n"
        "12. Be confident, concise, and executive in tone.\n\n"
        "GOOD EXAMPLES:\n"
        "The latest attendance trends provide a clear view of personnel participation this month.\n"
        "These assessment results highlight the strongest performers in BEPT.\n"
        "Current leave patterns offer insight into personnel availability across the unit.\n"
        "Recent performance data reveals the trainees making the most significant progress.\n"
        "This overview captures the current medical status across affected personnel.\n"
        "The latest rankings reflect those leading the selected evaluation.\n"
        "Attendance records paint a clear picture of unit readiness this month.\n"
        "Performance outcomes highlight the individuals setting the standard in this assessment cycle.\n"
        "Current statistics offer a concise snapshot of activity across the selected group.\n\n"
        "BAD EXAMPLES:\n"
        "Data retrieved successfully.\n"
        "Monthly attendance statistics have been retrieved.\n"
        "The report is shown below.\n"
        "Here are the results.\n"
        "I have fetched the requested data.\n\n"
        f"Admin question: {question}\n\n"
        f"Context:\n{context_str}\n\n"
        "Generate only the sentence and nothing else."
    )


def _sanitize_intro(text: str) -> str:
    """
    Post-process the LLM output to enforce clean single-sentence output.
    Strips notes, qualifiers, follow-up questions, and LLM meta-commentary.
    """
    import re

    text = text.strip().strip('"\'')

    # --- Strip common LLM meta-commentary prefixes ---
    # Catches: "Here is a possible introduction:", "Here's the intro:",
    # "Here is the introductory sentence:", "Introduction:", etc.
    meta_prefixes = re.compile(
        r"^(?:"
        r"here(?:'s| is)(?: a| the| my)?(?: possible| suggested?)?(?: introductory?| intro)?(?: sentence| line| message| response)?[:\s]*|"
        r"(?:introductory?|intro|opening|response)\s+(?:sentence|line|message)[:\s]*|"
        r"(?:a possible|suggested?) introduction[:\s]*|"
        r"introduction[:\s]*|"
        r"answer[:\s]*|"
        r"response[:\s]*"
        r")",
        re.IGNORECASE,
    )
    text = meta_prefixes.sub("", text).strip()

    # --- Strip parenthetical notes at end ---
    # Catches: "(Note: I've written...)", "(I made this concise)"
    text = re.sub(r"\s*\([^)]{0,200}\)\s*$", "", text).strip()

    # --- Strip "Note:" lines anywhere ---
    text = re.sub(r"\s*[Nn]ote\s*[:—–-].*$", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"\s*[Pp]lease note.*$", "", text, flags=re.DOTALL).strip()

    # --- Keep only the first sentence ---
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if sentences:
        text = sentences[0].strip()

    # --- Remove trailing question marks ---
    if text.endswith("?"):
        text = text.rstrip("?").rstrip() + "."

    # --- Ensure ends with a period ---
    if text and not text[-1] in ".!":
        text += "."

    # --- Final cleanup: if result still looks like meta-commentary, discard it ---
    meta_check = re.compile(
        r"^(?:here(?:'s| is)|i(?:'ve| have)|based on|the following|as requested)",
        re.IGNORECASE,
    )
    if meta_check.match(text):
        return ""

    return text


def _generate_intro_message(
    question: str,
    intent: Dict[str, Any],
    dotnet_data: Any,
) -> str:
    """
    Generate a clean single-sentence intro using Ollama.
    Falls back to static template if Ollama is unavailable or returns bad output.
    Output is always exactly one sentence with no notes, qualifiers, or questions.
    """
    category    = intent.get("category", "")
    subcategory = intent.get("subcategory", "")

    try:
        import requests as _req
        from config import OLLAMA_URL, DEFAULT_MODEL

        prompt   = _build_intro_prompt(question, intent, dotnet_data)
        messages = [{"role": "user", "content": prompt}]
        payload  = {
            "model":    DEFAULT_MODEL,
            "messages": messages,
            "stream":   False,
            "options": {
                "temperature": 0.2,   # Lower temp = more deterministic / less creative
                "num_predict": 60,    # Hard cap — one sentence needs at most ~20 tokens
                "num_ctx":     512,
                "stop": ["Note:", "Please note", "\n", "?"],  # Stop tokens
            },
        }
        resp = _req.post(OLLAMA_URL, json=payload, timeout=(8, 30))
        resp.raise_for_status()
        raw_text = (
            resp.json()
            .get("message", {})
            .get("content", "")
            .strip()
        )

        # Sanitize the output
        clean_text = _sanitize_intro(raw_text)

        # Accept only if it's a valid single sentence of reasonable length
        if clean_text and 10 <= len(clean_text) <= 200:
            logger.debug("LLM intro (sanitized): %s", clean_text)
            return clean_text

        logger.debug("LLM intro rejected after sanitize (raw=%r clean=%r), using template", raw_text, clean_text)

    except Exception as exc:
        logger.debug("Ollama intro generation failed, using template: %s", exc)

    # Static fallback
    key = (category, subcategory)
    if key in _INTRO_TEMPLATES:
        return _INTRO_TEMPLATES[key]

    category_label = category or "requested"
    return f"These records outline the current {category_label.lower()} status across the unit."


# =============================================================================
# HELPERS
# =============================================================================

def _get_session_id(data: Dict) -> str:
    from config import SESSION_HEADER
    session_id = (
        data.get("session_id") or
        data.get("sessionId") or
        request.headers.get(SESSION_HEADER) or
        request.headers.get("X-Session-Id") or ""
    ).strip()
    return session_id or "admin-default"


def _get_id_filters(data: Dict) -> Dict[str, int]:
    def _safe_int(value) -> Optional[int]:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    filters = {}

    command_id = _safe_int(data.get("commandId", data.get("command_id")))
    batch_id   = _safe_int(data.get("batchId",   data.get("batch_id")))
    platoon_id = _safe_int(data.get("platoonId", data.get("platoon_id")))

    if command_id is not None:
        filters["commandId"] = command_id
    if batch_id is not None:
        filters["batchId"] = batch_id
    if platoon_id is not None:
        filters["platoonId"] = platoon_id

    return filters


def _get_full_name(data: Dict) -> str:
    return (
        data.get("fullName") or
        data.get("full_name") or
        ""
    ).strip()


def _call_dotnet(payload: Dict) -> tuple[Any, Optional[str]]:
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


def _success_response(data: Dict, http_status: int = 200, message: str = ""):
    """
    Unified success envelope.

    Shape:
      {
        "status":     true,
        "httpStatus": 200,
        "message":    "<single clean sentence>",
        "data": {
          "intent":        {...},
          "dotnetPayload": {...},
          "result":        <raw .NET response>,
          "sessionId":     "..."
        }
      }
    """
    return jsonify({
        "status":     True,
        "httpStatus": http_status,
        "message":    message,
        "data":       data,
    }), http_status


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

    return _success_response(
        {
            "pythonStatus":  "ok",
            "dotnetBackend": "reachable" if dotnet_ok else "unreachable",
            "dotnetUrl":     DOTNET_EXECUTE_URL,
            "pythonPort":    5000,
        },
        message="Admin health check complete.",
    )


@admin_bp.route("/classify", methods=["POST"])
def admin_classify():
    """
    Classify-only endpoint — returns intent JSON and .NET payload
    without calling .NET. Useful for debugging.
    """
    body       = request.get_json(force=True, silent=True) or {}
    message    = (body.get("message") or "").strip()

    if not message:
        return _error_response("message field is required.", 400)

    session_id = _get_session_id(body)

    # ── Greeting short-circuit ─────────────────────────────────────────────
    if _is_greeting(message.lower().strip().rstrip("!?.,;")):
        response_data, greeting_message = _build_greeting_response(body, session_id)
        return _success_response(response_data, message=greeting_message)

    id_filters    = _get_id_filters(body)
    full_name     = _get_full_name(body)

    message = admin_normalize_query(message)
    intent_result  = classify_admin_intent(message)
    dotnet_payload = format_admin_payload(intent_result)
    dotnet_payload.update(id_filters)

    if full_name:
        dotnet_payload["fullName"] = full_name

    response_data: Dict[str, Any] = {
        "intent":        intent_result,
        "dotnetPayload": dotnet_payload,
    }
    if session_id and session_id != "admin-default":
        response_data["sessionId"] = session_id

    return _success_response(
        response_data,
        message="Intent classified successfully.",
    )


@admin_bp.route("/chat", methods=["POST"])
def admin_chat():
    """
    Main admin chat endpoint.

    Full pipeline:
      1. Classify intent from admin's natural-language question
      2. Build .NET payload (camelCase), including fullName if provided
      3. POST to .NET AiCommand/execute
      4. Generate clean single-sentence intro via LLM (or static template)
      5. Return unified JSON response

    RESPONSE SHAPE:
      {
        "status":     true,
        "httpStatus": 200,
        "message":    "<single sentence — no notes, no questions>",
        "data": {
          "dotnetPayload": {
            "category":  "Performance",
            "operation": "Top",
            "n":         5,
            "section":   "BEPT"
          },
          "result":    <raw .NET response object>,
          "sessionId": "..."    (only present if provided by frontend)
        }
      }
    """
    start_time = time.time()
    body       = request.get_json(force=True, silent=True) or {}
    message    = (body.get("message") or "").strip()
    session_id = _get_session_id(body)
    id_filters = _get_id_filters(body)
    full_name  = _get_full_name(body)

    # ── Greeting short-circuit ─────────────────────────────────────────────
    if _is_greeting(message.lower().strip().rstrip("!?.,;")):
        response_data, greeting_message = _build_greeting_response(body, session_id)
        return _success_response(response_data, message=greeting_message)

    if not message:
        return _error_response("message field is required and cannot be empty.", 400)

    elapsed_ms = lambda: round((time.time() - start_time) * 1000)

    # ── Step 1: Classify intent ────────────────────────────────────────────
    message = admin_normalize_query(message)
    intent_result = classify_admin_intent(message)
    logger.info(
        "Admin intent: session=%s category=%s subcategory=%s confidence=%s",
        session_id,
        intent_result.get("category"),
        intent_result.get("subcategory"),
        intent_result.get("confidence"),
    )

    # ── Unrecognised query ─────────────────────────────────────────────────
    if intent_result.get("category") is None:
        return _success_response(
            {
                "dotnetPayload": {},
                "result":        None,
            },
            message=(
                "Please ask a performance, leave, attendance, medical, equipment, "
                "verification, distribution, or skills related question."
            ),
        )

    # ── Step 2: Build .NET payload ─────────────────────────────────────────
    dotnet_payload = format_admin_payload(intent_result)
    dotnet_payload.update(id_filters)

    if full_name:
        dotnet_payload["fullName"] = full_name

    logger.info("Sending to .NET: %s", json.dumps(dotnet_payload))

    # ── Step 3: Call .NET backend ──────────────────────────────────────────
    dotnet_data, dotnet_error = _call_dotnet(dotnet_payload)

    if dotnet_error:
        logger.warning("Admin .NET call failed: %s", dotnet_error)
        return _error_response(
            "Unable to fetch data at the moment. Please try again shortly.",
            502,
        )

    # ── Step 4: Generate clean single-sentence intro ───────────────────────
    intro_message = _generate_intro_message(
        question=message,
        intent=intent_result,
        dotnet_data=dotnet_data,
    )

    # ── Step 5: Build response ─────────────────────────────────────────────
    response_data: Dict[str, Any] = {
        "dotnetPayload": dotnet_payload,
        "result":        dotnet_data if dotnet_data is not None else {},
    }

    if session_id and session_id != "admin-default":
        response_data["sessionId"] = session_id

    logger.info(
        "Admin chat complete: session=%s elapsed=%dms",
        session_id,
        elapsed_ms(),
    )

    return _success_response(response_data, message=intro_message)