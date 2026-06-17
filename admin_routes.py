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
  2. Resolve company and platoon mentions from query to numeric IDs
  3. classify_admin_intent()    → structured intent dict (Python internal)
  4. format_admin_payload()     → camelCase JSON payload for .NET
     └─ includes commandId, batchId, platoonId, companyId, fullName from the frontend request
  5. POST payload to .NET       → https://<DOTNET_API_BASE_URL>/api/AiCommand/execute
  6. generate_intro_message()   → LLM generates a single clean intro sentence
  7. Return raw .NET data directly in data.result — NO formatting layer
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
from config import _is_greeting, _is_small_talk, _is_patriotic, GREETING_PHRASES
from query_planner import plan_query, QueryType
from result_combiner import intersect_results, merge_results, compare_results
from admin_context import AdminSessionContext

# ── (NEW IMPORT) Import Named Entity Resolver ─────────────────────────────────
from admin_entity_resolver import resolve_entities_from_query

_session_context = AdminSessionContext()

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
# CONVERSATIONAL DETECTION
# =============================================================================

def _is_admin_conversational(message: str) -> bool:
    """Return True if the message is casual / conversational and not a data query."""
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
            "performance", "leave", "attendance", "medical", "equipment",
            "verification", "distribution", "skills", "top", "bottom",
            "score", "marks", "grading", "bpet", "ppt", "firing", "drill",
            "performer", "performers", "overdue", "absconded", "bmi",
            "disease", "present", "strength", "issued", "procured",
            "pending", "completed", "sport", "blood", "unit", "overall",
            "average", "pass", "fail", "improvement", "drop", "attempt",
            "comparison", "compare", "summary", "ranking",
        }
        if not any(t in _ADMIN_SIGNAL_WORDS for t in tokens):
            return True

    return False


# =============================================================================
# GREETING BUILDER
# =============================================================================

def _build_greeting_response(body: Dict, session_id: str) -> tuple:
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
# CONVERSATIONAL RESPONSE BUILDER
# =============================================================================

def _build_conversational_response(message: str, body: Dict, session_id: str) -> tuple:
    """Generate a natural conversational reply for non-greeting casual messages."""
    import random
    import datetime as _dt

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
        from config import OLLAMA_URL, DEFAULT_MODEL

        prompt = (
            f"You are AgniAI Command Console — an intelligent admin assistant that helps "
            f"commanding officers review and analyze Agniveer data such as Performance, "
            f"Attendance, Leave, Medical, Equipment, Verification, Distribution, and Skills.\n"
            f"The admin officer's name/title is \"{admin_name}\". They sent this casual message: \"{message}\"\n"
            f"The current time of day greeting is \"{time_greeting}\".\n\n"
            f"IMPORTANT: Start your reply with \"{time_greeting}, {admin_name}!\" or \"{time_greeting}, {admin_name}.\" "
            f"then continue naturally.\n"
            f"Reply warmly, professionally, and naturally in 1-2 sentences as a command console assistant would. "
            f"Be respectful and military-professional in tone. If they asked how you are, respond naturally. "
            f"If they said thanks, acknowledge it warmly. If they said something patriotic, match that energy with pride. "
            f"End by offering to help with Agniveer data, reports, or analytics they may need.\n"
            f"Do NOT use markdown, bullets, or headers. Do NOT be robotic. Do NOT mention aspirants or recruitment."
        )
        payload = {
            "model":    DEFAULT_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream":   False,
            "options": {
                "temperature": 0.7,
                "num_predict": 80,
                "num_ctx":     512,
            },
        }
        resp = _req.post(OLLAMA_URL, json=payload, timeout=(8, 30))
        resp.raise_for_status()
        llm_reply = (
            resp.json()
            .get("message", {})
            .get("content", "")
            .strip()
            .strip('"\'')
        )
        if llm_reply and 5 <= len(llm_reply) <= 300:
            response_data: Dict[str, Any] = {"type": "conversational"}
            if session_id and session_id != "admin-default":
                response_data["sessionId"] = session_id
            return response_data, llm_reply
    except Exception as exc:
        logger.debug("LLM conversational reply failed, using fallback: %s", exc)

    fallbacks = [
        f"{time_greeting}, {admin_name}. I'm here and ready to help. What data would you like to review?",
        f"{time_greeting}, {admin_name}! Let me know if you need any reports or insights.",
        f"{time_greeting}, {admin_name}. All systems are operational. What would you like to look into today?",
        f"{time_greeting}, {admin_name}. At your service — feel free to ask about Performance, Attendance, Leave, or any other module.",
    ]

    response_data = {"type": "conversational"}
    if session_id and session_id != "admin-default":
        response_data["sessionId"] = session_id

    return response_data, random.choice(fallbacks)


# =============================================================================
# INTRO MESSAGE GENERATOR
# =============================================================================

_INTRO_TEMPLATES: Dict[tuple, str] = {
    # Performance
    ("Performance", "TopPerformers"):      "These assessment results highlight the strongest performers in the evaluation.",
    ("Performance", "LowestPerformers"):   "These results identify the individuals requiring additional training support.",
    ("Performance", "AverageScore"):       "The average scores outline overall achievement levels across the group.",
    ("Performance", "PassPercentage"):     "Pass rates reflect the percentage of trainees meeting the assessment standards.",
    ("Performance", "FailPercentage"):     "Fail rates identify the proportion of trainees currently below standard.",
    ("Performance", "GradeFilter"):        "The grade filter results show performance by the selected grading category.",
    ("Performance", "GradingSummary"):     "The grading summary provides a breakdown of performance achievements.",
    ("Performance", "OverallPerformers"):  "Overall performance metrics highlight trainee progress across all criteria.",
    ("Performance", "Improvement"):        "These records highlight the trainees showing positive performance growth.",
    ("Performance", "Drop"):               "These trends identify trainees experiencing a decline in assessment scores.",
    ("Performance", "SectionSummary"):     "The section summary provides a clear view of performance across individual modules.",
    ("Performance", "AttemptWise"):        "Attempt-wise statistics track trainee progress across successive evaluation cycles.",
    ("Performance", "BestAttempt"):        "Best attempt outcomes reflect peak trainee achievements in this evaluation.",
    ("Performance", "Comparison"):         "This comparison highlights achievement differences across the selected categories.",
    # Leave
    ("Leave", "MostLeaveTaken"):           "Leave patterns highlight the person with the highest absence rate.",
    ("Leave", "LeastLeaveTaken"):          "Leave summaries identify the person with the highest duty presence.",
    ("Leave", "CurrentLeave"):             "Current leave records outline person availability across the unit.",
    ("Leave", "AbscondedLeave"):           "These records flag persons currently absent without official leave.",
    # Medical
    ("Medical", "ActiveCases"):            "This summary captures current active cases undergoing medical attention.",
    ("Medical", "BMIAnalysis"):            "BMI records outline fitness levels and weight distribution across persons.",
    ("Medical", "DiseaseStats"):           "Health records highlight the most common medical cases reported recently.",
    # Attendance
    ("Attendance", "MonthlyAttendance"):   "Monthly attendance trends provide a clear view of person participation.",
    ("Attendance", "PresentToday"):        "Today's attendance records outline current person presence on campus.",
    ("Attendance", "StrengthBreakdown"):   "The strength breakdown captures unit headcount and active person counts.",
    # Verification
    ("Verification", "PendingVerification"):   "Verification files track documents currently awaiting official review.",
    ("Verification", "CompletedVerification"): "These records confirm files that have cleared the verification process.",
    # Equipment
    ("Equipment", "EquipmentStats"):           "This inventory summary reflects current equipment counts and status.",
    ("Equipment", "OverdueEquipment"):         "These records flag issued gear currently overdue for return.",
    ("Equipment", "ReturnedEquipment"):        "This quality review highlights equipment returned in sub-standard condition.",
    ("Equipment", "IssuedItems"):              "Here is the complete list of items issued to Agniveers.",
    ("Equipment", "ProcuredItems"):            "Here is the complete list of items procured by Agniveers.",
    # Distribution
    ("Distribution", "LatestDistribution"):    "Recent distribution logs track the latest issue of supplies and gear.",
    ("Distribution", "DistributionByUnit"):    "Distribution logs trace supply allocation across different units.",
    ("Distribution", "UnassignedItems"):       "Supply records outline items currently unassigned to any unit.",
    ("Distribution", "TopUnit"):               "This summary highlights the unit receiving the largest supply allocation.",
    # Skills
    ("Skills", "BySport"):                     "Sport rosters track athletic participation and team assignments.",
    ("Skills", "ByClass"):                     "Class rosters group persons by their administrative designations.",
    ("Skills", "BloodGroup"):                  "Medical profiles outline the blood group distribution across the group.",
}


def _build_intro_prompt(
    question: str,
    intent: Dict[str, Any],
    dotnet_data: Any,
) -> str:
    category    = intent.get("category", "")
    subcategory = intent.get("subcategory", "")
    number      = intent.get("number")
    section     = intent.get("section", "")
    leave_type  = intent.get("leave_type", "")
    grading     = intent.get("grading", "")
    unit_name   = intent.get("unit_name", "")
    sport       = intent.get("sport", "")
    class_name  = intent.get("class", "")

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
        context_parts.append(f"Leave type: {leave_type}")
    if grading:
        context_parts.append(f"Grading filter: {grading}")
    if unit_name:
        context_parts.append(f"Unit filter: {unit_name}")
    if sport:
        context_parts.append(f"Sport filter: {sport}")
    if class_name:
        context_parts.append(f"Class filter: {class_name}")

    context_str = "\n".join(context_parts)

    return (
        "You are AgniAI, an intelligent military training assistant.\n\n"
        "Generate ONE short introductory sentence for the data being shown to the admin.\n\n"
        "STRICT RULES:\n"
        "1. ONE sentence only. End with a period.\n"
        "2. 10 to 20 words maximum.\n"
        "3. NEVER mention any person's name, rank, or ID.\n"
        "4. NEVER mention any score, number, percentage, or statistic.\n"
        "5. NEVER say what the result is — only describe what type of data is shown.\n"
        "6. NEVER use: retrieved, fetched, generated, extracted, shown below, listed below.\n"
        "7. Do not ask questions.\n"
        "8. No markdown, no bullets, no quotes.\n"
        "9. Sound like a professional assistant introducing a report.\n\n"
        "GOOD EXAMPLES:\n"
        "Attempt-wise improvement data is ready for your review.\n"
        "Here is the leave status across persons for the current period.\n"
        "Performance rankings for the selected section are available below.\n\n"
        f"Admin question: {question}\n\n"
        f"Context:\n{context_str}\n\n"
        "Generate only the sentence."
    )


def _sanitize_intro(text: str) -> str:
    import re

    text = text.strip().strip('"\'')

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

    text = re.sub(r"\s*\([^)]{0,200}\)\s*$", "", text).strip()
    text = re.sub(r"\s*[Nn]ote\s*[:—–-].*$", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"\s*[Pp]lease note.*$", "", text, flags=re.DOTALL).strip()

    sentences = re.split(r"(?<=[.!?])\s+", text)
    if sentences:
        text = sentences[0].strip()

    if text.endswith("?"):
        text = text.rstrip("?").rstrip() + "."

    if text and text[-1] not in ".!":
        text += "."

    if re.search(r"\b\d+\b", text):
        return ""

    words = text.split()
    for word in words[1:]:
        clean_word = re.sub(r"[^A-Za-z]", "", word)
        if clean_word and clean_word[0].isupper() and clean_word.lower() not in {
            "agniveer", "bpet", "ppt", "drill", "firing", "medical",
            "attendance", "leave", "equipment", "performance", "verification",
            "distribution", "skills", "unit", "platoon", "batch",
        }:
            return ""

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
                "temperature": 0.2,
                "num_predict": 60,
                "num_ctx":     512,
                "stop": ["Note:", "Please note", "\n", "?"],
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

        clean_text = _sanitize_intro(raw_text)

        if clean_text and 10 <= len(clean_text) <= 200:
            logger.debug("LLM intro (sanitized): %s", clean_text)
            return clean_text

        logger.debug(
            "LLM intro rejected after sanitize (raw=%r clean=%r), using template",
            raw_text, clean_text,
        )

    except Exception as exc:
        logger.debug("Ollama intro generation failed, using template: %s", exc)

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
    company_id = _safe_int(data.get("companyId", data.get("company_id")))  # (UPDATED) Support explicit companyId

    if command_id is not None:
        filters["commandId"] = command_id
    if batch_id is not None:
        filters["batchId"] = batch_id
    if platoon_id is not None:
        filters["platoonId"] = platoon_id
    if company_id is not None:
        filters["companyId"] = company_id  # (UPDATED)

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


def _execute_multi_operation(
    query_plan,
    id_filters: Dict,
    full_name: str,
) -> tuple[Optional[Dict], Optional[str]]:
    """
    Execute a multi-operation query plan.

    Runs each sub-operation through _call_dotnet() sequentially,
    then combines results based on the plan's query_type.

    Returns (combined_result, error_message).
    """
    results = []
    labeled_results = []

    for i, op in enumerate(query_plan.operations):
        payload = dict(op.dotnet_payload)
        payload.update(id_filters)
        if full_name:
            payload["fullName"] = full_name

        logger.info(
            "Multi-op %d/%d: sending to .NET: %s",
            i + 1, len(query_plan.operations), json.dumps(payload),
        )

        data, error = _call_dotnet(payload)
        if error:
            logger.warning(
                "Multi-op %d/%d failed: %s", i + 1, len(query_plan.operations), error,
            )
            return None, f"Sub-query {i + 1} failed: {error}"

        results.append(data)
        label = op.intent_result.get("category", f"Query {i + 1}")
        labeled_results.append((label, data))

    # Combine based on query type
    if query_plan.query_type == QueryType.CROSS_FILTER:
        combined = intersect_results(results, primary_index=0)
    elif query_plan.query_type == QueryType.COMPARISON:
        combined = compare_results(labeled_results)
    elif query_plan.query_type == QueryType.MULTI_INDEPENDENT:
        combined = merge_results(labeled_results)
    else:
        # Should not happen, but fallback
        combined = results[0] if results else {}

    return combined, None


def _success_response(data: Dict, http_status: int = 200, message: str = ""):
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
    body    = request.get_json(force=True, silent=True) or {}
    message = (body.get("message") or "").strip()

    if not message:
        return _error_response("message field is required.", 400)

    session_id = _get_session_id(body)

    if _is_admin_conversational(message):
        cleaned = message.lower().strip().rstrip("!?.,;")
        if _is_greeting(cleaned):
            response_data, greeting_message = _build_greeting_response(body, session_id)
        else:
            response_data, greeting_message = _build_conversational_response(message, body, session_id)
        return _success_response(response_data, message=greeting_message)

    id_filters = _get_id_filters(body)
    full_name  = _get_full_name(body)

    # ── (NEW) Resolve Named Entities (Company / Platoon) ─────────────────────
    resolved_entities = resolve_entities_from_query(
        message,
        existing_company_id=id_filters.get("companyId"),
        existing_platoon_id=id_filters.get("platoonId")
    )
    if resolved_entities.get("companyId") is not None:
        id_filters["companyId"] = resolved_entities["companyId"]
    if resolved_entities.get("platoonId") is not None:
        id_filters["platoonId"] = resolved_entities["platoonId"]

    message        = admin_normalize_query(message)
    intent_result  = classify_admin_intent(message)
    dotnet_payload = format_admin_payload(intent_result)
    dotnet_payload.update(id_filters)

    if full_name:
        dotnet_payload["fullName"] = full_name

    # ── Query plan (for debugging) ─────────────────────────────────────────
    query_plan = plan_query(message)

    response_data: Dict[str, Any] = {
        "queryType":     query_plan.query_type.value,
        "confidence":    round(query_plan.confidence, 2),
        "queryPlan":     query_plan.to_dict(),
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
    """
    start_time = time.time()
    body       = request.get_json(force=True, silent=True) or {}
    message    = (body.get("message") or "").strip()
    session_id = _get_session_id(body)
    id_filters = _get_id_filters(body)
    full_name  = _get_full_name(body)

    # ── Greeting / conversational short-circuit ─────────────────────────────
    if _is_admin_conversational(message):
        cleaned = message.lower().strip().rstrip("!?.,;")
        if _is_greeting(cleaned):
            response_data, greeting_message = _build_greeting_response(body, session_id)
        else:
            response_data, greeting_message = _build_conversational_response(message, body, session_id)
        return _success_response(response_data, message=greeting_message)

    if not message:
        return _error_response("message field is required and cannot be empty.", 400)

    # ── (NEW) Resolve Named Entities (Company / Platoon) ─────────────────────
    resolved_entities = resolve_entities_from_query(
        message,
        existing_company_id=id_filters.get("companyId"),
        existing_platoon_id=id_filters.get("platoonId")
    )
    if resolved_entities.get("companyId") is not None:
        id_filters["companyId"] = resolved_entities["companyId"]
    if resolved_entities.get("platoonId") is not None:
        id_filters["platoonId"] = resolved_entities["platoonId"]

    elapsed_ms = lambda: round((time.time() - start_time) * 1000)

    # ── Step 1: Normalize & plan ────────────────────────────────────────────
    message    = admin_normalize_query(message)
    query_plan = plan_query(message)

    logger.info(
        "Query plan: session=%s type=%s confidence=%.2f ops=%d reason=%s",
        session_id,
        query_plan.query_type.value,
        query_plan.confidence,
        len(query_plan.operations),
        query_plan.reasoning,
    )

    # ── Multi-operation path (CROSS_FILTER / COMPARISON / MULTI_INDEPENDENT)
    if (query_plan.query_type != QueryType.SIMPLE
            and query_plan.confidence >= 0.5
            and len(query_plan.operations) >= 2):

        # Use the first operation's intent as the "primary" for intro/logging
        primary_intent  = query_plan.operations[0].intent_result
        primary_payload = query_plan.operations[0].dotnet_payload

        logger.info(
            "Admin multi-op: session=%s type=%s ops=%d",
            session_id, query_plan.query_type.value, len(query_plan.operations),
        )

        combined_data, multi_error = _execute_multi_operation(
            query_plan, id_filters, full_name,
        )

        if multi_error:
            logger.warning("Admin multi-op failed: %s", multi_error)
            return _error_response(
                "Unable to fetch data at the moment. Please try again shortly.",
                502,
            )

        intro_message = _generate_intro_message(
            question=message,
            intent=primary_intent,
            dotnet_data=combined_data,
        )

        if combined_data is not None:
            _session_context.update(session_id, message, primary_intent, combined_data)

        response_data: Dict[str, Any] = {
            "queryType":     query_plan.query_type.value,
            "confidence":    round(query_plan.confidence, 2),
            "queryPlan":     query_plan.to_dict(),
            "result":        combined_data if combined_data is not None else {},
            "intent":        primary_intent,
            "dotnetPayload": primary_payload,
        }

        if session_id and session_id != "admin-default":
            response_data["sessionId"] = session_id

        logger.info(
            "Admin multi-op complete: session=%s elapsed=%dms",
            session_id, elapsed_ms(),
        )

        return _success_response(response_data, message=intro_message)

    # ══════════════════════════════════════════════════════════════════════
    # SIMPLE PATH — existing flow, unchanged
    # ══════════════════════════════════════════════════════════════════════

    # ── Step 1: Classify intent ────────────────────────────────────────────
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
        response_data = {
            "queryType":     query_plan.query_type.value,
            "confidence":    round(query_plan.confidence, 2),
            "queryPlan":     query_plan.to_dict(),
            "dotnetPayload": {},
            "result":        None,
            "intent":        intent_result,
        }
        if session_id and session_id != "admin-default":
            response_data["sessionId"] = session_id
        return _success_response(
            response_data,
            message=(
                "Sorry, I was unable to understand your request. "
                "I can help with Performance, Leave, Attendance, Medical, Equipment, "
                "Verification, Distribution, and Skills information. "
                "Please ask a relevant question."
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

    # ── Step 4: Generate intro sentence ───────────────────────────────────
    intro_message = _generate_intro_message(
        question=message,
        intent=intent_result,
        dotnet_data=dotnet_data,
    )

    if dotnet_data is not None:
        _session_context.update(session_id, message, intent_result, dotnet_data)

    # ── Step 5: Build response ─────────────────────────────────────────────
    response_data: Dict[str, Any] = {
        "queryType":     query_plan.query_type.value,
        "confidence":    round(query_plan.confidence, 2),
        "queryPlan":     query_plan.to_dict(),
        "result":        dotnet_data if dotnet_data is not None else {},
        "intent":        intent_result,
        "dotnetPayload": dotnet_payload,
    }

    if session_id and session_id != "admin-default":
        response_data["sessionId"] = session_id

    logger.info(
        "Admin chat complete: session=%s elapsed=%dms",
        session_id,
        elapsed_ms(),
    )

    return _success_response(response_data, message=intro_message)