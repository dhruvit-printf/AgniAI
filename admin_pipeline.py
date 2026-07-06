"""
admin_pipeline.py
=================
Single source of truth for the AgniAI admin query execution pipeline.

Both HTTP (admin_routes.py) and WebSocket (websocket_routes.py) call
execute_admin_query() — the ONLY orchestration function.

This module owns:
  - .NET API configuration and communication
  - Conversational / greeting detection
  - The full pipeline: Planner → Intent → .NET → Combiner → Report → Response
  - Session context management
  - Audit logging
  - OpenTelemetry span instrumentation (when enabled)

No other module should contain pipeline orchestration logic.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple


from admin_context import AdminSessionContext
from admin_entity_resolver import resolve_entities_from_query
from audit_logger import set_audit_context, write_audit_log
from config import _is_greeting, _is_patriotic, _is_small_talk
from context_engine import context_engine
from conversation_detector import (
    build_conversational_response as build_conversation_payload,
)
from conversation_detector import (
    is_conversational_query,
)
from dotnet_executor import _call_dotnet
from feature_flags import get_flags
from intent_engine.admin_intent import (
    PayloadValidationError,
    classify_admin_intent,
    format_admin_payload,
)
from intent_engine.intent_schema import agniveer_no_required
from intent_engine.query_planner import QueryType, plan_query
from metadata_builder import build_metadata
from normalized_models import extract_records as _extract_records
from query_normalizer import admin_normalize_query, clean_query
from query_understanding_engine import understand_query
from report_generator import generate_report, get_fallback_report
from response_builder import build_response
from result_combiner import combine_results
from schedule_enrichment import enrich_schedule_by_company
from schemas import (
    AnalysisModel,
    CombinedResponseModel,
    ConclusionModel,
    DotNetPayloadModel,
    DotNetResponseModel,
    FinalResponseModel,
    IntentModel,
    MetadataModel,
    PredictionModel,
    SuggestedQuestionModel,
)
from suggested_question_engine import generate_suggested_questions
from telemetry import (
    SPAN_BUILD_RESPONSE,
    SPAN_CALL_DOTNET,
    SPAN_CLASSIFY_ADMIN_INTENT,
    SPAN_COMBINE_RESULTS,
    SPAN_GENERATE_REPORT,
    SPAN_PLAN_QUERY,
    request_id_var,
    session_id_var,
    span,
    trace_context,
    trace_id_var,
)
from visualization_intent import build_visualization_intent

logger = logging.getLogger(__name__)


SLOW_QUERY_THRESHOLD = float(os.getenv("SLOW_QUERY_THRESHOLD_SEC", "5.0"))
from metrics import metrics_collector

_session_context = AdminSessionContext()


# =============================================================================
# HELPERS
# =============================================================================


def ensure_agniveer_no_in_data(
    data: Any,
    max_depth: int = 10,
    visited: Optional[set] = None,
) -> None:
    """Ensure all nested records in data contain agniveerNo if agniveerId is present.

    Prevents infinite recursion on circular structures using a visited set and a max_depth limit.
    """
    if max_depth <= 0:
        return
    if visited is None:
        visited = set()

    data_id = id(data)
    if data_id in visited:
        return
    visited.add(data_id)

    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                id_val = None
                for key in ("agniveerId", "AgniveerId", "AgniVeerId", "id", "Id"):
                    if key in item and item[key] is not None:
                        id_val = item[key]
                        break
                if "agniveerNo" not in item and id_val is not None:
                    item["agniveerNo"] = str(id_val)
            ensure_agniveer_no_in_data(item, max_depth - 1, visited)

    elif isinstance(data, dict):
        id_val = None
        for key in ("agniveerId", "AgniveerId", "AgniVeerId", "id", "Id"):
            if key in data and data[key] is not None:
                id_val = data[key]
                break
        if "agniveerNo" not in data and id_val is not None:
            data["agniveerNo"] = str(id_val)

        for value in data.values():
            ensure_agniveer_no_in_data(value, max_depth - 1, visited)


def map_query_type(qt: QueryType) -> str:
    """Map QueryType enum to string label for the response."""
    if qt == QueryType.CROSS_FILTER:
        return "cross_filter"
    elif qt in (QueryType.COMPARE, QueryType.COMPARISON):
        return "compare"
    elif qt in (QueryType.MULTI_INDEPENDENT, QueryType.MULTI_OPERATION):
        return "multi_independent"
    elif qt == QueryType.TREND:
        return "trend"
    elif qt == QueryType.DISTRIBUTION:
        return "distribution"
    else:
        return "simple"


def _sanitize_error(err_msg: Any) -> str:
    """Scrub raw database response body or detail dumps from error messages."""
    if not err_msg:
        return ""
    err_str = str(err_msg)
    if "Backend returned HTTP" in err_str:
        # e.g. "Backend returned HTTP 400: <json-body>" -> "Backend returned HTTP 400"
        parts = err_str.split(":", 1)
        if len(parts) > 0:
            return parts[0]
    return err_str


# ID fields that must never be sent as null / empty / zero (Rule 9)
_ID_FIELDS: frozenset = frozenset({"companyId", "platoonId", "batchId", "agniveerNo"})

_NO_CARRY_FORWARD_CATEGORIES = frozenset(
    {
        "disqualified",
        "personaldetail",
    }
)


def _strip_empty_id_fields(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Final payload sanitiser — called immediately before every _call_dotnet() invocation.

    Removes any ID field whose value is None, empty string, or 0.
    This is the last line of defence against phantom IDs reaching the backend.

    Rules enforced:
      Rule 5  — payload must contain only meaningful filters.
      Rule 9  — validate and strip unresolved fields before calling .NET.
    """
    cleaned: Dict[str, Any] = {}
    for k, v in payload.items():
        if k in _ID_FIELDS:
            # Only include if it's a genuine, non-empty, non-zero value
            if v is None or v == "" or v == 0:
                continue
        cleaned[k] = v
    return cleaned


_AGNIVEER_NO_MISSING_MESSAGE = "Please provide agniveer number"

# Friendly, conversational fallbacks — the raw exception/status text is always
# logged server-side for diagnostics, but never shown to the user verbatim.
_BACKEND_UNAVAILABLE_MESSAGE = (
    "I'm having trouble reaching the records system right now. "
    "Please try again in a moment."
)
_REQUEST_UNPROCESSABLE_MESSAGE = (
    "I couldn't quite work out how to run that request. "
    "Could you try rephrasing your question?"
)


def _agniveer_no_missing_response(session_id: str) -> Dict[str, Any]:
    """Clarification response for categories/operations that require agniveerNo."""
    payload = build_conversation_payload(
        _AGNIVEER_NO_MISSING_MESSAGE,
        session_id=session_id,
        query_type="clarification",
    )
    return {
        "type": "clarification",
        "response_payload": payload,
        "combined_message": _AGNIVEER_NO_MISSING_MESSAGE,
        "execution_time_ms": 0,
    }


def _get_session_id(data: Dict) -> str:
    """Extract session ID from request body or headers."""
    import uuid as _uuid
    session_id = (data.get("session_id") or data.get("sessionId") or "").strip()
    if not session_id:
        session_id = _uuid.uuid4().hex
        logger.warning("request without session_id — generated ephemeral id %s", session_id)
    return session_id


def _get_id_filters(data: Dict) -> Dict[str, Any]:
    """Extract numeric ID filters from the request body."""

    def _safe_int(value) -> Optional[int]:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    filters = {}
    command_id = _safe_int(data.get("commandId", data.get("command_id")))
    batch_id = _safe_int(data.get("batchId", data.get("batch_id")))
    platoon_id = _safe_int(data.get("platoonId", data.get("platoon_id")))
    company_id = _safe_int(data.get("companyId", data.get("company_id")))

    if command_id is not None:
        filters["commandId"] = command_id
    if batch_id is not None:
        filters["batchId"] = batch_id
    if platoon_id is not None:
        filters["platoonId"] = platoon_id
    if company_id is not None:
        filters["companyId"] = company_id
    return filters


def _get_full_name(data: Dict) -> str:
    """Extract the admin's full name from the request body."""
    return (data.get("fullName") or data.get("full_name") or "").strip()


def _get_cache_scope(data: Dict) -> Dict[str, Any]:
    """Collect the request fields that can change cached answer scope."""
    scope = _get_id_filters(data)
    full_name = _get_full_name(data)
    if full_name:
        scope["fullName"] = full_name
    return scope


_INTENT_FIELD_ALIASES: Dict[str, Tuple[str, ...]] = {
    "category": ("category",),
    "operation": ("operation", "subcategory"),
    "section": ("section",),
    "sub_section": ("sub_section", "subSection"),
    "metric": ("metric",),
    "comparison_target": ("comparison_target", "comparisonTarget"),
    "batch_id": ("batch_id", "batchId"),
    "platoon_id": ("platoon_id", "platoonId"),
    "company_id": ("company_id", "companyId"),
    "agniveer_no": ("agniveer_no", "agniveerNo"),
    "class": ("class",),
    "sport": ("sport",),
    "gender": ("gender",),
    "rank": ("rank",),
    "date_range": ("date_range", "dateRange"),
    "query_type": ("query_type", "queryType"),
    "group_by": ("group_by", "groupBy"),
    "sort_by": ("sort_by", "sortBy"),
    "aggregation": ("aggregation",),
    "top_n": ("top_n", "topN"),
}

_VISUALIZATION_FIELD_ALIASES: Dict[str, Tuple[str, ...]] = {
    "presentation": ("presentation", "widget", "widgetType", "widget_type"),
    "chart_type": ("chart_type", "chartType"),
    "comparison": ("comparison",),
    "trend": ("trend",),
    "group_by": ("group_by", "groupBy"),
    "metric": ("metric",),
    "widget_hint": ("widget_hint", "widgetHint"),
    "data_type": ("data_type", "dataType", "type"),
}

_DISPLAY_ONLY_VISUALIZATION_FIELDS = {
    "presentation",
    "chart_type",
    "widget_hint",
}


def _extract_frontend_intent(body: Dict[str, Any]) -> Dict[str, Any]:
    intent: Dict[str, Any] = {}
    body_intent = body.get("intent")
    if isinstance(body_intent, dict):
        for key, value in body_intent.items():
            if value not in (None, "", [], {}):
                intent[key] = value
    for canonical, aliases in _INTENT_FIELD_ALIASES.items():
        if canonical in intent and intent[canonical] not in (None, "", [], {}):
            continue
        for alias in aliases:
            value = body.get(alias)
            if value not in (None, "", [], {}):
                intent[canonical] = value
                break
    filters = intent.get("filters")
    if not isinstance(filters, dict):
        filters = {}
    for field in (
        "batch_id",
        "platoon_id",
        "company_id",
        "agniveer_no",
        "section",
        "sub_section",
        "class",
        "sport",
    ):
        value = intent.get(field)
        if value not in (None, "", [], {}):
            filters[field] = value
    intent["filters"] = filters
    return intent


def _extract_frontend_visualization_intent(body: Dict[str, Any]) -> Dict[str, Any]:
    visual: Dict[str, Any] = {}
    frontend_hint_present = False
    body_intent = body.get("intent")
    if isinstance(body_intent, dict):
        for key, aliases in _VISUALIZATION_FIELD_ALIASES.items():
            for alias in aliases:
                value = body_intent.get(alias)
                if value not in (None, "", [], {}):
                    visual[key] = value
                    if key in _DISPLAY_ONLY_VISUALIZATION_FIELDS:
                        frontend_hint_present = True
                    break

    for canonical, aliases in _VISUALIZATION_FIELD_ALIASES.items():
        if canonical in visual and visual[canonical] not in (None, "", [], {}):
            continue
        for alias in aliases:
            value = body.get(alias)
            if value not in (None, "", [], {}):
                visual[canonical] = value
                if canonical in _DISPLAY_ONLY_VISUALIZATION_FIELDS:
                    frontend_hint_present = True
                break

    widgets_value = body.get("widgets")
    if isinstance(widgets_value, list) and widgets_value:
        visual.setdefault("widgets", widgets_value)

    widget_hint = str(visual.get("widget_hint") or "").strip().lower()
    presentation = str(visual.get("presentation") or "").strip().lower()
    if widget_hint and not presentation:
        if widget_hint in {"card", "cards"}:
            visual["presentation"] = "cards"
        elif widget_hint in {"table", "grid"}:
            visual["presentation"] = "table"
        elif widget_hint in {"chart", "bar", "line", "pie"}:
            visual["presentation"] = "chart"
            if widget_hint in {"bar", "line", "pie"}:
                visual["chart_type"] = widget_hint
    if presentation in {"card", "cards"}:
        visual["presentation"] = "cards"
    elif presentation in {"table", "grid"}:
        visual["presentation"] = "table"
    elif presentation in {"chart", "graph", "plot"}:
        visual["presentation"] = "chart"

    if frontend_hint_present:
        visual["frontend_override"] = True

    return visual


def _merge_intents(*sources: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key, value in source.items():
            if value in (None, "", [], {}):
                continue
            # Rule 7: Do not merge empty/zero/null ID fields
            if key in _ID_FIELDS and (value == "" or value == 0):
                continue
            if key == "filters" and isinstance(value, dict):
                existing = merged.get("filters")
                if not isinstance(existing, dict):
                    existing = {}
                for fk, fv in value.items():
                    if fv not in (None, "", [], {}):
                        # Rule 7: Do not merge empty/zero/null ID fields inside filters
                        if fk in _ID_FIELDS and (fv == "" or fv == 0):
                            continue
                        existing.setdefault(fk, fv)
                merged["filters"] = existing
                continue
            merged.setdefault(key, value)
    return merged


def _extract_combined_record_count(combined_result: Any) -> int:
    if isinstance(combined_result, dict):
        if isinstance(combined_result.get("records"), list):
            return len(combined_result["records"])
        if isinstance(combined_result.get("sections"), list):
            total = 0
            for section in combined_result["sections"]:
                if isinstance(section, dict) and isinstance(section.get("data"), list):
                    total += len(section["data"])
            return total
        if isinstance(combined_result.get("sides"), list):
            total = 0
            for side in combined_result["sides"]:
                if isinstance(side, dict) and isinstance(side.get("data"), list):
                    total += len(side["data"])
            return total
    if isinstance(combined_result, list):
        return len(combined_result)
    return 0


def _log_combination_summary(
    *,
    question: str,
    intent: Dict[str, Any],
    qtype: str,
    labeled_results: List[Tuple[str, Any]],
    raw_results: List[Any],
    combined_result: Any,
) -> None:
    if qtype not in (
        "cross_filter",
        "comparison",
        "compare",
        "multi_independent",
        "multi_operation",
    ):
        return

    input_counts = []
    for idx, (label, data) in enumerate(labeled_results):
        if isinstance(data, dict) and data.get("unavailable") is True:
            count = 0
        else:
            count = len(_extract_records(data))
        input_counts.append(
            {
                "index": idx + 1,
                "label": label,
                "count": count,
            }
        )

    output_count = _extract_combined_record_count(combined_result)
    logger.info(
        json.dumps(
            {
                "question": question,
                "category": intent.get("category"),
                "subcategory": intent.get("subcategory"),
                "type": intent.get("type") or qtype,
                "input": input_counts,
                "outputCount": output_count,
                "rawCount": len(raw_results),
            },
            ensure_ascii=False,
        )
    )


def _validate_model_payload(model_cls, payload: Any, context: str) -> bool:
    """Validate payload against schema. Returns True if valid, False if invalid.
    Logs drift but does NOT raise — schema mismatch must not crash the pipeline."""
    try:
        model_cls.model_validate(payload)
        return True
    except Exception as exc:
        logger.warning(
            json.dumps(
                {
                    "message": "Schema validation drift (non-fatal)",
                    "context": context,
                    "model": model_cls.__name__,
                    "error": str(exc),
                }
            )
        )
        return False


def _log_stage_duration(stage: str, duration_ms: float, **extra: Any) -> None:
    event = {"stage": stage, "duration_ms": round(duration_ms, 2)}
    event.update({k: v for k, v in extra.items() if v is not None})
    logger.info(event)


# =============================================================================
# CONVERSATIONAL DETECTION
# =============================================================================

_ADMIN_SIGNAL_WORDS = {
    "performance",
    "leave",
    "attendance",
    "medical",
    "equipment",
    "verification",
    "distribution",
    "skills",
    "top",
    "bottom",
    "score",
    "marks",
    "grading",
    "bpet",
    "ppt",
    "firing",
    "drill",
    "performer",
    "performers",
    "overdue",
    "absconded",
    "bmi",
    "disease",
    "present",
    "strength",
    "issued",
    "procured",
    "pending",
    "completed",
    "sport",
    "blood",
    "unit",
    "overall",
    "average",
    "pass",
    "fail",
    "improvement",
    "drop",
    "attempt",
    "comparison",
    "compare",
    "summary",
    "ranking",
    "verified",
    "verify",
    "verification",
    "approved",
    "cleared",
    "rejected",
    "responded",
    "agniveer",
    "agniveers",
}

_ADMIN_SIGNAL_PHRASES = (
    "verified agniveers",
    "completed verification",
    "pending verification",
    "not responded",
    "approved",
    "cleared",
    "rejected",
)


def _is_admin_conversational(message: str) -> bool:
    """Check if the message is a greeting or small talk (not a data query)."""
    cleaned = message.lower().strip().rstrip("!?.,;")

    if is_conversational_query(cleaned):
        return True
    if _is_greeting(cleaned):
        return True
    if _is_small_talk(cleaned):
        return True
    if _is_patriotic(cleaned):
        return True

    tokens = cleaned.split()
    if len(tokens) <= 3:
        if not any(t in _ADMIN_SIGNAL_WORDS for t in tokens) and not any(
            phrase in cleaned for phrase in _ADMIN_SIGNAL_PHRASES
        ):
            return True

    return False


# =============================================================================
# GREETING / CONVERSATIONAL BUILDERS
# =============================================================================


def _get_admin_name(body: Dict) -> str:
    """Extract admin name from request body using multiple field aliases."""
    return (
        body.get("fullName")
        or body.get("adminName")
        or body.get("userName")
        or body.get("commanderName")
        or body.get("commander_name")
        or body.get("name")
        or "Officer"
    ).strip()


def _build_greeting_response(body: Dict, session_id: str) -> Tuple[Dict, str]:
    """Build a time-aware greeting response."""
    import datetime as _dt
    import random

    admin_name = _get_admin_name(body)

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


def _build_conversational_response(
    message: str, body: Dict, session_id: str, trace_id: Optional[str] = None
) -> Tuple[Dict, str]:
    """Build a conversational response using LLM or fallback."""
    import datetime as _dt
    import random

    admin_name = _get_admin_name(body)

    hour = _dt.datetime.now().hour
    if 5 <= hour < 12:
        time_greeting = "Good Morning"
    elif 12 <= hour < 17:
        time_greeting = "Good Afternoon"
    else:
        time_greeting = "Good Evening"

    try:
        from config import DEFAULT_MODEL, OLLAMA_URL, ollama_session

        prompt = (
            f"You are AgniAI Command Console — an intelligent admin assistant that helps "
            f"commanding officers review and analyze Agniveer data.\n"
            f'The admin officer\'s name/title is "{admin_name}". They sent: "{message}"\n'
            f'Time greeting: "{time_greeting}".\n\n'
            f'Start with "{time_greeting}, {admin_name}!" then reply warmly and '
            f"professionally in 1-2 sentences. Offer to help with Agniveer data.\n"
            f"No markdown, no bullets."
        )
        payload: Dict[str, Any] = {
            "model": DEFAULT_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.7, "num_predict": 80, "num_ctx": 512},
        }
        resp = ollama_session.post(OLLAMA_URL, json=payload, timeout=(3, 10))
        resp.raise_for_status()
        llm_reply = (
            resp.json().get("message", {}).get("content", "").strip().strip("\"'")
        )
        if llm_reply and 5 <= len(llm_reply) <= 300:
            response_data: Dict[str, Any] = {"type": "conversational"}
            if session_id and session_id != "admin-default":
                response_data["sessionId"] = session_id
            return response_data, llm_reply
    except Exception as exc:
        logger.warning(
            json.dumps(
                {
                    "message": "LLM conversational reply failed, using fallback",
                    "trace_id": trace_id or "N/A",
                    "error": str(exc),
                }
            )
        )

    fallbacks = [
        f"{time_greeting}, {admin_name}. I'm here and ready to help. What data would you like to review?",
        f"{time_greeting}, {admin_name}! Let me know if you need any reports or insights.",
        f"{time_greeting}, {admin_name}. All systems are operational. What would you like to look into today?",
    ]

    response_data = {"type": "conversational"}
    if session_id and session_id != "admin-default":
        response_data["sessionId"] = session_id

    return response_data, random.choice(fallbacks)


# =============================================================================
# MAIN PIPELINE — THE SINGLE SOURCE OF TRUTH
# =============================================================================


def execute_admin_query(
    user_query: str,
    body: Dict[str, Any],
    session_id: Optional[str] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    trace_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute the complete admin query pipeline.

    This is the ONLY orchestration function. Both HTTP and WebSocket
    transports call this function.
    """
    if not trace_id:
        trace_id = uuid.uuid4().hex

    def _notify(stage: str) -> None:
        """Fire progress_callback if one was provided."""
        if progress_callback is not None:
            try:
                progress_callback(stage)
            except Exception as exc:
                logger.exception("progress_callback(%s) failed: %s", stage, exc)

    start_time = time.time()

    entity_resolution_duration = 0.0
    planning_duration = 0.0
    planner_duration = 0.0
    intent_duration = 0.0
    dotnet_duration = 0.0
    combiner_duration = 0.0
    widget_duration = 0.0
    response_assembly_duration = 0.0
    report_duration = 0.0
    total_duration = 0.0
    # The old code used 'filter_query' which is a stale query type name that does not match 'simple' used elsewhere in the pipeline and vocabulary.
    qtype_str = "simple"
    audit_success = True
    audit_error_type: Optional[str] = None

    # ── Pipeline state — declared up-front so all branches can reference them ─
    comparison_context = None
    comparison_datasets_info: List[Any] = []
    failed_filters: List[Any] = []
    widget_start: float = 0.0
    response_dotnet_payload: Any = []

    request_id = uuid.uuid4().hex
    if not session_id or session_id == "admin-default":
        session_id = uuid.uuid4().hex
        logger.warning(f"No persistent sessionId provided. Assigned random session ID: {session_id}. Conversational memory disabled for this request.")

    token_req = request_id_var.set(request_id)
    token_trace = trace_id_var.set(trace_id)
    token_sess = session_id_var.set(session_id)
    set_audit_context(question=user_query or "", intent={"type": "Unknown"})

    carry_forward_filters: Dict[str, Any] = {}

    # ── Context Resolution ────────────────────────────────────────────────────
    # Resolve follow-up queries against the last 10 interactions for this session.
    # Returns either the original query ("fresh") or a fully reconstructed one.
    _original_user_query = user_query
    _ctx_resolution = context_engine.resolve(session_id, user_query or "")
    if _ctx_resolution.needs_clarification:
        _clarification_msg = (
            _ctx_resolution.clarification_question or "Could you clarify your question?"
        )
        _clarification_payload = build_conversation_payload(
            _clarification_msg,
            session_id=session_id,
            query_type="clarification",
        )
        return {
            "type": "clarification",
            "response_payload": _clarification_payload,
            "combined_message": _clarification_msg,
            "execution_time_ms": 0,
        }
    if _ctx_resolution.context_source != "fresh":
        logger.info(
            json.dumps(
                {
                    "message": "Context engine resolved follow-up",
                    "original": _original_user_query,
                    "resolved": _ctx_resolution.resolved_query,
                    "source": _ctx_resolution.context_source,
                    "session_id": session_id,
                }
            )
        )
        user_query = _ctx_resolution.resolved_query
        carry_forward_filters = dict(_ctx_resolution.carry_forward_filters or {})

    # Initialise entity dict so it's always bound when add_interaction() fires
    resolved_entities: Dict[str, Any] = dict(_ctx_resolution.resolved_entities or {})



    try:
        message = clean_query(user_query or "").strip()
        frontend_intent = _extract_frontend_intent(body)
        frontend_visualization_intent = _extract_frontend_visualization_intent(body)
        id_filters = _get_id_filters(body)
        full_name = _get_full_name(body)
        semantic_understanding = understand_query(message)

        # ── Greeting / conversational short-circuit ──────────────────────────
        if semantic_understanding.get("conversational") or _is_admin_conversational(
            message
        ):
            intent_start = time.time()
            if _is_greeting(message):
                _, reply_text = _build_greeting_response(body, session_id)
                qtype = "greeting"
            else:
                reply_text = (
                    "I can help with administrative data, reports, and analysis."
                )
                qtype = "conversational"
            intent_duration = time.time() - intent_start
            total_duration = time.time() - start_time
            response_payload = build_conversation_payload(
                reply_text,
                session_id=session_id,
                query_type=qtype,
            )
            response_payload.setdefault("metadata", {})
            response_payload["metadata"].setdefault("timings", {})
            response_payload["metadata"]["timings"]["intentDurationMs"] = round(
                intent_duration * 1000, 2
            )
            response_payload["metadata"]["executionTimeMs"] = round(
                total_duration * 1000
            )
            set_audit_context(
                question=user_query or message,
                intent=response_payload.get("intent") or {"category": qtype},
            )
            metrics_collector.inc_requests(qtype)
            metrics_collector.inc_success(qtype)
            write_audit_log(
                question=user_query or message,
                intent=response_payload.get("intent") or {},
            )
            return {
                "type": qtype,
                "response_payload": response_payload,
                "combined_message": response_payload.get("message", ""),
                "execution_time_ms": round(total_duration * 1000),
            }

        if not message:
            return {
                "type": "error",
                "error_message": "Empty query provided.",
            }

        # ── Step 1: Resolve Named Entities ───────────────────────────────────
        resolved_agniveer_no = None

        with span(SPAN_PLAN_QUERY, trace_id=trace_id):
            planner_start = time.time()
            _notify("planner")
            entity_resolution_start = time.time()

            existing_co = id_filters.get("companyId")
            existing_pl = id_filters.get("platoonId")
            existing_ba = id_filters.get("batchId")

            # Load conversation carryover context only for follow-up turns.
            # get_recent() returns None once the entry exceeds its TTL, so a
            # stale single-slot record can't be read back arbitrarily later.
            if _ctx_resolution.context_source != "fresh":
                prev_history = _session_context.get_recent(session_id)
                if prev_history:
                    prev_intent = prev_history.get("intent") or {}

                    new_cat = semantic_understanding.get("category")
                    old_cat = prev_intent.get("category")

                    if new_cat and old_cat and new_cat != old_cat and new_cat != "Unknown":
                        logger.info(f"Category shift ({old_cat} -> {new_cat}). Dropping conversational filters.")
                        carry_forward_filters.clear()
                    else:
                        if existing_co is None:
                            existing_co = prev_intent.get("company_id") or prev_intent.get(
                                "companyId"
                            )
                        if existing_pl is None:
                            existing_pl = prev_intent.get("platoon_id") or prev_intent.get(
                                "platoonId"
                            )
                        if existing_ba is None:
                            existing_ba = prev_intent.get("batch_id") or prev_intent.get(
                                "batchId"
                            )
                        resolved_agniveer_no = prev_intent.get(
                            "agniveer_no"
                        ) or prev_intent.get("agniveerNo")

            new_resolved = resolve_entities_from_query(
                message,
                existing_company_id=existing_co,
                existing_platoon_id=existing_pl,
                existing_batch_id=existing_ba,
                trace_id=trace_id,
                session_id=session_id,
            )
            resolved_entities.update(new_resolved)
            entity_resolution_duration = time.time() - entity_resolution_start
            logger.info(
                {
                    "stage": "entity_resolution_time",
                    "duration_ms": round(entity_resolution_duration * 1000, 2),
                    "trace_id": trace_id,
                    "session_id": session_id,
                }
            )
            resolved_company = resolved_entities.get("companyId")
            if resolved_company is not None:
                id_filters["companyId"] = int(resolved_company)
            resolved_platoon = resolved_entities.get("platoonId")
            if resolved_platoon is not None:
                id_filters["platoonId"] = int(resolved_platoon)
            resolved_batch = resolved_entities.get("batchId")
            if resolved_batch is not None:
                id_filters["batchId"] = int(resolved_batch)
            # agniveerNo is a string filter — stored separately

            resolved_agniveer_no = (
                resolved_entities.get("agniveerNo") or resolved_agniveer_no
            )

            planning_start = time.time()
            message = admin_normalize_query(message)
            query_plan = plan_query(message)
            planning_duration = time.time() - planning_start
            planner_duration = time.time() - planner_start
            logger.info(
                {
                    "stage": "planner_time",
                    "duration_ms": round(planner_duration * 1000, 2),
                    "trace_id": trace_id,
                    "session_id": session_id,
                    "query_type": query_plan.query_type.value,
                }
            )

        _notify("intent")

        # ── Step 2: Intent Classification ────────────────────────────────────
        with span(SPAN_CLASSIFY_ADMIN_INTENT, trace_id=trace_id):
            intent_start = time.time()
            raw_results: List[Any] = []
            labeled_results: List[Tuple[str, Any]] = []
            primary_intent: Dict[str, Any] = {}
            operation_count: int = 1
            partial_failure = False
            failed_sections = []

            if (
                query_plan.query_type
                in (
                    QueryType.CROSS_FILTER,
                    QueryType.COMPARE,
                    QueryType.COMPARISON,
                    QueryType.MULTI_INDEPENDENT,
                    QueryType.MULTI_OPERATION,
                )
                and len(query_plan.operations) >= 2
            ):

                qtype_str = map_query_type(query_plan.query_type)
                operation_count = len(query_plan.operations)

                logger.info(
                    json.dumps(
                        {
                            "message": "Query plan compiled",
                            "trace_id": trace_id,
                            "session_id": session_id,
                            "query_type": qtype_str,
                            "confidence": query_plan.confidence,
                            "operation_count": operation_count,
                            "reasoning": query_plan.reasoning,
                        }
                    )
                )

                # Validate each sub-op intent
                for op in query_plan.operations:
                    _validate_model_payload(
                        IntentModel, op.intent_result, "multi.intent"
                    )

                # Categories like personaldetail (always) and Attendance
                # (except Present) require an agniveerNo on every operation.
                for op in query_plan.operations:
                    op_agniveer_no = (
                        op.intent_result.get("agniveer_no") or resolved_agniveer_no
                    )
                    if agniveer_no_required(
                        op.intent_result.get("category"),
                        op.intent_result.get("operation"),
                    ) and not op_agniveer_no:
                        return _agniveer_no_missing_response(session_id)

                intent_duration = time.time() - intent_start
                logger.info(
                    {
                        "stage": "classifier_time",
                        "duration_ms": round(intent_duration * 1000, 2),
                        "trace_id": trace_id,
                        "session_id": session_id,
                        "query_type": qtype_str,
                    }
                )

                # ── Step 3: Execute .NET API Call(s) ─────────────────────────
                with span(SPAN_CALL_DOTNET, trace_id=trace_id):
                    dotnet_start = time.time()
                    _notify("dotnet")

                    max_workers = min(
                        len(query_plan.operations),
                        int(os.getenv("DOTNET_MAX_PARALLEL", "4")),
                    )

                    response_dotnet_payload = [None] * len(query_plan.operations)

                    def run_op(idx, op):
                        payload = dict(op.dotnet_payload)
                        # Each comparison/multi-op fragment may name its own
                        # company/platoon/batch (e.g. "... of Arora" vs
                        # "... of Thorat company") — resolve per-fragment so
                        # the two sides don't collapse onto the same
                        # globally-resolved id. Fragments that don't mention
                        # a distinct entity fall back to the shared filters.
                        frag_entities = resolve_entities_from_query(
                            op.raw_fragment,
                            trace_id=trace_id,
                            session_id=session_id,
                        )
                        op_filters = dict(id_filters)
                        for key in ("companyId", "platoonId", "batchId"):
                            frag_value = frag_entities.get(key)
                            if frag_value is not None:
                                op_filters[key] = frag_value
                        payload.update(op_filters)
                        for k, v in carry_forward_filters.items():
                            if payload.get(k) in (None, ""):
                                payload[k] = v
                        if resolved_agniveer_no and not payload.get("agniveerNo"):
                            payload["agniveerNo"] = resolved_agniveer_no
                        if full_name:
                            payload["fullName"] = full_name

                        # ── Rule 9: strip empty/zero/null ID fields before sending ──
                        payload = _strip_empty_id_fields(payload)

                        # Validate DotNetPayloadModel
                        _validate_model_payload(
                            DotNetPayloadModel, payload, "multi.dotnet_payload"
                        )

                        logger.info(
                            json.dumps(
                                {
                                    "message": "Sending multi-op request to .NET",
                                    "trace_id": trace_id,
                                    "session_id": session_id,
                                    "query_type": qtype_str,
                                    "op_index": idx + 1,
                                    "total_ops": len(query_plan.operations),
                                }
                            )
                        )

                        response_dotnet_payload[idx] = dict(payload)

                        op_start = time.time()
                        try:
                            with trace_context(request_id, trace_id, session_id):
                                data, err = _call_dotnet(
                                    payload,
                                    trace_id=trace_id,
                                    session_id=session_id,
                                    query_type=qtype_str,
                                )
                            op_time_ms = round((time.time() - op_start) * 1000, 2)
                            if not err and data is not None:
                                # Validate DotNetResponseModel
                                if isinstance(data, dict):
                                    _validate_model_payload(
                                        DotNetResponseModel,
                                        data,
                                        "multi.dotnet_response",
                                    )
                                elif isinstance(data, list):
                                    _validate_model_payload(
                                        DotNetResponseModel,
                                        {
                                            "success": True,
                                            "commandLabel": op.intent_result.get(
                                                "subcategory"
                                            )
                                            or op.intent_result.get("category")
                                            or "",
                                            "data": data,
                                            "message": "",
                                        },
                                        "multi.dotnet_response_list",
                                    )
                            return idx, op, data, err, op_time_ms
                        except Exception as exc:
                            op_time_ms = round((time.time() - op_start) * 1000, 2)
                            return idx, op, None, str(exc), op_time_ms

                    try:
                        with ThreadPoolExecutor(max_workers=max_workers) as executor:
                            futures = [
                                executor.submit(run_op, i, op)
                                for i, op in enumerate(query_plan.operations)
                            ]
                            results = [f.result() for f in futures]
                    except BaseException as exc:
                        import traceback as _tb

                        total_duration = time.time() - start_time
                        logger.error(
                            json.dumps(
                                {
                                    "message": "Backend thread execution failed",
                                    "trace_id": trace_id,
                                    "session_id": session_id,
                                    "query_type": qtype_str,
                                    "exception": str(exc),
                                    "traceback": _tb.format_exc(),
                                }
                            )
                        )
                        metrics_collector.inc_errors(qtype_str)
                        metrics_collector.record_duration(
                            "pipeline_duration", round(total_duration * 1000, 2)
                        )
                        write_audit_log(
                            trace_id=trace_id,
                            session_id=session_id,
                            query_type=qtype_str,
                            query_duration=round(total_duration * 1000, 2),
                            success=False,
                            error_type="thread_exception",
                        )
                        return {
                            "type": "error",
                            "error_type": "thread_exception",
                            "error_message": _BACKEND_UNAVAILABLE_MESSAGE,
                        }

                    # ── Dependent Schedule enrichment ─────────────────────
                    # A "Schedule" leg with no company/platoon of its own
                    # (e.g. "top performers in BPET and their today's
                    # schedule") can't be resolved from its own fragment
                    # text — the company only becomes known from a peer
                    # leg's records (each carrying a platoonName). Re-fetch
                    # that leg scoped to the company(ies) actually present
                    # in the peer data instead of leaving it as a generic,
                    # unscoped schedule call.
                    for pos, (idx, op, _data, _err, op_time_ms) in enumerate(results):
                        if op.intent_result.get("category") != "Schedule":
                            continue
                        sent_payload = response_dotnet_payload[idx] or {}
                        if sent_payload.get("companyId") or sent_payload.get(
                            "platoonId"
                        ):
                            continue  # already scoped from its own fragment
                        for _, peer_op, peer_data, peer_err, _ in results:
                            if (
                                peer_err
                                or peer_data is None
                                or peer_op.intent_result.get("category")
                                == "Schedule"
                            ):
                                continue
                            enriched = enrich_schedule_by_company(
                                sent_payload,
                                peer_data,
                                trace_id=trace_id,
                                session_id=session_id,
                                query_type=qtype_str,
                            )
                            if enriched is not None:
                                results[pos] = (idx, op, enriched, None, op_time_ms)
                                logger.info(
                                    json.dumps(
                                        {
                                            "message": "Schedule leg enriched from peer records",
                                            "trace_id": trace_id,
                                            "session_id": session_id,
                                            "companies": enriched["data"][
                                                "totalCompanies"
                                            ],
                                        }
                                    )
                                )
                                break

                    dotnet_duration = time.time() - dotnet_start
                    logger.info(
                        {
                            "stage": "dotnet_time",
                            "duration_ms": round(dotnet_duration * 1000, 2),
                            "trace_id": trace_id,
                            "session_id": session_id,
                            "query_type": qtype_str,
                        }
                    )

                    # ── Task 2: Partial failure checks ──
                    all_failed = all(r[3] is not None for r in results)
                    if all_failed:
                        for idx, op, _, err, _ in results:
                            metrics_collector.inc_errors(qtype_str)
                        total_duration = time.time() - start_time
                        metrics_collector.record_duration(
                            "pipeline_duration", round(total_duration * 1000, 2)
                        )
                        write_audit_log(
                            trace_id=trace_id,
                            session_id=session_id,
                            query_type=qtype_str,
                            query_duration=round(total_duration * 1000, 2),
                            success=False,
                            error_type="dotnet_error",
                        )
                        return {
                            "type": "error",
                            "error_message": _BACKEND_UNAVAILABLE_MESSAGE,
                        }

                    # CROSS_FILTER primary failure check
                    if query_plan.query_type == QueryType.CROSS_FILTER:
                        primary_idx, primary_op, primary_data, primary_error, _ = (
                            results[0]
                        )
                        if primary_error:
                            metrics_collector.inc_errors(qtype_str)
                            total_duration = time.time() - start_time
                            metrics_collector.record_duration(
                                "pipeline_duration", round(total_duration * 1000, 2)
                            )
                            write_audit_log(
                                trace_id=trace_id,
                                session_id=session_id,
                                query_type=qtype_str,
                                query_duration=round(total_duration * 1000, 2),
                                success=False,
                                error_type="dotnet_error",
                            )
                            return {
                                "type": "error",
                                "error_message": _BACKEND_UNAVAILABLE_MESSAGE,
                            }

                    # Build raw_results and labeled_results preserving the order
                    failed_filters = []
                    comparison_datasets_info = []
                    for res in results:
                        idx = res[0]
                        op = res[1]
                        dotnet_data = res[2]
                        dotnet_error = res[3]
                        op_time_ms = res[4]

                        resolved_canonical = (
                            op.intent_result.get("section")
                            or op.intent_result.get("sport")
                            or op.intent_result.get("class")
                        )
                        plan_label = None
                        if (
                            hasattr(query_plan, "comparison_execution_plan")
                            and query_plan.comparison_execution_plan
                            and idx < len(query_plan.comparison_execution_plan)
                        ):
                            plan_label = query_plan.comparison_execution_plan[idx].get(
                                "label"
                            )

                        if resolved_canonical:
                            label = resolved_canonical
                        elif plan_label:
                            label = plan_label
                        else:
                            label = (
                                op.intent_result.get("category")
                                or op.raw_fragment.upper()
                            )

                        if dotnet_error:
                            partial_failure = True
                            failed_sections.append(label)

                        if query_plan.query_type == QueryType.CROSS_FILTER:
                            if dotnet_error:
                                metrics_collector.inc_errors(qtype_str)
                                category = op.intent_result.get(
                                    "category", f"Filter {idx + 1}"
                                )
                                failed_filters.append(category)
                            else:
                                ensure_agniveer_no_in_data(dotnet_data)
                                raw_results.append(dotnet_data)
                                label = op.intent_result.get(
                                    "category", f"Query {idx + 1}"
                                )
                                labeled_results.append((label, dotnet_data))
                        elif query_plan.query_type == QueryType.COMPARISON:
                            if dotnet_error:
                                metrics_collector.inc_errors(qtype_str)
                                data_placeholder = {"unavailable": True}
                                raw_results.append(data_placeholder)
                                labeled_results.append((label, data_placeholder))
                            else:
                                ensure_agniveer_no_in_data(dotnet_data)
                                raw_results.append(dotnet_data)
                                labeled_results.append((label, dotnet_data))

                            comparison_datasets_info.append(
                                {
                                    "id": f"dataset_{idx + 1}",
                                    "label": label,
                                    "intent": op.intent_result,
                                    "dotnetPayload": response_dotnet_payload[idx],
                                    "rawData": (
                                        dotnet_data
                                        if not dotnet_error
                                        else {"unavailable": True}
                                    ),
                                    "metadata": {
                                        "endpoint": "api/AiCommand/execute",
                                        "status": (
                                            "SUCCESS" if not dotnet_error else "FAILURE"
                                        ),
                                        "executionTimeMs": op_time_ms,
                                    },
                                }
                            )
                        elif query_plan.query_type == QueryType.MULTI_OPERATION:
                            label = op.intent_result.get(
                                "category", f"Section {idx + 1}"
                            )
                            if dotnet_error:
                                metrics_collector.inc_errors(qtype_str)
                                data_placeholder = {"unavailable": True}
                                raw_results.append(data_placeholder)
                                labeled_results.append((label, data_placeholder))
                            else:
                                ensure_agniveer_no_in_data(dotnet_data)
                                raw_results.append(dotnet_data)
                                labeled_results.append((label, dotnet_data))

                    if query_plan.query_type == QueryType.COMPARISON:
                        comparison_context = {"datasets": comparison_datasets_info}
                    else:
                        comparison_context = None

                    primary_intent = _merge_intents(
                        frontend_intent,
                        *(op.intent_result for op in query_plan.operations),
                    )
                    primary_intent["operations"] = [
                        op.intent_result for op in query_plan.operations
                    ]
                    primary_intent["filters"] = _merge_intents(
                        frontend_intent.get("filters", {}),
                        *(
                            op.intent_result.get("filters", {})
                            for op in query_plan.operations
                        ),
                    )
                    dotnet_duration = time.time() - dotnet_start

            else:
                # FILTER_QUERY / ANALYTICS: single .NET call
                qtype_str = map_query_type(query_plan.query_type)
                operation_count = 1

                classified_intent = (
                    query_plan.operations[0].intent_result
                    if (
                        query_plan.operations
                        and query_plan.operations[0].intent_result.get("category")
                    )
                    else classify_admin_intent(
                        message, resolved_entities=resolved_entities
                    )
                )
                primary_intent = _merge_intents(
                    frontend_intent,
                    classified_intent,
                )
                primary_intent["filters"] = _merge_intents(
                    frontend_intent.get("filters", {}),
                    classified_intent.get("filters", {}),
                )

                if primary_intent.get("category") in _NO_CARRY_FORWARD_CATEGORIES:
                    carry_forward_filters = {}

                # Validate IntentModel
                _validate_model_payload(IntentModel, primary_intent, "single.intent")

                logger.info(
                    json.dumps(
                        {
                            "message": "Query plan compiled",
                            "trace_id": trace_id,
                            "session_id": session_id,
                            "query_type": qtype_str,
                            "confidence": query_plan.confidence,
                            "operation_count": operation_count,
                            "reasoning": query_plan.reasoning,
                        }
                    )
                )

                # Low-confidence or unrecognised query
                if primary_intent.get("category") is None or float(
                    query_plan.confidence
                ) < float(os.getenv("INTENT_CONFIDENCE_THRESHOLD", "0.35")):
                    intent_duration = time.time() - intent_start
                    unrecognised_msg = (
                        "I couldn't understand the query clearly. "
                        "Could you please rephrase it?"
                    )

                    total_duration = time.time() - start_time
                    durations = {
                        "entity_resolution_ms": round(
                            entity_resolution_duration * 1000, 2
                        ),
                        "planning_ms": round(planning_duration * 1000, 2),
                        "planner_duration": round(planner_duration * 1000, 2),
                        "intent_duration": round(intent_duration * 1000, 2),
                        "dotnet_duration": round(dotnet_duration * 1000, 2),
                        "combiner_duration": round(combiner_duration * 1000, 2),
                        "widget_duration": 0.0,
                        "response_assembly_duration": 0.0,
                        "report_duration": round(report_duration * 1000, 2),
                        "total_duration": round(total_duration * 1000, 2),
                    }

                    response_payload = build_conversation_payload(
                        unrecognised_msg,
                        session_id=session_id,
                        query_type="unclear",
                    )
                    response_payload.setdefault("metadata", {})
                    response_payload["metadata"].setdefault("timings", {})
                    response_payload["metadata"]["timings"].update(
                        {
                            "entityResolutionMs": round(
                                entity_resolution_duration * 1000
                            ),
                            "planningMs": round(planning_duration * 1000),
                            "plannerDurationMs": round(planner_duration * 1000),
                            "intentDurationMs": round(intent_duration * 1000),
                            "dotnetDurationMs": 0,
                            "combineDurationMs": 0,
                            "widgetMs": 0,
                            "responseAssemblyMs": 0,
                            "analysisDurationMs": 0,
                            "predictionDurationMs": 0,
                            "conclusionDurationMs": 0,
                            "totalDurationMs": round(total_duration * 1000),
                            "executionTimeMs": round(total_duration * 1000),
                        }
                    )
                    response_payload["metadata"].setdefault("metrics", {})
                    response_payload["metadata"]["metrics"]["confidence"] = round(
                        float(query_plan.confidence), 2
                    )
                    response_payload["intent"] = {
                        "category": primary_intent.get("category") or "unclear",
                        "confidence": round(float(query_plan.confidence), 2),
                        "operation": primary_intent.get("operation"),
                        "query_type": "unrecognised",
                    }

                    logger.info(
                        json.dumps(
                            {
                                "message": "Admin pipeline complete",
                                "question": user_query,
                                "query_type": "unrecognised",
                                "intent_formed": primary_intent,
                                "trace_id": trace_id,
                            }
                        )
                    )

                    metrics_collector.inc_requests("unrecognised")
                    metrics_collector.record_duration(
                        "planner_duration", durations["planner_duration"]
                    )
                    metrics_collector.record_duration(
                        "intent_duration", durations["intent_duration"]
                    )
                    metrics_collector.record_duration(
                        "dotnet_duration", durations["dotnet_duration"]
                    )
                    metrics_collector.record_duration(
                        "report_duration", durations["report_duration"]
                    )
                    metrics_collector.record_duration(
                        "pipeline_duration", durations["total_duration"]
                    )

                    if total_duration > SLOW_QUERY_THRESHOLD:
                        logger.warning(
                            json.dumps(
                                {
                                    "message": f"Query exceeded {int(SLOW_QUERY_THRESHOLD)} seconds.",
                                    "trace_id": trace_id,
                                    "session_id": session_id,
                                    "query_type": "unrecognised",
                                    "duration_ms": round(total_duration * 1000, 2),
                                }
                            )
                        )

                    write_audit_log(
                        trace_id=trace_id,
                        session_id=session_id,
                        query_type="unrecognised",
                        query_duration=durations["total_duration"],
                        success=True,
                    )

                    combined_message = response_payload.get("message", "")
                    metrics_collector.inc_success("unrecognised")
                    return {
                        "type": "unrecognised",
                        "response_payload": response_payload,
                        "combined_message": combined_message,
                    }

                # Categories like personaldetail (always) and Attendance
                # (except Present) require an agniveerNo before calling .NET.
                _primary_agniveer_no = (
                    primary_intent.get("agniveer_no") or resolved_agniveer_no
                )
                if agniveer_no_required(
                    primary_intent.get("category"), primary_intent.get("operation")
                ) and not _primary_agniveer_no:
                    return _agniveer_no_missing_response(session_id)

                dotnet_payload = format_admin_payload(primary_intent)
                dotnet_payload.update(id_filters)
                for k, v in carry_forward_filters.items():
                    if dotnet_payload.get(k) in (None, ""):
                        dotnet_payload[k] = v
                if full_name:
                    dotnet_payload["fullName"] = full_name
                # Wire in agniveerNo from entity resolution if not already set by intent
                if resolved_agniveer_no and not dotnet_payload.get("agniveerNo"):
                    dotnet_payload["agniveerNo"] = resolved_agniveer_no

                # ── Rule 9: strip empty/zero/null ID fields before sending ──
                dotnet_payload = _strip_empty_id_fields(dotnet_payload)

                # Validate DotNetPayloadModel
                _validate_model_payload(
                    DotNetPayloadModel, dotnet_payload, "single.dotnet_payload"
                )

                if (
                    query_plan.query_type == QueryType.ANALYTICS
                    and query_plan.operations
                ):
                    op = query_plan.operations[0]
                    if getattr(op, "group_by", None):
                        dotnet_payload["groupBy"] = op.group_by
                    if query_plan.analytics_hint:
                        dotnet_payload["analyticsHint"] = query_plan.analytics_hint

                response_dotnet_payload = [dict(dotnet_payload)]

                intent_duration = time.time() - intent_start

                # ── Step 3: Execute .NET API Call ─────────────────────────────
                with span(SPAN_CALL_DOTNET, trace_id=trace_id):
                    dotnet_start = time.time()
                    _notify("dotnet")

                    logger.info(
                        json.dumps(
                            {
                                "message": "Sending simple request to .NET",
                                "trace_id": trace_id,
                                "session_id": session_id,
                                "query_type": qtype_str,
                            }
                        )
                    )

                    with trace_context(request_id, trace_id, session_id):
                        dotnet_data, dotnet_error = _call_dotnet(
                            dotnet_payload,
                            trace_id=trace_id,
                            session_id=session_id,
                            query_type=qtype_str,
                        )
                    if dotnet_error:
                        sanitized_error = _sanitize_error(dotnet_error)
                        logger.warning(
                            json.dumps(
                                {
                                    "message": "Admin .NET call failed",
                                    "trace_id": trace_id,
                                    "session_id": session_id,
                                    "query_type": qtype_str,
                                    "error": sanitized_error,
                                }
                            )
                        )

                        metrics_collector.inc_requests(qtype_str)
                        metrics_collector.inc_errors(qtype_str)
                        total_duration = time.time() - start_time
                        metrics_collector.record_duration(
                            "pipeline_duration", round(total_duration * 1000, 2)
                        )

                        if total_duration > SLOW_QUERY_THRESHOLD:
                            logger.warning(
                                json.dumps(
                                    {
                                        "message": f"Query exceeded {int(SLOW_QUERY_THRESHOLD)} seconds.",
                                        "trace_id": trace_id,
                                        "session_id": session_id,
                                        "query_type": qtype_str,
                                        "duration_ms": round(total_duration * 1000, 2),
                                    }
                                )
                            )

                        write_audit_log(
                            trace_id=trace_id,
                            session_id=session_id,
                            query_type=qtype_str,
                            query_duration=round(total_duration * 1000, 2),
                            success=False,
                            error_type="dotnet_error",
                        )

                        # Every .NET call failure — a transient outage or a
                        # rejected request — is surfaced as one friendly,
                        # conversational message. The raw status/body is only
                        # ever written to the log above, never to the user.
                        unavailable_msg = _BACKEND_UNAVAILABLE_MESSAGE
                        availability_payload = build_conversation_payload(
                            unavailable_msg,
                            session_id=session_id,
                            query_type="service_unavailable",
                        )
                        availability_payload.setdefault("metadata", {})
                        availability_payload["metadata"].setdefault("timings", {})
                        availability_payload["metadata"]["timings"][
                            "dotnetDurationMs"
                        ] = round(dotnet_duration * 1000, 2)
                        availability_payload["metadata"][
                            "executionTimeMs"
                        ] = round(total_duration * 1000)
                        return {
                            "type": "service_unavailable",
                            "response_payload": availability_payload,
                            "combined_message": unavailable_msg,
                            "execution_time_ms": round(total_duration * 1000),
                        }

                    ensure_agniveer_no_in_data(dotnet_data)

                    # Validate DotNetResponseModel
                    if dotnet_data is not None:
                        if isinstance(dotnet_data, dict):
                            _validate_model_payload(
                                DotNetResponseModel,
                                dotnet_data,
                                "single.dotnet_response",
                            )
                        elif isinstance(dotnet_data, list):
                            _validate_model_payload(
                                DotNetResponseModel,
                                {
                                    "success": True,
                                    "commandLabel": primary_intent.get("subcategory")
                                    or primary_intent.get("category")
                                    or "",
                                    "data": dotnet_data,
                                    "message": "",
                                },
                                "single.dotnet_response_list",
                            )

                    raw_results = [dotnet_data]
                    labeled_results = [
                        (primary_intent.get("category", "Result"), dotnet_data)
                    ]
                    dotnet_duration = time.time() - dotnet_start

        # ── Step 4: Result Combiner ───────────────────────────────────────────
        with span(SPAN_COMBINE_RESULTS, trace_id=trace_id):
            combiner_start = time.time()
            _notify("combiner")
            combined_result = combine_results(
                raw_results,
                labeled_results,
                qtype_str,
                primary_intent,
                comparison_context=comparison_context,
            )
            _log_combination_summary(
                question=user_query,
                intent=primary_intent,
                qtype=qtype_str,
                labeled_results=labeled_results,
                raw_results=raw_results,
                combined_result=combined_result,
            )

            # Validate CombinedResponseModel
            if isinstance(combined_result, dict):
                _validate_model_payload(
                    CombinedResponseModel, combined_result, "combined.result"
                )

            if qtype_str == "cross_filter" and bool(failed_filters) and failed_filters:
                combined_result["degraded"] = True
                combined_result["failedFilters"] = failed_filters
            combiner_duration = time.time() - combiner_start
            logger.info(
                {
                    "stage": "combiner_time",
                    "duration_ms": round(combiner_duration * 1000, 2),
                    "trace_id": trace_id,
                    "session_id": session_id,
                    "query_type": qtype_str,
                }
            )

        # ── Step 5: Report Generator ──────────────────────────────────────────
        report: Dict[str, Any] = {
            "message": "I could only prepare a partial summary for this request.",
            "analysis": None,
            "prediction": {
                "trend": "Insufficient Data",
                "projection": "A reliable prediction is not available yet.",
                "heuristicEstimate": "A reliable prediction is not available yet.",
                "shortTerm": "insufficient data",
                "futureTrends": ["A reliable prediction is not available yet."],
            },
            "conclusion": None,
            "durations": {
                "analysisDurationMs": 0.0,
                "predictionDurationMs": 0.0,
                "conclusionDurationMs": 0.0,
            },
        }
        try:
            with span(SPAN_GENERATE_REPORT, trace_id=trace_id):
                report_start = time.time()
                _notify("report")
                report = generate_report(
                    combined_result=combined_result,
                    query_type=qtype_str,
                    intent=primary_intent,
                    user_query=message,
                    trace_id=trace_id,
                )

                # If the report layer intentionally returns blanks because the
                # result set is empty, replace that with a grounded no-data report
                # so the user still gets an explanation instead of a silent payload.
                if (
                    not (report.get("introMessage") or "").strip()
                    and not report.get("analysis")
                    and not report.get("prediction")
                    and not report.get("conclusion")
                ):
                    report = get_fallback_report(
                        combined_result, qtype_str, primary_intent
                    )
                    records = _extract_records(combined_result)
                    pred_cat = str(primary_intent.get("category") or "Agniveer").strip()
                    if pred_cat.lower() in ("unclear", "unknown", "none"):
                        pred_cat = "Agniveer"
                    else:
                        pred_cat = pred_cat[0].upper() + pred_cat[1:]

                    report["prediction"] = {
                        "trend": "Stable" if records else "Insufficient Data",
                        "projection": (
                            f"Future {pred_cat} results should remain broadly stable unless the underlying records change."
                            if records
                            else f"Future projection is unavailable because no {pred_cat} records were returned."
                        ),
                        "heuristicEstimate": (
                            f"Future {pred_cat} results should remain broadly stable unless the underlying records change."
                            if records
                            else f"Future projection is unavailable because no {pred_cat} records were returned."
                        ),
                        "shortTerm": "stable",
                        "futureTrends": [
                            (
                                f"Future {pred_cat} results should remain broadly stable unless the underlying records change."
                                if records
                                else f"No stable trend can be estimated because no {pred_cat} records were returned."
                            )
                        ],
                    }
                report_duration = time.time() - report_start
                logger.info(
                    json.dumps(
                        {
                            "message": "Report generator stage finished",
                            "stage": "report",
                            "duration_ms": round(report_duration * 1000, 2),
                            "trace_id": trace_id,
                            "session_id": session_id,
                            "query_type": qtype_str,
                            "output_shape": {
                                k: type(v).__name__ for k, v in report.items()
                            },
                        }
                    )
                )

                # Validate report models (non-fatal)
                if report.get("analysis"):
                    _validate_model_payload(
                        AnalysisModel, report.get("analysis"), "report.analysis"
                    )
                if report.get("prediction"):
                    _validate_model_payload(
                        PredictionModel, report.get("prediction"), "report.prediction"
                    )
                if report.get("conclusion"):
                    _validate_model_payload(
                        ConclusionModel, report.get("conclusion"), "report.conclusion"
                    )
        except Exception as report_exc:
            import traceback as _tb

            report_duration = time.time() - start_time
            logger.error(
                json.dumps(
                    {
                        "message": "Report generator stage failed",
                        "stage": "report",
                        "duration_ms": round(report_duration * 1000, 2),
                        "trace_id": trace_id,
                        "session_id": session_id,
                        "query_type": qtype_str,
                        "exception": str(report_exc),
                        "traceback": _tb.format_exc(),
                    }
                )
            )
            # report keeps its safe defaults — pipeline continues

        # ── Update session context ────────────────────────────────────────────
        _session_context.update(session_id, message, primary_intent, combined_result)

        # ── Store interaction in context engine (for follow-up resolution) ────
        try:
            _intent_filters = {
                **(primary_intent.get("filters") or {}),
                **_get_id_filters(body),
            }
            _payload_summary = (
                f"{primary_intent.get('operation', 'lookup')} "
                f"{primary_intent.get('category') or primary_intent.get('section') or ''}".strip()
            )
            context_engine.add_interaction(
                session_id,
                user_message=_original_user_query,
                resolved_query=message,
                intent=primary_intent,
                entities=(
                    resolved_entities if isinstance(resolved_entities, dict) else {}
                ),
                filters=_intent_filters,
                category=primary_intent.get("category"),
                section=primary_intent.get("section"),
                operation=primary_intent.get("operation", "lookup"),
                payload_summary=_payload_summary,
            )
        except Exception as _ctx_exc:
            logger.debug("context_engine.add_interaction failed: %s", _ctx_exc)

        total_duration = time.time() - start_time

        # Extract report durations
        report_durations = report.get("durations") or {}
        analysis_ms = report_durations.get("analysisDurationMs", 0.0)
        prediction_ms = report_durations.get("predictionDurationMs", 0.0)
        conclusion_ms = report_durations.get("conclusionDurationMs", 0.0)

        # Durations mapping
        durations_payload = {
            "entityResolutionMs": round(entity_resolution_duration * 1000),
            "planningMs": round(planning_duration * 1000),
            "plannerDurationMs": round(planner_duration * 1000),
            "intentDurationMs": round(intent_duration * 1000),
            "dotnetDurationMs": round(dotnet_duration * 1000),
            "combineDurationMs": round(combiner_duration * 1000),
            "widgetMs": round(widget_duration * 1000),
            "responseAssemblyMs": round(response_assembly_duration * 1000),
            "analysisDurationMs": round(analysis_ms),
            "predictionDurationMs": round(prediction_ms),
            "conclusionDurationMs": round(conclusion_ms),
            "totalDurationMs": round(total_duration * 1000),
            "executionTimeMs": round(total_duration * 1000),
        }

        # For backward compatibility
        durations = {
            "entity_resolution_ms": round(entity_resolution_duration * 1000, 2),
            "planning_ms": round(planning_duration * 1000, 2),
            "planner_duration": round(planner_duration * 1000, 2),
            "intent_duration": round(intent_duration * 1000, 2),
            "dotnet_duration": round(dotnet_duration * 1000, 2),
            "combiner_duration": round(combiner_duration * 1000, 2),
            "widget_duration": round(widget_duration * 1000, 2),
            "response_assembly_duration": round(response_assembly_duration * 1000, 2),
            "report_duration": round(report_duration * 1000, 2),
            "total_duration": round(total_duration * 1000, 2),
        }

        # ── Step 6: Widget Engine (independent) ─────────────────────────────
        formatted_data_payload = None
        try:
            widget_start = time.time()
            from widget_engine import build_widget_list

            visualization_intent = build_visualization_intent(
                message, primary_intent, combined_result, query_type_override=qtype_str
            )
            if frontend_visualization_intent:
                visualization_intent = {
                    **visualization_intent,
                    **frontend_visualization_intent,
                }

            # ── Audit: log visualization intent ─────────────────────────────
            logger.info(
                json.dumps(
                    {
                        "message": "Pipeline stage: visualization_intent",
                        "stage": "visualization_intent",
                        "trace_id": trace_id,
                        "session_id": session_id,
                        "presentation": visualization_intent.get("presentation"),
                        "chart_type": visualization_intent.get("chart_type"),
                        "comparison": visualization_intent.get("comparison"),
                        "trend": visualization_intent.get("trend"),
                        "record_count": visualization_intent.get("record_count"),
                        "frontend_override": visualization_intent.get(
                            "frontend_override"
                        ),
                    }
                )
            )

            formatted_data_payload = build_widget_list(
                combined_result=combined_result,
                query_type=qtype_str,
                intent=primary_intent,
                analysis=report.get("analysis"),
                visualization_intent=visualization_intent,
            )

            # ── Audit: log full widget output ────────────────────────────────
            widget_duration = time.time() - widget_start
            _record_count = (
                len(_extract_records(combined_result)) if combined_result else 0
            )
            _widget_count = (
                len(formatted_data_payload)
                if isinstance(formatted_data_payload, list)
                else 0
            )
            _widget_types = [
                w.get("type")
                for w in (formatted_data_payload or [])
                if isinstance(w, dict)
            ]
            _has_null_widget = any(
                w.get("type") is None or w.get("title") is None or w.get("data") is None
                for w in (formatted_data_payload or [])
                if isinstance(w, dict)
            )
            if _has_null_widget:
                logger.error(
                    json.dumps(
                        {
                            "message": "PIPELINE BUG: null-field widget detected before response_builder",
                            "stage": "widget",
                            "trace_id": trace_id,
                            "session_id": session_id,
                            "widgets": formatted_data_payload,
                        }
                    )
                )
                if isinstance(formatted_data_payload, list):
                    formatted_data_payload = [
                        (
                            w
                            if isinstance(w, dict)
                            and w.get("type")
                            and w.get("title")
                            and w.get("data") is not None
                            else {
                                "type": "null",
                                "title": "No Data Available",
                                "data": [],
                            }
                        )
                        for w in formatted_data_payload
                    ]
            logger.info(
                json.dumps(
                    {
                        "message": "Pipeline stage: widget_engine",
                        "stage": "widget",
                        "duration_ms": round(widget_duration * 1000, 2),
                        "trace_id": trace_id,
                        "session_id": session_id,
                        "query_type": qtype_str,
                        "record_count": _record_count,
                        "widget_count": _widget_count,
                        "widget_types": _widget_types,
                        "formatted_data_count": _widget_count,
                        "has_null_widget": _has_null_widget,
                        "output_is_list": isinstance(formatted_data_payload, list),
                    }
                )
            )
        except Exception as widget_exc:
            import traceback as _tb

            widget_duration = time.time() - widget_start if 'widget_start' in locals() else 0.0
            logger.error(
                json.dumps(
                    {
                        "message": "Widget stage failed",
                        "stage": "widget",
                        "duration_ms": (
                            round(widget_duration * 1000, 2) if widget_duration else 0
                        ),
                        "trace_id": trace_id,
                        "session_id": session_id,
                        "query_type": qtype_str,
                        "exception": str(widget_exc),
                        "traceback": _tb.format_exc(),
                    }
                )
            )

        if not formatted_data_payload:
            formatted_data_payload = [{
                "type": "null",
                "title": "No Data Available",
                "data": [],
            }]

        # ── Step 6b: Build suggested questions (independent) ──────────────
        suggested = []
        try:
            suggested = generate_suggested_questions(
                qtype_str, primary_intent, combined_result
            )
            # Validate SuggestedQuestionModel (non-fatal)
            if suggested:
                for q in suggested:
                    _validate_model_payload(
                        SuggestedQuestionModel, {"question": q}, "suggested_questions"
                    )
        except Exception as sq_exc:
            logger.error(
                json.dumps(
                    {
                        "message": "Suggested questions generation failed",
                        "stage": "suggested_questions",
                        "trace_id": trace_id,
                        "session_id": session_id,
                        "exception": str(sq_exc),
                    }
                )
            )
            suggested = []

        # ── Step 7: Response Builder (independent) ────────────────────────
        try:
            with span(SPAN_BUILD_RESPONSE, trace_id=trace_id):
                response_assembly_start = time.time()
                response_metadata = build_metadata(
                    session_id=session_id,
                    confidence=query_plan.confidence,
                    query_type=qtype_str,
                    operation_count=operation_count,
                    durations=durations_payload,
                )
                if query_plan.query_type in (
                    QueryType.COMPARE,
                    QueryType.COMPARISON,
                ) and bool(comparison_datasets_info):
                    response_metadata["queryType"] = "COMPARISON"
                    response_metadata["comparisonType"] = ""
                    response_metadata["visualization"] = (
                        combined_result.get("visualizationType")
                        if isinstance(combined_result, dict)
                        else "COMPARE_CARD"
                    )
                    response_metadata["datasets"] = [
                        {"id": d["id"], "label": d["label"]}
                        for d in comparison_datasets_info
                    ]
                    if (
                        isinstance(combined_result, dict)
                        and "comparisonMetrics" in combined_result
                    ):
                        response_metadata["comparisonMetrics"] = combined_result[
                            "comparisonMetrics"
                        ]

                    response_dotnet_payload = [
                        d["dotnetPayload"] for d in comparison_datasets_info
                    ]

                response_payload = build_response(
                    message=report.get("message", ""),
                    formatted_data=formatted_data_payload,
                    metadata=response_metadata,
                    session_id=session_id,
                    analysis=report.get("analysis"),
                    prediction=report.get("prediction"),
                    conclusion=report.get("conclusion"),
                    suggested_questions=suggested,
                    dotnet_payload=response_dotnet_payload,
                    overall_confidence=query_plan.confidence,
                    partial_failure=partial_failure,
                    failed_sections=failed_sections,
                )
                response_assembly_duration = time.time() - response_assembly_start
                logger.info(
                    json.dumps(
                        {
                            "message": "Response builder stage finished",
                            "stage": "response_builder",
                            "duration_ms": round(response_assembly_duration * 1000, 2),
                            "trace_id": trace_id,
                            "session_id": session_id,
                            "query_type": qtype_str,
                            "output_shape": (
                                list(response_payload.keys())
                                if isinstance(response_payload, dict)
                                else type(response_payload).__name__
                            ),
                        }
                    )
                )
        except Exception as rb_exc:
            import traceback as _tb

            logger.error(
                json.dumps(
                    {
                        "message": "Response builder stage failed",
                        "stage": "response_builder",
                        "trace_id": trace_id,
                        "session_id": session_id,
                        "query_type": qtype_str,
                        "exception": str(rb_exc),
                        "traceback": _tb.format_exc(),
                    }
                )
            )
            # Build minimal valid response preserving .NET data
            response_payload = build_response(
                message=report.get("message", ""),
                formatted_data=formatted_data_payload,
                metadata=build_metadata(
                    session_id=session_id,
                    confidence=query_plan.confidence,
                    query_type=qtype_str,
                    operation_count=operation_count,
                    durations=durations_payload,
                ),
                session_id=session_id,
                analysis=report.get("analysis"),
                prediction=report.get("prediction"),
                conclusion=report.get("conclusion"),
                suggested_questions=suggested or [],
                dotnet_payload=response_dotnet_payload,
                overall_confidence=query_plan.confidence,
                partial_failure=partial_failure,
                failed_sections=failed_sections,
            )

        execution_time_ms = round(total_duration * 1000)
        response_payload.setdefault("metadata", {})
        response_payload["metadata"]["executionTimeMs"] = execution_time_ms
        response_payload["sessionId"] = session_id
        response_payload["queryType"] = qtype_str
        logger.info(
            {
                "stage": "total_time",
                "duration_ms": round(total_duration * 1000, 2),
                "trace_id": trace_id,
                "session_id": session_id,
                "query_type": qtype_str,
            }
        )

        # Extract combiner strategy details
        if qtype_str in ("multi_independent", "multi_operation"):
            combiner_strategy = "merge"
        elif qtype_str == "comparison":
            combiner_strategy = "comparison"
        elif qtype_str == "cross_filter":
            combiner_strategy = "intersect"
        else:
            combiner_strategy = "passthrough"

        # Get record count and report strategy
        record_count = len(_extract_records(combined_result))
        flags = get_flags()
        if record_count == 0:
            report_strategy = "skip_llm"
        elif flags.ENABLE_REPORTS and flags.ENABLE_OLLAMA:
            report_strategy = "llm"
        else:
            report_strategy = "fallback"

        # Full intent (including the per-operation breakdown) goes to the
        # dedicated audit.log file via set_audit_context; the console line
        # only carries question / queryType / dotnetPayload / responseType /
        # visualizationType / sessionId to keep stdout readable.
        set_audit_context(question=user_query, intent=primary_intent)
        logger.info(
            json.dumps(
                {
                    "message": "Admin query audit",
                    "question": user_query,
                    "queryType": qtype_str,
                    "dotnetPayload": response_dotnet_payload,
                    "responseType": primary_intent.get("responseType") or "",
                    "visualizationType": (
                        _widget_types[0]
                        if "_widget_types" in locals() and _widget_types
                        else ""
                    ),
                    "sessionId": session_id,
                },
                ensure_ascii=False,
                default=str,
            )
        )

        metrics_collector.inc_requests(qtype_str)
        metrics_collector.record_duration(
            "entity_resolution_ms", durations["entity_resolution_ms"]
        )
        metrics_collector.record_duration("planning_ms", durations["planning_ms"])
        metrics_collector.record_duration(
            "planner_duration", durations["planner_duration"]
        )
        metrics_collector.record_duration(
            "intent_duration", durations["intent_duration"]
        )
        metrics_collector.record_duration(
            "dotnet_duration", durations["dotnet_duration"]
        )
        metrics_collector.record_duration(
            "report_duration", durations["report_duration"]
        )
        metrics_collector.record_duration(
            "widget_duration", durations["widget_duration"]
        )
        metrics_collector.record_duration(
            "response_assembly_duration", durations["response_assembly_duration"]
        )
        metrics_collector.record_duration(
            "pipeline_duration", durations["total_duration"]
        )

        if total_duration > SLOW_QUERY_THRESHOLD:
            logger.warning(
                json.dumps(
                    {
                        "message": f"Query exceeded {int(SLOW_QUERY_THRESHOLD)} seconds.",
                        "trace_id": trace_id,
                        "session_id": session_id,
                        "query_type": qtype_str,
                        "duration_ms": round(total_duration * 1000, 2),
                    }
                )
            )

        write_audit_log(question=user_query, intent=primary_intent)

        # Validate FinalResponseModel and MetadataModel
        if "metadata" in response_payload:
            _validate_model_payload(
                MetadataModel, response_payload["metadata"], "final.metadata"
            )
        _validate_model_payload(FinalResponseModel, response_payload, "final.response")

        combined_message = response_payload.get("message", "")
        metrics_collector.inc_success(qtype_str)



        return {
            "type": "query",
            "response_payload": response_payload,
            "combined_message": combined_message,
            "execution_time_ms": execution_time_ms,
        }

    except PayloadValidationError as exc:
        total_duration = time.time() - start_time
        error_intent = {"type": "error"}
        set_audit_context(question=user_query, intent=error_intent)
        logger.warning(
            json.dumps(
                {
                    "message": "Payload validation failed",
                    "question": user_query,
                    "trace_id": trace_id,
                    "session_id": session_id,
                    "errors": exc.errors,
                }
            )
        )
        metrics_collector.inc_requests("error")
        metrics_collector.inc_errors("error")
        metrics_collector.record_duration(
            "pipeline_duration", round(total_duration * 1000, 2)
        )
        write_audit_log(question=user_query, intent=error_intent)
        return {
            "type": "error",
            "error_type": "payload_validation",
            "error_message": _REQUEST_UNPROCESSABLE_MESSAGE,
        }

    except Exception as exc:
        import traceback as _tb

        total_duration = time.time() - start_time
        error_intent = {"type": "error"}
        set_audit_context(question=user_query, intent=error_intent)
        logger.error(
            json.dumps(
                {
                    "message": "Admin pipeline outer catch-all",
                    "question": user_query,
                    "intent": error_intent,
                    "type": "error",
                    "intentFormed": False,
                    "exception": str(exc),
                    "traceback": _tb.format_exc(),
                },
                ensure_ascii=False,
            )
        )

        metrics_collector.inc_requests("error")
        metrics_collector.inc_errors("error")
        metrics_collector.record_duration(
            "pipeline_duration", round(total_duration * 1000, 2)
        )

        if total_duration > SLOW_QUERY_THRESHOLD:
            logger.warning(
                json.dumps(
                    {
                        "message": f"Query exceeded {int(SLOW_QUERY_THRESHOLD)} seconds.",
                        "trace_id": trace_id,
                        "session_id": session_id or "admin-default",
                        "query_type": "error",
                        "duration_ms": round(total_duration * 1000, 2),
                    }
                )
            )

        write_audit_log(question=user_query, intent=error_intent)

        return {
            "type": "error",
            "error_type": "internal_error",
            "error_message": (
                "Something went wrong on my end while processing that. "
                "Please try again in a moment."
            ),
        }
    finally:
        request_id_var.reset(token_req)
        trace_id_var.reset(token_trace)
        session_id_var.reset(token_sess)
