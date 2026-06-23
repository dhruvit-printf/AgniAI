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

import requests as _requests

from admin_context import AdminSessionContext
from admin_entity_resolver import resolve_entities_from_query
from admin_intent import (
    admin_normalize_query,
    classify_admin_intent,
    clean_query,
    format_admin_payload,
)
from audit_logger import write_audit_log
from audit_logger import reset_audit_context, set_audit_context
from conversation_detector import build_conversational_response as build_conversation_payload
from conversation_detector import is_conversational_query
from config import GREETING_PHRASES, _is_greeting, _is_patriotic, _is_small_talk
from dotnet_executor import _call_dotnet
from feature_flags import get_flags
from query_planner import QueryType, plan_query
from query_understanding_engine import understand_query
from report_generator import generate_report, get_fallback_report
from response_builder import build_response, build_answer
from normalized_models import extract_records as _extract_records
from result_combiner import combine_results
from suggested_question_engine import generate_suggested_questions
from visualization_intent import build_visualization_intent
from telemetry import (
    SPAN_BUILD_RESPONSE,
    SPAN_CALL_DOTNET,
    SPAN_CLASSIFY_ADMIN_INTENT,
    SPAN_COMBINE_RESULTS,
    SPAN_GENERATE_REPORT,
    SPAN_PLAN_QUERY,
    span,
    request_id_var,
    trace_id_var,
    session_id_var,
    trace_context,
)
from schemas import (
    IntentModel,
    DotNetPayloadModel,
    DotNetResponseModel,
    CombinedResponseModel,
    AnalysisModel,
    PredictionModel,
    ConclusionModel,
    SuggestedQuestionModel,
    MetadataModel,
    FinalResponseModel,
)

logger = logging.getLogger(__name__)


SLOW_QUERY_THRESHOLD = float(os.getenv("SLOW_QUERY_THRESHOLD_SEC", "5.0"))
from metrics import metrics_collector

_session_context = AdminSessionContext()


# =============================================================================
# HELPERS
# =============================================================================


def ensure_agniveer_no_in_data(data: Any) -> None:
    """Recursively ensure that all records inside data contain agniveerNo if agniveerId is present."""
    if isinstance(data, list):
        for item in data:
            ensure_agniveer_no_in_data(item)
    elif isinstance(data, dict):
        id_val = None
        for key in ("agniveerId", "AgniveerId", "AgniVeerId", "id", "Id"):
            if key in data and data[key] is not None:
                id_val = data[key]
                break
        if "agniveerNo" not in data and id_val is not None:
            data["agniveerNo"] = str(id_val)

        for v in data.values():
            if isinstance(v, (dict, list)):
                ensure_agniveer_no_in_data(v)


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


def _get_session_id(data: Dict) -> str:
    """Extract session ID from request body or headers."""
    session_id = (data.get("session_id") or data.get("sessionId") or "").strip()
    return session_id or "admin-default"


def _get_id_filters(data: Dict) -> Dict[str, int]:
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
    for field in ("batch_id", "platoon_id", "company_id", "agniveer_no", "section", "sub_section", "class", "sport"):
        value = intent.get(field)
        if value not in (None, "", [], {}):
            filters[field] = value
    intent["filters"] = filters
    return intent


def _merge_intents(*sources: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key, value in source.items():
            if value in (None, "", [], {}):
                continue
            if key == "filters" and isinstance(value, dict):
                existing = merged.get("filters")
                if not isinstance(existing, dict):
                    existing = {}
                for fk, fv in value.items():
                    if fv not in (None, "", [], {}):
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
    if qtype not in ("cross_filter", "comparison", "compare", "multi_independent", "multi_operation"):
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
                "intent": intent,
                "type": intent.get("type") or qtype,
                "input": input_counts,
                "outputCount": output_count,
                "rawCount": len(raw_results),
            },
            ensure_ascii=False,
        )
    )


def _validate_model_payload(model_cls, payload: Any, context: str) -> None:
    """Validate payload against schema.  Log drift but NEVER raise — schema
    mismatch must not crash the pipeline."""
    try:
        model_cls.model_validate(payload)
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


def _log_stage_duration(stage: str, duration_ms: float, **extra: Any) -> None:
    event = {"stage": stage, "duration_ms": round(duration_ms, 2)}
    event.update({k: v for k, v in extra.items() if v is not None})
    logger.info(event)


# =============================================================================
# CONVERSATIONAL DETECTION
# =============================================================================


def _is_admin_conversational(message: str) -> bool:
    """Check if the message is a greeting or small talk (not a data query)."""
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
        if not any(t in _ADMIN_SIGNAL_WORDS for t in tokens) and not any(
            phrase in cleaned for phrase in _ADMIN_SIGNAL_PHRASES
        ):
            return True

    return False


# =============================================================================
# GREETING / CONVERSATIONAL BUILDERS
# =============================================================================


def _build_greeting_response(body: Dict, session_id: str) -> Tuple[Dict, str]:
    """Build a time-aware greeting response."""
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


def _build_conversational_response(
    message: str, body: Dict, session_id: str, trace_id: Optional[str] = None
) -> Tuple[Dict, str]:
    """Build a conversational response using LLM or fallback."""
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

    try:
        import requests as _req

        from config import DEFAULT_MODEL, OLLAMA_URL

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
        resp = _req.post(OLLAMA_URL, json=payload, timeout=(8, 30))
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

    request_id = uuid.uuid4().hex
    if not trace_id:
        trace_id = uuid.uuid4().hex
    if session_id is None:
        session_id = _get_session_id(body)

    token_req = request_id_var.set(request_id)
    token_trace = trace_id_var.set(trace_id)
    token_sess = session_id_var.set(session_id)
    set_audit_context(question=user_query or "", intent={"type": "Unknown"})

    # ── Check Cache ──
    from cache_manager import cache_manager
    query_hash = cache_manager.get_query_hash(user_query, scope=_get_cache_scope(body))
    import sys
    in_testing = "pytest" in sys.modules or "unittest" in sys.modules or os.getenv("ENV") == "testing"
    bypass_cache = (
        body.get("bypass_cache")
        or body.get("bypassCache")
        or os.getenv("BYPASS_CACHE", "false").lower() in ("true", "1", "yes")
        or in_testing
    )

    if not bypass_cache:
        cached_val = cache_manager.get(query_hash)
        if cached_val:
            metrics_collector.inc_cache_hits()
            request_id_var.reset(token_req)
            trace_id_var.reset(token_trace)
            session_id_var.reset(token_sess)
            if isinstance(cached_val, dict) and "response_payload" in cached_val:
                payload = dict(cached_val["response_payload"])
                if "metadata" in payload:
                    payload["metadata"] = dict(payload["metadata"])
                    payload["metadata"]["requestId"] = request_id
                    payload["metadata"]["traceId"] = trace_id
                    payload["metadata"]["sessionId"] = session_id
                if "sessionId" in payload:
                    payload["sessionId"] = session_id
                cached_val = dict(cached_val)
                cached_val["response_payload"] = payload
            cached_intent = {}
            if isinstance(cached_val, dict):
                cached_intent = (
                    cached_val.get("response_payload", {}).get("intent") or {}
                )
            write_audit_log(question=user_query, intent=cached_intent)
            return cached_val
        else:
            metrics_collector.inc_cache_misses()
    else:
        metrics_collector.inc_cache_misses()

    try:
        message = clean_query(user_query or "").strip()
        frontend_intent = _extract_frontend_intent(body)
        if session_id is None:
            session_id = _get_session_id(body)
        id_filters = _get_id_filters(body)
        full_name = _get_full_name(body)
        semantic_understanding = understand_query(message)

        # ── Greeting / conversational short-circuit ──────────────────────────
        if is_conversational_query(message) or semantic_understanding.get("conversational"):
            intent_start = time.time()
            if _is_greeting(message):
                _, reply_text = _build_greeting_response(body, session_id)
                qtype = "greeting"
            else:
                reply_text = "I can help with administrative data, reports, and analysis."
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
            response_payload["metadata"]["timings"]["intentDurationMs"] = round(intent_duration * 1000, 2)
            response_payload["metadata"]["executionTimeMs"] = round(total_duration * 1000)
            set_audit_context(
                question=user_query or message,
                intent=response_payload.get("intent") or {"category": qtype},
            )
            metrics_collector.inc_requests(qtype)
            metrics_collector.inc_success(qtype)
            write_audit_log(question=user_query or message, intent=response_payload.get("intent") or {})
            return {
                "type": qtype,
                "response_payload": response_payload,
                "combined_message": response_payload.get("message", ""),
                "execution_time_ms": round(total_duration * 1000),
            }

        if not message:
            return {
                "type": "error",
                "error_message": "Failed to process request.",
            }

        # ── Step 1: Resolve Named Entities ───────────────────────────────────
        resolved_agniveer_no = None

        with span(SPAN_PLAN_QUERY, trace_id=trace_id):
            planner_start = time.time()
            _notify("planner")
            entity_resolution_start = time.time()
            resolved_entities = resolve_entities_from_query(
                message,
                existing_company_id=id_filters.get("companyId"),
                existing_platoon_id=id_filters.get("platoonId"),
                existing_batch_id=id_filters.get("batchId"),
                trace_id=trace_id,
                session_id=session_id,
            )
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

            resolved_agniveer_no = resolved_entities.get("agniveerNo")

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
                    QueryType.COMPARISON,
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

                    def run_op(idx, op):
                        payload = dict(op.dotnet_payload)
                        payload.update(id_filters)
                        if resolved_agniveer_no and not payload.get("agniveerNo"):
                            payload["agniveerNo"] = resolved_agniveer_no
                        if full_name:
                            payload["fullName"] = full_name

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

                        try:
                            with trace_context(request_id, trace_id, session_id):
                                data, err = _call_dotnet(payload, trace_id=trace_id)
                            if not err and data is not None:
                                # Validate DotNetResponseModel
                                if isinstance(data, dict):
                                    _validate_model_payload(
                                        DotNetResponseModel, data, "multi.dotnet_response"
                                    )
                                elif isinstance(data, list):
                                    _validate_model_payload(
                                        DotNetResponseModel,
                                        {
                                            "success": True,
                                            "commandLabel": op.intent_result.get("subcategory")
                                            or op.intent_result.get("category")
                                            or "",
                                            "data": data,
                                            "message": "",
                                        },
                                        "multi.dotnet_response_list",
                                    )
                            return idx, op, data, err
                        except Exception as exc:
                            return idx, op, None, str(exc)

                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = [
                            executor.submit(run_op, i, op)
                            for i, op in enumerate(query_plan.operations)
                        ]
                        results = [f.result() for f in futures]

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
                        for idx, op, _, err in results:
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
                            "error_message": "Failed to process request.",
                        }

                    # CROSS_FILTER primary failure check
                    if query_plan.query_type == QueryType.CROSS_FILTER:
                        primary_idx, primary_op, primary_data, primary_error = results[
                            0
                        ]
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
                                "error_message": "Failed to process request.",
                            }

                    # Build raw_results and labeled_results preserving the order
                    failed_filters = []
                    for idx, op, dotnet_data, dotnet_error in results:
                        label = (
                            op.intent_result.get("category")
                            or op.intent_result.get("section")
                            or op.intent_result.get("sport")
                            or op.intent_result.get("class")
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
                            label = (
                                op.intent_result.get("section")
                                or op.intent_result.get("sport")
                                or op.intent_result.get("class")
                                or op.raw_fragment.upper()
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

                    primary_intent = _merge_intents(
                        frontend_intent,
                        query_plan.operations[0].intent_result,
                    )
                    dotnet_duration = time.time() - dotnet_start

            else:
                # FILTER_QUERY / ANALYTICS: single .NET call
                qtype_str = map_query_type(query_plan.query_type)
                operation_count = 1

                classified_intent = (
                    query_plan.operations[0].intent_result
                    if (
                        query_plan.query_type == QueryType.ANALYTICS
                        and query_plan.operations
                        and query_plan.operations[0].intent_result.get("category")
                    )
                    else classify_admin_intent(message)
                )
                primary_intent = _merge_intents(
                    frontend_intent,
                    classified_intent,
                )
                primary_intent["filters"] = _merge_intents(
                    frontend_intent.get("filters", {}),
                    classified_intent.get("filters", {}),
                )

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
                    semantic_understanding.get("confidence") or 0.0
                ) < float(os.getenv("INTENT_CONFIDENCE_THRESHOLD", "0.35")):
                    intent_duration = time.time() - intent_start
                    unrecognised_msg = (
                        "I couldn't understand the query clearly. "
                        "Could you please rephrase it?"
                    )

                    total_duration = time.time() - start_time
                    durations = {
                        "entity_resolution_ms": round(entity_resolution_duration * 1000, 2),
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
                            "entityResolutionMs": round(entity_resolution_duration * 1000),
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
                        float(semantic_understanding.get("confidence") or 0.0), 2
                    )
                    response_payload["intent"] = {
                        "category": "unclear",
                        "confidence": round(
                            float(semantic_understanding.get("confidence") or 0.0), 2
                        ),
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

                dotnet_payload = format_admin_payload(primary_intent)
                dotnet_payload.update(id_filters)
                if full_name:
                    dotnet_payload["fullName"] = full_name
                # Wire in agniveerNo from entity resolution if not already set by intent
                if resolved_agniveer_no and not dotnet_payload.get("agniveerNo"):
                    dotnet_payload["agniveerNo"] = resolved_agniveer_no

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
                            dotnet_payload, trace_id=trace_id
                        )
                    if dotnet_error:
                        logger.warning(
                            json.dumps(
                                {
                                    "message": "Admin .NET call failed",
                                    "trace_id": trace_id,
                                    "session_id": session_id,
                                    "query_type": qtype_str,
                                    "error": _sanitize_error(dotnet_error),
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

                        return {
                            "type": "error",
                            "error_message": "Failed to process request.",
                        }

                    ensure_agniveer_no_in_data(dotnet_data)

                    # Validate DotNetResponseModel
                    if dotnet_data is not None:
                        if isinstance(dotnet_data, dict):
                            _validate_model_payload(
                                DotNetResponseModel, dotnet_data, "single.dotnet_response"
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
                raw_results, labeled_results, qtype_str, primary_intent
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
            if (
                qtype_str == "cross_filter"
                and isinstance(combined_result, dict)
                and combined_result.get("status") is False
            ):
                # Return empty/no matches status false response
                total_duration = time.time() - start_time
                response_payload = {
                    "status": False,
                    "queryType": qtype_str,
                    "message": combined_result.get("message", "No matching records found")
                }
                return {
                    "type": qtype_str,
                    "response_payload": response_payload,
                    "combined_message": combined_result.get("message", "No matching records found"),
                    "execution_time_ms": round(total_duration * 1000)
                }

            if (
                qtype_str == "cross_filter"
                and "failed_filters" in locals()
                and failed_filters
            ):
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
        report = {
            "introMessage": "Report generated with partial metrics.",
            "analysis": None,
            "prediction": None,
            "conclusion": None,
            "durations": {"analysisDurationMs": 0.0, "predictionDurationMs": 0.0, "conclusionDurationMs": 0.0},
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
                    report = get_fallback_report(combined_result, qtype_str, primary_intent)
                    records = _extract_records(combined_result)
                    report["prediction"] = {
                        "trend": "Stable" if records else "Insufficient Data",
                        "projection": (
                            f"Future {str(primary_intent.get('category') or 'agniveer').lower()} results should remain broadly stable unless the underlying records change."
                            if records
                            else f"Future projection is unavailable because no {str(primary_intent.get('category') or 'agniveer').lower()} records were returned."
                        ),
                        "heuristicEstimate": (
                            f"Future {str(primary_intent.get('category') or 'agniveer').lower()} results should remain broadly stable unless the underlying records change."
                            if records
                            else f"Future projection is unavailable because no {str(primary_intent.get('category') or 'agniveer').lower()} records were returned."
                        ),
                        "shortTerm": "stable",
                        "futureTrends": [
                            (
                                f"Future {str(primary_intent.get('category') or 'agniveer').lower()} results should remain broadly stable unless the underlying records change."
                                if records
                                else f"No stable trend can be estimated because no {str(primary_intent.get('category') or 'agniveer').lower()} records were returned."
                            )
                        ],
                    }
                report_duration = time.time() - report_start
                logger.info(
                    json.dumps({
                        "stage": "report",
                        "duration_ms": round(report_duration * 1000, 2),
                        "trace_id": trace_id,
                        "session_id": session_id,
                        "query_type": qtype_str,
                        "output_shape": {k: type(v).__name__ for k, v in report.items()},
                    })
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
                json.dumps({
                    "stage": "report",
                    "duration_ms": round(report_duration * 1000, 2),
                    "trace_id": trace_id,
                    "session_id": session_id,
                    "query_type": qtype_str,
                    "exception": str(report_exc),
                    "traceback": _tb.format_exc(),
                })
            )
            # report keeps its safe defaults — pipeline continues

        # ── Update session context ────────────────────────────────────────────
        _session_context.update(session_id, message, primary_intent, combined_result)

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
            # Snake case for backward compatibility
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
            from widget_engine import build_formatted_data
            visualization_intent = build_visualization_intent(
                message, primary_intent, combined_result
            )
            formatted_data_payload = build_formatted_data(
                combined_result=combined_result,
                query_type=qtype_str,
                intent=primary_intent,
                analysis=report.get("analysis"),
                prediction=report.get("prediction"),
                conclusion=report.get("conclusion"),
                visualization_intent=visualization_intent,
            )
            widget_duration = time.time() - widget_start
            logger.info(
                json.dumps({
                    "stage": "widget",
                    "duration_ms": round(widget_duration * 1000, 2),
                    "trace_id": trace_id,
                    "session_id": session_id,
                    "query_type": qtype_str,
                    "output_shape": type(formatted_data_payload).__name__,
                })
            )
        except Exception as widget_exc:
            import traceback as _tb
            widget_duration = time.time() - widget_start if 'widget_start' in dir() else 0.0
            logger.error(
                json.dumps({
                    "stage": "widget",
                    "duration_ms": round(widget_duration * 1000, 2) if widget_duration else 0,
                    "trace_id": trace_id,
                    "session_id": session_id,
                    "query_type": qtype_str,
                    "exception": str(widget_exc),
                    "traceback": _tb.format_exc(),
                })
            )

        # ── Step 6b: Build answer & suggested questions (independent) ─────
        answer = build_answer(qtype_str, combined_result, primary_intent)

        suggested = []
        try:
            suggested = generate_suggested_questions(qtype_str, primary_intent, answer)
            # Validate SuggestedQuestionModel (non-fatal)
            if suggested:
                for q in suggested:
                    _validate_model_payload(
                        SuggestedQuestionModel, {"question": q}, "suggested_questions"
                    )
        except Exception as sq_exc:
            logger.error(
                json.dumps({
                    "stage": "suggested_questions",
                    "trace_id": trace_id,
                    "session_id": session_id,
                    "exception": str(sq_exc),
                })
            )
            suggested = []

        # ── Step 7: Response Builder (independent) ────────────────────────
        try:
            with span(SPAN_BUILD_RESPONSE, trace_id=trace_id):
                response_assembly_start = time.time()
                response_payload = build_response(
                    query_type=qtype_str,
                    intro_message=report.get("introMessage", ""),
                    combined_result=combined_result,
                    analysis=report.get("analysis"),
                    conclusion=report.get("conclusion"),
                    intent=primary_intent,
                    raw_results=raw_results,
                    confidence=query_plan.confidence,
                    operation_count=operation_count,
                    formatted_data=formatted_data_payload,
                    session_id=session_id,
                    durations=durations_payload,
                    widgets=None,
                    suggested_questions=suggested,
                    prediction=report.get("prediction"),
                    partial_failure=partial_failure,
                    failed_sections=failed_sections,
                    answer_dict=answer,
                )
                response_assembly_duration = time.time() - response_assembly_start
                logger.info(
                    json.dumps({
                        "stage": "response_builder",
                        "duration_ms": round(response_assembly_duration * 1000, 2),
                        "trace_id": trace_id,
                        "session_id": session_id,
                        "query_type": qtype_str,
                        "output_shape": list(response_payload.keys()) if isinstance(response_payload, dict) else type(response_payload).__name__,
                    })
                )
        except Exception as rb_exc:
            import traceback as _tb
            logger.error(
                json.dumps({
                    "stage": "response_builder",
                    "trace_id": trace_id,
                    "session_id": session_id,
                    "query_type": qtype_str,
                    "exception": str(rb_exc),
                    "traceback": _tb.format_exc(),
                })
            )
            # Build minimal valid response preserving .NET data
            from normalized_models import build_answer as _ba
            response_payload = {
                "status": True,
                "sessionId": session_id,
                "message": report.get("introMessage", ""),
                "queryType": qtype_str,
                "answer": answer,
                "result": {"processedData": combined_result},
                "widgets": [],
                "widget": "table",
                "records": _extract_records(combined_result),
                "analysis": None,
                "prediction": None,
                "conclusion": None,
                "intent": primary_intent,
                "formattedData": formatted_data_payload,
                "suggestedQuestions": [],
                "metadata": {"timings": durations_payload, "metrics": {"confidence": query_plan.confidence, "queryType": qtype_str}},
                "overallConfidence": round(float(query_plan.confidence), 2),
                "partialFailure": True,
                "failedSections": ["response_builder"],
            }

        execution_time_ms = round(total_duration * 1000)
        response_payload.setdefault("metadata", {})
        response_payload["metadata"]["executionTimeMs"] = execution_time_ms
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

        # Log all 10 required audit metrics
        set_audit_context(question=user_query, intent=primary_intent)
        logger.info(
            json.dumps(
                {
                    "message": "Admin query audit",
                    "question": user_query,
                    "intent": primary_intent,
                    "type": primary_intent.get("type") or "",
                    "intentFormed": bool(primary_intent.get("category")),
                },
                ensure_ascii=False,
            )
        )

        metrics_collector.inc_requests(qtype_str)
        metrics_collector.record_duration(
            "entity_resolution_ms", durations["entity_resolution_ms"]
        )
        metrics_collector.record_duration(
            "planning_ms", durations["planning_ms"]
        )
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

        # Cache successful query response
        is_cacheable = cache_manager.is_cacheable_category(primary_intent.get("category"))
        if is_cacheable and not bypass_cache and response_payload.get("status"):
            result_to_cache = {
                "type": "query",
                "response_payload": response_payload,
                "combined_message": combined_message,
                "execution_time_ms": execution_time_ms,
            }
            cache_manager.set(query_hash, result_to_cache, category=primary_intent.get("category"))

        return {
            "type": "query",
            "response_payload": response_payload,
            "combined_message": combined_message,
            "execution_time_ms": execution_time_ms,
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
            "error_message": "Failed to process request.",
        }
    finally:
        request_id_var.reset(token_req)
        trace_id_var.reset(token_trace)
        session_id_var.reset(token_sess)
