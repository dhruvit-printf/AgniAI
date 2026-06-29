"""
telemetry.py
============
OpenTelemetry distributed tracing for AgniAI.

Instruments the full pipeline with spans:
  HTTP Request → plan_query → classify_admin_intent → _call_dotnet
  → combine_results → generate_report → build_response → Response Send
  (and WebSocket Request)

Controlled by ENABLE_OPENTELEMETRY env flag. When disabled, all functions
are no-ops — zero overhead, zero import errors.

Example trace output (Jaeger/Zipkin compatible):
  TraceID: abc123
  ├─ http_request                    [342ms]
  │  ├─ plan_query                   [12ms]
  │  ├─ classify_admin_intent        [8ms]
  │  ├─ _call_dotnet                 [280ms]
  │  ├─ combine_results              [2ms]
  │  ├─ generate_report              [35ms]
  │  └─ build_response               [1ms]
  └─ response_send                   [2ms]
"""

from __future__ import annotations

import contextlib
import functools
import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, Optional
from contextvars import ContextVar

logger = logging.getLogger(__name__)

# ── ContextVars for Request Tracing ────────────────────────────────────────
request_id_var: ContextVar[str] = ContextVar("request_id", default="N/A")
trace_id_var: ContextVar[str] = ContextVar("trace_id", default="N/A")
session_id_var: ContextVar[str] = ContextVar("session_id", default="N/A")


@contextmanager
def trace_context(
    request_id: str, trace_id: str, session_id: str
) -> Generator[None, None, None]:
    """Context manager to bind request, trace, and session contexts to the current thread."""
    t_req = request_id_var.set(request_id)
    t_trace = trace_id_var.set(trace_id)
    t_sess = session_id_var.set(session_id)
    try:
        yield
    finally:
        request_id_var.reset(t_req)
        trace_id_var.reset(t_trace)
        session_id_var.reset(t_sess)


# ── Feature gate ───────────────────────────────────────────────────────────
_OTEL_ENABLED = os.getenv("ENABLE_OPENTELEMETRY", "false").lower() in (
    "1",
    "true",
    "yes",
)

# ── Lazy OTel imports ──────────────────────────────────────────────────────
_tracer = None


def _get_tracer():
    global _tracer
    if _tracer is not None:
        return _tracer
    if not _OTEL_ENABLED:
        return None
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider()
        exporter = OTLPSpanExporter(
            endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"),
            insecure=True,
        )
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer("agniai", "1.0.0")
        logger.info("OpenTelemetry tracer initialized.")
        return _tracer
    except ImportError:
        logger.warning("opentelemetry packages not installed — tracing disabled.")
        return None
    except Exception as exc:
        logger.warning("OTel tracer init failed: %s", exc)
        return None


# ── Span names (matches super-prompt spec) ─────────────────────────────────
SPAN_HTTP_REQUEST = "http_request"
SPAN_WEBSOCKET_REQUEST = "websocket_request"
SPAN_EXECUTE_ADMIN_QUERY = "admin_pipeline.execute_admin_query"
SPAN_PLAN_QUERY = "plan_query"
SPAN_CLASSIFY_ADMIN_INTENT = "classify_admin_intent"
SPAN_CALL_DOTNET = "_call_dotnet"
SPAN_COMBINE_RESULTS = "combine_results"
SPAN_GENERATE_REPORT = "generate_report"
SPAN_BUILD_RESPONSE = "build_response"
SPAN_RESPONSE_SEND = "response_send"


# ── Context manager ────────────────────────────────────────────────────────


@contextmanager
def span(
    name: str,
    *,
    trace_id: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
) -> Generator[None, None, None]:
    """
    Context manager that creates an OTel span when tracing is enabled,
    and is a no-op when disabled. Never raises.

    Usage:
        with span(SPAN_PLAN_QUERY, trace_id=trace_id):
            result = plan_query(message)
    """
    tracer = _get_tracer()
    if tracer is None:
        yield
        return

    try:
        from opentelemetry import trace as otel_trace

        with tracer.start_as_current_span(name) as s:
            if trace_id:
                s.set_attribute("trace_id", trace_id)
            if attributes:
                for k, v in attributes.items():
                    try:
                        s.set_attribute(k, str(v))
                    except Exception:
                        pass
            try:
                yield
            except Exception as exc:
                s.record_exception(exc)
                s.set_status(otel_trace.StatusCode.ERROR, str(exc))
                raise
    except ImportError:
        yield
    except Exception:
        yield


def instrument(span_name: str):
    """
    Decorator that wraps a function in an OTel span when tracing is enabled.

    Usage:
        @instrument(SPAN_PLAN_QUERY)
        def plan_query(message):
            ...
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with span(span_name):
                return fn(*args, **kwargs)

        return wrapper

    return decorator


# ── Timing helpers ─────────────────────────────────────────────────────────


class SpanTimer:
    """
    Lightweight alternative when OTel is not available.
    Records start/end time and emits a structured log entry.
    """

    def __init__(self, name: str, trace_id: str = "N/A") -> None:
        self.name = name
        self.trace_id = trace_id
        self._start = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.perf_counter() - self._start) * 1000
        status = "error" if exc_type else "ok"
        logger.debug(
            '{"span":"%s","trace_id":"%s","duration_ms":%.2f,"status":"%s"}',
            self.name,
            self.trace_id,
            duration_ms,
            status,
        )


# ── Example trace structure (for documentation) ───────────────────────────
EXAMPLE_TRACE = {
    "traceId": "abc123def456",
    "spans": [
        {"name": "http_request", "duration_ms": 342, "status": "ok"},
        {
            "name": "plan_query",
            "duration_ms": 12,
            "status": "ok",
            "attributes": {"query_type": "simple", "confidence": 0.95},
        },
        {
            "name": "classify_admin_intent",
            "duration_ms": 8,
            "status": "ok",
            "attributes": {"category": "Attendance", "subcategory": "PresentToday"},
        },
        {
            "name": "_call_dotnet",
            "duration_ms": 280,
            "status": "ok",
            "attributes": {"dotnet_url": "https://host/api/AiCommand/execute"},
        },
        {"name": "combine_results", "duration_ms": 2, "status": "ok"},
        {"name": "generate_report", "duration_ms": 35, "status": "ok"},
        {"name": "build_response", "duration_ms": 1, "status": "ok"},
        {"name": "response_send", "duration_ms": 2, "status": "ok"},
    ],
}
