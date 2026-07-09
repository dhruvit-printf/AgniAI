"""
tests/test_telemetry.py
Unit tests for telemetry.py — OpenTelemetry span instrumentation.
"""

import os
import time
from unittest.mock import MagicMock, patch

import pytest


class TestSpanContextManager:
    def test_span_is_noop_when_otel_disabled(self):
        """span() must be a no-op when ENABLE_OPENTELEMETRY is false."""
        import telemetry

        original = telemetry._OTEL_ENABLED
        try:
            telemetry._OTEL_ENABLED = False
            telemetry._tracer = None
            result = []
            with telemetry.span("test_span"):
                result.append("ran")
            assert result == ["ran"]
        finally:
            telemetry._OTEL_ENABLED = original

    def test_span_does_not_raise_on_exception_inside(self):
        """span() must not suppress exceptions from wrapped code."""
        import telemetry

        telemetry._OTEL_ENABLED = False
        telemetry._tracer = None
        with pytest.raises(ValueError, match="expected"):
            with telemetry.span("test_span"):
                raise ValueError("expected")

    def test_span_accepts_trace_id_kwarg(self):
        """trace_id kwarg must be accepted without error."""
        import telemetry

        telemetry._OTEL_ENABLED = False
        telemetry._tracer = None
        with telemetry.span("test", trace_id="abc123"):
            pass  # no exception

    def test_span_accepts_attributes_kwarg(self):
        """attributes dict must be accepted without error."""
        import telemetry

        telemetry._OTEL_ENABLED = False
        telemetry._tracer = None
        with telemetry.span("test", attributes={"key": "value"}):
            pass  # no exception

    def test_span_noop_with_none_trace_id(self):
        import telemetry

        telemetry._OTEL_ENABLED = False
        telemetry._tracer = None
        with telemetry.span("test", trace_id=None):
            pass


class TestInstrumentDecorator:
    def test_instrument_wraps_function(self):
        """instrument() must call the wrapped function."""
        import telemetry

        telemetry._OTEL_ENABLED = False
        telemetry._tracer = None

        @telemetry.instrument("test_span")
        def my_func(x, y):
            return x + y

        result = my_func(2, 3)
        assert result == 5

    def test_instrument_preserves_function_name(self):
        import telemetry

        telemetry._OTEL_ENABLED = False

        @telemetry.instrument("test_span")
        def my_named_function():
            pass

        assert my_named_function.__name__ == "my_named_function"

    def test_instrument_propagates_exceptions(self):
        import telemetry

        telemetry._OTEL_ENABLED = False

        @telemetry.instrument("test_span")
        def failing_func():
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            failing_func()

    def test_instrument_works_with_kwargs(self):
        import telemetry

        telemetry._OTEL_ENABLED = False

        @telemetry.instrument("test_span")
        def func_with_kwargs(a, b=10):
            return a + b

        assert func_with_kwargs(1, b=20) == 21


class TestSpanTimer:
    def test_span_timer_records_duration(self):
        import telemetry

        with telemetry.SpanTimer("test_op", trace_id="tid"):
            time.sleep(0.01)
        # No assertion on duration — just verify no exception

    def test_span_timer_context_manager_protocol(self):
        import telemetry

        timer = telemetry.SpanTimer("test")
        result = timer.__enter__()
        assert result is timer
        timer.__exit__(None, None, None)

    def test_span_timer_handles_exception(self):
        import telemetry

        try:
            with telemetry.SpanTimer("test"):
                raise ValueError("test error")
        except ValueError:
            pass  # exception propagated — timer handled gracefully


class TestSpanConstants:
    def test_all_span_constants_are_strings(self):
        import telemetry

        constants = [
            telemetry.SPAN_HTTP_REQUEST,
            telemetry.SPAN_EXECUTE_ADMIN_QUERY,
            telemetry.SPAN_PLAN_QUERY,
            telemetry.SPAN_CLASSIFY_ADMIN_INTENT,
            telemetry.SPAN_CALL_DOTNET,
            telemetry.SPAN_COMBINE_RESULTS,
            telemetry.SPAN_GENERATE_REPORT,
            telemetry.SPAN_BUILD_RESPONSE,
            telemetry.SPAN_RESPONSE_SEND,
        ]
        for c in constants:
            assert isinstance(c, str) and len(c) > 0

    def test_example_trace_structure(self):
        import telemetry

        assert "traceId" in telemetry.EXAMPLE_TRACE
        assert "spans" in telemetry.EXAMPLE_TRACE
        assert len(telemetry.EXAMPLE_TRACE["spans"]) > 0
