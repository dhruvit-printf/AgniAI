"""
tests/test_websocket_routes.py
Tests for websocket_routes.py transport layer.
"""

from unittest.mock import MagicMock, call, patch

import pytest


class TestRunPipelineSuccessPath:
    def _make_pipeline_result(self, qtype="query"):
        return {
            "type": qtype,
            "response_payload": {
                "intro": {"title": "Test", "description": "Test intro"},
                "introMessage": {"title": "Test", "description": "Test intro"},
                "formattedData": {"sections": []},
                "widgets": [{"section": "Result", "type": "TABLE", "widgetType": "TABLE"}],
                "analysis": {"summary": "Summary", "observations": [], "insights": []},
                "prediction": {"trend": "Stable", "projection": "Stable"},
                "conclusion": {"summary": "Done"},
                "suggestedQuestions": ["Q1"],
            },
        }

    def test_success_path_emits_all_events_in_order(self):
        sent = []
        with patch(
            "websocket_routes.ws_manager.send_json",
            side_effect=lambda sid, p: sent.append(p["type"]),
        ), patch(
            "websocket_routes.execute_admin_query",
            return_value=self._make_pipeline_result(),
        ):
            import websocket_routes

            websocket_routes._run_pipeline("sid1", "show attendance", {}, "trace-001")
        assert sent == [
            "intro",
            "formattedData",
            "widgets",
            "analysis",
            "prediction",
            "conclusion",
            "suggestedQuestions",
            "done",
        ]

    def test_done_is_last_event_on_success(self):
        sent = []
        with patch(
            "websocket_routes.ws_manager.send_json",
            side_effect=lambda sid, p: sent.append(p["type"]),
        ), patch(
            "websocket_routes.execute_admin_query",
            return_value=self._make_pipeline_result(),
        ):
            import websocket_routes

            websocket_routes._run_pipeline("sid1", "q", {}, "t")
        assert sent[-1] == "done"

    def test_greeting_type_still_emits_done(self):
        sent = []
        with patch(
            "websocket_routes.ws_manager.send_json",
            side_effect=lambda sid, p: sent.append(p["type"]),
        ), patch(
            "websocket_routes.execute_admin_query",
            return_value=self._make_pipeline_result("greeting"),
        ):
            import websocket_routes

            websocket_routes._run_pipeline("sid1", "hello", {}, "t")
        assert "done" in sent
        assert sent[-1] == "done"


class TestRunPipelineErrorPath:
    def test_error_path_emits_error_then_done(self):
        sent = []
        with patch(
            "websocket_routes.ws_manager.send_json",
            side_effect=lambda sid, p: sent.append(p["type"]),
        ), patch(
            "websocket_routes.execute_admin_query",
            return_value={"type": "error", "error_message": "Failed"},
        ):
            import websocket_routes

            websocket_routes._run_pipeline("sid1", "bad query", {}, "t")
        assert "error" in sent
        assert "done" in sent
        error_idx = sent.index("error")
        done_idx = sent.index("done")
        assert done_idx > error_idx, "done must come after error"
        assert sent[-1] == "done"

    def test_exception_in_pipeline_emits_error_then_done(self):
        sent = []
        with patch(
            "websocket_routes.ws_manager.send_json",
            side_effect=lambda sid, p: sent.append(p["type"]),
        ), patch(
            "websocket_routes.execute_admin_query",
            side_effect=RuntimeError("unexpected crash"),
        ):
            import websocket_routes

            websocket_routes._run_pipeline("sid1", "q", {}, "t")
        assert "error" in sent
        assert "done" in sent
        assert sent[-1] == "done"

    def test_error_message_never_exposes_internal_detail(self):
        payloads = []
        with patch(
            "websocket_routes.ws_manager.send_json",
            side_effect=lambda sid, p: payloads.append(p),
        ), patch(
            "websocket_routes.execute_admin_query",
            return_value={"type": "error", "error_message": "Failed"},
        ):
            import websocket_routes

            websocket_routes._run_pipeline("sid1", "q", {}, "t")
        error_events = [p for p in payloads if p.get("type") == "error"]
        assert len(error_events) == 1
        assert error_events[0]["message"] == "Failed to process request."


class TestProgressCallback:
    def test_progress_stages_are_emitted(self):
        progress_stages = []

        def capture_send(sid, payload):
            if payload.get("type") == "progress":
                progress_stages.append(payload.get("stage"))

        mock_result = {
            "type": "query",
            "response_payload": {
                "intro": {"title": "", "description": ""},
                "introMessage": {"title": "", "description": ""},
                "formattedData": {},
                "widgets": [],
                "analysis": None,
                "prediction": None,
                "conclusion": None,
                "suggestedQuestions": [],
            },
        }

        with patch(
            "websocket_routes.ws_manager.send_json", side_effect=capture_send
        ), patch(
            "websocket_routes.execute_admin_query", return_value=mock_result
        ) as mock_exec:

            def fake_pipeline(user_query, body, progress_callback, trace_id):
                for stage in ["planner", "intent", "dotnet", "combiner", "report"]:
                    progress_callback(stage)
                return mock_result

            mock_exec.side_effect = fake_pipeline

            import websocket_routes

            websocket_routes._run_pipeline("sid1", "q", {}, "t")

        for stage in ["planner", "intent", "dotnet", "combiner", "report"]:
            assert stage in progress_stages
