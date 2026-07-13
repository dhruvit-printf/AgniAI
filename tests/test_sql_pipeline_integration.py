"""
tests/test_sql_pipeline_integration.py
========================================
Integration tests for the hybrid SQL-fallback wiring in admin_pipeline.py
(Task 5) and the no-raw-leak invariant on the SQL path (Task 6).

Uses controlled mocks for planning/classification so the low-confidence /
unclassifiable branch is reached deterministically, independent of NLP
heuristic drift.
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

from admin_pipeline import execute_admin_query
from intent_engine.query_planner import QueryPlan, QueryType, SubOperation


def _unclassifiable_plan(message: str) -> QueryPlan:
    op = SubOperation(raw_fragment=message, intent_result={}, dotnet_payload={})
    return QueryPlan(
        query_type=QueryType.SIMPLE,
        operations=[op],
        confidence=0.1,
        raw_query=message,
        reasoning="test: forced unclassifiable",
    )


def _patched_flags(enable_sql: bool) -> MagicMock:
    flags = MagicMock()
    flags.ENABLE_SQL_EXECUTOR = enable_sql
    flags.ENABLE_REPORTS = True
    flags.ENABLE_OLLAMA = True
    return flags


def _base_patches(message: str, enable_sql: bool):
    """Common patch set that deterministically routes `message` into the
    low-confidence / unclassifiable branch of admin_pipeline's single-op
    path, regardless of real NLP classification behavior."""
    return [
        patch(
            "admin_pipeline.understand_query", return_value={"conversational": False}
        ),
        patch("admin_pipeline._is_admin_conversational", return_value=False),
        patch(
            "admin_pipeline.qie_process",
            return_value=types.SimpleNamespace(canonical_text=message),
        ),
        patch("admin_pipeline.plan_query", return_value=_unclassifiable_plan(message)),
        patch("admin_pipeline.classify_admin_intent", return_value={}),
        patch("admin_pipeline.get_flags", return_value=_patched_flags(enable_sql)),
    ]


class TestHybridFallbackFlagOff:
    def test_unclassifiable_query_falls_to_unrecognised_when_flag_off(self):
        message = "asdlkj qwoie zxcv random gibberish text here"
        patches = _base_patches(message, enable_sql=False)
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            with (
                patch("sql_query_plan.run_sql_plan") as mock_run_sql,
                patch("admin_pipeline._call_dotnet") as mock_call_dotnet,
            ):
                result = execute_admin_query(message, {}, session_id="s1")

        assert result["type"] == "unrecognised"
        mock_run_sql.assert_not_called()
        mock_call_dotnet.assert_not_called()


class TestHybridFallbackFlagOn:
    def test_sql_backend_serves_unclassifiable_query(self):
        message = "asdlkj qwoie zxcv random gibberish text here"
        section = {
            "success": True,
            "records": [{"agniveerNo": "A1", "fullName": "X"}],
            "data": [{"agniveerNo": "A1", "fullName": "X"}],
            "count": 1,
        }
        patches = _base_patches(message, enable_sql=True)
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            with (
                patch(
                    "sql_query_plan.run_sql_plan", return_value=(section, None)
                ) as mock_run_sql,
                patch("admin_pipeline._call_dotnet") as mock_call_dotnet,
                patch("admin_pipeline.generate_report") as mock_report,
            ):
                mock_report.return_value = {
                    "introMessage": "Here you go.",
                    "analysis": {
                        "summary": "Summary.",
                        "observations": [],
                        "insights": [],
                    },
                    "conclusion": {"summary": "Done."},
                }
                result = execute_admin_query(message, {}, session_id="s2")

        assert result["type"] == "query"
        mock_run_sql.assert_called_once()
        mock_call_dotnet.assert_not_called()

        response_payload = result["response_payload"]
        # No raw SQL / prompt ever reaches the response — only the safe
        # {"backend": "sql"} marker, mirroring the .NET dotnetPayload field.
        assert response_payload["dotnetPayload"] == [{"backend": "sql"}]
        assert "SELECT" not in str(response_payload)
        assert "FROM AgniveerMaster" not in str(response_payload)

    def test_sql_backend_failure_falls_back_to_unrecognised(self):
        message = "asdlkj qwoie zxcv random gibberish text here"
        patches = _base_patches(message, enable_sql=True)
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            with (
                patch(
                    "sql_query_plan.run_sql_plan", return_value=(None, "CANNOT_ANSWER")
                ) as mock_run_sql,
                patch("admin_pipeline._call_dotnet") as mock_call_dotnet,
            ):
                result = execute_admin_query(message, {}, session_id="s3")

        assert result["type"] == "unrecognised"
        mock_run_sql.assert_called_once()
        mock_call_dotnet.assert_not_called()

    def test_sql_backend_exception_falls_back_to_unrecognised(self):
        message = "asdlkj qwoie zxcv random gibberish text here"
        patches = _base_patches(message, enable_sql=True)
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            with (
                patch("sql_query_plan.run_sql_plan", side_effect=RuntimeError("boom")),
                patch("admin_pipeline._call_dotnet") as mock_call_dotnet,
            ):
                result = execute_admin_query(message, {}, session_id="s4")

        assert result["type"] == "unrecognised"
        mock_call_dotnet.assert_not_called()


class TestGreetingAndHighConfidenceUnaffected:
    def test_greeting_still_short_circuits_regardless_of_flag(self):
        with patch("admin_pipeline.get_flags", return_value=_patched_flags(True)):
            with (
                patch("sql_query_plan.run_sql_plan") as mock_run_sql,
                patch("admin_pipeline._call_dotnet") as mock_call_dotnet,
            ):
                result = execute_admin_query("hello", {}, session_id="s5")

        assert result["type"] == "greeting"
        mock_run_sql.assert_not_called()
        mock_call_dotnet.assert_not_called()

    def test_high_confidence_query_still_uses_dotnet_with_flag_on(self):
        with patch("admin_pipeline.get_flags", return_value=_patched_flags(True)):
            with (
                patch("admin_pipeline._call_dotnet") as mock_call_dotnet,
                patch("admin_pipeline.generate_report") as mock_report,
                patch("sql_query_plan.run_sql_plan") as mock_run_sql,
            ):
                mock_call_dotnet.return_value = ({"records": []}, None)
                mock_report.return_value = {
                    "introMessage": "Intro",
                    "analysis": {"summary": "Sum", "observations": [], "insights": []},
                    "conclusion": {"summary": "Conc"},
                }
                result = execute_admin_query(
                    "Show attendance for agniveer 12345", {}, session_id="s6"
                )

        assert result["type"] == "query"
        mock_call_dotnet.assert_called_once()
        mock_run_sql.assert_not_called()
