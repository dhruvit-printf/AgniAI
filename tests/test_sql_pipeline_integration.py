"""
tests/test_sql_pipeline_integration.py
========================================
Integration tests for the SQL-only pipeline wiring in admin_pipeline.py.

Post text-to-SQL migration, admin_pipeline.py calls fetch_sql_results()
unconditionally for every non-conversational query — there is no .NET
fallback, no confidence threshold, and no feature flag gating it. These
tests use controlled mocks for planning so a fixed query_plan reaches the
SQL-execution stage deterministically, independent of NLP heuristic drift.
"""

from __future__ import annotations

import types
from unittest.mock import patch

from admin_pipeline import execute_admin_query
from intent_engine.query_planner import QueryPlan, QueryType, SubOperation


def _simple_plan(message: str, confidence: float = 0.1) -> QueryPlan:
    op = SubOperation(raw_fragment=message, intent_result={}, dotnet_payload={})
    return QueryPlan(
        query_type=QueryType.SIMPLE,
        operations=[op],
        confidence=confidence,
        raw_query=message,
        reasoning="test: forced simple plan",
    )


def _base_patches(message: str, confidence: float = 0.1):
    """Common patch set that deterministically routes `message` into the
    single-op SQL-execution stage of admin_pipeline, regardless of real NLP
    classification behavior."""
    return [
        patch(
            "admin_pipeline.understand_query", return_value={"conversational": False}
        ),
        patch("admin_pipeline._is_admin_conversational", return_value=False),
        patch(
            "admin_pipeline.qie_process",
            return_value=types.SimpleNamespace(canonical_text=message),
        ),
        patch(
            "admin_pipeline.plan_query",
            return_value=_simple_plan(message, confidence=confidence),
        ),
    ]


class TestSqlIsTheOnlyBackend:
    def test_sql_backend_always_serves_the_query(self):
        message = "show me something"
        section = {
            "success": True,
            "records": [{"agniveerNo": "A1", "fullName": "X"}],
            "data": [{"agniveerNo": "A1", "fullName": "X"}],
            "count": 1,
        }
        patches = _base_patches(message)
        with patches[0], patches[1], patches[2], patches[3]:
            with (
                patch(
                    "admin_pipeline.fetch_sql_results",
                    return_value=([section], [("Result", section)], None),
                ) as mock_fetch_sql,
                patch("admin_pipeline.generate_report") as mock_report,
            ):
                mock_report.return_value = {
                    "message": "Here you go.",
                    "analysis": {
                        "summary": "Summary.",
                        "observations": [],
                        "insights": [],
                    },
                    "conclusion": {"summary": "Done."},
                }
                result = execute_admin_query(message, {}, session_id="s2")

        assert result["type"] == "query"
        mock_fetch_sql.assert_called_once()

        response_payload = result["response_payload"]
        # No raw SQL / prompt ever reaches the response — only the safe
        # {"backend": "sql"} marker.
        assert response_payload["dotnetPayload"] == [{"backend": "sql"}]
        assert "SELECT" not in str(response_payload)
        assert "FROM AgniveerMaster" not in str(response_payload)

    def test_high_confidence_query_still_uses_sql(self):
        """Unlike the retired hybrid design, a high classifier confidence
        does not skip the SQL backend — there is no other backend left."""
        message = "Show attendance for agniveer 12345"
        section = {"success": True, "records": [], "data": [], "count": 0}
        patches = _base_patches(message, confidence=0.95)
        with patches[0], patches[1], patches[2], patches[3]:
            with (
                patch(
                    "admin_pipeline.fetch_sql_results",
                    return_value=([section], [("Result", section)], None),
                ) as mock_fetch_sql,
                patch("admin_pipeline.generate_report") as mock_report,
            ):
                mock_report.return_value = {
                    "message": "Intro",
                    "analysis": {"summary": "Sum", "observations": [], "insights": []},
                    "conclusion": {"summary": "Conc"},
                }
                result = execute_admin_query(message, {}, session_id="s6")

        assert result["type"] == "query"
        mock_fetch_sql.assert_called_once()

    def test_sql_backend_failure_falls_back_to_unrecognised(self):
        message = "asdlkj qwoie zxcv random gibberish text here"
        patches = _base_patches(message)
        with patches[0], patches[1], patches[2], patches[3]:
            with patch(
                "admin_pipeline.fetch_sql_results",
                return_value=([], [], "CANNOT_ANSWER"),
            ) as mock_fetch_sql:
                result = execute_admin_query(message, {}, session_id="s3")

        assert result["type"] == "unrecognised"
        mock_fetch_sql.assert_called_once()

    def test_sql_backend_exception_falls_back_to_unrecognised(self):
        message = "asdlkj qwoie zxcv random gibberish text here"
        patches = _base_patches(message)
        with patches[0], patches[1], patches[2], patches[3]:
            with patch(
                "admin_pipeline.fetch_sql_results",
                side_effect=RuntimeError("boom"),
            ):
                result = execute_admin_query(message, {}, session_id="s4")

        assert result["type"] == "unrecognised"


class TestGreetingUnaffected:
    def test_greeting_still_short_circuits(self):
        with patch("admin_pipeline.fetch_sql_results") as mock_fetch_sql:
            result = execute_admin_query("hello", {}, session_id="s5")

        assert result["type"] == "greeting"
        mock_fetch_sql.assert_not_called()
