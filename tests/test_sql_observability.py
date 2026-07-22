"""
tests/test_sql_observability.py
=================================
Task 9 — observability for the SQL backend: metrics counters increment on
the corresponding paths, and the audit log records which backend served a
query (with no raw SQL/prompt/PII).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from audit_logger import write_audit_log
from metrics import Metrics


class TestSqlMetricsCounters:
    def test_performance_executor_increments_generated_counter(self):
        m = Metrics()
        with (
            patch("metrics.metrics_collector", m),
            patch("sql_executor.run_readonly", return_value=([], None)),
        ):
            from sql_executor import execute_sql_query

            execute_sql_query(
                question="top performers",
                intent={
                    "category": "Performance",
                    "subcategory": "TopPerformers",
                    "operation": "Top",
                },
            )
        assert m.sql_generated_total == 1
        assert m.sql_llm_fallback_total == 0
        assert m.sql_capability_gap_fallback_total == 0

    # The tests below exercise the generic AST-planner/text2sql tiered
    # pipeline via the mocked query_planner_v2/sql_builder/sql_validator/
    # generate_sql chain, so the intent's "category" must be one with no
    # deterministic fast-path in execute_sql_query (Attendance/Performance/
    # Medical/... all have one now and would return before ever reaching
    # the mocks) — "UnroutedCategory" is a synthetic value that falls
    # through every fast-path check by construction.

    def test_generated_increments_counter(self):
        m = Metrics()
        from ast_models import ASTNode
        with (
            patch("metrics.metrics_collector", m),
            patch("query_planner_v2.query_planner_v2.plan_query", return_value=ASTNode(base_table="AttendanceMaster")),
            patch("sql_validator.sql_validator.validate_ast", return_value=(True, None)),
            patch("sql_builder.sql_builder.build", return_value=("SELECT 1", [])),
            patch("sql_validator.sql_validator.validate_sql", return_value=(True, None)),
            patch("sql_executor.run_readonly", return_value=([], None)),
        ):
            from sql_executor import execute_sql_query

            execute_sql_query(question="anything", intent={"category": "UnroutedCategory"})
        assert m.sql_generated_total == 1

    def test_cannot_answer_increments_counter(self):
        m = Metrics()
        with (
            patch("metrics.metrics_collector", m),
            patch("query_planner_v2.query_planner_v2.plan_query", side_effect=Exception("AST Failed")),
            patch("sql_executor.generate_sql", return_value=(None, "Cannot answer")),
        ):
            from sql_executor import execute_sql_query

            execute_sql_query(question="anything", intent={"category": "UnroutedCategory"})
        assert m.sql_cannot_answer_total == 1

    def test_validator_rejected_increments_counter(self):
        m = Metrics()
        with (
            patch("metrics.metrics_collector", m),
            patch("query_planner_v2.query_planner_v2.plan_query", return_value="mock_ast"),
            patch("sql_validator.sql_validator.validate_ast", return_value=(False, "Invalid AST")),
            patch("sql_executor.generate_sql", return_value=(None, "Fallback failed")),
        ):
            from sql_executor import execute_sql_query

            execute_sql_query(question="delete everyone", intent={"category": "UnroutedCategory"})
        assert m.sql_validator_rejected_total == 1

    def test_exec_error_increments_counter(self):
        m = Metrics()
        with (
            patch("metrics.metrics_collector", m),
            patch("query_planner_v2.query_planner_v2.plan_query", return_value="mock_ast"),
            patch("sql_validator.sql_validator.validate_ast", return_value=(True, None)),
            patch("sql_builder.sql_builder.build", return_value=("SELECT 1", [])),
            patch("sql_validator.sql_validator.validate_sql", return_value=(True, None)),
            patch(
                "sql_executor.run_readonly",
                return_value=(
                    None,
                    "Database query execution failed.",
                ),
            ),
            patch("sql_executor.generate_sql", return_value=(None, "Fallback failed")),
        ):
            from sql_executor import execute_sql_query

            execute_sql_query(question="anything", intent={"category": "UnroutedCategory"})
        assert m.sql_exec_error_total == 1

    def test_metrics_hook_never_raises_on_broken_collector(self):
        from sql_executor import metrics_hook

        with patch("metrics.metrics_collector", None):
            metrics_hook("generated")  # must not raise

    def test_record_sql_latency(self):
        m = Metrics()
        m.record_sql_latency(0.25)
        m.record_sql_latency(0.75)
        assert m.sql_latency_seconds["sum"] == 1.0
        assert m.sql_latency_seconds["count"] == 2.0

    def test_prometheus_export_includes_sql_metrics(self):
        m = Metrics()
        m.inc_sql_generated()
        m.inc_sql_capability_gap_fallback()
        text = m.generate_prometheus_text()
        assert "sql_generated_total 1" in text
        assert "sql_capability_gap_fallback_total 1" in text
        assert "sql_latency_seconds_sum" in text

    def test_performance_average_increments_generated_counter(self):
        m = Metrics()
        with (
            patch("metrics.metrics_collector", m),
            patch("sql_executor.run_readonly", return_value=([], None)),
        ):
            from sql_executor import execute_sql_query

            execute_sql_query(
                question="average marks per section",
                intent={
                    "category": "Performance",
                    "operation": "Average",
                },
            )
        assert m.sql_generated_total == 1
        assert m.sql_capability_gap_fallback_total == 0
        assert m.sql_llm_fallback_total == 0

    def test_structural_reject_fallback_increments_counter(self):
        m = Metrics()
        with (
            patch("metrics.metrics_collector", m),
            patch("query_planner_v2.query_planner_v2.plan_query", return_value="mock_ast"),
            patch("sql_validator.sql_validator.validate_ast", return_value=(False, "Invalid AST")),
            patch("sql_executor.generate_sql", return_value=("SELECT 1", None)),
            patch("sql_validator.sql_validator.validate_sql", return_value=(True, None)),
            patch("sql_executor.run_readonly", return_value=([], None)),
        ):
            from sql_executor import execute_sql_query

            execute_sql_query(question="anything", intent={"category": "UnroutedCategory"})
        assert m.sql_structural_reject_fallback_total == 1
        assert m.sql_validator_rejected_total == 1
        assert m.sql_llm_fallback_total == 1

    def test_database_error_bubbles_without_fallback(self):
        m = Metrics()
        from ast_models import ASTNode
        mock_generate = MagicMock()
        with (
            patch("metrics.metrics_collector", m),
            patch("query_planner_v2.query_planner_v2.plan_query", return_value=ASTNode(base_table="AttendanceMaster")),
            patch("sql_validator.sql_validator.validate_ast", return_value=(True, None)),
            patch("sql_builder.sql_builder.build", return_value=("SELECT 1", [])),
            patch("sql_validator.sql_validator.validate_sql", return_value=(True, None)),
            patch("sql_executor.run_readonly", return_value=(None, "Database is down")),
            patch("sql_executor.generate_sql", mock_generate),
        ):
            from sql_executor import execute_sql_query

            res, err = execute_sql_query(question="anything", intent={"category": "UnroutedCategory"})
            assert err is not None
            assert "Database query execution failed" in err
            assert res is None
            
        mock_generate.assert_not_called()
        assert m.sql_exec_error_total == 1
        assert m.sql_llm_fallback_total == 0



class TestAuditLogBackendField:
    def test_backend_defaults_to_dotnet(self):
        with patch("audit_logger._get_audit_logger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            write_audit_log(
                question="Show attendance", intent={"category": "UnroutedCategory"}
            )

        logged = json.loads(mock_logger.info.call_args[0][0])
        assert logged["backend"] == "dotnet"

    def test_backend_sql_is_recorded(self):
        with patch("audit_logger._get_audit_logger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            write_audit_log(
                question="some long-tail query",
                intent={"category": None},
                backend="sql",
            )

        logged = json.loads(mock_logger.info.call_args[0][0])
        assert logged["backend"] == "sql"

    def test_invalid_backend_falls_back_to_dotnet(self):
        with patch("audit_logger._get_audit_logger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            write_audit_log(question="q", intent={}, backend="not-a-real-backend")

        logged = json.loads(mock_logger.info.call_args[0][0])
        assert logged["backend"] == "dotnet"

    def test_no_raw_sql_or_pii_in_audit_entry(self):
        with patch("audit_logger._get_audit_logger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            write_audit_log(
                question="who has fever",
                intent={"category": "Medical"},
                backend="sql",
                dotnet_payload=[{"backend": "sql"}],
            )

        logged = json.loads(mock_logger.info.call_args[0][0])
        assert logged["dotnet_payload"] == [{"backend": "sql"}]
        assert "SELECT" not in json.dumps(logged)
