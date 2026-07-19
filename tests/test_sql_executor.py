"""
tests/test_sql_executor.py
===========================
Unit tests for sql_executor.py — the text-to-SQL fetch backend.

Covers: validate_sql regression, _extract_sql fence/prose stripping,
_to_section normalization (feeds universal_normalizer / result_combiner
without hitting their raw-row fallback), golden fast-path validation, and
no-raw-SQL-leak into the returned section.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from sql_executor import (
    _extract_sql,
    _to_camel_case,
    _to_section,
    execute_sql_query,
    validate_sql,
)

# ── validate_sql — safety regression ────────────────────────────────────────


class TestValidateSqlAllows:
    def test_allows_plain_select(self):
        assert validate_sql("SELECT AgniveerNo FROM AgniveerMaster") is None

    def test_allows_with_cte_select(self):
        sql = "WITH F1 AS (SELECT AgniveerId FROM AgniveerLeaveMaster) SELECT * FROM F1"
        assert validate_sql(sql) is None

    def test_allows_trailing_semicolon(self):
        assert validate_sql("SELECT 1;") is None


class TestValidateSqlBlocks:
    @pytest.mark.parametrize(
        "sql",
        [
            "INSERT INTO AgniveerMaster (FullName) VALUES ('x')",
            "UPDATE AgniveerMaster SET FullName = 'x'",
            "DELETE FROM AgniveerMaster",
            "MERGE INTO AgniveerMaster USING x ON 1=1",
            "DROP TABLE AgniveerMaster",
            "ALTER TABLE AgniveerMaster ADD COLUMN x INT",
            "TRUNCATE TABLE AgniveerMaster",
            "CREATE TABLE x (id INT)",
            "GRANT SELECT ON AgniveerMaster TO public",
            "EXEC sp_who",
            "EXECUTE sp_configure",
            "SELECT * FROM AgniveerMaster; EXEC xp_cmdshell 'dir'",
        ],
    )
    def test_blocks_dml_ddl_and_exec(self, sql):
        assert validate_sql(sql) is not None

    def test_blocks_statement_chaining(self):
        sql = "SELECT 1; SELECT 2"
        assert validate_sql(sql) is not None

    def test_blocks_line_comment(self):
        sql = "SELECT AgniveerNo FROM AgniveerMaster -- drop everything"
        assert validate_sql(sql) is not None

    def test_blocks_block_comment(self):
        sql = "SELECT AgniveerNo /* sneaky */ FROM AgniveerMaster"
        assert validate_sql(sql) is not None

    def test_blocks_password_column(self):
        sql = "SELECT Password FROM UserMaster"
        assert validate_sql(sql) is not None

    def test_blocks_logintoken_table(self):
        sql = "SELECT Token FROM LoginToken"
        assert validate_sql(sql) is not None

    def test_blocks_defaultlog_table(self):
        sql = "SELECT * FROM DefaultLog"
        assert validate_sql(sql) is not None

    def test_blocks_empty_sql(self):
        assert validate_sql("") is not None
        assert validate_sql("   ") is not None

    def test_blocks_non_select_statement(self):
        assert validate_sql("SHOW TABLES") is not None


class TestValidateSqlBestAttemptRule:
    """R7 (business_rules.LLM_HARD_RULES): aggregating MarksObtained from
    AgniveerScoreAttempt without scoping to one attempt per Agniveer/sub-item
    double-counts retakes. Regression for a real generated query that summed
    marks for a sport filter without `IsBestAttempt = 1`."""

    def test_blocks_marks_sum_without_best_attempt_or_attempt_no(self):
        sql = (
            "SELECT a.AgniveerNo, a.FullName, SUM(sa.MarksObtained) AS BestTotal "
            "FROM AgniveerMaster a "
            "INNER JOIN AgniveerScoreAttempt sa ON a.Id = sa.AgniveerId "
            "WHERE (a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL) "
            "AND a.Sports = 'Volleyball' "
            "GROUP BY a.AgniveerNo, a.FullName ORDER BY BestTotal DESC"
        )
        err = validate_sql(sql)
        assert err is not None
        assert "MarksObtained" in err

    def test_allows_marks_sum_with_is_best_attempt(self):
        sql = (
            "SELECT a.AgniveerNo, SUM(sa.MarksObtained) AS BestTotal "
            "FROM AgniveerMaster a "
            "INNER JOIN AgniveerScoreAttempt sa ON a.Id = sa.AgniveerId "
            "WHERE sa.IsBestAttempt = 1 AND a.Sports = 'Volleyball' "
            "GROUP BY a.AgniveerNo"
        )
        assert validate_sql(sql) is None

    def test_allows_marks_sum_grouped_by_attempt_no(self):
        sql = (
            "SELECT sa.AgniveerId, sa.AttemptNo, SUM(sa.MarksObtained) AS TotalMarks "
            "FROM AgniveerScoreAttempt sa "
            "GROUP BY sa.AgniveerId, sa.AttemptNo"
        )
        assert validate_sql(sql) is None

    def test_allows_query_without_marks_column(self):
        sql = "SELECT AgniveerId, AttemptedDate FROM AgniveerScoreAttempt"
        assert validate_sql(sql) is None


# ── _extract_sql — fence / prose stripping ──────────────────────────────────


class TestExtractSql:
    def test_strips_markdown_fence(self):
        text = "```sql\nSELECT 1\n```"
        assert _extract_sql(text) == "SELECT 1"

    def test_strips_leading_prose(self):
        text = "Here is the query:\nSELECT AgniveerNo FROM AgniveerMaster"
        assert _extract_sql(text).startswith("SELECT AgniveerNo")

    def test_keeps_with_cte(self):
        text = "Sure!\nWITH F1 AS (SELECT 1) SELECT * FROM F1"
        assert _extract_sql(text).startswith("WITH F1")

    def test_no_sql_found_returns_stripped_text(self):
        text = "  CANNOT_ANSWER  "
        assert _extract_sql(text) == "CANNOT_ANSWER"


# ── _to_section — result-shape adapter (Task 3) ─────────────────────────────


class TestToSection:
    def test_shape(self):
        rows = [
            {"agniveerNo": "A1", "fullName": "X"},
            {"agniveerNo": "A2", "fullName": "Y"},
        ]
        section = _to_section(rows, intent={"category": "Attendance"})
        assert section["success"] is True
        assert section["records"] == rows
        assert section["data"] == rows
        assert section["count"] == 2

    def test_empty_rows(self):
        section = _to_section([], intent=None)
        assert section["records"] == []
        assert section["count"] == 0

    def test_normalize_response_resolves_rows_directly(self):
        # Resolves the rows found by _walk directly (agniveerNo present),
        # rather than falling back to _fallback_raw_rows scanning. The
        # walker also inherits the envelope's top-level "success" scalar
        # into each record (existing universal_normalizer context-inheritance
        # behavior, unrelated to this adapter) — so every field of the
        # original row is preserved, plus that one inherited key.
        from universal_normalizer import normalize_response

        rows = [{"agniveerNo": "A1", "fullName": "X", "score": 91}]
        section = _to_section(rows, intent={"category": "Performance"})
        result = normalize_response(section)
        assert len(result) == len(rows)
        assert result[0] == {**rows[0], "success": True}

    def test_result_combiner_extract_records_resolves_rows_directly(self):
        from result_combiner import _extract_records

        rows = [{"agniveerNo": "A1", "fullName": "X"}]
        section = _to_section(rows, intent=None)
        result = _extract_records(section)
        assert len(result) == len(rows)
        assert result[0] == {**rows[0], "success": True}

    def test_sql_is_not_inherited_into_row_records(self):
        from universal_normalizer import normalize_response

        rows = [{"agniveerNo": "A1", "fullName": "X"}]
        section = _to_section(rows, intent=None, sql="SELECT 1")
        result = normalize_response(section)
        assert len(result) == 1
        assert "sql" not in result[0]
        assert result[0]["agniveerNo"] == "A1"


# ── Golden fast-path (Task 7) ────────────────────────────────────────────────


class TestTwoTierPipeline:
    def test_ast_success_does_not_call_llm(self):
        from ast_models import ASTNode
        with (
            patch("query_planner_v2.query_planner_v2.plan_query") as mock_plan,
            patch("sql_validator.sql_validator.validate_ast") as mock_val_ast,
            patch("sql_builder.sql_builder.build") as mock_build,
            patch("sql_validator.sql_validator.validate_sql") as mock_val_sql,
            patch("sql_executor.generate_sql") as mock_generate,
            patch("sql_executor.run_readonly") as mock_run,
        ):
            mock_plan.return_value = ASTNode(base_table="AgniveerMaster")
            mock_val_ast.return_value = (True, None)
            mock_build.return_value = ("SELECT AgniveerNo FROM AgniveerMaster", [])
            mock_val_sql.return_value = (True, None)
            mock_run.return_value = ([{"agniveerNo": "A1"}], None)
            
            data, err = execute_sql_query(question="who is A1", intent={"category": "Agniveer"})
            assert err is None
            mock_generate.assert_not_called()
            assert data["records"] == [{"agniveerNo": "A1"}]

    def test_ast_failure_falls_back_to_llm(self):
        with (
            patch("query_planner_v2.query_planner_v2.plan_query") as mock_plan,
            patch("sql_executor.generate_sql") as mock_generate,
            patch("sql_validator.sql_validator.validate_sql") as mock_val_sql,
            patch("sql_executor.run_readonly") as mock_run,
            patch("sql_executor.metrics_hook") as mock_metrics,
        ):
            mock_plan.side_effect = Exception("Planner error")
            mock_generate.return_value = ("SELECT AgniveerNo FROM AgniveerMaster", None)
            mock_val_sql.return_value = (True, None)
            mock_run.return_value = ([{"agniveerNo": "A2"}], None)
            
            data, err = execute_sql_query(question="who is A2", intent={"category": "Agniveer"})
            assert err is None
            mock_generate.assert_called_once()
            assert data["records"] == [{"agniveerNo": "A2"}]
            mock_metrics.assert_any_call("llm_fallback")


# ── execute_sql_query — end-to-end contract ─────────────────────────────────


class TestExecuteSqlQuery:
    def test_returns_section_on_success(self):
        from ast_models import ASTNode
        with (
            patch("query_planner_v2.query_planner_v2.plan_query") as mock_plan,
            patch("sql_validator.sql_validator.validate_ast") as mock_val_ast,
            patch("sql_builder.sql_builder.build") as mock_build,
            patch("sql_validator.sql_validator.validate_sql") as mock_val_sql,
            patch("sql_executor.run_readonly") as mock_run,
        ):
            mock_plan.return_value = ASTNode(base_table="AgniveerMaster")
            mock_val_ast.return_value = (True, None)
            mock_build.return_value = ("SELECT AgniveerNo FROM AgniveerMaster", [])
            mock_val_sql.return_value = (True, None)
            mock_run.return_value = ([{"agniveerNo": "A1"}], None)
            data, err = execute_sql_query(question="who is A1", intent={"category": "Agniveer"})
            assert err is None
            assert data["success"] is True
            assert data["records"] == [{"agniveerNo": "A1"}]

    def test_cannot_answer_bubbles_error(self):
        with (
            patch("query_planner_v2.query_planner_v2.plan_query") as mock_plan,
            patch("sql_executor.generate_sql") as mock_generate,
        ):
            mock_plan.side_effect = Exception("CANNOT_ANSWER")
            mock_generate.return_value = (None, "CANNOT_ANSWER")
            data, err = execute_sql_query(question="what's the weather", intent={"category": "Agniveer"})
            assert data is None
            assert "CANNOT_ANSWER" in err

    def test_validator_rejection_bubbles_error(self):
        with (
            patch("query_planner_v2.query_planner_v2.plan_query") as mock_plan,
            patch("sql_validator.sql_validator.validate_ast") as mock_val_ast,
            patch("sql_executor.generate_sql") as mock_generate,
        ):
            mock_plan.return_value = None
            mock_val_ast.return_value = (False, "AST rejected")
            mock_generate.return_value = (None, "AST rejected")
            data, err = execute_sql_query(question="delete everyone", intent={"category": "Agniveer"})
            assert data is None
            assert err is not None

    def test_exec_error_bubbles_error(self):
        from ast_models import ASTNode
        with (
            patch("query_planner_v2.query_planner_v2.plan_query") as mock_plan,
            patch("sql_validator.sql_validator.validate_ast") as mock_val_ast,
            patch("sql_builder.sql_builder.build") as mock_build,
            patch("sql_validator.sql_validator.validate_sql") as mock_val_sql,
            patch("sql_executor.run_readonly") as mock_run,
            patch("sql_executor.generate_sql") as mock_generate,
        ):
            mock_plan.return_value = ASTNode(base_table="AgniveerMaster")
            mock_val_ast.return_value = (True, None)
            mock_build.return_value = ("SELECT 1", [])
            mock_val_sql.return_value = (True, None)
            mock_run.return_value = (
                None,
                "The generated query could not be executed against the database.",
            )
            mock_generate.return_value = (None, "Fallback failed")
            data, err = execute_sql_query(question="anything", intent={"category": "Agniveer"})
            assert data is None
            assert err

    def test_executed_sql_is_surfaced_for_dotnet_payload_transparency(self):
        from ast_models import ASTNode
        sql_text = "SELECT AgniveerNo, FullName FROM AgniveerMaster WHERE Height > 170"
        with (
            patch("query_planner_v2.query_planner_v2.plan_query") as mock_plan,
            patch("sql_validator.sql_validator.validate_ast") as mock_val_ast,
            patch("sql_builder.sql_builder.build") as mock_build,
            patch("sql_validator.sql_validator.validate_sql") as mock_val_sql,
            patch("sql_executor.run_readonly") as mock_run,
        ):
            mock_plan.return_value = ASTNode(base_table="AgniveerMaster")
            mock_val_ast.return_value = (True, None)
            mock_build.return_value = (sql_text, [])
            mock_val_sql.return_value = (True, None)
            mock_run.return_value = ([{"agniveerNo": "A1", "fullName": "X"}], None)
            data, err = execute_sql_query(question="tall agniveers", intent={"category": "Agniveer"})
            assert err is None
            assert data["sql"] == sql_text
            # execution_metadata was also added in the new pipeline
            assert "execution_metadata" in data

    @pytest.mark.parametrize(
        "intent, expected_fragment",
        [
            ({"category": "Agniveer", "company_id": 2}, "Company.Id"),
            ({"category": "Agniveer", "platoon_id": 3}, "Agniveer.PlatoonId"),
            ({"category": "Agniveer", "batch_id": 4}, "Agniveer.BatchId"),
            ({"category": "Agniveer", "agniveer_no": "A0701905F"}, "Agniveer.AgniveerNo"),
        ],
    )
    def test_legacy_scope_filters_are_translated_into_v2_filters(self, intent, expected_fragment):
        with (
            patch("query_planner_v2.query_planner_v2.plan_query") as mock_plan,
            patch("sql_validator.sql_validator.validate_ast") as mock_val_ast,
            patch("sql_builder.sql_builder.build") as mock_build,
            patch("sql_validator.sql_validator.validate_sql") as mock_val_sql,
            patch("sql_executor.run_readonly") as mock_run,
        ):
            from ast_models import ASTNode

            mock_plan.return_value = ASTNode(base_table="AgniveerMaster")
            mock_val_ast.return_value = (True, None)
            mock_build.return_value = ("SELECT 1", [])
            mock_val_sql.return_value = (True, None)
            mock_run.return_value = ([{"agniveerNo": "A1"}], None)

            data, err = execute_sql_query(question="test", intent=intent)
            assert err is None
            assert mock_plan.call_args is not None
            planned_intent = mock_plan.call_args.args[0]
            assert expected_fragment in planned_intent["filters"]


# ── _to_camel_case — System.Text.Json-compatible acronym handling ───────────
class TestToCamelCase:
    """Regression coverage for the acronym-run casing fix.

    _to_camel_case must match System.Text.Json's camelCase naming policy —
    which lowercases an entire leading run of uppercase letters up to (but
    not including) the capital that starts the next word — not just the
    first character. A naive "lowercase only the first letter" rule turns
    "OMRInputTotal" into "oMRInputTotal", which silently fails to match
    utils.py's _SCORE_FIELDS entry "omrInputTotal", so any row containing
    that column has its score dropped from every aggregate/comparison/chart
    without ever raising an error.
    """

    @pytest.mark.parametrize(
        "column_name, expected",
        [
            ("OMRInputTotal", "omrInputTotal"),
            ("FullName", "fullName"),
            ("AgniveerNo", "agniveerNo"),
            ("Id", "id"),
            ("IsBestAttempt", "isBestAttempt"),
            ("ID", "id"),
            ("MarksObtained", "marksObtained"),
            ("BestTotal", "bestTotal"),
            ("PPT", "ppt"),
            ("AgniveerId", "agniveerId"),
            ("A", "a"),
            ("PlatoonNo", "platoonNo"),
            ("CompanyId", "companyId"),
            ("BMICategory", "bmiCategory"),
        ],
    )
    def test_camel_cases_correctly(self, column_name, expected):
        assert _to_camel_case(column_name) == expected

    def test_empty_string_is_unchanged(self):
        assert _to_camel_case("") == ""

    def test_omr_score_row_is_findable_after_camel_casing(self):
        """End-to-end: a row shaped like a real SQL Server cursor result
        (PascalCase columns) must, after camelCasing, expose the exact key
        utils.get_score looks for — proving the fix actually unblocks score
        extraction, not just the isolated string transform."""
        from utils import get_score

        row = {"AgniveerNo": "A1", "FullName": "Amit", "OMRInputTotal": 87}
        camel_row = {_to_camel_case(k): v for k, v in row.items()}

        assert camel_row["omrInputTotal"] == 87
        assert get_score(camel_row) == 87.0