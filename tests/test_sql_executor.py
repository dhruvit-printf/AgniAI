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

    def test_medical_bmi_uses_formula_not_stored_column(self):
        with (
            patch("sql_executor.run_readonly") as mock_run,
            patch("sql_validator.sql_validator.validate_sql") as mock_validate_sql,
            patch("sql_executor.metrics_hook"),
        ):
            mock_validate_sql.return_value = (True, None)
            mock_run.return_value = ([{"AgniveerNo": "A1", "FullName": "X"}], None)
            data, err = execute_sql_query(
                question="who is overweight",
                intent={
                    "category": "Medical",
                    "operation": "BMI",
                    "bmiCategory": "Overweight",
                },
            )
            assert err is None
            assert data["success"] is True
            sql = mock_run.call_args[0][0]
            assert "MedicalRecordMaster" in sql
            assert "POWER(EffHeight / 100.0, 2)" in sql
            assert "a.BmiValue" not in sql
            assert "BmiCategory" in sql
            assert "BmiValue >= 25.0 AND BmiValue < 30.0" in sql

    def test_medical_bmi_unfit_covers_overweight_and_obese(self):
        with (
            patch("sql_executor.run_readonly") as mock_run,
            patch("sql_validator.sql_validator.validate_sql") as mock_validate_sql,
            patch("sql_executor.metrics_hook"),
        ):
            mock_validate_sql.return_value = (True, None)
            mock_run.return_value = ([{"AgniveerNo": "A1", "FullName": "X"}], None)
            data, err = execute_sql_query(
                question="who is unfit",
                intent={
                    "category": "Medical",
                    "operation": "BMI",
                    "bmiCategory": "Unfit",
                },
            )
            assert err is None
            assert data["success"] is True
            sql = mock_run.call_args[0][0]
            assert "WHERE BmiValue >= 25.0" in sql
            # Unfit must not carry the Overweight-only upper bound — it's the
            # union of Overweight and Obese, not Overweight alone.
            assert "BmiValue >= 25.0 AND BmiValue < 30.0" not in sql

    def test_medical_blood_group_detail_uses_person_and_scope(self):
        with (
            patch("sql_executor.run_readonly") as mock_run,
            patch("sql_validator.sql_validator.validate_sql") as mock_validate_sql,
            patch("sql_executor.metrics_hook"),
        ):
            mock_validate_sql.return_value = (True, None)
            mock_run.return_value = ([{"AgniveerNo": "A0701882L", "FullName": "X"}], None)
            data, err = execute_sql_query(
                question="What is the blood group of Agniveer A0701882L in Alpha company?",
                intent={
                    "category": "Medical",
                    "operation": "BloodGroup",
                    "agniveer_no": "A0701882L",
                    "companyName": "Alpha",
                },
            )
            assert err is None
            assert data["success"] is True
            sql = mock_run.call_args[0][0]
            assert "AgniveerMaster" in sql
            assert "BloodGroup" in sql
            assert "COUNT(*) AS AgniveerCount" not in sql
            assert "a.AgniveerNo" in sql
            assert "a.FullName" in sql
            assert "COALESCE(NULLIF(a.BloodGroup, ''), 'Unknown')" in sql
            assert "LOWER(a.AgniveerNo) LIKE" in sql
            assert "LOWER(c.Name) = LOWER(?)" in sql

    def test_medical_blood_group_report_counts_each_group(self):
        with (
            patch("sql_executor.run_readonly") as mock_run,
            patch("sql_validator.sql_validator.validate_sql") as mock_validate_sql,
            patch("sql_executor.metrics_hook"),
        ):
            mock_validate_sql.return_value = (True, None)
            mock_run.return_value = ([{"BloodGroup": "A+", "AgniveerCount": 10}], None)
            data, err = execute_sql_query(
                question="Show the blood group report",
                intent={
                    "category": "Medical",
                    "operation": "BloodGroup",
                },
            )
            assert err is None
            assert data["success"] is True
            sql = mock_run.call_args[0][0]
            assert "GROUP BY BloodGroup" in sql
            assert "COUNT(*) AS AgniveerCount" in sql
            assert "AgniveerNo, FullName, BloodGroup" not in sql

    def test_schedule_by_company_id_skips_lookup(self):
        with (
            patch("sql_executor.run_readonly") as mock_run,
            patch("sql_validator.sql_validator.validate_sql") as mock_validate_sql,
        ):
            mock_validate_sql.return_value = (True, None)
            mock_run.return_value = (
                [{"CompanyName": "Alpha", "ScheduleDate": "2026-07-20", "Pd": "1"}],
                None,
            )
            data, err = execute_sql_query(
                question="schedule for company 5",
                intent={"category": "Schedule", "operation": "bycompany", "company_id": 5},
            )
            assert err is None
            assert data["success"] is True
            assert mock_run.call_count == 1
            sql, params = mock_run.call_args[0]
            assert "CompanySchedule" in sql
            assert "s.CompanyId = ?" in sql
            assert params == [5]

    def test_schedule_by_company_name_resolves_id_first(self):
        with (
            patch("sql_executor.run_readonly") as mock_run,
            patch("sql_validator.sql_validator.validate_sql") as mock_validate_sql,
        ):
            mock_validate_sql.return_value = (True, None)
            mock_run.side_effect = [
                ([{"CompanyId": 5}], None),
                ([{"CompanyName": "Lakhwinder", "ScheduleDate": "2026-07-20"}], None),
            ]
            data, err = execute_sql_query(
                question="give schedule for lakhwinder company",
                intent={"category": "Schedule", "operation": "bycompany", "company_name": "Lakhwinder"},
            )
            assert err is None
            assert data["success"] is True
            assert mock_run.call_count == 2
            lookup_sql, lookup_params = mock_run.call_args_list[0][0]
            assert "CompanyMaster" in lookup_sql
            assert lookup_params == ["Lakhwinder"]
            final_sql, final_params = mock_run.call_args_list[1][0]
            assert "CompanySchedule" in final_sql
            assert final_params == [5]

    def test_schedule_by_platoon_resolves_company_via_platoon(self):
        with (
            patch("sql_executor.run_readonly") as mock_run,
            patch("sql_validator.sql_validator.validate_sql") as mock_validate_sql,
        ):
            mock_validate_sql.return_value = (True, None)
            mock_run.side_effect = [
                ([{"CompanyId": 7}], None),
                ([{"CompanyName": "Bravo"}], None),
            ]
            data, err = execute_sql_query(
                question="schedule for platoon 3",
                intent={"category": "Schedule", "operation": "bycompany", "platoon_id": 3},
            )
            assert err is None
            lookup_sql, lookup_params = mock_run.call_args_list[0][0]
            assert "PlatoonMaster" in lookup_sql
            assert lookup_params == [3]
            final_sql, final_params = mock_run.call_args_list[1][0]
            assert final_params == [7]

    def test_schedule_by_agniveer_no_resolves_company_via_platoon_join(self):
        with (
            patch("sql_executor.run_readonly") as mock_run,
            patch("sql_validator.sql_validator.validate_sql") as mock_validate_sql,
        ):
            mock_validate_sql.return_value = (True, None)
            mock_run.side_effect = [
                ([{"CompanyId": 9}], None),
                ([{"CompanyName": "Charlie"}], None),
            ]
            data, err = execute_sql_query(
                question="what is the schedule for agniveer A0701882L today",
                intent={
                    "category": "Schedule",
                    "operation": "byagniveer",
                    "agniveer_no": "A0701882L",
                    "date": "2026-07-20",
                },
            )
            assert err is None
            lookup_sql, lookup_params = mock_run.call_args_list[0][0]
            assert "AgniveerMaster" in lookup_sql
            assert "PlatoonMaster" in lookup_sql
            assert lookup_params == ["A0701882L"]
            final_sql, final_params = mock_run.call_args_list[1][0]
            assert "CAST(s.ScheduleDate AS DATE) = CAST(? AS DATE)" in final_sql
            assert final_params == [9, "2026-07-20"]

    def test_schedule_bytoday_filters_on_resolved_date(self):
        with (
            patch("sql_executor.run_readonly") as mock_run,
            patch("sql_validator.sql_validator.validate_sql") as mock_validate_sql,
        ):
            mock_validate_sql.return_value = (True, None)
            mock_run.return_value = ([{"ScheduleDate": "2026-07-20"}], None)
            data, err = execute_sql_query(
                question="what is today's schedule for company 5",
                intent={
                    "category": "Schedule",
                    "operation": "bytoday",
                    "company_id": 5,
                    "date": "2026-07-20",
                },
            )
            assert err is None
            sql, params = mock_run.call_args[0]
            assert "CAST(s.ScheduleDate AS DATE) = CAST(? AS DATE)" in sql
            assert params == [5, "2026-07-20"]

    def test_schedule_bydate_range_uses_from_to(self):
        with (
            patch("sql_executor.run_readonly") as mock_run,
            patch("sql_validator.sql_validator.validate_sql") as mock_validate_sql,
        ):
            mock_validate_sql.return_value = (True, None)
            mock_run.return_value = ([{"ScheduleDate": "2026-07-20"}], None)
            data, err = execute_sql_query(
                question="schedule for company 5 this week",
                intent={
                    "category": "Schedule",
                    "operation": "bydate",
                    "company_id": 5,
                    "from_date": "2026-07-20",
                    "to_date": "2026-07-26",
                },
            )
            assert err is None
            sql, params = mock_run.call_args[0]
            assert "CAST(s.ScheduleDate AS DATE) >= CAST(? AS DATE)" in sql
            assert "CAST(s.ScheduleDate AS DATE) <= CAST(? AS DATE)" in sql
            assert params == [5, "2026-07-20", "2026-07-26"]

    def test_schedule_bycompany_without_date_returns_full_schedule(self):
        with (
            patch("sql_executor.run_readonly") as mock_run,
            patch("sql_validator.sql_validator.validate_sql") as mock_validate_sql,
        ):
            mock_validate_sql.return_value = (True, None)
            mock_run.return_value = ([{"ScheduleDate": "2026-07-20"}], None)
            data, err = execute_sql_query(
                question="schedule for company 5",
                intent={"category": "Schedule", "operation": "bycompany", "company_id": 5},
            )
            assert err is None
            sql, params = mock_run.call_args[0]
            assert "ScheduleDate AS DATE) =" not in sql
            assert params == [5]

    def test_schedule_without_any_scope_queries_all_companies(self):
        # No company/platoon/agniveer at all -> no lookup, falls back to an
        # unscoped (all-companies) schedule query still bound by the date.
        with (
            patch("sql_executor.run_readonly") as mock_run,
            patch("sql_validator.sql_validator.validate_sql") as mock_validate_sql,
        ):
            mock_validate_sql.return_value = (True, None)
            mock_run.return_value = ([{"ScheduleDate": "2026-07-20"}], None)
            data, err = execute_sql_query(
                question="show me the schedule",
                intent={"category": "Schedule", "operation": "bytoday", "date": "2026-07-20"},
            )
            assert err is None
            assert data["success"] is True
            sql, params = mock_run.call_args[0]
            assert "s.CompanyId = ?" not in sql
            assert params == ["2026-07-20"]

    def test_schedule_unresolvable_entity_returns_empty_section(self):
        with (
            patch("sql_executor.run_readonly") as mock_run,
            patch("sql_validator.sql_validator.validate_sql") as mock_validate_sql,
        ):
            mock_validate_sql.return_value = (True, None)
            mock_run.return_value = ([], None)
            data, err = execute_sql_query(
                question="schedule for nonexistent company",
                intent={"category": "Schedule", "operation": "bycompany", "company_name": "Nonexistent"},
            )
            assert err is None
            assert data["success"] is True
            assert data["records"] == []
            # Only the company-lookup call happened — no second query was run
            # once the name failed to resolve to a company.
            assert mock_run.call_count == 1

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


# ── _jsonable — type normalisation ──────────────────────────────────────────


class TestJsonable:
    """Verify that _jsonable converts pyodbc raw types to JSON-safe Python types.

    pyodbc returns datetime.date / datetime.datetime for datetime2/date columns
    and decimal.Decimal for decimal(18,2) columns. Flask's DefaultJSONProvider
    serializes these as RFC 822 strings and JSON strings respectively, neither
    of which matches what .NET System.Text.Json produces (ISO 8601 and JSON
    numbers).  _jsonable fixes this at the point rows leave run_readonly.
    """

    def test_date_becomes_iso_string(self):
        import datetime
        from sql_executor import _jsonable

        result = _jsonable(datetime.date(2001, 5, 14))
        assert result == "2001-05-14"
        assert isinstance(result, str)

    def test_datetime_becomes_iso_string(self):
        import datetime
        from sql_executor import _jsonable

        result = _jsonable(datetime.datetime(2023, 11, 1, 9, 30, 0))
        assert result == "2023-11-01T09:30:00"
        assert isinstance(result, str)

    def test_datetime_is_checked_before_date(self):
        """datetime is a subclass of date; if we checked date first, a
        datetime would be serialized without its time component."""
        import datetime
        from sql_executor import _jsonable

        dt = datetime.datetime(2024, 6, 15, 14, 22, 59)
        result = _jsonable(dt)
        assert "T" in result, "datetime must include the time component"
        assert result == dt.isoformat()

    def test_decimal_becomes_float(self):
        import decimal
        from sql_executor import _jsonable

        result = _jsonable(decimal.Decimal("87.50"))
        assert isinstance(result, float)
        assert result == 87.5

    def test_decimal_zero_becomes_float(self):
        import decimal
        from sql_executor import _jsonable

        assert _jsonable(decimal.Decimal("0.00")) == 0.0

    def test_decimal_large_value(self):
        import decimal
        from sql_executor import _jsonable

        result = _jsonable(decimal.Decimal("12345.99"))
        assert isinstance(result, float)
        assert abs(result - 12345.99) < 1e-6

    def test_int_passthrough(self):
        from sql_executor import _jsonable

        assert _jsonable(42) == 42
        assert isinstance(_jsonable(42), int)

    def test_float_passthrough(self):
        from sql_executor import _jsonable

        assert _jsonable(3.14) == 3.14

    def test_str_passthrough(self):
        from sql_executor import _jsonable

        assert _jsonable("hello") == "hello"

    def test_none_passthrough(self):
        from sql_executor import _jsonable

        assert _jsonable(None) is None

    def test_bool_passthrough(self):
        from sql_executor import _jsonable

        assert _jsonable(True) is True
        assert _jsonable(False) is False


# ── _to_section with raw pyodbc types ────────────────────────────────────────


class TestToSectionTypeNormalization:
    """Regression tests: _to_section must produce a dict whose values are all
    JSON-serializable standard Python types (no datetime.date / Decimal objects)
    because they flow directly to jsonify() via admin_pipeline -> admin_routes.

    These tests mock the raw pyodbc row types (datetime.date, datetime.datetime,
    decimal.Decimal) that pyodbc returns for SQL Server datetime2/date and
    decimal(18,2) columns, and assert on the final values in the returned
    section dict.
    """

    def test_date_column_serialized_as_iso_string(self):
        import datetime

        rows = [{"AgniveerNo": "A1", "DateOfBirth": datetime.date(2001, 5, 14)}]
        section = _to_section(rows)
        record = section["records"][0]

        assert record["dateOfBirth"] == "2001-05-14"
        assert isinstance(record["dateOfBirth"], str)

    def test_datetime_column_serialized_as_iso_string(self):
        import datetime

        rows = [
            {
                "AgniveerNo": "A1",
                "AttendanceDateTime": datetime.datetime(2024, 3, 15, 8, 0, 0),
            }
        ]
        section = _to_section(rows)
        record = section["records"][0]

        assert record["attendanceDateTime"] == "2024-03-15T08:00:00"
        assert isinstance(record["attendanceDateTime"], str)

    def test_decimal_marks_serialized_as_float(self):
        import decimal

        rows = [{"AgniveerNo": "A1", "MarksObtained": decimal.Decimal("87.50")}]
        section = _to_section(rows)
        record = section["records"][0]

        assert isinstance(record["marksObtained"], float)
        assert record["marksObtained"] == 87.5

    def test_decimal_zero_serialized_as_float(self):
        import decimal

        rows = [{"AgniveerNo": "A1", "Cutoff": decimal.Decimal("0.00")}]
        section = _to_section(rows)
        record = section["records"][0]

        assert isinstance(record["cutoff"], float)

    def test_mixed_row_all_types_clean(self):
        """A realistic row with all three problem types at once."""
        import datetime
        import decimal

        rows = [
            {
                "AgniveerNo": "A123",
                "FullName": "Amit Kumar",
                "DateOfBirth": datetime.date(2001, 5, 14),
                "MarksObtained": decimal.Decimal("87.50"),
                "Score": 95,
            }
        ]
        section = _to_section(rows)
        record = section["records"][0]

        assert record["agniveerNo"] == "A123"
        assert record["fullName"] == "Amit Kumar"
        assert record["dateOfBirth"] == "2001-05-14"
        assert isinstance(record["marksObtained"], float)
        assert record["marksObtained"] == 87.5
        assert record["score"] == 95

        # Verify no raw non-JSON-serializable types remain
        import json

        try:
            json.dumps(section)
        except TypeError as exc:
            raise AssertionError(
                f"section is not JSON-serializable after _to_section: {exc}"
            ) from exc

    def test_get_score_finds_decimal_marks_after_normalization(self):
        """utils.get_score must return a numeric value from a row that came
        through _to_section with a Decimal MarksObtained column."""
        import decimal
        from utils import get_score

        rows = [{"AgniveerNo": "A1", "MarksObtained": decimal.Decimal("73.25")}]
        section = _to_section(rows)
        record = section["records"][0]

        score = get_score(record)
        assert score == pytest.approx(73.25)

    def test_null_values_not_affected(self):
        """None values in rows should be preserved unchanged."""
        rows = [{"AgniveerNo": "A1", "DateOfBirth": None, "MarksObtained": None}]
        section = _to_section(rows)
        record = section["records"][0]

        assert record["dateOfBirth"] is None
        assert record["marksObtained"] is None


# ── Precision audit: Decimal→float in _jsonable ──────────────────────────────
#
# FINDING (2026-07-20): NOT AN ACTIVE BUG — MITIGATED BY DESIGN.
#
# ALL threshold/boundary logic (Excellent/Good/SAT/Fail cutoffs at 90/75/60/45)
# lives in SQL Server CASE WHEN expressions that run BEFORE _jsonable; Python
# only receives the already-decided string labels.  Python-side sum() calls in
# analysis_engine / conclusion_engine / prediction_engine operate on individual
# float scores for statistics (average, band counts) — never summing to
# recompute a total and then comparing against a grading threshold.
#
# For decimal(18,2) columns (at most 2 decimal places), any individual value
# that IS compared against the Python thresholds 50.0 / 75.0 converts without
# a precision-flipping error because: all representable 2dp values of the form
# x.00, x.25, x.50, x.75 are EXACT IEEE 754 doubles; values with other 2dp
# suffixes (.10/.20/.30 etc.) have representation error < 1e-11, orders of
# magnitude smaller than the 0.01-unit gap between any score and the nearest
# threshold literal.
#
# The tests below document and lock this invariant.


class TestDecimalPrecisionAudit:
    """Precision audit for _jsonable's Decimal->float conversion.

    Verifies that decimal(18,2) Decimal values produced by pyodbc, converted
    via _jsonable, compare correctly against Python-side threshold literals
    (50.0, 75.0) and that float arithmetic on many such values does not produce
    a sum/average that drifts far enough to flip a boundary comparison.
    """

    def test_exact_threshold_values_convert_without_error(self):
        """The four PPT/BPET grade boundaries are exactly representable."""
        import decimal

        from sql_executor import _jsonable

        for boundary in ("90.00", "75.00", "60.00", "45.00"):
            result = _jsonable(decimal.Decimal(boundary))
            expected = float(boundary)
            assert result == expected, f"Boundary {boundary} has float error"
            assert isinstance(result, float)

    def test_half_decimal_values_are_exact_floats(self):
        """x.00, x.25, x.50, x.75 — always exact in IEEE 754."""
        import decimal

        from sql_executor import _jsonable

        for val_str in ("87.50", "73.25", "60.75", "45.00", "90.00", "0.50", "99.75"):
            result = _jsonable(decimal.Decimal(val_str))
            assert result == pytest.approx(float(val_str), abs=1e-12)

    def test_tenth_decimal_values_have_bounded_representation_error(self):
        """x.10/.20/.30 ... values have IEEE 754 error < 1e-9 after round(,10)."""
        import decimal

        from sql_executor import _jsonable

        for val_str in ("87.60", "73.30", "60.10", "44.90", "50.20", "75.80"):
            result = _jsonable(decimal.Decimal(val_str))
            error = abs(result - float(val_str))
            assert error < 1e-9, f"{val_str}: float error {error} exceeds 1e-9"

    def test_individual_score_threshold_comparisons_hold(self):
        """Individual decimal(18,2) scores converted via _jsonable compare
        correctly against the 50.0 and 75.0 threshold literals used in
        conclusion_engine / analysis_engine.
        """
        import decimal

        from sql_executor import _jsonable

        for above in ("75.01", "76.00", "90.00", "99.99"):
            assert _jsonable(decimal.Decimal(above)) > 75.0, f"{above} not > 75"

        for below in ("74.99", "74.00", "50.00", "0.00"):
            assert _jsonable(decimal.Decimal(below)) < 75.0, f"{below} not < 75"

        for below in ("49.99", "49.00", "0.01"):
            assert _jsonable(decimal.Decimal(below)) < 50.0, f"{below} not < 50"

        for above in ("50.01", "51.00", "75.00", "100.00"):
            assert _jsonable(decimal.Decimal(above)) > 50.0, f"{above} not > 50"

    def test_average_of_many_floats_near_boundary_is_stable(self):
        """100 rows (50x80.00, 50x70.00) whose Decimal mean = exactly 75.00.

        Verifies that summing 100 floats converted via _jsonable and dividing
        gives a result within 1e-10 of the true mean 75.00 — proving that even
        hypothetical future Python-side averaging cannot drift far enough to flip
        a > 75.0 comparison at the boundary.
        """
        import decimal

        from sql_executor import _to_section

        rows = (
            [{"AgniveerNo": f"A{i}", "MarksObtained": decimal.Decimal("80.00")}
             for i in range(50)]
            + [{"AgniveerNo": f"B{i}", "MarksObtained": decimal.Decimal("70.00")}
               for i in range(50)]
        )
        section = _to_section(rows)
        float_scores = [r["marksObtained"] for r in section["records"]]

        assert len(float_scores) == 100
        assert all(isinstance(s, float) for s in float_scores)

        avg = sum(float_scores) / len(float_scores)
        assert abs(avg - 75.0) < 1e-10, f"avg={avg} has too much float error"

    def test_tenth_value_average_near_boundary_is_stable(self):
        """10 rows (5x90.10, 5x59.90) whose Decimal mean = exactly 75.00.

        Uses the .10/.90 suffix values that have IEEE 754 representation error.
        Verifies the error does not compound to flip the comparison.
        """
        import decimal

        from sql_executor import _to_section

        rows = (
            [{"AgniveerNo": f"A{i}", "MarksObtained": decimal.Decimal("90.10")}
             for i in range(5)]
            + [{"AgniveerNo": f"B{i}", "MarksObtained": decimal.Decimal("59.90")}
               for i in range(5)]
        )
        section = _to_section(rows)
        float_scores = [r["marksObtained"] for r in section["records"]]

        avg = sum(float_scores) / len(float_scores)
        # True mean = (5*90.10 + 5*59.90) / 10 = 75.00
        assert abs(avg - 75.0) < 1e-9, f"Mean {avg} drifted > 1e-9 from 75.0"

    def test_aggregate_records_averagescore_is_display_only(self):
        """result_combiner.aggregate_records sums per-row floats for display.
        Verifies the output averageScore is close to the true value and confirms
        this value is NOT compared against any hard grade boundary in the return
        value — it is a display-only field keyed 'averageScore'.
        """
        import decimal

        from result_combiner import aggregate_records
        from sql_executor import _jsonable

        records = (
            [{"agniveerNo": f"A{i}", "platoon": "Alpha",
              "bestTotal": _jsonable(decimal.Decimal("90.00"))}
             for i in range(5)]
            + [{"agniveerNo": f"B{i}", "platoon": "Bravo",
                "bestTotal": _jsonable(decimal.Decimal("60.00"))}
               for i in range(5)]
        )
        result = aggregate_records(records, group_by="platoon",
                                   metric="average_score")

        alpha = next((r for r in result if r["group"] == "Alpha"), None)
        bravo = next((r for r in result if r["group"] == "Bravo"), None)

        assert alpha is not None
        assert abs(alpha["averageScore"] - 90.0) < 0.01
        assert bravo is not None
        assert abs(bravo["averageScore"] - 60.0) < 0.01

        # The return dict contains only display fields — no grade boundary check.
        for row in result:
            assert set(row.keys()) <= {"group", "count", "averageScore"}
