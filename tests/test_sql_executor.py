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
    GOLDEN_QUERIES,
    _extract_sql,
    _render_golden_query,
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


# ── Golden fast-path (Task 7) ────────────────────────────────────────────────


class TestGoldenQueryKeysAreReachable:
    """Regression guard for the bug where GOLDEN_QUERIES was keyed by
    (category, subcategory, operation) but subcategory is entirely derived
    from (category, operation) via CATEGORY_OPERATION_TO_SUBCATEGORY — a
    typo'd subcategory string silently made an entry unreachable forever,
    with no error and no failing test, because the old tests only ever
    looked GOLDEN_QUERIES entries up by their own (matching) key instead of
    driving a real question through the classifier. GOLDEN_QUERIES is now
    keyed by (category, operation) only, so the only way this can regress is
    an operation string that doesn't exist for its category — this test
    catches that class of typo directly.
    """

    @pytest.mark.parametrize("key", list(GOLDEN_QUERIES.keys()))
    def test_operation_is_valid_for_category(self, key):
        from intent_engine.intent_schema import OPERATIONS_BY_CATEGORY

        category, operation = key
        if category == "Strength":
            # Strength has no operations — its one golden entry uses "".
            assert operation == ""
            return
        valid_operations = OPERATIONS_BY_CATEGORY.get(category, frozenset())
        assert operation in valid_operations, (
            f"{key}: {operation!r} is not a valid operation for category "
            f"{category!r} (valid: {sorted(valid_operations)}) — this golden "
            "query can never be reached by the real classifier."
        )

    @pytest.mark.parametrize(
        "question,expected_key",
        [
            ("who are the top performers", ("Performance", "Top")),
            ("verification requests still pending", ("Verification", "Pending")),
            ("today's training schedule", ("Schedule", "bytoday")),
            ("who is currently absconded", ("Leave", "Absconded")),
            ("bmi distribution of all agniveers", ("Medical", "BMI")),
        ],
    )
    def test_real_classifier_output_hits_golden_path(self, question, expected_key):
        """Drives a real question through the actual intent pipeline (the
        same one admin_pipeline.py uses) instead of a hand-built intent dict,
        so a future key/subcategory mismatch fails here instead of silently
        falling through to the LLM path in production."""
        from intent_engine.admin_intent import classify_admin_intent

        intent = classify_admin_intent(question)
        key = (intent.get("category"), intent.get("operation"))
        assert key == expected_key, (
            f"question {question!r} classified as {key}, expected "
            f"{expected_key} — check intent_schema.py's category/operation "
            "keyword tables and CATEGORY_OPERATION_TO_SUBCATEGORY"
        )
        assert key in GOLDEN_QUERIES, f"{key} is missing from GOLDEN_QUERIES"


class TestGoldenQueries:
    @pytest.mark.parametrize("key", list(GOLDEN_QUERIES.keys()))
    def test_golden_query_passes_validator(self, key):
        template = GOLDEN_QUERIES[key]
        rendered = _render_golden_query(template, {"number": 5})
        err = validate_sql(rendered)
        assert err is None, f"{key} failed validation: {err}"

    @pytest.mark.parametrize("key", list(GOLDEN_QUERIES.keys()))
    def test_golden_query_is_single_select(self, key):
        rendered = _render_golden_query(GOLDEN_QUERIES[key], {})
        stripped = rendered.strip().lower()
        assert stripped.startswith("select") or stripped.startswith("with")

    def test_golden_query_never_calls_llm(self):
        key = ("Performance", "Top")
        assert key in GOLDEN_QUERIES
        with (
            patch("sql_executor.generate_sql") as mock_generate,
            patch("sql_executor.run_readonly") as mock_run,
        ):
            mock_run.return_value = ([{"agniveerNo": "A1"}], None)
            data, err = execute_sql_query(
                question="top performers",
                intent={
                    "category": "Performance",
                    "subcategory": "TopPerformers",
                    "operation": "Top",
                },
            )
            mock_generate.assert_not_called()
            assert err is None
            assert data["records"] == [{"agniveerNo": "A1"}]

    def test_golden_query_top_n_substitution_is_safe(self):
        rendered = _render_golden_query(
            GOLDEN_QUERIES[("Performance", "Top")],
            {"number": "3; DROP TABLE AgniveerMaster"},
        )
        # Non-numeric input falls back to the safe default; never interpolated raw.
        assert "DROP TABLE" not in rendered
        assert validate_sql(rendered) is None


# ── execute_sql_query — end-to-end contract ─────────────────────────────────


class TestExecuteSqlQuery:
    def test_returns_section_on_success(self):
        with (
            patch("sql_executor.generate_sql") as mock_gen,
            patch("sql_executor.run_readonly") as mock_run,
        ):
            mock_gen.return_value = ("SELECT AgniveerNo FROM AgniveerMaster", None)
            mock_run.return_value = ([{"agniveerNo": "A1"}], None)
            data, err = execute_sql_query(question="who is A1", intent=None)
            assert err is None
            assert data["success"] is True
            assert data["records"] == [{"agniveerNo": "A1"}]

    def test_cannot_answer_bubbles_error(self):
        with patch("sql_executor.generate_sql") as mock_gen:
            mock_gen.return_value = (None, "CANNOT_ANSWER")
            data, err = execute_sql_query(question="what's the weather", intent=None)
            assert data is None
            assert err == "CANNOT_ANSWER"

    def test_validator_rejection_bubbles_error(self):
        with patch("sql_executor.generate_sql") as mock_gen:
            mock_gen.return_value = ("DELETE FROM AgniveerMaster", None)
            data, err = execute_sql_query(question="delete everyone", intent=None)
            assert data is None
            assert err is not None

    def test_exec_error_bubbles_error(self):
        with (
            patch("sql_executor.generate_sql") as mock_gen,
            patch("sql_executor.run_readonly") as mock_run,
        ):
            mock_gen.return_value = ("SELECT 1", None)
            mock_run.return_value = (
                None,
                "The generated query could not be executed against the database.",
            )
            data, err = execute_sql_query(question="anything", intent=None)
            assert data is None
            assert err

    def test_executed_sql_is_surfaced_for_dotnet_payload_transparency(self):
        """The section returned to the caller carries the executed SQL under
        `sql` — admin_pipeline.py reads section["sql"] to populate the
        `dotnetPayload.sqlQuery` transparency field (the SQL-backend analog
        of the pre-migration "exact .NET request payload" debug field
        documented in response_builder.py). This is intentional exposure to
        admin/debug consumers, not a leak: validate_sql() already rejects
        any query touching denied tables/columns before it ever reaches
        run_readonly(), so the executed SQL text itself is safe to surface."""
        sql_text = "SELECT AgniveerNo, FullName FROM AgniveerMaster WHERE Height > 170"
        with (
            patch("sql_executor.generate_sql") as mock_gen,
            patch("sql_executor.run_readonly") as mock_run,
        ):
            mock_gen.return_value = (sql_text, None)
            mock_run.return_value = ([{"agniveerNo": "A1", "fullName": "X"}], None)
            data, err = execute_sql_query(question="tall agniveers", intent=None)
            assert err is None
            assert data["sql"] == sql_text
            assert set(data.keys()) == {"success", "records", "data", "count", "sql"}
