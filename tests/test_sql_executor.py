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
        key = ("Performance", "TopPerformers", "Top")
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
            GOLDEN_QUERIES[("Performance", "TopPerformers", "Top")],
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

    def test_no_raw_sql_in_returned_section(self):
        """The section returned to the caller must never contain the raw SQL
        text — only rows/records/count, mirroring the .NET no-leak invariant."""
        sql_text = "SELECT AgniveerNo, FullName FROM AgniveerMaster WHERE Height > 170"
        with (
            patch("sql_executor.generate_sql") as mock_gen,
            patch("sql_executor.run_readonly") as mock_run,
        ):
            mock_gen.return_value = (sql_text, None)
            mock_run.return_value = ([{"agniveerNo": "A1", "fullName": "X"}], None)
            data, err = execute_sql_query(question="tall agniveers", intent=None)
            assert err is None
            assert sql_text not in str(data)
            assert set(data.keys()) == {"success", "records", "data", "count"}
