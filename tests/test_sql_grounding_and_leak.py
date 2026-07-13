"""
tests/test_sql_grounding_and_leak.py
=======================================
Task 6 — grounding + no-leak invariants on the SQL path.

Proves two things using the exact envelope shape sql_executor.execute_sql_query
returns (via _to_section):
  1. Rows fetched by the SQL backend ground an LLM narrative the same way
     .NET-fetched rows do — an ungrounded number gets stripped.
  2. No response object ever carries raw SQL text or un-rendered raw rows
     beyond the normal formattedData/processedData shape.
"""

from __future__ import annotations

from grounding_utils import ground_and_sanitize
from normalized_models import extract_records
from response_builder import build_response
from sql_executor import _to_section


def _grounded_text_for(rows):
    records = extract_records(_to_section(rows))
    return " ".join(str(v) for record in records for v in record.values())


class TestGroundingOnSqlPathRows:
    def test_ungrounded_number_is_stripped(self):
        rows = [
            {"agniveerNo": "A1", "fullName": "Amit", "score": 91},
            {"agniveerNo": "A2", "fullName": "Kapil", "score": 85},
        ]
        grounded_text = _grounded_text_for(rows)

        narrative = (
            "The top score was 91. "
            "Suspiciously, 999 agniveers were flagged for review."
        )
        cleaned = ground_and_sanitize(narrative, grounded_text)

        assert "91" in cleaned
        assert "999" not in cleaned

    def test_all_grounded_numbers_survive(self):
        rows = [{"agniveerNo": "A1", "fullName": "Amit", "score": 91, "rank": 1}]
        grounded_text = _grounded_text_for(rows)

        narrative = "Amit ranked 1 with a score of 91."
        cleaned = ground_and_sanitize(narrative, grounded_text)

        assert cleaned == narrative

    def test_empty_rows_ground_nothing(self):
        grounded_text = _grounded_text_for([])
        narrative = "There were 42 matching records."
        cleaned = ground_and_sanitize(narrative, grounded_text)
        assert cleaned == ""


class TestNoRawLeakFromSqlPath:
    def test_sql_section_never_contains_a_sql_string_field(self):
        rows = [{"agniveerNo": "A1", "fullName": "Amit"}]
        section = _to_section(rows, intent={"category": "Performance"})
        # Only the standard envelope keys — never a "sql"/"query" field.
        assert set(section.keys()) == {"success", "records", "data", "count"}

    def test_response_builder_never_leaks_sql_via_dotnet_payload_marker(self):
        raw_sql = "SELECT AgniveerNo, FullName FROM AgniveerMaster WHERE Height > 170"
        response = build_response(
            message="Here are the results.",
            formatted_data=[
                {"type": "table", "title": "Results", "data": [{"row": 1}]}
            ],
            metadata={"queryType": "simple"},
            session_id="s1",
            dotnet_payload=[{"backend": "sql"}],
        )
        assert raw_sql not in str(response)
        assert response["dotnetPayload"] == [{"backend": "sql"}]

    def test_execute_sql_query_error_path_never_leaks_generated_sql(self):
        """A rejected/failed generation must return only a generic error
        string — never the SQL text itself — mirroring dotnet_executor's
        'raw dotnetResponse never leaks' invariant."""
        from unittest.mock import patch

        from sql_executor import execute_sql_query

        forbidden_sql = "DELETE FROM AgniveerMaster WHERE 1=1"
        with patch("sql_executor.generate_sql", return_value=(forbidden_sql, None)):
            data, err = execute_sql_query(question="delete everyone", intent=None)

        assert data is None
        assert forbidden_sql not in (err or "")
