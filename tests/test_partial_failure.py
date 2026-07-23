import unittest
from unittest.mock import patch

from admin_pipeline import execute_admin_query
from intent_engine.query_planner import QueryPlan, QueryType, SubOperation


def _multi_independent_plan(message: str) -> QueryPlan:
    op1 = SubOperation(
        raw_fragment="attendance stats",
        intent_result={"category": "Attendance", "operation": "Present"},
        dotnet_payload={},
    )
    op2 = SubOperation(
        raw_fragment="equipment overdue records",
        intent_result={"category": "Equipment"},
        dotnet_payload={},
    )
    return QueryPlan(
        query_type=QueryType.MULTI_INDEPENDENT,
        operations=[op1, op2],
        confidence=0.9,
        raw_query=message,
        reasoning="test: forced multi_independent plan",
    )


class TestPartialFailure(unittest.TestCase):
    """Post text-to-SQL migration, partial failure works differently than
    the old parallel .NET-calls model:

    - cross_filter is still all-or-nothing at the response level: each
      condition is fetched separately and the combiner intersects the
      returned sets, so a failure in any leg still fails the whole
      cross-filter together. There is no "2 of 3 conditions succeeded"
      degraded mode for cross_filter.
    - compare issues one SQL call per side (sql_query_plan._fetch_compare)
      and bubbles an error if EITHER side fails — there is no more
      "unavailable" placeholder side for compare.
    - multi_independent is the one shape that still degrades gracefully:
      each leg is fetched independently and a failed leg becomes an
      {"unavailable": True} placeholder rather than failing the whole
      request (sql_query_plan._fetch_multi_independent).
    """

    @patch("admin_pipeline.generate_report")
    @patch("admin_pipeline.fetch_sql_results")
    @patch("admin_pipeline.plan_query")
    def test_multi_independent_partial_failure_degrades_gracefully(
        self, mock_plan_query, mock_fetch_sql, mock_generate_report
    ):
        message = "Show attendance stats and equipment overdue records"
        mock_plan_query.return_value = _multi_independent_plan(message)

        attendance_section = {
            "success": True,
            "records": [{"agniveerNo": "101"}],
            "data": [{"agniveerNo": "101"}],
            "count": 1,
        }
        unavailable = {"unavailable": True}
        mock_fetch_sql.return_value = (
            [attendance_section, unavailable],
            [("Attendance", attendance_section), ("Equipment", unavailable)],
            None,
        )
        mock_generate_report.return_value = {
            "message": "Report.",
            "analysis": {"summary": "Analysis", "observations": [], "insights": []},
            "conclusion": {"summary": "Conclusion"},
        }

        result = execute_admin_query(message, {})

        self.assertEqual(result["type"], "query")
        response_payload = result["response_payload"]
        self.assertTrue(response_payload.get("partialFailure"))
        self.assertEqual(response_payload.get("failedSections"), ["Equipment"])

    @patch("admin_pipeline.fetch_sql_results")
    def test_cross_filter_failure_is_all_or_nothing(self, mock_fetch_sql):
        # A cross-filter failure still means the whole request fails, never a
        # partial intersection.
        mock_fetch_sql.return_value = ([], [], "CANNOT_ANSWER")

        result = execute_admin_query(
            "Show top performer in PPT who plays cricket and is currently on leave",
            {},
        )

        self.assertEqual(result["type"], "unrecognised")

    @patch("admin_pipeline.fetch_sql_results")
    def test_compare_failure_is_all_or_nothing(self, mock_fetch_sql):
        # Likewise, a compare bubbles an error if either side fails — there
        # is no more "unavailable" placeholder side.
        mock_fetch_sql.return_value = ([], [], "Timeout on BEPT")

        result = execute_admin_query("Compare PPT and BEPT", {})

        self.assertEqual(result["type"], "unrecognised")

    @patch("admin_pipeline.fetch_sql_results")
    def test_all_intents_fail(self, mock_fetch_sql):
        mock_fetch_sql.return_value = ([], [], "Error 1")
        result = execute_admin_query("Compare PPT and BEPT", {})
        self.assertEqual(result["type"], "unrecognised")
        self.assertIn("rephrase", result["response_payload"]["message"].lower())


if __name__ == "__main__":
    unittest.main()
