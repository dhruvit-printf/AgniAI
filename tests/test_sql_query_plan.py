"""
tests/test_sql_query_plan.py
=============================
Unit tests for sql_query_plan.py — drives sql_executor per query-plan type
and hands results to the EXISTING result_combiner so the SQL path's output
shape matches the .NET path exactly (Task 4 accept criteria).
"""

from __future__ import annotations

from unittest.mock import patch

from intent_engine.query_planner import QueryPlan, QueryType, SubOperation
from result_combiner import combine_results
from sql_query_plan import run_sql_plan


def _op(fragment: str, category: str, **extra) -> SubOperation:
    intent = {"category": category, **extra}
    return SubOperation(raw_fragment=fragment, intent_result=intent, dotnet_payload={})


def _plan(query_type: QueryType, operations, confidence=0.9) -> QueryPlan:
    return QueryPlan(
        query_type=query_type,
        operations=operations,
        confidence=confidence,
        raw_query="q",
        reasoning="test",
    )


class TestSimple:
    def test_simple_matches_combine_results_passthrough(self):
        op = _op("top performers in bpet", "Performance")
        plan = _plan(QueryType.SIMPLE, [op])
        section = {
            "success": True,
            "records": [{"agniveerNo": "A1"}],
            "data": [{"agniveerNo": "A1"}],
            "count": 1,
        }

        with patch("sql_query_plan.execute_sql_query") as mock_exec:
            mock_exec.return_value = (section, None)
            data, err = run_sql_plan(
                plan, "top performers in bpet", {"category": "Performance"}
            )

        assert err is None
        expected = combine_results(
            [section], [("Performance", section)], "simple", {"category": "Performance"}
        )
        assert data == expected == section

    def test_simple_bubbles_error(self):
        op = _op("gibberish query", "Performance")
        plan = _plan(QueryType.SIMPLE, [op])

        with patch("sql_query_plan.execute_sql_query") as mock_exec:
            mock_exec.return_value = (None, "CANNOT_ANSWER")
            data, err = run_sql_plan(plan, "gibberish query", {})

        assert data is None
        assert err == "CANNOT_ANSWER"


class TestCrossFilter:
    def test_cross_filter_single_query_combined_via_combine_results(self):
        op1 = _op("suffered from fever", "Medical")
        op2 = _op("failed bpet", "Performance")
        plan = _plan(QueryType.CROSS_FILTER, [op1, op2])
        section = {
            "success": True,
            "records": [{"agniveerNo": "A1"}, {"agniveerNo": "A2"}],
            "data": [{"agniveerNo": "A1"}, {"agniveerNo": "A2"}],
            "count": 2,
        }

        with patch("sql_query_plan.execute_sql_query") as mock_exec:
            mock_exec.return_value = (section, None)
            data, err = run_sql_plan(
                plan, "who suffered from fever and failed bpet", {}
            )

        assert err is None
        # One execute_sql_query call for the whole intersection (HARD RULE R4).
        assert mock_exec.call_count == 1
        _, kwargs = mock_exec.call_args
        assert "fever" in kwargs["question"] and "bpet" in kwargs["question"]
        assert kwargs["intent"]["query_type"] == "cross_filter"

        assert data["status"] is True
        assert data["filterDepth"] == 1
        assert {r["agniveerNo"] for r in data["records"]} == {"A1", "A2"}

    def test_cross_filter_bubbles_error(self):
        op1 = _op("suffered from fever", "Medical")
        op2 = _op("failed bpet", "Performance")
        plan = _plan(QueryType.CROSS_FILTER, [op1, op2])

        with patch("sql_query_plan.execute_sql_query") as mock_exec:
            mock_exec.return_value = (None, "Statement contains a forbidden keyword.")
            data, err = run_sql_plan(plan, "q", {})

        assert data is None
        assert err is not None


class TestCompare:
    def test_compare_two_labeled_sides(self):
        op1 = _op("bpet scores", "Performance", section="BPET")
        op2 = _op("firing scores", "Performance", section="Firing")
        plan = _plan(QueryType.COMPARE, [op1, op2])
        section1 = {
            "success": True,
            "records": [{"agniveerNo": "A1", "score": 80}],
            "data": [],
            "count": 1,
        }
        section2 = {
            "success": True,
            "records": [{"agniveerNo": "A2", "score": 70}],
            "data": [],
            "count": 1,
        }

        with patch("sql_query_plan.execute_sql_query") as mock_exec:
            mock_exec.side_effect = [(section1, None), (section2, None)]
            data, err = run_sql_plan(plan, "compare bpet vs firing", {})

        assert err is None
        assert mock_exec.call_count == 2
        expected = combine_results(
            [section1, section2],
            [("BPET", section1), ("Firing", section2)],
            "compare",
            {},
        )
        assert data == expected
        assert data["queryType"] == "comparison"

    def test_compare_bubbles_error_on_either_side(self):
        op1 = _op("bpet scores", "Performance", section="BPET")
        op2 = _op("firing scores", "Performance", section="Firing")
        plan = _plan(QueryType.COMPARE, [op1, op2])

        with patch("sql_query_plan.execute_sql_query") as mock_exec:
            mock_exec.side_effect = [
                ({"success": True, "records": [], "data": [], "count": 0}, None),
                (None, "CANNOT_ANSWER"),
            ]
            data, err = run_sql_plan(plan, "q", {})

        assert data is None
        assert err == "CANNOT_ANSWER"


class TestMultiIndependent:
    def test_multi_independent_never_merges(self):
        op1 = _op("attendance summary", "Attendance")
        op2 = _op("current leave status", "Leave")
        plan = _plan(QueryType.MULTI_INDEPENDENT, [op1, op2])
        section1 = {
            "success": True,
            "records": [{"agniveerNo": "A1"}],
            "data": [],
            "count": 1,
        }
        section2 = {
            "success": True,
            "records": [{"agniveerNo": "A2"}],
            "data": [],
            "count": 1,
        }

        with patch("sql_query_plan.execute_sql_query") as mock_exec:
            mock_exec.side_effect = [(section1, None), (section2, None)]
            data, err = run_sql_plan(plan, "q", {})

        assert err is None
        assert data["queryType"] == "multi_independent"
        assert data["sectionCount"] == 2
        labels = [s["label"] for s in data["sections"]]
        assert labels == ["Attendance", "Leave"]
        # Never merged/intersected — each section keeps its own records
        # (normalize_response also inherits the envelope's "success" scalar
        # into each record — existing, unrelated normalizer behavior).
        assert data["sections"][0]["data"] == [{"agniveerNo": "A1", "success": True}]
        assert data["sections"][1]["data"] == [{"agniveerNo": "A2", "success": True}]

    def test_multi_independent_partial_failure_is_degraded_not_bubbled(self):
        op1 = _op("attendance summary", "Attendance")
        op2 = _op("current leave status", "Leave")
        plan = _plan(QueryType.MULTI_INDEPENDENT, [op1, op2])
        section1 = {
            "success": True,
            "records": [{"agniveerNo": "A1"}],
            "data": [],
            "count": 1,
        }

        with patch("sql_query_plan.execute_sql_query") as mock_exec:
            mock_exec.side_effect = [(section1, None), (None, "CANNOT_ANSWER")]
            data, err = run_sql_plan(plan, "q", {})

        assert err is None
        assert data["degraded"] is True
        assert data["failedSections"] == ["Leave"]
        assert data["sectionCount"] == 2

    def test_multi_independent_all_failed_bubbles_error(self):
        op1 = _op("attendance summary", "Attendance")
        op2 = _op("current leave status", "Leave")
        plan = _plan(QueryType.MULTI_INDEPENDENT, [op1, op2])

        with patch("sql_query_plan.execute_sql_query") as mock_exec:
            mock_exec.side_effect = [(None, "CANNOT_ANSWER"), (None, "CANNOT_ANSWER")]
            data, err = run_sql_plan(plan, "q", {})

        assert data is None
        assert err is not None
